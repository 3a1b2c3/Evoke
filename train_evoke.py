import os


os.environ["HF_ENABLE_PARALLEL_LOADING"] = "yes"
os.environ["HF_PARALLEL_LOADING_WORKERS"] = "8"

import argparse
import copy
import json
import logging
import math
import random
import re
import shutil
from datetime import timedelta
from pathlib import Path

import deepspeed
import numpy as np
import torch
import torch.distributed.checkpoint as dcp
import transformers
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import (
    DeepSpeedPlugin,
    DistributedDataParallelKwargs,
    InitProcessGroupKwargs,
    ProjectConfiguration,
    broadcast,
    set_seed,
)
from evoke.modules.evoke_kernels import (
    replace_all_norms_with_flash_norms,
    replace_rmsnorm_with_fp32,
    replace_rope_with_flash_rope,
)
from evoke.modules.transformer_evoke import EvokeTransformer3DModel
from evoke.pipelines.pipeline_evoke import EvokePipeline
from evoke.scheduler.scheduling_evoke import EvokeScheduler
from evoke.utils.create_ema_zero3 import _z3_params_to_fetch
from evoke.utils.create_ema_zero3_lora import create_ema_final, gather_zero3ema
from evoke.utils.train_config import Args
from evoke.utils.utils_base import (
    NORM_LAYER_PREFIXES,
    compare_configs,
    encode_prompt,
    get_optimizer,
    load_extra_components,
    load_model_checkpoint,
    organize_checkpoint_weights_view,
    save_extra_components,
    save_model_checkpoint,
)
from evoke.utils.utils_evoke_base import (
    _flow_loss,
    prepare_stage1_clean_input_from_latents,
    prepare_stage1_noise_input,
    prepare_stage2_noise_input,
)
from evoke.utils.utils_evoke_post import (
    OptimizedLowVRAMManager,
    _critic_loss,
    _generator_loss,
    _sf_prof,
    sf_prof_mark,
    sf_prof_accum,
    sf_prof_step_begin,
    sf_prof_step_end,
    _ode_regression_loss,
    _offload_frozen_params_to,
    _evoke_teacher_base_to,
    merge_dict_list,
    sample_dynamic_dmd_num_latent_sections,
    sf_curriculum_lookup,
)
from evoke.utils.utils_recycle_batch import get_timesteps
from evoke.videoalign.inference import VideoVLMRewardInference
from packaging import version
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoTokenizer,
    UMT5EncoderModel,
)

import diffusers
from diffusers import (
    AutoencoderKLWan,
    FlowMatchEulerDiscreteScheduler,
    UniPCMultistepScheduler,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    _collate_lora_metadata,
    cast_training_params,
    free_memory,
)
from diffusers.utils import (
    check_min_version,
    convert_unet_state_dict_to_peft,
    export_to_video,
    is_wandb_available,
)
from diffusers.utils.import_utils import is_torch_npu_available, is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module


if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.36.0.dev0")

logger = get_logger(__name__)

if is_torch_npu_available():
    torch.npu.config.allow_internal_format = False


def main(args):
    # / fail-fast validation (must be called explicitly).
    # the two validators self-guard and are mutually exclusive: sf10s returns early when arch!=evoke_teacher; sf_evoke returns early when arch==evoke_teacher or sf is off.
    from evoke.utils.train_config import (
        validate_cloud_warp_backend, validate_sf10s_evoke_teacher_config, validate_sf_evoke_config)
    validate_sf10s_evoke_teacher_config(args)
    validate_sf_evoke_config(args)
    # Depth-backend validation: also checks that the backend's assets exist, so a bad path fails here
    # rather than as a skipped ingest window that leaves an all-black warp behind.
    validate_cloud_warp_backend(args)
    _sf_evoke_teacher = getattr(args.model_config, "real_score_arch", "evoke") == "evoke_teacher"
    # evoke-backbone self-forcing (prefix-anchor multi-section rollout + tier-conditioned scoring.
    _sf_evoke = (not _sf_evoke_teacher) and bool(getattr(args.training_config, "sf_self_forcing", False))
    # the generic self-forcing machinery (data sf keys / rollout prefix / per-section prompt) applies to both paths.
    _sf_any = _sf_evoke_teacher or _sf_evoke
    # dual real-score teacher: besides the evoke_teacher long-range teacher, also attach a frozen Evoke-Base
    # pose teacher (camera force); only active with evoke_teacher + dual_teacher.enabled, otherwise single-teacher bit-identical.
    _sf_dual = bool(_sf_evoke_teacher and getattr(getattr(args.model_config, "dual_teacher", None), "enabled", False))
    # front large-window decoupling: with dual, the front part [prefix|pred_front] goes to the EvokeTeacher critic, the tail block pred_tail
    #   goes to the Evoke critic (a second _critic_loss on evoke_critic_accelerator). dual off or flag off -> single critic, bit-identical.
    _sf_front_window = bool(_sf_dual and getattr(args.training_config, "sf_evoke_teacher_front_window", False))
    # dual_teacher.offload: during compute_kl_grad scoring Evoke goes to GPU and the EvokeTeacher **frozen base** is swapped out to CPU
    #   (frees 28G for Evoke, avoids EvokeTeacher56+Evoke28+student28+activations=156 hitting the 141 wall); after scoring Evoke is swapped out and the base swapped back.
    #   NOTE: only Evoke (frozen) + the EvokeTeacher frozen base are moved (not DeepSpeed flat-buffers); the trainable critic-LoRA stays on GPU the whole time --
    # moving DeepSpeed-managed params leaks ZeRO-2 .data re-pointers (measured +116GB/step -> host OOM).
    #   when on, Evoke initially stays on CPU; build vram_manager (dmd_is_low_vram_mode is always false: warp-tail needs the vae so low_vram is off, orthogonal).
    _dual_offload = bool(_sf_dual and getattr(args.model_config.dual_teacher, "offload", False))
    # EvokeTeacher per-expert offload (high<->low residency swap), independent of the camera-teacher switch:
    #   the dual experts (2x14B) need it to avoid OOM even with dual_teacher.enabled=false. Every Evoke-specific
    #   offload call site is guarded by `real_score_model_hb is not None`, so this stays a no-op when hb=None.
    _et_offload = bool(_sf_evoke_teacher and getattr(getattr(args.model_config, "evoke_teacher", None), "offload", False))

    # detect whether the 3 DMD engines are ZeRO-3 sharded (zero_optimization.stage==3 in
    #   dmd_generator/critic_deepspeed_config). under ZeRO-3 DeepSpeed manages a sharded flat-buffer for **all** params (frozen base included) ->
    #   manual offload (per-param `.data` re-pointering in _offload_frozen_params_to / bulk `.to()` in OptimizedLowVRAMManager)
    #   breaks DeepSpeed shard bookkeeping (empty shards with numel=0 get moved + fall out of partition accounting). NOTE: sharding itself already replaces offload
    #   (3x14B -> /num_gpus resident, ~5.25G/card on 16 cards), so ZeRO-3 must turn dual_teacher.offload off. fail-fast here to prevent that footgun.
    def _ds_zero_stage(cfg_path):
        try:
            with open(cfg_path, "r") as _f:
                return int(json.load(_f).get("zero_optimization", {}).get("stage", 0))
        except Exception:
            return 0
    _dmd_zero3 = False
    if args.training_config.is_train_dmd and args.training_config.dmd_generator_deepspeed_config:
        _gen_stage = _ds_zero_stage(args.training_config.dmd_generator_deepspeed_config)
        _crit_stage = _ds_zero_stage(args.training_config.dmd_critic_deepspeed_config)
        _dmd_zero3 = (_gen_stage == 3 or _crit_stage == 3)
        if _dmd_zero3:
            assert _gen_stage == 3 and _crit_stage == 3, (
                f"[ZeRO-3] zero stage of the generator/critic ds config must match (the 3 engines share the process group), "
                f"got gen={_gen_stage}, critic={_crit_stage}")
    if _dmd_zero3 and _dual_offload:
        raise ValueError(
            "dual_teacher.offload must be false: ZeRO-3 already shards the 3x14B resident weights "
            "(sharding replaces offload); manual offload (_offload_frozen_params_to / OptimizedLowVRAMManager) breaks DeepSpeed "
            "shard management. please set model_config.dual_teacher.offload=false in the config.")

    if args.data_config.use_multi_dataset:
        # Online multi-dataset mode; VAE+T5 encoding done inline in the main process.
        from evoke.dataset.dataloader_multi_dataset import (
            BucketedFeatureDataset,
            BucketedSampler,
            collate_fn,
        )
    elif args.data_config.use_stage3_dataset:
        from evoke.dataset.dataloader_dmd import (
            BucketedFeatureDataset,
            BucketedSampler,
            collate_fn,
        )
    elif args.data_config.use_stage1_dataset:
        from evoke.dataset.dataloader_history_latents_dist import (
            BucketedFeatureDataset,
            BucketedSampler,
            collate_fn,
        )
    else:
        from evoke.dataset.dataloader_mp4_dist import (
            BucketedFeatureDataset,
            BucketedSampler,
            collate_fn,
        )

    if torch.backends.mps.is_available() and args.training_config.mixed_precision == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    # load dmd reward model
    reward_model = None
    if args.training_config.is_use_reward_model:
        reward_model = VideoVLMRewardInference(args.model_config.reward_model_name_or_path)
        reward_model.model.requires_grad_(False)
        reward_model.model.eval()

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    init_kwargs = InitProcessGroupKwargs(backend="nccl", timeout=timedelta(seconds=1800))

    # Support 2 models training using deepspeed.
    # https://huggingface.co/docs/accelerate/usage_guides/deepspeed_multiple_model
    deepspeed_plugins = None
    dmd_deepspeed_training = (
        args.training_config.is_train_dmd
        and args.training_config.dmd_generator_deepspeed_config is not None
        and args.training_config.dmd_critic_deepspeed_config is not None
    )
    if dmd_deepspeed_training:
        generator_zero_plugin = DeepSpeedPlugin(hf_ds_config=args.training_config.dmd_generator_deepspeed_config)
        critic_zero_plugin = DeepSpeedPlugin(hf_ds_config=args.training_config.dmd_critic_deepspeed_config)
        deepspeed_plugins = {"generator": generator_zero_plugin, "critic_model": critic_zero_plugin}
        # dual-teacher: register a third DeepSpeed engine -- the critic dedicated to the Evoke backbone.
        #   separate plugin (reuses the critic ds config; each plugin is parsed independently) + separate accelerator (below). NOTE: one accelerator per
        #   engine: do not prepare a second time on critic_accelerator (that breaks its engine.step()). not registered when dual is off -> bit-identical.
        if _sf_dual:
            evoke_critic_zero_plugin = DeepSpeedPlugin(hf_ds_config=args.training_config.dmd_critic_deepspeed_config)
            deepspeed_plugins["critic_evoke"] = evoke_critic_zero_plugin

    accelerator = Accelerator(
        gradient_accumulation_steps=args.training_config.gradient_accumulation_steps,
        mixed_precision=args.training_config.mixed_precision,
        log_with=args.report_to.report_to,
        project_config=accelerator_project_config,
        deepspeed_plugins=deepspeed_plugins,
        kwargs_handlers=[kwargs, init_kwargs],
    )
    if (
        accelerator.distributed_type == DistributedType.DEEPSPEED
        and args.training_config.is_train_dmd
        and not args.training_config.dmd_generator_deepspeed_config
        and not args.training_config.dmd_critic_deepspeed_config
    ):
        raise ValueError("`--deepspeed_config` is required for DMD distillation.")

    if dmd_deepspeed_training:
        # Keep the shared Accelerate GradientState on the same accumulation
        # schedule for all three engines.  With a default Accelerator() here,
        # GA>1 on the primary accelerator can be overwritten back to 1.
        # Existing production configs use GA=1, so their behavior is unchanged.
        critic_accelerator = Accelerator(
            gradient_accumulation_steps=args.training_config.gradient_accumulation_steps
        )
        # accelerator dedicated to the Evoke critic (separate engine; picks the "critic_evoke" plugin @ prepare).
        if _sf_dual:
            evoke_critic_accelerator = Accelerator(
                gradient_accumulation_steps=args.training_config.gradient_accumulation_steps
            )

    # sequence-parallel (SP) subgroup init: G consecutive cards inside a group share one clip and split, along the frame dim,
    #   the EvokeTeacher teacher/critic full-front-window long-sequence activations (~109G/card -> /G); across groups = ZeRO-3 DP (effective bs=world/G).
    #   must be called on all ranks (new_group is a collective; every rank builds each subgroup in the same order); only the dual EvokeTeacher critic consumes it.
    # G=1 -> no groups built, sp_runtime is_sp_enabled()=False -> all identity, byte-identical.
    _sp_world_size = int(getattr(args.training_config, "sf_critic_sp_world_size", 1) or 1)
    if _sp_world_size > 1:
        import torch.distributed as _sp_dist
        assert _sp_dist.is_initialized(), "[SP] distributed must already be initialized (after the accelerator is built)"
        _sp_ws = _sp_dist.get_world_size()
        assert _sp_ws % _sp_world_size == 0, \
            f"[SP] world_size={_sp_ws} is not divisible by sf_critic_sp_world_size={_sp_world_size}"
        from evoke.modules.evoke_teacher.sp_runtime import init_sequence_parallel
        init_sequence_parallel(_sp_world_size)
        if accelerator.is_main_process:
            print(f"[SP] init_sequence_parallel(G={_sp_world_size}): "
                  f"{_sp_ws // _sp_world_size} groups x {_sp_world_size} cards, effective bs={_sp_ws // _sp_world_size}", flush=True)

    # student-side parallelism: second-level split of the same SP group, G = G_p x G_u (teacher/critic SP-G untouched).
    #   mechanism A (chunk parallel): the N backward subgraphs are dispatched by k % G_p, zero communication.
    #   mechanism B (Ulysses): head-dim all-to-all on attention inside the U-subgroup (G_u consecutive cards).
    #   Must be called on **all** ranks (new_group is a collective), after init_sequence_parallel, and must sit
    #   outside `if _sp_world_size > 1:` -- `_stu_sp` is read unconditionally downstream. Both flags off -> identity.
    from evoke.modules import student_sp as _stu_sp
    _stu_sp.init_student_sp(
        sp_world_size=_sp_world_size,
        chunk_parallel=bool(getattr(args.training_config, "sf_student_chunk_parallel", False)),
        ulysses_size=int(getattr(args.training_config, "sf_student_sp_ulysses", 1) or 1),
        diag=bool(getattr(args.training_config, "sf_student_cp_diag", False)),
        skip_first_chunk=bool(getattr(args.training_config, "dmd_score_skip_first_chunk", False)),
    )

    # turn off zero.Init inside transformers/diffusers from_pretrained: building a DeepSpeedPlugin(stage3) sets the
    #   transformers global zero3 flag, which shards models at construction time. That breaks the frozen T5/VAE (never
    #   prepare'd, so no all-gather hook -> `weight must be 2-D` in forward) and this repo's manual `.to(device)`/dtype
    #   cast before prepare. After the unset, from_pretrained loads full weights and accelerator.prepare shards the
    #   three transformers per the plugin ds_config; T5/VAE stay fully resident.
    if _dmd_zero3:
        from transformers.integrations.deepspeed import unset_hf_deepspeed_config
        unset_hf_deepspeed_config()
        if accelerator.is_main_process:
            print("[ZeRO-3] unset_hf_deepspeed_config(): from_pretrained no longer zero.Init's (frozen T5/VAE stay full; "
                  "the 3 transformers are sharded by accelerator.prepare)", flush=True)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        config_path = os.path.join(args.output_dir, "config.json")
        current_conf = OmegaConf.to_container(args, resolve=True)
        # [v2v ODE dump] gen mode is not resumable training, so skip the config consistency guard (smoke->full changing max_samples is not a conflict, just overwrite)
        if os.path.exists(config_path) and not args.training_config.is_dump_ode_traj:
            with open(config_path, "r") as f:
                existing_conf = json.load(f)

            ignore_keys = {"training_config.local_rank"}
            mismatches = compare_configs(existing_conf, current_conf, ignore_keys=ignore_keys)
            if mismatches:
                print("Config mismatches found:")
                for mismatch in mismatches:
                    print(f"  - {mismatch}")
                raise ValueError("Configuration mismatch detected!")
        else:
            with open(config_path, "w") as f:
                json.dump(current_conf, f, indent=4)

    if args.training_config.use_ema:
        args.training_config.ema_zero3_port = os.environ.get("MASTER_PORT", "12345")

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    if args.report_to.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Validate GEO sample-level conditioning ratios at startup.
    _geo_t2v_r = float(getattr(args.training_config, "geo_condition_t2v_ratio", 0.0) or 0.0)
    _geo_i2v_r = float(getattr(args.training_config, "geo_condition_i2v_ratio", 0.0) or 0.0)
    assert 0.0 <= _geo_t2v_r <= 1.0, f"geo_condition_t2v_ratio must be in [0,1], got {_geo_t2v_r}"
    assert 0.0 <= _geo_i2v_r <= 1.0, f"geo_condition_i2v_ratio must be in [0,1], got {_geo_i2v_r}"
    assert _geo_t2v_r + _geo_i2v_r <= 1.0 + 1e-6, (
        f"geo_condition_t2v_ratio + geo_condition_i2v_ratio must be ≤ 1, "
        f"got {_geo_t2v_r} + {_geo_i2v_r} = {_geo_t2v_r + _geo_i2v_r:.3f}"
    )
    _geo_full_r = max(0.0, 1.0 - _geo_t2v_r - _geo_i2v_r)
    _use_geo_train_cfg = bool(
        getattr(args.training_config, "use_geometric_state", None)
        if getattr(args.training_config, "use_geometric_state", None) is not None
        else getattr(args.data_config, "use_geometric_state", False)
    )
    if accelerator.is_main_process:
        if _use_geo_train_cfg:
            logger.info(
                f"[GEO-mix] sample-level conditioning mode effective ratios: "
                f"t2v={_geo_t2v_r:.2f}, i2v={_geo_i2v_r:.2f}, full_geo={_geo_full_r:.2f}"
            )
            if bool(args.training_config.is_random_drop):
                logger.info(
                    "[GEO-mix] WARN: is_random_drop=true but _use_geo_train=true -> the old random_drop_* is disabled automatically, "
                    "everything goes through the geo_condition_*_ratio path. suggest turning is_random_drop off in the yaml to avoid confusion."
                )
        else:
            if _geo_t2v_r + _geo_i2v_r > 0:
                logger.info(
                    f"[GEO-mix] WARN: geo_condition_*_ratio is set but use_geometric_state=false -> has no effect. "
                    f"(t2v={_geo_t2v_r}, i2v={_geo_i2v_r})"
                )


    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load the tokenizers
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_config.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.model_config.revision,
    )

    # Cast non-trainable weights to half-precision for inference.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Load scheduler and models
    if args.training_config.is_enable_stage2:
        noise_scheduler = EvokeScheduler(
            shift=args.training_config.stage2_timestep_shift,
            stages=args.training_config.stage2_num_stages,
            stage_range=args.training_config.stage2_stage_range,
            gamma=args.training_config.stage2_scheduler_gamma,
        )
        noise_scheduler_copy = copy.deepcopy(noise_scheduler)
    else:
        # NOTE: resolve relative to __file__: this used to be a relative path, which raises FileNotFoundError when launched from a cwd outside the repo root.
        _sched_cfg = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "configs", "scheduler", "scheduler_config.json")
        noise_scheduler = UniPCMultistepScheduler.from_pretrained(_sched_cfg)
        noise_scheduler_copy = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000)
        if args.training_config.is_train_dmd:
            noise_scheduler.config.flow_shift = args.training_config.dmd_timestep_shift

    if args.training_config.is_train_dmd:
        if args.training_config.is_enable_stage2:
            critic_noise_scheduler = EvokeScheduler(
                shift=args.training_config.stage2_timestep_shift,
                stages=args.training_config.stage2_num_stages,
                stage_range=args.training_config.stage2_stage_range,
                gamma=args.training_config.stage2_scheduler_gamma,
            )
        else:
            critic_noise_scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000)

    vae = AutoencoderKLWan.from_pretrained(
        args.model_config.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.model_config.revision,
        variant=args.model_config.variant,
        torch_dtype=(torch.float32 if args.model_config.upcast_vae else weight_dtype),
        device_map=(None if _dmd_zero3 else accelerator.device),  # [ZeRO-3] from_pretrained rejects device_map under the global zero3 flag; None -> placed by .to(target_device) below
    )
    if args.model_config.enable_slicing:
        vae.enable_slicing()
    if args.model_config.enable_tiling:
        vae.enable_tiling()

    # Precompute VAE normalization constants for online encoding.
    online_latents_mean = None
    online_latents_std = None
    if args.data_config.use_multi_dataset:
        online_latents_mean = (
            torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1)
            .to(accelerator.device, vae.dtype)
        )
        online_latents_std = (
            1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1)
            .to(accelerator.device, vae.dtype)
        )

    text_encoder = UMT5EncoderModel.from_pretrained(
        args.model_config.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.model_config.revision,
        variant=args.model_config.variant,
        dtype=weight_dtype,
        device_map=(None if _dmd_zero3 else accelerator.device),  # [ZeRO-3] transformers from_pretrained rejects device_map under the global zero3 flag; None -> placed by .to(target_device) below
    )
    if _dmd_zero3:
        # [ZeRO-3] device_map=None makes T5/VAE load onto CPU; but encode_prompt(device=accelerator.device) below uses
        #   text_encoder **immediately**, far earlier than the .to(target_device) at L~810 -> must move it to the card manually first (mirroring the
        #   immediate placement that ZeRO-2 device_map does), otherwise the embedding index_select hits "cpu and cuda two devices". the .to(target_device)
        #   at L~810 re-places it per data mode (online multi-dataset=GPU -> no-op; offline=CPU -> moved back). the VAE goes to the card too (its forward runs in rollout, also placed by L~810).
        text_encoder = text_encoder.to(accelerator.device)
        vae = vae.to(accelerator.device)
    # For negative prompt
    with torch.no_grad():
        negative_prompt_embeds, _ = encode_prompt(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            prompt=args.data_config.negative_prompt,
            device=accelerator.device,
        )

    transformer_additional_kwargs = {
        "has_multi_term_memory_patch": args.training_config.has_multi_term_memory_patch,
        "zero_history_timestep": args.training_config.zero_history_timestep,
        "restrict_self_attn": args.training_config.restrict_self_attn,
        "guidance_cross_attn": args.training_config.guidance_cross_attn,
        "is_train_restrict_lora": args.training_config.is_train_restrict_lora,
        "restrict_lora": args.training_config.restrict_lora,
        "restrict_lora_rank": args.training_config.restrict_lora_rank,
        "is_amplify_history": args.training_config.is_amplify_history,
        "history_scale_mode": args.training_config.history_scale_mode,
        # Passed through to saved checkpoint so the inference pipeline can apply the same sink slicing.
        "use_raw_sink_frames": args.training_config.use_raw_sink_frames,
    }
    # Inject camera-control kwargs into transformer init (no-op when enabled=False).
    _cam_cfg = getattr(args.model_config, "camera_control", None)
    if _cam_cfg is not None and _cam_cfg.enabled:
        transformer_additional_kwargs.update({
            "enable_cam_control": True,
            "cam_rank": _cam_cfg.cam_rank,
            "cam_ctrl_layers": _cam_cfg.cam_ctrl_layers,
        })
    # GEO additive Plucker switch (independent of camera_control). Builds the shared Plucker encoder and
    # enables the additive noise/warp injection inside the transformer forward.
    _geo_cfg = getattr(args.model_config, "geometric_state", None)
    # GENERATOR (student) plucker build: per-model override. None → use geo_warp_plucker_enabled (legacy).
    # Set False (student=Evoke-Distilled) so NO plucker submodule is built on the generator, while the teacher
    # build (below) + cam-pose data path keep using the config-level geo_warp_plucker_enabled.
    _gen_plk_override = getattr(_geo_cfg, "generator_geo_warp_plucker_enabled", None)
    _gen_plk = bool(getattr(_geo_cfg, "geo_warp_plucker_enabled", False)) if _gen_plk_override is None else bool(_gen_plk_override)
    transformer_additional_kwargs["geo_warp_plucker_enabled"] = _gen_plk
    if accelerator.is_main_process:
        print(f"[plucker] generator={_gen_plk} teacher/critic={bool(getattr(_geo_cfg, 'geo_warp_plucker_enabled', False))}", flush=True)
    transformer = EvokeTransformer3DModel.from_pretrained(
        args.model_config.transformer_model_name_or_path,
        subfolder=args.model_config.subfolder or "transformer",
        transformer_additional_kwargs=transformer_additional_kwargs,
    )

    # Optionally attach a warp residual MLP to the transformer (zero-init, identity at startup).
    _geo_cfg_for_mlp = getattr(args.model_config, "geometric_state", None)
    if _geo_cfg_for_mlp is not None and bool(getattr(_geo_cfg_for_mlp, "geo_warp_residual_mlp_enabled", False)):
        _inner_dim = int(transformer.config.num_attention_heads) * int(transformer.config.attention_head_dim)
        _hidden_mult = float(getattr(_geo_cfg_for_mlp, "geo_warp_residual_mlp_hidden_mult", 2.0))
        _hidden_dim = int(_inner_dim * _hidden_mult)
        _warp_mlp = torch.nn.Sequential(
            torch.nn.Linear(_inner_dim, _hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(_hidden_dim, _inner_dim),
        )
        # Zero-init last layer so the MLP starts as identity.
        torch.nn.init.zeros_(_warp_mlp[-1].weight)
        torch.nn.init.zeros_(_warp_mlp[-1].bias)
        transformer.warp_residual_mlp = _warp_mlp
        _wmlp_params = sum(p.numel() for p in _warp_mlp.parameters())
        if accelerator.is_main_process:
            print(f"[GEO-MLP] Created warp_residual_mlp: inner_dim={_inner_dim}, hidden_dim={_hidden_dim}, "
                  f"params={_wmlp_params/1e6:.2f}M (zero-init)", flush=True)

    # Initialize camera-control modules after pretrained load, before LoRA adapter creation.
    if _cam_cfg is not None and _cam_cfg.enabled:
        from evoke.modules.camera_control import reinit_cam_modules
        reinit_cam_modules(
            transformer,
            num_layers=transformer.config.num_layers,
            cam_weight_path=_cam_cfg.cam_ckpt_path,
            cam_ctrl_layers=_cam_cfg.cam_ctrl_layers,
            strict=_cam_cfg.strict_camera_ckpt,
        )
        # Sanity-check plucker encoder weights; warn if both layer1 and layer2 are all-zero.
        if accelerator.is_main_process:
            _l1 = getattr(transformer, "c2ws_hidden_states_layer1", None)
            _l2 = getattr(transformer, "c2ws_hidden_states_layer2", None)
            _pe = getattr(transformer, "patch_embedding_wancamctrl", None)
            if _l1 is not None and _l2 is not None and _pe is not None:
                _l1_max = float(_l1.weight.detach().abs().max())
                _l2_max = float(_l2.weight.detach().abs().max())
                _pe_max = float(_pe.weight.detach().abs().max())
                print(f"[CamCtrl SANITY] patch_emb.w abs_max={_pe_max:.3e}  "
                      f"c2ws_layer1.w abs_max={_l1_max:.3e}  c2ws_layer2.w abs_max={_l2_max:.3e}")
                if _l1_max < 1e-8 and _l2_max < 1e-8:
                    print(f"[CamCtrl SANITY] WARNING: both layer1 + layer2 weights are zero → plucker encoder deadlock!")
                    print(f"[CamCtrl SANITY]   Fix: use a fresh output_dir; check cam_ckpt_path.")

    transformer = replace_rmsnorm_with_fp32(transformer)
    transformer = replace_all_norms_with_flash_norms(transformer)
    replace_rope_with_flash_rope()

    # load dmd real score model
    if args.training_config.is_train_dmd:
        if args.model_config.real_score_model_name_or_path is None:
            args.model_config.real_score_model_name_or_path = args.model_config.transformer_model_name_or_path
        # DMD per-model restrict_self_attn: teacher(real_score)/critic may differ from generator (student).
        # None → fall back to generator's restrict_self_attn (legacy, both equal).
        _real_score_restrict = args.training_config.real_score_restrict_self_attn
        if _real_score_restrict is None:
            _real_score_restrict = args.training_config.restrict_self_attn
        if accelerator.is_main_process:
            print(f"[restrict] generator={args.training_config.restrict_self_attn} "
                  f"real_score/critic={_real_score_restrict}", flush=True)
        critic_transformer_additional_kwargs = {
            "has_multi_term_memory_patch": args.training_config.has_multi_term_memory_patch,
            "zero_history_timestep": args.training_config.zero_history_timestep,
            "restrict_self_attn": _real_score_restrict,
            "guidance_cross_attn": args.training_config.guidance_cross_attn,
            "is_train_restrict_lora": args.training_config.is_train_restrict_lora,
            "restrict_lora": args.training_config.restrict_lora,
            "restrict_lora_rank": args.training_config.restrict_lora_rank,
            "is_use_gan": args.training_config.is_use_gan,
            "is_use_gan_hooks": args.training_config.is_use_gan_hooks,
            "is_use_gan_final": args.training_config.is_use_gan_final,
            "gan_cond_map_dim": args.training_config.gan_cond_map_dim,
            "gan_hooks": args.training_config.gan_hooks,
            # teacher(real_score)+critic(fake_score) are stage1 full-res, trained WITH plk.
            # Build their Plucker submodules so the stage1 ckpt's plk weights load (else silently dropped).
            "geo_warp_plucker_enabled": bool(getattr(_geo_cfg, "geo_warp_plucker_enabled", False)),
        }

        if _sf_evoke_teacher:
            # real/fake score backbone = EvokeTeacher sparse teacher (dual expert + critic-LoRA live inside the wrapper),
            # skip the evoke-specific norm replacement (the wrapper carries EvokeTeacher's own RMSNorm).
            from evoke.modules.evoke_teacher import EvokeTeacherScoreWrapper
            _et_cfg = args.model_config.evoke_teacher
            # push the two sparse keys from the teacher's own training config into the construction kwargs: the shipped
            #   weights are cs9/select1 while loader.py's defaults are cs8/select4. Neither param changes param shape, so
            #   a strict=False load reports missing=0 and the mismatch is silent.
            #   Explicit model_cfg_overrides win (same-name keys are not overwritten).
            _et_ovr = dict(_et_cfg.model_cfg_overrides) if _et_cfg.model_cfg_overrides else {}
            for _et_k in ("chunk_size", "num_select_frames"):
                _et_v = getattr(_et_cfg, _et_k, None)
                if _et_v is not None and _et_k not in _et_ovr:
                    _et_ovr[_et_k] = int(_et_v)
            real_score_model = EvokeTeacherScoreWrapper(
                high_dir=_et_cfg.high_dir,
                low_dir=_et_cfg.low_dir,
                boundary=float(_et_cfg.boundary),
                model_cfg_overrides=(_et_ovr or None),
                torch_dtype=torch.bfloat16,
                critic_lora_rank=args.model_config.critic_lora_rank,
                critic_lora_alpha=args.model_config.critic_lora_alpha,
                critic_lora_dropout=args.model_config.critic_lora_dropout,
                single_expert=getattr(_et_cfg, "single_expert", None),
            )
            # NOTE: print the sparse params that are **actually in effect after construction** (not what the yaml says).
            #   motivation: chunk_size used to be a dead key and ran a whole formal at 8 with nobody able to see it. from now on any
            #   "configured but not in effect" problem of this kind shows up on this line.
            if accelerator.is_main_process:
                _et_b0 = (real_score_model.dit_low or real_score_model.dit_high).blocks[0]
                print(f"[LW-ALIGN] teacher sparse EFFECTIVE: chunk_size={_et_b0.chunk_size} "
                      f"num_select_frames={_et_b0.num_select_frames} "
                      f"num_nearby_frames={_et_b0.num_nearby_frames} "
                      f"overlap_size={_et_b0.overlap_size} per_frame_tokens={_et_b0.per_frame_tokens} "
                      f"select_scales={list(_et_b0.select_scales)} "
                      f"select_gate_mode={_et_b0.select_gate_mode} sink_decay_mode={_et_b0.sink_decay_mode} "
                      f"| boundary_t={real_score_model.boundary_t:.0f}", flush=True)
            assert int(_et_ovr.get("chunk_size", -1)) == int(
                (real_score_model.dit_low or real_score_model.dit_high).blocks[0].chunk_size), \
                "[LW-ALIGN] chunk_size did not reach the model (model_cfg_overrides channel broken?)"
            # SP: set sp_enabled on both experts' dit (_forward_core uses it to split activations along the frame dim).
            #   G=1 -> not set -> byte-identical. set at construction time; after prepare the wrapper still accesses the same dit object.
            if _sp_world_size > 1:
                for _sp_e in (real_score_model.dit_high, real_score_model.dit_low):
                    if _sp_e is not None:
                        _sp_e.sp_enabled = True
                if accelerator.is_main_process:
                    print(f"[SP] EvokeTeacher critic dit sp_enabled=True (G={_sp_world_size})", flush=True)
            # set when dual expert (single_expert=None) + offload: the scoring forward keeps only
            #   the **routed** expert base (28GB) resident on GPU and leaves the other on CPU -> peak = single-expert level (otherwise both experts
            #   sit resident at 28GB each, +28GB, and hit the 141G H200 wall). single expert / no offload -> False -> old whole-wrapper offload path, byte-id.
            real_score_model._per_expert_offload = (
                getattr(_et_cfg, "single_expert", None) is None and bool(_et_offload))
            if real_score_model._per_expert_offload and accelerator.is_main_process:
                print(f"[DUAL-EXPERT] per-expert offload ON: scoring keeps only the routed expert on GPU "
                      f"(t>=boundary {real_score_model.boundary_t:.0f} -> high, else low)", flush=True)
            # share the host-offloaded frozen expert base within a node (28GB/rank -> one copy per node): otherwise every
            #   rank keeps a byte-identical read-only copy of the non-routed expert and a 56-card run trips the host
            #   OOM-killer. LOCAL_RANK0 writes the base into /dev/shm once; each rank mmap-attaches and frees its own copy,
            #   so every later swap-out is p.data pointing back at the shared tensor. Off (default) -> no-op.
            if bool(getattr(args.training_config, "sf_evoke_teacher_shared_host_base", False)):
                assert real_score_model._per_expert_offload, (
                    "[SHARED-BASE] sf_evoke_teacher_shared_host_base requires per-expert offload (dual expert + offload)")
                from evoke.modules.evoke_teacher.shared_host_base import publish_and_attach
                _sb_local_rank = int(os.environ.get("LOCAL_RANK", 0))
                _sb_barrier = (torch.distributed.barrier
                               if torch.distributed.is_available() and torch.distributed.is_initialized()
                               else None)
                for _sb_tag, _sb_dit, _sb_dir in (
                    ("et_high", real_score_model.dit_high, _et_cfg.high_dir),
                    ("et_low", real_score_model.dit_low, _et_cfg.low_dir),
                ):
                    if _sb_dit is None:
                        continue
                    publish_and_attach(_sb_dit, tag=_sb_tag, extra_id=str(_sb_dir),
                                       local_rank=_sb_local_rank, barrier=_sb_barrier,
                                       verbose=accelerator.is_local_main_process)
        else:
            real_score_model = EvokeTransformer3DModel.from_pretrained(
                args.model_config.real_score_model_name_or_path,
                subfolder=args.model_config.critic_subfolder or "transformer",
                transformer_additional_kwargs=critic_transformer_additional_kwargs,
            )
            real_score_model = replace_rmsnorm_with_fp32(real_score_model)
            real_score_model = replace_all_norms_with_flash_norms(real_score_model)

        # second real-score teacher (Evoke-Base pose, camera force): frozen inference module, no adapter. Separate
        # forward inside compute_kl_grad, convex-weighted with the evoke_teacher score (see utils_evoke_post).
        # None when dual is off -> single-teacher path, bit-identical.
        # NOTE: construction is deferred until transformer + EvokeTeacher are already on the card. Loading here would
        #   hold all three (28 + 56 + 28 GB/rank) in CPU at once and hit the host-RAM cgroup; deferred, the CPU peak
        #   stays at transformer+EvokeTeacher = 84GB/rank.
        real_score_model_hb = None

    # We only train the additional adapter LoRA layers
    transformer.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    vae.eval()
    text_encoder.eval()
    if args.training_config.is_train_dmd and not _sf_evoke_teacher:
        real_score_model.requires_grad_(False)  # evoke_teacher: the wrapper already froze the base and opened up the LoRA at construction

    # Camera-only mode: skip LoRA adapter creation entirely.
    _cam_only_mode = (
        _cam_cfg is not None
        and _cam_cfg.enabled
        and (
            _cam_cfg.train_only_camera
            or set(args.training_config.trainable_models or []) == {"cam"}
        )
    )
    # GEO-only mode: trainable_models contains only "geo_lora"; skip main LoRA.
    _geo_only_mode = set(args.training_config.trainable_models or []) == {"geo_lora"}
    # Full SFT mode: trainable_models contains "full" — full fine-tune the transformer, no LoRA adapter.
    _full_finetune_mode = "full" in set(args.training_config.trainable_models or [])
    # DiT-freeze mode: trainable_models given but does not contain "lora"; skip main LoRA.
    _no_main_lora_mode = (
        bool(args.training_config.trainable_models)
        and "lora" not in set(args.training_config.trainable_models)
        and not _cam_only_mode
        and not _geo_only_mode
        and not _full_finetune_mode
    )

    if _cam_only_mode:
        print("[CamCtrl] camera-only training: skip LoRA adapter creation.")
    elif _geo_only_mode:
        print("[GEO] geo_lora-only training: skip main LoRA adapter creation (GEO LoRA is added separately).")
    elif _full_finetune_mode:
        # Full SFT: unfreeze the entire transformer, no LoRA adapter.
        transformer.requires_grad_(True)
        # Keep the input patch_embedding frozen unless is_train_full_patch_embedding=True
        # (preserve the pretrained input projection during full SFT).
        if not args.training_config.is_train_full_patch_embedding:
            _n_pe_frozen = 0
            for _pe_name, _pe_param in transformer.named_parameters():
                if _pe_name.startswith("patch_embedding."):
                    _pe_param.requires_grad = False
                    _n_pe_frozen += 1
            print(f"[Train] full SFT: patch_embedding frozen ({_n_pe_frozen} params).")
        print("[Train] full SFT: all transformer params trainable (no LoRA adapter).")
    elif _no_main_lora_mode:
        print(f"[Train] trainable_models={args.training_config.trainable_models} does not contain 'lora' "
              f"-> skip the main LoRA adapter (DiT body frozen).")
    elif args.model_config.lora_layers is not None:
        if args.model_config.lora_layers != "all-linear":
            target_modules = [layer.strip() for layer in args.model_config.lora_layers.split(",")]
            if args.training_config.is_train_lora_patch_embedding and "patch_embedding" not in target_modules:
                target_modules.append("patch_embedding")

            if args.training_config.is_train_lora_multi_term_memory_patchg:
                for patch_name in ["patch_short", "patch_mid", "patch_long"]:
                    if patch_name not in target_modules:
                        target_modules.append(patch_name)
        elif args.model_config.lora_layers == "all-linear":
            target_modules = set()
            for name, module in transformer.named_modules():
                if isinstance(module, torch.nn.Linear):
                    # Exclude warp_residual_mlp so it is trained full rather than wrapped by PEFT.
                    if "warp_residual_mlp" in name:
                        continue
                    target_modules.add(name)
            target_modules = list(target_modules)
            if args.training_config.is_train_lora_patch_embedding and "patch_embedding" not in target_modules:
                target_modules.append("patch_embedding")

            if args.training_config.is_train_lora_multi_term_memory_patchg:
                for patch_name in ["patch_short", "patch_mid", "patch_long"]:
                    if patch_name not in target_modules:
                        target_modules.append(patch_name)
        # Exclude norm layers and cam modules from main LoRA target list.
        target_modules = [t for t in target_modules if "norm" not in t]
        if _cam_cfg is not None and _cam_cfg.enabled:
            from evoke.modules.camera_control import is_cam_param_name as _is_cam
            target_modules = [t for t in target_modules if not _is_cam(t)]
    else:
        target_modules = args.model_config.lora_target_modules

    # Attach main LoRA adapter unless a special training mode skips it (incl. full SFT).
    if not _cam_only_mode and not _geo_only_mode and not _no_main_lora_mode and not _full_finetune_mode:
        transformer_lora_config = LoraConfig(
            r=args.model_config.lora_rank,
            lora_alpha=args.model_config.lora_alpha,
            lora_dropout=args.model_config.lora_dropout,
            init_lora_weights="gaussian",
            target_modules=list(target_modules),
            exclude_modules=list(args.model_config.lora_exclude_modules),
        )
        transformer.add_adapter(transformer_lora_config)

    if args.model_config.train_norm_layers:
        for name, param in transformer.named_parameters():
            if any(k in name for k in NORM_LAYER_PREFIXES):
                param.requires_grad = True

    # Attach a separate GEO LoRA adapter when "geo_lora" is in trainable_models.
    _geo_cfg = getattr(args.model_config, "geometric_state", None)
    _geo_lora_enabled = (
        _geo_cfg is not None
        and bool(getattr(_geo_cfg, "enabled", False))
        and "geo_lora" in set(args.training_config.trainable_models or [])
    )
    if _geo_lora_enabled:
        from peft import LoraConfig as _LoraConfig_GEO
        _geo_target = str(getattr(_geo_cfg, "lora_target_modules", "to_q,to_k,to_v")).split(",")
        _geo_target = [t.strip() for t in _geo_target if t.strip()]
        geo_lora_config = _LoraConfig_GEO(
            r=int(getattr(_geo_cfg, "lora_rank", 1)),
            lora_alpha=int(getattr(_geo_cfg, "lora_alpha", 1)),
            lora_dropout=float(getattr(_geo_cfg, "lora_dropout", 0.0)),
            init_lora_weights=True,
            target_modules=_geo_target,
        )
        transformer.add_adapter(geo_lora_config, adapter_name="geo")
        # Re-activate both adapters; add_adapter deactivates the previously active one.
        _pcfg = getattr(transformer, "peft_config", None) or {}
        if "default" in _pcfg:
            transformer.set_adapter(["default", "geo"])
            if accelerator.is_main_process:
                print(f"[GEO] active_adapters → ['default', 'geo']")
        else:
            if accelerator.is_main_process:
                print(f"[GEO] active_adapters → ['geo'] only (no main LoRA)")
        if accelerator.is_main_process:
            print(f"[GEO] LoRA adapter 'geo' attached: rank={geo_lora_config.r}, target={_geo_target}")

    trainable_modules = []
    _train_tags = set(args.training_config.trainable_models or [])
    # Add cam module names when jointly training cam + LoRA (cam-only uses a dedicated path).
    if "cam" in _train_tags and _cam_cfg is not None and _cam_cfg.enabled and not _cam_only_mode:
        from evoke.modules.camera_control import CAM_PARAM_TOKENS as _CAM_TOKENS
        trainable_modules.extend(list(_CAM_TOKENS))
    # Legacy per-flag trainable module additions (used when trainable_models is empty or as additions).
    if args.training_config.is_train_full_multi_term_memory_patchg:
        trainable_modules.extend(["patch_short", "patch_mid", "patch_long"])
    if args.training_config.is_train_full_patch_embedding:
        trainable_modules.append("patch_embedding")
    if args.training_config.is_train_restrict_lora:
        trainable_modules.extend(["q_loras", "k_loras", "v_loras"])
    if args.training_config.is_amplify_history:
        trainable_modules.append("history_key_scale")
    # warp_residual_mlp is full-tuned (not wrapped by LoRA).
    if _geo_cfg_for_mlp is not None and bool(getattr(_geo_cfg_for_mlp, "geo_warp_residual_mlp_enabled", False)):
        trainable_modules.append("warp_residual_mlp")
    for name, param in transformer.named_parameters():
        for trainable_module_name in trainable_modules:
            if trainable_module_name in name:
                param.requires_grad = True
                break

    # Camera-only: freeze everything except cam parameters.
    if _cam_only_mode:
        from evoke.modules.camera_control import set_camera_only_trainable
        set_camera_only_trainable(transformer, verbose=accelerator.is_main_process)
        from evoke.modules.camera_control import is_cam_param_name as _is_cam_assert
        _leaked = [n for n, p in transformer.named_parameters() if p.requires_grad and not _is_cam_assert(n)]
        assert not _leaked, (
            f"[CamCtrl] camera-only training: non-cam params leaked into trainable set ({len(_leaked)}): "
            f"{_leaked[:5]}"
        )

    if args.training_config.use_ema:
        model_cls = EvokeTransformer3DModel
        transformer_cpu = copy.deepcopy(transformer)
        with open(args.training_config.ema_deepspeed_config_file, "r") as f:
            ds_config = json.load(f)

    # get fake score model
    if args.training_config.is_train_dmd and _sf_evoke_teacher:
        # the critic LoRA was already injected into both experts when EvokeTeacherScoreWrapper was built; nothing to do here.
        pass
    elif args.training_config.is_train_dmd:
        critic_target_modules = [
            m for m in target_modules if m not in ["patch_short", "patch_mid", "patch_long", "patch_embedding"]
        ]
        critic_exclude_modules = list(args.model_config.lora_exclude_modules) + [
            "patch_short",
            "patch_mid",
            "patch_long",
            "patch_embedding",
            "gan_heads",
            "gan_final_head",
        ]
        critic_transformer_lora_config = LoraConfig(
            r=args.model_config.critic_lora_rank,
            lora_alpha=args.model_config.critic_lora_alpha,
            lora_dropout=args.model_config.critic_lora_dropout,
            init_lora_weights="gaussian",
            target_modules=critic_target_modules,
            exclude_modules=critic_exclude_modules,
        )

        real_score_model.add_adapter(critic_transformer_lora_config)

        if args.model_config.train_norm_layers:
            for name, param in real_score_model.named_parameters():
                if any(k in name for k in NORM_LAYER_PREFIXES):
                    param.requires_grad = True

        if args.training_config.is_use_gan:
            critic_trainable_modules = ["gan_heads", "gan_final_head"]
            for name, param in real_score_model.named_parameters():
                for trainable_module_name in critic_trainable_modules:
                    if trainable_module_name in name:
                        param.requires_grad = True
                        break

    if args.model_config.load_checkpoints_custom:
        load_model_checkpoint(
            args=args,
            checkpoint_path=args.model_config.load_model_path,
            transformer=transformer,
            pipeline_class=EvokePipeline,
            norm_layer_prefixes=NORM_LAYER_PREFIXES,
            convert_unet_state_dict_to_peft_fn=convert_unet_state_dict_to_peft,
            set_peft_model_state_dict_fn=set_peft_model_state_dict,
            cast_training_params_fn=cast_training_params,
        )
        if args.training_config.is_train_dmd:
            assert args.model_config.critic_lora_name_or_path is not None
            assert args.model_config.load_dcp

    if args.model_config.critic_lora_name_or_path is not None:
        load_model_checkpoint(
            args=args,
            checkpoint_path=args.model_config.critic_lora_name_or_path,
            transformer=real_score_model,
            pipeline_class=EvokePipeline,
            norm_layer_prefixes=NORM_LAYER_PREFIXES,
            convert_unet_state_dict_to_peft_fn=convert_unet_state_dict_to_peft,
            set_peft_model_state_dict_fn=set_peft_model_state_dict,
            cast_training_params_fn=cast_training_params,
        )

    # -- NOTE: weight-level warm-start (must happen before accelerator.prepare) --
    #   restores the three sets of weights from a save_checkpoints_custom artifact: generator LoRA / memory patch / **critic LoRA**.
    #   optimizer/RNG/scheduler are not restored (that ckpt simply does not contain them) => no conflict with accelerate resume, and no need to relax
    #   SF10S's `resume_from_checkpoint is None` assert. any of the three missing is fail-fast, never a silent half-inherit.
    _ws_dir = getattr(args.training_config, "sf_warmstart_dir", None)
    _ws_critic_only = bool(getattr(args.training_config, "sf_warmstart_critic_only", False))
    if _ws_dir:
        assert os.path.isdir(_ws_dir), f"[LW-WARMSTART] directory does not exist: {_ws_dir}"
        _ws_ngen = _ws_nmem = 0
        if not _ws_critic_only:
            # (1) generator LoRA -- same key transform as load_model_hook.
            #   weight_name must be passed explicitly: training jobs carry HF_HUB_OFFLINE=1, and diffusers'
            #   _best_guess_weight_name raises under offline mode. load_model_hook omits it because it only runs on the
            #   accelerate resume path, which SF10S forbids.
            _ws_lora_f = os.path.join(_ws_dir, "pytorch_lora_weights.safetensors")
            if not os.path.exists(_ws_lora_f):
                _ws_lora_f = os.path.join(_ws_dir, "weights", "lora.safetensors")
            assert os.path.exists(_ws_lora_f), f"[LW-WARMSTART] generator LoRA not found: {_ws_dir}"
            _ws_sd = EvokePipeline.lora_state_dict(
                os.path.dirname(_ws_lora_f), weight_name=os.path.basename(_ws_lora_f))
            _ws_tsd = {k.replace("transformer.", "", 1): v for k, v in _ws_sd.items() if k.startswith("transformer.")}
            assert _ws_tsd, f"[LW-WARMSTART] {_ws_lora_f} contains no LoRA key with a transformer.* prefix"
            _ws_tsd = convert_unet_state_dict_to_peft(_ws_tsd)
            _ws_inc = set_peft_model_state_dict(transformer, _ws_tsd, adapter_name="default")
            _ws_unexp = list(getattr(_ws_inc, "unexpected_keys", None) or []) if _ws_inc is not None else []
            assert not _ws_unexp, f"[LW-WARMSTART] generator LoRA has {len(_ws_unexp)} unexpected key(s): {_ws_unexp[:5]}"
            _ws_ngen = len(_ws_tsd)
            del _ws_sd, _ws_tsd
            # (2) memory patch (the trainable params of is_train_lora_multi_term_memory_patchg)
            _ws_extra = os.path.join(_ws_dir, "transformer_partial.pth")
            if not os.path.exists(_ws_extra):
                _ws_extra = os.path.join(_ws_dir, "weights", "memory.pth")
            assert os.path.exists(_ws_extra), (
                f"[LW-WARMSTART] memory patch not found (transformer_partial.pth / weights/memory.pth): {_ws_dir}")
            load_extra_components(args, transformer, _ws_extra)
            _ws_nmem = 1
        # (3) critic LoRA (the EvokeTeacher wrapper's trainable_state_dict on disk; not part of the merged generator weights)
        _ws_ncri = 0
        _ws_critic = os.path.join(_ws_dir, "critic", "critic_evoke_teacher_lora.safetensors")
        if _sf_evoke_teacher:
            assert os.path.exists(_ws_critic), (
                f"[LW-WARMSTART] the evoke_teacher path must be able to restore the critic LoRA, but {_ws_critic} was not found. "
                f"without it the critic starts from scratch -> fake-score first hands the student a stretch of wrong gradient (exactly what we want to avoid)")
            from safetensors.torch import load_file as _ws_load_file
            _ws_csd = _ws_load_file(_ws_critic)
            # we are before accelerator.prepare here => real_score_model is still a bare module, so unwrap_model is not needed (and not usable)
            _ws_tgt = set(real_score_model.trainable_state_dict().keys())
            _ws_miss = sorted(_ws_tgt - set(_ws_csd.keys()))
            assert not _ws_miss, (
                f"[LW-WARMSTART] critic LoRA is missing {len(_ws_miss)}/{len(_ws_tgt)} key(s) (critic_lora_rank/target changed?):"
                f" {_ws_miss[:5]}")
            _ws_ret = real_score_model.load_state_dict(_ws_csd, strict=False)
            assert not list(getattr(_ws_ret, "unexpected_keys", []) or []), \
                f"[LW-WARMSTART] critic LoRA has unexpected key(s): {list(_ws_ret.unexpected_keys)[:5]}"
            _ws_ncri = len(_ws_tgt)
            del _ws_csd
        if accelerator.is_main_process:
            print(f"[LW-WARMSTART] weights restored (no optimizer/RNG/scheduler): dir={_ws_dir}\n"
                  f"  critic_only={_ws_critic_only}"
                  f"{'  => generator and memory patch come from the merged weights of transformer_model_name_or_path' if _ws_critic_only else ''}\n"
                  f"  (1) generator LoRA: {_ws_ngen if not _ws_critic_only else 'skipped (already folded into merged)'}\n"
                  f"  (2) memory patch: {'restored' if _ws_nmem else 'skipped (merged already contains patch_short/mid/long)'}\n"
                  f"  (3) critic LoRA: {_ws_ncri} tensor(s)\n"
                  f"  WARNING: Adam momentum is zeroed => pair with sf_gen_freeze_steps"
                  f"(currently = {int(getattr(args.training_config, 'sf_gen_freeze_steps', 0) or 0)} steps)", flush=True)

    if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    # For offline datasets move VAE/T5 to CPU to save VRAM; online mode keeps them on GPU.
    _data_uses_cpu_models = (
        (args.data_config.use_stage1_dataset or args.data_config.use_stage3_dataset)
        and not args.data_config.use_multi_dataset
    )
    target_device = "cpu" if _data_uses_cpu_models else accelerator.device
    vae.to(target_device)
    text_encoder.to(target_device)
    if args.training_config.is_use_reward_model:
        reward_model.model.to(target_device)
    free_memory()

    # Cast transformer parameters and move to accelerator device.
    for name, param in transformer.named_parameters():
        should_keep_fp32 = any(pattern in name for pattern in transformer.__class__._keep_in_fp32_modules)
        if should_keep_fp32:
            param.data = param.data.to(torch.float32)
        else:
            param.data = param.data.to(weight_dtype)
    transformer.to(accelerator.device)

    if args.training_config.is_train_dmd:
        for name, param in real_score_model.named_parameters():
            should_keep_fp32 = any(pattern in name for pattern in real_score_model.__class__._keep_in_fp32_modules)
            if should_keep_fp32:
                param.data = param.data.to(torch.float32)
            else:
                param.data = param.data.to(weight_dtype)
        real_score_model.to(accelerator.device)
        # construction of the second teacher is deferred to here: transformer + EvokeTeacher are both already on the card (above), so now
        #   from_pretrained'ing Evoke onto CPU -> the instantaneous CPU peak is only = Evoke 28GB (the rest is on GPU), and the CPU peak over the whole load = the
        #   84GB/rank of the transformer+EvokeTeacher window above (Evoke does not stack on top), avoiding three models resident in CPU hitting the host cgroup (smoke2/4 SIGKILL).
        if _sf_dual:
            _dt_cfg = args.model_config.dual_teacher
            assert _dt_cfg.evoke_model_path, "[DUAL-TEACHER] dual_teacher.evoke_model_path is not set"
            real_score_model_hb = EvokeTransformer3DModel.from_pretrained(
                _dt_cfg.evoke_model_path,
                subfolder=(_dt_cfg.evoke_subfolder or "transformer"),
                transformer_additional_kwargs=critic_transformer_additional_kwargs,
            )
            real_score_model_hb = replace_rmsnorm_with_fp32(real_score_model_hb)
            real_score_model_hb = replace_all_norms_with_flash_norms(real_score_model_hb)
            # critic-LoRA dedicated to the Evoke backbone: adapters-off=teacher (s_hb) / adapters-on=critic
            #   (s_fake_hb) -> grad_hb = s_fake_hb(warp) - s_hb(warp), a self-consistent camera force. target/exclude reuse the construction
            #   rules of the existing critic (see L712-722; but _sf_dual implies _sf_evoke_teacher -> that elif never runs, so they are rebuilt here by the same rules); rank/alpha/dropout
            #   come from dual_teacher.evoke_critic_lora_*. NOTE: freeze base first, then add_adapter (EvokeTransformer3DModel mixes in
            #   PeftAdapterMixin -> add/disable/enable_adapters are available); afterwards do **not** blanket-freeze again (that would freeze the LoRA).
            _hb_critic_target_modules = [
                m for m in target_modules if m not in ["patch_short", "patch_mid", "patch_long", "patch_embedding"]
            ]
            _hb_critic_exclude_modules = list(args.model_config.lora_exclude_modules) + [
                "patch_short",
                "patch_mid",
                "patch_long",
                "patch_embedding",
                "gan_heads",
                "gan_final_head",
                # the camera Plucker encoder (6 keys) must stay a frozen base with no critic-LoRA attached:
                #   teacher (adapters-off) and critic (adapters-on) must share the same frozen plucker -> grad_hb stays self-consistent.
                #   when the generator plucker is off, target_modules does not contain them anyway (no-op); this is insurance against flipping it on.
                "patch_embedding_wancamctrl",
                "c2ws_hidden_states_layer1",
                "c2ws_hidden_states_layer2",
            ]
            evoke_critic_lora_config = LoraConfig(
                r=_dt_cfg.evoke_critic_lora_rank,
                lora_alpha=_dt_cfg.evoke_critic_lora_alpha,
                lora_dropout=_dt_cfg.evoke_critic_lora_dropout,
                init_lora_weights="gaussian",
                target_modules=_hb_critic_target_modules,
                exclude_modules=_hb_critic_exclude_modules,
            )
            real_score_model_hb.requires_grad_(False)
            real_score_model_hb.add_adapter(evoke_critic_lora_config)
            real_score_model_hb.eval()
            # confirm the 6 global Plucker keys really loaded (non-zero). max~=0 -> silently dropped/never built = zero camera force (a fatal silent failure).
            if accelerator.is_main_process:
                _plk_names, _plk_max = [], 0.0
                for _pn, _pp in real_score_model_hb.named_parameters():
                    if ("patch_embedding_wancamctrl" in _pn) or ("c2ws_hidden_states_layer" in _pn):
                        _plk_names.append(_pn)
                        _plk_max = max(_plk_max, float(_pp.detach().float().abs().max().item()))
                print(f"[CAMERA-TEACHER] hb teacher plucker: keys={len(_plk_names)} max|w|={_plk_max:.4g} "
                      f"(expect keys=6 and max>0 -> camera weights really loaded; if max~=0 -> dropped/never built, stop!)", flush=True)
        # same dtype for the second teacher. with offload on it initially stays on CPU (compute_kl_grad swaps it in/out on demand,
        #   so the generate/backward phase does not carry the extra 28G); with offload off it goes straight to the card (single node with lots of VRAM, old behavior).
        if real_score_model_hb is not None:
            for name, param in real_score_model_hb.named_parameters():
                _keep_fp32 = any(p in name for p in real_score_model_hb.__class__._keep_in_fp32_modules)
                param.data = param.data.to(torch.float32 if _keep_fp32 else weight_dtype)
            real_score_model_hb.to("cpu" if _dual_offload else accelerator.device)
    free_memory()

    if args.training_config.enable_npu_flash_attention:
        if is_torch_npu_available():
            accelerator.print("npu flash attention enabled.")
            transformer.enable_npu_flash_attention()
            if args.training_config.is_train_dmd:
                real_score_model.enable_npu_flash_attention()
        else:
            raise ValueError("npu flash attention requires torch_npu extensions and is supported only on npu devices.")

    if args.training_config.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            transformer.enable_xformers_memory_efficient_attention()
            if args.training_config.is_train_dmd:
                real_score_model.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if args.training_config.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
        if args.training_config.is_train_dmd:
            real_score_model.enable_gradient_checkpointing()
            if real_score_model_hb is not None:
                real_score_model_hb.enable_gradient_checkpointing()

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # under ZeRO-3 the LoRA/trainable params are sharded, so a plain rank0 read = a partial shard.
    #   deepspeed.zero.GatheredParameters is a collective -> **all ranks** must enter the returned context (all-gather onto every rank),
    #   while only rank0 writes to disk. ZeRO-2: _z3_params_to_fetch returns [] -> enabled=False -> the context is a no-op -> byte-identical to the original
    #   rank0-only save. defined here so periodic saving (inside the training loop) and the final save (after the loop) can share it.
    def _z3_gather_trainable(_m):
        _fetch = _z3_params_to_fetch([p for p in _m.parameters() if p.requires_grad])
        return deepspeed.zero.GatheredParameters(_fetch, enabled=len(_fetch) > 0)

    # Custom hooks to serialize LoRA weights in the expected format.
    def save_model_hook(models, weights, output_dir):
        if _sf_evoke_teacher:
            raise RuntimeError("[SF10S] accelerate save_state does not support the evoke_teacher wrapper, use save_checkpoints_custom")
        if accelerator.is_main_process:
            transformer_lora_layers_to_save = None
            modules_to_save = {}

            for model in models:
                if isinstance(unwrap_model(model), type(unwrap_model(transformer))):
                    model = unwrap_model(model)
                    if not _cam_only_mode:
                        transformer_lora_layers_to_save = get_peft_model_state_dict(model)
                        if args.model_config.train_norm_layers:
                            transformer_norm_layers_to_save = {
                                f"transformer.{name}": param
                                for name, param in model.named_parameters()
                                if any(k in name for k in NORM_LAYER_PREFIXES)
                            }
                            transformer_lora_layers_to_save = {
                                **transformer_lora_layers_to_save,
                                **transformer_norm_layers_to_save,
                            }
                    modules_to_save["transformer"] = model
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")

                # Pop so accelerate does not fall back to its default save path.
                if weights:
                    weights.pop()

            if not _cam_only_mode:
                EvokePipeline.save_lora_weights(
                    output_dir,
                    transformer_lora_layers=transformer_lora_layers_to_save,
                    **_collate_lora_metadata(modules_to_save),
                )

            save_extra_components(args, model=unwrap_model(model), output_dir=output_dir)

    def load_model_hook(models, input_dir):
        if _sf_evoke_teacher:
            raise RuntimeError("[SF10S] accelerate save_state resume does not support the evoke_teacher wrapper (the config validator should have caught this)")
        transformer_ = None

        if not accelerator.distributed_type == DistributedType.DEEPSPEED:
            while len(models) > 0:
                model = models.pop()

                if isinstance(unwrap_model(model), type(unwrap_model(transformer))):
                    model = unwrap_model(model)
                    transformer_ = model
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")
        else:
            transformer_ = EvokeTransformer3DModel.from_pretrained(
                args.model_config.transformer_model_name_or_path,
                subfolder=(
                    args.model_config.critic_subfolder if "critic" in input_dir else args.model_config.subfolder
                )
                or "transformer",
                transformer_additional_kwargs=critic_transformer_additional_kwargs
                if "critic" in input_dir
                else transformer_additional_kwargs,
            )
            transformer_.add_adapter(
                critic_transformer_lora_config if "critic" in input_dir else transformer_lora_config
            )

        lora_state_dict = EvokePipeline.lora_state_dict(input_dir)

        transformer_state_dict = {
            f"{k.replace('transformer.', '')}": v for k, v in lora_state_dict.items() if k.startswith("transformer.")
        }
        transformer_state_dict = convert_unet_state_dict_to_peft(transformer_state_dict)
        incompatible_keys = set_peft_model_state_dict(transformer_, transformer_state_dict, adapter_name="default")
        if incompatible_keys is not None:
            # check only for unexpected keys
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            if unexpected_keys:
                logger.warning(
                    f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                    f" {unexpected_keys}. "
                )

        if args.model_config.train_norm_layers:
            transformer_norm_state_dict = {
                k: v
                for k, v in lora_state_dict.items()
                if k.startswith("transformer.") and any(norm_k in k for norm_k in NORM_LAYER_PREFIXES)
            }
            transformer_._transformer_norm_layers = EvokePipeline._load_norm_into_transformer(
                transformer_norm_state_dict,
                transformer=transformer_,
                discard_original_layers=False,
            )

        load_extra_components(args, transformer_, os.path.join(input_dir, "transformer_partial.pth"))

        # Upcast trainable LoRA parameters to fp32.
        if args.training_config.mixed_precision != "fp32":
            models = [transformer_]
            cast_training_params(models)

        dcp_dir = os.path.join(input_dir, "distributed_checkpoint")
        if "critic" not in dcp_dir:
            states = {
                "dataloader": train_dataloader,
            }
            dcp.load(states, checkpoint_id=dcp_dir)

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    if args.training_config.is_train_dmd:
        critic_accelerator.register_save_state_pre_hook(save_model_hook)
        critic_accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable TF32 for faster training on Ampere GPUs.
    if args.training_config.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.training_config.scale_lr:
        args.training_config.learning_rate = (
            args.training_config.learning_rate
            * args.training_config.gradient_accumulation_steps
            * args.training_config.train_batch_size
            * accelerator.num_processes
        )

        if args.training_config.is_train_dmd:
            args.training_config.critic_learning_rate = (
                args.training_config.critic_learning_rate
                * args.training_config.gradient_accumulation_steps
                * args.training_config.train_batch_size
                * accelerator.num_processes
            )

    # Upcast trainable params to fp32 for mixed-precision training.
    if args.training_config.mixed_precision != "fp32":
        models = [transformer]
        if args.training_config.is_train_dmd:
            models.append(real_score_model)
        # the Evoke-specific critic-LoRA is also trained in fp32 (mirrors the existing critic).
        if _sf_dual and real_score_model_hb is not None:
            models.append(real_score_model_hb)
        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params(models, dtype=torch.float32)

    # Full SFT: re-assert the trainable set immediately before collecting optimizer params, so it is
    # robust to any requires_grad churn during loading / norm-replacement / casting above. Train the
    # whole transformer except patch_embedding (unless is_train_full_patch_embedding=True), then
    # fail fast if most of the model is still frozen (avoids silently wasting a multi-node run).
    if _full_finetune_mode:
        transformer.requires_grad_(True)
        if not args.training_config.is_train_full_patch_embedding:
            for _n, _p in transformer.named_parameters():
                if _n.startswith("patch_embedding."):
                    _p.requires_grad = False
        _n_tr = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
        _n_tot = sum(p.numel() for p in transformer.parameters())
        print(f"[Train] full SFT (pre-optimizer re-assert): trainable={_n_tr/1e9:.3f}B / {_n_tot/1e9:.3f}B", flush=True)
        assert _n_tr > 0.9 * _n_tot, (
            f"full SFT expected ~all params trainable, got {_n_tr}/{_n_tot} ({_n_tr/_n_tot:.1%}); "
            f"most of the model is frozen before optimizer setup."
        )

    # Freeze text cross-attention (attn2) so prompt semantics stay at the base calibration.
    # Runs after the full-SFT re-assert above (which validated ~all params trainable first) and after any
    # checkpoint load. SFT: freeze attn2 base params. LoRA: zero the attn2 LoRA delta (-> base, dropping any
    # warm-started drift) then freeze it. Applies to both modes.
    if getattr(args.training_config, "freeze_cross_attn", False):
        _n_attn2 = 0
        for _n, _p in transformer.named_parameters():
            if ".attn2." in _n:
                if "lora_" in _n:
                    with torch.no_grad():
                        _p.zero_()
                _p.requires_grad = False
                _n_attn2 += 1
        _n_tr_f = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
        print(f"[Train] freeze_cross_attn: attn2 frozen at base ({_n_attn2} params); "
              f"trainable now {_n_tr_f/1e9:.3f}B", flush=True)

    transformer_parameters_with_lr = {
        "params": [p for p in transformer.parameters() if p.requires_grad],
        "lr": args.training_config.learning_rate,
    }
    params_to_optimize = [transformer_parameters_with_lr]
    # NOTE: the **total** trainable param count must be recorded here, before prepare. `Optimizer.add_param_group`
    #   mutates this dict in place and shares the object with `self.param_groups`; DeepSpeed ZeRO-2 then replaces
    #   `param_group['params']` with this rank's fp32 partition, so counting after prepare yields the sharded value
    #   (total/W plus alignment padding), which shrinks as W grows.
    _stu_gen_numel_total = sum(p.numel() for p in transformer_parameters_with_lr["params"])

    use_deepspeed_optimizer = (
        accelerator.state.deepspeed_plugin is not None
        and "optimizer" in accelerator.state.deepspeed_plugin.deepspeed_config
    )
    use_deepspeed_scheduler = (
        accelerator.state.deepspeed_plugin is not None
        and "scheduler" in accelerator.state.deepspeed_plugin.deepspeed_config
    )

    optimizer = get_optimizer(args, accelerator, params_to_optimize, use_deepspeed=use_deepspeed_optimizer)

    if args.training_config.is_train_dmd:
        critic_model_lora_parameters = list(filter(lambda p: p.requires_grad, real_score_model.parameters()))
        critic_model_lr_parameters_with_lr = {
            "params": critic_model_lora_parameters,
            "lr": args.training_config.critic_learning_rate,
        }
        critic_model_params_to_optimize = [critic_model_lr_parameters_with_lr]
        critic_optimizer = get_optimizer(
            args, critic_accelerator, critic_model_params_to_optimize, use_deepspeed=use_deepspeed_optimizer
        )
        # Evoke-specific critic optimizer (mirrors the critic above; separate accelerator).
        #   lr = dual_teacher.evoke_critic_learning_rate (None -> reuse critic_learning_rate).
        if _sf_dual:
            evoke_critic_lora_parameters = [p for p in real_score_model_hb.parameters() if p.requires_grad]
            evoke_critic_lr_parameters_with_lr = {
                "params": evoke_critic_lora_parameters,
                "lr": (args.model_config.dual_teacher.evoke_critic_learning_rate
                       or args.training_config.critic_learning_rate),
            }
            evoke_critic_params_to_optimize = [evoke_critic_lr_parameters_with_lr]
            evoke_critic_optimizer = get_optimizer(
                args, evoke_critic_accelerator, evoke_critic_params_to_optimize,
                use_deepspeed=use_deepspeed_optimizer,
            )

    # Dataset and DataLoaders creation:
    dataset_sampling_ratios = {}
    if args.data_config.dataset_sampling_ratios:
        for temp_key, temp_value in zip(args.data_config.instance_data_root, args.data_config.dataset_sampling_ratios):
            clean_path = temp_key.rstrip("/")
            dataset_sampling_ratios[clean_path] = temp_value

    if args.data_config.use_multi_dataset:
        # Online multi-dataset: raw video and prompt are encoded inline per step.
        assert args.data_config.data_yaml_path, "data_yaml_path must be set when use_multi_dataset=true"
        dataset_kwargs = {
            "data_yaml_path": args.data_config.data_yaml_path,
            "single_height": args.data_config.single_height,
            "single_width": args.data_config.single_width,
            "num_frames": args.data_config.num_frames,
            "target_fps": args.data_config.target_fps,
            "history_sizes": args.training_config.history_sizes,
            "is_keep_x0": True,
            "seed": args.seed,
            # whether the random subsampling driven by ratio is redrawn every epoch (default false = old behavior:
            #   drawn once at construction and locked forever => sources with ratio<1 are frozen into a fixed subset).
            "resample_ratio_each_epoch": bool(
                getattr(args.data_config, "resample_ratio_each_epoch", False)),
        }
    elif args.data_config.use_stage3_dataset:
        dataset_kwargs = {
            "gan_folders": args.data_config.gan_data_root
            if args.training_config.is_use_gan or args.training_config.is_use_gt_history
            else None,
            "ode_folders": args.data_config.ode_data_root if args.training_config.is_use_ode_regression else None,
            "text_folders": args.data_config.text_data_root
            if not args.training_config.is_only_ode_regression
            else None,
            "is_use_gt_history": args.training_config.is_use_gt_history,
            "return_secondary": args.training_config.is_use_gt_history,
            "single_res": args.data_config.single_res,
            "single_length": args.data_config.single_length,
            "single_num_frame": args.data_config.single_num_frame,
            "single_height": args.data_config.single_height,
            "single_width": args.data_config.single_width,
            "force_rebuild": args.data_config.force_rebuild,
            "seed": args.seed,
        }
        assert any(
            [
                dataset_kwargs["gan_folders"],
                dataset_kwargs["ode_folders"],
                dataset_kwargs["text_folders"],
            ]
        ), "Invalid dataset config: at least one of `gan_folders`, `ode_folders`, or `text_folders` must be non-empty."
    elif args.data_config.use_stage1_dataset:
        dataset_kwargs = {
            "feature_folders": args.data_config.instance_data_root,
            "single_res": args.data_config.single_res,
            "single_height": args.data_config.single_height,
            "single_width": args.data_config.single_width,
            "return_prompt_raw": args.training_config.is_use_reward_model,
            "return_all_vae_latent": (
                args.training_config.dmd_teacher_forcing and args.training_config.dmd_teacher_forcing_ratio > 0
            )
            or args.training_config.is_use_gan,
            "history_sizes": args.training_config.history_sizes,
            "is_keep_x0": True,
            "force_rebuild": args.data_config.force_rebuild,
            "seed": args.seed,
        }
    else:
        raise NotImplementedError
        dataset_kwargs = {
            "json_files": args.data_config.instance_data_root,
            "video_folders": args.data_config.instance_video_root,
            "force_rebuild": args.data_config.force_rebuild,
            "stride": args.data_config.stride,
            "resolution": args.data_config.resolution,
            "single_res": args.data_config.single_res,
            "single_length": args.data_config.single_length,
            "single_num_frame": args.data_config.single_num_frame,
            "single_height": args.data_config.single_height,
            "single_width": args.data_config.single_width,
            "multi_res": args.data_config.multi_res,
            "id_token": args.data_config.id_token,
        }

    train_dataset = BucketedFeatureDataset(**dataset_kwargs)

    sampler = BucketedSampler(
        train_dataset,
        batch_size=args.training_config.train_batch_size,
        drop_last=True,  # TODO need to be true now
        shuffle=args.data_config.use_shuffle,
        seed=args.seed,
        dataset_sampling_ratios=dataset_sampling_ratios,
        num_sp_groups=accelerator.num_processes // _sp_world_size,
        sp_world_size=_sp_world_size,
        global_rank=accelerator.process_index,
        decouple_rollout=bool(getattr(args.training_config, "sf_decouple_rollout", False)),
    )

    train_dataloader = StatefulDataLoader(
        train_dataset,
        batch_sampler=sampler,
        pin_memory=args.data_config.pin_memory,
        prefetch_factor=args.data_config.prefetch_factor if args.data_config.prefetch_factor > 0 else None,
        persistent_workers=args.data_config.persistent_workers,
        collate_fn=collate_fn,
        num_workers=args.data_config.dataloader_num_workers,
    )

    if args.model_config.load_dcp:
        if args.model_config.load_dcp_path is not None:
            dcp_dir = os.path.join(args.model_config.load_dcp_path, "distributed_checkpoint")
        else:
            dcp_dir = os.path.join(args.model_config.load_model_path, "distributed_checkpoint")
        states = {
            "dataloader": train_dataloader,
        }
        dcp.load(states, checkpoint_id=dcp_dir)
        print(f"load dcp from {dcp_dir} successfully!")

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.training_config.gradient_accumulation_steps)
    if args.training_config.max_train_steps is None:
        args.training_config.max_train_steps = args.training_config.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    if use_deepspeed_scheduler:
        from accelerate.utils import DummyScheduler

        lr_scheduler = DummyScheduler(
            name=args.training_config.lr_scheduler,
            optimizer=optimizer,
            total_num_steps=args.training_config.max_train_steps * accelerator.num_processes,
            num_warmup_steps=args.training_config.lr_warmup_steps * accelerator.num_processes,
        )

        if args.training_config.is_train_dmd:
            critic_lr_scheduler = DummyScheduler(
                name=args.training_config.lr_scheduler,
                optimizer=critic_optimizer,
                total_num_steps=args.training_config.max_train_steps * accelerator.num_processes,
                num_warmup_steps=args.training_config.lr_warmup_steps * accelerator.num_processes,
            )
            # Evoke critic scheduler (mirror; deepspeed branch).
            if _sf_dual:
                evoke_critic_lr_scheduler = DummyScheduler(
                    name=args.training_config.lr_scheduler,
                    optimizer=evoke_critic_optimizer,
                    total_num_steps=args.training_config.max_train_steps * accelerator.num_processes,
                    num_warmup_steps=args.training_config.lr_warmup_steps * accelerator.num_processes,
                )
    else:
        lr_scheduler = get_scheduler(
            args.training_config.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=args.training_config.lr_warmup_steps * accelerator.num_processes,
            num_training_steps=args.training_config.max_train_steps * accelerator.num_processes,
            num_cycles=args.training_config.lr_num_cycles,
            power=args.training_config.lr_power,
        )

        if args.training_config.is_train_dmd:
            critic_lr_scheduler = get_scheduler(
                args.training_config.lr_scheduler,
                optimizer=critic_optimizer,
                num_warmup_steps=args.training_config.lr_warmup_steps * accelerator.num_processes,
                num_training_steps=args.training_config.max_train_steps * accelerator.num_processes,
                num_cycles=args.training_config.lr_num_cycles,
                power=args.training_config.lr_power,
            )
            # Evoke critic scheduler (mirror; non-deepspeed branch).
            if _sf_dual:
                evoke_critic_lr_scheduler = get_scheduler(
                    args.training_config.lr_scheduler,
                    optimizer=evoke_critic_optimizer,
                    num_warmup_steps=args.training_config.lr_warmup_steps * accelerator.num_processes,
                    num_training_steps=args.training_config.max_train_steps * accelerator.num_processes,
                    num_cycles=args.training_config.lr_num_cycles,
                    power=args.training_config.lr_power,
                )

    # Prepare everything with our `accelerator`.
    accelerator.wait_for_everyone()
    if accelerator.state.deepspeed_plugin is not None:
        accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = (
            args.training_config.train_batch_size
        )
    if args.training_config.is_train_dmd:
        if dmd_deepspeed_training:
            accelerator.state.select_deepspeed_plugin("generator")
        transformer, optimizer, lr_scheduler = accelerator.prepare(transformer, optimizer, lr_scheduler)
        if dmd_deepspeed_training:
            critic_accelerator.state.select_deepspeed_plugin("critic_model")
            critic_accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = (
                args.training_config.train_batch_size
            )
        # under 2D SP (sp<world) the critic ZeRO-3 must shard/all-gather/reduce-scatter over the **DP-stride-G group**
        #   (excluding the opposite-SP ranks) so it never interleaves with the SP subgroup all_to_all -- that is the
        #   forward+backward deadlock. accelerate does not expose mpu, so deepspeed.initialize is temporarily
        #   monkeypatched to inject critic_mpu while the critic is prepared. Preconditions: broadcast the trainable
        #   LoRA so all ranks agree, and save/restore the global groups.mpu (engine.py only ever sets it).
        #   Postcondition: wrap engine.step to all-reduce the grad shards across the SP group. G=1 or sp==world ->
        #   critic_mpu=None -> original path, byte-identical.
        from evoke.modules.evoke_teacher.sp_runtime import is_2d_sp as _is_2d_sp
        _critic_mpu = None
        # the mpu (DP-stride-G ZeRO-3) is only used on the ZeRO-3 path; the ZeRO-2 path (the final answer to the un-staggerable 2D SP x ZeRO-3
        #   backward deadlock) uses EvokeTeacher manual staggering (critic backward below) and injects no mpu.
        if _is_2d_sp() and _dmd_zero3:
            import torch.distributed as _mpu_dist
            import deepspeed as _mpu_ds
            from deepspeed.utils import groups as _ds_groups
            from evoke.modules.evoke_teacher.sp_zero3 import (
                build_critic_mpu as _build_critic_mpu,
                wrap_critic_engine_step as _wrap_critic_step,
            )
            _critic_mpu = _build_critic_mpu()
            # (1) LoRA consistent across all ranks (the base loads deterministically from disk and is already consistent; only the gaussian LoRA can drift -> broadcast trainable params).
            _bcast_n = 0
            for _p in real_score_model.parameters():
                if _p.requires_grad and _p.data.is_cuda:
                    _mpu_dist.broadcast(_p.data, src=0)
                    _bcast_n += 1
            # (2) monkeypatch + save groups.mpu
            _orig_ds_init = _mpu_ds.initialize
            _orig_groups_mpu = getattr(_ds_groups, "mpu", None)

            def _mpu_patched_init(*_a, **_kw):
                assert "mpu" not in _kw or _kw["mpu"] is None, \
                    "[mpu] accelerate unexpectedly brought its own mpu (should not happen: accelerate parallelism_config is not used)"
                _kw["mpu"] = _critic_mpu
                # under mpu DeepSpeed's world_size = dp_size (=world/G), but accelerate already budgeted
                #   train_batch_size = micro x ga x WORLD -> DeepSpeedConfig._batch_assertion blows up (8 != 1*1*4).
                #   recompute train_batch_size from dp_size (effective batch = dp_size clips, semantically correct).
                _cfg = _kw.get("config_params") or _kw.get("config")
                if isinstance(_cfg, dict):
                    _dp_sz = _mpu_dist.get_world_size() // _sp_world_size
                    _mb = _cfg.get("train_micro_batch_size_per_gpu", 1)
                    _ga = _cfg.get("gradient_accumulation_steps", 1)
                    _mb = 1 if _mb in ("auto", None) else int(_mb)
                    _ga = 1 if _ga in ("auto", None) else int(_ga)
                    _cfg["train_batch_size"] = _mb * _ga * _dp_sz
                return _orig_ds_init(*_a, **_kw)

            _mpu_ds.initialize = _mpu_patched_init
            if accelerator.is_main_process:
                print(f"[SP §14 mpu] critic ZeRO-3 -> DP-stride-{_sp_world_size} groups "
                      f"(dp_size={_mpu_dist.get_world_size()//_sp_world_size}); "
                      f"broadcast {_bcast_n} trainable LoRA tensors WORLD-consistent", flush=True)
        try:
            real_score_model, critic_optimizer, critic_lr_scheduler = critic_accelerator.prepare(
                real_score_model, critic_optimizer, critic_lr_scheduler
            )
        finally:
            if _critic_mpu is not None:
                _mpu_ds.initialize = _orig_ds_init          # restore deepspeed.initialize
                _ds_groups.mpu = _orig_groups_mpu           # restore the global mpu (so the later evoke-critic sees the default WORLD)
        if _critic_mpu is not None:
            # postcondition: wrap the critic engine.step -> all-reduce (SUM) the grad shards across the SP group before stepping. the engine is the prepared model.
            _wrap_critic_step(real_score_model)
        # prepare of the third engine -- Evoke critic with its own accelerator + the "critic_evoke" plugin.
        #   NOTE: it must be prepared on evoke_critic_accelerator (do not reuse critic_accelerator: one accelerator per engine,
        #   preparing twice on the same accelerator breaks engine.step()). with offload on, the base starts on CPU and deepspeed.initialize
        #   moves it to the card ((5)f two-stage offload is not wired up yet, see the risks in the report).
        if _sf_dual:
            if dmd_deepspeed_training:
                evoke_critic_accelerator.state.select_deepspeed_plugin("critic_evoke")
                evoke_critic_accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = (
                    args.training_config.train_batch_size
                )
            real_score_model_hb, evoke_critic_optimizer, evoke_critic_lr_scheduler = (
                evoke_critic_accelerator.prepare(
                    real_score_model_hb, evoke_critic_optimizer, evoke_critic_lr_scheduler
                )
            )
    else:
        transformer, optimizer, lr_scheduler = accelerator.prepare(transformer, optimizer, lr_scheduler)

    # DeepSpeed's Accelerate wrapper calls
    # engine.step() inside accelerator.backward(), so p.grad is already cleared
    # when the normal training loop regains control.  The env-gated test hook
    # captures full ZeRO-2 gradients immediately before each engine's first step.
    if os.environ.get("SF_DECOUPLE_EQUIV_MODE"):
        from scripts.training.tmp.test_decouple_equivalence import (
            install_engine_capture,
            prewarm_da3_singleton,
        )
        prewarm_da3_singleton(args, accelerator.device)
        install_engine_capture("generator", accelerator.deepspeed_engine_wrapped.engine)
        install_engine_capture("critic_evoke_teacher", critic_accelerator.deepspeed_engine_wrapped.engine)
        if _sf_dual:
            install_engine_capture(
                "critic_evoke", evoke_critic_accelerator.deepspeed_engine_wrapped.engine
            )

    # Recalculate steps/epochs after dataloader size is known.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.training_config.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.training_config.max_train_steps = args.training_config.num_train_epochs * num_update_steps_per_epoch
    args.training_config.num_train_epochs = math.ceil(
        args.training_config.max_train_steps / num_update_steps_per_epoch
    )

    # Initialize experiment trackers on the main process.
    if accelerator.is_main_process:
        tracker_name = args.report_to.tracker_name or "wanvideo-train"
        wandb_name = args.report_to.wandb_name or "custom-wandb-run-name"
        accelerator.init_trackers(
            tracker_name,
            config=OmegaConf.to_container(args, resolve=True),
            init_kwargs={"wandb": {"name": wandb_name}},
        )

    # Train!
    total_batch_size = (
        args.training_config.train_batch_size
        * accelerator.num_processes
        * args.training_config.gradient_accumulation_steps
    )
    num_trainable_parameters = sum(param.numel() for model in params_to_optimize for param in model["params"])
    if args.training_config.is_train_dmd:
        critic_num_trainable_parameters = sum(
            param.numel() for model in critic_model_params_to_optimize for param in model["params"]
        )

    # implicit invariant -> explicit assert: a backward window free of DP reduction relies on
    #   (1) enable_backward_allreduce=False (skips the epilogue at the end of backward) and (2) the IPG bucket never
    #   overflowing (ZeRO-2 reduces on the spot once elements_in_ipg_bucket + numel > reduce_bucket_size). Raising
    #   lora_rank or switching to full SFT breaks it, and the symptom is a hang or a silently wrong gradient, so it
    #   has to fail at startup.
    if _stu_sp.is_any_enabled():
        import json as _stu_json
        with open(args.training_config.dmd_generator_deepspeed_config, "r") as _f:
            _stu_rbs = int(_stu_json.load(_f)["zero_optimization"]["reduce_bucket_size"])
        # mechanism B only: with the true (pre-prepare) param count, 625,721,344 > reduce_bucket_size, so
        #   reduce_ipg_grads() necessarily fires several times inside one backward -- enable_backward_allreduce=False
        #   only skips the epilogue, ZeRO-2's param-hook reduction is always active.
        #   Harmless under A-only (G_u=1): DeepSpeed asserts each param is reduced exactly once per backward, so N
        #   partial reductions equal one whole reduction, and with no U-subgroup all-to-all in the window there is no
        #   deadlock surface. Under G_u>1 those in-window DP reductions would interleave with the U-subgroup
        #   all-to-all, so this only fail-fasts for mechanism B.
        if _stu_sp.get_G_u() > 1:
            assert _stu_gen_numel_total < _stu_rbs, (
                f"[STU-SP §9.1-3] mechanism B (G_u={_stu_sp.get_G_u()}) requires the IPG bucket not to overflow, but the total trainable param count "
                f"{_stu_gen_numel_total} >= reduce_bucket_size {_stu_rbs} => there will be "
                f"{_stu_gen_numel_total // _stu_rbs + 1} DP reductions inside the backward window, interleaving with the U-subgroup all-to-all => deadlock risk. "
                f"please raise reduce_bucket_size in the ds json to >= {_stu_gen_numel_total}"
                f"(with overlap_comm:false that only costs one extra ipg_buffer), or shrink the trainable param count.")
        _stu_nred = 0 if _stu_gen_numel_total < _stu_rbs else _stu_gen_numel_total // _stu_rbs + 1
        accelerator.print(
            f"  [STU-SP] G_p={_stu_sp.get_G_p()} × G_u={_stu_sp.get_G_u()}, loss_scale=×{_stu_sp.loss_scale()}; "
            f"the /G_u for redundant params is **done in the graph** (the _ScatterTokens.backward + tail _ScaleGrad pair), no longer via a param hook "
            f"-- the latter would wrongly divide text_embedder (which is actually a partial sum) and wrongly divide the GEO-REG term; "
            f"total trainable params={_stu_gen_numel_total} (this rank after sharding={num_trainable_parameters}) "
            f"vs reduce_bucket_size={_stu_rbs} => about {_stu_nred} in-bucket reductions inside the backward window"
            + ("(G_u=1, no U-subgroup collective to interleave with => harmless; bucket order guarded by check_ipg_bucket_order)"
               if _stu_sp.get_G_u() <= 1 else "(NOTE: mechanism B already asserted no overflow)"))

    accelerator.print("***** Running training *****")
    accelerator.print(f"  Num generator trainable parameters = {num_trainable_parameters}")
    if args.training_config.is_train_dmd:
        accelerator.print(f"  Num fake_score_model trainable parameters = {critic_num_trainable_parameters}")
    accelerator.print(f"  Num examples = {len(train_dataset)}")
    accelerator.print(f"  Num batches each epoch = {len(train_dataloader)}")
    accelerator.print(f"  Num Epochs = {args.training_config.num_train_epochs}")
    accelerator.print(f"  Instantaneous batch size per device = {args.training_config.train_batch_size}")
    accelerator.print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    accelerator.print(f"  Gradient Accumulation steps = {args.training_config.gradient_accumulation_steps}")
    accelerator.print(f"  Total optimization steps = {args.training_config.max_train_steps}")
    global_step = 0
    first_epoch = 0

    ema_transformer = None
    vram_manager = None
    # dual_teacher.offload reuses the same manager for the EvokeTeacher<->Evoke residency swap (dmd_is_low_vram_mode is always false).
    if args.training_config.is_train_dmd and (args.training_config.dmd_is_low_vram_mode or _et_offload):
        vram_manager = OptimizedLowVRAMManager()

    # Resume from checkpoint if requested.
    if args.training_config.resume_from_checkpoint:
        if args.training_config.resume_from_checkpoint != "latest":
            resume_path = args.training_config.resume_from_checkpoint
            if os.path.isabs(resume_path):
                path = resume_path
            else:
                path = os.path.join(args.output_dir, resume_path)
        else:
            # Find the most recent checkpoint.
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = os.path.join(args.output_dir, dirs[-1]) if len(dirs) > 0 else None

        if path is None or not os.path.exists(path):
            accelerator.print(
                f"Checkpoint '{args.training_config.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.training_config.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(path, load_kwargs={"weights_only": False})
            if args.training_config.is_train_dmd:
                critic_accelerator.load_state(os.path.join(path, "critic"), load_kwargs={"weights_only": False})
                # Evoke critic checkpoint (mirror; separate subdirectory critic_evoke).
                if _sf_dual:
                    evoke_critic_accelerator.load_state(
                        os.path.join(path, "critic_evoke"), load_kwargs={"weights_only": False}
                    )
            global_step = int(os.path.basename(path).split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

            if args.training_config.use_ema:
                if args.training_config.is_train_dmd and args.training_config.dmd_is_low_vram_mode:
                    vram_manager.move_to_cpu(transformer, non_blocking=False)
                    vram_manager.move_to_cpu(real_score_model, non_blocking=False)

                transformer_cpu.load_state_dict(unwrap_model(transformer).state_dict())
                ema_transformer = create_ema_final(
                    accelerator=accelerator,
                    args=args,
                    transformer_cpu=transformer_cpu,
                    model_cls=model_cls,
                    ds_config=ds_config,
                    transformer_lora_config=transformer_lora_config,
                    resume_checkpoint_path=os.path.join(path, "model_ema"),
                    transformer_additional_kwargs=transformer_additional_kwargs,
                )
                accelerator.wait_for_everyone()

                transformer_cpu = None
                del transformer_cpu

                if args.training_config.is_train_dmd and args.training_config.dmd_is_low_vram_mode:
                    vram_manager.move_to_gpu(transformer, accelerator.device)
                    vram_manager.move_to_gpu(real_score_model, accelerator.device)
    else:
        initial_global_step = 0

    if args.model_config.load_checkpoints_custom:
        assert initial_global_step == 0

    progress_bar = tqdm(
        range(0, args.training_config.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    if (
        args.training_config.is_train_dmd and args.training_config.dmd_is_low_vram_mode
    ) or args.data_config.use_stage3_dataset:
        # online multi_dataset needs vae/text_encoder encoding in every step's materialize,
        # so they cannot be set to None the way offline DMD does (otherwise the first batch materialize crashes with NoneType.dtype). the low-vram
        # teacher swap-out orchestration (vram_manager swapping real_score_model) is orthogonal to keeping vae/te, and still saves ~56G.
        _sf_keep_encoders = args.data_config.use_multi_dataset
        if (
            (
                not args.training_config.is_dmd_vae_decode
                and not args.training_config.is_use_reward_model
                and not args.training_config.is_smoothness_loss
            )
            or args.training_config.is_use_gt_history
        ) and not _sf_keep_encoders:
            vae = None
        if not _sf_keep_encoders:
            text_encoder = None
        free_memory()

    # Initialize EMA model.
    if ema_transformer is None and args.training_config.use_ema:
        if args.training_config.is_train_dmd and args.training_config.dmd_is_low_vram_mode:
            vram_manager.move_to_cpu(transformer, non_blocking=False)
            vram_manager.move_to_cpu(real_score_model, non_blocking=False)
        else:
            transformer.to("cpu", non_blocking=False)

        transformer_cpu.load_state_dict(unwrap_model(transformer).state_dict())
        ema_transformer = create_ema_final(
            accelerator=accelerator,
            args=args,
            transformer_cpu=transformer_cpu,
            model_cls=model_cls,
            ds_config=ds_config,
            transformer_lora_config=transformer_lora_config,
            update_after_step=args.training_config.ema_start_step,
        )
        accelerator.wait_for_everyone()

        transformer_cpu = None
        del transformer_cpu

        if args.training_config.is_train_dmd and args.training_config.dmd_is_low_vram_mode:
            vram_manager.move_to_gpu(transformer, accelerator.device)
            vram_manager.move_to_gpu(real_score_model, accelerator.device)
        else:
            transformer.to(accelerator.device, non_blocking=False)

    # Collect GAN critic trainable parameter sets for selective gradient toggling.
    gan_critic_trainable_params = None
    gan_base_critic_trainable_params = None
    gan_extra_critic_trainable_params = None
    if args.training_config.is_use_gan:
        gan_critic_trainable_params = {
            name for name, param in real_score_model.named_parameters() if param.requires_grad
        }
        gan_extra_critic_trainable_params = {
            name
            for name, param in real_score_model.named_parameters()
            if param.requires_grad and any(module in name for module in critic_trainable_modules)
        }
        gan_base_critic_trainable_params = gan_critic_trainable_params - gan_extra_critic_trainable_params

    # Initialize error-recycling buffers.
    recycle_vars = None
    if args.training_config.use_error_recycling:
        from types import SimpleNamespace

        num_grids = args.training_config.num_grids

        recycle_vars = SimpleNamespace()
        recycle_vars.recycle_inferece_timesteps, recycle_vars.recycle_sigmas = get_timesteps(
            num_inference_steps=num_grids, denoising_strength=1, shift=1.0
        )

        resolutions = set()
        for t, h, w in sampler.buckets.keys():
            base_h = h // 8
            base_w = w // 8
            resolutions.add((base_h, base_w))
            if args.training_config.is_enable_stage2:
                resolutions.add((base_h // 2, base_w // 2))
                resolutions.add((base_h // 4, base_w // 4))

        recycle_vars.latent_error_buffer = {
            resolution: {i: [] for i in range(num_grids)} for resolution in resolutions
        }
        recycle_vars.y_error_buffer = {resolution: {i: [] for i in range(num_grids)} for resolution in resolutions}

    def safe_item(value):
        return value.item() if hasattr(value, "item") else value

    accelerator.wait_for_everyone()

    prof = None
    if args.training_config.profile_out_dir is not None:
        prof = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(skip_first=2, wait=1, warmup=1, active=2, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(args.training_config.profile_out_dir),
            profile_memory=True,
            with_stack=True,
            record_shapes=True,
        )

    # [v2v ODE dump] lazily build the collect-pipe (once);
    _dump_ode_pipe = None
    _dump_ode_count = 0

    for epoch in range(first_epoch, args.training_config.num_train_epochs):
        transformer.train()
        if args.training_config.is_train_dmd:
            real_score_model.train()
            # the Evoke-specific critic must also be train() to be able to backward: construction ((5)a) set .eval()
            #   and never switched back; the ZeRO-2 optimizer does not check module.training so smoke#3 got away with it, but ZeRO-3 stage3 backward asserts
            #   module.training=True -> not switching crashes with "backward pass is invalid for module in evaluation mode". mirrors the EvokeTeacher
            #   critic; the frozen base has requires_grad=False and only the critic-LoRA is trainable, so train() does not change any values (no dropout/BN, LoRA dropout=0).
            if _sf_dual and real_score_model_hb is not None:
                real_score_model_hb.train()
        sampler.set_epoch(epoch)
        train_dataset.set_epoch(epoch)

        for step, batch in enumerate(train_dataloader):
            # measure two things SF-PROFILE cannot see (zero overhead when the env is off, bit-wise no effect on training):
            #   - _pp_fetch = from the end of the previous step to this batch being in hand = **waiting on the worker** (this is where a starved worker shows up);
            #   - _pp_t_prep = all the time between this line and sf_prof_step_begin() = total data-preparation cost.
            #   measured, the two together are ~= 30-34s/step, 22% of the real wall clock, and SF-PROFILE total does not count any of it.
            from evoke.utils import sf_prep_profile as _pp_mod
            _pp_fetch = None
            if _pp_mod.enabled():
                _pp_prev = globals().get("_PP_LAST_END")
                _pp_fetch = (_pp_mod.mark() - _pp_prev) if _pp_prev else None
            _pp_mod.step_reset()
            _pp_t_prep = _pp_mod.mark()

            # on-demand GT encoding: the data side must know which chunk j this step's GEO-REG will use so it can encode
            #   only up to pixel frame 36j+33 -- hence j is sampled here rather than inside the loss. It uses a local
            #   random.Random (touches no global/CUDA RNG), is per-rank independent (the GEO-REG branch has no collective),
            #   and whether GEO-REG fires is a pure function of (global_step, config).
            _gt_ondemand = bool(getattr(args.training_config, "sf_gt_encode_on_demand", False))
            _sf_geo_j = None
            _sf_gt_encode_px = None
            if _gt_ondemand:
                import random as _rnd_geo
                _wk = max(1, int(getattr(args.training_config, "sf_geo_reg_every_k", 1) or 1))
                _ww = float(getattr(args.training_config, "sf_geo_reg_weight", 0.0) or 0.0)
                _wfz = int(getattr(args.training_config, "sf_gen_freeze_steps", 0) or 0)
                _wN = int(args.training_config.dmd_num_latent_sections_max)
                _win_c = int(args.training_config.latent_window_size[0])
                _geo_maybe = (_ww > 0.0 and int(global_step) % _wk == 0 and int(global_step) >= _wfz)
                _P_sec = int(getattr(args.training_config, "rollout_prefix_sections", 1) or 1)
                if _geo_maybe:
                    # the seed deliberately excludes process_index: the baseline drew j from a CUDA generator seeded identically
                    #   on every rank, so all ranks used the same GT chunk. Making it per-rank would change the GEO-term gradient
                    #   (48 chunks instead of one) while staying invisible in the logs, since j is only logged on the main process.
                    # the domain must match the `randint(_wr_P, _wr_T)` inside GEO-REG: _wr_P = rollout_prefix_sections,
                    #   _wr_T = N+1, and Python's randint is inclusive on both ends. The lower bound must not be hardcoded to 1:
                    #   with P>=2 that draws j<P, on-demand encoding then stops at chunk j, and the legal interval inside GEO-REG
                    #   becomes empty.
                    _sf_geo_j = _rnd_geo.Random(
                        int(global_step) * 7919 + 20260729
                    ).randint(_P_sec, _wN)
                    _sf_gt_encode_px = (_sf_geo_j * _win_c + _win_c - 1) * 4 + 1   # pixel frames needed for latent [0, 9j+9)
                else:
                    _sf_gt_encode_px = (_P_sec * _win_c - 1) * 4 + 1               # prefix only (33 @ P=1,win=9)
            if os.environ.get("SF_DECOUPLE_EQUIV_MODE"):
                from scripts.training.tmp.test_decouple_equivalence import begin_batch
                begin_batch(batch)
            models_to_accumulate = [transformer]
            if args.training_config.is_train_dmd:
                models_to_accumulate.append(real_score_model)

            with torch.no_grad():
                latent_window_size = args.training_config.latent_window_size[0]

                # Get data samples
                gt_history_latents = None
                gt_target_latents = None
                gt_x0_latents = None
                gt_history_latents_2 = None
                gt_target_latents_2 = None
                gt_x0_latents_2 = None
                history_latents = None
                target_latents = None
                x0_latents = None
                model_input = None
                prompt_raws = None
                prompt_embeds = None
                indices_hidden_states = None
                indices_latents_history_short = None
                indices_latents_history_mid = None
                indices_latents_history_long = None
                latents_history_short = None
                latents_history_mid = None
                latents_history_long = None
                gan_vae_latents = None
                gan_prompt_embeds = None
                ode_latents = None
                ode_prompt_embeds = None
                # [v2v ODE] warp conditions -> 9-tuple gt_all_data (read back from the .pt; a t2v .pt has none -> None -> the loss takes the t2v path)
                ode_gt_all_data = None
                ode_attention_kwargs = None
                ode_cam_Ks = None
                ode_cam_c2ws = None
                ode_cam_base_h = None
                ode_cam_base_w = None
                ode_cam_strategy = "scale_ks"
                # [v2v ODE] `_use_geo_train` gates the batch-MATERIALIZED GEO common code below
                # (warp_video_latents / geo_condition_mode / sink / DMD-from-materialize). The
                # use_stage3_dataset (.pt) path threads its warp via `ode_gt_all_data` into
                # `_ode_regression_loss` instead, so it must NOT take that path → default False.
                # Only the use_stage1_dataset (online materialize) branch sets it True (:~1524).
                _use_geo_train = False
                text_prompt_raws = None
                text_prompt_embeds = None

                # [review#2] the _sf_* must be initialized outside the data branches: the loss call in the common code references them unconditionally,
                # so assigning them only inside the stage1 branch would make the current use_stage3_dataset DMD recipe NameError at step0.
                _sf_prefix_latents = None
                _sf_gt_latents = None
                _sf_prompt_embeds_list = None
                _sf_score_prompt_embeds = None
                _sf_teacher_y = None
                _sf_segment_frame_ranges = None
                _sf_pose_Ks = None
                _sf_pose_c2ws = None
                # same reason as above (the loss call in the common code references them unconditionally): set None/False outside the data branches.
                _sf_prompt_embeds_list_i2v = None
                _sf_i2v_hist_latent_full = None
                _sf_i2v_active = False
                _sf_i2v_hist_latent = None
                # NOTE: the sample type must be "copied out of batch into a local": this loop body contains `batch = None; del batch` (to save memory,
                # search for the first "del batch"), while the mode dispatch happens **after** it => a batch.get() there would
                # UnboundLocalError (measured crash on 48 cards,). same as the existing _dump_uttid convention: any batch field
                #   needed after the del must first be copied into an _sf_*/_dump_* local.
                #   (an AST guard that scans for batch reads after the del lives on the long-dmd-formal branch, not here.)
                _sf_sample_is_i2v = False
                if args.data_config.use_stage3_dataset:
                    noisy_model_input_shape = (
                        args.training_config.train_batch_size,
                        16,
                        latent_window_size,
                        args.data_config.single_height // 8,
                        args.data_config.single_width // 8,
                    )

                    # For ODE
                    if args.training_config.is_use_ode_regression:
                        ode_latent_window_size = batch["ode_latent_window_size"][0]
                        ode_latents = batch["ode_latents"][0]
                        ode_prompt_embeds = batch["ode_prompt_embeds"][:1].to(
                            accelerator.device, dtype=weight_dtype, non_blocking=True
                        )
                        assert args.training_config.train_batch_size == 1
                        assert ode_latent_window_size == latent_window_size
                        # [v2v ODE] read back the saved warp conditions -> rebuild the 9-tuple (same layout as inference_with_trajectory_stage2);
                        # tensors are moved to device (dtype is handled by .to(prompt dtype) inside the loss forward). a t2v .pt lacks this key -> stays None.
                        _raw_geo = batch["ode_geo"][0] if "ode_geo" in batch else None
                        if _raw_geo is not None:
                            _w = {
                                _k: (_v.to(accelerator.device, non_blocking=True) if torch.is_tensor(_v) else _v)
                                for _k, _v in _raw_geo.items()
                            }
                            ode_gt_all_data = (
                                None,
                                _w["indices_hidden_states"],
                                _w["indices_latents_history_short"],
                                _w["indices_latents_history_mid"],
                                _w["indices_latents_history_long"],
                                _w["latents_history_short"],
                                _w["latents_history_mid"],
                                _w["latents_history_long"],
                                None,
                            )
                            ode_attention_kwargs = _w.get("attention_kwargs")
                            # [v2v ODE plucker] read back the camera poses -> the loss rebuilds cam_plucker_emb per stage (same convention as the dump)
                            ode_cam_Ks = _w.get("cam_Ks")
                            ode_cam_c2ws = _w.get("cam_c2ws")
                            ode_cam_base_h = _w.get("cam_base_h")
                            ode_cam_base_w = _w.get("cam_base_w")
                            ode_cam_strategy = _w.get("cam_strategy", "scale_ks")

                    # For Text
                    if dataset_kwargs["text_folders"] and not args.training_config.is_only_ode_regression:
                        text_prompt_raws = batch["text_prompt_raws"]
                        text_prompt_embeds = batch["text_prompt_embeds"].to(
                            accelerator.device, dtype=weight_dtype, non_blocking=True
                        )

                    # For GAN
                    if args.training_config.is_use_gan or args.training_config.is_use_gt_history:
                        gan_vae_latents = batch["gan_vae_latents"].to(
                            accelerator.device, dtype=weight_dtype, non_blocking=True
                        )
                        gan_prompt_embeds = batch["gan_prompt_embeds"].to(
                            accelerator.device, dtype=weight_dtype, non_blocking=True
                        )
                        if args.training_config.is_use_gt_history:
                            text_prompt_raws = batch["gan_prompt_raws"]
                            text_prompt_embeds = gan_prompt_embeds
                            gt_target_latents = gan_vae_latents.to(
                                accelerator.device, dtype=weight_dtype, non_blocking=True
                            )
                            gt_x0_latents = batch["gan_x0_latents"].to(
                                accelerator.device, dtype=weight_dtype, non_blocking=True
                            )
                            gt_history_latents = batch["gan_history_latents"].to(
                                accelerator.device, dtype=weight_dtype, non_blocking=True
                            )

                            gt_target_latents_2 = batch["gan_vae_latents_2"].to(
                                accelerator.device, dtype=weight_dtype, non_blocking=True
                            )
                            gt_x0_latents_2 = batch["gan_x0_latents_2"].to(
                                accelerator.device, dtype=weight_dtype, non_blocking=True
                            )
                            gt_history_latents_2 = batch["gan_history_latents_2"].to(
                                accelerator.device, dtype=weight_dtype, non_blocking=True
                            )
                            assert gt_target_latents_2.shape[2] == args.training_config.num_critic_input_frames
                        assert gan_vae_latents.shape[2] == args.training_config.num_critic_input_frames

                elif args.data_config.use_stage1_dataset:
                    # Stage-1 dataset: prepare short/mid/long history latents and optional raw sink frames.
                    sink_latents = None
                    nearby_sink_latents = None
                    nearby_sink_indices = None
                    # Resolve GEO training flag (training_config takes priority over data_config for compatibility).
                    _use_geo_train = bool(
                        getattr(args.training_config, "use_geometric_state", None)
                        if getattr(args.training_config, "use_geometric_state", None) is not None
                        else getattr(args.data_config, "use_geometric_state", False)
                    )
                    _geo_cfg = getattr(args.model_config, "geometric_state", None)
                    _geo_retrieve_cfg = getattr(_geo_cfg, "retrieve", None) if _geo_cfg is not None else None
                    # Online mode: materialize raw video and prompt into latents inline.
                    if args.data_config.use_multi_dataset:
                        from evoke.dataset.online_materialize import materialize_online_batch
                        batch = materialize_online_batch(
                            batch, vae=vae, tokenizer=tokenizer, text_encoder=text_encoder,
                            history_sizes=args.training_config.history_sizes,
                            latent_window_size=args.training_config.latent_window_size[0],
                            latents_mean=online_latents_mean, latents_std=online_latents_std,
                            device=accelerator.device, weight_dtype=weight_dtype,
                            is_keep_x0=True, seed=args.seed, epoch=epoch,
                            use_geometric_state=_use_geo_train,
                            geo_keep_clean_anchor=bool(getattr(_geo_cfg, "warp_keep_clean_anchor", False)) if _geo_cfg is not None else False,
                            geo_retrieve_cfg=_geo_retrieve_cfg,
                            geo_condition_t2v_ratio=float(getattr(args.training_config, "geo_condition_t2v_ratio", 0.0) or 0.0),
                            geo_condition_i2v_ratio=float(getattr(args.training_config, "geo_condition_i2v_ratio", 0.0) or 0.0),
                            geo_cloud_warp_cfg=getattr(_geo_cfg, "cloud_warp", None) if _geo_cfg is not None else None,
                            geo_visibility_aware_noise=bool(getattr(_geo_cfg, "visibility_aware_noise", False)) if _geo_cfg is not None else False,
                            geo_sigma_invisible=float(getattr(_geo_cfg, "warp_noise_sigma_invisible", 0.8)) if _geo_cfg is not None else 0.8,
                            geo_sigma_visible_min=float(getattr(_geo_cfg, "warp_noise_sigma_min", 0.111)) if _geo_cfg is not None else 0.111,
                            geo_sigma_visible_max=float(getattr(_geo_cfg, "warp_noise_sigma_max", 0.135)) if _geo_cfg is not None else 0.135,
                            # Error-bank warp injection (err-then-noise); no-op unless enabled + bank warmed.
                            recycle_vars=recycle_vars,
                            args=args,
                            geo_warp_error_inject_enabled=bool(getattr(_geo_cfg, "warp_error_inject_enabled", False)) if _geo_cfg is not None else False,
                            geo_warp_error_prob=float(getattr(_geo_cfg, "warp_error_prob", 0.0)) if _geo_cfg is not None else 0.0,
                            # Train-only warp pose jitter (warp render pose only; plucker + loss stay at GT). None => off.
                            geo_pose_jitter_cfg=getattr(_geo_cfg, "warp_pose_jitter", None) if _geo_cfg is not None else None,
                            # whole-clip rollout + interleave data mode
                            sf_full_rollout_interleave=args.data_config.use_full_rollout_interleave,
                            sf_prefix_sections=args.training_config.rollout_prefix_sections,
                            # y is consumed by evoke_teacher only; evoke does not build it (avoids importing the vendored model)
                            sf_build_teacher_y=_sf_evoke_teacher,
                            # dual: skip whole-clip encoding, encode only the prefix window (sf_gt_latents=None);
                            #   N/T_px are derived from full_clip_num_frames. dual off -> sf_skip_full_encode=False -> whole-clip encoding, bit-identical.
                            sf_skip_full_encode=_sf_dual,
                            full_clip_num_frames=args.data_config.num_frames,
                            # master switch for the i2v path (>0 = on): the data side then
                            #   (1) additionally produces i2v-layout section prompts + a 1x-slot latent for video samples (prefix/y are sliced by the train loop);
                            #   (2) routes image-only samples (a single jpg frame) entirely through materialize_i2v_image_only.
                            #   =0 -> the data side never computes anything extra, and an image-only sample fail-fasts (no silent misbehavior).
                            sf_i2v_prefix_latent_frames=int(
                                getattr(args.training_config, "sf_i2v_prefix_latent_frames", 0) or 0),
                            sf_i2v_hist_latent_mode=str(
                                getattr(args.training_config, "sf_i2v_hist_latent_mode", "static_repeat")),
                            # ratio=0 => video samples are always v2v => the data side skips that 33-frame 1x latent nobody reads.
                            sf_i2v_ratio=float(getattr(args.training_config, "sf_i2v_ratio", 0.0) or 0.0),
                            # up to which pixel frame this step needs encoding (None=whole clip=old behavior)
                            sf_gt_encode_px=_sf_gt_encode_px,
                            # an image-only sample has no clip to infer N from (fixed-N path: min==max, guaranteed by the validator)
                            sf_num_generated_sections=int(
                                args.training_config.dmd_num_latent_sections_max),
                        )

                    # Prepare prompt embeds
                    prompt_embeds = batch["prompt_embeds"].to(accelerator.device)

                    # / whole-clip rollout + per-section prompt data (produced by the use_full_rollout_interleave mode).
                    if _sf_any:
                        _sf_prefix_latents = batch["sf_prefix_latents"].to(accelerator.device, dtype=weight_dtype)
                        # full-length GT latents (used to swap in GT for the teacher long/mid tier; defaults to None for backward compat with old data)
                        _sf_gt_latents = batch.get("sf_gt_latents")
                        if _sf_gt_latents is not None:
                            _sf_gt_latents = _sf_gt_latents.to(accelerator.device, dtype=weight_dtype)
                        _sf_prompt_embeds_list = [
                            t.to(accelerator.device, dtype=weight_dtype) for t in batch["sf_prompt_embeds_list"]
                        ]
                        # evoke_teacher-only conditions (the evoke tier scoring does not consume them, so use a tolerant get; (4)).
                        _sf_score_prompt_embeds = batch.get("sf_score_prompt_embeds")
                        if _sf_score_prompt_embeds is not None:
                            _sf_score_prompt_embeds = _sf_score_prompt_embeds.to(
                                accelerator.device, dtype=weight_dtype)
                        _sf_teacher_y = batch.get("sf_teacher_y")
                        if _sf_teacher_y is not None:
                            _sf_teacher_y = _sf_teacher_y.to(accelerator.device, dtype=weight_dtype)
                        _sf_segment_frame_ranges = batch.get("sf_segment_frame_ranges")
                        if _sf_evoke_teacher:
                            assert _sf_score_prompt_embeds is not None and _sf_teacher_y is not None, \
                                "[SF10S] evoke_teacher scoring needs sf_score_prompt_embeds + sf_teacher_y"
                        # the second set of section prompts for i2v mixed training (P_lat=1 layout) plus the 1x-slot latent.
                        #   with ratio=0 the data side always returns None -> both variables stay None -> the i2v block below is skipped entirely -> old path bit-identical.
                        _sf_prompt_embeds_list_i2v = batch.get("sf_prompt_embeds_list_i2v")
                        if _sf_prompt_embeds_list_i2v is not None:
                            _sf_prompt_embeds_list_i2v = [
                                t.to(accelerator.device, dtype=weight_dtype) for t in _sf_prompt_embeds_list_i2v
                            ]
                        _sf_i2v_hist_latent_full = batch.get("sf_i2v_hist_latent")
                        if _sf_i2v_hist_latent_full is not None:
                            _sf_i2v_hist_latent_full = _sf_i2v_hist_latent_full.to(
                                accelerator.device, dtype=weight_dtype)
                        # image-only sample marker (set True by materialize_i2v_image_only). NOTE: it must be copied
                        #   into a local here -- the downstream mode dispatch runs after `del batch`, where reading batch would UnboundLocalError.
                        _sf_sample_is_i2v = bool(batch.get("sf_sample_is_i2v", False))
                        # GT pose for warp-in-rollout (returned by the materialize side; defaults to None when warp is off)
                        _sf_pose_Ks = batch.get("sf_pose_Ks")
                        _sf_pose_c2ws = batch.get("sf_pose_c2ws")
                        if _sf_pose_Ks is not None:
                            _sf_pose_Ks = _sf_pose_Ks.to(accelerator.device)
                        if _sf_pose_c2ws is not None:
                            _sf_pose_c2ws = _sf_pose_c2ws.to(accelerator.device)

                    # DMD on the stage1/GEO data path: define the per-window noise shape that
                    # _generator_loss/_critic_loss expect (same formula the use_stage3_dataset branch sets);
                    # otherwise `noisy_model_input_shape` is unbound here → UnboundLocalError at the DMD call.
                    noisy_model_input_shape = (
                        args.training_config.train_batch_size,
                        16,
                        latent_window_size,
                        args.data_config.single_height // 8,
                        args.data_config.single_width // 8,
                    )

                    # Prepare stage1 clean data
                    history_latents = batch["history_latents"].to(accelerator.device)
                    target_latents = batch["target_latents"].to(accelerator.device)
                    x0_latents = batch["x0_latents"].to(accelerator.device)
                    # Disable legacy random_drop when GEO sample-level conditioning is active.
                    _old_random_drop_active = (
                        bool(args.training_config.is_random_drop) and not _use_geo_train
                    )
                    (
                        model_input,
                        indices_hidden_states,
                        indices_latents_history_short,
                        indices_latents_history_mid,
                        indices_latents_history_long,
                        latents_history_short,
                        latents_history_mid,
                        latents_history_long,
                        sink_latents,
                        nearby_sink_latents,
                    ) = prepare_stage1_clean_input_from_latents(
                        history_latents=history_latents,
                        target_latents=target_latents,
                        x0_latents=x0_latents,
                        latent_window_size=latent_window_size,
                        history_sizes=args.training_config.history_sizes,
                        is_random_drop=_old_random_drop_active,
                        random_drop_i2v_ratio=args.training_config.random_drop_i2v_ratio,
                        random_drop_v2v_ratio=args.training_config.random_drop_v2v_ratio,
                        random_drop_t2v_ratio=args.training_config.random_drop_t2v_ratio,
                        is_keep_x0=True,
                        use_raw_sink_frames=args.training_config.use_raw_sink_frames,
                        dtype=weight_dtype,
                        device=accelerator.device,
                    )
                    # Discriminator "real" for the use_stage1_dataset / GEO path: model_input is the clean 9-frame target chunk,
                    # already (x-mean)*std normalized and on device, shape (B,16,latent_window_size,H/8,W/8) -- exactly what the
                    # generator's fake is distilled toward. Captured here before target_latents is del'd below.
                    # With is_use_gt_history=false the gt_*/_2 variants are unused (the critic takes the GEO branch).
                    if args.training_config.is_use_gan:
                        gan_vae_latents = model_input.to(accelerator.device, dtype=weight_dtype, non_blocking=True)
                        gan_prompt_embeds = prompt_embeds.to(accelerator.device, dtype=weight_dtype, non_blocking=True)
                        assert gan_vae_latents.shape[2] == args.training_config.num_critic_input_frames, (
                            f"[GAN-GEO] real chunk T={gan_vae_latents.shape[2]} must == num_critic_input_frames="
                            f"{args.training_config.num_critic_input_frames} (= latent_window_size) for the "
                            f"fake/real dim-0 concat in the discriminator forward"
                        )
                    history_latents = None
                    target_latents = None
                    x0_latents = None
                    del history_latents
                    del target_latents
                    del x0_latents
                else:
                    raise NotImplementedError

                # Stash camera pose tensors before batch is deleted.
                _stashed_target_pose_Ks = batch.get("target_pose_Ks") if isinstance(batch, dict) else None
                _stashed_target_pose_c2ws = batch.get("target_pose_c2ws") if isinstance(batch, dict) else None

                # base resolution + Ks-scaling strategy for the score models' Plucker
                # (teacher/critic are stage1 full-res, trained WITH plk). Mirror the ODE-path logic:
                # camera_control settings when present, else GEO-only data-config defaults.
                _cc = getattr(args.model_config, "camera_control", None)
                if _cc is not None and getattr(_cc, "enabled", False):
                    _dmd_cam_base_h = _cc.base_height_pix or args.data_config.single_height
                    _dmd_cam_base_w = _cc.base_width_pix or args.data_config.single_width
                    _dmd_cam_strategy = _cc.pc_resolution_strategy
                else:
                    _dmd_cam_base_h = args.data_config.single_height
                    _dmd_cam_base_w = args.data_config.single_width
                    _dmd_cam_strategy = "scale_ks"

                # GEO training: prepare short-tier warp override and visibility masks.
                _geo_history_visible_mask_short = None
                _geo_history_visible_mask_mid = None
                _geo_history_visible_mask_long = None
                _geo_visibility_mask = None   # warp visibility of the target chunk (latent resolution), for masked-region loss weighting + per-region logging
                # Leading short-tier frames [prefix | warp] that error-recycling must NOT inject mem
                # error into (warp gets its own err-then-noise injection in materialize). 0 = no GEO warp.
                _geo_protect_short_frames = 0
                # Stage2 warp compression mode (default fixed_mem = legacy; None kwargs preserves bit-identical path).
                _stage2_warp_mode = str(
                    getattr(args.training_config, "stage2_warp_compression_mode", "fixed_mem") or "fixed_mem"
                )
                # warp_rope_noise_center_align is fixed_mem-only (needs the full-res warp as the coord reference;
                # synchronized downsamples warp per stage so there is no single full-res frame). Fail-fast.
                _geo_cfg_for_nc = getattr(args.model_config, "geometric_state", None)
                _geo_nc_on = bool(getattr(_geo_cfg_for_nc, "warp_rope_noise_center_align", False))
                if _geo_nc_on and _stage2_warp_mode != "fixed_mem":
                    raise ValueError(
                        f"warp_rope_noise_center_align=True requires stage2_warp_compression_mode='fixed_mem' "
                        f"(mutually exclusive with '{_stage2_warp_mode}')."
                    )
                _geo_train_attention_kwargs = (
                    None if _stage2_warp_mode == "fixed_mem"
                    else {"stage2_warp_compression_mode": _stage2_warp_mode}
                )
                _geo_train_nearby_sink_indices = None
                # `_rtc_on`/`_geo_saturate` used to be defined only inside the single-chunk GEO assembly block (which needs batch to
                # carry warp_video_latents), but the `if _use_geo_train:` post-processing block at 2107 references them unconditionally -> the sf multi-section mode
                # (_use_geo_train=true while batch has no warp keys) UnboundLocalError'd at step0. defaults as fallback: False/no-op,
                # and the assembly block reassigns them when reached (default path bit-identical).
                _rtc_on = False
                _geo_saturate = lambda _lat, _tier: _lat  # noqa: E731 -- no-op; redefined to the real implementation inside the assembly block
                # GEO condition mode sampled by online_materialize: t2v / i2v / full_geo.
                _geo_condition_mode = (
                    batch.get("geo_condition_mode") if (_use_geo_train and isinstance(batch, dict)) else None
                ) or "full_geo"
                _geo_err_inject_depth = None  # stage2 err-bank: max injected-error depth (warp/prev_short) -> per_item_depth
                if _use_geo_train and isinstance(batch, dict) and batch.get("warp_video_latents") is not None:
                    _geo_warp_lat = batch["warp_video_latents"].to(
                        device=accelerator.device, dtype=weight_dtype, non_blocking=True
                    )
                    _geo_visibility_mask = batch["warp_visibility_mask"].to(
                        device=accelerator.device, dtype=torch.float32, non_blocking=True
                    )
                    # Train-only warp-token drop: stochastically zero the warp visibility mask (full / per-frame /
                    # per-patch). Couples token-drop (visible_token_drop) + loss weighting via this single mask.
                    _wtd_cfg = getattr(_geo_cfg, "warp_token_drop", None) if _geo_cfg is not None else None
                    if _wtd_cfg is not None and getattr(_wtd_cfg, "enabled", False):
                        from evoke.dataset.online_materialize import apply_warp_token_drop
                        _geo_visibility_mask = apply_warp_token_drop(_geo_visibility_mask, _wtd_cfg)
                    # short tier layout: [prefix(1) | warp(W) | prev_short(1)] -> protect 1+W leading frames.
                    _geo_protect_short_frames = 1 + int(_geo_warp_lat.shape[2])
                    # short tier layout: [prefix(1) | warp(W) | prev_short(1)] (built below).
                    assert batch.get("geo_source_image_latent") is not None, (
                        "GEO training needs batch['geo_source_image_latent'] (produced by online_materialize); "
                        "use_geometric_state may have been left off in the config."
                    )
                    _geo_prefix_latent_clean = batch["geo_source_image_latent"].to(
                        device=accelerator.device, dtype=weight_dtype, non_blocking=True
                    )  # [B, 16, 1, H_lat, W_lat]
                    # Determine prefix noise sigma range from short_tier_noise config (fallback to defaults).
                    _st_noise_cfg = getattr(args.model_config, "geometric_state", None)
                    _st_noise_cfg = getattr(_st_noise_cfg, "short_tier_noise", None) if _st_noise_cfg is not None else None
                    _st_enabled = bool(getattr(_st_noise_cfg, "enabled", False)) if _st_noise_cfg is not None else False
                    _st_targets = list(getattr(_st_noise_cfg, "target_tiers", []) or []) if _st_noise_cfg is not None else []
                    if _st_enabled and "prefix" in _st_targets:
                        _geo_sigma_low = float(getattr(_st_noise_cfg, "sigma_min", 0.2))
                        # Per-tier override: prefix_sigma_max (None → shared sigma_max). 0.0 → clean prefix.
                        _prefix_hi = getattr(_st_noise_cfg, "prefix_sigma_max", None)
                        _geo_sigma_high = float(_prefix_hi) if _prefix_hi is not None else float(getattr(_st_noise_cfg, "sigma_max", 0.6))
                        _geo_sigma_source = "Plan A v3"
                    else:
                        _geo_sigma_low, _geo_sigma_high = 0.111, 0.135  # default image-conditioning noise range
                        _geo_sigma_source = "default"
                    _geo_image_sigma = (
                        torch.rand(1, device=_geo_prefix_latent_clean.device) * (_geo_sigma_high - _geo_sigma_low) + _geo_sigma_low
                    ).view(1, 1, 1, 1, 1).to(dtype=_geo_prefix_latent_clean.dtype)
                    _geo_prefix_latent = (
                        _geo_image_sigma * torch.randn_like(_geo_prefix_latent_clean)
                        + (1 - _geo_image_sigma) * _geo_prefix_latent_clean
                    )
                    if accelerator.is_main_process and global_step < 3:
                        print(f"[ShortTier-Noise TRAIN] prefix sigma={float(_geo_image_sigma.item()):.3f} from [{_geo_sigma_low:.3f}, {_geo_sigma_high:.3f}] ({_geo_sigma_source})")

                    # Recover prev_short (the real previous latent) from the ORIGINAL short tier
                    # before the GEO override discards it; build [prefix | warp | prev_short] to match inference
                    # (pipeline_evoke builds the same layout). prev_short = continuity anchor with the ref tail.
                    _geo_prev_short_clean = latents_history_short[:, :, -1:, :, :].clone()
                    # [i2v] no prior chunk → prev_short = the ref image itself (= clean prefix), matching inference
                    # i2v chunk0 (pipeline uses the ref-image latent as prev_short). Without this, an i2v sample
                    # drawn at choice_idx>0 would leak the section's REAL last frame into prev_short.
                    if _geo_condition_mode == "i2v":
                        _geo_prev_short_clean = _geo_prefix_latent_clean.clone()
                    # Snapshot pre-errbank (CLEAN) warp + prev_short for the asymmetric DM teacher.
                    # _geo_prev_short_clean gets errbank-mutated at ~:1779 (misnomer); keep the true clean copy.
                    # _geo_warp_lat is rebound (not in-place) by the errbank '+=' so the alias stays clean. None
                    # unless toggled -> no extra memory on the default path.
                    _rtc_on = bool(getattr(args.training_config, "recycle_teacher_clean", False))
                    _geo_warp_lat_clean_src = _geo_warp_lat if _rtc_on else None
                    _geo_prev_short_pre_eb = latents_history_short[:, :, -1:, :, :].clone() if _rtc_on else None
                    # [i2v] teacher's clean prev_short must also be the ref image (not the leaked section frame),
                    # mirroring the student override above — else Option B teacher sees a leaked real prev frame.
                    if _rtc_on and _geo_condition_mode == "i2v":
                        _geo_prev_short_pre_eb = _geo_prefix_latent_clean.clone()
                    _geo_prev_short_idx_old = indices_latents_history_short[:, -1:].clone()
                    # PER-FRAME probabilistic latent saturation: each frame of an enabled tier independently, w.p. `prob`, draws
                    #    a sat_factor 50/50 over/under in [ratio_min,1)u(1,ratio_max] and applies (x - chanmean)*f + chanmean.
                    #    ratio_max caps ~1.4 (f>=1.8 flips into brightening). Applied only to the degraded tiers
                    #    (warp/prev_short/mid/long), never the prefix. Requires recycle_teacher_clean, else the teacher sees the
                    #    saturation too and there is no signal.
                    _wsat_cfg = getattr(_geo_cfg, "warp_saturation_corrupt", None) if _geo_cfg is not None else None
                    _wsat_on = bool(getattr(_wsat_cfg, "enabled", False)) if _wsat_cfg is not None else False
                    _wsat_tiers = set(getattr(_wsat_cfg, "target_tiers", []) or []) if _wsat_on else set()
                    _wsat_step_on = False   # per-STEP gate, decided ONCE & shared across tiers (whole history clean or not)
                    if _wsat_on:
                        # recycle_teacher_clean is only needed on the DMD dual-engine path (stage3); stage1 has no teacher/critic,
                        # where saturation corruption acts directly on the warp history of AR continuation training and needs no Option B asymmetry.
                        if args.training_config.is_train_dmd:
                            assert _rtc_on, (
                                "[GEO-sat DMD] warp_saturation_corrupt.enabled=True requires recycle_teacher_clean=True on the DMD path "
                                "(otherwise the teacher also eats the saturated history and the supervision target becomes meaningless)"
                            )
                        _wsat_min = float(getattr(_wsat_cfg, "ratio_min", 0.5))
                        _wsat_max = float(getattr(_wsat_cfg, "ratio_max", 1.4))
                        _wsat_frame_prob = float(getattr(_wsat_cfg, "frame_prob", 0.6))   # per-FRAME (given step on)
                        # per-STEP enable: w.p. step_prob saturate this step, else WHOLE history clean (model still
                        # sees clean histories ~ (1-step_prob) of the time -> avoids over-correcting).
                        _wsat_step_on = random.random() < float(getattr(_wsat_cfg, "step_prob", 0.6))

                    def _geo_saturate(_lat, _tier):
                        # Per-frame independent saturation on the degraded copy of an enabled tier (only if step is on).
                        if _lat is None or not _wsat_step_on or _tier not in _wsat_tiers:
                            return _lat
                        _T = int(_lat.shape[2])
                        _facs = []
                        for _ in range(_T):
                            if random.random() < _wsat_frame_prob:
                                if random.random() < 0.5:
                                    _facs.append(random.uniform(_wsat_min, 1.0 - 1e-3))
                                else:
                                    _facs.append(random.uniform(1.0 + 1e-3, _wsat_max))
                            else:
                                _facs.append(1.0)
                        if all(_f == 1.0 for _f in _facs):
                            return _lat   # no frame selected -> exact no-op
                        if accelerator.is_main_process and (global_step < 3 or global_step % 200 == 0):
                            _nz = [round(_f, 2) for _f in _facs if _f != 1.0]
                            _over = sum(1 for _f in _nz if _f > 1.0)
                            print(f"[GEO-sat] step={global_step} tier={_tier} saturated {len(_nz)}/{_T} frames "
                                  f"(over={_over}/under={len(_nz)-_over}) f={_nz}", flush=True)
                        _fac = torch.tensor(_facs, device=_lat.device, dtype=_lat.dtype).view(1, 1, _T, 1, 1)
                        _m = _lat.mean(dim=1, keepdim=True)
                        return (_lat - _m) * _fac + _m
                    # ── Stage2 err-bank injection (warp + prev_short). Samples a banked REAL y_error at full-res:
                    #    warp += full window (NOTE: _geo_warp_lat is ALREADY visibility-noised in materialize -> this is
                    #    after-noise, not err-then-noise; an accepted second-order difference, see EXP README); prev_short += last frame
                    #    BEFORE its short_tier_noise (proper err-then-noise). Tracks max depth -> per_item_depth (banking +1).
                    if (
                        bool(getattr(args.training_config, "use_error_recycling", False))
                        and bool(getattr(args.training_config, "allow_error_recycling_stage2", False))
                        and bool(getattr(args.training_config, "is_enable_stage2", False))
                        and recycle_vars is not None
                        and global_step > int(getattr(args.training_config, "buffer_warmup_iter", 50))
                    ):
                        _eb_tiers = set(getattr(_geo_cfg, "error_inject_tiers", []) or []) if _geo_cfg is not None else set()
                        _eb_prob = float(getattr(_geo_cfg, "error_inject_prob", 0.0)) if _geo_cfg is not None else 0.0
                        _eb_ybuf = getattr(recycle_vars, "y_error_buffer", None)
                        _eb_h, _eb_w = int(_geo_warp_lat.shape[-2]), int(_geo_warp_lat.shape[-1])
                        if (
                            _eb_tiers and _eb_prob > 0 and random.random() < _eb_prob
                            and _eb_ybuf is not None and (_eb_h, _eb_w) in _eb_ybuf
                        ):
                            from evoke.utils.utils_recycle_batch import sample_y_error_from_latent_buffer
                            _eb_depth = torch.zeros(_geo_warp_lat.shape[0], dtype=torch.long, device=_geo_warp_lat.device)
                            if "warp" in _eb_tiers:
                                _eb_e, _eb_d = sample_y_error_from_latent_buffer(
                                    args, recycle_vars, _geo_warp_lat, dtype=_geo_warp_lat.dtype, device=_geo_warp_lat.device,
                                )
                                _geo_warp_lat = _geo_warp_lat + _eb_e
                                _eb_depth = torch.maximum(_eb_depth, _eb_d.to(_eb_depth.device))
                            if "prev_short" in _eb_tiers:
                                # Sample at warp shape (full window) then take the LAST frame for the 1-frame prev_short.
                                _eb_e2, _eb_d2 = sample_y_error_from_latent_buffer(
                                    args, recycle_vars, _geo_warp_lat, dtype=_geo_prev_short_clean.dtype,
                                    device=_geo_prev_short_clean.device,
                                )
                                _geo_prev_short_clean = _geo_prev_short_clean + _eb_e2[:, :, -1:, :, :]
                                _eb_depth = torch.maximum(_eb_depth, _eb_d2.to(_eb_depth.device))
                            _geo_err_inject_depth = _eb_depth
                            if global_step < int(getattr(args.training_config, "buffer_warmup_iter", 50)) + 6 or global_step % 200 == 0:
                                print(f"[errbank-s2] step={global_step} inject tiers={sorted(_eb_tiers)} "
                                      f"depth={_eb_depth.tolist()}", flush=True)
                    if _st_enabled and "prev_short" in _st_targets:
                        # Per-tier override: prev_short_sigma_max (None → shared sigma_max). Independent of prefix.
                        _ps_lo = float(getattr(_st_noise_cfg, "sigma_min", 0.2))
                        _ps_hi_cfg = getattr(_st_noise_cfg, "prev_short_sigma_max", None)
                        _ps_hi = float(_ps_hi_cfg) if _ps_hi_cfg is not None else float(getattr(_st_noise_cfg, "sigma_max", 0.6))
                        _ps_sigma = (
                            torch.rand(1, device=_geo_prev_short_clean.device) * (_ps_hi - _ps_lo)
                            + _ps_lo
                        ).view(1, 1, 1, 1, 1).to(dtype=_geo_prev_short_clean.dtype)
                        _ps_eps = torch.randn_like(_geo_prev_short_clean)
                        _geo_prev_short = (
                            _ps_sigma * _ps_eps + (1 - _ps_sigma) * _geo_prev_short_clean
                        )
                        # same sigma + same noise eps on the CLEAN (pre-errbank) prev_short.
                        if _rtc_on:
                            _geo_prev_short_teacher = (
                                _ps_sigma * _ps_eps + (1 - _ps_sigma) * _geo_prev_short_pre_eb
                            )
                    else:
                        _geo_prev_short = _geo_prev_short_clean
                        if _rtc_on:
                            _geo_prev_short_teacher = _geo_prev_short_pre_eb
                    # saturate the DEGRADED warp + prev_short (rebinds, no in-place → the Option B
                    # teacher copies _geo_warp_lat_clean_src / _geo_prev_short_teacher stay un-saturated).
                    _geo_warp_lat = _geo_saturate(_geo_warp_lat, "warp")
                    _geo_prev_short = _geo_saturate(_geo_prev_short, "prev_short")
                    # short tier physical layout: [prefix | warp | prev_short] — matches the RoPE position order
                    # (prefix < warp < prev_short < noise). prev_short is the frame CLOSEST to noise.
                    _geo_short_new = torch.cat([_geo_prefix_latent, _geo_warp_lat, _geo_prev_short], dim=2)
                    # CLEAN short tier for the DM teacher: same prefix, pre-errbank warp, clean prev_short.
                    # Same physical layout/RoPE order -> shares all the index/rope logic below; only the history
                    # CONTENT differs (no errbank). None unless toggled.
                    if _rtc_on:
                        _geo_short_clean = torch.cat(
                            [_geo_prefix_latent, _geo_warp_lat_clean_src, _geo_prev_short_teacher], dim=2
                        )
                    # RoPE index assignment: prefix_idx_mode + rope_alignment + warp_rope_mode.
                    _geo_prefix_idx_mode = (
                        str(getattr(_geo_cfg, "prefix_idx_mode", "zero")) if _geo_cfg is not None else "zero"
                    )
                    _geo_rope_align = bool(getattr(_geo_cfg, "rope_alignment", True)) if _geo_cfg is not None else True
                    _warp_rope_mode = (
                        str(getattr(_geo_cfg, "warp_rope_mode", "overlap_noise")) if _geo_cfg is not None else "overlap_noise"
                    )

                    if _warp_rope_mode == "before_prev_short":
                        # Plan 16: warp takes the prev_short rope slot [noise[0]-1 .. +W-1]; prev_short pushed to
                        # noise[0]-1 (= base+W); noise += W; prefix = 0. RoPE order: prefix < warp < prev_short < noise.
                        assert not _geo_rope_align, "warp_rope_mode='before_prev_short' requires rope_alignment=False."
                        assert _geo_prefix_idx_mode == "zero", "warp_rope_mode='before_prev_short' requires prefix_idx_mode='zero'."
                        _W_p16 = int(args.training_config.latent_window_size[0])
                        _base_short_idx = indices_hidden_states[:, :1].to(accelerator.device) - 1   # [B,1] base noise[0]-1
                        _geo_warp_idx = (
                            _base_short_idx + torch.arange(_W_p16, device=accelerator.device).unsqueeze(0)
                        ).long()
                        _geo_prev_short_idx = (_base_short_idx + _W_p16).long()   # prev_short -> noise[0]-1 after shift
                        indices_hidden_states = indices_hidden_states + _W_p16
                        _geo_prefix_idx = torch.zeros(
                            indices_hidden_states.shape[0], 1, dtype=torch.long, device=accelerator.device
                        )
                    elif _warp_rope_mode == "before_prev_mid":
                        raise NotImplementedError(
                            "warp_rope_mode='before_prev_mid' (Plan 22) is not ported to this codebase; "
                            "use 'overlap_noise' or 'before_prev_short'."
                        )
                    else:
                        # overlap_noise (default): prefix_idx_mode + rope_alignment as before.
                        if _geo_prefix_idx_mode == "adjacent":
                            # prefix.idx = noise.idx[0] - 1, clamped at 0.
                            if _geo_rope_align:
                                _geo_prefix_idx = (
                                    indices_hidden_states[:, :1].to(accelerator.device) - 1
                                ).clamp(min=0).long()
                            else:
                                _W_offset_pre = int(args.training_config.latent_window_size[0])
                                _geo_prefix_idx = (
                                    indices_hidden_states[:, :1].to(accelerator.device) + _W_offset_pre - 1
                                ).clamp(min=0).long()
                        else:
                            # Default: prefix.idx = 0.
                            _geo_prefix_idx = torch.zeros(
                                indices_hidden_states.shape[0], 1, dtype=torch.long, device=accelerator.device
                            )
                        # rope_alignment=True: warp idx == target idx (overlap). False: target idx shifted by W.
                        if _geo_rope_align:
                            _geo_warp_idx = indices_hidden_states.to(accelerator.device)
                        else:
                            _geo_warp_idx = indices_hidden_states.to(accelerator.device).clone()
                            _W_offset = int(args.training_config.latent_window_size[0])
                            indices_hidden_states = indices_hidden_states + _W_offset
                        # overlap_noise: prev_short keeps its original rope index.
                        _geo_prev_short_idx = _geo_prev_short_idx_old.to(accelerator.device)
                    # short tier indices: [prefix | warp | prev_short] (same physical order as the latent).
                    _geo_short_idx_new = torch.cat([_geo_prefix_idx, _geo_warp_idx, _geo_prev_short_idx], dim=1)
                    if accelerator.is_main_process and global_step < 3:
                        print(
                            f"[GEO-RoPE TRAIN] rope_align={_geo_rope_align} "
                            f"prefix_idx={_geo_prefix_idx[0].tolist()} (mode={_geo_prefix_idx_mode}) "
                            f"warp_idx={_geo_warp_idx[0].tolist()[:5]}... "
                            f"target_idx={indices_hidden_states[0].tolist()[:5]}...",
                            flush=True,
                        )

                    # Build visibility mask: [prefix(ones) | warp(from renderer) | prev_short(ones, clean memory)]
                    # — same physical order as the latent / index tensors.
                    _H_lat = _geo_visibility_mask.shape[-2]
                    _W_lat = _geo_visibility_mask.shape[-1]
                    _geo_prefix_vis = torch.ones(
                        _geo_visibility_mask.shape[0], 1, 1, _H_lat, _W_lat,
                        device=accelerator.device, dtype=torch.float32,
                    )
                    _geo_prev_short_vis = torch.ones_like(_geo_prefix_vis)
                    _geo_history_visible_mask_short = torch.cat(
                        [_geo_prefix_vis, _geo_visibility_mask, _geo_prev_short_vis], dim=2
                    )

                    # Replace short tier (mid/long keep prepare_stage1_clean_input_from_latents values).
                    latents_history_short = _geo_short_new
                    indices_latents_history_short = _geo_short_idx_new
                    # warp frames = W (latent_window_size); prev_short frames = 1 (trailing). Layout
                    # [prefix | warp(W) | prev_short(1)]: transformer warp_mlp's + per-stage compresses ONLY the
                    # middle W warp frames; prefix(leading) + prev_short(trailing 1) = uncompressed anchor.
                    _geo_train_attention_kwargs = {
                        "history_visible_token_threshold": float(getattr(getattr(args.model_config, "geometric_state", None), "visible_token_threshold", 0.1) or 0.1),
                        "geo_warp_frames": int(args.training_config.latent_window_size[0]),
                        "geo_prev_short_frames": 1,
                        # NOTE: STAGE0-ONLY WARP (default False): the generator strips warp on the fine pyramid stages (i_s>0); consumed only by the generator rollout
                        #   (inference_with_trajectory_stage2), the single teacher/critic forward is unaffected = warp on all stages.
                        "geo_warp_stage0_only": bool(getattr(getattr(args.model_config, "geometric_state", None), "warp_stage0_only", False)),
                    }
                    if _stage2_warp_mode != "fixed_mem":
                        _geo_train_attention_kwargs["stage2_warp_compression_mode"] = _stage2_warp_mode
                    # warp_rope_noise_center_align (fixed_mem only): center coarse-stage noise rope into the
                    # full-res warp frame. On = full centering directly (no interpolation/ramp knob).
                    if _geo_nc_on:
                        _geo_train_attention_kwargs["warp_rope_noise_center_align"] = True
                        if accelerator.is_main_process and global_step < 5:
                            print(f"[GEO-noise-center TRAIN] step={global_step} ON (full centering)", flush=True)
                    # [STAGE0-ONLY] one-off confirmation that the flag reached the generator attention_kwargs; the actual strip happens in
                    # inference_with_trajectory_stage2 (i_s>0), see the "[stage0-only] i_s=..." print. the single teacher/critic
                    # forward does not go through the pyramid loop -> no strip = full warp (by design).
                    if accelerator.is_main_process and global_step < 3:
                        _s0_dbg = bool(_geo_train_attention_kwargs.get("geo_warp_stage0_only", False))
                        print(
                            f"[STAGE0-ONLY TRAIN] step={global_step} geo_warp_stage0_only={_s0_dbg} "
                            f"geo_warp_frames={_geo_train_attention_kwargs.get('geo_warp_frames')} "
                            f"geo_prev_short_frames={_geo_train_attention_kwargs.get('geo_prev_short_frames')} "
                            f"-> student strips warp on the fine stages (i_s>0); teacher/critic=full warp",
                            flush=True,
                        )

                    # When raw_sink is active, use the clean (pre-noise) prefix as nearby_sink anchor.
                    if args.training_config.use_raw_sink_frames and nearby_sink_latents is not None:
                        nearby_sink_latents = _geo_prefix_latent_clean.clone()
                        _geo_train_nearby_sink_indices = torch.zeros(
                            1, 1, dtype=torch.long, device=accelerator.device
                        )
                # GEO per-mode dispatch: t2v clears all visual tokens; i2v forces mid/long to image-only
                # (no prior-chunk history — i2v = "from one image"); full_geo unchanged.
                if _use_geo_train:
                    if _geo_condition_mode == "t2v":
                        latents_history_short = None
                        latents_history_mid = None
                        latents_history_long = None
                        indices_latents_history_short = None
                        indices_latents_history_mid = None
                        indices_latents_history_long = None
                        sink_latents = None
                        nearby_sink_latents = None
                        _geo_history_visible_mask_short = None
                        _geo_history_visible_mask_mid = None
                        _geo_history_visible_mask_long = None
                        _geo_train_nearby_sink_indices = None
                    # full_geo: no change. i2v: handled below (short tier = [prefix|single-src warp|prefix-as-prev_short]
                    # built above; mid/long FORCED to invisible here regardless of choice_idx → no real-history leak).

                    # i2v: FORCE mid/long to pure invisible noise (sigma_invisible) — i2v is "from one image", so
                    # there is NO prior-chunk history. UNCONDITIONAL for i2v (NOT gated on geo_invisible_history_noise):
                    # i2v semantics require no-history regardless, and this overwrites the section's REAL history when
                    # choice_idx>0 (fixing the leak). Mirrors inference i2v chunk0.
                    if _geo_condition_mode == "i2v":
                        _sig_inv = float(getattr(_geo_cfg, "warp_noise_sigma_invisible", 0.8)) if _geo_cfg is not None else 0.8
                        if latents_history_long is not None:
                            latents_history_long = _sig_inv * torch.randn_like(latents_history_long)
                        if latents_history_mid is not None:
                            latents_history_mid = _sig_inv * torch.randn_like(latents_history_mid)

                    # GEO-native train-only aug: extend short_tier_noise to the mid/long history tiers. Kept separate from the
                    # original corrupt_history, which runs after the short-tier override and would double-noise the assembled
                    # [prefix|warp|prev_short] tier, re-noise warp and fight geo_invisible_history_noise on mid/long. Same
                    # sigma*randn+(1-sigma)*clean as prefix/prev_short, gated per tier via target_tiers. Train-only.
                    # Default target_tiers=[prefix,prev_short] -> no-op (opt-in).
                    _mlt_cfg = getattr(args.model_config, "geometric_state", None)
                    _mlt_cfg = getattr(_mlt_cfg, "short_tier_noise", None) if _mlt_cfg is not None else None
                    _mlt_en = bool(getattr(_mlt_cfg, "enabled", False)) if _mlt_cfg is not None else False
                    _mlt_tg = list(getattr(_mlt_cfg, "target_tiers", []) or []) if _mlt_cfg is not None else []
                    if _mlt_en and ("mid" in _mlt_tg or "long" in _mlt_tg):
                        _mlt_lo = float(getattr(_mlt_cfg, "sigma_min", 0.0))
                        # Per-tier override: mid_long_sigma_max (None → shared sigma_max). Lets mid/long sit a
                        # notch below prev_short while sharing the same sigma·randn+(1-sigma)·clean formula.
                        _mlt_hi_cfg = getattr(_mlt_cfg, "mid_long_sigma_max", None)
                        _mlt_hi = float(_mlt_hi_cfg) if _mlt_hi_cfg is not None else float(getattr(_mlt_cfg, "sigma_max", 0.135))

                        def _geo_apply_tier_noise(_lat):
                            if _lat is None:
                                return _lat
                            _sig = (
                                torch.rand(1, device=_lat.device) * (_mlt_hi - _mlt_lo) + _mlt_lo
                            ).view(1, 1, 1, 1, 1).to(dtype=_lat.dtype)
                            return _sig * torch.randn_like(_lat) + (1 - _sig) * _lat

                        if "mid" in _mlt_tg:
                            latents_history_mid = _geo_apply_tier_noise(latents_history_mid)
                        if "long" in _mlt_tg:
                            latents_history_long = _geo_apply_tier_noise(latents_history_long)
                        if accelerator.is_main_process and global_step < 3:
                            print(
                                f"[ShortTier-Noise TRAIN] mid/long tier noise from "
                                f"[{_mlt_lo:.3f}, {_mlt_hi:.3f}] (targets={_mlt_tg})",
                                flush=True,
                            )

                    # mid/long: snapshot CLEAN (post short_tier_noise, pre-saturation) for the
                    # Option B teacher, THEN saturate the degraded mid/long for student+critic. Clean clone only on
                    # the _rtc_on path; when saturation is off these equal the degraded tensors (no-op for teacher).
                    _geo_mid_clean = latents_history_mid.clone() if (_rtc_on and latents_history_mid is not None) else None
                    _geo_long_clean = latents_history_long.clone() if (_rtc_on and latents_history_long is not None) else None
                    latents_history_mid = _geo_saturate(latents_history_mid, "mid")
                    latents_history_long = _geo_saturate(latents_history_long, "long")

                # [v2v ODE dump] grab uttid/prompt before `del batch` (the dump block runs after `del batch`, when batch is already released)
                _dump_uttid = None
                _dump_prompt_raw = None
                if args.training_config.is_dump_ode_traj and isinstance(batch, dict):
                    _dump_uttid = batch["uttid"][0] if "uttid" in batch else None
                    _dump_prompt_raw = batch["prompt"][0] if "prompt" in batch else None

                batch = None
                del batch

                # Offload VAE/T5 to CPU for offline stage-1 to save VRAM; keep on GPU for online mode.
                _can_offload_vae_t5 = (
                    not args.data_config.use_stage3_dataset
                    and (args.training_config.offload or args.data_config.use_stage1_dataset)
                    and not args.data_config.use_multi_dataset
                )
                if _can_offload_vae_t5:
                    if vae is not None:
                        vae.to("cpu", non_blocking=True)
                    if text_encoder is not None:
                        text_encoder.to("cpu", non_blocking=True)
                    free_memory()

                # Set NULL Text
                if prompt_embeds is not None:
                    dropout_mask = (
                        torch.rand(prompt_embeds.shape[0], device=prompt_embeds.device)
                        < args.data_config.caption_dropout_p
                    )
                    prompt_embeds[dropout_mask] = 0

                # Move training tensors to device (may be None in HE mode; guarded by if-not-None).
                if not args.training_config.is_train_dmd and not args.training_config.is_use_ode_regression:
                    model_input = model_input.to(device=accelerator.device, dtype=weight_dtype, non_blocking=True)
                    indices_hidden_states = indices_hidden_states.to(accelerator.device, non_blocking=True)
                    if indices_latents_history_short is not None:
                        indices_latents_history_short = indices_latents_history_short.to(
                            accelerator.device, non_blocking=True
                        )
                    if indices_latents_history_mid is not None:
                        indices_latents_history_mid = indices_latents_history_mid.to(
                            accelerator.device, non_blocking=True
                        )
                    if indices_latents_history_long is not None:
                        indices_latents_history_long = indices_latents_history_long.to(
                            accelerator.device, non_blocking=True
                        )
                    if latents_history_short is not None:
                        latents_history_short = latents_history_short.to(
                            device=accelerator.device, dtype=weight_dtype, non_blocking=True
                        )
                    if latents_history_mid is not None:
                        latents_history_mid = latents_history_mid.to(
                            device=accelerator.device, dtype=weight_dtype, non_blocking=True
                        )
                    if latents_history_long is not None:
                        latents_history_long = latents_history_long.to(
                            device=accelerator.device, dtype=weight_dtype, non_blocking=True
                        )
                    if sink_latents is not None:
                        sink_latents = sink_latents.to(
                            device=accelerator.device, dtype=weight_dtype, non_blocking=True
                        )
                    if nearby_sink_latents is not None:
                        nearby_sink_latents = nearby_sink_latents.to(
                            device=accelerator.device, dtype=weight_dtype, non_blocking=True
                        )
                if prompt_embeds is not None:
                    prompt_embeds = prompt_embeds.to(accelerator.device, non_blocking=True)

                # Prepare final data for training
                use_clean_input = False
                per_item_depth = None  # set by stage1 prepare; None for stage2/dmd/ode (recycling is stage1-only)
                # Preserve the stage2 warp short-tier ([prefix|warp|prev_short]) + mid/long
                # history latents BEFORE the is_train_dmd block below nulls latents_history_*, so they can be
                # handed to the DMD generator/critic for warp conditioning. None when GEO inactive (byte-identical).
                _geo_keep_hist_short = latents_history_short
                _geo_keep_hist_mid = latents_history_mid
                _geo_keep_hist_long = latents_history_long
                # clean (pre-errbank) short tier, built only when recycle_teacher_clean is on.
                # Same device/dtype as _geo_keep_hist_short (DMD path skips the .to() block above; both are
                # built from the same sources). None on every other path -> teacher falls back to degraded.
                _geo_keep_hist_short_clean = locals().get("_geo_short_clean", None)
                # clean (pre-saturation) mid/long for the Option B teacher. None unless _rtc_on
                # (then == _geo_keep_hist_mid/long when saturation is off -> teacher unchanged).
                _geo_keep_hist_mid_clean = locals().get("_geo_mid_clean", None)
                _geo_keep_hist_long_clean = locals().get("_geo_long_clean", None)
                # -- [v2v ODE dump] the teacher collects the pyramid trajectory under warp conditions + saves it (no training, continue) --
                # reuses pipeline_evoke_ode.stage2_sample (which already collects ode_stages_tensor + accepts the full set of warp conditions);
                # saves the "already assembled tiers" directly (short=[prefix|warp|prev_short] + mid/long) + indices + kwargs -> zero reconstruction drift on the training side.
                if args.training_config.is_dump_ode_traj:
                    from evoke.pipelines.pipeline_evoke_ode import EvokePipeline as _EvokeODEPipeline
                    from diffusers.utils.torch_utils import randn_tensor as _randn_tensor

                    if _dump_ode_pipe is None:
                        _unwrapped_tf = accelerator.unwrap_model(transformer)
                        _unwrapped_tf.eval()
                        _dump_ode_pipe = _EvokeODEPipeline(
                            tokenizer=tokenizer,
                            text_encoder=text_encoder,
                            vae=vae,
                            scheduler=noise_scheduler_copy,
                            transformer=_unwrapped_tf,
                        )
                        os.makedirs(args.training_config.dump_ode_traj_out, exist_ok=True)
                        if accelerator.is_main_process:
                            logger.info(f"[dump_ode_traj] → {args.training_config.dump_ode_traj_out}")

                    # [v2v ODE dump] the uttid carries the epoch -> multiple epochs = several different random time windows per clip, each unique and not skipped (multi-crop)
                    _uttid = (
                        f"{_dump_uttid}_e{epoch}" if _dump_uttid is not None
                        else f"{accelerator.process_index}_{step}_e{epoch}"
                    )
                    _out_path = os.path.join(args.training_config.dump_ode_traj_out, f"{_uttid}.pt")
                    if os.path.exists(_out_path):
                        progress_bar.update(1)
                        continue

                    assert (
                        _geo_keep_hist_short is not None
                        and _geo_keep_hist_mid is not None
                        and _geo_keep_hist_long is not None
                    ), "[dump_ode_traj] warp tiers must not be None (needs complete v2v GEO; check use_geometric_state + geo_condition_*_ratio=0)"
                    # calling stage2_sample directly: fill in the pipe state that only generate() would set (the CFG gate self._guidance_scale / the interrupt flag)
                    _dump_ode_pipe._guidance_scale = args.training_config.dump_ode_guidance_scale
                    _dump_ode_pipe._interrupt = False
                    _pbar = _dump_ode_pipe.progress_bar(total=sum(args.training_config.dump_ode_steps_per_stage))

                    with torch.no_grad():
                        _dump_noise = _randn_tensor(
                            noisy_model_input_shape,
                            generator=None,
                            device=accelerator.device,
                            dtype=torch.float32,
                        )
                        _, _ode_stages = _dump_ode_pipe.stage2_sample(
                            is_first_section=True,
                            latents=_dump_noise,
                            stage2_num_stages=args.training_config.stage2_num_stages,
                            stage2_num_inference_steps_list=args.training_config.dump_ode_steps_per_stage,
                            prompt_embeds=prompt_embeds,
                            negative_prompt_embeds=negative_prompt_embeds,
                            guidance_scale=args.training_config.dump_ode_guidance_scale,
                            indices_hidden_states=indices_hidden_states,
                            indices_latents_history_short=indices_latents_history_short,
                            indices_latents_history_mid=indices_latents_history_mid,
                            indices_latents_history_long=indices_latents_history_long,
                            latents_history_short=_geo_keep_hist_short,
                            latents_history_mid=_geo_keep_hist_mid,
                            latents_history_long=_geo_keep_hist_long,
                            attention_kwargs=_geo_train_attention_kwargs,
                            cam_Ks=_stashed_target_pose_Ks,
                            cam_c2ws=_stashed_target_pose_c2ws,
                            cam_base_h=args.data_config.single_height,
                            cam_base_w=args.data_config.single_width,
                            cam_strategy="scale_ks",
                            device=accelerator.device,
                            transformer_dtype=weight_dtype,
                            scheduler_type=args.training_config.dump_ode_scheduler_type,
                            use_dynamic_shifting=args.training_config.use_dynamic_shifting,
                            time_shift_type=args.training_config.time_shift_type,
                            generator=None,
                            progress_bar=_pbar,
                        )
                    _pbar.close()

                    # trajectory point selection (reuses the get_ode-pairs convention: K=4 intermediate points per window + [-1] as the clean endpoint; noise_pred dropped)
                    _section_proc = []
                    for _item in _ode_stages:
                        _n = int(_item["timesteps"].shape[0])
                        _k = max(1, min(4, _n - 1))
                        _hi = max(0, _n - 2)
                        _idxs = [0] if _k == 1 else sorted({round(_i * _hi / (_k - 1)) for _i in range(_k)})
                        _section_proc.append(
                            {
                                "latents": _item["latents"][_idxs + [-1]],
                                "timesteps": _item["timesteps"][_idxs],
                            }
                        )
                    _ode_latents_to_save = [_section_proc]  # v1: a single section

                    def _cpu_or_none(_t):
                        return _t.detach().cpu() if _t is not None else None

                    _geo_save = {
                        "latents_history_short": _cpu_or_none(_geo_keep_hist_short),
                        "latents_history_mid": _cpu_or_none(_geo_keep_hist_mid),
                        "latents_history_long": _cpu_or_none(_geo_keep_hist_long),
                        "indices_hidden_states": _cpu_or_none(indices_hidden_states),
                        "indices_latents_history_short": _cpu_or_none(indices_latents_history_short),
                        "indices_latents_history_mid": _cpu_or_none(indices_latents_history_mid),
                        "indices_latents_history_long": _cpu_or_none(indices_latents_history_long),
                        "attention_kwargs": _geo_train_attention_kwargs,
                        # [v2v ODE plucker] save the camera poses (Ks/c2ws are small) + base/strategy -> the training side rebuilds plk per stage
                        "cam_Ks": _cpu_or_none(_stashed_target_pose_Ks),
                        "cam_c2ws": _cpu_or_none(_stashed_target_pose_c2ws),
                        "cam_base_h": args.data_config.single_height,
                        "cam_base_w": args.data_config.single_width,
                        "cam_strategy": "scale_ks",
                    }
                    torch.save(
                        {
                            "latent_window_size": latent_window_size,
                            "prompt_raw": _dump_prompt_raw,
                            "prompt_embed": prompt_embeds.detach().cpu(),
                            "ode_latents": _ode_latents_to_save,
                            "geo": _geo_save,
                        },
                        _out_path,
                    )
                    _dump_ode_count += 1
                    progress_bar.update(1)
                    if accelerator.is_main_process and _dump_ode_count % 10 == 0:
                        logger.info(f"[dump_ode_traj] saved {_dump_ode_count} (last={_uttid})")
                    if (
                        args.training_config.dump_ode_max_samples
                        and _dump_ode_count >= args.training_config.dump_ode_max_samples
                    ):
                        logger.info(
                            f"[dump_ode_traj] rank {accelerator.process_index} reached the limit of "
                            f"{args.training_config.dump_ode_max_samples}, exiting."
                        )
                        return
                    continue

                if args.training_config.is_train_dmd or args.training_config.is_use_ode_regression:
                    noisy_model_input_list = None
                    sigmas_list = None
                    timesteps_list = None
                    targets_list = None
                    latents_history_short = None
                    latents_history_mid = None
                    latents_history_long = None
                else:
                    if args.training_config.is_enable_stage2:
                        (
                            noisy_model_input_list,
                            sigmas_list,
                            timesteps_list,
                            targets_list,
                            latents_history_short,
                            latents_history_mid,
                            latents_history_long,
                        ) = prepare_stage2_noise_input(
                            args=args,
                            scheduler=noise_scheduler_copy,
                            latents=model_input,
                            pyramid_stage_num=args.training_config.stage2_num_stages,
                            stage2_sample_ratios=args.training_config.stage2_sample_ratios,
                            latents_history_short=latents_history_short,
                            latents_history_mid=latents_history_mid,
                            latents_history_long=latents_history_long,
                            latent_window_size=latent_window_size,
                            is_navit_pyramid=args.training_config.is_navit_pyramid,
                            is_efficient_sample=args.training_config.efficient_sample,
                        )
                        # Stage2 err-bank: carry warp/prev_short injected depth into banking (output error -> depth+1).
                        if _geo_err_inject_depth is not None:
                            per_item_depth = _geo_err_inject_depth
                    else:
                        (
                            noisy_model_input_list,
                            sigmas_list,
                            timesteps_list,
                            targets_list,
                            latents_history_short,
                            latents_history_mid,
                            latents_history_long,
                            use_clean_input,
                            per_item_depth,
                        ) = prepare_stage1_noise_input(
                            args=args,
                            model_input=model_input,
                            noise_scheduler=noise_scheduler_copy,
                            recycle_vars=recycle_vars,
                            latents_history_short=latents_history_short,
                            latents_history_mid=latents_history_mid,
                            latents_history_long=latents_history_long,
                            latent_window_size=latent_window_size,
                            is_keep_x0=True,
                            geo_protect_short_frames=_geo_protect_short_frames,
                        )

            # Build Plucker embeddings aligned to noisy_model_input_list. Built when EITHER the camera_control
            # AdaLN path OR the GEO additive Plucker path is on (both consume cam_plucker_emb in forward).
            cam_plucker_emb_list = None
            _cam_ctrl_cfg = getattr(args.model_config, "camera_control", None)
            _cam_ctrl_on = _cam_ctrl_cfg is not None and _cam_ctrl_cfg.enabled
            _geo_plk_on = bool(
                getattr(getattr(args.model_config, "geometric_state", None), "geo_warp_plucker_enabled", False)
            )
            if (
                (_cam_ctrl_on or _geo_plk_on)
                and _stashed_target_pose_Ks is not None
                and _stashed_target_pose_c2ws is not None
                and noisy_model_input_list is not None
            ):
                from evoke.modules.camera_control import prepare_cam_plucker_for_list
                # Use camera_control's settings when present; otherwise fall back to data-config defaults
                # (the GEO-only case where camera_control is absent/disabled).
                if _cam_ctrl_on:
                    _base_h = _cam_ctrl_cfg.base_height_pix or args.data_config.single_height
                    _base_w = _cam_ctrl_cfg.base_width_pix or args.data_config.single_width
                    _strategy = _cam_ctrl_cfg.pc_resolution_strategy
                else:
                    _base_h = args.data_config.single_height
                    _base_w = args.data_config.single_width
                    _strategy = "scale_ks"
                _Ks_f = _stashed_target_pose_Ks.to(accelerator.device, dtype=torch.float32)
                _c2ws_f = _stashed_target_pose_c2ws.to(accelerator.device, dtype=torch.float32)

                def _build_cam_emb_for(item):
                    """Build Plucker embedding for a tensor or a NaViT pyramid inner list."""
                    if isinstance(item, list):
                        per_seg = prepare_cam_plucker_for_list(
                            item, _Ks_f, _c2ws_f,
                            base_height_pix=_base_h, base_width_pix=_base_w,
                            strategy=_strategy,
                        )
                        return [e.to(dtype=weight_dtype) for e in per_seg]
                    # Tensor case: wrap in single-element list, then unwrap.
                    e_list = prepare_cam_plucker_for_list(
                        [item], _Ks_f, _c2ws_f,
                        base_height_pix=_base_h, base_width_pix=_base_w,
                        strategy=_strategy,
                    )
                    return e_list[0].to(dtype=weight_dtype)

                cam_plucker_emb_list = [_build_cam_emb_for(it) for it in noisy_model_input_list]

            with accelerator.accumulate(models_to_accumulate):
                # Predict the noise residual
                if not args.training_config.is_train_dmd and not args.training_config.is_use_ode_regression:
                    assert len(noisy_model_input_list) == len(sigmas_list) == len(timesteps_list) == len(targets_list)
                    logs = _flow_loss(
                        args=args,
                        accelerator=accelerator,
                        lr_scheduler=lr_scheduler,
                        transformer=transformer,
                        prompt_embeds=prompt_embeds,
                        prompt_attention_masks=None,
                        noisy_model_input_list=noisy_model_input_list,
                        sigmas_list=sigmas_list,
                        timesteps_list=timesteps_list,
                        targets_list=targets_list,
                        indices_hidden_states=indices_hidden_states,
                        indices_latents_history_short=indices_latents_history_short,
                        indices_latents_history_mid=indices_latents_history_mid,
                        indices_latents_history_long=indices_latents_history_long,
                        latents_history_short=latents_history_short,
                        latents_history_mid=latents_history_mid,
                        latents_history_long=latents_history_long,
                        sink_latents=sink_latents,
                        nearby_sink_latents=nearby_sink_latents,
                        nearby_sink_indices=_geo_train_nearby_sink_indices if _use_geo_train else None,
                        # GEO visibility filter per tier.
                        history_visible_mask_short=_geo_history_visible_mask_short,
                        history_visible_mask_mid=_geo_history_visible_mask_mid,
                        history_visible_mask_long=_geo_history_visible_mask_long,
                        attention_kwargs=_geo_train_attention_kwargs,
                        recycle_vars=recycle_vars,
                        global_step=global_step,
                        noise_scheduler_copy=noise_scheduler_copy,
                        use_clean_input=use_clean_input,
                        per_item_depth=per_item_depth,
                        cam_plucker_emb_list=cam_plucker_emb_list,
                        target_visible_mask=_geo_visibility_mask,   # masked-region loss weighting + visible/invisible per-region logging
                    )
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                elif args.training_config.is_use_ode_regression and args.training_config.is_only_ode_regression:
                    if vae is not None:
                        vae.to("cpu", non_blocking=True)
                    if text_encoder is not None:
                        text_encoder.to("cpu", non_blocking=True)

                    _, logs = _ode_regression_loss(
                        args=args,
                        accelerator=accelerator,
                        transformer=transformer,
                        scheduler=noise_scheduler_copy,
                        noise=torch.randn(noisy_model_input_shape, device=accelerator.device, dtype=weight_dtype),
                        weight_dtype=weight_dtype,
                        # For Stage 1
                        is_keep_x0=True,
                        history_sizes=args.training_config.history_sizes,
                        # For Stage 2
                        stage2_num_stages=args.training_config.stage2_num_stages,
                        # For ODE Main
                        last_step_only=args.training_config.dmd_last_step_only,
                        use_dynamic_shifting=args.training_config.use_dynamic_shifting,
                        time_shift_type=args.training_config.time_shift_type,
                        is_backward_grad=True,
                        ode_regression_weight=args.training_config.ode_regression_weight,
                        ode_latents=ode_latents,
                        ode_prompt_embeds=ode_prompt_embeds,
                        # [v2v ODE] warp pass-through (t2v .pt -> None/False -> bit-compatible)
                        gt_all_data=ode_gt_all_data,
                        attention_kwargs=ode_attention_kwargs,
                        is_use_gt_history=(ode_gt_all_data is not None),
                        cam_Ks=ode_cam_Ks,
                        cam_c2ws=ode_cam_c2ws,
                        cam_base_h=ode_cam_base_h,
                        cam_base_w=ode_cam_base_w,
                        cam_strategy=ode_cam_strategy,
                        ode_num_latent_sections_min=args.training_config.ode_num_latent_sections_min,
                        ode_num_latent_sections_max=args.training_config.ode_num_latent_sections_max,
                        # For Dynamic Num Sections
                        ode_dynamic_alpha=args.training_config.ode_dynamic_alpha,
                        ode_dynamic_beta=args.training_config.ode_dynamic_beta,
                        ode_dynamic_sample_type=args.training_config.ode_dynamic_sample_type,
                        global_step=global_step,
                        ode_dynamic_step=args.training_config.ode_dynamic_step,
                    )
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                else:
                    TRAIN_GENERATOR = global_step % args.training_config.dfake_gen_update_ratio == 0
                    # -- train only the critic for the first K steps: student params do not move at all --
                    #   This cannot be done with TRAIN_GENERATOR=False: that also skips the student's rollout forward, leaving
                    #     _sf_rollout_shared empty, so the critic re-rolls warp-free and fits a distribution the student never
                    #     produces. Instead keep the forward and skip only the generator's backward, engine.step() and
                    #     lr_scheduler.step(). K is a config constant, identical on all ranks; K=0 restores the old behaviour.
                    _GEN_FROZEN = int(global_step) < int(getattr(args.training_config, "sf_gen_freeze_steps", 0) or 0)
                    # the two "run once on the first step" STU-SP structural self-checks inside gen_bwd (IPG bucket order / collective sequence number) must land on
                    #   **the first step that really does a backward**, otherwise they never get a chance to run during the freeze -> the self-checks are silently lost.
                    _GEN_BWD_DIAG_NOW = int(global_step) == int(
                        getattr(args.training_config, "sf_gen_freeze_steps", 0) or 0)
                    if _GEN_FROZEN and accelerator.is_main_process and global_step % max(
                            1, int(args.training_config.log_iters)) == 0:
                        print(f"[GEN-FROZEN] step={global_step}/{int(args.training_config.sf_gen_freeze_steps)} "
                              f"student params are not updated (the forward still runs so the critic can reuse it), only the critic is trained", flush=True)
                    # shared rollout container: filled by the generator step (detached), reused by the critic in the same iteration.
                    # the validator already forces dfake=1 when sharing -> every step has a generator rollout available to share.
                    # the sharing gate is extended to evoke_teacher+warp: under warp the critic must reuse the gen warp rollout
                    #   (dfake=1 is guaranteed by the validator -> every step has a gen rollout). warp-off evoke_teacher (the parent config) -> condition False -> None
                    #   -> the critic re-rolls on its own (warp-free, bit-identical). the evoke branch is unchanged.
                    _sf_rollout_shared = (
                        {} if (
                            ((_sf_evoke and bool(getattr(args.training_config, "sf_share_rollout", False)))
                             or (_sf_evoke_teacher and bool(getattr(args.training_config, "use_geometric_state", False))))
                            and TRAIN_GENERATOR
                        ) else None
                    )
                    USE_GAN = args.training_config.is_use_gan and global_step >= args.training_config.gan_start_step
                    USE_REWARD = (
                        args.training_config.is_use_reward_model
                        and global_step >= args.training_config.reward_start_step
                    )
                    USE_GT_HIST = (
                        args.training_config.is_use_gt_history
                        and random.random() < args.training_config.use_gt_history_ratio
                    )

                    VISUALIZE = (
                        global_step % args.training_config.log_iters == 0 and not args.training_config.no_visualize
                    )
                    logs = {}

                    if accelerator.is_main_process:
                        if (
                            args.training_config.is_enable_cold_start
                            and global_step < args.training_config.cold_start_step
                        ):
                            num_rollout_sections = (
                                args.training_config.dmd_num_latent_sections_min + 1
                                if args.training_config.stage_cold_start_step is not None
                                and global_step >= args.training_config.stage_cold_start_step
                                else args.training_config.dmd_num_latent_sections_min
                            )
                        else:
                            num_rollout_sections = sample_dynamic_dmd_num_latent_sections(
                                min_sections=args.training_config.dmd_num_latent_sections_min,
                                max_sections=args.training_config.dmd_num_latent_sections_max,
                                dmd_dynamic_alpha=args.training_config.dmd_dynamic_alpha,
                                dmd_dynamic_beta=args.training_config.dmd_dynamic_beta,
                                dmd_dynamic_sample_type=args.training_config.dmd_dynamic_sample_type,
                                global_step=global_step,
                                dmd_dynamic_step=args.training_config.dmd_dynamic_step,
                                device=accelerator.device,
                            )
                        num_rollout_sections = torch.tensor(num_rollout_sections, device=accelerator.device)
                    else:
                        num_rollout_sections = torch.tensor(0, device=accelerator.device)

                    num_rollout_sections = broadcast(num_rollout_sections, from_process=0).item()
                    logs["num_rollout_sections"] = num_rollout_sections

                    # section curriculum: every rank looks up (N, W) **deterministically** from global_step, overriding the
                    # sampled value above (not placed inside the is_main_process guard -- if non-main ranks got W/ncif via broadcast-N-only they would
                    # diverge in shape -> NCCL hang, see review SHOULD-FIX#4). the tail window ncif = W*win overrides the static config in lockstep.
                    _sf_ncif = None
                    if _sf_any and bool(getattr(args.training_config, "sf_curriculum_enabled", False)):
                        num_rollout_sections, _sf_cur_W, _sf_cur_stage = sf_curriculum_lookup(
                            args.training_config.sf_curriculum_schedule, global_step)
                        _sf_ncif = int(_sf_cur_W) * int(latent_window_size)
                        logs["num_rollout_sections"] = num_rollout_sections
                        logs["sf_curriculum_stage"] = _sf_cur_stage
                        logs["sf_curriculum_W"] = _sf_cur_W
                        # instrumentation (acceptance 3: section -> N -> tail-window interval all consistent + budget advance cross-check)
                        if accelerator.is_main_process and global_step % args.training_config.log_iters == 0:
                            logger.info(
                                f"[SF-EVOKE curriculum] step={global_step} stage={_sf_cur_stage} "
                                f"N={num_rollout_sections} W={_sf_cur_W} ncif={_sf_ncif} "
                                f"tail_window_sections=[{num_rollout_sections - _sf_cur_W},{num_rollout_sections})")
                        # variable N: the data is materialized at the deepest N, sliced to the current N here
                        # (take the first N_cur sections adjacent to the prefix, matching the caption mapping of rollout gen_0..gen_{N-1},
                        # see review SHOULD-FIX#6; the prefix itself is N-independent and not sliced).
                        if _sf_prompt_embeds_list is not None:
                            assert len(_sf_prompt_embeds_list) >= num_rollout_sections, (
                                f"[SF-EVOKE] number of data-section prompts {len(_sf_prompt_embeds_list)} < curriculum N "
                                f"{num_rollout_sections} (does num_frames not cover the deepest N?)")
                            _sf_prompt_embeds_list = _sf_prompt_embeds_list[:num_rollout_sections]
                        # teacher-y is no longer head-sliced here: the window y is rebuilt in place by _generator_loss/_critic_loss
                        #   per the v2v window (anchor + VAE zero placeholder) (the is_evoke_teacher_score block in utils_evoke_post.py). only the full-clip y is passed through here.

                    # geometric state DMD conditioning: when GEO built the stage2 short-tier
                    # ([prefix|warp|prev_short]) + attention_kwargs above, hand them to the DMD generator/critic
                    # so the score-model forwards condition on warp (fixes empty-history RoPE reshape crash under
                    # restrict_self_attn). v2v single-chunk only; None when GEO inactive -> byte-identical path.
                    _gt_geo_attention_kwargs = None
                    _gt_geo_all_data = None
                    _gt_geo_all_data_teacher_clean = None  # DM-teacher tuple (clean short tier)
                    if (
                        _use_geo_train
                        and _geo_train_attention_kwargs is not None
                        and _geo_keep_hist_short is not None
                        and indices_latents_history_short is not None
                    ):
                        _gt_geo_attention_kwargs = _geo_train_attention_kwargs
                        # 9-tuple matching the gt_all_data layout consumed by inference_with_trajectory_stage2 / compute_kl_grad.
                        # Uses the PRESERVED warp short-tier latents (latents_history_* were nulled in the is_train_dmd block above;
                        # indices_* survive).
                        # elem8 is inference_with_trajectory_stage2's history_latents accumulator, sliced as
                        # history_latents[:, :, sum(history_sizes):] *after* the generated chunk is appended, so its frame count sets
                        # the offset and MUST equal sum(history_sizes)=19 for the slice to pick the freshly generated chunk
                        # (latent_window_size=9). An 11-frame short tier there gives slice[19:20] = 1 frame, so DMD and the critic
                        # denoising loss supervise only 1 of the 9 generated frames. Rebuild the real [long|mid|short_tail] context
                        # like the non-GEO gt path; only the 19-frame count matters, short_tail = the trailing prev_short frame.
                        assert _geo_keep_hist_mid is not None and _geo_keep_hist_long is not None, (
                            "[GEO-DMD] mid/long history tiers required to build the output-window accumulator"
                        )
                        _geo_dmd_hist_accum = torch.cat(
                            [_geo_keep_hist_long, _geo_keep_hist_mid, _geo_keep_hist_short[:, :, -1:]], dim=2
                        )
                        _hsz_sum = int(sum(args.training_config.history_sizes))
                        assert _geo_dmd_hist_accum.shape[2] == _hsz_sum, (
                            f"[GEO-DMD] output-window accumulator must be sum(history_sizes)={_hsz_sum} frames "
                            f"(long={_geo_keep_hist_long.shape[2]}+mid={_geo_keep_hist_mid.shape[2]}+short_tail=1), "
                            f"got {_geo_dmd_hist_accum.shape[2]}"
                        )
                        _gt_geo_all_data = (
                            None,
                            indices_hidden_states,
                            indices_latents_history_short,
                            indices_latents_history_mid,
                            indices_latents_history_long,
                            _geo_keep_hist_short,
                            _geo_keep_hist_mid,
                            _geo_keep_hist_long,
                            _geo_dmd_hist_accum,
                        )
                        # Asymmetric DM teacher: the same tuple except the history CONTENT is the clean copy -- elem5 short tier =
                        # pre-errbank/pre-saturation warp+prev_short, elem6/elem7 mid/long = pre-saturation. Indices unchanged;
                        # elem8's content is sliced off. Built only when recycle_teacher_clean produced the clean tier, else None
                        # and compute_kl_grad falls back to the degraded tuple.
                        if _geo_keep_hist_short_clean is not None:
                            _t_mid = _geo_keep_hist_mid_clean if _geo_keep_hist_mid_clean is not None else _geo_keep_hist_mid
                            _t_long = _geo_keep_hist_long_clean if _geo_keep_hist_long_clean is not None else _geo_keep_hist_long
                            _gt_geo_all_data_teacher_clean = (
                                None,
                                indices_hidden_states,
                                indices_latents_history_short,
                                indices_latents_history_mid,
                                indices_latents_history_long,
                                _geo_keep_hist_short_clean,
                                _t_mid,
                                _t_long,
                                _geo_dmd_hist_accum,
                            )

                    if args.data_config.use_stage3_dataset:
                        prompt_raws = text_prompt_raws
                        prompt_embeds = text_prompt_embeds

                    if TRAIN_GENERATOR:
                        extras_list = []

                        if USE_GAN:
                            for name, param in real_score_model.named_parameters():
                                if name in gan_critic_trainable_params:
                                    param.requires_grad = False

                        if args.training_config.is_use_ode_regression:
                            if args.training_config.dmd_is_low_vram_mode:
                                vram_manager.move_to_cpu(real_score_model)
                                vram_manager.move_to_gpu(transformer, accelerator.device)

                            _, ode_log_dict = _ode_regression_loss(
                                args=args,
                                accelerator=accelerator,
                                transformer=transformer,
                                scheduler=noise_scheduler_copy,
                                noise=torch.randn(
                                    noisy_model_input_shape, device=accelerator.device, dtype=weight_dtype
                                ),
                                # For Stage 1
                                is_keep_x0=True,
                                history_sizes=args.training_config.history_sizes,
                                # For Stage 2
                                stage2_num_stages=args.training_config.stage2_num_stages,
                                stage2_num_inference_steps_list=args.validation_config.stage2_simulated_inference_steps,
                                # For ODE Main
                                last_step_only=args.training_config.dmd_last_step_only,
                                use_dynamic_shifting=args.training_config.use_dynamic_shifting,
                                time_shift_type=args.training_config.time_shift_type,
                                is_backward_grad=True,
                                ode_regression_weight=args.training_config.ode_regression_weight,
                                ode_latents=ode_latents,
                                ode_prompt_embeds=ode_prompt_embeds,
                                # [v2v ODE] warp pass-through (t2v .pt -> None/False -> bit-compatible)
                                gt_all_data=ode_gt_all_data,
                                attention_kwargs=ode_attention_kwargs,
                                is_use_gt_history=(ode_gt_all_data is not None),
                                cam_Ks=ode_cam_Ks,
                                cam_c2ws=ode_cam_c2ws,
                                cam_base_h=ode_cam_base_h,
                                cam_base_w=ode_cam_base_w,
                                cam_strategy=ode_cam_strategy,
                                ode_num_latent_sections_min=args.training_config.ode_num_latent_sections_min,
                                ode_num_latent_sections_max=args.training_config.ode_num_latent_sections_max,
                                # For Dynamic ODE Length
                                ode_dynamic_alpha=args.training_config.ode_dynamic_alpha,
                                ode_dynamic_beta=args.training_config.ode_dynamic_beta,
                                ode_dynamic_sample_type=args.training_config.ode_dynamic_sample_type,
                                global_step=global_step,
                                ode_dynamic_step=args.training_config.ode_dynamic_step,
                            )
                            logs.update(ode_log_dict)

                            ode_log_dict = None
                            del ode_log_dict

                        # snapshot of host RSS + node available memory at the start of the step (before generation): the host-RAM
                        #   timeline diagnostic for dual-expert offload (does the generation phase warp/DA3/dataloader + Evoke-on-CPU hit the cgroup?).
                        #   compare against the end-of-step VRAM-PROBE -> locates the host peak phase. no env -> bit-identical.
                        if os.environ.get("SF_VRAM_PROBE") and accelerator.is_main_process:
                            _hm_rss = _hm_av = -1.0
                            try:
                                for _l in open("/proc/self/status"):
                                    if _l.startswith("VmRSS:"): _hm_rss = float(_l.split()[1]) / 1e6; break
                                for _l in open("/proc/meminfo"):
                                    if _l.startswith("MemAvailable:"): _hm_av = float(_l.split()[1]) / 1e6; break
                            except Exception:
                                pass
                            _hm_g = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else -1.0
                            print(f"[HOST-MEM step-start] step={global_step} rank0_rss={_hm_rss:.1f}GB "
                                  f"node_mem_avail={_hm_av:.1f}GB gpu_alloc={_hm_g:.1f}GB", flush=True)

                        # -- NOTE: sample the window start chunk s each step (rank-consistent) --
                        #   window = _sf_win_chunks consecutive chunks inside the generated region, start s in [2, N-wc+1] (never covers g1).
                        #   critic forward+backward only eats [GT prefix | window]; the gen DMD gradient only covers this window (g1 excluded).
                        #   rank-consistent: seeded from global_step -> same s on all ranks -> consistent sequence splitting inside the SP group (no collective). synchronized between
                        #   the generator and critic losses (same global_step -> same s). 0=OFF -> _sf_score_window=None (all 189).
                        _sf_score_window = None
                        _sf_win_chunks = int(getattr(args.training_config, "sf_score_window_chunks", 0) or 0)
                        if _sf_win_chunks > 0:
                            import random as _rnd_win
                            _n_gen_chunks = int(args.training_config.dmd_num_latent_sections_max)
                            _s_lo = 2                                      # lower bound of the start (g1 excluded)
                            _s_hi = _n_gen_chunks - _sf_win_chunks + 1     # upper bound of the start (window stays in range)
                            # sampling s uniformly trains mid-sequence chunks every step while the two ends (especially the tail g20) are only
                            #   trained with probability ~1/(number of positions) (10x undertrained). the fix: let the start "slide off the edge and then clamp back into the legal range" -> the edge positions
                            #   s_lo/s_hi accumulate the slid-off mass -> end coverage floor rises from ~10% to ~36%. the middle is still covered every step; g1 is always excluded.
                            # the slide-off amount is asymmetric between the high and low side (tilt): high side (wc-1)+tilt, low side (wc-1)-tilt -> tail coverage > head coverage
                            #   (compensating for autoregressive drift being worst at the tail). tilt=0 -> symmetric. rank-consistent (seeded from global_step, no collective).
                            _tilt = int(getattr(args.training_config, "sf_score_window_tail_tilt", 0) or 0)
                            _raw_lo = _s_lo - ((_sf_win_chunks - 1) - _tilt)   # low-side slide-off (head)
                            _raw_hi = _s_hi + ((_sf_win_chunks - 1) + _tilt)   # high-side slide-off (tail, tilt extra slots)
                            _s_raw = _rnd_win.Random(int(global_step) + 20260725).randint(_raw_lo, _raw_hi)
                            _s_win = min(max(_s_raw, _s_lo), _s_hi)        # clamp back into [2, N-wc+1]
                            _sf_score_window = (_s_win, _sf_win_chunks)
                            if accelerator.is_main_process and global_step < 3:
                                print(f"[SF-WINDOW] step={global_step} sampled window (edge-weighted, tail tilt={_tilt}): start chunk s={_s_win} "
                                      f"covers chunk [{_s_win},{_s_win + _sf_win_chunks - 1}] ({_sf_win_chunks} chunks total; "
                                      f"N={_n_gen_chunks}, s in [{_s_lo},{_s_hi}] clamp; g1 excluded)", flush=True)

                        # -- i2v or v2v for this step (student i2v/v2v mixed training) --
                        #   an i2v step gives the student 1 latent (pixel frame 0 = the reference image) as condition, scoring
                        #   sequence [ref image|g1..gN] = P_lat + N*win = 181. The teacher's cond frame is that same frame, so
                        #   student and teacher share the reference image and latent 0 (sink).
                        #   The data flow is unchanged: same clip, i2v just slices prefix[:, :, :1] + teacher_y[:, :, :181].
                        #   Rank consistency comes from a seeded RNG (global_step[+SP group id]) -- no collective, no global RNG
                        #   consumption. Within a group the mode must match, or the SP frame-slice lengths differ and NCCL sees a
                        #   shape mismatch; with scope=group different groups may pick different modes.
                        #   Routing is per sample: image-only samples are always i2v (no temporal GT/pose, and no slicing needed);
                        #   video samples are v2v unless switched to i2v with probability sf_i2v_ratio.
                        _sf_i2v_hist_latent = None
                        _sf_i2v_ratio = float(getattr(args.training_config, "sf_i2v_ratio", 0.0) or 0.0)
                        _sf_i2v_on = int(getattr(args.training_config, "sf_i2v_prefix_latent_frames", 0) or 0) > 0
                        # _sf_sample_is_i2v was set to False outside the data branches and copied out of batch inside the `if _sf_any:` extraction block
                        #   -- batch **must not** be read here (it was del'd earlier in this loop body, so reading it = UnboundLocalError;
                        # the measured 48-card crash on was on exactly this line).
                        _sf_i2v_active = _sf_sample_is_i2v
                        _sf_i2v_need_slice = False
                        if (not _sf_sample_is_i2v) and _sf_i2v_on and _sf_i2v_ratio > 0.0 and _sf_evoke_teacher:
                            import random as _rnd_i2v
                            _i2v_scope = str(getattr(args.training_config, "sf_i2v_mode_scope", "group"))
                            if _i2v_scope == "group":
                                from evoke.modules.evoke_teacher.sp_runtime import (
                                    get_sp_size as _i2v_gsz, is_sp_enabled as _i2v_sp_on,
                                )
                                _i2v_G = _i2v_gsz() if _i2v_sp_on() else 1
                                _i2v_grp = int(accelerator.process_index) // max(1, int(_i2v_G))
                            else:
                                _i2v_grp = 0
                            _sf_i2v_active = _rnd_i2v.Random(
                                int(global_step) * 10007 + _i2v_grp * 97 + 20260729).random() < _sf_i2v_ratio
                            _sf_i2v_need_slice = _sf_i2v_active
                        if _sf_sample_is_i2v:
                            # the data side already produced the i2v layout (prefix=1 frame / y=181 frames / a single section prompt), so just take the 1x-slot latent.
                            _sf_i2v_hist_latent = _sf_i2v_hist_latent_full
                        if _sf_i2v_need_slice:
                            _i2v_P_lat = int(getattr(args.training_config, "sf_i2v_prefix_latent_frames", 1) or 1)
                            _i2v_T = _i2v_P_lat + int(num_rollout_sections) * int(latent_window_size)
                            assert _sf_prefix_latents.shape[2] > _i2v_P_lat, \
                                f"[LW-I2V] prefix has only {_sf_prefix_latents.shape[2]} frames, cannot slice a {_i2v_P_lat}-frame i2v prefix"
                            assert _sf_teacher_y is not None and _sf_teacher_y.shape[2] >= _i2v_T, \
                                f"[LW-I2V] teacher_y frame count {None if _sf_teacher_y is None else _sf_teacher_y.shape[2]} < {_i2v_T}"
                            # latent frame 0 = pixel frame 0 (Wan-VAE encodes a single frame independently) = the reference image, same source as the cond of teacher y.
                            _sf_prefix_latents = _sf_prefix_latents[:, :, :_i2v_P_lat].contiguous()
                            # y's mask has the 4 channels of latent frame 0 set to 1, and truncation does not touch it; the VAE is causal => the first _i2v_T frames are
                            #   value-for-value identical to "encoding only the first (T-1)*4+1 pixel frames".
                            _sf_teacher_y = _sf_teacher_y[:, :, :_i2v_T].contiguous()
                            if _sf_prompt_embeds_list_i2v is not None:
                                _sf_prompt_embeds_list = _sf_prompt_embeds_list_i2v[:num_rollout_sections]
                            if str(getattr(args.training_config, "sf_i2v_hist_latent_mode", "static_repeat")) \
                                    == "static_repeat":
                                assert _sf_i2v_hist_latent_full is not None, \
                                    "[LW-I2V] hist_latent_mode=static_repeat requires the data side to produce sf_i2v_hist_latent"
                                _sf_i2v_hist_latent = _sf_i2v_hist_latent_full
                        # the g1 gate and the mechanism-A skip must be toggled as a pair (otherwise the mask covers it but nobody builds the graph -> 1/N of the supervision is silently lost).
                        if bool(getattr(args.training_config, "sf_i2v_score_g1", False)):
                            from evoke.modules import student_sp as _stu_sp_g1
                            _stu_sp_g1.set_skip_first_chunk(
                                bool(getattr(args.training_config, "dmd_score_skip_first_chunk", False))
                                and not _sf_i2v_active)
                        # print the routing result step by step for the first 8 steps; afterwards once every log_iters (so long runs can be checked for the real i2v/v2v ratio).
                        if _sf_i2v_on and accelerator.is_main_process and (
                                global_step < 8 or global_step % max(1, int(args.training_config.log_iters)) == 0):
                            print(f"[LW-I2V] step={global_step} mode={'i2v' if _sf_i2v_active else 'v2v'} "
                                  f"src={'image-only' if _sf_sample_is_i2v else 'video'} "
                                  # None guard: every other quantity on this line is guarded, only this one was not -- on the use_stage3_dataset path
                                  # _sf_prefix_latents is always None and an AttributeError would kill the diagnostic we came here to read.
                                  f"prefix_T={None if _sf_prefix_latents is None else _sf_prefix_latents.shape[2]} "
                                  f"teacher_y_T={None if _sf_teacher_y is None else _sf_teacher_y.shape[2]} "
                                  f"gt_latents={'None' if _sf_gt_latents is None else int(_sf_gt_latents.shape[2])} "
                                  f"pose={'None' if _sf_pose_c2ws is None else 'yes'} "
                                  f"hist_latent={'static_repeat' if _sf_i2v_hist_latent is not None else 'prefix'} "
                                  f"(ratio={_sf_i2v_ratio} only applies to video samples, "
                                  f"scope={getattr(args.training_config, 'sf_i2v_mode_scope', 'group')})", flush=True)

                        # settle and print the "outside the window" part before the timing window starts.
                        if _pp_mod.enabled() and accelerator.is_main_process:
                            _pp_mod.accum("__prep_total__", _pp_t_prep)
                            print(_pp_mod.report(
                                f" step={global_step} mode={'i2v' if _sf_i2v_active else 'v2v'}"
                                f" fetch={('%.2fs' % _pp_fetch) if _pp_fetch is not None else 'n/a'}"),
                                flush=True)
                        sf_prof_step_begin()          # reset + start timing on every optimizer step
                        _sf_t_gl = sf_prof_mark()
                        if os.environ.get("SF_DECOUPLE_EQUIV_MODE"):
                            from scripts.training.tmp.test_decouple_equivalence import (
                                record_phase as _equiv_record_phase,
                                seed_phase as _equiv_seed_phase,
                            )
                            _equiv_record_phase(
                                "materialized",
                                tensors={
                                    "model_input": model_input,
                                    "prompt_embeds": prompt_embeds,
                                    "negative_prompt_embeds": negative_prompt_embeds,
                                    "sf_prefix_latents": _sf_prefix_latents,
                                    "sf_teacher_y": _sf_teacher_y,
                                    "sf_score_prompt_embeds": _sf_score_prompt_embeds,
                                    "sf_pose_Ks": _sf_pose_Ks,
                                    "sf_pose_c2ws": _sf_pose_c2ws,
                                    "target_pose_Ks": _stashed_target_pose_Ks,
                                    "target_pose_c2ws": _stashed_target_pose_c2ws,
                                },
                            )
                            _equiv_seed_phase("rollout")
                        generator_loss, generator_log_dict = _generator_loss(
                            args=args,
                            accelerator=accelerator,
                            real_fake_score_model=real_score_model,
                            # second real-score teacher (camera force) + convex weights; hb=None when dual is off -> inert.
                            real_fake_score_model_hb=real_score_model_hb,
                            w_lw=args.model_config.dual_teacher.w_lw,
                            w_hb=args.model_config.dual_teacher.w_hb,
                            # the two teachers offload alternately (dual_teacher.offload); when off both stay resident (old behavior).
                            dual_teacher_offload=_et_offload,
                            # (s,wc)=window in the generated region (start chunk s, wc chunks); None=all 189 (bit-id).
                            #   generator side: the teacher still does a full forward, only the DMD gradient_mask covers this window (g1 excluded).
                            sf_score_window=_sf_score_window,
                            # decoupled dual-loss per-teacher params: lambda_hb weights the Evoke tail section (not convex) and
                            #   evoke_score_timestep_max caps that section's band (None -> all bands, stage0 camera included). Both are
                            #   inert with dual off (hb=None). evoke_teacher_score_timestep_max is NOT inert: it is read under plain
                            #   `if _sf_front_window` and caps the EvokeTeacher front window's scoring band, which is why it lives on
                            #   evoke_teacher.score_timestep_max (default None = full band) rather than in the dual block.
                            lambda_hb=args.model_config.dual_teacher.lambda_hb,
                            evoke_teacher_score_timestep_max=args.model_config.evoke_teacher.score_timestep_max,
                            evoke_score_timestep_max=args.model_config.dual_teacher.evoke_score_timestep_max,
                            transformer=transformer,
                            scheduler=noise_scheduler_copy,
                            noise=torch.randn(noisy_model_input_shape, device=accelerator.device, dtype=weight_dtype),
                            prompt_embeds=prompt_embeds,
                            negative_prompt_embeds=negative_prompt_embeds,
                            # For VRAM manager
                            dmd_is_low_vram_mode=args.training_config.dmd_is_low_vram_mode,
                            vram_manager=vram_manager,
                            dmd_is_offload_grad=args.training_config.dmd_is_offload_grad,
                            is_gan_low_vram_mode=args.training_config.is_gan_low_vram_mode,
                            # For Stage 1
                            is_keep_x0=True,
                            history_sizes=args.training_config.history_sizes,
                            # For Stage 2
                            is_enable_stage2=args.training_config.is_enable_stage2,
                            stage2_num_stages=args.training_config.stage2_num_stages,
                            stage2_num_inference_steps_list=args.validation_config.stage2_simulated_inference_steps,
                            # For DMD Main
                            denoising_step_list=list(args.training_config.dmd_denoising_step_list),
                            last_step_only=args.training_config.dmd_last_step_only,
                            last_section_grad_only=args.training_config.dmd_last_section_grad_only,
                            timestep_shift=args.training_config.dmd_timestep_shift,
                            use_dynamic_shifting=args.training_config.use_dynamic_shifting,
                            time_shift_type=args.training_config.time_shift_type,
                            fake_guidance_scale=args.training_config.fake_guidance_scale,
                            real_guidance_scale=args.training_config.real_guidance_scale,
                            # the curriculum takes over the tail-window size (falls back to the static config without a curriculum)
                            num_critic_input_frames=(_sf_ncif if _sf_ncif is not None
                                                     else args.training_config.num_critic_input_frames),
                            num_rollout_sections=num_rollout_sections,
                            is_skip_first_section=args.training_config.is_skip_first_section,
                            is_amplify_first_chunk=args.training_config.is_amplify_first_chunk,
                            # For Easy Anti-Drifting
                            is_corrupt_history_latents=args.training_config.corrupt_history,
                            is_add_saturation=args.training_config.is_add_saturation,
                            # For GT History
                            is_use_gt_history=USE_GT_HIST,
                            gt_history_latents=gt_history_latents,
                            gt_target_latents=gt_target_latents,
                            gt_x0_latents=gt_x0_latents,
                            # For VAE Re-Encode
                            vae=vae,
                            is_dmd_vae_decode=args.training_config.is_dmd_vae_decode,
                            # For Multi Stage Backward Simulated
                            is_multi_pyramid_stage_backward_simulated=args.training_config.is_multi_pyramid_stage_backward_simulated,
                            # For Consistency Align
                            is_consistency_align=args.training_config.is_consistency_align,
                            consistentcy_align_weight=args.training_config.consistentcy_align_weight,
                            # For Smoothness
                            is_smoothness_loss=args.training_config.is_smoothness_loss,
                            smoothness_loss_weight=args.training_config.smoothness_loss_weight,
                            # For KV Cache
                            use_kv_cache=args.validation_config.use_kv_cache,
                            # For Mean-Variance Regularization
                            is_mean_var_regular=args.training_config.is_mean_var_regular,
                            mean_var_regular_weight=args.training_config.mean_var_regular_weight,
                            regular_mean=args.training_config.regular_mean,
                            regular_var=args.training_config.regular_var,
                            is_x0_mean_var_regular=args.training_config.is_x0_mean_var_regular,
                            mean_var_regular_x0_weight=args.training_config.mean_var_regular_x0_weight,
                            regular_x0_mean=args.training_config.regular_x0_mean,
                            regular_x0_var=args.training_config.regular_x0_var,
                            #
                            is_chunk_mean_var_regular=args.training_config.is_chunk_mean_var_regular,
                            chunk_mean_var_regular_weight=args.training_config.chunk_mean_var_regular_weight,
                            chunk_regular_mean=args.training_config.chunk_regular_mean,
                            chunk_regular_var=args.training_config.chunk_regular_var,
                            is_chunk_x0_mean_var_regular=args.training_config.is_chunk_x0_mean_var_regular,
                            chunk_mean_var_regular_x0_weight=args.training_config.chunk_mean_var_regular_x0_weight,
                            chunk_regular_x0_mean=args.training_config.chunk_regular_x0_mean,
                            chunk_regular_x0_var=args.training_config.chunk_regular_x0_var,
                            # For GAN
                            is_use_gan=USE_GAN,
                            gan_prompt_embeds=gan_prompt_embeds,
                            gan_g_weight=args.training_config.gan_g_weight,
                            # For Reward
                            is_use_reward_model=USE_REWARD,
                            reward_model=reward_model,
                            reward_weight_vq=args.training_config.reward_weight_vq,
                            reward_weight_mq=args.training_config.reward_weight_mq,
                            reward_weight_ta=args.training_config.reward_weight_ta,
                            reward_texts=prompt_raws,
                            # For Decouple DMD
                            is_decouple_dmd=args.training_config.is_decouple_dmd,
                            decouple_ca_start_step=args.training_config.decouple_ca_start_step,
                            decouple_ca_end_step=args.training_config.decouple_ca_end_step,
                            # For Dynamic Timestep
                            is_forcing_low_renoise=args.training_config.generator_is_forcing_low_renoise,
                            # explicit scoring band (anchored at pyramid stage boundaries; None/0 = old behavior)
                            min_score_timestep=int(getattr(args.training_config, "dmd_score_timestep_min", 0) or 0),
                            max_score_timestep=getattr(args.training_config, "dmd_score_timestep_max", None),
                            # thin high-band mixed sampling (prob=0 old behavior; actual-t semantics)
                            highband_prob=float(getattr(args.training_config, "dmd_score_highband_prob", 0.0) or 0.0),
                            highband_min=int(getattr(args.training_config, "dmd_score_highband_min", 666) or 666),
                            highband_max=int(getattr(args.training_config, "dmd_score_highband_max", 1000) or 1000),
                            dynamic_alpha=args.training_config.generator_dynamic_alpha,
                            dynamic_beta=args.training_config.generator_dynamic_beta,
                            dynamic_sample_type=args.training_config.generator_dynamic_sample_type,
                            global_step=global_step,
                            dynamic_step=args.training_config.generator_dynamic_step,
                            # GEO v2v DMD conditioning (None when GEO inactive).
                            gt_geo_all_data=_gt_geo_all_data,
                            # clean-warp tuple for the DM teacher only (None -> teacher uses degraded).
                            gt_geo_all_data_teacher_clean=_gt_geo_all_data_teacher_clean,
                            gt_geo_attention_kwargs=_gt_geo_attention_kwargs,
                            # EvokeTeacher full-sequence scoring branch
                            is_evoke_teacher_score=_sf_evoke_teacher,
                            sf_windowed_score=(_sf_evoke_teacher and bool(getattr(args.training_config, "sf_curriculum_enabled", False))),
                            sf_prefix_latents=_sf_prefix_latents,
                            sf_gt_latents=_sf_gt_latents,   # used to swap in GT for the teacher long/mid tier
                            # the j paired with the data-side encoding window (None=GEO-REG draws it itself)
                            sf_geo_j=_sf_geo_j,
                            sf_prompt_embeds_list=_sf_prompt_embeds_list,
                            sf_score_prompt_embeds=_sf_score_prompt_embeds,
                            sf_teacher_y=_sf_teacher_y,
                            sf_segment_frame_ranges=_sf_segment_frame_ranges,
                            # this step's mode + the i2v-step 1x-slot latent (always False/None on v2v steps -> old path bit-identical)
                            sf_i2v_active=_sf_i2v_active,
                            sf_i2v_hist_latent=_sf_i2v_hist_latent,
                            # skip GEO-REG during the freeze (a separate student forward + DA3 render that is never backwarded => wasted compute)
                            sf_gen_frozen=_GEN_FROZEN,
                            # tier scoring branch + shared rollout out-param + warp pose
                            sf_evoke_tier_score=_sf_evoke,
                            sf_rollout_out=_sf_rollout_shared,
                            sf_pose_Ks=_sf_pose_Ks,
                            sf_pose_c2ws=_sf_pose_c2ws,
                            # camera Plucker for student(per-stage)+teacher(full-res).
                            # the evoke teacher does not consume plucker (Evoke-Base has no such weights) -> no pose passed.
                            cam_Ks=None if _sf_evoke else _stashed_target_pose_Ks,
                            cam_c2ws=None if _sf_evoke else _stashed_target_pose_c2ws,
                            cam_base_h=_dmd_cam_base_h,
                            cam_base_w=_dmd_cam_base_w,
                            cam_strategy=_dmd_cam_strategy,
                        )
                        sf_prof_accum("genloss", _sf_t_gl)   # genloss = rollout (warp included) + scoring forward
                        if os.environ.get("SF_DECOUPLE_EQUIV_MODE"):
                            from scripts.training.tmp.test_decouple_equivalence import (
                                record_phase as _equiv_record_phase,
                            )
                            _equiv_record_phase(
                                "generator_out",
                                tensors={
                                    "rollout_pred_video": (
                                        _sf_rollout_shared.get("pred_video")
                                        if _sf_rollout_shared else None
                                    ),
                                    "rollout_tail_cam_Ks": (
                                        _sf_rollout_shared.get("tail_cam_Ks")
                                        if _sf_rollout_shared else None
                                    ),
                                    "rollout_tail_cam_c2ws": (
                                        _sf_rollout_shared.get("tail_cam_c2ws")
                                        if _sf_rollout_shared else None
                                    ),
                                },
                                scalars={
                                    "generator_loss": generator_loss,
                                    "dmd_loss_lw": generator_log_dict.get("dmd_loss_lw"),
                                    "dmd_loss_hb": generator_log_dict.get("dmd_loss_hb"),
                                },
                            )

                        # when mechanism A/B is on: manually stagger the DP allreduce so the backward window holds only the
                        #   U-subgroup all-to-all (including inside checkpoint recomputation) and no ZeRO DP reduction -- two PGs
                        #   over the same rank set interleaving inside backward is the deadlock. All three steps are needed;
                        #   leaving one out fails silently:
                        #   (1) enable_backward_allreduce=False -- the real switch. set_gradient_accumulation_boundary(False) does
                        #      nothing for ZeRO-2's hook reduction (partition_gradients is always True there); the window is
                        #      reduction-free thanks to (1) plus the IPG bucket never overflowing, which is asserted at startup.
                        #   (2) a separate allreduce_gradients() afterwards -- a single IPG epilogue.
                        #   (3) an explicit engine.step(): under accelerate+DeepSpeed optimizer.step()/zero_grad() are `pass` and the
                        #      only optimizer step normally comes from the engine.step() inside accelerator.backward(), so omitting
                        #      it computes gradients but never updates params while the loss curve still looks normal. It stays
                        #      inside the same _sf_prof("gen_bwd") block so gen_bwd remains comparable to the baseline.
                        # during the freeze: no backward, no engine.step() => student params are bit-wise unchanged.
                        #   generator_loss is detached at once, else the rollout graph stays alive until the critic finishes.
                        if _GEN_FROZEN:
                            generator_loss = generator_loss.detach()
                        with _sf_prof("gen_bwd"):
                            if _GEN_FROZEN:
                                pass          # no backward during the freeze (the forward already ran and the shared rollout is filled in for the critic)
                            elif _stu_sp.is_any_enabled():
                                _g_eng = accelerator.deepspeed_engine_wrapped.engine
                                _g_sync = accelerator.sync_gradients
                                _g_eng.set_gradient_accumulation_boundary(is_boundary=False)
                                _g_eng.enable_backward_allreduce = False
                                _g_eng.backward(generator_loss)   # GAS=1 (forced by the validator) => no longer divided by GAS
                                _g_eng.enable_backward_allreduce = True
                                _g_eng.set_gradient_accumulation_boundary(is_boundary=_g_sync)
                                # NOTE: must run before allreduce_gradients() (the bucket is emptied afterwards). once on the first step, at the cost of one
                                #   all_reduce (4 longs), to rule out "silent gradient scrambling", a class of error that cannot be repaired after the fact.
                                #   not gated on diag: it guards against a catastrophic error, not diagnostic information.
                                # switched to _GEN_BWD_DIAG_NOW: no backward during the freeze => the self-check must move to the first real backward step.
                                if _GEN_BWD_DIAG_NOW:
                                    _stu_h = _stu_sp.check_ipg_bucket_order(_g_eng, tag="generator")
                                    accelerator.print(f"  [STU-SP §8-(6)] IPG bucket order hash={_stu_h} consistent across all ranks, ok")
                                _g_eng.allreduce_gradients()
                                if _g_sync:
                                    _g_eng.step()
                            else:
                                accelerator.backward(generator_loss)

                        generator_grad_norm = None
                        if accelerator.sync_gradients and not _GEN_FROZEN:
                            # NOTE: must be after eng.step(): _global_grad_norm is only assigned inside _take_model_step
                            #   (engine.py) and starts as None. also, under DeepSpeed accelerator.clip_grad_norm_
                            #   **does not clip**, it only returns get_global_grad_norm() (accelerator.py) =>
                            #   max_grad_norm is a dead config and the real clipping follows gradient_clipping=1.0 in the ds json;
                            #   this value is the total_norm **before** clipping (worth knowing when L1/L3 use it as a criterion).
                            generator_params_to_clip = transformer.parameters()
                            generator_grad_norm = accelerator.clip_grad_norm_(
                                generator_params_to_clip, args.training_config.max_grad_norm
                            )
                        # the in-bucket param_id sequence hash is consistent in-group + the collective sequence numbers match.
                        #   the in-bucket order is decided by the insertion order of params_in_ipg_bucket (= the order the backward graph is reached), and with different
                        #   branch counts per rank (5 sections vs 4) it is **not proven** consistent; the moment it is not => the same offset holds a different param =>
                        #   silent gradient scrambling, which staggering does nothing about.
                        if _stu_sp.is_any_enabled() and _stu_sp.is_diag() and _GEN_BWD_DIAG_NOW:
                            _stu_sp.check_seq_in_group("gen_bwd")

                        generator_log_dict["generator_loss"] = generator_loss
                        if generator_grad_norm is not None:
                            generator_log_dict["generator_grad_norm"] = generator_grad_norm

                        extra = generator_log_dict
                        extras_list.append(extra)
                        generator_log_dict = merge_dict_list(extras_list)
                        # skip these three lines during the freeze: optimizer.step()/zero_grad() are a pass under DeepSpeed,
                        #   but lr_scheduler.step() is not -- skipping it keeps the student LR schedule from advancing on steps it was not trained on
                        #   (no difference under a constant lr, but this line matters once warmup/cosine is used).
                        if not _GEN_FROZEN:
                            optimizer.step()
                            lr_scheduler.step()
                            optimizer.zero_grad(set_to_none=True)

                        base_logs = {
                            # "generator_lr": lr_scheduler.get_last_lr()[0],
                            "generator_loss": generator_log_dict["generator_loss"].mean().item(),
                        }
                        # a non-boundary micro-step with grad_accum>1 does not run clip_grad_norm_, hence the key is absent.
                        # behavior is unchanged for the shipped configs, which all use grad_accum=1.
                        if "generator_grad_norm" in generator_log_dict:
                            base_logs["generator_grad_norm"] = safe_item(
                                generator_log_dict["generator_grad_norm"]
                            )
                        if args.training_config.is_decouple_dmd:
                            base_logs.update(
                                {
                                    "dmdtrain_ca_gradient_norm": safe_item(
                                        generator_log_dict["dmdtrain_ca_gradient_norm"]
                                    ),
                                    "dmdtrain_dm_gradient_norm": safe_item(
                                        generator_log_dict["dmdtrain_dm_gradient_norm"]
                                    ),
                                }
                            )
                        else:
                            base_logs["dmdtrain_gradient_norm"] = safe_item(
                                generator_log_dict["dmdtrain_gradient_norm"]
                            )
                        logs.update(base_logs)
                        base_logs = None
                        del base_logs

                        # off-sequence observation (only the SF jitter path has this key)
                        if "sf_window_offset" in generator_log_dict:
                            logs["sf_window_offset"] = generator_log_dict["sf_window_offset"]

                        # observability of the decoupled dual-teacher losses: dmd_loss_lw (front section, EvokeTeacher long-range) /
                        #   dmd_loss_hb (tail section, Evoke camera force -> student camera gradient). dmd_loss_hb>0 = the camera signal really reaches the student (core requirement 3).
                        if "dmd_loss_hb" in generator_log_dict:
                            logs["dmd_loss_hb"] = generator_log_dict["dmd_loss_hb"]
                        if "dmd_loss_lw" in generator_log_dict:
                            logs["dmd_loss_lw"] = generator_log_dict["dmd_loss_lw"]

                        # the eq.8 denominator D (utils_evoke_post compute_kl_grad). dmd_loss_lw = 0.5*mean[(N/D)^2] is a ratio, so
                        #   one curve cannot show where N is heading. N = mean|s_fake-s_real| = dmdtrain_gradient_norm *
                        #   dmdtrain_normalizer (exact for B=1). N falling = the critic/teacher disagreement really shrank;
                        #   D rising = the student is drifting away from the teacher. On the decouple-DMD path and with
                        #   normalization=False the key is absent, so nothing is logged.
                        if generator_log_dict.get("dmdtrain_normalizer") is not None:
                            logs["dmdtrain_normalizer"] = safe_item(
                                generator_log_dict["dmdtrain_normalizer"]
                            )
                        # control quantity for the masked denominator: the all-frame mean. `sf_dmd_normalizer_masked=true` changes
                        #   the eq.8 denominator from a mean over all 189 frames to a mean inside the gradient interval, which really
                        #   rescales the gradient. The ratio of the two curves shows how much the GT prefix (9 frames, no gradient)
                        #   diluted the denominator.
                        if generator_log_dict.get("dmdtrain_normalizer_full") is not None:
                            logs["dmdtrain_normalizer_full"] = safe_item(
                                generator_log_dict["dmdtrain_normalizer_full"]
                            )

                        if args.training_config.is_smoothness_loss or USE_GAN or USE_REWARD:
                            logs["dmd_loss_raw"] = generator_log_dict["dmd_loss_raw"]

                        if args.training_config.is_consistency_align:
                            logs["consistency_align_loss"] = generator_log_dict["consistency_align_loss"]

                        if args.training_config.is_smoothness_loss:
                            logs["smoothness_loss"] = generator_log_dict["smoothness_loss"]

                        if args.training_config.is_mean_var_regular:
                            logs["kl_mean_var_loss"] = generator_log_dict["kl_mean_var_loss"]
                            logs["pred_mean_avg"] = generator_log_dict["pred_mean_avg"]
                            logs["pred_var_avg"] = generator_log_dict["pred_var_avg"]

                            if args.training_config.is_x0_mean_var_regular:
                                logs["kl_mean_var_x0_loss"] = generator_log_dict["kl_mean_var_x0_loss"]
                                logs["pred_x0_mean_avg"] = generator_log_dict["pred_x0_mean_avg"]
                                logs["pred_x0_var_avg"] = generator_log_dict["pred_x0_var_avg"]

                        if args.training_config.is_chunk_mean_var_regular:
                            logs["kl_chunk_mean_var_loss"] = generator_log_dict["kl_chunk_mean_var_loss"]
                            logs["pred_chunk_mean_avg"] = generator_log_dict["pred_chunk_mean_avg"]
                            logs["pred_chunk_var_avg"] = generator_log_dict["pred_chunk_var_avg"]

                            if args.training_config.is_chunk_x0_mean_var_regular:
                                logs["kl_chunk_mean_var_x0_loss"] = generator_log_dict["kl_chunk_mean_var_x0_loss"]
                                logs["pred_chunk_x0_mean_avg"] = generator_log_dict["pred_chunk_x0_mean_avg"]
                                logs["pred_chunk_x0_var_avg"] = generator_log_dict["pred_chunk_x0_var_avg"]

                        if USE_GAN:
                            logs["gan_G_loss"] = generator_log_dict["gan_G_loss"]

                        if USE_REWARD:
                            logs["reward_score_vq"] = generator_log_dict["reward_score_vq"]
                            logs["reward_score_mq"] = generator_log_dict["reward_score_mq"]
                            logs["reward_score_ta"] = generator_log_dict["reward_score_ta"]

                        generator_loss = None
                        generator_grad_norm = None
                        if (
                            bool(getattr(
                                args.training_config, "sf_decouple_rollout", False
                            ))
                            and int(getattr(
                                args.training_config,
                                "sf_critic_steps_per_student",
                                1,
                            ) or 1) > 1
                        ):
                            # The merged log dict still owns the original loss
                            # tensor (and graph nodes) after backward.  Multi-step critic does
                            # not visualize between student and critic; drop it
                            # before the 42B EvokeTeacher activation peak.
                            generator_log_dict = None
                            extra = None
                            extras_list = None
                        del generator_loss
                        del generator_grad_norm
                        free_memory()
                        if (
                            os.environ.get("SF_VRAM_PROBE")
                            and bool(getattr(
                                args.training_config, "sf_decouple_rollout", False
                            ))
                            and int(getattr(
                                args.training_config,
                                "sf_critic_steps_per_student",
                                1,
                            ) or 1) > 1
                            and global_step < 3
                            and torch.cuda.is_available()
                        ):
                            torch.cuda.synchronize()
                            print(
                                "[VRAM-PHASE student-released] "
                                f"step={global_step} "
                                f"rank={torch.distributed.get_rank()} "
                                f"alloc={torch.cuda.memory_allocated() / 2**30:.1f}GB "
                                f"reserved={torch.cuda.memory_reserved() / 2**30:.1f}GB",
                                flush=True,
                            )

                    if USE_GAN:
                        for name, param in real_score_model.named_parameters():
                            if name in gan_critic_trainable_params:
                                param.requires_grad = True

                    # Train the critic
                    _sf_t_critic = sf_prof_mark()   # start of timing for the critic phase (forward+backward+step)
                    extras_list = []
                    # (6)a before the EvokeTeacher critic: the EvokeTeacher base returns to GPU and the Evoke base
                    #   is swapped out to CPU (only frozen bases are moved, the trainable critic-LoRA stays on GPU). the _generator_loss exit already left this state -> idempotent confirmation.
                    #   device comes from critic_accelerator.device (= where the critic forward inputs live). only active on the front-window+offload path.
                    #   [per-expert] the EvokeTeacher bring-to-GPU is skipped; the critic's wrapper.forward keeps only the routed expert resident (avoiding dual-expert OOM).
                    if _dual_offload and _sf_front_window:
                        _evoke_teacher_base_to(real_score_model, critic_accelerator.device)
                        if real_score_model_hb is not None:
                            _offload_frozen_params_to(real_score_model_hb, "cpu")
                    # the EvokeTeacher critic scores SP frame slices: under decouple it rotates through owners _jc and scores G
                    #   distinct clips (for each clip all cards in the group cooperate on the frame slices -> a complete clip gradient), accumulating per clip and doing
                    #   one allreduce+step after the loop. off / _Gc=1 -> a single pass (byte-identical). section-stacked prompts (shape varies with the section count) +
                    #   segment ranges are broadcast per owner as varshape/objects; video/noise/timestep/y go through sync_tensor inside _critic_loss (owner-block).
                    from evoke.modules.evoke_teacher.sp_runtime import (
                        is_2d_sp as _is2d_bwd, get_sp_size as _sp_gsz_c, get_sp_rank as _sp_grk_c,
                        sp_score_owner as _sp_owner_c, sp_decouple_scope as _sp_dc_scope_c,
                        broadcast_from_owner as _sp_bft_c,
                        broadcast_varshape_from_owner as _sp_bvs_c,
                        broadcast_object_from_owner as _sp_bobj_c,
                    )
                    _sf_decouple_c = (bool(getattr(args.training_config, "sf_decouple_rollout", False))
                                      and _sp_world_size > 1)
                    if _sf_decouple_c:
                        assert _is2d_bwd() and not _dmd_zero3, (
                            "[THROUGHPUT-B] critic decouple currently only supports 2D SP x ZeRO-2 route-B backward "
                            "(is_2d_sp and not zero3)")
                        _dc_scope_obj_c = _sp_dc_scope_c(); _dc_scope_obj_c.__enter__()
                    _Gc = _sp_gsz_c() if _sf_decouple_c else 1
                    _lrc = _sp_grk_c() if _sf_decouple_c else 0
                    _critic_steps = int(getattr(
                        args.training_config, "sf_critic_steps_per_student", 1
                    ) or 1)
                    _c3_enabled = _sf_decouple_c and _critic_steps > 1
                    _c3_plan = None
                    _c3_local_steps = None
                    _c3_substep_losses = []
                    if _c3_enabled:
                        assert not USE_GAN, (
                            "[THROUGHPUT-B MULTI-CRITIC] multi-step critic does not support the GAN critic yet"
                        )
                        from evoke.utils.critic_update_schedule import build_critic_update_plan
                        _c_world = torch.distributed.get_world_size()
                        _c_group_index = torch.distributed.get_rank() // _Gc
                        _c3_plan = build_critic_update_plan(
                            world_size=_c_world,
                            sp_size=_Gc,
                            num_critic_steps=_critic_steps,
                            outer_step=global_step,
                        )
                        _c3_local_steps = _c3_plan.for_group(_c_group_index)
                        _expected_c_bs = int(getattr(
                            args.training_config,
                            "sf_critic_expected_global_batch_size",
                            0,
                        ) or 0)
                        if _expected_c_bs:
                            assert _c3_plan.bucket_sizes() == (
                                _expected_c_bs,
                            ) * _critic_steps, (
                                "[THROUGHPUT-B MULTI-CRITIC] critic global batch does not match the config: "
                                f"actual={_c3_plan.bucket_sizes()} "
                                f"expected={_expected_c_bs}"
                            )
                        if accelerator.is_main_process and global_step < 3:
                            print(
                                "[THROUGHPUT-B MULTI-CRITIC] "
                                f"W={_c_world} G={_Gc} Q={_c3_plan.num_groups} "
                                f"critic_steps={_critic_steps} "
                                f"slots={_c3_plan.slots_per_step} "
                                f"batch={_c3_plan.bucket_sizes()}",
                                flush=True,
                            )
                        _c_work = [
                            slot
                            for step_slots in _c3_local_steps
                            for slot in step_slots
                        ]
                        _c3_metric_sum = torch.zeros(
                            (), device=critic_accelerator.device, dtype=torch.float32
                        )
                        _c3_metric_count = torch.zeros_like(_c3_metric_sum)
                    else:
                        # Keep the validated C1/off path byte-for-byte in ordering:
                        # one owner pass per SP local rank, followed by one engine step.
                        _c_work = [None] * _Gc
                    critic_loss = None
                    for _c_work_index, _c_slot in enumerate(_c_work):
                        _jc = (
                            _c_slot.owner_local_rank
                            if _c3_enabled
                            else _c_work_index
                        )
                        if os.environ.get("SF_DECOUPLE_EQUIV_MODE"):
                            from scripts.training.tmp.test_decouple_equivalence import seed_phase as _equiv_seed_phase
                            _equiv_seed_phase(
                                "critic_evoke_teacher_score",
                                owner_local_rank=(_jc if _sf_decouple_c else None),
                            )
                        if _sf_decouple_c:
                            # NOTE: the clip-dependent bundle of owner _jc: section-stacked prompts (shape varies with the section count) + segment ranges (section count varies per clip).
                            #   video(rollout)/noise/timestep/y go through sync_tensor inside _critic_loss (broadcast from _jc inside the owner-block).
                            _oc_c = _sp_owner_c(_jc); _oc_c.__enter__()
                            _prompt_jc = _sp_bvs_c(
                                _sf_score_prompt_embeds, _jc, critic_accelerator.device,
                                _sf_score_prompt_embeds.dtype if _sf_score_prompt_embeds is not None else None)
                            _ranges_jc = _sp_bobj_c(_sf_segment_frame_ranges, _jc)
                            # EvokeTeacher warp-5600 critic consumes camera Plücker too.  Keep the
                            # complete clip-dependent bundle coherent across the owner SP pass.
                            # Broadcast cloned buffers: dist.broadcast is in-place and the local
                            # pose is still needed by later owners / the Evoke critic.
                            _cam_Ks_jc = (
                                _sp_bft_c(
                                    _stashed_target_pose_Ks.detach().to(critic_accelerator.device).clone(),
                                    _jc,
                                )
                                if _stashed_target_pose_Ks is not None else None
                            )
                            _cam_c2ws_jc = (
                                _sp_bft_c(
                                    _stashed_target_pose_c2ws.detach().to(critic_accelerator.device).clone(),
                                    _jc,
                                )
                                if _stashed_target_pose_c2ws is not None else None
                            )
                            # EvokeTeacher wrapper broadcasts its stored y in-place; isolate every
                            # owner iteration so owner-0 cannot destroy owner-1's local source.
                            _teacher_y_jc = (
                                _sp_bft_c(
                                    _sf_teacher_y.detach().to(critic_accelerator.device).clone(),
                                    _jc,
                                )
                                if _sf_teacher_y is not None else None
                            )
                        else:
                            _prompt_jc = _sf_score_prompt_embeds
                            _ranges_jc = _sf_segment_frame_ranges
                            _cam_Ks_jc = _stashed_target_pose_Ks
                            _cam_c2ws_jc = _stashed_target_pose_c2ws
                            _teacher_y_jc = _sf_teacher_y
                        # runtime fallback for four-interval consistency: the sample is i2v, but this step never went through the generator's mode dispatch
                        #   (that block sits inside `if TRAIN_GENERATOR:`) => the sf_i2v_* handed to the critic below are the defaults,
                        #   so the critic would score with v2v semantics -> a different source than generator/teacher. better to blow up here.
                        assert not (_sf_sample_is_i2v and not _sf_i2v_active), (
                            "[LW-I2V] an i2v sample reached a critic-only step: sf_i2v_* did not go through the mode dispatch (defaults False/None), "
                            "so the critic would skip the frame-0 replacement while the generator did it => a silent four-interval mismatch. "
                            "set dfake_gen_update_ratio back to 1, or hoist the mode dispatch out of the TRAIN_GENERATOR block")
                        critic_loss, critic_log_dict = _critic_loss(
                            args=args,
                            critic_accelerator=critic_accelerator,
                            fake_score_model=real_score_model,
                            transformer=transformer,
                            scheduler=critic_noise_scheduler,
                            # EvokeTeacher scoring branch (with a curriculum, sf_windowed_score=True -> v2v windowed scoring)
                            is_evoke_teacher_score=_sf_evoke_teacher,
                            sf_windowed_score=(_sf_evoke_teacher and bool(getattr(args.training_config, "sf_curriculum_enabled", False))),
                            # (s,wc); None=critic sees all 189 (bit-id). critic side: it only eats [prefix|window]
                            #   for forward+backward (the memory fix). same global_step as the generator -> same window.
                            sf_score_window=_sf_score_window,
                            sf_prefix_latents=_sf_prefix_latents,
                            sf_prompt_embeds_list=_sf_prompt_embeds_list,
                            sf_score_prompt_embeds=_prompt_jc,
                            sf_teacher_y=_teacher_y_jc,
                            sf_segment_frame_ranges=_ranges_jc,
                            # same mode / same 1x-slot convention as this step's generator (four-interval consistency).
                            #  These two quantities are computed inside `if TRAIN_GENERATOR:` and read here outside it, so a critic-only
                            #    step (dfake_gen_update_ratio>1) would see the defaults (False/None) and skip the frame-0 replacement the
                            #    generator did -- a silent four-interval mismatch. The validator forbids i2v+ratio>1; this assert is the
                            #    second line of defence.
                            sf_i2v_active=_sf_i2v_active,
                            sf_i2v_hist_latent=_sf_i2v_hist_latent,
                            # tier scoring + shared rollout (reused when the generator step already filled it)
                            sf_evoke_tier_score=_sf_evoke,
                            sf_shared_rollout=(
                                ((
                                     _sf_rollout_shared["pred_video"].detach().clone()
                                     if _sf_decouple_c
                                     else _sf_rollout_shared["pred_video"]
                                 ),
                                 _sf_rollout_shared["score_history"],
                                 _sf_rollout_shared["denoised_timestep_from"],
                                 _sf_rollout_shared["denoised_timestep_to"])
                                if _sf_rollout_shared else None
                            ),
                            noise=torch.randn(
                                noisy_model_input_shape, device=critic_accelerator.device, dtype=weight_dtype
                            ),
                            prompt_embeds=prompt_embeds,
                            # For VRAM manager
                            dmd_is_low_vram_mode=args.training_config.dmd_is_low_vram_mode,
                            vram_manager=vram_manager,
                            is_gan_low_vram_mode=args.training_config.is_gan_low_vram_mode,
                            # For Stage 1
                            is_keep_x0=True,
                            history_sizes=args.training_config.history_sizes,
                            # For Stage 2
                            is_enable_stage2=args.training_config.is_enable_stage2,
                            stage2_num_stages=args.training_config.stage2_num_stages,
                            stage2_num_inference_steps_list=args.validation_config.stage2_simulated_inference_steps,
                            # For DMD Main
                            denoising_step_list=list(args.training_config.dmd_denoising_step_list),
                            last_step_only=args.training_config.dmd_last_step_only,
                            last_section_grad_only=args.training_config.dmd_last_section_grad_only,
                            timestep_shift=args.training_config.dmd_timestep_shift,
                            use_dynamic_shifting=args.training_config.use_dynamic_shifting,
                            time_shift_type=args.training_config.time_shift_type,
                            # critic training band cap (actual-t semantics; None=old behavior, full band)
                            min_score_timestep=int(getattr(args.training_config, "critic_score_timestep_min", 0) or 0),
                            max_score_timestep=getattr(args.training_config, "critic_score_timestep_max", None),
                            # the curriculum takes over the tail-window size (same as the generator side)
                            num_critic_input_frames=(_sf_ncif if _sf_ncif is not None
                                                     else args.training_config.num_critic_input_frames),
                            num_rollout_sections=num_rollout_sections,
                            is_skip_first_section=args.training_config.is_skip_first_section,
                            is_amplify_first_chunk=args.training_config.is_amplify_first_chunk,
                            # For Easy Anti-Drifting
                            is_corrupt_history_latents=args.training_config.corrupt_history,
                            is_add_saturation=args.training_config.is_add_saturation,
                            # GT History
                            is_use_gt_history=USE_GT_HIST,
                            gt_history_latents=gt_history_latents_2,
                            gt_target_latents=gt_target_latents_2,
                            gt_x0_latents=gt_x0_latents_2,
                            # For VAE Re-Encode
                            vae=vae,
                            is_dmd_vae_decode=args.training_config.is_dmd_vae_decode,
                            # For Multi Stage Backward Simulated
                            is_multi_pyramid_stage_backward_simulated=args.training_config.is_multi_pyramid_stage_backward_simulated,
                            # For KV Cache
                            use_kv_cache=args.validation_config.use_kv_cache,
                            # For GAN
                            is_use_gan=USE_GAN,
                            is_separate_gan_grad=args.training_config.is_separate_gan_grad,
                            gan_base_critic_trainable_params=gan_base_critic_trainable_params,
                            gan_extra_critic_trainable_params=gan_extra_critic_trainable_params,
                            gan_vae_latents=gan_vae_latents,
                            gan_prompt_embeds=gan_prompt_embeds,
                            gan_d_weight=args.training_config.gan_d_weight,
                            aprox_r1=args.training_config.aprox_r1,
                            aprox_r2=args.training_config.aprox_r2,
                            r1_weight=args.training_config.r1_weight,
                            r2_weight=args.training_config.r2_weight,
                            r1_sigma=args.training_config.r1_sigma,
                            r2_sigma=args.training_config.r2_sigma,
                            # For Dynamic Timestep
                            dynamic_alpha=args.training_config.critic_dynamic_alpha,
                            dynamic_beta=args.training_config.critic_dynamic_beta,
                            dynamic_sample_type=args.training_config.critic_dynamic_sample_type,
                            global_step=global_step,
                            dynamic_step=args.training_config.critic_dynamic_step,
                            # GEO v2v DMD conditioning (None when GEO inactive).
                            gt_geo_all_data=_gt_geo_all_data,
                            gt_geo_attention_kwargs=_gt_geo_attention_kwargs,
                            # camera Plucker for critic (fake_score=stage1 full-res).
                            # the Evoke-Base critic has no plucker weights -> no pose passed.
                            cam_Ks=None if _sf_evoke else _cam_Ks_jc,
                            cam_c2ws=None if _sf_evoke else _cam_c2ws_jc,
                            cam_base_h=_dmd_cam_base_h,
                            cam_base_w=_dmd_cam_base_w,
                            cam_strategy=_dmd_cam_strategy,
                        )
                        if (
                            os.environ.get("SF_DECOUPLE_EQUIV_MODE")
                            and ((not _sf_decouple_c) or (_jc == _lrc))
                        ):
                            from scripts.training.tmp.test_decouple_equivalence import (
                                record_phase as _equiv_record_phase,
                            )
                            _equiv_record_phase(
                                "critic_evoke_teacher_out",
                                scalars={"critic_evoke_teacher_loss": critic_loss},
                            )
                        if (
                            os.environ.get("SF_VRAM_PROBE")
                            and _c3_enabled
                            and _c_work_index == 0
                            and global_step < 3
                            and torch.cuda.is_available()
                        ):
                            torch.cuda.synchronize()
                            print(
                                "[VRAM-PHASE teacher-first-forward] "
                                f"step={global_step} "
                                f"rank={torch.distributed.get_rank()} "
                                f"alloc={torch.cuda.memory_allocated() / 2**30:.1f}GB "
                                f"reserved={torch.cuda.memory_reserved() / 2**30:.1f}GB",
                                flush=True,
                            )
                        if not (
                            USE_GAN
                            and (args.training_config.is_gan_aprox_grad or args.training_config.is_gan_low_vram_mode)
                        ):
                            # the 2D SP x ZeRO-3 backward deadlock cannot be staggered away (the reduce-scatter
                            #   hook cannot be deferred out of the backward window; 5 smokes proved the critic backward hangs) -> switch to the only 2D SP configuration
                            #   EvokeTeacher has validated: ZeRO-2 + manual staggering (runner.py).
                            if _is2d_bwd() and not _dmd_zero3:
                                # disable ZeRO reduction during backward (enable_backward_allreduce=False) so only SP subgroup collectives
                                #   run and no ZeRO-2 DP reduction interleaves; allreduce_gradients() then reduces separately, staggered
                                #   against the SP collectives. x sp_world_size: the WORLD all-reduce averages the SP frame-slice partials as
                                #   if they were DP replicas (/G), and xG corrects that back to the clip-mean. gen/evoke are not SP and
                                #   compute redundantly within the group, so their WORLD average is already correct and is not scaled.
                                # decouple: every clip does a backward(critic_loss), accumulating G different clips. G must not be multiplied
                                #   in again -- the owner rotation already feeds all G clips' SP partials into the WORLD average. The off
                                #   path still needs x sp_world_size to cancel the /G applied to a single clip's SP partial.
                                #   The ZeRO-2 IPG epilogue must follow every backward: it completes the partition reduce, writes
                                #   averaged_gradients and resets params_already_reduced; deferring it to the last iteration makes the second
                                #   backward report "parameter already been reduced". engine.step() only on the last owner and at an
                                #   accelerate accumulation boundary. off/_Gc=1 is still a single pass.
                                _c_eng = critic_accelerator.deepspeed_engine_wrapped.engine
                                _c_sync = critic_accelerator.sync_gradients
                                _c_eng.set_gradient_accumulation_boundary(is_boundary=False)
                                _c_eng.enable_backward_allreduce = False
                                if _c3_enabled:
                                    # Every active EvokeTeacher sample is an SP partial spread
                                    # over G ranks.  DeepSpeed averages each slot over WORLD,
                                    # so W/bs restores the mean of this critic substep.
                                    # Padding groups execute identical hooks/collectives with
                                    # a zero-weight loss.
                                    _c_loss_scale = (
                                        _c_slot.loss_scale if _c_slot.active else 0.0
                                    )
                                else:
                                    _c_loss_scale = 1 if _sf_decouple_c else _sp_world_size
                                _c_eng.backward(critic_loss * _c_loss_scale)
                                _c_eng.enable_backward_allreduce = True
                                _c_last = (
                                    _c_slot.slot == _c3_plan.slots_per_step - 1
                                    if _c3_enabled
                                    else ((not _sf_decouple_c) or (_jc == _Gc - 1))
                                )
                                _c_boundary = _c_sync and _c_last
                                _c_eng.set_gradient_accumulation_boundary(is_boundary=_c_boundary)
                                _c_eng.allreduce_gradients()   # do the ZeRO-2 DP reduction / IPG epilogue separately after every backward
                                if (
                                    _c3_enabled
                                    and _c_slot.active
                                    and _lrc == 0
                                ):
                                    _c3_metric_sum.add_(
                                        critic_loss.detach().float().mean()
                                    )
                                    _c3_metric_count.add_(1)
                                if _c_boundary:
                                    _c_eng.step()                  # optimizer step + zero_grad (engine level)
                                    if _c3_enabled:
                                        critic_lr_scheduler.step()
                                        _c3_metric = torch.stack(
                                            (_c3_metric_sum, _c3_metric_count)
                                        )
                                        torch.distributed.all_reduce(_c3_metric)
                                        _c3_mean_loss = (
                                            _c3_metric[0]
                                            / _c3_metric[1].clamp_min(1)
                                        )
                                        _c3_substep_losses.append(_c3_mean_loss)
                                        logs[
                                            f"critic_loss_c{_c_slot.critic_substep}"
                                        ] = _c3_mean_loss.item()
                                        if (
                                            accelerator.is_main_process
                                            and global_step < 3
                                        ):
                                            print(
                                                "[MULTI-CRITIC-LW-STEP] "
                                                f"outer={global_step} "
                                                f"substep={_c_slot.critic_substep} "
                                                f"global_bs={_c_slot.global_batch_size} "
                                                f"loss={_c3_mean_loss.item():.6g}",
                                                flush=True,
                                            )
                                        _c3_metric_sum.zero_()
                                        _c3_metric_count.zero_()
                            else:
                                # mpu (ZeRO-3, the SP-SUM lives inside the engine.step of wrap_critic_engine_step) or G=1 (byte-id).
                                critic_accelerator.backward(critic_loss)
                        if _sf_decouple_c:
                            _oc_c.__exit__(None, None, None)
                        if _c3_enabled:
                            # Do not retain the final owner's large clip-dependent
                            # tensors or autograd log tensors into the next slot/step.
                            critic_loss = None
                            critic_log_dict = None
                            _prompt_jc = None
                            _ranges_jc = None
                            _cam_Ks_jc = None
                            _cam_c2ws_jc = None
                            _teacher_y_jc = None
                            if _c_last:
                                free_memory()
                    if _sf_decouple_c:
                        _dc_scope_obj_c.__exit__(None, None, None)

                    # pin: active on the ZeRO-3 mpu path (removes the block-loop all-gather interleaving), a no-op on the ZeRO-2 path (params are not sharded so there is no
                    #   ds_id), and a no-op at G=1. the pin on the critic (grad) path does not unpin itself across backward -> unpin_all() here is the fallback against
                    #   leaking across steps (with ZeRO-2/G=1 _PINNED is always empty -> no-op).
                    if _sp_world_size > 1:
                        from evoke.modules.evoke_teacher.sp_zero3 import unpin_all as _sp_unpin_all
                        _sp_unpin_all()

                    critic_grad_norm = None
                    if _c3_enabled:
                        assert len(_c3_substep_losses) == _critic_steps
                        critic_log_dict = {
                            "critic_loss": torch.stack(_c3_substep_losses).mean()
                        }
                        # DeepSpeedEngine.step already performed optimizer step,
                        # clipping, and zero_grad once per critic substep.
                    else:
                        if critic_accelerator.sync_gradients:
                            critic_params_to_clip = real_score_model.parameters()
                            critic_grad_norm = critic_accelerator.clip_grad_norm_(
                                critic_params_to_clip, args.training_config.max_grad_norm_critic
                            )

                        critic_log_dict["critic_loss"] = critic_loss
                        if critic_grad_norm is not None:
                            critic_log_dict["critic_grad_norm"] = critic_grad_norm

                        extra = critic_log_dict
                        extras_list.append(extra)
                        critic_log_dict = merge_dict_list(extras_list)
                        critic_optimizer.step()
                        critic_lr_scheduler.step()
                        critic_optimizer.zero_grad(set_to_none=True)

                    sf_prof_accum("critic", _sf_t_critic)   # accumulate the critic phase (all of the EvokeTeacher critic)
                    # second critic training -- the critic dedicated to the Evoke backbone (tail block pred_tail, keep-warp tier).
                    #   It has its own evoke_critic_accelerator and engine, so there is no conflict with the EvokeTeacher critic's
                    #   engine above; backward/step happen at the same grad-accum boundary (sync_gradients is shared across
                    #   accelerators via the GradientState singleton).
                    #   It reuses the gen shared rollout: warp cannot be re-rendered, so pred_tail is the tail of the full rollout
                    #   and the tail-block snapshot is score_history. adapters-on=critic, so this forward uses the critic-LoRA,
                    #   consistent with the EvokeTeacher critic above.
                    if (_sf_front_window and real_score_model_hb is not None
                            and _sf_rollout_shared and "pred_video" in _sf_rollout_shared):
                        _hb_front_frames = int(_sf_rollout_shared.get(
                            "sf_front_frames", (num_rollout_sections - 1) * latent_window_size))
                        _hb_pred_tail = _sf_rollout_shared["pred_video"][:, :, _hb_front_frames:]
                        _hb_tail_snapshot = _sf_rollout_shared["score_history"]
                        # [(5)f offload] (6)b before the Evoke critic: the Evoke base returns to GPU and the EvokeTeacher base is swapped out to CPU (frozen bases only).
                        #   device comes from evoke_critic_accelerator.device (= where this pass' noise / forward inputs live).
                        if _dual_offload:
                            _evoke_teacher_base_to(real_score_model, "cpu")   # per-expert: both experts go to CPU, freeing GPU for the Evoke critic
                            _offload_frozen_params_to(real_score_model_hb, evoke_critic_accelerator.device)
                        # the Evoke critic is non-SP full-sequence scoring: under decouple every card scores its own distinct
                        #   clip (sf_shared_rollout=this card's tail-block pred/snapshot) -> entering sp_decouple_scope makes the sync_tensor at 4580 inside _critic_loss
                        #   (generated/noise/timestep) skip the broadcast (using its own clip); there is no owner rotation (not SP, frame slices are not split);
                        #   backward = plain DP (24 cards each with a distinct clip -> mean-24). off -> broadcast from rank0 as usual (byte-identical).
                        _hb_dc_c = bool(getattr(args.training_config, "sf_decouple_rollout", False)) and _sp_world_size > 1
                        if _hb_dc_c:
                            from evoke.modules.evoke_teacher.sp_runtime import sp_decouple_scope as _sp_dc_scope_hb
                            _hb_dc_cm = _sp_dc_scope_hb(); _hb_dc_cm.__enter__()
                        if (
                            os.environ.get("SF_DECOUPLE_EQUIV_MODE")
                            and not _c3_enabled
                        ):
                            from scripts.training.tmp.test_decouple_equivalence import seed_phase as _equiv_seed_phase
                            _equiv_seed_phase("critic_evoke_score")
                        _hb_critic_kwargs = dict(
                            args=args,
                            critic_accelerator=evoke_critic_accelerator,
                            fake_score_model=real_score_model_hb,
                            transformer=transformer,
                            scheduler=critic_noise_scheduler,
                            # [(6)b] Evoke critic = tail-block tier scoring (keep-warp), a scope mutually exclusive with the EvokeTeacher critic front section.
                            is_evoke_teacher_score=False,
                            sf_evoke_tier_score=True,
                            sf_shared_rollout=(
                                _hb_pred_tail,
                                _hb_tail_snapshot,
                                _sf_rollout_shared["denoised_timestep_from"],
                                _sf_rollout_shared["denoised_timestep_to"],
                            ),
                            sf_prefix_latents=_sf_prefix_latents,
                            sf_prompt_embeds_list=_sf_prompt_embeds_list,
                            # Filled immediately before each critic substep so C3
                            # receives independent FM randomness (and C1 consumes
                            # exactly the same single draw as before).
                            noise=None,
                            prompt_embeds=prompt_embeds,
                            dmd_is_low_vram_mode=args.training_config.dmd_is_low_vram_mode,
                            vram_manager=vram_manager,
                            is_gan_low_vram_mode=args.training_config.is_gan_low_vram_mode,
                            is_keep_x0=True,
                            history_sizes=args.training_config.history_sizes,
                            is_enable_stage2=args.training_config.is_enable_stage2,
                            stage2_num_stages=args.training_config.stage2_num_stages,
                            stage2_num_inference_steps_list=args.validation_config.stage2_simulated_inference_steps,
                            denoising_step_list=list(args.training_config.dmd_denoising_step_list),
                            last_step_only=args.training_config.dmd_last_step_only,
                            last_section_grad_only=args.training_config.dmd_last_section_grad_only,
                            timestep_shift=args.training_config.dmd_timestep_shift,
                            use_dynamic_shifting=args.training_config.use_dynamic_shifting,
                            time_shift_type=args.training_config.time_shift_type,
                            # [(6)b] Evoke critic score-t cap (dual_teacher.evoke_critic_score_timestep_max; null -> all bands)
                            min_score_timestep=int(getattr(args.training_config, "critic_score_timestep_min", 0) or 0),
                            max_score_timestep=args.model_config.dual_teacher.evoke_critic_score_timestep_max,
                            # the tail block = 1 chunk = win frames; the prompt picks the last section (on the tier path mid picks the section -> N-1 = the tail section, consistent with (4)b).
                            num_critic_input_frames=latent_window_size,
                            num_rollout_sections=num_rollout_sections,
                            is_skip_first_section=args.training_config.is_skip_first_section,
                            is_amplify_first_chunk=args.training_config.is_amplify_first_chunk,
                            is_corrupt_history_latents=args.training_config.corrupt_history,
                            is_add_saturation=args.training_config.is_add_saturation,
                            is_use_gt_history=False,
                            vae=vae,
                            is_dmd_vae_decode=False,
                            is_multi_pyramid_stage_backward_simulated=args.training_config.is_multi_pyramid_stage_backward_simulated,
                            use_kv_cache=args.validation_config.use_kv_cache,
                            is_use_gan=False,
                            dynamic_alpha=args.training_config.critic_dynamic_alpha,
                            dynamic_beta=args.training_config.critic_dynamic_beta,
                            dynamic_sample_type=args.training_config.critic_dynamic_sample_type,
                            global_step=global_step,
                            dynamic_step=args.training_config.critic_dynamic_step,
                            # the camera teacher has plucker -> feed it the tail-block GT camera trajectory (the same slice as
                            #   teacher/critic on the (4)b compute_kl_grad side, keeping s_fake_hb(warp,plk)-s_hb(warp,plk) self-consistent).
                            #   with Evoke-Base (no plucker), sf_pose in _generator_loss still writes tail_cam_*, but the fwd ignores it -> harmless.
                            cam_Ks=_sf_rollout_shared.get("tail_cam_Ks"),
                            cam_c2ws=_sf_rollout_shared.get("tail_cam_c2ws"),
                            cam_base_h=_dmd_cam_base_h,
                            cam_base_w=_dmd_cam_base_w,
                            cam_strategy=_dmd_cam_strategy,
                        )
                        _hb_num_steps = _critic_steps if _c3_enabled else 1
                        _hb_sample_substep = (
                            (global_step + _c_group_index + _lrc) % _critic_steps
                            if _c3_enabled
                            else 0
                        )
                        _hb_substep_losses = []
                        for _hb_substep in range(_hb_num_steps):
                            # Independent FM noise/timestep sampling for each real
                            # critic optimizer update.
                            _hb_critic_kwargs["noise"] = torch.randn(
                                noisy_model_input_shape,
                                device=evoke_critic_accelerator.device,
                                dtype=weight_dtype,
                            )
                            evoke_critic_loss, evoke_critic_log_dict = _critic_loss(
                                **_hb_critic_kwargs
                            )
                            if (
                                os.environ.get("SF_DECOUPLE_EQUIV_MODE")
                                and not _c3_enabled
                            ):
                                from scripts.training.tmp.test_decouple_equivalence import (
                                    record_phase as _equiv_record_phase,
                                )
                                _equiv_record_phase(
                                    "critic_evoke_out",
                                    scalars={"critic_evoke_loss": evoke_critic_loss},
                                )
                            if _hb_dc_c and not _c3_enabled:
                                # Preserve the validated C1 context boundary.
                                _hb_dc_cm.__exit__(None, None, None)

                            _hb_active = (
                                (not _c3_enabled)
                                or _hb_substep == _hb_sample_substep
                            )
                            _hb_loss_scale = (
                                (
                                    _c_world
                                    / _c3_plan.bucket_sizes()[_hb_substep]
                                )
                                if _c3_enabled and _hb_active
                                else (0.0 if _c3_enabled else 1.0)
                            )
                            evoke_critic_accelerator.backward(
                                evoke_critic_loss * _hb_loss_scale
                            )
                            _hb_critic_grad_norm = None
                            if (
                                not _c3_enabled
                                and evoke_critic_accelerator.sync_gradients
                            ):
                                _hb_critic_grad_norm = (
                                    evoke_critic_accelerator.clip_grad_norm_(
                                        real_score_model_hb.parameters(),
                                        args.training_config.max_grad_norm_critic,
                                    )
                                )

                            # DeepSpeed backward already performs the real engine
                            # step at accumulation boundary; wrapper optimizer.step
                            # remains a no-op.  The scheduler must advance once per
                            # real critic update.
                            evoke_critic_optimizer.step()
                            evoke_critic_lr_scheduler.step()
                            evoke_critic_optimizer.zero_grad(set_to_none=True)

                            if _c3_enabled:
                                _hb_metric = torch.stack((
                                    (
                                        evoke_critic_loss.detach().float().mean()
                                        if _hb_active
                                        else torch.zeros(
                                            (),
                                            device=evoke_critic_accelerator.device,
                                            dtype=torch.float32,
                                        )
                                    ),
                                    torch.ones(
                                        (),
                                        device=evoke_critic_accelerator.device,
                                        dtype=torch.float32,
                                    ) if _hb_active else torch.zeros(
                                        (),
                                        device=evoke_critic_accelerator.device,
                                        dtype=torch.float32,
                                    ),
                                ))
                                torch.distributed.all_reduce(_hb_metric)
                                _hb_mean_loss = (
                                    _hb_metric[0] / _hb_metric[1].clamp_min(1)
                                )
                                _hb_substep_losses.append(_hb_mean_loss)
                                logs[
                                    f"evoke_critic_loss_c{_hb_substep}"
                                ] = _hb_mean_loss.item()
                                if (
                                    accelerator.is_main_process
                                    and global_step < 3
                                ):
                                    print(
                                        "[MULTI-CRITIC-HB-STEP] "
                                        f"outer={global_step} "
                                        f"substep={_hb_substep} "
                                        "global_bs="
                                        f"{_c3_plan.bucket_sizes()[_hb_substep]} "
                                        f"loss={_hb_mean_loss.item():.6g}",
                                        flush=True,
                                    )
                            else:
                                logs["evoke_critic_loss"] = (
                                    evoke_critic_loss.detach().mean().item()
                                )
                                if _hb_critic_grad_norm is not None:
                                    logs["evoke_critic_grad_norm"] = safe_item(
                                        _hb_critic_grad_norm
                                    )

                            evoke_critic_loss = None
                            evoke_critic_log_dict = None
                            _hb_critic_kwargs["noise"] = None
                            free_memory()

                        if _hb_dc_c and _c3_enabled:
                            _hb_dc_cm.__exit__(None, None, None)
                        if _c3_enabled:
                            logs["evoke_critic_loss"] = torch.stack(
                                _hb_substep_losses
                            ).mean().item()
                        _hb_critic_kwargs = None
                        _hb_pred_tail = None
                        _hb_tail_snapshot = None
                        free_memory()

                    if args.training_config.use_ema and ema_transformer is not None:
                        if (
                            global_step < args.training_config.ema_start_step
                            or not args.training_config.is_train_dmd
                            or TRAIN_GENERATOR
                        ):
                            if args.training_config.dmd_is_low_vram_mode:
                                vram_manager.move_to_cpu(real_score_model)
                                vram_manager.move_to_gpu(transformer, accelerator.device)

                    logs["critic_loss"] = critic_log_dict["critic_loss"].mean().item()
                    # same as the generator: a non-accumulation boundary has no grad norm, and a pure logging item must not interrupt training.
                    if "critic_grad_norm" in critic_log_dict:
                        logs["critic_grad_norm"] = safe_item(critic_log_dict["critic_grad_norm"])
                    if USE_GAN:
                        logs.update(
                            {
                                "denoising_loss": critic_log_dict["denoising_loss"],
                                "gan_D_loss": critic_log_dict["gan_D_loss"],
                                "r1_loss": critic_log_dict["r1_loss"],
                                "r2_loss": critic_log_dict["r2_loss"],
                            }
                        )

                    critic_loss = None
                    critic_grad_norm = None
                    if _c3_enabled:
                        assert bool(getattr(
                            args.training_config, "no_visualize", False
                        )), (
                            "[THROUGHPUT-B MULTI-CRITIC] tensor logs were disabled to save memory, "
                            "training_config.no_visualize=true is required"
                        )
                        critic_log_dict = None
                        extras_list = None
                        _sf_rollout_shared = None
                        _c_work = None
                        _c3_local_steps = None
                        _c3_plan = None
                        _c3_substep_losses = None
                    del critic_loss
                    del critic_grad_norm
                    free_memory()

                batch = None
                model_input = None
                prompt_embeds = None
                indices_hidden_states = None
                indices_latents_history_short = None
                indices_latents_history_mid = None
                indices_latents_history_long = None
                latents_history_short = None
                latents_history_mid = None
                latents_history_long = None
                gan_vae_latents = None
                gan_prompt_embeds = None
                gt_history_latents = None
                gt_target_latents = None
                gt_x0_latents = None
                gt_history_latents_2 = None
                gt_target_latents_2 = None
                gt_x0_latents_2 = None
                ode_latents = None
                ode_prompt_embeds = None
                text_prompt_raws = None
                text_prompt_embeds = None
                del batch
                del model_input
                del prompt_embeds
                del indices_hidden_states
                del indices_latents_history_short
                del indices_latents_history_mid
                del indices_latents_history_long
                del latents_history_short
                del latents_history_mid
                del latents_history_long
                del gan_vae_latents
                del gan_prompt_embeds
                del gt_history_latents
                del gt_target_latents
                del gt_x0_latents
                del gt_history_latents_2
                del gt_target_latents_2
                del gt_x0_latents_2
                del ode_latents
                del ode_prompt_embeds
                del text_prompt_raws
                del text_prompt_embeds
                free_memory()

            if accelerator.sync_gradients:
                if args.training_config.use_ema and ema_transformer is not None:
                    if (
                        global_step < args.training_config.ema_start_step
                        or not args.training_config.is_train_dmd
                        or TRAIN_GENERATOR
                    ):
                        ema_transformer.step(transformer.parameters())

                progress_bar.update(1)
                global_step += 1

                # per-step phase timings (only with SF_PROFILE=1); rollout_tf=rollout-warp; score=genloss-rollout;
                #   critic+misc=total-genloss-gen_bwd. used to judge whether putting SP on the student rollout (option A) is worth it.
                _sf_snap = sf_prof_step_end()
                if _sf_snap is not None and accelerator.is_main_process:
                    _tot = _sf_snap.get("__total__", 0.0) or 1e-9
                    _roll = _sf_snap.get("rollout", 0.0); _warp = _sf_snap.get("warp", 0.0)
                    _gl = _sf_snap.get("genloss", 0.0); _gbwd = _sf_snap.get("gen_bwd", 0.0)
                    _geor = _sf_snap.get("georeg", 0.0); _crit = _sf_snap.get("critic", 0.0)
                    # genloss = rollout (warp included) + teacher-score-fwd + georeg-fwd (the GEO backward is folded into gen_bwd).
                    _rtf = _roll - _warp
                    _score = _gl - _roll - _geor            # teacher scoring forward = genloss - rollout - georeg forward
                    _other = _tot - _gl - _gbwd - _crit     # remainder = dataload + checkpoint + misc
                    print(f"[SF-PROFILE] step={global_step} total={_tot:.1f}s | "
                          f"rollout={_rtf:.1f}s({100*_rtf/_tot:.0f}%) warp={_warp:.1f}s({100*_warp/_tot:.0f}%) "
                          f"score_fwd={_score:.1f}s({100*_score/_tot:.0f}%) gen_bwd={_gbwd:.1f}s({100*_gbwd/_tot:.0f}%) "
                          f"georeg_fwd={_geor:.1f}s({100*_geor/_tot:.0f}%) critic={_crit:.1f}s({100*_crit/_tot:.0f}%) "
                          f"other={_other:.1f}s({100*_other/_tot:.0f}%)", flush=True)

                # record the moment "this step ended" -- the start of the next round minus it is the time spent **waiting on the worker**.
                #   written straight into globals() to avoid adding a global declaration for one diagnostic value. not entered when the env is off.
                if _pp_mod.enabled():
                    globals()["_PP_LAST_END"] = _pp_mod.mark()

                if args.training_config.is_train_dmd:
                    if accelerator.is_main_process and VISUALIZE:
                        phase_name = "dmd_visualize"
                        if args.training_config.dmd_is_low_vram_mode:
                            vram_manager.move_to_cpu(transformer)
                            vram_manager.move_to_cpu(real_score_model)

                        if vae is None:
                            vae = AutoencoderKLWan.from_pretrained(
                                args.model_config.pretrained_model_name_or_path,
                                subfolder="vae",
                                revision=args.model_config.revision,
                                variant=args.model_config.variant,
                                torch_dtype=(torch.float32 if args.model_config.upcast_vae else weight_dtype),
                                device_map=(None if _dmd_zero3 else accelerator.device),  # [ZeRO-3] transformers/diffusers reject device_map (global zero3 flag); set None, placed later by .to(target_device)
                            )
                            if args.model_config.enable_slicing:
                                vae.enable_slicing()
                            if args.model_config.enable_tiling:
                                vae.enable_tiling()

                        if args.training_config.dmd_is_low_vram_mode and args.training_config.is_dmd_vae_decode:
                            vram_manager.move_to_gpu(vae, accelerator.device)
                        else:
                            vae.to(accelerator.device, non_blocking=True)
                        latents_mean = (
                            torch.tensor(vae.config.latents_mean)
                            .view(1, vae.config.z_dim, 1, 1, 1)
                            .to(vae.device, vae.dtype)
                        )
                        latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(
                            vae.device, vae.dtype
                        )

                        for tracker in accelerator.trackers:
                            if tracker.name == "wandb":
                                video_logs = []

                                def decode_latent(latent):
                                    with torch.no_grad():
                                        latent = latent[0:1]  # [1, C, T, H, W]
                                        latent = latent / latents_std + latents_mean
                                        return vae.decode(latent)[0]  # [1, C, T, H, W]

                                def prepare_for_saving(tensor, fps=30, caption=None):
                                    tensor = (tensor * 0.5 + 0.5).clamp(0, 1).detach()
                                    tensor = tensor.permute(0, 2, 1, 3, 4)
                                    video_array = (tensor * 255).cpu().numpy().astype(np.uint8)
                                    return wandb.Video(video_array, fps=fps, format="mp4", caption=caption)

                                log_configs = [
                                    (
                                        critic_log_dict,
                                        ["critictrain_latent", "critictrain_noisy_latent", "critictrain_pred_image"],
                                    ),
                                ]
                                generator_keys = [
                                    "dmdtrain_clean_latent",
                                    "dmdtrain_pred_real_image",
                                    "dmdtrain_pred_fake_image",
                                ]
                                if args.training_config.is_decouple_dmd:
                                    generator_keys.extend(["dmdtrain_ca_noisy_latent", "dmdtrain_dm_noisy_latent"])
                                else:
                                    generator_keys.append("dmdtrain_noisy_latent")
                                log_configs.append((generator_log_dict, generator_keys))
                                for log_dict, keys in log_configs:
                                    for key in keys:
                                        if key in log_dict:
                                            with torch.no_grad():
                                                decoded = decode_latent(log_dict[key])
                                            video_logs.append(prepare_for_saving(decoded, fps=30, caption=key))
                                            del decoded

                                tracker.log({phase_name: video_logs}, step=global_step)

                        if (
                            args.training_config.is_train_dmd and args.training_config.dmd_is_low_vram_mode
                        ) or args.data_config.use_stage3_dataset:
                            if (
                                not args.training_config.is_dmd_vae_decode
                                and not args.training_config.is_use_reward_model
                                and not args.training_config.is_smoothness_loss
                            ):
                                vae = None
                            free_memory()

                        if vae is not None:
                            vae.to("cpu", non_blocking=True)

                    optimizer.zero_grad(set_to_none=True)
                    critic_optimizer.zero_grad(set_to_none=True)
                    if "generator_log_dict" in locals():
                        if generator_log_dict is not None:
                            generator_log_dict.clear()
                        del generator_log_dict
                    if "critic_log_dict" in locals():
                        if critic_log_dict is not None:
                            critic_log_dict.clear()
                        del critic_log_dict
                    if "video_logs" in locals():
                        del video_logs
                    if "log_configs" in locals():
                        del log_configs
                    free_memory()

                if global_step % args.training_config.checkpointing_steps == 0:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")

                    states = {
                        "dataloader": train_dataloader,
                    }
                    dcp_dir = os.path.join(save_path, "distributed_checkpoint")
                    dcp.save(states, checkpoint_id=dcp_dir)
                    states = None
                    del states
                    free_memory()

                    if accelerator.is_main_process or accelerator.distributed_type == DistributedType.DEEPSPEED:
                        # Checkpoint cleanup runs on rank-0 only to avoid concurrent rmtree races.
                        if accelerator.is_main_process and args.training_config.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            # Only count standard checkpoint-{N} directories (exclude checkpoint-N-final etc.).
                            checkpoints = [
                                d for d in checkpoints
                                if d.startswith("checkpoint-") and d.split("-")[-1].isdigit()
                            ]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            if len(checkpoints) >= args.training_config.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.training_config.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                accelerator.print(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint, ignore_errors=True)
                        accelerator.wait_for_everyone()

                        if args.training_config.save_checkpoints_custom:
                            # see the _z3_gather_trainable definition above: all ranks must enter the
                            #   GatheredParameters with-block (only rank0 writes to disk); under ZeRO-2 it is a no-op, byte-identical.
                            with _z3_gather_trainable(transformer):
                                if accelerator.is_main_process:
                                    save_model_checkpoint(
                                        transformer=transformer,
                                        args=args,
                                        save_path=save_path,
                                        weight_dtype=weight_dtype,
                                        unwrap_model_fn=unwrap_model,
                                        get_peft_model_state_dict_fn=get_peft_model_state_dict,
                                        collate_lora_metadata_fn=_collate_lora_metadata,
                                        save_extra_components_fn=save_extra_components,
                                        pipeline_class=EvokePipeline,
                                        norm_layer_prefixes=NORM_LAYER_PREFIXES,
                                    )
                            if args.training_config.is_train_dmd and _sf_evoke_teacher:
                                # the wrapper's critic-LoRA is written to disk directly (the evoke-PEFT save semantics do not apply)
                                with _z3_gather_trainable(real_score_model):
                                    if accelerator.is_main_process:
                                        _et_critic_dir = os.path.join(save_path, "critic")
                                        os.makedirs(_et_critic_dir, exist_ok=True)
                                        from safetensors.torch import save_file as _sf_save_file
                                        _et_sd = {k: v.detach().to("cpu", torch.float32).contiguous()
                                                  for k, v in unwrap_model(real_score_model).trainable_state_dict().items()}
                                        _sf_save_file(_et_sd, os.path.join(_et_critic_dir, "critic_evoke_teacher_lora.safetensors"))
                                        del _et_sd
                            # write the Evoke-specific critic-LoRA to disk (mirrors the EvokeTeacher one above;
                            #   EvokeTransformer3DModel has no trainable_state_dict -> take the requires_grad params == the critic-LoRA).
                            # hoisted out of the `sf_warmstart_dir` if below: the dual-teacher critic has nothing to do with
                            #   warm-start provenance, and nesting made it (a) skipped entirely on any dual run without sf_warmstart_dir -> silent critic-LoRA loss,
                            #   (b) enter _z3_gather_trainable on rank0 only (the enclosing if carried is_main_process) -> a ZeRO-3 collective mismatch.
                            if _sf_dual and real_score_model_hb is not None:
                                with _z3_gather_trainable(real_score_model_hb):
                                    if accelerator.is_main_process:
                                        _hb_critic_dir = os.path.join(save_path, "critic_evoke")
                                        os.makedirs(_hb_critic_dir, exist_ok=True)
                                        from safetensors.torch import save_file as _sf_save_file
                                        _hb_sd = {k: v.detach().to("cpu", torch.float32).contiguous()
                                                  for k, v in unwrap_model(real_score_model_hb).named_parameters()
                                                  if v.requires_grad}
                                        _sf_save_file(_hb_sd, os.path.join(_hb_critic_dir, "critic_evoke_lora.safetensors"))
                                        del _hb_sd
                            # the evoke (non-evoke_teacher) critic PEFT save. It must be an independent `if` with an explicit
                            #   `not _sf_evoke_teacher`, since the evoke_teacher critic is written by the block above. Hanging it off the
                            #   `sf_warmstart_dir` branch instead made it fire on the EvokeTeacherScoreWrapper (no peft_config -> rank0
                            #   raises while the other ranks sit in the collective) or skip the critic LoRA silently.
                            if args.training_config.is_train_dmd and not _sf_evoke_teacher:
                                with _z3_gather_trainable(real_score_model):
                                    if accelerator.is_main_process:
                                        save_model_checkpoint(
                                            transformer=real_score_model,
                                            args=args,
                                            save_path=os.path.join(save_path, "critic"),
                                            weight_dtype=weight_dtype,
                                            unwrap_model_fn=unwrap_model,
                                            get_peft_model_state_dict_fn=get_peft_model_state_dict,
                                            collate_lora_metadata_fn=_collate_lora_metadata,
                                            save_extra_components_fn=save_extra_components,
                                            pipeline_class=EvokePipeline,
                                            norm_layer_prefixes=NORM_LAYER_PREFIXES,
                                        )
                            # the continued-training marker is written inside the ckpt, not left to the output_dir name alone --
                            #   this run's global_step restarts from 0, so the checkpoint-N it produces is named the same as a "from scratch" one and becomes indistinguishable once copied elsewhere.
                            #   this file records the provenance + the "equivalent steps = source steps + N" conversion.
                            if getattr(args.training_config, "sf_warmstart_dir", None) and accelerator.is_main_process:
                                _ws_src = str(args.training_config.sf_warmstart_dir).rstrip("/")
                                _ws_base = os.path.basename(_ws_src)
                                # the step count is parsed from an anchored step marker rather than by scraping every digit out of the
                                #   directory name: joining all digits only works for a literal `checkpoint-N`, while a real source such as
                                #   `dmdfinal_v3_stusp_a_fulln_48g_formal_ck1600` also swallows the 3 of v3 and the 48 of 48g.
                                #   findall + [-1] because chained warm-starts carry several markers (..._ck150_adapter_ck450 -> 450), and
                                #   anchoring on checkpoint/step/ck keeps unrelated trailing digits out (..._ck1600_bf16 -> 1600).
                                #   No marker -> "" -> the text form ("source steps + N") is written instead of inventing a number.
                                _ws_marks = re.findall(r"(?:checkpoint|step|ck)[-_]?(\d+)", _ws_base, re.IGNORECASE)
                                _ws_n = _ws_marks[-1] if _ws_marks else ""
                                _ws_frz = int(getattr(args.training_config, "sf_gen_freeze_steps", 0) or 0)
                                # re-read in place rather than relying on the loader-side local still being in scope.
                                _ws_co_doc = bool(getattr(args.training_config, "sf_warmstart_critic_only", False))
                                # the three provenance lines are built here instead of inside nested
                                #   f-string conditionals, because what they may truthfully claim depends on which load path ran.
                                # NOTE: the equivalent step count **differs** between critic and student: the student is frozen for the
                                #   first K steps (critic only), so K must be subtracted for it. writing a single number over-reported the student by K.
                                _ws_stu_adv = max(0, int(global_step) - _ws_frz)
                                _ws_eq_critic = (f"**{int(_ws_n) + int(global_step)}**  (= {_ws_n} + {global_step})"
                                                 if _ws_n else f"**source steps + {global_step}**")
                                if _ws_co_doc:
                                    # critic_only never loaded generator weights from the warm-start source, so counting the student's
                                    #   steps from that source would be plain wrong -- its lineage is the merged start instead.
                                    _ws_eq_student = (f"**merged start + {_ws_stu_adv}** (critic_only: no generator weight came from the "
                                                      f"warm-start source; the lineage is `{args.model_config.transformer_model_name_or_path}`)")
                                    _ws_inherited = (f"critic LoRA only (sf_warmstart_critic_only=true; generator + memory patch come from "
                                                     f"the merged start `{args.model_config.transformer_model_name_or_path}`)")
                                else:
                                    _ws_eq_student = ((f"**{int(_ws_n) + _ws_stu_adv}**" if _ws_n else f"**source steps + {_ws_stu_adv}**")
                                                      + f"  (= the line above - {_ws_frz} frozen steps; student params did not move bit-wise during the freeze)")
                                    _ws_inherited = "generator LoRA + memory patch + critic LoRA (all three)"
                                with open(os.path.join(save_path, "WARMSTART.md"), "w", encoding="utf-8") as _wf:
                                    _wf.write(
                                        f"# this ckpt is a **continued-training** artifact, not trained from scratch\n\n"
                                        f"- warm-start source: `{_ws_src}`\n"
                                        f"- this run's global_step: **{global_step}** (restarted from 0)\n"
                                        f"- equivalent cumulative steps (critic): {_ws_eq_critic}\n"
                                        f"- equivalent cumulative steps (student/generator): {_ws_eq_student}\n"
                                        f"- inherited weights: {_ws_inherited} -- fail-fast validated at startup\n"
                                        f"- **not** inherited: Adam momentum / lr_scheduler progress / dataloader position / RNG\n"
                                        f"- student freeze steps (critic only): {int(getattr(args.training_config, 'sf_gen_freeze_steps', 0) or 0)}\n"
                                        f"- training config: see this run's `{os.path.join(args.output_dir, 'config.json')}`\n")
                        else:
                            accelerator.save_state(save_path)
                            if args.training_config.is_train_dmd:
                                critic_accelerator.save_state(os.path.join(save_path, "critic"))
                        accelerator.print(f"Saved state to {save_path}")
                        # Create weights/ symlink view for inference convenience.
                        if accelerator.is_main_process:
                            organize_checkpoint_weights_view(save_path)
                            if args.training_config.is_train_dmd:
                                organize_checkpoint_weights_view(os.path.join(save_path, "critic"))

                    if args.training_config.use_ema and ema_transformer is not None:
                        ema_transformer.save_pretrained(
                            args,
                            os.path.join(save_path, "model_ema"),
                            args.model_config.transformer_model_name_or_path,
                            lora_config=transformer_lora_config,
                            transformer_additional_kwargs=transformer_additional_kwargs,
                        )

                if (
                    args.validation_config.validation_prompts is not None
                    and global_step % args.validation_config.validation_steps == 0
                ) or (
                    args.validation_config.first_step_valid
                    and global_step == (initial_global_step + 1)
                ):
                    if args.training_config.is_train_dmd and args.training_config.dmd_is_low_vram_mode:
                        vram_manager.move_to_cpu(real_score_model)

                    if args.training_config.is_train_dmd:
                        optimizer.zero_grad(set_to_none=True)
                        critic_optimizer.zero_grad(set_to_none=True)

                        if "generator_log_dict" in locals():
                            if generator_log_dict is not None:
                                generator_log_dict.clear()
                            del generator_log_dict
                        if "critic_log_dict" in locals():
                            if critic_log_dict is not None:
                                critic_log_dict.clear()
                            del critic_log_dict

                        free_memory()

                    if (
                        args.training_config.use_ema_validation
                        and args.training_config.use_ema
                        and ema_transformer is not None
                        and global_step >= args.training_config.ema_start_step
                    ):
                        accelerator.print("Starting EMA store and copy_to...")
                        ema_transformer.store(transformer.parameters())
                        ema_state_dict = gather_zero3ema(accelerator, ema_transformer)
                        transformer.load_state_dict({"module." + k: v for k, v in ema_state_dict.items()})
                        accelerator.print("EMA store and copy_to completed")
                        ema_state_dict = None
                        del ema_state_dict

                    free_memory()
                    if accelerator.is_main_process:
                        with torch.no_grad():
                            if vae is None:
                                vae = AutoencoderKLWan.from_pretrained(
                                    args.model_config.pretrained_model_name_or_path,
                                    subfolder="vae",
                                    revision=args.model_config.revision,
                                    variant=args.model_config.variant,
                                    torch_dtype=(torch.float32 if args.model_config.upcast_vae else weight_dtype),
                                    device_map=(None if _dmd_zero3 else accelerator.device),  # [ZeRO-3] transformers/diffusers reject device_map (global zero3 flag); set None, placed later by .to(target_device)
                                )
                                if args.model_config.enable_slicing:
                                    vae.enable_slicing()
                                if args.model_config.enable_tiling:
                                    vae.enable_tiling()

                            if text_encoder is None:
                                text_encoder = UMT5EncoderModel.from_pretrained(
                                    args.model_config.pretrained_model_name_or_path,
                                    subfolder="text_encoder",
                                    revision=args.model_config.revision,
                                    variant=args.model_config.variant,
                                    dtype=weight_dtype,
                                    device_map=(None if _dmd_zero3 else accelerator.device),  # [ZeRO-3] transformers/diffusers reject device_map (global zero3 flag); set None, placed later by .to(target_device)
                                )

                            if args.data_config.use_stage1_dataset or args.training_config.offload:
                                vae.to(accelerator.device, non_blocking=True)
                                text_encoder.to(accelerator.device, non_blocking=True)

                            pipe = EvokePipeline.from_pretrained(
                                args.model_config.pretrained_model_name_or_path,
                                vae=vae,
                                transformer=unwrap_model(transformer),
                                tokenizer=tokenizer,
                                text_encoder=text_encoder,
                                scheduler=noise_scheduler,
                                revision=args.model_config.revision,
                                variant=args.model_config.variant,
                                torch_dtype=weight_dtype,
                            )

                            # Inject GEO noise configs into the validation pipeline.
                            _inject_geo_cfg = getattr(args.model_config, "geometric_state", None)
                            _inject_st_cfg = getattr(_inject_geo_cfg, "short_tier_noise", None) if _inject_geo_cfg is not None else None
                            if _inject_st_cfg is not None:
                                pipe._short_tier_noise_cfg = {
                                    "enabled": bool(getattr(_inject_st_cfg, "enabled", False)),
                                    "sigma_min": float(getattr(_inject_st_cfg, "sigma_min", 0.2)),
                                    "sigma_max": float(getattr(_inject_st_cfg, "sigma_max", 0.6)),
                                    "target_tiers": list(getattr(_inject_st_cfg, "target_tiers", ["prefix", "prev_short"]) or []),
                                    "apply_at_inference": bool(getattr(_inject_st_cfg, "apply_at_inference", True)),
                                    "sigma_lock_per_rollout": bool(getattr(_inject_st_cfg, "sigma_lock_per_rollout", False)),
                                }
                            if _inject_geo_cfg is not None:
                                pipe._geo_vsnoise_cfg = {
                                    "enabled": bool(getattr(_inject_geo_cfg, "visibility_aware_noise", False)),
                                    "sigma_invisible": float(getattr(_inject_geo_cfg, "warp_noise_sigma_invisible", 0.8)),
                                    "sigma_min": float(getattr(_inject_geo_cfg, "warp_noise_sigma_min", 0.111)),
                                    "sigma_max": float(getattr(_inject_geo_cfg, "warp_noise_sigma_max", 0.135)),
                                    "visible_token_threshold": float(getattr(_inject_geo_cfg, "visible_token_threshold", 0.1)),
                                    "rope_alignment": bool(getattr(_inject_geo_cfg, "rope_alignment", True)),
                                    "prefix_idx_mode": str(getattr(_inject_geo_cfg, "prefix_idx_mode", "zero")),
                                    "warp_rope_mode": str(getattr(_inject_geo_cfg, "warp_rope_mode", "overlap_noise")),
                                    "warp_keep_clean_anchor": bool(getattr(_inject_geo_cfg, "warp_keep_clean_anchor", False)),
                                    "invisible_history_noise": bool(getattr(_inject_geo_cfg, "geo_invisible_history_noise", False)),
                                    "warp_lag_chunks": int(getattr(_inject_geo_cfg, "warp_lag_chunks", 0)),
                                    # val uses the same convention as training: coarse-stage noise RoPE is center-aligned (switch on = full centering, no alpha).
                                    "warp_rope_noise_center_align": bool(getattr(_inject_geo_cfg, "warp_rope_noise_center_align", False)),
                                    # warp is injected only at pyramid stage0 (the Geo convention; default False = injected at all three stages, like stage1).
                                    # keeps val consistent with inference; combined with persistent decoding it shows when training pulls the pred first-frame distribution into continuity (flicker -> no flicker).
                                    "geo_warp_stage0_only": bool(getattr(_inject_geo_cfg, "warp_stage0_only", False)),
                                }
                                # Cloud recall backend: validation is fed the same warp distribution as
                                # training. resolve_cloud_warp is shared by all four producers of this
                                # dict, so they cannot drift apart.
                                _cw_val = getattr(_inject_geo_cfg, "cloud_warp", None)
                                if _cw_val is not None and bool(getattr(_cw_val, "enabled", False)):
                                    from evoke.utils.train_config import resolve_cloud_warp
                                    pipe._geo_vsnoise_cfg.update(resolve_cloud_warp(_cw_val))
                                    pipe._geo_vsnoise_cfg.update({
                                        "recon_backend": "da3",   # cloud pipeline switch; the depth backend is depth_backend
                                        "warp_lag_chunks": 0,     # for v2v the ref is fully ready + ingest is synchronous -> lag0 uses the nearest frames, giving higher coverage
                                    })

                            all_videos = []
                            all_prompts = []
                            # Per-prompt GEO dump dirs for 3-column visualization.
                            all_geo_dump_dirs = []
                            for _vp_idx, validation_prompt in enumerate(args.validation_config.validation_prompts):
                                pipeline_args = {
                                    "prompt": args.data_config.id_token + validation_prompt,
                                    "negative_prompt": "oversaturated, garish colors, color shift, hue shift, color drift, inconsistent colors, color cast, color banding, flickering, jittery motion, abrupt transitions, sudden scene changes, temporal inconsistency, static, still picture, blurred details, subtitles, style, works, paintings, images, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, three legs, many people in the background, walking backwards, messy background",
                                    "guidance_scale": args.validation_config.validation_guidance_scale,
                                    "num_frames": args.validation_config.validation_max_num_frames,
                                    "height": args.validation_config.validation_height,
                                    "width": args.validation_config.validation_width,
                                    "num_inference_steps": args.validation_config.num_inference_steps,
                                    # ---- Dynamic Shifting ----
                                    "use_dynamic_shifting": args.validation_config.use_dynamic_shifting,
                                    "time_shift_type": args.validation_config.time_shift_type,
                                    # For Stage 1
                                    "history_sizes": args.training_config.history_sizes,
                                    "latent_window_size": args.validation_config.validation_latent_window_size[0],
                                    "is_keep_x0": True,
                                    "use_kv_cache": args.validation_config.use_kv_cache,
                                    # For Stage 2
                                    "is_enable_stage2": args.training_config.is_enable_stage2,
                                    "stage2_num_stages": args.training_config.stage2_num_stages,
                                    "stage2_num_inference_steps_list": args.validation_config.stage2_simulated_inference_steps,
                                    "vae_decode_type": args.training_config.vae_decode_type,
                                    # For Stage 3
                                    "use_dmd": args.training_config.is_train_dmd,
                                    "is_amplify_first_chunk": args.training_config.is_amplify_first_chunk,
                                }

                                pipeline_args = _inject_v2v_cam_into_pipeline_args(args, _vp_idx, pipeline_args)

                                # Enable GEO intermediate dump only when warp is active.
                                _geo_dump_dir = None
                                if pipeline_args.get("use_geometric_state", False):
                                    _geo_dump_dir = os.path.join(
                                        args.output_dir,
                                        f"_geo_viz_step{global_step}_validation_vp{_vp_idx}",
                                    )
                                    os.makedirs(_geo_dump_dir, exist_ok=True)
                                    pipe._geo_dump_dir = _geo_dump_dir
                                else:
                                    pipe._geo_dump_dir = None
                                all_geo_dump_dirs.append(_geo_dump_dir)

                                videos, prompt = log_validation(
                                    pipe=pipe,
                                    args=args,
                                    accelerator=accelerator,
                                    pipeline_args=pipeline_args,
                                )

                                # Clear dump handle to prevent accidental triggers.
                                pipe._geo_dump_dir = None

                                all_videos.extend(videos)
                                all_prompts.extend([prompt] * len(videos))

                            for tracker in accelerator.trackers:
                                phase_name = "validation"
                                if tracker.name == "wandb":
                                    video_logs = []

                                    _vp_videos_log = args.validation_config.validation_videos or []
                                    _vp_poses_log = args.validation_config.validation_pose_paths or []
                                    # Enable cam viz HUD when either camera_control or GEO is active.
                                    _cam_enabled_log = (
                                        (getattr(args.model_config, "camera_control", None) is not None
                                         and args.model_config.camera_control.enabled)
                                        or getattr(args.training_config, "use_geometric_state", False)
                                    )
                                    os.makedirs(args.output_dir, exist_ok=True)
                                    for i, (video, prompt) in enumerate(zip(all_videos, all_prompts)):
                                        filename = os.path.join(
                                            args.output_dir,
                                            f"global_step{global_step}_{phase_name}_video_{i}_{prompt[:25].replace(' ', '_')}.mp4",
                                        )
                                        _gt_path = _vp_videos_log[i] if i < len(_vp_videos_log) else None
                                        _pose_path = _vp_poses_log[i] if i < len(_vp_poses_log) else None
                                        # Use save_rgb_video (cv2 backend) to avoid R/B channel swap.
                                        if _cam_enabled_log and _gt_path and _pose_path:
                                            from evoke.utils.ev_validation import (
                                                combine_gt_pred_with_cam_viz,
                                                save_rgb_video,
                                            )
                                            _geo_dd = all_geo_dump_dirs[i] if i < len(all_geo_dump_dirs) else None
                                            _video_to_save = combine_gt_pred_with_cam_viz(
                                                pred_video=video,
                                                gt_video_path=_gt_path,
                                                pose_path=_pose_path,
                                                height=args.validation_config.validation_height,
                                                width=args.validation_config.validation_width,
                                                num_frames=args.validation_config.validation_max_num_frames,
                                                ref_seconds=args.validation_config.validation_video_seconds,
                                                start_seconds=args.validation_config.validation_video_start_seconds,
                                                latent_window_size=args.validation_config.validation_latent_window_size[0],
                                                vae_stride_t=4,
                                                target_fps=args.data_config.target_fps,
                                                source_fps=args.validation_config.validation_pose_source_fps,
                                                source_resolution=tuple(args.validation_config.validation_pose_source_resolution),
                                                pose_type=args.validation_config.validation_pose_type,
                                                warp_dump_dir=_geo_dd,
                                                max_rotation_deg=getattr(args.validation_config, "validation_pose_max_rotation_deg", 0.0),
                                            )
                                            save_rgb_video(_video_to_save, filename, fps=args.data_config.target_fps)
                                            # Remove GEO dump dir after combining to avoid accumulation.
                                            if _geo_dd and os.path.isdir(_geo_dd):
                                                import shutil as _shutil
                                                try:
                                                    _shutil.rmtree(_geo_dd)
                                                except Exception as _e:
                                                    logger.warning(f"failed to remove GEO viz dump dir {_geo_dd}: {_e}")
                                        else:
                                            export_to_video(video, filename, fps=args.data_config.target_fps)
                                        video_logs.append(
                                            wandb.Video(filename, caption=f"{i}: {prompt}", format="mp4")
                                        )

                                    tracker.log({phase_name: video_logs}, step=global_step)

                            videos = None
                            prompt = None
                            all_videos = None
                            all_prompts = None
                            video_logs = None
                            del videos
                            del prompt
                            del all_videos
                            del all_prompts
                            del video_logs
                            free_memory()

                            if (
                                args.training_config.is_train_dmd and args.training_config.dmd_is_low_vram_mode
                            ) or args.data_config.use_stage3_dataset:
                                if (
                                    not args.training_config.is_dmd_vae_decode
                                    and not args.training_config.is_use_reward_model
                                    and not args.training_config.is_smoothness_loss
                                ):
                                    vae = None
                                text_encoder = None
                                free_memory()

                            del pipe
                            free_memory()

                    # Synchronize all ranks so non-main ranks wait for rank-0 validation to finish.
                    accelerator.wait_for_everyone()

                    if (
                        args.training_config.use_ema_validation
                        and args.training_config.use_ema
                        and ema_transformer is not None
                        and global_step >= args.training_config.ema_start_step
                    ):
                        accelerator.wait_for_everyone()
                        ema_transformer.restore(transformer.parameters())

            # Offload VAE/T5 after each step for offline stage-1.
            if args.data_config.use_stage1_dataset and not args.data_config.use_multi_dataset:
                if vae is not None:
                    vae.to("cpu", non_blocking=True)
                if text_encoder is not None:
                    text_encoder.to("cpu", non_blocking=True)
                free_memory()

            if args.training_config.offload:
                if vae is not None:
                    vae.to(accelerator.device, non_blocking=True)
                if text_encoder is not None:
                    text_encoder.to(accelerator.device, non_blocking=True)

            if prof is not None:
                prof.step()

            # when SF_VRAM_PROBE is set, rank0 prints this step's (gen+critic backward) peak VRAM and resets the window
            #   (the per-step peak = the max alloc/reserved within that step). used for dual-expert offload memory acceptance: the measured peak decides 16-card feasibility.
            #   NOTE: it also prints host RSS (this rank) + node MemAvailable: offload moves EvokeTeacher (~56G) to CPU, and with 8 ranks/node
            # that is ~448G host RAM worst case (a host-OOM vector invisible in the GPU peak); node_mem_avail dropping near 0 = danger.
            #   no env -> the branch is not entered, bit-identical.
            if os.environ.get("SF_VRAM_PROBE") and accelerator.is_main_process and torch.cuda.is_available():
                _vp_alloc = torch.cuda.max_memory_allocated() / 1e9
                _vp_resv = torch.cuda.max_memory_reserved() / 1e9
                _vp_rss = _vp_avail = -1.0
                try:
                    with open("/proc/self/status") as _vpf:
                        for _vpl in _vpf:
                            if _vpl.startswith("VmRSS:"):
                                _vp_rss = float(_vpl.split()[1]) / 1e6  # kB -> GB
                                break
                    with open("/proc/meminfo") as _vpf:
                        for _vpl in _vpf:
                            if _vpl.startswith("MemAvailable:"):
                                _vp_avail = float(_vpl.split()[1]) / 1e6  # kB -> GB
                                break
                except Exception:
                    pass
                print(f"[VRAM-PROBE] step={global_step} peak_alloc={_vp_alloc:.1f}GB "
                      f"peak_reserved={_vp_resv:.1f}GB rank0_rss={_vp_rss:.1f}GB "
                      f"node_mem_avail={_vp_avail:.1f}GB", flush=True)
                torch.cuda.reset_peak_memory_stats()

            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.training_config.max_train_steps:
                break

            logs = None
            del logs
            free_memory()

    if prof is not None:
        prof.stop()
        print(f"Profiler stopped. Check results in: {args.training_config.profile_out_dir}")

    # Final save: LoRA weights and extra components.
    if args.training_config.is_train_dmd and not _dmd_zero3:
        # ZeRO-3: real_score_model is a sharded engine, and .to("cpu") would move the sharded .data and break shard management; sharding already saves memory, so no swap-out is needed.
        real_score_model.to("cpu", non_blocking=True)
    accelerator.wait_for_everyone()
    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}-final")
    if args.training_config.use_ema and ema_transformer is not None:
        ema_transformer.save_pretrained(
            args,
            os.path.join(save_path, "model_ema"),
            args.model_config.transformer_model_name_or_path,
            lora_config=transformer_lora_config,
            transformer_additional_kwargs=transformer_additional_kwargs,
        )
    # generator LoRA final save: **all ranks** must enter the GatheredParameters with-block
    #   (only rank0 writes to disk); under ZeRO-2 _z3_params_to_fetch -> [] -> no-op -> byte-identical.
    with _z3_gather_trainable(transformer):
        if accelerator.is_main_process:
            modules_to_save = {}
            model_to_save = unwrap_model(transformer)
            original_dtype = next(model_to_save.parameters()).dtype
            # ZeRO-3: skip the whole-model dtype .to() (it would re-point the sharded base .data and break shard bookkeeping; inside the gather the LoRA is already full,
            #   and its dtype is the training dtype, so no further cast is needed for saving). ZeRO-2 keeps the original cast behavior.
            if args.model_config.bnb_quantization_config_path is None and not _dmd_zero3:
                if args.training_config.upcast_before_saving:
                    model_to_save.to(torch.float32)
                else:
                    model_to_save.to(weight_dtype)

            # Camera-only mode has no LoRA adapter to save.
            if not _cam_only_mode:
                transformer_lora_layers = get_peft_model_state_dict(model_to_save)
                if args.model_config.train_norm_layers:
                    transformer_norm_layers = {
                        f"transformer.{name}": param
                        for name, param in model_to_save.named_parameters()
                        if any(k in name for k in NORM_LAYER_PREFIXES)
                    }
                    transformer_lora_layers = {
                        **transformer_lora_layers,
                        **transformer_norm_layers,
                    }
                modules_to_save["transformer"] = model_to_save

                EvokePipeline.save_lora_weights(
                    save_directory=save_path,
                    transformer_lora_layers=transformer_lora_layers,
                    **_collate_lora_metadata(modules_to_save),
                )

                # save_lora_weights only dumps the default adapter; dump GEO adapter separately.
                from evoke.utils.utils_base import _dump_geo_adapter as _dump_geo_adapter_helper
                _dump_geo_adapter_helper(model_to_save, save_path, get_peft_model_state_dict)
            save_extra_components(args, model=model_to_save, output_dir=save_path)
            if not _dmd_zero3:
                model_to_save.to(original_dtype)

    # the final save must write a critic too, or `checkpoint-{N}-final` is a dead end for the evoke_teacher path:
    #   the warm-start loader hard-asserts `critic/critic_evoke_teacher_lora.safetensors`, and whenever max_steps
    #   is not a multiple of checkpointing_steps the last stretch of critic training is lost outright.
    #   The three branches below mirror the periodic save (evoke_teacher / evoke / dual-teacher evoke critic).
    #   real_score_model was moved to CPU above only under `not _dmd_zero3`; a state_dict from CPU is fine, and
    #   under ZeRO-3 it stayed on GPU so the gather behaves as in the periodic path. Every `with` is entered under
    #   a rank-independent condition (only the disk write is rank0) -> no collective mismatch.
    if args.training_config.is_train_dmd and args.training_config.save_checkpoints_custom:
        if _sf_evoke_teacher:
            with _z3_gather_trainable(real_score_model):
                if accelerator.is_main_process:
                    _fin_critic_dir = os.path.join(save_path, "critic")
                    os.makedirs(_fin_critic_dir, exist_ok=True)
                    from safetensors.torch import save_file as _sf_save_file
                    _fin_sd = {k: v.detach().to("cpu", torch.float32).contiguous()
                               for k, v in unwrap_model(real_score_model).trainable_state_dict().items()}
                    _sf_save_file(_fin_sd, os.path.join(_fin_critic_dir, "critic_evoke_teacher_lora.safetensors"))
                    del _fin_sd
        else:
            with _z3_gather_trainable(real_score_model):
                if accelerator.is_main_process:
                    save_model_checkpoint(
                        transformer=real_score_model,
                        args=args,
                        save_path=os.path.join(save_path, "critic"),
                        weight_dtype=weight_dtype,
                        unwrap_model_fn=unwrap_model,
                        get_peft_model_state_dict_fn=get_peft_model_state_dict,
                        collate_lora_metadata_fn=_collate_lora_metadata,
                        save_extra_components_fn=save_extra_components,
                        pipeline_class=EvokePipeline,
                        norm_layer_prefixes=NORM_LAYER_PREFIXES,
                    )
        if _sf_dual and real_score_model_hb is not None:
            with _z3_gather_trainable(real_score_model_hb):
                if accelerator.is_main_process:
                    _fin_hb_dir = os.path.join(save_path, "critic_evoke")
                    os.makedirs(_fin_hb_dir, exist_ok=True)
                    from safetensors.torch import save_file as _sf_save_file
                    _fin_hb_sd = {k: v.detach().to("cpu", torch.float32).contiguous()
                                  for k, v in unwrap_model(real_score_model_hb).named_parameters()
                                  if v.requires_grad}
                    _sf_save_file(_fin_hb_sd, os.path.join(_fin_hb_dir, "critic_evoke_lora.safetensors"))
                    del _fin_hb_sd

    if accelerator.is_main_process:
        if args.training_config.use_ema and ema_transformer is not None:
            ema_state_dict = gather_zero3ema(accelerator, ema_transformer)
            transformer.load_state_dict(ema_state_dict)

        # Run final validation.
        if args.validation_config.validation_prompts is not None:
            with torch.no_grad():
                if vae is None:
                    vae = AutoencoderKLWan.from_pretrained(
                        args.model_config.pretrained_model_name_or_path,
                        subfolder="vae",
                        revision=args.model_config.revision,
                        variant=args.model_config.variant,
                        torch_dtype=(torch.float32 if args.model_config.upcast_vae else weight_dtype),
                        device_map=(None if _dmd_zero3 else accelerator.device),  # [ZeRO-3] transformers/diffusers reject device_map (global zero3 flag); set None, placed later by .to(target_device)
                    )
                    if args.model_config.enable_slicing:
                        vae.enable_slicing()
                    if args.model_config.enable_tiling:
                        vae.enable_tiling()

                if text_encoder is None:
                    text_encoder = UMT5EncoderModel.from_pretrained(
                        args.model_config.pretrained_model_name_or_path,
                        subfolder="text_encoder",
                        revision=args.model_config.revision,
                        variant=args.model_config.variant,
                        dtype=weight_dtype,
                        device_map=(None if _dmd_zero3 else accelerator.device),  # [ZeRO-3] transformers/diffusers reject device_map (global zero3 flag); set None, placed later by .to(target_device)
                    )

                if args.data_config.use_stage1_dataset:
                    vae.to(accelerator.device, non_blocking=True)
                    text_encoder.to(accelerator.device, non_blocking=True)

                pipe = EvokePipeline.from_pretrained(
                    args.model_config.pretrained_model_name_or_path,
                    vae=vae,
                    transformer=unwrap_model(transformer),
                    tokenizer=tokenizer,
                    text_encoder=text_encoder,
                    scheduler=noise_scheduler,
                    revision=args.model_config.revision,
                    variant=args.model_config.variant,
                    torch_dtype=weight_dtype,
                )

                # Inject GEO noise configs into the final-step pipeline.
                _inject_geo_cfg_f = getattr(args.model_config, "geometric_state", None)
                _inject_st_cfg_f = getattr(_inject_geo_cfg_f, "short_tier_noise", None) if _inject_geo_cfg_f is not None else None
                if _inject_st_cfg_f is not None:
                    pipe._short_tier_noise_cfg = {
                        "enabled": bool(getattr(_inject_st_cfg_f, "enabled", False)),
                        "sigma_min": float(getattr(_inject_st_cfg_f, "sigma_min", 0.2)),
                        "sigma_max": float(getattr(_inject_st_cfg_f, "sigma_max", 0.6)),
                        "target_tiers": list(getattr(_inject_st_cfg_f, "target_tiers", ["prefix", "prev_short"]) or []),
                        "apply_at_inference": bool(getattr(_inject_st_cfg_f, "apply_at_inference", True)),
                        "sigma_lock_per_rollout": bool(getattr(_inject_st_cfg_f, "sigma_lock_per_rollout", False)),
                    }
                if _inject_geo_cfg_f is not None:
                    pipe._geo_vsnoise_cfg = {
                        "enabled": bool(getattr(_inject_geo_cfg_f, "visibility_aware_noise", False)),
                        "sigma_invisible": float(getattr(_inject_geo_cfg_f, "warp_noise_sigma_invisible", 0.8)),
                        "sigma_min": float(getattr(_inject_geo_cfg_f, "warp_noise_sigma_min", 0.111)),
                        "sigma_max": float(getattr(_inject_geo_cfg_f, "warp_noise_sigma_max", 0.135)),
                        "visible_token_threshold": float(getattr(_inject_geo_cfg_f, "visible_token_threshold", 0.1)),
                        "rope_alignment": bool(getattr(_inject_geo_cfg_f, "rope_alignment", True)),
                        "prefix_idx_mode": str(getattr(_inject_geo_cfg_f, "prefix_idx_mode", "zero")),
                        "warp_rope_mode": str(getattr(_inject_geo_cfg_f, "warp_rope_mode", "overlap_noise")),
                        "warp_keep_clean_anchor": bool(getattr(_inject_geo_cfg_f, "warp_keep_clean_anchor", False)),
                        "invisible_history_noise": bool(getattr(_inject_geo_cfg_f, "geo_invisible_history_noise", False)),
                        "warp_lag_chunks": int(getattr(_inject_geo_cfg_f, "warp_lag_chunks", 0)),
                        # val uses the same convention as training: coarse-stage noise RoPE is center-aligned (switch on = full centering, no alpha).
                        "warp_rope_noise_center_align": bool(getattr(_inject_geo_cfg_f, "warp_rope_noise_center_align", False)),
                    }
                    # Shared resolver with the mid-run validation block above. Note this block used to omit
                    # geo_warp_warm_encode, so final validation ignored cloud_warp.warp_warm_encode; it is
                    # picked up now (inert while no config sets it).
                    _cw_val_f = getattr(_inject_geo_cfg_f, "cloud_warp", None)
                    if _cw_val_f is not None and bool(getattr(_cw_val_f, "enabled", False)):
                        from evoke.utils.train_config import resolve_cloud_warp
                        pipe._geo_vsnoise_cfg.update(resolve_cloud_warp(_cw_val_f))
                        pipe._geo_vsnoise_cfg.update({
                            "recon_backend": "da3",   # cloud pipeline switch; the depth backend is depth_backend
                            "warp_lag_chunks": 0,     # for v2v the ref is fully ready + ingest is synchronous -> lag0 uses the nearest frames, giving higher coverage
                        })

                all_videos = []
                all_prompts = []
                all_geo_dump_dirs = []
                for _vp_idx, validation_prompt in enumerate(args.validation_config.validation_prompts):
                    pipeline_args = {
                        "prompt": args.data_config.id_token + validation_prompt,
                        "negative_prompt": "oversaturated, garish colors, color shift, hue shift, color drift, inconsistent colors, color cast, color banding, flickering, jittery motion, abrupt transitions, sudden scene changes, temporal inconsistency, static, still picture, blurred details, subtitles, style, works, paintings, images, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, three legs, many people in the background, walking backwards, messy background",
                        "guidance_scale": args.validation_config.validation_guidance_scale,
                        "num_frames": args.validation_config.validation_max_num_frames,
                        "height": args.validation_config.validation_height,
                        "width": args.validation_config.validation_width,
                        "num_inference_steps": args.validation_config.num_inference_steps,
                        # ---- Dynamic Shifting ----
                        "use_dynamic_shifting": args.validation_config.use_dynamic_shifting,
                        "time_shift_type": args.validation_config.time_shift_type,
                        # For Stage 1
                        "history_sizes": args.training_config.history_sizes,
                        "latent_window_size": args.validation_config.validation_latent_window_size[0],
                        "is_keep_x0": True,
                        "use_kv_cache": args.validation_config.use_kv_cache,
                        # For Stage 2
                        "is_enable_stage2": args.training_config.is_enable_stage2,
                        "stage2_num_stages": args.training_config.stage2_num_stages,
                        "stage2_num_inference_steps_list": args.validation_config.stage2_simulated_inference_steps,
                        "vae_decode_type": args.training_config.vae_decode_type,
                        # For Stage 3
                        "use_dmd": args.training_config.is_train_dmd,
                        "is_amplify_first_chunk": args.training_config.is_amplify_first_chunk,
                    }
                    pipeline_args = _inject_v2v_cam_into_pipeline_args(args, _vp_idx, pipeline_args)

                    _geo_dump_dir = None
                    if pipeline_args.get("use_geometric_state", False):
                        _geo_dump_dir = os.path.join(
                            args.output_dir,
                            f"_geo_viz_step{global_step}_final_step_validation_vp{_vp_idx}",
                        )
                        os.makedirs(_geo_dump_dir, exist_ok=True)
                        pipe._geo_dump_dir = _geo_dump_dir
                    else:
                        pipe._geo_dump_dir = None
                    all_geo_dump_dirs.append(_geo_dump_dir)

                    videos, prompt = log_validation(
                        pipe=pipe,
                        args=args,
                        accelerator=accelerator,
                        pipeline_args=pipeline_args,
                    )
                    pipe._geo_dump_dir = None

                    all_videos.extend(videos)
                    all_prompts.extend([prompt] * len(videos))

                for tracker in accelerator.trackers:
                    phase_name = "final_step_validation"
                    if tracker.name == "wandb":
                        video_logs = []

                        _vp_videos_f = args.validation_config.validation_videos or []
                        _vp_poses_f = args.validation_config.validation_pose_paths or []
                        # Enable cam viz HUD when either camera_control or GEO is active.
                        _cam_enabled_f = (
                            (getattr(args.model_config, "camera_control", None) is not None
                             and args.model_config.camera_control.enabled)
                            or getattr(args.training_config, "use_geometric_state", False)
                        )
                        os.makedirs(args.output_dir, exist_ok=True)
                        for i, (video, prompt) in enumerate(zip(all_videos, all_prompts)):
                            filename = os.path.join(
                                args.output_dir,
                                f"global_step{global_step}_{phase_name}_video_{i}_{prompt[:25].replace(' ', '_')}.mp4",
                            )
                            _gt_path = _vp_videos_f[i] if i < len(_vp_videos_f) else None
                            _pose_path = _vp_poses_f[i] if i < len(_vp_poses_f) else None
                            if _cam_enabled_f and _gt_path and _pose_path:
                                from evoke.utils.ev_validation import (
                                    combine_gt_pred_with_cam_viz,
                                    save_rgb_video,
                                )
                                _geo_dd = all_geo_dump_dirs[i] if i < len(all_geo_dump_dirs) else None
                                _vid_save = combine_gt_pred_with_cam_viz(
                                    pred_video=video,
                                    gt_video_path=_gt_path,
                                    pose_path=_pose_path,
                                    height=args.validation_config.validation_height,
                                    width=args.validation_config.validation_width,
                                    num_frames=args.validation_config.validation_max_num_frames,
                                    ref_seconds=args.validation_config.validation_video_seconds,
                                    start_seconds=args.validation_config.validation_video_start_seconds,
                                    latent_window_size=args.validation_config.validation_latent_window_size[0],
                                    vae_stride_t=4,
                                    target_fps=args.data_config.target_fps,
                                    source_fps=args.validation_config.validation_pose_source_fps,
                                    source_resolution=tuple(args.validation_config.validation_pose_source_resolution),
                                    pose_type=args.validation_config.validation_pose_type,
                                    warp_dump_dir=_geo_dd,
                                    max_rotation_deg=getattr(args.validation_config, "validation_pose_max_rotation_deg", 0.0),
                                )
                                save_rgb_video(_vid_save, filename, fps=args.data_config.target_fps)
                                # Remove GEO dump dir after combining.
                                if _geo_dd and os.path.isdir(_geo_dd):
                                    import shutil as _shutil
                                    try:
                                        _shutil.rmtree(_geo_dd)
                                    except Exception as _e:
                                        logger.warning(f"failed to remove GEO viz dump dir {_geo_dd}: {_e}")
                            else:
                                export_to_video(video, filename, fps=args.data_config.target_fps)
                            video_logs.append(wandb.Video(filename, caption=f"{i}: {prompt}", format="mp4"))

                        tracker.log({phase_name: video_logs}, step=global_step)

    accelerator.end_training()


def _inject_v2v_cam_into_pipeline_args(args, vp_idx, pipeline_args):
    """Inject ref video and camera pose into pipeline_args for V2V+cam validation."""
    _vp_videos = args.validation_config.validation_videos or []
    _vp_poses = args.validation_config.validation_pose_paths or []
    _vid = _vp_videos[vp_idx] if vp_idx < len(_vp_videos) else None
    _pose = _vp_poses[vp_idx] if vp_idx < len(_vp_poses) else None

    _start_s = float(getattr(args.validation_config, "validation_video_start_seconds", 0.0) or 0.0)
    if _vid:
        from evoke.utils.ev_validation import load_ref_video_for_v2v
        pipeline_args["video"] = load_ref_video_for_v2v(
            _vid,
            height=args.validation_config.validation_height,
            width=args.validation_config.validation_width,
            seconds=args.validation_config.validation_video_seconds,
            target_fps=args.data_config.target_fps,
            source_fps=args.validation_config.validation_pose_source_fps,
            start_seconds=_start_s,
        )

    if _pose:
        from evoke.utils.ev_validation import load_pose_for_v2v
        # Compute the required pose length matching pipeline's num_latent_sections logic.
        _W = args.validation_config.validation_latent_window_size[0]
        _vae_stride_t = 4
        _window_pix = (_W - 1) * _vae_stride_t + 1
        _num_secs = max(1, (args.validation_config.validation_max_num_frames + _window_pix - 1) // _window_pix)
        _ref_pix = max(0, int(round(args.validation_config.validation_video_seconds * args.data_config.target_fps))) if _vid else 0
        _ref_lat = ((_ref_pix - 1) // _vae_stride_t + 1) if _ref_pix > 0 else 0
        _needed_lat = _ref_lat + _num_secs * _W
        _pose_num_target_frames = (_needed_lat - 1) * _vae_stride_t + 1
        _Ks, _c2ws = load_pose_for_v2v(
            _pose,
            target_height=args.validation_config.validation_height,
            target_width=args.validation_config.validation_width,
            source_resolution=tuple(args.validation_config.validation_pose_source_resolution),
            pose_type=args.validation_config.validation_pose_type,
            num_target_frames=_pose_num_target_frames,
            target_fps=args.data_config.target_fps,
            source_fps=args.validation_config.validation_pose_source_fps,
            start_seconds=_start_s,
            max_rotation_deg=getattr(args.validation_config, "validation_pose_max_rotation_deg", 0.0),
        )
        pipeline_args["lingbot_Ks"] = _Ks
        pipeline_args["lingbot_c2ws"] = _c2ws

    # GEO validation: mirror training setting when validation flag is not explicitly set.
    _val_geo = getattr(args.validation_config, "use_geometric_state", None)
    _train_geo = getattr(args.training_config, "use_geometric_state", False)
    _geo_enabled = _val_geo if _val_geo is not None else _train_geo
    if _geo_enabled:
        _video_tensor = pipeline_args.get("video", None)
        assert _video_tensor is not None and _vid, (
            "validation use_geometric_state=True requires validation_videos[vp_idx] to provide a ref video "
            "(the Pi3X source pixels are taken from the first frame)"
        )
        pipeline_args["use_geometric_state"] = True
        # Use first video frame as Pi3X source image.
        pipeline_args["image"] = _video_tensor[0:1].clone()
        # Load FrameBank retrieve config from yaml (same as training and inference).
        _geo_cfg = getattr(args.model_config, "geometric_state", None)
        _geo_retrieve_cfg = getattr(_geo_cfg, "retrieve", None) if _geo_cfg is not None else None
        if _geo_retrieve_cfg is not None:
            import math as _math
            pipeline_args["geo_score"] = str(getattr(_geo_retrieve_cfg, "score", "v1"))
            pipeline_args["geo_nearby_k"] = int(getattr(_geo_retrieve_cfg, "nearby_k", 0))
            pipeline_args["geo_select_k"] = int(getattr(_geo_retrieve_cfg, "select_k", 5))
            pipeline_args["geo_top_k"] = int(getattr(_geo_retrieve_cfg, "select_k", 5))  # legacy alias
            pipeline_args["geo_bank_max"] = int(getattr(_geo_retrieve_cfg, "bank_max", 0)) or None
            pipeline_args["geo_init_k"] = int(getattr(_geo_retrieve_cfg, "init_k", 10))
            if pipeline_args["geo_score"] == "v3":
                pipeline_args["geo_score_kwargs"] = {
                    "depth": float(getattr(_geo_retrieve_cfg, "v3_depth", 5.0)),
                    "fov_rad": _math.radians(float(getattr(_geo_retrieve_cfg, "v3_fov_deg", 60.0))),
                }
            else:
                pipeline_args["geo_score_kwargs"] = {}

    return pipeline_args


@torch.no_grad()
def log_validation(
    pipe,
    args,
    accelerator,
    pipeline_args,
):
    logger.info(
        f"Running validation... \n Generating {args.validation_config.num_validation_videos} videos with prompt: {pipeline_args['prompt']}."
    )

    pipe = pipe.to(accelerator.device)

    # run inference
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None

    videos = []
    for _ in range(args.validation_config.num_validation_videos):
        video = pipe(**pipeline_args, generator=generator, output_type="np").frames[0]
        videos.append(video)

    del pipe
    free_memory()

    return videos, pipeline_args["prompt"]


if __name__ == "__main__":
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None,
                        help="Single legacy yaml containing all fields.")
    parser.add_argument("--training_config", type=str, default=None,
                        help="Primary training yaml (split-yaml style).")
    parser.add_argument("--data_config", type=str, default=None,
                        help="Data yaml path; stored as data_config.data_yaml_path.")
    parser.add_argument("--model_config", type=str, default=None,
                        help="Optional extra model_config override yaml.")
    args = parser.parse_args()

    from evoke.utils.yaml_inherit import load_yaml_with_inheritance as _load_yaml

    schema = OmegaConf.structured(Args)
    if args.config is not None:
        conf = OmegaConf.merge(schema, _load_yaml(args.config))
    else:
        assert args.training_config is not None, "Either --training_config or --config is required."
        pieces = [_load_yaml(args.training_config)]
        if args.model_config:
            pieces.append(_load_yaml(args.model_config))
        conf = OmegaConf.merge(schema, *pieces)
        if args.data_config:
            conf.data_config.data_yaml_path = args.data_config

    global_rank = int(os.environ.get("RANK", -1))
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != conf.training_config.local_rank:
        conf.training_config.local_rank = env_local_rank

    assert (
        len(conf.validation_config.validation_latent_window_size) == 1
        and len(conf.validation_config.validation_stream_chunk_size) == 1
    ), "Only a single value is currently supported for validation_latent_window_size and validation_stream_chunk_size"

    assert not (conf.data_config.use_stage1_dataset and conf.training_config.offload), (
        "use_stage1_dataset and offload cannot both be True"
    )

    assert not (conf.data_config.use_stage1_dataset and conf.training_config.offload), (
        "use_stage1_dataset and offload cannot both be True"
    )

    if conf.model_config.lora_layers is not None:
        assert len(conf.model_config.lora_target_modules) == 0, (
            f"Error: lora_target_modules length is {len(conf.model_config.lora_target_modules)}, expected 0 when lora_layers is not None."
        )

    if conf.training_config.efficient_sample:
        assert conf.training_config.pyramid_sample_mode == "full", (
            f"efficient_sample requires pyramid_sample_mode='full', got '{conf.training_config.pyramid_sample_mode}'"
        )

    if conf.data_config.dataset_sampling_ratios:
        assert conf.data_config.use_stage1_dataset, (
            "dataset_sampling_ratios is only supported when use_stage1_dataset=True"
        )
        if len(conf.data_config.instance_data_root) != len(conf.data_config.dataset_sampling_ratios):
            raise ValueError(
                f"Length mismatch: instance_data_root ({len(conf.data_config.instance_data_root)}) "
                f"vs dataset_sampling_ratios ({len(conf.data_config.dataset_sampling_ratios)})"
            )

        basenames = []
        for temp_key, temp_value in zip(conf.data_config.instance_data_root, conf.data_config.dataset_sampling_ratios):
            basename = temp_key.rstrip("/")
            if basename in basenames:
                raise ValueError(f"Duplicate dataset name: {basename}")
            basenames.append(basename)

    if conf.data_config.single_res:
        assert conf.data_config.force_rebuild, "force_rebuild must be True when single_res is enabled"

    # Multi-term memory patch must be enabled when any memory-patch training flag is set.
    if (
        conf.training_config.is_train_full_multi_term_memory_patchg
        or conf.training_config.is_train_lora_multi_term_memory_patchg
        or conf.training_config.zero_history_timestep
    ):
        assert conf.training_config.has_multi_term_memory_patch, "Missing clean patch embedding configuration."
        assert conf.training_config.is_enable_stage1, (
            "is_enable_stage1 must be enabled when using clean patch embedding."
        )

    if conf.training_config.restrict_lora:
        assert conf.training_config.restrict_self_attn, (
            "Self-attention restriction must be enabled when restricting LoRA."
        )

    if conf.training_config.is_train_restrict_lora:
        assert conf.training_config.restrict_lora, (
            "LoRA restriction must be enabled when training with LoRA restriction."
        )

    assert not (
        conf.training_config.is_train_full_multi_term_memory_patchg
        and conf.training_config.is_train_lora_multi_term_memory_patchg
    ), (
        "Both 'is_train_full_multi_term_memory_patchg' and 'is_train_lora_multi_term_memory_patchg' cannot be True at the same time."
    )
    assert not (
        conf.training_config.is_train_full_patch_embedding and conf.training_config.is_train_lora_patch_embedding
    ), "Both 'is_train_full_patch_embedding' and 'is_train_lora_patch_embedding' cannot be True at the same time."

    assert not (conf.training_config.use_error_recycling and conf.training_config.corrupt_history), (
        "Both 'use_error_recycling' and 'corrupt_history' cannot be True at the same time."
    )

    if conf.training_config.is_enable_stage2:
        if (
            not conf.training_config.is_train_dmd
            and not conf.training_config.is_use_ode_regression
            and not conf.training_config.is_dump_ode_traj
        ):
            assert conf.training_config.use_dynamic_shifting is False, (
                "Dynamic shifting cannot be used with pyramid sampling unless "
                "is_train_dmd / is_use_ode_regression / is_dump_ode_traj is True."
            )

    if conf.training_config.is_use_ode_regression:
        assert conf.training_config.use_dynamic_shifting, (
            "use_dynamic_shifting must be True when is_use_ode_regression is enabled."
        )

    if conf.validation_config.use_kv_cache:
        assert conf.training_config.restrict_self_attn, "When use_kv_cache=True, restrict_self_attn must also be True!"

    assert not (conf.training_config.use_error_recycling and conf.training_config.corrupt_history), (
        "Both 'use_error_recycling' and 'corrupt_history' cannot be True at the same time."
    )

    assert not (conf.training_config.use_error_recycling and conf.training_config.corrupt_model_input), (
        "Both 'use_error_recycling' and 'corrupt_model_input' cannot be True at the same time."
    )

    # Error recycling: history-tier apply_error_injection is stage1-single-tensor only. The WARP-injection
    # path (_geo_inject_warp_error in materialize) + per-pyramid-stage banking (_flow_loss list-branch) +
    # 3-resolution bucket registration ARE stage2-ready, so stage2/NaViT is allowed iff explicitly opted in
    # via allow_error_recycling_stage2 (warp-injection only; mem/history-tier injection still won't fire).
    # See
    assert not (
        conf.training_config.use_error_recycling
        and (conf.training_config.is_enable_stage2 or conf.training_config.is_navit_pyramid)
        and not conf.training_config.allow_error_recycling_stage2
    ), (
        "use_error_recycling on stage2/NaViT pyramid requires allow_error_recycling_stage2=true "
        "(warp-injection path only; history-tier apply_error_injection remains stage1-only)."
    )

    # depth_sample_ratio length must match max_error_depth when depth bucketing is active.
    if conf.training_config.use_error_recycling and int(conf.training_config.max_error_depth) > 1:
        assert len(conf.training_config.depth_sample_ratio) == int(conf.training_config.max_error_depth), (
            f"depth_sample_ratio length {len(conf.training_config.depth_sample_ratio)} must equal "
            f"max_error_depth {conf.training_config.max_error_depth}."
        )

    if conf.training_config.is_multi_pyramid_stage_backward_simulated:
        assert conf.training_config.is_enable_stage2, (
            "Multi_Pyramid_Stage_Backward_Simulated requires is_enable_stage2 to be enabled"
        )

    if conf.training_config.use_ema_validation:
        assert conf.training_config.use_ema, "EMA validation requires use_ema to be enabled"

    if conf.training_config.is_use_reward_model:
        assert conf.training_config.reward_weight_vq > 0 or conf.training_config.reward_weight_mq > 0, (
            "At least one of reward_weight_vq or reward_weight_mq must be greater than 0 when using reward model"
        )

    if conf.training_config.is_use_gan:
        assert conf.training_config.is_train_dmd, "GAN training requires is_train_dmd to be enabled"
        assert conf.training_config.is_use_gan_hooks or conf.training_config.is_use_gan_final, (
            "GAN training requires either is_use_gan_hooks or is_use_gan_final to be enabled"
        )

    if conf.training_config.stage_cold_start_step is not None:
        assert conf.training_config.stage_cold_start_step <= conf.training_config.cold_start_step, (
            f"stage_cold_start_step ({conf.training_config.stage_cold_start_step}) must be less than or equal to cold_start_step ({conf.training_config.cold_start_step})"
        )

    if conf.training_config.is_decouple_dmd:
        assert conf.training_config.decouple_ca_start_step >= conf.training_config.generator_dynamic_step, (
            "decouple_ca_start_step must be greater than or equal to generator_dynamic_step"
        )

        assert conf.training_config.decouple_ca_end_step >= conf.training_config.generator_dynamic_step, (
            "decouple_ca_end_step must be greater than or equal to generator_dynamic_step"
        )

    main(conf)
