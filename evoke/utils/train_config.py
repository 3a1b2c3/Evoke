import os
from dataclasses import dataclass, field, fields
from typing import Any, List, Optional, Tuple, Union


@dataclass
class ReportTo:
    tracker_name: str = field(default="evoke")
    wandb_name: str = field(default="test_run")
    report_to: str = field(
        default="wandb",
        metadata={"choices": ["wandb", "tensorboard", "comet_ml", "all"]},
    )


@dataclass
class DataConfig:
    use_shuffle: bool = field(default=False)
    pin_memory: bool = field(default=False)
    persistent_workers: bool = field(default=False)
    # whether the "random subsample" part of the data yaml ratio (the whole source when ratio<1 plus the
    #   fractional remainder when ratio>1) is re-drawn every epoch.
    #   false (default): random.sample in SubsampledDataset.__init__ draws once and is locked forever, so that
    #     source sees the same videos every epoch and set_epoch only reshuffles them -- 80k raw clips at ratio=0.024
    #     means only ever seeing 2010 of them.
    #   true: a new draw every epoch (subset size, epoch length and sampling share unchanged), so distinct coverage
    #     grows linearly with epochs. seed = f(seed, epoch, part_id) with no rank term, so every rank draws the same
    #     subset (required for sharing a clip inside an SP group). Needs persistent_workers=false (validator
    #     fail-fast): workers must re-fork each epoch to pick up the new indices.
    resample_ratio_each_epoch: bool = field(default=False)
    instance_data_root: list = field(default_factory=list)
    instance_video_root: list = field(default_factory=list)
    dataset_sampling_ratios: list = field(default_factory=list)
    dataloader_num_workers: int = field(default=0)
    prefetch_factor: int = field(default=2)
    force_rebuild: bool = field(default=False)
    stride: int = field(default=1)
    resolution: int = field(default=640)
    single_res: bool = field(default=False)
    single_res: bool = field(default=False)
    single_height: int = field(default=384)
    single_width: int = field(default=640)
    single_length: bool = field(default=False)
    single_num_frame: int = field(default=81)
    multi_res: bool = field(default=False)
    caption_dropout_p: float = field(default=0.00)
    id_token: str = field(default="")
    # CFG negative prompt (actually used).
    negative_prompt: str = field(
        default="oversaturated, garish colors, color shift, hue shift, color drift, inconsistent colors, color cast, color banding, flickering, jittery motion, abrupt transitions, sudden scene changes, temporal inconsistency, static, still picture, blurred details, subtitles, style, works, paintings, images, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, three legs, many people in the background, walking backwards, messy background"
    )
    # Stage 1 dataset toggle
    use_stage1_dataset: bool = field(default=False)
    # Online multi-dataset loading
    use_multi_dataset: bool = field(default=False)
    data_yaml_path: Optional[str] = field(default=None)
    num_frames: int = field(default=105)
    target_fps: int = field(default=24)
    # Stage 3 dataset toggle
    use_stage3_dataset: bool = field(default=False)
    gan_data_root: Optional[list] = field(default_factory=list)
    ode_data_root: Optional[list] = field(default_factory=list)
    text_data_root: Optional[list] = field(default_factory=list)
    # Fallback GEO data switch; training_config.use_geometric_state takes priority.
    use_geometric_state: bool = field(default=False)
    # full-rollout + interleave caption data mode (table 8):
    # per-section caption T5 encoding + segment_frame_ranges + prefix latent + teacher y.
    use_full_rollout_interleave: bool = field(default=False)


@dataclass
class EvokeTeacherTeacherConfig:
    """EvokeTeacher sparse teacher (real/fake score backbone) config.
    weights = merged full safetensors directory (glob load, no config.json);
    boundary follows the EvokeTeacher inference semantics switch_dit_boundary (t >= boundary*1000 -> high-noise expert;
    note the training-side yaml timestep_boundary 0.358 is a timestep-index fraction, do not mix the two up)."""
    high_dir: Optional[str] = field(default=None)   # high-noise expert merged directory
    low_dir: Optional[str] = field(default=None)    # low-noise expert merged directory
    boundary: float = field(default=0.9)
    # the two keys below are sparse_attn.{chunk_size,num_select_frames} from the model yaml of the teacher's own
    #   training; train_evoke.py feeds them into EvokeTeacherScoreWrapper's model_cfg_overrides, which overrides
    #   loader.py's EVOKE_TEACHER_NOCAM_SPARSE defaults (cs8/select4).
    #   Neither param changes any param shape (only the chunk grid and the topk count), so a strict=False load
    #     reports missing=0 and a mismatch is silent. The default matches the cs9/select1 weights in use; switching
    #     back to a cs8-family teacher means setting 8/4 explicitly in the yaml.
    chunk_size: int = field(default=9)              # teacher sparse chunk frame count (cs9 weights = 9)
    num_select_frames: int = field(default=1)       # far-history frames importance-selected per chunk (select=1 weights = 1)
    # load a single expert only, saving 28G of GPU (smoke): None=both experts / "high" / "low".
    # all t route to that expert, so the scoring regime is imprecise for the other half of t - pipeline check only, draw no quality conclusions.
    single_expert: Optional[str] = field(default=None)
    # override the loader model construction args (e.g. the lazy_qkv/chunk_batch_size memory switches)
    model_cfg_overrides: Optional[dict] = field(default=None)
    # per-expert residency swap for the two 2x14B experts: scoring keeps only the
    #   routed expert on GPU and leaves the other on CPU. With both resident a rank needs +28GB and hits the H200 wall.
    #   Only has an effect with dual experts (single_expert=null); a single expert has no second copy to swap.
    #   Previously spelled `dual_teacher.offload`, which was read regardless of `dual_teacher.enabled` -- the switch
    #   never belonged to the (now retired) Evoke-Base camera teacher, it has always driven the EvokeTeacher experts.
    offload: bool = field(default=True)
    # scoring noise cap for the front big window.
    #   null = full band, which is what enables dual-expert routing (t < boundary*1000 -> low expert, t >= -> high);
    #   a concrete value caps t below it (850 in the single-low era = low noise only, never touching stage0).
    #   Previously spelled `dual_teacher.evoke_teacher_score_timestep_max` (default 850 there); the default is None here
    # because every recipe in this repo runs the full band -- for why the default moved.
    score_timestep_max: Optional[int] = field(default=None)


@dataclass
class DualTeacherConfig:
    """decoupled dual real-score teacher (disjoint scope, two independent DMD losses summed):
    - EvokeTeacher-low (nocam, long-range quality): scores the **front N-K sections as one big window**, low-noise cap (evoke_teacher_score_timestep_max); critic=EvokeTeacher backbone.
    - Evoke-Base (pose, camera force): scores the **last 1 warp-ON section** (W=1), full band; critic=**a critic-LoRA dedicated to the Evoke backbone**.
    total generator gradient = grad_lw + lambda_hb * grad_hb (NOTE: non-convex: disjoint chunks, no scatter).
    only active when real_score_arch=='evoke_teacher' and enabled; when off the whole chain is bit-identical (single teacher).
    (v2.1) + """
    enabled: bool = field(default=False)
    evoke_model_path: Optional[str] = field(default=None)   # Evoke-Base pose teacher (camera force) weight directory
    evoke_subfolder: Optional[str] = field(default=None)    # None -> "transformer"
    offload: bool = field(default=True)                       # two-phase alternating offload (EvokeTeacher scoring -> swap in Evoke scoring)
    # -- v2.1: decoupled composition --
    lambda_hb: float = field(default=0.5)                     # relative weight of grad_hb (camera) (NOTE: non-convex, independent of grad_lw)
    # -- v2.1: critic-LoRA dedicated to the Evoke backbone (adapters-off=teacher / adapters-on=critic; camera force self-consistent) --
    evoke_critic_lora_rank: int = field(default=128)
    evoke_critic_lora_alpha: float = field(default=128.0)
    evoke_critic_lora_dropout: float = field(default=0.0)
    evoke_critic_learning_rate: Optional[float] = field(default=None)   # None -> reuse training_config.critic_learning_rate
    # -- per-teacher scoring noise cap --
    # [DEPRECATED - ignored by code] moved to evoke_teacher.score_timestep_max. It never belonged here: the cap
    #   applies to the EvokeTeacher front big window and was read under plain `if _sf_front_window`, regardless of
    #   dual_teacher.enabled, so it was a live knob parked on the retired camera-teacher block. The field is kept as
    #   a no-op only so legacy yamls that still set it survive the OmegaConf struct merge; do not add it to new
    #   configs and do not read it. Its default stayed 850 while the new key defaults to None (full band, what every
    #   recipe here runs), so a legacy yaml relying on this default must set evoke_teacher.score_timestep_max: 850.
    evoke_teacher_score_timestep_max: Optional[int] = field(default=850)
    evoke_score_timestep_max: Optional[int] = field(default=None)      # tail section full band (None -> num_train_timestep, includes the stage0 camera)
    evoke_critic_score_timestep_max: Optional[int] = field(default=None)  # Evoke critic training cap (None -> same as evoke_score_timestep_max)
    # -- deprecated (v1/v2 convex mix, dropped in v2.1; fields kept so old configs do not error, no longer consumed) --
    w_lw: float = field(default=0.5)
    w_hb: float = field(default=0.5)


@dataclass
class ModelConfig:
    pretrained_model_name_or_path: Optional[str] = field(default=None)
    transformer_model_name_or_path: Optional[str] = field(default=None)
    siglip_model_name_or_path: Optional[str] = field(default=None)
    lora_paths: Optional[list[str]] = field(default_factory=list)
    subfolder: Optional[str] = field(default=None)
    revision: Optional[str] = field(default=None)
    variant: Optional[str] = field(default=None)
    load_checkpoints_custom: bool = field(default=False)
    load_model_path: Optional[str] = field(default=None)
    load_dcp: bool = field(default=False)
    load_dcp_path: Optional[str] = field(default=None)
    # VAE options
    upcast_vae: bool = field(default=True)
    enable_slicing: bool = field(default=False)
    enable_tiling: bool = field(default=False)
    # LoRA hyperparameters
    lora_rank: int = field(default=128)
    lora_alpha: float = field(default=128.0)
    lora_dropout: float = field(default=0.0)
    lora_layers: Optional[str] = field(default=None)
    lora_target_modules: list = field(default_factory=list)
    lora_exclude_modules: list = field(default_factory=list)
    train_norm_layers: bool = field(default=False)
    bnb_quantization_config_path: Optional[str] = field(default=None)
    # Stage 3 critic / reward paths
    critic_lora_name_or_path: Optional[str] = field(default=None)
    critic_subfolder: Optional[str] = field(default=None)
    critic_lora_rank: int = field(default=128)
    critic_lora_alpha: float = field(default=128.0)
    critic_lora_dropout: float = field(default=0.0)
    real_score_model_name_or_path: Optional[str] = field(default=None)
    # real/fake score backbone arch: "evoke" (default, current path) | "evoke_teacher" (the sparse teacher).
    # with evoke_teacher, real_score_model_name_or_path is unused;
    # the weight paths come from the evoke_teacher sub-config.
    real_score_arch: str = field(default="evoke")
    evoke_teacher: "EvokeTeacherTeacherConfig" = field(default_factory=lambda: EvokeTeacherTeacherConfig())
    # dual real-score teacher (evoke_teacher long-range + Evoke-Base camera force); disabled=single teacher, bit-identical.
    dual_teacher: "DualTeacherConfig" = field(default_factory=lambda: DualTeacherConfig())
    reward_model_name_or_path: Optional[str] = field(default=None)
    # Lingbot-style AdaLN camera control; no extra params when disabled.
    camera_control: "CameraControlConfig" = field(default_factory=lambda: CameraControlConfig())
    # geometric state adapter; no extra params when disabled.
    geometric_state: "WarpAsHistoryConfig" = field(default_factory=lambda: WarpAsHistoryConfig())


@dataclass
class GeoRetrieveConfig:
    """FrameBank retrieve config shared by training and inference."""
    score: str = field(default="v1")              # scoring variant: "v1" / "v2" / "v3"
    nearby_k: int = field(default=0)               # number of temporally nearest frames to include
    select_k: int = field(default=5)               # number of metric-selected frames
    v3_depth: float = field(default=5.0)            # V3: canonical scene depth in metres
    v3_fov_deg: float = field(default=60.0)         # V3: camera field of view in degrees
    bank_max: int = field(default=0)                # max bank size; 0 = unlimited
    init_k: int = field(default=10)                 # max frames seeded into bank at v2v init


@dataclass
class ShortTierNoiseConfig:
    """Adds configurable sigma noise to prefix and prev_short tokens to balance attention trust scores."""
    enabled: bool = field(default=False)                                          # master switch
    sigma_min: float = field(default=0.2)                                          # lower sigma bound (shared default)
    sigma_max: float = field(default=0.6)                                          # upper sigma bound (shared default)
    target_tiers: List[str] = field(default_factory=lambda: ["prefix", "prev_short"])  # empty list equals disabled
    apply_at_inference: bool = field(default=True)                                 # apply noise at inference for train/infer consistency
    sigma_lock_per_rollout: bool = field(default=False)                            # lock sigma per rollout for temporal continuity
    # Per-tier sigma_max overrides (None → fall back to the shared sigma_max above). Lets prefix /
    # prev_short / mid+long carry different noise strengths. Set a tier's value to 0.0 for a clean
    # (un-noised) tier — e.g. prefix_sigma_max=0.0 keeps the prefix clean while it stays in target_tiers.
    prefix_sigma_max: Optional[float] = field(default=None)                        # None → shared sigma_max
    prev_short_sigma_max: Optional[float] = field(default=None)                    # None → shared sigma_max
    mid_long_sigma_max: Optional[float] = field(default=None)                      # None → shared sigma_max (applies to both mid & long)


@dataclass
class SamplingConfig:
    """Discrete sampling distribution: pick a value from `choices` with weights `probs`.
    `probs` accepts an explicit list[float] OR the string 'uniform' (equal weights).
    Field is `choices` (NOT `values`) to avoid colliding with OmegaConf DictConfig.values()."""
    choices: List[int] = field(default_factory=list)
    probs: Any = field(default="uniform")


@dataclass
class Da3BackendConfig:
    """DepthAnything3 estimator params; read when cloud_warp.backend == 'da3'.

    DA3 is pose-conditioned: the GT extrinsics and intrinsics are network inputs, so its depth is
    solved inside the GT camera model and only a single Umeyama scale is left to recover."""
    ckpt_path: str = field(default="models/DA3")          # weights dir (from_pretrained)
    process_res: int = field(default=644)                 # internal long-side res (controls cloud density)
    src: Optional[str] = field(default=None)              # source tree; None -> EVOKE_DA3_SRC / vendored default


@dataclass
class VigeoBackendConfig:
    """ViGeo estimator params; read when cloud_warp.backend == 'vigeo'.

    ViGeo takes no pose or intrinsics input: it predicts its own poses and focal length and its depth is
    up-to-scale, so the depth is rescaled by the Umeyama scale between ViGeo's predicted trajectory and
    the GT one, then unprojected with the GT K -- the same output contract as DA3. See
    evoke/modules/geometric_state/vigeo_cloud.py. Defaults are the recipe validated in
    (streaming with an anchored scale)."""
    weights: str = field(default="models/ViGeo1.1")       # dir holding vigeo.pt
    process_res: int = field(default=644)                 # per-backend, so ViGeo's resolution can be raised alone
    src: Optional[str] = field(default=None)              # source tree; None -> EVOKE_VIGEO_SRC / the vendored evoke/third_party/vigeo
    # 'chunk' holds the kv-cache across ingest windows so one stream shares one coordinate frame and one
    # scale (ViGeo's design intent); 'offline' treats each window independently and drifts at the tail.
    mode: str = field(default="chunk")
    chunk_size: int = field(default=16)                   # ViGeo internal chunk length; mode=chunk/online only
    # 'anchor' locks the median scale of the first anchor_windows windows: later windows are fed
    # self-generated frames whose apparent motion already contains generation drift, and re-solving there
    # writes that drift into depth. 'per_window' solves independently each window.
    scale_mode: str = field(default="anchor")
    anchor_windows: int = field(default=4)
    cache_keep_frames: int = field(default=6)             # streaming: context frames kept per global block
    # 0 = derive from cache_keep_frames. Do NOT hand-tune; see ViGeoDepthEstimator._resolve_budget.
    total_budget: int = field(default=0)
    intr_source: str = field(default="gt")                # 'gt' = GT K rescaled to output res (as DA3) | 'vigeo' = self-estimated focal (ablation)
    conf_transform: str = field(default="exp")            # ViGeo conf is raw logits (can be negative); 'exp' maps it positive
    num_tokens: Optional[int] = field(default=None)       # None -> resolution from process_res; int -> ViGeo token budget (native 1369)


@dataclass
class CloudWarpConfig:
    """Known-trajectory point-cloud warp backend (replaces Pi3X retrieve/pose-estimation).
    Consumed by the inference render and by training; """
    enabled: bool = field(default=False)                  # master switch (cloud backend on/off)
    # Depth-estimation backend behind the otherwise unchanged FrameBank / render pipeline, and the single switch for
    # it: 'da3' (default) | 'vigeo'. Everything downstream -- render_mode, splat/density/recall params,
    # conf_percentile -- is shared, so an A/B isolates the geometry model alone; backend-specific paths live in the
    # `da3` / `vigeo` sub-blocks. Orthogonal to render_mode, which selects the renderer, not least because
    # render_mode_mix_prob_zbuf re-draws render_mode per training sample.
    backend: str = field(default="da3")
    da3: "Da3BackendConfig" = field(default_factory=lambda: Da3BackendConfig())
    vigeo: "VigeoBackendConfig" = field(default_factory=lambda: VigeoBackendConfig())
    # [DEPRECATED — ignored by code] Unprojection always uses the GT c2w; there is no pose-estimation
    # branch to switch to. Kept as a no-op; do NOT add it to new configs.
    use_gt_pose: bool = field(default=True)
    # [DEPRECATED — ignored by code] The depth scale is always solved against the GT trajectory
    # (_align_pred_robust); no other mode exists. Kept as a no-op; do NOT add it to new configs.
    scale: str = field(default="gt_metric")
    splat_radius: int = field(default=2)                  # render splat radius (px)
    update_frames_per_chunk: int = field(default=12)      # frames per chunk ingested to DA3 (=ingest_n, >=3); 12 keeps DA3 fwd <1s
    train_batch_windows: bool = field(default=True)       # (legacy, dump-all path only; recall path ignores)
    lag_sampling: SamplingConfig = field(default_factory=SamplingConfig)            # cloud lag (chunks): infer fixes lag=1; train samples for robustness
    history_chunks_sampling: SamplingConfig = field(default_factory=SamplingConfig)  # recall pool depth (chunks): pool = chunk[k-1-lag-history+1 .. k-1-lag]
    # -- multi-source priority-fused warp (default; replaces point-cloud recall; train/infer/val share _render_multisrc) --
    # render_mode: 'multisrc'=multi-source priority fusion (default) | 'backward'=grid_sample backward warp (single main source + recall hole fill + fill_iters dilation)
    #   | 'backward_zbuf'=per-pixel multi-source z-buffer fusion (no dilation / holes left invalid), i.e. multisrc_zbuf; steadier at inference / no chunk-boundary edge jumps
    #   | 'recall'=legacy point cloud (fallback)
    render_mode: str = field(default="multisrc")
    bw_fill_iters: int = field(default=12)   # backward warp gap-fill iterations (closes small interior holes); used by render_mode=backward (backward_zbuf has no dilation and ignores it)
    # warp_warm_encode: VAE-encode the warp with the first frame warm-padded (prepend vae_t copies -> encode ->
    # drop the warm I-frame latent) so warp[0] is a CONTINUATION-distributed latent, not the VAE first-frame
    # (I-frame) distribution. Lets the DiT learn a continuation pred[0] -> one-shot/persistent decode w/o flicker.
    # Off (default) = per-chunk vae.encode (I-frame first latent). Used by training (online_materialize) + val/infer.
    warp_warm_encode: bool = field(default=False)
    # -- backward_zbuf hybrid despeckle (off by default; validated in) --
    #   a zbuf single-pixel splat with no hole filling leaves ~17% salt-and-pepper (scattered invalid pixels), which the VAE 8x average turns into gray mush = a poisoned training label.
    #   when on, per-frame morphological open->close: clears scattered boundary valid pixels (-> holes, left to the model) + fills fully surrounded interior points (splat misses whose content is known), real large holes untouched.
    #   train/infer/val must use the same renderer with the same params -> all three read them from this config (off by default = byte-identical with today).
    zbuf_despeckle: bool = field(default=False)             # master switch: per-frame despeckle post-processing on the backward_zbuf path
    zbuf_despeckle_ksize: int = field(default=3)            # morphological kernel size (open->close); 3x3
    zbuf_despeckle_fill_iters: int = field(default=4)       # interior-point neighborhood mean color-fill iterations
    # render_mode_mix: train-only per-sample render_mode mixing (0=off, use the fixed render_mode).
    #   >0 = each sample takes backward_zbuf (despeckled) with this probability, else backward -> a mix such as 0.8 backward : 0.2 zbuf,
    #   which fixes the zbuf salt-and-pepper poison label while keeping backward's generation ability. val/inference still use the fixed render_mode (no mixing).
    render_mode_mix_prob_zbuf: float = field(default=0.0)
    nsrc: int = field(default=8)                          # source frames fused per target frame (nearest-frame main source used fully + the others only fill holes)
    nearby_window: int = field(default=16)                # nearest candidate window (the main source is picked from it)
    multisrc_splat: int = field(default=1)                # forward splat radius per source
    dens_thresh: float = field(default=0.45)              # density gate: sparse points with local coverage < this -> hole (left to the DiT)
    dens_win: int = field(default=7)                      # density gate window
    recall_min_cov: float = field(default=0.5)            # recall trigger: absolute covis threshold for old frames (+ pose proximity gate)
    recall_margin: float = field(default=0.15)            # and must be > the best nearby + margin
    # -- FrameBank recall-style warp --
    # NOTE: voxel_size/max_points removed -- recall keeps no persistent point cloud, render only draws the ~recall_k recalled frames, so the points are naturally bounded
    # (must keep recall_k x process_res pixels x (1-conf filtering) < 2^24, see design).
    recall_k: int = field(default=12)                     # frames recalled per chunk to render
    n_nearby: int = field(default=4)                      # fixed most-recent frames (continuity; adaptively dropped if covis≈0)
    n_tframe: int = field(default=6)                      # target frames sampled for covis coverage scoring
    recall_grid_div: int = field(default=8)               # covis coverage grid downsample (H/div × W/div)
    recall_mask_pts: int = field(default=8000)            # point subsample for covis scoring
    conf_percentile: float = field(default=30.0)          # conf filter percentile at ingest (percentile => backend-agnostic)
    recall_k_sampling: SamplingConfig = field(default_factory=SamplingConfig)        # train robustness: recall_k sampling
    n_nearby_sampling: SamplingConfig = field(default_factory=SamplingConfig)        # train robustness: n_nearby sampling


def resolve_cloud_warp(cw) -> dict:
    """Flatten a CloudWarpConfig into the wire dict the warp pipeline consumes.

    The yaml is nested per backend but `_geo_vsnoise_cfg` and the inference CLI are flat, so the nesting
    is absorbed here, once, on the producer side; consumers keep reading `da3_process_res` /
    `da3_weights` and only need to branch on `depth_backend`. Every producer -- both validation blocks
    in train_evoke, online_materialize and sf_warp_rollout -- goes through this function so they cannot
    drift apart.

    `da3_process_res` / `da3_weights` / `da3_src` keep their historical names but carry the values for
    the **active** backend, so the downstream estimator construction never has to know which sub-block
    they came from. The names are deliberately unchanged: they are what makes the resolved dict
    comparable key-for-key against the pre-refactor values.
    """
    if cw is None:
        return {}
    if isinstance(cw, dict):
        # Attribute lookups on a plain mapping would silently miss every key and return a full set of
        # defaults, i.e. a plausible-looking but wrong warp recipe. OmegaConf nodes support getattr and
        # are not dict subclasses, so they pass.
        raise TypeError("resolve_cloud_warp expects a CloudWarpConfig / DictConfig node, got a plain "
                        "dict; attribute access on a mapping would silently yield all defaults.")
    backend = str(getattr(cw, "backend", "da3") or "da3").lower()
    da3 = getattr(cw, "da3", None)
    vg = getattr(cw, "vigeo", None)
    active = vg if backend == "vigeo" else da3
    # Asymmetric on purpose: the da3 block kept `ckpt_path` from the pre-nesting flat key.
    weights_attr = "weights" if backend == "vigeo" else "ckpt_path"
    out = {
        "depth_backend": backend,
        "da3_process_res": int(getattr(active, "process_res", 644)),
        "da3_weights": str(getattr(active, weights_attr, "models/ViGeo1.1" if backend == "vigeo" else "models/DA3")),
        "da3_src": getattr(active, "src", None) or None,
        "render_mode": str(getattr(cw, "render_mode", "multisrc")),
        "bw_fill_iters": int(getattr(cw, "bw_fill_iters", 12)),
        "zbuf_despeckle": bool(getattr(cw, "zbuf_despeckle", False)),
        "zbuf_despeckle_ksize": int(getattr(cw, "zbuf_despeckle_ksize", 3)),
        "zbuf_despeckle_fill_iters": int(getattr(cw, "zbuf_despeckle_fill_iters", 4)),
        "update_frames_per_chunk": int(getattr(cw, "update_frames_per_chunk", 12)),
        "recall_k": int(getattr(cw, "recall_k", 12)),
        "n_nearby": int(getattr(cw, "n_nearby", 4)),
        "n_tframe": int(getattr(cw, "n_tframe", 6)),
        "recall_grid_div": int(getattr(cw, "recall_grid_div", 8)),
        "recall_mask_pts": int(getattr(cw, "recall_mask_pts", 8000)),
        "conf_percentile": float(getattr(cw, "conf_percentile", 30.0)),
        "cloud_splat_radius": int(getattr(cw, "splat_radius", 2)),
        "nsrc": int(getattr(cw, "nsrc", 8)),
        "nearby_window": int(getattr(cw, "nearby_window", 16)),
        "multisrc_splat": int(getattr(cw, "multisrc_splat", 1)),
        "dens_thresh": float(getattr(cw, "dens_thresh", 0.45)),
        "dens_win": int(getattr(cw, "dens_win", 7)),
        "recall_min_cov": float(getattr(cw, "recall_min_cov", 0.5)),
        "recall_margin": float(getattr(cw, "recall_margin", 0.15)),
        "geo_warp_warm_encode": bool(getattr(cw, "warp_warm_encode", False)),
    }
    if backend == "vigeo":
        # ViGeo-only knobs, consumed by depth_backend.build_estimator; ignored by the da3 path.
        out.update({f"vigeo_{k}": getattr(vg, k, None) for k in (
            "mode", "chunk_size", "scale_mode", "anchor_windows", "cache_keep_frames",
            "total_budget", "intr_source", "conf_transform", "num_tokens")})
    return out


def vigeo_opts_from_cfg(cfg: dict) -> dict:
    """Extract the `vigeo_*` knobs from a resolved/flat cfg into build_estimator's `vigeo_opts`."""
    keys = ("mode", "chunk_size", "scale_mode", "anchor_windows", "cache_keep_frames",
            "total_budget", "intr_source", "conf_transform", "num_tokens",
            # baseline-free scale modes; None for callers that never set them, which build_estimator drops
            "scale_value", "depth_median_target")
    return {k: cfg.get(f"vigeo_{k}") for k in keys}


@dataclass
class WarpTokenDropConfig:
    """Train-only stochastic drop on the WARP visibility mask (regularization / mask diversity).

    One categorical draw per sample over [none, full, per_frame, per_patch] (mode_probs, normalized).
    Drops by zeroing the warp visibility mask, so those tokens fall under visible_token_threshold and
    the loss weighting treats them as invisible (coupled by design).
    """
    enabled: bool = field(default=False)
    mode_probs: List[float] = field(default_factory=lambda: [0.5, 0.2, 0.2, 0.1])  # none/full/per_frame/per_patch
    frame_drop_ratio: float = field(default=0.5)   # per_frame: per-frame Bernoulli drop prob
    patch_drop_ratio: float = field(default=0.3)   # per_patch: per visible 2x2-latent-patch Bernoulli drop prob


@dataclass
class WarpPoseJitterConfig:
    """Train-only rigid pose jitter on the warp RENDER target poses (warp = rough reference, not 1:1 copy).

    With prob (per-sample / per-chunk, B=1), a single small camera-frame rigid offset DeltaT is sampled and
    right-multiplied onto the TARGET pose window fed to the warp renderer ONLY -> the rendered warp is slightly
    spatially misaligned from the GT target, so the model learns to treat warp as a coarse reference rather than
    a pixel-exact copy target. finding: ROTATION (yaw+pitch) moves the warp
    effectively; pure TRANSLATION is ineffective in this backward/multisrc warp setup -> jitter is rotation-based,
    translation optional (default off). CRITICAL: only the warp-render target pose is jittered; the plucker
    (camera-control) and the target latent / loss stay at the TRUE GT pose (jitter touches a local clone only).
    Default disabled -> warp render takes the exact original GT-pose path (byte-identical)."""
    enabled: bool = field(default=False)
    prob: float = field(default=0.0)                                       # per-sample(=per-chunk, B=1) trigger probability
    yaw_deg_range: List[float] = field(default_factory=lambda: [0.5, 2.0])    # |yaw|   ~ U(range) when triggered (sign random)
    pitch_deg_range: List[float] = field(default_factory=lambda: [0.5, 2.0])  # |pitch| ditto
    roll_deg_range: List[float] = field(default_factory=lambda: [0.0, 0.5])   # small roll
    trans_frac_range: List[float] = field(default_factory=lambda: [0.0, 0.0]) # fraction of per-frame motion; default OFF (EXP: translation ineffective)


@dataclass
class WarpSaturationCorruptConfig:
    """Train-only GEO-native latent saturation corruption on the history tiers (cures inference over-saturation drift).

    Latent-space saturation (= the same kernel as evoke add_saturation_to_history_latents), TWO-LEVEL gate:
      (1) per-STEP w.p. step_prob saturation is enabled (else the WHOLE history stays clean -> model still sees
          clean histories often enough to NOT over-correct; pure per-frame would make all-clean ~0 prob);
      (2) given enabled, each FRAME of an enabled tier w.p. frame_prob picks a sat_factor in
          [ratio_min,1)∪(1,ratio_max] (50/50 over/under) and applies `(x - x.mean(dim=channel))*f + x.mean`.
    Applied ONLY to the DEGRADED tiers (student rollout + critic/fake_score); the Option B teacher (pred_real,
    recycle_teacher_clean) keeps the un-saturated copy -> teacher target = correct saturation, student must learn
    to NOT amplify a saturated history. prefix never touched. target_tiers ⊆ {warp, prev_short, mid, long}.
    Requires recycle_teacher_clean=True. Default disabled -> byte-identical.
    """
    enabled: bool = field(default=False)
    ratio_min: float = field(default=0.5)          # < 1.0 branch (desaturate); experiments confirm 0.5 still faithfully tracks pixel saturation
    # > 1.0 branch (over-saturate). NOTE: cap 1.4: measured that latent saturation only equals
    # true pixel saturation for f in [0.5,~1.5]; f>=1.8 breaks down (VAE nonlinearity -> brightening + blocky artifacts, S drops instead, no longer simulating over-saturation).
    ratio_max: float = field(default=1.4)
    step_prob: float = field(default=0.6)           # PER-STEP: prob saturation is enabled (else whole history clean)
    frame_prob: float = field(default=0.6)          # PER-FRAME (given enabled): prob each frame gets saturated
    target_tiers: List[str] = field(default_factory=lambda: ["warp", "prev_short", "mid", "long"])


@dataclass
class WarpAsHistoryConfig:
    """pose-addressed geometric state training/inference config.

    Injects Pi3X-rendered warp_latents into the short tier with shared RoPE indices,
    trained via rank-1 LoRA on to_q/to_k/to_v for attention re-routing.
    """
    enabled: bool = field(default=False)
    # GEO LoRA params
    lora_rank: int = field(default=1)
    lora_alpha: float = field(default=1.0)
    lora_dropout: float = field(default=0.0)
    lora_target_modules: str = field(default="to_q,to_k,to_v")
    # Pi3X renderer checkpoint path
    pi3x_ckpt_path: Optional[str] = field(default=None)
    # [DEPRECATED — ignored by code] The only reader is WarpAsHistoryPipeline, which is never
    # instantiated anywhere in the repo. The live visibility filter is visible_token_threshold below,
    # which the active paths read directly. Kept as a no-op; do NOT add it to new configs.
    visible_token_drop: bool = field(default=True)
    # Visibility filter for warp tokens.
    visible_token_threshold: float = field(default=0.1)
    # FrameBank retrieve config shared across training and inference.
    retrieve: "GeoRetrieveConfig" = field(default_factory=lambda: GeoRetrieveConfig())
    # Short-tier noise config; disabled by default.
    short_tier_noise: "ShortTierNoiseConfig" = field(default_factory=lambda: ShortTierNoiseConfig())
    # DA3 known-trajectory point-cloud warp backend; disabled by default (parsing-only until P1/P2).
    cloud_warp: "CloudWarpConfig" = field(default_factory=lambda: CloudWarpConfig())
    # Train-only warp-token drop (modifies the warp visibility mask); disabled by default.
    warp_token_drop: "WarpTokenDropConfig" = field(default_factory=lambda: WarpTokenDropConfig())
    # Train-only GEO-native latent saturation corruption on history tiers (student/critic only); disabled by default.
    warp_saturation_corrupt: "WarpSaturationCorruptConfig" = field(default_factory=lambda: WarpSaturationCorruptConfig())
    # Train-only rigid pose jitter on the warp RENDER target poses (warp = rough reference); disabled by default.
    # Only the warp-render pose is jittered; plucker + target latent/loss stay at TRUE GT pose.
    warp_pose_jitter: "WarpPoseJitterConfig" = field(default_factory=lambda: WarpPoseJitterConfig())
    # Spatial visibility-aware noise for warp_latents; uses per-pixel sigma map when enabled.
    visibility_aware_noise: bool = field(default=False)
    warp_noise_sigma_invisible: float = field(default=0.8)
    # Per-frame sigma range for visible warp pixels (also used when visibility_aware_noise=False).
    warp_noise_sigma_min: float = field(default=0.111)
    warp_noise_sigma_max: float = field(default=0.135)
    # Error-bank injection into warp (err-then-noise). Samples a low-noise y_error from the recycle
    # bank and adds it to the CLEAN warp latent BEFORE visibility-aware noising. Default off ->
    # bit-identical to legacy path. See
    warp_error_inject_enabled: bool = field(default=False)
    warp_error_prob: float = field(default=0.0)
    # Stage2 err-bank per-tier injection (downstream unified path in train_evoke, before short_tier_noise).
    # Which history tiers receive a banked real y_error: subset of {"warp", "prev_short"}. Empty = off.
    # prefix/mid/long are NEVER injected. error_inject_prob = per-step injection probability (shared).
    error_inject_tiers: List[str] = field(default_factory=list)
    error_inject_prob: float = field(default=0.0)
    # True: warp RoPE idx overlaps with target idx. False: warp idx shifted to history slot.
    rope_alignment: bool = field(default=True)
    # "zero": prefix.idx=0 (baseline). "adjacent": prefix.idx=noise.idx[0]-1 (sequential).
    prefix_idx_mode: str = field(default="zero")
    # Optional 2-layer residual MLP applied to warp tokens after patch_short; zero-init output layer.
    geo_warp_residual_mlp_enabled: bool = field(default=False)
    geo_warp_residual_mlp_hidden_mult: float = field(default=2.0)
    # Warp RoPE index mode (Plan 16). "overlap_noise": warp takes base noise slot, noise shifts +W.
    # "before_prev_short"/"before_prev_mid": warp takes prev_short/mid slot, requires rope_alignment=False + prefix_idx_mode=zero.
    warp_rope_mode: str = field(default="overlap_noise")
    # Inject sigma_invisible noise into zeros-padding history (long/mid) frames; mirrors inference (v4).
    geo_invisible_history_noise: bool = field(default=False)
    # Overwrite warp[0] with the clean previous-chunk last frame instead of the Pi3X render (v20).
    warp_keep_clean_anchor: bool = field(default=False)
    # [DEPRECATED — ignored by code] i2v used to render NO warp ([prefix]-only). i2v now always goes through the
    # DA3 single-source warp path (build_single_source_warp), producing the same [prefix|warp(9)|prev_short] tier
    # as v2v. The field is kept (no-op) only so legacy yamls that still set it don't fail OmegaConf struct merge;
    # do NOT add it to new configs.
    geo_i2v_zero_warp: bool = field(default=False)
    # Lag (in chunks) before a decoded frame is fused into the warp point cloud; 0 = synchronous.
    warp_lag_chunks: int = field(default=0)
    # Additive Plucker injection: when True, project the per-frame Plucker field via a 2-layer MLP
    # (process_cam_plucker_to_tokens) and ADD the resulting tokens to BOTH the warp tokens and the noise
    # tokens (no AdaLN). Single switch; independent of camera_control. Zero-init -> warm-start safe.
    geo_warp_plucker_enabled: bool = field(default=False)
    # DMD per-model override: plucker for the GENERATOR (student) build only. None = use geo_warp_plucker_enabled
    # (legacy, both equal). Set False when student=Evoke-Distilled (no plucker) but teacher=warp ckpt (has plucker):
    # keep geo_warp_plucker_enabled=true so teacher builds+loads plucker and the cam-pose data path stays on, and set
    # generator_geo_warp_plucker_enabled=false so NO plucker submodule is built on the student (it would be a dead
    # zero-init layer). Student forward ignores any cam_plucker_emb (gated by self.geo_warp_plucker_enabled at fwd).
    generator_geo_warp_plucker_enabled: Optional[bool] = field(default=None)
    # warp_rope_noise_center_align (fixed_mem only): at coarse pyramid stages, center the NOISE spatial RoPE
    # into the full-res warp coordinate frame (coord -> coord*scale + (scale-1)/2, scale = full/stage), i.e. each
    # coarse cell at its centroid -- the same coordinate convention Pyramid-Flow uses. The warp/history rope stays
    # native -> uniform history block -> restrict_self_attn / KV-cache stay valid. Off (default) = native coords
    # (= legacy baseline). On = full centering directly (no interpolation knob).
    warp_rope_noise_center_align: bool = field(default=False)
    # warp_stage0_only: inject warp into the short tier ONLY at the coarsest pyramid stage (stage0), matching
    # the Geo reference + stage0-only training (loss only on stage0). At finer pyramid stages (i_s>0) the warp
    # segment is stripped -> [prefix|prev_short] only. Off (default) = warp injected at all stages (stage1-style).
    # Consumed at inference/val (stage2_sample) via _geo_vsnoise_cfg; val mirrors it for train/infer parity.
    warp_stage0_only: bool = field(default=False)

    def __post_init__(self):
        # Validate prefix_idx_mode enum.
        if self.prefix_idx_mode not in ("zero", "adjacent"):
            raise ValueError(
                f"WarpAsHistoryConfig.prefix_idx_mode must be 'zero' or 'adjacent', "
                f"got '{self.prefix_idx_mode}'."
            )
        # Validate warp_rope_mode enum + its rope_alignment/prefix_idx_mode requirements (Plan 16).
        if self.warp_rope_mode not in ("overlap_noise", "before_prev_short", "before_prev_mid"):
            raise ValueError(
                f"WarpAsHistoryConfig.warp_rope_mode must be 'overlap_noise' / 'before_prev_short' / "
                f"'before_prev_mid', got '{self.warp_rope_mode}'."
            )
        if self.warp_rope_mode in ("before_prev_short", "before_prev_mid"):
            if self.rope_alignment:
                raise ValueError(
                    f"warp_rope_mode='{self.warp_rope_mode}' requires rope_alignment=False "
                    f"(warp uses a different RoPE idx than prev_short/mid/noise)."
                )
            if self.prefix_idx_mode != "zero":
                raise ValueError(
                    f"warp_rope_mode='{self.warp_rope_mode}' requires prefix_idx_mode='zero' "
                    f"(prev_short takes the short-term anchor; prefix stays at idx=0)."
                )
        # Fail-fast sigma_invisible range check.
        _sigma_inv = float(self.warp_noise_sigma_invisible)
        if not (0.0 < _sigma_inv <= 1.0):
            raise ValueError(
                f"WarpAsHistoryConfig.warp_noise_sigma_invisible must be in (0.0, 1.0], "
                f"got {_sigma_inv}. Recommended ablation range: [0.5, 0.95]."
            )
        # Validate visible sigma range.
        _sm = float(self.warp_noise_sigma_min)
        _sM = float(self.warp_noise_sigma_max)
        if not (0.0 <= _sm <= _sM <= 1.0):
            raise ValueError(
                f"warp_noise_sigma_min/max must satisfy 0 <= min <= max <= 1.0, "
                f"got [{_sm}, {_sM}]."
            )


@dataclass
class CameraControlConfig:
    """Lingbot-style camera control via AdaLN affine modulation on noise tokens.

    pc_resolution_strategy: "scale_ks" (closed-form intrinsic scaling) or "resample" (interpolate Plucker field).
    base_height_pix/base_width_pix: defaults to data resolution during training, pipeline height/width at inference.
    """
    enabled: bool = field(default=False)
    cam_rank: int = field(default=128)
    cam_ctrl_layers: Optional[list[int]] = field(default=None)
    cam_ckpt_path: Optional[str] = field(default=None)
    train_only_camera: bool = field(default=False)
    strict_camera_ckpt: bool = field(default=True)
    pc_resolution_strategy: str = field(default="scale_ks")
    base_height_pix: Optional[int] = field(default=None)
    base_width_pix: Optional[int] = field(default=None)


@dataclass
class ValidationConfig:
    validation_steps: int = field(default=100)
    validation_height: int = field(default=480)
    validation_width: int = field(default=832)
    validation_max_num_frames: int = field(default=81)
    validation_prompts: Optional[list[str]] = field(default_factory=lambda: ["A frog jumps on a lotus leaf."])
    validation_images: Optional[list[str]] = field(default_factory=lambda: ["examples/i2v/image.jpg"])
    validation_guidance_scale: float = field(default=9.0)
    validation_latent_window_size: list[int] = field(default_factory=lambda: [9])
    validation_stream_chunk_size: list[int] = field(default_factory=lambda: [3])
    first_step_valid: bool = field(default=True)
    num_validation_videos: int = field(default=1)
    num_inference_steps: int = field(default=30)
    # Dynamic shifting schedule
    use_dynamic_shifting: bool = field(default=False)
    time_shift_type: str = field(
        default="linear",
        metadata={"choices": ["exponential", "linear"]},
    )
    # Stage 1
    use_kv_cache: bool = field(default=False)
    # Stage 2
    stage2_simulated_inference_steps: list[int] = field(default_factory=lambda: [10, 10, 10])

    # V2V validation: ref video paths aligned 1-to-1 with validation_prompts.
    validation_videos: list[str] = field(default_factory=list)
    # Duration in seconds of the ref video used for V2V conditioning.
    validation_video_seconds: float = field(default=3.0)
    # Start offset in seconds into the ref video.
    validation_video_start_seconds: float = field(default=0.0)

    # Camera-control validation: pose npz paths aligned 1-to-1 with validation_prompts.
    validation_pose_paths: list[str] = field(default_factory=list)
    # Source resolution [H, W] for pose npz intrinsic rescaling.
    validation_pose_source_resolution: list[int] = field(default_factory=lambda: [1080, 1920])  # [H, W]
    validation_pose_source_fps: int = field(default=30)
    validation_pose_type: str = field(default="vipe")
    # Clamp each val frame's rotation deviation from frame 0 to <= this many degrees (0 = disabled).
    # Keeps the val camera trajectory high-overlap so warp coverage stays dense (warp-following test).
    validation_pose_max_rotation_deg: float = field(default=0.0)

    # None: mirrors training_config.use_geometric_state. True/False: explicit override.
    use_geometric_state: Optional[bool] = field(default=None)


@dataclass
class TrainingConfig:
    # High-level trainable model selector; overrides is_train_* flags. "lora" trains the main transformer with LoRA.
    trainable_models: list[str] = field(default_factory=list)
    # Promote x0 and the last short-tier frame to raw sink tokens via the main patch_embedding.
    use_raw_sink_frames: bool = field(default=False)
    # Enable GEO forward path in the training loop; requires model_config.geometric_state.enabled.
    use_geometric_state: bool = field(default=False)
    # Environment
    local_rank: int = field(default=-1)
    allow_tf32: bool = field(default=False)
    gradient_checkpointing: bool = field(default=True)
    enable_xformers_memory_efficient_attention: bool = field(default=False)
    enable_npu_flash_attention: bool = field(default=False)
    upcast_before_saving: bool = field(default=False)
    offload: bool = field(default=False)
    mixed_precision: str = field(
        default="bf16",
        metadata={"choices": ["no", "fp16", "bf16"]},
    )
    profile_out_dir: Optional[str] = field(default=None)
    # Training resource
    num_train_epochs: int = field(default=1)
    max_train_steps: Optional[int] = field(default=None)
    train_batch_size: int = field(default=1)
    gradient_accumulation_steps: int = field(default=1)
    checkpointing_steps: int = field(default=500)
    checkpoints_total_limit: Optional[int] = field(default=None)
    resume_from_checkpoint: Optional[str] = field(default=None)
    save_checkpoints_custom: bool = field(default=False)
    # Optimizer
    learning_rate: float = field(default=2e-4)
    scale_lr: bool = field(default=False)
    lr_scheduler: str = field(
        default="constant",
        metadata={
            "choices": [
                "linear",
                "cosine",
                "cosine_with_restarts",
                "polynomial",
                "constant",
                "constant_with_warmup",
            ]
        },
    )
    lr_warmup_steps: int = field(default=500)
    lr_num_cycles: int = field(default=1)
    lr_power: float = field(default=1.0)
    optimizer: str = field(
        default="adamw",
        metadata={
            "choices": ["adam", "adamw", "prodigy"],
        },
    )
    use_8bit_adam: bool = field(default=False)
    adam_beta1: float = field(default=0.9)
    adam_beta2: float = field(default=0.999)
    prodigy_beta3: Optional[float] = field(default=None)
    prodigy_decouple: bool = field(default=True)
    prodigy_use_bias_correction: bool = field(default=True)
    prodigy_safeguard_warmup: bool = field(default=True)
    adam_weight_decay: float = field(default=1e-04)
    adam_epsilon: float = field(default=1e-08)
    max_grad_norm: float = field(default=1.0)
    # warp visibility-differentiated loss weighting (default 1.0/1.0 = legacy behavior). the visible region (warp coverage, the model can copy) gets visible_loss_weight;
    # the invisible region (occlusion / newly exposed view from camera motion, must be generated) gets invisible_loss_weight. amplifying invisible -> forces the model to learn camera-consistent generation.
    # _flow_loss also logs loss_visible / loss_invisible per region to diagnose copy shortcuts.
    visible_loss_weight: float = field(default=1.0)
    invisible_loss_weight: float = field(default=1.0)
    weighting_scheme: str = field(
        default="logit_normal",
        metadata={
            "choices": ["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"],
        },
    )
    logit_mean: float = field(default=0.0)
    logit_std: float = field(default=1.0)
    mode_scale: float = field(default=1.29)
    # Dynamic shifting
    use_dynamic_shifting: bool = field(default=False)
    time_shift_type: str = field(
        default="linear",
        metadata={"choices": ["exponential", "linear"]},
    )
    base_seq_len: Optional[int] = field(default=256)
    max_seq_len: Optional[int] = field(default=4096)
    base_shift: Optional[float] = field(default=0.5)
    max_shift: Optional[float] = field(default=1.15)
    # VAE decode
    vae_decode_type: str = field(
        default="persistent",
        metadata={
            # 'persistent' carries the first-chunk cold-start fix (warm cache); others do not.
            "choices": ["default", "default_batch", "long", "persistent"],
        },
    )
    # EMA
    use_ema: bool = field(default=False)
    use_ema_validation: bool = field(default=False)
    ema_decay: float = field(default=0.999)
    ema_start_step: int = field(default=0)
    ema_zero3_port: int = field(default=10543)
    ema_deepspeed_config_file: str = field(default="configs/deepspeed/zero3.json")
    # Stage 1
    is_enable_stage1: bool = field(default=False)
    history_sizes: list[int] = field(default_factory=lambda: [16, 2, 1])
    latent_window_size: list[int] = field(default_factory=lambda: [9])
    is_random_drop: bool = field(default=False)
    random_drop_i2v_ratio: float = field(default=0)
    random_drop_v2v_ratio: float = field(default=0)
    random_drop_t2v_ratio: float = field(default=0)
    # GEO mixed-training conditioning ratios; remainder probability goes to full_geo mode.
    geo_condition_t2v_ratio: float = field(default=0)
    geo_condition_i2v_ratio: float = field(default=0)
    is_amplify_history: bool = field(default=False)
    history_scale_mode: str = field(
        default="per_head",
        metadata={
            "choices": ["scalar", "per_head"],
        },
    )
    has_multi_term_memory_patch: bool = field(default=False)
    is_train_full_multi_term_memory_patchg: bool = field(default=False)
    is_train_lora_multi_term_memory_patchg: bool = field(default=False)
    is_train_full_patch_embedding: bool = field(default=False)
    # Freeze text cross-attention (attn2) so prompt understanding stays at the base calibration.
    # SFT: freeze attn2 base params. LoRA: zero attn2 LoRA delta (-> base) then freeze it.
    freeze_cross_attn: bool = field(default=False)
    # Stage2 pyramid: how the short-tier warp interacts with per-stage resolution.
    # "fixed_mem" = legacy (warp prepended once at short-tier 2x, shared by all pyramid stages).
    # "synchronized" = warp downsampled per pyramid stage to match that stage's noise grid.
    stage2_warp_compression_mode: str = field(default="fixed_mem")
    is_train_lora_patch_embedding: bool = field(default=False)
    zero_history_timestep: bool = field(default=False)
    restrict_self_attn: bool = field(default=False)
    # DMD per-model override: restrict_self_attn for the real_score/critic (teacher) model build, decoupled
    # from the generator's restrict_self_attn. None = fall back to restrict_self_attn (legacy, both equal).
    # Use case: student(generator)=Evoke-Distilled native restrict=false (forcing true causes jumps) -> restrict_self_attn:false;
    #           teacher(real_score)=warp ckpt native restrict=true (forcing false causes jumps) -> real_score_restrict_self_attn:true.
    real_score_restrict_self_attn: Optional[bool] = field(default=None)
    guidance_cross_attn: bool = field(default=False)
    is_train_restrict_lora: bool = field(default=False)
    restrict_lora: bool = field(default=False)
    restrict_lora_rank: int = field(default=128)
    # Anti-drifting corruption for model input
    corrupt_model_input: bool = field(default=False)
    corrupt_mode_model_input: str = field(
        default="noise",
        metadata={
            "choices": ["noise", "downsample", "random"],
        },
    )
    corrupt_mode_prob_model_input: float = field(default=0.9)
    is_frame_independent_corrupt_model_input: bool = field(default=False)
    is_chunk_independent_corrupt_model_input: bool = field(default=False)
    noise_corrupt_ratio_model_input: float = field(default=1 / 3)
    noise_corrupt_clean_prob_model_input: float = field(default=0.1)
    downsample_min_corrupt_ratio_model_input: float = field(default=0.9)
    downsample_max_corrupt_ratio_model_input: float = field(default=1.0)
    # Anti-drifting corruption for history
    corrupt_history: bool = field(default=False)
    corrupt_mode_history: str = field(
        default="noise",
        metadata={
            "choices": ["noise", "downsample", "random"],
        },
    )
    corrupt_mode_prob_history: float = field(default=0.9)
    is_frame_independent_corrupt_history: bool = field(default=False)
    is_chunk_independent_corrupt_history: bool = field(default=False)
    noise_corrupt_ratio_history_short: float = field(default=1 / 3)
    noise_corrupt_ratio_history_mid: float = field(default=1 / 3)
    noise_corrupt_ratio_history_long: float = field(default=1 / 3)
    noise_corrupt_clean_prob_history: float = field(default=0.1)
    downsample_min_corrupt_ratio_history: float = field(default=0.9)
    downsample_max_corrupt_ratio_history: float = field(default=1.0)
    is_add_saturation: bool = field(default=False)
    saturation_ratio_min: float = field(default=0.3)
    saturation_ratio_max: float = field(default=1.7)
    saturation_ratio_clean_prob: float = field(default=0.1)
    # Stage 2
    is_enable_stage2: bool = field(default=False)
    is_navit_pyramid: bool = field(default=False)
    stage2_num_stages: int = field(default=3)
    stage2_timestep_shift: float = field(default=1.0)
    stage2_scheduler_gamma: float = field(default=1 / 3)
    stage2_stage_range: list[float] = field(default_factory=lambda: [0.0, 1 / 3, 2 / 3, 1])
    stage2_sample_ratios: list[int] = field(default_factory=lambda: [1, 2, 1])
    efficient_sample: bool = field(default=False)
    # Stage 3 VRAM options
    dmd_is_low_vram_mode: bool = field(default=False)
    is_gan_low_vram_mode: bool = field(default=False)
    dmd_is_offload_grad: bool = field(default=False)
    # Stage 3
    log_iters: int = field(default=200)
    no_visualize: bool = field(default=False)
    is_train_dmd: bool = field(default=False)
    max_grad_norm_critic: float = field(default=1.0)
    dmd_generator_deepspeed_config: Optional[str] = field(default=None)
    dmd_critic_deepspeed_config: Optional[str] = field(default=None)
    critic_learning_rate: Optional[float] = field(default=2e-6)
    dfake_gen_update_ratio: Optional[int] = field(default=5)
    dmd_denoising_step_list: list[int] = field(default_factory=lambda: [1000, 750, 500, 250])
    num_critic_input_frames: Optional[int] = field(default=21)
    dmd_timestep_shift: Optional[float] = field(default=5.0)
    dmd_last_step_only: bool = field(default=False)
    dmd_last_section_grad_only: bool = field(default=False)
    dmd_teacher_forcing: bool = field(default=False)
    dmd_teacher_forcing_ratio: float = field(default=0.2)
    fake_guidance_scale: float = field(default=0.0)
    real_guidance_scale: float = field(default=3.0)
    is_skip_first_section: bool = field(default=False)
    is_amplify_first_chunk: bool = field(default=False)
    # GT history conditioning
    is_use_gt_history: bool = field(default=False)
    use_gt_history_ratio: float = field(default=1.0)
    is_use_gt_coherence_dmd: bool = field(default=False)
    # VAE re-encode for DMD
    is_dmd_vae_decode: bool = field(default=False)
    # Multi-stage backward simulation
    is_multi_pyramid_stage_backward_simulated: bool = field(default=False)
    # Consistency alignment loss
    is_consistency_align: bool = field(default=False)
    consistentcy_align_weight: float = field(default=0.25)
    # Temporal smoothness loss
    is_smoothness_loss: bool = field(default=False)
    smoothness_loss_weight: float = field(default=1e-2)
    # Mean-variance regularization
    is_mean_var_regular: bool = field(default=False)
    mean_var_regular_weight: float = field(default=1.0)
    regular_mean: Optional[float] = field(default=0.00657021)
    regular_var: Optional[float] = field(default=0.85126512)
    is_x0_mean_var_regular: bool = field(default=False)
    mean_var_regular_x0_weight: float = field(default=1.0)
    regular_x0_mean: Optional[float] = field(default=-0.01618061)
    regular_x0_var: Optional[float] = field(default=0.27996052)
    is_chunk_mean_var_regular: bool = field(default=False)
    chunk_mean_var_regular_weight: float = field(default=1.0)
    chunk_regular_mean: Optional[float] = field(default=0.01906107)
    chunk_regular_var: Optional[float] = field(default=0.81397036)
    is_chunk_x0_mean_var_regular: bool = field(default=False)
    chunk_mean_var_regular_x0_weight: float = field(default=1.0)
    chunk_regular_x0_mean: Optional[float] = field(default=-0.01578601)
    chunk_regular_x0_var: Optional[float] = field(default=0.29913200)
    # ODE regression
    is_use_ode_regression: bool = field(default=False)
    is_only_ode_regression: bool = field(default=False)
    ode_regression_weight: float = field(default=0.25)
    ode_num_latent_sections_min: int = field(default=3)
    ode_num_latent_sections_max: int = field(default=3)
    # -- [v2v ODE dump] offline teacher trajectory generation (gen mode, not training;) --
    is_dump_ode_traj: bool = field(default=False)          # switch: collect teacher trajectories under warp conditioning and save them
    dump_ode_traj_out: str = field(default="")             # .pt output directory
    dump_ode_steps_per_stage: Optional[list] = field(default_factory=lambda: [20, 20, 20])  # teacher denoising steps per pyramid stage
    dump_ode_guidance_scale: float = field(default=5.0)    # teacher rollout CFG (matches inference)
    dump_ode_scheduler_type: str = field(default="euler")  # euler avoids unipc pyramid singularities
    dump_ode_max_samples: int = field(default=0)           # >0: each rank exits once it has stored N (smoke); 0=all
    # GAN
    is_use_gan: bool = field(default=False)
    gan_start_step: int = field(default=0)
    is_separate_gan_grad: bool = field(default=False)
    # GAN "approximate gradient" mode toggle, read in train_evoke.py's critic-backward guard
    # (... and (is_gan_aprox_grad or is_gan_low_vram_mode)). Was referenced but never declared here →
    # ConfigAttributeError once a GAN run actually reached that guard. Default False (the implemented
    # GAN path is is_gan_low_vram_mode).
    is_gan_aprox_grad: bool = field(default=False)
    is_use_gan_hooks: bool = field(default=False)
    is_use_gan_final: bool = field(default=False)
    gan_cond_map_dim: int = field(default=768)
    gan_hooks: list[int] = field(default_factory=lambda: [5, 15, 25, 35])
    gan_g_weight: float = field(default=1e-2)
    gan_d_weight: float = field(default=1e-2)
    aprox_r1: bool = field(default=False)
    aprox_r2: bool = field(default=False)
    r1_weight: float = field(default=0.0)
    r2_weight: float = field(default=0.0)
    r1_sigma: float = field(default=0.1)
    r2_sigma: float = field(default=0.1)
    # Reward model
    is_use_reward_model: bool = field(default=False)
    reward_start_step: int = field(default=0)
    reward_weight_vq: float = field(default=2.0)
    reward_weight_mq: float = field(default=2.0)
    reward_weight_ta: float = field(default=2.0)
    # DMD decoupling
    is_decouple_dmd: bool = field(default=False)
    decouple_ca_start_step: int = field(default=2000)
    decouple_ca_end_step: int = field(default=3000)
    # Cold start
    is_enable_cold_start: bool = field(default=False)
    cold_start_step: int = field(default=1000)
    stage_cold_start_step: Optional[int] = field(default=None)
    # Dynamic timestep sampling
    generator_is_forcing_low_renoise: bool = field(default=False)
    generator_dynamic_alpha: float = field(default=4.0)
    generator_dynamic_beta: float = field(default=1.5)
    generator_dynamic_sample_type: str = field(
        default="uniform",
        metadata={
            "choices": ["uniform", "beta"],
        },
    )
    generator_dynamic_step: int = field(default=1000)
    critic_dynamic_alpha: float = field(default=4.0)
    critic_dynamic_beta: float = field(default=1.5)
    critic_dynamic_sample_type: str = field(
        default="uniform",
        metadata={
            "choices": ["uniform", "beta"],
        },
    )
    critic_dynamic_step: int = field(default=1000)
    # Dynamic DMD section sampling
    dmd_num_latent_sections_min: Optional[int] = field(default=3)
    dmd_num_latent_sections_max: Optional[int] = field(default=3)
    dmd_dynamic_alpha: float = field(default=1.5)
    dmd_dynamic_beta: float = field(default=4.0)
    dmd_dynamic_sample_type: str = field(
        default="uniform",
        metadata={
            "choices": ["uniform", "beta"],
        },
    )
    dmd_dynamic_step: int = field(default=1000)
    # number of GT prefix chunks before rollout (shared by the evoke_teacher and sf-evoke paths; 0 = no injection)
    rollout_prefix_sections: int = field(default=1)
    # -- Evoke-Base teacher x segmented-curriculum self-forcing++ --
    #   everything defaults to off/empty -> no new branch is entered and the legacy path is bit-identical. validation lives in validate_sf_evoke_config (called explicitly at the entry point).
    # generic self-forcing switch (decouples the prefix-anchor + multi-section rollout scoring mechanism from the evoke_teacher gate onto the evoke teacher).
    sf_self_forcing: bool = field(default=False)
    # segmented curriculum: advance the rollout depth N + the tail scoring window W with step. each schedule entry = [N, W, step_budget] (cumulative).
    sf_curriculum_enabled: bool = field(default=False)
    sf_curriculum_schedule: List[List[int]] = field(default_factory=list)
    # -- decoupled dual-teacher rollout/gradient switches (consumed only when dual_teacher.enabled; bit-identical when off) --
    # front N-K sections = EvokeTeacher big window (grad on all of them); last 1 warp-ON section = Evoke. v2.1 /
    sf_evoke_teacher_front_window: bool = field(default=False)      # grad on all front sections (start_gradient_section_index=0), EvokeTeacher scores the front big window
    sf_detach_history_between_chunks: bool = field(default=False)  # T2: history appends pred_x0.detach() -> no BPTT across sections (mandatory when all front sections carry grad)
    sf_stage0_stopgrad_front: bool = field(default=False)       # front-section chunks detach at the pyramid stage0->1 boundary -> EvokeTeacher trains stages1-2 only, never touching stage0 (camera)
    # with dual experts, if front-section scoring routes to the **high-noise expert** (coarse-structure supervision) then do **not** detach stage0
    #   (lets high-expert gradients train stage0 long-range/coarse structure through all 3 stages); routing to the low expert still detaches (stages1-2 only, protecting the camera).
    #   precondition: the single front-section scoring t is sampled before rollout, used to decide the detach, and reused for scoring (_generator_loss). single-expert /
    #   non-dual / False -> always detach per sf_stage0_stopgrad_front (byte-identical).
    sf_front_stage0_high_keep: bool = field(default=True)
    sf_return_full_rollout: bool = field(default=False)         # rollout returns the whole generated region [sum(history_sizes):], _generator_loss slices front/tail
    # per-section recompute (chunk-level recompute): in the front big window only the exit-step forward of each section carries grad, and N sections
    #   accumulate O(N) activations, hitting the 141G H200 wall (the N=20 rollout peaks at ~140G). with use_kv_cache=False a section forward is a pure function of its inputs -> use
    #   torch.utils.checkpoint(use_reentrant=False) to drop activations in the forward and recompute in backward, so only 1 section is resident at any time -> the peak is decoupled from N
    #   (N=20 then fits on 16 GPUs). cost = 1 extra exit-step forward recompute per section in backward. see utils_evoke_post.run_generator.
    # (CPU offload/save_on_cpu was tried and dropped after a pin_memory host OOM SIGKILL)
    sf_recompute_sections: bool = field(default=False)
    # recompute only the top K pyramid stages by "activation footprint" (1 step per stage under 3-NFE; the top stage high-resolution tokens are ~16x base = the bulk of a section activation).
    #   default 1 = checkpoint the top stage only -> low-resolution stage activations stay resident (cheap) and backward recomputes only 1 (the most expensive) per section -> overhead halved.
    #   <=0 = recompute every grad stage (most memory-saving / most expensive). only active when sf_recompute_sections=true.
    sf_recompute_top_stages: int = field(default=1)
    # sequence-parallel (SP) group size G: G consecutive ranks in a group share one clip and split, along the frame dim, the activations
    #   of that whole front-window long sequence scored by the EvokeTeacher teacher/critic (~109G/rank -> /G); across groups = ZeRO-3 DP (effective bs = world/G).
    # 1 = off (byte-identical, all SP primitives are identity no-ops). >1 requires world%G==0.
    #   revives the vendor's sp_runtime + the DiT dead code; only the EvokeTeacher critic engine is split (the generator/Evoke are not).
    sf_critic_sp_world_size: int = field(default=1)
    sf_decouple_rollout: bool = field(default=False)   # the G ranks in a group each run a different clip -> effective bs x G (6->24, removing rollout redundancy); default off = byte-identical
    # -- student(generator)-side parallelism: a second-level decomposition of the same SP group, G = G_p x G_u ---------
    # (the teacher/critic SP-G is not touched at all)
    # mechanism A -- chunk-parallel gradients: history tiers are already detached between sections => the N sections are N disconnected subgraphs, dispatched whole by k % G_p,
    #   with **zero communication**. only backward is split (76% of the student side); forward values stay redundant on all ranks (the autoregressive dependency cannot be saved).
    #   101.9s -> 15.3s (8x1). default false = byte-identical.
    sf_student_chunk_parallel: bool = field(default=False)
    # mechanism B -- Ulysses token-SP width G_u (1 = off, target state 2; G_p = G//G_u is derived automatically). student attention is
    #   dense with no mask => **zero halo**, so it uses an all-to-all over the head dim (pure permutation, zero reduction, no memory blowup). rollout 32.0 -> 17.4s.
    sf_student_sp_ulysses: int = field(default=1)
    # runtime diag (on for smoke, off for formal): pred_video consistent within the group / S consistent within the group / collective ordinals consistent /
    #   assert the teacher get_sp_size()==G. runs on the first step only, zero overhead; it catches hangs and silent mis-scoring (unrecoverable after the fact).
    sf_student_cp_diag: bool = field(default=False)
    # share the "frozen expert base offloaded to host" within a node: under per-expert offload each rank
    #   holds its own copy of the un-routed expert (14B bf16 ~= 28GB), 8 ranks per node = 224GB of which 196GB is pure duplication -> the 56-GPU r1
    #   was SIGKILLed by the host OOM-killer at step 3. when on, LOCAL_RANK0 writes a single copy to /dev/shm and every rank
    #   mmap-attaches; swap-out becomes "point p.data back at the shared tensor" (zero-copy, which also removes the 28GB D2H -> no per-step slowdown).
    #   needs per-expert offload (dual expert + offload); default off = byte-identical. see evoke/modules/evoke_teacher/shared_host_base.py
    sf_evoke_teacher_shared_host_base: bool = field(default=False)
    # after each student optimizer step, every critic
    # independently takes a few optimizer steps. 1 = the validated large-batch B behavior; formal C2S1=2:
    # after student 1xbs24, the EvokeTeacher/Evoke critics each do 2xbs12, consuming all 24 rollouts exactly once.
    sf_critic_steps_per_student: int = field(default=1)
    # >0: runtime-assert each critic substep's global batch against the actual WORLD. formal C2S1=12;
    # a scaled-down smoke can set 0, letting the same planner bucket automatically by the actual world size.
    sf_critic_expected_global_batch_size: int = field(default=0)
    # F6 single shared rollout: the critic reuses the generator's detached rollout (no independent re-roll); when true it forces dfake_gen_update_ratio=1.
    sf_share_rollout: bool = field(default=False)
    # whether teacher/critic scoring carries warp. default false = drop warp before the DiT (Evoke-Base has no warp;).
    sf_teacher_warp: bool = field(default=False)
    # whether the critic carries warp; None = follow sf_teacher_warp.
    sf_critic_warp: Optional[bool] = field(default=None)
    # number of tail chunks for which the student renders warp (>= the scoring window W; the front sections are a pure latent rollout with no warp Q3b).
    #   0/None = render warp throughout (K=N, no tail optimization).
    sf_warp_tail_chunks: Optional[int] = field(default=None)
    # skip the first latent frame of every section when scoring (DMD gradient mask + a symmetric mask on the critic denoising loss;
    # the teacher conditioning window is unchanged). motivation: the latent[0] Evoke-Base predicts per chunk is an I-frame/first-frame distribution, whereas the student
    # (cont1800 lineage) is a continuation distribution -- not masking it would pull the student first latent toward I-frame and reintroduce chunk-boundary flicker.
    sf_score_skip_first_latent: bool = field(default=False)
    # alternating misaligned scoring window, which fixes skip-first's frame0 vacuum slot (under a pure mask frame0
    # gets zero supervision and shared-weight updates drift into blocky/grid patterns). 50/50 per step,
    # rank-synchronized: off=0 is the original window [f0..f8] (masking f0); off=1 shifts left one frame to
    # [p8|f0..f7] (masking the sacrificial p8, the previous section's last frame taken misaligned from the student
    # rollout rather than spliced in; prev_short shifts to p7 to avoid a duplicate frame and a RoPE collision). The
    # teacher's I-frame slot always lands on a masked frame, so every frame is supervised in expectation.
    # Requires sf_score_skip_first_latent=true + sf_teacher_warp=false + W=1 (asserted).
    sf_score_window_jitter: bool = field(default=False)
    # upper bound on the jitter left shift: off in {0..max_off}, P(0)=1/2, {1..max_off} share the other 1/2.
    # With off in {0,1} f0 is only supervised at slot1, so the teacher's slot1 I-frame halo accumulates on f0 at full
    # concentration; spreading it over slot1..8 dilutes it ~max_off x. Cost: mid-section frame coverage goes from 1 to
    # 9/16..15/16 (f0/f8 stay at 1/2, no zero-coverage slot). Default 1 = the legacy off in {0,1} behaviour,
    # bit-identical. >1 requires N>=2 for every curriculum stage (with an N=1 stage, off=8 would re-slice the video
    # I-frame latent p0 into the prev slot).
    sf_score_window_jitter_max_off: int = field(default=1)
    # fit the critic denoising loss on all frames (lifts sf_score_skip_first_latent's symmetric mask on the critic
    # side; the generator-side DMD gradient mask is unaffected). motivation: the critic loss does not flow back into the generator (its input is already detached),
    # so the slot0 mask only makes the critic keep Base's I-frame worldview forever at the window head (probes measured fake[s0] tracking the teacher, not the
    # student), which blinds the brake (fake tracking the student drift) at the window head. default false = the legacy symmetric-mask behavior.
    sf_critic_full_frame: bool = field(default=False)
    # mask the first k slots of the scoring window (generator DMD gradient + the symmetric critic mask); for k>=2 the
    # jitter becomes off in {0,k} 50/50, supervising f_k..f8 or f0..f_{8-k}. Design invariant: live frames never sit
    # on the poisoned slots 0..1 (slot0 = I-frame template, slot1 = halo; slots 2-8 clean), so the positional bias on
    # live frames goes to zero and the detached sacrificial frame absorbs it; coverage is 1/2 head/tail and 1
    # mid-section, with no vacuum, preserving the student's all-frames-same-distribution initialization. k=1 is the
    # original skip_first+jitter{0,1} behaviour. k>=2 is mutually exclusive with sf_score_window_jitter_max_off>1.
    sf_score_skip_first_k: int = field(default=1)
    # when the teacher(real) scores, swap the long/mid tiers for same-timeline GT latent slices (short/prev keep
    # student frames to preserve visual continuity; the prefix anchor is GT anyway). The critic(fake) keeps
    # all-student tiers, so the asymmetry becomes a restoring force toward the GT colour tone, curing the low-band
    # regime's anchorless blue-white drift. Self-limiting: when the student tone matches the GT tone the difference
    # goes to zero, so there is no white-fog runaway. Default false = legacy behaviour, bit-identical.
    sf_teacher_gt_longmid: bool = field(default=False)
    # [M9 stage-2 inverse mask] DMD gradients act only on the first latent frame of each section (frame1-8 get zero DMD gradient;
    # GAN/critic stay on all frames). the motivation is the opposite of sf_score_skip_first_latent: the student (after being pulled off by the Evoke-Base teacher)
    # has a first latent biased toward the I-frame distribution, so a tiny LoRA with the warp-5600 teacher pulls only frame0 back to the continuation distribution --
    # scoring all frames dilutes the frame0 signal to 1/9 and risks appearance erasure on frame1-8, and the concentrated mask solves both.
    # note: the masked mean gives frame0 an extra ~win x gradient amplification (smaller denominator), which is the expected "turn it up" effect. the two switches are mutually exclusive.
    dmd_score_first_latent_only: bool = field(default=False)
    # first-latent inverse mask for the flat-distill (non-SF) path: DMD gradients mask out the first latent of each chunk (GAN/critic stay on all
    # frames). same motivation as sf_score_skip_first_latent (that switch only takes effect on the SF tier scoring path and never reaches flat distill): the Evoke-Base
    # teacher's latent[0] is an I-frame distribution, and not masking pulls the student first frame back toward I-frame. mutually exclusive with dmd_score_first_latent_only.
    dmd_score_skip_first_latent: bool = field(default=False)
    # cut the DMD gradient of the student's entire first generated chunk (GAN/critic stay on all frames). When the
    # EvokeTeacher teacher scores [prefix|20chunks] at once, the Wan-VAE independent first-frame distribution exists
    # only at the very first latent, so the teacher's per-chunk separable score pulls the student's first chunk toward
    # that distribution and flickers at chunk boundaries. This masks the first generated chunk
    # (gradient_mask[:,:,_sf_P:_sf_P+win]=False). Unlike sf_score_skip_first_latent (per-chunk first frame, Evoke tier
    # path), it masks only the whole first chunk of the generated region. Default False -> bit-identical.
    dmd_score_skip_first_chunk: bool = field(default=False)
    # -- the critic forward+backward runs on a random contiguous window only --
    # N=20 critic backward OOMs at 189 frames. Design:
    #   - the student still rolls out all 20 chunks (autoregressive dynamics unchanged) and the teacher still forwards
    #     all 189 frames (no_grad, cheap); the generator DMD gradient mask covers the window only, excluding g1.
    #   - the **trainable critic** (the grad-carrying forward + backward in _critic_loss) eats only
    #     [GT prefix | random window] = (1+win_chunks)*win frames per step, the window being sf_score_window_chunks
    #     contiguous chunks inside the generated region, never covering g1 (start chunk s in [2, N-wc+1]). This is the
    #     real memory fix: critic forward frames 189 -> ~99.
    #   - s is seeded-random per step and rank-consistent (train_evoke seeds with global_step).
    # The critic trains on the rebased window while the gen-loss critic query still spans all 189 frames (no_grad), so
    #   there is a train/query context-length asymmetry -- a mild approximation, since the per-frame denoising task
    #   generalizes. 0 = OFF (bit-identical); >0 = window chunk count. Needs sf_evoke_teacher_front_window + warp.
    sf_score_window_chunks: int = field(default=0)
    # the **asymmetry** of the clamped window-start sampling: tilt more slots slide off the high side, tilt fewer off the low side ->
    #   tail chunks (high s) get a higher coverage probability than head ones (low s), compensating for autoregressive drift being worst at the tail. 0=symmetric (both ends equally likely);
    #   1=slight tail tilt (mean coverage of the second half of the chunks ~= the first half x1.15, g20~=0.39 vs g2~=0.32); larger = more tail-biased (about +0.18 on the back/front ratio per +1).
    #   constraint 0 <= tilt <= wc-1 (the low-side slide-off amount wc-1-tilt must be >=0). only active when sf_score_window_chunks>0.
    sf_score_window_tail_tilt: int = field(default=0)
    # -- **weight-level** warm-start from save_checkpoints_custom artifacts --
    # The evoke_teacher path forbids accelerate resume, and save_checkpoints_custom ckpts carry no
    #   optimizer/RNG/scheduler state, so weights are the only thing that can be inherited. This switch restores all
    #   three artifacts:
    #     (1) generator LoRA  <dir>/pytorch_lora_weights.safetensors (a hard link of weights/lora.safetensors)
    #     (2) memory patch    <dir>/transformer_partial.pth (= weights/memory.pth, the multi-term memory patch)
    #     (3) critic LoRA     <dir>/critic/critic_evoke_teacher_lora.safetensors
    #   (3) is the critical one: without it the critic starts from scratch and the DMD fake-score first hands the
    #   student a stretch of wrong gradient. Not inherited: optimizer momentum, lr_scheduler progress, dataloader
    #   position, RNG. Under constant lr the first two matter little, but zeroing Adam's moments makes the first
    #   several steps noisier, so pair this with sf_gen_freeze_steps and let the critic settle first.
    sf_warmstart_dir: Optional[str] = field(default=None)
    # True = load **only** the critic LoRA, skipping the generator LoRA and the memory patch. For the case where the
    #   generator starts from a merged directory (transformer_model_name_or_path): the merged weights already have the
    #   LoRA folded in and carry the memory patch modules, so loading the generator LoRA again would apply it twice.
    #   The critic LoRA is not in the merged directory (it hangs off the teacher wrapper) and must be loaded back
    #   separately, or the critic starts from scratch. This path also sidesteps the HF-offline
    #   lora_state_dict(weight_name) trap, since it only uses safetensors.load_file.
    sf_warmstart_critic_only: bool = field(default=False)
    # for the first K steps **only the critic is updated and the student params do not move at all** (K=0 = off, byte-identical with legacy).
    #   implementation: the student rollout/DMD forward still runs (the critic has to reuse its shared rollout, and the warp convention must match),
    #   only the generator backward + engine.step() + lr_scheduler.step() are skipped => the student params stay bit-identical.
    #   it also saves gen_bwd (~18.5s of the 96s). K is one constant across all ranks => it introduces no rank divergence.
    sf_gen_freeze_steps: int = field(default=0)
    # -- student i2v / v2v mixed training (byte-identical to the legacy v2v path when ratio=0) --
    # v2v: the data side hands over P=1 GT prefix **chunk** (9 latents = 33 pixel frames) as the rollout start; the
    #   teacher i2v cond is the first of those 9 frames and the scoring sequence is [GT chunk 9 | g1..g20] = 189.
    # i2v: the student takes **1 latent** (pixel frame 0 = the reference image) as condition, scoring sequence
    #   [ref image 1 | g1..g20] = 181. The teacher cond frame is that same frame, so four-region consistency holds.
    #   181 is native to the teacher: its cs9 30s stage had num_frames=721 -> T_lat=181=20x9+1 and the 60s stage
    #   (the current weights) 1441 -> 361=40x9+1, so a 1-frame tail chunk is a shape it trained on.
    # The data flow is untouched: num_frames stays 753 and both modes are slices of the same clip (i2v uses
    #   prefix[:, :, :1] + teacher_y[:, :, :181]), so the sample set is identical to a pure v2v run.
    # Dispatch is **per sample**, not drawn by ratio:
    #   - image-only samples (a jpg -> 1 frame) always go i2v; they cannot provide temporal GT/pose;
    #   - video samples go v2v by default and switch to i2v with probability sf_i2v_ratio (reference image = frame 0).
    #     0.0 leaves all video samples on v2v (the i2v count is then set by the image sources' share); 1.0 moves all.
    sf_i2v_ratio: float = field(default=0.0)
    # NOTE: the **master switch** of the i2v path, doubling as a layout param: the number of GT prefix latent frames on i2v steps.
    #   0 (default) = the i2v path is fully off -- the data side produces no i2v tensors and training enters no i2v branch => byte-identical with legacy configs;
    #     in that state an image-only sample appearing in the data will fail-fast (rather than silently misbehave).
    #   1 = a single reference image (Wan-VAE latent 0 is exactly the independently encoded pixel frame 0). must be < rollout_prefix_sections*latent_window_size.
    sf_i2v_prefix_latent_frames: int = field(default=0)
    # mode sampling granularity: "group" = each SP group samples independently (necessarily consistent inside the group, num_groups mixed samples per step, recommended);
    #   "step" = one global mode per step (more conservative, only one mode per step). both use a seeded RNG (seeded with global_step)
    #   -> rank-consistent with **zero collective communication** and zero global RNG consumption (the same recipe as _sf_score_window, utils_evoke_post.py).
    #   WARNING: never let ranks call torch.rand independently: inconsistency inside an SP group causes an NCCL shape mismatch and breaks mechanism A's intra-group RNG symmetry.
    sf_i2v_mode_scope: str = field(default="group")
    # what to feed the student history 1x slot (the most recent history frame) on i2v steps:
    #   "static_repeat" (default, matching i2v inference verbatim) = VAE(reference image repeated 33 frames)[..., -1:],
    #     i.e. the pipeline's fake_image_latents. It is a **continuation-distribution** latent, from the same source
    #     as the short tier's training distribution; the x0 anchor is still the single-frame I-frame latent.
    #   "iframe" = 1x reuses the I-frame latent, saving one 33-frame VAE encode but making the slot an I-frame
    #     distribution, which is inconsistent with inference unless fake_latents=image_latents is passed there.
    sf_i2v_hist_latent_mode: str = field(default="static_repeat")
    # whether i2v steps unmask the DMD gradient of g1. Under i2v the teacher's I-frame slot is the true reference
    #   image and the student's g1 is the continuation section right after it -- the native teacher training layout --
    #   so dmd_score_skip_first_chunk's rationale no longer holds and all 20/20 chunks can take gradient (v2v steps
    #   still mask g1). Must be paired with mechanism A's skip: student_sp._skip_first_chunk structurally skips g1's
    #   backward, so lifting only the mask would silently mean covered-but-no-gradient. This switch toggles both
    #   (train_evoke calls set_skip_first_chunk every step). Default False = both modes mask g1.
    sf_i2v_score_g1: bool = field(default=False)
    # -- GT window supervision regularizer --
    # Appended once every every_k steps at the end of _generator_loss: draw a random window from sf_gt_latents (1 GT
    # prefix chunk + 1 target chunk), render warp conditioning from GT (one-shot SFWarpRollout, pure GT sources), take
    # an in-stage0 target, and add weight*L_geo to dmd_loss in the same backward. Without the camera teacher both
    # experts are nocam and DMD exerts no camera-following pressure; this brings it back with real-data warp.
    # t_min/t_max are in-stage0 sigma x1000 semantics, not the global t axis, so they do not interact with expert
    # routing. weight=0 is fully OFF (branch not entered, no RNG consumed). Needs use_geometric_state=true +
    # dual_teacher.enabled=false.
    # sf_dmd_normalizer_masked: whether the eq.8 normalizer averages only the frames that enter the loss. False
    #   (legacy) averages |p_real| over all 189 frames while the loss covers [18:189), so the denominator also counts
    #   the 9 GT prefix + 9 g1 frames (9.5%); GT frames are real data that the teacher reconstructs with smaller
    #   error, dragging the denominator down and scaling gradients up. True shares the region with gradient_mask.
    sf_dmd_normalizer_masked: bool = field(default=False)
    # encode only the required pixel prefix on demand instead of the full-length GT VAE encode.
    #   False encodes all num_frames (753) frames every step (16.0s/step measured), but the consumers of the 189
    #   latents are only:
    #     - sf_prefix_latents = latent[0:9)          every step (9 frames = pixels 0..32)
    #     - GEO-REG           = {0} u [9j-19, 9j+9)  once every 3 steps, j random in [1,20], mean 28 frames
    #     (GT-ANCHOR is off, x0/history/target are placeholders, and the critic does not eat them)
    #   so only ~8% is consumed in expectation. True encodes 33 pixel frames on non-GEO steps and 36j+33 on GEO
    #   steps; Wan-VAE is strictly causal in time (see the cond_y_fastpath docstring), so this is **bit-identical**
    #   to the full-length encode. GEO-REG's j must be sampled before materialize, which changes the RNG consumption
    #   order -- not bit-comparable against a run with this off. SF_GT_VERIFY=1 bit-checks the first few calls.
    sf_gt_encode_on_demand: bool = field(default=False)
    sf_geo_reg_weight: float = field(default=0.0)
    sf_geo_reg_every_k: int = field(default=1)
    sf_geo_reg_t_min: int = field(default=666)
    sf_geo_reg_t_max: int = field(default=899)
    # strip warp before teacher/critic scoring on the flat-distill path (the short tier drops its middle warp frames + the score kwargs set
    # geo_warp_frames=0): warp-free models such as teacher=Evoke-Base have never seen warp tokens, so keeping them makes the scoring OOD.
    # the generator rollout side (gt_all_data -> run_generator) is unaffected and the student still carries warp. GAN D real/fake stay symmetric on the same side.
    # on the SF path use sf_teacher_warp instead (this switch auto-no-ops in the SF tier scoring branch).
    dmd_teacher_strip_warp: bool = field(default=False)
    # explicit upper/lower bound on the generator DMD scoring t band (anchored to the pyramid stage boundaries rather than forcing_low's
    # hardcoded 500). stage_range [0,1/3,2/3,1] -> the band handled by stage0 (coarse level, warp injection) = (666,1000], stage1=(333,666],
    # stage2=(0,333]. max=666 means "the teacher never scores in the band handled by stage0" (structural supervision is structurally excluded).
    # WARNING: an explicit max requires dmd_timestep_shift<=1 (the shift warp is applied after the band mapping, and >1 would break the cap; fail-fast asserted at the use site).
    # None/0 = legacy behavior (min=0 -> clamp20, max follows forcing_low or 1000). the critic training band is unaffected (full band).
    dmd_score_timestep_max: Optional[int] = field(default=None)
    dmd_score_timestep_min: int = field(default=0)
    # thin high-band mixed sampling: low-band (sigma<=0.5) scoring has no restoring force on DC low-frequency
    # statistics such as colour tone and brightness (at low sigma the low frequencies pass through teacher and critic
    # unchanged), so the residual bias becomes an anchorless push and drifts blue-white. Colour tone is decided by the
    # high-noise t (spectral autoregression), so a small dose of high-band scoring restores a tone anchor; the
    # probability stays low because high-noise structure erasure is a dose effect. Each generator DMD scoring switches
    # the whole band to [highband_min, highband_max] with probability prob (both endpoints are actual t = sigma x1000,
    # inverse-warped when shift>1). prob=0 -> legacy, bit-identical, no RNG consumed. Prerequisite:
    # dmd_score_timestep_max must already be set -- the thin high band is a finisher for the low-band cap.
    dmd_score_highband_prob: float = field(default=0.0)
    dmd_score_highband_min: int = field(default=666)
    dmd_score_highband_max: int = field(default=1000)
    # explicit upper/lower bound on the critic(fake_score) denoising training t band. Without a cap the critic's
    # training mass concentrates at actual t~=833 under shift5 while DMD only queries it inside the
    # dmd_score_timestep_max band (actual <=500), so in-band underfitting is structural, fake ~= native Base, and the
    # DMD gradient degenerates into the teacher's open-loop CFG field with no brake. Aligning the caps puts 100% of
    # the critic training mass inside the queried band.
    # max is actual-t semantics (= sigma x1000), inverse-warped at the use site when shift>1, like
    # dmd_score_timestep_max; min is nominal semantics (passed straight into sample_dynamic_timestep and then
    # shift-warped), so the two are asymmetric and min=0 has no effect. None = full band. DMD denoising path only:
    # recipes with GAN on are forbidden (the GAN branch reuses critic_timestep, so the cap would move its noising t).
    critic_score_timestep_max: Optional[int] = field(default=None)
    critic_score_timestep_min: int = field(default=0)
    # Dynamic ODE section sampling
    ode_dynamic_alpha: float = field(default=1.5)
    ode_dynamic_beta: float = field(default=4.0)
    ode_dynamic_sample_type: str = field(
        default="uniform",
        metadata={
            "choices": ["uniform", "beta"],
        },
    )
    ode_dynamic_step: int = field(default=1000)
    # Error recycling
    use_error_recycling: bool = field(default=False)
    # Opt-in: allow error recycling on stage2 / NaViT pyramid. Injection = the DOWNSTREAM unified path in
    # train_evoke (warp + prev_short; see model_config.geometric_state.error_inject_tiers), which TRACKS
    # per_item_depth -> banking compounds depth, so max_error_depth>1 IS supported. Per-pyramid-stage banking
    # (_flow_loss list-branch) + 3-resolution bucket registration are all wired; only the entrypoint assert
    # blocked it. (Legacy stage1 history-tier apply_error_injection is NOT used on stage2.)
    allow_error_recycling_stage2: bool = field(default=False)
    # Asymmetric DMD teacher conditioning: feed the DM teacher (pred_real, adapters-OFF forwards
    # in compute_kl_grad) a CLEAN warp history = warp/prev_short BEFORE the err-bank "+=" injection (keeps
    # materialization visibility/warp-noise). Student rollout + critic (fake_score) + GAN keep the DEGRADED
    # (errbank) history. Default False -> byte-identical; no-op when errbank inactive (clean==degraded).
    recycle_teacher_clean: bool = field(default=False)
    y_error_sample_from_all_grids: bool = field(default=True)

    error_buffer_size: int = field(default=500)
    buffer_replacement_strategy: str = field(default="l2_batch")
    buffer_warmup_iter: int = field(default=50)
    timestep_grid_size: int = field(default=25)
    num_grids: int = field(default=50)

    y_error_num: int = field(default=6)
    error_modulate_factor: float = field(default=0.0)
    error_setting: int = field(default=1)
    noise_prob: float = field(default=0.01)
    y_prob: float = field(default=0.9)
    latent_prob: float = field(default=0.9)
    clean_prob: float = field(default=0.2)
    clean_buffer_update_prob: float = field(default=0.1)
    # —— Error-bank depth + low-noise grid sampling ——
    # y_error sampling grid restriction for mem/warp injection. grid 0 = highest noise,
    # grid (num_grids-1) = lowest noise; "low_noise" samples the LAST `ref_inject_grid_topk` grids.
    # "all" = legacy behavior (all grids). "current_grid" = SVI default (current timestep's grid only).
    ref_inject_grid_mode: str = field(default="all")
    ref_inject_grid_topk: int = field(default=8)
    # Depth bank: cap recursive error compounding. max_error_depth=1 => only clean->depth1 errors
    # (no compounding). depth_sample_ratio length must equal max_error_depth (normalized internally).
    max_error_depth: int = field(default=1)
    depth_sample_ratio: List[float] = field(default_factory=lambda: [1.0])
    # Norm-cap fallback : reject banked errors whose L2 norm exceeds k * sampled-error norm.
    # 0.0 disables. Companion/alternative to depth bounding.
    error_norm_cap_k: float = field(default=0.0)
    # Cross-GPU buffer gather during warmup. Default OFF: the manual accelerator.gather() interleaves
    # with DeepSpeed ZeRO-2 grad-reduction collectives under grad-accumulation and desyncs the
    # collective order across ranks -> NCCL allreduce timeout/hang at the first warmup steps. Local-only
    # banking needs no collective and is safe; it just fills buffers ~world_size× slower during warmup.
    error_buffer_distributed_warmup: bool = field(default=False)


@dataclass
class Args:
    output_dir: str = field(default="Evoke")
    seed: int = field(default=42)
    report_to: ReportTo = field(default_factory=ReportTo)
    data_config: DataConfig = field(default_factory=DataConfig)
    model_config: ModelConfig = field(default_factory=ModelConfig)
    validation_config: ValidationConfig = field(default_factory=ValidationConfig)
    training_config: TrainingConfig = field(default_factory=TrainingConfig)
    logging_dir: str = field(default="logs")


def validate_cloud_warp_backend(args) -> None:
    """Fail-fast validation of cloud_warp.backend and its sub-block.

    WARNING: must be called explicitly at the train_evoke entry point -- OmegaConf loading does not
    trigger __post_init__, so validation placed there would never run.
    """
    cw = getattr(getattr(getattr(args, "model_config", None), "geometric_state", None), "cloud_warp", None)
    if cw is None or not bool(getattr(cw, "enabled", False)):
        return
    backend = str(getattr(cw, "backend", "da3") or "da3").lower()
    if backend not in ("da3", "vigeo"):
        raise ValueError(f"[cloud_warp] backend must be 'da3' or 'vigeo', got {backend!r}")
    vg = getattr(cw, "vigeo", None)
    if backend != "vigeo":
        # Guard against editing the vigeo block while still running da3, which would look like it took
        # effect. Every field is compared against the schema default, so a knob added later is covered
        # without editing this check.
        ref = VigeoBackendConfig()
        changed = [f.name for f in fields(ref)
                   if vg is not None and getattr(vg, f.name, None) != getattr(ref, f.name)]
        if changed:
            raise ValueError(
                f"[cloud_warp] backend={backend} but cloud_warp.vigeo has non-default {changed}; "
                f"those values are ignored. Set backend: vigeo or revert them.")
        return
    if str(getattr(vg, "intr_source", "gt")) not in ("gt", "vigeo"):
        raise ValueError(f"[cloud_warp.vigeo] intr_source must be 'gt' or 'vigeo', got {vg.intr_source!r}")
    if str(getattr(vg, "conf_transform", "exp")) not in ("exp", "none"):
        raise ValueError(f"[cloud_warp.vigeo] conf_transform must be 'exp' or 'none', got {vg.conf_transform!r}")
    mode = str(getattr(vg, "mode", "chunk"))
    if mode not in ("offline", "chunk", "online"):
        raise ValueError(f"[cloud_warp.vigeo] mode must be offline/chunk/online, got {mode!r}")
    scale_mode = str(getattr(vg, "scale_mode", "anchor"))
    if scale_mode not in ("per_window", "anchor"):
        raise ValueError(f"[cloud_warp.vigeo] scale_mode must be 'per_window' or 'anchor', got {scale_mode!r}")
    if scale_mode == "anchor" and mode == "offline":
        # Anchoring locks one scale for a whole stream; with each window independent there is no stream
        # to anchor, so the lock would just freeze the first window's scale for unrelated windows.
        raise ValueError("[cloud_warp.vigeo] scale_mode=anchor requires mode=chunk or online")
    if int(getattr(vg, "total_budget", 0)) > 0:
        print("[cloud_warp.vigeo] WARNING: total_budget is set explicitly; if the per-global-block share "
              "falls to one frame's token count or below, ViGeo's cache eviction silently stops and the "
              "kv-cache grows without bound. Prefer total_budget=0 with cache_keep_frames.", flush=True)
    # Assets must exist now, not at the first forward: a bad path otherwise surfaces as a skipped window
    # and the run completes with an all-black warp.
    from evoke.modules.geometric_state.depth_backend import check_assets
    check_assets(backend, weights=getattr(vg, "weights", None), src=getattr(vg, "src", None))


def validate_sf10s_evoke_teacher_config(args) -> None:
    """fail-fast validation for real_score_arch=evoke_teacher (PLAN table 5 / review M7/S7).

    WARNING: must be called explicitly at the train_evoke entry point: OmegaConf loading does not trigger __post_init__ (memory:
    omegaconf-skips-post-init), so putting it in post_init means it normally never runs.
    """
    mc, tc = args.model_config, args.training_config
    # -- preconditions for re-drawing the ratio subset each epoch (default false -> the whole block no-ops) --
    #   placed **before** the arch early-return: so misconfiguring it on a non-evoke_teacher arch also fail-fasts.
    if bool(getattr(args.data_config, "resample_ratio_each_epoch", False)):
        assert not bool(getattr(args.data_config, "persistent_workers", False)), (
            "[EPOCH-RESAMPLE] resample_ratio_each_epoch=true requires persistent_workers=false -- "
            "resident workers do not re-fork on a new epoch, so the indices swapped in by the main process never reach them => the re-draw **silently does nothing**")
        assert bool(getattr(args.data_config, "use_multi_dataset", False)), (
            "[EPOCH-RESAMPLE] only effective on the online multi-source path with use_multi_dataset (ratio/SubsampledDataset are concepts of that path)")
    # preconditions for the dual real-score teacher (only effective when dual_teacher.enabled;
    #   when off all asserts below no-op -> single teacher, whole chain bit-identical). NOTE: deliberately placed before the arch early-return:
    #   so "dual misconfigured on a non-evoke_teacher arch" also fail-fasts -- otherwise this function returns early when arch!=evoke_teacher,
    #   while validate_sf_evoke_config does not check dual (and dual is only built in the evoke_teacher branch) -> dual_teacher.enabled
    # would be silently ignored.
    _dt = getattr(mc, "dual_teacher", None)
    if _dt is not None and bool(getattr(_dt, "enabled", False)):
        assert getattr(mc, "real_score_arch", "evoke") == "evoke_teacher", \
            "[DUAL-TEACHER] dual_teacher.enabled only supports real_score_arch=evoke_teacher (the Evoke pose teacher is built inside the evoke_teacher dual-teacher branch)"
        # the convex-weight assert is void (decoupled: two independent DMD losses summed, lambda_hb non-convex); the W==1 curriculum assert is void
        #   (EvokeTeacher scores the front N-K big window, Evoke the last 1 section; they no longer share a window).
        assert _dt.evoke_model_path and os.path.exists(_dt.evoke_model_path), \
            f"[DUAL-TEACHER] dual_teacher.evoke_model_path must be set and must exist, got {_dt.evoke_model_path!r}"
        assert getattr(tc, "use_geometric_state", False) is True, \
            "[DUAL-TEACHER] requires training_config.use_geometric_state=true (the camera force relies on the warp tail; " \
            "Evoke keep-warp asserts geo_warp_frames>0 at runtime)"
        # gt-anchor must be off: on the dual path sf_gt_latents=None (full-clip encoding skipped), so enabling gt-anchor would hit the _generator_loss
        #   "gt_latents_history_long is not None" assert -> hard crash.
        assert not bool(getattr(tc, "sf_teacher_gt_longmid", False)), \
            "sf_teacher_gt_longmid must be false (the dual path skips full-clip encoding, sf_gt_latents=None; gt-anchor conflicts with the long-range objective)"
        # N(entry[0]) >= 2 for every curriculum stage: needs >=1 front section (EvokeTeacher) + 1 tail warp-ON section (Evoke).
        _dt_sched = list(getattr(tc, "sf_curriculum_schedule", []) or [])
        for _i, _ent in enumerate(_dt_sched):
            assert int(_ent[0]) >= 2, \
                f"[DUAL-TEACHER v2.1] the N(entry[0]) of curriculum[{_i}] must be >=2 (needs >=1 front EvokeTeacher section + 1 tail Evoke section), got {list(_ent)}"
        # consistency of the decoupled dual-loss / stop-grad switches (all recommended on for the dual path; without T2 the front big window OOMs on O(N) BPTT).
        if bool(getattr(tc, "sf_evoke_teacher_front_window", False)):
            assert bool(getattr(tc, "sf_detach_history_between_chunks", False)), \
                "sf_evoke_teacher_front_window=true must also enable sf_detach_history_between_chunks (T2; otherwise the front big window does cross-section BPTT -> OOM)"
            assert bool(getattr(tc, "sf_return_full_rollout", False)), \
                "sf_evoke_teacher_front_window=true must also enable sf_return_full_rollout (the whole generated region is needed to slice front/tail)"
        # SP group size validation (world%G divisibility is checked against the actual world when train_evoke calls init_sequence_parallel).
        _g = int(getattr(tc, "sf_critic_sp_world_size", 1) or 1)
        assert _g >= 1, f"[SP] sf_critic_sp_world_size must be >=1, got {_g}"
        if _g > 1:
            assert getattr(mc, "real_score_arch", "evoke") == "evoke_teacher", \
                "[SP] sf_critic_sp_world_size>1 is only for the EvokeTeacher teacher/critic (real_score_arch=evoke_teacher)"
            assert bool(getattr(tc, "sf_evoke_teacher_front_window", False)), \
                "[SP] sf_critic_sp_world_size>1 targets splitting the front big-window long sequence, so sf_evoke_teacher_front_window must be on too"
            # the GAN discriminator does its own internal backward, bypassing the SP-group grad-shard all-reduce of critic engine.step ->
            #   SP-partials are never merged -> the two DP replicas diverge / the gradients are wrong. forbid the SP + GAN combination (belt and braces).
            assert not bool(getattr(tc, "is_use_gan", False)), \
                "[SP] sf_critic_sp_world_size>1 is mutually exclusive with is_use_gan (the GAN discriminator internal backward bypasses the SP-group grad-shard all-reduce)"
        # decouple-rollout: the G ranks in a group each run a different clip (effective bs x G), and the EvokeTeacher front section/critic
        #   score with SP rotating through owners. only meaningful on the SP-on + front-big-window path (the whole chain no-ops when off/SP-off, byte-identical); explicit fail-fast against misconfiguration.
        if bool(getattr(tc, "sf_decouple_rollout", False)):
            assert _g > 1, "[THROUGHPUT-B] sf_decouple_rollout=true requires sf_critic_sp_world_size>1 (SP on; otherwise decoupling is meaningless)"
            assert bool(getattr(tc, "sf_evoke_teacher_front_window", False)), \
                "[THROUGHPUT-B] sf_decouple_rollout=true must also enable sf_evoke_teacher_front_window (front big-window SP scoring is the only SP path of decouple)"
            _critic_steps = int(getattr(tc, "sf_critic_steps_per_student", 1) or 1)
            _critic_expected_bs = int(
                getattr(tc, "sf_critic_expected_global_batch_size", 0) or 0
            )
            assert 1 <= _critic_steps <= _g, (
                "[THROUGHPUT-B MULTI-CRITIC] sf_critic_steps_per_student must be in "
                f"[1,G={_g}], got {_critic_steps}"
            )
            assert _critic_expected_bs >= 0, (
                "[THROUGHPUT-B MULTI-CRITIC] sf_critic_expected_global_batch_size "
                f"must be >=0, got {_critic_expected_bs}"
            )
            if _critic_steps > 1:
                assert int(getattr(tc, "gradient_accumulation_steps", 1) or 1) == 1, (
                    "[THROUGHPUT-B MULTI-CRITIC] multiple critic optimizer steps currently only support "
                    "gradient_accumulation_steps=1"
                )
                assert int(getattr(tc, "dfake_gen_update_ratio", 1) or 1) == 1, (
                    "[THROUGHPUT-B MULTI-CRITIC] student-first multiple critic steps need a shared "
                    "warp rollout every round, so dfake_gen_update_ratio must be 1"
                )
                assert bool(getattr(tc, "no_visualize", False)), (
                    "[THROUGHPUT-B MULTI-CRITIC] to release the critic full-latent tensor logs, "
                    "the current implementation requires no_visualize=true"
                )
    # -- student-side parallelism fail-fast (one-time at startup, zero runtime cost) ---------------------
    # it guards against "one config change and it silently breaks". all no-ops when off.
    _stu_cp = bool(getattr(tc, "sf_student_chunk_parallel", False))
    _stu_gu = int(getattr(tc, "sf_student_sp_ulysses", 1) or 1)
    assert _stu_gu >= 1, f"[STU-SP] sf_student_sp_ulysses must be >=1 (1=off), got {_stu_gu}"
    if _stu_gu > 1:
        # 16: mechanism B depends on the subgroup split of mechanism A (G_p=1 goes through the same code)
        assert _stu_cp, "[STU-SP] sf_student_sp_ulysses>1 must also enable sf_student_chunk_parallel (B depends on the second-level decomposition of A)"
    if _stu_cp:
        _sg = int(getattr(tc, "sf_critic_sp_world_size", 1) or 1)
        # 1: topology
        assert _sg > 1, "[STU-SP] sf_student_chunk_parallel=true requires sf_critic_sp_world_size>1 (the student reuses the same SP group for the second-level decomposition)"
        assert _sg % _stu_gu == 0, f"[STU-SP] G={_sg} must be divisible by G_u={_stu_gu} (G_p = G//G_u)"
        assert _sg // _stu_gu >= 1, f"[STU-SP] G_p = {_sg}//{_stu_gu} must be >=1"
        # 2: the group must share one clip -- otherwise the per-rank section gradients are stitched from different clips
        assert not bool(getattr(tc, "sf_decouple_rollout", False)), \
            "[STU-SP] sf_student_chunk_parallel is mutually exclusive with sf_decouple_rollout (mechanism A requires one clip inside the group)"
        # 3: the foundation of mechanism A -- sections are disconnected. without it BPTT crosses into a section "this rank never built a graph for" => **silently dropped gradients**
        assert bool(getattr(tc, "sf_detach_history_between_chunks", False)), \
            "[STU-SP] sf_student_chunk_parallel=true must also enable sf_detach_history_between_chunks (otherwise BPTT crosses into sections this rank never graphed -> silently dropped gradients)"
        # 4: full-window supervision (start_gradient_section_index=0), otherwise the section indices and the ownership function disagree
        assert bool(getattr(tc, "sf_evoke_teacher_front_window", False)), \
            "[STU-SP] sf_student_chunk_parallel=true must also enable sf_evoke_teacher_front_window (all sections graphed, consistent section indexing)"
        # 5: the only class that breaks "gradients decompose per section" -- forward values / normalization denominators that depend on "which sections carry gradient".
        #    is_consistency_align: the append gate includes should_compute_grad and .mean() divides by a varying length.
        #    the other four also eat pred_video (the same chunk-sharded graph); if enabled they must be x G together with dmd_loss => the scaling insertion point would miss them.
        #    is_use_reward_model: the reward_score at is **multiplicative**, so only turning it off is safe.
        for _k in ("is_consistency_align", "is_mean_var_regular", "is_chunk_mean_var_regular",
                   "is_smoothness_loss", "is_use_reward_model"):
            assert not bool(getattr(tc, _k, False)), \
                f"[STU-SP] sf_student_chunk_parallel=true requires {_k}=false (§7.2 scaling insertion point / §2.1 decomposability)"
        # 6: the GAN discriminator does its own internal backward; vae_decode introduces a cross-section global term
        assert not bool(getattr(tc, "is_use_gan", False)), "[STU-SP] mutually exclusive with is_use_gan"
        assert not bool(getattr(tc, "is_dmd_vae_decode", False)), "[STU-SP] mutually exclusive with is_dmd_vae_decode"
        # 7: it replaces output with pyramid_stage_videos and changes the frame accounting, conflicting with the 189-frame convention of sf_return_full_rollout
        assert not bool(getattr(tc, "is_multi_pyramid_stage_backward_simulated", False)), \
            "[STU-SP] mutually exclusive with is_multi_pyramid_stage_backward_simulated (it changes the output convention and the frame accounting)"
        # 8: these two insert a WORLD broadcast into the chunk loop, breaking the time-shared communication domains
        assert not bool(getattr(tc, "sf_score_window_jitter", False)), \
            "[STU-SP] sf_score_window_jitter inserts a WORLD broadcast into the chunk loop (breaking the time-shared communication domains)"
        assert not bool(getattr(tc, "is_amplify_first_chunk", False)), \
            "[STU-SP] is_amplify_first_chunk inserts a WORLD broadcast into the chunk loop (breaking the time-shared communication domains)"
        # 9: a hard numerical precondition (not just "unsupported for now"): enable_backward_allreduce=False makes engine.backward skip
        #    _scale_loss_by_gas => with GAS>1 the loss is divided by GAS one time too few.
        assert int(getattr(tc, "gradient_accumulation_steps", 1) or 1) == 1, \
            "the staggered allreduce requires gradient_accumulation_steps=1 (enable_backward_allreduce=False skips _scale_loss_by_gas)"
        # 10: guarantees image_latents always = GT prefix frame 0 => latents_prefix (the only history channel that is not detached) does not carry a graph across sections
        assert int(getattr(tc, "rollout_prefix_sections", 0) or 0) >= 1, \
            "[STU-SP] requires rollout_prefix_sections>=1 (otherwise latents_prefix carries a graph across sections and the sections are no longer disconnected)"
        # 11: the curriculum makes N vary with step => the G_p bucketing of the ownership function is inconsistent across steps inside one group
        assert not bool(getattr(tc, "sf_curriculum_enabled", False)), \
            "[STU-SP] sf_curriculum_enabled is not supported yet (N varies with step => section ownership drifts across steps)"
        # 12: must use the dedicated staggered ds json (overlap_comm:false), not the non-SP ZeRO-2 config
        _gds = str(getattr(tc, "dmd_generator_deepspeed_config", "") or "")
        assert _gds.endswith("_gen_sp.json"), \
            f"[STU-SP §9.1] dmd_generator_deepspeed_config must point at *_gen_sp.json (overlap_comm:false + staggered), got {_gds!r}"
        if _stu_gu > 1:
            # 15: mechanism B requires attention to be dense, mask-free and full-sequence -- each of these brings absolute-position slicing that would need changes
            assert not bool(getattr(tc, "restrict_self_attn", False)), \
                "sf_student_sp_ulysses>1 requires restrict_self_attn=false (the history/noise chunked branch carries absolute-position slicing)"
            assert not bool(getattr(tc, "is_amplify_history", False)), \
                "sf_student_sp_ulysses>1 requires is_amplify_history=false (history_seq_len at processor:305 goes negative after sharding, currently blocked only by a >0 coincidence)"
            # Both of these live under model_config, not under args / training_config. Reading them
            # off the wrong object made getattr fall back to False, so the two asserts always passed.
            assert not bool(getattr(getattr(mc, "camera_control", None), "enabled", False)), \
                "sf_student_sp_ulysses>1 requires camera_control.enabled=false (cam slots are absolute slots)"
            # Ulysses shards the GENERATOR forward, so the predicate must be the plucker value the generator is actually
            # built with. Testing the raw geo_warp_plucker_enabled would over-block: every evoke_teacher config pairs
            # geo_warp_plucker_enabled=true (teacher builds plucker, cam-pose data path on) with
            # generator_geo_warp_plucker_enabled=false (no plucker submodule on the student), so the assert would fail
            # configs whose student has no plucker at all -- and a false alarm here gets the whole gate ignored as noise.
            _geo_mc = getattr(mc, "geometric_state", None)
            _gen_plk_ov = getattr(_geo_mc, "generator_geo_warp_plucker_enabled", None)
            _gen_plk_eff = (bool(getattr(_geo_mc, "geo_warp_plucker_enabled", False))
                            if _gen_plk_ov is None else bool(_gen_plk_ov))
            assert not _gen_plk_eff, \
                "sf_student_sp_ulysses>1 requires the generator-side plucker to be off " \
                "(generator_geo_warp_plucker_enabled=false, or geo_warp_plucker_enabled=false when it is unset) " \
                "-- the generator-side Plucker is added per absolute noise slot"
            # 14/NAViT: num_attention_heads and enable_navit are only known at runtime => asserted inside transformer.forward

    if float(getattr(tc, "sf_geo_reg_weight", 0.0) or 0.0) > 0.0:
        assert getattr(mc, "real_score_arch", "evoke") == "evoke_teacher", \
            "[GEOREG] sf_geo_reg_weight>0 only supports real_score_arch=evoke_teacher (the branch is gated on is_evoke_teacher_score, otherwise it silently does nothing)"
    if getattr(mc, "real_score_arch", "evoke") != "evoke_teacher":
        return
    et = mc.evoke_teacher
    assert et.high_dir and et.low_dir, "[SF10S] evoke_teacher.high_dir/low_dir are required (merged directories)"
    assert tc.is_train_dmd, "[SF10S] real_score_arch=evoke_teacher is only for DMD training"
    assert tc.is_enable_stage2, "[SF10S] only the stage2 path is supported (the stage1 rollout was never reworked for prefix/segmented prompts)"
    assert not tc.is_use_gan, "[SF10S] the evoke_teacher path does not support GAN (the wrapper has no gan_mode)"
    assert not tc.is_use_gt_history, "[SF10S] mutually exclusive with gt-history single-section distillation (the prefix is injected via rollout)"
    assert not tc.is_use_reward_model and not tc.is_dmd_vae_decode, \
        "[SF10S] the evoke_teacher path does not support reward/vae_decode"
    # tail-chunk warp ported into the evoke_teacher path.
    #   the teacher stays nocam, but the student rollout may render warp on its tail sections (SFWarpRollout),
    #   while the teacher side stays nocam/i2v (scoring does not consume warp). with warp off all asserts below no-op.
    if bool(tc.use_geometric_state):
        # warp-on needs sf_rollout_out (the shared rollout) to be populated, otherwise the critic silently re-rolls warp-free and mismatches.
        #   both paths populate it: (a) the curriculum windowed scoring path (sf_windowed_score, the is_evoke_teacher_score and sf_curriculum_enabled block);
        # (b) the front-window path (sf_evoke_teacher_front_window block d; the train_evoke _sf_rollout_shared gate only looks at
        #   _sf_evoke_teacher and use_geometric_state, independent of the curriculum). so it is relaxed to "curriculum OR front-window", either one suffices.
        #   only the full-sequence else path ("no curriculum AND no front-window") is unwired -> it still fail-fasts.
        assert bool(getattr(tc, "sf_curriculum_enabled", False)) or bool(getattr(tc, "sf_evoke_teacher_front_window", False)), \
            "[LW-WARP] use_geometric_state=true requires sf_curriculum_enabled=true or sf_evoke_teacher_front_window=true (" \
            "both populate the shared rollout; the full-sequence path with neither curriculum nor front-window is unwired and the critic would silently go warp-free)"
        # under warp the critic must reuse the gen warp rollout (an independent re-roll has no pose and cannot render warp, so it would mismatch the gen-side conditioning)
        # -> sharing requires a gen rollout on every step -> dfake_gen_update_ratio=1.
        assert int(tc.dfake_gen_update_ratio) == 1, \
            "[LW-WARP] use_geometric_state=true requires dfake_gen_update_ratio=1 (the critic reuses the gen warp rollout; " \
            "with dfake>1 the critic-only steps have no shared rollout -> they fall back to a warp-free re-roll and mismatch the gen conditioning)"
        assert not bool(getattr(tc, "dmd_is_low_vram_mode", False)), \
            "[LW-WARP] use_geometric_state=true requires a vae (SFWarpRollout decode/encode); dmd_is_low_vram_mode sets vae=None and is incompatible"
        _geo_cfg = getattr(mc, "geometric_state", None)
        assert _geo_cfg is not None and bool(getattr(_geo_cfg, "enabled", False)), \
            "[LW-WARP] use_geometric_state=true requires model_config.geometric_state.enabled=true (SFWarpRollout reads the cloud_warp/sigma config)"
        # sf_warp_tail_chunks: None=default (W+2); 0=render warp throughout (K=N, always >=W so no check needed); >0=explicit K, must be >= the deepest curriculum W.
        _tail_v = getattr(tc, "sf_warp_tail_chunks", None)
        if _tail_v is not None and int(_tail_v) > 0:
            _Ws = [int(e[1]) for e in (list(getattr(tc, "sf_curriculum_schedule", []) or []))
                   if hasattr(e, "__len__") and len(e) >= 2]
            if _Ws:
                assert int(_tail_v) >= max(_Ws), \
                    f"[LW-WARP] sf_warp_tail_chunks({_tail_v}) must be >= the deepest curriculum W({max(_Ws)}) (the scoring window must land inside the warp-ON tail)"
    # -- NOTE: GT window supervision regularizer fail-fast (weight=0 all no-op -> bit-identical) --
    # shared frozen expert base: must be a per-expert offload setup (dual expert + offload),
    #   otherwise there is no swapped-out 28GB on the host to share -> enabling it does nothing, so fail-fast against misconfiguration.
    #   WARNING: must sit at the **function top level**: the whole SP/THROUGHPUT-B block above lives inside `if dual_teacher.enabled:`,
    #     and dmd-final happens to have enabled=false -> placed inside it would never run (already stepped on).
    if bool(getattr(tc, "sf_evoke_teacher_shared_host_base", False)):
        assert getattr(mc, "real_score_arch", "evoke") == "evoke_teacher", \
            "[SHARED-BASE] sf_evoke_teacher_shared_host_base is only for the EvokeTeacher teacher (real_score_arch=evoke_teacher)"
        assert getattr(getattr(mc, "evoke_teacher", None), "single_expert", None) is None, \
            "[SHARED-BASE] sf_evoke_teacher_shared_host_base needs dual experts (single_expert=null); a single expert has no swapped-out copy to share"
        assert bool(getattr(getattr(mc, "evoke_teacher", None), "offload", False)), \
            "[SHARED-BASE] sf_evoke_teacher_shared_host_base needs evoke_teacher.offload=true (the per-expert offload switch)"
        print("[SHARED-BASE] validation passed: sharing the offloaded frozen expert base within the node (saves ~28GB per rank; "
              "swap-out becomes zero-copy). it is backed by /dev/shm, so the container SharedMemory must be large enough (256Gi for formal jobs).", flush=True)
    _wrw = float(getattr(tc, "sf_geo_reg_weight", 0.0) or 0.0)
    if _wrw > 0.0:
        assert bool(tc.use_geometric_state), \
            "[GEOREG] sf_geo_reg_weight>0 requires use_geometric_state=true (the regularizer branch reuses SFWarpRollout to render GT warp conditioning)"
        # sf_skip_full_encode is driven by dual_teacher.enabled (a train_evoke materialize kwarg); enabled=true -> sf_gt_latents=None.
        # DMD-FINAL already removed the camera teacher -> dual_teacher.enabled=false -> sf_gt_latents is available, so the two changes are mutually consistent.
        assert not bool(getattr(getattr(mc, "dual_teacher", None), "enabled", False)), \
            "[GEOREG] requires the full-clip GT latents (sf_gt_latents); with dual_teacher.enabled=true the sf_skip_full_encode path leaves it None"
        assert int(getattr(tc, "sf_geo_reg_every_k", 1)) >= 1, "[GEOREG] sf_geo_reg_every_k must be >=1"
        _wr_tmn = int(getattr(tc, "sf_geo_reg_t_min", 666))
        _wr_tmx = int(getattr(tc, "sf_geo_reg_t_max", 899))
        # lower bound 20->1: in-stage sigma x1000=1 -> feeds the model model-t=743.5 (=the stage0 high-noise floor). to make GEO span
        #   the full stage0 high-noise band [743.5,999] you need in-stage [1,999] (in-stage != global: in-stage is this stage's [0,1] progress, nonlinearly mapped to model-t).
        assert 1 <= _wr_tmn < _wr_tmx <= 999, \
            f"[GEOREG] illegal t band: need 1<=t_min<t_max<=999 (in-stage0 sigma x1000 semantics), got [{_wr_tmn},{_wr_tmx}]"
        print(f"[GEOREG] validation passed: lambda={_wrw} every_k={int(getattr(tc, 'sf_geo_reg_every_k', 1))} "
              f"band=in-stage0 sigma x1000 in [{_wr_tmn},{_wr_tmx}] (in-stage semantics, not the global t axis; no interaction with the expert routing band)", flush=True)
    assert float(tc.dmd_timestep_shift) == 5.0 and not tc.use_dynamic_shifting, \
        "[SF10S] the teacher t<->sigma mapping is locked to shift=5.0 / no dynamic shifting"
    assert not mc.train_norm_layers, "[SF10S] train_norm_layers would unfreeze teacher params inside the wrapper, must be false"
    assert mc.critic_lora_name_or_path is None, "[SF10S] critic reloading uses evoke-PEFT semantics, unsupported by evoke_teacher"
    assert not tc.enable_npu_flash_attention, "[SF10S] the wrapper has no npu flash attention method"
    assert not tc.enable_xformers_memory_efficient_attention, "[SF10S] the wrapper has no xformers method (review S2)"
    assert not tc.is_enable_cold_start, "[SF10S] cold-start makes the rollout section count < N, conflicting with the fixed prefix/ncif (review S2)"
    assert not tc.is_decouple_dmd, "[SF10S] decoupled DMD was never adapted to the evoke_teacher branch (review S2)"
    assert int(tc.train_batch_size) == 1, "[SF10S] the data mode is limited to B=1 (materialize section-mapping constraint)"
    assert args.data_config.use_stage1_dataset and args.data_config.use_multi_dataset, \
        "[SF10S] must use the use_stage1_dataset + use_multi_dataset online data path (it produces the sf_* keys)"
    assert not tc.is_mean_var_regular and not tc.is_chunk_mean_var_regular, \
        "[SF10S] the mean/var regularizer does not mask the prefix, so the evoke_teacher branch does not support it yet (review S5)"
    assert tc.resume_from_checkpoint is None, \
        "[SF10S] accelerate save_state resume is unsupported (the save/load hooks fail-fast on the wrapper)"
    win = tc.latent_window_size
    if not isinstance(win, (int, float)):  # list/tuple/OmegaConf ListConfig
        win = win[0]
    win = int(win)
    P = int(tc.rollout_prefix_sections)
    assert P >= 1, "[SF10S] needs at least 1 GT prefix chunk (same source as the teacher i2v first frame)"
    # data side: the full-rollout interleave mode must be on (it produces prefix/y/segment)
    assert args.data_config.use_full_rollout_interleave, \
        "[SF10S] data_config.use_full_rollout_interleave must be true"
    if not bool(getattr(tc, "sf_curriculum_enabled", False)):
        # -- fixed-N path --
        n_sec = int(tc.dmd_num_latent_sections_min)
        assert n_sec == int(tc.dmd_num_latent_sections_max), "[SF10S] the fixed-N path uses a fixed section count"
        assert int(tc.num_critic_input_frames) == n_sec * win, (
            f"[SF10S] num_critic_input_frames must = the total generated frames {n_sec}*{win} (review M1: larger trips the rollout assert, "
            f"smaller shrinks the loss window), got {tc.num_critic_input_frames}")
        expected_frames = ((P + n_sec) * win - 1) * 4 + 1
        assert int(args.data_config.num_frames) == expected_frames, (
            f"[SF10S] num_frames should be the {expected_frames} implied by (P+N)*win (P={P}, "
            f"N={n_sec}, win={win}), got {args.data_config.num_frames}")
        # -- NOTE: fail-fast (0=OFF all no-op -> bit-identical) --
        _wc = int(getattr(tc, "sf_score_window_chunks", 0) or 0)
        if _wc > 0:
            # precondition ①: the front-window path (both the generator and the critic window slicing hang off it; nothing is wired for non-front-window).
            assert bool(getattr(tc, "sf_evoke_teacher_front_window", False)), \
                "[SF-WINDOW] sf_score_window_chunks>0 requires sf_evoke_teacher_front_window=true (window slicing hangs off the front-window path)"
            # precondition ②: the shared rollout must be in effect (the critic eats the detached full rollout of gen and then slices the window; otherwise a critic re-roll has no window bookkeeping).
            assert bool(tc.use_geometric_state), \
                "[SF-WINDOW] sf_score_window_chunks>0 requires use_geometric_state=true (the critic reuses the full gen rollout and then slices the window)"
            # precondition ③: the window must be able to exclude g1 and stay inside the generated region -> 2 <= wc <= N-1 (leaving g1 out and at least 1 possible start).
            assert 2 <= _wc <= n_sec - 1, (
                f"[SF-WINDOW] sf_score_window_chunks({_wc}) needs 2<=wc<=N-1 (N={n_sec}; leaves g1 uncovered + at least 1 window position)")
            # precondition ④ [review R4]: the windowed critic only supports a camera-free teacher (the critic-side _hb_absent_c assert + the generator-side mask
            #   would go out of bounds because front only covers N-1 chunks when hb is present). intercepted at config time so a misconfig does not crash at runtime.
            assert not bool(getattr(getattr(mc, "dual_teacher", None), "enabled", False)), \
                "[SF-WINDOW] sf_score_window_chunks>0 requires dual_teacher.enabled=false (the windowed critic only supports the camera-free-teacher option-A path)"
            # precondition ⑤ : the asymmetry tilt in [0, wc-1] (the low-side slide-off amount (wc-1)-tilt must be >=0).
            _tilt = int(getattr(tc, "sf_score_window_tail_tilt", 0) or 0)
            assert 0 <= _tilt <= _wc - 1, \
                f"[SF-WINDOW] sf_score_window_tail_tilt({_tilt}) needs 0<=tilt<=wc-1({_wc - 1})"
            if _tilt > 0:
                print(f"[SF-WINDOW] tail tilt={_tilt}: tail chunk coverage > head (second-half mean ~= first-half x{1 + 0.15 * _tilt:.2f} order)", flush=True)
            print(f"[SF-WINDOW] validation passed: critic forward+backward window={_wc} chunks ({(1 + _wc) * win} frames total "
                  f"=[prefix {win}|window {_wc * win}]); start s in [2,{n_sec - _wc + 1}] (g1 excluded); "
                  f"teacher still all {(1 + n_sec) * win} frames, the gen gradient covers the window only", flush=True)
    else:
        # -- deep curriculum: N grows via the schedule; data is materialized for max_N,
        #   rollout goes N deep (the front N-W sections detached), and the teacher only scores the last W sections in a window (v2v self-anchored, RoPE re-based from 0) -> scoring memory
        #   is O(W), independent of N. wiring: train_evoke.py gate widened to _sf_any; the utils_evoke_post.py is_evoke_teacher_score block (gen+critic).
        #   WARNING: W<N is supported now (gradient_mask changed to anchor-relative [:,:,1:], matching the gradients of the last W rollout sections); the OOM wall is gone thanks to windowing,
        #   so the new N ceiling = rollout memory + source clip length (not statically validated, felt out via smoke).
        sched = list(getattr(tc, "sf_curriculum_schedule", []) or [])
        assert len(sched) >= 1, "[LW-CUR] sf_curriculum_schedule must not be empty when sf_curriculum_enabled"
        for i, ent in enumerate(sched):
            assert len(ent) == 3, f"[LW-CUR] curriculum[{i}] must be [N, W, step_budget], got {ent}"
            N_i, W_i, b_i = int(ent[0]), int(ent[1]), int(ent[2])
            assert 1 <= W_i <= N_i, f"[LW-CUR] curriculum[{i}] needs 1 <= W({W_i}) <= N({N_i})"
            assert b_i >= 1, f"[LW-CUR] curriculum[{i}] step_budget must be >=1, got {b_i}"
        max_N = max(int(e[0]) for e in sched)
        # num_frames must cover the deepest N (data is materialized for max_N, rollout requests the current N)
        need_frames = ((P + max_N) * win - 1) * 4 + 1
        assert int(args.data_config.num_frames) >= need_frames, (
            f"[LW-CUR] num_frames({args.data_config.num_frames}) must be >= {need_frames}, which covers the deepest N={max_N}"
            f"(P={P}, win={win})")

    # -- NOTE: weight-level warm-start + student freeze period fail-fast --
    _ws_d = getattr(tc, "sf_warmstart_dir", None)
    _ws_k = int(getattr(tc, "sf_gen_freeze_steps", 0) or 0)
    assert _ws_k >= 0, f"[LW-WARMSTART] sf_gen_freeze_steps must be >=0, got {_ws_k}"
    if _ws_d:
        assert os.path.isdir(_ws_d), f"[LW-WARMSTART] sf_warmstart_dir does not exist: {_ws_d}"
        # all three weight pieces must be present: missing any one is a "half inheritance", harder to debug than no inheritance at all (especially a missing critic LoRA).
        _ws_co = bool(getattr(tc, "sf_warmstart_critic_only", False))
        _ws_cri = os.path.join(_ws_d, "critic", "critic_evoke_teacher_lora.safetensors")
        if not _ws_co:
            _ws_gen_ok = any(os.path.exists(os.path.join(_ws_d, p)) for p in
                             ("pytorch_lora_weights.safetensors", "weights/lora.safetensors"))
            _ws_mem_ok = any(os.path.exists(os.path.join(_ws_d, p)) for p in
                             ("transformer_partial.pth", "weights/memory.pth"))
            assert _ws_gen_ok, f"[LW-WARMSTART] missing generator LoRA (pytorch_lora_weights.safetensors / weights/lora.safetensors): {_ws_d}"
            assert _ws_mem_ok, f"[LW-WARMSTART] missing memory patch (transformer_partial.pth / weights/memory.pth): {_ws_d}"
        else:
            # critic_only: the generator/memory patch must come from a merged starting point => transformer_model_name_or_path has to point at
            #   an already-merged directory (otherwise the student starts from base while the critic loaded a step-N one -- the two sides are from different eras).
            assert "merged" in str(mc.transformer_model_name_or_path or "").lower() or \
                   os.path.isdir(os.path.join(str(mc.transformer_model_name_or_path or ""), "transformer")), (
                "[LW-WARMSTART] critic_only=true requires transformer_model_name_or_path to point at an already-merged starting directory "
                f"(currently {mc.transformer_model_name_or_path})")
        assert os.path.exists(_ws_cri), (
            f"[LW-WARMSTART] missing critic LoRA {_ws_cri} -- the evoke_teacher path must load it back, otherwise the critic starts from scratch and "
            f"the fake-score first hands the student a stretch of wrong gradient")
        assert tc.resume_from_checkpoint is None, (
            "[LW-WARMSTART] mutually exclusive with resume_from_checkpoint: this switch is a **weight-level** warm-start (that ckpt has no "
            "optimizer/RNG/scheduler), and accelerate resume is already forbidden on the SF10S path")
        print(f"[LW-WARMSTART] validation passed: dir={_ws_d}; critic_only={_ws_co}"
              f"{' => critic LoRA only (generator+memory patch come from the merged start ' + str(mc.transformer_model_name_or_path) + ')' if _ws_co else ' => all three of generator LoRA/memory patch/critic LoRA'}; "
              f"student frozen for {_ws_k} steps (critic only), then joint training", flush=True)
    if _ws_k > 0:
        assert tc.is_train_dmd, "[LW-WARMSTART] sf_gen_freeze_steps>0 only makes sense under DMD training (the freeze period relies on critic updates)"
        assert int(tc.dfake_gen_update_ratio) == 1, (
            f"[LW-WARMSTART] sf_gen_freeze_steps>0 requires dfake_gen_update_ratio=1 (otherwise some steps in the freeze period do not update the critic either), "
            f"got {tc.dfake_gen_update_ratio}")

    # -- NOTE: student i2v/v2v mixed-training fail-fast (ratio=0 -> all no-op, legacy path bit-identical) --
    _i2v_r = float(getattr(tc, "sf_i2v_ratio", 0.0) or 0.0)
    assert 0.0 <= _i2v_r <= 1.0, f"[LW-I2V] sf_i2v_ratio must be in [0,1], got {_i2v_r}"
    _i2v_on = int(getattr(tc, "sf_i2v_prefix_latent_frames", 0) or 0) > 0
    assert _i2v_on or _i2v_r == 0.0, (
        "[LW-I2V] sf_i2v_ratio>0 requires sf_i2v_prefix_latent_frames>0 (the latter is the master switch of the i2v path)")
    if _i2v_on:
        _i2v_pf = int(getattr(tc, "sf_i2v_prefix_latent_frames", 1) or 1)
        assert 1 <= _i2v_pf < P * win, (
            f"[LW-I2V] sf_i2v_prefix_latent_frames({_i2v_pf}) must be in [1, P*win)={1}..{P * win - 1}"
            f"(=P*win is v2v itself; P={P}, win={win})")
        assert getattr(tc, "sf_i2v_mode_scope", "group") in ("group", "step"), \
            f"[LW-I2V] sf_i2v_mode_scope must be group|step, got {getattr(tc, 'sf_i2v_mode_scope', None)}"
        # only static_repeat is allowed: the latent in the 1x slot must be a non-first-frame (continuation)
        #   distribution, for i2v and v2v alike. The "iframe" mode makes 1x reuse frame 0 of the prefix, which is an
        #   I-frame distribution and violates that directly:
        #     encode(1 frame)[0]   = I-frame distribution (no causal context)
        #     encode(33 frames)[8] = continuation distribution  <- the only thing this slot has ever held
        #       (v2v k=0 is prefix[8]; v2v/i2v k>=1 is pred_{k-1}[8]; i2v k=0 is VAE(img.repeat(33))[-1:])
        #   It saves 33 frames of VAE (2% of prep), nowhere near worth trading the train/inference convention for.
        assert getattr(tc, "sf_i2v_hist_latent_mode", "static_repeat") == "static_repeat", (
            f"[LW-I2V] sf_i2v_hist_latent_mode only allows static_repeat, got "
            f"{getattr(tc, 'sf_i2v_hist_latent_mode', None)} -- 'iframe' would turn the history 1x slot into "
            f"an **I-frame distribution**, while that slot must be a **continuation distribution** in any non-degenerate case (user hard constraint); "
            f"it only saves 33 frames of VAE (2% of prep), which is not worth a train/inference mismatch")
        # precondition ①: only wired on the front-window option-A path (K_tail=0, pred includes the prefix and aligns frame-by-frame with teacher_y).
        assert bool(getattr(tc, "sf_evoke_teacher_front_window", False)), \
            "[LW-I2V] requires sf_evoke_teacher_front_window=true (i2v frame accounting hangs off the front-window path)"
        assert not bool(getattr(getattr(mc, "dual_teacher", None), "enabled", False)), \
            "[LW-I2V] requires dual_teacher.enabled=false (the K_tail=1 front/tail split would be off by one frame when T_lat is not 0 mod win)"
        # precondition ②: the rollout must return the whole [prefix|generated region] (the non-full branch aligns its output window to multiples of win, and P_lat=1 would silently misalign and drop frames).
        assert bool(getattr(tc, "sf_return_full_rollout", False)), \
            "[LW-I2V] requires sf_return_full_rollout=true (otherwise the output window aligns to multiples of win -> silent misalignment with a 1-frame prefix)"
        # precondition ③: the curriculum / sliding window / jitter paths all account by section or by multiples of win and are incompatible with a 1-frame prefix.
        assert not bool(getattr(tc, "sf_curriculum_enabled", False)), \
            "[LW-I2V] requires sf_curriculum_enabled=false (the curriculum accounts by section)"
        assert int(getattr(tc, "sf_score_window_chunks", 0) or 0) == 0, \
            "[LW-I2V] requires sf_score_window_chunks=0 (the sliding-window start is _sf_P+(s-1)*win, which does not land on a chunk boundary with a 1-frame prefix)"
        assert not bool(getattr(tc, "sf_score_window_jitter", False)), \
            "[LW-I2V] requires sf_score_window_jitter=false (borrowing the sacrificial frame at k=0 requires the prefix to be >= 1 whole section)"
        # precondition ④: these two have a hard `% latent_window_size == 0` assert when T_lat is not 0 mod win (utils_evoke_post:3690/5016).
        assert not bool(tc.is_smoothness_loss) and not bool(tc.is_dmd_vae_decode), \
            "[LW-I2V] requires is_smoothness_loss=false and is_dmd_vae_decode=false (both contain a hard %win==0 assert)"
        # precondition ⑤ : the i2v history mid/long **stay all-zero** (k=0 all zero, k=1 partly zero, k>=2 real latents),
        #   matching the i2v inference layout of `pipeline_evoke_diffusers` frame by frame (history_latents = [zeros(18)|fake_image_latents]).
        #   but geo_invisible_history_noise=true makes **inference** replace the all-zero mid/long with sigma_inv*randn
        #   (pipeline_evoke.py:_geo_maybe_noise_invisible_history) => train(zero)/infer(noise) mismatch, and silently so.
        #   using that inference path would require changing the training side to noise as well (train_evoke.py, the same formula); explicitly not doing that => blocked here.
        assert not bool(getattr(getattr(mc, "geometric_state", None), "geo_invisible_history_noise", False)), (
            "[LW-I2V] with the i2v path on, geometric_state.geo_invisible_history_noise must be false -- the training-side i2v "
            "mid/long stay all-zero (matching the i2v inference of pipeline_evoke_diffusers); enabling it makes inference use sigma_inv*randn => train/infer mismatch")
        # precondition ⑤b [four-region consistency]: the mode dispatch (_sf_i2v_active / _sf_i2v_hist_latent) is computed **inside** the
        #   `if TRAIN_GENERATOR:` block of train_evoke, while the critic call reads them **outside** it => on critic-only steps (which only exist when ratio>1)
        #   they would take the defaults False/None, so the critic skips the frame-0 replacement while the generator did it -> the teacher input and the critic
        #   query come from different sources, and **silently** (no error, just mis-scoring). either block it here or hoist the dispatch out of the block; blocking is chosen.
        #   note: with use_geometric_state=true another rule already forces ratio==1; this one is the fallback for when that is off.
        assert int(getattr(tc, "dfake_gen_update_ratio", 1)) == 1, (
            f"[LW-I2V] with the i2v path on, dfake_gen_update_ratio must be 1, got "
            f"{getattr(tc, 'dfake_gen_update_ratio', None)} -- >1 produces critic-only steps, and on those steps "
            f"sf_i2v_* never goes through the mode dispatch => a silent four-region mismatch (the critic call site in train_evoke.py has the same assert as a fallback)")
        # precondition: the sf_prompt_embeds_list of the image-only branch is **the same tensor object repeated N
        #   times**, and sf_score_prompt_embeds = prompt_embeds.unsqueeze(1) **shares storage** with it (see
        #   online_materialize.materialize_i2v_image_only), while caption dropout in train_evoke is an in-place
        #   `prompt_embeds[dropout_mask] = 0`. One draw would therefore zero both the 20 student section prompts and
        #   the teacher/critic scoring prompt, whereas the v2v path's sf_score_prompt_embeds is a torch.stack copy and
        #   only affects the student. The two paths have asymmetric dropout semantics, so the aliasing must be broken
        #   before dropout can be enabled.
        _i2v_cdp = float(getattr(args.data_config, "caption_dropout_p", 0.0) or 0.0)
        assert _i2v_cdp == 0.0, (
            f"[LW-I2V] with the i2v path on, caption_dropout_p must be 0, got {_i2v_cdp} "
            f"-- the section prompts of the image-only branch and the scoring prompt share storage, so an in-place dropout would zero the teacher conditioning too "
            f"(the v2v path would not) => the two paths are asymmetric and silent. to use dropout, clone in materialize_i2v_image_only first")
        # precondition ⑥: unmasking g1 must also be able to toggle the skip of mechanism A (they are paired, otherwise "covered but no gradient").
        if bool(getattr(tc, "sf_i2v_score_g1", False)):
            assert bool(getattr(tc, "sf_student_chunk_parallel", False)) or \
                bool(getattr(tc, "dmd_score_skip_first_chunk", False)), \
                "[LW-I2V] sf_i2v_score_g1=true only makes sense when mechanism A or skip_first_chunk is in effect"
        _T_i2v = _i2v_pf + n_sec * win      # the curriculum is asserted off above => n_sec is necessarily defined
        print(f"[LW-I2V] validation passed: i2v path ON (prefix_latent_frames={_i2v_pf}); "
              f"image-only samples always go i2v, video samples go i2v at ratio={_i2v_r}"
              f"(scope={getattr(tc, 'sf_i2v_mode_scope', 'group')}); "
              f"hist_latent={getattr(tc, 'sf_i2v_hist_latent_mode', 'static_repeat')} "
              f"score_g1={bool(getattr(tc, 'sf_i2v_score_g1', False))}; "
              f"i2v step scoring sequence = {_i2v_pf}+{n_sec}x{win} = {_T_i2v} latents "
              f"(v2v steps still {P * win}+{n_sec}x{win} = {(P + n_sec) * win}); num_frames={args.data_config.num_frames} unchanged",
              flush=True)


def validate_sf_evoke_config(args) -> None:
    """fail-fast validation for real_score_arch=evoke and sf_self_forcing
    (table 1② / review MUST-FIX#4).

    WARNING: independent of and mutually exclusive with validate_sf10s_evoke_teacher_config (that one early-returns when arch!=evoke_teacher; this one only handles evoke+sf).
    must be called explicitly at the train_evoke entry point, choosing one of the two by arch (OmegaConf skips __post_init__, see memory omegaconf-skips-post-init).
    it independently re-asserts the invariants of this path (it does not reuse the sf10s ones, since those never apply to evoke).
    """
    mc, tc, dc = args.model_config, args.training_config, args.data_config
    # only active when the backbone is evoke and sf_self_forcing is on; if either fails, nothing is validated (default path bit-identical).
    if getattr(mc, "real_score_arch", "evoke") == "evoke_teacher":
        return
    if not bool(getattr(tc, "sf_self_forcing", False)):
        return

    win = tc.latent_window_size
    if not isinstance(win, (int, float)):  # list/tuple/OmegaConf ListConfig
        win = win[0]
    win = int(win)

    assert tc.is_train_dmd, "[SF-EVOKE] sf_self_forcing is only for DMD training"
    assert tc.is_enable_stage2, "[SF-EVOKE] only the stage2 pyramid path is supported (the stage1 rollout was never reworked for prefix/segmentation)"
    assert dc.use_full_rollout_interleave, "[SF-EVOKE] data_config.use_full_rollout_interleave must be true"
    assert int(tc.train_batch_size) == 1, "[SF-EVOKE] the data mode is limited to B=1 (materialize section-mapping constraint)"
    assert not tc.is_use_gt_history, "[SF-EVOKE] mutually exclusive with gt-history single-section distillation (the prefix is injected via rollout)"
    assert not tc.is_use_gan, "[SF-EVOKE] v1 does not enable GAN (F4; deferred)"
    assert not tc.is_use_reward_model and not tc.is_dmd_vae_decode, "[SF-EVOKE] reward/dmd_vae_decode not supported"
    assert float(tc.dmd_timestep_shift) == 5.0 and not tc.use_dynamic_shifting, \
        "[SF-EVOKE] the teacher t<->sigma mapping is locked to shift=5.0 / no dynamic shifting"
    assert int(tc.rollout_prefix_sections) >= 1, "[SF-EVOKE] needs >=1 GT prefix chunk (i2v anchor + warp seed from the same source)"
    # teacher/critic backbone carries no plucker weights -> the geo plucker must be off when building the teacher.
    _geo = getattr(mc, "geometric_state", None)
    if _geo is not None and bool(getattr(_geo, "enabled", False)):
        assert not bool(getattr(_geo, "geo_warp_plucker_enabled", False)), \
            "[SF-EVOKE] the Evoke-Base teacher has no plucker weights, geo_warp_plucker_enabled must be false"
    # teacher/critic warp: false=strip before scoring (for warp-free teachers such as Evoke-Base); true=keep warp while scoring
    # (for warp-native teachers such as warp-5600, where the camera-following restoring force needs the teacher to see warp; wired in v1.1).
    # v1.1 limitation: critic and teacher must be on the same side (critic=None follows the teacher).
    _t_warp = bool(getattr(tc, "sf_teacher_warp", False))
    _c_warp = getattr(tc, "sf_critic_warp", None)
    assert _c_warp is None or bool(_c_warp) == _t_warp, \
        "[SF-EVOKE] v1.1 requires the critic warp to be on the same side as the teacher (sf_critic_warp=None means follow)"
    if _t_warp:
        assert bool(getattr(tc, "use_geometric_state", False)), \
            "[SF-EVOKE] sf_teacher_warp=true requires use_geometric_state=true (only then does the scoring window have a warp tier)"
    # preconditions for the alternating misaligned scoring window
    if bool(getattr(tc, "sf_score_window_jitter", False)):
        assert bool(getattr(tc, "sf_score_skip_first_latent", False)), \
            "[SF-JITTER] requires sf_score_skip_first_latent=true (the slot-0 mask is the protection against the sacrificial frame / I-frame mismatch)"
        assert not _t_warp, \
            "[SF-JITTER] only strip scoring is supported (sf_teacher_warp=false): aligning the left-shifted window with the warp tier is not implemented"
        _sched = getattr(tc, "sf_curriculum_schedule", None)
        if _sched:
            assert all(int(seg[1]) == 1 for seg in _sched), \
                "[SF-JITTER] only W=1 is supported (under W>1 the misaligned-mask semantics would mask the last frame of the previous section, not implemented)"
        assert not bool(getattr(tc, "corrupt_history", False)) and not bool(getattr(tc, "is_add_saturation", False)), \
            "[SF-JITTER] mutually exclusive with the corrupt_history/is_add_saturation augmentations (off=1 re-slices the tiers and would bypass the augmentation transform)"
        # for an N=1 single-chunk section, off=1 borrows the sacrificial frame from the last GT prefix frame, so the prefix needs at least 1 whole section (P=prefix_sections x win >= win).
        # defensive hoist: catch the bad config "P < a whole section -> runtime window assert" at startup (the current rollout_prefix_sections=1 always satisfies it).
        assert int(getattr(tc, "rollout_prefix_sections", 0)) >= 1, \
            "[SF-JITTER] requires rollout_prefix_sections>=1 (off=1 of an N=1 section borrows the sacrificial frame from the GT prefix)"
    # preconditions for phase-diffusion jitter (the default max_off=1 passes everything trivially)
    _max_off = int(getattr(tc, "sf_score_window_jitter_max_off", 1) or 1)
    assert 1 <= _max_off <= win - 1, \
        f"[BRAKEFIX R4] sf_score_window_jitter_max_off must be in [1, win-1]=[1,{win-1}], got {_max_off}"
    if _max_off > 1:
        assert bool(getattr(tc, "sf_score_window_jitter", False)), \
            "[BRAKEFIX R4] sf_score_window_jitter_max_off>1 requires sf_score_window_jitter=true"
        _sched_mo = getattr(tc, "sf_curriculum_schedule", None)
        if _sched_mo:
            assert all(int(seg[0]) >= 2 for seg in _sched_mo), \
                "[BRAKEFIX R4] max_off>1 requires N>=2 for every curriculum stage (a large off on an N=1 section would re-slice the GT prefix video " \
                "I-frame latent p0 into the prev slot, exactly the distribution that is meant to be isolated)"
    # preconditions for critic full-frame fitting
    if bool(getattr(tc, "sf_critic_full_frame", False)):
        assert bool(getattr(tc, "sf_score_skip_first_latent", False)), \
            "[BRAKEFIX R3] sf_critic_full_frame only makes sense with sf_score_skip_first_latent=true" \
            "(otherwise the critic is already full-frame, so this looks like a misconfiguration)"
    # [TWO-SLOT MASK] preconditions for k mode (the default k=1 passes everything trivially)
    _skip_k = int(getattr(tc, "sf_score_skip_first_k", 1) or 1)
    assert 1 <= _skip_k <= win - 2, \
        f"[TWO-SLOT MASK] sf_score_skip_first_k must be in [1, win-2]=[1,{win-2}], got {_skip_k}"
    if _skip_k >= 2:
        assert bool(getattr(tc, "sf_score_skip_first_latent", False)), \
            "[TWO-SLOT MASK] k>=2 requires sf_score_skip_first_latent=true (the master mask switch)"
        assert bool(getattr(tc, "sf_score_window_jitter", False)), \
            "[TWO-SLOT MASK] k>=2 requires sf_score_window_jitter=true (off in {0,k} alternation, otherwise f0..f_{k-1} are a zero-supervision vacuum)"
        assert _max_off == 1, \
            "[TWO-SLOT MASK] k>=2 is mutually exclusive with sf_score_window_jitter_max_off>1 (k mode replaces phase diffusion)"
        _sched_k = getattr(tc, "sf_curriculum_schedule", None)
        if _sched_k:
            assert all(int(seg[0]) >= 2 for seg in _sched_k), \
                "[TWO-SLOT MASK] k>=2 requires N>=2 for every curriculum stage (same reason as max_off>1, guarding against an N=1 section borrowing deep into the GT prefix)"
    # preconditions (the default false passes trivially)
    if bool(getattr(tc, "sf_teacher_gt_longmid", False)):
        assert dc.use_full_rollout_interleave, \
            "[GT-ANCHOR] requires use_full_rollout_interleave=true (the data side produces sf_gt_latents in that mode)"
        assert int(getattr(tc, "rollout_prefix_sections", 0)) >= 1, \
            "[GT-ANCHOR] requires rollout_prefix_sections>=1 (GT and the rollout timeline are aligned via the prefix)"
        assert bool(getattr(tc, "sf_share_rollout", False)), \
            "[GT-ANCHOR] requires sf_share_rollout=true (the critic reuses the snapshot; an independent re-roll has no GT tier wired)"
    # preconditions for the critic scoring band cap (None = legacy behavior, not validated)
    _c_max = getattr(tc, "critic_score_timestep_max", None)
    if _c_max is not None:
        _c_min = int(getattr(tc, "critic_score_timestep_min", 0) or 0)
        assert tc.is_train_dmd, "[BRAKEFIX R1] critic_score_timestep_max is only for DMD training"
        assert 0 <= _c_min < int(_c_max) <= 1000, \
            f"[BRAKEFIX R1] need 0 <= min({_c_min}) < max({int(_c_max)}) <= 1000"
    # preconditions for thin high-band mixed sampling (prob=0 passes trivially)
    _hb_p = float(getattr(tc, "dmd_score_highband_prob", 0.0) or 0.0)
    assert 0.0 <= _hb_p <= 1.0, f"[HIGHBAND-ANCHOR] dmd_score_highband_prob must be in [0,1], got {_hb_p}"
    if _hb_p > 0.0:
        assert tc.is_train_dmd, "[HIGHBAND-ANCHOR] only for DMD training"
        _lb_max = getattr(tc, "dmd_score_timestep_max", None)
        assert _lb_max is not None, \
            "[HIGHBAND-ANCHOR] the thin high band is a finisher for the low-band cap: dmd_score_timestep_max must already be set (otherwise scoring is full-band anyway)"
        _hb_min = int(getattr(tc, "dmd_score_highband_min", 666))
        _hb_max = int(getattr(tc, "dmd_score_highband_max", 1000))
        assert int(_lb_max) <= _hb_min < _hb_max <= 1000, \
            f"[HIGHBAND-ANCHOR] need lowband_cap({int(_lb_max)}) <= hb_min({_hb_min}) < hb_max({_hb_max}) <= 1000" \
            "(both endpoints are actual-t semantics; an hb_min below the low-band cap would overlap the low band, which looks like a misconfiguration)"
    # F6 shared rollout: the generator must roll on every step (the critic has no independent re-roll available; forces 1:1).
    if bool(getattr(tc, "sf_share_rollout", False)):
        assert int(tc.dfake_gen_update_ratio) == 1, \
            "[SF-EVOKE] with sf_share_rollout=true, dfake_gen_update_ratio must be 1 (the critic reuses the generator rollout)"
    # with student warp-ON the shared rollout is mandatory: an independent critic re-roll cannot render warp (no pose/helper) -> conditioning mismatch.
    if bool(getattr(tc, "use_geometric_state", False)):
        assert bool(getattr(tc, "sf_share_rollout", False)), \
            "[SF-EVOKE] with use_geometric_state=true, sf_share_rollout=true is mandatory (review: a critic re-roll cannot render warp)"
        assert not bool(getattr(tc, "dmd_is_low_vram_mode", False)), \
            "[SF-EVOKE] warp-in-rollout needs the vae resident, incompatible with dmd_is_low_vram_mode"

    # [review SHOULD-FIX#4] v1 forces the curriculum on: the curriculum-free sf mode (fixed N) lacks the num_frames/N/W consistency validation,
    # so errors would be deferred to the step0 rollout assert; v1 was designed to be curriculum-driven anyway, so just require it (a later fixed-N mode can relax this + add validation).
    assert bool(getattr(tc, "sf_curriculum_enabled", False)), \
        "[SF-EVOKE] v1 requires sf_curriculum_enabled=true (the fixed-N mode has no startup validation)"
    # curriculum: each schedule entry is [N, W, step_budget]; validate 1 <= W <= N entry by entry.
    if bool(getattr(tc, "sf_curriculum_enabled", False)):
        sched = list(getattr(tc, "sf_curriculum_schedule", []) or [])
        assert len(sched) >= 1, "[SF-EVOKE] sf_curriculum_schedule must not be empty when sf_curriculum_enabled"
        for i, ent in enumerate(sched):
            assert len(ent) == 3, f"[SF-EVOKE] curriculum[{i}] must be [N, W, step_budget], got {ent}"
            N_i, W_i, budget_i = int(ent[0]), int(ent[1]), int(ent[2])
            assert 1 <= W_i <= N_i, f"[SF-EVOKE] curriculum[{i}] needs 1 <= W({W_i}) <= N({N_i})"
            assert budget_i >= 1, f"[SF-EVOKE] curriculum[{i}] step_budget must be >=1, got {budget_i}"
        max_N = max(int(e[0]) for e in sched)
        # with W>1 the gradient window must = the scoring window (otherwise dmd_last_section_grad_only only gives the last section a gradient, misaligned with the W sections; review SHOULD-FIX).
        if any(int(e[1]) > 1 for e in sched):
            assert not tc.dmd_last_section_grad_only, \
                "[SF-EVOKE] when the curriculum contains W>1, dmd_last_section_grad_only must be false (gradient window = scoring window)"
        # num_frames must cover the deepest N (data is materialized for the deepest N, then the train side slices the current N;).
        P = int(tc.rollout_prefix_sections)
        need_frames = ((P + max_N) * win - 1) * 4 + 1
        assert int(dc.num_frames) >= need_frames, (
            f"[SF-EVOKE] num_frames({dc.num_frames}) must be >= {need_frames}, which covers the deepest N={max_N}"
            f"(P={P}, win={win})")

    # tail-section warp: sf_warp_tail_chunks >= the deepest scoring window W (otherwise the scored sections have no warp, inconsistent with student warp-on generation).
    tail = getattr(tc, "sf_warp_tail_chunks", None)
    if tail is not None and int(tail) > 0 and bool(getattr(tc, "sf_curriculum_enabled", False)):
        max_W = max(int(e[1]) for e in (getattr(tc, "sf_curriculum_schedule", []) or []))
        assert int(tail) >= max_W, \
            f"[SF-EVOKE] sf_warp_tail_chunks({tail}) must be >= the deepest scoring window W({max_W})"
