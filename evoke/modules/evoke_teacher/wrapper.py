"""EvokeTeacherScoreWrapper -- wraps the two EvokeTeacher experts into a score model that the evoke DMD engine can consume directly.

Design (:
  - the external call surface mimics a evoke score model exactly:
      forward(hidden_states, timestep, encoder_hidden_states, indices_*, latents_history_*,
              return_dict=False, attention_kwargs=None, ...) -> (flow_pred,)
    so that compute_kl_grad / compute_distribution_matching_loss / _critic_loss are reused with zero changes.
  - encoder_hidden_states 3-D [B,L,D] = a single prompt (the negative prompt takes this path);
    4-D [B,S,L,D] = segmented prompts (set_condition must supply segment_frame_ranges first).
  - the i2v condition y ([B,20,T,H,W] = 4ch mask + 16ch VAE (first frame + zero frames), encoded on the data side, PLAN M8)
    is cached per batch through set_condition and concatenated with hidden_states (16ch) inside forward into in_dim=36.
  - the two experts are routed by t: t >= boundary*1000 (default 0.9 -> 900) goes to high-noise, otherwise low-noise
    (following EvokeTeacher's inference semantics switch_dit_boundary, PLAN M3).
  - the critic-LoRA is injected into both experts with peft; enable/disable_adapters is compatible with the
    teacher (adapters OFF) / critic (adapters ON) switching of compute_kl_grad.
  - the core of forward is a port of the nocam / non-SP path of EvokeTeacher
    pipelines/wan_video.py::model_fn_wan_video (only that function holds the segment encoding / chunk_context_map logic, dit.forward does not).

Numerical conventions (verified by the P0 score comparison): x_t=(1-sigma)x0+sigma*eps, the model predicts v=eps-x0, and the model
input timestep=sigma*1000 (Wan convention, consistent with x0=xt-sigma*v in evoke convert_flow_pred_to_x0 and with the shift5.0 schedule, so timestep is passed straight through).
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from .loader import build_evoke_teacher_dit, load_merged_weights
from .dit_sparse_14b import sinusoidal_embedding_1d

# global forward call counter (for the per-rank trace points that locate a 2D hang).
_SP_FWD_CALL = 0

# Matches the target surface of EvokeTeacher's training LoRA recipe (lora.target_modules).
EVOKE_TEACHER_LORA_TARGETS = [
    "self_attn.q", "self_attn.k", "self_attn.v", "self_attn.o",
    "cross_attn.q", "cross_attn.k", "cross_attn.v", "cross_attn.o",
    "ffn.0", "ffn.2",
]


def build_i2v_y(cond_latent_norm: torch.Tensor, num_cond_px_frames: int = 1) -> torch.Tensor:
    """Assemble y = [4ch mask | 16ch cond latent] following EvokeTeacher WanVideoUnit (wan_video.py).

    cond_latent_norm: [B,16,T_lat,H',W'] -- the data-side pixel-domain [cond frames | zero frames] passed through the VAE and
      normalized as (z-mean)/std (the VAE normalization constants on both sides were checked value by value);
    num_cond_px_frames: number of conditioning pixel frames (i2v=1, v2v=N).
    Returns [B,20,T_lat,H',W'].
    """
    B, C, T_lat, Hl, Wl = cond_latent_norm.shape
    assert C == 16
    F_px = (T_lat - 1) * 4 + 1
    msk = torch.ones(1, F_px, Hl, Wl, device=cond_latent_norm.device, dtype=cond_latent_norm.dtype)
    msk[:, num_cond_px_frames:] = 0
    msk = torch.cat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
    msk = msk.view(1, msk.shape[1] // 4, 4, Hl, Wl)
    msk = msk.transpose(1, 2)[0]  # [4, T_lat, H', W']
    y = torch.cat([msk.unsqueeze(0).expand(B, -1, -1, -1, -1), cond_latent_norm], dim=1)
    return y


def _orig_to_latent(orig_frame: int) -> int:
    # copied from model_fn_wan_video: pixel frame index -> latent frame index (Wan VAE 4x temporal compression, frame 0 stands alone).
    if orig_frame <= 0:
        return 0
    return (orig_frame - 1) // 4 + 1


class EvokeTeacherScoreWrapper(nn.Module):
    def __init__(
        self,
        high_dir: str,
        low_dir: str,
        boundary: float = 0.9,
        model_cfg_overrides: dict = None,
        torch_dtype=torch.bfloat16,
        critic_lora_rank: int = 0,
        critic_lora_alpha: float = 0.0,
        critic_lora_dropout: float = 0.0,
        single_expert: str = None,  # None=both experts; "high"/"low"=load only one (saves 28G GPU in a smoke run, no quality conclusions from its scores)
    ):
        super().__init__()
        self._torch_dtype = torch_dtype
        self.boundary_t = float(boundary) * 1000.0
        self._single_expert = single_expert
        assert single_expert in (None, "high", "low"), single_expert

        if single_expert != "low":
            print(f"[EvokeTeacherScoreWrapper] building high-noise expert from {high_dir}")
            self.dit_high = build_evoke_teacher_dit(model_cfg_overrides, torch_dtype)
            load_merged_weights(self.dit_high, high_dir, torch_dtype)
        else:
            self.dit_high = None
        if single_expert != "high":
            print(f"[EvokeTeacherScoreWrapper] building low-noise expert from {low_dir}")
            self.dit_low = build_evoke_teacher_dit(model_cfg_overrides, torch_dtype)
            load_merged_weights(self.dit_low, low_dir, torch_dtype)
        else:
            self.dit_low = None
        if single_expert is not None:
            print(f"[EvokeTeacherScoreWrapper] SINGLE-EXPERT={single_expert} (saves 28G GPU; all t route to this expert, "
                  f"so the scoring regime is inaccurate for the other half of t -- smoke pipeline check only)")

        self.requires_grad_(False)

        self._has_critic_lora = critic_lora_rank > 0
        if self._has_critic_lora:
            self._inject_critic_lora(critic_lora_rank, critic_lora_alpha, critic_lora_dropout)

        self._use_gradient_checkpointing = False
        # train_evoke sets this True for dual-expert (single_expert=None) + offload:
        #   the scoring forward keeps only **the routed expert** (28GB) resident on GPU and leaves the other expert's base on
        #   CPU -> peak = the single-expert level (otherwise both experts resident at 28GB each = +28GB -> hits the 141G H200
        #   wall, see the dual-expert-offload memory). Default False
        #   -> byte-identical (single expert / no offload / the old whole-wrapper offload path are unchanged).
        self._per_expert_offload = False
        # per-batch condition cache (set_condition): the y latent and the segment pixel-frame ranges.
        self._cond_y = None
        self._cond_segment_frame_ranges = None

    # ------------------------------------------------------------------ setup

    def _inject_critic_lora(self, rank: int, alpha: float, dropout: float):
        from peft import LoraConfig
        try:
            from peft import inject_adapter_in_model
        except ImportError:  # path used by older peft versions
            from peft.mapping import inject_adapter_in_model

        cfg = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout,
            init_lora_weights="gaussian",
            target_modules=list(EVOKE_TEACHER_LORA_TARGETS),
        )
        for name in ("dit_high", "dit_low"):
            m = getattr(self, name)
            if m is not None:  # the other one is None in single-expert mode
                inject_adapter_in_model(cfg, m, adapter_name="critic")
        n_train = 0
        for pname, p in self.named_parameters():
            if "lora_" in pname:
                p.requires_grad = True
                # NOTE: this cast is overridden by train_evoke's blanket bf16 downcast; what really takes
                # effect is the later cast_training_params(models, fp32) (review S4); it is kept only for
                # correctness when the wrapper is used standalone.
                p.data = p.data.to(torch.float32)
                n_train += p.numel()
            else:
                p.requires_grad = False
        print(f"[EvokeTeacherScoreWrapper] critic LoRA injected on both experts: "
              f"rank={rank} trainable={n_train/1e6:.1f}M params")

    def _iter_lora_layers(self):
        from peft.tuners.tuners_utils import BaseTunerLayer
        for m in self.modules():
            if isinstance(m, BaseTunerLayer):
                yield m

    def enable_adapters(self):
        if self._has_critic_lora:
            for m in self._iter_lora_layers():
                m.enable_adapters(True)

    def disable_adapters(self):
        if self._has_critic_lora:
            for m in self._iter_lora_layers():
                m.enable_adapters(False)

    def enable_gradient_checkpointing(self):
        # EvokeTeacher's GC is a block-loop-level behaviour (the checkpoint wrapping inside model_fn), not a module method;
        # this only sets a flag that takes effect inside forward (compatible with the call site at train_evoke.py).
        self._use_gradient_checkpointing = True

    def trainable_state_dict(self):
        return {k: v for k, v in self.state_dict().items() if "lora_" in k}

    # evoke reads the fp32 keep-list off this class attribute (train_evoke.py).
    _keep_in_fp32_modules: list = []

    @property
    def dtype(self):
        return self._torch_dtype

    @property
    def device(self):
        # the frozen base may live on CPU while the trainable critic-LoRA is always on GPU -> report the GPU (compute)
        #   device, so the caller moves the scoring inputs to GPU (the caller derives the input device from real_score_model.device, cdml:2758). On the
        #   non-per-expert path (base and LoRA always co-resident) -> next(parameters()) is that same device, byte-identical.
        if self._per_expert_offload:
            for p in self.parameters():
                if p.device.type == "cuda":
                    return p.device
        return next(self.parameters()).device

    # ---------------------------------------------------- per-expert offload
    @staticmethod
    def _first_base_device(m):
        """Device of the first **frozen base** (non-lora_) parameter of the routed expert (the base is co-resident as a whole -> the first one represents all)."""
        for n, p in m.named_parameters():
            if "lora_" not in n:
                return p.device
        return None

    def _ensure_routed_expert_on_gpu(self, dit, device):
        """keep only the routed expert `dit`'s frozen base resident on `device` (GPU) and swap
        the other expert's base out to CPU. Idempotent (it checks the actual device -> several forwards inside one scoring call
        do not move it repeatedly). Only active under _per_expert_offload; otherwise a no-op (whole-wrapper offload is handled
        externally by _offload_frozen_params_to). The move only touches the frozen base (names without 'lora_') plus buffers;
        the trainable critic-LoRA stays on GPU untouched (moving a DeepSpeed flat-buffer would re-point .data and leak host
        memory, see _offload_frozen_params_to)."""
        if not self._per_expert_offload:
            return
        from evoke.utils.utils_evoke_post import _offload_frozen_params_to
        _dev = device if isinstance(device, torch.device) else torch.device(device)
        other = self.dit_low if dit is self.dit_high else self.dit_high
        if other is not None:
            _od = self._first_base_device(other)
            if _od is not None and _od.type != "cpu":
                _offload_frozen_params_to(other, "cpu")
        _dd = self._first_base_device(dit)
        if _dd is not None and _dd != _dev:
            _offload_frozen_params_to(dit, _dev)

    # ------------------------------------------------------------ conditioning

    def set_condition(
        self,
        y: torch.Tensor,
        segment_frame_ranges: Optional[List[Tuple[int, int]]] = None,
    ):
        """Call once per batch. y: [B,20,T_lat,H_lat,W_lat] (4ch mask + 16ch first-frame VAE cond,
        encoded in the pixel domain on the data side); segment_frame_ranges: [(start_px, end_px), ...]
        in one-to-one correspondence with the S dimension of the 4-D encoder_hidden_states."""
        assert y is not None and y.dim() == 5 and y.shape[1] == 20, \
            f"y must be [B,20,T,H,W], got {None if y is None else tuple(y.shape)}"
        self._cond_y = y
        self._cond_segment_frame_ranges = segment_frame_ranges

    # ----------------------------------------------------------------- forward

    def _route_expert(self, timestep: torch.Tensor):
        if self._single_expert == "high":
            return self.dit_high
        if self._single_expert == "low":
            return self.dit_low
        # route by timestep: t >= boundary*1000 (default 900) -> high-noise expert, otherwise low-noise
        #   (matching EvokeTeacher's inference semantics switch_dit_boundary). forward() has already called
        #   sync_tensor_in_sp_group(timestep) **before** this function -> t0 is a timestep that is **consistent inside the SP group**:
        #     - non-decouple (whole group on the same clip / all ranks on the same seed) -> group-wide = globally consistent -> routing agrees;
        #     - throughput-B decouple (each group scores a different clip) -> t0 = this group's owner timestep -> this group routes by the band of
        #       its own clip and every other group by its own clip (correct). NOTE: it is no longer broadcast from WORLD rank0: that would force
        #       every group onto rank0's expert, which under decoupling means scoring this group's low/high-noise clip with the
        #       **wrong-band expert** (OOD). Group-wide consistency is guaranteed by forward's timestep sync;
        # ZeRO-3 pin (the original motivation for the old broadcast) died together with the ZeRO-3 plan, and under ZeRO-2 pin is a no-op -> no WORLD collective can get misaligned.
        t0 = float(timestep.flatten()[0])
        use_high = t0 >= self.boundary_t
        return self.dit_high if use_high else self.dit_low

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        indices_hidden_states=None,
        indices_latents_history_short=None,
        indices_latents_history_mid=None,
        indices_latents_history_long=None,
        latents_history_short=None,
        latents_history_mid=None,
        latents_history_long=None,
        return_dict: bool = False,
        attention_kwargs: dict = None,
        **kwargs,
    ):
        # EvokeTeacher scores the whole sequence and does not consume the evoke history tiers; a non-None value means the caller took the wrong branch.
        assert all(
            v is None
            for v in (indices_hidden_states, indices_latents_history_short,
                      indices_latents_history_mid, indices_latents_history_long,
                      latents_history_short, latents_history_mid, latents_history_long)
        ), "[EvokeTeacherScoreWrapper] evoke history tiers must not be passed into the evoke_teacher scoring path"
        assert self._cond_y is not None, "call set_condition() first to supply the i2v y condition"
        assert not return_dict

        # per-rank trace point to locate a 2D hang (SP off -> no-op).
        from .sp_runtime import sp_diag as _sp_diag
        global _SP_FWD_CALL
        _SP_FWD_CALL += 1
        _sp_diag(f"wrapper.forward#{_SP_FWD_CALL} ENTER grad={torch.is_grad_enabled()} "
                 f"t0={float(timestep.flatten()[0]):.0f}")

        # === [SP] broadcast every input of the SP forward inside the group (from group rank0) ===
        # the G ranks of a group share one clip, but the non-deterministic bf16 / flash-attn kernels of the student rollout make
        # each rank's generated video drift slightly -> the frame-dimension scatter/gather would not stitch consistently. This
        # forward is the only entry point of the SP forward, so everything is broadcast here -> the group's scoring forward is
        # fully coherent (regardless of rollout drift / upstream RNG misalignment). In DMD, hidden_states (the noisy latent is
        # already .detach()ed) / timestep / encoder are all detached / no-grad (the generator gradient is applied on the separate
        # KL-grad path of compute_kl_grad, not through the scoring model inputs) -> broadcasting does not break the gradient path.
        # timestep is synced first (expert routing depends on it; a disagreement inside the group would route to different dits
        # -> misaligned SP collectives -> deadlock). sync_tensor_in_sp_group: SP off = no-op -> byte-identical.
        from .sp_runtime import is_sp_enabled, sync_tensor_in_sp_group
        if is_sp_enabled():
            timestep = sync_tensor_in_sp_group(timestep.contiguous())
            hidden_states = sync_tensor_in_sp_group(hidden_states.contiguous())
            if encoder_hidden_states is not None:
                encoder_hidden_states = sync_tensor_in_sp_group(encoder_hidden_states.contiguous())

        dit = self._route_expert(timestep)
        # keep only the routed expert (28GB) resident on GPU and swap the other expert's base out
        #   to CPU -> peak = the single-expert level. The device comes from the inputs' GPU (the caller already moved the scoring
        #   inputs to self.device=GPU). no-op unless _per_expert_offload. NOTE: must come before pin / _forward_core (the base has to be on GPU first).
        self._ensure_routed_expert_on_gpu(dit, hidden_states.device)
        # before the scoring SP forward, pre-gather the routed expert (dit_high/low, one 14B ~= 28GB) into a
        # resident state and pin it, removing the deadlock where ZeRO-3's per-parameter WORLD all-gather interleaves with the
        #   SP-subgroup all_to_all (analysed in sp_zero3.py). SP off / not ZeRO-3 -> no-op. Teacher scoring is entirely no_grad
        #   (utils_evoke_post:2642) -> the unpin at the end of this forward is self-contained; critic scoring is grad-enabled ->
        #   the pin is kept across backward and train_evoke backstops it with unpin_all() after the critic backward.
        from .sp_zero3 import pin_module_params, unpin_module_params
        _sp_pinned = pin_module_params(dit)
        x = hidden_states.to(self._torch_dtype)
        y = self._cond_y.to(device=x.device, dtype=x.dtype)
        # [SP] y (the i2v condition) is concatenated with x into in_dim=36 and goes through SP together; y is already on GPU here
        #   -> the broadcast removes the non-deterministic VAE drift and keeps the concat input consistent inside the group (synced from the same source as hidden_states). SP off = no-op.
        if is_sp_enabled():
            y = sync_tensor_in_sp_group(y.contiguous())
        assert y.shape[2:] == x.shape[2:], f"y spatio-temporal shape {tuple(y.shape)} does not match x {tuple(x.shape)}"
        if y.shape[0] != x.shape[0]:
            y = y.expand(x.shape[0], -1, -1, -1, -1)
        x = torch.cat([x, y], dim=1)  # 16 + 20 = in_dim 36

        if timestep.dim() == 0:
            timestep = timestep[None]
        # [review#1] the training engine hands in an int64 timestep (sample_dynamic_timestep .round().long());
        # the last line of sinusoidal_embedding_1d does to(position.dtype), which truncates cos/sin to integers -> it must be cast to fp32 first
        # (do not go straight to bf16: for integers around t~900 the bf16 step is 4, which would introduce a sigma offset).
        timestep = timestep.flatten().to(device=x.device, dtype=torch.float32)
        if timestep.shape[0] != x.shape[0]:
            timestep = timestep.expand(x.shape[0])

        flow_pred = self._forward_core(dit, x, timestep, encoder_hidden_states)
        # no_grad (teacher DMD scoring) -> the unpin here is self-contained (gather-and-hold-and-release inside one forward);
        #   grad-enabled (critic scoring) -> the pin is kept and train_evoke calls unpin_all() after the critic backward.
        if _sp_pinned and not torch.is_grad_enabled():
            unpin_module_params(dit)
        return (flow_pred,)

    def _forward_core(self, dit, x, timestep, encoder_hidden_states):
        """Port of the nocam / non-SP / non-teacache / non-repr path of model_fn_wan_video."""
        # Timestep modulation
        t = dit.time_embedding(
            sinusoidal_embedding_1d(dit.freq_dim, timestep).to(self._torch_dtype))
        t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))

        # Patchify
        x = dit.patchify(x)
        f, h, w = x.shape[2:]
        x = x.flatten(2).transpose(1, 2).contiguous()  # b c f h w -> b (f h w) c

        # text conditioning: 3-D single prompt / 4-D segmented
        segment_contexts_encoded = None
        chunk_context_map = None
        frame_to_seg = None  # [SP] keep the global frame->segment map; used to recompute the local chunk_context_map under SP
        if encoder_hidden_states.dim() == 4:
            assert self._cond_segment_frame_ranges is not None, \
                "4-D encoder_hidden_states (segmented mode) requires set_condition to supply segment_frame_ranges"
            B_sc, num_seg, L_text, dim_text = encoder_hidden_states.shape
            assert num_seg == len(self._cond_segment_frame_ranges), (
                f"segment count mismatch: embeds S={num_seg} vs ranges {len(self._cond_segment_frame_ranges)}")
            seg_flat = encoder_hidden_states.reshape(B_sc * num_seg, L_text, dim_text)
            seg_flat = seg_flat.to(self._torch_dtype)
            seg_encoded = dit.text_embedding(seg_flat)
            segment_contexts_encoded = seg_encoded.reshape(B_sc, num_seg, L_text, -1)
            # the main cross-attn context falls back to segment 0 (copied from model_fn semantics: when chunk_context_map is
            # present the block swaps context per chunk and the main context serves only as a fallback).
            context = segment_contexts_encoded[:, 0]
            # latent frame -> segment map (pixel-frame ranges -> latent frames, taking the chunk mid-frame)
            frame_to_seg = torch.zeros(f, dtype=torch.long, device=x.device)
            for si, (sf, ef) in enumerate(self._cond_segment_frame_ranges):
                lat_s = _orig_to_latent(int(sf))
                lat_e = min(_orig_to_latent(int(ef)), f)
                frame_to_seg[lat_s:lat_e] = si
            chunk_f = dit.blocks[0].chunk_size if len(dit.blocks) > 0 else 8
            num_chunks = (f + chunk_f - 1) // chunk_f
            chunk_map = []
            for ci in range(num_chunks):
                mid = min((ci * chunk_f + min((ci + 1) * chunk_f, f)) // 2, f - 1)
                chunk_map.append(int(frame_to_seg[mid]))
            chunk_context_map = torch.tensor(chunk_map, dtype=torch.long, device=x.device)
        else:
            context = dit.text_embedding(encoder_hidden_states.to(self._torch_dtype))

        # RoPE freqs
        freqs = torch.cat([
            dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
        ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)

        # === [SP] Sequence Parallel: scatter along the frame dimension before the block loop (ghost frames included) ===
        # ported from the SP path of EvokeTeacher pipelines/wan_video.py::model_fn_wan_video
        # (training mode blend_f=0 -> hard-trim gather). sp_enabled unset -> _sp_active=False -> byte-identical.
        _sp_active = getattr(dit, "sp_enabled", False)
        _sp_total_seq_len = x.shape[1]           # total global token count f*h*w
        _sp_f_start = 0
        _sp_num_frames_global = f
        _ghost_before = 0
        _freqs_full = freqs
        if _sp_active:
            from .sp_runtime import get_sp_frame_info
            pf_sp = h * w
            _freqs_full = freqs
            frames_per_rank, _sp_f_start, _sp_f_end, _sp_f_local = get_sp_frame_info(f)
            # ghost frames align to the chunk grid, so the local chunk split matches the global single-GPU one (model_fn L1161-1171)
            _chunk_f = dit.blocks[0].chunk_size if len(dit.blocks) > 0 and hasattr(dit.blocks[0], "chunk_size") else 8
            _aligned_start = (_sp_f_start // _chunk_f) * _chunk_f
            _ghost_f_start = max(0, _aligned_start - _chunk_f)
            _sp_f_end_real = min(f, _sp_f_start + frames_per_rank)
            _aligned_end = ((_sp_f_end_real + _chunk_f - 1) // _chunk_f) * _chunk_f
            _ghost_f_end = min(f, _aligned_end + _chunk_f)
            _ghost_before = _sp_f_start - _ghost_f_start
            x = x[:, _ghost_f_start * pf_sp:_ghost_f_end * pf_sp]
            freqs = _freqs_full[_ghost_f_start * pf_sp:_ghost_f_end * pf_sp]
            _sp_f_start = _ghost_f_start
            # recompute the local chunk_context_map (global chunk indices != local ones; model_fn L1189-1200)
            if frame_to_seg is not None and chunk_context_map is not None:
                _local_f = _ghost_f_end - _ghost_f_start
                _local_num_chunks = (_local_f + _chunk_f - 1) // _chunk_f
                _local_map = []
                for _ci in range(_local_num_chunks):
                    _local_mid = min((_ci * _chunk_f + min((_ci + 1) * _chunk_f, _local_f)) // 2, _local_f - 1)
                    _global_mid = min(_local_mid + _ghost_f_start, f - 1)
                    _local_map.append(int(frame_to_seg[_global_mid]))
                chunk_context_map = torch.tensor(_local_map, dtype=torch.long, device=x.device)

        extra_kw = {
            "tokens_per_frame": h * w,
            "spatial_hw": (h, w),
            "freqs_3d": dit.freqs,
            "select_gate_t_frac": (timestep.detach().float() / 1000.0).clamp(0.0, 1.0),
        }
        if segment_contexts_encoded is not None:
            extra_kw["segment_contexts_encoded"] = segment_contexts_encoded
            extra_kw["chunk_context_map"] = chunk_context_map
        if _sp_active:
            extra_kw["sp_num_frames_global"] = _sp_num_frames_global
            extra_kw["sp_frame_offset"] = _sp_f_start
            extra_kw["freqs_full"] = _freqs_full

        use_gc = self._use_gradient_checkpointing and torch.is_grad_enabled()
        for block in dit.blocks:
            if use_gc:
                x = torch.utils.checkpoint.checkpoint(
                    block, x, context, t_mod, freqs,
                    use_reentrant=False, hidden_h=h, hidden_w=w, **extra_kw)
            else:
                x = block(x, context, t_mod, freqs, hidden_h=h, hidden_w=w, **extra_kw)
            if isinstance(x, tuple):
                x = x[0]

        # === [SP] trim ghost -> head on local -> gather (hard trim, training path model_fn L1402-1415) ===
        if _sp_active:
            from .sp_runtime import get_sp_frame_info, gather_frames
            _fpr, _orig_f_start, _orig_f_end, _ = get_sp_frame_info(f)
            pf_sp = h * w
            trim_start = _ghost_before * pf_sp
            orig_local_tokens = (_orig_f_end - _orig_f_start) * pf_sp
            x = x[:, trim_start:trim_start + orig_local_tokens].contiguous()
            x = dit.head(x, t)
            if x.shape[1] < _fpr * pf_sp:
                x = torch.nn.functional.pad(x, (0, 0, 0, _fpr * pf_sp - x.shape[1]))
            from .sp_runtime import sp_diag as _sp_diag
            _sp_diag("forward_core pre-gather (block loop done)")
            x = gather_frames(x, _sp_total_seq_len)  # all-gather back to the full sequence (autograd-aware)
            _sp_diag("forward_core gather done")
        else:
            x = dit.head(x, t)
        x = dit.unpatchify(x, (f, h, w))
        return x
