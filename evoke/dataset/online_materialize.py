"""Online batch materialization: raw_video + prompt -> x0/history/target latents + prompt_embeds."""
from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from diffusers.utils.torch_utils import randn_tensor
from einops import rearrange

from evoke.utils.utils_base import encode_prompt


# Module-level singleton for the cloud-warp depth estimator (P2; ~5 GB, loaded once).
# Keyed on the construction params: an unkeyed singleton would hand back a DA3 estimator to a caller
# that asked for ViGeo (or one built at another resolution) and silently ignore the request.
_DEPTH_ESTIMATORS: dict = {}

# NOTE: circuit breaker for on-demand GT encoding. once a check refutes the premise => set True => fall back to full-length encoding for the rest of this run.
#   a dict instead of a bare bool + `global`: avoids getting the placement constraints of `global` wrong again (see MEMORY: edits to big files must be AST-located).
#   **never assert-kill training** -- the right action when a verification check fails is "fall back to the slow path", not shooting down 8/48 GPUs
# (hard lesson: the GT-VERIFY assert killed the 8-GPU smoke at step 2, and it was itself a broken check that always false-alarms).
_GT_OD = {"off": False, "n_warn": 0}


def _get_da3_estimator(process_res: int, device, weights=None, backend="da3", src=None, vigeo_opts=None):
    """Return the process-local cloud-warp depth estimator, constructing it on first use.

    Cached per (backend, process_res, weights, src, vigeo_opts) so two callers asking for different
    backends -- or the same backend on a different recipe -- do not silently share one instance. ViGeo
    instances carry per-stream state (kv-cache, locked depth scale), so every stream boundary must call
    depth_backend.reset_stream() -- the builders in da3_cloud do this for their callers.
    """
    from evoke.modules.geometric_state.depth_backend import build_estimator
    key = (str(backend or "da3").lower(), int(process_res), str(weights or ""), str(src or ""),
           tuple(sorted((k, str(v)) for k, v in (vigeo_opts or {}).items())))
    if key not in _DEPTH_ESTIMATORS:
        _DEPTH_ESTIMATORS[key] = build_estimator(
            backend, device, int(process_res), weights=weights, src=src, vigeo_opts=vigeo_opts)
    return _DEPTH_ESTIMATORS[key]


def _geo_inject_warp_error(args, recycle_vars, w_lat):
    """In-place additive injection of a banked low-noise y_error onto the CLEAN warp latent.

    w_lat: [B, C, T_lat, H_lat, W_lat]. Resolution-key guarded (no-op if (H,W) not registered in the
    bank or the bank is empty). Honors ref_inject_grid_mode/topk + depth_sample_ratio via the sampler.
    Frame-count mismatch between the warp window and stored y_error is handled by the sampler's
    per-item shape guard (skips => zeros => no-op)."""
    from evoke.utils.utils_recycle_batch import sample_y_error_from_latent_buffer

    ybuf = getattr(recycle_vars, "y_error_buffer", None)
    _, _, _, h, w = w_lat.shape
    if ybuf is None or (h, w) not in ybuf:
        return  # resolution not in the bank -> no-op
    err, _depths = sample_y_error_from_latent_buffer(
        args, recycle_vars, w_lat, dtype=w_lat.dtype, device=w_lat.device
    )
    w_lat.add_(err.to(w_lat.device, dtype=w_lat.dtype))


def apply_warp_token_drop(vis, cfg, generator=None):
    """Train-only stochastic drop on the WARP visibility mask `vis` [B,1,T,H,W] (float in [0,1]).

    One categorical draw per batch item over [none, full, per_frame, per_patch] (cfg.mode_probs, normalized),
    then zeros the chosen region of the mask. Returns a modified clone (input untouched). Coupled by design:
    zeroing the warp visibility -> existing visible_token_drop removes those tokens AND loss weighting treats
    them as invisible. Patch granularity = 2x2 latent (matches short-tier visible-mask pooling). Dropping an
    already-invisible region is a no-op. Requires visible_token_drop=True to actually remove tokens."""
    if vis is None or not getattr(cfg, "enabled", False):
        return vis
    probs = [max(0.0, float(p)) for p in cfg.mode_probs]
    s = sum(probs)
    if s <= 0:
        return vis
    probs = [p / s for p in probs]
    c_full = probs[0] + probs[1]          # cumulative: [none | full | per_frame | per_patch]
    c_frame = c_full + probs[2]
    B, _, T, H, W = vis.shape
    dev = vis.device
    fr, pr = float(cfg.frame_drop_ratio), float(cfg.patch_drop_ratio)
    out = vis.clone()
    for b in range(B):
        r = torch.rand((), generator=generator, device=dev).item()
        if r < probs[0]:
            continue                                  # none
        elif r < c_full:
            out[b] = 0.0                              # full warp drop
        elif r < c_frame:                             # per-frame drop
            drop = torch.rand(T, generator=generator, device=dev) < fr
            out[b, :, drop] = 0.0
        else:                                         # per-patch drop (2x2 latent; visible-only is automatic)
            ph = pw = 2
            gh, gw = (H + ph - 1) // ph, (W + pw - 1) // pw
            pd = torch.rand(T, gh, gw, generator=generator, device=dev) < pr
            pd = pd.repeat_interleave(ph, -2).repeat_interleave(pw, -1)[:, :H, :W]   # [T,H,W]
            out[b, 0] = out[b, 0] * (~pd).to(out.dtype)
    return out


def _pose_jitter_rot_xyz(deg_x, deg_y, deg_z):
    """Small-angle intrinsic-XYZ rotation matrix, degrees -> [3,3] float32 np.
    Mirrors (Rz @ Ry @ Rx)."""
    rx, ry, rz = np.radians([deg_x, deg_y, deg_z])
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], np.float64)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], np.float64)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], np.float64)
    return (Rz @ Ry @ Rx).astype(np.float32)


def _sample_warp_pose_jitter_DT(cfg, mean_interframe_trans=0.0):
    """Sample a single camera-frame rigid offset DeltaT [4,4] (torch float32) from WarpPoseJitterConfig.

    Rotation: pitch(x), yaw(y), roll(z) each |val| ~ U(its range) with random +/- sign (deg). Translation
    (optional, default off): only sampled if trans_frac_range != [0,0]; magnitude = trans_frac * mean_interframe_trans
    distributed over a FORWARD(+z)+lateral(+x)+down(+y) cam-frame drift (mirrors EXP build_DT_cam's vector split).
    Returns DT such that jittered = target_c2w @ DT (right-multiply, camera frame) — exactly EXP's convention."""
    def _signed(rng):
        lo, hi = float(rng[0]), float(rng[1])
        mag = random.uniform(lo, hi)
        return mag if random.random() < 0.5 else -mag
    pitch = _signed(getattr(cfg, "pitch_deg_range", [0.5, 2.0]))   # about x
    yaw = _signed(getattr(cfg, "yaw_deg_range", [0.5, 2.0]))       # about y
    roll = _signed(getattr(cfg, "roll_deg_range", [0.0, 0.5]))     # about z
    DT = np.eye(4, dtype=np.float32)
    DT[:3, :3] = _pose_jitter_rot_xyz(pitch, yaw, roll)
    tr = getattr(cfg, "trans_frac_range", [0.0, 0.0])
    if not (float(tr[0]) == 0.0 and float(tr[1]) == 0.0):
        tf = random.uniform(float(tr[0]), float(tr[1]))
        tmag = tf * float(mean_interframe_trans)
        # cam-frame: +x right, +y down, +z forward (split like EXP build_DT_cam: lateral/vertical/forward).
        DT[:3, 3] = np.array([tmag * 0.5, tmag * 0.2, tmag], np.float32)
    return torch.from_numpy(DT)


def _geo_add_noise_to_warp_latents(
    warp_latents: torch.Tensor,
    device: torch.device,
    generator=None,
    sigma_min: float = 0.111,
    sigma_max: float = 0.135,
    visibility_aware_noise: bool = False,
    sigma_invisible: float = 0.8,
    visibility_mask_lat: torch.Tensor | None = None,
) -> torch.Tensor:
    """Add noise to warp latents with either frame-uniform or spatially-varying sigma.

    Args:
      visibility_mask_lat: shape `[B, 1, T_lat, H_lat, W_lat]` float in [0,1]. T_lat must equal warp_latents.shape[2].
    """
    chunk_frames = int(warp_latents.shape[2])
    rand_generator = generator[0] if isinstance(generator, list) else generator
    frame_sigmas = (
        torch.rand(chunk_frames, device=device, generator=rand_generator) * (sigma_max - sigma_min) + sigma_min
    ).to(dtype=warp_latents.dtype)

    if visibility_aware_noise and visibility_mask_lat is not None:
        # spatial-sigma path: invisible pixels get high noise, visible pixels get low noise
        assert 0.0 < float(sigma_invisible) <= 1.0, (
            f"sigma_invisible must be in (0, 1.0], got {sigma_invisible}"
        )
        mask = visibility_mask_lat.to(device=device, dtype=warp_latents.dtype)
        if mask.shape[2] != chunk_frames:
            raise ValueError(
                f"visibility_mask_lat temporal dim {mask.shape[2]} != warp_latents {chunk_frames}"
            )
        # [B,1,T,H,W] · [1,1,T,1,1] + (1-mask) · scalar  → [B,1,T,H,W]
        sigma_visible_5d = frame_sigmas.view(1, 1, chunk_frames, 1, 1)
        sigmas = mask * sigma_visible_5d + (1.0 - mask) * float(sigma_invisible)
        noise = randn_tensor(warp_latents.shape, generator=generator, device=device, dtype=warp_latents.dtype)
        return sigmas * noise + (1.0 - sigmas) * warp_latents
    else:
        # frame-uniform sigma path
        frame_sigmas = frame_sigmas.view(1, 1, chunk_frames, 1, 1)
        return (
            frame_sigmas
            * randn_tensor(warp_latents.shape, generator=generator, device=device, dtype=warp_latents.dtype)
            + (1 - frame_sigmas) * warp_latents
        )


def _geo_resize_visibility_to_latent(
    visibility_mask_pix: torch.Tensor,
    num_lat_per_chunk: int,
    H_lat: int,
    W_lat: int,
    vae_t_stride: int = 4,
    patch_hw: tuple = (2, 2),
    cov_thresh: float = 0.5,
) -> torch.Tensor:
    """Downsample pixel-domain visibility mask [1,1,33,H,W] to a BLOCKY latent-domain mask
    [1,1,9,H_lat,W_lat].

    coverage-style patch mask: pool pixel coverage onto the DiT patch grid (H_lat//ph, W_lat//pw)
    via adaptive_avg_pool3d, hard-threshold at cov_thresh (a patch is visible iff >cov_thresh of its
    pixel footprint is covered; 0.5=default, higher=stricter, keeping only near-fully-visible blocks), then nearest-upsample
    back so the mask is CONSTANT within each patch.
    This replaces the old trilinear (per-latent-pixel, continuous) downsample, which produced
    salt-and-pepper visibility — feeding the warp conditioning + loss a dotty signal. Used for
    both visibility-aware noise (a) and the visible/invisible loss weighting (c), since both
    consume this same returned mask.
    """
    device = visibility_mask_pix.device
    sample_ids = torch.arange(num_lat_per_chunk, device=device) * int(vae_t_stride)
    sample_ids = sample_ids.clamp(max=visibility_mask_pix.shape[2] - 1)
    sampled = visibility_mask_pix.index_select(2, sample_ids)                       # [1,1,num_lat,H_pix,W_pix]
    ph, pw = max(1, int(patch_hw[0])), max(1, int(patch_hw[1]))
    gh, gw = max(1, int(H_lat) // ph), max(1, int(W_lat) // pw)
    patch_cov = F.adaptive_avg_pool3d(sampled, (num_lat_per_chunk, gh, gw))         # patch coverage in [0,1]
    patch_bin = (patch_cov > float(cov_thresh)).to(sampled.dtype)                   # blocky binary (visible only where coverage >cov_thresh)
    return F.interpolate(patch_bin, size=(num_lat_per_chunk, H_lat, W_lat), mode="nearest")


def prepare_stage1_latent_v2(
    vae_latent: torch.Tensor,
    history_sizes,
    choice_idx: int,
    is_keep_x0: bool = True,
    base_vae_latent: torch.Tensor | None = None,
):
    """Pure-function equivalent of BucketedFeatureDataset.prepare_stage1_latent.

    Args:
        vae_latent: `[B_section, C, T_per_section, H, W]` — video split into B_section chunks.
        history_sizes: [long, mid, short] history lengths (sum = history_window_size).
        choice_idx: which section is the target.
        is_keep_x0: whether to return x0_latent (first frame of section 0).
        base_vae_latent: optional higher-res latent for history; defaults to vae_latent.

    Returns:
        (x0_latent, history_latent, target_latent)
            x0_latent:      [C, 1, H, W]                 (or None)
            history_latent: [C, history_window, H, W]
            target_latent:  [C, latent_window, H, W]
    """
    source_latent = base_vae_latent if base_vae_latent is not None else vae_latent

    x0_latent = source_latent[0, :, :1, :, :].clone() if is_keep_x0 else None

    total_sections = source_latent.shape[0]
    latent_window_size = source_latent.shape[2]
    history_window_size = sum(history_sizes)
    section_size = history_window_size + latent_window_size

    # Flatten sections into time dim and prepend zero-padding for history.
    temp_source = rearrange(source_latent, "b c t h w -> c (b t) h w")
    pad_src = torch.zeros(
        temp_source.shape[0], history_window_size, temp_source.shape[2], temp_source.shape[3],
        device=temp_source.device, dtype=temp_source.dtype,
    )
    continue_source = torch.cat([pad_src, temp_source], dim=1)

    temp_vae = rearrange(vae_latent, "b c t h w -> c (b t) h w")
    pad_vae = torch.zeros_like(pad_src)
    continue_vae = torch.cat([pad_vae, temp_vae], dim=1)

    # Section 0 has no history; zero out x0 to match offline behavior.
    if choice_idx == 0 and x0_latent is not None:
        x0_latent = torch.zeros_like(x0_latent)

    assert 0 <= choice_idx < total_sections
    start = choice_idx * latent_window_size
    end = start + section_size
    history_latent = continue_source[:, start : start + history_window_size, :, :]
    target_latent = continue_vae[:, start + history_window_size : end, :, :]
    return x0_latent, history_latent, target_latent


def _find_segment_aligned_choice_idx(
    segment_prompts,
    initial_choice_idx: int,
    n_section: int,
    latent_window_size: int,
    vae_temporal_ratio: int = 4,
):
    """Find a choice_idx whose target window falls entirely within one interleave_caption_5 segment.

    Searches from initial_choice_idx outward by distance; falls back to (initial, None) if no
    valid segment is found, so a bad sample does not crash the training batch.
    """
    if not segment_prompts:
        return initial_choice_idx, None

    # Pixel frame count per chunk: (W-1)*ratio + 1
    chunk_pix_frames = (latent_window_size - 1) * vae_temporal_ratio + 1  # 9 frames latent = 33 pixel frames

    def _try(ci):
        if ci < 0 or ci >= n_section:
            return None
        target_start = ci * chunk_pix_frames
        target_end = target_start + chunk_pix_frames
        for seg in segment_prompts:
            if seg["start_frame"] <= target_start and target_end <= seg["end_frame"]:
                return seg["prompt"]
        return None

    # Search order: initial, initial+1, initial-1, initial+2, ...
    candidates = [initial_choice_idx]
    for d in range(1, n_section):
        candidates.extend([initial_choice_idx + d, initial_choice_idx - d])

    for ci in candidates:
        cap = _try(ci)
        if cap is not None:
            return ci, cap

    # Graceful fallback: no segment fits; warn and return overall caption.
    import warnings as _warnings
    _warnings.warn(
        f"[interleave_caption] no segment fits {chunk_pix_frames}-frame target window; "
        f"falling back to overall caption. segments={[(s['start_frame'], s['end_frame']) for s in segment_prompts]}",
        RuntimeWarning,
        stacklevel=2,
    )
    return initial_choice_idx, None



def materialize_full_rollout_interleave(
    batch: dict,
    latent: "torch.Tensor",           # [B,C,T_lat,H,W] normalized
    raw_video: "torch.Tensor",        # [B,3,T_pix,H,W] [-1,1]
    vae,
    tokenizer,
    text_encoder,
    latents_mean: "torch.Tensor",
    latents_std: "torch.Tensor",
    latent_window_size: int,
    prefix_sections: int,
    history_sizes,
    device,
    weight_dtype,
    is_keep_x0: bool = True,
    sf_build_teacher_y: bool = True,   # evoke-tier scoring does not consume y -> pass False to avoid importing the evoke_teacher wrapper
    # skip the full-clip encode: the incoming latent is prefix only (T_lat==P*win) and sf_gt_latents is set to None;
    # N/T_px are derived from full_clip_num_frames (no longer from latent.shape). default OFF -> original full-clip behaviour unchanged.
    sf_skip_full_encode: bool = False,
    # True = the incoming latent is the on-demand encoded **prefix** (not the full clip), but still serves as sf_gt_latents.
    sf_gt_partial: bool = False,
    full_clip_num_frames: Optional[int] = None,
    # student i2v mixed training: when >0, **additionally** produce the i2v-layout section prompts (sf_prompt_embeds_list_i2v)
    #   and the 1x slot latent (sf_i2v_hist_latent) for the train loop to swap in on i2v steps. 0 = produce nothing -> old path bit-identical
    #   (no extra VAE encode, the uniq caption set is unchanged -> T5 encoding stays value-identical).
    sf_i2v_prefix_latent_frames: int = 0,
    sf_i2v_hist_latent_mode: str = "static_repeat",
    # probability that a video sample takes i2v. at 0 video samples are always v2v => that 33-frame 1x latent has no reader at all
    #   => it is skipped below. image-only samples ignore this (they compute their own inside materialize_i2v_image_only).
    sf_i2v_ratio: float = 0.0,
):
    """full-sequence rollout + interleave caption mode (table 8).

    Produces (for the self-forcing DMD with real_score_arch=evoke_teacher):
      - sf_prefix_latents [B,C,P*w,H,W]: GT prefix chunk (student rollout start point / teacher context);
      - sf_prompt_embeds_list (len=N): per-generated-section T5 embeds (caption of the segment holding the section midpoint frame,
        all = overall caption when there is no interleave);
      - sf_score_prompt_embeds [B,S,L,D] + sf_segment_frame_ranges: the segment stack used for teacher/critic scoring
        (covers the whole clip including the prefix; S=1 whole-sequence overall when there is no interleave);
      - sf_teacher_y [B,20,T_lat,H,W]: teacher i2v y (pixel-domain [first frame|zero frames] through the VAE + mask, recheck M8);
      - the standard keys existing train-loop code expects (x0/history/target_latents use the last section as a placeholder; the DMD path never consumes their contents).
    """
    B, C_lat, T_lat, H_lat, W_lat = latent.shape
    assert B == 1, f"[SF10S] train_batch_size=1 (matches the GEO constraint and simplifies segment mapping), got B={B}"
    P = int(prefix_sections)
    if sf_skip_full_encode:
        # the incoming latent is prefix only (T_lat == P*win); the full-clip VAE encode was skipped -> N/T_px come from the
        # training num_frames, no longer from the encoded tensor shape. n_section is set to P (the latent holds only the P prefix
        # sections), so the rearrange / placeholder choice_idx=n_section-1 below naturally lands on the prefix.
        assert full_clip_num_frames is not None, (
            "sf_skip_full_encode=True requires full_clip_num_frames to derive N/T_px"
        )
        full_T_lat = (full_clip_num_frames - 1) // 4 + 1
        full_n_section = full_T_lat // latent_window_size
        n_section = P
        N = full_n_section - P
    else:
        n_section = T_lat // latent_window_size
        full_T_lat = T_lat
        N = n_section - P
    assert P >= 1 and N >= 1, f"[SF10S] invalid section count: n_section={n_section}, prefix={P}"

    sf_prefix_latents = latent[:, :, : P * latent_window_size].contiguous()

    # ---- segment mapping: segment_prompts (pixel-frame ranges, already aligned to the clip window) -> per-generated-section caption + scoring segment stack ----
    segs_raw = batch.get("segment_prompts", [None])[0] if batch.get("segment_prompts") else None
    overall_caption = list(batch["prompt"])[0]
    T_px = (full_T_lat - 1) * 4 + 1   # use the full-clip latent length (when skipping, T_lat holds only the prefix and cannot be used)

    def _seg_of_px(px_frame):
        if segs_raw:
            for si, seg in enumerate(segs_raw):
                if int(seg["start_frame"]) <= px_frame < int(seg["end_frame"]):
                    return si
        return None

    # the section-caption mapping depends only on "how many latent frames the prefix takes": generated section k occupies latent [P_lat+k*win, P_lat+(k+1)*win).
    #   v2v: P_lat = P*win (=9) -> mid_lat = (P+k)*win + win//2, character-for-character the same as the old formula; i2v: P_lat = 1 -> mid_lat = 1+k*win+win//2.
    #   without parameterizing it, i2v steps would use the v2v midpoint (off by P*win-P_lat = 8 latent = 32 pixel frames, ~1.33s @24fps), diverging semantically from the
    #   teacher-side chunk_context_map (which follows the true latent ownership from _orig_to_latent).
    def _captions_for_prefix_latents(p_lat: int, tag: str):
        out = []
        for k in range(N):
            mid_lat = int(p_lat) + k * latent_window_size + latent_window_size // 2
            mid_px = min(mid_lat * 4, T_px - 1)
            si = _seg_of_px(mid_px)
            if segs_raw and si is None:
                print(f"[SF10S][warn]{tag} section {k} midpoint frame {mid_px} falls inside no segment (captions do not cover the clip tail?), "
                      f"student uses the overall caption while teacher chunk_context_map records segment0 -- the two sides may diverge semantically")
            out.append(segs_raw[si]["prompt"] if si is not None else overall_caption)
        return out

    per_section_caption = _captions_for_prefix_latents(P * latent_window_size, "")
    # the second copy, in i2v layout (computed only when the flag is on; its captions always belong to score_captions + {overall} => no new entry in uniq =>
    #   the T5 batch composition and padding are completely unchanged => embeds on the old path stay value-identical).
    per_section_caption_i2v = (
        _captions_for_prefix_latents(int(sf_i2v_prefix_latent_frames), "[i2v]")
        if int(sf_i2v_prefix_latent_frames) > 0 else None
    )

    # scoring segment stack: with interleave use its segment ranges (clipped into the clip), otherwise a single overall segment.
    if segs_raw:
        score_captions = [seg["prompt"] for seg in segs_raw]
        sf_segment_frame_ranges = [
            (max(0, int(seg["start_frame"])), min(T_px, int(seg["end_frame"]))) for seg in segs_raw
        ]
    else:
        score_captions = [overall_caption]
        sf_segment_frame_ranges = [(0, T_px)]

    # ---- T5 encoding (deduplicated, one encode) ----
    # the i2v copy is appended at the **end**: its elements already appear in score_captions/overall => dict.fromkeys adds no entry and
    #   the prefix order is unchanged => with the flag off or on, embeds of the old keys stay value-identical (the append is only a defensive guard against KeyError).
    uniq = list(dict.fromkeys(
        per_section_caption + score_captions + [overall_caption] + (per_section_caption_i2v or [])))
    from evoke.utils import sf_prep_profile as _pp          # zero overhead when off
    _t = _pp.mark()
    uniq_embeds, _ = encode_prompt(
        tokenizer=tokenizer, text_encoder=text_encoder, prompt=uniq, device=device, dtype=weight_dtype,
    )  # [len(uniq), L, D]
    _pp.accum("t5", _t, f"{len(uniq)} captions")
    emb_of = {c: uniq_embeds[i : i + 1] for i, c in enumerate(uniq)}
    sf_prompt_embeds_list = [emb_of[c] for c in per_section_caption]
    sf_prompt_embeds_list_i2v = (
        [emb_of[c] for c in per_section_caption_i2v] if per_section_caption_i2v is not None else None
    )
    sf_score_prompt_embeds = torch.stack([emb_of[c] for c in score_captions], dim=1)  # [1,S,L,D]
    prompt_embeds = emb_of[overall_caption]

    # ---- teacher i2v y: pixel-domain [first frame | zero frames x (T_px-1)] through the VAE (a zero pixel frame latent != a zero latent, recheck M8) ----
    # evoke-tier scoring does not consume y -> sf_build_teacher_y=False skips it (avoids importing the vendored evoke_teacher model).
    sf_teacher_y = None
    if sf_build_teacher_y:
        from evoke.dataset.cond_y_fastpath import cond_y_latent as _cond_y_fast
        from evoke.modules.evoke_teacher.wrapper import build_i2v_y
        _T_px_y = int(raw_video.shape[2])
        _t = _pp.mark()
        # NOTE: fast path: encode only the 113-frame prefix + concatenate the constant tail computed once at startup; **bit-identical** to full-length encoding.
        #   this used to be `vae.encode(zeros_like(raw_video) with the first frame filled in)` -- 752 of the 753 frames are zeros, measured 15.8s/step.
        #   basis: Wan-VAE is strictly causal in time and R=112 => latent j>=29 is independent of the image and bit-identical to each other (see
        # the docstring of cond_y_fastpath.py and).
        #   `SF_CONDY_VERIFY=1` additionally runs a full-length encode for a bit-exact comparison (on for smoke, off for formal runs).
        with torch.no_grad():
            cond_lat = _cond_y_fast(vae, raw_video, _T_px_y)
        _pp.accum("vae_cond_y", _t, f"fast path {_cond_y_fast.__module__.split('.')[-1]}: "
                                    f"encoded 113 frames + constant tail (originally {_T_px_y} frames)")
        cond_px = raw_video          # only its shape is used by the probe below (no longer builds a 753-frame zero tensor)
        if os.environ.get("SF_VAE_RF_PROBE") == "1" and not getattr(_pp, "_rf_done", False):
            _pp._rf_done = True
            # NOTE: try/except is mandatory: on this probe was itself buggy (flatten collapsed the time dim -> IndexError)
            #   and killed the whole 8-GPU smoke. **diagnostic code may never kill training** -- on error just print one line and move on.
            try:
                print(_pp.vae_rf_probe(vae, raw_video, int(cond_px.shape[2]),
                                       latents_mean, latents_std), flush=True)
            except Exception as _e:
                print(f"[VAE-RF-PROBE] the probe itself errored, skipped (training unaffected): {type(_e).__name__}: {_e}", flush=True)
        cond_lat = ((cond_lat - latents_mean) * latents_std).to(dtype=weight_dtype)
        sf_teacher_y = build_i2v_y(cond_lat, num_cond_px_frames=1)

    # ---- latent for the student history 1x slot on i2v steps ----
    # On the inference side the i2v 1x slot is not a single-frame encode of the reference image but
    #   VAE(reference image repeated 33 frames)[..., -1:] (the pipeline's fake_image_latents): a
    #   **continuation-distribution** latent, from the same distribution the short tier saw in stage1/2, while the
    #   x0 anchor is the single-frame I-frame latent. Training copies that recipe exactly so the student input on
    #   i2v steps lines up slot by slot with i2v inference. Flag off -> always None, no VAE call.
    #   (the "iframe" mode is banned by the validator: it would make 1x I-frame distributed.)
    # `sf_i2v_ratio > 0` is part of the gate because this is the **video sample** path: at ratio=0 video samples
    #   always take v2v, so train_evoke's `_sf_i2v_hist_latent` is None on those steps and this [B,16,1,h,w] tensor
    #   has no reader at all -- utils_evoke_post asserts `_hist_tail.shape[2] == prefix_latents.shape[2]` (1 vs 9),
    #   so it is structurally unusable for v2v. No slot distribution changes either way: image-only samples use
    #   their own copy from materialize_i2v_image_only, and when ratio>0 this encode runs as before.
    sf_i2v_hist_latent = None
    if (int(sf_i2v_prefix_latent_frames) > 0 and str(sf_i2v_hist_latent_mode) == "static_repeat"
            and float(sf_i2v_ratio) > 0.0):
        _min_f = (int(latent_window_size) - 1) * 4 + 1          # 33 @ win=9, same formula as inference
        _static_px = raw_video[:, :, :1].repeat(1, 1, _min_f, 1, 1)
        _t = _pp.mark()
        with torch.no_grad():
            # same deterministic .mode() encoding as teacher y (conditioning tensors should carry no sampling noise)
            _static_lat = vae.encode(_static_px.to(vae.dtype)).latent_dist.mode()
        _pp.accum("vae_hist1x", _t, f"{_static_px.shape[2]} frames")
        _static_lat = ((_static_lat - latents_mean) * latents_std).to(dtype=weight_dtype)
        sf_i2v_hist_latent = _static_lat[:, :, -1:].contiguous()   # [B,C,1,H,W]

    # ---- compatibility placeholders (last section as target). The DMD path does not read their contents; this
    #      only keeps shapes and dataflow from crashing. ----
    # With `sf_gt_partial=True`, `latent` is the on-demand encoded GT prefix: non-GEO steps have T_lat = 9, which
    #   matches n_section=P=1, but GEO steps have T_lat = 9(j+1) = 18..180 != 9, so a plain rearrange raises
    #   `EinopsError: 18 != 9` (the skip branch hardcodes n_section to P, assuming the incoming latent holds only
    #   the prefix). The fix touches only the placeholder: take the first n_section*win frames. n_section itself is
    #   left alone, so the section picked by choice_idx=n_section-1 and the history/target contents do not change.
    #   Note this is not the same content as with sf_gt_partial off (there the full-length path gives n_section=21
    #   and the placeholder comes from section 20, versus n_section=1 taken from the prefix here). That is harmless
    #   only because nothing reads them: under is_train_dmd latents_history_* is set to None wholesale, model_input
    #   is consumed only by is_use_gan (false), and the GEO-train block is skipped since the batch has no
    #   warp_video_latents key.
    #   Sliced only when sf_gt_partial: on the other paths n_section*win == T_lat so the slice is a no-op, but the
    #   explicit gate keeps a real error such as "T_lat is not divisible by win" from being silently truncated.
    _lat_ph = latent[:, :, : n_section * latent_window_size] if sf_gt_partial else latent
    latent_sec = rearrange(_lat_ph, "b c (n w) h s -> b n c w h s", n=n_section, w=latent_window_size)
    x0, hist, tgt = prepare_stage1_latent_v2(
        latent_sec[0], history_sizes=history_sizes, choice_idx=n_section - 1, is_keep_x0=is_keep_x0,
    )

    return {
        "prompt_embeds": prompt_embeds,
        "prompt_attention_masks": None,
        "x0_latents": x0.unsqueeze(0) if is_keep_x0 else None,
        "history_latents": hist.unsqueeze(0),
        "target_latents": tgt.unsqueeze(0),
        "clean_all_latents": None,
        "prompt": [overall_caption],
        "uttid": batch.get("uttid"),
        "dataset_name": batch.get("dataset_name"),
        "bucket_key": batch.get("bucket_key"),
        # dedicated keys
        "sf_prefix_latents": sf_prefix_latents,
        # full-length GT latent sequence (scaled, same source and scale as the prefix): used during teacher scoring to
        # swap the long/mid tiers for same-timeline GT (sf_teacher_gt_longmid). ~40MB/batch; without the flag it is only transferred, never consumed.
        # when the full-clip encode is skipped the latent holds only the prefix and cannot act as full-length GT -> None (must be paired with sf_teacher_gt_longmid=false).
        # with sf_gt_partial=True the latent is the "on-demand encoded prefix" (still real GT latent, bit-identical to the same span of
        #   the full-length encode) => handed to GEO-REG as sf_gt_latents as usual; only skip_full_encode (the v2.1 dual-teacher path) sets None.
        "sf_gt_latents": None if (sf_skip_full_encode and not sf_gt_partial) else latent,
        "sf_prompt_embeds_list": sf_prompt_embeds_list,
        "sf_score_prompt_embeds": sf_score_prompt_embeds,
        "sf_teacher_y": sf_teacher_y,
        "sf_segment_frame_ranges": sf_segment_frame_ranges,
        "sf_num_generated_sections": N,
        # keys dedicated to i2v mixed training (flag off -> always None, just two extra Nones passed along):
        #   sf_prompt_embeds_list_i2v = section prompts recomputed for the P_lat=1 layout (the v2v copy is kept as is, for v2v steps);
        #   sf_i2v_hist_latent      = static-repeat last latent for the history 1x slot on i2v steps (inference recipe).
        #   the prefix itself **adds no key**: on i2v steps the train loop slices sf_prefix_latents[:, :, :P_lat] (latent frame 0 =
        #   pixel frame 0 = the reference image, encoded independently by Wan-VAE) => dataflow / mixing ratio / require_full_length all unchanged.
        "sf_prompt_embeds_list_i2v": sf_prompt_embeds_list_i2v,
        "sf_i2v_hist_latent": sf_i2v_hist_latent,
        # video samples: v2v by default; the train loop decides whether to switch to i2v according to sf_i2v_ratio.
        "sf_sample_is_i2v": False,
        # warp-in-rollout uses the full-length GT pose (recheck MUST-FIX#1: the old early return dropped pose;
        # None when there is no pose source -> the warp-off path is unaffected)
        "sf_pose_Ks": batch.get("lingbot_Ks"),
        "sf_pose_c2ws": batch.get("lingbot_c2ws"),
    }


def materialize_i2v_image_only(
    batch: dict,
    raw_video: "torch.Tensor",        # [B,3,1,H,W] in [-1,1] -- single-frame reference image
    vae,
    tokenizer,
    text_encoder,
    latents_mean: "torch.Tensor",
    latents_std: "torch.Tensor",
    latent_window_size: int,
    history_sizes,
    device,
    weight_dtype,
    is_keep_x0: bool = True,
    num_generated_sections: int = 20,
    prefix_latent_frames: int = 1,
    hist_latent_mode: str = "static_repeat",
    build_teacher_y: bool = True,
):
    """image-only sample -> SF10S batch in i2v layout (key-for-key isomorphic to a video sample taking i2v).

    The only structural difference from a video sample's i2v step: there is **no GT clip** here, so
      - sf_gt_latents = None   -> GEO-REG warns + skips automatically (it is a pure GT regularizer)
      - sf_pose_*     = None   -> the tail-section warp is not built (no pose)
      - there is only one section prompt (the inline prompt from the jsonl), reused for every section
    Everything else (prefix / 1x slot static-repeat / teacher y / frame accounting) is exactly as in a video sample's i2v step:
      scoring sequence = P_lat + N*win, teacher y is composed from [image | zero pixel frames] (build_i2v_y only ever looks at frame 0).

    NOTE: cost: 2 VAE encodes -- 33 frames (static-repeat) + T_px_i2v frames (y's [image|zero frames]). the latter costs exactly the same
      as building y for a video sample (which also uses zeros_like(raw_video) with the first frame filled in), and it **skips the full real-video encode** =>
      image-only samples are cheaper than video samples.
    """
    B, C_pix, T_pix, H_pix, W_pix = raw_video.shape
    assert T_pix == 1, f"[LW-I2V-SAMPLE] single-frame input required, got T_pix={T_pix}"
    assert B == 1, f"[LW-I2V-SAMPLE] B=1 required, got {B}"
    win = int(latent_window_size)
    N = int(num_generated_sections)
    P_lat = int(prefix_latent_frames)
    assert N >= 1 and P_lat >= 1, f"[LW-I2V-SAMPLE] invalid N={N} P_lat={P_lat}"
    T_lat_i2v = P_lat + N * win                     # 181 @ P_lat=1,N=20,win=9
    T_px_i2v = (T_lat_i2v - 1) * 4 + 1              # 721
    Hl, Wl = H_pix // 8, W_pix // 8
    img = raw_video[:, :, :1]                       # [B,3,1,H,W]

    from evoke.utils import sf_prep_profile as _ppi          # zero overhead when off

    def _enc(px):
        _t = _ppi.mark()
        with torch.no_grad():
            # same deterministic .mode() encoding as teacher y (conditioning tensors should carry no sampling noise)
            z = vae.encode(px.to(vae.dtype)).latent_dist.mode()
        # of the three i2v encodes the 721-frame one (teacher y) dominates by far -- same nature as the 2nd encode on the v2v path.
        _ppi.accum(f"vae_i2v_{px.shape[2]}f", _t, f"{px.shape[2]} frames")
        return ((z - latents_mean) * latents_std).to(dtype=weight_dtype)

    # ---- prefix: single-frame latent of the reference image (I-frame distribution) = student x0 anchor + frame 0 of the scoring sequence (the teacher's sink) ----
    sf_prefix_latents = _enc(img)[:, :, :P_lat].contiguous()
    assert sf_prefix_latents.shape[2] == P_lat, (
        f"[LW-I2V-SAMPLE] the single-frame encode yields only {sf_prefix_latents.shape[2]} latents, need {P_lat}")

    # ---- history 1x slot: static-repeat last latent (continuation distribution), same formula as fake_image_latents in i2v inference ----
    sf_i2v_hist_latent = None
    if str(hist_latent_mode) == "static_repeat":
        _min_f = (win - 1) * 4 + 1                  # 33
        sf_i2v_hist_latent = _enc(img.repeat(1, 1, _min_f, 1, 1))[:, :, -1:].contiguous()

    # ---- teacher i2v y: pixel-domain [image | zero frames x (T_px_i2v-1)] through the VAE (a zero pixel frame latent != a zero latent, recheck M8) ----
    sf_teacher_y = None
    if build_teacher_y:
        from evoke.dataset.cond_y_fastpath import cond_y_latent as _cond_y_fast
        from evoke.modules.evoke_teacher.wrapper import build_i2v_y
        # NOTE: same as the v2v path: encode only 113 frames + the constant tail, **bit-identical** to encoding all T_px_i2v (=721) frames.
        #   measured: this one encode used to take 15.4s = 95% of the entire i2v prep.
        _t = _ppi.mark()
        with torch.no_grad():
            _cond_lat = _cond_y_fast(vae, img, int(T_px_i2v))
        _cond_lat = ((_cond_lat - latents_mean) * latents_std).to(dtype=weight_dtype)
        _ppi.accum("vae_i2v_condy_fast", _t, f"encoded 113 frames + constant tail (originally {T_px_i2v} frames)")
        sf_teacher_y = build_i2v_y(_cond_lat, num_cond_px_frames=1)
        assert sf_teacher_y.shape[2] == T_lat_i2v, (
            f"[LW-I2V-SAMPLE] y frame count {sf_teacher_y.shape[2]} != scoring sequence {T_lat_i2v}")

    # ---- prompt: a single inline caption, reused for every section (image samples have no segments) ----
    overall_caption = list(batch["prompt"])[0]
    _emb, _ = encode_prompt(
        tokenizer=tokenizer, text_encoder=text_encoder, prompt=[overall_caption],
        device=device, dtype=weight_dtype,
    )
    prompt_embeds = _emb[0:1]
    sf_prompt_embeds_list = [prompt_embeds for _ in range(N)]

    # ---- standard compatibility placeholders (the DMD path does not read their contents, only shapes/flow must hold; all zeros, no VAE cost) ----
    _n_ph = 1 + N
    _ph = torch.zeros(1, sf_prefix_latents.shape[1], _n_ph * win, Hl, Wl,
                      device=sf_prefix_latents.device, dtype=sf_prefix_latents.dtype)
    _ph_sec = rearrange(_ph, "b c (n w) h s -> b n c w h s", n=_n_ph, w=win)
    x0, hist, tgt = prepare_stage1_latent_v2(
        _ph_sec[0], history_sizes=history_sizes, choice_idx=_n_ph - 1, is_keep_x0=is_keep_x0,
    )

    return {
        "prompt_embeds": prompt_embeds,
        "prompt_attention_masks": None,
        "x0_latents": x0.unsqueeze(0) if is_keep_x0 else None,
        "history_latents": hist.unsqueeze(0),
        "target_latents": tgt.unsqueeze(0),
        "clean_all_latents": None,
        "prompt": [overall_caption],
        "uttid": batch.get("uttid"),
        "dataset_name": batch.get("dataset_name"),
        "bucket_key": batch.get("bucket_key"),
        # dedicated keys (i2v layout)
        "sf_prefix_latents": sf_prefix_latents,
        "sf_gt_latents": None,                       # no GT clip => GEO-REG / gt-anchor skip automatically
        "sf_prompt_embeds_list": sf_prompt_embeds_list,
        "sf_prompt_embeds_list_i2v": sf_prompt_embeds_list,   # the same copy (this sample is always i2v)
        "sf_score_prompt_embeds": prompt_embeds.unsqueeze(1),  # [B,1,L,D] single segment
        "sf_teacher_y": sf_teacher_y,
        "sf_segment_frame_ranges": [(0, T_px_i2v)],
        "sf_num_generated_sections": N,
        "sf_i2v_hist_latent": sf_i2v_hist_latent,
        # NOTE: this sample can **only** take i2v (no temporal GT / no pose): the train loop dispatches on this and it never enters sf_i2v_ratio sampling.
        "sf_sample_is_i2v": True,
        "sf_pose_Ks": None,                          # no pose => the tail-section warp is not built
        "sf_pose_c2ws": None,
    }


def materialize_online_batch(
    batch: dict,
    vae,
    tokenizer,
    text_encoder,
    history_sizes,
    latent_window_size: int,
    latents_mean: torch.Tensor,
    latents_std: torch.Tensor,
    device: torch.device,
    weight_dtype: torch.dtype,
    is_keep_x0: bool = True,
    seed: int = 42,
    epoch: int = 0,
    # geometric state training; when enabled adds warp_video_latents/warp_visibility_mask fields.
    use_geometric_state: bool = False,
    # v20: overwrite warp[0] with the immediate previous frame (clean seam anchor) + mask[0]=1.
    geo_keep_clean_anchor: bool = False,
    # FrameBank retrieve config; None = defaults.
    geo_retrieve_cfg=None,
    # Per-sample GEO conditioning mode sampling ratios (t2v / i2v / full_geo).
    geo_condition_t2v_ratio: float = 0.0,
    geo_condition_i2v_ratio: float = 0.0,
    # DA3 cloud-warp config (P2); when enabled, full_geo warp comes from the DA3 known-pose point cloud
    # instead of Pi3X retrieve. None / disabled = legacy Pi3X path. Object with attrs (CloudWarpConfig).
    geo_cloud_warp_cfg=None,
    # Spatial visibility-aware noise; False = frame-uniform sigma (default).
    geo_visibility_aware_noise: bool = False,
    geo_sigma_invisible: float = 0.8,
    geo_sigma_visible_min: float = 0.111,
    geo_sigma_visible_max: float = 0.135,
    # Error-bank warp injection (err-then-noise). When enabled, samples a low-noise y_error from the
    # recycle bank and adds it to the CLEAN warp latent BEFORE visibility noising. None bank => no-op.
    recycle_vars=None,
    args=None,
    geo_warp_error_inject_enabled: bool = False,
    geo_warp_error_prob: float = 0.0,
    # Train-only rigid pose jitter on the warp RENDER target poses (warp = rough reference, not 1:1 copy).
    # When enabled+triggered, a single camera-frame rigid DeltaT is right-multiplied onto a CLONE of the
    # per-sample target poses fed to the warp renderer ONLY; the returned target_pose_c2ws (plucker + loss)
    # is never mutated. None / disabled => exact original GT-pose path. Object with attrs (WarpPoseJitterConfig).
    geo_pose_jitter_cfg=None,
    # full-sequence rollout + interleave caption mode (use_full_rollout_interleave); returns early when triggered.
    sf_full_rollout_interleave: bool = False,
    sf_prefix_sections: int = 1,
    sf_build_teacher_y: bool = True,   # pass False for evoke-tier scoring (y is consumed by evoke_teacher only)
    # with dual teachers and gt-anchor off: encode only the prefix pixel window, skipping the useless full-clip VAE encode of the N-P sections.
    # default OFF -> the original full-clip encoding path is bit-identical. full_clip_num_frames = the training num_frames (downstream derives N/T_px).
    sf_skip_full_encode: bool = False,
    full_clip_num_frames: Optional[int] = None,
    # when >0, additionally produce the i2v-layout section prompts + 1x slot latent (see materialize_full_rollout_interleave).
    #   0 = off -> nothing extra is computed, old path bit-identical.
    sf_i2v_prefix_latent_frames: int = 0,
    sf_i2v_hist_latent_mode: str = "static_repeat",
    # probability that a video sample takes i2v. at 0 video samples are always v2v => that 33-frame 1x latent has no reader at all
    #   => it is skipped below. image-only samples ignore this (they compute their own inside materialize_i2v_image_only).
    sf_i2v_ratio: float = 0.0,
    # up to which pixel frame this step needs GT encoded (None=full clip=old behaviour). computed by train_evoke from
    #   (whether GEO-REG is needed, which j was drawn); since the VAE is strictly causal => prefix encoding is bit-identical to full length.
    sf_gt_encode_px: Optional[int] = None,
    # image-only samples have no clip to derive N from (video samples use T_lat//win - P), so the caller must pass it
    #   (= dmd_num_latent_sections_max on the fixed-N path; the validator already guarantees min==max).
    sf_num_generated_sections: int = 0,
):
    """Materialize raw_video + prompt into all latent fields needed by the stage-1 training loop.

    Args:
        batch: dict with "raw_video" `[B, C, T_pix, H_pix, W_pix]` fp32 in [-1, 1] and "prompt" list[str].
        vae: AutoencoderKLWan, already on device.
        latents_mean / latents_std: shape `[1, z_dim, 1, 1, 1]`; std is 1/raw_std.

    Returns:
        dict with fields matching offline BucketedFeatureDataset.__getitem__:
            prompt_embeds, prompt_attention_masks,
            x0_latents [B, C_lat, 1, H_lat, W_lat],
            history_latents [B, C_lat, history_window, H_lat, W_lat],
            target_latents [B, C_lat, latent_window, H_lat, W_lat],
            clean_all_latents (None for stage-1),
            uttid / dataset_name / bucket_key / num_frame / height / width.
    """
    raw_video = batch["raw_video"].to(device=device, dtype=vae.dtype, non_blocking=True)
    B, C_pix, T_pix, H_pix, W_pix = raw_video.shape

    # ── NOTE: image-only sample -> take the i2v branch (dispatched per **sample**, not drawn by ratio) ──
    #   the data config mounts both video sources (scene/gameverse/dl3dv, with time) and image sources (vbench2_i2v: jpg -> LoadImage
    #   >> ToList -> **1 frame**). image samples cannot supply the num_frames temporal supervision and have no pose => they can only be i2v:
    #   reference image as the condition, student rolls out N sections, teacher scores exactly those N sections. DMD is a distribution-matching loss
    #   (the teacher scores the student rollout, no GT target needed) => image-only samples are **complete** for DMD, they only lose the GT-dependent
    #   extras GEO-REG / warp / gt-anchor (each of which skips automatically through its own None gate).
    #   NOTE: rank consistency: the 8 GPUs of an SP group share the same index => the same sample => the same branch, **zero collective communication**.
    #   threshold: a sample that cannot even fill one chunk (win latent frames = (win-1)*4+1 pixel frames) = image-only.
    _px_per_chunk = (int(latent_window_size) - 1) * 4 + 1
    if sf_full_rollout_interleave and T_pix < _px_per_chunk:
        # NOTE: the error must pinpoint **which sample**: these three asserts live in the training loop, where the dataset-level same-source
        #   resampling (data_config.ConfigAwareDataset) cannot catch them => one trigger crashes the whole 48-GPU job. printing only T_pix
        #   tells you "some sample has T_pix=17" but not which source or which line. uttid/dataset_name are right there in the batch.
        def _who():
            def _one(k):
                v = batch.get(k)
                return (v[0] if isinstance(v, (list, tuple)) and v else v)
            return f"uttid={_one('uttid')} dataset={_one('dataset_name')} bucket={_one('bucket_key')}"

        assert T_pix == 1, (
            f"[LW-I2V-SAMPLE] only 1-frame image samples are supported, got T_pix={T_pix} (<{_px_per_chunk} but >1): "
            f"filter short video samples out with require_full_length, or extend this branch explicitly. sample: {_who()}")
        assert B == 1, f"[LW-I2V-SAMPLE] train_batch_size=1 required (same constraint as SF10S), got B={B}. sample: {_who()}"
        assert sf_i2v_prefix_latent_frames > 0, (
            "[LW-I2V-SAMPLE] the data contains image-only samples but sf_i2v_ratio=0 (i2v path not enabled) => "
            "sf_i2v_prefix_latent_frames=0, so no i2v conditioning can be built for them. either enable the i2v path or drop the image source from select. "
            f"sample: {_who()}")
        return materialize_i2v_image_only(
            batch, raw_video=raw_video, vae=vae, tokenizer=tokenizer, text_encoder=text_encoder,
            latents_mean=latents_mean, latents_std=latents_std,
            latent_window_size=latent_window_size, history_sizes=history_sizes,
            device=device, weight_dtype=weight_dtype, is_keep_x0=is_keep_x0,
            num_generated_sections=int(sf_num_generated_sections),
            prefix_latent_frames=int(sf_i2v_prefix_latent_frames),
            hist_latent_mode=str(sf_i2v_hist_latent_mode),
            build_teacher_y=bool(sf_build_teacher_y),
        )

    # with dual teachers and gt-anchor off: skip the full-clip VAE encode, encode only the prefix pixel window
    # P_px = (P*win - 1)*4 + 1, saving the useless encoding/memory of the N-P sections; N/T_px are derived downstream from full_clip_num_frames.
    # default OFF (triggers only when both conditions hold) -> the full-clip encoding path below is bit-identical.
    if sf_full_rollout_interleave and sf_skip_full_encode:
        assert full_clip_num_frames is not None, (
            "sf_skip_full_encode=True requires full_clip_num_frames"
        )
        P = int(sf_prefix_sections)
        P_px = (P * latent_window_size - 1) * 4 + 1
        with torch.no_grad():
            prefix_latent = vae.encode(raw_video[:, :, :P_px]).latent_dist.sample()
        prefix_latent = (prefix_latent - latents_mean) * latents_std
        prefix_latent = prefix_latent.to(dtype=weight_dtype)
        return materialize_full_rollout_interleave(
            batch, latent=prefix_latent, raw_video=raw_video, vae=vae,
            tokenizer=tokenizer, text_encoder=text_encoder,
            latents_mean=latents_mean, latents_std=latents_std,
            latent_window_size=latent_window_size, prefix_sections=sf_prefix_sections,
            history_sizes=history_sizes, device=device, weight_dtype=weight_dtype,
            is_keep_x0=is_keep_x0,
            sf_build_teacher_y=sf_build_teacher_y,
            sf_skip_full_encode=True,
            full_clip_num_frames=full_clip_num_frames,
            sf_i2v_prefix_latent_frames=sf_i2v_prefix_latent_frames,
            sf_i2v_hist_latent_mode=sf_i2v_hist_latent_mode,
            sf_i2v_ratio=sf_i2v_ratio,
        )

    # encode on demand only the pixel prefix that is needed -- **bit-identical** to the full-length `.sample()`,
    # RNG stream included.
    #
    # Why `vae.encode(prefix).latent_dist.sample()` is not enough: the baseline encodes the full length, and
    #   diffusers draws `sample = randn_tensor(self.mean.shape, ...)` then `x = mean + std * sample`, so the shape
    #   of the noise is the full-length [B,16,T_lat_full,h,w]. A prefix encode would draw only [B,16,9,h,w], consume
    #   a different count of random values and shift the whole RNG stream -- same distribution, not bit-identical.
    #
    # Approach: compute mu/sigma from the prefix encode, draw eps at the **full-length shape** (same randn_tensor,
    #   same device/dtype/order), then take the prefix: z = mu_part + sigma_part * eps_full[:, :, :n]. As long as
    #   mu_part == mu_full[:, :, :n] and sigma_part == sigma_full[:, :, :n] (the causal premise), z is bit-identical
    #   to the baseline's z_full[:, :, :n] and the RNG state advances by exactly as much. The extra eps tail is
    #   dropped on the spot -- the baseline draws those numbers too, so it is not even added overhead.
    from evoke.utils import sf_prep_profile as _ppc
    if sf_gt_encode_px is not None and int(sf_gt_encode_px) < T_pix and not _GT_OD["off"]:
        _gt_px = max(1, int(sf_gt_encode_px))
        _T_lat_full = 1 + (int(T_pix) - 1) // 4
        # key=_gt_px => the first N calls are verified for every prefix length (the 33 frames of non-GEO steps / the 36j+33 frames of each GEO step counted separately).
        # with one global counter the first 3 calls would all land on non-GEO steps during the freeze period => long-prefix shapes would never get verified.
        _do_verify = os.environ.get("SF_GT_VERIFY") == "1" and _ppc.gt_verify_take(int(_gt_px))
        _dev = raw_video.device
        # save the RNG state from **before** ε is drawn: the check rewinds here to recompute once with the baseline recipe; the breaker also needs it so the
        # fallback path draws the same ε as the baseline (otherwise one trip would skew the RNG stream permanently).
        _st_cpu = torch.get_rng_state()
        _st_cu = torch.cuda.get_rng_state(_dev) if (_dev.type == "cuda") else None
        _tc = _ppc.mark()
        with torch.no_grad():
            _dist = vae.encode(raw_video[:, :, :_gt_px]).latent_dist   # VAE encode consumes no random numbers
            _mu, _sd = _dist.mean, _dist.std
            _n = int(_mu.shape[2])
            _eps = randn_tensor(                                       # <- **the same function and the same shape** as the baseline
                (_mu.shape[0], _mu.shape[1], _T_lat_full, _mu.shape[3], _mu.shape[4]),
                generator=None, device=_dist.parameters.device, dtype=_dist.parameters.dtype)
            _part_raw = _mu + _sd * _eps[:, :, :_n]
            del _eps                    # the full-length ε (46MB fp32) only aligns RNG consumption; free it right after use
        _part = ((_part_raw - latents_mean) * latents_std).to(dtype=weight_dtype)
        _ppc.accum("vae_gt_ondemand", _tc, f"{_gt_px} frames -> {_n} lat (full clip {T_pix} -> {_T_lat_full} lat)")
        # the RNG state after path C = the state after the baseline (same shapes, same order) => the check must restore to here when done.
        _st_cpu_after = torch.get_rng_state()
        _st_cu_after = torch.cuda.get_rng_state(_dev) if (_dev.type == "cuda") else None

        if _do_verify:
            # NOTE: the checks (three of them, all required to be strictly 0; on failure it does **not** raise, it trips the breaker and falls back):
            #   1. per-latent-frame μ Δ=0   2. per-latent-frame σ Δ=0   -- these two are the premise of "truncation does not change already produced chunks"
            #   3. after rewinding the RNG, the baseline-recipe `encode(full length).sample()` is bit-identical to _part_raw on the first n frames
            #      -- the strongest one: it directly proves "C's output == the baseline's output", down to the very same batch of ε.
            #   WARNING: never compare two `.sample()` calls (the first version's mistake): each draws its own ε, so they structurally cannot match; max|Δ| there measures sqrt(2)*σ.
            _fail = None
            try:
                torch.set_rng_state(_st_cpu)
                if _st_cu is not None:
                    torch.cuda.set_rng_state(_st_cu, _dev)
                with torch.no_grad():
                    _fd = vae.encode(raw_video).latent_dist
                    _full_raw = _fd.sample()                            # baseline recipe, drawing the very same batch of ε
                _dmu = (_fd.mean[:, :, :_n] - _mu).abs().amax(dim=(0, 1, 3, 4))
                _dsd = (_fd.std[:, :, :_n] - _sd).abs().amax(dim=(0, 1, 3, 4))
                _dz = (_full_raw[:, :, :_n] - _part_raw).abs().amax(dim=(0, 1, 3, 4))
                # NOTE: release the full-length distribution object immediately: _fd carries parameters+mean+logvar+std+var (fp32, 189x60x104)
                #   plus _full_raw ~ 0.5GB. they are locals of materialize_online_batch, so without del they stay
                #   alive until the function returns, spanning the later T5 / cond-y / i2v encodes -- a real risk against the 138/142GB wall.
                #   in the baseline that dist statement frees them as soon as it ends, so this is memory the check **adds** and must clean up itself.
                del _fd, _full_raw
                _bad = {k: [i for i in range(_n) if float(v[i]) != 0.0]
                        for k, v in (("μ", _dmu), ("σ", _dsd), ("z", _dz))}
                _fmt = lambda v: "[" + " ".join(f"{float(v[i]):.3e}" for i in range(_n)) + "]"
                _detail = (f"encoded {_gt_px} frames -> {_n} latent (full clip {T_pix} -> {_T_lat_full} latent) | "
                           f"per-frame max|Δ|: μ={_fmt(_dmu)} σ={_fmt(_dsd)} z={_fmt(_dz)}")
                if any(_bad.values()):
                    _fail = (f"frames where μ differs={_bad['μ']} where σ differs={_bad['σ']} where z differs={_bad['z']}\n"
                             f"    {_detail}\n"
                             f"    => 'the VAE is strictly causal in time, truncating the input does not change already produced chunks' is refuted"
                             f" (differences concentrated in the last few frames => the receptive field is longer than R=112; uniform differences over all frames => there is another non-causal operator)")
                else:
                    print(f"[GT-VERIFY] OK bit-identical to the baseline (μ/σ/z all 0): {_detail}", flush=True)
            except Exception as _e:
                print(f"[GT-VERIFY] the check itself errored, skipped (training unaffected): {type(_e).__name__}: {_e}", flush=True)
            finally:
                # whatever the checks conclude, restore the RNG -- on pass, back to the "C finished" state (identical to the baseline);
                # on failure the code below resets to `_st_cpu/_st_cu` so the fallback path draws the baseline's batch of ε.
                torch.set_rng_state(_st_cpu_after)
                if _st_cu_after is not None:
                    torch.cuda.set_rng_state(_st_cu_after, _dev)
            if _fail is not None:
                _GT_OD["off"] = True
                print(f"[GT-VERIFY] FAIL premise refuted -- **on-demand GT encoding is disabled for the rest of this run, falling back to full-length encoding**"
                      f" (training is not killed, it just goes back to 16s/step)\n    {_fail}", flush=True)
                torch.set_rng_state(_st_cpu)          # rewind so the full-length .sample() below draws the baseline's batch of ε
                if _st_cu is not None:
                    torch.cuda.set_rng_state(_st_cu, _dev)

    if sf_gt_encode_px is not None and int(sf_gt_encode_px) < T_pix and not _GT_OD["off"]:
        return materialize_full_rollout_interleave(
            batch, latent=_part, raw_video=raw_video, vae=vae,
            tokenizer=tokenizer, text_encoder=text_encoder,
            latents_mean=latents_mean, latents_std=latents_std,
            latent_window_size=latent_window_size, prefix_sections=sf_prefix_sections,
            history_sizes=history_sizes, device=device, weight_dtype=weight_dtype,
            is_keep_x0=is_keep_x0, sf_build_teacher_y=sf_build_teacher_y,
            sf_skip_full_encode=True, sf_gt_partial=True, full_clip_num_frames=T_pix,
            sf_i2v_prefix_latent_frames=sf_i2v_prefix_latent_frames,
            sf_i2v_hist_latent_mode=sf_i2v_hist_latent_mode,
            sf_i2v_ratio=sf_i2v_ratio,
        )

    # VAE encode full video and normalize.
    from evoke.utils import sf_prep_profile as _pp0          # zero overhead when off
    _t0 = _pp0.mark()
    with torch.no_grad():
        latent = vae.encode(raw_video).latent_dist.sample()
    _pp0.accum("vae_gt_full", _t0, f"{raw_video.shape[2]} real video frames")
    latent = (latent - latents_mean) * latents_std
    latent = latent.to(dtype=weight_dtype)

    _, C_lat, T_lat, H_lat, W_lat = latent.shape
    assert T_lat % latent_window_size == 0, (
        f"VAE output T_lat={T_lat} must be divisible by latent_window_size={latent_window_size}. "
        f"check num_frames in the training yaml (= (latent_window * n_section - 1) * 4 + 1)."
    )
    n_section = T_lat // latent_window_size

    # full-sequence rollout mode returns early (it does not take the single-window choice_idx path).
    if sf_full_rollout_interleave:
        return materialize_full_rollout_interleave(
            batch, latent=latent, raw_video=raw_video, vae=vae,
            tokenizer=tokenizer, text_encoder=text_encoder,
            latents_mean=latents_mean, latents_std=latents_std,
            latent_window_size=latent_window_size, prefix_sections=sf_prefix_sections,
            history_sizes=history_sizes, device=device, weight_dtype=weight_dtype,
            is_keep_x0=is_keep_x0,
            sf_build_teacher_y=sf_build_teacher_y,
            sf_i2v_prefix_latent_frames=sf_i2v_prefix_latent_frames,
            sf_i2v_hist_latent_mode=sf_i2v_hist_latent_mode,
            sf_i2v_ratio=sf_i2v_ratio,
        )

    # Rearrange into section view for prepare_stage1_latent_v2.
    latent_sec = rearrange(latent, "b c (n w) h s -> b n c w h s", n=n_section, w=latent_window_size)

    # Per-sample segment prompts from dataloader (interleave_caption_5 mode); None otherwise.
    segs_per_sample = batch.get("segment_prompts", [None] * B)
    if segs_per_sample is None:
        segs_per_sample = [None] * B
    # Skill (grok event/VFX) sample: drop-warp + event-window target. Default OFF → identical to the
    # original path for every non-skill (scene/gameverse/dl3dv) sample. GEO forces B==1, so is_skill /
    # event_window arrive as length-B lists; read element 0. event_window is in LOADED-frame (pixel) space.
    _is_skill_raw = batch.get("is_skill", False)
    _is_skill = bool(_is_skill_raw[0]) if isinstance(_is_skill_raw, (list, tuple)) and _is_skill_raw else bool(_is_skill_raw)
    _event_win_raw = batch.get("event_window", None)
    _event_win = _event_win_raw[0] if isinstance(_event_win_raw, (list, tuple)) and _event_win_raw else _event_win_raw
    # Build per-sample latents; for interleave mode, align choice_idx to a matching segment.
    x0_list, hist_list, tgt_list = [], [], []
    sample_prompts = list(batch["prompt"])   # defaults to overall caption; replaced by matched segment below
    choice_idx_per_sample = [None] * B    # for downstream pose slicing
    for b in range(B):
        g = torch.Generator().manual_seed(seed + epoch * 1_000_003 + b)
        initial_choice_idx = int(torch.randint(0, n_section, (1,), generator=g).item())
        segs = segs_per_sample[b] if b < len(segs_per_sample) else None
        if segs:
            choice_idx, matched_caption = _find_segment_aligned_choice_idx(
                segs, initial_choice_idx, n_section, latent_window_size,
            )
            # Keep overall caption when no segment matched (graceful fallback).
            if matched_caption is not None:
                sample_prompts[b] = matched_caption
        else:
            choice_idx = initial_choice_idx
        # Skill: land the target chunk inside the event window (pixel space). target pix start of
        # section s = s*latent_window_size*4; pick a random section whose start is in [ev_lo, ev_hi].
        # History naturally reaches back into the pre-event scene. Fallback: keep random choice_idx.
        if _is_skill and _event_win is not None:
            ev_lo, ev_hi = int(_event_win[0]), int(_event_win[1])
            valid = [s for s in range(n_section) if ev_lo <= s * latent_window_size * 4 <= ev_hi]
            if valid:
                choice_idx = valid[int(torch.randint(0, len(valid), (1,), generator=g).item())]
        choice_idx_per_sample[b] = choice_idx
        x0, hist, tgt = prepare_stage1_latent_v2(
            latent_sec[b], history_sizes=history_sizes,
            choice_idx=choice_idx, is_keep_x0=is_keep_x0,
        )
        x0_list.append(x0)
        hist_list.append(hist)
        tgt_list.append(tgt)
    x0_latents = torch.stack(x0_list, dim=0) if is_keep_x0 else None
    history_latents = torch.stack(hist_list, dim=0)
    target_latents = torch.stack(tgt_list, dim=0)

    # Slice per-sample target window poses when pose data is present in batch.
    target_pose_Ks = None
    target_pose_c2ws = None
    # Full-length poses needed by GEO renderer to select source pose.
    lingbot_c2ws_full = None
    lingbot_Ks_full = None
    if "lingbot_Ks" in batch and "lingbot_c2ws" in batch:
        lingbot_Ks_full = batch["lingbot_Ks"]      # [B, 4]
        lingbot_c2ws_full = batch["lingbot_c2ws"]  # [B, N_pix, 4, 4]
        N_pix = lingbot_c2ws_full.shape[1]
        pix_window_len = (latent_window_size - 1) * 4 + 1
        pose_per_sample = []
        for b in range(B):
            latent_start = choice_idx_per_sample[b] * latent_window_size
            pix_start = latent_start * 4
            pix_end = pix_start + pix_window_len
            # Clamp-pad if c2ws is shorter than requested range.
            if pix_end > N_pix:
                need = pix_end - N_pix
                seg = torch.cat([
                    lingbot_c2ws_full[b, pix_start:N_pix],
                    lingbot_c2ws_full[b, N_pix - 1:N_pix].expand(need, -1, -1),
                ], dim=0)
            else:
                seg = lingbot_c2ws_full[b, pix_start:pix_end]
            pose_per_sample.append(seg)
        target_pose_c2ws = torch.stack(pose_per_sample, dim=0).to(device=device, dtype=weight_dtype)
        target_pose_Ks = lingbot_Ks_full.to(device=device, dtype=weight_dtype)

    # Skill source has no real pose → synthesize a STATIC camera (identity c2w on every frame) so the
    # downstream Plucker is computed for a non-moving rig ("event camera control signal = static"). The warp itself
    # is dropped (below); this pose feeds ONLY the plucker. Ks: default normalized intrinsic → pixel
    # (fx=W, fy=H, c=center) at target resolution, matching the fallback_default_intrinsic convention.
    if _is_skill and target_pose_c2ws is None:
        _pix_window_len = (latent_window_size - 1) * 4 + 1
        _eye = torch.eye(4, device=device, dtype=weight_dtype)
        target_pose_c2ws = _eye.view(1, 1, 4, 4).expand(B, _pix_window_len, 4, 4).contiguous()
        _bk = batch["bucket_key"]
        _h = int(_bk[1]) if isinstance(_bk, (tuple, list)) else int(_bk[1].item())
        _w = int(_bk[2]) if isinstance(_bk, (tuple, list)) else int(_bk[2].item())
        target_pose_Ks = torch.tensor(
            [float(_w), float(_h), _w / 2.0, _h / 2.0], device=device, dtype=weight_dtype
        ).view(1, 4).expand(B, 4).contiguous()

    # T5 text encode using per-sample prompts (interleave-matched captions applied above).
    prompt_embeds, prompt_attention_mask = encode_prompt(
        tokenizer=tokenizer, text_encoder=text_encoder,
        prompt=sample_prompts, device=device, dtype=weight_dtype,
    )

    bkey = batch["bucket_key"]
    num_frame = int(bkey[0]) if isinstance(bkey, (tuple, list)) else int(bkey[0].item())
    height = int(bkey[1]) if isinstance(bkey, (tuple, list)) else int(bkey[1].item())
    width = int(bkey[2]) if isinstance(bkey, (tuple, list)) else int(bkey[2].item())

    # Sample per-batch GEO conditioning mode (t2v / i2v / full_geo).
    # Skill samples are FORCED to full_geo: they carry the drop-warp layout ([prefix|prev_short]+real
    # mid/long + static-pose plucker) and must NOT be randomly routed to t2v (clears all history) or
    # i2v (forces mid/long invisible), which would discard the very prefix/plk/mid-long they rely on.
    import random as _random_mode
    _geo_mode = "full_geo"
    if use_geometric_state and not _is_skill and (geo_condition_t2v_ratio + geo_condition_i2v_ratio) > 0:
        _r = _random_mode.random()
        if _r < geo_condition_t2v_ratio:
            _geo_mode = "t2v"
        elif _r < geo_condition_t2v_ratio + geo_condition_i2v_ratio:
            _geo_mode = "i2v"
        else:
            _geo_mode = "full_geo"

    # GEO render + VAE encode; skipped entirely when mode is t2v. (i2v now goes through the normal warp
    # path below via DA3 single-source — the old zero-warp [prefix]-only branch was removed.)
    warp_video_latents = None
    warp_visibility_mask = None
    geo_source_image_latent = None   # source frame latent aligned with the warp anchor (= chunk first frame)
    # Skill samples skip the DA3 warp render entirely (no pose, warp is dropped); their warp/prefix
    # tensors are synthesized below. `and not _is_skill` keeps every non-skill path byte-for-byte.
    if use_geometric_state and _geo_mode != "t2v" and not _is_skill:
        assert B == 1, (
            f"GEO training requires train_batch_size=1 (Bug 3 cross-batch mask consistency constraint), got B={B}. "
            f"set in the yaml: train_batch_size: 1, gradient_accumulation_steps: 4."
        )
        assert target_pose_c2ws is not None, (
            "GEO training requires target_pose_c2ws (i.e. the batch contains lingbot_c2ws). check pose_dir in the yaml."
        )

        pix_window_len = (latent_window_size - 1) * 4 + 1

        warp_lat_list = []
        warp_mask_list = []
        source_latent_list = []   # prefix latent from the Pi3X anchor frame
        # Pixel stride between candidate pool frames.
        _geo_pix_stride = latent_window_size * 4  # 36
        # FrameBank retrieve settings from config (mirrors inference defaults).
        import math as _math
        _r_init_k = int(getattr(geo_retrieve_cfg, "init_k", 10)) if geo_retrieve_cfg is not None else 10
        _r_bank_max = int(getattr(geo_retrieve_cfg, "bank_max", 0)) if geo_retrieve_cfg is not None else 0
        _r_score = str(getattr(geo_retrieve_cfg, "score", "v1")) if geo_retrieve_cfg is not None else "v1"
        _r_nearby_k = int(getattr(geo_retrieve_cfg, "nearby_k", 0)) if geo_retrieve_cfg is not None else 0
        _r_select_k = int(getattr(geo_retrieve_cfg, "select_k", 5)) if geo_retrieve_cfg is not None else 5
        _r_score_kwargs = {}
        if _r_score == "v3" and geo_retrieve_cfg is not None:
            _r_score_kwargs = {
                "depth": float(getattr(geo_retrieve_cfg, "v3_depth", 5.0)),
                "fov_rad": _math.radians(float(getattr(geo_retrieve_cfg, "v3_fov_deg", 60.0))),
            }
        _geo_init_max = _r_init_k                   # = init_k from the config
        # Import FrameBank only in mirror-inference mode.
        # ── DA3 cloud-warp (P2): full_geo warp from known-pose point cloud (replaces Pi3X) ──
        _cloud_enabled = bool(getattr(geo_cloud_warp_cfg, "enabled", False)) if geo_cloud_warp_cfg is not None else False
        if _cloud_enabled:
            import random as _random_mode
            # FrameBank recall-style warp (replaces the dump-all build_training_cloud_warp;)
            # render_mode=multisrc: multi-source priority fusion (default);
            #             recall: old point-cloud recall (render_cloud_batched, deprecated).
            from evoke.modules.geometric_state.da3_cloud import build_recall_cloud_warp, build_multisrc_warp, build_backward_warp
            _cw = geo_cloud_warp_cfg
            _cw_render_mode = str(getattr(_cw, "render_mode", "multisrc"))
            _cw_nsrc = int(getattr(_cw, "nsrc", 8))                       # source frames fused per target frame
            _cw_nearby_win = int(getattr(_cw, "nearby_window", 16))       # nearest-candidate window
            _cw_ms_splat = int(getattr(_cw, "multisrc_splat", 1))         # multi-source forward splat radius
            _cw_dens_thresh = float(getattr(_cw, "dens_thresh", 0.45))
            _cw_dens_win = int(getattr(_cw, "dens_win", 7))
            _cw_recall_min_cov = float(getattr(_cw, "recall_min_cov", 0.5))
            _cw_recall_margin = float(getattr(_cw, "recall_margin", 0.15))
            _cw_bw_fill_iters = int(getattr(_cw, "bw_fill_iters", 12))    # backward warp hole-filling iterations
            # backward_zbuf blended despeckle (default off; same renderer and same params as infer/val)
            _cw_zbuf_despeckle = bool(getattr(_cw, "zbuf_despeckle", False))
            _cw_zbuf_despeckle_ksize = int(getattr(_cw, "zbuf_despeckle_ksize", 3))
            _cw_zbuf_despeckle_fill_iters = int(getattr(_cw, "zbuf_despeckle_fill_iters", 4))
            # train-only per-sample render_mode mixing (0=off, use the fixed _cw_render_mode); >0 = one draw of P(backward_zbuf) per sample
            _cw_render_mode_mix_prob_zbuf = float(getattr(_cw, "render_mode_mix_prob_zbuf", 0.0))
            # Depth backend (da3 | vigeo) from the nested per-backend config block. Only the estimator
            # keys are taken from here; the render params above keep reading _cw directly.
            from evoke.utils.train_config import resolve_cloud_warp, vigeo_opts_from_cfg
            _est_cfg = resolve_cloud_warp(_cw)
            _da3_est = _get_da3_estimator(
                int(_est_cfg.get("da3_process_res", 644)), device,
                weights=_est_cfg.get("da3_weights"), backend=_est_cfg.get("depth_backend", "da3"),
                src=_est_cfg.get("da3_src"), vigeo_opts=vigeo_opts_from_cfg(_est_cfg))
            _cw_ingest_n = int(getattr(_cw, "update_frames_per_chunk", 12))   # =ingest_n
            _cw_splat = int(getattr(_cw, "splat_radius", 2))
            _cw_n_tframe = int(getattr(_cw, "n_tframe", 6))
            _cw_grid_div = int(getattr(_cw, "recall_grid_div", 8))
            _cw_mask_pts = int(getattr(_cw, "recall_mask_pts", 8000))
            _cw_conf_pct = float(getattr(_cw, "conf_percentile", 30.0))
            _cw_recall_k_default = int(getattr(_cw, "recall_k", 12))
            _cw_n_nearby_default = int(getattr(_cw, "n_nearby", 4))

            def _choices_probs(samp, default):
                ch = list(getattr(samp, "choices", []) or []) if samp is not None else []
                if not ch:
                    return [int(default)], "uniform"
                return ch, getattr(samp, "probs", "uniform")
            _cw_lag_choices, _cw_lag_probs = _choices_probs(getattr(_cw, "lag_sampling", None), 1)
            _cw_hist_choices, _cw_hist_probs = _choices_probs(getattr(_cw, "history_chunks_sampling", None), 16)
            _cw_rk_choices, _cw_rk_probs = _choices_probs(getattr(_cw, "recall_k_sampling", None), _cw_recall_k_default)
            _cw_nn_choices, _cw_nn_probs = _choices_probs(getattr(_cw, "n_nearby_sampling", None), _cw_n_nearby_default)

            def _sample_choice(choices, probs):
                if isinstance(probs, str) or probs is None:          # "uniform"
                    return int(_random_mode.choice(choices))
                return int(_random_mode.choices(choices, weights=[float(p) for p in probs], k=1)[0])
        for b in range(B):
            # Pixel index of the target chunk start.
            latent_start = choice_idx_per_sample[b] * latent_window_size
            pix_start = latent_start * 4
            target_poses_b = target_pose_c2ws[b].to(dtype=torch.float32)  # [33, 4, 4]

            # ── Train-only warp pose jitter (warp = rough reference, not 1:1 copy) ──
            # Jitter ONLY the pose fed to the warp RENDER below; target_pose_c2ws (plucker + loss) is NEVER
            # mutated -- we right-multiply a single camera-frame rigid DeltaT onto a LOCAL CLONE for the whole
            # chunk. Disabled / not-triggered => target_poses_b is exactly the original (byte-for-byte).
            if (
                geo_pose_jitter_cfg is not None
                and bool(getattr(geo_pose_jitter_cfg, "enabled", False))
                and random.random() < float(getattr(geo_pose_jitter_cfg, "prob", 0.0))
            ):
                _tp = target_poses_b                                  # [F, 4, 4] true GT poses for this window
                # mean per-frame translation (for optional translation jitter; rotation ignores it)
                _t = _tp[:, :3, 3].detach().to(torch.float64)
                _mean_mot = (
                    float(torch.median(torch.linalg.norm(_t[1:] - _t[:-1], dim=1)).item())
                    if _tp.shape[0] >= 2 else 0.0
                )
                _DT = _sample_warp_pose_jitter_DT(geo_pose_jitter_cfg, mean_interframe_trans=_mean_mot).to(
                    device=_tp.device, dtype=_tp.dtype)
                # ONE DeltaT for ALL frames in the window; right-multiply (camera frame): target_c2w @ DT_cam.
                target_poses_b = _tp.clone() @ _DT

            if _cloud_enabled and _geo_mode != "i2v":
                # ── DA3 cloud warp (P2 main path): full_geo warp from known-pose point cloud ──
                assert lingbot_c2ws_full is not None and lingbot_Ks_full is not None, (
                    "cloud_warp requires lingbot_c2ws + lingbot_Ks (GT trajectory + pixel intrinsics)"
                )
                first_frame_pix = raw_video[b:b+1, :, pix_start].to(dtype=torch.float32)  # prefix anchor [1,3,H,W]
                _lag = _sample_choice(_cw_lag_choices, _cw_lag_probs)        # fixed at 1 for inference; sampled during training for robustness
                _hist = _sample_choice(_cw_hist_choices, _cw_hist_probs)     # recall pool depth
                _recall_k = _sample_choice(_cw_rk_choices, _cw_rk_probs)
                _n_nearby = _sample_choice(_cw_nn_choices, _cw_nn_probs)
                # lingbot_Ks = [fx,fy,cx,cy] pixels @ (H_pix,W_pix) -> 3x3.
                _kn = lingbot_Ks_full[b].cpu().numpy()
                _Kpix = np.array([[_kn[0], 0.0, _kn[2]], [0.0, _kn[1], _kn[3]], [0.0, 0.0, 1.0]], dtype=np.float32)
                # train-only per-sample render_mode mixing (when off, _rm_sample == _cw_render_mode, identical to the original behaviour).
                #   >0 = this sample draws backward_zbuf (despeckle) with P(backward_zbuf), otherwise backward -> a mix like 0.8:0.2.
                _rm_sample = _cw_render_mode
                if _cw_render_mode_mix_prob_zbuf > 0:
                    _rm_sample = "backward_zbuf" if torch.rand(1).item() < _cw_render_mode_mix_prob_zbuf else "backward"
                if _rm_sample in ("backward", "backward_zbuf"):
                    # backward (single main source) / backward_zbuf (per-pixel multi-source z-buffer) share build_backward_warp,
                    # and render_mode picks _render_backward / _render_backward_multisrc_zbuf internally.
                    warp_video_b, visibility_mask_b = build_backward_warp(
                        _da3_est, raw_video[b].to(torch.float32), lingbot_c2ws_full[b].to(torch.float32),
                        _Kpix, target_poses_b,
                        pix_start=int(pix_start), pix_stride=int(_geo_pix_stride), window_pix=int(pix_window_len),
                        height=int(H_pix), width=int(W_pix), ingest_n=_cw_ingest_n,
                        lag=_lag, history=_hist, nearby=_cw_nearby_win, fill_iters=_cw_bw_fill_iters,
                        recall_min_cov=_cw_recall_min_cov, recall_margin=_cw_recall_margin,
                        render_mode=_rm_sample,
                        zbuf_despeckle=_cw_zbuf_despeckle, zbuf_despeckle_ksize=_cw_zbuf_despeckle_ksize,
                        zbuf_despeckle_fill_iters=_cw_zbuf_despeckle_fill_iters, device=device)
                elif _cw_render_mode == "multisrc":
                    warp_video_b, visibility_mask_b = build_multisrc_warp(
                        _da3_est, raw_video[b].to(torch.float32), lingbot_c2ws_full[b].to(torch.float32),
                        _Kpix, target_poses_b,
                        pix_start=int(pix_start), pix_stride=int(_geo_pix_stride), window_pix=int(pix_window_len),
                        height=int(H_pix), width=int(W_pix), ingest_n=_cw_ingest_n,
                        lag=_lag, history=_hist, nsrc=_cw_nsrc, nearby=_cw_nearby_win,
                        splat_radius=_cw_ms_splat, dens_thresh=_cw_dens_thresh, dens_win=_cw_dens_win,
                        recall_min_cov=_cw_recall_min_cov, recall_margin=_cw_recall_margin, device=device)
                else:
                    warp_video_b, visibility_mask_b = build_recall_cloud_warp(
                        _da3_est, raw_video[b].to(torch.float32), lingbot_c2ws_full[b].to(torch.float32),
                        _Kpix, target_poses_b,
                        pix_start=int(pix_start), pix_stride=int(_geo_pix_stride), window_pix=int(pix_window_len),
                        height=int(H_pix), width=int(W_pix), ingest_n=_cw_ingest_n,
                        recall_k=_recall_k, n_nearby=_n_nearby, lag=_lag, history=_hist,
                        n_tframe=_cw_n_tframe, grid_div=_cw_grid_div, mask_pts=_cw_mask_pts,
                        conf_pct=_cw_conf_pct, splat_radius=_cw_splat, device=device)
                warp_video_b = warp_video_b.to(device=device, dtype=torch.float32)
                visibility_mask_b = visibility_mask_b.to(device=device, dtype=torch.float32)
            elif _geo_mode == "i2v":
                # i2v: DA3 single-source warp -- source = first frame of the target chunk, gt-metric depth (Umeyama solved from >=3 frames inside the chunk) + raw GT
                # poses, the same recipe as v2v -> warp parallax automatically matches the real GT motion. replaces the old Pi3X single source (this repo uses DA3 only).
                # degenerate / too few frames -> build_single_source_warp returns a blank warp (all holes), which still flows into the shared post-processing below.
                assert _cloud_enabled and lingbot_c2ws_full is not None and lingbot_Ks_full is not None, (
                    "i2v DA3 single-source warp requires cloud_warp.enabled=true + lingbot_c2ws_full + lingbot_Ks_full"
                )
                from evoke.modules.geometric_state.da3_cloud import build_single_source_warp
                first_frame_pix = raw_video[b:b+1, :, pix_start].to(dtype=torch.float32)  # [1,3,H,W] source = chunk first frame (=prefix)
                _kn = lingbot_Ks_full[b].cpu().numpy()
                _Kpix = np.array([[_kn[0], 0.0, _kn[2]], [0.0, _kn[1], _kn[3]], [0.0, 0.0, 1.0]], dtype=np.float32)
                warp_video_b, visibility_mask_b = build_single_source_warp(
                    _da3_est, raw_video[b].to(torch.float32), lingbot_c2ws_full[b].to(torch.float32),
                    _Kpix, target_poses_b,
                    pix_start=int(pix_start), pix_stride=int(_geo_pix_stride), window_pix=int(pix_window_len),
                    height=int(H_pix), width=int(W_pix), ingest_n=_cw_ingest_n, splat_radius=_cw_ms_splat, device=device)
                warp_video_b = warp_video_b.to(device=device, dtype=torch.float32)
                visibility_mask_b = visibility_mask_b.to(device=device, dtype=torch.float32)
            else:
                raise RuntimeError(
                    "[GEO] warp requires cloud_warp.enabled=true (DA3 backend); the Pi3X mirror path has been removed."
                )
            # the DA3 paths (cloud / i2v single-source) directly yield the 33-frame warp_video_b / visibility_mask_b.

            # v20: overwrite warp[0] with the immediate previous frame (clean seam anchor, lag-independent) + mask[0]=1.
            if bool(geo_keep_clean_anchor):
                _anc = first_frame_pix.to(device=device)
                _anc = _anc[0:1] if _anc.ndim == 4 else _anc.unsqueeze(0)
                if tuple(_anc.shape[-2:]) != tuple(warp_video_b.shape[-2:]):
                    _anc = torch.nn.functional.interpolate(
                        _anc, size=tuple(warp_video_b.shape[-2:]), mode="bilinear", align_corners=False
                    )
                warp_video_b[:, :, 0] = _anc.to(dtype=warp_video_b.dtype).clamp(-1, 1)
                visibility_mask_b[:, :, 0] = 1.0

            # VAE encode the Pi3X anchor frame as the DiT prefix latent.
            source_pix_5d = first_frame_pix.unsqueeze(2) if first_frame_pix.ndim == 4 else first_frame_pix.unsqueeze(0).unsqueeze(2)
            with torch.no_grad():
                source_lat_dist = vae.encode(source_pix_5d.to(dtype=vae.dtype)).latent_dist
                source_lat_b = source_lat_dist.sample()
            source_lat_b = ((source_lat_b - latents_mean) * latents_std).to(dtype=torch.float32)
            source_latent_list.append(source_lat_b)

            # VAE encode 33 pixel frames -> 9 latent frames.
            # warp_warm_encode (cloud_warp switch, default off): prepend vae_t copies of the warp's first
            # frame so its FIRST latent is encoded as a CONTINUATION frame (not the VAE I-frame/first-frame
            # distribution), then drop that warm I-frame latent. The GT target is already continuation-
            # distributed (full-video one-shot encode), so making warp continuation too lets the model learn
            # a continuation-distributed pred[0]; inference can then one-shot/persistent decode seamlessly.
            _cw_warm_encode = bool(getattr(_cw, "warp_warm_encode", False))
            with torch.no_grad():
                if _cw_warm_encode:
                    _vt = 4
                    _min_f = (int(latent_window_size) - 1) * _vt + 1
                    _vid = warp_video_b[:, :, -_min_f:]
                    _warm = _vid[:, :, :1].repeat(1, 1, _vt, 1, 1)
                    _padded = torch.cat([_warm, _vid], dim=2).to(dtype=vae.dtype)
                    w_lat = vae.encode(_padded).latent_dist.sample()[:, :, 1:]   # drop warm I-frame latent
                else:
                    w_lat = vae.encode(warp_video_b.to(dtype=vae.dtype)).latent_dist.sample()
            w_lat = (w_lat - latents_mean) * latents_std
            w_lat = w_lat.to(dtype=torch.float32)

            # Trim to latent_window_size if VAE output has extra frames.
            if w_lat.shape[2] > latent_window_size:
                w_lat = w_lat[:, :, -latent_window_size:]

            # Error-bank injection (err-then-noise): add a banked low-noise y_error to the CLEAN warp
            # latent BEFORE visibility noising, so err rides the (1-sigma) scaling and is zeroed in
            # invisible regions (when sigma_invisible=1.0). Gated; no-op until bank warmed.
            if (
                geo_warp_error_inject_enabled
                and recycle_vars is not None
                and args is not None
                and random.random() < float(geo_warp_error_prob)
            ):
                with torch.no_grad():
                    _geo_inject_warp_error(args, recycle_vars, w_lat)

            # Resize visibility mask to latent domain then add noise (spatial or frame-uniform).
            vis_lat = _geo_resize_visibility_to_latent(
                visibility_mask_b, num_lat_per_chunk=latent_window_size,
                H_lat=H_lat, W_lat=W_lat, vae_t_stride=4,
            )
            w_lat = _geo_add_noise_to_warp_latents(
                w_lat,
                device=device,
                sigma_min=float(geo_sigma_visible_min),
                sigma_max=float(geo_sigma_visible_max),
                visibility_aware_noise=bool(geo_visibility_aware_noise),
                sigma_invisible=float(geo_sigma_invisible),
                visibility_mask_lat=vis_lat,
            )

            warp_lat_list.append(w_lat)
            warp_mask_list.append(vis_lat)
        warp_video_latents = torch.cat(warp_lat_list, dim=0).to(dtype=weight_dtype)
        warp_visibility_mask = torch.cat(warp_mask_list, dim=0).to(dtype=torch.float32)
        # Collect source latents for train_evoke prefix injection.
        geo_source_image_latent = torch.cat(source_latent_list, dim=0).to(dtype=weight_dtype)

    # ── Skill (grok event/VFX): drop warp, keep plucker. We emit a ZERO warp tier with an all-INVISIBLE
    # mask: the existing short-tier machinery (history_visible_mask_short = [prefix(1) | warp_vis | prev_short(1)])
    # then FILTERS the warp tokens out entirely (filter_history_tokens_by_mask) → short tier collapses to
    # [prefix | prev_short] + real mid/long, exactly the trained warp_token_drop=full layout. The static-pose
    # plucker is added to the NOISE tokens independently (transformer line ~1479), so it survives the drop.
    # prefix is produced HERE (independent of the skipped warp block) from the loaded-window head frame =
    # the pre-event scene anchor ("walk around a bit first, then cast the skill").
    if use_geometric_state and _is_skill:
        W = int(latent_window_size)
        warp_video_latents = torch.zeros(B, C_lat, W, H_lat, W_lat, device=device, dtype=weight_dtype)
        warp_visibility_mask = torch.zeros(B, 1, W, H_lat, W_lat, device=device, dtype=torch.float32)
        _src_list = []
        for b in range(B):
            _head = raw_video[b:b + 1, :, 0:1]   # [1, C, 1, H_pix, W_pix] = loaded-window head (pre-event)
            with torch.no_grad():
                _sl = vae.encode(_head.to(dtype=vae.dtype)).latent_dist.sample()
            _sl = ((_sl - latents_mean) * latents_std).to(dtype=torch.float32)
            _src_list.append(_sl)
        geo_source_image_latent = torch.cat(_src_list, dim=0).to(dtype=weight_dtype)

    return {
        "prompt_embeds": prompt_embeds,
        "prompt_attention_masks": prompt_attention_mask,
        "x0_latents": x0_latents,
        "history_latents": history_latents,
        "target_latents": target_latents,
        "clean_all_latents": None,
        # Present only when yaml has pose_dir configured.
        "target_pose_Ks": target_pose_Ks,
        "target_pose_c2ws": target_pose_c2ws,
        # GEO fields; all None when use_geometric_state=False.
        "warp_video_latents": warp_video_latents,
        "warp_visibility_mask": warp_visibility_mask,
        # Source frame latent aligned with Pi3X anchor, used as short-tier prefix.
        "geo_source_image_latent": geo_source_image_latent,
        # GEO conditioning mode for train loop dispatch; None when GEO is disabled.
        # Skill stays "full_geo" (real mid/long kept); warp is dropped via the zero-visibility tier above.
        "geo_condition_mode": _geo_mode if use_geometric_state else None,
        # Skill (grok event/VFX) marker — for logging / inference symmetry. Training needs no branch:
        # the zero-visibility warp tier already collapses the short tier to [prefix | prev_short].
        "is_skill": _is_skill,
        "uttid": batch.get("uttid", []),
        "dataset_name": batch.get("dataset_name", []),
        "bucket_key": bkey,
        "num_frame": num_frame,
        "height": height,
        "width": width,
    }
