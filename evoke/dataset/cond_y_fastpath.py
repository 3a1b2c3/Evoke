"""Fast path for the teacher's i2v condition y: encode a 113-frame prefix and append a constant
tail computed once at startup. Bit-identical to the full encode, not an approximation.

The condition latent was built as vae.encode(cond_px).mode() over cond_px = zeros with the first
frame written in -- 753 frames, 752 of them zero. A zero pixel frame does not encode to a zero
latent, so zeros cannot replace it. But Wan-VAE is strictly causal in time with a finite
receptive field:

    latent frame j depends on pixel frames [4j - 112, 4j]   (R = 112, clipped at 0)
    independent of the first frame  <=>  4j - 112 >= 1  <=>  j >= 29

Past j = 29 the input window is all zeros, and the chunk period equals the temporal stride, so
the network is shift-equivariant there and every such frame is the same constant K -- a function
of (H, W, dtype, device, tiling params) alone. So head = encode(cond_px[:, :, :113]) yields
latents 0..28 and the tail is K expanded. `num_frame` only sets the chunk count and chunk c
depends on chunks <= c, so truncating the input leaves the earlier chunks bit-identical.

SF_CONDY_VERIFY=1 also runs the full encode and compares bitwise. A mismatch does not abort: it
logs, disables the fast path for the rest of the run and returns the full-length result already
in hand. A failing check must fall back to the slow path, never kill the job.
"""
from __future__ import annotations

import os

import torch

# latent frame j >= M_INDEP is independent of the image; from R=112, confirmed by measurement.
M_INDEP = 29
# Pixel frames needed for latents 0..M_INDEP-1: 1 + 4*(M_INDEP-1). chunk 0 = pixel[0:1], chunk i = [4i-3, 4i].
HEAD_PX = 1 + 4 * (M_INDEP - 1)          # = 113

_K_CACHE: dict = {}
# Verify only the first N calls, counted per T_px, then run the fast path unchecked: a check costs
#   one extra full-length encode (18.15s against ~2.3s), and the premises -- receptive field,
#   tiling params, VAE implementation -- cannot change within a run, so a violation shows up on
#   the very first call.
_VERIFY_DONE: dict = {}
_VERIFY_N = int(os.environ.get("SF_CONDY_VERIFY_N", "3") or 3)

# Tripped when a check refutes the premises, or the fast path itself fails: the run then uses `_full_encode`.
_FAST = {"off": False, "why": ""}


def _disable(why: str):
    if not _FAST["off"]:
        _FAST["off"] = True
        _FAST["why"] = why
        print(f"[COND-Y-FAST] x disabling the fast path for this run, falling back to the full encode (training continues, ~13s/step slower): {why}", flush=True)


def _full_encode(vae, raw_video: torch.Tensor, T_px: int) -> torch.Tensor:
    """The slow path the fast path replaces: encode [first frame | zeros*(T_px-1)] in full, `.mode()`.

    Structurally identical to the original code, so tripping the breaker restores the old behaviour.
    """
    B, C_pix, _, H, W = raw_video.shape
    full_px = torch.zeros(1, C_pix, int(T_px), H, W, device=raw_video.device, dtype=raw_video.dtype)
    full_px[:, :, :1] = raw_video[:1, :, :1]
    with torch.no_grad():
        return vae.encode(full_px.to(vae.dtype)).latent_dist.mode()


def _tiling_key(vae):
    """K depends on the spatial tiling split/blend, so those have to be part of the cache key."""
    return (
        bool(getattr(vae, "use_tiling", False)),
        bool(getattr(vae, "use_slicing", False)),
        getattr(vae, "tile_sample_min_height", None), getattr(vae, "tile_sample_min_width", None),
        getattr(vae, "tile_sample_stride_height", None), getattr(vae, "tile_sample_stride_width", None),
    )


def _zero_tail_const(vae, H: int, W: int, C_pix: int, dtype, device) -> torch.Tensor:
    """Compute (and cache) the fixed-point latent frame that every j >= M_INDEP collapses to under an
    all-zero input; shape [1,16,1,H/8,W/8].

    Returned after `.mode()` but before the (x-mean)*std normalisation, which the caller applies,
    exactly as on the full-length path.
    """
    key = (H, W, C_pix, str(dtype), str(device)) + _tiling_key(vae)
    if key in _K_CACHE:
        return _K_CACHE[key]
    # Encode 1+4*M_INDEP = 117 zero frames -> T_lat = 30, then take frame M_INDEP, already independent.
    n_px = 1 + 4 * M_INDEP
    with torch.no_grad():
        z = torch.zeros(1, C_pix, n_px, H, W, device=device, dtype=dtype)
        lat = vae.encode(z).latent_dist.mode()
    assert lat.shape[2] == M_INDEP + 1, (
        f"[COND-Y-FAST] expected {n_px} frames to give {M_INDEP + 1} latents, got {lat.shape[2]} "
        f"-- the VAE temporal compression changed, so the fast path premise no longer holds")
    K = lat[:, :, M_INDEP:M_INDEP + 1].contiguous().clone()
    _K_CACHE[key] = K
    print(f"[COND-Y-FAST] constant tail precomputed and cached: M={M_INDEP} HEAD_PX={HEAD_PX} "
          f"K.shape={tuple(K.shape)} key=({H}x{W}, {dtype}, tiling={_tiling_key(vae)[0]}) "
          f"-- each step now encodes {HEAD_PX} frames instead of the full length", flush=True)
    return K


def cond_y_latent(vae, raw_video: torch.Tensor, T_px: int, verify: bool | None = None) -> torch.Tensor:
    """A latent bit-identical to `vae.encode([first frame|zeros*(T_px-1)]).latent_dist.mode()`, but
    encoding only HEAD_PX frames.

    raw_video: [B,C,T,H,W] (only its first frame is used); T_px: target pixel frame count.
    Returns [1,16,T_lat,H/8,W/8] with T_lat = 1 + (T_px-1)//4.
    """
    # These two are input contracts, not runtime checks: they are step-independent, so they can only
    #   fire on step 0, and the slow path has the same contract -- there is nothing to fall back to.
    assert raw_video.dim() == 5, f"[COND-Y-FAST] expected [B,C,T,H,W], got {tuple(raw_video.shape)}"
    B, C_pix, _, H, W = raw_video.shape
    T_lat = 1 + (T_px - 1) // 4
    if T_lat <= M_INDEP:                       # too short to have an independent region; use the full path
        return _full_encode(vae, raw_video, T_px)
    if _FAST["off"]:                           # breaker tripped
        return _full_encode(vae, raw_video, T_px)
    if verify is None:
        # Only the first _VERIFY_N calls, counted per T_px; see _VERIFY_DONE.
        _n = _VERIFY_DONE.get(T_px, 0)
        verify = os.environ.get("SF_CONDY_VERIFY") == "1" and _n < _VERIFY_N
        if verify:
            _VERIFY_DONE[T_px] = _n + 1

    try:
        head_px = torch.zeros(1, C_pix, HEAD_PX, H, W, device=raw_video.device, dtype=raw_video.dtype)
        head_px[:, :, :1] = raw_video[:1, :, :1]
        with torch.no_grad():
            head = vae.encode(head_px.to(vae.dtype)).latent_dist.mode()
        if head.shape[2] != M_INDEP:
            _disable(f"encoding {HEAD_PX} frames should give {M_INDEP} latents, got {head.shape[2]} -- VAE temporal compression changed")
            return _full_encode(vae, raw_video, T_px)
        K = _zero_tail_const(vae, H, W, C_pix, raw_video.dtype, raw_video.device)
        tail = K.expand(head.shape[0], -1, T_lat - M_INDEP, -1, -1)
        out = torch.cat([head, tail], dim=2)
    except Exception as _e:
        _disable(f"the fast path itself raised: {type(_e).__name__}: {_e}")
        return _full_encode(vae, raw_video, T_px)

    if verify:
        # Bitwise check: "close enough" is not accepted, since the claim is exact equality. Both sides
        # use `.mode()`, so it is deterministic and consumes no random numbers.
        #
        # The whole block stays inside the try: `_full_encode` allocates another full-length zero tensor
        #   (~2.1GB) plus full-length VAE activations while head_px / head / out are still referenced,
        #   so OOM is the likeliest way this check fails -- and a check meant to avoid killing the job
        #   must not kill it. head_px is freed first for the same reason.
        try:
            del head_px                        # free the 113-frame tensor before the full-length peak
            ref = _full_encode(vae, raw_video, T_px)
            if ref.shape != out.shape:
                _disable(f"shape mismatch: ref={tuple(ref.shape)} fast={tuple(out.shape)}")
                return ref
            d_head = (ref[:, :, :M_INDEP] - out[:, :, :M_INDEP]).abs().max().item()
            d_tail = (ref[:, :, M_INDEP:] - out[:, :, M_INDEP:]).abs().max().item()
            if d_head != 0.0 or d_tail != 0.0:
                _disable(
                    f"differs from the full encode: first {M_INDEP} frames max|delta|={d_head:.3e}, "
                    f"last {T_lat - M_INDEP} frames max|delta|={d_tail:.3e} -> "
                    f"{'receptive field R>112, or the chunking changed' if d_head else ''}"
                    f"{' and ' if (d_head and d_tail) else ''}"
                    f"{'the M=' + str(M_INDEP) + ' independence boundary does not hold (tiling params changed?)' if d_tail else ''}")
                return ref                    # use the slow-path result for this step; it is already computed
            print(f"[COND-Y-VERIFY] ok bit-identical T_px={T_px} T_lat={T_lat}: first {M_INDEP} frames delta=0, "
                  f"last {T_lat - M_INDEP} frames delta=0  ({_VERIFY_DONE.get(T_px, 0)}/{_VERIFY_N} checks; "
                  f"unchecked from here on)", flush=True)
            del ref
        except Exception as _e:
            # The check itself failed, most likely OOM: log once and still return the fast-path result.
            #   No trip -- nothing refuted the premises, and the slow path needs more memory, not less.
            print(f"[COND-Y-VERIFY] the check itself failed and was skipped; training is unaffected and the fast-path result is still used: "
                  f"{type(_e).__name__}: {_e}", flush=True)
    return out
