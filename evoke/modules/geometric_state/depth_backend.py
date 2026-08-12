"""Single construction point for the cloud-warp depth estimator ('da3' | 'vigeo').

The estimator is the only backend-dependent object in the warp pipeline -- everything downstream takes it
as a parameter and only calls `depth_window` / `depth_windows_batched` -- so routing both construction
sites (`online_materialize._get_da3_estimator` for training, `pipeline_evoke._geo_da3_setup` for
inference and validation) through here keeps the branch in one place.
"""
from __future__ import annotations

from typing import Optional

_VIGEO_KEYS = ("num_tokens", "mode", "chunk_size", "intr_source", "conf_transform",
               "scale_mode", "anchor_windows", "total_budget", "cache_keep_frames",
               # only read by the baseline-free scale modes: scale_value for scale_mode=fixed,
               # depth_median_target for scale_mode=depth_median (see vigeo_cloud._resolve_scale_baseline_free)
               "scale_value", "depth_median_target")


def build_estimator(backend: str, device, process_res: int,
                    weights: Optional[str] = None, src: Optional[str] = None,
                    vigeo_opts: Optional[dict] = None):
    """Construct the depth estimator for `backend` ('da3' | 'vigeo').

    `weights` / `src` / `process_res` are already resolved for the active backend by the caller, so this
    function never has to know which nested config block they came from. `vigeo_opts` carries the
    ViGeo-only knobs and is ignored for da3; an unrecognised key raises, because silently dropping it
    means a misspelled knob leaves the estimator on a different recipe than the config describes.
    """
    b = (backend or "da3").lower()
    if b == "vigeo":
        from .vigeo_cloud import ViGeoDepthEstimator, _VIGEO_WEIGHTS
        kwargs = dict(device=device, process_res=int(process_res),
                      weights=(weights or str(_VIGEO_WEIGHTS)))
        if src:
            kwargs["src"] = src
        unknown = sorted(set(vigeo_opts or {}) - set(_VIGEO_KEYS))
        if unknown:
            raise ValueError(f"unknown ViGeo option(s) {unknown}; expected any of {sorted(_VIGEO_KEYS)}")
        for k in _VIGEO_KEYS:
            v = (vigeo_opts or {}).get(k)
            if v is not None:
                kwargs[k] = v
        return ViGeoDepthEstimator(**kwargs)
    if b != "da3":
        raise ValueError(f"unknown cloud_warp depth backend {backend!r}; expected 'da3' or 'vigeo'")
    from .da3_cloud import DA3DepthEstimator, _DA3_WEIGHTS
    kwargs = dict(device=device, process_res=int(process_res),
                  weights=(weights or str(_DA3_WEIGHTS)))
    if src:
        kwargs["src"] = src
    return DA3DepthEstimator(**kwargs)


def reset_stream(estimator) -> None:
    """Drop per-stream estimator state at a stream boundary; a no-op for stateless backends.

    A stream is one contiguous video: one training sample, one self-forcing rollout, or one generation.
    ViGeo holds a kv-cache and a locked depth scale across windows, and the training-side estimator is
    a process-wide singleton shared by every sample, so without this the first video's cache and scale
    leak into all later ones -- silently, since ViGeo does not raise on an inconsistent stream.
    """
    fn = getattr(estimator, "reset_stream", None)
    if callable(fn):
        fn()


def check_assets(backend: str, weights: Optional[str] = None, src: Optional[str] = None) -> None:
    """Fail fast if the backend's assets are missing, before any model loads.

    vigeo only: DA3 checks its own source tree in `DA3DepthEstimator._lazy`, and its weights path is a
    from_pretrained argument that raises on its own. Parameter order matches build_estimator.
    """
    if (backend or "da3").lower() != "vigeo":
        return
    from .vigeo_cloud import check_assets as _check
    _check(src, weights)
