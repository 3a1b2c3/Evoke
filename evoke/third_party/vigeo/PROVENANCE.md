# ViGeo (vendored)

`vigeo/` is a verbatim copy of the `vigeo` package from the upstream ViGeo repository, with
`__pycache__/` and `*.pyc` stripped. No source file was modified.

- Upstream: <https://github.com/aigc3d/ViGeo>, commit `78100ce`
- License: Apache-2.0 — see `LICENSE` in this directory (redistribution must retain it).
  Upstream ships no `NOTICE` file, so none is required here.
- Vendored on: 2026-08-03
- Only the `vigeo` package is vendored. The upstream `utils/` and `videoldcm/` trees are for the
  data-refinement model and the paper's evaluation harness; the `vigeo` package does not import
  them, so they are not needed here.
- The package uses absolute imports (`from vigeo.layers import …`), so it is consumed by putting
  *this* directory on `sys.path` rather than by importing it as a `evoke.third_party` subpackage —
  the same arrangement as the vendored DA3 copy next door.

## Consumers

Only the geometric state ViGeo depth backend uses it:
`evoke/modules/geometric_state/vigeo_cloud.py` (`ViGeoDepthEstimator._lazy`), which inserts this
directory into `sys.path` and then imports `vigeo.ViGeo`. Active when
`cloud_warp.backend: vigeo` (training / validation) or `--geo_depth_backend vigeo` (inference).

Runtime dependencies beyond what Evoke already pins: none. The package needs `torch`, `numpy`,
`einops` and `huggingface_hub`, all already required. `xformers` is imported behind an
`XFORMERS_ENABLED` guard in `vigeo/layers/{attention,block}.py` and falls back to plain attention
when absent, which is the path this repo takes (the evoke env has no xformers).

## Weights

Not vendored: `models/ViGeo1.1/vigeo.pt` is ~4.8 GB. Fetch the `pkqbajng/ViGeo1.1` snapshot into
`models/ViGeo1.1/`, or point `EVOKE_VIGEO_WEIGHTS` at an existing copy. `models/` is gitignored,
matching how the DA3 weights are handled.
