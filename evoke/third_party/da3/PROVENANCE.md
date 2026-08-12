# Depth Anything 3 (vendored)

`depth_anything_3/` is a verbatim copy of the `src/depth_anything_3` package from the
upstream Depth Anything 3 repository, with `__pycache__/` and `*.pyc` stripped. No source
file was modified.

- Upstream: <https://github.com/ByteDance-Seed/Depth-Anything-3>, commit `41736238`
- License: Apache-2.0 — see `LICENSE` in this directory (redistribution must retain it).
  Upstream ships no `NOTICE` file, so none is required here.
- Vendored on: 2026-07-29
- The package uses absolute imports (`import depth_anything_3.…`) and PEP 420 implicit
  namespace packages, so it is consumed by putting *this* directory on `sys.path` rather
  than by importing it as a `evoke.third_party` subpackage.

## Consumers

Only the geometric state DA3 point-cloud path uses it:
`evoke/modules/geometric_state/da3_cloud.py` (`DA3DepthEstimator._lazy`), which inserts
this directory into `sys.path` and then imports `depth_anything_3.api`.

Before importing, `_lazy()` installs a no-op stub for `depth_anything_3.utils.export` so
that the export-only dependencies (`moviepy`, `plyfile`, `gsplat`) are not required at
runtime. On that path `depth_anything_3.api` pulls in `addict`, `antlr4`, `attrs`, `einops`,
`evo`, `huggingface_hub`, `imageio` and `omegaconf`; `evo` and `addict` are imported by this
vendored code and by nothing else in the repo, so they are pinned in the top-level
`requirements.txt`. Upstream's own `requirements.txt` is much wider (`fastapi`, `open3d`,
`pycolmap`, `typer`, `xformers`, …) because it also covers `services/`, `bench/`, `cli.py`
and the export path — none of which this repo uses.

To use an external checkout instead, set `EVOKE_DA3_SRC=/path/to/Depth-Anything-3/src`
(that directory must contain the `depth_anything_3/` package). If neither the vendored copy
nor `EVOKE_DA3_SRC` resolves to a directory containing `depth_anything_3/`,
`DA3DepthEstimator._lazy()` raises `FileNotFoundError` instead of a bare `ImportError`.

## Weights

The code contains no weights. `DA3DepthEstimator` reads `models/DA3` by default
(`models/` is not version-controlled); `EVOKE_DA3_WEIGHTS` overrides the location.
The checkpoint in use is `da3-giant`:

```bash
huggingface-cli download depth-anything/da3-giant --local-dir models/DA3
```

> **Licensing note:** the `da3-giant` weights are released under **CC-BY-NC-4.0**
> (non-commercial), which is *not* the Apache-2.0 license of the code in this directory.
> The weights are not redistributed here — only downloaded by the user — but the
> non-commercial restriction applies to anyone running this path.
