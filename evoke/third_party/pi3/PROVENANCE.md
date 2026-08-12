# Pi3 / Pi3X (vendored)

This directory is a byte-for-byte copy of the `pi3/` package from the official Pi3
implementation, keeping only the model code (upstream `examples/`, `assets/`, and the demo and
benchmark scripts -- 42MB of demo data in total -- have been removed).

- Upstream: Pi3 (`third_party/Pi3` of geometric state), commit `b56ef4b`
- License: BSD 3-Clause, see `LICENSE` in this directory (the copyright notice must be retained on
  redistribution)
- Vendored on: 2026-07-28
- The package uses relative imports throughout, so it can be used directly as a subpackage:
  `from evoke.third_party.pi3.models.pi3x import Pi3X`

## Consumers

Only the camera-control warp path uses it (`Pi3XWarpRenderer.estimate_*_geometry` in
`evoke/modules/geometric_state/camera_warp.py`, lazily imported via `_import_pi3()`).
The main training and inference path of geometric state uses DA3 point clouds
(`evoke/modules/geometric_state/da3_cloud.py`) and does not depend on this directory.

## Weights

The code contains no weights. `Pi3XWarpRenderer` reads `models/Pi3X/model.safetensors` by default
(`models/` is not version-controlled):

```bash
aria2c -x 16 -s 16 -d models/Pi3X -o model.safetensors \
  https://huggingface.co/yyfz233/Pi3X/resolve/main/model.safetensors
```

To override with an external checkout: set `EVOKE_PI3_SRC=/path/to/Pi3` (that directory must
contain a `pi3/` package), or explicitly pass `pi3_repo` from callers outside
`Pi3XWarpRendererConfig`.
