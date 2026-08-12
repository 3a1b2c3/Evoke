"""Build the EvokeTeacher nocam teacher and load its merged weights.

Two facts copied from upstream:
  - architecture base kwargs = training_config.get_model_extra_kwargs (fixed Wan2.2 A14B values);
  - sparse params = configs/model/384*640/nocam_{hn,ln}.yaml. hn and ln differ only in
    timestep_boundary, which the wrapper handles, so construction is identical.
A merged weight directory holds only safetensors shards -- no config.json, no index -- so it
is loaded by glob.
"""

import glob
import json
import os

import torch

from .dit_sparse_cam_14b import WanModelCam

# Fixed Wan2.2 A14B architecture params, copied from upstream get_model_extra_kwargs.
EVOKE_TEACHER_A14B_ARCH = {
    "has_image_input": False,
    "patch_size": (1, 2, 2),
    "in_dim": 36,
    "dim": 5120,
    "ffn_dim": 13824,
    "freq_dim": 256,
    "text_dim": 4096,
    "out_dim": 16,
    "num_heads": 40,
    "num_layers": 40,
    "eps": 1e-06,
    "require_clip_embedding": False,
}

# 384x640 nocam sparse config, copied from upstream nocam_hn.yaml.
EVOKE_TEACHER_NOCAM_SPARSE = {
    "sparse_attn": True,
    "chunk_size": 8,
    "overlap_size": 1,
    "per_frame_tokens": 960,  # 384x640: (384/8/2)*(640/8/2)=24*40
    "num_select_frames": 4,
    "num_nearby_frames": 3,
    "select_scales": ["1x", "2x", "4x", "8x"],
}


def build_evoke_teacher_dit(overrides: dict = None, torch_dtype=torch.bfloat16) -> WanModelCam:
    """Build a nocam WanModelCam (cam_ctrl defaults to False, so no cam modules).

    Constructed under a default_dtype context to avoid the 56GB peak of fp32 14B init.
    """
    kwargs = {**EVOKE_TEACHER_A14B_ARCH, **EVOKE_TEACHER_NOCAM_SPARSE, **(overrides or {})}
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch_dtype)
    try:
        model = WanModelCam(**kwargs)
    finally:
        torch.set_default_dtype(prev_dtype)
    # forward() restores fp32 semantics for non-parameter buffers (freqs etc.) via .to(); cast the
    # parameters once here to cover the few layers that init explicitly in float32.
    model = model.to(torch_dtype)
    return model


def load_merged_weights(model: WanModelCam, merged_dir: str, torch_dtype=torch.bfloat16) -> dict:
    """Glob and load the safetensors shards of a merged directory.

    Tolerates a "pipe.dit." prefix defensively. strict=False, so leftover cam_*/repr_* keys from
    the base checkpoint are allowed to be unexpected -- but `missing` must be empty, since a
    missing key means the model has weights the checkpoint does not cover, i.e. the construction
    params disagree with it. Returns {"missing": [...], "unexpected": [...]} for the caller to log.
    """
    from safetensors.torch import load_file

    shards = sorted(glob.glob(os.path.join(merged_dir, "*.safetensors")))
    assert shards, f"[evoke_teacher] no safetensors found under {merged_dir}"
    state = {}
    for p in shards:
        sd = load_file(p)
        for k, v in sd.items():
            if k.startswith("pipe.dit."):
                k = k[len("pipe.dit."):]
            state[k] = v.to(torch_dtype)
        del sd
    missing, unexpected = model.load_state_dict(state, strict=False)
    del state
    if missing:
        raise RuntimeError(
            f"[evoke_teacher] {len(missing)} missing keys loading {merged_dir} — "
            f"construction params disagree with the checkpoint; first 5: {missing[:5]}"
        )
    if unexpected:
        print(f"[evoke_teacher] {len(unexpected)} unexpected keys ignored "
              f"(leftover cam/repr etc.), first 5: {unexpected[:5]}")
    return {"missing": list(missing), "unexpected": list(unexpected)}


def dump_load_report(report: dict, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({k: v[:50] for k, v in report.items()}, f, ensure_ascii=False, indent=2)
