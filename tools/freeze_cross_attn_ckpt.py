"""Make a "freezeCross" checkpoint: reset text cross-attention (attn2) to Evoke-Base so a warm-start
keeps the warp progress (attn1/ffn/...) but starts attn2 from the clean base calibration.

SFT mode:  copy <src>/transformer shards, overwrite every '.attn2.' tensor with the Evoke-Base value,
           keep config.json/index.json/transformer_partial.pth unchanged.
LoRA mode: drop every '.attn2.' key from pytorch_lora_weights.safetensors, keep transformer_partial.pth.

Usage:
  python tools/freeze_cross_attn_ckpt.py sft  <src_ckpt> <base_transformer_dir> <out_ckpt>
  python tools/freeze_cross_attn_ckpt.py lora <src_ckpt> <out_ckpt>
"""
import argparse
import json
import os
import shutil

from safetensors import safe_open
from safetensors.torch import save_file, load_file

INDEX = "diffusion_pytorch_model.safetensors.index.json"


def _base_attn2(base_tf):
    """Load all '.attn2.' tensors from the Evoke-Base sharded transformer."""
    idx = json.load(open(os.path.join(base_tf, INDEX)))
    wmap = idx["weight_map"]
    per_shard = {}
    for k, shard in wmap.items():
        if ".attn2." in k:
            per_shard.setdefault(shard, []).append(k)
    out = {}
    for shard, keys in per_shard.items():
        with safe_open(os.path.join(base_tf, shard), framework="pt") as f:
            for k in keys:
                out[k] = f.get_tensor(k)
    print(f"[base] loaded {len(out)} attn2 tensors")
    return out


def do_sft(src_ckpt, base_tf, out_ckpt):
    src_tf = os.path.join(src_ckpt, "transformer")
    out_tf = os.path.join(out_ckpt, "transformer")
    os.makedirs(out_tf, exist_ok=True)
    base_attn2 = _base_attn2(base_tf)

    # config.json + index.json unchanged (same keys/shapes; only attn2 values change).
    shutil.copy(os.path.join(src_tf, "config.json"), os.path.join(out_tf, "config.json"))
    shutil.copy(os.path.join(src_tf, INDEX), os.path.join(out_tf, INDEX))

    wmap = json.load(open(os.path.join(src_tf, INDEX)))["weight_map"]
    shards = sorted(set(wmap.values()))
    n_repl = 0
    for shard in shards:
        tensors, meta = {}, None
        with safe_open(os.path.join(src_tf, shard), framework="pt") as f:
            meta = f.metadata()
            for k in f.keys():
                if ".attn2." in k:
                    assert k in base_attn2, f"attn2 key missing in base: {k}"
                    t = f.get_tensor(k)
                    tensors[k] = base_attn2[k].to(dtype=t.dtype)
                    n_repl += 1
                else:
                    tensors[k] = f.get_tensor(k)
        save_file(tensors, os.path.join(out_tf, shard), metadata=meta or {"format": "pt"})
        print(f"[sft] wrote shard {shard} ({len(tensors)} tensors)")
    print(f"[sft] replaced {n_repl} attn2 tensors with base")

    # warp_residual_mlp partial is not attn2 -> keep as-is.
    _copy_if(os.path.join(src_ckpt, "transformer_partial.pth"), os.path.join(out_ckpt, "transformer_partial.pth"))
    print(f"[sft] DONE -> {out_ckpt}")


def do_lora(src_ckpt, out_ckpt):
    os.makedirs(out_ckpt, exist_ok=True)
    src = os.path.join(src_ckpt, "pytorch_lora_weights.safetensors")
    sd = load_file(src)
    kept = {k: v for k, v in sd.items() if ".attn2." not in k}
    dropped = len(sd) - len(kept)
    with safe_open(src, framework="pt") as f:
        meta = f.metadata()
    save_file(kept, os.path.join(out_ckpt, "pytorch_lora_weights.safetensors"), metadata=meta or {"format": "pt"})
    print(f"[lora] dropped {dropped} attn2 lora keys, kept {len(kept)}")
    _copy_if(os.path.join(src_ckpt, "transformer_partial.pth"), os.path.join(out_ckpt, "transformer_partial.pth"))
    print(f"[lora] DONE -> {out_ckpt}")


def _copy_if(src, dst):
    if os.path.exists(src):
        shutil.copy(src, dst)
        print(f"[copy] {os.path.basename(src)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    s = sub.add_parser("sft"); s.add_argument("src"); s.add_argument("base_tf"); s.add_argument("out")
    l = sub.add_parser("lora"); l.add_argument("src"); l.add_argument("out")
    a = ap.parse_args()
    if a.mode == "sft":
        do_sft(a.src, a.base_tf, a.out)
    else:
        do_lora(a.src, a.out)
