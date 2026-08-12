#!/usr/bin/env python3
"""Merge LoRA adapter and transformer_partial.pth into the base DiT, producing a full transformer/ directory."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from peft import LoraConfig, set_peft_model_state_dict
from safetensors.torch import load_file

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)   # tools/ lives one level below the repo root
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from diffusers.utils import convert_unet_state_dict_to_peft

from evoke.modules.transformer_evoke import EvokeTransformer3DModel


def build_target_modules(transformer, lora_layers: str, include_patch_embedding: bool,
                         include_multi_term_memory_patchg_lora: bool,
                         exclude_modules: list[str]) -> list[str]:
    """Build LoRA target module list, mirroring train_evoke.py logic."""
    if lora_layers == "all-linear":
        target_modules = set()
        for name, module in transformer.named_modules():
            if isinstance(module, torch.nn.Linear):
                target_modules.add(name)
        target_modules = list(target_modules)
    else:
        target_modules = [t.strip() for t in lora_layers.split(",")]

    if include_patch_embedding and "patch_embedding" not in target_modules:
        target_modules.append("patch_embedding")
    if include_multi_term_memory_patchg_lora:
        for p in ("patch_short", "patch_mid", "patch_long"):
            if p not in target_modules:
                target_modules.append(p)
    target_modules = [t for t in target_modules if "norm" not in t]
    return target_modules


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="Base DiT root (with transformer/ subdir and vae/ etc)")
    ap.add_argument("--ckpt", required=True, help="Training checkpoint dir (with pytorch_lora_weights.safetensors + transformer_partial.pth)")
    ap.add_argument("--dst", required=True, help="Output directory")
    ap.add_argument("--subfolder", default="transformer")
    # Do NOT default this back to bf16. The base is fp32 while the LoRA delta has a per-element
    #   magnitude of |delta|/|W| ~ 9e-5, about 22x below a bf16 half-ulp (2^-9 = 2.0e-3). bf16(W0) already
    #   sits on a grid point, so bf16(W0 + delta) rounds straight back to bf16(W0) for ~98% of the
    #   elements and the update is quantised away. Measured retention (<q,d>/||d||^2 over the same set of
    #   modules): saved as bf16 = 0.055, saved as fp32 = 0.308, not fused at all (LoRA kept as an
    #   adapter) = 1.000.
    #   "Training ran in mixed_precision: bf16" is NOT a reason to store the base in bf16 -- compute
    #   precision and storage precision are two different things.
    #   Note that even fp32 only retains ~0.31, because the weights are cast to bf16 again when loaded.
    #   So merging is for exporting a standalone deployable model, NOT for warm-starting a continued
    #   run: for that, keep the base untouched and re-attach the LoRA as an adapter (retention 1.000).
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16", "fp16"])
    # LoRA construction args
    ap.add_argument("--lora_layers", default="all-linear")
    ap.add_argument("--lora_rank", type=int, default=128)
    ap.add_argument("--lora_alpha", type=float, default=128.0)
    ap.add_argument("--lora_dropout", type=float, default=0.0)
    ap.add_argument("--lora_exclude_modules", nargs="*", default=["down", "up"])
    ap.add_argument("--include_patch_embedding", action="store_true", default=True)
    ap.add_argument("--no_include_patch_embedding", dest="include_patch_embedding", action="store_false")
    ap.add_argument("--include_multi_term_memory_patchg_lora", action="store_true", default=False,
                    help="Include multi-term memory patch modules in LoRA target set")
    ap.add_argument("--has_multi_term_memory_patch", action="store_true", default=True)
    ap.add_argument("--use_raw_sink_frames", action="store_true", default=True)
    # Safetensors shard size
    ap.add_argument("--max_shard_size", default="5GB")
    args = ap.parse_args()

    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
    base_root = Path(args.base).resolve()
    ckpt_dir = Path(args.ckpt).resolve()
    dst_root = Path(args.dst).resolve()
    base_transformer = base_root / args.subfolder
    assert base_transformer.exists(), f"base transformer not found: {base_transformer}"
    assert ckpt_dir.exists(), f"ckpt dir not found: {ckpt_dir}"

    # 1. Load base DiT.
    print(f"[merge] Loading base DiT from {base_transformer} ...")
    transformer = EvokeTransformer3DModel.from_pretrained(
        str(base_root),
        subfolder=args.subfolder,
        torch_dtype=dtype,
        transformer_additional_kwargs={
            "has_multi_term_memory_patch": args.has_multi_term_memory_patch,
            "use_raw_sink_frames": args.use_raw_sink_frames,
        },
    )
    transformer.requires_grad_(False)
    print(f"  total params: {sum(p.numel() for p in transformer.parameters()) / 1e9:.3f} B")

    # 2. Build LoRA config and add adapter.
    target_modules = build_target_modules(
        transformer,
        lora_layers=args.lora_layers,
        include_patch_embedding=args.include_patch_embedding,
        include_multi_term_memory_patchg_lora=args.include_multi_term_memory_patchg_lora,
        exclude_modules=args.lora_exclude_modules,
    )
    print(f"[merge] LoRA target_modules: {len(target_modules)} entries (rank={args.lora_rank})")
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        init_lora_weights="gaussian",
        target_modules=list(target_modules),
        exclude_modules=list(args.lora_exclude_modules),
    )
    transformer.add_adapter(lora_config)

    # 3. Load LoRA weights from checkpoint.
    lora_path = ckpt_dir / "pytorch_lora_weights.safetensors"
    assert lora_path.exists(), f"LoRA weights not found: {lora_path}"
    print(f"[merge] Loading LoRA weights from {lora_path} ...")
    sd = load_file(str(lora_path))
    transformer_sd = {k.replace("transformer.", "", 1): v for k, v in sd.items() if k.startswith("transformer.")}
    transformer_sd = convert_unet_state_dict_to_peft(transformer_sd)
    incompat = set_peft_model_state_dict(transformer, transformer_sd, adapter_name="default")
    if incompat is not None:
        unexp = getattr(incompat, "unexpected_keys", [])
        if unexp:
            print(f"  WARN: {len(unexp)} unexpected LoRA keys (first 5): {unexp[:5]}")
    print(f"  loaded {len(transformer_sd)} LoRA tensors")

    # 4. Load transformer_partial.pth for fully-trained modules (patch_short/mid/long etc).
    partial_path = ckpt_dir / "transformer_partial.pth"
    if partial_path.exists():
        print(f"[merge] Loading transformer_partial.pth ...")
        partial_sd = torch.load(str(partial_path), map_location="cpu", weights_only=False)
        if partial_sd:
            partial_sd = {k: v.to(dtype) if torch.is_tensor(v) else v for k, v in partial_sd.items()}
            missing, unexpected = transformer.load_state_dict(partial_sd, strict=False)
            loaded = len(partial_sd) - len(unexpected)
            print(f"  loaded {loaded}/{len(partial_sd)} partial keys, unexpected={len(unexpected)}")
            if unexpected:
                print(f"  unexpected sample: {list(unexpected)[:3]}")
        else:
            print(f"  partial.pth empty, skip")
    else:
        print(f"[merge] No transformer_partial.pth in ckpt dir, skip.")

    # 5. Fuse LoRA delta into base weights and drop LoRA modules.
    # Snapshot (base_layer, A, B, scaling) for a few modules BEFORE fusing, so the retention factor can be
    #   computed after. Why this and not the obvious checks: "the fused weight differs from the base" and
    #   "residual/|W| < 2e-4" both pass unconditionally and can never fail -- an fp32->bf16 downcast changes
    #   100% of the elements, and |delta|/|W| ~ 9e-5 is already below that 2e-4 threshold, so the assertion
    #   holds even when the whole delta is lost. The only meaningful test normalises by ||delta||:
    #   keep = <q, delta> / ||delta||^2, where q = (fused weight) - (base weight).
    _chk = {}
    for _n, _m in transformer.named_modules():
        if len(_chk) >= 12:
            break
        if hasattr(_m, "base_layer") and hasattr(_m, "lora_A"):
            try:
                _chk[_n] = (_m.base_layer.weight.detach().float().clone(),
                            _m.lora_A["default"].weight.detach().float().clone(),
                            _m.lora_B["default"].weight.detach().float().clone(),
                            float(getattr(_m, "scaling", {}).get("default", 1.0)))
            except Exception:
                pass
    print("[merge] Calling fuse_lora() ...")
    transformer.fuse_lora(lora_scale=1.0, safe_fusing=False)
    print("[merge] Calling unload_lora() ...")
    transformer.unload_lora()

    # How much of the delta actually survives, after fusing AND at the dtype it will be saved in.
    if _chk:
        _sd = dict(transformer.named_parameters())
        _keeps = []
        for _n, (_W0, _A, _B, _sc) in _chk.items():
            _p = _sd.get(_n + ".weight")
            if _p is None:
                continue
            _d = (_B.flatten(1) @ _A.flatten(1)).reshape(_W0.shape) * _sc
            _dn = float((_d * _d).sum())
            if _dn <= 0:
                continue
            # Simulate the on-disk rounding: save_pretrained stores at `dtype`.
            _q = _p.detach().float().to(dtype).float() - _W0.to(dtype).float()
            _keeps.append(float((_q * _d).sum()) / _dn)
        if _keeps:
            _avg = sum(_keeps) / len(_keeps)
            print(f"[merge] LoRA retention self-check: mean {_avg:.3f} "
                  f"(n={len(_keeps)}, 1.0 = fully inherited, dtype={args.dtype})")
            assert _avg > 0.9, (
                f"[merge] only {_avg:.1%} of the LoRA delta survived the merge -- rounding ate it.\n"
                f"  Cause: |delta|/|W| is far below the ulp of {args.dtype}, so {args.dtype}(W0+delta) rounds back to {args.dtype}(W0).\n"
                f"  - Exporting for inference: re-run with --dtype fp32.\n"
                f"  - Warm-starting a continued run: do not merge at all -- even fp32 retains only ~0.31\n"
                f"    (the weights are cast to bf16 again on load). Keep the base as-is and re-attach the\n"
                f"    LoRA as an adapter instead, which retains 1.000.")

    # 6. Save merged transformer.
    out_transformer = dst_root / args.subfolder
    out_transformer.mkdir(parents=True, exist_ok=True)
    print(f"[merge] save_pretrained -> {out_transformer} (dtype={args.dtype}, shard={args.max_shard_size})")
    transformer.save_pretrained(
        str(out_transformer),
        max_shard_size=args.max_shard_size,
        safe_serialization=True,
    )

    # 7. Symlink base top-level entries (vae, text_encoder, tokenizer, scheduler, etc) into dst.
    for entry in os.listdir(base_root):
        if entry == args.subfolder:
            continue
        src = base_root / entry
        dst = dst_root / entry
        if dst.exists() or dst.is_symlink():
            continue
        rel = os.path.relpath(src, dst_root)
        os.symlink(rel, dst)
        print(f"  symlink {entry} -> {rel}")

    print("[merge] Done.")
    print(f"  merged DiT: {out_transformer}")
    print(f"  full root:  {dst_root}")


if __name__ == "__main__":
    main()
