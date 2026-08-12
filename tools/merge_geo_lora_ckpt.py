"""Merge a GEO LoRA checkpoint into the base transformer for a clean warm-start base.

Loads base transformer + LoRA (pytorch_lora_weights.safetensors) + extra components (transformer_partial.pth,
incl. warp_residual_mlp which load_extra_components auto-creates), fuses the LoRA into the weights, and saves a
full diffusers transformer/ that downstream configs can use via `transformer_model_name_or_path`.

Usage:
  python tools/merge_geo_lora_ckpt.py \
    --base models/evoke-base --ckpt <ckpt_dir> --out <out_dir>
"""
import argparse
from argparse import Namespace

from evoke.modules.transformer_evoke import EvokeTransformer3DModel
from evoke.pipelines.pipeline_evoke import EvokePipeline
from evoke.utils.utils_base import load_extra_components


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="base model dir (has transformer/ + pipeline components)")
    ap.add_argument("--ckpt", required=True, help="LoRA ckpt dir with pytorch_lora_weights.safetensors + transformer_partial.pth")
    ap.add_argument("--out", required=True, help="output dir; merged weights written to <out>/transformer")
    ap.add_argument("--plucker", choices=["auto", "on", "off"], default="auto",
                    help="build patch_embedding_wancamctrl + c2ws_hidden_states_layer{1,2} so the LoRA's plucker "
                         "deltas fuse (else silently dropped). 'auto' = detect from the LoRA keys (matches infer).")
    args_cli = ap.parse_args()

    # Match the v22 GEO transformer config (no restrict/gan; multi-term memory + guidance cross-attn).
    transformer_additional_kwargs = {
        "has_multi_term_memory_patch": True,
        "zero_history_timestep": True,
        "guidance_cross_attn": True,
        "restrict_self_attn": False,
        "is_train_restrict_lora": False,
        "restrict_lora": False,
        "restrict_lora_rank": 128,
    }

    # Plücker (plk): the additive-Plücker submodules are built only when geo_warp_plucker_enabled=True.
    # If the LoRA carries plucker deltas but we build without the flag, those deltas are silently dropped
    # ("unexpected keys") → merged model loses camera control. Auto-detect from the LoRA (mirrors infer).
    _lora_file = f"{args_cli.ckpt}/pytorch_lora_weights.safetensors"
    _plk_on = args_cli.plucker == "on"
    if args_cli.plucker == "auto":
        try:
            from safetensors import safe_open
            with safe_open(_lora_file, framework="pt") as _f:
                _plk_on = any("patch_embedding_wancamctrl" in k for k in _f.keys())
        except Exception as _e:
            print(f"[merge] plk auto-detect skipped: {type(_e).__name__}: {_e}", flush=True)
    if _plk_on:
        transformer_additional_kwargs["geo_warp_plucker_enabled"] = True
        print("[merge] geo_warp_plucker_enabled=True (plucker deltas will fuse)", flush=True)

    transformer = EvokeTransformer3DModel.from_pretrained(
        args_cli.base, subfolder="transformer",
        transformer_additional_kwargs=transformer_additional_kwargs,
    )

    # plk submodules are fresh in __init__ + absent from the base ckpt → under low_cpu_mem_usage their
    # zero-init base stays on `meta`. Materialize to zeros before LoRA fuse (else fuse/save hits meta tensors).
    # Effective weight = 0 (base) + LoRA delta, so fusing yields exactly the trained plucker delta.
    if _plk_on:
        import torch.nn as _nn
        _ref_dtype = next(transformer.parameters()).dtype
        for _m in ("patch_embedding_wancamctrl", "c2ws_hidden_states_layer1", "c2ws_hidden_states_layer2"):
            _mod = getattr(transformer, _m, None)
            if _mod is not None and any(p.is_meta for p in _mod.parameters()):
                _mod.to_empty(device="cpu")
                _nn.init.zeros_(_mod.weight)
                if getattr(_mod, "bias", None) is not None:
                    _nn.init.zeros_(_mod.bias)
                _mod.to(dtype=_ref_dtype)
                print(f"[merge] plk: materialized meta base {_m} -> zeros ({_ref_dtype})", flush=True)

    pipe = EvokePipeline.from_pretrained(args_cli.base, transformer=transformer)

    # Pass dir + explicit weight_name (required under HF_HUB_OFFLINE=1; matches scripts/inference/infer_single.py).
    pipe.load_lora_weights(
        args_cli.ckpt, weight_name="pytorch_lora_weights.safetensors", adapter_name="default",
    )
    pipe.set_adapters(["default"], adapter_weights=[1.0])

    # Load extra components (warp_residual_mlp + multi-term patch); auto-creates warp_residual_mlp if absent.
    args = Namespace()
    args.training_config = Namespace()
    args.training_config.is_enable_stage1 = True
    args.training_config.restrict_self_attn = False
    args.training_config.is_amplify_history = False
    args.training_config.is_use_gan = False
    load_extra_components(args, transformer, f"{args_cli.ckpt}/transformer_partial.pth")

    pipe.fuse_lora()
    pipe.unload_lora_weights()
    out_tf = f"{args_cli.out}/transformer"
    pipe.transformer.save_pretrained(out_tf)
    print(f"[merge] DONE -> {out_tf}")


if __name__ == "__main__":
    main()
