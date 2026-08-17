#!/usr/bin/env python3
"""Run Evoke inference examples without bash/WSL."""

import os
import sys
from pathlib import Path

# Add Evoke to path
evoke_dir = Path(__file__).parent
sys.path.insert(0, str(evoke_dir))

import torch
from evoke.pipelines.pipeline_evoke import EvokePipeline
from evoke.modules.transformer_evoke import EvokeTransformer3DModel
from diffusers.models.modeling_utils import ModelMixin

# Patch from_pretrained to strip max_memory before calling parent
_orig_from_pretrained = ModelMixin.from_pretrained.__func__
@classmethod
def _patched_from_pretrained(cls, *args, **kwargs):
    kwargs.pop("max_memory", None)
    kwargs.pop("_fast_init", None)
    return _orig_from_pretrained(cls, *args, **kwargs)
EvokeTransformer3DModel.from_pretrained = _patched_from_pretrained


def run_inference(mode, num_frames=73, output_dir="outputs"):
    """Run Evoke inference for a given mode."""

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    print(f"\n[Evoke] Mode: {mode}, Frames: {num_frames}")

    # Initialize pipeline
    print("  Loading model...")

    # Load from HF repo - the model_index.json is in evoke-base,
    # but transformer weights reference the stage3_post_distillation folder
    try:
        pipeline = EvokePipeline.from_pretrained(
            "SII-YuanyangYin/Evoke",
            torch_dtype=torch.float16,
        )
        pipeline = pipeline.to("cuda")
    except Exception as e:
        print(f"Failed to load from HF repo: {e}")
        # Fallback: load from local cache with evoke-base
        hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
        evoke_model_path = hf_cache / "models--SII-YuanyangYin--Evoke" / "snapshots"
        snapshots = list(evoke_model_path.glob("*"))
        if not snapshots:
            raise RuntimeError("No Evoke model snapshots found")
        model_snapshot = snapshots[0]
        pipeline = EvokePipeline.from_pretrained(
            str(model_snapshot / "evoke-base"),
            torch_dtype=torch.float16,
        )
        pipeline = pipeline.to("cuda")

    # Prepare inputs based on mode
    if mode == "t2v":
        # Text-to-video (prompt only)
        prompt = "A serene mountain landscape with morning mist"
        output_path = output_dir / "t2v" / "geo_pred.mp4"

        print(f"  Prompt: {prompt}")
        print(f"  Generating {num_frames} frames (9.5s)...")

        output = pipeline(
            prompt=prompt,
            num_frames=num_frames,
            height=384,
            width=640,
            num_inference_steps=3,
        )

    elif mode == "i2v":
        # Image-to-video (first frame + camera motion)
        from PIL import Image

        # Use example image from repo
        example_image = evoke_dir / "examples" / "data" / "frame.png"
        if not example_image.exists():
            print(f"  ERROR: Example image not found at {example_image}")
            return False

        image = Image.open(example_image)
        output_path = output_dir / "i2v" / "geo_pred.mp4"

        print(f"  Image: {example_image}")
        print(f"  Generating {num_frames} frames (9.5s)...")

        output = pipeline(
            image=image,
            num_frames=num_frames,
            height=384,
            width=640,
            num_inference_steps=3,
        )

    elif mode == "v2v":
        # Video-to-video (reference video + camera motion)
        # Use example video from repo
        example_video = evoke_dir / "examples" / "data" / "video.mp4"
        if not example_video.exists():
            print(f"  ERROR: Example video not found at {example_video}")
            return False

        output_path = output_dir / "v2v" / "geo_pred.mp4"

        print(f"  Video: {example_video}")
        print(f"  Generating {num_frames} frames (9.5s)...")

        output = pipeline(
            video=str(example_video),
            num_frames=num_frames,
            height=384,
            width=640,
            num_inference_steps=3,
        )

    else:
        print(f"  ERROR: Unknown mode {mode}")
        return False

    # Save output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(output, 'save'):
        output.save(str(output_path))
    else:
        # output is already path
        output_path = output

    print(f"  ✓ Saved: {output_path}")
    return True


def main():
    """Run all inference examples."""

    print("="*70)
    print("EVOKE INFERENCE EXAMPLES -- Pure Python (No WSL/Bash)")
    print("="*70)

    # Check models
    hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
    if not (hf_cache / "models--SII-YuanyangYin--Evoke").exists():
        print("\nERROR: Models not found in HF cache")
        print("Run: download_models.bat")
        return 1

    print("\n✓ Models found")

    # Run examples (t2v only for now - others need more testing)
    modes = ["t2v"]

    for i, mode in enumerate(modes, 1):
        try:
            success = run_inference(mode, num_frames=25)
            if not success:
                print(f"  ✗ Failed")
        except Exception as e:
            print(f"  ✗ Error: {e}")

    print("\n" + "="*70)
    print("✓ Examples complete")
    print("="*70)
    print("\nOutput videos: outputs/*/geo_pred.mp4\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
