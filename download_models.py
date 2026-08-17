#!/usr/bin/env python3
"""Download Evoke models from HuggingFace."""

import sys
from pathlib import Path
import subprocess

def run_cmd(cmd, description):
    """Run command and return success status."""
    print(f"\n[Download] {description}")
    print(f"  Command: {cmd}")
    print()
    result = subprocess.run(cmd, shell=True)
    return result.returncode == 0

def main():
    print("=" * 80)
    print("EVOKE - Model Downloader")
    print("=" * 80)

    repo_root = Path(__file__).parent
    models_dir = repo_root / "models"
    models_dir.mkdir(exist_ok=True)

    print(f"\nModels directory: {models_dir.absolute()}")
    print()

    # Check huggingface-cli
    try:
        subprocess.run("huggingface-cli --version", shell=True, capture_output=True, check=True)
    except subprocess.CalledProcessError:
        print("⚠️  huggingface-cli not found. Installing huggingface-hub...")
        subprocess.run("pip install huggingface-hub", shell=True, check=True)

    # Download models
    downloads = [
        {
            "name": "EVOKE Models",
            "repo": "SII-YuanyangYin/Evoke",
            "local_dir": str(models_dir),
            "required": True,
            "description": "VAE, text encoder, tokenizer, scheduler, and all model stages",
        },
        {
            "name": "ViGeo Depth Backend",
            "repo": "pkqbajng/ViGeo",
            "local_dir": str(models_dir / "ViGeo1.1"),
            "required": True,
            "description": "REQUIRED: Depth estimation for world state bank",
        },
        {
            "name": "Depth-Anything-3",
            "repo": "depth-anything/Depth-Anything-3",
            "local_dir": str(models_dir / "DA3"),
            "required": False,
            "description": "OPTIONAL: Alternative depth backend (manual download recommended)",
        },
    ]

    failed = []
    skipped = []

    for i, dl in enumerate(downloads, 1):
        print("=" * 80)
        print(f"[{i}/{len(downloads)}] {dl['name']}")
        print("=" * 80)
        print(f"Repo:     {dl['repo']}")
        print(f"Location: {dl['local_dir']}")
        print(f"Status:   {'REQUIRED' if dl['required'] else 'OPTIONAL'}")
        print(f"Description: {dl['description']}")
        print()

        if not dl["required"]:
            response = input("Download this optional model? (y/n, default n): ").lower()
            if response != "y":
                print(f"⊘ Skipped: {dl['name']}")
                skipped.append(dl['name'])
                continue

        cmd = f"huggingface-cli download {dl['repo']} --local-dir {dl['local_dir']}"
        if run_cmd(cmd, f"Downloading {dl['name']}..."):
            print(f"✓ {dl['name']} downloaded successfully")
        else:
            print(f"✗ Failed to download {dl['name']}")
            if dl["required"]:
                failed.append(dl['name'])

    # Summary
    print("\n" + "=" * 80)
    print("DOWNLOAD SUMMARY")
    print("=" * 80)

    if not failed:
        print("\n✓ All required models downloaded successfully!")
    else:
        print(f"\n✗ Failed to download {len(failed)} required model(s):")
        for name in failed:
            print(f"  - {name}")
        return 1

    if skipped:
        print(f"\n⊘ Skipped {len(skipped)} optional model(s):")
        for name in skipped:
            print(f"  - {name}")

    print(f"\nModel locations:")
    print(f"  All models: {models_dir.absolute()}")
    print(f"  EVOKE:      {models_dir / 'evoke'}")
    print(f"  ViGeo:      {models_dir / 'ViGeo1.1'}")
    print(f"  DA3:        {models_dir / 'DA3'}")

    print(f"\nNext steps:")
    print(f"  1. cd {repo_root / 'scripts' / 'inference'}")
    print(f"  2. bash infer_post_distill.sh")
    print(f"  3. See scripts/inference/README.md for all modes")

    print(f"\nInference modes:")
    print(f"  MODE=t2v     NUM_CHUNKS=20  bash scripts/inference/infer_post_distill.sh")
    print(f"  MODE=i2v     NUM_CHUNKS=20  bash scripts/inference/infer_post_distill.sh")
    print(f"  MODE=v2v     NUM_CHUNKS=20  bash scripts/inference/infer_post_distill.sh")
    print(f"  MODE=segment NUM_CHUNKS=6   bash scripts/inference/infer_post_distill.sh")

    print("\n" + "=" * 80)

    return 0

if __name__ == "__main__":
    sys.exit(main())
