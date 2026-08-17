# Evoke Setup Guide

**Alaya-EVOKE: From Linear-Scaling Supervision to Endless World**

A three-step, CFG-free world model that generates 1.5s of 384×640 video every 2.11s on one H200.

## Quick Start

```bash
cd C:\workspace\world\Evoke
setup_evoke.bat
```

This will:
1. Create `.venv` (Python 3.10+)
2. Install torch 2.4.0+ (CUDA 12.8)
3. Install all dependencies
4. Download all model weights (~26 GB minimal)

## Space Requirements

**This setup downloads MINIMAL weights only (~14 GB):**

| Component | Size | Status | Purpose |
|-----------|------|--------|---------|
| **evoke-base** | ~10 GB | ✓ Downloaded | VAE encoder/decoder, text encoder, tokenizer, scheduler |
| **stage3_post_distillation** | ~14 GB | ✓ Downloaded | **Shipped model** (3-step, CFG-free) |
| **ViGeo depth backend** | ~2 GB | ✓ Downloaded | Monocular depth estimation for world state |
| **stage1_camera_control** | ~14 GB | ✗ Skipped | 50-step multi-step model (baseline) |
| **stage2_few_step_training** | ~14 GB | ✗ Skipped | Pyramid distillation checkpoint |
| **stage3_long_distillation** | ~14 GB | ✗ Skipped | 30s long-video distillation init |
| **evoke_teacher** | ~28 GB | ✗ Skipped | DMD teacher (50-step, training only) |
| **DA3 depth backend** | ~12 GB | ✗ Skipped | Alternative depth backend (Depth-Anything-3) |
| **Total (this setup)** | **~26 GB** | — | evoke-base + shipped model + ViGeo only |

**To download full model suite later** (all stages + teacher + depth backends), see "Manual Model Download" section.

## Installation Steps

### 1. Prerequisites
- **Python 3.10** (3.11+ may work, 3.12 has torch issues on Windows)
- **CUDA 12.8** (GPU required; no CPU-only inference)
- **~50 GB free disk** (source repos + venv + minimal models + outputs)
- **HuggingFace account** (for `SII-YuanyangYin/Evoke` gated access; request approval on HF)

### 2. Clone and Enter Directory
```bash
cd C:\workspace\world\Evoke
```

### 3. Run Setup Script
```bash
setup_evoke.bat
```

The script will:
- Create `.venv` with Python 3.10
- Install PyTorch 2.4.0 + CUDA 12.4
- Install diffusers (development version from git)
- Download all models from HuggingFace

### 4. Manual Model Download (if setup script fails)

**Minimal download** (what setup_evoke.bat gets):

```bash
# Activate venv
.venv\Scripts\activate.bat

# Install HF CLI
pip install huggingface-hub

# Set token (one-time, required for gated repo)
huggingface-cli login

# Download evoke-base + shipped model only (~12 GB)
huggingface-cli download SII-YuanyangYin/Evoke --local-dir models --include "evoke-base/*" --include "evoke/stage3_post_distillation/*"

# Download ViGeo depth backend (~2 GB, REQUIRED)
huggingface-cli download pkqbajng/ViGeo --local-dir models/ViGeo1.1
```

**Full suite** (if you want all models for comparison):

```bash
# Download everything
huggingface-cli download SII-YuanyangYin/Evoke --local-dir models

# Optional: Depth-Anything-3 alternative (~12 GB)
# Get config.json + model.safetensors from:
#   https://huggingface.co/spaces/depth-anything/depth-anything-3
# Place in: models/DA3/
```

### 5. Check Installation
```bash
.venv\Scripts\activate.bat
python -c "import torch; import diffusers; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')"
```

## Model Descriptions

### Shipped Model (stage3_post_distillation)
- **Steps:** 3 (CFG-free)
- **Speed:** 2.11s per chunk on H200 (1.5s of video)
- **Use case:** Real-time inference, demos
- **Quality:** State-of-the-art on WBench (80.8), competitive on VBench-Long (85.11)

### Stage 1 (stage1_camera_control)
- **Steps:** 50 with CFG 5.0
- **Speed:** ~120s per chunk (slow, for reference)
- **Use case:** Baseline comparison, training initialization
- **Conditioning:** v2v / i2v / t2v all in distribution

### Teacher (evoke_teacher)
- **Steps:** 50 with CFG 5.0
- **Size:** ~28 GB (two experts: high_noise + low_noise)
- **Use case:** Training only, not inference
- **Role:** DMD (Diffusion to Merging) teacher for knowledge distillation

### Depth Backends

**ViGeo (default, REQUIRED)**
- ~2 GB, all recipes use this
- Fast monocular depth from single frames
- Enables world state bank (geometry continuity)

**Depth-Anything-3 (optional)**
- ~12 GB (da3-giant)
- Higher quality depth but slower
- Swap via config: `DEPTH_BACKEND=da3` in scripts

## Running Inference

All examples use bundled data in `examples/` (no external dataset needed). Activate venv first:

```bash
.venv\Scripts\activate.bat
```

Then run any of these:

```bash
# Prompt-to-video (no reference) — 20 chunks = 30s
MODE=t2v NUM_CHUNKS=20 bash scripts/inference/infer_post_distill.sh

# Image-to-video (first frame + camera motion)
MODE=i2v NUM_CHUNKS=20 bash scripts/inference/infer_post_distill.sh

# Video-to-video (reference video + camera motion)
MODE=v2v NUM_CHUNKS=20 bash scripts/inference/infer_post_distill.sh

# Re-prompt mid-rollout (switch prompt at chunk 3 of 6)
MODE=segment NUM_CHUNKS=6 MAX_CASES=0 bash scripts/inference/infer_post_distill.sh
```

**Key parameters:**
- One chunk = 36 frames = 1.5s
- `NUM_CHUNKS=20` = 30s video
- `NUM_CHUNKS=6` = ~9.5s (default demo length)
- Resolution: 384×640 (fixed in recipe)

**Output:** `outputs/<case>/geo_pred.mp4`

For full options and custom data, see `scripts/inference/README.md`.

## Troubleshooting

### "ModuleNotFoundError: No module named 'diffusers'"
```bash
pip install git+https://github.com/huggingface/diffusers.git
```

### "ImportError: cannot import name 'ffmpeg'" or "postprocess_viz.py failed"
ffmpeg is missing. Install from:
- https://ffmpeg.org/download.html
- Or via Chocolatey: `choco install ffmpeg`
- Or via Scoop: `scoop install ffmpeg`

Ensure `ffmpeg` is on PATH:
```bash
where ffmpeg
```

### "GatedRepoError" when downloading Evoke
The `SII-YuanyangYin/Evoke` repo is gated (requires approval). Options:
1. Request access on HuggingFace
2. Login: `huggingface-cli login`
3. Or use setup script with `HF_TOKEN` env var

### "CUDA out of memory" during inference
- Reduce `NUM_CHUNKS` (default: 20)
- Reduce resolution in config (currently 384×640)
- Close other GPU applications

### "ffmpeg: not found" in inference
```bash
# Linux / WSL
sudo apt install ffmpeg

# Windows
choco install ffmpeg
# or download from ffmpeg.org
```

## Directory Structure

```
models/
├── evoke-base/                              # VAE + text encoder
│   ├── vae_model.safetensors
│   ├── text_encoder/
│   ├── tokenizer/
│   └── scheduler/
├── evoke/
│   ├── stage1_camera_control/transformer/   # 50-step baseline
│   ├── stage2_few_step_training/transformer/
│   ├── stage3_long_distillation/transformer/
│   ├── stage3_post_distillation/transformer/ # Shipped model (3-step)
│   └── evoke_teacher/                       # Training only
│       ├── high_noise/
│       └── low_noise/
├── ViGeo1.1/
│   └── vigeo.pt                             # Depth backend (REQUIRED)
└── DA3/                                     # Optional
    ├── config.json
    └── model.safetensors
```

## Performance

Benchmark results from paper (arxiv 2608.13546):

| Benchmark | Score | Rank | Notes |
|-----------|-------|------|-------|
| **WBench (Navigation)** | 80.8 | 1st | Video Quality (82.8), Setting (83.8), Physical (72.1) |
| **VBench-2.0** | 66.77 | 1st | Nearest: Veo 3 (66.72) |
| **VBench-Long** | 85.11 | 7th | Extended generation (max: IPOW 88.26) |
| **Throughput** | 2.11s | — | Per 1.5s chunk on H200 |
| **Long-session** | 2 hours | — | Uninterrupted generation |

## Notes

- **Warp/attention recipe must match** between training and inference — mismatch silently degrades quality
- **Resolution must be multiple of 64** for the long tier and pyramid stage to divide evenly
- **Diffusers version is pinned** — development fork, not PyPI (see requirements.txt)
- **Teacher is training-only** — distilled models (`stage3_*`) are for inference
- **ViGeo is required** — world state bank depends on it for monocular depth

## Links

- **Project page:** https://evoke-world.github.io/Evoke/
- **Paper:** https://arxiv.org/abs/2608.13546
- **Models:** https://huggingface.co/SII-YuanyangYin/Evoke
- **License:** Apache 2.0 (depth backends are CC-BY-NC-4.0, more restrictive)

## Environment

- Python 3.10 (3.11+ tolerated, 3.12 breaks torch on Windows)
- PyTorch 2.4.0 (load-bearing, pinned for SDPA semantics)
- CUDA 12.8
- Diffusers: 0.39.0.dev0 (development version, not PyPI)
- DeepSpeed: 0.14.5 (training only, load-bearing for ZeRO-2)
- Flash-Attention: 2.8.3 (load-bearing, FA2 backend)
- Triton: 3.0.0 (load-bearing, fused kernels)

**All packages fully compatible with CUDA 12.8.** Install torch first from cu128 index, then requirements.txt automatically pulls CUDA 12.8 wheels for GPU dependencies.
