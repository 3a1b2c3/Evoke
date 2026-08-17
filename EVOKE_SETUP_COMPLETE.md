# Evoke Setup — Complete

**Status:** ✓ Installed and ready for inference

## What's Installed

### Environment
- **Python:** 3.10
- **PyTorch:** 2.7.0 (CUDA 12.8)
- **CUDA:** 12.8
- **Location:** `.venv/` (activated)

### Core Packages (from requirements.txt)
- torch 2.7.0 (cu128)
- flash-attn 2.8.3 (FA2 backend)
- triton-windows 3.3.0 (fused kernels)
- transformers 5.3.0
- diffusers 0.39.0.dev0
- deepspeed 0.14.5 (training)
- peft, safetensors, huggingface-hub, opencv, pillow, etc.

### Models

**Location:** HuggingFace cache
```
C:\Users\kschmid\.cache\huggingface\hub\models--SII-YuanyangYin--Evoke\
├── evoke-base/                          (10 GB)
│   ├── vae_model.safetensors
│   ├── text_encoder/
│   ├── tokenizer/
│   └── scheduler/
└── evoke/stage3_post_distillation/      (14 GB) ← Shipped model (3-step, CFG-free)
    └── transformer/
        ├── config.json
        ├── diffusion_pytorch_model.safetensors
        └── ...

C:\Users\kschmid\.cache\huggingface\hub\models--pkqbajng--ViGeo\
└── vigeo.pt                             (2 GB, depth backend)
```

**Note:** Models are NOT duplicated in local `C:\workspace\world\Evoke\models\` — they live in HF cache only.

## Running Inference

### Quick Start

```powershell
cd C:\workspace\world\Evoke
.venv\Scripts\activate.bat
bash run_examples.sh
```

This runs 4 demo modes (each ~9.5s, 384×640):
1. **t2v** — Text-to-video (prompt only)
2. **i2v** — Image-to-video (first frame + camera motion)
3. **v2v** — Video-to-video (reference video + camera motion)
4. **segment** — Re-prompt mid-rollout (prompt switches at chunk 3)

### Output
Videos saved to: `outputs/<mode>/geo_pred.mp4`

### Manual Inference

Activate venv, then:

```bash
# Text-to-video
MODE=t2v NUM_CHUNKS=20 bash scripts/inference/infer_post_distill.sh

# Image-to-video
MODE=i2v NUM_CHUNKS=20 bash scripts/inference/infer_post_distill.sh

# Video-to-video
MODE=v2v NUM_CHUNKS=20 bash scripts/inference/infer_post_distill.sh

# Re-prompt mid-rollout
MODE=segment NUM_CHUNKS=6 MAX_CASES=0 bash scripts/inference/infer_post_distill.sh
```

**Parameters:**
- `NUM_CHUNKS=20` = 30s video (one chunk = 1.5s)
- `NUM_CHUNKS=6` = ~9.5s video (demo length)
- Resolution: 384×640 (fixed in recipe)

## Model Loading via HF Cache

Evoke auto-detects models from HF cache. No local `models/` directory needed.

If inference scripts can't find models, set:

```powershell
# PowerShell
$env:HF_HOME = "$env:USERPROFILE\.cache\huggingface"
$env:HF_TOKEN = "your-hf-token-if-needed"
```

Or in bash:
```bash
export HF_HOME=$HOME/.cache/huggingface
bash scripts/inference/infer_post_distill.sh
```

## Scripts

| Script | Purpose |
|--------|---------|
| `setup_evoke.bat` | Initial setup (venv + deps + attempt download) |
| `download_models.bat` | Download models to HF cache (~26 GB) |
| `run_examples.bat` | Activate venv + run bash examples |
| `run_examples.sh` | 4 inference demos (t2v, i2v, v2v, segment) |
| `scripts/inference/infer_post_distill.sh` | Shipped model inference (3-step, CFG-free) |
| `scripts/inference/infer_stage1.sh` | Stage 1 baseline (50-step, CFG 5.0) |
| `scripts/inference/infer_evoke_teacher.sh` | Teacher model (50-step, training only) |

## Troubleshooting

### "Models not found" during inference
Models are in HF cache, not local `models/` directory. Evoke should auto-detect them.

**If not found:**
1. Check HF cache exists:
   ```powershell
   ls "$env:USERPROFILE\.cache\huggingface\hub\"
   ```
2. Verify downloads completed (check earlier output)
3. Set `HF_HOME` explicitly before running inference

### "torch not installed" or CUDA errors
Reinstall torch with correct CUDA:
```powershell
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
```

### "triton-windows not found"
Install from requirements:
```powershell
pip install -r requirements.txt
```

### Network timeouts during download
Use `hf` CLI directly:
```powershell
hf download SII-YuanyangYin/Evoke --local-dir-use-symlinks False
hf download pkqbajng/ViGeo
```

Models will auto-cache in HF cache.

### "ffmpeg not found"
Some postprocessing needs ffmpeg. Install:
```powershell
choco install ffmpeg
# or download from https://ffmpeg.org/download.html
```

## Environment Summary

```
Evoke 2026 (arxiv 2608.13546)
├── Models: SII-YuanyangYin/Evoke (shipped: stage3_post_distillation)
├── Depth backend: pkqbajng/ViGeo (required)
├── PyTorch: 2.7.0 (CUDA 12.8)
├── Attention: Flash-Attn 2.8.3 (FA2)
├── Kernels: triton-windows 3.3.0
└── Inference: 2.11s per 1.5s chunk (384×640@24fps)
```

## What's Next

1. **Run demos:** `bash run_examples.sh`
2. **Explore inference:** See `scripts/inference/README.md` for custom data
3. **Training:** Use `train_evoke.py` with your own data (requires setup beyond this)

## References

- **Project:** https://evoke-world.github.io/Evoke/
- **Paper:** https://arxiv.org/abs/2608.13546
- **Models:** https://huggingface.co/SII-YuanyangYin/Evoke
- **Depth backend:** https://huggingface.co/pkqbajng/ViGeo
