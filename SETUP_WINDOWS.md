# Evoke Setup — Windows (Local RTX 5090, 32GB)

**Date**: 2026-08-16  
**Status**: Ready for inference  
**Platform**: Windows 11, Python 3.10, CUDA 13.0, PyTorch 2.7.0

---

## Quick Start

### 1. Create Virtual Environment

```cmd
cd C:\workspace\world\Evoke
python -m venv .venv
.\.venv\Scripts\activate
```

### 2. Install Dependencies

```cmd
C:\Users\kschmid\.local\bin\uv.exe pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128 --python .\.venv\Scripts\python.exe
C:\Users\kschmid\.local\bin\uv.exe pip install -r requirements.txt --python .\.venv\Scripts\python.exe
```

**Note**: Requirements pinned to torch 2.7.0 (CUDA 12.8 compatible) with cu128 index.

### 3. Download Models (~12-14 GB)

```cmd
python -c "from huggingface_hub import snapshot_download; snapshot_download('SII-YuanyangYin/Evoke', local_dir='models', allow_patterns=['evoke-base/**', 'evoke/stage3_post_distillation/**'])"
python -c "from huggingface_hub import snapshot_download; snapshot_download('pkqbajng/ViGeo', local_dir='models/ViGeo1.1', allow_patterns=['*.pt'])"
```

### 4. Run Examples

```cmd
.\run_examples.bat
```

---

## Environment Details

| Component | Version | Index | Notes |
|-----------|---------|-------|-------|
| Python | 3.10 | system | |
| PyTorch | 2.7.0 | cu128 | cu128 for RTX 5090 |
| CUDA | 13.0 | - | System CUDA toolkit |
| torchvision | 0.22.0 | cu128 | Compatible with torch 2.7.0 |

---

## What Each Step Does

### Step 1: Virtual Environment
- Creates isolated Python environment at `.venv/`
- Activates it for dependency installation

### Step 2: Install Dependencies

**PyTorch for CUDA 12.8** (local RTX 5090):
```cmd
C:\Users\kschmid\.local\bin\uv.exe pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128 --python .\.venv\Scripts\python.exe
```

Key dependencies (from requirements.txt):
- `flash-attn==2.8.3` — optimized attention backend
- `diffusers==0.39.0.dev0` — diffusion pipelines
- `transformers==5.3.0` — text encoders
- `deepspeed==0.14.5` — training optimization
- `triton==3.0.0` — kernel compilation

### Step 3: Download Models

Two models needed:

**Evoke Base** (VAE + text encoder):
- Repo: `SII-YuanyangYin/Evoke`
- Download: `evoke-base/` + `evoke/stage3_post_distillation/`
- Size: ~12 GB

**ViGeo** (monocular depth backend):
- Repo: `pkqbajng/ViGeo`
- Download: `models/ViGeo1.1/*.pt`
- Size: ~2 GB

### Step 4: Run Examples

```cmd
.\run_examples.bat
```

Runs inference scripts in `scripts/inference/`.

---

## Troubleshooting

### ModuleNotFoundError: No module named 'einops'

**Cause**: Dependencies not installed

**Fix**:
```cmd
C:\Users\kschmid\.local\bin\uv.exe pip install -r requirements.txt --python .\.venv\Scripts\python.exe
```

### CUDA out of memory

**Cause**: Model too large for 32GB GPU

**Fix**: Reduce batch size or resolution in config.

### ModuleNotFoundError: No module named 'huggingface_hub'

**Cause**: huggingface_hub not installed

**Fix**:
```cmd
C:\Users\kschmid\.local\bin\uv.exe pip install huggingface-hub --python .\.venv\Scripts\python.exe
```

---

## Inference Modes

Supported modes (set `MODE` environment variable):

```cmd
set MODE=t2v
.\run_examples.bat

set MODE=i2v
.\run_examples.bat

set MODE=v2v
.\run_examples.bat
```

See `scripts/inference/README.md` for full options.

---

## Configuration Files

- `requirements.txt` — Python dependencies (pinned versions)
- `setup_evoke.bat` — Old setup script (may be outdated)
- `run_examples.bat` — Example inference runner
- `evoke/config/` — Model and pipeline configs

---

## Notes

- **CUDA 12.8 vs 13.0**: System has CUDA 13.0, but PyTorch 2.7.0 uses cu128 wheels (compatible via PTX JIT)
- **Flash-Attn**: Requires `triton-windows` for Windows (not standard `triton`)
- **Memory**: 32GB RTX 5090 is tight; monitor VRAM during inference
- **Training**: setup.txt references training on H200 + CUDA 12.4 + torch 2.4.0 (different from inference)

---

## References

- Original setup guide: `setup_evoke.bat`
- Model cards: https://huggingface.co/SII-YuanyangYin/Evoke
- Inference modes: `scripts/inference/README.md`

