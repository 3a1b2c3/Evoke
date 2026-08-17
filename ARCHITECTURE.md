# Evoke Inference Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      EVOKE INFERENCE STACK                      │
└─────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│                     USER INTERFACE LAYER                         │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  .bat Scripts (Windows)          Python Scripts                 │
│  ├─ setup_evoke.bat             ├─ run_examples_python.py       │
│  ├─ download_models.bat         ├─ evaluate_evoke.py (MIND)     │
│  ├─ run_examples.bat            └─ infer_streaming.py           │
│  └─ build_extension.bat                                         │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
                                  ↓
┌──────────────────────────────────────────────────────────────────┐
│                    INFERENCE ENGINE LAYER                        │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Evoke Pipeline (diffusers)                                    │
│  ├─ Text Encoder (from evoke-base)                             │
│  ├─ DiT (Diffusion Transformer, stage3_post_distillation)      │
│  ├─ VAE Decoder (from evoke-base)                              │
│  ├─ Scheduler (DDPM)                                           │
│  └─ World State Bank (Geometry Management)                     │
│      ├─ Depth Estimation (ViGeo 1.1)                           │
│      ├─ Point Cloud Storage                                    │
│      └─ Camera Warp Rendering                                  │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
                                  ↓
┌──────────────────────────────────────────────────────────────────┐
│                   COMPUTATION & RUNTIME LAYER                    │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  PyTorch 2.7.0                                                  │
│  ├─ CUDA 13.0 (RTX 5090, sm_120)                               │
│  ├─ Flash-Attention 2.8.3 (FA2 backend)                        │
│  ├─ Triton-Windows 3.3.0 (fused kernels)                       │
│  └─ DeepSpeed 0.14.5 (training only)                           │
│                                                                  │
│  GPU Memory:                                                    │
│  ├─ Model weights: ~14 GB (DiT + text encoder + VAE)           │
│  ├─ KV cache: ~2-3 GB per chunk (32 frames)                    │
│  ├─ Activations: ~1-2 GB                                       │
│  └─ Total: ~18-20 GB per inference                             │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
                                  ↓
┌──────────────────────────────────────────────────────────────────┐
│                    DATA & STORAGE LAYER                          │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  HuggingFace Cache                                              │
│  └─ ~/.cache/huggingface/hub/                                   │
│     ├─ models--SII-YuanyangYin--Evoke/                          │
│     │  ├─ evoke-base/ (10 GB)                                   │
│     │  └─ stage3_post_distillation/ (14 GB)                     │
│     └─ models--pkqbajng--ViGeo/                                 │
│        └─ vigeo.pt (2 GB)                                       │
│                                                                  │
│  Local Outputs                                                  │
│  └─ outputs/                                                    │
│     ├─ t2v/geo_pred.mp4                                         │
│     ├─ i2v/geo_pred.mp4                                         │
│     ├─ v2v/geo_pred.mp4                                         │
│     └─ segment/geo_pred.mp4                                     │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

## Inference Flow

```
USER INPUT
    ↓
┌─────────────────────────┐
│  Prepare Conditioning   │
├─────────────────────────┤
│ • Parse prompt/image    │
│ • Extract camera pose   │
│ • Encode text (CLIP)    │
└─────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────┐
│  STREAMING CHUNK GENERATION (3-step diffusion)      │
├─────────────────────────────────────────────────────┤
│                                                     │
│ FOR each chunk (36 frames):                         │
│                                                     │
│ 1. Read World State Bank                            │
│    └─ Retrieve geometry from previous chunks        │
│       using camera pose (monocular depth)           │
│                                                     │
│ 2. Denoise Loop (3 steps)                           │
│    └─ Step 1-3: Diffusion Transformer (DiT)         │
│       ├─ Input: noise + conditioning                │
│       ├─ Attention: KV cache (previous blocks)      │
│       └─ Output: latent frame                       │
│                                                     │
│ 3. Decode Latents                                   │
│    └─ VAE decode: latent → pixel space              │
│                                                     │
│ 4. Update World State Bank                          │
│    └─ Estimate depth (ViGeo)                        │
│       Unproject into 3D point cloud                 │
│       Store indexed by camera pose                  │
│                                                     │
│ 5. Yield Frame Block (36 frames)                    │
│    └─ Send to video encoder                         │
│                                                     │
└─────────────────────────────────────────────────────┘
    ↓ (repeat until all chunks done)
┌─────────────────────────┐
│  Encode to Video        │
├─────────────────────────┤
│ • H.264 codec           │
│ • 384×640, 24fps        │
│ • 1.5s per chunk        │
└─────────────────────────┘
    ↓
┌─────────────────────────┐
│  OUTPUT VIDEO           │
├─────────────────────────┤
│ • geo_pred.mp4          │
│ • ~50-100 MB per chunk  │
└─────────────────────────┘
```

## Memory Profile

```
GPU Memory Timeline (per inference)

Time 0ms: Model Load
├─ DiT weights: 14 GB
├─ Text encoder: 1 GB
├─ VAE: 0.5 GB
└─ Total: ~15.5 GB (peak during load)

Time 100ms - Chunk 0 Start
├─ Model weights: 15.5 GB (resident)
├─ Chunk 0 KV cache: 2.5 GB
├─ Activations: 1 GB
├─ Intermediate buffers: 1 GB
└─ Total: ~20 GB

Time 2100ms - Chunk 0 Complete, Chunk 1 Start
├─ Model weights: 15.5 GB
├─ Chunk 1 KV cache: 2.5 GB (chunk 0 cache freed)
├─ Activations: 1 GB
└─ Total: ~19 GB (bounded)

... (repeats for each chunk, memory stays bounded)
```

## Inference Modes

```
┌─────────────────────────────────────────────────────────┐
│                  T2V (Text-to-Video)                    │
├─────────────────────────────────────────────────────────┤
│ Input:  prompt (text)                                   │
│ Output: 384×640 video, 24 fps                           │
│ Camera: Fixed (free to move generated)                  │
│ Use:    World generation from text                      │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                  I2V (Image-to-Video)                   │
├─────────────────────────────────────────────────────────┤
│ Input:  first frame (image) + camera trajectory         │
│ Output: continuation video, matching camera motion      │
│ Camera: Constrained by trajectory                       │
│ Use:    Extending image with camera control             │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                  V2V (Video-to-Video)                   │
├─────────────────────────────────────────────────────────┤
│ Input:  reference video + camera trajectory             │
│ Output: edited video following new camera path          │
│ Camera: Constrained by trajectory                       │
│ Use:    Video-driven generation with camera control     │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│              SEGMENT (Re-prompt Mid-Flight)             │
├─────────────────────────────────────────────────────────┤
│ Input:  initial prompt + schedule (chunk → prompt)      │
│ Output: video with prompt switches (no cut)             │
│ Camera: Continuous, driven by scheduling                │
│ Use:    Scene transitions without restart               │
└─────────────────────────────────────────────────────────┘
```

## Component Interactions

```
run_examples.bat
    ↓
run_examples_python.py
    ↓
evoke/pipelines/pipeline_evoke.py
    ├─ Load models from HF cache
    ├─ Initialize EvokePipeline
    └─ Call pipeline(prompt/image/video, num_chunks=6)
        ↓
        ├─ Text Encoder: prompt → conditioning
        ├─ FOR each chunk:
        │   ├─ Diffusion Loop (3 steps)
        │   │   └─ Transformer3DModel (DiT)
        │   │       ├─ Read KV cache
        │   │       ├─ Denoising pass
        │   │       └─ Write KV cache
        │   ├─ VAE Decode: latent → frames
        │   ├─ Depth Estimate (ViGeo)
        │   ├─ World State Update
        │   └─ Yield frames to video encoder
        └─ Save output.mp4

evaluate_evoke.py (MIND metrics)
    ├─ Extract frames
    ├─ Optical flow (motion)
    ├─ Frame diff (consistency)
    ├─ Laplacian (sharpness)
    └─ Save metrics.json
```

## Deployment Checklist

- [x] Virtual environment (.venv)
- [x] PyTorch 2.7.0 (CUDA 13.0)
- [x] Flash-Attention 2.8.3
- [x] Triton-Windows 3.3.0
- [x] Diffusers (dev version)
- [x] Models cached in HF (~26 GB)
- [x] Inference pipeline (run_examples.bat)
- [x] Evaluation suite (MIND metrics)
- [ ] Training setup (optional, requires more GPU mem)
