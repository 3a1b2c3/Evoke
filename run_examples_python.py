#!/usr/bin/env python3
"""Run Evoke inference examples without bash/WSL."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Add Evoke to path
evoke_dir = Path(__file__).parent
sys.path.insert(0, str(evoke_dir))

import cv2
import numpy as np
import torch
from diffusers import AutoencoderKLWan
from evoke.pipelines.pipeline_evoke import EvokePipeline
from evoke.modules.transformer_evoke import EvokeTransformer3DModel
from evoke.diffusers_version.scheduling_evoke_diffusers import EvokeScheduler
from evoke.utils.ev_validation import add_joystick_overlay_from_c2ws, load_pose_for_v2v, load_ref_video_for_v2v
from diffusers.models.modeling_utils import ModelMixin

# post_distill's real 3-step recipe (scripts/inference/infer_post_distill.sh): a 3-stage pyramid
# (1 step/stage), CFG off, and a scheduler rebuilt with stages=3 -- NOT the checkpoint's shipped
# stages=1 scheduler_config.json. Using num_inference_steps=50 with the default scheduler (as an
# earlier version of this script did) runs the pipeline's generic CFG path on a model that was
# specifically distilled to be CFG-free, which produced near-pure-noise output.
STAGE2_NUM_STAGES = 3
STAGE2_STAGE_RANGE = [0.0, 1 / 3, 2 / 3, 1.0]
STAGE2_STEPS = [1, 1, 1]
GUIDANCE_SCALE = 1.0  # CFG off; distillation removed the need for it

# The rollout advances one chunk per window_num_frames = (latent_window_size-1)*vae_temporal+1
# frames -- with latent_window_size=9 and Wan's temporal factor 4 that's 33 frames/chunk. A prompt
# schedule that switches at start_chunk 3 therefore does NOTHING unless the clip runs >=4 chunks, so
# for segment_prompts cases we size num_frames to the schedule's last switch plus a tail of chunks
# that actually show the post-switch scene.
CHUNK_FRAMES = 33
SEGMENT_TAIL_CHUNKS = 3  # chunks rendered AFTER the last prompt switch (so the transition is visible)

# Patch from_pretrained to strip max_memory before calling parent
_orig_from_pretrained = ModelMixin.from_pretrained.__func__
@classmethod
def _patched_from_pretrained(cls, *args, **kwargs):
    kwargs.pop("max_memory", None)
    kwargs.pop("_fast_init", None)
    return _orig_from_pretrained(cls, *args, **kwargs)
EvokeTransformer3DModel.from_pretrained = _patched_from_pretrained


def _load_case(evoke_dir, case_dir, name=None):
    """Read an examples/<case_dir>/cases.jsonl row (the repo's real inference examples).

    Defaults to the first row. Pass `name` to select a specific case by its "name" field --
    e.g. segment_prompts/cases.jsonl stacks 4 cases (aurora/meteor/crystalstorm/gateway)."""
    cases_path = evoke_dir / "examples" / case_dir / "cases.jsonl"
    with open(cases_path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if name is not None:
        for row in rows:
            if row.get("name") == name:
                return row
        raise ValueError(f"case '{name}' not found in {cases_path}")
    return rows[0]


def _load_prompt(evoke_dir, case):
    prompt_path = case.get("prompt_path")
    if prompt_path:
        full_path = evoke_dir / prompt_path
        if full_path.suffix == ".json":
            with open(full_path) as f:
                return json.load(f)["overall"]["short_prompt"]
        return full_path.read_text().strip()
    return case["prompt"]


def _load_pose(evoke_dir, case, height, width, num_frames):
    """Load a case's vipe pose.npz as (lingbot_Ks, lingbot_c2ws), sized for this run's resolution."""
    pose_path = evoke_dir / case["pose_path"]
    src_h, src_w = case.get("pose_source_resolution", (480, 832))
    return load_pose_for_v2v(
        str(pose_path),
        target_height=height,
        target_width=width,
        source_resolution=(src_h, src_w),
        pose_type=case.get("pose_type", "vipe"),
        num_target_frames=num_frames,
        target_fps=24,
        source_fps=case.get("pose_fps", 30),
    )


def _save_video(video_np, output_path, fps=24):
    """Encode a [T,H,W,3] uint8 RGB array to mp4 (matches scripts/inference/postprocess_viz.py)."""
    h, w = video_np.shape[1], video_np.shape[2]
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (int(w), int(h)))
    for frame in video_np:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def _measure_video_fps(video_path):
    """Read back the mp4's actual container fps/frame count -- don't just trust the encode call."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, frame_count


def _gpu_stats(label):
    """Print util/VRAM/power so a low-power+near-max-VRAM run (Windows UVM paging/thrashing) is
    visible in the log instead of only being caught by someone manually watching nvidia-smi."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw",
             "--format=csv,noheader,nounits"],
            text=True, timeout=10,
        ).strip()
        util, mem_used, mem_total, power = [x.strip() for x in out.split(",")]
        print(f"  [gpu:{label}] util={util}% mem={mem_used}/{mem_total} MiB power={power}W")
    except Exception as e:
        print(f"  [gpu:{label}] nvidia-smi query failed: {e}")


def run_inference(pipeline, engine_mode, case_dir=None, case_name=None, num_frames=25, output_dir="outputs",
                   height=256, width=448, image_noise_sigma_min=0.02, image_noise_sigma_max=0.05):
    """Run Evoke inference in `engine_mode` (t2v/i2v/v2v) using examples/<case_dir>/cases.jsonl.

    case_dir defaults to engine_mode (the repo's own examples/{t2v,i2v,v2v}/ layout), but can point
    at any directory with a compatible cases.jsonl -- e.g. examples/racer or examples/2, which are
    i2v-shaped cases living outside the repo's own examples/i2v/.

    case_name selects a specific row by "name" when the cases.jsonl stacks several (segment_prompts);
    output then goes to outputs/<case_dir>/<case_name>/ so the four cases don't overwrite each other.

    image_noise_sigma_min/max: pipeline defaults are 0.111/0.135; lower keeps the model anchored
    closer to the seed-image pixels instead of drifting toward the text prompt.
    """
    case_dir = case_dir or engine_mode
    out_sub = f"{case_dir}/{case_name}" if case_name else case_dir
    output_dir = Path(output_dir) / out_sub
    output_dir.mkdir(parents=True, exist_ok=True)

    pred_path = output_dir / "geo_pred.mp4"
    if pred_path.exists():
        print(f"\n[Evoke] Case: {out_sub} -- skip (already exists: {pred_path})")
        return True

    case = _load_case(evoke_dir, case_dir, case_name)
    prompt = _load_prompt(evoke_dir, case)

    # Per-chunk prompt schedule: segment_prompts cases ship a schedule_<name>.json
    # ([{start_chunk, prompt}, ...]). The pipeline switches prompt_embeds at each start_chunk
    # (pipeline_evoke.py:2019-2037,2412-2415) -- WITHOUT this the clip renders one static prompt
    # and the scene never transitions (why geo_pred_hud.mp4 looked like it "did nothing").
    chunk_prompts = None
    seg_path = case.get("segment_prompts_path")
    if seg_path:
        with open(evoke_dir / seg_path) as f:
            _sched = json.load(f)
        chunk_prompts = {int(e["start_chunk"]): e["prompt"] for e in _sched}
        # Grow the clip so the schedule can actually reach its last switch AND show it: run up to
        # (last switch chunk + tail) chunks. Never shrink an explicitly larger caller request.
        need_chunks = max(chunk_prompts) + 1 + SEGMENT_TAIL_CHUNKS
        num_frames = max(num_frames, need_chunks * CHUNK_FRAMES)
        print(f"  Segment schedule: {len(_sched)} prompt(s), switch at chunks {sorted(chunk_prompts)} "
              f"-> {need_chunks} chunks / {num_frames} frames")

    print(f"\n[Evoke] Mode: {engine_mode}, Case: {out_sub}, Frames: {num_frames}, Resolution: {height}x{width}")
    print(f"  Case: {case['name']}")
    print(f"  Prompt: {prompt}")

    kwargs = dict(
        prompt=prompt,
        num_frames=num_frames,
        height=height,
        width=width,
        is_enable_stage2=True,
        stage2_num_inference_steps_list=STAGE2_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        image_noise_sigma_min=image_noise_sigma_min,
        image_noise_sigma_max=image_noise_sigma_max,
    )
    if chunk_prompts:
        kwargs["chunk_prompts"] = chunk_prompts

    lingbot_Ks = lingbot_c2ws = None
    if engine_mode == "i2v":
        from PIL import Image
        # .convert("RGB") -- some example PNGs (e.g. racer/Screenshot.png) carry an alpha
        # channel, which crashes the VAE's first conv (expects 3 channels, got 4).
        kwargs["image"] = Image.open(evoke_dir / case["image_path"]).convert("RGB")
    elif engine_mode == "v2v":
        kwargs["video"] = load_ref_video_for_v2v(
            str(evoke_dir / case["video_path"]),
            height=height,
            width=width,
            seconds=num_frames / 24.0,
            target_fps=24,
            source_fps=int(case.get("video_fps", 30)),
        )

    if "pose_path" in case:
        lingbot_Ks, lingbot_c2ws = _load_pose(evoke_dir, case, height, width, num_frames)
        # enable_model_cpu_offload() moves submodules to cuda only during their own forward pass
        # and evicts them back to cpu afterward. Hardcoding "cuda" here raced with that on
        # whichever case ran first in the process (accelerate hadn't relocated the relevant
        # submodule yet) -- pipeline._execution_device is accelerate's own resolved device and
        # is safe across the offload hooks regardless of call order.
        exec_device = pipeline._execution_device
        lingbot_Ks = lingbot_Ks.to(exec_device)
        lingbot_c2ws = lingbot_c2ws.to(exec_device)
        kwargs["lingbot_Ks"] = lingbot_Ks
        kwargs["lingbot_c2ws"] = lingbot_c2ws
        print(f"  Pose: {case['pose_path']} -> c2ws {tuple(lingbot_c2ws.shape)}")

    print(f"  Generating {num_frames} frames (requested; chunked internally, actual count may differ)...")
    gen_start = time.time()
    try:
        output = pipeline(**kwargs)
    except RuntimeError as e:
        # Pipeline bug (pipeline_evoke.py:1938-1945): latents_mean/latents_std are placed on
        # self.vae.device read BEFORE the VAE's cpu-offload hook has ever fired, so the very
        # FIRST pipeline() call in a process can see a stale cpu device there while
        # self.vae.encode() itself already ran on cuda via its hook -- device mismatch. Once
        # any call has fired the hook (regardless of which case), later calls in the same
        # process don't hit this, so retry once rather than fighting the accelerate hook
        # internals directly (that approach broke a different op -- see git history).
        if "Expected all tensors to be on the same device" not in str(e):
            raise
        print(f"  [retry] device-mismatch on first use of VAE offload hook, retrying once: {e}")
        output = pipeline(**kwargs)
    gen_elapsed = time.time() - gen_start
    _gpu_stats(f"post-generation:{case_dir}")

    video_np = output.frames[0]
    if not (video_np.dtype == np.uint8):
        video_np = (np.clip(video_np, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    actual_frames = video_np.shape[0]
    print(f"  Generation time: {gen_elapsed:.2f}s for {actual_frames} actual frames "
          f"({actual_frames / gen_elapsed:.2f} frames/s, {actual_frames / 24.0:.2f}s @ 24fps)")

    pred_path = output_dir / "geo_pred.mp4"
    _save_video(video_np, pred_path)
    measured_fps, measured_frames = _measure_video_fps(pred_path)
    print(f"  Saved: {pred_path} (measured: {measured_fps:.3f} fps, {measured_frames} frames, "
          f"{measured_frames / measured_fps:.2f}s)")

    if lingbot_c2ws is not None:
        c2ws_np = lingbot_c2ws.detach().cpu().numpy() if isinstance(lingbot_c2ws, torch.Tensor) else np.asarray(lingbot_c2ws)
        n = min(video_np.shape[0], c2ws_np.shape[0])
        hud_frames = add_joystick_overlay_from_c2ws(list(video_np[:n]), c2ws_np[:n], label_left="Move", label_right="Rot")
        hud_np = np.stack(list(hud_frames) + list(video_np[n:])) if n < video_np.shape[0] else np.stack(list(hud_frames))
        hud_path = output_dir / "geo_pred_hud.mp4"
        _save_video(hud_np, hud_path)
        print(f"  Saved (with key overlay): {hud_path}")

    return True


def main():
    """Run all inference examples."""

    print("="*70)
    print("EVOKE INFERENCE EXAMPLES -- Pure Python (No WSL/Bash)")
    print("="*70)

    # Caps the growing warp point-cloud/history state (scripts/inference/README.md); cheap,
    # zero-risk, required beyond ~10min of rollout. Override by setting the env var before running.
    os.environ.setdefault("GEO_HIST_MAX_FRAMES", "720")

    _gpu_stats("baseline")

    # Check models
    hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
    if not (hf_cache / "models--SII-YuanyangYin--Evoke").exists():
        print("\nERROR: Models not found in HF cache")
        print("Run: download_models.bat")
        return 1

    print("\nModels found")

    print("  Loading model...")
    # The published model_index.json's HF-repo-root load always fails (the repo holds several
    # checkpoints, not one loadable pipeline -- see the README's own note), so go straight to the
    # local snapshot's evoke-base subfolder instead of trying the direct load first.
    evoke_model_path = hf_cache / "models--SII-YuanyangYin--Evoke" / "snapshots"
    snapshots = list(evoke_model_path.glob("*"))

    # DEBUG: Show what we're looking for
    print(f"\n=== EVOKE MODEL DEBUG ===")
    print(f"HF_HOME: {os.environ.get('HF_HOME', 'not set')}")
    print(f"HF_CACHE: {hf_cache}")
    print(f"Looking in: {evoke_model_path}")
    print(f"Path exists: {evoke_model_path.exists()}")
    print(f"Snapshots found: {len(snapshots)}")
    if snapshots:
        print(f"First snapshot: {snapshots[0]}")
    print(f"========================\n")

    if not snapshots:
        raise RuntimeError(f"No Evoke model snapshots found in {evoke_model_path}")
    ckpt_path = snapshots[0] / "evoke-base"

    # VAE must stay fp32 -- Wan-style VAEs are numerically unstable in fp16, which was producing
    # near-pure-noise output even with the correct step count (scripts/inference/infer_single.py:848).
    vae = AutoencoderKLWan.from_pretrained(str(ckpt_path), subfolder="vae", torch_dtype=torch.float32)
    # post_distill's pyramid scheduler: the checkpoint's shipped scheduler_config.json has
    # stages=1, which silently runs the pipeline's generic (CFG, 50-step) path instead of the
    # 3-step distilled one this checkpoint actually needs (infer_single.py:849-863).
    scheduler = EvokeScheduler(
        num_train_timesteps=1000,
        shift=1.0,
        stages=STAGE2_NUM_STAGES,
        stage_range=STAGE2_STAGE_RANGE,
        gamma=1 / 3,
        scheduler_type="unipc",
        use_dynamic_shifting=False,
        time_shift_type="exponential",
    )
    # disable_mmap=True bypasses safetensors mmap on Windows, where the default pagefile
    # (~2-4 GB auto-managed) is too small to reserve the tens-of-GB virtual address space this
    # transformer's shards need -- otherwise: OSError: paging file is too small (os error 1455).
    # Same fix as _helios_i2v_worker.py. Trade-off: shards load into host RAM instead of mmap.
    transformer = EvokeTransformer3DModel.from_pretrained(
        str(ckpt_path), subfolder="transformer", torch_dtype=torch.float16, disable_mmap=True,
    )
    pipeline = EvokePipeline.from_pretrained(
        str(ckpt_path), transformer=transformer, vae=vae, scheduler=scheduler, torch_dtype=torch.float16,
    )
    # VRAM check confirmed the fp16 model alone fills ~97.6% of a 32GB card (31842/32607 MiB)
    # right after a plain .to("cuda") -- leaving almost no headroom for denoising activations,
    # which is what was causing the 100%-util/near-max-VRAM/low-power paging/thrashing pattern.
    # enable_model_cpu_offload() keeps components on CPU and moves each to GPU only while active.
    pipeline.enable_model_cpu_offload()
    _gpu_stats("post-load")

    # (engine_mode, case_dir, case_name) -- engine_mode picks the pipeline branch, case_dir picks
    # examples/<dir>/cases.jsonl, case_name selects one stacked row (None = first row).
    # All 4 segment_prompts cases now run, each with its schedule_<name>.json prompt schedule
    # applied via chunk_prompts (aurora tundra->aurora, meteor, crystalstorm, gateway transitions).
    runs = [
        ("i2v", "i2v", None),
        ("v2v", "v2v", None),
        ("i2v", "racer", None),
        ("i2v", "2", None),
        ("i2v", "segment_prompts", "aurora"),
        ("i2v", "segment_prompts", "meteor"),
        ("i2v", "segment_prompts", "crystalstorm"),
        ("i2v", "segment_prompts", "gateway"),
    ]

    for engine_mode, case_dir, case_name in runs:
        try:
            # 33 = Evoke's own hard minimum: (latent_window_size-1)*4+1 (v2v's reference-video
            # encode errors below this; i2v silently rounds up, so use the same floor everywhere).
            kwargs = dict(num_frames=33)
            if case_dir == "i2v":
                # image_noise_sigma tuning (0.02/0.05 -> 0.0/0.02 -> 0.0/0.005) never brought the
                # turtle back -- consistent with the real bottleneck being the coarse-to-fine
                # pyramid's 12x20 first stage (paper Sec 3.5), not how clean the conditioning
                # anchor is. Testing the paper's own working resolution (384x640, vs our usual
                # 256x448) instead, since that raises every pyramid stage's absolute resolution.
                kwargs["height"] = 384
                kwargs["width"] = 640
            success = run_inference(pipeline, engine_mode, case_dir, case_name=case_name, **kwargs)
            if not success:
                print(f"  Failed: {case_dir}{('/' + case_name) if case_name else ''}")
        except Exception as e:
            print(f"  Error ({case_dir}{('/' + case_name) if case_name else ''}): {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*70)
    print("Examples complete")
    print("="*70)
    print("\nOutput videos: outputs/*/geo_pred.mp4 (+ geo_pred_hud.mp4 where pose is available)\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
