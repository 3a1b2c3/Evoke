#!/bin/bash
# ============================================================================
# Post-distilled model -- batched inference; rolls out to minute scale
#   Weights : models/evoke/stage3_post_distillation
#   Engine  : scripts/inference/infer_batch.py -> scripts/inference/infer_single.py
#
#   What distillation bought, and what this script pins:
#     * 3 sampling steps  -- a 3-stage pyramid, one step per stage (STAGE2_STEPS="1 1 1")
#     * no CFG            -- guidance_scale 1.0, i.e. one forward per step, not two
#   Both are properties of the released weights, not tuning knobs. Raising the step count
#   does not improve this model; it was distilled to be sampled exactly this way.
#
#   Every knob below mirrors the training config "configs/training/stage3/post_distill.yaml".
#   Inference and training MUST agree on the warp / attention recipe: a mismatch
#   silently degrades image quality, it is not just a speed difference.
#
#   MODE=v2v (default) | i2v | t2v | segment
#     v2v      reference video + pose track  -> camera-controlled continuation (needs the shared dataset)
#     i2v      first frame + pose track      -> camera-controlled generation (runs off in-repo examples)
#     t2v      prompt only                   -> NO camera control; warp is off (the CLI rejects warp+t2v)
#     segment  i2v + a per-chunk prompt schedule -> the prompt changes partway through the rollout
#              (not a conditioning mode of its own; the engine still runs i2v)
#
#   Usage: [MODE=i2v] [LOCAL_GPUS=2] [MAX_CASES=8] [NUM_FRAMES=...] [OUT_ROOT=...] bash scripts/inference/infer_post_distill.sh
# ============================================================================
set -e
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"      # repo root
[ -n "${EVOKE_PYTHON_BIN:-}" ] && export PATH="$EVOKE_PYTHON_BIN:$PATH"

# One interpreter check for the whole run (~10s for the torch import). infer_batch.py spawns
#   sys.executable per case, so if this python is right, every case gets the same one.
if ! python -c 'import torch, cv2' 2>/dev/null; then
  echo "[preflight] FATAL: the python on PATH ($(command -v python || echo none)) cannot import torch + cv2." >&2
  echo "[preflight]        Set EVOKE_PYTHON_BIN=<env>/bin to prepend the right interpreter." >&2
  exit 1
fi

usage() {
  cat <<'EOF'
infer_post_distill.sh -- post-distilled model (models/evoke/stage3_post_distillation)

  MODE=v2v      reference video + pose track  -> camera-controlled continuation (needs shared dataset)
  MODE=i2v      first frame + pose track      -> camera-controlled generation (runs on bundled examples)
  MODE=t2v      prompt only                   -> no camera control (warp is off; the CLI forbids warp+t2v)
  MODE=segment  i2v + a prompt that switches mid-rollout (engine still runs i2v)

  NOTE: trained on v2v conditioning only -> i2v / t2v / segment are ZERO-SHOT here.

Examples
  MODE=i2v MAX_CASES=1 NUM_FRAMES=721  bash scripts/inference/infer_post_distill.sh      # ~30s clip, no external data needed
  MODE=segment NUM_CHUNKS=6            bash scripts/inference/infer_post_distill.sh      # bundled 3-segment demo, no external data needed
  MODE=v2v LOCAL_GPUS=4                bash scripts/inference/infer_post_distill.sh      # full v2v eval over the sekai jsonl
  TRANSFORMER_PATH=<other-ckpt>        bash scripts/inference/infer_post_distill.sh      # same recipe, different checkpoint

Full sweep (needs the shared dataset mounted)
  MODE=v2v JSONL=<your-sweep>.jsonl MAX_CASES=0 \
    VROOT=<video-root> AROOT=<annotation-root> bash <this script>

Knobs   LOCAL_GPUS (shard by case)  MAX_CASES (0=all)
        NUM_CHUNKS  exact chunk count (1 chunk = 36 frames = 1.5s @24fps) -- prefer this
        NUM_FRAMES  picks the chunk count only, it does NOT bound the output:
                    chunks = ceil(NUM_FRAMES/33), frames = 36*chunks - 3
                    721 -> 22 chunks -> 789 frames (32.9s);  2877 -> 88 -> 3165 (131.9s)
        v2v only: REF_VIDEO_SEC (default 5) = how much of the reference video conditions the
                  model, i.e. generation continues from that point. Forced to 0 for i2v/t2v.
        START_SECONDS (default 0) offsets where that window is taken from; it also shifts the
                  pose track, so it applies to i2v too (NOT forced to 0).
        HEIGHT WIDTH FPS  GUIDANCE_SCALE  SEED  OUT_ROOT  JSONL  TRANSFORMER_PATH
        EVOKE_PYTHON_BIN  prepended to PATH, for picking the interpreter
Logs    LOCAL_GPUS=1 streams the run to the terminal and to logs/ (chunk-level progress bar
        included). QUIET=1 redirects it to the log only; multi-shard runs always redirect.
Long    Rollouts above ~5min must stream: STREAM_LONG=1 decodes per chunk and stitches the final
        mp4 from segments/ (the driver refuses to accumulate more than 7200 frames on GPU).
        GEO_HIST_MAX_FRAMES bounds the DA3 point cloud for hour-scale runs -- note it changes the
        warp input, so it is a recipe change, not just a memory knob. See the README.
Output  $OUT_ROOT/<case>/geo_pred.mp4                      generated video (always)
        $OUT_ROOT/<case>/gt_vs_pred_cam_viz.mp4            4-panel gt|warp|vis|pred (v2v only)
        $OUT_ROOT/<case>/segments/segment_NNN_pred.mp4     per-chunk segments (persistent decode)
        $OUT_ROOT/_logs/<case>.log                         per-case engine log
EOF
}
[ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ] && { usage; exit 0; }

# The pyramid takes its step count from STAGE2_STEPS (pipeline_evoke.stage2_sample ->
#   set_timesteps per stage); --num_inference_steps is NOT consulted there. Warn rather than
#   let an override look effective.
if [ -n "${NUM_INFERENCE_STEPS:-}" ] && [ "${NUM_INFERENCE_STEPS}" != "3" ]; then
  echo "[post_distill] WARNING: NUM_INFERENCE_STEPS=$NUM_INFERENCE_STEPS is ignored under the pyramid;"
  echo "[post_distill]          the step count comes from STAGE2_STEPS. Override that instead."
fi

export MODE=${MODE:-v2v}
# SHARDS_PER_GPU>1 puts several worker processes on each device. Worth it because a case alternates
#   between a GPU phase (diffusion, ~100% util) and a CPU phase (VAE post, PNG deflate, mp4 encode, the
#   CPU half of the warp render) during which the device sits idle -- sampled across a live 8-shard
#   sweep, roughly half the wall time had every GPU at 0%. Two processes per device interleave one's CPU
#   phase with the other's GPU phase. The cost is VRAM: measured peak 50.4 GB per process, so 2 fit in
#   an H200's 139.8 GB and 3 do not. Shard s runs on device s % LOCAL_GPUS.
LOCAL_GPUS=${LOCAL_GPUS:-1}
SHARDS_PER_GPU=${SHARDS_PER_GPU:-1}
NSHARD=${NSHARD:-$((LOCAL_GPUS * SHARDS_PER_GPU))}

# `segment` is not a fourth conditioning mode -- it is i2v plus a per-chunk prompt schedule. It gets
#   its own name so the bundled demo, the output dir and the log dir do not collide with a plain i2v
#   run, but the engine is told i2v. TAG is what the paths use; MODE is what infer_batch.py sees.
TAG=$MODE
if [ "$MODE" = "segment" ]; then
  export JSONL=${JSONL:-"examples/segment_prompts/cases.jsonl"}
  export MODE=i2v
fi

# This model was trained with v2v conditioning only (geo_condition_{i2v,t2v}_ratio = 0.0 in
# configs/training/stage3/post_distill.yaml), so i2v / t2v below are ZERO-SHOT extrapolation, not a trained capability.
if [ "$TAG" != "v2v" ]; then
  echo "[post_distill] WARNING: MODE=$TAG is zero-shot for this model (trained on v2v conditioning only)."
fi

# -- per-mode input set --
#    Paths in a jsonl row are resolved under the dataset roots (VROOT / AROOT) first, then
#    repo-relative -- so the bundled examples/* sets run as-is, and the full sekai sweep
#    works by pointing VROOT / AROOT at a mounted dataset.
#      v2v: video_path / pose_path / prompt_path (a caption json)
#      i2v: image_path / pose_path / prompt_path (a .txt)
#      t2v: prompt only; no pose, no warp
case "$MODE" in
  v2v) export JSONL=${JSONL:-"examples/v2v/cases.jsonl"} ;;
  i2v) export JSONL=${JSONL:-"examples/i2v/cases.jsonl"} ;;
  t2v) export JSONL=${JSONL:-"examples/t2v/cases.jsonl"} ;;
esac

# Remember whether the checkpoint was chosen or fell back to the default: an unexported
#   TRANSFORMER_PATH silently evaluates the default model, which reads as "my override worked".
TP_ORIGIN="this is YOUR override (TRANSFORMER_PATH was set)"
[ -z "${TRANSFORMER_PATH:-}" ] && TP_ORIGIN="this is the LAUNCHER DEFAULT -- you did not set TRANSFORMER_PATH, so another ckpt you meant to test is NOT running"
export TRANSFORMER_PATH=${TRANSFORMER_PATH:-"models/evoke/stage3_post_distillation"}
export MAX_CASES=${MAX_CASES:-8}       # first N cases only; MAX_CASES=0 runs the whole jsonl
export NUM_FRAMES=${NUM_FRAMES:-2877}  # -> 88 chunks -> 3165 frames ~= 131.9s @24fps (see -h)
export HEIGHT=${HEIGHT:-384} WIDTH=${WIDTH:-640} FPS=${FPS:-24}
export GUIDANCE_SCALE=${GUIDANCE_SCALE:-1.0}   # CFG off: distillation removed the need for it
# persistent: this ckpt's per-chunk first latent is CONTINUOUS-frame distributed (verified on real
# latents), so continuous cross-chunk decode is flicker-free (0.98-1.13x |dframe| at every boundary).
# default/default_warm0 decode each chunk head as an I-frame -> ~4.5x flicker at every boundary here.
export VAE_DECODE_TYPE=${VAE_DECODE_TYPE:-persistent}

# -- mirrors longvideo_30s_distill.yaml (same warp recipe as few-step); this exact
#    combination has been run end-to-end at 120s and 60min --
export IS_STAGE2=1 STAGE2_NUM_STAGES=3 STAGE2_STEPS="1 1 1"
export NUM_INFERENCE_STEPS=3
export RESTRICT=0                      # restrict_self_attn=false and use_kv_cache=false
export GEO_WARP_STAGE0_ONLY=1
export WARP_SIGMA_MAX=0.135
export NOISE_CENTER=0
export RENDER_MODE=backward_zbuf BW_FILL_ITERS=12
# Depth estimator behind the cloud warp (= training cloud_warp.backend); orthogonal to RENDER_MODE.
export DEPTH_BACKEND=${DEPTH_BACKEND:-vigeo}   # DEPTH_BACKEND=da3 falls back to DepthAnything3
export ZBUF_DESPECKLE=0
export WARP_MODE=fixed_mem
# To run a different checkpoint, point TRANSFORMER_PATH at the parent of its transformer/ directory:
#   e.g. TRANSFORMER_PATH=models/evoke/stage3_long_distillation bash <this script>

export OUT_ROOT=${OUT_ROOT:-"output_evoke/infer/post_distill_$TAG"}
# An unexported variable inside OUT_ROOT (OUT_ROOT=output_evoke/infer/$TAG/v2v with TAG never set)
#   collapses to an empty path segment and quietly writes to a different directory than intended.
case "$OUT_ROOT" in
  *//*) echo "[post_distill] FATAL: OUT_ROOT='$OUT_ROOT' has an empty path segment -- a" >&2
        echo "[post_distill]        variable in it expanded to nothing (unexported \$TAG etc.)." >&2
        echo "[post_distill]        Export it, or drop it from OUT_ROOT." >&2
        exit 1 ;;
  */)   echo "[post_distill] WARNING: OUT_ROOT='$OUT_ROOT' ends in a slash; if that is a"
        echo "[post_distill]          variable that expanded to nothing, results land one level up."
        export OUT_ROOT="${OUT_ROOT%/}" ;;
esac
# NUM_CHUNKS is consumed by infer_batch.py (NUM_FRAMES = 33*NUM_CHUNKS), so echoing this script's own
#   NUM_FRAMES would report the default while a different length actually runs.
EFF_FRAMES=$NUM_FRAMES; EFF_NOTE=""
if [ -n "${NUM_CHUNKS:-}" ] && [ "${NUM_CHUNKS}" != "0" ]; then
  EFF_FRAMES=$((33 * NUM_CHUNKS)); EFF_NOTE=" (NUM_CHUNKS=$NUM_CHUNKS -> $((36 * NUM_CHUNKS - 3)) px frames)"
fi

SLOG="logs/infer_post_distill_$TAG"; mkdir -p "$SLOG"
echo "=============================================================================="
echo "[post_distill] CHECKPOINT : $TRANSFORMER_PATH"
echo "[post_distill]              ^-- $TP_ORIGIN"
if [ "$GUIDANCE_SCALE" = "1" ] || [ "$GUIDANCE_SCALE" = "1.0" ]; then
    CFG_STATE=off
else
    CFG_STATE=on
fi
MODE_NOTE=""
if [ "$TAG" != "$MODE" ]; then
    MODE_NOTE=" (engine sample_type=$MODE)"
fi
echo "[post_distill] mode=$TAG$MODE_NOTE steps=3 (pyramid 1+1+1, guidance_scale=$GUIDANCE_SCALE, CFG $CFG_STATE)"
echo "[post_distill] jsonl=$JSONL max_cases=$MAX_CASES frames=$EFF_FRAMES$EFF_NOTE shards=$NSHARD"
echo "[post_distill] out=$OUT_ROOT"
echo "=============================================================================="

# One shard is the interactive case: stream the worker to the terminal *and* the log, so a live
#   rollout is distinguishable from a hang. Several shards keep the redirect -- interleaving N
#   progress bars on one terminal is unreadable. QUIET=1 forces the redirect back on.
if [ "$LOCAL_GPUS" = "1" ] && [ "${QUIET:-0}" != "1" ]; then
  g=${GPU_OFFSET:-0}
  echo "[post_distill] shard 0 -> gpu $g -> terminal + $SLOG/shard_0.log"
  set +e
  CUDA_VISIBLE_DEVICES=$g SHARD=0 NSHARD=$NSHARD \
      python scripts/inference/infer_batch.py 2>&1 | tee "$SLOG/shard_0.log"
  fail=${PIPESTATUS[0]}   # `set -e` on a pipe would see tee's status, not the worker's
  set -e
else
  # SHARD_STAGGER_S: delay between launches. Building a pipeline streams the whole checkpoint through
  #   host RAM (54 GB transformer + 22 GB text encoder for the post_distill ckpt), so starting every
  #   shard at once multiplies that peak by NSHARD -- at NSHARD=16 on a 512 GB box the kernel SIGKILLs
  #   the workers mid-load, which shows up as a per-case log that stops after the "transformer:" line
  #   with no traceback. Staggering spreads the peaks; after loading, a worker's host footprint drops.
  #   0 disables it (fine for NSHARD <= LOCAL_GPUS, which is what the historical default did).
  # Default 0: the throttle that actually matters lives in infer_single._acquire_load_slot
  #   (EVOKE_MAX_CONCURRENT_LOADS), which gates the pipeline BUILD rather than the launch -- a shard
  #   whose cases are all already done then exits at once instead of sitting out a stagger it does not
  #   need. Kept as an override for the rare case where even process startup needs spreading out.
  SHARD_STAGGER_S=${SHARD_STAGGER_S:-0}
  # How many shards may stream a checkpoint through host RAM at the same time. 54 GB transformer +
  #   22 GB text encoder per build, so all 16 at once SIGKILLs the workers on a 512 GB box.
  export EVOKE_MAX_CONCURRENT_LOADS=${EVOKE_MAX_CONCURRENT_LOADS:-$LOCAL_GPUS}
  pids=()
  for s in $(seq 0 $((NSHARD-1))); do
      # GPU_OFFSET shifts the whole shard set onto a contiguous range starting at that GPU (default 0).
      #   CUDA_VISIBLE_DEVICES=$g below is an **absolute** index and overrides any outer setting, so
      #   freeing a card by exporting CUDA_VISIBLE_DEVICES has no effect -- shard g=0 still lands on
      #   physical GPU 0. To coexist with another job, LOCAL_GPUS=7 GPU_OFFSET=1 pins this run to GPU 1-7.
      g=$(( (s % LOCAL_GPUS) + ${GPU_OFFSET:-0} ))
      CUDA_VISIBLE_DEVICES=$g SHARD=$s NSHARD=$NSHARD \
          python scripts/inference/infer_batch.py > "$SLOG/shard_${s}.log" 2>&1 &
      pids+=($!); echo "[post_distill] shard $s -> gpu $g -> $SLOG/shard_${s}.log"
      [ "$s" -lt "$((NSHARD-1))" ] && [ "$SHARD_STAGGER_S" -gt 0 ] && sleep "$SHARD_STAGGER_S"
  done
  fail=0; for p in "${pids[@]}"; do wait "$p" || fail=$((fail+1)); done
fi
echo "[post_distill] DONE failed_shards=$fail -> $OUT_ROOT"; exit $fail
