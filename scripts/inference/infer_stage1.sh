#!/bin/bash
# ============================================================================
# Stage-1 (multi-step geometric state) batched inference
#   Weights : models/evoke/stage1_camera_control  (holds transformer/; loaded with subfolder="transformer")
#   Engine  : scripts/inference/infer_batch.py -> scripts/inference/infer_single.py
#
#   Every knob below mirrors the training config "configs/training/stage1/stage1.yaml".
#   Inference and training MUST agree on the warp / attention recipe: a mismatch
#   silently degrades image quality, it is not just a speed difference.
#
#   MODE=v2v (default) | i2v | t2v
#     v2v  reference video + pose track  -> camera-controlled continuation (needs the shared dataset)
#     i2v  first frame + pose track      -> camera-controlled generation (runs off in-repo examples)
#     t2v  prompt only                   -> NO camera control; warp is off (the CLI rejects warp+t2v)
#
#   Usage: [MODE=i2v] [LOCAL_GPUS=2] [MAX_CASES=8] [NUM_FRAMES=...] [OUT_ROOT=...] bash scripts/inference/infer_stage1.sh
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
infer_stage1.sh -- Stage-1 multi-step GEO model (models/evoke/stage1_camera_control)

  MODE=v2v   reference video + pose track  -> camera-controlled continuation (needs shared dataset)
  MODE=i2v   first frame + pose track      -> camera-controlled generation (runs on bundled examples)
  MODE=t2v   prompt only                   -> no camera control (warp is off; the CLI forbids warp+t2v)

  All three modes are in-distribution for this model (t2v_ratio=0.1, i2v_ratio=0.2).

Examples
  MODE=i2v MAX_CASES=1 NUM_FRAMES=721  bash scripts/inference/infer_stage1.sh      # ~30s clip, no external data needed
  MODE=v2v LOCAL_GPUS=4                bash scripts/inference/infer_stage1.sh      # full v2v eval over the sekai jsonl
  TRANSFORMER_PATH=<other-ckpt>        bash scripts/inference/infer_stage1.sh      # same recipe, different checkpoint

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

export MODE=${MODE:-v2v}
LOCAL_GPUS=${LOCAL_GPUS:-1}; NSHARD=${NSHARD:-$LOCAL_GPUS}

# All three modes are in-distribution here (geo_condition_t2v_ratio=0.1, i2v_ratio=0.2).

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

# TRANSFORMER_PATH is the PARENT of the transformer/ dir: it is loaded with
#   from_pretrained(TRANSFORMER_PATH, subfolder="transformer") -- see infer_single.py.
# Remember whether the checkpoint was chosen or fell back to the default: an unexported
#   TRANSFORMER_PATH silently evaluates the default model, which reads as "my override worked".
TP_ORIGIN="this is YOUR override (TRANSFORMER_PATH was set)"
[ -z "${TRANSFORMER_PATH:-}" ] && TP_ORIGIN="this is the LAUNCHER DEFAULT -- you did not set TRANSFORMER_PATH, so another ckpt you meant to test is NOT running"
export TRANSFORMER_PATH=${TRANSFORMER_PATH:-"models/evoke/stage1_camera_control"}
export MAX_CASES=${MAX_CASES:-8}       # first N cases only; MAX_CASES=0 runs the whole jsonl
export NUM_FRAMES=${NUM_FRAMES:-1437}  # -> 44 chunks -> 1581 frames ~= 65.9s @24fps (see -h)
export HEIGHT=${HEIGHT:-384} WIDTH=${WIDTH:-640} FPS=${FPS:-24}
export GUIDANCE_SCALE=${GUIDANCE_SCALE:-1.0}
export VAE_DECODE_TYPE=${VAE_DECODE_TYPE:-persistent}

# -- mirrors stage1.yaml: no pyramid / 8 steps / restrict+kv-cache / despeckle ON --
export IS_STAGE2=0                     # stage1 has no NaViT pyramid
export NUM_INFERENCE_STEPS=${NUM_INFERENCE_STEPS:-8}   # = validation_config.num_inference_steps
export RESTRICT=1                      # training_config.restrict_self_attn=true -> required at inference
                                       #   (kv-cache is then free speedup); running without it is a
                                       #   train/infer attention mismatch, not just slower
export GEO_WARP_STAGE0_ONLY=0          # warp_stage0_only unset in config
export WARP_SIGMA_MAX=0.135            # = geometric_state.warp_noise_sigma_max
export NOISE_CENTER=0                  # warp_rope_noise_center_align unset in config
export RENDER_MODE=backward_zbuf BW_FILL_ITERS=3   # cloud_warp.render_mode / bw_fill_iters
# Depth estimator behind the cloud warp (= training cloud_warp.backend); orthogonal to RENDER_MODE.
export DEPTH_BACKEND=${DEPTH_BACKEND:-vigeo}   # DEPTH_BACKEND=da3 falls back to DepthAnything3
export ZBUF_DESPECKLE=1                # cloud_warp.zbuf_despeckle=true

export OUT_ROOT=${OUT_ROOT:-"output_evoke/infer/stage1_$MODE"}
# An unexported variable inside OUT_ROOT (OUT_ROOT=output_evoke/infer/$TAG/v2v with TAG never set)
#   collapses to an empty path segment and quietly writes to a different directory than intended.
case "$OUT_ROOT" in
  *//*) echo "[stage1] FATAL: OUT_ROOT='$OUT_ROOT' has an empty path segment -- a" >&2
        echo "[stage1]        variable in it expanded to nothing (unexported \$TAG etc.)." >&2
        echo "[stage1]        Export it, or drop it from OUT_ROOT." >&2
        exit 1 ;;
  */)   echo "[stage1] WARNING: OUT_ROOT='$OUT_ROOT' ends in a slash; if that is a"
        echo "[stage1]          variable that expanded to nothing, results land one level up."
        export OUT_ROOT="${OUT_ROOT%/}" ;;
esac
# NUM_CHUNKS is consumed by infer_batch.py (NUM_FRAMES = 33*NUM_CHUNKS), so echoing this script's own
#   NUM_FRAMES would report the default while a different length actually runs.
EFF_FRAMES=$NUM_FRAMES; EFF_NOTE=""
if [ -n "${NUM_CHUNKS:-}" ] && [ "${NUM_CHUNKS}" != "0" ]; then
  EFF_FRAMES=$((33 * NUM_CHUNKS)); EFF_NOTE=" (NUM_CHUNKS=$NUM_CHUNKS -> $((36 * NUM_CHUNKS - 3)) px frames)"
fi

SLOG="logs/infer_stage1_$MODE"; mkdir -p "$SLOG"
echo "=============================================================================="
echo "[stage1] CHECKPOINT : $TRANSFORMER_PATH"
echo "[stage1]              ^-- $TP_ORIGIN"
echo "[stage1] mode=$MODE steps=8 (multi-step, no pyramid)"
echo "[stage1] jsonl=$JSONL max_cases=$MAX_CASES frames=$EFF_FRAMES$EFF_NOTE shards=$NSHARD"
echo "[stage1] out=$OUT_ROOT"
echo "=============================================================================="

# One shard is the interactive case: stream the worker to the terminal *and* the log, so a live
#   rollout is distinguishable from a hang. Several shards keep the redirect -- interleaving N
#   progress bars on one terminal is unreadable. QUIET=1 forces the redirect back on.
if [ "$LOCAL_GPUS" = "1" ] && [ "${QUIET:-0}" != "1" ]; then
  echo "[stage1] shard 0 -> terminal + $SLOG/shard_0.log"
  set +e
  CUDA_VISIBLE_DEVICES=0 SHARD=0 NSHARD=$NSHARD \
      python scripts/inference/infer_batch.py 2>&1 | tee "$SLOG/shard_0.log"
  fail=${PIPESTATUS[0]}   # `set -e` on a pipe would see tee's status, not the worker's
  set -e
else
  pids=()
  for g in $(seq 0 $((LOCAL_GPUS-1))); do
      CUDA_VISIBLE_DEVICES=$g SHARD=$g NSHARD=$NSHARD \
          python scripts/inference/infer_batch.py > "$SLOG/shard_${g}.log" 2>&1 &
      pids+=($!); echo "[stage1] shard $g -> $SLOG/shard_${g}.log"
  done
  fail=0; for p in "${pids[@]}"; do wait "$p" || fail=$((fail+1)); done
fi
echo "[stage1] DONE failed_shards=$fail -> $OUT_ROOT"; exit $fail
