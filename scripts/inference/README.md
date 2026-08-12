# Inference

The headline model is the 30s long-video distilled one: **3 sampling steps, no CFG**, rolls out to
minute scale. All four commands below run on data bundled in `examples/` — no external dataset.

```bash
# t2v -- prompt only, no camera control
MODE=t2v NUM_CHUNKS=20 bash scripts/inference/infer_post_distill.sh

# i2v -- first frame + camera trajectory
MODE=i2v NUM_CHUNKS=20 bash scripts/inference/infer_post_distill.sh

# v2v -- reference video + its camera trajectory
MODE=v2v NUM_CHUNKS=20 bash scripts/inference/infer_post_distill.sh

# segment -- i2v plus a per-chunk prompt schedule; the prompt switches mid-rollout
MODE=segment NUM_CHUNKS=6 MAX_CASES=0 bash scripts/inference/infer_post_distill.sh
```

Each mode already points at its own bundled jsonl, so `JSONL=` is only needed for your own data.

Result lands in `<OUT_ROOT>/<case>/geo_pred.mp4`; `OUT_ROOT` defaults to
`output_evoke/infer/post_distill_<mode>` here and `output_evoke/infer/stage1_<mode>` for stage 1.

If the launcher stops at `[preflight] FATAL: ... cannot import torch + cv2`, point it at the right
interpreter: `EVOKE_PYTHON_BIN=<your-env>/bin bash scripts/inference/...`.

Each launcher opens with the checkpoint it resolved and whether that came from `TRANSFORMER_PATH` or
from its own default — check that line first, an unexported override silently evaluates the default
model. The settings behind a result are also recorded in `<OUT_ROOT>/run_info.json`.

## Watching a run

A single-GPU run (`LOCAL_GPUS=1`, the default) streams to the terminal *and* to `logs/`. The rollout
reports two levels: a bar over chunks, and one line per chunk splitting it into its phases.

```
  chunk   3/20   warp   1.17s (pool=  72 cov=0.877)  |  diffusion   2.05s (3 steps)  |  decode+dump   5.78s  |  total   9.01s
chunks:  15%|█▌        | 3/20 [00:27<02:33,  9.19s/chunk]
```

`warp` is the point-cloud render (ViGeo depth by default) plus its encode to latents, `diffusion` the denoising steps
themselves, and `decode+dump` the VAE decode and the per-chunk segment videos — on the bundled v2v
case that last part costs more than the generation it follows. `cov` is warp coverage and `pool` the
number of source frames fused, the two numbers worth watching over a long rollout.

`QUIET=1` sends the run to the log only. Multi-shard runs always redirect and draw no bar: nothing
reaches a terminal, so the engine's plain per-chunk lines are the better record.

The log file gets the same lines with the bar's redraws stripped, so it stays greppable — progress
bars are terminal UI and a file full of `\r` frames is not readable. `EVOKE_INFER_DEBUG=1` restores
everything the bar replaces: the per-step DiT timings, the per-chunk sampling bar, the raw
`[GEO-da3]` / segment-dump lines (the tag is historical; the backend is whatever
`DEPTH_BACKEND` selects), and the vendored depth logger.

## Length: use NUM_CHUNKS

One chunk = 36 pixel frames = 1.5s @24fps. `NUM_CHUNKS=20` gives a 30s clip.

`NUM_FRAMES` is the older knob and it is **not** an output length — it only picks the chunk count,
and nothing trims the result back down:

```
chunks = ceil(NUM_FRAMES / 33)        # 33 = (latent_window_size-1) * vae_temporal_stride + 1
frames = 36 * chunks - 3              # i2v / v2v: the final length is quantised down to 4k+1
frames = 36 * chunks - 4              # t2v: same, minus the dropped frame 0 (see below)
```

So `NUM_FRAMES=721` yields 22 chunks = 789 frames = 32.9s, not 30s. `NUM_CHUNKS` sets
`NUM_FRAMES = 33 * chunks`, which is the only way to get an exact length. (The formula above is for
the shipped `VAE_DECODE_TYPE=persistent`; plain `default` decode yields `33 * chunks`.)

The `-3` is not a per-chunk trim: every chunk decodes 36 pixel frames (chunk 0 decodes 33 under t2v,
where the cache is not warmed), and the accumulated clip is then cut back to the nearest `4k+1`.
`segments/segment_NNN_pred.mp4` holds each chunk as decoded, so the segments can total more frames
than `geo_pred.mp4` — stitching them is not the same as the final video.

t2v drops frame 0 from the final `geo_pred.mp4`. With no real prior frame to warm the causal VAE
cache, chunk 0's first pixel is the I-frame slot and carries a colour cast that the rest of the clip
does not; the frame is still present in `segments/segment_000_pred.mp4`.

## Modes

| MODE | Input | Camera control |
|---|---|---|
| `v2v` | reference video + pose track | yes |
| `i2v` | first frame + pose track | yes |
| `t2v` | prompt only | no — warp carries the trajectory, and the engine forbids warp+t2v |

For v2v, generation continues **after** the reference window: `REF_VIDEO_SEC` (default 5s) is how
much of the clip conditions the model, so the default run conditions on `[0s, 5s]` and generates
from there. It is forced to 0 for i2v/t2v.

`START_SECONDS` (default 0) offsets where that window is taken from. It also shifts the **pose
track**, so unlike `REF_VIDEO_SEC` it applies to i2v as well and is never forced to 0.

### i2v camera scale

Two i2v-only knobs, both about how far the camera moves. `v2v`/`t2v` are unaffected: v2v seeds its
frame bank from the reference video and solves its own depth scale from real camera motion, t2v has no
warp. Calibration tables live next to the values in the code.

| env | default | what it is |
|---|---|---|
| `GEO_CHUNK0_TARGET_DISP_PX` | `135` px | chunk 0's depth scale comes from a *single* frame, and pinning its median depth does not pin the near field — so its amplitude used to swing with the reference photo, running ~6× the flow of its own later chunks. This rescales chunk-0 depth so that at its largest commanded step, 90 % of pixels stay inside an N-pixel budget. `0` restores the old behaviour. |
| `VIGEO_DEPTH_MEDIAN_TARGET` | `5` | how deep the world is, in pose units: after scaling, the scene's median depth *is* this number. At `1.0` (the bare unit definition, and the default until it was calibrated) a 5 s sekai track commands ~1.8 scene-depths per chunk, so the camera leaves the geometry, the warp collapses to holes and the model re-invents the scene — which reads as a sudden acceleration even though the commanded motion is flat or falling. v2v solves ≈9.9 for the same quantity. |

Both defaults are calibrated, so i2v needs no flags. 5 vs 10 for the world scale is a look call: 10 is
the value v2v measures and is safer on worst-case warp coverage, 5 keeps twice the parallax. `1.0`
reproduces pre-calibration runs.

Diagnosing it: read per-chunk `cov=` in `_logs/<case>.log` — below ~0.35 the model is filling holes from
the prompt. Optical flow will not show it, because a mid-clip reset leaves the tail/body flow ratio near
1.0. Resolved values are logged as `chunk0_target_disparity_px = ...` and `chunk0 disparity rescale: k=`.

## Models

| Launcher | Weights | Steps |
|---|---|---|
| `infer_stage1.sh` | `models/evoke/stage1_camera_control` | 50 (CFG 5.0) |
| `infer_post_distill.sh` | `models/evoke/stage3_post_distillation` | 3 |

Any other checkpoint runs on the same launcher and recipe — pass
`TRANSFORMER_PATH=models/evoke/stage3_long_distillation` (or any other parent of a `transformer/`
directory) and give it its own `OUT_ROOT`.

`TRANSFORMER_PATH` is always the **parent** of a `transformer/` directory — the weights load as
`from_pretrained(TRANSFORMER_PATH, subfolder="transformer")`.

### Sampling from the teacher

`infer_evoke_teacher.py` runs the dual-expert teacher in `models/evoke/evoke_teacher` directly. The
teacher is normally only a frozen scorer inside stage-3 DMD, but that scoring forward is a v-prediction,
so wrapping it in a flow-match Euler loop is enough to sample from it.

`infer_evoke_teacher.sh` is the worked i2v example — it runs on the bundled `examples/i2v/` data with
no arguments, ~5s @24fps:

```bash
bash scripts/inference/infer_evoke_teacher.sh                    # i2v, 121 frames, 50 steps
CLIP_SECONDS=10 STEPS=30 bash scripts/inference/infer_evoke_teacher.sh
IMAGE= PROMPT="a drone shot over a snowy mountain village" \
  bash scripts/inference/infer_evoke_teacher.sh                  # t2v instead
```

`CLIP_SECONDS` is rounded up to the nearest valid length: the VAE is temporal stride 4, so pixel
frames must be `4k+1` (5s @24fps → 121). The engine underneath takes the same knobs as flags:

```bash
python scripts/inference/infer_evoke_teacher.py \
  --prompt "a drone shot flying over a snowy mountain village at sunrise" \
  --image_path examples/i2v/image.jpg \
  --num_frames 121 --num_inference_steps 50 --offload \
  --output output/evoke_teacher/i2v.mp4
```

Result lands in `output/evoke_teacher/i2v.mp4`; the engine log is `logs/infer_evoke_teacher/run.log`.

It is an **example**, not a supported launcher, and differs from the two above in three ways worth
knowing before you read anything into its output:

- nocam, non-SP, single process — `EvokeTeacherScoreWrapper._forward_core` only ports that one path of
  the teacher's forward, so there is no camera conditioning and no sequence-parallel inference;
- not validated against the teacher's own sampler — the schedule, shift, v-prediction sign and expert
  boundary come from the training path, where they were checked for a *single* scoring step. A sampling
  loop compounds any mismatch, so A/B one prompt+seed against the upstream sampler before trusting it;
- 2 x 14B is ~56 GB of bf16 weights — pass `--offload` to keep only the routed expert resident, or
  `--single_expert high|low` to check the plumbing on one expert (which is wrong for half the schedule).

Both distilled models were trained on v2v conditioning only, so `MODE=i2v|t2v` on them is zero-shot;
the launchers say so at startup. All three modes are in distribution for stage1.

3 steps and CFG-off are properties of the distilled weights, not knobs: the step count comes from
`STAGE2_STEPS`, and raising `NUM_INFERENCE_STEPS` does nothing (the launcher warns).

`stage1` is the opposite case — it is **not** distilled, so it is sampled like an ordinary diffusion
model: 50 steps with CFG 5.0. Do not copy `validation_config.num_inference_steps` (=8) out of
`stage1_camera_control.yaml`; that is the cheap sanity check run during training, and 8 steps with CFG off visibly
smears the result.

## Hour-scale rollouts

Without streaming, the pipeline accumulates the whole clip as one fp32 GPU tensor (~2.95 MB/frame at
384x640), so a long run OOMs deep into the job. The driver therefore **refuses** more than 7200
frames unless you opt in:

- `STREAM_LONG=1` — decode per chunk, never accumulate, stitch the final `geo_pred.mp4` from
  `segments/`. Needs `VAE_DECODE_TYPE=persistent` (the launcher default). The generated pixels are
  identical, but the *side outputs* differ: the 4-panel video becomes a concatenation of the
  per-chunk panels rather than the whole-video comparison, and `sample_frames/` holds segment head
  frames instead of 5 evenly spaced ones.
- `GEO_HIST_MAX_FRAMES=<N>` — slide the point cloud so it keeps only the last N pixel frames.
  Required beyond ~10min (the cloud otherwise grows by 12 dense depth frames per chunk), but note
  this **changes what the warp sees**, i.e. it is a recipe change and not just a memory knob. Must
  be much larger than `WARP_LAG * 36`; 720 (~30s) is a typical value.

## Bring your own data

Copy the layout of **`examples/`** — it has one working case per mode, and every launcher points at
it by default:

```
examples/t2v/cases.jsonl              {"name", "prompt"}
examples/i2v/cases.jsonl              + image_path, pose_path, prompt_path
examples/v2v/cases.jsonl              + video_path, pose_path, prompt_path, video_fps
examples/segment_prompts/cases.jsonl  + schedule_*.json (prompt switches mid-rollout)
```

Point `JSONL` at your own file with the same fields. Paths inside are resolved relative to the repo
root. Poses are vipe `.npz` (`cam_c2w [T,4,4]` + `intrinsics`); declare `pose_fps` and
`pose_source_resolution` per row when they differ from the i2v defaults (24 / `[480, 832]`) — a 30 fps
track declared as 24 renders 25 % long and 25 % slow, and nothing detects it.

## Output

```
<OUT_ROOT>/<case>/geo_pred.mp4                  generated video (always; also the resume marker)
<OUT_ROOT>/<case>/geo_pred_hud.mp4              same video with the joystick overlay (JOYSTICK_HUD=both)
<OUT_ROOT>/<case>/segments/segment_NNN_pred.mp4 per-chunk segments (SAVE_SEGMENTS=0 to skip)
<OUT_ROOT>/<case>/gt_vs_pred_cam_viz.mp4        4-panel gt | warp | visibility | pred (v2v only)
<OUT_ROOT>/_logs/<case>.log                     engine log
<OUT_ROOT>/run_info.json                        checkpoint + recipe that produced this directory
```

Finished cases are skipped on re-run, so an interrupted sweep resumes. Knobs: `LOCAL_GPUS` ·
`GPU_OFFSET` · `MAX_CASES` (0 = all) · `TRANSFORMER_PATH` · `OUT_ROOT` · `SEED` · `NUM_CHUNKS` ·
`REF_VIDEO_SEC` / `START_SECONDS` · `JOYSTICK_HUD` · `EVOKE_PYTHON_BIN` — or run any launcher with
`-h`. For sweeps of hundreds of cases, `IN_PROCESS_BATCH=1 BG_POSTPROC=1 SHARDS_PER_GPU=1` is the tuned
recipe (~3.4 s/case against ~31 s/case out of the box).

Launcher knobs mirror `configs/training/**` of the same model. A train/infer mismatch in the warp or
attention recipe degrades quality silently instead of failing, so do not tune them ad hoc.
