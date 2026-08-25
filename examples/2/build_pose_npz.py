#!/usr/bin/env python3
"""Synthesize a vipe-format cam_c2w pose track from 2.json's discrete WASD action sequence.

2.json is a {"action_seq": ["w",...], "action_speed_list": [2,...], "motion_speed": 0.075} record
(DreamX-World format) with only 4 entries -- one coarse action per generation CHUNK, not per
frame (matches the README's "one chunk = 36 frames = 1.5s"; events.json's fractional "at" times
over the whole clip also imply a multi-chunk rollout, not a 4-frame one). This is NOT a real vipe
reconstruction: each action tick is held for CHUNK_FRAMES frames and dead-reckoned into a plausible
synthetic trajectory, same convention as examples/racer/build_pose_npz.py: w/s translate along the
camera's local -Z/+Z (forward/back), a/d rotate about Y (turn left/right). Per-frame step size is
motion_speed * action_speed_list[i], scaled by FORWARD_STEP/YAW_STEP_DEG to land in the same rough
magnitude range as real vipe trajectories (examples/i2v/pose.npz).

There's no way to recover true metric scale or the exact turn-vs-strafe semantics of a/d from key
presses alone -- treat this pose track as approximate camera *guidance*, not ground truth.
"""
import json
from pathlib import Path

import numpy as np

FORWARD_STEP = 0.01   # world units per frame per unit of (motion_speed * action_speed)
YAW_STEP_DEG = 1.0     # degrees per frame per unit of (motion_speed * action_speed)
CHUNK_FRAMES = 36      # frames held per action_seq entry (one generation chunk, per README)

case_dir = Path(__file__).parent
record = json.loads((case_dir / "2.json").read_text())[0]
action_seq = record["action_seq"]
action_speed_list = record["action_speed_list"]
motion_speed = record["motion_speed"]

ticks = [(key, speed) for key, speed in zip(action_seq, action_speed_list) for _ in range(CHUNK_FRAMES)]

yaw = 0.0
pos = np.zeros(3, dtype=np.float64)
c2ws = np.zeros((len(ticks), 4, 4), dtype=np.float32)

for i, (key, speed) in enumerate(ticks):
    step = motion_speed * speed

    if key == "a":
        yaw += np.deg2rad(YAW_STEP_DEG * step)
    elif key == "d":
        yaw -= np.deg2rad(YAW_STEP_DEG * step)

    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    R = np.array([
        [cos_y, 0.0, sin_y],
        [0.0, 1.0, 0.0],
        [-sin_y, 0.0, cos_y],
    ])

    if key == "w":
        pos = pos + R @ np.array([0.0, 0.0, -FORWARD_STEP * step])
    elif key == "s":
        pos = pos + R @ np.array([0.0, 0.0, FORWARD_STEP * step])

    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = R
    c2w[:3, 3] = pos
    c2ws[i] = c2w

# Normalized default intrinsic (auto-rescaled to source_resolution by load_pose_for_v2v).
intrinsic = np.array([[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]], dtype=np.float32)

out_path = case_dir / "pose.npz"
np.savez(out_path, cam_c2w=c2ws, intrinsics=intrinsic)
print(f"Wrote {out_path}: cam_c2w {c2ws.shape}, intrinsics {intrinsic.shape}")
