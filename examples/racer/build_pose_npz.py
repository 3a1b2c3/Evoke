#!/usr/bin/env python3
"""Synthesize a vipe-format cam_c2w pose track from 0001.json's discrete move/view actions.

0001.json is a per-tick {"move": "go forward"|"no-op", "view": "turn left"|"turn right"|"no-op"}
sequence (racer RC-car recording) -- NOT a real vipe reconstruction. There is no ground-truth
camera trajectory for this clip, so this builds a plausible synthetic one by dead-reckoning:
each tick accumulates a fixed yaw step (turn) then a fixed forward step along the camera's new
local -Z (matches the axis convention observed in examples/i2v/pose.npz: forward motion shows up
as a growing negative local-Z translation, turns as a rotation about Y with R[0,2]=sin(yaw)).

Step sizes are tuned by eye to roughly match real vipe trajectory magnitudes (examples/i2v/pose.npz
advances ~0.001 units/frame under gentle motion); there's no way to recover true metric scale from
key presses alone, so treat this pose track as approximate camera *guidance*, not ground truth.
"""
import json
from pathlib import Path

import numpy as np

FORWARD_STEP = 0.02   # world units per "go forward" tick
YAW_STEP_DEG = 2.0    # degrees per "turn left/right" tick; +yaw = left (right-handed, Y-up)

racer_dir = Path(__file__).parent
actions = json.loads((racer_dir / "0001.json").read_text())

yaw = 0.0
pos = np.zeros(3, dtype=np.float64)
c2ws = np.zeros((len(actions), 4, 4), dtype=np.float32)

for i, a in enumerate(actions):
    if a["view"] == "turn left":
        yaw += np.deg2rad(YAW_STEP_DEG)
    elif a["view"] == "turn right":
        yaw -= np.deg2rad(YAW_STEP_DEG)

    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    R = np.array([
        [cos_y, 0.0, sin_y],
        [0.0, 1.0, 0.0],
        [-sin_y, 0.0, cos_y],
    ])

    if a["move"] == "go forward":
        pos = pos + R @ np.array([0.0, 0.0, -FORWARD_STEP])

    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = R
    c2w[:3, 3] = pos
    c2ws[i] = c2w

# Normalized default intrinsic (auto-rescaled to source_resolution by load_pose_for_v2v).
intrinsic = np.array([[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]], dtype=np.float32)

out_path = racer_dir / "pose.npz"
np.savez(out_path, cam_c2w=c2ws, intrinsics=intrinsic)
print(f"Wrote {out_path}: cam_c2w {c2ws.shape}, intrinsics {intrinsic.shape}")
