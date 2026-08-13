"""
YAML multi-dataset configuration system. Parses a config file and returns a ConcatDataset.

ratio semantics: >=1 repeat multiplier, <1 random subsample fraction, 1 exact one pass.
Epoch size = sum(size_i * ratio_i), computed automatically.

A `ratio` below 1 is subsampled once at construction by default, which locks that source to a
fixed subset for the whole run -- 80k raw clips does not mean 80k reachable clips. With
resample_each_epoch=True the subset is re-drawn every epoch: same size, so the epoch length is
unchanged, but different videos, so distinct coverage grows with the epoch count. The seed is
f(base_seed, epoch, part_id) and excludes rank, because sharing a clip within an SP group relies
on "same index -> same video". It also makes epoch 0 deterministic, which construction-time
sampling was not. Requires persistent_workers=false so workers re-fork and see the new indices.
"""
import gc
import math
import os
import random
import yaml
import torch
from .unified_dataset import UnifiedDataset
from .caption_operator import ResolveCaptionFile
from .operators import ToAbsolutePath, LoadVideo, ImageCropAndResize, RouteByType, RouteByExtensionName, LoadGIF, LoadImage, ToList


class ConfigAwareDataset(torch.utils.data.Dataset):
    """Wraps UnifiedDataset with per-dataset config, caption dict unpacking, and clip-start alignment."""

    # Force a cyclic GC every N __getitem__ calls. Large per-sample objects (717-frame video
    # tensors / PIL frame lists) land in reference cycles that refcounting can't reclaim, so RSS
    # creeps across steps → dataloader-worker OOM (SIGKILL) → NCCL heartbeat timeout → SIGABRT.
    # This only bit short-clip-heavy real data (many require_full_length retries whose exception
    # tracebacks add cycles); sekai_game had zero retries so the natural GC cadence sufficed.
    # Verified locally: a gc.collect() per sample keeps RSS flat. N=2 bounds RSS at negligible cost.
    _GC_EVERY = 2

    def __init__(self, inner, ds_meta, target_fps=24):
        self.inner = inner
        self.ds_meta = ds_meta
        self.target_fps = target_fps
        self._gc_counter = 0

    def __getitem__(self, idx):
        # Retry with a random index on transient load errors (corrupt video, missing file, etc.)
        MAX_RETRIES = 20
        last_err = None
        tried_idx = idx
        try:
            for attempt in range(MAX_RETRIES):
                try:
                    data = self._load_and_process(tried_idx)
                    if attempt > 0:
                        print(f"[BadSample recovered] orig_idx={idx} → final_idx={tried_idx} after {attempt} retries", flush=True)
                    return data
                except (OSError, IOError, RuntimeError, ValueError, AttributeError, KeyError,
                        IndexError, StopIteration) as e:
                    # Keep only the message, NOT the exception object — retaining `e` keeps its
                    # __traceback__ → frames → big locals (video tensors / PIL frames / readers)
                    # alive in reference cycles that refcounting can't free, leaking RSS across the
                    # many retries on short-clip-heavy data.
                    last_err = f"{type(e).__name__}: {str(e)[:200]}"
                    if attempt == 0:
                        # Log the first failure to help locate bad samples
                        print(f"[BadSample] idx={tried_idx}: {last_err} → retrying with random idx", flush=True)
                    tried_idx = random.randrange(len(self.inner))
            # All retries exhausted — propagate the error
            raise RuntimeError(
                f"ConfigAwareDataset.__getitem__: gave up after {MAX_RETRIES} retries starting from idx={idx}. "
                f"Last error: {last_err}"
            )
        finally:
            # Reclaim cycle-trapped per-sample memory periodically (see _GC_EVERY note above).
            self._gc_counter += 1
            if self._gc_counter >= self._GC_EVERY:
                self._gc_counter = 0
                gc.collect()

    def _load_and_process(self, idx):
        data = self.inner[idx]
        data["__ds_config__"] = self.ds_meta

        # Extract clip start time from either the new __start_time__ key or a legacy tuple field
        clip_start_time = data.get("__start_time__", 0) or 0
        if isinstance(data.get("video"), tuple):
            if len(data["video"]) == 3:
                data["video"], clip_start_time, data["__frame_indices__"] = data["video"]
            elif len(data["video"]) == 2:
                data["video"], clip_start_time = data["video"]
        # Keep __start_time__ so downstream memory_rollout tasks can compute mem_start_lat
        data["__start_time__"] = clip_start_time

        # Unpack caption dict into prompt string and optional segment_prompts
        if isinstance(data.get("prompt"), dict):
            cap = data["prompt"]
            data["prompt"] = cap["text"]
            if "segment_prompts" in cap:
                # Align segment_prompts to the randomly cropped clip range
                data["segment_prompts"] = self._align_segment_prompts(
                    cap["segment_prompts"], clip_start_time, len(data.get("video", []))
                )
        return data

    def _align_segment_prompts(self, segment_prompts, clip_start_time, num_frames):
        """Align segment_prompts frame ranges to the cropped clip window."""
        if not segment_prompts or clip_start_time == 0:
            return segment_prompts

        clip_start_frame = int(clip_start_time * self.target_fps)
        clip_end_frame = clip_start_frame + num_frames
        aligned = []
        for seg in segment_prompts:
            seg_start = seg["start_frame"]
            seg_end = seg["end_frame"]
            # Skip segments outside the clip window
            if seg_end <= clip_start_frame or seg_start >= clip_end_frame:
                continue
            # Clamp and shift to [0, num_frames)
            aligned.append({
                "start_frame": max(0, seg_start - clip_start_frame),
                "end_frame": min(num_frames, seg_end - clip_start_frame),
                "prompt": seg["prompt"],
            })
        return aligned if aligned else None

    def __len__(self):
        return len(self.inner)


class SubsampledDataset(torch.utils.data.Dataset):
    """Randomly samples subset_size items from inner dataset (used when ratio < 1).

    With resample_each_epoch=True, set_epoch(e) swaps in a different subset of the same size.
    part_id distinguishes the several SubsampledDatasets of one epoch, so two parts over the same
    inner dataset do not draw identical subsets.
    """

    def __init__(self, inner, subset_size, part_id: int = 0,
                 resample_each_epoch: bool = False, base_seed: int = 0):
        self.inner = inner
        self.subset_size = int(min(subset_size, len(inner)))
        self.part_id = int(part_id)
        self.resample_each_epoch = bool(resample_each_epoch)
        self.base_seed = int(base_seed)
        self._epoch = None
        if self.resample_each_epoch:
            # Sample deterministically for epoch 0 as well, so (seed, epoch) -> subset is reproducible
            # instead of depending on how much global random state was consumed before construction.
            self.set_epoch(0)
        else:
            self.indices = random.sample(range(len(inner)), self.subset_size)   # legacy behaviour, kept verbatim

    def set_epoch(self, epoch: int):
        """Re-draw this epoch's subset; a no-op unless resample_each_epoch=True.

        The seed depends only on (base_seed, epoch, part_id) -- not on rank -- and does not touch
        the global random state, so every rank draws the same subset and no other random stream
        (rollout, crop offsets, ...) is perturbed.
        """
        if not self.resample_each_epoch:
            return
        e = int(epoch)
        if e == self._epoch:
            return
        rng = random.Random((self.base_seed & 0x7FFFFFFF) * 1000003
                            + e * 1000033 + self.part_id * 7919 + 20260729)
        self.indices = rng.sample(range(len(self.inner)), self.subset_size)
        self._epoch = e

    def __getitem__(self, idx):
        return self.inner[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


def set_dataset_epoch(dataset, epoch: int) -> int:
    """Push `epoch` to every SubsampledDataset in a ConcatDataset; returns how many responded.

    Matched by isinstance, so nothing else is assumed to have set_epoch. The integer part of a
    ratio is a repeated reference to the same inner dataset and has no subset to swap, so it is
    skipped. With resample_each_epoch=False every set_epoch is a no-op and this returns 0.
    """
    n = 0
    for part in (getattr(dataset, "datasets", None) or []):
        if isinstance(part, SubsampledDataset):
            part.set_epoch(epoch)
            n += 1
    return n


def load_data_config(config_path, max_pixels=1024 * 1024, height=None, width=None,
                     num_frames=None, target_fps=None,
                     resample_each_epoch: bool = False, seed: int = 0):
    """
    Parse a YAML data config and return a ConcatDataset.

    Args:
        config_path: path to the YAML config file
        max_pixels: max pixels per frame
        height, width: fixed resolution (None = dynamic)
        num_frames: clip length; falls back to data yaml defaults then 193
        target_fps: clip fps; falls back to data yaml defaults then 24
        resample_each_epoch: True re-draws the random subsample of each ratio every epoch, at the
            same size; False draws once at construction and keeps it for the whole run
        seed: base seed for the re-draw, rank-independent (usually args.seed)

    Returns:
        ConcatDataset
    """
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    all_datasets = cfg["datasets"]
    defaults = cfg.get("defaults", {})   # legacy yaml compat; newer yamls omit this field
    select = cfg["select"]
    ratio = cfg["ratio"]

    # Priority: caller arg > data yaml defaults > hardcoded fallback
    global_num_frames = num_frames if num_frames is not None else defaults.get("num_frames", 193)
    global_target_fps = target_fps if target_fps is not None else defaults.get("target_fps", 24)

    assert len(select) == len(ratio), f"select ({len(select)}) and ratio ({len(ratio)}) must have same length"

    sub_datasets = []
    raw_sizes = []

    for name in select:
        assert name in all_datasets, f"Dataset '{name}' not found in datasets config"
        # Merge defaults with per-dataset overrides
        ds_cfg = {**defaults, **all_datasets[name]}
        # Deep-merge nested dicts that are not present in the per-dataset block
        for nested_key in ("v2v", "task_ratio", "caption_key"):
            if nested_key in defaults and nested_key not in all_datasets[name]:
                ds_cfg[nested_key] = defaults[nested_key]

        source_fps = ds_cfg.get("source_fps", None)    # None lets LoadVideo read from metadata
        source_resolution = ds_cfg.get("source_resolution", None)   # [h, w] or None
        if source_resolution is not None:
            source_h, source_w = int(source_resolution[0]), int(source_resolution[1])
        else:
            source_h, source_w = None, None

        # Build video operator; memory_rollout tasks need the first frame as a global anchor
        task_ratio = ds_cfg.get("task_ratio", {}) or {}
        needs_first_frame = "memory_rollout" in task_ratio
        frame_processor = ImageCropAndResize(height, width, max_pixels, 16, 16)
        # Hoisted so UnifiedDataset can set its per-sample event_hint (skill/event sources).
        video_loader = LoadVideo(
            global_num_frames, 4, 1,
            frame_processor=frame_processor,
            target_fps=global_target_fps,
            source_fps=source_fps,
            random_start=True,
            return_first_frame=needs_first_frame,
            # Skip clips too short to fill num_frames (keeps latent frame count aligned
            # to the training latent_window); enabled per-dataset in the data yaml.
            require_full_length=ds_cfg.get("require_full_length", False),
        )
        video_operator = RouteByType(operator_map=[
            (str, ToAbsolutePath(ds_cfg["video_dir"]) >> RouteByExtensionName(operator_map=[
                (("jpg", "jpeg", "png", "webp"), LoadImage() >> frame_processor >> ToList()),
                (("gif",), LoadGIF(global_num_frames, 4, 1, frame_processor=frame_processor)),
                (("mp4", "avi", "mov", "wmv", "mkv", "flv", "webm"), video_loader),
            ])),
        ])

        # Skill (VFX event) source flags: drop-warp training + event-window targeting.
        skill_source = bool(ds_cfg.get("skill_source", False))
        event_window_keys = ds_cfg.get("event_window_keys", None)

        caption_operator = ResolveCaptionFile(
            caption_dir=ds_cfg.get("caption_dir", ""),
            caption_key=ds_cfg.get("caption_key", "overall_caption"),
            target_fps=global_target_fps,
            # inline_caption=True: prompt field holds the caption text itself, not a path.
            inline=ds_cfg.get("inline_caption", False),
            # Text field of a segmented caption, and whether to strip camera tags; either may be set
            #   in defaults or per dataset. Matching the EvokeTeacher teacher's own training means
            #   short_prompt + true.
            segment_text_key=ds_cfg.get("segment_text_key", "full_prompt"),
            strip_camera_tags=ds_cfg.get("strip_camera_tags", False),
        )
        # Print the caption semantics actually in effect once at construction (rank 0). These two keys
        #   live in the data yaml's defaults, and a broken merge chain would silently fall back to
        #   full_prompt -- no error, just a different text domain -- so it has to be visible.
        if os.environ.get("RANK", "0") == "0":
            print(f"[LW-ALIGN][data] {name}: caption_key={ds_cfg.get('caption_key', 'overall_caption')} "
                  f"segment_text_key={ds_cfg.get('segment_text_key', 'full_prompt')} "
                  f"strip_camera_tags={bool(ds_cfg.get('strip_camera_tags', False))} "
                  f"require_full_length={bool(ds_cfg.get('require_full_length', False))}", flush=True)

        # Include pose key when pose_dir is configured
        has_pose = "pose_dir" in ds_cfg and ds_cfg["pose_dir"]
        data_file_keys = ("video", "prompt", "pose") if has_pose else ("video", "prompt")

        # Map dataset-specific jsonl key names to the standard video/pose/prompt keys
        jsonl_key_map = ds_cfg.get("jsonl_key_map", None)
        if jsonl_key_map is None:
            # Auto-detect common _path suffix variants
            jsonl_key_map = {
                "video_path": "video",
                "pose_path": "pose",
                "prompt_path": "prompt",
            }

        ds = UnifiedDataset(
            base_path=ds_cfg["video_dir"],
            metadata_path=ds_cfg["jsonl"],
            data_file_keys=data_file_keys,
            main_data_operator=video_operator,
            special_operator_map={"prompt": caption_operator},
            jsonl_key_map=jsonl_key_map,
            pose_dir=ds_cfg.get("pose_dir", None),
            target_h=height,
            target_w=width,
            source_h=source_h,
            source_w=source_w,
            # Use a default intrinsic when the pose npz has none (some sources store only c2w).
            fallback_default_intrinsic=ds_cfg.get("fallback_default_intrinsic", False),
            event_window_keys=event_window_keys,
            skill_source=skill_source,
            video_loader=video_loader,
        )

        # Apply path_replace to fix jsonl path prefixes so they match actual disk layout.
        # Format in yaml:
        #   path_replace:
        #     video:  {'PusaV1/': 'PusaV1/train/'}
        #     prompt: {'Annotation/': ''}
        #     pose:   {'Annotation/': ''}
        # Keys refer to the post-remap standard names (video/prompt/pose).
        # Each (old, new) uses str.replace(old, new, 1) to replace only the first occurrence.
        path_replace = ds_cfg.get("path_replace", None)
        if path_replace:
            for entry in ds.data:
                for key, replace_map in path_replace.items():
                    val = entry.get(key)
                    if not isinstance(val, str):
                        continue
                    for old, new in replace_map.items():
                        val = val.replace(old, new, 1)
                    entry[key] = val

        raw_sizes.append(len(ds))

        ds_meta = {
            "task_ratio": ds_cfg.get("task_ratio", {"t2v": 0.1, "i2v": 0.7, "v2v": 0.2}),
            "v2v": ds_cfg.get("v2v", {}),
            "has_pose": has_pose,
            "skill": skill_source,
        }
        sub_datasets.append(ConfigAwareDataset(ds, ds_meta, target_fps=global_target_fps))

    # Build final ConcatDataset according to ratio:
    # ratio >= 1: repeat full passes plus fractional random subsample
    # ratio < 1:  random subsample at that fraction
    parts = []
    total_effective = 0

    # part_id is a stable index of "which SubsampledDataset this is", fed into the sampling seed so
    #   different parts of one epoch draw different subsets even over the same inner dataset. It
    #   follows only from select/ratio, hence is identical on every rank.
    _sub_part_id = 0
    _n_resampled = 0

    for ds, r, name, raw_sz in zip(sub_datasets, ratio, select, raw_sizes):
        effective = int(raw_sz * r)

        def _mk_sub(_inner, _size):
            nonlocal _sub_part_id, _n_resampled
            _sub = SubsampledDataset(_inner, _size, part_id=_sub_part_id,
                                     resample_each_epoch=resample_each_epoch, base_seed=seed)
            _sub_part_id += 1
            if resample_each_epoch:
                _n_resampled += 1
            return _sub

        if r >= 1:
            full_repeats = int(r)
            frac = r - full_repeats
            # Full passes
            for _ in range(full_repeats):
                parts.append(ds)
            # Fractional remainder as random subsample
            if frac > 0:
                frac_size = int(raw_sz * frac)
                if frac_size > 0:
                    parts.append(_mk_sub(ds, frac_size))
        else:
            # ratio < 1: random subsample
            subset_size = max(1, int(raw_sz * r))
            parts.append(_mk_sub(ds, subset_size))

        total_effective += effective
        # pool = how many clips of this source are reachable per epoch. With resampling off that is a
        #   permanent ceiling; with it on the subset changes each epoch and coverage accumulates.
        print(f"[DataConfig] {name}: {raw_sz} x {r} = {effective} effective samples"
              f"{'  [re-drawn each epoch]' if (resample_each_epoch and r != int(r)) else ''}")

    dataset = torch.utils.data.ConcatDataset(parts)
    print(f"[DataConfig] Total: {len(dataset)} samples/epoch  (select={select}, ratio={ratio})")
    if resample_each_epoch:
        print(f"[EPOCH-RESAMPLE] on: {_n_resampled} randomly subsampled parts are re-drawn every epoch"
              f" (same size, so the epoch length does not change); seed = f(seed={seed}, epoch, part_id), rank-independent."
              f" Requires persistent_workers=false for workers to see the new subset each epoch.", flush=True)
    return dataset
