"""Online multi-dataset loader; returns raw video + prompt for materialize_online_batch to encode."""
from __future__ import annotations

import numpy as np
import torch
import torch.multiprocessing as mp
from torch.utils.data import Dataset

from evoke.dataset.evoke_data import load_data_config

# Use file_system sharing to avoid exhausting /dev/shm with large video tensors.
try:
    if mp.get_sharing_strategy() != "file_system":
        mp.set_sharing_strategy("file_system")
except Exception:
    pass


def _stack_pil_frames(frames) -> torch.Tensor:
    # Convert list of PIL images to [C, T, H, W] float32 tensor in [-1, 1].
    arr_list = []
    for f in frames:
        a = np.asarray(f, dtype=np.float32) / 255.0
        if a.ndim == 2:
            a = np.stack([a] * 3, axis=-1)
        arr_list.append(a)
    arr = np.stack(arr_list, axis=0)                              # [T, H, W, C]
    t = torch.from_numpy(arr).permute(3, 0, 1, 2).contiguous()    # [C, T, H, W]
    return t * 2.0 - 1.0


class BucketedFeatureDataset(Dataset):
    """Thin wrapper around EvokeTeacher ConcatDataset with fixed resolution and frame count."""

    def __init__(
        self,
        data_yaml_path: str,
        single_height: int = 384,
        single_width: int = 640,
        num_frames: int = 105,
        target_fps: int = 24,
        history_sizes=(16, 2, 1),
        is_keep_x0: bool = True,
        seed: int = 42,
        # True: re-draw the random subsample of each `ratio` entry every epoch.
        #   False: drawn once at construction, then fixed for the whole run.
        resample_ratio_each_epoch: bool = False,
        # Compatibility with offline dataset_kwargs; unused in online path.
        **_unused,
    ):
        self.inner = load_data_config(
            data_yaml_path,
            height=single_height,
            width=single_width,
            num_frames=num_frames,
            target_fps=target_fps,
            resample_each_epoch=bool(resample_ratio_each_epoch),
            seed=int(seed),
        )
        self.resample_ratio_each_epoch = bool(resample_ratio_each_epoch)
        self.single_height = single_height
        self.single_width = single_width
        self.num_frames = num_frames
        self.target_fps = target_fps
        self.history_sizes = list(history_sizes)
        self.is_keep_x0 = is_keep_x0
        self.seed = seed
        self._epoch = 0

        # Compatibility attributes expected by train_evoke legacy code paths.
        bkey = (num_frames, single_height, single_width)
        self.buckets = {bkey: list(range(len(self.inner)))}
        self.samples = [{"dataset_name": "online_multi", "bucket_key": bkey} for _ in range(len(self.inner))]

    def __len__(self):
        return len(self.inner)

    def set_epoch(self, epoch: int):
        self._epoch = epoch
        # Push the epoch down to the subsampled parts so they swap in a different set of videos
        #   (same size). Must run before this epoch's DataLoader iterator is created, so the
        #   workers re-forked each epoch (persistent_workers=false) pick up the new indices.
        if getattr(self, "resample_ratio_each_epoch", False):
            from evoke.dataset.evoke_data.data_config import set_dataset_epoch
            _n = set_dataset_epoch(self.inner, epoch)
            if __import__("os").environ.get("RANK", "0") == "0":
                print(f"[EPOCH-RESAMPLE] epoch={epoch}: re-drew {_n} subsampled parts"
                      f" (total sample count {len(self.inner)} unchanged)", flush=True)

    def __getitem__(self, idx: int):
        # The online loader randomly crops before the
        # training loop sees uttid.  During the GPU equivalence test only, seed
        # this access by dataset index so R/B get identical bytes even when a
        # clip is assigned to a different rank.  Default path is unchanged.
        if __import__("os").environ.get("SF_DECOUPLE_EQUIV_MODE"):
            from scripts.training.tmp.test_decouple_equivalence import deterministic_dataset_item
            data = deterministic_dataset_item(self.inner, idx)
        else:
            data = self.inner[idx]
        video = _stack_pil_frames(data["video"])    # [C, T, H, W]
        out = {
            "raw_video": video,
            "prompt": data["prompt"],
            # Per-segment prompts for interleave mode; None when not applicable.
            "segment_prompts": data.get("segment_prompts"),
            "bucket_key": (self.num_frames, self.single_height, self.single_width),
            "uttid": f"online_{idx}",
            "dataset_name": "online_multi",
        }
        # Pass camera pose tensors when present (only if data yaml includes pose_dir).
        if "lingbot_Ks" in data and "lingbot_c2ws" in data:
            out["lingbot_Ks"] = torch.as_tensor(data["lingbot_Ks"], dtype=torch.float32)    # [4]
            out["lingbot_c2ws"] = torch.as_tensor(data["lingbot_c2ws"], dtype=torch.float32) # [N_pix, 4, 4]
        # Skill (grok event/VFX) sample: drop-warp + event-window target. event_window is in
        # LOADED-frame (pixel) positions; None when the loaded window missed the event.
        if data.get("__skill__"):
            out["is_skill"] = True
            out["event_window"] = data.get("__event_window__")
        return out


class BucketedSampler:
    """Batch sampler with shuffle, DDP sharding, and fixed batch size."""

    def __init__(
        self,
        dataset: BucketedFeatureDataset,
        batch_size: int,
        drop_last: bool = True,
        shuffle: bool = True,
        seed: int = 42,
        num_sp_groups: int = 1,
        sp_world_size: int = 1,
        global_rank: int = 0,
        decouple_rollout: bool = False,
        **_unused,
    ):
        self.dataset = dataset
        # Mirror dataset buckets onto the sampler (matches history_latents_dist / mp4_dist
        # samplers); error-recycling buffer init reads sampler.buckets to enumerate resolutions.
        self.buckets = dataset.buckets
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed
        self.num_sp_groups = num_sp_groups
        self.sp_world_size = sp_world_size
        self.global_rank = global_rank
        # All ranks in the same sp_group share a batch (the default).
        self.ith_sp_group = global_rank // max(sp_world_size, 1)
        # decouple_rollout: one distinct clip per rank, so shard by global_rank (stride = world
        #   size). Off: shard by sp_group, so the G ranks of a group share one clip.
        self.decouple_rollout = bool(decouple_rollout)
        if self.decouple_rollout:
            self.shard_key = int(global_rank)
            self.num_shards = int(num_sp_groups) * max(int(sp_world_size), 1)
        else:
            self.shard_key = self.ith_sp_group
            self.num_shards = int(num_sp_groups)
        self._epoch = 0

    def set_epoch(self, epoch: int):
        self._epoch = epoch
        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(epoch)

    def __iter__(self):
        n = len(self.dataset)
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self._epoch * 1000003)
            indices = torch.randperm(n, generator=g).tolist()
        else:
            indices = list(range(n))
        # Shard indices: by sp_group (shared clip), or by global_rank when decoupled.
        shard = indices[self.shard_key :: max(self.num_shards, 1)]
        for i in range(0, len(shard), self.batch_size):
            batch = shard[i : i + self.batch_size]
            if len(batch) < self.batch_size and self.drop_last:
                break
            yield batch

    def __len__(self):
        per_rank = len(self.dataset) // max(self.num_sp_groups, 1)
        if self.drop_last:
            return per_rank // self.batch_size
        return (per_rank + self.batch_size - 1) // self.batch_size


def collate_fn(batch):
    out = {
        "raw_video": torch.stack([b["raw_video"] for b in batch], dim=0),  # [B, C, T, H, W]
        "prompt": [b["prompt"] for b in batch],
        # Per-sample segment prompts for interleave mode; None otherwise.
        "segment_prompts": [b.get("segment_prompts") for b in batch],
        "uttid": [b["uttid"] for b in batch],
        "dataset_name": [b["dataset_name"] for b in batch],
        "bucket_key": batch[0]["bucket_key"],
    }
    # Stack camera poses if present; all samples in a batch must carry them.
    if batch[0].get("lingbot_Ks") is not None:
        out["lingbot_Ks"] = torch.stack([b["lingbot_Ks"] for b in batch], dim=0)    # [B, 4]
        out["lingbot_c2ws"] = torch.stack([b["lingbot_c2ws"] for b in batch], dim=0)  # [B, N_pix, 4, 4]
    # Skill flag + event windows (per-sample). GEO forces train_batch_size=1, so these are length-B
    # lists; materialize reads element [0]. Absent (non-skill batches) → keys omitted → default off.
    if any(b.get("is_skill") for b in batch):
        out["is_skill"] = [bool(b.get("is_skill", False)) for b in batch]
        out["event_window"] = [b.get("event_window") for b in batch]
    return out
