import torch, torchvision, imageio, os
import imageio.v3 as iio
from PIL import Image
import numpy as np

class DataProcessingPipeline:
    def __init__(self, operators=None):
        self.operators: list[DataProcessingOperator] = [] if operators is None else operators
        
    def __call__(self, data):
        for operator in self.operators:
            data = operator(data)
        return data
    
    def __rshift__(self, pipe):
        if isinstance(pipe, DataProcessingOperator):
            pipe = DataProcessingPipeline([pipe])
        return DataProcessingPipeline(self.operators + pipe.operators)


class DataProcessingOperator:
    def __call__(self, data):
        raise NotImplementedError("DataProcessingOperator cannot be called directly.")
    
    def __rshift__(self, pipe):
        if isinstance(pipe, DataProcessingOperator):
            pipe = DataProcessingPipeline([pipe])
        return DataProcessingPipeline([self]).__rshift__(pipe)


class DataProcessingOperatorRaw(DataProcessingOperator):
    def __call__(self, data):
        return data


class ToInt(DataProcessingOperator):
    def __call__(self, data):
        return int(data)


class ToFloat(DataProcessingOperator):
    def __call__(self, data):
        return float(data)


class ToStr(DataProcessingOperator):
    def __init__(self, none_value=""):
        self.none_value = none_value
    
    def __call__(self, data):
        if data is None: data = self.none_value
        return str(data)


class LoadImage(DataProcessingOperator):
    def __init__(self, convert_RGB=True, convert_RGBA=False):
        self.convert_RGB = convert_RGB
        self.convert_RGBA = convert_RGBA
    
    def __call__(self, data: str):
        image = Image.open(data)
        if self.convert_RGB: image = image.convert("RGB")
        if self.convert_RGBA: image = image.convert("RGBA")
        return image


class ImageCropAndResize(DataProcessingOperator):
    def __init__(self, height=None, width=None, max_pixels=None, height_division_factor=1, width_division_factor=1):
        self.height = height
        self.width = width
        self.max_pixels = max_pixels
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor

    def crop_and_resize(self, image, target_height, target_width):
        width, height = image.size
        scale = max(target_width / width, target_height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        image = torchvision.transforms.functional.center_crop(image, (target_height, target_width))
        return image
    
    def get_height_width(self, image):
        if self.height is None or self.width is None:
            width, height = image.size
            if width * height > self.max_pixels:
                scale = (width * height / self.max_pixels) ** 0.5
                height, width = int(height / scale), int(width / scale)
            height = height // self.height_division_factor * self.height_division_factor
            width = width // self.width_division_factor * self.width_division_factor
        else:
            height, width = self.height, self.width
        return height, width
    
    def __call__(self, data: Image.Image):
        image = self.crop_and_resize(data, *self.get_height_width(data))
        return image


class ToList(DataProcessingOperator):
    def __call__(self, data):
        return [data]
    

class LoadVideo(DataProcessingOperator):
    def __init__(self, num_frames=81, time_division_factor=4, time_division_remainder=1,
                 frame_processor=lambda x: x, target_fps=None,
                 source_fps=None, random_start=False, return_first_frame=False,
                 require_full_length=False):
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        # frame_processor is applied per-frame inside the loader.
        self.frame_processor = frame_processor
        # target_fps: resample source video to this frame rate.
        self.target_fps = target_fps
        # source_fps: explicit source fps from config; falls back to video metadata.
        self.source_fps = source_fps
        # random_start: randomly select clip start instead of always using frame 0.
        self.random_start = random_start
        # return_first_frame: also return source frame 0 as a global anchor.
        self.return_first_frame = return_first_frame
        # require_full_length: if the video can't supply the full num_frames (at target_fps),
        # raise so the dataset retries another sample. Guarantees fixed-length clips → latent
        # frame count stays aligned to the training latent_window (short clips would misalign).
        self.require_full_length = require_full_length
        # event_hint: transient per-sample (event_start_src, event_end_src) in SOURCE frame indices,
        # set by UnifiedDataset just before a skill-source sample is loaded. When present, the loaded
        # window is biased so the event sits in its latter half (pre-event scene stays as history).
        # None (default) → behaviour is byte-for-byte the original random_start path.
        self.event_hint = None

    def _event_biased_start(self, es_t, ee_t, num_output, max_start):
        """Pick a start (target-frame space) so [start, start+num_output) covers the event with
        pre-event history before it. Falls back to clamped placement; jittered for diversity."""
        import random as _random
        if max_start <= 0:
            return 0
        desired = int(es_t) - num_output // 2          # event begins ~mid-window → history precedes it
        jit = num_output // 8
        if jit > 0:
            desired += _random.randint(-jit, jit)
        return min(max(0, desired), max_start)

    def _adjust_num_frames(self, num_frames):
        """Adjust num_frames to satisfy time_division constraints."""
        while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
            num_frames -= 1
        return num_frames

    def get_frame_indices(self, reader):
        """Compute frame indices to load, with optional fps-based temporal subsampling and random start."""
        import random as _random
        total_frames = int(reader.count_frames())

        if self.target_fps is not None:
            # prefer configured source_fps; fall back to video metadata
            source_fps = self.source_fps
            if source_fps is None:
                meta = reader.get_meta_data()
                source_fps = meta.get('fps', None)
            if source_fps is not None and source_fps != self.target_fps:
                # stride: source frames per target frame (>1 = downsample, <1 = upsample)
                stride = source_fps / self.target_fps
                video_duration = total_frames / source_fps
                max_available = int(video_duration * self.target_fps) + 1
                num_output = min(self.num_frames, max_available)
                num_output = self._adjust_num_frames(num_output)

                if self.require_full_length and num_output < self.num_frames:
                    raise RuntimeError(
                        f"LoadVideo: video too short for require_full_length "
                        f"(resampled max_available={max_available} < num_frames={self.num_frames}); skip & retry")

                if self.event_hint is not None and max_available > num_output:
                    # Skill source: bias the window to include the event (source frames -> target via stride).
                    max_start = max_available - num_output
                    es_t, ee_t = int(self.event_hint[0] / stride), int(self.event_hint[1] / stride)
                    start_idx = self._event_biased_start(es_t, ee_t, num_output, max_start)
                elif self.random_start and max_available > num_output:
                    max_start = max_available - num_output
                    start_idx = _random.randint(0, max_start)
                else:
                    start_idx = 0

                # map target frame index to nearest source frame
                indices = [min(int(round((start_idx + i) * stride)), total_frames - 1) for i in range(num_output)]
                self._last_start_time = start_idx / self.target_fps
                self._last_source_fps = source_fps
                return indices

        # Fallback: sequential frames
        num_frames = min(self.num_frames, total_frames)
        if total_frames < self.num_frames:
            num_frames = self._adjust_num_frames(num_frames)

        if self.require_full_length and num_frames < self.num_frames:
            raise RuntimeError(
                f"LoadVideo: video too short for require_full_length "
                f"(total_frames={total_frames} < num_frames={self.num_frames}); skip & retry")

        if self.event_hint is not None and total_frames > num_frames:
            # Skill source (no fps resample): event_hint already in source-frame space (stride=1).
            max_start = total_frames - num_frames
            start_idx = self._event_biased_start(self.event_hint[0], self.event_hint[1], num_frames, max_start)
        elif self.random_start and total_frames > num_frames:
            max_start = total_frames - num_frames
            start_idx = _random.randint(0, max_start)
        else:
            start_idx = 0

        source_fps = None
        try:
            meta = reader.get_meta_data()
            source_fps = meta.get('fps', None)
        except Exception:
            pass
        self._last_start_time = start_idx / source_fps if source_fps else 0
        self._last_source_fps = source_fps
        return list(range(start_idx, start_idx + num_frames))

    def __call__(self, data: str):
        reader = imageio.get_reader(data)
        self._last_start_time = 0
        self._last_source_fps = None
        try:
            # get_frame_indices may raise (e.g. require_full_length skips a short clip); the
            # finally below still closes the reader so the ffmpeg process is never leaked.
            frame_indices = self.get_frame_indices(reader)
            frames = []
            first_frame = None  # source frame 0, populated when return_first_frame is set
            # reuse frames[0] as the anchor when the clip already starts at source frame 0
            first_frame_is_clip_start = (len(frame_indices) > 0 and frame_indices[0] == 0)
            for frame_id in frame_indices:
                # guard against count_frames() over-reporting on VFR/truncated videos;
                # wrap as RuntimeError so the upstream retry handler can skip the sample.
                try:
                    frame = reader.get_data(frame_id)
                except (IndexError, StopIteration) as e:
                    raise RuntimeError(
                        f"LoadVideo: truncated video '{data}' — count_frames over-reported "
                        f"(asked frame {frame_id}, decoder ended early)"
                    ) from e
                frame = Image.fromarray(frame)
                frame = self.frame_processor(frame)
                frames.append(frame)
            # fetch source frame 0 for global anchor when requested
            if self.return_first_frame:
                if first_frame_is_clip_start:
                    first_frame = frames[0]
                else:
                    try:
                        raw0 = reader.get_data(0)
                        first_frame = self.frame_processor(Image.fromarray(raw0))
                    except (IndexError, StopIteration) as e:
                        raise RuntimeError(
                            f"LoadVideo: failed to fetch source[0] from '{data}' for first_frame anchor"
                        ) from e
        finally:
            # ALWAYS release the ffmpeg reader — critical on the raise paths above. A short-clip-heavy
            # region (many require_full_length skips) would otherwise leak readers → dataloader worker
            # OOM-kill → NCCL heartbeat timeout → SIGABRT (root cause of the v4 step-283 crash).
            reader.close()
        # return 3-tuple (frames, start_time, frame_indices) or 4-tuple when return_first_frame is set
        if self.random_start:
            if self.return_first_frame:
                return (frames, self._last_start_time, frame_indices, first_frame)
            return (frames, self._last_start_time, frame_indices)
        return frames


class SequencialProcess(DataProcessingOperator):
    def __init__(self, operator=lambda x: x):
        self.operator = operator
        
    def __call__(self, data):
        return [self.operator(i) for i in data]


class LoadGIF(DataProcessingOperator):
    def __init__(self, num_frames=81, time_division_factor=4, time_division_remainder=1, frame_processor=lambda x: x):
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        # frame_processor is applied per-frame inside the loader.
        self.frame_processor = frame_processor

    def get_num_frames(self, path):
        num_frames = self.num_frames
        images = iio.imread(path, mode="RGB")
        if len(images) < num_frames:
            num_frames = len(images)
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        return num_frames
        
    def __call__(self, data: str):
        num_frames = self.get_num_frames(data)
        frames = []
        images = iio.imread(data, mode="RGB")
        for img in images:
            frame = Image.fromarray(img)
            frame = self.frame_processor(frame)
            frames.append(frame)
            if len(frames) >= num_frames:
                break
        return frames


class RouteByExtensionName(DataProcessingOperator):
    def __init__(self, operator_map):
        self.operator_map = operator_map
        
    def __call__(self, data: str):
        file_ext_name = data.split(".")[-1].lower()
        for ext_names, operator in self.operator_map:
            if ext_names is None or file_ext_name in ext_names:
                return operator(data)
        raise ValueError(f"Unsupported file: {data}")


class RouteByType(DataProcessingOperator):
    def __init__(self, operator_map):
        self.operator_map = operator_map
        
    def __call__(self, data):
        for dtype, operator in self.operator_map:
            if dtype is None or isinstance(data, dtype):
                return operator(data)
        raise ValueError(f"Unsupported data: {data}")


class LoadTorchPickle(DataProcessingOperator):
    def __init__(self, map_location="cpu"):
        self.map_location = map_location
        
    def __call__(self, data):
        return torch.load(data, map_location=self.map_location, weights_only=False)


class ToAbsolutePath(DataProcessingOperator):
    def __init__(self, base_path=""):
        self.base_path = base_path
        
    def __call__(self, data):
        return os.path.join(self.base_path, data)


# Camera parameter normalization utilities.

def normalize_intrinsic(
    intrinsic: np.ndarray,
    width: float = 584.0,
    height: float = 328.0
) -> np.ndarray:
    """Normalize a 3x3 camera intrinsic matrix to [0, 1] range by image dimensions."""
    assert intrinsic.ndim == 2, "intrinsic matrix must be 2-D"

    intrinsic = intrinsic.copy()
    intrinsic[0, 0] = intrinsic[0, 0] / width   # fx
    intrinsic[1, 1] = intrinsic[1, 1] / height  # fy
    intrinsic[0, 2] = intrinsic[0, 2] / width   # cx
    intrinsic[1, 2] = intrinsic[1, 2] / height  # cy

    return intrinsic


def normalize_cam_c2w(extrinsic: np.ndarray, normalize_translation: bool = True) -> np.ndarray:
    """Align all extrinsics to frame 0 and optionally scale translations to max-norm 1."""
    assert extrinsic.ndim == 3, "extrinsic matrix must be 3-D (N, 4, 4)"

    extrinsic0_inv = np.linalg.inv(extrinsic[0])
    extrinsic_aligned = [extrinsic0_inv @ e for e in extrinsic]

    if normalize_translation:
        translations = np.array([e[:3, 3] for e in extrinsic_aligned])
        max_translation_norm = np.linalg.norm(translations, axis=1).max()
        scale = 1.0 / max_translation_norm if max_translation_norm > 1e-10 else 1.0
        extrinsic_normalized = []
        for e in extrinsic_aligned:
            en = e.copy()
            en[:3, 3] *= scale
            extrinsic_normalized.append(en)
        return np.array(extrinsic_normalized)
    else:
        return np.array(extrinsic_aligned)


def compute_relative_pose(pose_a, pose_b, use_torch=False):
    """Compute relative pose matrix of camera B with respect to camera A"""
    assert pose_a.shape == (4, 4), f"Camera A extrinsic matrix should be (4,4), got {pose_a.shape}"
    assert pose_b.shape == (4, 4), f"Camera B extrinsic matrix should be (4,4), got {pose_b.shape}"
    
    if use_torch:
        if not isinstance(pose_a, torch.Tensor):
            pose_a = torch.from_numpy(pose_a).float()
        if not isinstance(pose_b, torch.Tensor):
            pose_b = torch.from_numpy(pose_b).float()
        
        pose_a_inv = torch.inverse(pose_a)
        relative_pose = torch.matmul(pose_b, pose_a_inv)
    else:
        if not isinstance(pose_a, np.ndarray):
            pose_a = np.array(pose_a, dtype=np.float32)
        if not isinstance(pose_b, np.ndarray):
            pose_b = np.array(pose_b, dtype=np.float32)
        
        pose_a_inv = np.linalg.inv(pose_a)
        relative_pose = np.matmul(pose_b, pose_a_inv)
    
    return relative_pose


# LingBot-World compatible camera utilities.

def SE3_inverse_torch(T):
    """Batched SE3 inverse in torch. Args: T [B,4,4]. Returns T_inv [B,4,4]."""
    Rot = T[:, :3, :3]       # [B, 3, 3]
    trans = T[:, :3, 3:]     # [B, 3, 1]
    R_inv = Rot.transpose(-1, -2)
    t_inv = -torch.bmm(R_inv, trans)
    T_inv = torch.eye(4, device=T.device, dtype=T.dtype).unsqueeze(0).expand(T.shape[0], -1, -1).clone()
    T_inv[:, :3, :3] = R_inv
    T_inv[:, :3, 3:] = t_inv
    return T_inv


def compute_relative_poses_lingbot(c2ws_mat, framewise=True, normalize_trans=True):
    """Compute relative poses from absolute c2w matrices, optionally framewise and translation-normalized."""
    # step 1: all frames relative to frame 0
    ref_w2cs = SE3_inverse_torch(c2ws_mat[0:1])  # [1, 4, 4]
    relative_poses = torch.matmul(ref_w2cs, c2ws_mat)  # [F, 4, 4]
    relative_poses[0] = torch.eye(4, device=c2ws_mat.device, dtype=c2ws_mat.dtype)

    # step 2: convert to per-frame relative poses
    if framewise:
        relative_poses_framewise = torch.bmm(
            SE3_inverse_torch(relative_poses[:-1]), relative_poses[1:])
        relative_poses[1:] = relative_poses_framewise

    # step 3: normalize translations to max-norm 1
    if normalize_trans:
        translations = relative_poses[:, :3, 3]  # [F, 3]
        max_norm = torch.norm(translations, dim=-1).max()
        if max_norm > 0:
            relative_poses[:, :3, 3] = translations / max_norm

    return relative_poses


def resolve_intrinsic_source_resolution(K_3x3, h_org, w_org, tag=""):
    """Pick the resolution K_3x3 was calibrated at, for feeding transform_intrinsic_for_crop_resize.

    A wrong declared size rescales fx/fy and shifts the principal point off center, so the size implied
    by the principal point wins when BOTH axes disagree by > 10% at a consistent ratio (the signature of
    a resolution mismatch). A single-axis disagreement is more likely a real off-center principal point,
    where overriding would corrupt a correct intrinsic. Normalized intrinsics carry no size (pp <= 1) and
    always keep the declared value. Must stay shared between the training and val/infer pose loaders.
    """
    try:
        h_inferred = int(round(float(K_3x3[1, 2]) * 2)) if float(K_3x3[1, 2]) > 1 else None
        w_inferred = int(round(float(K_3x3[0, 2]) * 2)) if float(K_3x3[0, 2]) > 1 else None
    except Exception:
        h_inferred, w_inferred = None, None

    # Orientation flip = the pose was solved on a rotated frame, so K AND c2w are in a rotated camera
    # frame and no choice of source resolution repairs it. Raise so training skips the sample
    # (ConfigAwareDataset retries on ValueError) and val/infer fails loudly instead of warping garbage.
    # Requires the inferred aspect to be the declared aspect inverted, so an off-center principal point
    # (whose inferred size is meaningless) is not mistaken for a rotation.
    if h_inferred is not None and w_inferred is not None and h_org and w_org:
        a_inferred, a_declared = w_inferred / h_inferred, h_org / w_org
        if (w_inferred > h_inferred) != (w_org > h_org) and abs(a_inferred - a_declared) <= 0.05 * a_declared:
            raise ValueError(
                f"[CamCtrl] {tag}: intrinsic calibrated at {h_inferred}x{w_inferred}, opposite orientation "
                f"to the declared {h_org}x{w_org} -- pose solved on a rotated frame, sample unusable")

    h_final = h_org if h_org is not None else (h_inferred if h_inferred is not None else 720)
    w_final = w_org if w_org is not None else (w_inferred if w_inferred is not None else 1280)

    off_h = h_org is not None and h_inferred is not None and abs(h_org - h_inferred) / max(h_inferred, 1) > 0.1
    off_w = w_org is not None and w_inferred is not None and abs(w_org - w_inferred) / max(w_inferred, 1) > 0.1
    if off_h and off_w:
        s_h, s_w = h_inferred / h_org, w_inferred / w_org
        if abs(s_h - s_w) <= 0.05 * max(s_h, s_w):
            print(f"[CamCtrl] {tag}: declared {h_org}x{w_org} overridden by intrinsic-derived "
                  f"{h_inferred}x{w_inferred} (both axes off > 10% at scale {s_h:.3f}/{s_w:.3f})")
            return h_inferred, w_inferred
    if off_h:
        print(f"[CamCtrl] WARN {tag}: declared source_h={h_org} differs from intrinsic-derived {h_inferred} by > 10%")
    if off_w:
        print(f"[CamCtrl] WARN {tag}: declared source_w={w_org} differs from intrinsic-derived {w_inferred} by > 10%")
    return h_final, w_final


def transform_intrinsic_for_crop_resize(K_3x3, h_org, w_org, h_target, w_target):
    """Transform a 3x3 intrinsic matrix from original resolution to training resolution via resize+center-crop.

    Handles both pixel-unit and normalized intrinsics (all values <=2 are treated as normalized).
    Returns a torch [4] tensor [fx, fy, cx, cy] in pixel units at the target resolution.
    """
    fx = float(K_3x3[0, 0])
    fy = float(K_3x3[1, 1])
    cx = float(K_3x3[0, 2])
    cy = float(K_3x3[1, 2])

    # auto-detect normalized intrinsic (e.g. VIPE output where fx=0.5 means focal=W/2)
    if abs(fx) <= 2.0 and abs(fy) <= 2.0 and abs(cx) <= 2.0 and abs(cy) <= 2.0:
        fx = fx * w_org
        cx = cx * w_org
        fy = fy * h_org
        cy = cy * h_org

    scale_h = h_target / h_org
    scale_w = w_target / w_org
    scale = max(scale_h, scale_w)

    h_resize = round(h_org * scale)
    w_resize = round(w_org * scale)

    # apply uniform scale
    fx_r = fx * scale
    fy_r = fy * scale
    cx_r = cx * scale
    cy_r = cy * scale

    # shift principal point for center crop
    crop_offset_x = (w_resize - w_target) / 2.0
    crop_offset_y = (h_resize - h_target) / 2.0
    cx_f = cx_r - crop_offset_x
    cy_f = cy_r - crop_offset_y

    return torch.tensor([fx_r, fy_r, cx_f, cy_f], dtype=torch.float32)