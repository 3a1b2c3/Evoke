import os
import json
import random
import re
from .operators import DataProcessingOperator

# Mirrors the teacher's strip_camera_tags: drop whole <camera>...</camera> spans (DOTALL,
#   case-insensitive), then leftover bare tags, then normalise whitespace.
_CAMERA_BLOCK_RE = re.compile(r"<camera>.*?</camera>", re.DOTALL | re.IGNORECASE)
_CAMERA_BARE_RE = re.compile(r"</?camera>", re.IGNORECASE)
_WS_RE = re.compile(r"\s+")


def strip_camera_tags_text(text: str) -> str:
    """Strip <camera>...</camera> spans and bare tags, normalise whitespace. Non-str passes through."""
    if not isinstance(text, str):
        return text
    t = _CAMERA_BLOCK_RE.sub(" ", text)
    t = _CAMERA_BARE_RE.sub(" ", t)
    return _WS_RE.sub(" ", t).strip()


class ResolveCaptionFile(DataProcessingOperator):
    """Resolve a caption JSON file path to a text prompt (and optional segment_prompts)."""

    # Key prefixes that indicate interleave-style captions.
    INTERLEAVE_PREFIXES = ("interleave_caption",)
    # Segmented caption schema: a list of {time_range_s: [s, e], full_prompt, ...}, parsed into
    # the same segment_prompts shape as the interleave format.
    SEGMENTS_LIST_KEYS = ("segments", "merged_segments")

    def __init__(self, caption_dir, caption_key="overall_caption", target_fps=24, inline=False,
                 segment_text_key="full_prompt", strip_camera_tags=False):
        self.caption_dir = caption_dir
        self.target_fps = target_fps
        # Which text field a segmented caption reads, and whether to strip camera tags. The
        #   defaults reproduce the behaviour that predates these options. The teacher was trained
        #   with short_prompt + strip_camera_tags, so distillation must set both to match it.
        self.segment_text_key = segment_text_key
        self.strip_camera_tags = bool(strip_camera_tags)
        # inline=True: the jsonl carries the caption text directly (no caption file to open),
        # sources whose lines carry an `overall_caption` string instead of a `prompt_path`.
        self.inline = inline

        # Normalize caption_key to {key: weight} dict.
        if isinstance(caption_key, str):
            self.caption_keys = {caption_key: 1.0}
        elif isinstance(caption_key, dict):
            self.caption_keys = caption_key
        else:
            self.caption_keys = {"overall_caption": 1.0}

        self._keys = list(self.caption_keys.keys())
        self._weights = list(self.caption_keys.values())

    def _is_interleave(self, key):
        return any(key.startswith(p) for p in self.INTERLEAVE_PREFIXES)

    def _is_segments_list(self, key):
        return key in self.SEGMENTS_LIST_KEYS

    def _parse_segments_list(self, segments_data, path=""):
        """List schema -> segment_prompts, same shape as _parse_interleave.

        time_range_s is in clip-relative seconds, like interleave's start/end_time;
        _align_segment_prompts crops to the load window downstream. The prompt comes from
        self.segment_text_key, and a missing field is a hard error, never a silent fallback."""
        segments = []
        for seg in sorted(segments_data, key=lambda x: float(x["time_range_s"][0])):
            if self.segment_text_key not in seg:
                raise ValueError(
                    f"segment is missing segment_text_key '{self.segment_text_key}' (no silent fallback). "
                    f"path={path}, segment_index={seg.get('segment_index')}, "
                    f"fields present in this segment={sorted(seg.keys())}. Fix: 1) change segment_text_key in "
                    f"the data yaml, or 2) regenerate the caption json so the field exists."
                )
            _p = seg[self.segment_text_key]
            if self.strip_camera_tags:
                _p = strip_camera_tags_text(_p)
            if not isinstance(_p, str) or not _p.strip():
                # Drop segments that strip to nothing instead of feeding T5 an empty prompt.
                continue
            segments.append({
                "start_frame": int(float(seg["time_range_s"][0]) * self.target_fps),
                "end_frame": int(float(seg["time_range_s"][1]) * self.target_fps),
                "prompt": _p,
            })
        return segments

    def _parse_interleave(self, interleave_data):
        """Convert interleave_caption dict to a segment_prompts list."""
        segments = []
        for seg_key in sorted(interleave_data.keys()):
            seg = interleave_data[seg_key]
            _p = seg["caption"]
            if self.strip_camera_tags:   # same handling as the segments path
                _p = strip_camera_tags_text(_p)
                if not isinstance(_p, str) or not _p.strip():
                    continue
            segments.append({
                "start_frame": int(seg["start_time"] * self.target_fps),
                "end_frame": int(seg["end_time"] * self.target_fps),
                "prompt": _p,
            })
        return segments

    @staticmethod
    def _lookup(cap, dotted_key):
        """Resolve a possibly-dotted caption key into nested dicts.

        'overall.description' walks cap['overall']['description']; a plain key with no
        dots behaves like cap.get(key). Returns (value, found).
        """
        cur = cap
        for part in dotted_key.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return None, False
            cur = cur[part]
        return cur, True

    def __call__(self, caption_value):
        # Inline mode: caption_value IS the caption text (jsonl-embedded), not a path.
        if self.inline:
            if not isinstance(caption_value, str) or not caption_value.strip():
                raise ValueError(f"inline caption must be a non-empty string, got: {caption_value!r}")
            return {"text": caption_value}

        # File mode: caption_value is a path to a caption JSON file.
        # Try full relative path first; fall back to basename for older datasets.
        path = os.path.join(self.caption_dir, caption_value)
        if not os.path.exists(path):
            path = os.path.join(self.caption_dir, os.path.basename(caption_value))
        with open(path, "r", encoding="utf-8") as f:
            cap = json.load(f)

        # Sample one caption key according to configured weights.
        chosen_key = random.choices(self._keys, weights=self._weights)[0]

        # Resolve the (possibly dotted, e.g. 'overall.description') key; fail fast if absent.
        value, found = self._lookup(cap, chosen_key)
        if not found:
            raise ValueError(
                f"caption_key '{chosen_key}' is missing from the caption JSON; no silent fallback. "
                f"path={path}, top_keys={sorted(list(cap.keys()))}. "
                f"configured caption_key={self._keys} (dotted nesting is supported, e.g. 'overall.description'). "
                f"Fix: 1) check the spelling / nesting of caption_key, 2) regenerate the caption json so the "
                f"field exists, or 3) use path_replace to point the prompt path at a directory that has it."
            )

        result = {}

        if self._is_segments_list(chosen_key):
            # List segment schema.
            if not isinstance(value, list) or not value:
                raise ValueError(
                    f"caption_key '{chosen_key}' must be a non-empty list (segmented schema), "
                    f"path={path}, value type={type(value).__name__}"
                )
            result["segment_prompts"] = self._parse_segments_list(value, path=path)
            if not result["segment_prompts"]:
                raise ValueError(
                    f"caption_key '{chosen_key}' parsed to no usable segments: every segment's "
                    f"'{self.segment_text_key}' is empty, or became empty after stripping camera tags. path={path}"
                )
            result["text"] = result["segment_prompts"][0]["prompt"]
        elif self._is_interleave(chosen_key):
            # Parse interleave caption into segment_prompts; text is a placeholder.
            interleave_data = value
            if not isinstance(interleave_data, dict) or not interleave_data:
                raise ValueError(
                    f"caption_key '{chosen_key}' is of interleave type, but the JSON field is not a non-empty dict. "
                    f"path={path}, value type={type(interleave_data).__name__}"
                )
            result["segment_prompts"] = self._parse_interleave(interleave_data)
            if not result["segment_prompts"]:
                raise ValueError(
                    f"caption_key '{chosen_key}' parsed to an empty segment list. path={path}"
                )
            # Use first segment caption as placeholder text.
            result["text"] = result["segment_prompts"][0]["prompt"]
        else:
            # Plain caption: must be a non-empty string.
            text = value
            if not isinstance(text, str) or not text.strip():
                raise ValueError(
                    f"caption_key '{chosen_key}' is not a non-empty string. path={path}, value={text!r}"
                )
            if self.strip_camera_tags:   # the whole-clip caption carries tags too
                text = strip_camera_tags_text(text)
                if not text:
                    raise ValueError(
                        f"caption_key '{chosen_key}' became empty after stripping camera tags. path={path}"
                    )
            result["text"] = text

        return result
