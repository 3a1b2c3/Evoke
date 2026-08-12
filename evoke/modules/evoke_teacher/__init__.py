"""Vendored EvokeTeacher sparse teacher, used by stage-3 DMD via real_score_arch=evoke_teacher.

The model files (dit_sparse_14b.py, dit_sparse_cam_14b.py,
select_gate.py) are upstream copies with their imports rewritten to be relative. The lazy
imports for SP / history_encoder / cam data operators keep their original paths: this usage
never reaches them, and a loud ImportError is the intended outcome if it ever does.
"""

from .loader import EVOKE_TEACHER_A14B_ARCH, EVOKE_TEACHER_NOCAM_SPARSE, build_evoke_teacher_dit, load_merged_weights
from .wrapper import EVOKE_TEACHER_LORA_TARGETS, EvokeTeacherScoreWrapper
