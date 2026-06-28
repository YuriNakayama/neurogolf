"""case3 — per-task minimal ONNX solver bank + golf vs baseline.

ベースライン（case1, LB 7159.44）が解く高 cost タスクを、最小 cost の ONNX で
置き換えて Public Score を伸ばす build-case。solver で厳密に解けたタスクは安価な
ONNX を、それ以外はベースライン ONNX を採用する（merge は別モジュール）。
"""

from __future__ import annotations

from .run import Solved, run, solve_task, write_manifest

__all__ = ["Solved", "run", "solve_task", "write_manifest"]
