"""case2 — DSL-primitive ONNX solvers overriding the case1 baseline.

Build a submission bundle by re-solving tasks with minimal hand-built ONNX
(geometry / recolor / stencil primitives) and overriding the case1 baseline
wherever a case2 net is exactly correct and strictly cheaper. Submit until a
target Public Score is reached.
"""

from __future__ import annotations

from .build import TaskOutcome, build_override_bundle, summarize
from .solver import Solution, solve

TARGET_PUBLIC_SCORE = 7665.0

__all__ = [
    "TARGET_PUBLIC_SCORE",
    "Solution",
    "TaskOutcome",
    "build_override_bundle",
    "solve",
    "summarize",
]
