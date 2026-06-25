"""case0 — generate a per-task ONNX bundle from the built-in solvers.

Produces ``data/output/onnx/taskNNN.onnx`` for every task a solver can solve
exactly, forming a bundle that ``src/submit`` can zip and that ``case1`` can
blend against another bundle.
"""

from __future__ import annotations

from pipeline.case0.build import BuildSummary, TaskBuild, build_bundle, solve_task

__all__ = ["BuildSummary", "TaskBuild", "build_bundle", "solve_task"]
