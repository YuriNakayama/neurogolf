"""Build a per-task ONNX bundle by trying each solver and keeping what solves.

For every task, candidate solvers are tried in roughly cost-ascending order
(zero-param spatial transforms and identity first, then the 1x1-conv recolor).
The first model that solves the task exactly (verified via the official-scorer
mirror) is saved as ``taskNNN.onnx``. Tasks no solver can solve are skipped —
omitting a file is safer than shipping a wrong one (both earn zero, but a wrong
file risks a malformed bundle).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import onnx

from dataset.loader import available_task_ids, load_task
from dataset.types import Task
from solvers.identity import build_identity_model
from solvers.recolor import build_recolor_for_task
from solvers.run import audit_model
from solvers.spatial import (
    build_flip_lr_model,
    build_flip_ud_model,
    build_rot180_model,
    build_transpose_model,
)

# (name, factory) in try-order. A factory returns None when inapplicable.
Candidate = tuple[str, Callable[[Task], onnx.ModelProto | None]]

_CANDIDATES: tuple[Candidate, ...] = (
    ("identity", lambda _t: build_identity_model()),
    ("transpose", lambda _t: build_transpose_model()),
    ("flip_lr", lambda _t: build_flip_lr_model()),
    ("flip_ud", lambda _t: build_flip_ud_model()),
    ("rot180", lambda _t: build_rot180_model()),
    ("recolor", build_recolor_for_task),
)


@dataclass(frozen=True)
class TaskBuild:
    """Result of solving (or failing to solve) one task."""

    task_id: int
    solver: str | None
    cost: int | None
    points: float | None


@dataclass(frozen=True)
class BuildSummary:
    """Outcome of building a whole bundle."""

    builds: tuple[TaskBuild, ...]

    @property
    def solved(self) -> tuple[TaskBuild, ...]:
        return tuple(b for b in self.builds if b.solver is not None)

    @property
    def total_points(self) -> float:
        return sum(b.points or 0.0 for b in self.solved)


def solve_task(task: Task) -> tuple[str, onnx.ModelProto, TaskBuild] | None:
    """Try each candidate solver; return the first that solves ``task``."""
    for name, factory in _CANDIDATES:
        model = factory(task)
        if model is None:
            continue
        result = audit_model(model, task)
        if result.solved:
            build = TaskBuild(task.task_id, name, result.cost, result.points)
            return name, model, build
    return None


def build_bundle(task_dir: Path, out_dir: Path) -> BuildSummary:
    """Build ``out_dir/taskNNN.onnx`` for every solvable task under ``task_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    builds: list[TaskBuild] = []
    for task_id in available_task_ids(task_dir):
        task = load_task(task_dir, task_id)
        solved = solve_task(task)
        if solved is None:
            builds.append(TaskBuild(task_id, None, None, None))
            continue
        _name, model, build = solved
        onnx.save(model, str(out_dir / f"task{task_id:03d}.onnx"))
        builds.append(build)
    return BuildSummary(builds=tuple(builds))
