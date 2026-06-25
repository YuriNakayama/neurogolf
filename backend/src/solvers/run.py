"""Run a candidate ONNX model against a task and check it solves it exactly.

Correctness is delegated to :func:`evaluate.audit_one` — the official-scorer
mirror — so "passes locally" means "passes the competition scorer". A model
solves a task when every valid example matches (``n_fail == 0`` and a clean
``status``). The audit needs a file on disk and emits an onnxruntime profile in
the cwd, so it runs inside a temporary directory.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass

import onnx

from dataset.types import Task
from evaluate import audit_one


@dataclass(frozen=True)
class SolveResult:
    """Outcome of auditing one model against one task."""

    solved: bool
    cost: int | None
    points: float | None
    n_pass: int
    n_fail: int
    status: str


def audit_model(model: onnx.ModelProto, task: Task) -> SolveResult:
    """Audit ``model`` against ``task`` via the scorer mirror."""
    examples = task.as_scorer_dict()
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, f"task{task.task_id:03d}.onnx")
        onnx.save(model, path)
        cwd = os.getcwd()
        os.chdir(tmp)  # contain onnxruntime profile output
        try:
            res = audit_one(path, examples, run_correctness=True)
        finally:
            os.chdir(cwd)
    solved = res["status"] == "ok" and res["n_fail"] == 0 and res["n_pass"] > 0
    return SolveResult(
        solved=solved,
        cost=res["cost"],
        points=res["points"],
        n_pass=res["n_pass"],
        n_fail=res["n_fail"],
        status=res["status"],
    )


def solves_task(model: onnx.ModelProto, task: Task) -> bool:
    """True when ``model`` solves every valid example of ``task``."""
    return audit_model(model, task).solved
