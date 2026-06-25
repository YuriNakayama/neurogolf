"""Validate per-task ONNX files before building a `submission.zip`.

Each `taskNNN.onnx` must load, avoid the competition's banned ops, stay within
the 1.44 MB cap, and yield a finite `cost` / `score`. Validation is delegated to
the official-scorer mirror (`evaluate.audit_one`, with correctness off — a
bundle's correctness is verified at build time) so local numbers match Kaggle.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from evaluate import audit_one

MAX_ONNX_BYTES = int(1.44 * 1024 * 1024)


class ValidationError(RuntimeError):
    """Raised when a task ONNX fails local validation."""


@dataclass(frozen=True)
class TaskValidation:
    """Validation result for a single task ONNX."""

    path: Path
    size_bytes: int
    cost: int
    score: float


def validate_onnx_file(path: Path) -> TaskValidation:
    """Audit one `taskNNN.onnx`; return its cost / score.

    Raises :class:`ValidationError` on any load / banned-op / scoring failure.
    """
    size = path.stat().st_size
    if size > MAX_ONNX_BYTES:
        raise ValidationError(
            f"{path.name}: ファイルサイズ超過: {size} > {MAX_ONNX_BYTES} bytes"
        )
    # audit_one emits an onnxruntime profile in the cwd; contain it.
    with tempfile.TemporaryDirectory() as tmp:
        cwd = os.getcwd()
        os.chdir(tmp)
        try:
            res = audit_one(str(path.resolve()), None, run_correctness=False)
        finally:
            os.chdir(cwd)
    if res["status"] != "ok" or res["cost"] is None or res["points"] is None:
        raise ValidationError(f"{path.name}: 検証失敗: {res['status']}")
    return TaskValidation(
        path=path,
        size_bytes=size,
        cost=int(res["cost"]),
        score=float(res["points"]),
    )


def validate_onnx_files(paths: list[Path]) -> list[TaskValidation]:
    """Validate every task ONNX, collecting all failures into one error."""
    results: list[TaskValidation] = []
    failures: list[str] = []
    for path in paths:
        try:
            results.append(validate_onnx_file(path))
        except ValidationError as exc:
            failures.append(str(exc))
    if failures:
        joined = "\n".join(failures)
        raise ValidationError(f"{len(failures)} 件の ONNX が検証に失敗:\n{joined}")
    return results
