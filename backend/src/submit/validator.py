"""Validate per-task ONNX files before building a `submission.zip`.

Each `taskNNN.onnx` must load, avoid the competition's banned ops, and stay
within the 1.44 MB cap. Validation is delegated to the official-scorer mirror
(`evaluate.audit_one`, with correctness off) so local cost numbers match Kaggle.

**Hard vs soft failures.** `audit_one` may report a `score_error` /
`session_error` when this platform's onnxruntime cannot *load* an otherwise valid
graph for cost estimation (e.g. MaxPool / ConvTranspose with negative pads —
which Kaggle's scorer accepts and scores correctly). Such tasks are
**false-negatives locally**: the graph is structurally submittable, only the
local cost estimate is unavailable. Aborting the whole submission on these would
drop real Kaggle wins. So only the genuinely-disqualifying statuses
(`FILESIZE_OVER_LIMIT`, `load_error`, `BANNED_OP`, `sanitize_failed`) are hard
failures; `score_error` / `session_error` are soft — the task is kept in the zip
with an unknown cost. Kaggle is the final judge.
"""

from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from evaluate import audit_one

logger = logging.getLogger(__name__)

MAX_ONNX_BYTES = int(1.44 * 1024 * 1024)

# Statuses where the graph is structurally fine but this platform's onnxruntime
# could not run it to estimate cost. These are local false-negatives — keep the
# task in the submission (Kaggle scores it) but report an unknown cost.
_SOFT_STATUS_PREFIXES = ("score_error", "session_error")


class ValidationError(RuntimeError):
    """Raised when a task ONNX fails local validation (hard, disqualifying)."""


@dataclass(frozen=True)
class TaskValidation:
    """Validation result for a single task ONNX.

    `cost` / `score` are ``None`` for soft (locally-unscorable) tasks that are
    still kept in the submission.
    """

    path: Path
    size_bytes: int
    cost: int | None
    score: float | None
    scorable: bool


def _is_soft_status(status: str) -> bool:
    return any(status.startswith(prefix) for prefix in _SOFT_STATUS_PREFIXES)


def validate_onnx_file(path: Path) -> TaskValidation:
    """Audit one `taskNNN.onnx`; return its cost / score (None if unscorable).

    Raises :class:`ValidationError` only on hard, disqualifying failures
    (size cap, banned op, unloadable graph). A soft `score_error` /
    `session_error` returns a `scorable=False` result instead of raising.
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

    status = str(res["status"])
    if status == "ok" and res["cost"] is not None and res["points"] is not None:
        return TaskValidation(
            path=path,
            size_bytes=size,
            cost=int(res["cost"]),
            score=float(res["points"]),
            scorable=True,
        )
    if _is_soft_status(status):
        logger.warning(
            "%s: ローカルでコスト推定不可 (%s) だが構造は有効。"
            "Kaggle 採点に委ねて提出に含める。",
            path.name,
            status.split(":", 1)[0],
        )
        return TaskValidation(
            path=path, size_bytes=size, cost=None, score=None, scorable=False
        )
    raise ValidationError(f"{path.name}: 検証失敗: {status}")


def validate_onnx_files(paths: list[Path]) -> list[TaskValidation]:
    """Validate every task ONNX, collecting hard failures into one error.

    Soft (locally-unscorable) tasks are included with `scorable=False`. Only hard
    failures (size / banned-op / unloadable) raise :class:`ValidationError`.
    """
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
