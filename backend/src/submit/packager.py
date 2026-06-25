"""Pack per-task ONNX files into a Kaggle NeuroGolf `submission.zip`.

A submission is a flat zip containing at most one ONNX per task, named
`task001.onnx` … `task400.onnx` (docs/competition/abstract.md). This module
collects validated `taskNNN.onnx` files from a directory and zips them.
"""

from __future__ import annotations

import logging
import re
import zipfile
from collections.abc import Sequence
from pathlib import Path

logger = logging.getLogger(__name__)

SUBMISSION_NAME = "submission.zip"
MAX_TASKS = 400
_TASK_FILE_RE = re.compile(r"^task(\d{3})\.onnx$")


class PackagingError(RuntimeError):
    """Raised when a submission archive cannot be built."""


def _task_number(path: Path) -> int:
    match = _TASK_FILE_RE.match(path.name)
    if match is None:
        raise PackagingError(
            f"unexpected ONNX filename {path.name!r}; expected taskNNN.onnx"
        )
    return int(match.group(1))


def collect_onnx_files(onnx_dir: Path) -> list[Path]:
    """Return `taskNNN.onnx` files under `onnx_dir`, sorted by task number.

    Raises `PackagingError` if the directory is missing, has no task files, has
    duplicate task numbers, or has numbers outside `1..MAX_TASKS`.
    """
    onnx_dir = onnx_dir.resolve()
    if not onnx_dir.is_dir():
        raise PackagingError(f"ONNX ディレクトリが見つかりません: {onnx_dir}")

    files = sorted(p for p in onnx_dir.glob("task*.onnx") if p.is_file())
    if not files:
        raise PackagingError(f"taskNNN.onnx が 1 つもありません: {onnx_dir}")

    seen: dict[int, Path] = {}
    for path in files:
        number = _task_number(path)
        if not 1 <= number <= MAX_TASKS:
            raise PackagingError(
                f"task 番号 {number} が範囲外 (1..{MAX_TASKS}): {path.name}"
            )
        if number in seen:
            raise PackagingError(
                f"task{number:03d} が重複しています: {seen[number].name}, {path.name}"
            )
        seen[number] = path
    return [seen[n] for n in sorted(seen)]


def build_submission_zip(
    onnx_dir: Path,
    out_dir: Path,
    *,
    files: Sequence[Path] | None = None,
) -> Path:
    """Zip the task ONNX files in `onnx_dir` into `out_dir/submission.zip`.

    Args:
        onnx_dir: directory containing `taskNNN.onnx` files.
        out_dir: directory to write `submission.zip` into.
        files: optional pre-collected/validated file list (defaults to
            `collect_onnx_files(onnx_dir)`).

    Returns:
        Path to the written `submission.zip`.
    """
    members = list(files) if files is not None else collect_onnx_files(onnx_dir)
    if not members:
        raise PackagingError("提出する ONNX ファイルがありません")

    out_dir.mkdir(parents=True, exist_ok=True)
    archive_path = out_dir / SUBMISSION_NAME

    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in members:
            # Deterministic metadata: flat arcname, fixed timestamp.
            info = zipfile.ZipInfo(filename=path.name)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            zf.writestr(info, path.read_bytes())

    logger.info("built %s with %d task(s)", archive_path.name, len(members))
    return archive_path
