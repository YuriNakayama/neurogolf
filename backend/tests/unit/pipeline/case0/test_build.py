"""Tests for case0 bundle generation."""

from __future__ import annotations

import json
from pathlib import Path

from dataset.loader import load_task
from pipeline.case0.build import build_bundle, solve_task

_IDENTITY = {"train": [{"input": [[1, 2], [3, 4]], "output": [[1, 2], [3, 4]]}]}
_RECOLOR = {"train": [{"input": [[1, 2], [2, 1]], "output": [[2, 3], [3, 2]]}]}
_TRANSPOSE = {"train": [{"input": [[1, 2, 3], [4, 5, 6]], "output": [[1, 4], [2, 5], [3, 6]]}]}
_UNSOLVABLE = {"train": [{"input": [[1]], "output": [[2], [3]]}]}  # 1x1 -> 2x1


def _write(task_dir: Path, task_id: int, data: dict[str, object]) -> None:
    (task_dir / f"task{task_id:03d}.json").write_text(json.dumps(data))


def test_solve_task_picks_identity(tmp_path: Path) -> None:
    _write(tmp_path, 1, _IDENTITY)
    result = solve_task(load_task(tmp_path, 1))
    assert result is not None
    assert result[0] == "identity"


def test_solve_task_picks_recolor(tmp_path: Path) -> None:
    _write(tmp_path, 1, _RECOLOR)
    result = solve_task(load_task(tmp_path, 1))
    assert result is not None
    assert result[0] == "recolor"


def test_solve_task_picks_transpose(tmp_path: Path) -> None:
    _write(tmp_path, 1, _TRANSPOSE)
    result = solve_task(load_task(tmp_path, 1))
    assert result is not None
    assert result[0] == "transpose"


def test_solve_task_returns_none_when_unsolvable(tmp_path: Path) -> None:
    _write(tmp_path, 1, _UNSOLVABLE)
    assert solve_task(load_task(tmp_path, 1)) is None


def test_build_bundle_writes_only_solved(tmp_path: Path) -> None:
    src = tmp_path / "tasks"
    src.mkdir()
    _write(src, 1, _IDENTITY)
    _write(src, 2, _UNSOLVABLE)
    out = tmp_path / "onnx"
    summary = build_bundle(src, out)
    assert (out / "task001.onnx").is_file()
    assert not (out / "task002.onnx").exists()
    assert len(summary.solved) == 1
    assert summary.total_points > 0
