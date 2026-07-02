"""run() のタスクフィルタ動作を検証するユニットテスト（TDD: RED → GREEN）。"""

from __future__ import annotations

import json
from pathlib import Path

from pipeline.case3.run import run


def _write_task(task_dir: Path, num: int, inp: list[list[int]], out: list[list[int]]) -> None:
    data = {
        "train": [{"input": inp, "output": out}],
        "test": [],
        "arc-gen": [],
    }
    (task_dir / f"task{num:03d}.json").write_text(json.dumps(data))


def test_run_task_filter_runs_only_specified_tasks(tmp_path: Path) -> None:
    """tasks=[1] を渡すと task001 のみ処理し task002 は生成しない。"""
    task_dir = tmp_path / "tasks"
    task_dir.mkdir()
    out_dir = tmp_path / "out"

    g = [[1, 2], [3, 4]]
    _write_task(task_dir, 1, g, g)  # identity → 必ず solve
    _write_task(task_dir, 2, g, g)  # identity だが tasks=[1] で除外

    solved = run(task_dir, out_dir, tasks=[1])

    assert len(solved) == 1
    assert solved[0].task == 1
    assert (out_dir / "task001.onnx").is_file()
    assert not (out_dir / "task002.onnx").exists()


def test_run_no_filter_runs_all_listed_tasks(tmp_path: Path) -> None:
    """tasks=None (デフォルト) は指定したタスク番号すべてを処理する。"""
    task_dir = tmp_path / "tasks"
    task_dir.mkdir()
    out_dir = tmp_path / "out"

    g = [[1, 2], [3, 4]]
    _write_task(task_dir, 1, g, g)
    _write_task(task_dir, 2, g, g)

    solved = run(task_dir, out_dir, tasks=[1, 2])

    tasks_solved = {s.task for s in solved}
    assert tasks_solved == {1, 2}
