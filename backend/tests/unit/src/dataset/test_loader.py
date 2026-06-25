"""Tests for ARC-AGI task loading."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dataset.loader import (
    TaskLoadError,
    available_task_ids,
    load_task,
    task_filename,
)

_TASK = {
    "train": [{"input": [[1]], "output": [[2]]}],
    "test": [{"input": [[3]], "output": [[4]]}],
    "arc-gen": [{"input": [[5]], "output": [[6]]}, {"input": [[7]], "output": [[8]]}],
}


def _write_task(task_dir: Path, task_id: int, data: dict[str, object]) -> None:
    (task_dir / task_filename(task_id)).write_text(json.dumps(data))


def test_task_filename_zero_pads() -> None:
    assert task_filename(7) == "task007.json"


def test_load_task_parses_all_subsets(tmp_path: Path) -> None:
    _write_task(tmp_path, 1, _TASK)
    task = load_task(tmp_path, 1)
    assert task.task_id == 1
    assert len(task.train) == 1
    assert len(task.test) == 1
    assert len(task.arc_gen) == 2
    assert len(task.all_examples()) == 4
    assert task.train[0].input == [[1]]
    assert task.train[0].output == [[2]]


def test_as_scorer_dict_uses_hyphen_key(tmp_path: Path) -> None:
    _write_task(tmp_path, 1, _TASK)
    d = load_task(tmp_path, 1).as_scorer_dict()
    assert set(d) == {"train", "test", "arc-gen"}
    assert d["arc-gen"][0] == {"input": [[5]], "output": [[6]]}


def test_missing_subset_defaults_empty(tmp_path: Path) -> None:
    _write_task(tmp_path, 2, {"train": [{"input": [[0]], "output": [[1]]}]})
    task = load_task(tmp_path, 2)
    assert task.test == ()
    assert task.arc_gen == ()


def test_load_task_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(TaskLoadError):
        load_task(tmp_path, 99)


def test_available_task_ids(tmp_path: Path) -> None:
    _write_task(tmp_path, 3, _TASK)
    _write_task(tmp_path, 1, _TASK)
    assert available_task_ids(tmp_path) == [1, 3]
