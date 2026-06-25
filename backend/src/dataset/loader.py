"""Load ARC-AGI tasks from JSON.

Each task ships as ``task{NNN}.json`` with ``train`` / ``test`` / ``arc-gen``
example lists (note the hyphen in ``arc-gen``). The loader takes the directory
holding those files and the 1-based task id.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dataset.types import Example, Task

TASK_MIN = 1
TASK_MAX = 400


class TaskLoadError(RuntimeError):
    """Raised when a task JSON is missing or malformed."""


def task_filename(task_id: int) -> str:
    """Filename for a task id, e.g. ``7 -> 'task007.json'``."""
    return f"task{task_id:03d}.json"


def _parse_examples(raw: list[dict[str, Any]]) -> tuple[Example, ...]:
    return tuple(Example(input=ex["input"], output=ex["output"]) for ex in raw)


def load_task(task_dir: Path, task_id: int) -> Task:
    """Load one task; raise :class:`TaskLoadError` if missing or malformed."""
    path = task_dir / task_filename(task_id)
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise TaskLoadError(f"タスク JSON が見つかりません: {path}") from exc
    except json.JSONDecodeError as exc:
        raise TaskLoadError(f"タスク JSON の解析に失敗: {path}: {exc}") from exc
    try:
        return Task(
            task_id=task_id,
            train=_parse_examples(data.get("train", [])),
            test=_parse_examples(data.get("test", [])),
            arc_gen=_parse_examples(data.get("arc-gen", [])),
        )
    except (KeyError, TypeError) as exc:
        raise TaskLoadError(f"タスク JSON の形式が不正: {path}: {exc}") from exc


def available_task_ids(task_dir: Path) -> list[int]:
    """Task ids (1-400) whose JSON exists under ``task_dir``, sorted."""
    return [
        t
        for t in range(TASK_MIN, TASK_MAX + 1)
        if (task_dir / task_filename(t)).is_file()
    ]


def load_tasks(task_dir: Path) -> list[Task]:
    """Load every available task under ``task_dir``."""
    return [load_task(task_dir, t) for t in available_task_ids(task_dir)]
