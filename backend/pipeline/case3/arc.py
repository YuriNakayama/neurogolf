"""ARC task loading + grid encoding for case3 (self-contained).

case 独立の原則に従い ``src/`` も他 case も import しない。公式
``neurogolf_utils.convert_to_numpy`` と byte-for-byte 整合な encode を持つ:
グリッドを ``[1, 10, 30, 30]`` の one-hot テンソルにし、枠外は zero-hot。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

NUM_COLORS = 10
GRID_MAX = 30
GRID_SHAPE = (1, NUM_COLORS, GRID_MAX, GRID_MAX)

Grid = list[list[int]]


@dataclass(frozen=True)
class Example:
    """1 つの入出力グリッドペア。"""

    input: Grid
    output: Grid


@dataclass(frozen=True)
class Task:
    """1 タスク分の全 example（train / test / arc-gen）。"""

    num: int
    train: tuple[Example, ...]
    test: tuple[Example, ...]
    arc_gen: tuple[Example, ...]

    def all_examples(self) -> tuple[Example, ...]:
        return self.train + self.test + self.arc_gen

    def valid_examples(self) -> tuple[Example, ...]:
        """30x30 を超えるグリッドを含まない example のみ（公式は >30 を無視）。"""
        return tuple(e for e in self.all_examples() if _within_bounds(e))


def _within_bounds(ex: Example) -> bool:
    for g in (ex.input, ex.output):
        if not g or max(len(g), len(g[0])) > GRID_MAX:
            return False
    return True


def _to_examples(raw: list[dict[str, Grid]]) -> tuple[Example, ...]:
    return tuple(Example(input=e["input"], output=e["output"]) for e in raw)


def load_task(task_dir: Path, num: int) -> Task:
    """``task{num:03d}.json`` を読み込む。"""
    path = task_dir / f"task{num:03d}.json"
    raw = json.loads(path.read_text())
    return Task(
        num=num,
        train=_to_examples(raw.get("train", [])),
        test=_to_examples(raw.get("test", [])),
        arc_gen=_to_examples(raw.get("arc-gen", [])),
    )


def encode_grid(grid: Grid) -> np.ndarray:
    """グリッド → ``[1, 10, 30, 30]`` float32 one-hot（公式 convert_to_numpy 同等）。"""
    arr = np.zeros(GRID_SHAPE, dtype=np.float32)
    for r, row in enumerate(grid):
        for c, color in enumerate(row):
            arr[0, color, r, c] = 1.0
    return arr


def decode_grid(arr: np.ndarray, height: int, width: int) -> Grid:
    """``[1, 10, 30, 30]`` (>0 でしきい値化済み) → height x width グリッド。"""
    out: Grid = []
    for r in range(height):
        row: list[int] = []
        for c in range(width):
            colors = [ch for ch in range(NUM_COLORS) if arr[0, ch, r, c] > 0.0]
            row.append(colors[0] if len(colors) == 1 else 0)
        out.append(row)
    return out


def grid_shape(grid: Grid) -> tuple[int, int]:
    return len(grid), len(grid[0])
