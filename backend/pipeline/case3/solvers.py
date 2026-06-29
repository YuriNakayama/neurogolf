"""Solver bank: タスクごとに候補 ONNX を生成する builder 群。

各 solver は ``Task`` を受け取り、適用可能なら ``onnx.ModelProto`` を返す（不可なら
None）。正答性は呼び出し側が ``audit_one`` で全 example 厳密検証するので、ここでは
「train から推定した変換を ONNX 化する」ことに専念する。cost 最小が目的。
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import onnx

from . import builders as B
from .arc import GRID_MAX, NUM_COLORS, Example, Task, grid_shape
from .floodfill import build_floodfill_8conn
from .lookup import _build_table
from .residual import build_residual
from .smalllookup import build_small_lookup

Solver = Callable[[Task], onnx.ModelProto | None]


def _np(grid: list[list[int]]) -> np.ndarray:
    return np.array(grid, dtype=np.int64)


def _all(task: Task, pred: Callable[[Example], bool]) -> bool:
    exs = task.valid_examples()
    return bool(exs) and all(pred(e) for e in exs)


def _same_shape(e: Example) -> bool:
    return (len(e.input), len(e.input[0])) == (len(e.output), len(e.output[0]))


def _const_dim(task: Task, *, axis: int) -> int | None:
    """全 example で入力グリッドの ``axis`` 次元が一定なら返す。"""
    dims = {grid_shape(e.input)[axis] for e in task.valid_examples()}
    return next(iter(dims)) if len(dims) == 1 else None


# --- geometric / trivial -----------------------------------------------------
def solve_identity(task: Task) -> onnx.ModelProto | None:
    if _all(task, lambda e: e.input == e.output):
        return B.build_identity()
    return None


def solve_flip_v(task: Task) -> onnx.ModelProto | None:
    h = _const_dim(task, axis=0)
    if h is None:
        return None
    if _all(
        task,
        lambda e: (
            _same_shape(e) and _np(e.output).tolist() == _np(e.input)[::-1].tolist()
        ),
    ):
        return B.build_flip(h, 2)
    return None


def solve_flip_h(task: Task) -> onnx.ModelProto | None:
    w = _const_dim(task, axis=1)
    if w is None:
        return None
    if _all(
        task,
        lambda e: (
            _same_shape(e) and _np(e.output).tolist() == _np(e.input)[:, ::-1].tolist()
        ),
    ):
        return B.build_flip(w, 3)
    return None


def solve_rot180(task: Task) -> onnx.ModelProto | None:
    h = _const_dim(task, axis=0)
    w = _const_dim(task, axis=1)
    if h is None or w is None:
        return None
    if _all(
        task,
        lambda e: (
            _same_shape(e)
            and _np(e.output).tolist() == _np(e.input)[::-1, ::-1].tolist()
        ),
    ):
        return B.build_rot180(h, w)
    return None


def solve_transpose(task: Task) -> onnx.ModelProto | None:
    if _all(
        task,
        lambda e: (
            len(e.input) == len(e.input[0])
            and _np(e.output).tolist() == _np(e.input).T.tolist()
        ),
    ):
        return B.build_permute_axes((0, 1, 3, 2))
    return None


def solve_rot90(task: Task) -> onnx.ModelProto | None:
    """rot90 CW: output[r][c] = input[h-1-c][r]。"""
    h = _const_dim(task, axis=0)
    if h is None:
        return None
    if _all(
        task,
        lambda e: _np(e.output).tolist() == np.rot90(_np(e.input), k=-1).tolist(),
    ):
        return B.build_rot90(h)
    return None


def solve_rot270(task: Task) -> onnx.ModelProto | None:
    """rot270 CCW: output[r][c] = input[c][w-1-r]。"""
    w = _const_dim(task, axis=1)
    if w is None:
        return None
    if _all(
        task,
        lambda e: _np(e.output).tolist() == np.rot90(_np(e.input), k=1).tolist(),
    ):
        return B.build_rot270(w)
    return None


# --- recolor (global color map) ---------------------------------------------
def _recolor_mapping(task: Task) -> dict[int, int] | None:
    if not _all(task, _same_shape):
        return None
    mapping: dict[int, int] = {}
    for e in task.valid_examples():
        a, b = _np(e.input).flatten(), _np(e.output).flatten()
        for iv, ov in zip(a.tolist(), b.tolist(), strict=True):
            if iv in mapping and mapping[iv] != ov:
                return None
            mapping[iv] = ov
    for c in range(NUM_COLORS):
        mapping.setdefault(c, c)
    return mapping


def solve_recolor_gather(task: Task) -> onnx.ModelProto | None:
    """単射 recolor なら Gather(axis=1)（params=10）で最安。"""
    mapping = _recolor_mapping(task)
    if mapping is None:
        return None
    if len(set(mapping.values())) != NUM_COLORS:  # 非単射は Gather 不可
        return None
    return B.build_recolor_gather(mapping)


def solve_recolor(task: Task) -> onnx.ModelProto | None:
    mapping = _recolor_mapping(task)
    if mapping is None:
        return None
    return B.build_recolor(mapping)


# --- constant output (output identical across all examples) ------------------
def solve_constant(task: Task) -> onnx.ModelProto | None:
    outs = {tuple(map(tuple, e.output)) for e in task.valid_examples()}
    if len(outs) != 1:
        return None
    grid = task.valid_examples()[0].output
    return _build_constant(grid)


def _build_constant(grid: list[list[int]]) -> onnx.ModelProto:
    """input を無視して定数グリッドを出力。Mul で 0 にして Add で定数を足す代わりに、
    Identity 経由でなく定数テンソルを直接 output へ（Constant ノード）。

    output one-hot を直接 initializer 化すると memory に乗らない（output 名は除外）。
    """
    from onnx import helper

    oh = np.zeros(B.GRID_SHAPE, dtype=np.float32)
    for r, row in enumerate(grid):
        for c, color in enumerate(row):
            oh[0, color, r, c] = 1.0
    const = helper.make_node(
        "Constant",
        [],
        ["output"],
        value=helper.make_tensor("c", B._DTYPE, B.GRID_SHAPE, oh.flatten()),
    )
    return B._model([const], [])


# --- tiling -------------------------------------------------------------------
def solve_tile(task: Task) -> onnx.ModelProto | None:
    """出力が入力グリッドの整数倍タイリングであれば ONNX を構築。

    Slice + Tile + Pad の 3 ノード構成。cost = 12(params) + 40*h*w*(1+rh*rw) bytes。
    rh=rw=1 は identity に任せるため対象外。
    """
    exs = task.valid_examples()
    if not exs:
        return None
    in_shapes = {grid_shape(e.input) for e in exs}
    if len(in_shapes) != 1:
        return None
    h, w = next(iter(in_shapes))
    out_shapes = {grid_shape(e.output) for e in exs}
    if len(out_shapes) != 1:
        return None
    out_h, out_w = next(iter(out_shapes))
    if out_h % h != 0 or out_w % w != 0:
        return None
    rh, rw = out_h // h, out_w // w
    if rh == 1 and rw == 1:
        return None
    if out_h > GRID_MAX or out_w > GRID_MAX:
        return None
    for e in exs:
        if _np(e.output).tolist() != np.tile(_np(e.input), (rh, rw)).tolist():
            return None
    return B.build_tile(h, w, rh, rw)


# --- neighborhood lookup / residual (algorithmic, small-space) ---------------
def _min_k(task: Task) -> int | None:
    for k in (1, 3, 5):
        if _build_table(task.valid_examples(), k) is not None:
            return k
    return None


def solve_small_lookup(task: Task) -> onnx.ModelProto | None:
    k = _min_k(task)
    if k is None:
        return None
    return build_small_lookup(task.valid_examples(), k)


def solve_residual3(task: Task) -> onnx.ModelProto | None:
    return build_residual(task.valid_examples(), 3)


def solve_residual5(task: Task) -> onnx.ModelProto | None:
    return build_residual(task.valid_examples(), 5)


def solve_floodfill(task: Task) -> onnx.ModelProto | None:
    return build_floodfill_8conn(task.valid_examples())


# 適用順: cost が小さいものを先に（同点なら先勝ち）。検証側が cost 最小を選ぶので順序は目安。
SOLVERS: list[tuple[str, Solver]] = [
    ("identity", solve_identity),
    ("transpose", solve_transpose),
    ("flip_v", solve_flip_v),
    ("flip_h", solve_flip_h),
    ("rot180", solve_rot180),
    ("rot90", solve_rot90),
    ("rot270", solve_rot270),
    ("recolor_gather", solve_recolor_gather),
    ("recolor", solve_recolor),
    ("constant", solve_constant),
    ("tile", solve_tile),
    ("residual3", solve_residual3),
    ("residual5", solve_residual5),
    ("small_lookup", solve_small_lookup),
    ("floodfill", solve_floodfill),
]
