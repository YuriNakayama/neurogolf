"""enclosed-region 塗りつぶし solver（unrolled flood-fill）。

ARC 頻出: 「線（色 L）で囲まれた背景（色 0）の内側を色 F で塗る」。flood-fill は
本来反復だが ``Loop``/``Scan`` 禁止なので、**固定回数のダイレーション層に展開**する。

アルゴリズム（全 30×30 空間、枠外は 0=背景なので外周と地続き）:
1. free = (input==0)            # 塗れる背景
2. reach0 = free を外周シードから 4 連結ダイレーションで N 回伝播（線で遮断）
   reach = free AND (border or 任意の隣接 reach)。MaxPool(3x3,stride1) で近傍 OR、
   free でマスク、を N 回。
3. enclosed = free AND NOT reach      # 外周に届かない背景 = 囲まれた内側
4. output = input - 0ch(enclosed) + Fch(enclosed)

memory を抑えるため reach は 1 チャネル [1,1,30,30]。N 層ぶんの中間が乗るので、
N は「実際に伝播が必要な最大距離」に抑える（タスクの最大グリッド辺長で十分）。
"""

from __future__ import annotations

import numpy as np
import onnx
from onnx import TensorProto, helper

from .arc import NUM_COLORS, Example

_DTYPE = TensorProto.FLOAT
GRID = 30
GRID_SHAPE = [1, NUM_COLORS, GRID, GRID]


def _detect(examples: tuple[Example, ...]) -> tuple[int, int] | None:
    """(line_color, fill_color) を全 example から推定。flood-fill 規則に合致すれば返す。

    規則: out==in の所はそのまま。変化セルは「in==0 かつ 4 連結で外周に届かない」セルで、
    全て同一の fill_color に変わる。線色は変化しない非ゼロ色。
    """
    fill_colors: set[int] = set()
    for e in examples:
        a = np.array(e.input, dtype=np.int64)
        b = np.array(e.output, dtype=np.int64)
        if a.shape != b.shape:
            return None
        diff = a != b
        if not diff.any():
            continue
        if not np.all(a[diff] == 0):  # 変化前は必ず背景 0
            return None
        fill_colors.update(b[diff].tolist())
    if len(fill_colors) != 1:
        return None
    fill = fill_colors.pop()
    # 検証: 各 example で「enclosed(in==0 & not border-reachable) == 変化セル」か
    line_for = _infer_line_color(examples, fill)
    if line_for is None:
        return None
    return line_for, fill


def _border_reachable(free: np.ndarray) -> np.ndarray:
    """free(bool) の外周シードから 4 連結で到達可能な集合を BFS で求める。"""
    h, w = free.shape
    reach = np.zeros_like(free)
    from collections import deque

    dq: deque[tuple[int, int]] = deque()
    for r in range(h):
        for c in (0, w - 1):
            if free[r, c] and not reach[r, c]:
                reach[r, c] = True
                dq.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if free[r, c] and not reach[r, c]:
                reach[r, c] = True
                dq.append((r, c))
    while dq:
        r, c = dq.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and free[nr, nc] and not reach[nr, nc]:
                reach[nr, nc] = True
                dq.append((nr, nc))
    return reach


def _infer_line_color(examples: tuple[Example, ...], fill: int) -> int | None:
    """flood-fill 規則（enclosed==変化セル）が全 example で成立するか検証。

    成立すれば「囲い線」を成す非ゼロ色（複数なら任意/混在可）を 1 つ返す。規則が
    崩れたら None。実際の塗り判定はグリッド全体の連結性で行うので line_color は
    実装上不要だが、検証のために確認する。
    """
    for e in examples:
        a = np.array(e.input, dtype=np.int64)
        b = np.array(e.output, dtype=np.int64)
        free = a == 0
        reach = _border_reachable(free)
        enclosed = free & ~reach
        expected = a.copy()
        expected[enclosed] = fill
        if not np.array_equal(expected, b):
            return None
    return 0  # マーカー（line 色は単一とは限らないため 0 を返す）


def _max_dim(examples: tuple[Example, ...]) -> int:
    return max(max(len(e.input), len(e.input[0])) for e in examples)


def build_floodfill(
    examples: tuple[Example, ...], n_steps: int | None = None
) -> onnx.ModelProto | None:
    """enclosed 領域を fill 色で塗る unrolled flood-fill ネットを構成。解けねば None。

    4 連結ダイレーション（十字 = 縦 MaxPool(3x1) と横 MaxPool(1x3) の Max）を、内容が
    収まる小空間 [m×m] で n_steps 回展開する。reach は 1ch なので中間は m×m 要素。
    枠外（30×30 の m より外）は背景 0 = 外周に地続きなので enclosed にならない。
    """
    det = _detect(examples)
    if det is None:
        return None
    _line, fill = det
    m = _max_dim(examples)
    if m < 3:
        return None
    # 伝播に必要な最大ステップは内容の周長程度。安全に 2*m とする。
    steps = n_steps if n_steps is not None else max(8, 2 * m)

    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    # free = input channel 0, 小空間 [1,1,m,m] に Slice
    inits += [
        helper.make_tensor("c0s", TensorProto.INT64, [3], [0, 0, 0]),
        helper.make_tensor("c0e", TensorProto.INT64, [3], [1, m, m]),
        helper.make_tensor("c0a", TensorProto.INT64, [3], [1, 2, 3]),
    ]
    nodes.append(helper.make_node("Slice", ["input", "c0s", "c0e", "c0a"], ["free"]))

    # border seed: free * border_mask (m×m の外周)
    bm = np.zeros((1, 1, m, m), dtype=np.float32)
    bm[0, 0, 0, :] = 1
    bm[0, 0, -1, :] = 1
    bm[0, 0, :, 0] = 1
    bm[0, 0, :, -1] = 1
    inits.append(helper.make_tensor("bmask", _DTYPE, [1, 1, m, m], bm.flatten()))
    nodes.append(helper.make_node("Mul", ["free", "bmask"], ["reach0"]))

    # iterate: reach_{k+1} = free * dilate4(reach_k)
    # dilate4 = Max( MaxPool(3x1), MaxPool(1x3) ) で十字 = 4 連結
    cur = "reach0"
    for k in range(steps):
        v = f"v{k}"
        h = f"h{k}"
        nodes.append(
            helper.make_node(
                "MaxPool",
                [cur],
                [v],
                kernel_shape=[3, 1],
                pads=[1, 0, 1, 0],
                strides=[1, 1],
            )
        )
        nodes.append(
            helper.make_node(
                "MaxPool",
                [cur],
                [h],
                kernel_shape=[1, 3],
                pads=[0, 1, 0, 1],
                strides=[1, 1],
            )
        )
        cross = f"cr{k}"
        nodes.append(helper.make_node("Max", [v, h], [cross]))
        nxt = f"reach{k + 1}"
        nodes.append(helper.make_node("Mul", [cross, "free"], [nxt]))
        cur = nxt

    # enclosed = relu(free - reach)
    nodes.append(helper.make_node("Sub", ["free", cur], ["encs"]))
    nodes.append(helper.make_node("Relu", ["encs"], ["enc"]))

    # delta(小空間,10ch): -enc on ch0, +enc on ch fill
    w = np.zeros((NUM_COLORS, 1, 1, 1), dtype=np.float32)
    w[0, 0, 0, 0] = -1.0
    w[fill, 0, 0, 0] += 1.0
    inits.append(helper.make_tensor("Wd", _DTYPE, [NUM_COLORS, 1, 1, 1], w.flatten()))
    nodes.append(
        helper.make_node(
            "Conv", ["enc", "Wd"], ["dsmall"], kernel_shape=[1, 1], pads=[0, 0, 0, 0]
        )
    )
    # pad delta [1,10,m,m] -> [1,10,30,30]
    nodes.append(
        helper.make_node(
            "Pad",
            ["dsmall"],
            ["delta"],
            mode="constant",
            pads=[0, 0, 0, 0, 0, 0, GRID - m, GRID - m],
            value=0.0,
        )
    )
    nodes.append(helper.make_node("Add", ["input", "delta"], ["output"]))

    x = helper.make_tensor_value_info("input", _DTYPE, GRID_SHAPE)
    y = helper.make_tensor_value_info("output", _DTYPE, GRID_SHAPE)
    graph = helper.make_graph(nodes, "floodfill", [x], [y], inits)
    return helper.make_model(
        graph, ir_version=10, opset_imports=[helper.make_opsetid("", 10)]
    )
