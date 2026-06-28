"""近傍ルックアップ solver: 出力が k×k 入力近傍の決定的関数なタスクを厳密に解く。

構成（2 段、中間テンソル 1 個）:
1. ``Conv`` (k×k, in=10ch, out=P ch): 各ユニーク近傍パターン p の **完全一致検出器**。
   重み ``W1[p,i,dr,dc] = 1 if pattern_p[dr,dc]==i else 0``、bias ``-(match_count_p - 0.5)``
   とすると、score_p(pixel) = (一致セル数) - match_count_p + 0.5 で、完全一致のとき
   +0.5(>0)、1 セルでも違えば <= -0.5。ReLU で 0/正に。
2. ``Conv`` (1x1, in=P ch, out=10ch): パターン p -> 出力色 o_p のワンホット写像。
   ``W2[o,p,0,0] = 1 if o_p==o else 0``。

枠外は zero-hot（全 ch 0）なので、近傍に枠外(-1)を含むパターンは「該当 ch すべて 0」
で表現する（one-hot に -1 チャネルは無い → 枠外セルは「全 10ch が 0」を要求）。これを
検出するには「そのセルでどの色も立っていない」= sum_i input[i]==0 を使う必要があるが、
完全一致検出器の重みを 1（該当色）とし、枠外セルには寄与 0 とすることで、match_count を
「枠内かつ色一致セル数」で数え、ちょうどその数で threshold すれば一致を判定できる。
"""

from __future__ import annotations

import numpy as np
import onnx
from onnx import TensorProto, helper

from .arc import NUM_COLORS, Example

GRID_SHAPE = [1, NUM_COLORS, 30, 30]
_DTYPE = TensorProto.FLOAT


def _build_table(examples: tuple[Example, ...], k: int) -> dict[bytes, int] | None:
    pad = k // 2
    table: dict[bytes, int] = {}
    for e in examples:
        a = np.array(e.input, dtype=np.int64)
        b = np.array(e.output, dtype=np.int64)
        if a.shape != b.shape:
            return None
        h, w = a.shape
        ap = np.full((h + 2 * pad, w + 2 * pad), -1, dtype=np.int64)
        ap[pad : pad + h, pad : pad + w] = a
        for r in range(h):
            for c in range(w):
                key = ap[r : r + k, c : c + k].tobytes()
                if key in table and table[key] != b[r, c]:
                    return None
                table[key] = int(b[r, c])
    return table


def detector_weights(
    pat_grids: list[np.ndarray], k: int
) -> tuple[np.ndarray, np.ndarray]:
    """各パターンの完全一致検出器の Conv 重み [P,10,k,k] と bias [P] を構成。

    score_p(pixel) = Σ_{枠内セル} [色一致] − Σ_{枠外指定セル} [何か色がある]
    完全一致（全枠内セル一致 かつ 枠外指定セルが本当に枠外=全 ch 0）のとき score=match_count、
    それ以外は < match_count。bias = −(match_count−1) で、Relu 後は完全一致のみ +1。

    枠外指定セル（pat==−1）に **全 10ch −1** の重みを置くことで、「そこに色があれば減点」。
    これにより「より枠外の多い部分パターン」が部分集合として誤発火するのを防ぐ。
    """
    p = len(pat_grids)
    w = np.zeros((p, NUM_COLORS, k, k), dtype=np.float32)
    match_count = np.zeros(p, dtype=np.float32)
    for pi, g in enumerate(pat_grids):
        for dr in range(k):
            for dc in range(k):
                col = int(g[dr, dc])
                if col >= 0:
                    w[pi, col, dr, dc] = 1.0
                    match_count[pi] += 1.0
                else:
                    w[pi, :, dr, dc] = -1.0  # 枠外指定: 何か色があれば減点
    b = -(match_count - 1.0)
    return w, b


def build_lookup(examples: tuple[Example, ...], k: int) -> onnx.ModelProto | None:
    """近傍ルックアップを 2 層 Conv で厳密に構成。解けなければ None。"""
    table = _build_table(examples, k)
    if table is None:
        return None
    patterns = list(table.items())  # (key bytes, out_color)
    p = len(patterns)
    pad = k // 2

    # パターン配列を復元: key -> (k,k) int grid（-1=枠外）
    pat_grids = [
        np.frombuffer(key, dtype=np.int64).reshape(k, k) for key, _ in patterns
    ]
    out_colors = [oc for _, oc in patterns]

    w1, b1 = detector_weights(pat_grids, k)

    # W2[o, p, 0, 0] = 1 if out_color_p == o
    w2 = np.zeros((NUM_COLORS, p, 1, 1), dtype=np.float32)
    for pi, oc in enumerate(out_colors):
        w2[oc, pi, 0, 0] = 1.0

    w1_t = helper.make_tensor("W1", _DTYPE, [p, NUM_COLORS, k, k], w1.flatten())
    b1_t = helper.make_tensor("B1", _DTYPE, [p], b1.flatten())
    w2_t = helper.make_tensor("W2", _DTYPE, [NUM_COLORS, p, 1, 1], w2.flatten())

    conv1 = helper.make_node(
        "Conv", ["input", "W1", "B1"], ["h0"], kernel_shape=[k, k], pads=[pad] * 4
    )
    relu = helper.make_node("Relu", ["h0"], ["h1"])
    conv2 = helper.make_node(
        "Conv", ["h1", "W2"], ["output"], kernel_shape=[1, 1], pads=[0, 0, 0, 0]
    )

    x = helper.make_tensor_value_info("input", _DTYPE, GRID_SHAPE)
    y = helper.make_tensor_value_info("output", _DTYPE, GRID_SHAPE)
    graph = helper.make_graph(
        [conv1, relu, conv2], "lookup", [x], [y], [w1_t, b1_t, w2_t]
    )
    return helper.make_model(
        graph, ir_version=10, opset_imports=[helper.make_opsetid("", 10)]
    )
