"""残差ルックアップ solver: output = input + (局所的に決まる変更分)。

スパース編集タスク（出力≒入力、数セルのみ変化）向け。**変化セルの近傍パターンのみ**を
検出器にすることで P（パターン数）を小さく抑え、memory を削る。

構成（中間 1 個 = P×30×30 の検出器出力。P 小ならば安価）:
1. ``Conv`` (k×k, 10->P): 各「変化を起こす近傍パターン」の完全一致検出器（``detector_weights``）。
2. ``Conv`` (1x1, P->10): 検出器 p -> 出力色 o_p の **差分** ワンホット。
   ただし「input を保ったまま上書き」するため、出力 = input をベースに、変化セルのみ
   検出器が新色を立て、旧色を消す必要がある。これを 1 本の線形和で表すのは難しいので、
   **出力 = max(input_masked, detector_color)** ではなく、各変化セルで「元色を引き、新色を足す」
   方式を採る:
       output = input + Σ_p relu(detect_p) * (onehot(new_p) - onehot(old_p))
   detect_p は変化セルでのみ 1。old_p はそのパターンの中心セル色。

検出器が **非変化セルで誤発火しない** ことが厳密性の鍵。``detector_weights`` の枠外ペナルティ
で部分集合誤発火は防がれるが、「変化パターンと同一近傍だが出力不変」のセルがあると ambiguous
になる（その場合 build は None を返す＝この solver は使えない）。
"""

from __future__ import annotations

import numpy as np
import onnx
from onnx import TensorProto, helper

from .arc import NUM_COLORS, Example
from .lookup import detector_weights

_DTYPE = TensorProto.FLOAT
GRID = 30
GRID_SHAPE = [1, NUM_COLORS, GRID, GRID]


def _change_table(
    examples: tuple[Example, ...], k: int
) -> tuple[dict[bytes, tuple[int, int]], bool] | None:
    """変化セルの近傍 -> (旧色, 新色)。非変化セルが同一近傍を持てば ambiguous。

    返り値: (table, ok)。table が None なら同一近傍で異なる結果（厳密化不能）。
    """
    pad = k // 2
    change: dict[bytes, tuple[int, int]] = {}
    seen_keep: dict[bytes, int] = {}
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
                if a[r, c] != b[r, c]:
                    val = (int(a[r, c]), int(b[r, c]))
                    if key in change and change[key] != val:
                        return None
                    change[key] = val
                else:
                    seen_keep[key] = int(a[r, c])
    # 変化パターンが非変化セルにも現れたら誤発火する -> 厳密化不能
    for key in change:
        if key in seen_keep:
            return None
    return change, True


def build_residual(examples: tuple[Example, ...], k: int) -> onnx.ModelProto | None:
    """残差 = (新色 - 旧色) を変化検出器で加える。厳密化不能なら None。"""
    res = _change_table(examples, k)
    if res is None:
        return None
    change, _ = res
    if not change:
        return None  # 変化なし = identity（別 solver が拾う）
    patterns = list(change.items())
    p = len(patterns)
    pad = k // 2
    pat_grids = [
        np.frombuffer(key, dtype=np.int64).reshape(k, k) for key, _ in patterns
    ]
    w1, b1 = detector_weights(pat_grids, k)

    # W2[o, p, 0, 0] = +1 if new_p==o, -1 if old_p==o (差分ワンホット)
    w2 = np.zeros((NUM_COLORS, p, 1, 1), dtype=np.float32)
    for pi, (_key, (old, new)) in enumerate(patterns):
        w2[new, pi, 0, 0] += 1.0
        w2[old, pi, 0, 0] -= 1.0

    w1_t = helper.make_tensor("W1", _DTYPE, [p, NUM_COLORS, k, k], w1.flatten())
    b1_t = helper.make_tensor("B1", _DTYPE, [p], b1.flatten())
    w2_t = helper.make_tensor("W2", _DTYPE, [NUM_COLORS, p, 1, 1], w2.flatten())

    conv1 = helper.make_node(
        "Conv", ["input", "W1", "B1"], ["d0"], kernel_shape=[k, k], pads=[pad] * 4
    )
    relu = helper.make_node("Relu", ["d0"], ["d1"])
    conv2 = helper.make_node(
        "Conv", ["d1", "W2"], ["delta"], kernel_shape=[1, 1], pads=[0, 0, 0, 0]
    )
    add = helper.make_node("Add", ["input", "delta"], ["output"])

    x = helper.make_tensor_value_info("input", _DTYPE, GRID_SHAPE)
    y = helper.make_tensor_value_info("output", _DTYPE, GRID_SHAPE)
    graph = helper.make_graph(
        [conv1, relu, conv2, add], "residual", [x], [y], [w1_t, b1_t, w2_t]
    )
    return helper.make_model(
        graph, ir_version=10, opset_imports=[helper.make_opsetid("", 10)]
    )
