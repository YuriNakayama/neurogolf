"""パネル合成 solver: 入力を 2 分割し、論理演算で出力パネルを作る族。

ARC 頻出: 入力が「左右 or 上下 2 パネル（区切り列/行あり/なし）」で、出力は
両パネルのセル単位の論理合成（AND/OR/XOR/DIFF）を 1 色で塗ったもの。

出力サイズはパネル 1 枚ぶん。区切り（セパレータ列/行）の有無を含め全 example から
分割位置と論理演算・出力色を推定し、合致すれば小空間グラフで厳密構成する。

ONNX 構成（小空間 oh×ow）:
  L = Slice(left panel any-color mask),  R = Slice(right panel mask)
  combined = op(L, R)  ->  out color C where combined else 0
mask = 1 - input[:,0]（背景 0 以外）。op は加算/乗算/排他で表現。
"""

from __future__ import annotations

import numpy as np
import onnx
from onnx import TensorProto, helper

from .arc import NUM_COLORS, Example

_DTYPE = TensorProto.FLOAT
GRID = 30
GRID_SHAPE = [1, NUM_COLORS, GRID, GRID]

_OPS = ("or", "and", "xor", "diff")


def _combine(left: np.ndarray, right: np.ndarray, op: str) -> np.ndarray:
    lm = left != 0
    rm = right != 0
    if op == "or":
        out = lm | rm
    elif op == "and":
        out = lm & rm
    elif op == "xor":
        out = lm ^ rm
    else:  # diff
        out = lm & ~rm
    return np.asarray(out)


def detect_panels(
    examples: tuple[Example, ...],
) -> tuple[str, str, int, int, int] | None:
    """(layout, op, out_color, oh, ow) を推定。layout in {LR,LRsep,TB,TBsep}。"""
    # 出力サイズ一定が前提
    osz = {(len(e.output), len(e.output[0])) for e in examples}
    if len(osz) != 1:
        return None
    oh, ow = next(iter(osz))
    # 出力色一定（非ゼロは 1 色）
    colors: set[int] = set()
    for e in examples:
        b = np.array(e.output)
        colors.update(int(v) for v in np.unique(b) if v != 0)
    if len(colors) != 1:
        return None
    color = colors.pop()
    # レイアウト × op を総当たりで全 example 合致を探す
    layouts = ("LR", "LRsep", "TB", "TBsep")
    for layout in layouts:
        for op in _OPS:
            if _check(examples, layout, op, color, oh, ow):
                return layout, op, color, oh, ow
    return None


def _panels_for(
    a: np.ndarray, layout: str, oh: int, ow: int
) -> tuple[np.ndarray, np.ndarray] | None:
    ih, iw = a.shape
    if layout == "LR" and ih == oh and iw == 2 * ow:
        return a[:, :ow], a[:, ow:]
    if layout == "LRsep" and ih == oh and iw == 2 * ow + 1:
        return a[:, :ow], a[:, ow + 1 :]
    if layout == "TB" and iw == ow and ih == 2 * oh:
        return a[:oh, :], a[oh:, :]
    if layout == "TBsep" and iw == ow and ih == 2 * oh + 1:
        return a[:oh, :], a[oh + 1 :, :]
    return None


def _check(
    examples: tuple[Example, ...], layout: str, op: str, color: int, oh: int, ow: int
) -> bool:
    for e in examples:
        a = np.array(e.input)
        b = np.array(e.output)
        pr = _panels_for(a, layout, oh, ow)
        if pr is None:
            return False
        comb = _combine(pr[0], pr[1], op)
        pred = np.where(comb, color, 0)
        if not np.array_equal(pred, b):
            return False
    return True


def _panel_slice(
    name_in: str, r0: int, c0: int, oh: int, ow: int, tag: str
) -> tuple[list[onnx.NodeProto], list[onnx.TensorProto]]:
    s = helper.make_tensor(f"s{tag}", TensorProto.INT64, [2], [r0, c0])
    e = helper.make_tensor(f"e{tag}", TensorProto.INT64, [2], [r0 + oh, c0 + ow])
    ax = helper.make_tensor(f"a{tag}", TensorProto.INT64, [2], [2, 3])
    node = helper.make_node(
        "Slice", [name_in, f"s{tag}", f"e{tag}", f"a{tag}"], [f"p{tag}"]
    )
    return [node], [s, e, ax]


def build_panels(examples: tuple[Example, ...]) -> onnx.ModelProto | None:
    det = detect_panels(examples)
    if det is None:
        return None
    layout, op, color, oh, ow = det
    # パネルのオフセット
    if layout == "LR":
        off2 = (0, ow)
    elif layout == "LRsep":
        off2 = (0, ow + 1)
    elif layout == "TB":
        off2 = (oh, 0)
    else:
        off2 = (oh + 1, 0)

    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []
    # 各パネルの「非背景マスク」= 1 - input[:,0]。まず channel0 を slice したいが、
    # パネルは空間 slice。各パネルで全 10ch を slice し、background ch0 の補数でマスク化。
    for tag, (r0, c0) in (("L", (0, 0)), ("R", off2)):
        ns, ins = _panel_slice("input", r0, c0, oh, ow, tag)
        nodes += ns
        inits += ins
        # mask = 1 - p[:,0:1]
        s0 = helper.make_tensor(f"z0s{tag}", TensorProto.INT64, [1], [0])
        e0 = helper.make_tensor(f"z0e{tag}", TensorProto.INT64, [1], [1])
        a0 = helper.make_tensor(f"z0a{tag}", TensorProto.INT64, [1], [1])
        inits += [s0, e0, a0]
        nodes.append(
            helper.make_node(
                "Slice",
                [f"p{tag}", f"z0s{tag}", f"z0e{tag}", f"z0a{tag}"],
                [f"bg{tag}"],
            )
        )
        one = helper.make_tensor(f"one{tag}", _DTYPE, [1, 1, 1, 1], [1.0])
        inits.append(one)
        nodes.append(helper.make_node("Sub", [f"one{tag}", f"bg{tag}"], [f"m{tag}"]))

    # combine masks (mL, mR in {0,1})
    if op == "or":
        # or = max(mL,mR)
        nodes.append(helper.make_node("Max", ["mL", "mR"], ["comb"]))
    elif op == "and":
        nodes.append(helper.make_node("Mul", ["mL", "mR"], ["comb"]))
    elif op == "xor":
        # xor = |mL - mR|
        nodes.append(helper.make_node("Sub", ["mL", "mR"], ["d"]))
        nodes.append(helper.make_node("Abs", ["d"], ["comb"]))
    else:  # diff = mL * (1 - mR)
        oneD = helper.make_tensor("oneD", _DTYPE, [1, 1, 1, 1], [1.0])
        inits.append(oneD)
        nodes.append(helper.make_node("Sub", ["oneD", "mR"], ["notR"]))
        nodes.append(helper.make_node("Mul", ["mL", "notR"], ["comb"]))

    # to 10ch one-hot: channel `color`=comb, channel 0 = 1-comb（出力域内の背景は ch0=1）。
    # Conv 1x1 (1->10) with bias: out[c] = W[c]*comb + B[c].
    #   color: W=1, B=0  -> comb
    #   0    : W=-1, B=1 -> 1-comb
    w = np.zeros((NUM_COLORS, 1, 1, 1), dtype=np.float32)
    b = np.zeros((NUM_COLORS,), dtype=np.float32)
    w[color, 0, 0, 0] = 1.0
    if color != 0:
        w[0, 0, 0, 0] = -1.0
        b[0] = 1.0
    inits.append(helper.make_tensor("Wp", _DTYPE, [NUM_COLORS, 1, 1, 1], w.flatten()))
    inits.append(helper.make_tensor("Bp", _DTYPE, [NUM_COLORS], b.flatten()))
    nodes.append(
        helper.make_node(
            "Conv",
            ["comb", "Wp", "Bp"],
            ["small"],
            kernel_shape=[1, 1],
            pads=[0, 0, 0, 0],
        )
    )
    # pad small [1,10,oh,ow] -> [1,10,30,30]
    nodes.append(
        helper.make_node(
            "Pad",
            ["small"],
            ["output"],
            mode="constant",
            pads=[0, 0, 0, 0, 0, 0, GRID - oh, GRID - ow],
            value=0.0,
        )
    )

    x = helper.make_tensor_value_info("input", _DTYPE, GRID_SHAPE)
    y = helper.make_tensor_value_info("output", _DTYPE, GRID_SHAPE)
    graph = helper.make_graph(nodes, "panels", [x], [y], inits)
    return helper.make_model(
        graph, ir_version=10, opset_imports=[helper.make_opsetid("", 10)]
    )
