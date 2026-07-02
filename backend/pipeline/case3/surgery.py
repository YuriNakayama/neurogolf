"""グラフサージェリ: 既存バンドルの各 ONNX を意味不変のまま縮約して cost を削る。

seddiktrk "All Graph Surgeries" ノートブックの cost 削減パスを再実装したもの。
外部ライブラリ（onnxsim / onnxscript）に頼らず、``onnx`` の API のみで完結する
**決定的かつ意味保存**のパスだけを実装する:

- ``narrow_int64_to_int32``   : int64 初期化子を、全消費 op が int32 を受け付ける
  場合のみ int32 へ。
- ``cleanup``                 : 未使用初期化子の除去・完全一致初期化子の重複排除・
  Identity ノード除去（value_info も掃除）。
- ``index_surgery``           : tiny shape-index 初期化子の重複排除・Slice の冗長
  steps/axes 入力除去・default-axes hole-punch。
- ``broadcast_compress``      : 一様/ブロードキャスト軸の初期化子を rank 保存で縮約。
- ``conv1x1_to_gather``       : 10×10×1×1 permutation Conv を Gather(axis=1) へ。
- ``fp16_surgery``            : value-preserving 部分グラフを fp16 化（境界 Cast 挿入）。

**最重要の安全装置**: 各パスの出力は ``src/evaluate.audit_one`` で
**全 example の正答 (n_fail==0) かつ cost 減**を確認できた場合のみ採用する。
ローカル採点が Kaggle 実 LB と一致しないことが実証済みのため、検証できない
（または cost が減らない）場合は元ファイルをそのまま残す（タスクを落とさない）。
"""

from __future__ import annotations

import copy
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper

# --- int64 -> int32 narrowing -------------------------------------------------

INT32_SAFE_INPUTS: set[tuple[str, int]] = {
    ("Gather", 1),
    ("GatherElements", 1),
    ("ScatterElements", 1),
    ("OneHot", 0),
    ("Cast", 0),
    ("Add", 0),
    ("Add", 1),
    ("Sub", 0),
    ("Sub", 1),
    ("Mul", 0),
    ("Mul", 1),
    ("Div", 0),
    ("Div", 1),
    ("Mod", 0),
    ("Mod", 1),
    ("Equal", 0),
    ("Equal", 1),
    ("Less", 0),
    ("Less", 1),
    ("Greater", 0),
    ("Greater", 1),
    ("LessOrEqual", 0),
    ("LessOrEqual", 1),
    ("GreaterOrEqual", 0),
    ("GreaterOrEqual", 1),
    ("Clip", 0),
    ("Clip", 1),
    ("Clip", 2),
    ("Where", 1),
    ("Where", 2),
    ("Min", 0),
    ("Min", 1),
    ("Min", 2),
    ("Min", 3),
    ("Max", 0),
    ("Max", 1),
    ("Max", 2),
    ("Max", 3),
    ("Concat", 0),
    ("Concat", 1),
    ("Concat", 2),
    ("Concat", 3),
    ("Concat", 4),
    ("Concat", 5),
    ("Concat", 6),
    ("Concat", 7),
    ("Squeeze", 0),
    ("Unsqueeze", 0),
    ("ReduceSum", 0),
    ("ReduceMax", 0),
    ("ReduceMin", 0),
    ("ReduceMean", 0),
    ("ReduceProd", 0),
    ("Abs", 0),
    ("Neg", 0),
    ("Sign", 0),
    ("Flatten", 0),
}

INT64_REQUIRED: set[tuple[str, int]] = {
    ("Reshape", 1),
    ("Slice", 1),
    ("Slice", 2),
    ("Slice", 3),
    ("Slice", 4),
    ("Pad", 1),
    ("Tile", 1),
    ("Expand", 1),
    ("GatherND", 1),
    ("ScatterND", 1),
    ("Squeeze", 1),
    ("Unsqueeze", 1),
    ("ReduceSum", 1),
    ("ReduceMax", 1),
    ("ReduceMin", 1),
}

SHAPE_INDEX_OP_INPUTS: set[tuple[str, int]] = {
    ("Gather", 1),
    ("GatherElements", 1),
    ("ScatterElements", 1),
    ("Slice", 1),
    ("Slice", 2),
    ("Slice", 3),
    ("Slice", 4),
    ("Reshape", 1),
    ("Unsqueeze", 1),
    ("Squeeze", 1),
    ("Pad", 1),
    ("Tile", 1),
    ("Expand", 1),
}

SAFE_AXIS_BROADCAST_OPS: set[str] = {
    "Greater",
    "Less",
    "Equal",
    "LessOrEqual",
    "GreaterOrEqual",
    "Add",
    "Sub",
    "Mul",
    "Div",
    "Max",
    "Min",
    "Sum",
    "Where",
}
SAFE_CONST_AXIS_BROADCAST_OPS: set[str] = {"Where"}

F32 = TensorProto.FLOAT
F16 = TensorProto.FLOAT16
FP16_MAX = 65504.0
VALUE_PRESERVING_FP16_OPS: set[str] = {
    "Slice",
    "Gather",
    "Transpose",
    "Reshape",
    "Squeeze",
    "Unsqueeze",
    "Identity",
}


def _consumers(graph: onnx.GraphProto) -> dict[str, list[tuple[onnx.NodeProto, int]]]:
    out: dict[str, list[tuple[onnx.NodeProto, int]]] = defaultdict(list)
    for node in graph.node:
        for pos, name in enumerate(node.input):
            if name:
                out[name].append((node, pos))
    return out


def _init_arrays(graph: onnx.GraphProto) -> dict[str, np.ndarray]:
    arrs: dict[str, np.ndarray] = {}
    for init in graph.initializer:
        try:
            arrs[init.name] = numpy_helper.to_array(init)
        except Exception:
            pass
    return arrs


def narrow_int64_to_int32(model: onnx.ModelProto) -> onnx.ModelProto:
    m = copy.deepcopy(model)
    graph = m.graph
    cons = _consumers(graph)
    out_names = {o.name for o in graph.output}

    def safe(name: str) -> bool:
        if name not in cons:
            return True
        for node, pos in cons[name]:
            key = (node.op_type, pos)
            if key in INT64_REQUIRED or key not in INT32_SAFE_INPUTS:
                return False
        return True

    for i, init in enumerate(graph.initializer):
        arr = numpy_helper.to_array(init)
        if arr.dtype != np.int64:
            continue
        if arr.size and (
            arr.min() < np.iinfo(np.int32).min or arr.max() > np.iinfo(np.int32).max
        ):
            continue
        if not safe(init.name):
            continue
        graph.initializer[i].CopyFrom(
            numpy_helper.from_array(arr.astype(np.int32), init.name)
        )

    new_nodes: list[onnx.NodeProto] = []
    for node in graph.node:
        new_nodes.append(node)
        if node.op_type not in ("ArgMax", "ArgMin"):
            continue
        old = node.output[0]
        if old in out_names or not safe(old):
            continue
        new = old + "_i64"
        node.output[0] = new
        new_nodes.append(helper.make_node("Cast", [new], [old], to=TensorProto.INT32))
    del graph.node[:]
    graph.node.extend(new_nodes)
    return m


# --- identity-elementwise elimination -----------------------------------------


def _const_array(graph: onnx.GraphProto, name: str) -> np.ndarray | None:
    """name が初期化子か Constant ノード出力なら値配列を返す。"""
    for init in graph.initializer:
        if init.name == name:
            try:
                return numpy_helper.to_array(init)
            except Exception:
                return None
    for node in graph.node:
        if node.op_type == "Constant" and node.output and node.output[0] == name:
            for attr in node.attribute:
                if attr.name == "value":
                    try:
                        return numpy_helper.to_array(attr.t)
                    except Exception:
                        return None
    return None


def eliminate_identity_elementwise(model: onnx.ModelProto) -> onnx.ModelProto:
    """恒等な二項 elementwise を除去（中間テンソルを削減）。

    ``Max(x, 0)`` / ``Add(x, 0)`` / ``Mul(x, 1)`` / ``Min(x, 1)`` は、定数側が
    全要素 0（Max/Add）または 1（Mul/Min）なら ``x`` に等しい。0/1 マスク用途で
    bundle に残る冗長な Max/Min を畳み込み、その出力中間テンソルを消す。
    グラフ出力ノードは触らない。実効性は呼び出し側が audit_one で検証する。
    """
    m = copy.deepcopy(model)
    graph = m.graph
    out_names = {o.name for o in graph.output}
    rewire: dict[str, str] = {}
    kept: list[onnx.NodeProto] = []
    for node in graph.node:
        drop = False
        if (
            node.op_type in ("Max", "Min", "Mul", "Add")
            and len(node.input) == 2
            and len(node.output) == 1
            and node.output[0] not in out_names
        ):
            ca = _const_array(graph, node.input[0])
            cb = _const_array(graph, node.input[1])

            def _is_zero(c: np.ndarray | None) -> bool:
                return c is not None and bool(np.all(c == 0))

            def _is_one(c: np.ndarray | None) -> bool:
                return c is not None and bool(np.all(c == 1))

            other: str | None = None
            if node.op_type in ("Max", "Add"):
                other = (
                    node.input[1]
                    if _is_zero(ca)
                    else (node.input[0] if _is_zero(cb) else None)
                )
            elif node.op_type == "Mul":
                other = (
                    node.input[1]
                    if _is_one(ca)
                    else (node.input[0] if _is_one(cb) else None)
                )
            elif node.op_type == "Min":
                other = (
                    node.input[1]
                    if _is_one(ca)
                    else (node.input[0] if _is_one(cb) else None)
                )
            if other is not None:
                rewire[node.output[0]] = other
                drop = True
        if not drop:
            kept.append(node)
    if not rewire:
        return m

    def _resolve(name: str) -> str:
        seen: set[str] = set()
        while name in rewire and name not in seen:
            seen.add(name)
            name = rewire[name]
        return name

    for node in kept:
        for i in range(len(node.input)):
            node.input[i] = _resolve(node.input[i])
    del graph.node[:]
    graph.node.extend(kept)
    _prune_unused_initializers(graph)
    del graph.value_info[:]
    return m


# --- cleanup ------------------------------------------------------------------


def _prune_unused_initializers(graph: onnx.GraphProto) -> None:
    used = {x for node in graph.node for x in node.input if x}
    used |= {x.name for x in graph.input}
    keep = [init for init in graph.initializer if init.name in used]
    del graph.initializer[:]
    graph.initializer.extend(keep)


def _init_key(init: onnx.TensorProto) -> tuple[str, tuple[int, ...], bytes]:
    arr = numpy_helper.to_array(init)
    return (arr.dtype.str, tuple(arr.shape), arr.tobytes())


def cleanup(model: onnx.ModelProto) -> onnx.ModelProto:
    m = copy.deepcopy(model)
    graph = m.graph
    _prune_unused_initializers(graph)

    groups: dict[tuple[str, tuple[int, ...], bytes], list[str]] = defaultdict(list)
    for init in graph.initializer:
        groups[_init_key(init)].append(init.name)
    replace: dict[str, str] = {}
    for names in groups.values():
        if len(names) <= 1:
            continue
        canonical = sorted(names, key=lambda s: (len(s), s))[0]
        for name in names:
            if name != canonical:
                replace[name] = canonical
    if replace:
        for node in graph.node:
            for i, name in enumerate(node.input):
                if name in replace:
                    node.input[i] = replace[name]
        _prune_unused_initializers(graph)

    out_names = {o.name for o in graph.output}
    aliases: dict[str, str] = {}
    for node in graph.node:
        if (
            node.op_type == "Identity"
            and len(node.input) == 1
            and len(node.output) == 1
            and node.output[0] not in out_names
        ):
            aliases[node.output[0]] = node.input[0]
    if aliases:

        def resolve(name: str) -> str:
            seen: set[str] = set()
            while name in aliases and name not in seen:
                seen.add(name)
                name = aliases[name]
            return name

        kept: list[onnx.NodeProto] = []
        for node in graph.node:
            for i, name in enumerate(node.input):
                if name in aliases:
                    node.input[i] = resolve(name)
            if node.op_type == "Identity" and node.output and node.output[0] in aliases:
                continue
            kept.append(node)
        del graph.node[:]
        graph.node.extend(kept)
        del graph.value_info[:]
    return m


# --- index surgery ------------------------------------------------------------


def dedup_tiny_shape_index_initializers(model: onnx.ModelProto) -> onnx.ModelProto:
    m = copy.deepcopy(model)
    graph = m.graph
    cons = _consumers(graph)
    key_to_name: dict[tuple[str, tuple[int, ...], bytes], str] = {}
    rewires: dict[str, str] = {}
    remove: set[str] = set()
    for init in graph.initializer:
        arr = numpy_helper.to_array(init)
        if arr.dtype not in (np.int32, np.int64) or arr.size > 8:
            continue
        if any(
            (node.op_type, pos) not in SHAPE_INDEX_OP_INPUTS
            for node, pos in cons.get(init.name, [])
        ):
            continue
        key = (arr.dtype.str, arr.shape, arr.tobytes())
        if key not in key_to_name:
            key_to_name[key] = init.name
        else:
            rewires[init.name] = key_to_name[key]
            remove.add(init.name)
    if not remove:
        return m
    for node in graph.node:
        for i, name in enumerate(node.input):
            if name in rewires:
                node.input[i] = rewires[name]
    keep = [init for init in graph.initializer if init.name not in remove]
    del graph.initializer[:]
    graph.initializer.extend(keep)
    return m


def remove_redundant_slice_index_inputs(model: onnx.ModelProto) -> onnx.ModelProto:
    m = copy.deepcopy(model)
    graph = m.graph
    arrs = _init_arrays(graph)
    for node in graph.node:
        if node.op_type != "Slice" or len(node.input) < 4:
            continue
        original = list(node.input)
        starts = arrs.get(node.input[1])
        axes = (
            arrs.get(node.input[3]) if len(node.input) >= 4 and node.input[3] else None
        )
        steps = (
            arrs.get(node.input[4]) if len(node.input) >= 5 and node.input[4] else None
        )
        can_steps = steps is not None and steps.size > 0 and bool(np.all(steps == 1))
        can_axes = False
        if axes is not None and starts is not None:
            default = np.arange(starts.size, dtype=axes.dtype)
            can_axes = axes.shape == default.shape and bool(
                np.array_equal(axes, default)
            )
        if can_axes and (len(original) < 5 or can_steps):
            del node.input[:]
            node.input.extend(original[:3])
            continue
        if can_steps and len(original) == 5:
            del node.input[:]
            node.input.extend(original[:4])
    _prune_unused_initializers(graph)
    return m


def default_slice_axes_hole_punch(model: onnx.ModelProto) -> onnx.ModelProto:
    m = copy.deepcopy(model)
    graph = m.graph
    arrs = _init_arrays(graph)
    for node in graph.node:
        if node.op_type != "Slice" or len(node.input) < 5:
            continue
        if node.input[3] == "" or node.input[4] == "":
            continue
        starts = arrs.get(node.input[1])
        axes = arrs.get(node.input[3])
        steps = arrs.get(node.input[4])
        if starts is None or axes is None or steps is None:
            continue
        if steps.size > 0 and np.all(steps == 1):
            continue
        default = np.arange(starts.size, dtype=axes.dtype)
        if axes.shape == default.shape and np.array_equal(axes, default):
            node.input[3] = ""
    _prune_unused_initializers(graph)
    return m


def index_surgery(model: onnx.ModelProto) -> onnx.ModelProto:
    m = narrow_int64_to_int32(model)
    m = dedup_tiny_shape_index_initializers(m)
    m = remove_redundant_slice_index_inputs(m)
    m = default_slice_axes_hole_punch(m)
    return m


# --- broadcast compression ----------------------------------------------------


def _compress_axes_keep_rank(arr: np.ndarray) -> np.ndarray | None:
    if arr.ndim == 0 or arr.size <= 1:
        return None
    out = arr
    changed = False
    for axis in range(arr.ndim):
        if out.shape[axis] <= 1:
            continue
        first = np.take(out, [0], axis=axis)
        if np.all(out == first):
            out = first
            changed = True
    if not changed or out.size >= arr.size:
        return None
    return out.astype(arr.dtype, copy=False)


def broadcast_compress(model: onnx.ModelProto) -> onnx.ModelProto:
    m = copy.deepcopy(model)
    graph = m.graph
    consumer_ops: dict[str, set[str]] = defaultdict(set)
    for node in graph.node:
        for name in node.input:
            if name:
                consumer_ops[name].add(node.op_type)
    out_names = {o.name for o in graph.output}
    in_names = {i.name for i in graph.input}
    changed = False
    for init in graph.initializer:
        if init.name in out_names or init.name in in_names:
            continue
        ops = consumer_ops.get(init.name, set())
        if not ops or not ops <= SAFE_AXIS_BROADCAST_OPS:
            continue
        try:
            arr = numpy_helper.to_array(init)
        except Exception:
            continue
        comp = _compress_axes_keep_rank(arr)
        if comp is None:
            continue
        init.CopyFrom(numpy_helper.from_array(comp, init.name))
        changed = True
    for node in graph.node:
        if node.op_type != "Constant" or len(node.output) != 1:
            continue
        if node.output[0] in out_names:
            continue
        ops = consumer_ops.get(node.output[0], set())
        if not ops or not ops <= SAFE_CONST_AXIS_BROADCAST_OPS:
            continue
        attr = next(
            (a for a in node.attribute if a.name == "value" and a.HasField("t")), None
        )
        if attr is None:
            continue
        try:
            arr = numpy_helper.to_array(attr.t)
        except Exception:
            continue
        comp = _compress_axes_keep_rank(arr)
        if comp is None:
            continue
        attr.t.CopyFrom(numpy_helper.from_array(comp, attr.t.name))
        changed = True
    if changed:
        del graph.value_info[:]
    return m


# --- conv1x1 -> gather --------------------------------------------------------


def _conv1x1_perm_indices(w: np.ndarray) -> np.ndarray | None:
    if w.shape != (10, 10, 1, 1):
        return None
    mat = w[:, :, 0, 0]
    if not np.all(np.isclose(mat, 0) | np.isclose(mat, 1)):
        return None
    if not np.all(mat.sum(axis=1) == 1):
        return None
    return np.argmax(mat, axis=1).astype(np.int64)


def conv1x1_to_gather(model: onnx.ModelProto) -> onnx.ModelProto:
    m = copy.deepcopy(model)
    graph = m.graph
    inits = {init.name: init for init in graph.initializer}
    used = {x for n in graph.node for x in list(n.input) + list(n.output)}
    used |= {i.name for i in graph.initializer}
    new_nodes: list[onnx.NodeProto] = []
    remove: set[str] = set()
    counter = 0
    for node in graph.node:
        if node.op_type != "Conv" or len(node.input) < 2:
            new_nodes.append(node)
            continue
        w_init = inits.get(node.input[1])
        if w_init is None:
            new_nodes.append(node)
            continue
        try:
            w = numpy_helper.to_array(w_init)
        except Exception:
            new_nodes.append(node)
            continue
        idx = _conv1x1_perm_indices(w)
        if idx is None:
            new_nodes.append(node)
            continue
        b_name = node.input[2] if len(node.input) >= 3 and node.input[2] else None
        if b_name is not None:
            b_init = inits.get(b_name)
            if b_init is None or not np.allclose(numpy_helper.to_array(b_init), 0):
                new_nodes.append(node)
                continue
            remove.add(b_name)
        idx_name = f"{node.output[0]}_gidx_{counter}"
        while idx_name in used:
            counter += 1
            idx_name = f"{node.output[0]}_gidx_{counter}"
        used.add(idx_name)
        counter += 1
        graph.initializer.append(numpy_helper.from_array(idx, idx_name))
        new_nodes.append(
            helper.make_node(
                "Gather",
                [node.input[0], idx_name],
                list(node.output),
                axis=1,
                name=node.name or f"{node.output[0]}_gather",
            )
        )
        remove.add(node.input[1])
    del graph.node[:]
    graph.node.extend(new_nodes)
    if remove:
        keep = [init for init in graph.initializer if init.name not in remove]
        del graph.initializer[:]
        graph.initializer.extend(keep)
    del graph.value_info[:]
    return m


# --- fp16 surgery -------------------------------------------------------------


def _fp16_fits(arr: np.ndarray) -> bool:
    return arr.size == 0 or float(np.nanmax(np.abs(arr))) <= FP16_MAX


def _all_tensor_names(graph: onnx.GraphProto) -> set[str]:
    names: set[str] = set()
    for x in list(graph.input) + list(graph.output) + list(graph.value_info):
        names.add(x.name)
    for init in graph.initializer:
        names.add(init.name)
    for node in graph.node:
        names.update(x for x in node.input if x)
        names.update(x for x in node.output if x)
    return names


def _guard_fp16_range(graph: onnx.GraphProto) -> None:
    for init in graph.initializer:
        if init.data_type == F32 and not _fp16_fits(numpy_helper.to_array(init)):
            raise ValueError(f"initializer exceeds fp16 range: {init.name}")
    for node in graph.node:
        for attr in node.attribute:
            if attr.type == AttributeProto.TENSOR and attr.t.data_type == F32:
                if not _fp16_fits(numpy_helper.to_array(attr.t)):
                    raise ValueError("constant tensor exceeds fp16 range")
            if attr.name == "value_floats" and attr.floats:
                if max(abs(x) for x in attr.floats) > FP16_MAX:
                    raise ValueError("value_floats exceeds fp16 range")


def fp16_surgery(model: onnx.ModelProto) -> onnx.ModelProto:
    m = copy.deepcopy(model)
    graph = m.graph
    _guard_fp16_range(graph)

    input_name = graph.input[0].name
    out_names = {o.name for o in graph.output}
    init_names = {i.name for i in graph.initializer}

    region = {input_name}
    changed = True
    while changed:
        changed = False
        for node in graph.node:
            if node.op_type not in VALUE_PRESERVING_FP16_OPS:
                continue
            if any(o in region for o in node.output):
                continue
            data_in = [x for x in node.input if x and x not in init_names]
            if data_in and all(x in region for x in data_in):
                for o in node.output:
                    if o:
                        region.add(o)
                        changed = True
    region -= out_names

    def region_node(node: onnx.NodeProto) -> bool:
        return node.op_type in VALUE_PRESERVING_FP16_OPS and any(
            o in region for o in node.output
        )

    boundary: set[str] = set()
    for node in graph.node:
        if region_node(node):
            continue
        for x in node.input:
            if x in region and x != input_name:
                boundary.add(x)
        if input_name in node.input:
            boundary.add(input_name)

    for init in graph.initializer:
        if init.data_type == F32:
            arr = numpy_helper.to_array(init).astype(np.float16)
            init.CopyFrom(numpy_helper.from_array(arr, init.name))

    used = _all_tensor_names(graph)
    cast_map: dict[str, str] = {}
    for t in sorted(boundary):
        name = f"{t}__h16"
        k = 1
        while name in used:
            name = f"{t}__h16_{k}"
            k += 1
        used.add(name)
        cast_map[t] = name

    new_nodes: list[onnx.NodeProto] = []
    if input_name in cast_map:
        new_nodes.append(
            helper.make_node(
                "Cast",
                [input_name],
                [cast_map[input_name]],
                to=F16,
                name=cast_map[input_name],
            )
        )
    for node in graph.node:
        new_nodes.append(node)
        for o in node.output:
            if o in cast_map:
                new_nodes.append(
                    helper.make_node(
                        "Cast", [o], [cast_map[o]], to=F16, name=cast_map[o]
                    )
                )
    del graph.node[:]
    graph.node.extend(new_nodes)

    inserted = set(cast_map.values())
    for node in graph.node:
        if node.op_type == "Cast" and node.output and node.output[0] in inserted:
            continue
        if region_node(node):
            continue
        for i, x in enumerate(node.input):
            if x in cast_map:
                node.input[i] = cast_map[x]

    for node in graph.node:
        if node.op_type == "Cast" and not (node.output and node.output[0] in inserted):
            for attr in node.attribute:
                if attr.name == "to" and attr.i == F32:
                    attr.i = F16
        for attr in node.attribute:
            if attr.type == AttributeProto.TENSOR and attr.t.data_type == F32:
                arr = numpy_helper.to_array(attr.t).astype(np.float16)
                attr.t.CopyFrom(numpy_helper.from_array(arr, attr.t.name))

    del graph.value_info[:]
    for out in graph.output:
        if out.type.tensor_type.elem_type == F32:
            out.type.tensor_type.elem_type = F16

    onnx.checker.check_model(m, full_check=True)
    onnx.shape_inference.infer_shapes(m, strict_mode=True)
    return m


# --- runtime equivalence alias -----------------------------------------------


def _grid_to_input(grid: list[list[int]]) -> np.ndarray:
    arr = np.zeros((1, 10, 30, 30), dtype=np.float32)
    for r, row in enumerate(grid):
        for c, color in enumerate(row):
            if 0 <= color < 10:
                arr[0, color, r, c] = 1.0
    return arr


def equivalence_alias_runtime(
    model: onnx.ModelProto,
    examples: dict[str, Any],
    n_ex: int = 20,
) -> onnx.ModelProto:
    """実行時等値テンソルをエイリアス化して memory cost を削減。

    全中間テンソルを n_ex 例で実行して収集し、Union-Find で同値クラスタを構築する。
    各クラスタの canonical 以外の参照を canonical で書き換え、dead node を除去する。
    """
    import onnxruntime as ort

    m = onnx.shape_inference.infer_shapes(copy.deepcopy(model))
    graph = m.graph

    init_names: set[str] = {i.name for i in graph.initializer}
    out_names: set[str] = {o.name for o in graph.output}

    probe_names = [
        vi.name
        for vi in graph.value_info
        if vi.name not in out_names and vi.name not in init_names
    ]
    if not probe_names:
        return model

    m_ext = copy.deepcopy(m)
    vi_map = {vi.name: vi for vi in m.graph.value_info}
    for name in probe_names:
        if name in vi_map:
            m_ext.graph.output.append(copy.deepcopy(vi_map[name]))

    try:
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        sess = ort.InferenceSession(m_ext.SerializeToString(), opts)
    except Exception:
        return model

    out_name_list = [o.name for o in sess.get_outputs()]
    collected: dict[str, list[np.ndarray]] = defaultdict(list)
    count = 0
    for split in ("train", "test"):
        for ex in examples.get(split, []):
            grid = ex.get("input", [])
            if not grid or max(len(grid), len(grid[0])) > 30:
                continue
            inp = _grid_to_input(grid)
            try:
                outs = sess.run(None, {"input": inp})
            except Exception:
                continue
            for name, arr in zip(out_name_list, outs, strict=True):
                collected[name].append(arr)
            count += 1
            if count >= n_ex:
                break
        if count >= n_ex:
            break

    if count == 0:
        return model

    full = [n for n in probe_names if len(collected.get(n, [])) == count]
    if not full:
        return model

    # Union-Find: canonical = smallest index (earlier in graph order)
    parent = list(range(len(full)))

    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(a: int, b: int) -> None:
        pa, pb = _find(a), _find(b)
        if pa != pb:
            if pa < pb:
                parent[pb] = pa
            else:
                parent[pa] = pb

    for i in range(len(full)):
        arrs_i = collected[full[i]]
        for j in range(i + 1, len(full)):
            arrs_j = collected[full[j]]
            if arrs_i[0].shape != arrs_j[0].shape:
                continue
            if arrs_i[0].dtype != arrs_j[0].dtype:
                continue
            if all(np.array_equal(a, b) for a, b in zip(arrs_i, arrs_j, strict=True)):
                _union(i, j)

    rewire: dict[str, str] = {
        full[j]: full[_find(j)] for j in range(len(full)) if _find(j) != j
    }
    if not rewire:
        return model

    def _resolve(name: str) -> str:
        seen: set[str] = set()
        while name in rewire and name not in seen:
            seen.add(name)
            name = rewire[name]
        return name

    orig = copy.deepcopy(model)
    g = orig.graph
    for node in g.node:
        for k, inp in enumerate(node.input):
            if inp in rewire:
                node.input[k] = _resolve(inp)

    # Dead node elimination (iterative, reverse topological)
    changed = True
    while changed:
        needed: set[str] = {o.name for o in g.output}
        alive: list[onnx.NodeProto] = []
        changed = False
        for node in reversed(list(g.node)):
            if any(o in needed for o in node.output if o):
                alive.append(node)
                needed.update(inp for inp in node.input if inp)
            else:
                changed = True
        if changed:
            del g.node[:]
            g.node.extend(reversed(alive))

    _prune_unused_initializers(g)
    del g.value_info[:]
    return orig


# --- driver -------------------------------------------------------------------

Pass = Callable[[onnx.ModelProto], onnx.ModelProto]

PASSES: list[tuple[str, Pass]] = [
    ("cleanup", cleanup),
    ("identity_elementwise", eliminate_identity_elementwise),
    ("index_surgery", index_surgery),
    ("broadcast_compress", broadcast_compress),
    ("conv1x1_to_gather", conv1x1_to_gather),
    ("fp16_surgery", fp16_surgery),
]


def _load_task_json(task_dir: Path, task_num: int) -> dict[str, Any] | None:
    p = task_dir / f"task{task_num:03d}.json"
    if not p.is_file():
        return None
    import json

    with p.open() as f:
        data: dict[str, Any] = json.load(f)
    return data
