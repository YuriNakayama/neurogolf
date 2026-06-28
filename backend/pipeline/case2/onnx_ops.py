"""Minimal ONNX graph builders for case2 DSL primitives.

Each builder returns an ``onnx.ModelProto`` whose single input ``input`` and
single output ``output`` are ``[1, 10, 30, 30]`` float tensors (the competition
encoding: 10-color one-hot, 30x30, out-of-border cells zero-hot, grid anchored at
the top-left).

The grids are zero-padded to 30x30 and **anchored at the top-left**, so a
transform must keep its result anchored too. A naive full-axis reverse would move
an ``h x w`` grid to the opposite edge; instead reversals are expressed as a
``Gather`` whose index reverses only the occupied ``[0:w]`` (or ``[0:h]``) range
and leaves the padding in place. ``Transpose`` already re-anchors at the origin.

Graphs are hand-built with ``onnx.helper`` (never torch export) to keep
``cost = params + memory`` minimal. The official scorer (``src/evaluate``)
requires a fully static graph passing ``onnx.checker`` +
``shape_inference(strict_mode=True)``; these builders only emit such graphs.
"""

from __future__ import annotations

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

NUM_COLORS = 10
GRID_MAX = 30
_SHAPE = [1, NUM_COLORS, GRID_MAX, GRID_MAX]
_OPSET = 18
_IR_VERSION = 10


def _model(
    nodes: list[onnx.NodeProto], inits: list[onnx.TensorProto]
) -> onnx.ModelProto:
    """Wrap nodes/initializers into a checked, shape-inferred model."""
    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, _SHAPE)
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, _SHAPE)
    graph = helper.make_graph(nodes, "g", [inp], [out], initializer=inits)
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", _OPSET)], ir_version=_IR_VERSION
    )
    onnx.checker.check_model(model, full_check=True)
    return model


def _i64(name: str, arr: list[int]) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(arr, dtype=np.int64), name=name)


def _reverse_index(n: int) -> list[int]:
    """Index that reverses positions ``0..n-1`` and keeps ``n..29`` in place."""
    return list(range(n - 1, -1, -1)) + list(range(n, GRID_MAX))


def _gather(
    idx_name: str, idx: list[int], axis: int, out: str = "output"
) -> onnx.ModelProto:
    init = _i64(idx_name, idx)
    node = helper.make_node("Gather", ["input", idx_name], [out], axis=axis)
    return _model([node], [init])


def identity() -> onnx.ModelProto:
    """``output = input`` — one Identity node, cost 0."""
    return _model([helper.make_node("Identity", ["input"], ["output"])], [])


def recolor(perm: list[int]) -> onnx.ModelProto:
    """Permute color channels: ``output[c] = input[perm[c]]`` (Gather axis 1).

    ``perm[c]`` is the source color that becomes output color ``c``. Cost = 10.
    """
    if len(perm) != NUM_COLORS:
        raise ValueError(f"perm must have {NUM_COLORS} entries, got {len(perm)}")
    return _gather("perm", perm, axis=1)


def flip_h(w: int) -> onnx.ModelProto:
    """Mirror left-right within the occupied width ``w`` (Gather axis 3)."""
    return _gather("idx_w", _reverse_index(w), axis=3)


def flip_v(h: int) -> onnx.ModelProto:
    """Mirror top-bottom within the occupied height ``h`` (Gather axis 2)."""
    return _gather("idx_h", _reverse_index(h), axis=2)


def rot180(h: int, w: int) -> onnx.ModelProto:
    """Rotate 180: reverse rows within ``h`` then columns within ``w``."""
    g1 = helper.make_node("Gather", ["input", "idx_h"], ["t"], axis=2)
    g2 = helper.make_node("Gather", ["t", "idx_w"], ["output"], axis=3)
    return _model(
        [g1, g2], [_i64("idx_h", _reverse_index(h)), _i64("idx_w", _reverse_index(w))]
    )


def transpose() -> onnx.ModelProto:
    """Diagonal mirror / transpose H<->W (perm), cost 0. Maps ``h x w -> w x h``."""
    node = helper.make_node("Transpose", ["input"], ["output"], perm=[0, 1, 3, 2])
    return _model([node], [])


def rot90(h: int) -> onnx.ModelProto:
    """Rotate 90 CW (``h x w -> w x h``): transpose then reverse new width ``h``."""
    t = helper.make_node("Transpose", ["input"], ["t"], perm=[0, 1, 3, 2])
    g = helper.make_node("Gather", ["t", "idx"], ["output"], axis=3)
    return _model([t, g], [_i64("idx", _reverse_index(h))])


def rot270(w: int) -> onnx.ModelProto:
    """Rotate 90 CCW (``h x w -> w x h``): transpose then reverse new height ``w``."""
    t = helper.make_node("Transpose", ["input"], ["t"], perm=[0, 1, 3, 2])
    g = helper.make_node("Gather", ["t", "idx"], ["output"], axis=2)
    return _model([t, g], [_i64("idx", _reverse_index(w))])


def anti_transpose(h: int, w: int) -> onnx.ModelProto:
    """Anti-diagonal mirror: transpose then rot180 within the new ``w x h``."""
    t = helper.make_node("Transpose", ["input"], ["t"], perm=[0, 1, 3, 2])
    g1 = helper.make_node("Gather", ["t", "idx_w"], ["u"], axis=2)
    g2 = helper.make_node("Gather", ["u", "idx_h"], ["output"], axis=3)
    return _model(
        [t, g1, g2],
        [_i64("idx_w", _reverse_index(w)), _i64("idx_h", _reverse_index(h))],
    )


def subgrid(h0: int, h1: int, w0: int, w1: int) -> onnx.ModelProto:
    """Extract ``[h0:h1, w0:w1]`` and re-anchor it to the top-left.

    Gather rows then cols with an index that maps output position ``i`` to source
    ``offset + i`` for ``i`` inside the crop and to a guaranteed-empty source row
    (``GRID_MAX - 1`` is out-of-grid for any real grid) for the padding. Since the
    encoder leaves row/col 29 empty for sub-30 grids, gathering it yields zeros.
    Cost = numel of the two index initializers (= 60).
    """
    ch, cw = h1 - h0, w1 - w0
    rows = list(range(h0, h1)) + [GRID_MAX - 1] * (GRID_MAX - ch)
    cols = list(range(w0, w1)) + [GRID_MAX - 1] * (GRID_MAX - cw)
    g1 = helper.make_node("Gather", ["input", "rows"], ["t"], axis=2)
    g2 = helper.make_node("Gather", ["t", "cols"], ["output"], axis=3)
    return _model([g1, g2], [_i64("rows", rows), _i64("cols", cols)])


def _tile_index(n: int, reps: int) -> list[int]:
    """Index that repeats positions ``0..n-1`` ``reps`` times, padding to 30."""
    out = (list(range(n)) * reps)[:GRID_MAX]
    return out + [GRID_MAX - 1] * (GRID_MAX - len(out))


def tile(h: int, w: int, reps_h: int, reps_w: int) -> onnx.ModelProto:
    """Tile an ``h x w`` grid ``reps_h x reps_w`` times, anchored top-left.

    Gather rows then cols with a repeating index so the occupied block repeats;
    the padding gathers empty row/col 29. Requires ``reps_h*h <= 30`` and
    ``reps_w*w <= 30``. Cost = 60 (two index initializers).
    """
    rows = _tile_index(h, reps_h)
    cols = _tile_index(w, reps_w)
    g1 = helper.make_node("Gather", ["input", "rows"], ["t"], axis=2)
    g2 = helper.make_node("Gather", ["t", "cols"], ["output"], axis=3)
    return _model([g1, g2], [_i64("rows", rows), _i64("cols", cols)])


def _scale_index(n: int, factor: int) -> list[int]:
    """Index mapping output pos -> source pos for nearest-neighbour upscale."""
    out = [i // factor for i in range(min(n * factor, GRID_MAX))]
    return out + [GRID_MAX - 1] * (GRID_MAX - len(out))


def scale(h: int, w: int, sh: int, sw: int) -> onnx.ModelProto:
    """Block-replicate upscale by ``sh x sw`` (each cell -> ``sh x sw`` block).

    Pure index Gather on each axis (``out[i]=in[i//s]``). Requires the upscaled
    grid to fit 30x30. Cost = 60.
    """
    rows = _scale_index(h, sh)
    cols = _scale_index(w, sw)
    g1 = helper.make_node("Gather", ["input", "rows"], ["t"], axis=2)
    g2 = helper.make_node("Gather", ["t", "cols"], ["output"], axis=3)
    return _model([g1, g2], [_i64("rows", rows), _i64("cols", cols)])


def _mirror_tile_index(n: int, reps: int) -> list[int]:
    """Index for mirror-tiling: tile ``reps`` times, flipping every other copy."""
    seq: list[int] = []
    for r in range(reps):
        block = list(range(n)) if r % 2 == 0 else list(range(n - 1, -1, -1))
        seq.extend(block)
    seq = seq[:GRID_MAX]
    return seq + [GRID_MAX - 1] * (GRID_MAX - len(seq))


def mosaic(h: int, w: int, reps_h: int, reps_w: int) -> onnx.ModelProto:
    """Mirror-tile (kaleidoscope): adjacent tiles are flipped to share an edge."""
    rows = _mirror_tile_index(h, reps_h)
    cols = _mirror_tile_index(w, reps_w)
    g1 = helper.make_node("Gather", ["input", "rows"], ["t"], axis=2)
    g2 = helper.make_node("Gather", ["t", "cols"], ["output"], axis=3)
    return _model([g1, g2], [_i64("rows", rows), _i64("cols", cols)])


def symmetrize(h: int, w: int, axes: tuple[str, ...]) -> onnx.ModelProto:
    """Overlay mirror copies onto the grid via Max (background-fill symmetry).

    ``axes`` may contain ``"h"`` (left-right mirror), ``"v"`` (top-bottom). The
    grid's occupied cells are reversed within ``[0:w]`` / ``[0:h]`` and combined
    with ``Max`` (valid because background is channel-0 — see note). Because
    color 0 is channel-0-hot, ``Max`` over channels would corrupt background, so
    this is only exact when overlapping cells never conflict (the solver verifies
    via ``audit_one`` over all examples). Cost = ~30 per axis.
    """
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []
    terms = ["input"]
    if "h" in axes:
        nodes.append(helper.make_node("Gather", ["input", "idx_w"], ["mh"], axis=3))
        inits.append(_i64("idx_w", _reverse_index(w)))
        terms.append("mh")
    if "v" in axes:
        nodes.append(helper.make_node("Gather", ["input", "idx_h"], ["mv"], axis=2))
        inits.append(_i64("idx_h", _reverse_index(h)))
        terms.append("mv")
    nodes.append(helper.make_node("Max", terms, ["output"]))
    return _model(nodes, inits)


def keep_color(color: int) -> onnx.ModelProto:
    """Keep only ``color`` (and background); zero every other color channel.

    ``Mul`` the input by a per-channel mask that is 1 on channel 0 and ``color``,
    0 elsewhere. Cost = 10 (the mask initializer).
    """
    mask = np.zeros((1, NUM_COLORS, 1, 1), dtype=np.float32)
    mask[0, 0, 0, 0] = 1.0
    mask[0, color, 0, 0] = 1.0
    init = numpy_helper.from_array(mask, name="mask")
    node = helper.make_node("Mul", ["input", "mask"], ["output"])
    return _model([node], [init])


def einsum_remap(
    row_src: list[int], col_src: list[int], h_in: int, w_in: int
) -> onnx.ModelProto:
    """Any separable row-permute x col-permute via a single Einsum (no big middle).

    ``row_src[h]`` is the input row that becomes output row ``h``; ``col_src[w]``
    likewise for columns. Realized as ``Slice`` to the occupied ``h_in x w_in``
    block then ``Einsum('bcrs,hr,ws->bchw')`` with two tiny selector matrices, then
    ``Pad`` back to 30x30. This expresses flip / rotate / transpose / tile / mosaic
    / scale / crop with only the selector numel as params and a small ``h_out x
    w_out`` intermediate — far cheaper than full-30x30 Gather chains.
    """
    h_out, w_out = len(row_src), len(col_src)
    if h_out > GRID_MAX or w_out > GRID_MAX:
        raise ValueError("remapped grid exceeds 30x30")
    r_sel = np.zeros((h_out, h_in), dtype=np.float32)
    for h, r in enumerate(row_src):
        r_sel[h, r] = 1.0
    w_sel = np.zeros((w_out, w_in), dtype=np.float32)
    for w, c in enumerate(col_src):
        w_sel[w, c] = 1.0
    sl = helper.make_node("Slice", ["input", "s0", "e0", "ax0"], ["x"])
    es = helper.make_node("Einsum", ["x", "R", "W"], ["y"], equation="bcrs,hr,ws->bchw")
    pad = helper.make_node("Pad", ["y", "pads", "zero"], ["output"], mode="constant")
    inits = [
        numpy_helper.from_array(r_sel, "R"),
        numpy_helper.from_array(w_sel, "W"),
        _i64("s0", [0, 0, 0, 0]),
        _i64("e0", [1, NUM_COLORS, h_in, w_in]),
        _i64("ax0", [0, 1, 2, 3]),
        _i64("pads", [0, 0, 0, 0, 0, 0, GRID_MAX - h_out, GRID_MAX - w_out]),
        numpy_helper.from_array(np.array(0, dtype=np.float32), "zero"),
    ]
    return _model([sl, es, pad], inits)
