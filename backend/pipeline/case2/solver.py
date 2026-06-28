"""Per-task solver: infer a DSL primitive from examples, emit minimal ONNX.

Strategy: try each candidate primitive (geometry first — cheapest, then recolor,
then tiling). A primitive *solves* a task only if it reproduces **every** example
pair (train + test + arc-gen) exactly — hundreds of pairs, strong evidence the
inferred rule is the true transformation and will generalize to the hidden set.

The matched primitive is realized as the cheapest ONNX that reproduces it on the
encoded ``[1,10,30,30]`` tensor (anchored top-left). Geometry that needs a grid
dimension (flip/rot) is emitted only when that dimension is constant across all
examples (so the static graph is valid for every pair).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnx

from . import dsl, onnx_ops

Grid = np.ndarray
ExamplePair = tuple[Grid, Grid]


@dataclass(frozen=True)
class Solution:
    """A solved task: the primitive name and its minimal ONNX model."""

    name: str
    model: onnx.ModelProto


def load_examples(task_path: Path) -> list[ExamplePair]:
    """All (input, output) grid pairs from a task JSON (train+test+arc-gen)."""
    task = json.loads(task_path.read_text())
    pairs: list[ExamplePair] = []
    for key in ("train", "test", "arc-gen"):
        for ex in task.get(key, []):
            inp, out = np.array(ex["input"]), np.array(ex["output"])
            if max(inp.shape) > 30 or max(out.shape) > 30:
                continue
            pairs.append((inp, out))
    return pairs


def _matches(fn: Callable[[Grid], Grid], pairs: list[ExamplePair]) -> bool:
    for inp, out in pairs:
        try:
            got = fn(inp)
        except Exception:
            return False
        if got.shape != out.shape or not np.array_equal(got, out):
            return False
    return True


def _const_dim(
    pairs: list[ExamplePair], axis: int, on_input: bool = True
) -> int | None:
    """Return the grid dim along ``axis`` if constant across pairs, else None."""
    dims = {(p[0] if on_input else p[1]).shape[axis] for p in pairs}
    return next(iter(dims)) if len(dims) == 1 else None


def _infer_recolor(pairs: list[ExamplePair]) -> list[int] | None:
    """Infer a per-color permutation ``perm`` with ``out[c]=in[perm[c]]``.

    A recolor maps every source color to a fixed destination, same-shape grids.
    Returns the channel-gather permutation (``perm[dst]=src``) or None.
    """
    src_to_dst: dict[int, int] = {}
    for inp, out in pairs:
        if inp.shape != out.shape:
            return None
        for s, d in zip(inp.flatten(), out.flatten(), strict=True):
            s, d = int(s), int(d)
            if s in src_to_dst and src_to_dst[s] != d:
                return None
            src_to_dst[s] = d
    # The observed (src -> dst) must be injective so a single Gather can realize
    # it: each output channel ``d`` is gathered from exactly one input channel.
    dst_to_src: dict[int, int] = {}
    for s, d in src_to_dst.items():
        if d in dst_to_src and dst_to_src[d] != s:
            return None
        dst_to_src[d] = s
    # Fill output channels with no observed source from the unused input channels
    # (those channels are all-zero in every example, so any assignment is exact).
    perm: list[int] = [-1] * onnx_ops.NUM_COLORS
    for d, s in dst_to_src.items():
        perm[d] = s
    free_srcs = [c for c in range(onnx_ops.NUM_COLORS) if c not in dst_to_src.values()]
    it = iter(free_srcs)
    for d in range(onnx_ops.NUM_COLORS):
        if perm[d] == -1:
            perm[d] = next(it)
    return perm


def _infer_subgrid(pairs: list[ExamplePair]) -> tuple[int, int, int, int] | None:
    """Infer a fixed crop region ``(h0, h1, w0, w1)`` constant across all pairs.

    Returns the region if every output equals ``input[h0:h1, w0:w1]`` for the
    same coordinates; None otherwise.
    """
    region: tuple[int, int, int, int] | None = None
    for inp, out in pairs:
        ih, iw = inp.shape
        oh, ow = out.shape
        if oh > ih or ow > iw:
            return None
        match: tuple[int, int, int, int] | None = None
        for h0 in range(ih - oh + 1):
            for w0 in range(iw - ow + 1):
                if np.array_equal(inp[h0 : h0 + oh, w0 : w0 + ow], out):
                    match = (h0, h0 + oh, w0, w0 + ow)
                    break
            if match is not None:
                break
        if match is None:
            return None
        if region is None:
            region = match
        elif region != match:
            return None
    return region


def _infer_tile(pairs: list[ExamplePair]) -> tuple[int, int] | None:
    """Infer integer tile reps ``(rh, rw)`` constant across pairs, else None."""
    reps: set[tuple[int, int]] = set()
    for inp, out in pairs:
        ih, iw = inp.shape
        oh, ow = out.shape
        if ih == 0 or iw == 0 or oh % ih or ow % iw:
            return None
        reps.add((oh // ih, ow // iw))
    if len(reps) != 1:
        return None
    rh, rw = next(iter(reps))
    if (rh, rw) == (1, 1):
        return None
    if not _matches(lambda g: dsl.tile(g, rh, rw), pairs):
        return None
    return rh, rw


def _infer_factor(pairs: list[ExamplePair]) -> tuple[int, int] | None:
    """Infer a constant integer (h,w) output/input size ratio, else None."""
    facs: set[tuple[int, int]] = set()
    for inp, out in pairs:
        ih, iw = inp.shape
        oh, ow = out.shape
        if ih == 0 or iw == 0 or oh % ih or ow % iw:
            return None
        facs.add((oh // ih, ow // iw))
    if len(facs) != 1:
        return None
    fh, fw = next(iter(facs))
    return None if (fh, fw) == (1, 1) else (fh, fw)


def solve(task_path: Path) -> Solution | None:
    """Return the cheapest DSL solution for a task, or None if no primitive fits."""
    pairs = load_examples(task_path)
    if not pairs:
        return None

    # Geometry (ordered cheapest-first by typical cost).
    if _matches(dsl.identity, pairs):
        return Solution("identity", onnx_ops.identity())
    if _matches(dsl.transpose, pairs):
        return Solution("transpose", onnx_ops.transpose())

    perm = _infer_recolor(pairs)
    if perm is not None:
        return Solution("recolor", onnx_ops.recolor(perm))

    w = _const_dim(pairs, axis=1)
    h = _const_dim(pairs, axis=0)
    if w is not None and _matches(dsl.flip_h, pairs):
        return Solution("flip_h", onnx_ops.flip_h(w))
    if h is not None and _matches(dsl.flip_v, pairs):
        return Solution("flip_v", onnx_ops.flip_v(h))
    if h is not None and w is not None and _matches(dsl.rot180, pairs):
        return Solution("rot180", onnx_ops.rot180(h, w))
    if h is not None and _matches(dsl.rot90, pairs):
        return Solution("rot90", onnx_ops.rot90(h))
    if w is not None and _matches(dsl.rot270, pairs):
        return Solution("rot270", onnx_ops.rot270(w))
    if h is not None and w is not None and _matches(dsl.anti_transpose, pairs):
        return Solution("anti_transpose", onnx_ops.anti_transpose(h, w))

    # Size-changing structure: scale (block-replicate), tile, mosaic.
    fac = _infer_factor(pairs)
    if fac is not None and h is not None and w is not None:
        fh, fw = fac
        fits = fh * h <= onnx_ops.GRID_MAX and fw * w <= onnx_ops.GRID_MAX
        if fits and _matches(lambda g: dsl.scale(g, fh, fw), pairs):
            return Solution("scale", onnx_ops.scale(h, w, fh, fw))
        if fits and _matches(lambda g: dsl.tile(g, fh, fw), pairs):
            return Solution("tile", onnx_ops.tile(h, w, fh, fw))
        if fits and _matches(lambda g: dsl.mosaic(g, fh, fw), pairs):
            return Solution("mosaic", onnx_ops.mosaic(h, w, fh, fw))

    # Symmetry completion (overlay mirror copies onto the grid).
    if h is not None and w is not None:
        if _matches(dsl.symmetrize_h, pairs):
            return Solution("symmetrize_h", onnx_ops.symmetrize(h, w, ("h",)))
        if _matches(dsl.symmetrize_v, pairs):
            return Solution("symmetrize_v", onnx_ops.symmetrize(h, w, ("v",)))
        if _matches(dsl.symmetrize_all, pairs):
            return Solution("symmetrize_all", onnx_ops.symmetrize(h, w, ("h", "v")))

    # Fixed crop: extract the same rectangular region from every input.
    crop = _infer_subgrid(pairs)
    if crop is not None:
        h0, h1, w0, w1 = crop
        return Solution("subgrid", onnx_ops.subgrid(h0, h1, w0, w1))

    # Single-color keep (mask out all but one color + background).
    colors = {int(c) for inp, _ in pairs for c in np.unique(inp)} - {0}
    for color in sorted(colors):

        def _keep(g: Grid, c: int = color) -> Grid:
            return dsl.keep_color(g, c)

        if _matches(_keep, pairs):
            return Solution(f"keep_color_{color}", onnx_ops.keep_color(color))

    return None
