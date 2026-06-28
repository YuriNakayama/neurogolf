"""DSL primitives as numpy ``grid -> grid`` reference transforms.

Each primitive maps a raw ARC grid (a 2D int array of colors 0-9, up to 30x30)
to a transformed grid. The solver evaluates a candidate primitive against every
example pair (train + test + arc-gen) and, when it reproduces all outputs
exactly, emits the matching minimal ONNX from :mod:`onnx_ops`.

These are the *reference* semantics; the ONNX builders must produce byte-identical
results on the encoded ``[1,10,30,30]`` tensor (verified via ``src/evaluate``).
"""

from __future__ import annotations

import numpy as np

Grid = np.ndarray


def identity(g: Grid) -> Grid:
    return g


def flip_h(g: Grid) -> Grid:
    """Mirror left-right (reverse columns)."""
    return g[:, ::-1]


def flip_v(g: Grid) -> Grid:
    """Mirror top-bottom (reverse rows)."""
    return g[::-1, :]


def rot180(g: Grid) -> Grid:
    return g[::-1, ::-1]


def transpose(g: Grid) -> Grid:
    """Main-diagonal mirror: ``out[c, r] = in[r, c]``."""
    return g.T


def rot90(g: Grid) -> Grid:
    """Rotate 90 clockwise."""
    return np.rot90(g, k=-1)


def rot270(g: Grid) -> Grid:
    """Rotate 90 counter-clockwise."""
    return np.rot90(g, k=1)


def anti_transpose(g: Grid) -> Grid:
    """Anti-diagonal mirror."""
    return g[::-1, ::-1].T


def recolor(g: Grid, mapping: dict[int, int]) -> Grid:
    """Replace each color via ``mapping`` (missing colors map to themselves)."""
    out = g.copy()
    for src, dst in mapping.items():
        out[g == src] = dst
    return out


def tile(g: Grid, reps_h: int, reps_w: int) -> Grid:
    """Tile the grid ``reps_h`` x ``reps_w`` times."""
    return np.tile(g, (reps_h, reps_w))


def scale(g: Grid, sh: int, sw: int) -> Grid:
    """Nearest-neighbour upscale by ``sh`` x ``sw`` (block replication)."""
    return np.repeat(np.repeat(g, sh, axis=0), sw, axis=1)


def crop_bbox(g: Grid, bg: int = 0) -> Grid:
    """Crop to the bounding box of all non-``bg`` cells (re-anchored)."""
    mask = g != bg
    if not mask.any():
        return g
    rows, cols = np.where(mask.any(1))[0], np.where(mask.any(0))[0]
    return g[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]


def symmetrize_h(g: Grid) -> Grid:
    """Overlay the left-right mirror onto the grid (fill background)."""
    m = g[:, ::-1]
    return np.where(g == 0, m, g)


def symmetrize_v(g: Grid) -> Grid:
    """Overlay the top-bottom mirror onto the grid (fill background)."""
    m = g[::-1]
    return np.where(g == 0, m, g)


def symmetrize_all(g: Grid) -> Grid:
    """Overlay all four dihedral mirrors (fill background each step)."""
    out = g
    for m in (g[:, ::-1], g[::-1], g[::-1, ::-1]):
        out = np.where(out == 0, m, out)
    return out


def mosaic(g: Grid, reps_h: int, reps_w: int) -> Grid:
    """Mirror-tile: alternate flips so adjacent tiles share an edge (kaleidoscope)."""
    row_blocks = []
    for r in range(reps_h):
        blocks = []
        for c in range(reps_w):
            b = g
            if r % 2:
                b = b[::-1]
            if c % 2:
                b = b[:, ::-1]
            blocks.append(b)
        row_blocks.append(np.concatenate(blocks, axis=1))
    return np.concatenate(row_blocks, axis=0)


def gravity_down(g: Grid) -> Grid:
    """Drop all non-zero cells to the bottom of each column (stable order)."""
    out = np.zeros_like(g)
    h = g.shape[0]
    for c in range(g.shape[1]):
        col = g[:, c]
        vals = col[col != 0]
        out[h - len(vals) :, c] = vals
    return out


def keep_color(g: Grid, color: int) -> Grid:
    """Keep only ``color``; everything else becomes background 0."""
    return np.where(g == color, g, 0)


def subgrid(g: Grid, h0: int, h1: int, w0: int, w1: int) -> Grid:
    """Extract the subgrid at fixed coordinates ``[h0:h1, w0:w1]``."""
    return g[h0:h1, w0:w1]
