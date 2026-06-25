"""Grid <-> ONNX tensor encoding.

A grid is a rectangular matrix of color indices (0-9). The competition encodes
it as a ``[BATCH, NUM_COLORS, GRID_MAX, GRID_MAX]`` float tensor: cell ``(r, c)``
of color ``v`` sets channel ``v`` to ``1.0`` at ``(r, c)``; out-of-border cells
stay all-zero (zero-hot).

The encode / decode logic mirrors the official ``neurogolf_utils``
``convert_to_numpy`` / ``convert_from_numpy`` so a model that round-trips here
behaves identically under the scorer. Decoding uses the ``> 0`` rule the scorer
applies to network outputs.
"""

from __future__ import annotations

import numpy as np

NUM_COLORS = 10
GRID_MAX = 30
BATCH = 1
CLEAR = 10  # sentinel for an out-of-border / empty cell when decoding

Grid = list[list[int]]


class EncodingError(ValueError):
    """Raised when a grid cannot be encoded (e.g. exceeds ``GRID_MAX``)."""


def encode_grid(grid: Grid) -> np.ndarray:
    """Encode a grid to a ``[1, NUM_COLORS, GRID_MAX, GRID_MAX]`` float tensor."""
    if not grid or not grid[0]:
        raise EncodingError("グリッドが空です")
    height, width = len(grid), max(len(row) for row in grid)
    if max(height, width) > GRID_MAX:
        raise EncodingError(f"グリッドが {GRID_MAX} を超えています: {height}x{width}")
    tensor = np.zeros((BATCH, NUM_COLORS, GRID_MAX, GRID_MAX), dtype=np.float32)
    for r, row in enumerate(grid):
        for c, color in enumerate(row):
            tensor[0][color][r][c] = 1.0
    return tensor


def decode_grid(tensor: np.ndarray) -> Grid:
    """Decode a ``[1, NUM_COLORS, GRID_MAX, GRID_MAX]`` tensor back to a grid.

    A cell is the single color whose channel is active (``> 0``); a cell with no
    active channel is treated as clear and trimmed from trailing rows/columns,
    matching the official ``convert_from_numpy``.
    """
    active = tensor[0] > 0.0
    _, height, width = active.shape
    grid: Grid = []
    for row in range(height):
        cells: list[int] = []
        for col in range(width):
            colors = [c for c in range(active.shape[0]) if active[c][row][col]]
            cells.append(colors[0] if len(colors) == 1 else (11 if colors else CLEAR))
        while cells and cells[-1] == CLEAR:
            cells.pop()
        grid.append(cells)
    while grid and not grid[-1]:
        grid.pop()
    return grid
