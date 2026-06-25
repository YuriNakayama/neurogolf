"""Tests for grid <-> tensor encoding."""

from __future__ import annotations

import numpy as np
import pytest

from dataset.encoding import (
    GRID_MAX,
    NUM_COLORS,
    EncodingError,
    decode_grid,
    encode_grid,
)


def test_encode_shape_and_dtype() -> None:
    tensor = encode_grid([[0, 1], [2, 3]])
    assert tensor.shape == (1, NUM_COLORS, GRID_MAX, GRID_MAX)
    assert tensor.dtype == np.float32


def test_encode_sets_one_hot_channel() -> None:
    tensor = encode_grid([[5]])
    assert tensor[0][5][0][0] == 1.0
    assert tensor[0][:, 0, 0].sum() == 1.0  # exactly one channel hot


def test_out_of_border_is_zero_hot() -> None:
    tensor = encode_grid([[7]])
    # every cell except (0,0) is all-zero across channels
    assert tensor.sum() == 1.0


def test_round_trip_preserves_grid() -> None:
    grid = [[0, 4, 9], [1, 1, 0], [8, 0, 3]]
    assert decode_grid(encode_grid(grid)) == grid


def test_decode_trims_trailing_clear_cells() -> None:
    # a 1x1 grid of color 0 must come back as [[0]], not padded to 30x30
    assert decode_grid(encode_grid([[0]])) == [[0]]


def test_encode_rejects_oversized_grid() -> None:
    with pytest.raises(EncodingError):
        encode_grid([[0] * (GRID_MAX + 1)])


def test_encode_rejects_empty_grid() -> None:
    with pytest.raises(EncodingError):
        encode_grid([])
