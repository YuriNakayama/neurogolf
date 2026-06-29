"""LP-based linear separator → single-Conv ONNX builder for k-local tasks.

For each output channel j (0-9), solves an LP feasibility problem to find
w_j, b_j such that:
  (X @ w_j + b_j) >= 1  for all k×k window patches whose output color is j
  (X @ w_j + b_j) <= -1 for all others (including out-of-border cells)

If all 10 channels are separable, builds Conv[10,10,k,k] + bias[10] ONNX.

Cost breakdown (cost = params + memory):
  params = 10*10*k*k + 10  (W + B initializers)
  memory = 0               (Conv output is named "output", excluded by scorer)
  cost   = 100*k*k + 10    (k=3→910, k=5→2510, k=7→4910)
"""

from __future__ import annotations

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from scipy.optimize import linprog

from . import builders
from .arc import Example

_NUM = 10
_GMAX = 30


def _extract_windows(examples: list[Example], k: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, y): X[i] = flattened one-hot k×k patch, y[i] = output color (-1=OOB)."""
    pad = k // 2
    Xs: list[np.ndarray] = []
    ys: list[int] = []
    for e in examples:
        gi = np.array(e.input, dtype=np.int64)
        go = np.array(e.output, dtype=np.int64)
        if gi.shape != go.shape or max(gi.shape) > _GMAX:
            continue
        H, W = gi.shape
        oh = np.zeros((_NUM, _GMAX + 2 * pad, _GMAX + 2 * pad), dtype=np.float32)
        for r in range(H):
            for c in range(W):
                oh[gi[r, c], r + pad, c + pad] = 1.0
        for r in range(_GMAX):
            for c in range(_GMAX):
                Xs.append(oh[:, r : r + k, c : c + k].reshape(-1))
                ys.append(int(go[r, c]) if r < H and c < W else -1)
    D = _NUM * k * k
    if not Xs:
        return np.empty((0, D), dtype=np.float32), np.empty(0, dtype=np.int64)
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.int64)


def _dedup(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Deduplicate window patches by content hash.

    Returns None if any patch appears with conflicting labels (non-k-local task).
    """
    seen: dict[bytes, tuple[np.ndarray, int]] = {}
    for xi, yi in zip(X, y, strict=True):
        key = xi.tobytes()
        iy = int(yi)
        if key in seen:
            if seen[key][1] != iy:
                return None
        else:
            seen[key] = (xi, iy)
    if not seen:
        return np.empty((0, X.shape[1]), dtype=np.float32), np.empty(0, dtype=np.int64)
    xs = np.array([v[0] for v in seen.values()], dtype=np.float32)
    ys = np.array([v[1] for v in seen.values()], dtype=np.int64)
    return xs, ys


def _lp_binary_sep(
    X: np.ndarray, pos_mask: np.ndarray
) -> tuple[np.ndarray, float] | None:
    """Find w, b (float64) separating pos from neg with margin >=1.

    Constraints: X[pos] @ w + b >= 1,  X[~pos] @ w + b <= -1.
    Returns (w, b) if feasible and verified, else None.
    """
    D = X.shape[1]
    if not pos_mask.any():
        return np.zeros(D, dtype=np.float64), -1.0

    X64 = X.astype(np.float64)
    X_p = X64[pos_mask]
    X_n = X64[~pos_mask]

    # Variables: [w(D), b(1)]
    # pos: -X_p @ w - b <= -1
    # neg:  X_n @ w + b <= -1
    blocks = []
    rhs_parts = []
    if X_p.shape[0]:
        blocks.append(np.hstack([-X_p, -np.ones((X_p.shape[0], 1))]))
        rhs_parts.append(-np.ones(X_p.shape[0]))
    if X_n.shape[0]:
        blocks.append(np.hstack([X_n, np.ones((X_n.shape[0], 1))]))
        rhs_parts.append(-np.ones(X_n.shape[0]))

    A_ub = np.vstack(blocks)
    b_ub = np.concatenate(rhs_parts)
    c = np.zeros(D + 1)
    res = linprog(
        c, A_ub=A_ub, b_ub=b_ub, bounds=[(None, None)] * (D + 1), method="highs"
    )
    if res.status != 0:
        return None

    w, b = res.x[:D], float(res.x[D])
    if X_p.shape[0] and (X_p @ w + b).min() <= 0:
        return None
    if X_n.shape[0] and (X_n @ w + b).max() >= 0:
        return None
    return w, b


def try_linsep_conv(
    examples: list[Example], k: int
) -> tuple[np.ndarray, np.ndarray] | None:
    """Try LP separator for each of 10 channels.

    Returns (W[10,10,k,k], b[10]) float32 if all channels separable, else None.
    """
    X, y = _extract_windows(examples, k)
    if X.shape[0] == 0:
        return None
    deduped = _dedup(X, y)
    if deduped is None:
        return None
    Xd, yd = deduped

    W_rows: list[np.ndarray] = []
    B_vals: list[float] = []
    for j in range(_NUM):
        pos = yd == j
        result = _lp_binary_sep(Xd, pos)
        if result is None:
            return None
        w, b = result
        W_rows.append(w)
        B_vals.append(b)

    W = np.stack(W_rows, axis=0).astype(np.float32).reshape(_NUM, _NUM, k, k)
    B = np.array(B_vals, dtype=np.float32)
    return W, B


def build_linsep_conv(examples: list[Example], k: int) -> onnx.ModelProto | None:
    """Build single-Conv ONNX if k-local LP separation succeeds, else None."""
    result = try_linsep_conv(examples, k)
    if result is None:
        return None
    W, B = result
    W_init = numpy_helper.from_array(W, "W")
    B_init = numpy_helper.from_array(B, "B")
    node = helper.make_node(
        "Conv",
        ["input", "W", "B"],
        ["output"],
        kernel_shape=[k, k],
        pads=[k // 2] * 4,
        strides=[1, 1],
    )
    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, builders.GRID_SHAPE)
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, builders.GRID_SHAPE)
    graph = helper.make_graph([node], f"linsep_k{k}", [x], [y], [W_init, B_init])
    return helper.make_model(
        graph, ir_version=builders.IR_VERSION, opset_imports=builders.OPSET
    )
