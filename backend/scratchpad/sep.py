"""Test if a k-local task is exactly solvable by a single Conv[10,10,k,k]+bias
under the >0 per-channel decode. Per output channel j, check linear separability
of windows mapping-to-j (positive) vs not, via LinearSVC max-margin."""

import json
import sys

import numpy as np
from sklearn.svm import LinearSVC

LAKE = "../data/lake/neurogolf-2026"


def windows(task, k):
    """Full 30x30 frame: in-grid cells -> target color; out-of-grid -> -1 (no
    channel fires). Mirrors the real audit which compares the full [10,30,30]."""
    pad = k // 2
    Xs, ys = [], []
    for key in ["train", "test", "arc-gen"]:
        for ex in task.get(key, []):
            gi = np.array(ex["input"])
            go = np.array(ex["output"])
            if gi.shape != go.shape or max(gi.shape) > 30:
                continue
            H, W = gi.shape
            oh = np.zeros((10, 30 + 2 * pad, 30 + 2 * pad), dtype=np.float32)
            for r in range(H):
                for c in range(W):
                    oh[gi[r, c], r + pad, c + pad] = 1.0
            for r in range(30):
                for c in range(30):
                    Xs.append(oh[:, r : r + k, c : c + k].reshape(-1))
                    ys.append(int(go[r, c]) if (r < H and c < W) else -1)
    return np.array(Xs, dtype=np.float32), np.array(ys)


def dedup(X, y):
    seen = {}
    for xi, yi in zip(X, y, strict=False):
        seen[xi.tobytes()] = (xi, yi)
    xs = np.array([v[0] for v in seen.values()], dtype=np.float32)
    ys = np.array([v[1] for v in seen.values()])
    return xs, ys


def try_k(task, k):
    X, y = windows(task, k)
    if len(X) == 0:
        return None
    Xd, yd = dedup(X, y)
    D = Xd.shape[1]
    colors = sorted(set(yd))
    W = np.zeros((10, D), dtype=np.float64)
    b = np.zeros(10, dtype=np.float64)
    for j in range(10):
        pos = yd == j
        if not pos.any():
            b[j] = -1.0  # never fire
            continue
        if pos.all():
            b[j] = 1.0  # always fire
            continue
        clf = LinearSVC(C=1e6, max_iter=200000, tol=1e-6, dual=True)
        clf.fit(Xd, pos.astype(int))
        wj = clf.coef_[0]
        bj = clf.intercept_[0]
        s = Xd @ wj + bj
        mp, xn = s[pos].min(), s[~pos].max()
        if mp <= xn:
            return (k, len(Xd), colors, False, j)  # channel j not separable
        mid = 0.5 * (mp + xn)
        W[j] = wj
        b[j] = bj - mid
    return (k, len(Xd), colors, True, W, b)


def main():
    for t in [int(x) for x in sys.argv[1:]]:
        with open(f"{LAKE}/task{t:03d}.json") as f:
            task = json.load(f)
        for k in (3, 5, 7):
            r = try_k(task, k)
            if r is None:
                print(f"t{t:03d} k={k}: no data")
                break
            if r[3]:
                _, n, colors, _, W, b = r
                np.savez(f"scratchpad/sep_t{t:03d}_k{k}.npz", W=W, b=b)
                print(f"t{t:03d} k={k}: uniq={n} colors={colors} LINSEP=YES -> saved")
                break
            else:
                print(f"t{t:03d} k={k}: uniq={r[1]} colors={r[2]} linsep=NO (ch{r[4]})")


if __name__ == "__main__":
    main()
