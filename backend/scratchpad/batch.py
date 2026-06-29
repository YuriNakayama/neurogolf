"""For each k-local candidate: LinearSVC separability (full-frame) -> build
single-Conv ONNX -> faithful audit. Print win table (n_fail=0 and cost<floor)."""

import csv
import json
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

sys.path.insert(0, "src")
from sklearn.svm import LinearSVC  # noqa: E402

from evaluate.scorer import audit_one  # noqa: E402

LAKE = "../data/lake/neurogolf-2026"
OUT = "scratchpad/conv_wins"


def windows(task, k):
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
    return (
        np.array([v[0] for v in seen.values()], dtype=np.float32),
        np.array([v[1] for v in seen.values()]),
    )


def separate(Xd, yd):
    D = Xd.shape[1]
    W = np.zeros((10, D), dtype=np.float64)
    b = np.zeros(10, dtype=np.float64)
    for j in range(10):
        pos = yd == j
        if not pos.any():
            b[j] = -1.0
            continue
        clf = LinearSVC(C=1e6, max_iter=30000, tol=1e-4, dual=True)
        clf.fit(Xd, pos.astype(int))
        wj, bj = clf.coef_[0], clf.intercept_[0]
        s = Xd @ wj + bj
        mp, xn = s[pos].min(), s[~pos].max()
        if mp <= xn:
            return None
        W[j] = wj
        b[j] = bj - 0.5 * (mp + xn)
    return W, b


def build(t, k, W, b):
    pad = k // 2
    Wt = W.astype(np.float32).reshape(10, 10, k, k)
    bt = b.astype(np.float32)
    node = helper.make_node(
        "Conv",
        ["input", "W", "B"],
        ["output"],
        kernel_shape=[k, k],
        pads=[pad] * 4,
        strides=[1, 1],
    )
    g = helper.make_graph(
        [node],
        f"t{t:03d}",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])],
        [numpy_helper.from_array(Wt, "W"), numpy_helper.from_array(bt, "B")],
    )
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 9
    onnx.checker.check_model(m)
    import os

    os.makedirs(OUT, exist_ok=True)
    p = f"{OUT}/task{t:03d}.onnx"
    onnx.save(m, p)
    return p


def main():
    rows = list(csv.DictReader(open("scratchpad/klocal_cands.csv")))
    only = {int(x) for x in sys.argv[1:]} if len(sys.argv) > 1 else None
    rows = [r for r in rows if only is None or int(r["task"]) in only]
    rows.sort(key=lambda r: (int(r["k"]), -float(r["gain"])))  # quick k first
    res = open("scratchpad/conv_results.csv", "a")
    for row in rows:
        t, k, fc = int(row["task"]), int(row["k"]), int(row["cost"])
        task = json.load(open(f"{LAKE}/task{t:03d}.json"))
        X, y = windows(task, k)
        Xd, yd = dedup(X, y)
        sep = separate(Xd, yd)
        if sep is None:
            line = f"t{t:03d},{k},{fc},NO,,,linsep_NO"
        else:
            p = build(t, k, *sep)
            r = audit_one(p, task, run_correctness=True)
            ok = r["n_fail"] == 0 and r["cost"] and r["cost"] < fc
            line = (
                f"t{t:03d},{k},{fc},{'WIN' if ok else 'no'},{r['cost']},"
                f"{r['n_fail']},{r['status']}"
            )
        print(line, flush=True)
        res.write(line + "\n")
        res.flush()
    res.close()


if __name__ == "__main__":
    main()
