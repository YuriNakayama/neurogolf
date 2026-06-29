import sys

import numpy as np
import onnxruntime as ort

t = int(sys.argv[1])
N = int(sys.argv[2]) if len(sys.argv) > 2 else 400
so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
floor = ort.InferenceSession(f"../data/output/onnx/task{t:03d}.onnx", so)
cand = ort.InferenceSession(f"scratchpad/cand_t{t:03d}.onnx", so)
rng = np.random.default_rng(0)
# infer color palette from training windows file via sep npz? just use 0..9
disagree = 0
checked = 0
for _ in range(N):
    H = int(rng.integers(1, 31))
    W = int(rng.integers(1, 31))
    ncol = int(rng.integers(2, 7))
    pal = rng.choice(10, size=ncol, replace=False)
    g = pal[rng.integers(0, ncol, size=(H, W))]
    inp = np.zeros((1, 10, 30, 30), dtype=np.float32)
    for r in range(H):
        for c in range(W):
            inp[0, g[r, c], r, c] = 1
    of = (floor.run(["output"], {"input": inp})[0] > 0).astype(int)
    oc = (cand.run(["output"], {"input": inp})[0] > 0).astype(int)
    checked += 1
    if not np.array_equal(of, oc):
        disagree += 1
print(f"t{t:03d}: random grids checked={checked} disagree={disagree}")
