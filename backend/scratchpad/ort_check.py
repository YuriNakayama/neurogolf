import json
import sys

import numpy as np
import onnxruntime as ort

LAKE = "../data/lake/neurogolf-2026"
t = int(sys.argv[1])
k = int(sys.argv[2])
pad = k // 2
task = json.load(open(f"{LAKE}/task{t:03d}.json"))
so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
sess = ort.InferenceSession(f"scratchpad/cand_t{t:03d}.onnx", so)
ex = task["train"][0]
gi = np.array(ex["input"])
go = np.array(ex["output"])
H, W = gi.shape
inp = np.zeros((1, 10, 30, 30), dtype=np.float32)
for r in range(H):
    for c in range(W):
        inp[0, gi[r, c], r, c] = 1
out = sess.run(["output"], {"input": inp})[0]
print("out shape", out.shape)
pred = (out[0, :, :H, :W] > 0).astype(int)
tgt = np.zeros((10, H, W), dtype=int)
for r in range(H):
    for c in range(W):
        tgt[go[r, c], r, c] = 1
print("cell_err in-grid", (pred != tgt).sum())
# also check borders / full 30x30 predicted ones outside grid
full = (out[0] > 0).astype(int)
print("predicted-positive cells total", full.sum(), "; in-grid target ones", tgt.sum())
print("out-of-grid positive", full[:, :, :].sum() - pred.sum())
