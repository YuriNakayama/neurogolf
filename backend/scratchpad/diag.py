import json
import sys

import numpy as np

LAKE = "../data/lake/neurogolf-2026"
t = int(sys.argv[1])
k = int(sys.argv[2])
d = np.load(f"scratchpad/sep_t{t:03d}_k{k}.npz")
W = d["W"]
b = d["b"]
pad = k // 2
task = json.load(open(f"{LAKE}/task{t:03d}.json"))


def conv(oh, Wm, bm):
    # oh [10,H+2p,W+2p]; returns [10,H,W] scores
    H = oh.shape[1] - 2 * pad
    Wd = oh.shape[2] - 2 * pad
    out = np.zeros((10, H, Wd), dtype=Wm.dtype)
    Wr = Wm.reshape(10, 10, k, k)
    for r in range(H):
        for c in range(Wd):
            win = oh[:, r : r + k, c : c + k]
            out[:, r, c] = (Wr * win).sum(axis=(1, 2, 3)) + bm
    return out


for dt, name in [(np.float64, "f64"), (np.float32, "f32")]:
    Wm = W.astype(dt)
    bm = b.astype(dt)
    cellerr = 0
    cells = 0
    exfail = 0
    for key in ["train", "test", "arc-gen"]:
        for ex in task.get(key, []):
            gi = np.array(ex["input"])
            go = np.array(ex["output"])
            if gi.shape != go.shape or max(gi.shape) > 30:
                continue
            H, Wd = gi.shape
            oh = np.zeros((10, H + 2 * pad, Wd + 2 * pad), dtype=dt)
            for r in range(H):
                for c in range(Wd):
                    oh[gi[r, c], r + pad, c + pad] = 1
            s = conv(oh, Wm, bm)  # [10,H,Wd]
            pred = (s > 0).astype(int)
            tgt = np.zeros((10, H, Wd), dtype=int)
            for r in range(H):
                for c in range(Wd):
                    tgt[go[r, c], r, c] = 1
            e = (pred != tgt).sum()
            cellerr += e
            cells += H * Wd * 10
            if e > 0:
                exfail += 1
    print(f"{name}: cell_err={cellerr}/{cells} ex_fail={exfail}")
