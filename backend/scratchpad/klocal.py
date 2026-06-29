"""Find minimal k such that out[r,c] = f(in[r-k//2..,c-..]) deterministically
across ALL examples. High-cost + small-k = cheap single-conv redesign target."""

import csv
import json

import numpy as np

LAKE = "../data/lake/neurogolf-2026"


def grids(task):
    out = []
    for key in ["train", "test", "arc-gen"]:
        for ex in task.get(key, []):
            gi = np.array(ex["input"])
            go = np.array(ex["output"])
            if gi.shape == go.shape and max(gi.shape) <= 30:
                out.append((gi, go))
    return out


def klocal(pairs, k):
    """Return True if out cell is deterministic fn of k*k window (color ints)."""
    pad = k // 2
    table = {}
    for gi, go in pairs:
        gp = np.pad(gi, pad, constant_values=-1)
        H, W = gi.shape
        for r in range(H):
            for c in range(W):
                win = gp[r : r + k, c : c + k].tobytes()
                o = int(go[r, c])
                if win in table and table[win] != o:
                    return False
                table[win] = o
    return True


def main():
    costs = {}
    with open("scratchpad/rank.csv") as f:
        for row in csv.DictReader(f):
            costs[int(row["task"])] = int(row["cost"]) if row["cost"] else 0
    targets = [t for t, c in costs.items() if c >= 900]
    results = []
    for t in sorted(targets):
        with open(f"{LAKE}/task{t:03d}.json") as f:
            task = json.load(f)
        pairs = grids(task)
        total = sum(
            1 for key in ["train", "test", "arc-gen"] for _ in task.get(key, [])
        )
        if not pairs or len(pairs) < total:
            results.append((t, costs[t], None, "not-all-sameshape"))
            continue
        mink = None
        for k in (1, 3, 5, 7):
            if klocal(pairs, k):
                mink = k
                break
        results.append((t, costs[t], mink, f"npairs={len(pairs)}"))
    import math

    results.sort(key=lambda x: -x[1])
    cands = []
    for t, c, mink, note in results:
        flag = ""
        if mink is not None:
            newcost = 100 * mink * mink + 10  # single conv, mem 0
            if newcost < c:
                gain = (25 - math.log(newcost)) - (25 - math.log(c))
                flag = f"  *** k={mink} convcost={newcost} GAIN+{gain:.3f}"
                cands.append((t, mink, c, newcost, gain))
        print(f"t{t:03d} cost={c} mink={mink} {note}{flag}")
    with open("scratchpad/klocal_cands.csv", "w") as fh:
        fh.write("task,k,cost,convcost,gain\n")
        for t, mk, c, nc, g in sorted(cands, key=lambda x: -x[4]):
            fh.write(f"{t},{mk},{c},{nc},{g:.4f}\n")
    print(
        f"\n{len(cands)} candidates; total potential gain "
        f"{sum(x[4] for x in cands):.2f}"
    )


if __name__ == "__main__":
    main()
