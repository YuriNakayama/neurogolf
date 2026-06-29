import json
import sys

import numpy as np

LAKE = "../data/lake/neurogolf-2026"
t = int(sys.argv[1])
task = json.load(open(f"{LAKE}/task{t:03d}.json"))
for key in ["train", "test"]:
    print(
        f"--- {key} ({len(task.get(key, []))} ex), "
        f"arc-gen={len(task.get('arc-gen', []))} ---"
    )
for ex in task["train"][:2]:
    gi = np.array(ex["input"])
    go = np.array(ex["output"])
    print("IN", gi.shape, "colors", sorted(set(gi.flatten().tolist())))
    print(gi)
    print("OUT colors", sorted(set(go.flatten().tolist())))
    print(go)
    print("changed cells", int((gi != go).sum()), "/", gi.size)
