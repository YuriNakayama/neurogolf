---
paths:
  - "backend/pipeline/**"
---

# Pipeline Case Rules (`backend/pipeline/**`)

Conventions for working in `backend/pipeline/case<N>/` — directory layout, case independence, and how a case relates to the official scorer. NeuroGolf is an **offline ONNX golf** competition: a case either *builds* per-task ONNX (`taskNNN.onnx`) or *post-processes* existing bundles. There is no opponent, self-play, rating ladder, or long-running GPU training.

Path notation below anchors at `backend/` (`pipeline/case<N>/...`). `uv run ...` / `dev/submit ...` run with `backend/` as the working directory.

## What a case is

Each `case<N>` is one self-contained approach to producing or obtaining a submission bundle (a directory of `taskNNN.onnx`, or a ready `submission.zip`). Cases compose: a build-case emits a bundle, a reproduce-case fetches and locks a known-good one. Current cases:

| Case | Kind | Entry | Role |
|---|---|---|---|
| `case0` | build | `python -m pipeline.case0 build` | Try each solver against every task; save `taskNNN.onnx` for the ones solved **exactly** (verified via the official-scorer mirror). The output dir is a bundle. |
| `case1` | reproduce | `python -m pipeline.case1 submit` | Reproduce a known-good high-score bundle: fetch a public notebook's 400-task `submission.zip`, lock it by **byte count + SHA256**, and submit until a target Public Score is reached. Never builds a new model — a baseline that guarantees a known LB floor. (From `boristown/agi-neural-golf-visualization-baseline`, LB 7159.44.) |

Case numbers start at 0. A case's `README.md` records its origin, strategy, and any known score.

## Directory layout

```
case<N>/
├── __init__.py          # package marker + public re-exports
├── __main__.py          # `python -m pipeline.case<N>` entrypoint (typer + rich)
├── <logic>.py           # method-specific modules (e.g. case0/build.py, case1/reproduce.py, case1/submit_loop.py)
└── README.md            # purpose, origin (notebook URL etc.), strategy, latest known score
```

- **`__main__.py` is the entrypoint** — a typer app mirroring `src/submit/__main__.py` (typer `@app.command`, rich `Console`, Japanese help text, `from __future__ import annotations`). Keep it thin: argument parsing + orchestration, no business logic.
  - typer requires `Option(...)` in argument defaults, which trips ruff `B008`. Add a per-file ignore in `pyproject.toml` `[tool.ruff.lint.per-file-ignores]`, as done for `src/submit/__main__.py` and `pipeline/case1/__main__.py`. With multiple commands, add an `@app.callback()` so typer keeps subcommand routing (`... verify ...` / `... submit ...`).
- **Logic lives in sibling modules** under `case<N>/`, split by responsibility (one module = one job). Re-export the public API from `__init__.py`.
- `pipeline` is imported via the working directory (it is **not** in `[tool.hatch.build.targets.wheel].packages`), so `python -m pipeline.case<N>` resolves naturally from `backend/`.

## What goes in `src/` vs the case

`backend/src/` holds **only what is certain to be shared across cases and will not change when the method changes**. Method-specific logic stays in the case.

- ✅ in `src/`: the **official-scorer mirror** `src/evaluate` (`audit_one` / `audit_dir` / `convert_to_numpy` — the competition's fixed scoring, reproduced from `neurogolf_utils.py`; depends only on `onnx` / `onnxruntime` / `numpy`, nothing else in the repo).
- ❌ in the case: solver families (case0), reproduce fetch/digest-lock and the submit-until-target loop (case1), bundle resolution, the CLI — anything a different approach would rewrite.

When unsure, keep it in the case. Promoting a helper into `src/` later is cheap; un-coupling a leaky `src/` abstraction is not.

## Cross-case independence (mandatory)

Cases must remain self-contained. **Never import from another case** (`from pipeline.case0... import` inside `case1/`). If two cases need the same helper, **copy it** — duplication is preferred to cross-case coupling. The only shared import is `backend/src/`, and a case should depend only on the `src/` packages that import cleanly and are stable, so it never inherits breakage from unrelated parts of `src/`.

## Correctness & cost (the scoring contract)

A task earns `max(1, 25 - ln(cost))` **only when functionally correct** — every cell of every example pair (`train` / `test` / `arc-gen`) matches exactly. `cost = params + memory_bytes` (MACs no longer contribute). **Solve exactly first, then shrink cost.** A wrong file scores the same zero as a missing file but risks a malformed bundle, so **omitting a task is safer than shipping a wrong ONNX**.

- Always judge correctness and cost through `src/evaluate` (`audit_one`), never a bespoke check — it mirrors the official validator (sanitize → profiled session → `calculate_params` / `calculate_memory`), so a case must not reimplement scoring.
- Real task data lives at `data/lake/neurogolf-2026/` (`task001.json … task400.json` + the official `neurogolf_utils/neurogolf_utils.py`). Pass it via `--task-dir`. See `.claude/rules/data.md`.
- ONNX golf constraints (static shapes, banned ops `Loop`/`Scan`/`NonZero`/`Unique`/`Script`/`Function`/`Compress`, ≤ 1.44 MB, I/O `[1,10,30,30]`) are enforced by the scorer/submit path — see `.claude/rules/python.md` (ONNX Golf Conventions).

## Experiment discipline

**Change only what you are testing; hold everything else fixed.** State up front what a run tests, vary only that, and compare via the official-scorer mirror (`audit_dir` totals). If two things change at once, the score delta is unattributable. Score is the only currency — there is no Kaggle relative metric, opponent pool, or win-rate to interpret; a task's points are deterministic given its ONNX.

## Anti-patterns

- Cross-case imports — violates case independence (copy the helper instead).
- Putting method-specific logic (solvers, reproduce fetch/digest-lock, submit-until-target loop) in `src/` — `src/` is for cross-case invariants only.
- A hand-rolled correctness or cost check instead of `src/evaluate` — drifts from the official scorer.
- Shipping a not-exactly-correct ONNX to chase partial credit — there is none; omit the task.
- Business logic in `__main__.py`, or hardcoded paths as typer defaults without `--task-dir` / `--out` overrides (untestable in isolation).
