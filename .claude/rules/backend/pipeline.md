---
paths:
  - "backend/pipeline/**"
---

# Pipeline Experiment Rules

Conventions for working in `backend/pipeline/case*/` directories — directory layout, experiment discipline, long-running training, and evaluation. Case numbers are assigned starting from 1.


Path notation below uses `backend/` as the anchor (`pipeline/case<N>/...`). `uv run ...` / `dev/submit ...` are expected to execute with `backend/` as the working directory.

## Directory layout

### Per-case layout

```
case<N>/
├── main.py            # Kaggle entrypoint (thin wrapper — see submit.md)
├── __init__.py
├── README.md          # purpose, strategy summary, latest publicScore (if known)
└── baseline/  or  policy/   # agent body — must export `agent` callable
```

- `pipeline/case<N>/main.py` is always an entrypoint. Keep it as a thin wrapper of roughly 20 lines. Do not put business logic in it. (For the entrypoint template and import rules, see [`submit.md`](submit.md).)
- The implementation lives in subpackages under `pipeline/case<N>/<package>/` (e.g. `baseline/`, `policy/`). Maintain the hierarchy for readability and maintainability.
- Auxiliary directories such as `evaluation/`, `configs/`, `eda/`, `notebook/` may sit under `pipeline/case<N>/`. They are harmless on Kaggle as long as they are not imported from `main.py`, but the tar.gz size should still be kept small.

### Cross-case independence rule

Cases must remain self-contained. **Never** import from another case (`from pipeline.rulebase.case2.* import ...` inside `case1/`). When the same helper is needed in multiple cases, copy it into each — duplication is preferred to cross-case coupling. Shared development utilities live in `backend/src/` (e.g. `src/evaluate/`, `src/utils/repo_root.py`) and are imported only from `evaluation/` / `training/` (excluded from submission tar).

## Experiment discipline

**Change only what you are testing; hold everything else fixed.** An experiment exists to test one thing (a parameter, a method, a dataset, ...). Fix every other variable and vary only the subject under test, then compare. Always state up front what is being tested, and never change multiple items at once in a way that makes the result impossible to attribute — if two things changed, you cannot tell which one moved the metric.

## Anti-patterns

- Cross-case imports (`from pipeline.rulebase.case2.baseline import ...` inside `case1/`) — violates case independence.
- Changing more than the variable under test in one experiment — the result becomes impossible to attribute.
- Hardcoding paths as typer-Option defaults (`Path("data/.../foo.parquet")`) without making them overridable — use `--config`/`--out` / params.yaml hooks so the script is testable in isolation.

## Evaluation metric interpretation

Kaggle PTCG publicScore is a **relative metric** computed against other participants' submissions, and the opponent pool drifts over time, so **the same agent can produce very different publicScores depending on submission timing**. Do not use Kaggle-side numbers to judge the merit of a change. **Evaluate agents exclusively on local match results.**

