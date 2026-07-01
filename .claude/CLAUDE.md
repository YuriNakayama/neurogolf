# Kaggle NeuroGolf 2026

Project for **The 2026 NeuroGolf Championship** on Kaggle (IJCAI-ECAI 2026 Competitions Track). The task is **"neural golf"**: design the *smallest* neural networks (exported to ONNX) that solve ARC-AGI grid-transformation tasks. A submission is a `submission.zip` containing at most one ONNX file per task (`task001.onnx` ... `task400.onnx`). Scoring = **functional correctness** on ARC-AGI benchmarks (ARC-AGI-1 train + ARC-GEN-100K + a private suite) **plus minimizing parameter count and memory footprint**, estimated from the ONNX graph.

> ℹ️ **Migrated from a PTCG (Pokémon TCG battle-AI) scaffold.** The generic scaffolding (uv / ruff / mypy / pytest, `dev/*`, CI) is reused; PTCG-specific code (cabt engine, self-play, agent families, GPU/infra) was removed. The NeuroGolf core is **implemented**: ARC-AGI loader/encoding (`src/arc`), ONNX build/cost/constraints (`src/onnxgolf`), ONNX `submission.zip` packaging + validation (`src/submit`), and identity/recolor PoC solvers (`src/solvers`) — all green under `dev/test-bot`. Remaining work (real ARC data download, richer solvers) is tracked in [`docs/develop/MIGRATION.md`](../docs/develop/MIGRATION.md). Full competition spec: [`docs/competition/abstract.md`](../docs/competition/abstract.md).

## Task Overview

NeuroGolf is an **offline "neural golf"** competition: for each ARC-AGI grid-transformation task, build the *smallest* ONNX network that solves it exactly. There is no opponent, game engine, or rating ladder.

- **Input** (per task): an ARC grid encoded as a tensor `[BATCH=1, CHANNELS=10, HEIGHT=30, WIDTH=30]` (10-color one-hot, grids 1×1–30×30, out-of-border cells zero-hot).
- **Output**: the transformed grid in the same channel form (correct color channel = 1, others = 0; out-of-border = all 0).
- **Correct** = every cell of every example pair (train / test / arc-gen) matches exactly.
- **Score per task**: `max(1, 25 - ln(max(1, cost)))`, where `cost = params + memory_bytes`. Solve exactly first, then shrink `cost`.

## Technology Stack

- **Language**: Python 3.13
- **Submission format**: ONNX (per-task networks). Build/validate with `onnx` / `onnxruntime` (in `src/onnxgolf` + `src/submit`).
- **Kaggle**: `kaggle` CLI for submission; `dvc[s3]` optional for data/model management (undecided).
- **Testing**: Pytest, Ruff, Mypy
- **Package Management**: uv

## Folder Structure

```
backend/                Python implementation (pyproject.toml / uv.lock live here)
  src/                  Shared dev libs (submit, dataset, evaluate, simulate, utils, gpu/{vast,runpod,kaggle})
  pipeline/             Agent families only: rulebase / imitation / reinforce (reinforce/_bench は dev-only ベンチ)
  tests/                Pytest unit tests
data/                   4 layers (lake / processed / mart / output) (gitignored, DVC-managed)
dev/                    Development scripts (each cd's into backend and runs uv internally)
infra/                  Terraform/AWS: ECS 上の Claude 自律改善ループ + 15分毎 Kaggle submit (infra/README.md)
docs/
  develop/              
```

`uv run ...` is expected to run under `backend/`. From the repo root, use `dev/*` or `cd backend` first.

## Submission

A submission is a **`submission.zip`** containing **at most one ONNX file per task**, named `task001.onnx` … `task400.onnx`. Only solved tasks need be included. Constraints: each `.onnx` ≤ **1.44 MB**; all tensor/parameter shapes statically defined; **disallowed ONNX ops**: `Loop`, `Scan`, `NonZero`, `Unique`, `Script`, `Function`. Build/validate via `uv run python -m submit` (or `dev/submit`): it collects `taskNNN.onnx` from `--onnx-dir`, runs `onnx.checker` + the constraint/cost checks (`src/onnxgolf`, `src/submit/validator.py`), and zips them (`src/submit/packager.py`). 
Submissions can be made up to 100 times a day without verification.

## Glossary

| Term | Description |
|------|-------------|
| NeuroGolf | The 2026 NeuroGolf Championship — Kaggle / IJCAI-ECAI 2026 competition: solve ARC-AGI tasks with minimal ONNX networks |
| ARC-AGI | Abstraction and Reasoning Corpus — grid-transformation puzzles. Tasks come from the ARC-AGI-1 public training subset |
| ARC-GEN-100K | Procedurally generated ARC benchmark ([google/arc-gen](https://github.com/google/arc-gen)) used as an extra validation set per task |
| cost | `params + memory_bytes` of a task's ONNX net; MACs do not contribute; smaller = higher score |
| task score | `max(1, 25 - ln(max(1, cost)))`, awarded only when the net solves the task exactly |
| grid encoding | `[1, 10, 30, 30]` tensor — 10-color one-hot, out-of-border cells all-zero |

## Scoring

Per-task score is `max(1, 25 - ln(max(1, cost)))` and is only earned when the network is **functionally correct** — every cell of every example pair (train / test / arc-gen, plus a private benchmark) matches the expected output exactly. Total score is the sum across solved tasks. The competition rewards solving exactly while minimizing `cost` ("golf").

Priority for cost work is based on expected point gain: `delta_points = ln(old_cost / new_cost)`. Do not rank tasks by absolute cost alone. After `n_fail=0`, prefer candidates with large relative reduction, especially medium-to-high cost graphs where a whole intermediate tensor, dtype boundary, branch, or table can be removed. Low-cost simple tasks and many high-cost hard tasks are often already near-optimal, so absolute size is only a tie-breaker.

> ⚠️ **得点向上は独自実装を主軸とする。** harvest（公開ノートブックの取込）は per-task 最小値が
> 上限であり、上位入賞には独自ネットが必須。harvest は補助手段に過ぎず、**harvest 単独サイクルは
> 禁止**（各サイクルは独自実装の試行を1件以上含む）。詳細は `.claude/rules/backend/strategy.md`。

## Rules

| Rule file | Auto-loaded for | When to read manually |
|-----------|----------------|----------------------|
| `.claude/rules/python.md` | `**/*.py` | Python conventions (type hints, naming, error handling, lint). Note: the "Agent Performance Conventions" section is inherited PTCG content pending rework — see `docs/develop/MIGRATION.md` |
| `.claude/rules/data.md` | `data/**` | data/ 4-layer structure (lake/processed/mart/output). Note: selfplay/kaggle_episodes layer descriptions are inherited PTCG content pending rework |
| `.claude/rules/backend/strategy.md` | `backend/**` | **改善戦略（最優先）**: 独自実装を第一義務とし harvest 単独サイクルを禁止。サイクル構造（floor全監査→独自net→faithful検証→submit）、独自実装3手法、「不可能」判定の厳格基準、result doc の `independent_impl:` 必須欄 |
| `.claude/rules/backend/submit.md` | `backend/src/submit/**`, `dev/submit` | Kaggle submit conventions: zip must be named `submission.zip`, use the kaggle CLI path (not the SDK), LB-gated adoption, `dev/submit` usage |
| `.claude/rules/backend/pipeline.md` | `backend/pipeline/**` | `pipeline/case<N>/` layout, case independence, relation to the official scorer |
| `.claude/rules/backend/tests.md` | `backend/tests/**` | Pytest conventions, unit/integration/e2e classification |
| `.claude/rules/security.md` | Always loaded | Commits, secrets, CI/CD |

> The pipeline / command / docs / infra rules from the PTCG template were removed in cleanup (their subjects — agent families, GPU command catalog, experiment workflow, Terraform — no longer exist here). NeuroGolf ONNX-submit conventions are re-authored in `.claude/rules/backend/submit.md`; remaining pipeline conventions are tracked per `docs/develop/MIGRATION.md`.

## Response Language And Interface

- Answer user questions concisely, organizing the response as a table, chart, list, short sentence, ASCII art, or similar structured format.
- Keep user-facing replies under 800 characters, excluding tables, charts, code blocks, and ASCII art (which can exceed the limit when needed).
- Use the `AskUserQuestion` tool when asking questions to the user
- Internal reasoning, tool calls, and intermediate notes: English.
- User-facing output (final replies, reports, summaries): Japanese.(全てのユーザー向けの出力は日本語で行うこと)
