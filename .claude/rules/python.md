---
paths:
  - "**/*.py"
  - "**/*.ipynb"
---

# Python Rules

**General Python rules** for editing `.py` / `.ipynb` files in this repository. Auto-loaded across every region that contains Python code (`backend/`, `pipeline/`, tests, notebooks, etc.).

`pyproject.toml` / `uv.lock` / `.python-version` sit at `backend/` root, and `uv run ...` commands are expected to execute from `backend/`.

## Backend Module Architecture (`backend/src/**`)

`backend/src/` holds the shared **development** libraries (not part of the Kaggle submission).
Each subdirectory is exposed as a top-level package via `[tool.hatch.build.targets.wheel] packages`, so imports are bare (`from submit...`):

```
backend/src/
  submit/          Kaggle submission packaging / validation / quota (python -m submit)
```

> The PTCG template also had `dataset/`, `evaluate/`, `simulate/`, `gpu/`, `utils/` and a
> `pipeline/<family>/case<N>/` agent tree. These were removed (NeuroGolf has no game engine,
> self-play, or agent families). NeuroGolf-specific modules — an ARC-AGI dataset loader and
> the per-task ONNX build/validation logic — will be re-added per `docs/develop/MIGRATION.md`.

### Module Design Principles

- Each module owns a single responsibility
- Express inter-module dependencies via explicit imports
- Keep the ONNX build/export path loosely coupled so per-task solvers can be swapped
- Keep the submission artifact minimal (ONNX files only); avoid bundling dev-only code

## General Principles

- Comply with PEP 8 and write Pythonic code
- Methods should have referential transparency and idempotency
- Return early and keep nesting shallow
- Follow the Single Responsibility Principle
- Keep third-party libraries to a minimum
- Always import at the top of the file
- No backward compatibility concerns — remove unnecessary code
- Minimize lines of code
- Avoid excessive commenting and logging
- Don't implement temporary measures — make fundamental changes
- 200-400 lines per file typical, 800 max
- NEVER mutate objects — always create new instances

## Type Hints & Naming

- Use Python 3.13 standard types (`list[str]`, `str | None` instead of `List`, `Optional`)
- Avoid `Any` type, `cast`, and `type: ignore` comments
- Type hints for all function arguments and return values
- `snake_case` (functions/variables), `PascalCase` (classes), `UPPER_SNAKE_CASE` (constants)

```python
# GOOD
def select_action(obs: Observation) -> list[Action]:
    ...

# BAD
def select_action(obs) -> Any:
    ...
```

## ONNX Golf Conventions

NeuroGolf は ARC-AGI のグリッド変換を **タスクごとに 1 つの小さな ONNX ネット**で厳密に解く競技。スコアは `max(1, 25 - ln(cost))`（`cost` = パラメータ数 + メモリ + MAC 数）で、**正答を保ったまま cost を最小化**するのが本質。詳細仕様は [`docs/competition/abstract.md`](../../docs/competition/abstract.md)。

- **正答が最優先**: train / test / arc-gen の全ペアを全セル完全一致で構築できて初めて得点対象。まず厳密に解き、その後に cost を削る
- **cost 最小化**: 余分なパラメータ・チャネル・演算を持たせない。同じ変換を表せるなら最小構成を選ぶ
- **静的形状必須**: 全テンソル / パラメータは statically-defined shapes（動的形状不可）
- **禁止演算を使わない**: `Loop` / `Scan` / `NonZero` / `Unique` / `Script` / `Function`
- **ファイルサイズ上限**: 各 `taskNNN.onnx` は ≤ 1.44 MB
- **I/O 形状**: 入力 `[1, 10, 30, 30]`（10 色 one-hot, 30×30）、出力も同形式のチャネル表現
- 重い前計算・定数（カーネル, マスク等）はグラフ初期化時に 1 度だけ構築する
- マジックナンバー（`GRID_MAX=30`・`NUM_COLORS=10`・`MAX_ONNX_BYTES` 等）は定数として宣言する
- グリッド変換ロジックは NumPy / テンソル演算でまとめて表現し、Python ループを避ける
- Use `pathlib.Path` for file paths

## Error Handling

- Define appropriate exception classes
- Output structured logs
- Use exception chaining (`raise ... from e`)

```python
class ObservationParseError(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(f"Failed to parse observation: {reason}")
```

## Logging

- Use structured logging with JSON format
- Exclude sensitive information (API tokens)
- Use `logging.getLogger(__name__)`
- NEVER use `print()` for logging (do not pollute stdout, even from the submission entrypoint)

## Lint/Formatting

```bash
uv run ruff format .
uv run ruff check . --fix
uv run mypy .
```

## Code Quality Checklist

- [ ] Code is readable and well-named
- [ ] Functions are small (<50 lines), files are focused (<800 lines)
- [ ] No deep nesting (>4 levels)
- [ ] Proper error handling with exception chaining
- [ ] No `print()` statements — use structured logging
- [ ] No hardcoded values
- [ ] No mutation (immutable patterns used)
- [ ] Type hints for all functions (no `Any`)
- [ ] `ruff format`, `ruff check`, `mypy` pass
