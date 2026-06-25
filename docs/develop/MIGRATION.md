# NeuroGolf 2026 移行 TODO（PTCG スキャフォールド → NeuroGolf）

このリポジトリは Kaggle **PTCG**（ポケモンカード対戦 AI）コンペ用スキャフォールドを雛形として複製したものです。
現時点で完了しているのは **コピー + 識別子リネーム + スキャフォールドの健全化** までで、
ポケカ固有のドメインロジックは**そのまま残置**されています。

NeuroGolf は PTCG と**全く別ジャンル**のコンペです:

| | PTCG（雛形元） | NeuroGolf 2026（本コンペ） |
|---|---|---|
| 種別 | シミュレーション / ゲーム AI 対戦 | ARC-AGI を ONNX で最小規模に解く "neural golf"（IJCAI-ECAI 2026） |
| 提出物 | `tar.gz`（`main.py` + `deck.csv` + エンジン） | `submission.zip`（`task001.onnx`〜`task400.onnx`、1 タスク 1 ファイル） |
| エンジン | `cabt`（`kaggle-environments`） | なし（オフライン。対戦相手・ラダー無し） |
| 評価 | 自動対戦 Elo ラダー | ARC-AGI ベンチでの機能的正答 + パラメータ数 / 計算量 / メモリ最小化（ONNX グラフから推定） |
| データ | self-play ログ + Kaggle リプレイ | ARC-AGI-1 公開 train + ARC-GEN-100K + 非公開ベンチ |
| エージェント I/O | `agent(obs_dict) -> list[int]` | なし（タスクごとに ONNX グラフを生成） |

このため、以下のドメインロジックは**最終的に作り替え / 削除が必要**です。

---

## ✅ 完了済み

### 識別子リネーム（リポジトリ生成時）
- `ptcg`→`neurogolf` / `PTCG`→`NeuroGolf` / `PTCG_`→`NEUROGOLF_`（全 text ファイル）
- Kaggle slug `neurogolf-2026`（`backend/src/submit/kaggle_api.py` `COMPETITION`）
- `pyproject.toml` の `name` / `description`、README / CLAUDE.md の識別子ヘッダ + 移行バナー
- `bot/` → `backend/` 参照修正（`dev/*` + `.github/workflows/{ci,submit}.yml` + `submit/packager.py`）

### PTCG 固有コード・関連物の削除（クリーンアップ pass）
ユーザー判断により、NeuroGolf に無関係なポケカ/対戦/GPU/infra 関連を**削除済み**:

- **src**: `backend/src/{simulate,gpu,utils,dataset,evaluate}` を削除（残るのは `backend/src/submit/` のみ）
- **pipeline**: `backend/pipeline/{rulebase,imitation,reinforce}` の全ケースを削除（空の `pipeline/__init__.py` + `.submitignore` のみ残置 = 将来の ONNX pipeline 置き場）
- **infra**: `infra/`（Terraform）丸ごと削除
- **tests**: 削除 src に対応する全テスト（gpu / simulate / dataset / evaluate / pipeline cases）を削除。残るのは submit テストのみ（30 件、全 pass）
- **dev**: `dev/{dvc,vast,runpod,scrape,simulate,runtime-build,kaggle,loop}` 削除（残: setup / format / lint / test-bot / submit / create-worktree / delete-worktree / sync-data）
- **依存**: 削除コード専用だった依存を `pyproject.toml` から除去（torch / jax / cuda extras 機構 / vastai / runpod / numpy / pandas / polars / pyarrow / pydantic / psutil / scikit-learn / equinox / optax / kaggle-environments）。残: `kaggle` / `dvc[s3]` / `pyyaml` / `rich` / `typer` / `python-dotenv`。`uv.lock` 再生成済み（.venv 2GB→235M）
- **設定/スキル/docs**: 不要スキル（`replay-viewer` / `experiment*` / `python-to-jax` / `dvc` / `deploy` / `e2e-fix` / `playwright` / frontend 系）、不要 agents（`gpu-handler` / `e2e-runner` / `database-reviewer` / `solution-researcher`）、削除済み対象を解説する rules（`infra.md` / `command.md` / `docs.md` / `backend/{submit,pipeline,tests}.md`）、空 `docs/competition/`、旧ミラー `.agents/` `.codex/` を削除
- **検証**: `dev/lint`（ruff + mypy, 21 files）✅ / `dev/test-bot`（30 passed）✅

### NeuroGolf 本実装（実装 pass）
PTCG ドメインを置き換える NeuroGolf 中核を実装済み。`dev/test-bot` **70 passed**・ruff/mypy 全 green（44 files）:

- **依存追加**: `numpy` / `onnx` / `onnxruntime` を `pyproject.toml` に追加し `uv.lock` 再生成（#3 依存）
- **`backend/src/arc/`（新規）— ARC-AGI 基盤**:
  - `types.py`: `Grid` / `Example` / `Task`（frozen dataclass、`all_examples()` = train+test+arc_gen）
  - `encoding.py`: `encode_grid` / `decode_grid`（`[1,10,30,30]` one-hot ⇄ グリッド、枠外 zero-hot、定数 `NUM_COLORS=10` / `GRID_MAX=30`）
  - `loader.py`: ARC task JSON パース + `.arcgen.json` サイドカー + `taskNNN` id 採番
- **`backend/src/onnxgolf/`（新規）— ビルド/スコア基盤**:
  - `constraints.py`: I/O 形状・静的形状・禁止演算（`Loop`/`Scan`/`NonZero`/`Unique`/`Script`/`Function`）・1.44MB を `check_constraints` で検証
  - `cost.py`: `estimate_cost`（params + memory + MACs、`shape_inference` ベース）+ `task_score` = `max(1, 25 - ln(cost))`
  - `build.py`: `make_io` / `build_model`（`onnx.checker` 通過）/ `save_model`（制約検証付き保存）
- **`backend/src/submit/`（ONNX 化）**: `packager.py` を tar.gz → **`submission.zip`**（`taskNNN.onnx` を flat zip、決定的メタ）に作り替え。`validator.py` を ONNX 検証（checker + 制約 + cost/score）に。`__main__.py` を `submit` / `validate` / `submissions` サブコマンドに（`--onnx-dir`、`single_file`/`case`/`.submitignore` 廃止）。`kaggle_api.py` / `history.py` / `auth.py` は流用。
- **`backend/src/solvers/`（新規）— PoC**: `identity`（0 param / 0 MAC）+ `recolor`（1×1 conv の色置換）+ `run.py`（onnxruntime で全 example 完全一致を検証）。E2E テスト `test_e2e_poc.py` で「解く → 保存 → 検証 → submission.zip → 再読込」を通し確認。
- **`dev/submit` + `.github/workflows/submit.yml`**: ONNX フロー（`--onnx-dir` / `--dry-run` / `--wait`、`validate` サブコマンド）に更新
- **ドキュメント**: `.claude/CLAUDE.md` / `rules/python.md`（ONNX golf 規約）/ `rules/data.md` / `docs/competition/abstract.md` を NeuroGolf 仕様に更新

---

## 🔧 これから（精度・規模拡大）

### 1. ARC-AGI / ARC-GEN-100K の実データ取得
- `arc.loader` は実装済み。ARC-AGI-1 公開 train + ARC-GEN-100K を `data/lake/arc-agi-1/` にダウンロード/配置するステップが未実行（外部リソース）。

### 2. ソルバの拡充
- 現状 `identity` / `recolor` のみ。実タスク（タイル / 反転 / 対称 / オブジェクト操作 等）を解く ONNX ソルバ群を `backend/src/solvers/` に追加。各々 cost 最小化。
- タスク→ソルバの割当（どのタスクをどのソルバが解くか）と `data/output/onnx/taskNNN.onnx` 一括生成パイプライン。

### 3. cost 推定の精緻化
- `onnxgolf.cost` は Conv/Gemm/MatMul の MAC + initializer/activation memory を概算。公式スコアラとの差異が出たら係数・対象 op を調整。

---

## ⏳ 後回し（外部リソース確定後）

- **DVC を使うか自体が要検討**: GPU 学習・大規模データ管理用の `infra/`（Terraform S3 remote）・`gpu/` は削除済み。NeuroGolf がオフライン ONNX golf 中心なら DVC は不要かもしれない。使う場合のみ以下:
  - `.claude/rules/data.md` 等に残るプレースホルダ `neurogolf-dvc-000000000000` を実バケット名に
  - `dvc init` + `dvc remote add`（`.dvc/` は未初期化）
  - S3 バケット / IAM は手動 or 新 infra を作り直し（旧 Terraform は削除済み）
- **Kaggle 認証** `KAGGLE_USERNAME` / `KAGGLE_KEY` — `backend/.env`（コピー除外）にユーザーが設定
- **ARC-AGI-1 / ARC-GEN-100K のダウンロード** — ローダ実装後の実行時ステップ
