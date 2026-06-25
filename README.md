# NeuroGolf 2026

Kaggle [The 2026 NeuroGolf Championship](https://www.kaggle.com/competitions/neurogolf-2026)（IJCAI-ECAI 2026 Competitions Track）参戦リポジトリ。**ARC-AGI のグリッド変換タスクを「最小規模の ONNX ニューラルネット」で解く** "neural golf" コンペです。タスクごとに 1 つの ONNX ファイル（`task001.onnx`〜`task400.onnx`）を `submission.zip` にまとめて提出し、**機能的正答性 + パラメータ数 / 計算量 / メモリの最小化**（ONNX グラフから推定）で採点されます。データは ARC-AGI-1 公開 train + ARC-GEN-100K。

> ⚠️ **テンプレート移行中**: このリポジトリは Kaggle PTCG（ポケカ対戦 AI）コンペ用スキャフォールドを雛形として複製し、PTCG 固有のドメインコード（cabt エンジン・self-play・対戦エージェント・GPU 学習・DVC/Terraform infra）を削除した状態です。現在残っているのは汎用スキャフォールドと、Kaggle 提出まわりの `backend/src/submit/`（PTCG の tar.gz 形式のまま。ONNX `submission.zip` への作り替えは未実施）のみ。**NeuroGolf 仕様への作り替え項目は [`docs/develop/MIGRATION.md`](docs/develop/MIGRATION.md) に整理してあります。**

---

## 現在のリポジトリ構成

```
backend/                Python 実装 (pyproject.toml / uv.lock はここ)
  src/submit/           Kaggle 提出パッケージング (python -m submit)
                        ※ 現状 PTCG の tar.gz 形式。ONNX zip へ作り替え予定 (MIGRATION.md)
  pipeline/              NeuroGolf の提出物 (ONNX 生成) を置く予定の空プレースホルダ
  tests/                Pytest (submit のテストのみ)
data/                   4 層 (lake / processed / mart / output)。gitignore 済み
dev/                    開発スクリプト (setup / format / lint / test-bot / submit など)
docs/develop/           MIGRATION.md (テンプレート→NeuroGolf 移行 TODO)
.claude/, .github/      Claude Code 設定 + CI
```

## Technology Stack

- **Language**: Python 3.13
- **Submission**: ONNX (`onnx` / `onnxruntime` を MIGRATION.md で導入予定)
- **Kaggle API**: `kaggle` CLI (提出), `dvc[s3]` (成果物管理)
- **Testing**: Pytest + Ruff + Mypy
- **Package Management**: uv

## Commands

```bash
dev/setup            # 依存インストール (uv sync)
dev/format           # ruff format
dev/lint             # ruff + mypy
dev/test-bot         # CI 相当 (format check → lint → type check → pytest)
dev/submit           # Kaggle 提出 (※ ONNX 対応は MIGRATION.md 参照)
dev/create-worktree  # git worktree 作成
```

Python コマンドは `backend/` 配下で `uv run ...`、またはリポジトリルートから `dev/*` で実行します。

## 移行状況

このリポジトリはまだ NeuroGolf 用に完成していません。やるべき作り替え（submit を ONNX `submission.zip` 化、ARC-AGI データローダ追加、ドキュメント整備など）は [`docs/develop/MIGRATION.md`](docs/develop/MIGRATION.md) を参照してください。

## Links

- [Kaggle: The 2026 NeuroGolf Championship](https://www.kaggle.com/competitions/neurogolf-2026)
- [IJCAI 2026 Competitions](https://2026.ijcai.org/competitions/)
- [コンペ概要まとめ](docs/competition/abstract.md)
- [テンプレート移行 TODO](docs/develop/MIGRATION.md)
