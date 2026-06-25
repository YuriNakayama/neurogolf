# lake/neurogolf-2026 — 競技データ（生）

Kaggle 競技 **neurogolf-2026** のオフィシャル配布データ。`lake/` 層なので
**消失したら再取得が必要な不変の原本**（コードでは再生成できない）。

## 取得方法

競技ルールに同意済みであることが前提（同意は Kaggle Web UI で一度だけ）。

```bash
kaggle competitions download -c neurogolf-2026 -p data/lake/neurogolf-2026
cd data/lake/neurogolf-2026 && unzip -o neurogolf-2026.zip && rm neurogolf-2026.zip
```

## 内容

| パス | 説明 |
|---|---|
| `task001.json … task400.json` | 各タスクの例ペア。`train` / `test` / `arc-gen` の 3 サブセットを持ち、各例は `{"input": grid, "output": grid}`（grid は 0–9 の整数 2 次元配列、最大 30×30）。correctness はこの全例の完全一致で判定される。 |
| `neurogolf_utils/neurogolf_utils.py` | **公式スコアラー**（Apache-2.0, Google LLC）。`sanitize_model` / `calculate_params` / `calculate_memory` / `score_network` / `convert_to_numpy` 等。`src/evaluate/scorer.py`（ミラー）の照合基準。 |

`task001` の例数は train=5 / test=1 / arc-gen=262 のように、arc-gen が大半を占める。

## ミラーとの整合性

`src/evaluate/scorer.py` は公式 `neurogolf_utils.py` のスコアリングを再現した
fallback 実装。公式モジュールは `IPython` / `matplotlib` / `onnx_tool`（可視化用）
に依存するためローカルでは import できないが、スコアリング中核
（`calculate_params` / `calculate_memory`）は heavy-deps を stub すれば直接
ロードでき、identity / conv / add の各 ONNX で **params・memory がバイト一致**
することを確認済み。

## git / DVC

`data/**` は `.gitignore` 済みで、git に乗るのは `*.md` / `*.dvc` / `.gitkeep` のみ。
実データ（task JSON・`.py`）はコミットしない。DVC 採用は未確定（`docs/develop/MIGRATION.md`）。
