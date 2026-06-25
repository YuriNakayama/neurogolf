# case1 — per-task MAX BLEND

Kaggle ノートブック
[`biohack44/neurogolf-2026-blend-max`](https://www.kaggle.com/code/biohack44/neurogolf-2026-blend-max)
の移植。**まず元スコアを再現できるよう、計算ロジックは逐語移植**している。

## これは何か（重要）

ソルバーではなく **選択ブレンド (selection blend) の後処理**。新しいモデルは
作らない。既存の 2 つの提出バンドル **A / B** を入力に取り、タスクごとに
公式スコアラーのミラーで採点して **correct かつ cheaper** な ONNX を選び、
`submission.zip` を再パッケージする。

- 各タスクは A か B の実提出が既に含む ONNX のいずれか → private-set の挙動は保たれる。
- よって blend 後の LB は `max(A_LB, B_LB)` 以上になる（新規モデルなし＝private-set リスクなし）。

### tie-break ルール（per task）

1. 片方だけ correct（`n_fail==0`）→ それを採用
2. 両方 correct → 低 cost（高 points）を採用
3. 両方 correct で同 cost → A（任意・安定）
4. どちらも correct でない → fail 少を採用、なお同じなら A

## 入力の用意

A / B は **ディレクトリ**（`task001.onnx … task400.onnx` を含む）または
**`submission.zip`** のどちらでも良い。元ノートブックは Kaggle 上の
`/kaggle/input/...` を指していた。ローカルでは任意のパスを渡す。

タスク JSON（`task001.json …`）は競技データ
`/kaggle/input/competitions/neurogolf-2026` にある。

## 実行

```bash
cd backend
uv run python -m pipeline.case1 blend \
    --a-dir /path/to/A_bundle \
    --b-zip /path/to/B/submission.zip \
    --task-dir /kaggle/input/competitions/neurogolf-2026 \
    --out data/output/case1/submission
```

- `--a-dir` / `--a-zip`、`--b-dir` / `--b-zip` はどちらか一方を指定。
- `--out` は拡張子なしのパス（`shutil.make_archive` 規約）。`<out>.zip` が生成される。
- `--work` は zip 展開と stage の作業ディレクトリ（既定 `/kaggle/working`）。
- `--no-diff` で A/B 差分表示を抑制。

実行すると picks（A/B 採用数）・gain（A-only / B-only 比の局所改善）・staged 数を表示し、
`<out>.zip` と `blend_audit.csv`（最終 audit）を書き出す。

## 元ノートブック（13 セル）→ 配置 対応

| Notebook cell | 配置先 |
|---|---|
| Cell 2 deps | 不要（uv で依存管理） |
| Cell 4 audit harness（公式スコアラーのミラー） | `src/evaluate/scorer.py`（全ケース共通） |
| Cell 6 `_resolve`（バンドル解決） | `pipeline/case1/bundle.py` `resolve_bundle` |
| Cell 8 `score` / `better` / blend ループ | `pipeline/case1/blend.py` `score` / `better` / `blend` |
| Cell 10 差分表示 | `pipeline/case1/blend.py` `print_diff` |
| Cell 12 zip 化 + 最終 audit | `pipeline/case1/blend.py` `package` + `__main__.py` |

## 設計メモ

- **公式スコアラーのミラー**（Cell 4）は競技固定で全ケース不変のため
  `src/evaluate` に共通化した。リポジトリ内の他モジュールには依存せず
  `onnx` + `onnxruntime` + `numpy` のみで自己完結する（ノートブックと同じ）。
- **blend 選択ロジック**は手法変更で変わりうるため `src` に置かず
  この case ディレクトリに置く。
