# case1 — reproduce baseline（既知良好バンドルの再現提出）

公開ノートブック
[`boristown/agi-neural-golf-visualization-baseline`](https://www.kaggle.com/code/boristown/agi-neural-golf-visualization-baseline)
が出力する **400 タスク完成済みの `submission.zip`** を取得・検証し、そのまま
Kaggle に提出する後処理ケース。**ある程度の Public Score を担保するベースライン**
として使う。

## これは何か（重要）

ソルバーでも blend でもない。**新しいモデルは一切作らない。** 既知の高スコア
バンドルを「正しく掴んでいる」ことを **バイト数 + SHA256** で固定照合し、検証が
通ったものだけを提出する。

- 提出物は実績のある公開バンドルそのもの → private-set の挙動は保たれる。
- よって最低でも当該ノートブックの LB（**Public Score 7159.44**）が見込める。
- 1 日 100 回まで無検証提出できるため、pending / 一時失敗はループでリトライする。

## 固定対象（差し替え時はここを更新）

| 項目 | 値 |
|---|---|
| Kaggle kernel | `boristown/agi-neural-golf-visualization-baseline` |
| Public Score | 7159.44 |
| `submission.zip` バイト数 | 542,649 |
| SHA256 | `33a16642e139d04ad61d6edcccf1a72b26013e2aeee2c9070a7f1f095e9baa1e` |
| ONNX 数 | 400 |

固定値は `reproduce.py` の `TARGET_KERNEL` / `EXPECTED_BYTES` / `EXPECTED_SHA256`、
目標スコアは `__init__.py` の `TARGET_PUBLIC_SCORE`。

## 構成

```
case1/
├── reproduce.py     # fetch_target / verify_bundle / resolve_target（取得 + 固定照合）
├── submit_loop.py   # submit_once / run_until_target / reached（目標到達まで提出）
├── __main__.py      # typer CLI（verify / submit）
└── README.md
```

`src/submit` の Kaggle CLI ラッパー（提出・履歴ポーリング）に委譲し、case 固有の
「取得・固定照合・目標到達ループ」だけをここに置く。

## 実行

```bash
cd backend

# 検証だけ（ローカル zip を固定値と照合）
uv run python -m pipeline.case1 verify \
    --local-zip ../data/lake/case1-baseline/submission.zip

# Kaggle から取得 → 検証 → 目標スコアまで提出
uv run python -m pipeline.case1 submit

# 取得済みのローカル zip をそのまま提出
uv run python -m pipeline.case1 submit \
    --local-zip ../data/lake/case1-baseline/submission.zip
```

- `--local-zip` を省略すると `kaggle kernels output` で `--work`（既定
  `../data/lake/case1-baseline`）に取得してから検証する。
- `--dry-run` で提出せず検証のみ。`--target` / `--max-attempts` で到達条件と
  リトライ上限を変更。提出履歴は `data/output/submit/case1/submissions.jsonl`。

## 設計メモ

- **検証は固定照合**：バイト数と SHA256 の二重チェックで「別物を提出する」事故を
  防ぐ。バンドルを差し替えるときは取得し直して 3 値を更新する。
- **冪等な再提出**：同一バンドルの再提出は private-set 挙動を変えないので、
  目標未達（pending 等）のときに安全にリトライできる。
- **case 独立**：他 case を import しない。共有は `src/submit` のみ。
