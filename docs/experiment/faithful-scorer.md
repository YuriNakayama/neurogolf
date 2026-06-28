# Faithful scorer 環境（Kaggle 整合）

`src/evaluate/scorer.py` のミラーは onnx/onnxruntime のバージョンに敏感で、リポジトリの
デフォルト（py3.13 / onnx 1.22 / ort 1.27）は **新しすぎて** MaxUnpool / 負 pads の挙動が
Kaggle と異なり、bundle の一部タスクが誤って 0 点になる（local≠real +152 の真因）。

## Kaggle 整合スタック（実測で確定）

| 用途 | onnx | onnxruntime | python |
|---|---|---|---|
| **faithful 採点** | **1.20.0** | **1.24.1** | 3.12 |

このスタックで公開最高 bundle（7166.66）を採点すると **faithful local = 7166.55
（offset +0.11, 0 点タスク 0）** で実 LB とほぼ一致。**ローカル検証した cost 改善は実 LB に
ほぼ厳密に伝播する**。

## 実行方法

```bash
cd backend
uv run --python 3.12 \
  --with "onnxruntime==1.24.1" --with "onnx==1.20.0" --with "numpy<2.1" \
  python <script>   # script は src/evaluate.scorer.audit_dir/audit_one を import
```

採点ハーネス: `scratchpad/faithful_audit.py`（`audit_dir` を faithful スタックで実行し
総点・0 点タスクを出力）。bespoke ソルバや blend 候補を**提出前に faithful 検証**する。

## なぜ重要か

- local≠real(+152) は環境バージョン不一致が真因で、版を合わせれば +0.11 まで消える。
- これにより per-task golf / solver-search / surgery を**信頼できるローカル検証下**で回せる。
- ただし高コストタスク（cost>5000, 82 個）は非局所アルゴリズムで、faithful 検証下でも
  自前 solver では安価化できない（solver-search 0 wins, 6 タスク手分析で確認）。
