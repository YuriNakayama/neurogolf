# case3 — minimal-ONNX solver bank + per-task MAX-blend

ベースライン（case1, LB 7159.44）を上回る Public Score を狙う build-case。2 つの
レバーを持つ:

1. **solver bank**（`solvers.py` / `builders.py` / `lookup.py` / `smalllookup.py`）:
   タスクごとに最小 cost の ONNX を厳密構成する。
   - `identity`（cost 0 / 25 点）、`transpose`（cost 0）、`flip`（Slice 負ステップ）、
     `recolor`（Gather/1×1 Conv）、`localconv`、近傍ルックアップ（2 層 Conv）。
   - **小空間版**（`smalllookup.py`）: `Slice([1,10,30,30]→[1,10,h,w]) → 小空間処理 →
     Pad` で中間テンソルを h×w スケールに抑え memory を削る。
2. **MAX-blend**（`blend.py`）: 複数の公開バンドル + 自前 solver 出力から、タスク
   ごとに **公式スコアラミラー（`src/evaluate.audit_one`）で全 example 厳密検証**して、
   正答かつ cost 最小の ONNX を採用する。`>= max(各バンドル)` を保証。

## スコアリングの要点（`neurogolf_utils.py` 実コードより）

- `cost = params + memory`（**MACs は寄与しない**, 2026-05-04 改定）。`points =
  max(1, 25 - ln(cost))`。cost 0 → 25 点。
- **memory = 中間テンソルの静的形状要素数 × itemsize の総和**。`input`/`output` 名の
  テンソルは **除外**。→ 中間テンソルを増やさない / 小さく（小空間・小 dtype）するのが golf。
- 正答は train+test+arc-gen の全ペアを `(out>0)` 二値化で完全一致した場合のみ。

## ⚠️ ローカル採点と Kaggle 実 LB の不一致（最重要・実証済み）

**ローカル `audit_one` の正答判定は Kaggle 実 LB と一致しない。** frank7166 の一部タスク
（t045/127/135/146/149/240/384/347 など）はローカルの strict shape-inference / arc-gen
例で「不正解 or unscorable」になるが、**Kaggle 実採点では正答して得点している**。

実証: frank7166（実 LB 7166.10）の t347 をローカルで「正答かつ +17.8 点」のソルバに
差し替えて提出 → **実 LB は 7166.63 → 7161.55 に低下**（並行セッションでも同結果）。
つまりローカルの「勝ち」が実 LB では「負け」。

帰結:
- **frank7166 のタスクをローカル監査だけを根拠に差し替えてはならない**（回帰する）。
- ローカルで improvement を検証できないため、bespoke ソルバの実効性は提出するまで不明。
- これが「公開最高 7166 を超えるのが困難」な根本理由の一つ。安全策は **pure frank7166**。

## 測定された現実（重要）

- 公開バンドルの最高は **franksunp/7166-10**（LB 7166.10, local 7014.5）。
  ローカル監査と実 LB は **real ≈ local + 166** で安定対応。
- ベースラインは易タスクで既に最小 cost（recolor=cost10/22.7 点, transpose=cost0/25 点）。
  単純変換 solver では**ほぼ上回れない**。
- 局所決定的（k×k 近傍で出力が決まる）同形状タスクは 64 個。小空間ルックアップで golf
  可能だが、ベースラインも善戦するため純増は小さい。
- LB トップは**非公開チーム 7942.46**（公開ノートブック無し）。7665 は全公開を上回る水準。

## 実行

```bash
cd backend
# solver bank で自前 ONNX 生成（data/output/onnx/case3）
uv run python -m pipeline.case3 solve --task-dir ../data/lake/neurogolf-2026 --out ../data/output/onnx/case3
# 公開バンドル + 自前を MAX-blend（data/output/onnx/blend）
uv run python -m pipeline.case3 blend --task-dir ../data/lake/neurogolf-2026 \
    --bundle ../data/lake/bundles/franksunp_7166-10-lb-neurogolf-audit-variant-b/onnx \
    --bundle ... --out ../data/output/onnx/blend
```

## case 独立

`src/evaluate`（競技固定スコアラミラー）のみ共有依存。他 case は import しない。
