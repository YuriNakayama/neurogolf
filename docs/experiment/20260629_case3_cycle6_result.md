# 20260629 case3 cycle6 result — harishk2209/7178-23 harvest で 7180.45（ACCEPT, +3.96）★採用

## 実 LB / 採否

- **ACCEPT**。combined（floor + harishk 15-task cherry-pick）= ref 54166537 = **Public Score 7180.45**。
- floor 7176.49 から **+3.96**、公開最高 harishk2209/neurogolf-7178-23（7178.23）も **+2.22** 上回る
  新 **team best**。予測 7176.49+3.957 = 7180.45 と**完全一致**（faithful 環境が実 LB と厳密整合）。

## 手法

- LB 再スキャンで `harishk2209/neurogolf-7178-23`（LB 7178.23）が floor を上回る新規バンドル発見
  （`kokinnwakashuu/neurogolf-graph-golf-blend` は md5 一致＝同一物）。
- 各タスクで floor vs harishk を cost-only 比較 → harishk が安い 15 タスクのみ faithful 正答検証
  （onnx 1.20.0/ort 1.24.1/py3.12, official neurogolf_utils）→ 全 n_fail=0 確認 → harishk net を採用。
- combined を data/output/onnx へ overlay、dvc add/push 済み（infra も新 floor を pull 可能）。

## 採用した 15 win（floor cost → harishk cost）

| task | floor | harishk | task | floor | harishk |
|---|---|---|---|---|---|
| t001 | 1121 | 488 | t152 | 629 | 458 |
| t002 | 24447 | 21729 | t165 | 6150 | 6066 |
| t021 | 1170 | 656 | t173 | 23148 | 17646 |
| t025 | 14257 | 11664 | t190 | 2727 | 2527 |
| t035 | 3122 | 2262 | t191 | 31211 | 15072 |
| t076 | 23588 | 20040 | t205 | 15372 | 12914 |
| t101 | 17845 | 17755 | t286 | 52015 | 47013 |
| t364 | 24331 | 23013 | | | |

最大寄与は t191（31211→15072, −16139）と t001（1121→488）。faithful gain 合計 +3.957。

## 確定した教訓

1. **cycle4 の grader-faithfulness law を再実証**: graded 公開バンドル（実 LB を持つ）の
   cross-bundle cherry-pick は hidden 安全に転送する。予測 gain が実 LB に厳密一致（offset 0）。
2. **公開 LB/notebook の定期再スキャンが本コンテナ唯一の主力レバー**。前回 floor (boristown ベース)
   公開後に harishk が更に上を出しており、cherry-pick で **公開最高をも超える** team best を継続更新可能。
3. **隔離コンテナの注意点**: data/lake は並行プロセスに消去される（git clean / dvc checkout 由来か）。
   作業データ（task json/utils/floor onnx）は scratchpad（リポジトリ外）に退避してから処理する。
   official utils 利用には onnx_tool/ipython/matplotlib を --with で追加要。

## 次の一手

- 新 floor = **7180.45**, 7950 まで −769.6。次サイクルは引き続き公開 notebook 再スキャン
  （lucifer19/biohack44 等 DL 不安定バンドルの再取得 + 新規高 LB バンドルの cherry-pick）。
- harvest が枯れた場合は per-task 真アルゴリズム手 golf（generalization-gate 必須）に戻る。
