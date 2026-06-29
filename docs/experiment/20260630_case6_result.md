# 20260630 case6 result — 新 floor 全体への graph-surgery 再走 → 0/400 改善（提出なし、floor 維持）

## 結論
- **提出なし（floor 7180.58 維持）**。意味保存 surgery（int64→int32, cleanup, index_surgery,
  broadcast_compress, conv1x1→gather, fp16）を**現 floor 全 400 net に再走** → **0/400 改善**。
- cycle1 で新規採用した lucifer t233 を含め、公開バンドルは**発行前に既に等価 surgery 済み**。
  cycle2 の「surgery 飽和」が新 net にも及ぶことを実測確認。

## 検証内容
| 項目 | 結果 |
|---|---|
| surgery 再走 | `pipeline/case3 surgery --base data/output/onnx` 全 400 net |
| 改善 | **0/400**, local points delta +0.00 |
| t233（lucifer 由来, cycle1 採用）| 改善なし。発行元が既に縮約済み |

## 教訓
- 意味保存 surgery は公開バンドル取込み時点で飽和する（発行元が適用済み）。新規採用 net への
  再走も 0 win。**surgery レバーは完全枯渇**。
- harvest 飽和（cycle2/4）+ surgery 飽和（cycle6）+ flood-fill hand-golf 構造的不可（case5）
  により、自動・半自動レバーは全て枯渇。
- 残るのは **bit-pack 表現での per-task net 完全再設計**（floor と同系統でビット列演算をさらに
  圧縮する）のみ。これは expert-scale の研究課題で、1 net ずつ手作業が必要。

## 次の一手
- bit-pack per-task 設計に着手。floor net の bit-pack 構造を 1 タスク精読し、冗長な
  整数コード演算を削れるか調べる（高コスト bit-pack net = t002/t023 等を標的）。
- 7950 まで −769.4。
