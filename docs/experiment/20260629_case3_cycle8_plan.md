# 20260629 case3 cycle8 plan — 新規採用 17 グラフへ意味保存 ORT 最適化（hidden 安全 golf）

## 現状把握

- floor = 7180.46（cycle6+7, harishk 15 + frank 2 を cherry-pick）。公開 harvest は枯渇。
- per-task hand-golf は t233 解析で確認した通り非局所・param 余剰ゼロで削減不可。

## 仮説（過去結論への限定的反証）

過去の surgery 飽和テストは**旧 floor（boristown ベース）**で実施。今回 cycle6/7 で新規採用した
17 タスク（harishk 15 + frank 2）は**他者作の別グラフで surgery 未適用**。意味保存変換
（ORT オフライン最適化 = 定数畳み込み・冗長ノード除去）はこれら新グラフに未検証で、冗長があれば
hidden 安全に cost を下げられる可能性がある（ORT 最適化は出力不変＝hidden 安全, cycle1 の学習fitとは異なる）。

## 本サイクルで回す 1 改善

- 17 新グラフを ORT_ENABLE_BASIC でオフライン最適化して再保存 → 標準 ONNX op・静的形状・
  非禁止 op を確認 → faithful 監査（cost + n_fail=0）。cost<現在 かつ n_fail=0 のみ採用。
- 採用があれば overlay → submit → 実 LB。

## 採否

- 退行ゼロ最優先。faithful n_fail=0 かつ実 LB が 7180.46 を上回れば採用、否なら破棄。
- リスク: ORT BASIC が com.microsoft 等の非標準 op を挿入 → validator で弾く（提出前ゲート）。
