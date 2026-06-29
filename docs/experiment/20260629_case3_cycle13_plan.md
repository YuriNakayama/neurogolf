# 20260629 case3 cycle13 — plan: small-space residual 原設計 + floor ギャップ探索

## 仮説
1. 既存 residual ソルバは中間テンソルを 30×30 で持つため小グリッドで cost 過大。
   **small_residual**（Slice→h×w 空間で残差→Pad）で k-local かつ小グリッドのタスクの cost を
   下げ、公開 floor net を per-task で下回れる可能性。
2. floor が未解決のタスクがあれば、独自ソルバ追加で +最大25点/タスクの大利得。

## 対象・手法
- small_residual3/5 を全 400 タスクで構成し、faithful cost を floor net と比較
  （onnx 1.20.0 / ort 1.24.1 / official neurogolf_utils）。floor 未満かつ n_fail=0 のみ採用。
- floor の per-task 正答監査を 400 全数で実施し、未解決(gap)タスクを列挙。
- 高コスト上位タスクの実ルールを特徴づけ（geom/colormap/tile/ray）、公開 net が高コストで
  実装している「実は単純な変換」を探す。

## 期待
- small_residual で数タスクの cost 削減（+小）/ gap 充足で大利得、のいずれか。
