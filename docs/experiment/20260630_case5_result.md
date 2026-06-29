# 20260630 case5 result — t002 flood-fill 再構築は正答するが memory 爆発で REJECT（提出せず）

## 結論

- **REJECT（提出なし）**。t002 を 4連結 enclosed-fill の unrolled dilation ONNX で再構築 →
  **faithful n_fail=0/268（完全正答）だが cost=3,056,510**（K=64）。floor の bit-pack net
  **21089** に対し **約145倍**。K=25 まで削っても ~1.2M で floor に遠く及ばず。floor 維持。

## 検証内容

| 項目 | 結果 |
|---|---|
| enclosed-fill 規則網羅判定 | 高コスト同形状・少色タスクで **t002（fill=4）と t251（fill=1）** が 4連結 enclosed-fill と全 example 一致 |
| t002 規則特定 | free=非wall, 外周ring seed, K回 4連結 dilation∩free, enclosed=color0∧¬reach→4。**268/268 numpy 一致** |
| ONNX 構築 | Pad+Slice 方向 shift × Max（4方向）∩free を K=64 unroll、840 nodes |
| 当初バグ | 負方向 shift の Slice 窓が固定[0:30]で up/left が no-op → 2セル誤り。Slice 窓を pad 量だけずらして修正 → 268/268 |
| **cost** | **3,056,510**（K=64）。memory metric が 840 nodes の全中間テンソル peak を加算するため爆発 |
| 最小 K | visible set は K=25 で全通過。だが hidden の大グリッドで不足リスク + どの K でも cost >> 21089 |

## 確定した教訓（重要）

1. **空間的 flood-fill（unrolled dilation）は正答するが golf 不可能**。memory cost が
   unroll 深 K × 中間テンソル数で線形爆発する。K=25 でも ~1.2M >> floor 21089。
2. **floor の bit-pack（整数コード符号化）net は flood-fill タスクで既に near-optimal**。
   cycle13-14 の「整数コード符号化が公開 net の安さの核心」を実測で裏付け。10チャネル one-hot
   や 30×30 空間中間を避け、グリッドを整数ビット列に packing して列演算で解くのが安い。
3. **hand-golf で floor を破るには bit-pack 系の表現を使う必要がある**。空間 conv/dilation 系の
   素朴な真アルゴリズム実装は正答しても cost で負ける。次に hand-golf するなら bit-pack 表現で
   設計するか、bit-pack net の冗長ノード削減（surgery）を狙う（ただし surgery は cycle2 で飽和確認済み）。

## 次の一手

- 空間 flood-fill 路線は棄却。floor 7180.58 維持。
- hand-golf の現実的な勝ち筋は bit-pack 表現での per-task 設計のみ（高難度）。当面は
  **新規公開バンドルの定期監視 harvest**（受動だが実証済みレバー）を主とし、bit-pack 設計は
  研究課題として継続。7950 まで −769.4。
