# 20260629 case3 result — 単一 Conv 化（線形分離）は hidden private で死亡、REJECT

## 実 LB / 採否

- **REJECT**。t192 を単一 Conv[10,10,3,3]（cost 8621→910）に置換した提出 = ref 54164161 =
  **Public Score 7156.50**（floor 7172.43 から **−15.93**）。
- −15.93 ≈ t192 の floor 寄与点（25−ln(8621)=15.94）とほぼ完全一致 → **私の Conv は hidden
  private 集合で 0 点**。floor は未変更（候補は /tmp で構築・提出）。Kaggle LB は最良提出を
  保持するため順位退行なし（提出枠 1 消費のみ）。

## 検証した内容

| 項目 | 結果 |
|---|---|
| ARC データ復旧 | data/lake 空 → Kaggle から 400 タスク + 公式 neurogolf_utils.py 取得、faithful 基盤復旧 |
| k-局所スキャン | 「out[r,c]=入力 k×k 窓の決定的関数」を全 example で満たす 14 タスク発見（k=3/5/7, 潜在 gain +17.4）|
| 線形分離（max-margin LinearSVC, グリッド外を全オフ負例）| **t192(k3) のみ YES**。t222/t004/t293(k3)/t243(k5) は NO。k=5/7 は LinearSVC が過大に遅く timeout |
| t192 faithful 監査 | n_fail=0/265（train/test/arc-gen 全通過）, cost 910, pts 18.19（+2.25）|
| **t192 実 LB** | **0 点（hidden private 失敗）→ 全体 −15.93** |
| ランダム入力で floor net と一致 | **599/600 不一致**（提出前に出ていた決定的 red flag）|

## 確定した教訓（重要）

1. **faithful n_fail=0（train/test/arc-gen 全通過）は必要条件だが十分条件ではない**。
   Public Score は **hidden private 集合を含む**。可視データに fit しただけのネット（学習済み
   線形 Conv = lookup の補間）は hidden の未知窓で真ルールと乖離し 0 点になる。E19 の警告
   「安い版 swap は hidden で落ちる」を実 LB で実証。
2. **過去の redesign 成功（t002/t202/t364 等）が転送したのは、真のアルゴリズムを再構築した
   から**（flood-fill 等）。線形分離 fit は真ルールではないため転送しない。
3. **提出前ゲート: 候補 net を floor net とランダム/新規入力で出力比較し、高一致のときのみ
   提出する**。t192 は 599/600 不一致＝提出すべきでなかった。このゲートで提出枠を節約できた。
4. k-局所性（lookup-table 一貫）は「可視データで局所決定的」を示すのみ。**安価化には真の
   局所アルゴリズムの実装が要る**（学習 fit ではなく）。floor は既にそれを実装済み。

## 次の一手

- 単一 Conv 線形 fit は棄却。次サイクルは t192 等の denoise 系で **真の局所アルゴリズム
  （3×3 多数決等）を厳密に numpy 特定 → 最小 ONNX 化 → ランダム入力で floor net と一致確認
  → 提出**、という generalization-gated redesign を 1 タスク試す。一致しなければ floor 維持。
- 提出前のランダム入力一致ゲートを今後の必須手順とする。
</content>

## cycle 2 追記: 安全レバー(surgery)再確認 + 高コスト非局所タスクの確認

cycle 1 の教訓（hidden 安全に転送するのは ①意味保存 surgery ②真アルゴリズム再構築のみ）を
受け、安全レバーを再点検:

- **graph surgery 全パス（faithful, top-cost 標的）**: t233/t018 とも no reduction。過去
  E27/E34 で combined_best（=現 floor）に全 surgery wins 取込み済み → **飽和確定**（0 win）。
- **観測 observer 推奨タスクの精査**: t133(33573)=template-based stamping（key 形状検出→object
  展開、非局所, k-局所スキャンで mink=None）、t118=cross-arm gap fill（Einsum×3, 非局所）。
  いずれも floor が near-optimal 実装済みで安価な真アルゴリズムが存在しない → cheap win 無し。

**cycle 2 結論**: 安全レバー（surgery）は飽和、高コストタスクは非局所で floor が最小付近。
隔離コンテナ（並行 dsl session 無し）では harvest 不可。**floor 7172.43 が本コンテナで自動
反復が到達できる構造的上限**であることを cycle 1-2 で独立に再確認（8 セッション + 本セッション）。
残る唯一の道は generalization-gate を課した per-task 真アルゴリズム手 golf（expert-scale, 低確率）。
