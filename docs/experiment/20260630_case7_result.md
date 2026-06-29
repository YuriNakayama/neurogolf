# 20260630 case7 result — 公開フロント収束を確認、全レバー枯渇、floor 7180.58 維持

## 結論（提出なし、floor 維持）

3 クラスを網羅探索し、いずれも win なし:

| クラス | 結果 |
|---|---|
| 純幾何変換（flip/rot/transpose）| 該当 7 タスク全て cost≤158（floor 既最小）|
| 厳密 3×3-local（高コスト）| t192(table=11704), t222(61416), t004(1301)。巨大 table は hidden 汎化せず（cycle1 で t192 実証済み 0点）。提出不可 |
| boristown 新 revision（16:21, lucifer blend 込み）| **0 win**（floor に収束）|

## 公開フロント収束の確定

- **boristown / lucifer / harish / kojimar / frank が全て同一包絡線に収束**。最新 boristown は
  manifest 名 `v91_exact_lucifer_row1` の通り lucifer を取込み済みで、floor と一致。
- 私の floor **7180.58** は**この収束した公開包絡線そのもの**。どの公開バンドルも全タスクで floor 以上。
- 自動・半自動レバーの最終状態:
  - harvest: 飽和（cycle2/4/7, kojimar 283votes 含む全バンドル 0 win）
  - surgery: 完全枯渇（cycle6, 新 net 含め 0/400）
  - flood-fill hand-golf: cost∝node数 で構造的に不可（case5）
  - k-local fit: 巨大 table は hidden 落ち（cycle1 実証）
  - 簡単規則/幾何変換: floor 既最小（cycle4/7）

## 戦略的評価

7180.58 → 7950（−769.4）には、**公開コミュニティ全体が未到達の新規ソルバ**が必要。これは
per-cycle 増分ではなく研究規模の課題。当面の唯一の実証済み前進レバーは:
1. **新規公開バンドルの監視 harvest**（cycle1 で +0.03 実績、コミュニティ更新依存の受動レバー）。
2. **bit-pack 表現での novel solver 設計**（研究課題、低確率・高難度、継続検討）。

## 次の一手
- 公開フロント監視を継続（新バンドル/新 revision を定期再スキャン → 差分 cherry-pick）。
- 並行して bit-pack novel solver の設計検討を進める（厳密 k-local で table が小さく hidden 汎化
  し得るタスクの精査、または floor bit-pack net の冗長削減の深掘り）。
- floor 7180.58 を退行ゼロで維持。

## 追記: t004（3×3-local, table=1301）規則解析 → 単純形なし

- t004 の変化は 0↔6 双方向。3×3-majority / plus-majority とも 0/265 不一致。単純 CA/majority では
  ない複雑なパターン補完で、floor の table 実装が妥当。小 table でも単純閉形式が無く hidden 汎化
  保証が立たないため hand-golf 不可。
- bit-pack novel solver の即効候補は本セッションでは発見できず。公開フロント収束済みのため、
  novel solver は研究継続課題として残す。
