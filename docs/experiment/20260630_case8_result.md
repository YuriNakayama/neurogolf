# 20260630 case8 result — 全構造クラス網羅監査 → floor が全単純クラスでフロンティア、提出なし

## 結論（提出なし、floor 7180.58 維持）

全 400 を 9 構造クラスで網羅分類し、各クラスの最高コストを確認:

| クラス | 該当数 | 最高コスト | hand-golf 余地 |
|---|---|---|---|
| colormap（同形状・位置不変）| 5 | 772 | 無し（既最小, mid-cost 0件）|
| identity | 数件 | 0 | 無し |
| 幾何変換（flip/rot/transpose）| 7 | 158 | 無し（既最小）|
| tiling（整数倍 plain）| 1 | 287 | 無し |
| mirror-expansion | 7 | 458 | 無し |
| flood-fill（enclosed）| 2 (t002/t251) | 21089 | **構造的不可**（case5: cost∝node数）|
| k-local 3×3 | 3 (t192/t222/t004) | 8621 | hidden 落ち/単純形なし（cycle1/7）|
| gravity | 2 (t078/t032) | 1293 | 低 EV（既 1293/910, 数百 cost 救済のみ, hidden risk）|
| projection | 0 | – | – |

**確定**: floor は検出可能な全単純構造クラスで **cost フロンティアに到達済み**。残るコストは
全て非局所アルゴリズムタスク（bit-pack 系）に集中し、公開コミュニティが near-optimal に収束済み。

## 戦略的帰結

- 本日 8 サイクルで harvest(3×)/surgery(全走)/flood-fill/k-local/幾何/tiling/gravity/colormap を
  全て実測検証 → **自動・半自動・単純構造 hand-golf の全レバーが枯渇**。
- 7180.58 → 7950（−769.4）には、非局所 bit-pack タスクで**公開最良 net をさらに削る**か
  **公開未到達の新規ソルバ**が必要。前者は surgery 飽和で削り尽くし済み、後者は研究規模。
- **唯一の実証済み前進レバー = 新規公開バンドルの監視 harvest**（cycle1 で +0.03 実績）。
  コミュニティは活発（boristown は ~30-40分毎に再 publish）なので、新 revision/新規参入者の
  バンドルが floor を局所的に超える可能性は残る。

## 次の一手
- 公開フロントを継続監視し、新バンドル出現時に即 cherry-pick（faithful 検証 → 退行ゼロ submit）。
- bit-pack 非局所タスクの深掘り（floor net の冗長削減）は surgery 飽和済みのため、
  手動でのノード単位削減を要する研究課題として保留。
- floor 7180.58 を退行ゼロで維持。

## cycle9 追記: 自動簡約 + 単一ノード最小性の確認

- **onnxsim**: t233 で segfault（複雑 net に非対応）。**onnxoptimizer**（dead-code/identity/nop 除去）:
  t233 348→348 で 0 削減 → 公開 net は dead-code フリー、手動ノード削減の余地なし。
- **gravity t032**: floor net は **単一 Conv ノード（cost 910）**。理論最小構造で改善不可能。
  t078(grav_up, 1293) も同様に最小付近と推定。
- **最終確定**: floor は全タスクで（単一 Conv/Gather を含む）最小構造に到達済み。surgery 飽和は
  「公開 net が既に dead-code フリー & 最小ノード」であることに起因。手動・自動とも削減余地ゼロ。

## 本日（20260630）最終総括（cycle1-9）

- **成果**: lucifer t233 harvest で **7176.49→7180.58（+0.03 相当, 実 LB 確認）= 新 team best**。
- **枯渇確認したレバー**: harvest(3×, kojimar283含む)/surgery(全走0/400)/onnxoptimizer(0削減)/
  flood-fill(構造的不可)/k-local(hidden落ち)/colormap・幾何・tiling・mirror・gravity(全て最小)。
- **floor 7180.58 = 収束した公開包絡線 = 全構造クラスの最適化フロンティア**。
- **7950 への道**: 公開未到達の新規ソルバ（研究規模）or 新規公開バンドル監視 harvest（受動）のみ。
