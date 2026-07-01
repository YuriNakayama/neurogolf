# 20260701 priority update — cost 削減は相対削減率で優先

## 確認した事実

- 実装上の評価式は `backend/src/evaluate/scorer.py` の通り:
  `cost = params + memory_bytes`
  `points = max(1, 25 - ln(max(1, cost)))`
- 正答性は足切り。`n_fail>0` / `unscorable` は 0 点なので、cost が低くても採用しない。
- cost 削減の点差は `ln(old_cost / new_cost)`。

## 優先度変更

旧方針の「高 cost task を優先」は不正確。今後は以下で選ぶ。

1. `n_fail=0` を絶対条件にする。
2. 絶対 cost ではなく、期待相対削減率 `old_cost / new_cost` が大きい候補を優先する。
3. 同じ固定 cost 削減なら、小〜中 cost task の方が点効率が高い。
4. ただし低 cost 単純 task は既に最小化済みが多く、高 cost hard task も near-optimal が多い。
   狙い目は、中〜高 cost で冗長な中間テンソル・dtype 境界・branch・initializer table を
   丸ごと消せる task。

## docs/experiment との照合

- `handgolf-targets.md` 後半の E16 は、gain の主役が中 cost 帯であることを既に示している。
- E14/E15/E17 は、高 cost hard task の多くが bundle 側で bbox-crop/uint8/反復数まで
  golf 済みで、単純な再実装では勝てないことを示している。
- 個別 plan では 20260701 の複数ケースで既に `old_cost / new_cost` 優先へ移行済み。

結論: 大きい task を雑に削るより、小〜中 cost でも比率で大きく削れる候補を優先する。
