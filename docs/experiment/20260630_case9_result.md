# 20260630 case9 result — 「未解決タスク」探索は local 偽陽性、t347 swap は REJECT（−4.43）

## 実 LB / 採否

- **REJECT**。floor の t347（cost 143, ローカルで MaxUnpool クラッシュ）を OR-fold solver
  （cost 12014, faithful n_fail=0/269）に置換した提出 = **Public Score 7176.15**（floor 7180.58 から
  **−4.43**）。floor 未変更（/tmp で構築・提出, DVC `713fc67a` 維持, LB 最良 7180.58 を保持）。

## 重大な教訓: ローカル監査の偽陽性

- 全 400 faithful 監査で **t347 のみ INCORRECT（n_fail=269）** + 7 件が negative-pad shape-inference
  の score_error。t347 floor net はローカル ORT で **MaxUnpool クラッシュ**するため「未解決＝0点」と
  判断し、OR-fold（左右 3×3 の論理和→6）solver を新規実装した。
- **実 LB は逆**: floor t347（cost 143）は **Kaggle 上では正常動作し満点（~19.7pts）**。私の
  cost 12014 版（15.6pts）への置換で **約 −4.1pt 損**（実測 −4.43 と整合）。
- **確定**: ローカル ORT の MaxUnpool/negative-pad クラッシュは **ローカル固有**で、Kaggle scorer は
  正常採点する。prompt の「ローカル監査は LB と一致しない（false-negative あり）」を実 LB で実証。
  **ローカルで INCORRECT/crash でも、既存 floor net を置換してはならない**（特に公開バンドル由来）。

## 検証内容

| 項目 | 結果 |
|---|---|
| 全 400 faithful 監査 | t347 INCORRECT(nfail=269) + 7件 score_error（045/127/135/146/149/240/384, いずれも negative-pad）|
| t347 規則特定 | 3×6→3×3, out=6 where (左3×3 ∨ 右3×3 が nonzero), 背景=ch0。**269/269 numpy/onnx 一致** |
| OR-fold solver | Slice×4/Sub×2/Pad×2/Max/Concat, cost 12014, faithful n_fail=0, 許可 op のみ, 4512B |
| floor t347 実挙動 | ローカル ORT で MaxUnpool クラッシュ → 「0点」と誤判断 |
| **実 LB** | **7176.15（−4.43）→ floor t347 は Kaggle で満点だった。REJECT** |

## 確定した運用則

1. **ローカル監査の INCORRECT/crash を根拠に既存 floor net を置換しない**。Kaggle scorer は
   ローカル ORT より寛容（MaxUnpool/negative-pad 等を正常処理）。false-negative が実在する。
2. 「未解決タスク探索」は、**ローカル偽陽性**を真の未解決と取り違えるリスクが高い。floor が
   公開グレード済みバンドル由来なら、ローカルでクラッシュしても LB では解けている可能性が高い。
3. 新規 solver で加点を狙うなら、**floor が真に 0 点**（LB 実測 or 公開 manifest で未含）と
   確認できるタスクに限る。ローカル監査単独では不十分。

## 次の一手
- 「未解決探索」路線は local 偽陽性ゆえ棄却。floor 7180.58 維持。
- 7 件の score_error タスクも同様に Kaggle では正常採点されている（floor の一部）ので触らない。
- 残るレバーは公開バンドル監視 harvest（実証済み）と bit-pack novel solver（研究）のみ。7950 まで −769.4。
