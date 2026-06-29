# 20260629 case3 cycle9 result — boristown-v51 新版(11:00) harvest で 7180.47（ACCEPT, +0.01）

## 実 LB / 採否

- **ACCEPT**。combined（floor + boristown-v51 11:00版 t191）= ref 54168561 = **Public Score 7180.47**。
- floor 7180.46 から **+0.01**。退行なし、Kaggle 最良提出保持。

## 手法・結果

- 監視サイクル: 公開 kernel を再スキャン → boristown-v51 が 11:00 に再実行（私の 10:36 取得版と
  別 md5）、kokin も 10:49 更新を検出 → 最新版を再取得し combined 7180.46 基準で harvest。
- **boris2（11:00版）: t191 15072→14974 の 1 win**（faithful n_fail=0, +0.007）。kokin2 新版は 0 win。
- t191 を fold → submit → 7180.47。

## 教訓

- **harvest 監視は live レバー**: コンペは活発で、同じ作者が ~20-30 分毎にバンドルを微改善して
  再公開する。各更新版を再 harvest すると t191 のような数 byte 級の cost 減を継続的に拾える。
- 単発 gain は +0.007〜0.01 と微小だが、退行ゼロかつ Kaggle 最良保持のため zero-downside。
  新版が出る度に再スキャン→cherry-pick→faithful→submit を回すのが本コンテナの定常運用。

## 次の一手

- 公開バンドルの再スキャンを継続し、新版/新規バンドルの win を逐次 fold して submit。
- 7950 への構造的ギャップ（expert per-task 再設計）は不変（cycle8 で全レバー実測棄却済み）。
  現実的には harvest 監視で公開包絡線に追随し続けるのが最善。新 floor = **7180.47**。
