# v7: v6 + ロープのめり込み解消速度を v4 の既定に戻す

```bash
uv run train --task Isaac-Twisted-RL-Knot-v7 --headless           # 段ごとの MLP
uv run play  --task Isaac-Twisted-RL-Knot-Play-v7 --num_envs 16   # まっすぐなロープから段をつないで評価 (chain モード)
```

ログ: `logs/rsl_rl/knot_tying_v7/`。**学習には `Knot` プリセットを使う** (v3〜v6 と同じ)。

## なぜ作ったか

v4 / v5 / v6 の比較 ([`docs/experiments/2026-10-01_v4-v6_overnight.md`](../../../../../../docs/experiments/2026-10-01_v4-v6_overnight.md)) で、v5 / v6 は段 2 -> 固結び (tuck) を学べず、段 2 で
6 手を使い切る局所解に入っていた。v4 と v5 の違いは v5 の発散対策 2 つ (離す高さ 5〜10 cm、ロープの
めり込み解消速度 0.4 m/s) だけなので、v6 に 1 つずつ v4 の値を戻して切り分けた。あわせて、局所解への
直接の対策 2 つ (時間切れの罰則、段 2 から始める割合を上げる) も試した。

v6 に 1 つずつ入れた (seed 42、4096 環境、GPU 1 枚に 2 本)。学習ログの iteration 34〜43 の平均の表 (段ごとの成功・段 2 の
時間切れ・発散・`chain_est`) と chain 評価 (iteration 70) は [`docs/experiments/2026-10-01_v4-v6_overnight.md`](../../../../../../docs/experiments/2026-10-01_v4-v6_overnight.md) の末尾の追記にある (数値の正本)。
下の (b) は、その追記の「めり込み解消速度を v4 の既定 (0.96 m/s) に戻す」run (`abl_depB_s42`、= v7 の設定)。

* **段 2 の学習を止めていたのは、めり込み解消速度の上限 0.4 m/s だった。** 戻すと段 2 の成功率も
  時間切れも v4 と同じ水準になり、局所解に入らない。離す高さを戻した効果は小さい。
* 段 0 の時間切れも、学習前 (iteration 0〜9) から v6 29% に対して v7 5% と違う。0.4 m/s は学習の
  問題ではなく、物理の段階で交差を作りにくくしている (理由は未確認)。
* 局所解への直接の対策 2 つは、段 2 の成功率を上げなかった。成功が十分に出れば局所解には入らないので、
  原因 (物性) のほうを直した。

**ただし本番の指標 (chain 評価、iteration 70) では確かめられなかった。** (b) は 31.2% で、v6 の 2 シード
(35.2% / 16.0%) の間、v4 (29.7%) と同程度。(b) は 18 手のあいだの発散を v4 と同じ 17% に戻すので、段 2 を
学びやすくなった分と相殺している (発散しなかったエピソードに限ると (b)・v4・v6 s42 とも 36〜38%)。
**v7 の確認の学習はまだ回していない。** 詳しくは [`docs/experiments/2026-10-01_v4-v6_overnight.md`](../../../../../../docs/experiments/2026-10-01_v4-v6_overnight.md) の追記。

## v6 から何を・なぜ変えたか

| 項目 | v6 | v7 | 理由 |
|---|---|---|---|
| ロープのめり込み解消速度 (`rope_max_depenetration_velocity`) | 0.4 m/s | **ロープの既定** (`None`。`Knot` で 0.96 m/s、v4 と同じ) | 上の切り分け |
| 離す高さ (`grasp_limits.release_tip_z`) | 床上 5〜10 cm | v6 のまま | 学習を妨げず、v4 からの発散の減少分のおよそ半分を保つ |
| それ以外 (観測 `next_edge`・報酬・終了・アーカイブ・PPO・Play) | | v6 のまま | 1 版 1 変数 |

観測は v6 と同じ (197 次元) なので、v6 のチェックポイントから `--resume` で暖機開始できる
(ただし方策が学んだ物理とは違う)。

### 発散はどこまで戻るか

学習ログの「発散で終わったエピソード」は v4 5.5% / v7 3.7% / v6 1.6% (iteration 34〜43)。v5 の
発散対策のうち、めり込み解消速度の分を手放し、離す高さの分は残している。v5 の README の計測
(学習済みの v3 方策での 1 手あたりの発散) では、めり込み解消速度を下げた分の減少のほうが大きかったが、
その減少は学習を止める副作用と引き換えだった。

## 見るべき指標

v6 と同じ (`../v6/README.md` の「見るべき指標」)。特に:

* Play (chain モード) での固結び到達率 (`scripts/rsl_rl/eval_chain.py`)。版の比較はこれで行う
  (学習ログの `chain_est` は本番の成功率とずれる。[`docs/experiments/2026-10-01_v4-v6_overnight.md`](../../../../../../docs/experiments/2026-10-01_v4-v6_overnight.md) §4.5)。
* `Curriculum/stage_log/success_s2` と `timeout_s2`: 段 2 の学習が v4 並みに進み、時間切れが増えないこと。
* `Curriculum/stage_log/timeout_s0`: 学習の最初から v4 並み (約 5%) であること。

## 結果

**v7 としての学習はまだ無い** (2026-10-02 時点。`logs/rsl_rl/knot_tying_v7/` は無い)。v7 と同じ設定の run は v6 の切り分けの
`logs/rsl_rl/knot_tying_v6_ablation/2026-10-02_12-33-35_abl_depB_s42` (v6 + Hydra の上書き、seed 42、iteration 70) だけ。
その chain 評価は [`docs/experiments/2026-10-01_v4-v6_overnight.md`](../../../../../../docs/experiments/2026-10-01_v4-v6_overnight.md) の追記、run の一覧と保留した確認の学習の計画は
[`docs/experiments/2026-10-01_v4-v7_readme-notes.md`](../../../../../../docs/experiments/2026-10-01_v4-v7_readme-notes.md)。
次の候補 (シードを増やす、めり込み解消速度の中間値 0.6〜0.7 m/s など) は同じ報告書の「追記の後の次の候補」。

### 時間切れの罰則 (試して外したもの)

当初の v7 は「v6 + 時間切れの罰則 −0.2」だった。切り分けで段 2 の成功率を上げず (学習ログ 0.07)、chain 評価では段 2 に
届く割合を 46% に下げて発散を 20% に上げた (21.3%) ので、報酬項 (`stage_time_out`) ごと削除した。その run は
`logs/rsl_rl/knot_tying_v6_ablation/2026-10-02_12-34-00_abl_toP_s42`。
