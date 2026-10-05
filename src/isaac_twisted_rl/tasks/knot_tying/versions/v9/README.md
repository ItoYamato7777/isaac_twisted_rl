# v9: v7 + 学習の初期状態で段 2 を選ぶ重みを 2 倍に

```bash
uv run train --task Isaac-Twisted-RL-Knot-v9 --headless           # 段ごとの MLP
uv run play  --task Isaac-Twisted-RL-Knot-Play-v9 --num_envs 16   # まっすぐなロープから段をつないで評価 (chain モード)
```

ログ: `logs/rsl_rl/knot_tying_v9/`。**学習には `Knot` プリセットを使う** (v3〜v8 と同じ)。継承元は **v7** (v8 ではない)。

## なぜ作ったか

v7 の chain 評価 (iteration 100、512 エピソード、seed 7) では、段 2 以上に届くのが 98〜99% なのに対して、
固結びは 53.7% / 47.3% (seed 43 / 44) に留まる ([`docs/experiments/2026-10-04_v7_seeds.md`](../../../../../../docs/experiments/2026-10-04_v7_seeds.md))。
学習ログ (iteration 90〜99) の段ごとの成功も、段 0 が 0.97、段 1 が 0.57、段 2 が 0.19〜0.24 で、段 2 -> 固結び (tuck) が一番低い。

詰まりは段 2 の tuck の成功の少なさ (学習量と物理のばらつき) にあり、段 2 から始めるエピソードを増やせば段 2 の方策の学習が進む、
という見込みで作った (推測。確かめるのがこの版の目的)。

v8 (v7 + 対称性のデータ拡張) は chain 評価で v7 を大きく下回り、原因が分かっていない
([`docs/experiments/2026-10-05_v8_symmetry.md`](../../../../../../docs/experiments/2026-10-05_v8_symmetry.md))。v8 の変更を混ぜると
v9 の差が読めなくなるので、v7 から派生させる。

### 同じ変更を v6 で試したときとの違い

v6 の切り分け (2026-10-02、seed 42、iteration 70) で、同じ `stage_weights=[1.0,1.0,2.0]` を Hydra の上書きで試した run
`knot_tying_v6_ablation/2026-10-02_12-34-26_abl_sw2_s42` がある。学習ログの段 2 の成功は 0.04 のまま、chain 評価は 20.7% で、
改善は見えなかった ([`docs/experiments/2026-10-01_v4-v6_overnight.md`](../../../../../../docs/experiments/2026-10-01_v4-v6_overnight.md) の追記)。
ただし v6 では、めり込み解消速度の上限 0.4 m/s が段 2 の学習を止めていた (v7 を作った理由。v6 は段 2 の時間切れ 0.3 前後の局所解)。
v7 の物理 (0.96 m/s) では段 2 の成功が伸びるので、学習量を増やす効果は v6 とは別に測る必要がある。

## v7 から何を・なぜ変えたか

| 項目 | v7 | v9 | 理由 |
|---|---|---|---|
| 学習の初期状態の段の重み (`events.reset_to_stage.params["stage_weights"]`) | (1, 1, 1) (v4 の既定) | **(1, 1, 2)** | 段 2 から始めるエピソードを約 1/3 から約 1/2 に増やす |
| それ以外 (観測・報酬・終了・アーカイブの容量と読み込み・`straight_prob`・行動・物理・PPO・Play) | | v7 のまま | 1 版 1 変数 |

* v7 との差がこの 1 項目だけであることは `tests/test_v9_cfg.py` が静的に検査する。
* Play (chain モード) は v7 と同じ: 段の重みを (1, 0, 0) に、`straight_prob` を 1.0 にする (`v4/env_cfg.py` の `_play_post_init` が
  v9 の `__post_init__` の後に掛かる)。chain 評価の条件は v7 と変わらない。
* v8 の `algorithm.symmetry_cfg` は入れない (PPO の設定は v7 のクラスを継承し、ログ名だけを変える)。
* 観測は v7 と同じ (197 次元) なので、v7 のチェックポイントから `--resume` で暖機開始できる。

### 他の設定との関係

`ResetToStage` (`common/mdp/archive.py`) は空の段を選ばず、段 0 は空でも選ぶ (まっすぐなロープがあるため)。
どの段のアーカイブも空でないときの、学習のエピソードの始まりの割合:

| 始まり | v7 (1, 1, 1) | v9 (1, 1, 2) |
|---|---|---|
| まっすぐなロープ (段 0 のうち `straight_prob = 0.5`) | 1/6 (約 17%) | 1/8 (12.5%) |
| 段 0 のアーカイブ | 1/6 | 1/8 |
| 段 1 のアーカイブ | 1/3 | 1/4 |
| 段 2 のアーカイブ | 1/3 | **1/2** |

段 2 のアーカイブは、台本の途中状態 (約 130 件) で始まり、学習中に段 1 から上がった形状で増える (容量 16384)。
段 2 から上がった形状 (固結び) は保存しない。学習の最初は同じ約 130 件の形状から始める割合が増える。

## 見るべき指標

版の比較は chain 評価で行う ([`docs/system/evaluation.md`](../../../../../../docs/system/evaluation.md))。v7 の既存の run
(seed 43 / 44、iteration 70 / 100 の `eval_chain_it{N}.json`) と比べる。学習ログは代理の指標で、版の良し悪しの結論には使わない。

* Play (chain モード) の固結び到達率 (`scripts/rsl_rl/eval_chain.py`)。差は v7 のシード差 (iteration 100 で約 6 ポイント) と
  標準誤差 (約 2 ポイント) を超えるかで読む。
* `Curriculum/stage_log/start_frac_s2`: 段 2 から始めた割合が約 0.5 になっていること (設定が効いているかの確認)。
* `Curriculum/stage_log/success_s2`・`timeout_s2`: 段 2 -> 固結びの成功が v7 (iteration 90〜99 で 0.19〜0.24) より速く伸びるか。
* `Curriculum/stage_log/success_s0`・`success_s1`・`timeout_s0`: 段 0・1 から始める割合が減る (1/3 -> 1/4) ので、段 0・1 の成功が
  v7 (0.97 / 0.57) より落ちないか。**まっすぐなロープから始める割合が 1/6 から 1/8 に下がる**ので、段 0 の方策の崩れは
  `Curriculum/stage_log/straight_success` (まっすぐなロープから始めて段が上がった割合) でも見る。chain 評価はまっすぐなロープから始まるので、ここが崩れると
  段 2 の改善を打ち消す。
* `Episode_Termination/diverged` と `Curriculum/stage_log/diverged_s2`: 段 2 から始めるエピソードが増えるので、学習中の発散の割合が v7 (2.6〜2.8%) から増えないか。
