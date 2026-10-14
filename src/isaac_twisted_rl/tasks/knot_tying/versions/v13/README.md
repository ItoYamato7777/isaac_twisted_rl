# v13: v12 + PPO の損失に「方策の平均が行動の範囲 [−1, 1] の外へ出た分」の罰則を足す

```bash
uv run train --task Isaac-Twisted-RL-Knot-v13 --headless           # 段ごとの MLP + BoundedPPO
uv run play  --task Isaac-Twisted-RL-Knot-Play-v13 --num_envs 16   # まっすぐなロープから段をつないで評価 (chain モード)
```

ログ: `logs/rsl_rl/knot_tying_v13/`。**学習には `Knot` プリセットを使う** (v3〜v12 と同じ)。継承元は **v12**。
仕様と理由の正本は [`docs/decisions/0026-v13-action-bounds-loss.md`](../../../../../../docs/decisions/0026-v13-action-bounds-loss.md)。ここには配線と、見るべき指標を書く。
環境は v12 と同じなので、初期アーカイブの作り直しは要らない。

## なぜ作ったか

v7・v11・v12 のどれも、chain 評価の山が iteration 70〜100 にあり、その後に下がる。どの run でも、段ごとの方策の平均が
行動の範囲 [−1, 1] の外へ出る割合が学習とともに増え、chain 評価が大きく崩れた run では段 0 の置き方のチャネル
(dx・release_z・yaw) まで大半が範囲の外だった (相関のみ)。環境は行動をクリップし、PPO の log prob はクリップ前の行動で
計算するので、平均が範囲の外にあると、平均を範囲の中へ戻す勾配が無い。罰則でそれを与える。

## v12 から何を・なぜ変えたか

| 項目 | v12 | v13 | 理由 |
|---|---|---|---|
| PPO の algorithm (`algorithm.class_name`) | `PPO` | **`BoundedPPO`** (`common/policies/bounded_ppo.py`) | 損失に項を足す口が rsl-rl に無いため |
| 損失 | surrogate + value − entropy | 同じ + **`bounds_loss_coef · mean_batch(Σ_チャネル max(abs(μ) − 1, 0)²)`**、`bounds_loss_coef` = **0.01** | 上。範囲の中では 0 で、クリップ後の行動は変えない。係数の理由は 0026 の「係数」 |
| それ以外 (観測・報酬・終了・アーカイブの保存の規則 `stage_up`・物理・方策の構造・PPO の他の値) | | v12 のまま | 1 版 1 変数 |

μ はクリップ前の平均。境目の 1.0 は環境がクリップする値 (`PickPlaceAction.process_actions`) と同じ。
v12 との差がこれだけであることは `tests/test_v13_cfg.py` が静的に検査する。`BoundedPPO` の勾配が「損失に罰則を足した」ものと
一致することは `tests/test_bounded_ppo.py` が確かめる。

## 見るべき指標

版の比較は chain 評価で行う ([`docs/system/evaluation.md`](../../../../../../docs/system/evaluation.md))。比べる相手は v12 の同じ seed。
学習ログは代理の指標で、版の良し悪しの結論には使わない。

* chain 評価の `knot` を iteration 70 / 100 / 150 / 200 / 250 / 300 で v12 と並べる。0026 の予想では、2 シードとも it300 が it70・it100 の高い方から 5 ポイント以上は下がらない。
* 段 0 の置き方のチャネル (dx・release_z・yaw) で |μ| > 1 の割合 (`action_saturation.py --run v13_sS=<run>`)。0026 の予想では iteration 300 まで 20% を超えない。
* `Loss/bounds`: mini-batch の罰則 (係数を掛ける前) の平均。範囲の外への出方の学習のバッチでの値。
* `Curriculum/stage_log/timeout_s0`: 0026 の予想では s44 でも 0.1 を超えない。
