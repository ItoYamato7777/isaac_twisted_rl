# v14 の学習の前のゲート (DR の読み戻し・ゲート 1・発散の計測・スモークテスト) (2026-10-14)

- 状態: 凍結
- 読むべき状況: v14 を学習する前に、ロープの物理の DR の範囲を決めるとき。DR の端の物理で台本・ランダムな手がどうなるかを知りたいとき

## 目的と仮説

v14 ([0027](../decisions/0027-v14-s0-reshape-dr-priv-critic.md)) の 3 つの変更が配線どおりに動くかを確かめ、ロープの物理の DR の範囲 (6 軸とも公称の ×0.5〜×2) が
台本で固結びを結べ、発散が公称の 2 倍を超えない範囲かを測る (skill `physics-check`)。外れる軸があれば範囲を狭める案を出す。

## 条件

| 項目 | 値 |
|---|---|
| commit | 読み戻しは `7c09f7c`、ゲート 1・発散・スモークテストは `03480c9` (未コミットの差分: 未追跡の `src/isaac_twisted_rl/scripts/rsl_rl/dump_real_deploy_check.py` だけ。v14 と無関係) |
| タスク ID とプリセット | `Isaac-Twisted-RL-Knot-v14` (`Knot`、学習の設定。DR あり)。スモークテストの評価は `Isaac-Twisted-RL-Knot-Play-v14` |
| 物理の条件 | 下の表。公称は `events.rope_dr.params.enabled=False`、端の組は `events.rope_dr.params.ranges={...}` で全軸を 1 点にした |
| 読み戻し | `demos/rope_dr_check.py --headless --device cuda:0` (64 env、reset 2 回)。job `outputs/jobs/2026-10-14_23-17-53_rope_dr_check` |
| ゲート 1 | `overhand_script.py --task Isaac-Twisted-RL-Knot-v14 --num_envs 128 --jitter 0.25 --headless --out outputs/overhand_script/v14_<条件> --cfg "<上書き>"` (遅い尺、まっすぐなロープから。DR は 512 env)。job `outputs/jobs/2026-10-14_23-20-59_gate1_v14` (コマンドは `cmd.txt`) |
| 発散の計測 | `divergence_bench.py --task Isaac-Twisted-RL-Knot-v14 --num_envs 1024 --steps 24 --headless --label v14_<条件> --cfg "<上書き>"`。一様乱数の行動、始まりは学習と同じ (段 0〜2 のアーカイブとまっすぐ)。job `outputs/jobs/2026-10-14_23-20-59_div_v14`、出力 `outputs/divergence_bench/v14_<条件>/` |
| 発散の判定 | 終了項 `diverged` が立った手 (1 手あたり)。下の「エピソード」はエピソードの終わり方のうち `diverged` の割合 |
| スモークテスト | `run_job.sh --name smoke_v14 ... -- uv run train --task Isaac-Twisted-RL-Knot-v14 --headless --device cuda:0 --seed 42 --num_envs 1024 --max_iterations 2 --run_name smoke_v14 agent.device=cuda:0 env.rewards.record_stage_archive.params.history_every=4`。run `logs/rsl_rl/knot_tying_v14/2026-10-14_23-41-33_smoke_v14`。評価は `eval_chain.py --task Isaac-Twisted-RL-Knot-Play-v14 --num_envs 64 --seed 7 --checkpoint <run>/model_1.pt` (job `outputs/jobs/2026-10-15_00-16-19_smoke_eval_v14`) |
| 学習の GPU | v13 s43 / s44 の学習と同じ GPU で並べて回した (時間は参考にならない) |

## 結果

### 読み戻し

- 曲げ・捩りのばね定数と減衰、質量、慣性、ロープと指の摩擦の読み戻しは、公称値 × 引いた倍率と一致した (最大の相対誤差 8e-8)。reset を 2 回して 2 回とも。
- 捩りの DOF は 27 (`JointNN:0`)、曲げは 54。公称の摩擦はロープ・指とも 0.5 ([2026-10-14_friction-binding](2026-10-14_friction-binding.md))。
- 観測 `rope_params` (6) は `domain_rand.normalize(倍率)` と一致。1 手の後に NaN の env は 0 / 64。

### ゲート 1 と発散 (条件ごと)

| 条件 (倍率: 曲げ・捩り・ζ・質量・ロープ μ・指 μ) | ゲート 1 の固結び | ゲート 1 の発散 | 1 手あたりの発散 | 公称との比 | 把持 (carry の終わりに 5 cm 上がる) | エピソードの終わり方の発散 |
|---|---|---|---|---|---|---|
| 公称 (DR なし) | 126/128 | 0 | 2.90% | 1.0 | 99.2% | 6.0% |
| DR (6 軸とも ×0.5〜×2) | 416/512 (81%) | 18 | 4.58% | 1.6 | 97.2% | 9.5% |
| 剛い・重い・滑る (2・2・1・2・0.5・0.5) | 28/128 | 0 | 0.86% | 0.3 | **63.7%** | 2.1% |
| 柔らかい・軽い・滑る・減衰小 (全部 0.5) | 77/128 | 15 | **7.78%** | **2.7** | 99.1% | 15.2% |
| 剛い・軽い・減衰小・摩擦大 (2・2・0.5・0.5・2・2) | 38/128 | 16 | **11.42%** | **3.9** | 99.3% | 23.9% |

### DR の軸別 (DR の run を、軸ごとに倍率の 3 分位で分けた)

| 軸 | ゲート 1 の固結び (小 / 中 / 大) | 1 手あたりの発散 (小 / 中 / 大) |
|---|---|---|
| `bend_stiffness` | 73% / 88% / 82% | 4.14% / 4.41% / 5.20% |
| `twist_stiffness` | 77% / 83% / 84% | 5.70% / 4.52% / 3.53% |
| `damping_ratio` | 85% / 76% / 82% | **7.06%** / 4.11% / 2.58% |
| `mass` | 82% / 95% / **66%** | 4.88% / 4.33% / 4.53% |
| `rope_friction` | 87% / 85% / 72% | 3.37% / 4.43% / 5.94% |
| `finger_friction` | 82% / 82% / 79% | 3.98% / 4.19% / 5.58% |

3 分位の倍率の平均は、小 ×0.63〜0.66、中 ×1.00〜1.06、大 ×1.56〜1.65。

### スモークテスト

- `params/env.yaml` に `save_on: stage_up_s0_reshape`・`s0_min_change: 0.02`・`rope_dr` (`enabled: true`、6 軸の範囲)、`params/agent.yaml` に `obs_groups.critic: [rope_params, policy]`・`BoundedPPO`・`bounds_loss_coef: 0.01`。
- チェックポイントの actor の入力 197、critic の入力 203。`export_policy.py` で書き出せた。
- 段 0 (2 iteration、1024 env): `s0_reshape_saved` 126.5 -> 98 (= `saved_s0`)、`arc_size_s0` 128 -> 419 -> 887。`s0_change_mean` 0.080 / 0.105 m、`s0_change_frac` 0.74 / 0.99 (段 0 で交差なしのままだった手のうち 0.02 m 以上変わった割合。1 エピソード 1 回の制限の前)。
- `restore_mismatch`・`save_stage_mismatch` は 0。写しが `archive_history/archive_step{4,8}.pt` に書かれた。
- Play (chain 評価、64 エピソード) が v14 のチェックポイントで最後まで回った (固結び 0%。2 iteration の方策なので値に意味は無い)。

## 結論

- 分かった:
  - DR の書き込み・critic の特権観測・段 0 の保存の規則は、配線どおりに動く。
  - 公称の物理ではゲート 1 が 126/128 で、v14 の DR の範囲全体では 81%。発散は DR 全体で公称の 1.6 倍。
  - **減衰比 ζ の小さい側が発散の 2 倍の基準を超える。** DR の run の ζ の小さい 3 分の 1 (平均 ×0.64) で 7.06% (公称の 2.4 倍)。ζ ×0.5 を含む端の組は 2.7 倍・3.9 倍で、ζ と質量が両方小さい (動きの速い) 組ほど悪い。
  - 剛い・重い・滑るの組は把持が 64% に落ち、台本は 22% しか結べない (発散は少ない)。DR 全体の把持は 97%。
  - 段 0 で交差なしのまま形を変える手は多い (0.02 m を超えるのが 74〜99%)。閾値 0.02 m は「ほとんど動かさない手」を除くだけで、段 0 のアーカイブへの流入は 1 エピソード 1 回の制限で決まる。
- 推測:
  - ζ を小さくすると、臨界減衰より振動しやすくなり、軽いリンクほど速度が跳ねて発散の判定 (速さ 20 m/s など) に掛かる。
  - 端の組の台本の低さは、台本 (開ループ・公称の物理に合わせた手) が物理に合わないためで、学習した方策ではもっと高い可能性がある。6 軸がそろって端にある組は対数一様の DR ではほとんど引かれない。
- 未確認:
  - 学習の中での DR の端の発散率 (方策の手はランダムな手と違う)。
  - 段 0 のアーカイブに入った形の多様さ (2 iteration では見ていない)。

## 次の作業

- **学習の前に、DR の範囲をユーザーに確かめる。** 案: ζ (`damping_ratio`) の下限を ×0.5 から ×1.0 に上げる (臨界減衰より下げない)。他の 5 軸は ×0.5〜×2 のまま。v14 の `DR_RANGES` を変えるのは、まだ学習していないので v14 の中で直してよい。
- 範囲が決まったら、v13 の学習が終わって GPU が空いてから、v14 seed 43 / 44 を iteration 300 まで (skill `train-eval`)。
