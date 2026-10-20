# v17: v15 + 待ちのフェーズの尺を詰める

```bash
uv run train --task Isaac-Twisted-RL-Knot-v17 --headless           # v15 + 待ちのフェーズの尺を詰めた 1 手 (591 物理ステップ)
uv run play  --task Isaac-Twisted-RL-Knot-Play-v17 --num_envs 16   # v15 と同じ chain 評価 (18 手。尺は v17)
```

**不採用** (2026-10-20、ユーザー。固結びの成功率が v15 より下がった。結果は [`docs/experiments/2026-10-20_v17_shorter-wait-phases.md`](../../../../../../docs/experiments/2026-10-20_v17_shorter-wait-phases.md))。後の版は v17 を継承しない。下の hiro_knot の更新も要らない。

ログ: `logs/rsl_rl/knot_tying_v17/`。**学習には `Knot` プリセットを使う** (v3〜v16 と同じ)。継承元は **v15** (v16 の寸法の DR は継承しない。ユーザー、2026-10-19)。
仕様と理由の正本は [`docs/decisions/0030-v17-shorter-wait-phases.md`](../../../../../../docs/decisions/0030-v17-shorter-wait-phases.md)。ここには配線と、見るべき指標を書く。
初期アーカイブは v15 と同じ (形の静的な配置なので、尺を変えても作り直さない)。

## なぜ作ったか

学習の時間の 95〜97% は PhysX で、1 物理ステップの重さはフェーズによらないので、1 手の物理ステップ数がそのまま速さになる。物理を軽くする案 (反復・刻み) は v14 の方策の chain 評価で固結びが 4〜17 ポイント下がったが、待ちのフェーズを詰めた尺 (計測の D4) は −0.3 ポイントで発散も増えなかった ([計測](../../../../../../docs/experiments/2026-10-16_physics-speed.md))。ただしこれは v3 の尺で学んだ方策の評価で、詰めた尺で 0 から学んだ結果ではない。採否は学習と chain 評価で決める。

## v15 から何を・なぜ変えたか

| フェーズ | v3〜v16 (`KNOT_DURATIONS`) [s] | v17 [s] |
|---|---|---|
| close | 0.3 | **0.2** |
| hold | 0.2 | **0.1** |
| release | 0.25 | **0.15** |
| settle_out | 0.3 | **0.15** |
| relax | 0.5 | **0.25** |
| descend・lift・lower | 0.3 | 0.3 |
| carry | 0.6 | 0.6 |
| retreat | 0.105 | 0.105 |
| approach・clear | 0 (1 ステップ) | 0 (1 ステップ) |

- 1 手は 759 → **591 物理ステップ** (3.1625 s → 2.4625 s)。速さは 1.28 倍の見込み。
- 移動系は詰めない: 運ぶ速さが発散の原因になる (v3 の `KNOT_DURATIONS` の docstring)。
- **物理は変えない**: dt 1/240、ソルバの反復、めり込み解消速度、接触、摩擦、物理の DR は v15 のまま。行動・観測・報酬の項・終了・段の定義・PPO も v15 のまま。
- 配線は `durations` のフィールドの差し替えだけ。基底の `__post_init__` が `decimation`・`episode_length_s`・`render_interval` と報酬の重みの `step_dt` のスケールを計算し直すので、報酬の重みは「1 手あたり」の大きさのまま (v15 と同じ)。**尺を Hydra の上書きで変えない** (これらが追随せず、実効の報酬が 0.78 倍になる)。
- **Play は `episode_length_s` を 17.5 手ぶんにする**。Isaac Lab の手数の上限 `ceil(episode_length_s / (dt × decimation))` は、v4 の Play の式 (`18 × decimation × dt`) のままだと 591 で浮動小数の誤差のため 19 になる (759 では 18)。`demos/divergence_bench.py`・`bench_step_time.py`・0030 の計測の chain 評価と同じ対処。学習 (6 手) は 591 でも 6 ちょうどなので、基底の計算のまま。

尺を詰めると、1 手の終わりにまだ 0.05 m/s を超えて動いているロープが 7% → 20% に増え (速さの p90 は 0.1〜0.16 m/s)、settle_out の終わりに動いているロープは 29% → 68% になる (計測は v14 の方策)。段の判定と次の手の観測はほとんど変わらなかった (推測)。

## hiro_knot 側の更新の要否

**要る (v17 の方策を実機で使うとき)**。観測・行動・段の定義・`common/topology/` は変えていないので、方策の入出力の移植 (`observations`・`decode`・`topo`) はそのまま使える。尺だけが食い違う。`~/work/hiro_knot` (2026-10-19 の作業ツリー) を読んで確かめたこと:

| hiro_knot の箇所 | 尺の扱い | v17 の run で起きること |
|---|---|---|
| `hiro_knot/policy/constants.py` の `DURATIONS` | v3〜v16 の尺を直書き (`env.yaml` の `actions.pick_place.durations` と同じ値) | — |
| 同 `check_env_yaml()` (起動時の `app/run.py`、`tools/use_policy.py` の手順 5、`tests/policy/test_config_policy.py`) | `DURATIONS` を run の `env.yaml` と照らす | close・hold・release・settle_out・relax の 5 項目が食い違いとして出る (`use-policy` を通らない) |
| `tests/policy/test_config_policy.py` | `sum(phase_steps()) == decimation == 759` | v17 の値 (591) では落ちる |
| `hiro_knot/policy/trajectory.py` の `move_segments` | 各区間の `duration`・`steps` (sim の軌道の再現) | 尺を v17 にしないと sim の 1 手と合わない |
| `hiro_knot/robot/planner.py` の `real_time` | 待ち (`settings.WAIT_PHASES` = hold・settle_out・relax) は sim の尺そのまま、ほかは × `TIME_SCALE` (5.0) | v17 の尺を入れると実機の待ちは hold 0.2 → 0.1 s、settle_out 0.3 → 0.15 s (+ `EXTRA_RELEASE_WAIT` 0.5 s)、relax 0.5 → 0.25 s。指は close 1.5 → 1.0 s、release 1.25 → 0.75 s |
| `isaac_server/move.py` | `env.cfg.decimation` を起動したタスクから読む | `--task Isaac-Twisted-RL-Knot-Play-v17` で起動すれば追随する |

- 直し方の判断は hiro_knot 側に任せる (例: `DURATIONS` を run の `env.yaml` から読む。版ごとに定数を持つ)。このリポジトリからは変更しない。
- **実機の待ちを sim に合わせて詰めるかは別の判断**。実機の紐は sim の紐と落ち着き方が違いうる。v17 は sim でも 1 手の終わりに動いているロープが増える (上の 7% → 20%) ので、実機で待ちを短くすると、観測の時にまだ動いている紐が sim より多くなりうる。実機の待ちは sim の尺より長くてよい (待ちは方策の入出力ではない) が、settle_out を延ばすと「離したのに付いてくる」の起き方が sim と変わる ([0004](../../../../../../docs/decisions/0004-release-follow-physics.md))。

## 見るべき指標

- chain 評価の `knot` を v15 の同じシード (s43・s44、it150・200・250・300 の平均) と比べる (0030 の予想 1)。
- `bench_step_time.py` (4096 環境、GPU を取り合っていない時) の 1 手の時間が v15 の 0.78 倍前後 (予想 2)。学習ログの収集時間では比べない。
- `Curriculum/stage_log/restore_mismatch`・`save_stage_mismatch` が 0 のまま、`Episode_Termination/diverged` が v15 と 2 ポイント以内 (予想 4)。
- 起動の確認: `bench_step_time.py` の `meta` で `decimation` 591・`max_episode_length` 6 (Play は 18)、スモークテストの `params/env.yaml` で `durations` と `actions.pick_place.durations` の両方が上の表の値。
