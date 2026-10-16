# 0030 v17 は待ちのフェーズの尺を詰める (学習を速くするため。物理は変えない)

- 状態: 有効 (v17 以降。2026-10-16 にユーザーが実装の予約を指示した。**未実装**。継承元は v15 の結果が出てから決める)
- 読むべき状況: v17 を実装・学習・評価するとき。フェーズの尺 (`PhaseDurations`)・1 手の物理ステップ数 (`decimation`)・報酬の重みの `step_dt` のスケールを変えるとき。学習の速さのために物理の設定を変えたくなったとき
- 最終更新: 2026-10-16

## 決定 (案。数値は計測で決めた。ユーザーの承認は「新しい版として実装を予約する」まで)

v17 = 継承元 (v14 または v15。**v15 の結果が出てから決める**。目安: v15 が v14 と同じかそれ以上なら v15、崩れたら v14) + **待ちのフェーズの尺を詰める**。

| フェーズ | 継承元 (v3 の `KNOT_DURATIONS`) [s] | v17 [s] |
|---|---|---|
| close | 0.3 | **0.2** |
| hold | 0.2 | **0.1** |
| release | 0.25 | **0.15** |
| settle_out | 0.3 | **0.15** |
| relax | 0.5 | **0.25** |
| descend・lift・lower | 0.3 | 0.3 (そのまま) |
| carry | 0.6 | 0.6 (そのまま) |
| retreat | 0.105 | 0.105 (そのまま) |
| approach・clear | 0 (1 ステップ) | 0 (そのまま) |

- 1 手は 759 → **591 物理ステップ** (3.1625 s → 2.4625 s)。計測では 1 手のステップ数に比例して速くなる (**1.28 倍**の予測。[計測](../experiments/2026-10-16_physics-speed.md))。
- **物理は変えない**: dt 1/240、ソルバの位置反復 8・速度反復 1、めり込み解消速度 0.96 m/s、接触の設定、摩擦はすべて継承元のまま。行動・観測・報酬の項・終了・PPO・DR・段の定義も継承元のまま。
- 1 版 1 変数 ([0022](0022-p2-one-version-each.md))。反復 6 など物理を軽くする案はこの版に混ぜない (下の「検討して採らなかった案」)。
- 報酬の重みは、版の `durations` から基底の `__post_init__` が `step_dt` (= 591 / 240 = 2.4625 s) で割り直すので、「1 手あたり」の大きさのまま (継承元と同じ)。
- 学習: seed 43 / 44、0 から iteration 300 (継承元のチェックポイントからは再開しない)。比べる相手は継承元の同じシード。

## 理由

v14 の方策 (it300、2 シード) を、尺だけ変えた環境で chain 評価した ([計測](../experiments/2026-10-16_physics-speed.md))。

| 条件 | 速さ | chain の固結び s44 / s43 | 発散 s44 / s43 |
|---|---|---|---|
| 継承元 (v14 の尺) | 1.00 | 57.0% / 58.2% | 14.1% / 14.3% |
| D2 (settle_out 0.15・relax 0.25) | 1.14 | 57.0% / 56.3% | 12.7% / 11.9% |
| **D4 (上の表)** | **1.28** | **55.5% / 59.2%** | 14.3% / 11.7% |

- D4 は固結びが平均 −0.3 ポイント (95% 区間は ±4.3 ポイント程度) で、発散も増えなかった。学習の尺のゲート 1 (台本) も基準と同じくらい (64/128 対 68/128)。
- 同じ速さの変更でも、反復 6 は −3.8、200 Hz は −6.0、180 Hz は −11.2、反復 4 は −16.8 ポイント (どれも発散が増えた)。尺は、質が落ちない中で最も大きく速くなる。
- 時間の 95〜97% は PhysX で、1 ステップの重さはフェーズによらない。尺を詰めた分がそのまま速さになる。
- 尺を詰めると、1 手の終わりにまだ 0.05 m/s を超えて動いているロープが 7% → 20% に増える (速さの p90 は 0.1〜0.16 m/s)。それでも段の判定と次の手の観測がほとんど変わらなかった (推測)。ここは学習で確かめる (予想 4)。

**限界**: 方策は継承元の尺で学んだもので、尺を変えて **ゼロから学んだ結果ではない**。物性の変更は計測と学習で結果が逆になった前例がある ([0021](0021-measure-physics-changes-by-training.md))。尺は物性ではないが、離したロープが落ち着く前に次の手の観測を取るので、学習の信号が変わりうる。採否は学習と chain 評価で決める。

## 外れたら仮説が誤りと言える予想 (数値は暫定。ユーザーと決める)

v17 の 2 シードを、継承元の同じシードと比べる。chain 評価は標準の条件 (`--seed 7`、512 環境、公称の物理)。

1. **質が保たれる**: it150・200・250・300 の 4 点の `knot` の平均が、2 シードとも継承元の同じシードの平均から 5 ポイントを超えて下がらない。隣のチェックポイントは平均 9 ポイント動くので、1 点の差では見ない。
2. **速くなる**: `bench_step_time.py` (4096 環境、GPU が他のジョブと取り合っていない時) で 1 手の時間が継承元の 0.78 倍 (0.75〜0.81) になる。学習ログの 1 iteration の収集時間は、GPU の取り合いで 2 倍以上動くので比べない。
3. **同じ壁時計の時間で進む**: v17 の iteration N の chain 評価が、継承元の iteration 0.78 N 以下 (同じ時間で到達する iteration) と同じかそれ以上。1 と 2 が当たれば、この予想も当たる。
4. **段の判定がずれない**: 学習ログの `restore_mismatch`・`save_stage_mismatch` が 0 のまま (v14 のスモークテストは 0)。`Episode_Termination/diverged` が継承元と 2 ポイント以内 (v14 は約 3%)。

読み方: 1 が外れ、2 が当たれば、尺を詰めて速くなった分、学習の信号が悪くなった (落ちたフェーズを D2 などに戻して切り分ける)。1 と 2 の両方が当たれば、尺の変更は速さの利得だけを与えた。1 が当たって 2 が外れれば、1 手の時間は尺以外 (reset・ハンドの制御) で決まる部分が思ったより大きい。

## 検討して採らなかった案

| 案 | 採らなかった理由 |
|---|---|
| Hydra の上書きだけで尺を変える (版を作らない) | Hydra の上書きは `__post_init__` の後に効くので、`decimation`・`episode_length_s` と報酬の重みの `step_dt` のスケール (`_rescale_reward_weights`) が追随しない。実効の報酬が 0.78 倍になる。`env.durations.*` と `env.actions.pick_place.durations.*` の両方を書く必要もある。版の設定クラスの `durations` なら、基底が全部を計算し直す |
| D2 だけ (settle_out と relax だけを詰める) | 速さが 1.14 倍で小さい。D4 と質は同じだった |
| 反復 6 (1.25 倍) を同じ版に混ぜる | 1 版 1 変数。D4 + 反復 6 は s43 だけ大きく落ちた (固結び 59.2% → 43.8%、発散 34.2%)。それぞれを学習で確かめてから組み合わせる |
| 反復 4・180 Hz・200 Hz・反復 6 + 180 Hz | v14 の方策の chain 評価で −6〜−23 ポイント。ゲート 1 も 180 Hz・反復 4 は基準を下回った |
| 移動系 (descend・lift・carry・lower) をさらに詰める | `Knot` は carry 0.35 s で 1 手目に 13/32 が発散した ([gate1](../experiments/2026-09-27_gate1.md))。今の尺 (carry 0.6 s) の根拠。詰める余地は待ちのほうにある |
| 待ちを 0 近くまで詰める | settle_out は「離したのに付いてくる」を防ぐ物理の待ち ([0004](0004-release-follow-physics.md))。D4 の settle_out の終わりに動いているロープは 68% (v14 は 29%)。これ以上は計測していない |

## 実装の見通し (2026-10-16 のコードで確かめた)

(`kt/` = `src/isaac_twisted_rl/tasks/knot_tying/`。手順は skill `add-version`)

1. **`kt/versions/v17/`** を作る (`__init__.py`・`agents/__init__.py` は docstring だけ。Kit の起動前に import されるので isaaclab を import しない)。
   - `env_cfg.py`: 継承元 (`KnotTyingV15EnvCfg` か `KnotTyingV14EnvCfg`) を継承し、`durations` のフィールドだけを上書きする。v3 の `KNOT_DURATIONS` と同じ書き方で、全項目を明示する。

     ```python
     V17_DURATIONS = phases.PhaseDurations(
         approach=0.0, descend=0.3, close=0.2, hold=0.1, lift=0.3, carry=0.6,
         lower=0.3, release=0.15, settle_out=0.15, clear=0.0, retreat=0.105, relax=0.25,
     )
     @configclass
     class KnotTyingV17EnvCfg(<継承元>EnvCfg):
         durations: phases.PhaseDurations = V17_DURATIONS
     ```

     `__post_init__` は書かない。基底の `__post_init__` が `durations` から `decimation` (591)・`episode_length_s`・`render_interval`・報酬の重みのスケールを計算し、`make_preset_cfgs` が `_apply_rope` で `actions.pick_place.durations` に渡す。
   - `play_post_init`: 継承元のものをそのまま呼ぶ。Play の `episode_length_s` は `_play_post_init` が `cfg.decimation` から作り直すので、尺に追随する。
   - `agents/rsl_rl_ppo_cfg.py`: 継承元の Runner を継承し、`experiment_name = "knot_tying_v17"` だけを書く。
   - `README.md`: この文書の「決定」と「理由」の要約 (結果は書かない)。
2. **登録と一覧**: `kt/__init__.py` の `_VERSIONS` に `"v17"`、`kt/versions/__init__.py` の docstring の一覧と `kt/README.md` の版の系譜に 1 行ずつ、`docs/STATUS.md` の版の状態の表。**番号は v17** (v16 は [0029](0029-v16-rope-size-dr.md) の予約で、実装が v15 の結果の後なので、v17 を先に実装してもよい。v16 と v17 のどちらを先に学習するかはユーザーが決める)。
3. **確かめること** (学習の前。skill `physics-check` と skill `train-eval` のスモークテスト):
   - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run python -m pytest tests/ -q` (版の一覧の整合と `__init__.py` の import)。
   - `bench_step_time.py --task Isaac-Twisted-RL-Knot-v17 --num_envs 256 --moves 2` の `meta` が `decimation` 591、`max_episode_length` 6、`phase_steps` が上の表と合う。Play (`Isaac-Twisted-RL-Knot-Play-v17`) は `max_episode_length` が 18 になること。**積の浮動小数の誤差で手数が 1 つ増える事故** (`divergence_bench.py` の注: descend 0.5 s = 807 ステップで 12 手ちょうどが 13 手になった) がここで起きないことを見る。起きたら `episode_length_s` を半手ぶん引く。
   - ゲート 1 (台本、`--fast` で学習の尺): 継承元と同じくらい (v14 の尺で 68/128、D4 で 64/128)。初期アーカイブ (`outputs/archive_seed/overhand_28links.pt`) は形 (静的な配置) なので作り直さない。
   - 1 手あたりの発散 (`bench_step_time.py` か `divergence_bench.py`): 継承元の 3.4% に対して D4 は 3.8% だった。5% を超えたら学習に進まない。
   - スモークテスト (`--max_iterations 2`) で、`params/env.yaml` の `durations` と `actions.pick_place.durations` の両方が上の表の値、`decimation: 591` であること。
4. **学習と評価** (継承元の結果が出て、GPU が空いてから): 継承元と同じコマンドを、`--task Isaac-Twisted-RL-Knot-v17` と `--run_name v17_sS` に変えて回す。Hydra の上書きは継承元と同じものだけを付ける (尺は上書きしない)。評価は `eval_chain.py --task Isaac-Twisted-RL-Knot-Play-v17` に、学習と同じ Hydra の上書きと `--device cuda:N` `agent.device=cuda:N` を付ける。**学習を始める前にコードを commit する。**

## 実装の予約

- 実装・テスト・ゲートは先に進めてよい。ただし GPU を使うもの (ベンチ・ゲート 1・スモークテスト) は、v15 の学習 (2 本とも 10-16 17:07 起動。GPU 0・1 を使用中) が終わるか、ユーザーが空きを確かめてから。
- 学習は v15 の結果が出て、継承元を決めてから。
- v17 の結果が良ければ、次の候補は反復 6 (1.25 倍。Hydra の上書きで学習できる。別の学習として、v17 と同じシード・評価で比べる)。D4 + 反復 6 はそれぞれが学習で確かめられてから。

## 出典

- [2026-10-16_physics-speed.md](../experiments/2026-10-16_physics-speed.md): 計測の全体 (速さの内訳・chain 評価・ゲート 1・ロープの速さの推移)。
- [0020](0020-v4-solver-iterations-and-steps-per-iteration.md): 時間の大半は PhysX の反復という前の計測。
- [0021](0021-measure-physics-changes-by-training.md): 計測と学習の結果が逆になった前例。
- [0004](0004-release-follow-physics.md)・[0005](0005-measure-after-retreat.md): 待ちのフェーズを残している理由。
- `kt/common/mdp/phases.py` (`PhaseDurations` の docstring): 移動系を詰めたときの 2026-09-12 の実測。
- `kt/common/env_cfg_base.py` (`__post_init__`・`_rescale_reward_weights`): 尺から `decimation` と報酬の重みを計算する箇所。
