# common/: 全バージョンで共有する仕組み

シーン・物理・行動・終了判定・起動時イベントは全バージョンで共通で、ここに実装がある。
バージョン (`versions/vN/`) が決めるのは **観測・報酬・終了・リセット・エピソード長・フェーズの尺・PPO 設定**と、
一部の物理の値 (v4 のソルバの位置反復、v5 以降のめり込み解消速度など。`KnotTyingEnvCfg` のフィールドの上書き)。
方針は [`../README.md`](../README.md) と [`docs/decisions/`](../../../../../docs/decisions/README.md) の「版 = 設定」。

この文書は「どのバージョンでも変わらない仕組み」を 1 か所にまとめたもの。
バージョン固有の値は各 `versions/vN/README.md` を参照。

## 1. 「1 手 = 1 RL ステップ」

`lift` や `reach` のような一般的な manager-based タスクでは、1 env ステップは
「関節を少し動かす」程度の粒度になっている。この環境では **1 env ステップ = pick&place の
1 手 (掴む→運ぶ→置く→離す) 全部** で、Options Framework 的な粒度になっている
(`docs/history/next_action.md` §4.0 の決定。今も有効な判断は [`docs/decisions/`](../../../../../docs/decisions/README.md))。

実現しているのは Isaac Lab の `decimation` の使い方である。

```python
# ManagerBasedRLEnv.step() の構造 (isaaclab 側)
self.action_manager.process_action(action)   # 1 env ステップに 1 回
for _ in range(self.cfg.decimation):
    self.action_manager.apply_action()       # 毎物理ステップ
    self.scene.write_data_to_sim()
    self.sim.step(render=False)
    self.scene.update(dt=self.physics_dt)    # rope.data がここで更新される
# 以降 termination / reward / observation を 1 回だけ計算
```

[`env_cfg_base.py`](env_cfg_base.py) の `KnotTyingEnvCfg.__post_init__` は次のように設定する。

```python
self.sim.dt = 1.0 / 240.0
self.decimation = self.durations.decimation(self.sim.dt)   # 1 手ぶんの物理ステップ数
self.episode_length_s = self.actions_per_episode * self.decimation * self.sim.dt
```

`decimation` は「1 手を完了するのに必要な物理ステップ数」になる。フェーズの尺
(`durations`) を変えると `decimation` も自動で追随する。

### 物理は 240 Hz

120 Hz にすると、把持中の接触でロープ速度が 60〜175 m/s まで跳ね上がり NaN になる。
発散率は 120 Hz で 11.2%、240 Hz で 1.2% だった
(`docs/history/next_action.md` §4.5。判断は [`docs/decisions/`](../../../../../docs/decisions/README.md) の 240 Hz)。

## 2. フェーズ ([`mdp/phases.py`](mdp/phases.py))

1 手は 12 個のフェーズで構成される。フェーズ列は全バージョン共通で、尺はバージョンごとに
選べる。尺 0 のフェーズは 1 物理ステップで通過し、補間せずに目標位置へ直接書き込まれる。

| フェーズ | 内容 | `DEFAULT` (v0, v1) | `SHORT` (v2, v2-R2) | `KNOT` (v3〜v7、rope study) |
|---|---|---|---|---|
| `approach` | 掴む点の真上へ移動 | 0.175 | **0** | **0** |
| `descend` | 下降 | 0.14 | 0.14 | **0.3** |
| `close` | 指を握り込む | 0.3 | 0.3 | 0.3 |
| `hold` | 接触の安定待ち | 0.2 | 0.2 | 0.2 |
| `lift` | 持ち上げ (運搬高さ 0.20 m 固定) | 0.14 | 0.14 | **0.3** |
| `carry` | 置く場所まで運搬 (yaw もここで回す) | 0.35 | 0.35 | **0.6** |
| `lower` | 置く高さまで下降 | 0.14 | 0.14 | **0.3** |
| `release` | 指を開く | 0.25 | 0.25 | 0.25 |
| `settle_out` | 離した後ロープが落ち着くのを待つ | 0.3 | 0.3 | 0.3 |
| `clear` | 真上に上げる前に横へ抜ける | 0.105 | **0** | **0** |
| `retreat` | 待機高さ (0.30 m) へ退避 | 0.105 | 0.105 | 0.105 |
| `relax` | ロープが完全に落ち着くのを待つ (**報酬はこの直後に測る**) | 0.65 | **0.5** | **0.5** |
| **合計** | | **2.855 s / 686 step** | **2.4375 s / 585 step** | **3.1625 s / 759 step** |

尺の変更の経緯:

* **`DEFAULT`**: 移動系フェーズ (`approach` / `descend` / `lift` / `carry` / `lower` / `clear` / `retreat`)
  を 0.35 倍に詰めた値。1 手 5.0 s → 2.86 s にしても、把持成功率・交差生成率はほぼ変わらなかった
  (2026-09-12 の実測。表は `PhaseDurations` の docstring)。接触・物理待ち系
  (`close` / `hold` / `release` / `settle_out` / `relax`) は「離すと付いてくる」問題に直結するので据え置いた。
* **`SHORT`**: `approach` の非物理的な速度指令が発散源だったので、補間をやめて位置の書き換えにした
  (理由と安全性は `SHORT_DURATIONS` の docstring と [v2 README](../versions/v2/README.md) §2)。
* **`KNOT`**: 28 リンクの `Knot` を速く運ぶと発散するので、移動系を延ばした (ゲート 1、
  [`docs/experiments/2026-09-27_gate1.md`](../../../../../docs/experiments/2026-09-27_gate1.md))。定義は `versions/v3/env_cfg.py` で、v4 以降は継承して使う。

## 3. 行動空間 ([`mdp/actions.py`](mdp/actions.py) の `PickPlaceAction`)

6 次元・連続 `[-1, 1]` の行動を「1 手の指示」として解釈する。

| ch | 名前 | 意味 | 変換後 |
|---|---|---|---|
| 0 | `grasp_link` | 掴むリンク | 離散 `{2, ..., N-3}` (両端 2 個は接触面積不足で除外) |
| 1 | `delta_link` | 置き先までのリンク差 | 離散 `{-8, ..., +8}`。`target = clip(grasp + Δ, 0, N-1)` |
| 2 | `drop_x_rel` | 置き先の x 変位 | target リンクの**現在位置**からの相対値。±0.15 L (v3 は ±0.30 L。L = ロープ全長) |
| 3 | `drop_y_rel` | 同 y | 同上 |
| 4 | `drop_z_rel` | 置く高さ | 把持高さからの上乗せ、0〜3×ロープ直径 (指先が床上 2〜62 mm)。v5 は離すときの指先の高さ 50〜100 mm (`GraspLimits.release_tip_z`) |
| 5 | `end_yaw` | 置き終わりの手首角度 | ±π |

範囲は `GraspLimits` (`mdp/phases.py`) が持ち、バージョン側は `KnotTyingEnvCfg.grasp_limits` で変える。

* **絶対座標ではなくリンク相対にした理由**: pick&place 粒度では、離した瞬間にロープが物理で動く。
  そのため、次の手を決める時点で前回のリンク位置はもう当てにならない。キネマティックハンドには
  可達領域の制約も無いので、絶対座標の利点が無い。人間が書いた台本も「渡し先をリンクの現在位置から
  実行時に決める」形でしか組めなかった (`docs/history/next_action.md` §4.1。判断は [`docs/decisions/`](../../../../../docs/decisions/README.md) の行動空間)。
* **離散チャネルを連続値で受け取る理由**: RSL-RL の PPO は連続ガウス方策しか持たない。
  `grasp_link` / `delta_link` は `PickPlaceAction` の側で丸めて離散化する。
* **`process_actions` と `apply_actions`**:
  * `process_actions(actions)` は env ステップの最初に 1 回だけ呼ばれる。現在のロープ姿勢から
    掴む点と置く先のワールド座標を確定し、フェーズ列を `approach` から始める。ロープが NaN の環境では
    手を実行しない。
  * `apply_actions()` は毎物理ステップ呼ばれる。フェーズ内を smoothstep (`s = u²(3-2u)`) で補間し、
    ハンドの位置・yaw・指の開度を書き込む。速度も明示的に書き込む (`write_root_velocity_to_sim`)。
    位置だけ書くとテレポート扱いになって接触摩擦が働かず、掴んだロープが滑り落ちる。
* **把持は摩擦だけで成立させる**: 指には常に「全閉」を目標として指令し続ける。ロープの太さぶん
  閉じ切れないので、PD 制御の定常偏差がそのまま把持力になる。固定ジョイントで拘束しないので、
  どのリンクでも掴める。指の剛性はロープごとに `1500 * link_mass / capsule_radius` でスケールする
  ([`rope_task_cfg.py`](rope_task_cfg.py))。

## 4. 観測の共通事項 ([`mdp/observations.py`](mdp/observations.py))

* **p-data 文字列は観測に使わない。** 記号層は 4096 環境で 94 ms かかるのに対し、テンソル層
  (交差数・writhe・交差の表) は 1.6 ms で済む。p-data 文字列はログと Play の表示だけで使う。
* **ノイズは注入しない。** 位相量は離散なので、「1.2 個の交差」のような無意味な値になってしまう。
* **幾何と位相の両方を渡す。** 位相だけでは「次にどこを掴めばよいか」が分からない。

### `RopeTopologyExtractor` の共有と reset 時の罠

交差検出は env ステップにつき 1 回で足りる。`get_extractor(env)` が抽出器を env にキャッシュし、
観測・報酬・終了判定がそれぞれ呼んでも再計算しない。

`ManagerBasedRLEnv.step()` は `terminations -> rewards -> _reset_idx -> observations` の順に処理する。
reset された環境ではロープが初期姿勢に戻っているのに、キャッシュは reset 前の形状のまま残る。
そこで `PickPlaceAction.reset()` が `observations.invalidate_extractor(env)` を呼び、次の呼び出しで
必ず抽出し直させている。

## 5. 報酬の共通事項

* **成功判定は必ずハンドが退避した後に測る。** そうしないと「ハンドで吊って交差を見せかける」方策を
  学習してしまう。フェーズ列の最後は `clear` → `retreat` → `relax` なので、報酬を計算する時点では
  ハンドはロープから離れている。成功系の項は「ロープの最高点が直径の数倍以下」という高さ条件も
  二重に課している。
* **重みは `step_dt` で割られる。** Isaac Lab の `RewardManager.compute()` は各項の重みに `dt`
  (env ステップの秒数) を掛ける。この環境の 1 env ステップは 1 手 (2.4〜3.2 s) と長いので、
  `RewardsCfg` には「1 手あたりの値」を書く。`KnotTyingEnvCfg._rescale_reward_weights()` が
  `step_dt` で割ることで、書いた値がそのまま実効値になる。
* **重み 0 の項は呼ばれない。** `RewardManager` は重み 0 の項を計算しない。ログのためだけの項
  (`topology_stats` / `record_archive`) は、重み 1 で 0 を返す形にしてある。

## 6. 終了判定 ([`mdp/terminations.py`](mdp/terminations.py))

| 項 | 内容 |
|---|---|
| `time_out` | `actions_per_episode` 手を打ち切ったら終了 |
| `diverged` (`rope_diverged`) | ロープ座標が NaN、または高さが 5 m を超えた。v2-R2 以降は、env 原点からの水平距離 3.0 m とリンク速度 20 m/s も見る |

発散した環境はその場で reset され、報酬の `failure` (`action_failed`) が罰を与える。

版ごとに足す・置き換える終了項:

| 版 | 項 | 内容 |
|---|---|---|
| v3 | `knot_tied` | 成功 (固結び) で終了 (`time_out=False` なので PPO は bootstrap しない) |
| v4〜v7、rope study | `stage_up` / `stage_down` (`mdp.stage_changed`) | `knot_tied` を外し、段が上がった・下がったら終了 (1 遷移 = 1 エピソード)。Play では chain モード (固結びでだけ `stage_up`)。`stage_up` は `time_out` より後ろに置く (`versions/v4/env_cfg.py:124-140`) |

## 7. 起動時イベント ([`mdp/events.py`](mdp/events.py))

| イベント | タイミング | 内容 |
|---|---|---|
| `grasp_material` | `startup` | 指プリムに高摩擦材 (μ=4.0, `friction_combine_mode="max"`) を bind する。`bind_physics_material` は正規表現を受け付けないので、`find_matching_prim_paths` で env ごとに展開してから 1 つずつ bind する。取りこぼすと、**例外は出ないままロープが指をすり抜ける**。bind 数が 0 なら例外を投げる |
| `link_order` | `startup` | ロープの body の並びが定義順と一致しているかを、隣接リンク距離の比 (max/median ≈ 1.0) で検算する |
| `reset_all` | `reset` | シーンを初期状態に戻す。ロープは高さ 0.06 m から落として落ち着かせる |

## 8. 実行時の処理フロー

```
env.reset()
  └─ ハンドを待機姿勢へ、ロープを初期姿勢へ
     PickPlaceAction.reset() -> extractor キャッシュを無効化

for action_step in range(actions_per_episode):
    action = policy(obs)                            # 6 次元 [-1, 1]
    env.step(action)
        process_actions(action)                     # 掴む点・置き先を確定、フェーズ列を先頭へ
        for _ in range(decimation):                 # 1 手ぶんの物理ステップ
            apply_actions()                         # フェーズ内の smoothstep 補間でハンドを駆動
            sim.step()
            scene.update()                          # rope.data 更新
        terminations.compute()
        rewards.compute()
        (reset if terminated)
        observations.compute()
```

## 9. `uv run play` でロボットが止まって見える理由

学習用の設定では `render_interval = decimation` (1 手に 1 回だけ描画) になっている。
`apply_actions()` は毎物理ステップ滑らかにハンドを動かしているが、その途中経過が画面に反映されない。
そのため、1 手の終わりに最終姿勢へワープしたように見える。処理落ちではない。

`-Play-` 付きのタスクでは、[`rope_task_cfg.py`](rope_task_cfg.py) が `render_interval` を
8 物理ステップ (30 Hz 相当) に上書きする。変わるのは描画のタイミングだけで、物理と学習には影響しない。
起動時に出る `The render interval (...) is smaller than the decimation (...)` の警告は意図どおりなので、
無視してよい。

`--real-time` を付けると、`env.step_dt` (1 手の実時間) に合わせて速度を落とすので、1 手ずつ目で追える:

```bash
uv run play --task Isaac-Twisted-RL-Twist-Play-v0 --num_envs 1 --real-time
```
