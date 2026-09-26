# v0: 交差を 1 つ作る (基準バージョン)

```bash
uv run train --task Isaac-Twisted-RL-Twist-v0 --headless
```

を実行したときに何が起きているかを、MDP (Markov Decision Process) の構成要素ごとに説明する。

## このバージョンの位置づけ

**紐結びの最小版であり、以降のバージョンの基準点。**
ロープの一部を持ち上げ、ロープ本体の上をまたがせて置くと「交差」ができる。
これを位相不変量として検出し、報酬に変える。

このバージョンは `main` ブランチのコミット `1f5693f` 時点の実装と同一
(`experiment_name` を `knot_tying_v0` に分離した点を除く)。
**改良はここを書き換えるのではなく、v1 以降を足すこと**
(規約は [`../../README.md`](../../README.md))。

次のバージョン [`v1`](../v1/README.md) は、この課題を固結び
(overhand knot) へ拡張したもの。

## 1. タスクの目標

**ロープを掴んで、交差を 1 つ作って、手を離しても残す。**

- ロープ: D6 ジョイント連鎖で作った 20 リンクの紐 (`twist` プリセット)。
  柔らかい曲げ剛性 + 捩り自由度あり ([`rope_specs.py`](../../common/rope_model/rope_specs.py))。
- ロボット: **アームの無いキネマティックハンド**
  ([`parallel_gripper/`](../../common/parallel_gripper/))。IK 失敗・可達領域外・自己衝突・
  特異点が存在しないため、学習が「腕を動かす」ことに割かれず「紐を操作する」
  ことだけに集中できる。root 姿勢を直接書き込んで動かす。

## 2. 構成

シーン・物理・行動・終了判定は [`common/`](../../common/) と共有。
このバージョンが定義するのは **観測・報酬・エピソード長・PPO 設定**だけ。

```
versions/v0/
├── env_cfg.py                観測・報酬・actions_per_episode (5 手)
├── agents/rsl_rl_ppo_cfg.py  PPO ハイパーパラメータ
└── README.md                 このファイル
```

## 3. この環境で一番特徴的な設計: 「1 手 = 1 RL ステップ」

`lift` や `reach` のような一般的な manager-based タスクは 1 env ステップが
「関節を少し動かす」程度の粒度だが、この環境は **1 env ステップ = pick&place の
1 手 (掴む→運ぶ→置く→離す) 全部** という Options Framework 的な粒度になっている
([`next_action.md`](../../next_action.md) 4.0 の決定)。

これを実現しているのは Isaac Lab の `decimation` の使い方である。

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

[`common/env_cfg_base.py`](../../common/env_cfg_base.py) の `KnotTyingEnvCfg.__post_init__` が

```python
self.sim.dt = 1.0 / 240.0
self.decimation = self.durations.decimation(self.sim.dt)   # 1 手ぶんの物理ステップ数
self.episode_length_s = self.actions_per_episode * self.decimation * self.sim.dt
```

として `decimation` を「1 手を完了するのに必要な物理ステップ数」に設定する。
`durations` ([`mdp/phases.py`](../../common/mdp/phases.py) の `PhaseDurations`、既定合計 2.855 秒、
240 Hz で 686 ステップ) は 12 個のフェーズ (下表) の尺を持ち、これを変えると
`decimation` が自動で追随する。

| フェーズ | 尺 [s] | 内容 |
|---|---|---|
| `approach` | 0.175 | 掴む点の真上へ移動 |
| `descend` | 0.14 | 下降 |
| `close` | 0.3 | 指を握り込む |
| `hold` | 0.2 | 接触の安定待ち |
| `lift` | 0.14 | 持ち上げ |
| `carry` | 0.35 | 置く場所まで運搬 (yaw もここで回す) |
| `lower` | 0.14 | 置く高さまで下降 |
| `release` | 0.25 | 指を開く |
| `settle_out` | 0.3 | 離した後ロープが落ち着くのを待つ (物理の待ち) |
| `clear` | 0.105 | 真上に上げる前に横へ抜ける (指に残って持ち上がるのを防ぐ) |
| `retreat` | 0.105 | 待機高さへ退避 |
| `relax` | 0.65 | ロープが完全に落ち着くのを待つ (**報酬はこの直後に測る**) |

`approach` / `descend` / `lift` / `carry` / `lower` / `clear` / `retreat` の
移動系フェーズは、ロープと接触しない区間なので速度を上げても把持成功率・
交差生成率にほぼ影響しないことが実測で分かり (scale=0.35、64 環境 x 20 手の
ベンチマーク)、元の尺 (括弧内は旧値の目安、詳細は
[`mdp/phases.py`](../../common/mdp/phases.py) の `PhaseDurations` docstring) から約 65%
短縮してある。`close` / `hold` / `release` / `settle_out` / `relax` の
接触・物理待ち系フェーズは「離すと付いてくる」問題に直結するため未検証・
据え置き。

1 エピソードは `actions_per_episode = 5` (`env_cfg.py`) 手 (= 5 env ステップ、3430 物理ステップ)。

物理周波数は 120 Hz ではなく **240 Hz** を使う。120 Hz では把持中の接触で
ロープ速度が異常値 (60〜175 m/s) まで跳ね上がり NaN 化する発散率が 11.2% に
達したが、240 Hz では 1.2% まで下がる実測結果による ([`next_action.md`](../../next_action.md) 4.5)。

## 4. 行動空間 (`ActionsCfg.pick_place`, [`mdp/actions.py`](../../common/mdp/actions.py))

`PickPlaceAction` (`ActionTerm`) が 6 次元・連続 `[-1, 1]` のアクションを
「1 手の指示」として解釈する。

| ch | 名前 | 意味 | 変換後 |
|---|---|---|---|
| 0 | `grasp_link` | 掴むリンク | 離散 `{2, ..., N-3}` (両端 2 個は接触面積不足で除外) |
| 1 | `delta_link` | 置き先までのリンク差 | 離散 `{-8, ..., +8}`。`target = clip(grasp + Δ, 0, N-1)` |
| 2 | `drop_x_rel` | 置き先の x 変位 | target リンクの**現在位置**からの相対値、±0.15×ロープ全長 |
| 3 | `drop_y_rel` | 同 y | 同上 |
| 4 | `drop_z_rel` | 置く高さ | 把持高さからの上乗せ、0〜3×ロープ直径 |
| 5 | `end_yaw` | 置き終わりの手首角度 | ±π |

### なぜ絶対座標ではなくロープ (リンク) 相対なのか

pick&place 粒度では「離した瞬間にロープが物理で動く」ため、次の手を決める
時点で前回のリンク位置はもう当てにならない。またキネマティックハンドには
可達領域の制約が無いため、絶対座標を使う利点そのものが無い。人間が設計した
手本の手順 (`topology_debug_demo.py` の台本) も「渡し先をリンクの現在位置から
実行時に決める」形でしか組めなかった、という経験的根拠もある
([`next_action.md`](../../next_action.md) 4.1 に詳細)。

### 離散チャネルを連続値で表す理由

学習器 (RSL-RL の PPO) は連続ガウス方策しか持たないため、`grasp_link` /
`delta_link` も一旦 `[-1, 1]` の連続値として受け取り、`PickPlaceAction`
側で丸めて離散化する。方策から見れば全チャネル連続、環境に入る値は一部離散。

### `process_actions` と `apply_actions` の役割分担

- `process_actions(actions)` — env ステップの最初に 1 回。現在のロープ姿勢
  (`link_pos`) から掴む点・置く先のワールド座標を確定し、フェーズ列を
  `approach` から開始する。ロープが NaN 化している環境は手を実行せず、
  安全な位置 (現在のハンド位置) に潰して物理破綻を防ぐ。
- `apply_actions()` — 毎物理ステップ 1 回。フェーズ内の経過を smoothstep
  補間 (`s = u²(3-2u)`) してハンドの目標位置・yaw・指の開度を書き込む。
  等速補間ではなく smoothstep にしているのは、始点・終点での速度不連続が
  ロープを指から滑らせるため。速度も明示的に書き込む
  (`write_root_velocity_to_sim`) — 位置だけ書くとテレポート扱いになり
  接触摩擦が働かず、掴んだロープが滑り落ちる。

### 把持は固定ジョイントではなく摩擦のみ

指を常に「全閉」目標で指令し続けるが、ロープの太さ分だけ閉じ切れない。
その PD 制御の定常偏差がそのまま把持力になる。固定ジョイントで拘束しない
ため「どのリンクでも掴める」性質が得られ、これが `grasp_link` を自由に
選ばせる設計の前提になっている。指の目標剛性はロープごとにスケールされる
(`rope_task_cfg.py`: `stiffness = 1500 * link_mass / capsule_radius`)。

## 5. 観測空間 (`ObservationsCfg.policy`, [`mdp/observations.py`](../../common/mdp/observations.py))

方策への観測はテンソル層の数値のみで構成し、**p-data 文字列 (`'1O2+_2U1+'`
のような記号表現) は一切使わない**。理由は速度差: 記号層の生成は 4096 環境で
94 ms かかるのに対し、テンソル層 (交差数・writhe など) は 1.6 ms
(`topology/state_2_topology.py` の実測値)。毎ステップ計算するにはテンソル層
しか使えない。

| 項 | 内容 |
|---|---|
| `rope_link_pos` | 全リンク中心の env 原点相対座標 (ロープの「形」) |
| `crossings` | 現在の交差数 |
| `writhe` | 交差符号の総和 (位相の粗い指標) |
| `first_crossing` | 先頭交差の特徴量 `(over, sign, point_i/N, point_j/N)` |
| `hand_pos` | ハンド root の env 原点相対位置 |
| `actions` | 直前に取った行動 |

幾何 (`rope_link_pos`) と位相 (`crossings` 等) の両方を渡すのは、位相だけでは
「次にどこを掴めば交差を作れるか」が分からないため。ノイズは注入しない
(位相量は離散なので、ノイズを乗せると「1.2 個の交差」のような無意味な値に
なってしまう)。

### `RopeTopologyExtractor` の共有と、reset 時の罠

交差検出は env につき 1 回で足りるので `get_extractor(env)` が
`RopeTopologyExtractor` を env オブジェクトにキャッシュし、同じ env ステップ内
での再計算を防ぐ (観測・報酬・終了判定がそれぞれ独立に呼んでも 3 回走らない)。

`ManagerBasedRLEnv.step()` は `terminations -> rewards -> _reset_idx ->
observations` の順で処理するため、reset された環境はロープが初期姿勢に
戻っているのに観測計算時点でのキャッシュが reset 前の形状のままになる
落とし穴がある。これを避けるため `PickPlaceAction.reset()` が
`observations.invalidate_extractor(env)` を呼び、次の `get_extractor()` で
必ず抽出し直させている。

## 6. 報酬 ([`mdp/rewards.py`](../../common/mdp/rewards.py), 重みは `RewardsCfg`)

| 項 | 既定重み (1手あたり) | 内容 |
|---|---|---|
| `crossing_progress` | +0.3 | 交差数の増分 (密報酬)。手を打つ前後の差分を毎手評価。交差が減れば負 |
| `crossing_achieved` | +1.0 | 交差数 ≥1 **かつ** ロープが床付近 (吊られていない) なら成功 |
| `grasp_distance_penalty` | -0.05 | `close` 終了時の指先とロープの距離 (整形報酬) |
| `action_failed` | -0.2 | ロープ座標が NaN で手を実行できなかった / 発散した |
| `held_up_penalty` | -0.1 | 手の終わりにロープが浮いたまま (吊られている兆候) |

p-data は離散量のため一致/不一致だけでは勾配が出ない。そこで密報酬
(`crossing_progress`) と疎報酬 (`crossing_achieved`) を併用する。

**最重要: 成功判定は必ずハンドが退避した後に測る。** そうしないと
「ハンドで吊って交差を見せかける」方策を学習してしまう。フェーズ列の最後は
`clear`(横へ抜ける) → `retreat`(待機高さへ) → `relax`(静止待ち) なので、
報酬計算時点でハンドは既にロープから離れている。加えて `crossing_achieved` /
`held_up_penalty` は「ロープの最高点が直径の数倍以下」という高さ条件でも
二重に保険をかけている。

### 重みのスケーリングに関する注意

Isaac Lab の `RewardManager.compute()` は各項の重みに `dt` (env ステップの
秒数) を掛ける設計だが、この環境は 1 env ステップ = 1 手 = 約 2.9 秒と極端に長い。
`RewardsCfg` には「1 手あたりの値」(上表) をそのまま書き、
`KnotTyingEnvCfg._rescale_reward_weights()` が `__post_init__` で
`step_dt` (= `decimation * sim.dt`) で割ることで実効値を辻褄合わせしている。

## 7. 終了判定 ([`mdp/terminations.py`](../../common/mdp/terminations.py))

| 項 | 内容 |
|---|---|
| `time_out` | `episode_length_s` 経過 (= 5 手打ち切ったら終了) |
| `rope_diverged` | ロープ座標が NaN、または高さが 5 m を超えた (吹き飛び) |

発散した環境はその場で reset に回され、`rewards.action_failed` が罰を与える。
既定では「交差 1 つに到達したら即終了」はしていない (`terminations.py` の
`topology_reached` は用意されているが未使用) — 交差を作るだけでなく
**壊さずに保つ**ことも学習させたいため、時間切れまで継続する方針。

## 8. 起動時イベント ([`mdp/events.py`](../../common/mdp/events.py))

| イベント | タイミング | 内容 |
|---|---|---|
| `bind_grasp_material` | `startup` | 指プリムに高摩擦材 (μ=4.0, `friction_combine_mode="max"`) を bind。マルチ env では `find_matching_prim_paths` で `/World/envs/env_*/Gripper/left_finger` を展開してから 1 つずつ bind する必要がある (正規表現非対応のため)。取りこぼすと**例外無しでロープが指をすり抜ける** |
| `check_rope_link_order` | `startup` | ロープの body 配列順序が定義順と一致しているかを隣接リンク距離の比 (max/median ≈ 1.0) で検算する保険 |
| `reset_scene_to_default` | `reset` | シーンを初期状態に戻す (標準項) |

## 9. PPO 設定 ([`agents/rsl_rl_ppo_cfg.py`](agents/rsl_rl_ppo_cfg.py))

```python
num_steps_per_env = 10      # 2 エピソード分 (1 エピソード5ステップ)
max_iterations = 2000
gamma = 0.95                 # 割引率。地平が5ステップしかないので低め
lam = 0.95
entropy_coef = 0.01           # 離散選択を連続量子化で表すため探索崩壊を防ぐ
learning_rate = 1.0e-4
clip_param = 0.2
actor_hidden_dims = [256, 128, 64]
critic_hidden_dims = [256, 128, 64]
```

`lift` / `reach` 系タスクとの最大の違いは **エピソードが極端に短い
(5 ステップ) 一方、1 ステップの実時間コストが非常に重い (686 物理ステップ)**
こと。`num_steps_per_env` を小さめ (10) にして 1 イテレーションの物理ステップ数を
抑え、`gamma` も短い地平に合わせて低めにしている。

## 10. 実行時の処理フロー (まとめ)

```
env.reset()
  └─ ハンドを待機姿勢へ、ロープを初期姿勢へ (ROPE_INIT_Z=0.06 から落として settle)
     PickPlaceAction.reset() -> extractor キャッシュを無効化

for action_step in range(5):                      # 1 エピソード = 5 手
    action = policy(obs)                            # 6 次元 [-1, 1]
    env.step(action)
        process_actions(action)                      # 掴む点・置き先を確定、フェーズ列を先頭へ
        for _ in range(decimation):                  # 1 手 = 686 物理ステップ (240Hz*2.855s)
            apply_actions()                           # フェーズ内 smoothstep 補間でハンド駆動
            sim.step()
            scene.update()                            # rope.data 更新
        terminations.compute()                        # NaN/吹き飛びチェック
        rewards.compute()                             # 交差の増減・成功判定など
        (reset if terminated)
        observations.compute()                        # 次の行動のための観測
```

1 手 = 686 物理ステップ x 5 手 = 1 エピソード 3430 物理ステップ。
`num_envs = 1024` の並列環境で PPO のロールアウトを集め、`max_iterations = 2000`
まで学習を回す。

## 11. `uv run play` でロボットが止まって見える理由

```
uv run play --task Isaac-Twisted-RL-Twist-Play-v0 --num_envs 1
```

で GUI を見ると、ロボットが動かず固まって見えることがある。これは
**処理が重すぎて詰まっているのではなく、単に描画されていないだけ**。

`KnotTyingEnvCfg.__post_init__` ([`common/env_cfg_base.py`](../../common/env_cfg_base.py))
は `self.sim.render_interval = self.decimation` としている。`decimation` は
1 手ぶんの物理ステップ数 (240 Hz x 約 2.855 秒 = 686 ステップ) なので、
`ManagerBasedEnv.step()` の描画条件

```python
if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
    self.sim.render()
```

(isaaclab 側、`manager_based_env.py`) は **1 手 (実時間約 2.9 秒相当) につき
1 回しか真にならない**。`apply_actions()` 自体は毎物理ステップ (smoothstep
補間で) ハンドを滑らかに動かしているが、その途中経過が画面に一切
反映されず、1 手が終わった瞬間に最終姿勢へワープしたように見える。
これが「ロボットが止まっている」ように見える正体で、GPU 負荷や
処理落ちが原因ではない。

(学習中の GUI 表示は「1 手ごとに 1 回描画できれば十分」で、逆にこの
`render_interval` が小さいと毎物理ステップ描画が挟まり学習が大きく
遅くなるため、学習用の `TwistEnvCfg` 側はこのままにしてある。)

### 対策: Play 系クラスだけ描画間隔を短縮

[`rope_task_cfg.py`](../../common/rope_task_cfg.py) の `_apply_play_overrides()` が
`*PlayEnvCfg` (`TwistPlayEnvCfg` など) に対してのみ

```python
cfg.sim.render_interval = _PLAY_RENDER_STEPS  # 8 物理ステップ = 30 Hz 相当
```

を上書きする。学習用の `decimation` (= 686) やシミュレーション自体の
挙動には影響せず、**描画のタイミングだけ**を細かくするので、`-Play-v0`
タスクでは smoothstep 補間によるハンドの滑らかな動きがそのまま見える。
学習速度 (`train`) には一切影響しない。

なお `render_interval < decimation` にすると isaaclab が起動時に
`The render interval (...) is smaller than the decimation (...)` という
警告を出すが、これは Play 用に意図した設定であり無視してよい。

### さらに実時間に近づけたい場合

`play` はデフォルトでは実時間同期をせず、GPU が空いていれば物理・推論を
全力で回す。`--real-time` オプションを付けると `env.step_dt` (= 1 手の
実時間約 2.9 秒) に合わせて `time.sleep` で速度を落とすので、人間の目で
1 手ずつ追いたい場合はこちらも有効:

```
uv run play --task Isaac-Twisted-RL-Twist-Play-v0 --num_envs 1 --real-time
```
