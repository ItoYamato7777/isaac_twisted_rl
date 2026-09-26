# v2: 固結び (overhand knot) を作る — v1 の学習失敗を受けた作り直し

```bash
uv run train --task Isaac-Twisted-RL-Twist-v2 --headless        # MLP (ベースライン)
uv run train --task Isaac-Twisted-RL-Twist-v2-RNN --headless    # GRU
uv run train --task Isaac-Twisted-RL-Twist-v2-LSTM --headless   # LSTM
```

目標は [`v1`](../v1/README.md) と同じ **固結び (overhand knot / 三葉結び)**。
違うのは解き方で、v1 の学習が失敗した原因を実測から特定して潰している。

## 1. v1 はなぜ失敗したのか (TensorBoard の実測)

v1 の学習 (`logs/rsl_rl/knot_tying_v1/2026-09-15_21-49-25`) は
**4000 iteration の設定に対して 157 で打ち切られ、報酬は増えるどころか
下降を続けていた**。v0 (`knot_tying_v0/2026-09-12_13-51-53`、970 iteration)
と並べると違いは明確。

| 指標 | v0 (成功) | v1 (失敗) |
|---|---|---|
| `Train/mean_reward` | 1.73 → **5.45** | −0.009 → **−0.058 (下降中)** |
| `Episode_Reward/success` | 0.090 → 0.345 (1 手あたり 98.5% 発火) | 0.0001 → **厳密に 0.0** |
| `Episode_Termination/diverged` | 0.139 → **0.0006** | 0.250 → **0.480** |
| `action_failed` 率 | 3.3% → **0.05%** | 3.6% → **5.8%** |
| `Policy/mean_noise_std` | 0.998 → **0.610 (収束)** | 1.003 → **1.628 (拡大)** |
| `Loss/entropy` | 8.51 → **2.08** | 8.52 → **11.39 (上昇)** |
| 適応学習率 | 下限 1e-5 に張り付き | 5.1e-4 → **1.32e-3 (上昇)** |
| `Loss/value_function` | 0.17〜2.6 | **~1099 までスパイク** |

終端の報酬構成 (1 エピソードあたりの寄与):

* v1: `failure` **−0.0924** が最大絶対値の項で、`shaping` (+0.0496) を
  **1.86 倍上回る**。`success` は +0.0013。
* v0: `success` **+4.92 (91%)**、`progress` +0.49 (9%)、罰則は合計 0.05% 未満。

### 診断

1. **疎報酬が一度も発火しなかった。** `overhand_achieved` は 1 手あたり
   0.00005 回。PPO が受け取る勾配が実質「NaN を避けろ」だけになった。
2. **その唯一の勾配にも失敗した。** 発散終了が 25% → 48%。終盤は
   **エピソードの約半分がロープの発散で終了**している (ランダム方策の
   実測値 1.2% に対して 40 倍)。
3. **エントロピーボーナスが唯一プラスを払う項になった。** タスク報酬が
   純負なので `entropy_coef=0.02` の下では「最大限ランダムに振る舞う」のが
   最適になり、`noise_std` と `entropy` が揃って上昇した (v0 は両方下降)。
4. **critic がターゲットに追いつけていない。** value loss が v0 の 400 倍以上、
   適応学習率が上げられ続けている。

つまり v0 → v1 の変更のうち **「目標を一気に固結びまで上げ、到達可能な
中間目標を一つも置かなかった」ことが主因**。報酬ラダー
(`knot_class_shaping`)・観測拡張・ネットワーク拡大は、それ自体が
悪かったという証拠は無い。

## 2. v1 から何を・なぜ変えたか

**変更は 3 系統に絞った。** V3 以降も続くので、うまく行かなかったときに
どの変更が原因か切り分けられる範囲に留める。

| | v1 | v2 | なぜ変えたか |
|---|---|---|---|
| **疎報酬** | `overhand_achieved` のみ (3.0) | **3 段ラダー**: `success_c1` 0.5 / `success_c2` 1.5 / `success` 3.0 | 診断 1。各段に「実際に発火する」実測根拠がある (下表) |
| **`failure`** | −0.2 | **−0.1** | 診断 3。3 段が払うようになるので再校正。0 にはしない |
| **フェーズの尺** | `DEFAULT_DURATIONS` (686 step / 2.855 s) | **`SHORT_DURATIONS`** (585 step / 2.4375 s) | 診断 2。`approach` の速度指令を消す + 実時間 14.7% 短縮 |
| **エピソード長** | 8 手 | **6 手** | 位相的下限 (3 遷移 x 2 手) そのもの。信用割り当てを濃く、実時間を 36% 短縮 |
| **`entropy_coef`** | 0.02 | **0.01** (v0 の値) | 診断 3 |
| **`gamma`** | 0.98 | **0.95** (v0 の値) | 地平が 6 手に縮んだ + 診断 4 |
| **観測の正規化** | `empirical_normalization=True` | **per-policy で False** (v0 相当) | 診断 4 と同時発生。保守的に v0 側へ戻す。単変数の検証は v3 |
| **`num_steps_per_env`** | 16 | **12** | エピソード長 (6) の整数倍でなければならない。16 は 6 の倍数ではない |
| **`max_iterations`** | 4000 | **2000** | 4000 は約 14 日。実時間を本当に縮めるのは `num_envs` の測定 |
| **`obs_groups`** | 未設定 (deprecated fallback) | **明示** | 将来 critic に特権観測を足すときの変更点を明示 |
| **ネットワーク変種** | MLP のみ | **MLP / GRU / LSTM の 3 タスク** | 時系列モデルとの比較 (§4) |
| **観測** | 105 次元 | **据え置き** | v1 の失敗原因は観測ではない。v3 の暖機開始のため形状を固定する |
| **隠れ層** | [512,256,128] | **据え置き** | 同上 |

**変えていないもの**: シーン、物理 (240 Hz)、アクション空間 (6 次元)、
終了判定、イベント、`GraspLimits`、密報酬 `knot_class_shaping`。

### 3 段の疎報酬ラダー — なぜこの 3 つか

| 段 | 項 | 重み | 発火の実測根拠 |
|---|---|---|---|
| 交差 1 つ | `success_c1` (`crossing_achieved`) | +0.5 | **v0 が 1 手あたり 98.5% 発火で学習成功** |
| 2 交差の絡み | `success_c2` (`interleaved2_achieved`) | +1.5 | ランダム方策で 1.4% 到達 ([v1 README](../v1/README.md) §12) |
| 固結び | `success` (`overhand_achieved`) | +3.0 | ランダム方策で 512 回に 1 回 (単独では足場にならない) |

最下段は **新規コードが要らない**。v0 が使っていた `crossing_achieved` を
そのまま再利用している (`common/mdp/` を追加のみで削除・改名しない方針の恩恵)。

最下段を 1.0 ではなく **0.5** にしたのは、v0 でこの項が 1 手あたり 98.5% =
ほぼ飽和していたため。飽和した項を強くすると「交差 1 つを作って維持する
だけ」が局所最適になり、上の段へ登る動機が消える。

中段 `interleaved2_achieved` は v2 で追加した項
([`common/mdp/rewards.py`](../../common/mdp/rewards.py))。
`overhand_achieved` と同じ構造で判定だけ `is_interleaved_two`
(= `has_knot_subsequence(k=2)`) に替えたもの。この段を選んだのは
**「離しても輪が残るか」を問う最初の段**だから — 最下段の交差 1 つは飽和し、
最上段は疎すぎる。固結びに必要な「作った絡みが解けない」技能を初めて測る。

なお密報酬のポテンシャル整形はそのまま残している。疎報酬を段ごとに重ねても
「段に居座る」報酬ハッキングが起きないのは、各段が**その状態に到達している
あいだ毎手払われる**のではなく `settled` (ロープが床に落ちている) を
条件にしており、かつ上の段の重みが十分大きいため。ここは学習ログで
`success_c1` だけが伸びて `success_c2` が伸びない状態が続かないか監視する。

### `approach = 0.0` が「位置座標の変更による移動」になる仕組み

要求は「approach の移動を削除 (上の位置にグリッパを位置座標を変更することで
移動させる)」。これは **`actions.py` を変更せずに実現できる**。

`PhaseDurations.step_counts` は `max(int(round(d / sim_dt)), 1)` なので
尺 0 のフェーズは **1 物理ステップ**になる。`apply_actions` を 1 ステップ
だけ辿ると `dur = 1` → `u = 1.0` → smoothstep `s = 1·1·(3−2·1) = 1.0` なので、
初回ステップで `pos = _seg_start + (_seg_end − _seg_start)·1.0 = _seg_end`
となり、掴む点の真上 (`HOME_PALM_Z` = 0.30 m) へ **補間せず直接書き込まれる**。

さらに速度指令は `ds = 6·u·(1−u)/seg_time` で `u = 1.0` なので **厳密に 0**。
つまり「移動させる」のではなく「位置を書き換える」挙動そのものになる。

**これが診断 2 への対策の本命。** `approach` は前の手の終了位置から次の
掴む点まで最も長い距離を動くフェーズで、2026-09-12 に
「平均 78 m/s、最大 131 m/s」の非物理的な指令速度が実測されたのと同じ経路
にある ([`actions.py`](../../common/mdp/actions.py) の `reset()` のコメント)。
位置の書き換えにすれば、その速度指令自体が消える。

**安全性の根拠はテレポートの高度**。0.30 m はロープ直径の 15 倍なので、
瞬間移動してもロープとの接触は発生しない。**同じことを把持高さで
やってはいけない。**

`clear = 0.0` も同様に安全。「離した後に指に残って一緒に持ち上がる」対策で
入れたフェーズだが、その後 `OPEN_RATIO` が 1.0 (全開) に上がっており、
全開なら横抜けなしでも指から外れる。`settle_out` (0.3 s) はそのまま残る。

`relax` は 0.65 → 0.5 に留める (0 にはしない)。報酬を測る直前の静定待ちで、
`settle_out` と合わせて **離してから 0.8 s** は確保する。

## 3. 報酬・観測・行動の一覧

観測は v1 と同一 (105 次元)。行動も同一 (6 次元)。詳細は
[v1 README](../v1/README.md) §3-4 を参照。

報酬 (重みは「1 手あたり」。`step_dt` = 2.4375 で割られて実効値になる):

| 項 | 関数 | 重み | 実効値 |
|---|---|---|---|
| `shaping` | `knot_class_shaping` (γ=0.95, Φ=(0,.35,.70,1.0)) | +1.0 | 0.4103 |
| `success_c1` | `crossing_achieved` (min_crossings=1) | +0.5 | 0.2051 |
| `success_c2` | `interleaved2_achieved` | +1.5 | 0.6154 |
| `success` | `overhand_achieved` | +3.0 | 1.2308 |
| `offladder` | `offladder_penalty` | −0.05 | −0.0205 |
| `grasp_shaping` | `grasp_distance_penalty` | −0.05 | −0.0205 |
| `failure` | `action_failed` | −0.1 | −0.0410 |
| `held_up` | `held_up_penalty` | −0.1 | −0.0410 |

`step_dt` が v1 の 2.8583 から縮んだので、同じ重みでも実効値は
約 1.17 倍になっている (`success` は 1.0496 → 1.2308)。

## 4. 3 つのネットワーク変種

| タスク ID | Runner クラス | `experiment_name` | 方策 |
|---|---|---|---|
| `Isaac-Twisted-RL-Twist-v2` | `KnotTyingPPORunnerCfg` | `knot_tying_v2` | MLP [512,256,128] |
| `Isaac-Twisted-RL-Twist-v2-RNN` | `KnotTyingRNNPPORunnerCfg` | `knot_tying_v2_rnn` | GRU 256 x1 層 + MLP |
| `Isaac-Twisted-RL-Twist-v2-LSTM` | `KnotTyingLSTMPPORunnerCfg` | `knot_tying_v2_lstm` | LSTM 256 x1 層 + MLP |

`experiment_name` を分けるのは必須。`train.py` が
`logs/rsl_rl/{experiment_name}/` をログ根にし、`--resume` がそこから最新 run を
正規表現で拾うため、共有すると **GRU のチェックポイントを MLP に読ませる**
事故が起きる (`load_state_dict` は `strict=True` なので `RuntimeError`)。

### 環境は 1 つ、エージェントが 3 つという構成にした理由

`OnPolicyRunner` は `policy_cfg["class_name"]` から方策を **1 つだけ**
`eval()` で作る (`rsl_rl/runners/on_policy_runner.py:418`)。しかもその
`eval()` は rsl_rl 自身のモジュール名前空間で評価されるので、解決できるのは
import 済みの `ActorCritic` / `ActorCriticRecurrent` **のみ**。
自作の方策クラスを渡すことはライブラリを monkey-patch しない限りできない。

したがって「1 タスクの中でネットワークを切り替える」は不可能で、
**同じ env cfg に別の agent cfg を登録する**のが唯一の形になる。
仕組みは [`knot_tying/__init__.py`](../../__init__.py) の `_AGENT_VARIANTS`。

### 再帰方策がこの環境で有効とは考えにくい (それでも比べる価値はある)

正直に書いておく。観測はリンク 3D 位置 20 個 + 交差表 + クラス one-hot で
**ほぼ完全状態**であり、報酬は `settle_out` (0.3 s) + `relax` (0.5 s) の
静定後に測るので **覚えておくべき速度も残っていない** (ほぼマルコフ)。
さらに 1 エピソードが 6 手しかなく、`recurrent_mini_batch_generator` は
エピソード境界で軌跡を切ってパディングするため、隠れ状態が育つ窓が
そもそも 6 ステップしかない。

一応の理屈はある: `crossing_feats` は交差 5 個ぶんの固定長で、実際の交差数が
変わるとパディング部分の意味が曖昧になるので、再帰的な critic がそれを
平滑化する可能性はある。ただし薄い。

**MLP と同等以下になる想定**で回す。結果 (とくに負の結果) はこの節に追記する。

### rsl-rl 3.0.1 固有の注意点

この環境の rsl-rl は **3.0.1**、isaaclab は **2.3.0**
(`.venv/lib/python3.11/site-packages/`)。`/home/itoyama/work/IsaacLab` の
checkout は **rsl-rl 5.0.1 前提の別物**なので、そこの例をそのまま写すと落ちる。

* `RslRlPpoActorCriticRecurrentCfg` に **`state_dependent_std` は無い**。
  `RslRlRNNModelCfg` / `RslRlMLPModelCfg` も存在しない (4.0.0 以降の API)。
* **`rnn_hidden_dim`** であって `rnn_hidden_size` ではない。後者は
  `rnn_hidden_dim` が既定の 256 のときだけ上書きされる罠つき。
* **`rnn_type` は文字列一致で、間違えても例外が出ない。**
  `Memory` は `nn.GRU if type.lower() == "gru" else nn.LSTM` なので
  **"gru" 以外のあらゆる文字列が黙って LSTM になる**。
  学習開始ログで Actor/Critic の RNN が `GRU(...)` / `LSTM(...)` の
  どちらか**必ず目視確認すること**。
* `actor_obs_normalization` / `critic_obs_normalization` は `MISSING` なので
  **明示的に渡さないと configclass の検証で落ちる**。
* 再帰方策のミニバッチは **時間ではなく環境で切られる**
  (`rollout_storage.py:211` の `mini_batch_size = num_envs // num_mini_batches`)。
  そのため再帰版は `num_mini_batches` を 4 → **2** にしてある。

## 5. 移植元 TWISTED-RL の「交差数ごとにモデルを変える」について

移植元は交差数 x 移動種別ごとに最大 12 個の SAC を切り替える
([`diff_to_TwistedRL.md`](../v1/diff_to_TwistedRL.md) §2)。**v2 ではこれを
実装していない。** 理由は 2 つ:

1. **rsl_rl では表現できない** (§4 の `eval()` の制約)。
2. **off-policy 固有の仕組みに依存している。** 移植元の 12 個の SAC が
   1 個より安いのは `hindsight_sharing` が HER 由来の経験を段をまたいで
   共有するからで、これはリプレイバッファが前提。PPO には無い。

代わりに v2 は **報酬ラダーで段階的な難易度**を実現している。
「交差数ごとに別モデル」は **v3** で、段ごとに別タスク
(`-v3-Stage1/2/3`) を登録し `--resume` で前段のモデルから暖機開始する
カリキュラムとして実装する。これが PPO 下での正直な等価物。

そのために v2 は **観測次元とネットワーク形状を v1 から変えていない**。
`OnPolicyRunner.load` は `load_state_dict` を strict のまま呼ぶので
(`on_policy_runner.py:310`)、形状が揃っていないと暖機開始ができない。

## 6. 未検証の項目 (学習を始める前・並行して確認すること)

[`next_action.md`](../../next_action.md) §4.8.9 のゲートのうち、
**まだ一度も実行していないもの**:

1. **台本で固結びが結べるか (ゲート 1)。** 文書は「これが通らなければ
   どんな報酬設計でも PPO は学習できない」と書いている。現在の記録
   (`outputs/topology_debug/events.log`) は **交差 1 個まで**で、
   台本による固結びの記録は存在しない。

   **v2 の学習開始はこれを待たない。** v2 の最下段 `success_c1` は
   v0 が実証済みの目標なので、ゲート 1 の結果に依らず到達可能な学習信号が
   ある。ゲート 1 は別プロセスで並行実行し、結果で判断するのは
   「最上段 `overhand_achieved` が到達可能か」と
   「`drop_xy_rel_frac` を 0.15 → 0.30 に広げるか」。

   広げる判断を保留しているのは、これが**幾何の変更**で `carry` フェーズ
   0.35 s が覆う距離が倍になる = 把持中の指令速度が上がる (発散の機構
   そのもの) ため。発散対策と同時に変えると原因の切り分けができなくなる。

2. **クラス別の `rope_diverged` の監視。** 自己接触は交差数に対して
   超線形に増えるので、集計値ではクラス 3 固有の発散が隠れる。

## 7. 再現性

このバージョンの `common/` は、実装時点のコミットに対応する。`common/` は
バージョン間で共有されるため、後のバージョンが `common/` を触れば
v2 の挙動も変わり得る。厳密な再現には git のコミットまで戻ること。

v2 の実装で `common/` に加えた変更は 2 つ、**どちらも追加・一般化のみ**:

* [`common/mdp/rewards.py`](../../common/mdp/rewards.py) に
  `interleaved2_achieved` を**追加** (既存項は変更していない)。
* [`common/rope_task_cfg.py`](../../common/rope_task_cfg.py) の
  `rope_diameter` の充填を、項の名前 (`success` / `held_up`) の決め打ちから
  **`params` に `rope_diameter` を宣言している項すべてを埋める**形に一般化。
  以前の実装はバージョンが報酬項を足すたびにここも直さないと**例外なく
  既定値 0.02 が残る**地雷だった (v2 で疎報酬を 3 段に増やしたときに踏んだ)。
  v0/v1 と後方互換で、`fine` プリセット (直径 8 mm) では**バグ修正**になる。
