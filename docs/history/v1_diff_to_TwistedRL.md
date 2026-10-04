> **状態: 経緯の記録 (凍結)。v1 の時点で移植元 (MuJoCo 版) のコードと比べたスナップショット。通常は読まない**
> - 旧パス `src/isaac_twisted_rl/tasks/knot_tying/versions/v1/diff_to_TwistedRL.md`。v2 以降の実装とは食い違う記述がある (num_envs など)。論文との比較の後継は `paper/Report_Paper.md`。

# `Isaac-Twisted-RL-Twist-v1` と 移植元 TWISTED-RL の差分

移植元: `/home/itoyama/work/itoyama_twisted_rl`（ICRA 2026, Freund et al.,
"TWISTED-RL: Hierarchical Skilled Agents for Knot-Tying without Human
Demonstrations"）。

> **この文書は v1 時点で移植元の「コード」と比べたスナップショット。**
> v2-R2 時点で 3 本の「論文」と比べ直したのが [`paper/Report.md`](../../paper/Report_Paper.md)。
> その結論を受けて、v3 は論文の仕組みのうち次の 3 つを PPO のまま取り込んだ
> ([v3 README](../../src/isaac_twisted_rl/tasks/knot_tying/versions/v3/README.md)): 到達状態からのリセット (移植元のカリキュラム初期状態に相当)、
> 成功でのエピソード終了、p-data グラフの BFS 距離 (報酬のポテンシャルとして使う。高レベル計画は無し)。
> 以下の「v1 には無い」という記述は v1 時点のもの。

結論を先に書く: **現在の v1 は TWISTED-RL の「ロープのトポロジー表現」と
「MuJoCo → Isaac Lab の物理移植」だけを引き継ぎ、論文の核である
階層構造（BFS プランナー + 交差数/移動種別ごとの SAC 切り替え + HER）は
まるごと単一の PPO 方策に置き換えている**。タイトルは "Twisted-RL" だが、
実装としては別物と考えたほうが正確。

## 0. 一枚で見る全体像

| 観点 | TWISTED-RL (移植元) | `Isaac-Twisted-RL-Twist-v1` |
|---|---|---|
| 学習アルゴリズム | **SAC**（オフポリシー、リプレイバッファ、PER、エントロピー自動調整） | **PPO**（rsl_rl、オンポリシー、リプレイバッファ無し） |
| モデル構成 | **交差数 × 移動種別ごとに複数の SAC を切り替え**（`G0`,`G1_R2`,... 最大12種類） | **単一の actor-critic 1個だけ**。位相段階は観測に one-hot で見せるのみ |
| 階層構造 | **High-level (BFS+バンディット探索) と Low-level (SAC) の2層** | **1層のみ**。階層プランナー相当のコードは存在しない |
| ゴール設計 | **goal-conditioned**（`desired_goal` を辞書観測で受け取り、HER で再ラベリング） | **固定ゴール**（固結び1種類のみ、goal 入力なし） |
| 状態表現 | 生の MuJoCo configuration (93次元) + リンク3D位置 + シンボリックなゴールベクトル | テンソル層のみ（交差数・writhe・交差特徴・knot_class one-hot）+ リンク3D位置、105次元 |
| 行動空間 | 4次元連続 (`link, z, x, y`)、mocap/weld によるスプライン軌道追従 | 6次元連続 (`grasp_link, delta_link, drop_x/y/z, end_yaw`)、キネマティックハンド直接駆動 |
| 報酬 | スパース `{+1, 0, -1}` のみ（ゴール到達/禁止遷移/現状維持） | 密なポテンシャル整形 (`knot_class_shaping`) + スパース成功報酬 + 複数の補助罰則 |
| シミュレータ | MuJoCo (dm_control)、21リンク capsule composite rope | Isaac Sim/PhysX、D6ジョイント連鎖 (20リンク)、1024並列 |
| 実行粒度 | pick-and-place 1手 = 1 low-level episode（最大7ステップ） | pick-and-place 1手 = 1 env step（1エピソード8手） |

---

## 1. 学習アルゴリズム: SAC → PPO

**SAC ではなく PPO を使っている、という理解は正しい。**

- 移植元 `exploration/rl/cleanrl_scripts/sac_algorithm.py` は CleanRL の
  `sac_continuous_action.py` をベースにしたオフポリシー SAC。
  - Gaussian actor + tanh スカッシング、Twin Q-network (`DoubleQNetworks`)
    + target network（Polyak 更新 `tau=0.005`）。
  - エントロピー自動調整（`autotune=1` が既定、`target_entropy=-action_size`）。
  - Prioritized Experience Replay（`sac.yml`: `per_alpha=0.6`, `per_beta=0.4`）。
  - リプレイバッファ `buffer_size=700000`、`updates_per_env_step=2`
    （1 env step ごとに 2 回勾配更新するオフポリシー特有の高頻度更新）。
  - `learning_starts=2500`、`total_timesteps=2000000`、
    `policy_lr=1e-6` / `q_lr=1e-5`（極めて小さい学習率）。
  - Behavioral Cloning 補助損失、レイヤーフリーズによるカリキュラム間の
    転移学習機構（前段階の重みを一部凍結して次段階を学習）も用意されている。
  - `hindsight_sharing` により、ある遷移で生成された HER 由来の経験を
    **他のカリキュラム問題（R1/R2/cross 別の replay buffer ファイル）にも
    共有する**仕組みがある。

- v1 (`versions/v1/agents/rsl_rl_ppo_cfg.py`) は `isaaclab_rl.rsl_rl` の
  `RslRlOnPolicyRunnerCfg` / `RslRlPpoAlgorithmCfg`。
  - `clip_param=0.2`、GAE `lam=0.95`、`num_learning_epochs=5` で集めた
    ロールアウトを使い切って捨てるオンポリシー更新。
  - リプレイバッファ・ターゲットネットワーク・エントロピー自動調整・
    PER・HER・BC・レイヤーフリーズ、いずれも存在しない。
  - `num_steps_per_env=16`（8手 x 2エピソード分）ごとに1回更新。

この違いは実装上の些末な選択ではなく、**move/交差数ごとに複数エージェントを
使い分けてカリキュラム的に学習を積み上げる TWISTED-RL の設計と、
リプレイバッファで過去の経験を跨いで再利用する SAC は本質的に結びついている**
（`hindsight_sharing` はまさにそのための仕組み）。PPO に切り替えた時点で、
この節で挙げた仕組みは全部前提が崩れるため単純移植はできず、v1 はそれらを
引き継いでいない（後述 §2, §4）。

---

## 2. モデル構成: 交差数/移動種別ごとの複数 SAC → 単一 PPO 方策

**「結び目の遷移ごとに NN を変えている」という理解は正しい。v1 では変えていない、
という理解も正しい。**

### 移植元: 最大12種類のエージェントを動的に切り替える

`system_flow/low_level_class/rl_planner.py` の
`get_agent_by_crossing_number_and_move(crossing_number, move)` が中核:

```python
while crossing_number >= 0:
    problem_names = [f'G{crossing_number}_{move.capitalize()}',
                      f'G{crossing_number}', f'G_{move.capitalize()}', 'G']
    for problem_name in problem_names:
        if self.agents.get(problem_name) is not None:
            return self.agents[problem_name]
    crossing_number -= 1
raise ValueError(...)
```

- `move` は Reidemeister移動の種類 (`R1` / `R2` / `Cross`)。
- 優先順位は「交差数×move専用」→「交差数専用」→「move専用」→「万能モデル `G`」
  で、見つからなければ交差数を1つ下げて再試行するフォールバック。
- どこまで細分化するかはアブレーション設定 (`exploration/utils/config_utils.py`
  の `set_agents_by_ablation`) で選べる:
  - `G`: 万能モデル1個のみ
  - `A`: move種別ごとに1個（`G_R1`,`G_R2`,`G_Cross`）
  - `C`: 交差数ごとに1個（`G0`,`G1`,`G2` — README の "TWISTED-RL-C"）
  - `AC`: 交差数×move の全組み合わせ（最大12個）
- `system_flow/config/twisted_evaluation.yml` の既定 (`LOW_LEVEL.NAME:
  MultiRLPlanner`) は、これら複数アブレーションを**同時にロードして比較評価**
  する構成。
- 呼び出し元は `RLPlanner.generate_action`。high-level から「次にどの
  Reidemeister移動で遷移すべきか」を渡されるたびに、その時点の交差数と
  move種別に応じたエージェントへ**その場で切り替えて**推論する。

つまり移植元は「1つの巨大な方策が全部を学ぶ」のではなく、**易しい問題
(交差0→1) を解けるモデルをまず学習し、その経験を使ってより難しい問題
(交差1→2) 用のモデルを段階的に学習する**、カリキュラム的な分割統治になっている
（README の `--replay_buffer_files_path` オプションもこの段階的学習を支える）。

### v1: 単一 actor-critic、`knot_class` は観測に見えるだけ

- `versions/v1/agents/rsl_rl_ppo_cfg.py` の `RslRlPpoActorCriticCfg` は
  actor 1組・critic 1組のみ（`actor_hidden_dims=[512,256,128]`）。
  モデルを選択・切り替えるコードは環境側にも学習設定側にも存在しない。
- `knot_class`（位相段階 0〜3 の one-hot、4次元）は
  `common/mdp/observations.py` の `knot_class_one_hot()` が計算し、
  `versions/v1/env_cfg.py` の `ObservationsCfg.PolicyCfg` で他の観測項と
  `concatenate_terms=True` により**1本のベクトルに結合されるだけ**。
  この値でネットワークや推論経路が分岐する箇所はコード中に一切ない。
- `knot_class_shaping`（`common/mdp/rewards.py`）も同様に、位相段階を
  読んで**単一のスカラー報酬**を返すだけで、報酬関数自体を段階ごとに
  切り替えているわけではない。
- 階層プランナーに相当する仕組み（BFS グラフ探索、バンディット的な
  プラン再選択、`expand`によるランダム探索フォールバック）も存在しない。
  `common/mdp/rewards.py` や `common/topology/knot_detect.py` のコメントに
  「移植元の BFS」という言及があるが、これは**固結び判定条件を机上で
  導出するためにオフラインで使った**という記述であり、学習・推論時に
  動くランタイムの BFS 実装ではない。

**設計思想の違い**: 移植元は「シンボリックな高レベル探索（学習しない
古典的アルゴリズム）+ 局所的な低レベル運動を段階的に学習する分割統治」。
v1 は「単一方策に全遷移（交差0→1→2→3）を通しで学習させる end-to-end
アプローチ」。後者は設計としてシンプルだが、TWISTED-RL が分割統治を選んだ
動機（疎な成功信号のもとで難しい問題をいきなり解かせるのは困難）をそのまま
一手に引き受けることになる。v1 の報酬がスパース成功報酬だけでなく密な
ポテンシャル整形 (`knot_class_shaping`) を主体に据えているのは、
この一手引き受けを緩和するための代替策と位置づけられる（§5 参照）。

---

## 3. 階層構造 (High-level / Low-level): 2層 → 1層

移植元の `system_flow/high_level_class/high_level.py` の
`HighLevelPlanner.tie_knot` はおおよそ以下のループ:

1. `mujoco_infra/mujoco_utils/topology/BFS.bfs_all_path_new` で開始状態から
   目標状態までの**全経路を列挙**（各エッジが1回の Reidemeister移動）。
2. 訪問回数に基づくバンディット的な選択で初期状態・プランを選ぶ
   (`_bandit_select_topology_state`, `_bandit_select_trejctory_from_topology_state`)。
3. プランに沿って `low_planner.find_curve`（= 上記の交差数/move別 SAC）を
   1遷移ずつ呼び出し、成功したら `GraphManager`（`networkx.DiGraph`）に
   実際に到達したノード・エッジを記録。
4. 停滞したら一定確率でランダム探索 (`expand`) にフォールバック。
5. ゴール到達かタイムアウトまで 1〜4 を繰り返す（Anytime なクローズドループ探索）。

v1 にはこれに相当するものが無く、`ManagerBasedRLEnv` の1ステップ = 1手を
PPO 方策が毎回直接出力するフラットな構造。「次にどの Reidemeister 移動を
狙うか」という中間的な意思決定を明示的に行う層が無く、そこは方策の
価値関数と報酬シェーピング (`knot_class_shaping`) に暗黙的に埋め込まれている。

---

## 4. ゴール設計: goal-conditioned + HER → 固定ゴール

- 移植元は **goal-conditioned RL**。
  `exploration/rl/environment/exploration_gym_envs.py` の観測空間は
  `DictSpace({'observation': ..., 'desired_goal': ...})`。
  `exploration/goal_selector/goal_selector.py` が高レベルグラフのエッジ
  （= 「ある Reidemeister移動を1回実行して到達する状態」）をランダムに
  選び、それを目標として low-level SAC に渡す。
  失敗したエピソードも `apply_her`（final-strategy HER）で
  実際に到達した状態を目標として再ラベリングし、成功エピソードとして
  リプレイバッファに追加する。
- v1 は固定ゴール（固結びを作る、の1種類のみ）。観測にゴールIDや
  目標状態を示す入力は無く、`overhand_achieved` という単一の成功条件を
  常に目指す。目標を変える場合（v0: 交差1つ作る → v1: 固結びを作る）は
  **別バージョンとして新しい環境・新しい学習済みモデルを用意する**設計
  （[`../../README.md`](../../README.md) のバージョン制方針）。

HER は「同じ方策に多様なサブゴールを経験させることで、疎な最終報酬でも
学習を進める」ためのオフポリシー特有のテクニックであり、これも SAC→PPO の
変更とセットで失われている。

---

## 5. 報酬設計: スパース {+1,0,-1} → 密なポテンシャル整形+複合罰則

- 移植元 (`exploration_worker_env.py`) はスパースな3値報酬のみ:
  ゴール到達 `+1`、禁止遷移 `-1`、それ以外 `0`。シェーピングは無い。
  疎な信号の学習可能性は、上述の HER・goal-conditioning・カリキュラム的な
  モデル分割によって担保している。
- v1 (`versions/v1/env_cfg.py` の `RewardsCfg`) は密報酬が主体:
  - `knot_class_shaping`（重み+1.0）: ポテンシャル整形
    `F = γΦ(s') − Φ(s)`、`Φ=(0.0, 0.35, 0.70, 1.0)` の4段。
  - `overhand_achieved`（+3.0、スパース）: 固結び成立 かつ 床に落下済み。
  - `offladder_penalty`（-0.05）、`grasp_distance_penalty`（-0.05）、
    `action_failed`（-0.2）、`held_up_penalty`（-0.1）の4つの補助罰則。

  ポテンシャル整形の γ (`env_cfg.GAMMA=0.98`) は PPO の割引率
  (`rsl_rl_ppo_cfg.py` の `algorithm.gamma`) と共有され、Ng et al. (1999) の
  最適方策不変性が成り立つよう明示的に管理されている。

移植元がスパース報酬+HER+階層分割で学習可能性を担保するのに対し、
v1 は単一方策のまま密報酬で学習可能性を担保する、という設計上の対称性がある。

---

## 6. 状態表現・行動空間

### 状態表現

- 移植元: low-level は MuJoCo の生の `qpos+qvel`（93次元）とリンク3D位置
  (`Preprocessor.preprocess_qG`) を連結した連続値ベクトル。ゴールは
  対象リンク区間の one-hot マスク + Reidemeister移動の属性
  (`left, over_first, over_before_under, sign`) 等をベクトル化したもの。
  p-data 文字列（Gauss code的な記号表現）自体は評価時の初期/目標状態の
  **指定**にのみ使い、学習時の内部状態表現としては使っていない。
- v1: `common/mdp/observations.py` はテンソル層のみ（p-data文字列は
  ログ用途以外で一切生成しない。理由は速度差、4096環境で94ms対1.6ms）。
  105次元 = `rope_link_pos`(60) + `crossings`(1) + `writhe`(1) +
  `crossing_feats`(30, 交差5個分) + `knot_class`(4) + `hand_pos`(3) +
  `actions`(6)。

この点（記号表現を避けテンソル演算に閉じる）は移植の共通方針であり、
両者とも「p-data文字列を学習ループの中では使わない」という点で一致している
（移植元も低レベル学習では生の物理状態を使っている）。

### 行動空間

- 移植元: 4次元連続 (`link∈[0,1]`, `z`, `x`, `y`)。対象リンクを掴み、
  MuJoCo の mocap ボディ + weld拘束でスプライン軌道に沿って
  持ち上げ→移動→溶接解除、という粗粒度スキル。関節トルク制御ではない。
- v1: 6次元連続 (`grasp_link, delta_link, drop_x/y/z_rel, end_yaw`)。
  キネマティックハンド（アーム無し）の root 姿勢を直接書き込んで駆動。
  yaw（手首角度）を明示的に持つ点、置き先をリンク相対で指定する点が
  移植元の4次元行動より表現力が高い。

両者とも「pick-and-place 1手を1つの粗粒度アクションとして扱う」という
Options Framework的な粒度は共通している。

---

## 7. シミュレータ

- 移植元: MuJoCo (`dm_control.mujoco.Physics`)。ロープは
  `<composite type="rope" count="21 1 1">` の21リンク capsule連鎖。
  把持は mocapボディ + `<equality><weld>` による溶接拘束（動的アタッチ）。
- v1: Isaac Sim/PhysX、GPU並列 (`num_envs=1024`)。ロープは D6ジョイントで
  連結した20リンクの `Articulation`（`common/rope_model/`）、曲げ/捩り剛性は
  ジョイントの PD ドライブで表現。把持は固定ジョイントではなく高摩擦材
  (μ=4.0) のみで実現。

移植の目的が「MuJoCo単一環境の逐次評価」から「Isaac Lab GPU並列環境での
大規模学習」への転換であることが、把持方式（溶接 vs 摩擦）や
交差検出の実装（`common/topology/intersections.py` が shapely依存を除去し
バッチテンソル演算化）にも表れている。

---

## 8. まとめ: 何を継承し、何を捨てたか

**継承したもの**
- ロープのトポロジー（交差数・Reidemeister移動・p-data相当の位相構造）を
  使って知的タスクを定義するという発想そのもの。
- pick-and-place を1つの粗粒度アクションとして扱う設計。
- 固結び判定を厳密な交差数一致ではなく部分列的な条件で行う頑健化の方向性
  （移植元がBFSで全数列挙して導いた知見を再利用）。

**捨てた／置き換えたもの**
- SAC (オフポリシー・リプレイバッファ・PER・エントロピー自動調整) → PPO
  (オンポリシー、リプレイバッファ無し)。
- 交差数×move種別ごとの複数モデル切り替え → 単一の actor-critic。
- BFSベースのシンボリック high-level プランナー（バンディット探索+
  ランダムexpand）→ 存在しない（フラットな1層構造）。
- goal-conditioned 設計 + HER → 固定ゴール（バージョンごとに別モデル）。
- スパース {+1,0,-1} 報酬 → 密なポテンシャル整形＋複合罰則。
- MuJoCo → Isaac Sim/PhysX（GPU並列）。

要するに、v1 は TWISTED-RL の**問題設定（ロープでの結び目トポロジータスク）**
を引き継ぎつつ、**解法のアーキテクチャ（階層 SAC + goal-conditioning + HER）を
end-to-end の単一 PPO 方策に総取り換えした**実装である。論文の中心的な貢献
（交差数ごとのカリキュラム分割と階層プランナー）は、現状の v1 には存在しない。
