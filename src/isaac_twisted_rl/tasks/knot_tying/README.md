# knot_tying: 紐結びの強化学習

ロボットハンドでロープを操作し、結び目を作ることを学習させるタスク群。
TWISTED-RL (itoyama_twisted_rl) の MuJoCo 実装を Isaac Lab へ移植したもの。

```bash
# 最初に一度だけ: USD 資産 (ロープ 5 種 + グリッパー) を生成する
uv run python -u src/isaac_twisted_rl/tasks/knot_tying/demos/generate_assets.py

uv run train --task Isaac-Twisted-RL-Twist-v0 --headless   # 交差を 1 つ作る
uv run train --task Isaac-Twisted-RL-Twist-v1 --headless   # 固結び (学習失敗)
uv run train --task Isaac-Twisted-RL-Twist-v2 --headless   # 固結び (作り直し)
uv run train --task Isaac-Twisted-RL-Twist-v2-R2 --headless   # v2 + R2 除去で段を判定
uv run train --task Isaac-Twisted-RL-Knot-v3 --headless    # 論文の中核を取り込んだ版 (Knot で学習)
uv run train --task Isaac-Twisted-RL-Knot-v4 --headless    # 段ごとのエージェント + 1 遷移 = 1 エピソード
uv run train --task Isaac-Twisted-RL-Knot-v5 --headless    # v4 + pick&place の発散対策
uv run train --task Isaac-Twisted-RL-Knot-v6 --headless    # v5 + 次にたどる位相の辺のリンクマスクを観測に
uv run train --task Isaac-Twisted-RL-Knot-v7 --headless    # v6 + めり込み解消速度を v4 の既定に戻す
```

| 文書 | 内容 |
|---|---|
| この README | 使い方・バージョン一覧・フォルダ構成・規約・開発の約束 |
| [`common/README.md`](common/README.md) | 全バージョン共通の仕組み (1 手 = 1 RL ステップ、フェーズ、行動、終了判定、Play の描画) |
| `versions/vN/README.md` | 各バージョンの課題・観測・報酬・結果 |
| [`next_action.md`](next_action.md) | 設計判断と経緯の記録、**現在地と次の作業** |
| [`paper/Report.md`](../../../../paper/Report.md) | 3 本の論文と v2-R2 の比較 (v3 の根拠) |

## バージョン一覧

**課題はバージョン制**。実装を改良するときは既存バージョンを書き換えず、
新しいバージョンを足す。過去のバージョンをいつでも再学習・再評価でき、
改良の効果を比較できるようにするため。

| バージョン | 課題 | エピソード長 | 結果 | 詳細 |
|---|---|---|---|---|
| `v0` | ロープに**交差を 1 つ**作って、手を離しても残す | 5 手 | 学習成功 | [versions/v0/README.md](versions/v0/README.md) |
| `v1` | **固結び (overhand knot)** を作る | 8 手 | 失敗: 疎報酬が一度も発火しない | [versions/v1/README.md](versions/v1/README.md) |
| `v2` | 固結び。到達可能な中間目標を 3 段置き、発散源だった `approach` の速度指令を消した | 6 手 | 偽の C2 に居座った | [versions/v2/README.md](versions/v2/README.md) |
| `v2-R2` | v2 の報酬ラダーを **R2 除去後の位相**で判定する | 6 手 | 本物の C2 が稀で詰まる | [versions/v2_r2/README.md](versions/v2_r2/README.md) |
| `v3` | 論文の中核を取り込む: **到達状態アーカイブからのリセット・固結びで終了・BFS 距離のポテンシャル** (Report §7 P1)。学習は `Knot` | 12 手 (成功で終了) | smoke test のみ | [versions/v3/README.md](versions/v3/README.md) |
| `v4` | TWISTED-RL-C: **段 (固結びへの BFS 距離) ごとに独立したエージェント**、**1 遷移 = 1 エピソード** (+1 / 0 / −1)、段ごとのアーカイブ (台本の途中状態で初期化)。学習は `Knot` | 1 遷移 6 手 (Play は 18 手) | chain 評価 (まっすぐなロープから固結び) 41.6% (iteration 100、seed 42)。iteration 50 で頭打ち | [versions/v4/README.md](versions/v4/README.md) |
| `v5` | v4 (段ごとのエージェント・1 遷移 = 1 エピソード) に **pick&place の発散対策**を足す: 離すときの指先を床から 5〜10 cm に、ロープのめり込み解消速度の上限を 0.4 m/s に。学習は `Knot` | 1 遷移 6 手 (Play は 18 手) | chain 評価 9.6%。発散は v4 の半分以下だが、めり込み解消速度 0.4 m/s で段 2 -> 固結びを学べなかった | [versions/v5/README.md](versions/v5/README.md) |
| `v6` | v5 に **次にたどる位相の辺** (TWISTED-RL の ρ(g)) を観測として足す: over / under 区間のリンクマスク・移動の種類・符号など 64 次元 (段の one-hot の直前)。報酬・終了・アーカイブ・PPO は v5 のまま。学習は `Knot` | 1 遷移 6 手 (Play は 18 手) | chain 評価 37.9% / 22.9% (seed 42 / 43)。v5 の 2〜4 倍。v4 との差はシード差の範囲 ([結果](versions/v6/Result_Report_v6.md)) | [versions/v6/README.md](versions/v6/README.md) |
| `v7` | v6 の **ロープのめり込み解消速度を v4 の既定 (0.96 m/s) に戻す**。v6 の切り分けで、v5 の 0.4 m/s が学習中の段 2 -> 固結びを止めていたと分かった。離す高さ (5〜10 cm)・観測・報酬は v6 のまま。学習は `Knot` | 1 遷移 6 手 (Play は 18 手) | 未学習。同じ設定の切り分けの run (iteration 70) は 31.2% で v6 と差が出ず、確認の学習は保留 | [versions/v7/README.md](versions/v7/README.md) |

各バージョンには 5 つのロープのプリセットがある
(`Twist` / `Simple` / `Stiff` / `Fine` / `Knot`)。**v2-R2 までの学習は `Twist`、v3 以降は `Knot`** で、
`Simple` / `Stiff` / `Fine` は比較用ベースライン。`Knot` は `Twist` と同じ材料で
28 リンク (1.14 m) に延ばした固結び用で、**台本で固結びが結べることを確かめた
唯一のプリセット** (ゲート 1。`Twist` は長さが足りない。詳細は
[`next_action.md`](next_action.md) §4.8.10)。`-Play-` 付きは推論の様子を見る用
(環境数を絞り、描画間隔を短くしてある)。

```
Isaac-Twisted-RL-{Twist,Simple,Stiff,Fine,Knot}[-Play]-{v0,v1}          # 20 タスク
Isaac-Twisted-RL-{Twist,Simple,Stiff,Fine,Knot}[-Play]-v2[-RNN|-LSTM]   # 30 タスク
Isaac-Twisted-RL-{Twist,Simple,Stiff,Fine,Knot}[-Play]-v2-R2[-RNN|-LSTM]   # 30 タスク
Isaac-Twisted-RL-{Twist,Simple,Stiff,Fine,Knot}[-Play]-v3[-RNN|-LSTM]   # 30 タスク
Isaac-Twisted-RL-{Twist,Simple,Stiff,Fine,Knot}[-Play]-v4               # 10 タスク (MLP のみ)
Isaac-Twisted-RL-{Twist,Simple,Stiff,Fine,Knot}[-Play]-v5               # 10 タスク (MLP のみ)
Isaac-Twisted-RL-{Twist,Simple,Stiff,Fine,Knot}[-Play]-v6               # 10 タスク (MLP のみ)
Isaac-Twisted-RL-{Twist,Simple,Stiff,Fine,Knot}[-Play]-v7               # 10 タスク (MLP のみ)
```

`v2-R2` のようにハイフンを含むバージョン表記は、パッケージ名では
`versions/v2_r2/` になる (`__init__.py` の `_version_package`)。

v2〜v3 は **1 つの環境設定に対してエージェント設定が 3 つ**ある
(MLP / GRU / LSTM)。接尾辞なしが MLP、`-RNN` が GRU、`-LSTM` が LSTM。
`OnPolicyRunner` は方策を 1 つしか作れないので、ネットワーク構造を比べるには
同じ env に別の agent cfg を登録するしかない (`_AGENT_VARIANTS` 参照)。
v4 以降は段ごとの方策 (`StageGatedActorCritic`) に再帰版が無いので MLP だけ。

学習ログはバージョンごと、**さらに v2〜v3 は変種ごと**に分かれる:
`logs/rsl_rl/knot_tying_v0/` / `knot_tying_v1/` / `knot_tying_v2/` /
`knot_tying_v2_rnn/` / `knot_tying_v2_lstm/` / `knot_tying_v2_r2{,_rnn,_lstm}/` / `knot_tying_v3{,_rnn,_lstm}/` / `knot_tying_v4/` / `knot_tying_v5/` / `knot_tying_v6/` / `knot_tying_v7/`。混ざっていると `--resume` で
誤って別バージョン・別構造のチェックポイントを読んでしまうため
(`load_state_dict` は `strict=True` なので、構造が違えば `RuntimeError`)。
Hydra の上書きで 1 変数ずつ変えた切り分けの run は、通常の run と混ざらないよう別フォルダに置く
(例: `logs/rsl_rl/knot_tying_v6_ablation/`。各 run の上書きはフォルダの `README.md`)。

## v4 以降の評価 (chain 評価)

v4 以降は、学習ログの段ごとの成功率と `Curriculum/stage_log/chain_est` が本番の成功率を表さない
(段ごとに最大 6 手・アーカイブから始めるのに対して、本番はまっすぐなロープから 18 手で段をつなぐ。
`versions/v6/Result_Report_v6.md` §4.5)。**版の比較は chain 評価で行う:**

```bash
# Play 環境で、まっすぐなロープから固結び・発散・18 手のどれかで終わるまで回し、各環境の最初の 1 エピソードだけを数える
uv run python -u src/isaac_twisted_rl/scripts/rsl_rl/eval_chain.py --task Isaac-Twisted-RL-Knot-Play-v6 \
    --num_envs 512 --headless --device cuda:0 --seed 7 \
    --checkpoint logs/rsl_rl/knot_tying_v6/<run>/model_100.pt agent.device=cuda:0 --out eval.json
```

* 学習で物理を Hydra で上書きした run は、評価にも同じ上書きを付けること。
* 512 エピソードの標準誤差は 2 ポイント前後。**シード差 (同じ版で 15〜19 ポイント) のほうがずっと大きい**ので、
  版の比較は 2 シード以上で行う。
* これまでの結果は各 run のフォルダの `eval_chain_it{N}.json`。

**探索ありの成功率** (論文と同じ土俵。シミュレータで候補の手を試すビーム探索と、探索なしの best-of-N) は
`scripts/rsl_rl/eval_search.py` で測る。使い方とアルゴリズムは [`scripts/rsl_rl/README.md`](../../scripts/rsl_rl/README.md)、
v4〜v7 の結果は [`paper/Report_Search.md`](../../../../paper/Report_Search.md)。結果は各 run のフォルダの `eval_search_*_it{N}.json`。

## フォルダ構成

```
knot_tying/
├── __init__.py       gym.register (バージョン x プリセットの二重ループ)
│
├── next_action.md    設計判断と経緯の記録、現在地と次の作業
│
├── common/           ← バージョン非依存の共有層 (仕組みは common/README.md)
│   ├── env_cfg_base.py     シーン・物理・decimation (KnotTyingEnvCfg)
│   ├── rope_task_cfg.py    プリセット別クラスを作るファクトリ (make_preset_cfgs)
│   ├── rope_state.py       Articulation <-> topology/ の接続層
│   ├── state_archive.py    到達状態アーカイブ (純 torch、v3)
│   ├── mdp/                MDP 項のライブラリ (**全バージョンの項を集める**)
│   │   ├── phases.py         1 手を構成する 12 フェーズの尺と定数
│   │   ├── actions.py        PickPlaceAction (実装の中心)
│   │   ├── finger_geometry.py  指の真下の紐を探す幾何 (v5 の計測。純 torch)
│   │   ├── observations.py   観測項 + RopeTopologyExtractor の共有 (v6: 次にたどる辺 NextEdgeGoal)
│   │   ├── rewards.py        報酬項
│   │   ├── terminations.py   終了判定 (NaN 対策)
│   │   ├── topology_log.py   位相ラダーの集計ログと Play 時の p-data 表示 (v2-R2)、次にたどる辺のログ (v6)
│   │   ├── archive.py        アーカイブからのリセット・保存・ログ (v3、段ごとの版は v4)
│   │   ├── stages.py         段の観測・遷移の報酬・段が変わったら終了 (v4)
│   │   └── events.py         起動時イベント (摩擦材 bind, body 順序検算)
│   ├── policies/           独自の方策ネットワーク (v4: 段ごとのエージェント StageGatedActorCritic)
│   ├── topology/           3D 点列 -> 位相 (交差・p-data・結び目判定・R2 除去・BFS 距離・次にたどる辺)。
│   │                       isaaclab を import しないので Sim 無しでテストできる
│   ├── rope_model/         ロープの USD 生成と設定
│   └── parallel_gripper/   グリッパーの USD 生成と設定
│
├── versions/         ← バージョンごとの「配線」
│   ├── v0/{env_cfg.py, agents/rsl_rl_ppo_cfg.py, README.md}
│   ├── v1/{env_cfg.py, agents/rsl_rl_ppo_cfg.py, README.md, diff_to_TwistedRL.md}
│   ├── v2/{env_cfg.py, agents/rsl_rl_ppo_cfg.py, README.md}  ← 3 変種
│   ├── v2_r2/{env_cfg.py, agents/rsl_rl_ppo_cfg.py, README.md}  ← v2 を継承、判定器だけ差し替え
│   ├── v3/{env_cfg.py, agents/rsl_rl_ppo_cfg.py, README.md}  ← v2-R2 を継承、リセット・終了・報酬を差し替え
│   ├── v4/{env_cfg.py, agents/rsl_rl_ppo_cfg.py, README.md}  ← v3 を継承、段ごとのエージェントと 1 遷移 = 1 エピソード
│   ├── v5/{env_cfg.py, agents/rsl_rl_ppo_cfg.py, README.md}  ← v4 を継承、発散対策 (離す高さ・めり込み解消速度) だけ
│   ├── v6/{env_cfg.py, agents/rsl_rl_ppo_cfg.py, README.md}  ← v5 を継承、観測に次にたどる辺のリンクマスクを足すだけ
│   └── v7/{env_cfg.py, agents/rsl_rl_ppo_cfg.py, README.md}  ← v6 を継承、ロープのめり込み解消速度を v4 の既定に戻すだけ
│
└── demos/            ← RL 環境から import されない独立スクリプト
    ├── generate_assets.py   USD 資産の生成
    ├── rope_catch.py        ロープを掴むだけのデモ
    ├── topology_debug.py    台本で操作しながら位相抽出を目視検証
    ├── overhand_script.py   ゲート 1: 台本で固結びを結ぶ
    └── divergence_bench.py  1 手あたりの発散をフェーズ・段・手の幾何ごとに数える (v5)
```

## 設計方針: 「バージョン = 設定」であって「バージョン = コードのコピー」ではない

`versions/vN/` に置くのは **env 設定・PPO 設定・README だけ**。物理・位相
判定・MDP 項の実装は `common/` に共有で置く。バージョン間の差は
**「どの項を、どの重みで配線するか」**だけで表現する。

この形にできるのは、v1 が v0 の MDP 項を消さずに**追加**したから。
実際 `crossing_progress` (v0) と `knot_class_shaping` (v1) は
`common/mdp/rewards.py` に共存している。

**利点:** 物理まわりのバグ (発散対策、把持ゲイン) を直すと全バージョンに
効く。コピーしていると v0 だけ古いバグが残る。また v1 の README が
「v0 のこの項をこう差し替えた」と書けるのは、両者が同じ語彙で
書かれているから。

**トレードオフ:** `common/` を変えると v0 の挙動も変わりうる。これは
意図的な選択で、「過去バージョンの完全凍結」より「共通基盤の改善が全
バージョンに効く」ことを優先している。厳密な再現が要るときは、git の
コミットを遡ること (v0 は README に基準コミットを書いてある。他のバージョンは、
そのバージョンを追加したコミットが基準)。

## バージョンを追加するときの規約 (v2 以降)

1. `versions/vN/` を作り、`env_cfg.py` / `agents/rsl_rl_ppo_cfg.py` /
   `README.md` を置く。`__init__.py` の `_VERSIONS` に `"vN"` を足す
   (MLP / GRU / LSTM の変種を持たせるなら `_AGENT_VARIANTS` にも足す)。
2. **`common/mdp/` の項は「追加」する。既存の項を削除・改名しない。**
   過去バージョンの設定がその項を参照しているため。役目を終えた項は
   docstring に `.. deprecated::` を書いて残す
   (`first_crossing_feature` が既にこの形)。
3. 既存項の**挙動**を変えたいときは、新しい名前の項を足して新バージョン
   だけが使う。バグ修正は例外 (全バージョンに効かせる)。
4. `experiment_name` は `knot_tying_vN` にする (変種は `_rnn` / `_lstm` を付ける)。
5. `vN/README.md` に「v(N-1) から何を・なぜ変えたか」を表で書く。

## デモ / 検証

```bash
# 資産の生成 (最初に一度)
uv run python -u src/isaac_twisted_rl/tasks/knot_tying/demos/generate_assets.py

# ロープを掴むデモ
uv run python -u src/isaac_twisted_rl/tasks/knot_tying/demos/rope_catch.py --rope twist --headless

# 位相抽出が動く物理の上でも正しいかの目視検証 (xy 投影図を出力)
uv run python -u src/isaac_twisted_rl/tasks/knot_tying/demos/topology_debug.py --rope twist --headless

# ゲート 1: 台本で固結びを結ぶ (学習環境をそのまま使い、ハンドだけ台本で動かす)
uv run python -u src/isaac_twisted_rl/tasks/knot_tying/demos/overhand_script.py --task Isaac-Twisted-RL-Knot-v2-R2 --num_envs 128 --jitter 0.25 --headless
#   同じ台本を 6 次元の行動に写して env.step() で再生する
uv run python -u src/isaac_twisted_rl/tasks/knot_tying/demos/overhand_script.py --task Isaac-Twisted-RL-Knot-v2-R2 --num_envs 128 --jitter 0.25 --headless --replay_actions
#   台本の途中状態を段ごとに保存する (v4 の初期アーカイブ。outputs/archive_seed/)
uv run python -u src/isaac_twisted_rl/tasks/knot_tying/demos/overhand_script.py --task Isaac-Twisted-RL-Knot-v2-R2 --num_envs 128 --jitter 0.25 --headless --dump_archive

# 1 手あたりの発散を、どのフェーズで・どんな手で起きたか込みで数える (v5 README の「計測の方法」)
uv run python -u src/isaac_twisted_rl/tasks/knot_tying/demos/divergence_bench.py --task Isaac-Twisted-RL-Knot-v5 --num_envs 1024 --steps 24 --headless --archive_prob 0.0
```

単体テストの実行方法は、下の「開発の約束」を参照。

## 注意: Kit 起動前に import されるファイル

`tasks/__init__.py` の `import_packages` が、この配下の**パッケージの
`__init__.py` を Kit 起動前にすべて実行する**。そこで `isaaclab.sim` などを
import すると `list_envs` / `train` が丸ごと落ちる。

- 各 `__init__.py` は docstring だけにしておく。
- 環境設定は `gym.register` の**文字列 entry point** で参照し、実 import を
  `gym.make()` まで遅延させる。
- デモ本体はプレーンモジュールなので自動 import されない
  (`import_packages` はパッケージの `__init__` しか実行しない)。
- `common/mdp/` が除外されるのは、ブラックリスト `".mdp"` の**部分文字列
  一致**による (`isaaclab_tasks/utils/importer.py` の `_walk_packages`)。
  **`mdp` というフォルダ名は変えないこと。**

## 開発の約束

- **`common/topology/` と `common/state_archive.py` に `isaaclab` / `omni` / `pxr` を import しない。**
  崩すと単体テストが Isaac Sim 無しで走らなくなり、開発サイクルが桁違いに遅くなる。
  `Articulation` に触るのは `common/rope_state.py` と `common/mdp/` だけ。
- **テストを壊さない。** 変更したら必ず通すこと:
  ```bash
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/
  ```
  環境変数が要るのは、`/opt/ros/jazzy` の `launch_testing` が pytest のプラグイン自動読み込みに
  引っかかり、`ModuleNotFoundError: lark` で落ちるため (テスト自体とは無関係)。
- **lint / format を通す。** `.pre-commit-config.yaml` の ruff v0.14.10、line-length 120。
  `ruff check <path> && ruff format <path>`。
- **Isaac Sim を起動するスクリプトは `python -u` で回す。** 出力をファイルへリダイレクトすると
  stdout がブロックバッファリングされ、数分間なにも出ずハングしたように見える。
- **docstring は日本語で書き、「何をするか」より「なぜそうしたか」を書く。** 移植元と挙動を変えるときは、
  変えた理由と検証結果を docstring に残す。
- **rsl-rl は 3.0.1、isaaclab は 2.3.0** (`.venv` 内)。`/home/itoyama/work/IsaacLab` の checkout は
  rsl-rl 5.x 前提の別物なので、そこの例をそのまま写さない (注意点は [v2 README](versions/v2/README.md) §4)。
