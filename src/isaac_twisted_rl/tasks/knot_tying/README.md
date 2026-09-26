# knot_tying: 紐結びの強化学習

ロボットハンドでロープを操作し、結び目を作ることを学習させるタスク群。
TWISTED-RL (itoyama_twisted_rl) の MuJoCo 実装を Isaac Lab へ移植したもの。

```bash
# 最初に一度だけ: USD 資産 (ロープ 4 種 + グリッパー) を生成する
uv run python -u src/isaac_twisted_rl/tasks/knot_tying/demos/generate_assets.py

uv run train --task Isaac-Twisted-RL-Twist-v0 --headless   # 交差を 1 つ作る
uv run train --task Isaac-Twisted-RL-Twist-v1 --headless   # 固結び (学習失敗)
uv run train --task Isaac-Twisted-RL-Twist-v2 --headless   # 固結び (作り直し)
```

## バージョン一覧

**課題はバージョン制**。実装を改良するときは既存バージョンを書き換えず、
新しいバージョンを足す。過去のバージョンをいつでも再学習・再評価でき、
改良の効果を比較できるようにするため。

| バージョン | 課題 | エピソード長 | 詳細 |
|---|---|---|---|
| `v0` | ロープに**交差を 1 つ**作って、手を離しても残す | 5 手 | [versions/v0/README.md](versions/v0/README.md) |
| `v1` | **固結び (overhand knot)** を作る (**学習は失敗した**) | 8 手 | [versions/v1/README.md](versions/v1/README.md) |
| `v2` | 固結び。v1 の失敗を実測から特定して作り直したもの | 6 手 | [versions/v2/README.md](versions/v2/README.md) |

v1 は 4000 iteration の設定に対し 157 で打ち切られ、報酬が下降を続けた。
原因は「疎報酬 (`overhand_achieved`) が一度も発火せず、PPO が受け取る勾配が
失敗罰則だけになった」こと。v2 はそこに到達可能な中間目標を 3 段置き、
併せて発散源だった `approach` フェーズの速度指令を消している
(詳細と TensorBoard の実測値は [versions/v2/README.md](versions/v2/README.md))。

各バージョンには 4 つのロープのプリセットがある
(`Twist` / `Simple` / `Stiff` / `Fine`)。**学習に使うのは `Twist`** で、
他 3 つは比較用ベースライン。`-Play-` 付きは推論の様子を見る用
(環境数を絞り、描画間隔を短くしてある)。

```
Isaac-Twisted-RL-{Twist,Simple,Stiff,Fine}[-Play]-{v0,v1}          # 16 タスク
Isaac-Twisted-RL-{Twist,Simple,Stiff,Fine}[-Play]-v2[-RNN|-LSTM]   # 24 タスク
```

v2 だけは **1 つの環境設定に対してエージェント設定が 3 つ**ある
(MLP / GRU / LSTM)。接尾辞なしが MLP、`-RNN` が GRU、`-LSTM` が LSTM。
`OnPolicyRunner` は方策を 1 つしか作れないので、ネットワーク構造を比べるには
同じ env に別の agent cfg を登録するしかない (`_AGENT_VARIANTS` 参照)。

学習ログはバージョンごと、**さらに v2 は変種ごと**に分かれる:
`logs/rsl_rl/knot_tying_v0/` / `knot_tying_v1/` / `knot_tying_v2/` /
`knot_tying_v2_rnn/` / `knot_tying_v2_lstm/`。混ざっていると `--resume` で
誤って別バージョン・別構造のチェックポイントを読んでしまうため
(`load_state_dict` は `strict=True` なので、構造が違えば `RuntimeError`)。

## フォルダ構成

```
knot_tying/
├── __init__.py       gym.register (バージョン x プリセットの二重ループ)
│
├── common/           ← バージョン非依存の共有層
│   ├── env_cfg_base.py     シーン・物理・decimation (KnotTyingEnvCfg)
│   ├── rope_task_cfg.py    プリセット別クラスを作るファクトリ (make_preset_cfgs)
│   ├── rope_state.py       Articulation <-> topology/ の接続層
│   ├── mdp/                MDP 項のライブラリ (**全バージョンの項を集める**)
│   │   ├── phases.py         1 手を構成する 12 フェーズの尺と定数
│   │   ├── actions.py        PickPlaceAction (実装の中心)
│   │   ├── observations.py   観測項 + RopeTopologyExtractor の共有
│   │   ├── rewards.py        報酬項
│   │   ├── terminations.py   終了判定 (NaN 対策)
│   │   └── events.py         起動時イベント (摩擦材 bind, body 順序検算)
│   ├── topology/           3D 点列 -> 位相 (交差・p-data・結び目判定)。
│   │                       isaaclab を import しないので Sim 無しでテストできる
│   ├── rope_model/         ロープの USD 生成と設定
│   └── parallel_gripper/   グリッパーの USD 生成と設定
│
├── versions/         ← バージョンごとの「配線」
│   ├── v0/{env_cfg.py, agents/rsl_rl_ppo_cfg.py, README.md}
│   ├── v1/{env_cfg.py, agents/rsl_rl_ppo_cfg.py, README.md}
│   └── v2/{env_cfg.py, agents/rsl_rl_ppo_cfg.py, README.md}  ← 3 変種
│
└── demos/            ← RL 環境から import されない独立スクリプト
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
バージョンに効く」ことを優先している。厳密な再現が要るときは git の
コミットを遡ること (各バージョンの README に基準コミットを書いてある)。

## バージョンを追加するときの規約 (v2 以降)

1. `versions/vN/` を作り、`env_cfg.py` / `agents/rsl_rl_ppo_cfg.py` /
   `README.md` を置く。`__init__.py` の `_VERSIONS` に `"vN"` を足す
   (**変更はこの 1 行だけ**)。
2. **`common/mdp/` の項は「追加」する。既存の項を削除・改名しない。**
   過去バージョンの設定がその項を参照しているため。役目を終えた項は
   docstring に `.. deprecated::` を書いて残す
   (`first_crossing_feature` が既にこの形)。
3. 既存項の**挙動**を変えたいときは、新しい名前の項を足して新バージョン
   だけが使う。バグ修正は例外 (全バージョンに効かせる)。
4. `experiment_name` は `knot_tying_vN` にする。
5. `vN/README.md` に「v(N-1) から何を・なぜ変えたか」を表で書く。

## デモ / 検証

```bash
# 資産の生成 (最初に一度)
uv run python -u src/isaac_twisted_rl/tasks/knot_tying/demos/generate_assets.py

# ロープを掴むデモ
uv run python -u src/isaac_twisted_rl/tasks/knot_tying/demos/rope_catch.py --rope twist --headless

# 位相抽出が動く物理の上でも正しいかの目視検証 (xy 投影図を出力)
uv run python -u src/isaac_twisted_rl/tasks/knot_tying/demos/topology_debug.py --rope twist --headless

# 位相まわりの単体テスト (Isaac Sim を起動せずに走る)
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/test_knot_tying_topology.py tests/test_knot_detect.py
```

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
  一致**による。**`mdp` というフォルダ名は変えないこと。**

設計の経緯と判断の記録は [`next_action.md`](next_action.md) を参照。
