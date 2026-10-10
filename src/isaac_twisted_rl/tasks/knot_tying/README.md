# knot_tying: 紐結びの強化学習

- 状態: 現行 (パッケージの仕様: 版の系譜・タスク ID・フォルダ構成)
- 読むべき状況: このパッケージのどこに何があるかを知りたいとき、タスク ID を組むとき
- 現在地 (どの版が基準か、何が済んだか) と結果の数字は、ここには書かない: `docs/STATUS.md` と [`docs/experiments/`](../../../../docs/experiments/README.md)

ロボットハンドでロープを操作し、固結び (overhand knot) を作ることを学習させるタスク群。
TWISTED-RL (Freund ら) の MuJoCo 実装 (itoyama_twisted_rl) を Isaac Lab へ移植したもの。

```bash
# 最初に一度だけ: USD 資産 (ロープ 6 種 + グリッパー) を生成する。既存の USD を作り直すときは --rope <名前> で絞る
uv run python -u src/isaac_twisted_rl/tasks/knot_tying/demos/generate_assets.py

uv run train --task Isaac-Twisted-RL-Knot-v6 --headless   # 学習 (手順は skill train-eval)
```

| 文書 | 内容 |
|---|---|
| この README | 版の系譜・タスク ID の規則・ロープのプリセット・フォルダ構成 |
| [`common/README.md`](common/README.md) | 全版に共通の仕組み (1 手 = 1 RL ステップ、フェーズ、行動、終了判定、Play の描画) |
| `versions/vN/README.md` | 各版の課題・観測・報酬と、その理由 |
| [`docs/system/evaluation.md`](../../../../docs/system/evaluation.md) | chain 評価の定義と、版どうしの比べ方 |
| [`docs/system/logs.md`](../../../../docs/system/logs.md) | ログのフォルダ名・run の中身・ログのキーの意味 |
| [`docs/decisions/`](../../../../docs/decisions/README.md) | 版をまたぐ設計の判断と理由 (「版 = 設定」など) |
| [`rope_study/README.md`](rope_study/README.md) | 紐モデルの比較 (打ち切り中) の道具 |
| [`paper/Report_Paper.md`](../../../../paper/Report_Paper.md) | 3 本の論文と v2-R2 の比較 (v3 の根拠。2026-09-26 時点) |

## 版の系譜

**課題は版 (MDP の版) で管理する**。実装を改良するときは既存の版を書き換えず、新しい版を足す
(理由: [`docs/decisions/`](../../../../docs/decisions/README.md) の「版 = 設定」)。版の追加の手順は skill `add-version`。

| 版 | 継承元 | 課題と、前の版から変えたこと | エピソード長 | 詳細 |
|---|---|---|---|---|
| `v0` | — | ロープに**交差を 1 つ**作って、手を離しても残す | 5 手 | [versions/v0/README.md](versions/v0/README.md) |
| `v1` | v0 | **固結び**を作る (位相クラスのラダー + ポテンシャル整形) | 8 手 | [versions/v1/README.md](versions/v1/README.md) |
| `v2` | v1 | 固結び。到達可能な中間目標を 3 段置き、発散源だった `approach` の速度指令を消した | 6 手 | [versions/v2/README.md](versions/v2/README.md) |
| `v2-R2` | v2 | v2 の報酬ラダーを **R2 除去後の位相**で判定する | 6 手 | [versions/v2_r2/README.md](versions/v2_r2/README.md) |
| `v3` | v2-R2 | 論文の中核を取り込む: **到達状態アーカイブからのリセット・固結びで終了・BFS 距離のポテンシャル**。学習はロープ `Knot` | 12 手 (成功で終了) | [versions/v3/README.md](versions/v3/README.md) |
| `v4` | v3 | TWISTED-RL-C: **段ごとに独立したエージェント**、**1 遷移 = 1 エピソード** (+1 / 0 / −1)、段ごとのアーカイブ (台本の途中状態で初期化)。段 = round(3Φ) で 0 = 自明・3 = 固結び | 1 遷移 6 手 (Play は 18 手) | [versions/v4/README.md](versions/v4/README.md) |
| `v5` | v4 | **pick&place の発散対策**: 離すときの指先を床から 5〜10 cm に、ロープのめり込み解消速度の上限を 0.4 m/s に | 同上 | [versions/v5/README.md](versions/v5/README.md) |
| `v6` | v5 | **次にたどる位相の辺** (TWISTED-RL の ρ(g) の代わり) を観測に足す: over / under 区間のリンクマスク・移動の種類・符号など 64 次元。報酬・終了・アーカイブ・PPO は v5 のまま | 同上 | [versions/v6/README.md](versions/v6/README.md) |
| `v7` | v6 | **ロープのめり込み解消速度を v4 の既定 (0.96 m/s) に戻す**。離す高さ・観測・報酬は v6 のまま | 同上 | [versions/v7/README.md](versions/v7/README.md) |
| `v8` | v7 | **PPO の更新に対称性のデータ拡張** (鏡像・逆順で 4 倍)。環境 (観測・報酬・物理) は v7 のまま | 同上 | [versions/v8/README.md](versions/v8/README.md) |
| `v9` | v7 | **学習の初期状態で段 2 を選ぶ重みを 2 倍に** (`stage_weights` (1, 1, 1) -> (1, 1, 2))。観測・報酬・物理・PPO は v7 のまま (v8 の対称性は入れない) | 同上 | [versions/v9/README.md](versions/v9/README.md) |
| `v10` | v7 | **段を固結びへの正規の経路の上の p-data だけに絞る** (段 0〜2 は 7 個の p-data、外れた形は −1 で終了。段 3 は v7 と同じ判定)。`next_edge` の候補も経路の上に絞る。観測の形・報酬の値・物理・PPO・`stage_weights` は v7 のまま | 同上 | [versions/v10/README.md](versions/v10/README.md) |
| `v11` | v7 | **v7 の段のまま、Φ の表に無く R2 除去後のラダーでも 0 の「行き止まりの形」だけを −1 にする** (−1 で終了・アーカイブに入らない)。生が表に無く R2 除去後が表にある形は R2 除去後の形の段。`next_edge`・初期アーカイブ・観測の形・報酬の値・物理・PPO・`stage_weights` は v7 のまま | 同上 | [versions/v11/README.md](versions/v11/README.md) |
| `v12` | v7 | **到達状態アーカイブに「段が上がった手の後の形」だけを保存する** (`save_on = "stage_up"`。段 0 には学習中に何も足さない)。記録のためにアーカイブの写しを 50 iteration ごとに残す。段の定義・初期アーカイブ・観測・報酬・物理・PPO・`stage_weights` は v7 のまま | 同上 | [versions/v12/README.md](versions/v12/README.md) |

各版の状態 (学習したか、結果、比べた相手) は `docs/STATUS.md` の「版の状態」にある。

## ロープのプリセットとタスク ID

各版には 5 つのロープのプリセット (`Twist` / `Simple` / `Stiff` / `Fine` / `Knot`) がある。
**v2-R2 までの学習は `Twist`、v3 以降はロープ `Knot`** で、`Simple` / `Stiff` / `Fine` は比較用。
ロープ `Knot` は `Twist` と同じ材料で 28 リンク (1.14 m) に延ばした固結び用で、**台本で固結びが結べることを確かめた
唯一のプリセット** (ゲート 1。`Twist` は長さが足りない: [`docs/experiments/2026-09-27_gate1.md`](../../../../docs/experiments/2026-09-27_gate1.md))。
プリセットの物理パラメータは `common/rope_model/rope_specs.py` の `ROPE_SPECS` (USD を作るプリセットは 6 種で、タスクに登録するのは 5 種)。

```
Isaac-Twisted-RL-{Twist,Simple,Stiff,Fine,Knot}[-Play]-{v0,v1}          # 20 タスク
Isaac-Twisted-RL-{Twist,Simple,Stiff,Fine,Knot}[-Play]-v2[-RNN|-LSTM]   # 30 タスク
Isaac-Twisted-RL-{Twist,Simple,Stiff,Fine,Knot}[-Play]-v2-R2[-RNN|-LSTM]   # 30 タスク
Isaac-Twisted-RL-{Twist,Simple,Stiff,Fine,Knot}[-Play]-v3[-RNN|-LSTM]   # 30 タスク
Isaac-Twisted-RL-{Twist,Simple,Stiff,Fine,Knot}[-Play]-{v4,v5,v6,v7,v8,v9,v10,v11,v12} # 各 10 タスク (MLP のみ)
Isaac-Twisted-RL-RS-<紐モデル>[-Play]-v4                                # rope study (rope_study/__init__.py)
```

- `-Play` の意味は版で違う。v0〜v3 は推論の様子を見る環境 (環境数を絞り、描画間隔を短くしてある)、**v4 以降は chain 評価の環境** ([`docs/system/evaluation.md`](../../../../docs/system/evaluation.md) §3)。
- `v2-R2` のようにハイフンを含む版の表記は、パッケージ名では `versions/v2_r2/` になる (`__init__.py` の `_version_package`)。
- v2〜v3 は **1 つの環境設定に対してエージェント設定が 3 つ**ある (MLP / GRU / LSTM)。接尾辞なしが MLP、`-RNN` が GRU、`-LSTM` が LSTM。
  `OnPolicyRunner` は方策を 1 つしか作れないので、ネットワーク構造を比べるには同じ env に別の agent cfg を登録するしかない (`_AGENT_VARIANTS` 参照)。
  v4 以降は段ごとの方策 (`StageGatedActorCritic`) に再帰版が無いので MLP だけ。
- 学習ログは版ごと (v2〜v3 は変種ごと) のフォルダ `logs/rsl_rl/knot_tying_vN/` に分かれる。混ざっていると `--resume` で
  別の版・別の構造のチェックポイントを読んでしまうため (`load_state_dict` は `strict=True` なので、構造が違えば `RuntimeError`)。
  **プリセットはログ名に入らない。** 決まり方と注意は [`docs/system/logs.md`](../../../../docs/system/logs.md)。

## フォルダ構成

```
knot_tying/
├── __init__.py       gym.register (版 x プリセットの二重ループ)
│
├── common/           ← 版に依らない共有層 (仕組みは common/README.md)
│   ├── env_cfg_base.py     シーン・物理・decimation (KnotTyingEnvCfg)
│   ├── rope_task_cfg.py    プリセット別クラスを作るファクトリ (make_preset_cfgs)
│   ├── rope_state.py       Articulation <-> topology/ の接続層
│   ├── state_archive.py    到達状態アーカイブ (純 torch、v3)
│   ├── beam_search.py      推論時探索 (ビーム探索・到達木からの再開) の純 torch 部分。eval_search.py が使う
│   ├── mdp/                MDP 項のライブラリ (**全版の項を集める**。項は追加だけ)
│   │   ├── phases.py         1 手を構成する 12 フェーズの尺と定数
│   │   ├── actions.py        PickPlaceAction (実装の中心)
│   │   ├── finger_geometry.py  指の真下の紐を探す幾何 (v5 の計測。純 torch)
│   │   ├── observations.py   観測項 + RopeTopologyExtractor の共有 (v6: 次にたどる辺 NextEdgeGoal)
│   │   ├── rewards.py        報酬項
│   │   ├── terminations.py   終了判定 (NaN 対策)
│   │   ├── topology_log.py   位相ラダーの集計ログと Play 時の p-data 表示 (v2-R2)、次にたどる辺のログ (v6)
│   │   ├── archive.py        アーカイブからのリセット・保存・ログ (v3、段ごとの版は v4)
│   │   ├── stages.py         段の観測・遷移の報酬・段が変わったら終了 (v4)
│   │   ├── state_clone.py    環境の状態をそのまま別の環境へ写す (推論時探索)
│   │   └── events.py         起動時イベント (摩擦材 bind, body 順序検算)
│   ├── policies/           独自の方策ネットワーク (v4: 段ごとのエージェント StageGatedActorCritic)
│   ├── topology/           3D 点列 -> 位相 (交差・p-data・結び目判定・R2 除去・BFS 距離・次にたどる辺)。
│   │                       isaaclab を import しないので Sim 無しでテストできる
│   ├── rope_backends/      紐のシミュレーションモデル (バックエンド) を差し替える層 (rope study で追加)
│   ├── rope_model/         剛体鎖のロープの USD 生成と設定 (ROPE_SPECS)
│   └── parallel_gripper/   グリッパーの USD 生成と設定
│
├── versions/         ← 版ごとの「配線」(env_cfg.py・agents/rsl_rl_ppo_cfg.py・README.md)。継承元は上の表
│
├── rope_study/       ← 紐モデルの比較 (同じ MDP (v4) で紐モデルだけを差し替えたタスク群と、計測・評価の道具。打ち切り中)
│
└── demos/            ← RL 環境から import されない独立スクリプト
    ├── generate_assets.py   USD 資産の生成
    ├── rope_catch.py        ロープを掴むだけのデモ
    ├── multi_rope_catch.py  全プリセットを横並びで掴み比べるデモ
    ├── spawn_rope.py        ロープを出現させて GUI で確認するデモ
    ├── parallel_gripper.py  平行グリッパーを開閉させるデモ
    ├── topology_debug.py    台本で操作しながら位相抽出を目視検証
    ├── overhand_script.py   ゲート 1: 台本で固結びを結ぶ (初期アーカイブの作成も)
    └── divergence_bench.py  1 手あたりの発散をフェーズ・段・手の幾何ごとに数える (v5)
```

経緯の記録 (旧 `next_action.md` など) は `docs/history/` にある。通常は読まない。

## デモ / 検証

```bash
# 資産の生成 (最初に一度。作り直すときは --rope <名前> で絞る)
uv run python -u src/isaac_twisted_rl/tasks/knot_tying/demos/generate_assets.py

# ロープを掴むデモ
uv run python -u src/isaac_twisted_rl/tasks/knot_tying/demos/rope_catch.py --rope twist --headless

# 位相抽出が動く物理の上でも正しいかの目視検証 (xy 投影図を出力)
uv run python -u src/isaac_twisted_rl/tasks/knot_tying/demos/topology_debug.py --rope twist --headless

# ゲート 1: 台本で固結びを結ぶ (学習環境をそのまま使い、ハンドだけ台本で動かす)
uv run python -u src/isaac_twisted_rl/tasks/knot_tying/demos/overhand_script.py --task Isaac-Twisted-RL-Knot-v2-R2 --num_envs 128 --jitter 0.25 --headless
#   同じ台本を 6 次元の行動に写して env.step() で再生する
uv run python -u src/isaac_twisted_rl/tasks/knot_tying/demos/overhand_script.py --task Isaac-Twisted-RL-Knot-v2-R2 --num_envs 128 --jitter 0.25 --headless --replay_actions
#   台本の途中状態を段ごとに保存する (v4 の初期アーカイブ)。パスを省くと、v4〜v7 が読む共有の
#   outputs/archive_seed/overhand_{num_links}links.pt を上書きするので、新しいパスを明示する
uv run python -u src/isaac_twisted_rl/tasks/knot_tying/demos/overhand_script.py --task Isaac-Twisted-RL-Knot-v2-R2 --num_envs 128 --jitter 0.25 --headless --dump_archive outputs/archive_seed/<新しい名前>.pt

# 1 手あたりの発散を、どのフェーズで・どんな手で起きたか込みで数える (v5 README の「計測の方法」)
uv run python -u src/isaac_twisted_rl/tasks/knot_tying/demos/divergence_bench.py --task Isaac-Twisted-RL-Knot-v5 --num_envs 1024 --steps 24 --headless --archive_prob 0.0
```

物理を変えたときの確認の手順 (ゲート 1・初期アーカイブ・発散の計測) は skill `physics-check`。

## 構造上の制約

`tests/test_invariants.py` が静的に検査する。理由はそのファイルの docstring にある。

- `tasks/__init__.py` の `import_packages` が、この配下の**パッケージの `__init__.py` をすべて実行する**。そこで `isaaclab.sim` などを
  import すると `list_envs` / `train` が丸ごと落ちる。各 `__init__.py` は docstring だけにしておく。
  - 環境設定は `gym.register` の**文字列 entry point** で参照し、実 import を `gym.make()` まで遅延させる。
  - デモ本体はプレーンモジュールなので自動 import されない (`import_packages` はパッケージの `__init__` しか実行しない)。
- `common/mdp/` が除外されるのは、ブラックリスト `".mdp"` の**部分文字列一致**による (`isaaclab_tasks/utils/importer.py` の `_walk_packages`)。
  **`mdp` というフォルダ名は変えない。** 名前に `utils` や `.mdp` を含むパッケージを新しく作ると、黙って登録から外れる。
- `common/topology/` と `common/state_archive.py` に `isaaclab` / `omni` / `pxr` を import しない (単体テストを Isaac Sim 無しで回すため)。
  `Articulation` に触るのは `common/rope_state.py` と `common/mdp/` だけ。
- `common/mdp/` の項は追加だけ。既存の項を削除・改名しない (過去の版の設定がその項を参照している)。役目を終えた項は
  docstring に `.. deprecated::` を書いて残す (`first_crossing_feature` が既にこの形)。既存の項の**挙動**を変えたいときは、
  新しい名前の項を足して新しい版だけが使う。

開発の約束 (テスト・lint のコマンド、長時間のジョブ、docstring の書き方、ライブラリの版) は、リポジトリのルートの `CLAUDE.md` にある。
