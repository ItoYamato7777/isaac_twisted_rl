# 用語集: 正式名称・命名規則・互換名

- 状態: 現行
- 読むべき状況: 用語の意味が分からないとき・新しい名前を付けるとき・過去の名前を引くとき
- 最終更新: 2026-10-04

この文書の中の `kt/` は `src/isaac_twisted_rl/tasks/knot_tying/` の、`scripts/` は `src/isaac_twisted_rl/scripts/` の略。根拠の `file:line` は 2026-10-04 の作業ツリーで確かめた。

## 目次

1. [用語の表](#1-用語の表)
   - [1.1 課題・ロープ・紐モデル](#11-課題ロープ紐モデル)
   - [1.2 rope study の紐モデル (8 個)](#12-rope-study-の紐モデル-8-個)
   - [1.3 MDP・位相](#13-mdp位相)
   - [1.4 発散の判定](#14-発散の判定)
   - [1.5 評価・ログ・版](#15-評価ログ版)
2. [命名規則](#2-命名規則)
3. [互換名の表](#3-互換名の表)
   - [3.1 タスク ID](#31-タスク-id)
   - [3.2 ログのフォルダ名](#32-ログのフォルダ名)
   - [3.3 run 名と方策のラベル](#33-run-名と方策のラベル)
   - [3.4 使い直された名前](#34-使い直された名前)
   - [3.5 改名・移動されたファイル](#35-改名移動されたファイル)

## 1. 用語の表

列の意味: 「正式名称と意味」は文書で使う名前と、その語が指すもの (複数の意味があれば全部)。「コード上の識別子」は据え置きの名前 (タスク ID・ログ名・Hydra の上書きに出るので変えない)。「注意」は取り違えやすい点。

### 1.1 課題・ロープ・紐モデル

| 用語 | 正式名称と意味 | コード上の識別子 | 注意 |
|---|---|---|---|
| 紐 / ロープ | **ロープ**。新しい文書は「ロープ」に統一する。「紐」は同義 (rope study の文書は「紐」で書いてあり、書き換えない)。 | すべて `rope` (`rope_specs`、`rope_cfg`、`rope_backends` など) | 「紐モデル」「紐のバックエンド」は rope study 由来の固有の語として残す。 |
| 紐結び課題 | このリポジトリの課題 (ロープを固結びにする)。パッケージ名・クラス名・ログ名の接頭辞 | `knot_tying`、`KnotTyingEnvCfg`、`KnotTyingPPORunnerCfg`、ログ名 `knot_tying_vN` | ロープのプリセット `Knot` とは別。どのプリセットでも課題は同じ紐結び課題。 |
| 固結び (overhand knot) | 目標の結び目。開いたロープの overhand knot。位相クラス 3 = 段 3。 | `CLASS_OVERHAND`、`OVERHAND_P_DATA`、`is_overhand`、`distance_to_overhand`、`demos/overhand_script.py`、初期アーカイブ `overhand_{num_links}links.pt` | 「三葉結び」「trefoil」は両端を閉じた結び目を論じるときだけ使う。 |
| `Knot` | 4 つの意味がある。(1) **ロープ `Knot`**: ロープのプリセット。`twist` と同じ材料で 28 リンク・直径 20 mm・全長 1.14 m。v3 以降の学習に使う (2) 紐結び課題 (`knot_tying`) (3) 評価の結果の `knot` = 「固結びに達した」(4) 結び目一般 (`knot_detect`、`knot_graph`) | (1) `"Knot"` (タスク ID)、`"knot"` (`ROPE_SPECS`)、`KnotEnvCfg`、`KNOT_NUM_LINKS` (2) `knot_tying` (3) `eval_chain` の出力 `knot`、`beam_search` の結果 `knot`、v3 の終了項 `knot_tied` | ロープのプリセットには必ず「ロープ」を付ける。`Isaac-Twisted-RL-Knot-v4` の Knot はロープで、結び目ではない。`Knot-v2-R2` (ロープ `Knot` で v2-R2 の MDP) は v2-R2 の学習 (ロープ `Twist`) とは別の環境。 |
| `Twist` | 3 つの意味がある。(1) **ロープ `Twist`**: ロープのプリセット (20 リンク、曲げ剛性は `stiff` の 1/10、捩りを解放)。v0〜v2-R2 の学習に使った (2) 捩り (rotX) の剛性・ロック: 物理量 (3) 「独立したツイスト」= R1 の輪 (位相) | (1) `"Twist"`、`"twist"`、`TwistEnvCfg`、`_TWIST`、`TWIST_YOUNGS_MODULUS` (2) `twist_stiffness`、`twist_damping`、`twist_limit_deg`、`lock_twist` | (2) は「ねじり剛性」、(3) は「R1 ループ」と書き、「ツイスト」と呼ばない。 |
| Twisted 系 | 本プロジェクト =「**Isaac Twisted-RL**」。手法 =「**TWISTED-RL** (Freund ら)」。先行論文 =「**TWISTED** (Sudry ら、CoRL 2023)」。移植元 =「移植元の MuJoCo 実装 (twisted_rl)」 | パッケージ `isaac_twisted_rl`、pyproject の `isaac-twisted-rl`、タスク ID の接頭辞 `Isaac-Twisted-RL-`、`scripts/list_envs.py` の `"Twisted" in task_spec.id` | タスク ID `Isaac-Twisted-RL-Twist-v0` には「Twisted」(プロジェクト) と「Twist」(ロープ) の両方が入る。引用・比較の文では主語を明記する。 |
| ロープのプリセット | `ROPE_SPECS` のキー (小文字 6 種): `simple` / `stiff` / `twist` / `fine` / `knot` / `knot56`。タスク ID では先頭大文字の 5 種 (`Simple` / `Stiff` / `Twist` / `Fine` / `Knot`)。`knot56` はタスク ID に無く、rope study の `Art56` だけが使う | `ROPE_SPECS`、`get_spec`、`--rope`、`_ROPE_PRESETS` (`kt/__init__.py:54`)、`make_preset_cfgs`、`RopeModel.preset`、USD `rope_<preset>.usd` | 「全プリセット」の数は文脈で 5 (タスク ID) か 6 (`ROPE_SPECS`)。小文字 `knot` と大文字 `Knot` は同じもの。根拠: `kt/common/rope_model/rope_specs.py:16-22`、`:579-581`。 |
| 比較用ロープ `Simple` / `Stiff` / `Fine` | 比較用のベースラインのプリセット。20 / 20 / 48 リンク、`fine` だけ直径 8 mm。学習の記録は文書に無い | `_SIMPLE` / `_STIFF` / `_FINE`、`SimpleEnvCfg` など、`DEFAULT_ROPE = "simple"` | タスク ID は登録されている (各 30 本) が、追跡しているファイルからの参照は無い。v4 以降の初期アーカイブ `overhand_20links.pt`・`overhand_48links.pt` は `outputs/archive_seed/` に無い (読めなければ警告して飛ばす)。 |
| 紐モデル | rope study (`kt/rope_study/`) の `RopeModel` 1 個。紐の表現 (バックエンドの kind) + パラメータ + 摩擦の組。8 個 (§1.2) | `RopeModel`、`MODELS` / `MODEL_NAMES` (`kt/rope_study/models.py:97-132`)、`RopeStudyEnvCfg.rope_model` (名前の文字列) | 「紐モデル」は `RopeModel` だけに使う。`common/rope_model/` は紐モデルではない (下の「モデル」)。`MODELS` という名前の別の一覧が `make_eval_jobs.py`・`summarize_eval.py`・`run_physics_suite.sh` にもあり、中身が違う。 |
| 紐のバックエンド (kind) | ロープのシミュレーションの実体と、それを PhysX の Articulation と同じ読み書き口に見せるアダプタ。kind は `articulation` / `articulation_resampled` / `xpbd` / `fem` の 4 つ。観測・行動・位相は「仮想リンク」(弧長で等分した点、既定 28 本) で読む | `common/rope_backends/`、`RopeBackendCfg(kind, num_virtual_links=28, …)` (`kt/common/rope_backends/backend_cfg.py:8-22`)、`cfg.rope_backend`、`RopeStudyEnv`、`XpbdRope` / `FemRope` / `ResampledArticulationRope` | 紐モデル (kind + パラメータ + 摩擦) とは別の語。Hydra の上書きパス `env.rope_backend.xpbd_params.*` が文書のコマンドに出る。pyproject の `build-backend` とは無関係。 |
| 表現 | 2 つの意味がある。(1) **紐の表現**: rope study でロープのシミュレーションの離散化・ソルバ・接触の解き方 (= バックエンドの kind) (2) **位相の表現**: `topology/representation.py` の位相状態のデータ構造 (半辺構造・p-data) | (1) `RopeModel.kind` (2) `representation.py`、`AbstractState` | どちらか分かるように「紐の」「位相の」を付ける。 |
| モデル | 多義。(1) **紐モデル** (`RopeModel`) (2) `common/rope_model/` = **剛体鎖のロープの資産と設定** (USD の生成と ArticulationCfg。rope study より前からある) (3) XPBD の `"model"` = **棒の種類** (`cosserat` / `mass_spring`) (4) `model_<it>.pt` = **チェックポイント** (5) 方策のネットワーク | (1) `RopeModel` (2) `common/rope_model/{rope_cfg,rope_specs,generate_rope_usd}.py` (3) `_xpbd(model=…)`、`bench_physics --model` (4) `model_<it>.pt` | 「モデル」単独では書かない。 |
| 剛体鎖 | D6 関節でつないだカプセル剛体の鎖 (PhysX Articulation) のロープ。v0〜v7 のロープはすべてこれ。rope study の `Art` 系 | `articulation`、`common/rope_model/` | 「chain 評価」の chain とは無関係。 |
| `LowMu` | **LowMu (実物の範囲の摩擦)**。ロープどうし・ロープと床 0.5、指 1.0 にした変種。既定 (2.0 / 2.0 / 4.0) が非現実的に高いので、「Low」は既定に対する相対 | `LOW_MU` (`kt/rope_study/models.py:38-40`)、剛体鎖は `RopeModel.rope_friction` / `ground_friction` / `finger_friction`、XPBD は `friction_ground` / `friction_self` / `friction_finger` | 剛体鎖と XPBD では効かせるパラメータが違う。初出では 3 つの値を書く。 |
| 指の接触 (方式 1 / 2) | XPBD の指とロープの接触の解き方。方式 2 (今の既定) は挟む力の分だけ compliance で残りを剛にし、指の前後の面にも接触を持たせ、1 物理ステップで 2 cm より大きく動いた手を瞬間移動として扱う。方式 1 は全部の面が compliance で、挟む面と底面だけ | `XPBD_FINGER_V2` (`kt/rope_study/models.py:65-79`)、`finger_rigid_outside`・`finger_x_faces`・`finger_x_face_margin`・`hand_teleport_distance` | rope study の文書は方式を「v1 / v2」と書く (例 `docs/experiments/2026-10-02_rope_models.md:483` の「以下、XPBD の学習・評価はすべて v2」は方式 2 の意味で、MDP は v4)。MDP の版 vN と混同しない。 |
| 物理試験の回 | rope study の `run_physics_suite.sh` を回した回。文書では「第 n 回」か日付で呼ぶ | `kt/rope_study/run_physics_suite.sh`、出力 `outputs/rope_study/logs_suite_v{2,3}_gpu*.txt` | rope study の文書は「物理試験 v2 / v3」と書く (`docs/history/rope_study_HANDOFF.md:82`、`:98`)。MDP の版ではない。 |

### 1.2 rope study の紐モデル (8 個)

定義は `kt/rope_study/models.py:97-132`。どの紐モデルも同じロープ (直径 20 mm・全長 1.14 m・線密度 0.25 kg/m・曲げ剛性 EI = 7.85e-4 N m^2・ねじれ剛性 GJ = EI / 2) を表すように材料定数を揃え、表現と摩擦だけを変える (`models.py:1-5`)。名前の規則は「表現 + リンク (区間) 数 + 摩擦の変種」(§2)。

| 名前 | 表現 (バックエンドの kind) | リンク・区間の数 | 摩擦 (ロープどうし / 床 / 指) | 注意 |
|---|---|---|---|---|
| `Art28` | PhysX Articulation の剛体鎖 (`articulation`、プリセット `knot`) | カプセル 28 本 (4 cm) | 既定 2.0 / 2.0 / 4.0 | 基準。ロープ `Knot` と同じロープ。ただし v4 の `Knot` と同じ環境ではない (環境クラス `RopeStudyEnv`、初期アーカイブ `rope_study/Art28.pt`、ログ名が違う) |
| `Art28LowMu` | 同上 (`articulation`、`knot`) | 28 本 | LowMu: 0.5 / 0.5 / 1.0 | 剛体鎖では `rope_friction` などで効かせる |
| `Art56` | PhysX Articulation、仮想リンク 28 本に再標本化 (`articulation_resampled`、プリセット `knot56`) | カプセル 56 本 (2 cm)。1 本おきの接触を切る | 既定 | ログ (`knot_tying_rs_art56`) は無い |
| `XpbdC28` | Warp の XPBD Cosserat 棒 (`xpbd`、`"model": "cosserat"`) | 28 区間 | 既定 (`friction_self` 2.0 / `friction_ground` 2.0 / `friction_finger` 4.0) | 指の接触は方式 2 |
| `XpbdC56` | 同上 | 56 区間 | 既定 | |
| `XpbdMS28` | Warp の XPBD 質点-ばね (SoftGym / FleX 型) (`xpbd`、`"model": "mass_spring"`) | 28 区間 | 既定 | ねじれと正しい曲げ剛性を持たない |
| `XpbdC28LowMu` | XPBD Cosserat 棒 | 28 区間 | LowMu: 0.5 / 0.5 / 1.0 | XPBD では `friction_self` などで効かせる |
| `Fem` | PhysX の FEM 変形体 (四面体) (`fem`) | リンクは無い (観測は仮想リンク 28 本)。四面体の数は未確認 | 床・指は既定。ロープの摩擦は未確認 | 状態アーカイブの容量が 4096 (ほかは 16384)。ログ (`knot_tying_rs_fem`) は無い |

疑似の名前: `XpbdC28EIx0.5` / `XpbdC28EIx2` は評価で曲げ剛性を摂動した先の名前で、`MODELS` には無い。ログ名では小文字 (`knot_tying_rs_xpbdc28`)、run 名・初期アーカイブでは元の大文字小文字 (`rs_XpbdC28_s42`、`outputs/archive_seed/rope_study/XpbdC28.pt`) になる (§3.2)。

### 1.3 MDP・位相

| 用語 | 正式名称と意味 | コード上の識別子 | 注意 |
|---|---|---|---|
| 1 手 (action) | **1 手** = 6 次元の pick&place 1 回 (掴む → 運ぶ → 置く → 離す) = **1 env ステップ** = `decimation` 回の物理ステップ。物理ステップの数はフェーズの尺で変わる (686 / 585 / 759: `kt/common/README.md:66`) | `PickPlaceAction`、`actions_per_episode`、`_PLAY_ACTIONS`、`--max_moves`、`moves_at_solve`、`mean_actions_to_knot`、`beam_search`・`eval_search` の `moves` | 「1 手 = 1 RL ステップ」(`kt/common/README.md:10`)。`knot_graph` の `moves`・`MOVE_*` は 1 手ではなく位相移動。 |
| 位相移動 | 位相状態の上の移動: **R1 移動 / R2 移動 / cross** (と端の cross)。BFS グラフの辺 | `MOVE_R1` / `MOVE_R2` / `MOVE_CROSS` / `MOVE_END_CROSS`、`generate_moves`、`knot_graph` の `moves` | 1 回の位相移動に何手かかるかは決まっていない。 |
| 位相クラス (C0〜C3) | `topology_class` の値。0 = 交差なし、またはラダーから外れた位相 (独立ツイストなどの行き止まり)、1 = 交差 1 つ、2 = 2 交差・gap 2・同符号 (絡み)、3 = 固結び。v1〜v3 の報酬ラダーの「段」はこれ | `topology_class`、`CLASS_TRIVIAL` / `CLASS_SINGLE` / `CLASS_INTERLEAVED2` / `CLASS_OVERHAND`、`NUM_CLASSES` (`kt/common/topology/knot_detect.py:64-77`)、観測項 `knot_class`、ログ `div_cls{k}`、`success_c1` / `success_c2` | R2 除去の前 (v1・v2) と後 (v2-R2 以降) の 2 通りがあり、後なら「R2 除去後の位相クラス」と書く。`topology_class` の docstring は「報酬ラダーの段」と書くが、文書では「位相クラス」と呼ぶ。 |
| 段 (stage) | v4 以降の **`BfsPotential.stage` = round(3Φ)** (0〜3)。**0 = 自明、3 = 固結び** (`STAGE_GOAL = 3`)。BFS の表にある状態では 段 = 3 − d (d = 固結びまでの BFS 距離)。輪 1 つは 1、本物の C2 と cover 後は 2 (`kt/common/topology/knot_graph.py:548-559`) | `BfsPotential.stage`、`STAGE_GOAL`、観測項 `knot_stage` (`knot_stage_one_hot`)、`stage_transition_reward`、終了項 `stage_up` / `stage_down`、`ResetToStage`、`record_stage_archive`、`StageGatedActorCritic`、ログ `success_s{k}`、JSON `max_stage_ge{k}` | 「段」は v4 以降のこの量だけに使う。v1〜v3 の段は「位相クラス」。**「段 = 固結びへの BFS 距離」は向きが逆で誤り**。v4 の段 2 は C2 より広い (cover 後を含む)。`StateArchive(classes=…)` に v4 は段を入れる。 |
| ポテンシャル Φ | Φ = max(1 − d / 3, R2 除去後の位相クラス / 3)。d は R2 除去**前**の p-data で BFS の表を引いた固結びまでの距離。表に無い状態 (届かない・交差 5 個以上) は位相クラスの値になる (`kt/common/topology/knot_graph.py:272-276`、`:515-545`) | `BfsPotential`、`potential_table`、`distance_to_overhand` | d_max (自明な状態の距離) = 3 なので Φ は k/3 の値を取り、段はその 3 倍。 |
| p-data | 交差点をロープの始点側から並べた位相状態の文字列。**1 行形式** (`short_p_data`、例 `1U2+_2O1+`。BFS グラフのノード ID) が既定。複数行のものは「p-data (複数行形式)」 | `AbstractState.p_data` / `short_p_data`、`polyline_to_p_data(short=)`、`batch_p_data`、`p_data_key`、`OVERHAND_P_DATA` | 表記は「p-data」。コードは `p_data`。 |
| R2 除去 | 交差の列から、両方の紐の上で隣り合い、かつ同じ紐が両方とも上になっている 2 交差 (二角形) を、1 組ずつ繰り返し消す処理。**Gauss 符号の上での近似**で、厳密な R2 (二角形の内側が空) は見ていない (`kt/common/topology/knot_detect.py:249-290`) | `reduce_r2`、`_r2_pair_mask`、`extractor.reduced_batch`、パラメータ `reduce_r2`、ログ `r2_trunc` | 「R2」は 3 つを指す: **R2 除去** (この処理)、**R2 移動** (位相移動 `MOVE_R2`)、**v2-R2** (版の名前。v2 の報酬ラダーを R2 除去後の位相で判定する版)。 |
| 次にたどる辺 (next edge) / ρ(g) | v6 以降の観測項 `next_edge`: BFS グラフ上で固結びへ最短の次の辺 (位相移動) を、over / under 区間のリンクマスク・移動の種類・符号などで表したもの (ロープ `Knot` で 64 次元)。**論文の ρ(g) の代わり**として入れた (ρ(g) の中身は論文に書かれていない: `paper/explainer/twisted-rl.html:737`) | 観測項 `next_edge` (`NextEdgeGoal`)、`NextEdgeTable`、`NextEdgeStats`、`NEXT_EDGE_RULES`、`GraphEdge`、状態の複製の部品名 `"next_edge"` | ρ(g) は論文の記号としてだけ使う。`representation.py` の局所変数 `next_edge` (半辺の next ポインタ) とは別物。`representation.Edge` は半辺、`GraphEdge` は高レベルの辺。 |
| 状態アーカイブ | 到達したロープの形状のリングバッファ (v3 以降)。v4 以降は段ごと | `StateArchive` (`kt/common/state_archive.py`)、`ResetFromArchive`・`ResetToStage`・`record_archive`・`record_stage_archive` (`kt/common/mdp/archive.py`)、`archive_capacity`、`--archive_prob`、ログ `Curriculum/archive_log/*` | 保存したファイルは「アーカイブのファイル」(`{log_dir}/archive.pt`、v4 以降)。文書の置き場の名前に archive を使わない (古い文書は `docs/history/`)。 |
| 初期アーカイブ | 台本 (`demos/overhand_script.py --dump_archive`) で作った途中の状態の集まり。状態アーカイブの初期値に読む。版は `outputs/archive_seed/overhand_{num_links}links.pt`、rope study は `outputs/archive_seed/rope_study/<紐モデル>.pt` | `--seed_archive`、`SEED_ARCHIVE` (`kt/versions/v4/env_cfg.py:53`)、`SEED_DIR` (`kt/rope_study/env_cfg.py:43`)、`--seed_frac` | パスはコードに直書き。名前に seed が入るが乱数のシードとは無関係。 |
| シード | 乱数のシード。run 名の `_s42` など | `--seed`、run 名の `_s<シード>` | 「seed」は初期アーカイブ (`archive_seed`) にも使われているので、文書では「シード」と「初期アーカイブ」を書き分ける。 |
| `-RNN` / `-LSTM` (エージェントの変種) | 環境は同じで方策のネットワークだけを変えたタスク。**`-RNN` は GRU** (`rnn_type="gru"`)、`-LSTM` は LSTM。v2・v2-R2・v3 だけにある | `_AGENT_VARIANTS` (`kt/__init__.py:71-87`)、`KnotTyingRNNPPORunnerCfg` / `KnotTyingLSTMPPORunnerCfg`、ログ名 `_rnn` / `_lstm` | 文書では「`-RNN` (GRU)」と必ず併記する。`_rnn` / `_lstm` のログは無い (§3.2)。 |

### 1.4 発散の判定

「発散」の判定は 3 通りある。文書では次の名前で呼び分ける。chain 評価の出力 `diverged` は (1)。

| 名前 | 判定 | 識別子 | 根拠 |
|---|---|---|---|
| (1) **発散** (終了項 `diverged` が立つこと) | ロープの節点の NaN、または下の表の閾値を超えたこと。報酬・ログは `diverged_now` でこの項を呼び直して同じ基準を使う | 関数 `rope_diverged`、終了項名 `diverged`、`diverged_now`、TensorBoard `Episode_Termination/diverged`、評価 JSON の `diverged` | `kt/common/mdp/terminations.py:29-60`、`:68` |
| (2) **NaN** | ロープの節点が有限でないことだけ。`action_failed`・`TopologyStats` (`div_cls{k}`) は `diverged_term` を渡さないとき、`diverged_by_class` は常にこの判定 | `action_failed(diverged_term=None)`、`diverged_by_class`、`topology_log` の `diverged_term=None` | `kt/common/mdp/rewards.py:401-405`、`:427-458`、`kt/common/mdp/topology_log.py:117`、`:254` |
| (3) **spike** (`divergence_bench` の途中の暴れ) | 手の途中でリンク速度が 20 m/s を超えた (または NaN になった) が、手の終わりには (1) が立たなかったもの。発散とは数えない | `kt/demos/divergence_bench.py` の `spike` | `kt/demos/divergence_bench.py:24-30` |

終了項 `diverged` の閾値 (版ごと。高さ・水平距離は env 原点から測る):

| 版 | NaN | 高さ | 水平距離 | リンク速度 (手の終わり) | 根拠 |
|---|---|---|---|---|---|
| v0〜v2 | あり | 5 m 超 | 見ない | 見ない | `kt/common/env_cfg_base.py:112` (`rope_diverged` の既定) |
| v2-R2〜v7・rope study | あり | 5 m 超 | 3 m 超 | 20 m/s 超 | `kt/versions/v2_r2/env_cfg.py:48-52`、`:197-200`。v3 (`TerminationsCfg(V2R2TerminationsCfg)`)・v4 (`V3TerminationsCfg`)・v5〜v7 と rope study (v4 の設定を継承) はそのまま引き継ぐ |

注意: v2-R2 以降は報酬・ログの項にも `diverged_term="diverged"` を渡すので (1) で数える (`kt/versions/v2_r2/env_cfg.py:166`、`:184`)。v2 以前の `div_cls` 相当の数字 (`diverged_by_class`) は (2) なので、版をまたぐ発散率の比較では定義がずれる。

### 1.5 評価・ログ・版

| 用語 | 正式名称と意味 | コード上の識別子 | 注意 |
|---|---|---|---|
| 版 (vN) | **MDP の版**だけを指す: v0・v1・v2・v2-R2・v3・v4・v5・v6・v7。1 版 = `kt/versions/vN/` の 1 フォルダ (`v2-R2` は `v2_r2/`) | `_VERSIONS` (`kt/__init__.py:40`)、`_version_package`、`KnotTyingV{N}EnvCfg`、ログ名 `knot_tying_vN` | XPBD の指の接触は「方式 1 / 2」、物理試験は「第 n 回」、パッケージの版は「パッケージ版 1.2.0」(`pyproject.toml:3`) と呼び、vN を使わない。rope study は版ではなく紐モデルの軸 (タスク ID の末尾 `-v4` は MDP が v4 という意味)。`kt/versions/__init__.py` の docstring の一覧は v6 で止まっている (v7 は存在する)。版の番号は使い直さない (§3.4)。 |
| タスク ID | Gym に登録した環境の名前。規則は §3.1。版 150 本 + rope study 16 本 = 166 本 | `gym.register` (`kt/__init__.py:97-119`、`kt/rope_study/__init__.py:18-28`) | プリセットはタスク ID にしか現れない (ログ名・run 名・チェックポイントに無い)。 |
| Play タスク | タスク ID に `-Play` の付いた環境。中身は版で違う: **v0〜v2-R2** = 推論の様子を GUI で見る環境 (環境 9 個、描画を 8 物理ステップごと。v2-R2 は毎手 p-data を表示)、**v3** = それに加えて常にまっすぐなロープから始める、**v4 以降と rope study** = **chain 評価の環境** (chain モード、18 手) | `{Preset}PlayEnvCfg`、`{Model}PlayEnvCfg`、`play_post_init`、`_PLAY_NUM_ENVS`、`_PLAY_RENDER_STEPS` (`kt/common/rope_task_cfg.py:145-161`、`:221-227`)、`_PLAY_ACTIONS` (`kt/versions/v4/env_cfg.py:213-226`) | v4 以降は「Play タスク (chain 評価用)」と書く。v3 までと v4 以降の Play の数字を比べない。詳細は [`system/evaluation.md`](system/evaluation.md) §3。 |
| chain 評価 | `eval_chain.py` の評価だけを指す固有名。Play タスクでまっすぐなロープから 18 手以内に固結びに達した割合 (各環境の最初の 1 エピソード、決定論的方策)。定義と標準の条件は [`system/evaluation.md`](system/evaluation.md) §1 | `scripts/rsl_rl/eval_chain.py`、出力 `eval_chain_it{N}.json`、キー `knot` / `diverged` / `time_out` / `max_stage_ge{k}` / `mean_actions_to_knot` | chain は次のものにも使われているので区別する: **chain モード** (v4 以降の Play の終了条件。段をつなぎ、固結び・発散・18 手でだけ終わる: `stage_changed(chain=True)`)、**chain の見積り** (`Curriculum/stage_log/chain_est` = 段ごとの成功率の積。学習ログの粗い見積りで chain 評価ではない: `kt/common/mdp/archive.py:400-420`)、**剛体鎖** (Art 系のロープ)、**学習キュー** (`kt/rope_study/run_train_chain.sh` = 学習ジョブを 1 本ずつ順に回すスクリプト)。新しく chain の付く名前を作らない。 |
| ゲート 1 | **ゲート 1 (台本の固結び試験)**: 学習の前に、台本 (`kt/demos/overhand_script.py`) で固結びが結べるかを確かめる関門。ロープ `Knot` で 2026-09-27 に通過 (`docs/history/next_action.md:465-470`)。rope study では同じ台本を紐モデルごとに回す試験の名前でもある (遅い尺 `gate1/`、学習の尺 `gate1_fast/`) | `kt/rope_study/run_physics_suite.sh` の `gate1` / `gate1_fast` | 番号付きだがゲート 2 以降は無い。将来の関門には別の名前を付ける。`StageGatedActorCritic` の gating、LSTM のゲートとは無関係。手順は skill `physics-check`。 |
| ログ名 (experiment_name) | `logs/rsl_rl/<ログ名>/`。エージェントの設定の `experiment_name` で決まる。版ごとに `knot_tying_vN` (変種は `_rnn` / `_lstm`)、rope study は `knot_tying_rs_<紐モデルの小文字>`。詳細は [`system/logs.md`](system/logs.md) §1 | `experiment_name` (`kt/versions/*/agents/rsl_rl_ppo_cfg.py`、`kt/rope_study/agents/rsl_rl_ppo_cfg.py:21`)、`log_root_path` | ロープのプリセットはログ名に入らない (同じ版の 5 プリセットが同じフォルダを共有する)。変えるなら `--experiment_name <名前>` か Hydra の `agent.experiment_name=<名前>` (`scripts/rsl_rl/cli_args.py`)。 |
| run | ログ名の下の 1 回の学習のフォルダ `<日時>[_<run 名>]` | `run_name`、`--run_name`、`--load_run` (`scripts/rsl_rl/train.py:143-149`) | `train.py` のコメントの「experiment name」は run のフォルダ名のこと (Isaac Lab 由来)。 |
| run 名 | `--run_name` で付ける run のフォルダ名の接尾辞。規則は §2 | `--run_name` | 過去の run 名は §3.3。 |
| チェックポイント | run のフォルダの `model_<it>.pt` (`agent.save_interval` ごと)。rsl-rl の方策・最適化器の状態と iteration 数だけを持ち、タスク ID・ログ名・プリセットは入らない | `model_<it>.pt`、`--checkpoint` | 「モデル」と呼ばない。最後の番号は run ごとに `ls` で確かめる ([`system/logs.md`](system/logs.md) §3)。 |
| 実験 | 研究上の実験 (仮説・条件・結果の 1 組)。記録は `docs/experiments/` | — | ログ名 (`experiment_name`) や run を「実験」と呼ばない。 |

## 2. 命名規則

新しく名前を付けるときの規則。既存の名前 (タスク ID・ログ名・コードの識別子) は変えない。

| 対象 | 規則 | 根拠 |
|---|---|---|
| 紐モデル | 軸の順に連結: 表現 (`Art` / `XpbdC` / `XpbdMS` / `Fem`) + リンク数 + 摩擦の変種 (`LowMu`)。剛性などの軸を足すときは、末尾に変種として足す。既存の 8 個はこの規則にほぼ合う | C-9、C-20、BRIEF §4 Phase 2 の 3 |
| 版 | 整数を 1 つずつ増やす。番号は使い直さない。派生 (`-R2` のような) は作らず、次の番号にする | C-5、E-2 |
| run 名 | `--run_name <略>_s<シード>`。ロープ `Knot` 以外のプリセットで学習するときは、略にプリセット名を入れる (ログ名にプリセットが無いため: C-2) | E-1、C-2 |
| 実験記録 | `docs/experiments/YYYY-MM-DD_<題>.md` (日付は学習を始めた日) | §2 |
| 設計判断 | `docs/decisions/NNNN-<題>.md` | §2 |

根拠の列の記号 (C-n・E-n・§2・BRIEF) は refactor の調査と提案 (`refactor/audit/C.md`、`refactor/audit/E.md`、`refactor/proposal.md` §3.2) の番号。

## 3. 互換名の表

過去の名前と、タスク ID・ログ・ファイル名として外部に残る名前。ここ以外の文書では、§1 の正式名称を使う。

### 3.1 タスク ID

| 対象 | 規則 | 数 | 登録 |
|---|---|---|---|
| 版のタスク | `Isaac-Twisted-RL-{プリセット}[-Play]-vN[-RNN\|-LSTM]`。プリセットは `Twist` / `Simple` / `Stiff` / `Fine` / `Knot`、vN は `v0`〜`v7` と `v2-R2`。`-RNN` (GRU) / `-LSTM` は v2・v2-R2・v3 だけ | 150 本 (版 9 × プリセット 5 × 学習/Play 2 = 90 に、v2・v2-R2・v3 の `-RNN` 30 本と `-LSTM` 30 本) | [`kt/__init__.py`](../src/isaac_twisted_rl/tasks/knot_tying/__init__.py) (`:97-119`) |
| rope study のタスク | `Isaac-Twisted-RL-RS-<紐モデル>[-Play]-v4`。紐モデルは §1.2 の 8 個 (元の大文字小文字)。末尾の `-v4` は MDP が v4 という意味 | 16 本 (紐モデル 8 × 学習/Play 2) | [`kt/rope_study/__init__.py`](../src/isaac_twisted_rl/tasks/knot_tying/rope_study/__init__.py) (`:18-28`) |

合計 166 本、うち `-Play` は 83 本。`-Play` は版の前に入る (`…-Knot-Play-v4`、`…-RS-XpbdC28-Play-v4`)。変種の接尾辞は版の後ろに付く (`…-Twist-v2-RNN`)。全一覧は持たない。登録モジュールを `gymnasium` のスタブの下で実行すれば出せる (スクリプトは `refactor/audit/C.md` の付録 B)。

表記の対応 (層ごとに書き方が違う):

| タスク ID | パッケージ・フォルダ | クラス名 | ログ名 |
|---|---|---|---|
| `v2-R2` | `versions/v2_r2/` | `KnotTyingV2R2EnvCfg` | `knot_tying_v2_r2` |
| `-RNN` / `-LSTM` | — | `KnotTyingRNNPPORunnerCfg` / `KnotTyingLSTMPPORunnerCfg` | `_rnn` / `_lstm` |
| `RS-XpbdC28` | — | `XpbdC28EnvCfg` | `knot_tying_rs_xpbdc28` (run 名は `rs_XpbdC28_s42`) |
| `Knot` (プリセット) | `ROPE_SPECS["knot"]`、`rope_knot.usd` | `KnotEnvCfg` | (入らない) |

### 3.2 ログのフォルダ名

`logs/rsl_rl/<ログ名>/` とコードの名前の対応。`ls logs/rsl_rl` (2026-10-03 時点、refactor の調査 C §4.2) との突き合わせ。

| ログ名 | 版・変種・紐モデル | タスク ID | `logs/rsl_rl/` に |
|---|---|---|---|
| `knot_tying_v0` | v0 | `*-v0` | あり |
| `knot_tying_v1` | v1 | `*-v1` | あり |
| `knot_tying_v2` | v2 | `*-v2` | あり |
| `knot_tying_v2_rnn` / `knot_tying_v2_lstm` | v2 の `-RNN` (GRU) / `-LSTM` | `*-v2-RNN` / `*-v2-LSTM` | なし |
| `knot_tying_v2_r2` | v2-R2 | `*-v2-R2` | あり |
| `knot_tying_v2_r2_rnn` / `knot_tying_v2_r2_lstm` | v2-R2 の変種 | `*-v2-R2-RNN` / `*-v2-R2-LSTM` | なし |
| `knot_tying_v3` | v3 | `*-v3` | あり |
| `knot_tying_v3_rnn` / `knot_tying_v3_lstm` | v3 の変種 | `*-v3-RNN` / `*-v3-LSTM` | なし |
| `knot_tying_v4` | v4 | `*-v4` | あり |
| `knot_tying_v4_speed` | v4 の学習速度の比較 (コードに無い名前) | 未確認 (Hydra の上書き `agent.experiment_name` で名前を変えた) | あり |
| `knot_tying_v5` | v5 | `*-v5` | あり |
| `knot_tying_v6` | v6 | `*-v6` | あり |
| `knot_tying_v6_ablation` | v6 の切り分け (コードに無い名前)。旧 v7 の run と、今の v7 と同じ設定の参照 run `abl_depB_s42` を含む | (後から手で移した) | あり |
| `knot_tying_v7` | v7 | `*-v7` | なし (v7 は未学習) |
| `knot_tying_rs_art28` | `Art28` | `RS-Art28[-Play]-v4` | あり |
| `knot_tying_rs_art28lowmu` | `Art28LowMu` | `RS-Art28LowMu[-Play]-v4` | あり |
| `knot_tying_rs_art56` | `Art56` | `RS-Art56[-Play]-v4` | なし |
| `knot_tying_rs_xpbdc28` | `XpbdC28` | `RS-XpbdC28[-Play]-v4` | あり |
| `knot_tying_rs_xpbdc56` | `XpbdC56` | `RS-XpbdC56[-Play]-v4` | あり |
| `knot_tying_rs_xpbdms28` | `XpbdMS28` | `RS-XpbdMS28[-Play]-v4` | あり |
| `knot_tying_rs_xpbdc28lowmu` | `XpbdC28LowMu` | `RS-XpbdC28LowMu[-Play]-v4` | あり |
| `knot_tying_rs_fem` | `Fem` | `RS-Fem[-Play]-v4` | なし |

- 版のログ名は 5 プリセットで共用で、どのプリセットの run かはフォルダ名から分からない (run の `params/env.yaml` で確かめる)。
- コードに無い 2 つ (`knot_tying_v4_speed`・`knot_tying_v6_ablation`) の出どころは [`system/logs.md`](system/logs.md) §1。
- `logs/rsl_rl/eval_search_lane.sh` はフォルダではなくファイル。

### 3.3 run 名と方策のラベル

| 名前 | 何か |
|---|---|
| (接尾辞なし) | v0・v1・v2・v2-R2 と v3 の 2 本の run |
| `v3_newbuf_10it` | v3 の run |
| `3h_compare` | v4 の run |
| `overnight_s42`・`overnight_s43` | v4〜v6 の overnight の run (`overnight_s43` は v6 だけ。`docs/experiments/2026-10-01_v4-v6_overnight.md`) |
| `A_it16_T12`・`B_it16_T4`・`C_it8_T4` | `knot_tying_v4_speed` の 3 本 |
| `abl_relA_s42`・`abl_depB_s42`・`abl_toP_s42`・`abl_sw2_s42` | `knot_tying_v6_ablation` の 4 本。`abl_toP_s42` は旧 v7 (§3.4)、`abl_depB_s42` は今の v7 と同じ設定 |
| `rs_<紐モデル>_s42`・`rs_<紐モデル>_s43` | rope study の学習 (`kt/rope_study/run_train_chain.sh:25`)。紐モデル名は元の大文字小文字のまま (フォルダのログ名は小文字) |
| `v4Knot` | rope study の評価での、v4 のロープ `Knot` の方策のラベル (`kt/rope_study/summarize_eval.py:24`、`outputs/rope_study/eval/v4Knot/`) |
| `XpbdC28_s43`・`XpbdC56_s43` | rope study の評価での方策のラベル (同 `:24`) |
| `v4=…`・`v6s42=…` | 探索ありの評価の集計での方策のラベル (`scripts/rsl_rl/summarize_search.py:4`) |

### 3.4 使い直された名前

| 名前 | 過去の意味 | 今の意味 | どこに残るか |
|---|---|---|---|
| v7 | **旧 v7** = v6 + 時間切れの罰則 −0.2 (報酬項 `stage_time_out`: 段が変わらないまま 6 手で終わった手に罰)。1 本だけ学習し (`abl_toP_s42`)、効果が見えず削除した。コードはどのコミットにも残っていない | **v7** = v6 のロープのめり込み解消速度の上限を v4 の既定 (0.96 m/s) に戻した版 (`kt/versions/v7/`) | run `logs/rsl_rl/knot_tying_v6_ablation/2026-10-02_12-34-00_abl_toP_s42` (`params/agent.yaml` のログ名は `knot_tying_v7`)、`kt/versions/v7/README.md:77-81`、`docs/experiments/2026-10-01_v4-v6_overnight.md:226`、`:288` |
| v1 / v2 (rope study) | XPBD の指の接触の方式 1 / 2 | MDP の版 v1 / v2 とは無関係 (§1.1) | `kt/rope_study/models.py:65-79` (`XPBD_FINGER_V2`)、`docs/experiments/2026-10-02_rope_models.md:483` |
| 物理試験 v2 / v3 | rope study の物理試験の回 | MDP の版とは無関係 (§1.1) | `docs/history/rope_study_HANDOFF.md:82`、`:98` |
| 段 | v1〜v3 の文書では報酬ラダーの段 (= 位相クラス) | v4 以降の `BfsPotential.stage` だけ (§1.3) | `kt/versions/v1/README.md`、`kt/versions/v2/README.md`、`topology_class` の docstring |

### 3.5 改名・移動されたファイル

古い文書・コメント・会話の記録に残る旧パスを引くための表。

| 旧 | 新 | commit |
|---|---|---|
| `paper/Report.md` | [`paper/Report_Paper.md`](../paper/Report_Paper.md) | `dbf9a97` |
| `kt/next_action.md` | [`docs/history/next_action.md`](history/next_action.md) (節番号は同じ) | `62b251b` |
| `kt/rope_study/HANDOFF.md` | [`docs/history/rope_study_HANDOFF.md`](history/rope_study_HANDOFF.md) | `62b251b` |
| `kt/versions/v1/diff_to_TwistedRL.md` | [`docs/history/v1_diff_to_TwistedRL.md`](history/v1_diff_to_TwistedRL.md) | `62b251b` |
| `kt/versions/v3/HTML/*.html` | `docs/history/v3_html/` (`v3-implementation.html`、`v3-vs-papers.html`) | `62b251b` |
| `kt/versions/v3/Result_report_v3.md` | [`docs/experiments/2026-09-28_v3_main.md`](experiments/2026-09-28_v3_main.md) | `62b251b` |
| `kt/versions/v6/Result_Report_v6.md` | [`docs/experiments/2026-10-01_v4-v6_overnight.md`](experiments/2026-10-01_v4-v6_overnight.md) | `62b251b` |
| `kt/rope_study/Report_rope_models.md` | [`docs/experiments/2026-10-02_rope_models.md`](experiments/2026-10-02_rope_models.md) | `62b251b` |
| `paper/Report_Search.md` | [`docs/experiments/2026-10-03_inference_search.md`](experiments/2026-10-03_inference_search.md) | `62b251b` |

`scripts/rsl_rl/eval_chain.py:8` が参照する `eval_compare.py` は、追跡しているファイルにも git の履歴 (`git log --all -- '*eval_compare*'`) にも無い。
