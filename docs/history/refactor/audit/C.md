# C. 用語の監査

対象: `/home/itoyama/work/isaac_twisted_rl` の HEAD `6f41ef1` (ブランチ `refactor/claude-foundation`)。
`git ls-files` の 167 ファイルのうち `uv.lock` と PDF 3 本を除く 163 ファイルを数えた。`.venv` は追跡外なので対象外。
`logs/`・`outputs/` は `ls` だけで、中のファイルは開いていない。Isaac Sim は起動していない。

## 0. 要点 (重大な順。詳細は §6)

| 番号 | 一言で | 直し方 |
|---|---|---|
| C-1 | 「段」に 2 つの定義がある (v1〜v3 は位相クラス、v4 以降は `BfsPotential.stage`)。さらに文書は段を「固結びへの BFS 距離」と書くが、実際は向きが逆 (段 = 3 − 距離) | 文書で直せる |
| C-2 | ログ名 (`experiment_name`) がプリセットを含まない。同じ版の 5 プリセットが `logs/rsl_rl/knot_tying_vN/` を共有する。層ごとに表記の規則も違う (`v2-R2`↔`v2_r2`、`-RNN`↔`_rnn`、`RS-XpbdC28`↔`rs_xpbdc28`) | 識別子の変更が要る (ログに影響) |
| C-3 | 「Knot」がロープのプリセット・課題の名前 (`knot_tying`)・目標 (固結び)・評価結果のキー (`knot`) の 4 つを指す。`Isaac-Twisted-RL-Knot-v4` の Knot は「28 リンクのロープ」 | 文書で直せる (識別子はタスク ID なので据え置き) |
| C-4 | 「Twist」がロープのプリセット・捩り (物理量)・R1 の輪 (「独立ツイスト」) の 3 つを指す。「Twisted」は本リポジトリ・TWISTED-RL 論文・TWISTED 論文・移植元の MuJoCo 実装にまたがる | 文書で直せる |
| C-5 | `vN` が MDP の版のほかに、XPBD の指の接触 (v1/v2)・物理試験の回 (v2/v3) にも使われる。`versions/__init__.py` の一覧に v7 が無い | 文書で直せる (定数の改名は任意) |
| C-6 | `-Play` の意味が v4 で変わる (描画を見る環境 → chain 評価の環境) | 文書で直せる |
| C-7 | 「発散」の判定が 3 通りある (終了項 `diverged`、報酬側の NaN だけ、`divergence_bench` の spike)。終了項の閾値も版で変わる | 文書で直せる |
| C-8 | `--experiment_name` は定義だけで効かない。コードに無いログ名 (`knot_tying_v4_speed`・`knot_tying_v6_ablation`) が実在し、v7 の参照 run は `v6_ablation` の下にある | 文書 + コード |

ほかに C-9〜C-23 (§6.1)。識別子を変えるとタスク ID・ログ・初期アーカイブのファイル名に響くものは §6.3。

## 1. 方法

* 出現の数は次の関数で数えた。値は「一致した行の数 / ファイルの数」。1 行に 2 回出ても 1 と数える。
  コマンドと結果の全部は付録 A (`K1`〜`K140`)。表の中では `[K12]` のように参照する。

  ```bash
  cnt() {  # $1 = git grep の照合オプション (-F / -wF / -iF / -P)、$2 = パターン
    git -C /home/itoyama/work/isaac_twisted_rl grep -I -c $1 -e "$2" -- . ':!uv.lock' ':!*.pdf' \
      | awk -F: '{s+=$NF; n++} END {print s" 行 / "n" ファイル"}'
  }
  ```

* 「コード」は `.py` と `.sh`、「md」は `.md`、「html」は `paper/explainer/*.html` と `versions/v3/HTML/*.html`。
* タスク ID は登録モジュールを `gymnasium` のスタブの下で実行して抜き出した (Isaac Sim・isaaclab を import しない)。スクリプトは付録 B。
* 根拠の `path:line` はリポジトリのルートからの相対パス。`src/isaac_twisted_rl/tasks/knot_tying/` は長いので、表の中では `kt/` と略す
  (例: `kt/common/mdp/stages.py:1`)。`src/isaac_twisted_rl/scripts/` は `scripts/` と略す。

## 2. 用語の表

列: 現在の表記 (揺れ) / 意味 (文脈) / コード上の識別子 / 出現の数 / 推奨する正式名称 / 互換名として残す必要 / 根拠。

### 2.1 名前の層 (プリセット・プロジェクト・紐モデル)

| 用語 | 現在の表記 (揺れ) | 意味 (どの文脈でどの意味か) | コード上の識別子 | 出現の数 | 推奨する正式名称 | 互換名として残す必要 | 根拠 |
|---|---|---|---|---|---|---|---|
| Twist | `Twist` (タスク ID・クラス名)、`twist` (rope_specs)、twist (捩り)、「ツイスト」 | (1) ロープのプリセット: 20 リンク、ヤング率 1e5 Pa、捩り解放。v0〜v2-R2 の学習用 (2) 捩り (rotX) の剛性・ロック: 物理量 (3) 「独立したツイスト」= R1 の輪 (位相) | `_ROPE_PRESETS` の `"Twist"`、`TwistEnvCfg` / `TwistPlayEnvCfg`、`_TWIST`、`TWIST_YOUNGS_MODULUS`、`twist_stiffness` / `twist_damping` / `twist_limit_deg` / `lock_twist` | `Twist` 80 行/28 件 [K1]、`-Twist-` 52/21 [K2]、`` `twist` `` 41/9 [K3]、`twist_stiffness` 21/8 [K4]、ツイスト 9/6 [K5] | プリセットは「ロープ `Twist`」(文書で必ず「ロープ」を付ける)。物理量は「ねじり剛性」(識別子 `twist_stiffness` は据え置き)。位相は「R1 ループ」(ツイストと呼ばない) | 要: タスク ID 30 本 (`Isaac-Twisted-RL-Twist-*`)、`--rope twist`、`outputs/overhand_script/Twist-v2-R2/` (未追跡) | `kt/__init__.py:54`、`kt/common/rope_model/rope_specs.py:20`、`:443-452`、`:223`、`:334`、`kt/common/topology/knot_detect.py:67`、`kt/versions/v1/README.md:22` |
| Twisted 系 (プロジェクト・論文・移植元) | Twisted-RL、TWISTED-RL、TWISTED、twisted_rl、itoyama_twisted_rl、isaac_twisted_rl、isaac-twisted-rl、Isaac Twisted-RL、TwistedRL、`Isaac-Twisted-RL-` | (1) 本リポジトリ (README は「Twisted-RL の課題を実装」と書く) (2) Freund らの手法 TWISTED-RL (3) Sudry らの TWISTED (CoRL 2023) (4) 移植元の MuJoCo 実装 (twisted_rl / itoyama_twisted_rl) (5) Python パッケージ (6) タスク ID の接頭辞 | パッケージ `isaac_twisted_rl`、pyproject の `name = "isaac-twisted-rl"`、ID 接頭辞、`diff_to_TwistedRL.md`、`list_envs.py` の `"Twisted" in task_spec.id` | Twisted-RL 210/66 [K6]、TWISTED-RL 192/39 [K7]、TWISTED (-RL なし) 146/19 [K8]、isaac_twisted_rl 184/59 [K9]、twisted_rl (isaac_ なし) 17/12 [K10]、TwistedRL 7/6 [K11] | 本プロジェクト =「Isaac Twisted-RL」、手法 =「TWISTED-RL (Freund ら)」、先行論文 =「TWISTED (Sudry ら)」、移植元 =「移植元の MuJoCo 実装 (twisted_rl)」 | 要: タスク ID の接頭辞 (166 本すべて)、パッケージ名、`list_envs` の絞り込み | `README.md:8`、`README.md:135`、`paper/Report_Paper.md:7`、`kt/__init__.py:3`、`kt/common/topology/representation.py:3`、`scripts/list_envs.py:47`、`pyproject.toml:2-4` |
| Knot | `Knot` (ID・クラス)、`knot` (rope_specs)、knot_tying / KnotTying、`knot` (結果のキー)、`knot_tied` | (1) ロープのプリセット: `twist` と同じ材料で 28 リンク 1.14 m。v3〜v7 の学習用。**結び目ではない** (2) 課題のパッケージ名・ログ名・クラス名の接頭辞 (全プリセット共通) (3) 固結びに達したこと: `eval_chain` の出力 `knot`、ビーム探索の結果 `knot`、v3 の終了項 `knot_tied` (4) 結び目一般: `knot_detect`、`knot_graph` | `"Knot"`、`KnotEnvCfg` / `KnotPlayEnvCfg`、`_KNOT`、`KNOT_NUM_LINKS`、`KnotTyingEnvCfg` / `KnotTyingPPORunnerCfg`、`knot_tying_vN`、`OUTCOME_NAMES` の `"knot"`、`knot_tied` | `Knot` 238/57 [K12]、`-Knot-` 106/32 [K13]、`` `knot` `` 38/11 [K14]、knot_tying 272/85 [K15]、KnotTying 111/33 [K16] | プリセットは「ロープ `Knot` (28 リンク)」。課題は「紐結び課題 (knot_tying)」。目標は「固結び」 | 要: タスク ID 30 本、方策のラベル `v4Knot` (`summarize_eval.py:24`、`outputs/rope_study/eval/v4Knot/`)、`outputs/overhand_script/Knot-v2-R2/` | `kt/__init__.py:54`、`kt/common/rope_model/rope_specs.py:22`、`:501-510`、`scripts/rsl_rl/eval_chain.py:16`、`kt/common/beam_search.py:51`、`kt/versions/v3/env_cfg.py:12`、`:140` |
| 固結び / overhand | 固結び、overhand knot、overhand、OVERHAND、三葉結び、trefoil、結び目 | 目標の結び目 (開いた紐の overhand knot。両端を閉じれば三葉結び)。位相クラス 3 = 段 3 | `CLASS_OVERHAND`、`OVERHAND_P_DATA`、`is_overhand`、`distance_to_overhand`、`overhand_script.py`、`overhand_{num_links}links.pt` | 固結び 502/68 [K17]、overhand 185/42 [K18]、三葉 26/10 [K19]、trefoil 52/6 (ほぼテスト) [K20]、結び目 120/38 [K21] | 「固結び (overhand knot)」。三葉結び・trefoil は閉じた結び目を論じるときだけ | 要: 初期アーカイブのファイル名 `outputs/archive_seed/overhand_28links.pt` (コードがパスを組む) | `kt/common/topology/knot_detect.py:1`、`:75`、`kt/versions/v1/README.md:17`、`kt/versions/v4/env_cfg.py:53`、`kt/demos/overhand_script.py:117` |
| Simple / Stiff / Fine | `Simple`/`Stiff`/`Fine` (ID・クラス)、`simple`/`stiff`/`fine` (rope_specs) | 比較用ベースラインのロープ (20 / 20 / 48 リンク、`fine` は直径 8 mm)。学習の記録は文書に無い | `_SIMPLE` / `_STIFF` / `_FINE`、`SimpleEnvCfg` など、`DEFAULT_ROPE = "simple"` | `Simple` 12/2 [K22]、`Stiff` 14/4 [K23]、`Fine` 12/2 [K24]、`` `simple` `` 26/9 [K25]、`` `stiff` `` 12/5 [K26]、`` `fine` `` 33/11 [K27] | 「比較用ロープ `Simple` / `Stiff` / `Fine`」 | タスク ID 90 本は登録されているが、追跡ファイルで ID として参照されるのは 0 本 (§3.2)。ログ名にプリセットが無いので、ログに残っているかは区別できない (未確認) | `kt/__init__.py:52-54`、`kt/common/rope_model/rope_specs.py:18-21`、`:584`、`kt/common/rope_task_cfg.py:129-140` |
| Art28 系・Xpbd 系・Fem (rope_study のモデル名) | `Art28`、`Art28LowMu`、`Art56`、`XpbdC28`、`XpbdC56`、`XpbdMS28`、`XpbdC28LowMu`、`Fem`。ログ名では小文字 (`art28` など)。疑似モデル名 `XpbdC28EIx0.5` / `XpbdC28EIx2` (摂動先)。方策のラベル `v4Knot`・`XpbdC28_s43` | 紐モデル = 表現 (kind) + 区間数 + 摩擦。`Art` = PhysX Articulation (剛体鎖)、`C` = Cosserat 棒、`MS` = 質点-ばね、数字 = 区間 (リンク) 数、`Fem` = PhysX の FEM 変形体。`Art28` = 既存の `Knot` | `MODELS` / `MODEL_NAMES` / `RopeModel.name`、`{name}EnvCfg` / `{name}PlayEnvCfg`、`{name}PPORunnerCfg`、`knot_tying_rs_{name.lower()}`、run 名 `rs_{M}_s42`、`SEED_DIR/{name}.pt`。モデル名の一覧が `make_eval_jobs.py`・`summarize_eval.py`・`run_physics_suite.sh` にも別に書いてある | `Art28` 123/11 [K29]、`Art28LowMu` 68/7 [K30]、`Art56` 49/8 [K31]、`XpbdC28` 92/12 [K32]、`XpbdC56` 47/7 [K33]、`XpbdMS28` 41/8 [K34]、`XpbdC28LowMu` 55/7 [K35]、`Fem` 18/5 [K36]、小文字 `art28` 0 [K37]、剛体鎖 98/14 [K38] | 据え置き (正式名 = `models.py` の名前)。命名規則「表現 + 区間数 + 変種」を文書に書く | 要: タスク ID 16 本、ログ 6 フォルダ、run 名、初期アーカイブ `outputs/archive_seed/rope_study/<M>.pt` と `<M>_knots.pt`、`outputs/rope_study/{gate1,gate1_fast,physics,eval}/<M>/` | `kt/rope_study/models.py:7-16`、`:97-132`、`kt/rope_study/__init__.py:21`、`kt/rope_study/agents/rsl_rl_ppo_cfg.py:21`、`kt/rope_study/env_cfg.py:43`、`:202`、`kt/rope_study/run_train_chain.sh:25`、`kt/rope_study/make_eval_jobs.py:30-35`、`kt/rope_study/summarize_eval.py:22-24`、`kt/rope_study/run_physics_suite.sh:14` |
| LowMu | `LowMu` (モデル名の接尾辞)、`LOW_MU`、低摩擦、「摩擦を実物の範囲」 | 摩擦を実物の範囲 (紐 0.5・床 0.5・指 1.0) にした変種。既定 (2.0 / 2.0 / 4.0) が非現実的に高いので、「Low」は既定に対する相対。剛体鎖は `rope_friction` 等、XPBD は `friction_self` 等で、効かせ方が違う | `LOW_MU`、`RopeModel.rope_friction` / `ground_friction` / `finger_friction`、xpbd の `friction_ground` / `friction_self` / `friction_finger` | `LowMu` (部分一致) 116/7 [K39]、単語 `LowMu` 6/2 [K40]、`LOW_MU` 9/1 [K41]、低摩擦 13/4 [K42]、実物の範囲 14/3 [K43] | 「LowMu (実物の範囲の摩擦)」。初出で 3 つの値を書く | 要 (モデル名の一部。上の行と同じ) | `kt/rope_study/models.py:10`、`:38-40`、`:101-109`、`:113-126`、`kt/rope_study/Report_rope_models.md:166` |
| preset (プリセット) | プリセット、preset、`--rope`、「ロープ 5 種」「5 つのロープのプリセット」「4 プリセット」 | (1) `ROPE_SPECS` の名前 (小文字 6 種: simple / stiff / twist / fine / knot / knot56) (2) タスク ID 上の先頭大文字 5 種 (`knot56` を除く) (3) `RopeModel.preset` (剛体鎖の rope_specs 名)。文書の数は 4 / 5 / 6 と揺れる | `ROPE_SPECS`、`get_spec`、`add_rope_arg` (`--rope`)、`_ROPE_PRESETS`、`_PRESET_DOCS`、`make_preset_cfgs` (`preset.capitalize()`)、`RopeModel.preset` | プリセット 139/47 [K105]、`preset` 49/19 [K106]、`--rope ` 19/13 [K107]、`knot56` 25/6 [K28] | 「ロープのプリセット」= `ROPE_SPECS` のキー (6 種、小文字)。タスク ID の表記は先頭大文字 (5 種) と書き分ける | 要: タスク ID、`--rope` の値、USD のファイル名 `rope_<preset>.usd` | `kt/common/rope_model/rope_specs.py:5-6`、`:579-584`、`kt/__init__.py:54`、`kt/common/rope_task_cfg.py:193-194`、`kt/rope_study/models.py:52-53`、`kt/common/rope_model/__init__.py:15`、`kt/README.md:7`、`:47`、`kt/demos/generate_assets.py:36` |
| backend / rope_backends | バックエンド、backend、rope_backends、rope_backend、kind、「紐の表現」。別義: pyproject の `build-backend` | 紐のシミュレーションの実体と、それを Articulation と同じ読み書き口に見せるアダプタ。kind = articulation / articulation_resampled / xpbd / fem | `common/rope_backends/`、`RopeBackendCfg(kind, num_virtual_links, …)`、`cfg.rope_backend`、`RopeStudyEnv`、`XpbdRope` / `FemRope` / `ResampledArticulationRope` | `backend` 75/15 [K108]、rope_backends 54/10 [K109]、rope_backend 66/12 [K110]、バックエンド 11/8 [K111] | 「紐のバックエンド (kind)」。紐モデル (= kind + パラメータ + 摩擦) とは別の語として使う | 要: Hydra の上書きパス `env.rope_backend.xpbd_params.*` が文書のコマンドに出る | `kt/common/rope_backends/__init__.py:1-13`、`kt/common/rope_backends/backend_cfg.py:8-22`、`kt/rope_study/models.py:49-50`、`:78`、`kt/rope_study/env_cfg.py:70`、`pyproject.toml:41` |
| rope_model / 「モデル」 | モデル、紐モデル、ロープモデル、rope_model、RopeModel、`"model"`、`model_<it>.pt` | (1) rope_study の紐モデル (`RopeModel`、8 種) (2) `common/rope_model/` = 剛体鎖ロープの USD 生成と ArticulationCfg (rope_study より前からある) (3) `RopeStudyEnvCfg.rope_model` = モデル名の文字列 (4) XPBD の `"model"` = cosserat / mass_spring (5) チェックポイント `model_<it>.pt` (6) 方策のネットワーク | `RopeModel`、`MODELS`、`common/rope_model/{rope_cfg,rope_specs,generate_rope_usd}.py`、`RopeStudyEnvCfg.rope_model`、`_xpbd(model=…)`、`bench_physics --model` | モデル 371/44 [K112]、紐モデル 40/27 [K113]、rope_model 62/25 [K114]、RopeModel 13/2 [K115]、`model_` 60/22 [K116]、ロープモデル 1/1 [K117] | 「紐モデル」は `RopeModel` だけに使う。`common/rope_model/` は「剛体鎖ロープの資産」。XPBD の `"model"` は「棒の種類」 | パッケージ名は import パスだけ (タスク ID・ログ名に出ない)。`rope_model` フィールドがログの `params/env.yaml` に出るかは未確認 (開いていない) | `kt/rope_study/models.py:44-62`、`:84`、`kt/rope_study/env_cfg.py:71`、`:120`、`kt/common/rope_model/rope_cfg.py:62`、`kt/rope_study/__init__.py:1` |

### 2.2 MDP と位相の用語

| 用語 | 現在の表記 (揺れ) | 意味 (どの文脈でどの意味か) | コード上の識別子 | 出現の数 | 推奨する正式名称 | 互換名として残す必要 | 根拠 |
|---|---|---|---|---|---|---|---|
| stage / 段 | 段、stage、段 0〜3、「段 (固結びへの BFS 距離)」、「交差数に当たる」、「報酬ラダーの段」、3 段ラダー、最下段・中段・最上段 | (A) v1〜v3: 報酬ラダーの段 = 位相クラス (`topology_class` の 0〜3) (B) v4 以降: `BfsPotential.stage` = round(3Φ) (0〜3、3 = 固結び)。Φ = max(1 − d/3, 位相クラス/3) なので、BFS 表にある状態では段 = 3 − d (d = 固結びまでの距離)。**「段 (固結びへの BFS 距離)」は向きが逆** (C) v2 の「3 段の疎報酬ラダー」= 疎報酬の 3 項 | `BfsPotential.stage`、`STAGE_GOAL`、`current_stage`、観測項 `knot_stage` (`knot_stage_one_hot`)、`stage_transition(_reward)`、終了項 `stage_up` / `stage_down` (`stage_changed`)、`ResetToStage`、`record_stage_archive`、`StageGatedActorCritic(num_stages, stage_obs_dims)`、ログ `success_s{k}`・`Curriculum/stage_log/chain_est`、JSON `max_stage_ge{k}` | 段 (手段・段階を除く) 868/85 [K44]、`段 0〜3` 208/39 [K45]、単語 `stage` 150/24 [K46]、STAGE_GOAL 14/5 [K47]、BFS 距離 44/16 [K48]、ラダー 116/33 [K49] | 「段 (stage)」は (B) だけに使う。定義は「段 = round(3Φ)、0 = 自明 〜 3 = 固結び」と書き、「BFS 距離」とは書かない。(A) は「位相クラス」 | 識別子 (観測項名・終了項名・ログのキー・JSON のキー・`class_name` 文字列) は Hydra の上書きと過去ログの比較に現れるので据え置き | `kt/common/mdp/stages.py:1`、`:10`、`kt/common/topology/knot_graph.py:273-276`、`:548-559`、`kt/common/topology/knot_detect.py:233-234`、`kt/common/policies/stage_gated.py:6`、`kt/versions/v1/README.md:37-38`、`kt/versions/v2/README.md:83-85`、`kt/README.md:42` |
| topology class (位相クラス) | 位相クラス、class、クラス、C0〜C3 (C1・C2)、class 2、ラダーの段、knot_class、cls | `topology_class` の値。0 = 交差なし・ラダー外、1 = 交差 1、2 = 絡み 2 交差、3 = 固結び。R2 除去前 (v1・v2) と後 (v2-R2 以降) の 2 通り。v4 の「段 2」は「本物の C2 と cover 後」を含むので C2 より広い | `topology_class`、`CLASS_TRIVIAL` / `CLASS_SINGLE` / `CLASS_INTERLEAVED2` / `CLASS_OVERHAND`、`NUM_CLASSES`、観測項 `knot_class` (`knot_class_one_hot`)、`class_at_action_start`、`ladder_class_reached`、`diverged_by_class`、ログ `div_cls{k}`、`success_c1` / `success_c2`、`StateArchive(classes=…)` (v4 では段を入れる) | 位相クラス 27/11 [K50]、topology_class 47/16 [K51]、`CLASS_` 42/10 [K52]、`knot_class` 24/9 [K53]、`knot_stage` 12/7 [K54]、`C0〜C3` 198/28 (論文の記号を含む) [K55]、英語の「topology class」0 [K56] | 「位相クラス (C0〜C3)」。R2 除去後なら「R2 除去後の位相クラス」 | 観測項名 `knot_class`・ログのキー `div_cls*` は据え置き。`StateArchive` の引数名 `classes` は保存ファイルに出ない (`entries` の整数キーだけ) ので改名してよい | `kt/common/topology/knot_detect.py:64-77`、`kt/common/mdp/observations.py:195`、`kt/common/state_archive.py:35-38`、`:110-120`、`kt/common/mdp/archive.py:269`、`kt/versions/v4/README.md:49`、`kt/next_action.md:439`、`kt/common/topology/knot_graph.py:552` |
| p-data | p-data (文書)、p_data (コード)、short p-data / short_p_data、P_DATA。`pdata` は HTML の CSS クラス・アンカーだけ | 交差点を紐の始点側から並べた位相状態の文字列。複数行形式 (`p_data`) と 1 行形式 (`short_p_data`、例 `1U2+_2O1+`、グラフのノード ID)。文書の例はほぼ 1 行形式 | `AbstractState.p_data` / `short_p_data`、`polyline_to_p_data(short=)`、`batch_p_data`、`p_data_from_batch`、`p_data_key`、`OVERHAND_P_DATA`、`batch_keys` | p-data 297/56 [K57]、p_data 197/17 [K58]、short_p_data 51/6 [K59]、「p data」0 [K60] | 「p-data」。既定は 1 行形式とし、複数行は「p-data (複数行形式)」と書く | 不要 (タスク ID・ログ名に出ない) | `kt/common/topology/representation.py:21-30`、`:177-184`、`kt/common/topology/state_2_topology.py:184-196`、`kt/common/topology/knot_graph.py:473` |
| archive (状態アーカイブ) | アーカイブ、状態アーカイブ、到達状態アーカイブ、段ごとのアーカイブ、初期アーカイブ、seed archive、archive_seed、archive.pt、archive_log | (1) 到達したロープ形状のリングバッファ (v3 以降) (2) その保存ファイル `{log_dir}/archive.pt` (v4 以降) (3) 台本で作った初期アーカイブ `outputs/archive_seed/` (4) TensorBoard のキー `Curriculum/archive_log/*`。**文書の archive フォルダは現時点で無い** (`git ls-files` にも全履歴にも `archive/` が無い) | `common/state_archive.py` (`StateArchive`)、`common/mdp/archive.py` (`ResetFromArchive`、`ResetToStage`、`record_archive`、`record_stage_archive`)、`archive_capacity`、`--archive_prob`、`--seed_archive`、`SEED_ARCHIVE`、`SEED_DIR`、`--dump_archive` | アーカイブ 278/43 [K61]、単語 archive 46/16 [K62]、state_archive 17/13 [K63]、StateArchive 24/9 [K64]、archive_seed 30/12 [K65]、初期アーカイブ 54/15 [K66]、archive.pt 9/5 [K67] | 「状態アーカイブ」。ファイルは「アーカイブのファイル (`archive.pt`)」と「初期アーカイブ (`outputs/archive_seed/`)」。古い文書の置き場に `archive` の名前は使わない (例: `docs/history/`) | 要: `archive.pt` の名前、`outputs/archive_seed/` のパス (コードに直書き) | `kt/common/state_archive.py:1`、`:35`、`kt/common/mdp/archive.py:57`、`:231`、`:468`、`kt/versions/v4/env_cfg.py:53`、`:119`、`kt/rope_study/env_cfg.py:43`、`kt/versions/v3/README.md:113-118` |
| 発散 / divergence | 発散、diverged、divergence、rope_diverged、NaN、spike、「吹き飛んだ」 | (1) 終了項 `diverged` (= `rope_diverged`): NaN または高さ 5 m 超 (v0〜v2)。v2-R2 以降は水平 3 m・リンク速度 20 m/s を足す (2) 報酬側 (`action_failed`・`diverged_by_class`・`topology_log`): 既定は NaN だけ。`diverged_term` を渡したときだけ (1) と同じ (3) `divergence_bench` の spike: 手の途中で 20 m/s を超えたが、手の終わりには落ち着いたもの | `rope_diverged`、終了項名 `diverged`、`diverged_now`、`action_failed(diverged_term=)`、`diverged_by_class`、ログ `div_cls{k}`、`divergence_bench.py`、結果の `"diverged"` | 発散 567/69 [K79]、diverged 225/38 [K80]、divergence 25/15 [K81]、NaN 76/33 [K82] | 「発散 (終了項 `diverged` が立つこと)」。NaN だけの判定は「NaN」と呼び分ける | 要: 終了項名 `diverged` (TensorBoard の `Episode_Termination/diverged`、評価 JSON の `diverged`、`eval_search.py:214` の `get_term("diverged")`) | `kt/common/mdp/terminations.py:29-48`、`kt/common/env_cfg_base.py:112`、`kt/versions/v2_r2/env_cfg.py:48-51`、`:199`、`kt/common/mdp/rewards.py:401-405`、`:454-458`、`kt/demos/divergence_bench.py:26-30` |
| 1 手 / action / RL step | 1 手、手、手数、action、行動、move(s)、RL ステップ、env ステップ、env.step、step。別義: 物理ステップ、Step 1〜5 (開発の段階)、MOVE_R1/R2/CROSS (位相移動) | 1 手 = 6 次元の pick&place 1 回 = 1 env ステップ = decimation 回の物理ステップ (尺で変わる: 686 / 585 / 759)。`beam_search`・`eval_search` の `moves` も 1 手。`knot_graph` の `moves`・`MOVE_*` は位相の移動 (R1 / R2 / cross) で別物 | `PickPlaceAction`、`actions_per_episode`、`_PLAY_ACTIONS`、`--max_moves`、`moves_at_solve`、`mean_actions_to_knot`、`MOVE_R1` / `MOVE_R2` / `MOVE_CROSS`、`generate_moves` | 1 手 305/49 [K83]、手数 110/26 [K84]、RL ステップ 8/6 [K85]、env ステップ 27/14 [K86]、物理ステップ 108/37 [K87]、単語 moves 36/11 [K88]、max_moves 15/5 [K89]、`MOVE_` 63/6 [K90] | 「1 手 (action)」=「1 env ステップ」。位相の移動は「位相移動 (R1 / R2 / cross)」 | 要: CLI の `--max_moves`、JSON の `moves_per_minute_per_problem` (文書のコマンドに出る) | `kt/common/README.md:10-17`、`:34-39`、`:66`、`kt/common/topology/knot_detect.py:52`、`kt/common/beam_search.py:303`、`scripts/rsl_rl/eval_search.py:64`、`kt/common/topology/knot_graph.py:138-140`、`:201` |
| next edge / ρ(g) | 次にたどる辺、次にたどる位相の辺、次にたどる高レベルの辺、最短の次の辺、next_edge、NextEdge*、ρ(g)、next-edge。別義: `representation.py` の局所変数 `next_edge` (半辺の next ポインタ)、`Edge` (半辺) と `GraphEdge` (高レベルの辺) | v6 の観測: BFS グラフ上で固結びへ最短の次の辺 (位相移動) を、over / under 区間のリンクマスク・移動の種類・符号などで表したもの (Knot で 64 次元)。論文の ρ(g) に「当たる」とされるが、ρ(g) の中身は論文に書かれていない | 観測項 `next_edge` (`NextEdgeGoal`)、`NextEdgeTable`、`NextEdgeStats`、`NEXT_EDGE_RULES` (`RULES`)、`GraphEdge`、`state_clone.PARTS` の `"next_edge"` | next_edge 71/16 [K118]、NextEdge 34/12 [K119]、次にたどる 28/14 [K120]、最短の次の辺 5/4 [K121]、ρ(g) 14/7 [K122]、半辺 31/7 [K123]、単語 Edge 12/3 [K124] | 「次にたどる辺 (next edge)」。ρ(g) は論文の記号としてだけ使い、「v6 の next edge は ρ(g) の代わり」と書く | 要: 観測項名 `next_edge` (Hydra の `env.observations.policy.next_edge.params.rule`)、状態の複製の部品名 `"next_edge"` | `kt/common/topology/next_edge.py:3`、`:58`、`kt/common/mdp/observations.py:214-217`、`kt/common/topology/knot_graph.py:282-299`、`kt/common/topology/representation.py:96`、`:193`、`kt/versions/v6/env_cfg.py:12`、`:20`、`paper/explainer/twisted-rl.html:737`、`kt/common/mdp/state_clone.py:45` |
| R2 除去 | R2 除去、R2 除去後、R2 後、Reidemeister-II reduction、ライデマイスター II、reduce_r2、reduced_batch。別義: R2 = 位相移動 (`MOVE_R2`)、v2-R2 = 版の名前 | 交差の列から「同じ紐が両方とも上」の隣り合う 2 交差 (二角形) を繰り返し消す。Gauss 符号の上での近似で、厳密な R2 (二角形の内側が空) は見ていない | `reduce_r2`、`_r2_pair_mask`、`extractor.reduced_batch`、`ladder_batch(reduce_r2)`、パラメータ `reduce_r2`、`cls_r2`、ログ `r2_trunc` | R2 除去 143/32 [K125]、reduce_r2 70/16 [K126]、Reidemeister 28/12 [K127]、`_r2` 141/29 [K128] | 「R2 除去」。定義に「Gauss 符号の上での近似」と書く。位相移動の R2 は「R2 移動」 | 要: 版の名前 `v2-R2` (タスク ID・ログ名 `knot_tying_v2_r2`) | `kt/common/topology/knot_detect.py:249-290`、`README.md:61`、`kt/common/topology/knot_graph.py:139`、`kt/versions/v2_r2/README.md:61`、`kt/__init__.py:40` |

### 2.3 評価・ログ・版の用語

| 用語 | 現在の表記 (揺れ) | 意味 (どの文脈でどの意味か) | コード上の識別子 | 出現の数 | 推奨する正式名称 | 互換名として残す必要 | 根拠 |
|---|---|---|---|---|---|---|---|
| chain 評価 | chain 評価、chain eval、chain evaluation、chain モード、chain mode、eval_chain、chain_est、chain gap。別義: 剛体鎖、run_train_chain.sh | (1) Play 環境でまっすぐなロープから 18 手以内に固結びに達した割合 (各環境の最初の 1 エピソード、決定論的方策) (2) `chain_est` = 段ごとの成功率の積による粗い見積り (学習ログ) (3) chain モード = Play の終了条件 (段をつないで、固結び・発散・18 手でだけ終わる) (4) 別義: 剛体鎖 (Art 系の紐) (5) 別義: `run_train_chain.sh` = 学習ジョブを順に回すキュー | `eval_chain.py`、`stage_changed(chain=)`、`chain_est`、`eval_chain_it{N}.json`、`run_train_chain.sh` | chain 評価 114/20 [K68]、chain eval 6/3 [K69]、eval_chain 34/17 [K70]、chain_est 25/12 [K71]、chain モード 18/11 [K72]、train_chain 8/4 [K73]、剛体鎖 98/14 [K38] | 「chain 評価」(固有名として据え置き)。`chain_est` は「chain の見積り (学習ログ)」と明記。`run_train_chain.sh` は「学習キュー」と呼ぶ | 要: `eval_chain.py` の名前、run フォルダの `eval_chain_it{N}.json` | `scripts/rsl_rl/eval_chain.py:6-19`、`kt/next_action.md:34`、`kt/common/mdp/stages.py:163-184`、`kt/common/mdp/archive.py:402-420`、`kt/versions/v4/env_cfg.py:218-223`、`kt/rope_study/run_train_chain.sh:2`、`:28`、`kt/common/rope_backends/__init__.py:3` |
| gate / ゲート 1 | ゲート 1、ゲート1、ゲート (4.8.9 の見出し)、gate1、gate1_fast。別義: StageGated (gating)、LSTM のゲート | (1) 学習の前に「台本で固結びが結べるか」を確かめる関門 (`demos/overhand_script.py`)。2026-09-27 に `Knot` で通過。番号付きだが 2 以降は無い (2) rope_study では同じ台本をモデルごとに回す試験 (出力 `gate1/`・`gate1_fast/`) (3) 別義: `StageGatedActorCritic` の hard gating (4) 別義: LSTM のゲート | `run_physics_suite.sh` の `gate1` / `gate1_fast`、`StageGatedActorCritic`、`stage_gated.py` | ゲート 1 137/31 [K74]、ゲート1 1/1 [K75]、gate1 20/3 [K76]、gate (aggregate・gated を除く) 23/5 [K77]、StageGated 38/18 [K78] | 「ゲート 1 (台本の固結び試験)」と初出で定義する。将来の関門には別の名前を付ける | 未追跡の出力フォルダ (`outputs/rope_study/gate1/` など) だけ | `kt/next_action.md:465-470`、`:484`、`kt/README.md:150`、`kt/common/rope_model/rope_specs.py:467`、`kt/rope_study/run_physics_suite.sh:24-38`、`kt/common/policies/stage_gated.py:11`、`kt/versions/v2/agents/rsl_rl_ppo_cfg.py:219` |
| Play (タスク ID の `-Play`) | `-Play`、Play 環境、Play 系、`*PlayEnvCfg`、chain モード | v0〜v2: 推論を見る用 (環境 9、描画 8 物理ステップごと)。v2-R2: + p-data の表示。v3: + 常にまっすぐなロープから。v4 以降と rope_study: chain 評価の環境 (18 手、段をつなぐ、終了条件が違う) | `{Preset}PlayEnvCfg`、`{Model}PlayEnvCfg`、`play_post_init`、`_PLAY_NUM_ENVS`、`_PLAY_RENDER_STEPS`、`_PLAY_ACTIONS` | `-Play` 75/36 [K129]、Play 環境 20/10 [K130] | 「Play タスク」。v4 以降は「Play タスク (chain 評価用)」と書く | 要: タスク ID 83 本 (166 本の半分)、評価コマンド | `kt/common/rope_task_cfg.py:145`、`:161`、`:221-227`、`kt/versions/v2_r2/env_cfg.py:220-222`、`kt/versions/v3/env_cfg.py:205-208`、`kt/versions/v4/env_cfg.py:214-223`、`kt/rope_study/__init__.py:4` |
| version / vN | v0〜v7、v2-R2、v2_r2、V2R2、バージョン、版、MDP の版、紐の版。別義: 指の接触 v1 / v2 (`XPBD_FINGER_V2`)、物理試験 v2 / v3、pyproject `version = "1.2.0"`、CITATION の 1.2.0 | (1) MDP の版 (`versions/vN/`、タスク ID の末尾、ログ名) (2) XPBD の指の接触の改訂 (3) rope_study の物理試験の回 (4) パッケージの版。`Report_rope_models.md:483` の「以下、XPBD の学習・評価はすべて v2」は (2) の意味で、MDP は v4 | `_VERSIONS`、`_version_package` (`v2-R2` → `v2_r2`)、`versions/vN/`、`KnotTyingV{N}EnvCfg`、`knot_tying_vN`、`XPBD_FINGER_V2` | `v0〜v7` 1534/116 [K91]、v2-R2 208/41 [K92]、v2_r2 30/14 [K93]、V2R2 27/6 [K94]、バージョン 117/24 [K95]、版 173/62 [K96]、指 v 19/2 [K97]、物理試験 v 7/2 [K98]、XPBD_FINGER_V2 5/3 [K99]、1.2.0 3/2 [K100] | 「版 vN」は MDP の版だけに使う。指の接触は日付か内容で呼ぶ (例:「指の接触 (挟む面以外を剛)」)。物理試験は「第 n 回」。パッケージは「パッケージ版 1.2.0」 | 要: vN はタスク ID・ログ名・パッケージ名 | `kt/__init__.py:40`、`:43-49`、`kt/versions/__init__.py:6-13` (v7 が無い)、`kt/rope_study/models.py:65-79`、`kt/rope_study/HANDOFF.md:82`、`:98`、`kt/rope_study/Report_rope_models.md:483`、`pyproject.toml:3`、`kt/rope_study/__init__.py:8` |
| experiment / experiment_name / run | experiment_name、experiment、実験、run、run_name、切り分け、ablation | (1) `experiment_name` = `logs/rsl_rl/<名前>` (版ごと) (2) run = その下の `<日時>[_<run_name>]` (3) `train.py:146` の「experiment name」は run フォルダ名 (Isaac Lab 由来。Ray Tune が読むので変えない) (4) 文書の「実験」= 研究上の実験 (Discussion の E1〜E6) | `experiment_name`、`run_name`、`--experiment_name` (定義だけで適用されない)、`--load_run`、`log_root_path` | experiment_name 33/17 [K101]、experiment 40/17 [K102]、実験 41/8 [K103]、run_name 12/7 [K104] | 「ログ名 (experiment_name)」と「run」。「実験」は研究計画にだけ使う | 要: ログ名と run 名 (§4) | `scripts/rsl_rl/train.py:140-149`、`scripts/rsl_rl/cli_args.py:25-28`、`:70-91`、`kt/next_action.md:580`、`kt/README.md:186` |
| -RNN / -LSTM (変種) | `-RNN` (ID)、`_rnn` (ログ名)、`KnotTyingRNNPPORunnerCfg`、「再帰版 (GRU)」、GRU | `-RNN` は GRU (`rnn_type="gru"`)。`-LSTM` は LSTM。v2・v2-R2・v3 だけ | `_AGENT_VARIANTS`、`KnotTyingRNNPPORunnerCfg` / `KnotTyingLSTMPPORunnerCfg` | `-RNN` 19/12 [K131]、GRU 47/20 [K132] | 文書では「`-RNN` (GRU)」と必ず併記 | 要: タスク ID 18 本、ログ名 `*_rnn` / `*_lstm` (ただしログは実在しない) | `kt/__init__.py:71-87`、`kt/README.md:69-70`、`kt/versions/v2/agents/rsl_rl_ppo_cfg.py:155`、`:199`、`:208` |

### 2.4 ほかに見つけた衝突・揺れ (追加)

| 用語 | 現在の表記 (揺れ) | 意味 (どの文脈でどの意味か) | コード上の識別子 | 出現の数 | 推奨する正式名称 | 互換名として残す必要 | 根拠 |
|---|---|---|---|---|---|---|---|
| 紐 / ロープ | 紐、ロープ、縄 (1 行) | 同じもの。rope_study 以降の文書は「紐」、それより前は「ロープ」が多い (`Report_rope_models.md` は紐 124 行・ロープ 1 行) | 識別子はすべて `rope` | 紐 417/64 (コード 203・md 199) [K133]、ロープ 763/86 (コード 469・md 189) [K134] | どちらかに統一する。識別子 `rope` と多数派に合わせるなら「ロープ」 | 不要 | `kt/rope_study/models.py:3`、`kt/common/rope_task_cfg.py:1`、`kt/rope_study/Report_rope_models.md:166` |
| seed | seed、シード、s42 / s43、`--seed`。別義: 初期アーカイブ (`archive_seed`、`SEED_ARCHIVE`、`SEED_DIR`、`--seed_archive`、`--seed_frac`) | (1) 乱数のシード (2) 台本で作った初期アーカイブ | 上の識別子 | 単語 seed 195/41 [K135]、シード 49/16 [K136]、archive_seed 30/12 [K65] | 乱数は「シード」、アーカイブは「初期アーカイブ」 | 要: `outputs/archive_seed/` のパス (コードに直書き)、run 名の `_s42` | `kt/versions/v4/env_cfg.py:53`、`kt/rope_study/env_cfg.py:43`、`kt/demos/divergence_bench.py:15`、`kt/rope_study/run_train_chain.sh:25`、`README.md:84` |
| 表現 / representation | 表現、representation | (1) rope_study: 紐のシミュレーションの表現 (離散化・ソルバ・接触の解き方) (2) `topology/representation.py`: 位相状態のデータ構造 (半辺構造・p-data) | `representation.py`、`AbstractState` | 表現 91/30 [K137]、representation 10/6 [K138] | (1) は「紐の表現 (バックエンド)」、(2) は「位相の表現」 | 不要 | `kt/rope_study/models.py:4`、`kt/common/rope_backends/env.py:28`、`kt/common/topology/representation.py:1` |
| 古い文書名 | `paper/Report.md`、Report §7、`eval_compare.py` | `paper/Report.md` は commit `dbf9a97` で `paper/Report_Paper.md` に改名済み。`eval_compare.py` は追跡ファイルに無い | — | `paper/Report.md` 26/23 [K139]、eval_compare 2/2 [K140] | `paper/Report_Paper.md` に直す | 不要 | commit `dbf9a97`、`README.md:118`、`kt/README.md:27`、`kt/common/topology/knot_detect.py:91`、`scripts/rsl_rl/eval_chain.py:8` |
| 学習に使うプリセットの古い記述 | 「学習に使うのは `Twist` (v0〜v2-R2) と `Knot` (v3)」 | v4〜v7 も `Knot` で学習しているが、コメントが v3 で止まっている | `_ROPE_PRESETS` のコメント、`_PRESET_DOCS["knot"]` | — | 「v3 以降は `Knot`」 | 不要 | `kt/__init__.py:52-53`、`kt/common/rope_task_cfg.py:8`、`:142`、`kt/README.md:48` |

## 3. 登録されている Gym のタスク ID (機械的に抜き出した全一覧)

### 3.1 要約

* 登録は 2 か所だけ。合計 **166 本**。
  * `kt/__init__.py:103` と `:112` (二重ループ: 版 9 × プリセット 5 × 変種 × 学習/Play) → 150 本
  * `kt/rope_study/__init__.py:20` (モデル 8 × 学習/Play) → 16 本
* entry_point は 2 種類。`isaaclab.envs:ManagerBasedRLEnv` (150 本) と `…common.rope_backends.env:RopeStudyEnv` (16 本)。
* 環境設定のクラスはファクトリで作る。版は `make_preset_cfgs` (`kt/common/rope_task_cfg.py:165`、名前は `preset.capitalize()` + `EnvCfg`)、
  rope_study は `_make_cfgs` (`kt/rope_study/env_cfg.py:206-232`、`{name}EnvCfg`)。どちらも `class TwistEnvCfg` のような定義はソースに無い (名前は docstring・コメントにだけ出る: `kt/common/rope_task_cfg.py:172`、`kt/versions/v1/env_cfg.py:148`)。
* PPO 設定は、版ごとに `KnotTyingPPORunnerCfg` (+ v2・v2-R2・v3 は `KnotTyingRNNPPORunnerCfg` / `KnotTyingLSTMPPORunnerCfg`)。rope_study は `{name}PPORunnerCfg` (ファクトリ、`kt/rope_study/agents/rsl_rl_ppo_cfg.py:14-28`)。
* 版ごとの本数: v0・v1 は各 10、v2・v2-R2・v3 は各 30、v4〜v7 は各 10、RS は 16。`kt/README.md:56-63` の数と一致する (RS は README の表に無い)。
* **追跡ファイルで ID として参照されているのは 166 本のうち 28 本だけ** (表の最後の列)。`Simple` / `Stiff` / `Fine` の 90 本は 0 本。

### 3.2 抜き出しの方法

```bash
# gymnasium をスタブにして登録モジュール 2 つだけを exec し、gym.register の呼び出しを記録する (付録 B)
.venv/bin/python <scratchpad>/list_tasks.py > tasks.tsv        # 166 行
# 各 ID が追跡ファイルに何行出るか (後ろにハイフンや英数字が続く ID とは区別する)
cut -f1 tasks.tsv | while read id; do
  git grep -I -c -P "${id}(?![-\w])" -- . ':!uv.lock' | awk -F: '{s+=$NF} END {print s+0, "'"$id"'"}'
done
```

### 3.3 全一覧

列の注: entry_point・env_cfg・PPO 設定は、先頭の `isaac_twisted_rl.tasks.knot_tying.` を省いた。「ログ名」は PPO 設定の `experiment_name` (= `logs/rsl_rl/<ログ名>/`)。「参照」は追跡ファイルの中で ID がそのまま出る行の数。

| タスク ID | 登録 (file:line) | entry_point | env_cfg | PPO 設定 | ログ名 | 参照 |
|---|---|---|---|---|---|---|
| `Isaac-Twisted-RL-Twist-v0` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:103` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v0.env_cfg:TwistEnvCfg` | `versions.v0.agents.rsl_rl_ppo_cfg:KnotTyingPPORunnerCfg` | `knot_tying_v0` | 7 |
| `Isaac-Twisted-RL-Twist-Play-v0` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:112` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v0.env_cfg:TwistPlayEnvCfg` | `versions.v0.agents.rsl_rl_ppo_cfg:KnotTyingPPORunnerCfg` | `knot_tying_v0` | 5 |
| `Isaac-Twisted-RL-Simple-v0` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:103` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v0.env_cfg:SimpleEnvCfg` | `versions.v0.agents.rsl_rl_ppo_cfg:KnotTyingPPORunnerCfg` | `knot_tying_v0` | 0 |
| `Isaac-Twisted-RL-Simple-Play-v0` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:112` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v0.env_cfg:SimplePlayEnvCfg` | `versions.v0.agents.rsl_rl_ppo_cfg:KnotTyingPPORunnerCfg` | `knot_tying_v0` | 0 |
| `Isaac-Twisted-RL-Stiff-v0` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:103` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v0.env_cfg:StiffEnvCfg` | `versions.v0.agents.rsl_rl_ppo_cfg:KnotTyingPPORunnerCfg` | `knot_tying_v0` | 0 |
| `Isaac-Twisted-RL-Stiff-Play-v0` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:112` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v0.env_cfg:StiffPlayEnvCfg` | `versions.v0.agents.rsl_rl_ppo_cfg:KnotTyingPPORunnerCfg` | `knot_tying_v0` | 0 |
| `Isaac-Twisted-RL-Fine-v0` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:103` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v0.env_cfg:FineEnvCfg` | `versions.v0.agents.rsl_rl_ppo_cfg:KnotTyingPPORunnerCfg` | `knot_tying_v0` | 0 |
| `Isaac-Twisted-RL-Fine-Play-v0` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:112` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v0.env_cfg:FinePlayEnvCfg` | `versions.v0.agents.rsl_rl_ppo_cfg:KnotTyingPPORunnerCfg` | `knot_tying_v0` | 0 |
| `Isaac-Twisted-RL-Knot-v0` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:103` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v0.env_cfg:KnotEnvCfg` | `versions.v0.agents.rsl_rl_ppo_cfg:KnotTyingPPORunnerCfg` | `knot_tying_v0` | 0 |
| `Isaac-Twisted-RL-Knot-Play-v0` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:112` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v0.env_cfg:KnotPlayEnvCfg` | `versions.v0.agents.rsl_rl_ppo_cfg:KnotTyingPPORunnerCfg` | `knot_tying_v0` | 0 |
| `Isaac-Twisted-RL-Twist-v1` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:103` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v1.env_cfg:TwistEnvCfg` | `versions.v1.agents.rsl_rl_ppo_cfg:KnotTyingPPORunnerCfg` | `knot_tying_v1` | 7 |
| `Isaac-Twisted-RL-Twist-Play-v1` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:112` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v1.env_cfg:TwistPlayEnvCfg` | `versions.v1.agents.rsl_rl_ppo_cfg:KnotTyingPPORunnerCfg` | `knot_tying_v1` | 0 |
| `Isaac-Twisted-RL-Simple-v1` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:103` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v1.env_cfg:SimpleEnvCfg` | `versions.v1.agents.rsl_rl_ppo_cfg:KnotTyingPPORunnerCfg` | `knot_tying_v1` | 0 |
| `Isaac-Twisted-RL-Simple-Play-v1` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:112` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v1.env_cfg:SimplePlayEnvCfg` | `versions.v1.agents.rsl_rl_ppo_cfg:KnotTyingPPORunnerCfg` | `knot_tying_v1` | 0 |
| `Isaac-Twisted-RL-Stiff-v1` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:103` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v1.env_cfg:StiffEnvCfg` | `versions.v1.agents.rsl_rl_ppo_cfg:KnotTyingPPORunnerCfg` | `knot_tying_v1` | 0 |
| `Isaac-Twisted-RL-Stiff-Play-v1` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:112` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v1.env_cfg:StiffPlayEnvCfg` | `versions.v1.agents.rsl_rl_ppo_cfg:KnotTyingPPORunnerCfg` | `knot_tying_v1` | 0 |
| `Isaac-Twisted-RL-Fine-v1` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:103` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v1.env_cfg:FineEnvCfg` | `versions.v1.agents.rsl_rl_ppo_cfg:KnotTyingPPORunnerCfg` | `knot_tying_v1` | 0 |
| `Isaac-Twisted-RL-Fine-Play-v1` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:112` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v1.env_cfg:FinePlayEnvCfg` | `versions.v1.agents.rsl_rl_ppo_cfg:KnotTyingPPORunnerCfg` | `knot_tying_v1` | 0 |
| `Isaac-Twisted-RL-Knot-v1` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:103` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v1.env_cfg:KnotEnvCfg` | `versions.v1.agents.rsl_rl_ppo_cfg:KnotTyingPPORunnerCfg` | `knot_tying_v1` | 0 |
| `Isaac-Twisted-RL-Knot-Play-v1` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:112` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v1.env_cfg:KnotPlayEnvCfg` | `versions.v1.agents.rsl_rl_ppo_cfg:KnotTyingPPORunnerCfg` | `knot_tying_v1` | 0 |
| `Isaac-Twisted-RL-Twist-v2` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:103` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v2.env_cfg:TwistEnvCfg` | `versions.v2.agents.rsl_rl_ppo_cfg:KnotTyingPPORunnerCfg` | `knot_tying_v2` | 7 |
| `Isaac-Twisted-RL-Twist-Play-v2` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:112` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v2.env_cfg:TwistPlayEnvCfg` | `versions.v2.agents.rsl_rl_ppo_cfg:KnotTyingPPORunnerCfg` | `knot_tying_v2` | 0 |
| `Isaac-Twisted-RL-Twist-v2-RNN` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:103` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v2.env_cfg:TwistEnvCfg` | `versions.v2.agents.rsl_rl_ppo_cfg:KnotTyingRNNPPORunnerCfg` | `knot_tying_v2_rnn` | 4 |
| `Isaac-Twisted-RL-Twist-Play-v2-RNN` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:112` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v2.env_cfg:TwistPlayEnvCfg` | `versions.v2.agents.rsl_rl_ppo_cfg:KnotTyingRNNPPORunnerCfg` | `knot_tying_v2_rnn` | 0 |
| `Isaac-Twisted-RL-Twist-v2-LSTM` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:103` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v2.env_cfg:TwistEnvCfg` | `versions.v2.agents.rsl_rl_ppo_cfg:KnotTyingLSTMPPORunnerCfg` | `knot_tying_v2_lstm` | 4 |
| `Isaac-Twisted-RL-Twist-Play-v2-LSTM` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:112` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v2.env_cfg:TwistPlayEnvCfg` | `versions.v2.agents.rsl_rl_ppo_cfg:KnotTyingLSTMPPORunnerCfg` | `knot_tying_v2_lstm` | 0 |
| `Isaac-Twisted-RL-Simple-v2` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:103` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v2.env_cfg:SimpleEnvCfg` | `versions.v2.agents.rsl_rl_ppo_cfg:KnotTyingPPORunnerCfg` | `knot_tying_v2` | 0 |
| `Isaac-Twisted-RL-Simple-Play-v2` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:112` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v2.env_cfg:SimplePlayEnvCfg` | `versions.v2.agents.rsl_rl_ppo_cfg:KnotTyingPPORunnerCfg` | `knot_tying_v2` | 0 |
| `Isaac-Twisted-RL-Simple-v2-RNN` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:103` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v2.env_cfg:SimpleEnvCfg` | `versions.v2.agents.rsl_rl_ppo_cfg:KnotTyingRNNPPORunnerCfg` | `knot_tying_v2_rnn` | 0 |
| `Isaac-Twisted-RL-Simple-Play-v2-RNN` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:112` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v2.env_cfg:SimplePlayEnvCfg` | `versions.v2.agents.rsl_rl_ppo_cfg:KnotTyingRNNPPORunnerCfg` | `knot_tying_v2_rnn` | 0 |
| `Isaac-Twisted-RL-Simple-v2-LSTM` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:103` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v2.env_cfg:SimpleEnvCfg` | `versions.v2.agents.rsl_rl_ppo_cfg:KnotTyingLSTMPPORunnerCfg` | `knot_tying_v2_lstm` | 0 |
| `Isaac-Twisted-RL-Simple-Play-v2-LSTM` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:112` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v2.env_cfg:SimplePlayEnvCfg` | `versions.v2.agents.rsl_rl_ppo_cfg:KnotTyingLSTMPPORunnerCfg` | `knot_tying_v2_lstm` | 0 |
| `Isaac-Twisted-RL-Stiff-v2` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:103` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v2.env_cfg:StiffEnvCfg` | `versions.v2.agents.rsl_rl_ppo_cfg:KnotTyingPPORunnerCfg` | `knot_tying_v2` | 0 |
| `Isaac-Twisted-RL-Stiff-Play-v2` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:112` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v2.env_cfg:StiffPlayEnvCfg` | `versions.v2.agents.rsl_rl_ppo_cfg:KnotTyingPPORunnerCfg` | `knot_tying_v2` | 0 |
| `Isaac-Twisted-RL-Stiff-v2-RNN` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:103` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v2.env_cfg:StiffEnvCfg` | `versions.v2.agents.rsl_rl_ppo_cfg:KnotTyingRNNPPORunnerCfg` | `knot_tying_v2_rnn` | 0 |
| `Isaac-Twisted-RL-Stiff-Play-v2-RNN` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:112` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v2.env_cfg:StiffPlayEnvCfg` | `versions.v2.agents.rsl_rl_ppo_cfg:KnotTyingRNNPPORunnerCfg` | `knot_tying_v2_rnn` | 0 |
| `Isaac-Twisted-RL-Stiff-v2-LSTM` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:103` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v2.env_cfg:StiffEnvCfg` | `versions.v2.agents.rsl_rl_ppo_cfg:KnotTyingLSTMPPORunnerCfg` | `knot_tying_v2_lstm` | 0 |
| `Isaac-Twisted-RL-Stiff-Play-v2-LSTM` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:112` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v2.env_cfg:StiffPlayEnvCfg` | `versions.v2.agents.rsl_rl_ppo_cfg:KnotTyingLSTMPPORunnerCfg` | `knot_tying_v2_lstm` | 0 |
| `Isaac-Twisted-RL-Fine-v2` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:103` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v2.env_cfg:FineEnvCfg` | `versions.v2.agents.rsl_rl_ppo_cfg:KnotTyingPPORunnerCfg` | `knot_tying_v2` | 0 |
| `Isaac-Twisted-RL-Fine-Play-v2` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:112` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v2.env_cfg:FinePlayEnvCfg` | `versions.v2.agents.rsl_rl_ppo_cfg:KnotTyingPPORunnerCfg` | `knot_tying_v2` | 0 |
| `Isaac-Twisted-RL-Fine-v2-RNN` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:103` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v2.env_cfg:FineEnvCfg` | `versions.v2.agents.rsl_rl_ppo_cfg:KnotTyingRNNPPORunnerCfg` | `knot_tying_v2_rnn` | 0 |
| `Isaac-Twisted-RL-Fine-Play-v2-RNN` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:112` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v2.env_cfg:FinePlayEnvCfg` | `versions.v2.agents.rsl_rl_ppo_cfg:KnotTyingRNNPPORunnerCfg` | `knot_tying_v2_rnn` | 0 |
| `Isaac-Twisted-RL-Fine-v2-LSTM` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:103` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v2.env_cfg:FineEnvCfg` | `versions.v2.agents.rsl_rl_ppo_cfg:KnotTyingLSTMPPORunnerCfg` | `knot_tying_v2_lstm` | 0 |
| `Isaac-Twisted-RL-Fine-Play-v2-LSTM` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:112` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v2.env_cfg:FinePlayEnvCfg` | `versions.v2.agents.rsl_rl_ppo_cfg:KnotTyingLSTMPPORunnerCfg` | `knot_tying_v2_lstm` | 0 |
| `Isaac-Twisted-RL-Knot-v2` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:103` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v2.env_cfg:KnotEnvCfg` | `versions.v2.agents.rsl_rl_ppo_cfg:KnotTyingPPORunnerCfg` | `knot_tying_v2` | 0 |
| `Isaac-Twisted-RL-Knot-Play-v2` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:112` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v2.env_cfg:KnotPlayEnvCfg` | `versions.v2.agents.rsl_rl_ppo_cfg:KnotTyingPPORunnerCfg` | `knot_tying_v2` | 0 |
| `Isaac-Twisted-RL-Knot-v2-RNN` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:103` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v2.env_cfg:KnotEnvCfg` | `versions.v2.agents.rsl_rl_ppo_cfg:KnotTyingRNNPPORunnerCfg` | `knot_tying_v2_rnn` | 0 |
| `Isaac-Twisted-RL-Knot-Play-v2-RNN` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:112` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v2.env_cfg:KnotPlayEnvCfg` | `versions.v2.agents.rsl_rl_ppo_cfg:KnotTyingRNNPPORunnerCfg` | `knot_tying_v2_rnn` | 0 |
| `Isaac-Twisted-RL-Knot-v2-LSTM` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:103` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v2.env_cfg:KnotEnvCfg` | `versions.v2.agents.rsl_rl_ppo_cfg:KnotTyingLSTMPPORunnerCfg` | `knot_tying_v2_lstm` | 0 |
| `Isaac-Twisted-RL-Knot-Play-v2-LSTM` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:112` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v2.env_cfg:KnotPlayEnvCfg` | `versions.v2.agents.rsl_rl_ppo_cfg:KnotTyingLSTMPPORunnerCfg` | `knot_tying_v2_lstm` | 0 |
| `Isaac-Twisted-RL-Twist-v2-R2` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:103` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v2_r2.env_cfg:TwistEnvCfg` | `versions.v2_r2.agents.rsl_rl_ppo_cfg:KnotTyingPPORunnerCfg` | `knot_tying_v2_r2` | 7 |
| `Isaac-Twisted-RL-Twist-Play-v2-R2` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:112` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v2_r2.env_cfg:TwistPlayEnvCfg` | `versions.v2_r2.agents.rsl_rl_ppo_cfg:KnotTyingPPORunnerCfg` | `knot_tying_v2_r2` | 4 |
| `Isaac-Twisted-RL-Twist-v2-R2-RNN` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:103` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v2_r2.env_cfg:TwistEnvCfg` | `versions.v2_r2.agents.rsl_rl_ppo_cfg:KnotTyingRNNPPORunnerCfg` | `knot_tying_v2_r2_rnn` | 3 |
| `Isaac-Twisted-RL-Twist-Play-v2-R2-RNN` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:112` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v2_r2.env_cfg:TwistPlayEnvCfg` | `versions.v2_r2.agents.rsl_rl_ppo_cfg:KnotTyingRNNPPORunnerCfg` | `knot_tying_v2_r2_rnn` | 0 |
| `Isaac-Twisted-RL-Twist-v2-R2-LSTM` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:103` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v2_r2.env_cfg:TwistEnvCfg` | `versions.v2_r2.agents.rsl_rl_ppo_cfg:KnotTyingLSTMPPORunnerCfg` | `knot_tying_v2_r2_lstm` | 3 |
| `Isaac-Twisted-RL-Twist-Play-v2-R2-LSTM` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:112` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v2_r2.env_cfg:TwistPlayEnvCfg` | `versions.v2_r2.agents.rsl_rl_ppo_cfg:KnotTyingLSTMPPORunnerCfg` | `knot_tying_v2_r2_lstm` | 0 |
| `Isaac-Twisted-RL-Simple-v2-R2` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:103` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v2_r2.env_cfg:SimpleEnvCfg` | `versions.v2_r2.agents.rsl_rl_ppo_cfg:KnotTyingPPORunnerCfg` | `knot_tying_v2_r2` | 0 |
| `Isaac-Twisted-RL-Simple-Play-v2-R2` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:112` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v2_r2.env_cfg:SimplePlayEnvCfg` | `versions.v2_r2.agents.rsl_rl_ppo_cfg:KnotTyingPPORunnerCfg` | `knot_tying_v2_r2` | 0 |
| `Isaac-Twisted-RL-Simple-v2-R2-RNN` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:103` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v2_r2.env_cfg:SimpleEnvCfg` | `versions.v2_r2.agents.rsl_rl_ppo_cfg:KnotTyingRNNPPORunnerCfg` | `knot_tying_v2_r2_rnn` | 0 |
| `Isaac-Twisted-RL-Simple-Play-v2-R2-RNN` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:112` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v2_r2.env_cfg:SimplePlayEnvCfg` | `versions.v2_r2.agents.rsl_rl_ppo_cfg:KnotTyingRNNPPORunnerCfg` | `knot_tying_v2_r2_rnn` | 0 |
| `Isaac-Twisted-RL-Simple-v2-R2-LSTM` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:103` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v2_r2.env_cfg:SimpleEnvCfg` | `versions.v2_r2.agents.rsl_rl_ppo_cfg:KnotTyingLSTMPPORunnerCfg` | `knot_tying_v2_r2_lstm` | 0 |
| `Isaac-Twisted-RL-Simple-Play-v2-R2-LSTM` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:112` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v2_r2.env_cfg:SimplePlayEnvCfg` | `versions.v2_r2.agents.rsl_rl_ppo_cfg:KnotTyingLSTMPPORunnerCfg` | `knot_tying_v2_r2_lstm` | 0 |
| `Isaac-Twisted-RL-Stiff-v2-R2` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:103` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v2_r2.env_cfg:StiffEnvCfg` | `versions.v2_r2.agents.rsl_rl_ppo_cfg:KnotTyingPPORunnerCfg` | `knot_tying_v2_r2` | 0 |
| `Isaac-Twisted-RL-Stiff-Play-v2-R2` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:112` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v2_r2.env_cfg:StiffPlayEnvCfg` | `versions.v2_r2.agents.rsl_rl_ppo_cfg:KnotTyingPPORunnerCfg` | `knot_tying_v2_r2` | 0 |
| `Isaac-Twisted-RL-Stiff-v2-R2-RNN` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:103` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v2_r2.env_cfg:StiffEnvCfg` | `versions.v2_r2.agents.rsl_rl_ppo_cfg:KnotTyingRNNPPORunnerCfg` | `knot_tying_v2_r2_rnn` | 0 |
| `Isaac-Twisted-RL-Stiff-Play-v2-R2-RNN` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:112` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v2_r2.env_cfg:StiffPlayEnvCfg` | `versions.v2_r2.agents.rsl_rl_ppo_cfg:KnotTyingRNNPPORunnerCfg` | `knot_tying_v2_r2_rnn` | 0 |
| `Isaac-Twisted-RL-Stiff-v2-R2-LSTM` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:103` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v2_r2.env_cfg:StiffEnvCfg` | `versions.v2_r2.agents.rsl_rl_ppo_cfg:KnotTyingLSTMPPORunnerCfg` | `knot_tying_v2_r2_lstm` | 0 |
| `Isaac-Twisted-RL-Stiff-Play-v2-R2-LSTM` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:112` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v2_r2.env_cfg:StiffPlayEnvCfg` | `versions.v2_r2.agents.rsl_rl_ppo_cfg:KnotTyingLSTMPPORunnerCfg` | `knot_tying_v2_r2_lstm` | 0 |
| `Isaac-Twisted-RL-Fine-v2-R2` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:103` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v2_r2.env_cfg:FineEnvCfg` | `versions.v2_r2.agents.rsl_rl_ppo_cfg:KnotTyingPPORunnerCfg` | `knot_tying_v2_r2` | 0 |
| `Isaac-Twisted-RL-Fine-Play-v2-R2` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:112` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v2_r2.env_cfg:FinePlayEnvCfg` | `versions.v2_r2.agents.rsl_rl_ppo_cfg:KnotTyingPPORunnerCfg` | `knot_tying_v2_r2` | 0 |
| `Isaac-Twisted-RL-Fine-v2-R2-RNN` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:103` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v2_r2.env_cfg:FineEnvCfg` | `versions.v2_r2.agents.rsl_rl_ppo_cfg:KnotTyingRNNPPORunnerCfg` | `knot_tying_v2_r2_rnn` | 0 |
| `Isaac-Twisted-RL-Fine-Play-v2-R2-RNN` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:112` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v2_r2.env_cfg:FinePlayEnvCfg` | `versions.v2_r2.agents.rsl_rl_ppo_cfg:KnotTyingRNNPPORunnerCfg` | `knot_tying_v2_r2_rnn` | 0 |
| `Isaac-Twisted-RL-Fine-v2-R2-LSTM` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:103` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v2_r2.env_cfg:FineEnvCfg` | `versions.v2_r2.agents.rsl_rl_ppo_cfg:KnotTyingLSTMPPORunnerCfg` | `knot_tying_v2_r2_lstm` | 0 |
| `Isaac-Twisted-RL-Fine-Play-v2-R2-LSTM` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:112` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v2_r2.env_cfg:FinePlayEnvCfg` | `versions.v2_r2.agents.rsl_rl_ppo_cfg:KnotTyingLSTMPPORunnerCfg` | `knot_tying_v2_r2_lstm` | 0 |
| `Isaac-Twisted-RL-Knot-v2-R2` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:103` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v2_r2.env_cfg:KnotEnvCfg` | `versions.v2_r2.agents.rsl_rl_ppo_cfg:KnotTyingPPORunnerCfg` | `knot_tying_v2_r2` | 8 |
| `Isaac-Twisted-RL-Knot-Play-v2-R2` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:112` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v2_r2.env_cfg:KnotPlayEnvCfg` | `versions.v2_r2.agents.rsl_rl_ppo_cfg:KnotTyingPPORunnerCfg` | `knot_tying_v2_r2` | 0 |
| `Isaac-Twisted-RL-Knot-v2-R2-RNN` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:103` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v2_r2.env_cfg:KnotEnvCfg` | `versions.v2_r2.agents.rsl_rl_ppo_cfg:KnotTyingRNNPPORunnerCfg` | `knot_tying_v2_r2_rnn` | 0 |
| `Isaac-Twisted-RL-Knot-Play-v2-R2-RNN` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:112` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v2_r2.env_cfg:KnotPlayEnvCfg` | `versions.v2_r2.agents.rsl_rl_ppo_cfg:KnotTyingRNNPPORunnerCfg` | `knot_tying_v2_r2_rnn` | 0 |
| `Isaac-Twisted-RL-Knot-v2-R2-LSTM` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:103` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v2_r2.env_cfg:KnotEnvCfg` | `versions.v2_r2.agents.rsl_rl_ppo_cfg:KnotTyingLSTMPPORunnerCfg` | `knot_tying_v2_r2_lstm` | 0 |
| `Isaac-Twisted-RL-Knot-Play-v2-R2-LSTM` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:112` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v2_r2.env_cfg:KnotPlayEnvCfg` | `versions.v2_r2.agents.rsl_rl_ppo_cfg:KnotTyingLSTMPPORunnerCfg` | `knot_tying_v2_r2_lstm` | 0 |
| `Isaac-Twisted-RL-Twist-v3` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:103` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v3.env_cfg:TwistEnvCfg` | `versions.v3.agents.rsl_rl_ppo_cfg:KnotTyingPPORunnerCfg` | `knot_tying_v3` | 0 |
| `Isaac-Twisted-RL-Twist-Play-v3` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:112` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v3.env_cfg:TwistPlayEnvCfg` | `versions.v3.agents.rsl_rl_ppo_cfg:KnotTyingPPORunnerCfg` | `knot_tying_v3` | 0 |
| `Isaac-Twisted-RL-Twist-v3-RNN` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:103` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v3.env_cfg:TwistEnvCfg` | `versions.v3.agents.rsl_rl_ppo_cfg:KnotTyingRNNPPORunnerCfg` | `knot_tying_v3_rnn` | 0 |
| `Isaac-Twisted-RL-Twist-Play-v3-RNN` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:112` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v3.env_cfg:TwistPlayEnvCfg` | `versions.v3.agents.rsl_rl_ppo_cfg:KnotTyingRNNPPORunnerCfg` | `knot_tying_v3_rnn` | 0 |
| `Isaac-Twisted-RL-Twist-v3-LSTM` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:103` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v3.env_cfg:TwistEnvCfg` | `versions.v3.agents.rsl_rl_ppo_cfg:KnotTyingLSTMPPORunnerCfg` | `knot_tying_v3_lstm` | 0 |
| `Isaac-Twisted-RL-Twist-Play-v3-LSTM` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:112` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v3.env_cfg:TwistPlayEnvCfg` | `versions.v3.agents.rsl_rl_ppo_cfg:KnotTyingLSTMPPORunnerCfg` | `knot_tying_v3_lstm` | 0 |
| `Isaac-Twisted-RL-Simple-v3` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:103` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v3.env_cfg:SimpleEnvCfg` | `versions.v3.agents.rsl_rl_ppo_cfg:KnotTyingPPORunnerCfg` | `knot_tying_v3` | 0 |
| `Isaac-Twisted-RL-Simple-Play-v3` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:112` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v3.env_cfg:SimplePlayEnvCfg` | `versions.v3.agents.rsl_rl_ppo_cfg:KnotTyingPPORunnerCfg` | `knot_tying_v3` | 0 |
| `Isaac-Twisted-RL-Simple-v3-RNN` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:103` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v3.env_cfg:SimpleEnvCfg` | `versions.v3.agents.rsl_rl_ppo_cfg:KnotTyingRNNPPORunnerCfg` | `knot_tying_v3_rnn` | 0 |
| `Isaac-Twisted-RL-Simple-Play-v3-RNN` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:112` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v3.env_cfg:SimplePlayEnvCfg` | `versions.v3.agents.rsl_rl_ppo_cfg:KnotTyingRNNPPORunnerCfg` | `knot_tying_v3_rnn` | 0 |
| `Isaac-Twisted-RL-Simple-v3-LSTM` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:103` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v3.env_cfg:SimpleEnvCfg` | `versions.v3.agents.rsl_rl_ppo_cfg:KnotTyingLSTMPPORunnerCfg` | `knot_tying_v3_lstm` | 0 |
| `Isaac-Twisted-RL-Simple-Play-v3-LSTM` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:112` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v3.env_cfg:SimplePlayEnvCfg` | `versions.v3.agents.rsl_rl_ppo_cfg:KnotTyingLSTMPPORunnerCfg` | `knot_tying_v3_lstm` | 0 |
| `Isaac-Twisted-RL-Stiff-v3` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:103` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v3.env_cfg:StiffEnvCfg` | `versions.v3.agents.rsl_rl_ppo_cfg:KnotTyingPPORunnerCfg` | `knot_tying_v3` | 0 |
| `Isaac-Twisted-RL-Stiff-Play-v3` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:112` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v3.env_cfg:StiffPlayEnvCfg` | `versions.v3.agents.rsl_rl_ppo_cfg:KnotTyingPPORunnerCfg` | `knot_tying_v3` | 0 |
| `Isaac-Twisted-RL-Stiff-v3-RNN` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:103` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v3.env_cfg:StiffEnvCfg` | `versions.v3.agents.rsl_rl_ppo_cfg:KnotTyingRNNPPORunnerCfg` | `knot_tying_v3_rnn` | 0 |
| `Isaac-Twisted-RL-Stiff-Play-v3-RNN` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:112` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v3.env_cfg:StiffPlayEnvCfg` | `versions.v3.agents.rsl_rl_ppo_cfg:KnotTyingRNNPPORunnerCfg` | `knot_tying_v3_rnn` | 0 |
| `Isaac-Twisted-RL-Stiff-v3-LSTM` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:103` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v3.env_cfg:StiffEnvCfg` | `versions.v3.agents.rsl_rl_ppo_cfg:KnotTyingLSTMPPORunnerCfg` | `knot_tying_v3_lstm` | 0 |
| `Isaac-Twisted-RL-Stiff-Play-v3-LSTM` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:112` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v3.env_cfg:StiffPlayEnvCfg` | `versions.v3.agents.rsl_rl_ppo_cfg:KnotTyingLSTMPPORunnerCfg` | `knot_tying_v3_lstm` | 0 |
| `Isaac-Twisted-RL-Fine-v3` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:103` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v3.env_cfg:FineEnvCfg` | `versions.v3.agents.rsl_rl_ppo_cfg:KnotTyingPPORunnerCfg` | `knot_tying_v3` | 0 |
| `Isaac-Twisted-RL-Fine-Play-v3` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:112` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v3.env_cfg:FinePlayEnvCfg` | `versions.v3.agents.rsl_rl_ppo_cfg:KnotTyingPPORunnerCfg` | `knot_tying_v3` | 0 |
| `Isaac-Twisted-RL-Fine-v3-RNN` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:103` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v3.env_cfg:FineEnvCfg` | `versions.v3.agents.rsl_rl_ppo_cfg:KnotTyingRNNPPORunnerCfg` | `knot_tying_v3_rnn` | 0 |
| `Isaac-Twisted-RL-Fine-Play-v3-RNN` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:112` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v3.env_cfg:FinePlayEnvCfg` | `versions.v3.agents.rsl_rl_ppo_cfg:KnotTyingRNNPPORunnerCfg` | `knot_tying_v3_rnn` | 0 |
| `Isaac-Twisted-RL-Fine-v3-LSTM` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:103` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v3.env_cfg:FineEnvCfg` | `versions.v3.agents.rsl_rl_ppo_cfg:KnotTyingLSTMPPORunnerCfg` | `knot_tying_v3_lstm` | 0 |
| `Isaac-Twisted-RL-Fine-Play-v3-LSTM` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:112` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v3.env_cfg:FinePlayEnvCfg` | `versions.v3.agents.rsl_rl_ppo_cfg:KnotTyingLSTMPPORunnerCfg` | `knot_tying_v3_lstm` | 0 |
| `Isaac-Twisted-RL-Knot-v3` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:103` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v3.env_cfg:KnotEnvCfg` | `versions.v3.agents.rsl_rl_ppo_cfg:KnotTyingPPORunnerCfg` | `knot_tying_v3` | 21 |
| `Isaac-Twisted-RL-Knot-Play-v3` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:112` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v3.env_cfg:KnotPlayEnvCfg` | `versions.v3.agents.rsl_rl_ppo_cfg:KnotTyingPPORunnerCfg` | `knot_tying_v3` | 4 |
| `Isaac-Twisted-RL-Knot-v3-RNN` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:103` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v3.env_cfg:KnotEnvCfg` | `versions.v3.agents.rsl_rl_ppo_cfg:KnotTyingRNNPPORunnerCfg` | `knot_tying_v3_rnn` | 3 |
| `Isaac-Twisted-RL-Knot-Play-v3-RNN` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:112` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v3.env_cfg:KnotPlayEnvCfg` | `versions.v3.agents.rsl_rl_ppo_cfg:KnotTyingRNNPPORunnerCfg` | `knot_tying_v3_rnn` | 0 |
| `Isaac-Twisted-RL-Knot-v3-LSTM` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:103` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v3.env_cfg:KnotEnvCfg` | `versions.v3.agents.rsl_rl_ppo_cfg:KnotTyingLSTMPPORunnerCfg` | `knot_tying_v3_lstm` | 3 |
| `Isaac-Twisted-RL-Knot-Play-v3-LSTM` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:112` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v3.env_cfg:KnotPlayEnvCfg` | `versions.v3.agents.rsl_rl_ppo_cfg:KnotTyingLSTMPPORunnerCfg` | `knot_tying_v3_lstm` | 0 |
| `Isaac-Twisted-RL-Twist-v4` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:103` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v4.env_cfg:TwistEnvCfg` | `versions.v4.agents.rsl_rl_ppo_cfg:KnotTyingPPORunnerCfg` | `knot_tying_v4` | 0 |
| `Isaac-Twisted-RL-Twist-Play-v4` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:112` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v4.env_cfg:TwistPlayEnvCfg` | `versions.v4.agents.rsl_rl_ppo_cfg:KnotTyingPPORunnerCfg` | `knot_tying_v4` | 0 |
| `Isaac-Twisted-RL-Simple-v4` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:103` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v4.env_cfg:SimpleEnvCfg` | `versions.v4.agents.rsl_rl_ppo_cfg:KnotTyingPPORunnerCfg` | `knot_tying_v4` | 0 |
| `Isaac-Twisted-RL-Simple-Play-v4` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:112` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v4.env_cfg:SimplePlayEnvCfg` | `versions.v4.agents.rsl_rl_ppo_cfg:KnotTyingPPORunnerCfg` | `knot_tying_v4` | 0 |
| `Isaac-Twisted-RL-Stiff-v4` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:103` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v4.env_cfg:StiffEnvCfg` | `versions.v4.agents.rsl_rl_ppo_cfg:KnotTyingPPORunnerCfg` | `knot_tying_v4` | 0 |
| `Isaac-Twisted-RL-Stiff-Play-v4` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:112` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v4.env_cfg:StiffPlayEnvCfg` | `versions.v4.agents.rsl_rl_ppo_cfg:KnotTyingPPORunnerCfg` | `knot_tying_v4` | 0 |
| `Isaac-Twisted-RL-Fine-v4` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:103` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v4.env_cfg:FineEnvCfg` | `versions.v4.agents.rsl_rl_ppo_cfg:KnotTyingPPORunnerCfg` | `knot_tying_v4` | 0 |
| `Isaac-Twisted-RL-Fine-Play-v4` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:112` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v4.env_cfg:FinePlayEnvCfg` | `versions.v4.agents.rsl_rl_ppo_cfg:KnotTyingPPORunnerCfg` | `knot_tying_v4` | 0 |
| `Isaac-Twisted-RL-Knot-v4` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:103` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v4.env_cfg:KnotEnvCfg` | `versions.v4.agents.rsl_rl_ppo_cfg:KnotTyingPPORunnerCfg` | `knot_tying_v4` | 14 |
| `Isaac-Twisted-RL-Knot-Play-v4` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:112` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v4.env_cfg:KnotPlayEnvCfg` | `versions.v4.agents.rsl_rl_ppo_cfg:KnotTyingPPORunnerCfg` | `knot_tying_v4` | 8 |
| `Isaac-Twisted-RL-Twist-v5` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:103` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v5.env_cfg:TwistEnvCfg` | `versions.v5.agents.rsl_rl_ppo_cfg:KnotTyingPPORunnerCfg` | `knot_tying_v5` | 0 |
| `Isaac-Twisted-RL-Twist-Play-v5` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:112` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v5.env_cfg:TwistPlayEnvCfg` | `versions.v5.agents.rsl_rl_ppo_cfg:KnotTyingPPORunnerCfg` | `knot_tying_v5` | 0 |
| `Isaac-Twisted-RL-Simple-v5` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:103` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v5.env_cfg:SimpleEnvCfg` | `versions.v5.agents.rsl_rl_ppo_cfg:KnotTyingPPORunnerCfg` | `knot_tying_v5` | 0 |
| `Isaac-Twisted-RL-Simple-Play-v5` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:112` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v5.env_cfg:SimplePlayEnvCfg` | `versions.v5.agents.rsl_rl_ppo_cfg:KnotTyingPPORunnerCfg` | `knot_tying_v5` | 0 |
| `Isaac-Twisted-RL-Stiff-v5` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:103` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v5.env_cfg:StiffEnvCfg` | `versions.v5.agents.rsl_rl_ppo_cfg:KnotTyingPPORunnerCfg` | `knot_tying_v5` | 0 |
| `Isaac-Twisted-RL-Stiff-Play-v5` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:112` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v5.env_cfg:StiffPlayEnvCfg` | `versions.v5.agents.rsl_rl_ppo_cfg:KnotTyingPPORunnerCfg` | `knot_tying_v5` | 0 |
| `Isaac-Twisted-RL-Fine-v5` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:103` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v5.env_cfg:FineEnvCfg` | `versions.v5.agents.rsl_rl_ppo_cfg:KnotTyingPPORunnerCfg` | `knot_tying_v5` | 0 |
| `Isaac-Twisted-RL-Fine-Play-v5` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:112` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v5.env_cfg:FinePlayEnvCfg` | `versions.v5.agents.rsl_rl_ppo_cfg:KnotTyingPPORunnerCfg` | `knot_tying_v5` | 0 |
| `Isaac-Twisted-RL-Knot-v5` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:103` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v5.env_cfg:KnotEnvCfg` | `versions.v5.agents.rsl_rl_ppo_cfg:KnotTyingPPORunnerCfg` | `knot_tying_v5` | 7 |
| `Isaac-Twisted-RL-Knot-Play-v5` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:112` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v5.env_cfg:KnotPlayEnvCfg` | `versions.v5.agents.rsl_rl_ppo_cfg:KnotTyingPPORunnerCfg` | `knot_tying_v5` | 3 |
| `Isaac-Twisted-RL-Twist-v6` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:103` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v6.env_cfg:TwistEnvCfg` | `versions.v6.agents.rsl_rl_ppo_cfg:KnotTyingPPORunnerCfg` | `knot_tying_v6` | 0 |
| `Isaac-Twisted-RL-Twist-Play-v6` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:112` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v6.env_cfg:TwistPlayEnvCfg` | `versions.v6.agents.rsl_rl_ppo_cfg:KnotTyingPPORunnerCfg` | `knot_tying_v6` | 0 |
| `Isaac-Twisted-RL-Simple-v6` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:103` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v6.env_cfg:SimpleEnvCfg` | `versions.v6.agents.rsl_rl_ppo_cfg:KnotTyingPPORunnerCfg` | `knot_tying_v6` | 0 |
| `Isaac-Twisted-RL-Simple-Play-v6` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:112` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v6.env_cfg:SimplePlayEnvCfg` | `versions.v6.agents.rsl_rl_ppo_cfg:KnotTyingPPORunnerCfg` | `knot_tying_v6` | 0 |
| `Isaac-Twisted-RL-Stiff-v6` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:103` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v6.env_cfg:StiffEnvCfg` | `versions.v6.agents.rsl_rl_ppo_cfg:KnotTyingPPORunnerCfg` | `knot_tying_v6` | 0 |
| `Isaac-Twisted-RL-Stiff-Play-v6` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:112` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v6.env_cfg:StiffPlayEnvCfg` | `versions.v6.agents.rsl_rl_ppo_cfg:KnotTyingPPORunnerCfg` | `knot_tying_v6` | 0 |
| `Isaac-Twisted-RL-Fine-v6` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:103` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v6.env_cfg:FineEnvCfg` | `versions.v6.agents.rsl_rl_ppo_cfg:KnotTyingPPORunnerCfg` | `knot_tying_v6` | 0 |
| `Isaac-Twisted-RL-Fine-Play-v6` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:112` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v6.env_cfg:FinePlayEnvCfg` | `versions.v6.agents.rsl_rl_ppo_cfg:KnotTyingPPORunnerCfg` | `knot_tying_v6` | 0 |
| `Isaac-Twisted-RL-Knot-v6` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:103` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v6.env_cfg:KnotEnvCfg` | `versions.v6.agents.rsl_rl_ppo_cfg:KnotTyingPPORunnerCfg` | `knot_tying_v6` | 10 |
| `Isaac-Twisted-RL-Knot-Play-v6` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:112` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v6.env_cfg:KnotPlayEnvCfg` | `versions.v6.agents.rsl_rl_ppo_cfg:KnotTyingPPORunnerCfg` | `knot_tying_v6` | 13 |
| `Isaac-Twisted-RL-Twist-v7` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:103` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v7.env_cfg:TwistEnvCfg` | `versions.v7.agents.rsl_rl_ppo_cfg:KnotTyingPPORunnerCfg` | `knot_tying_v7` | 0 |
| `Isaac-Twisted-RL-Twist-Play-v7` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:112` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v7.env_cfg:TwistPlayEnvCfg` | `versions.v7.agents.rsl_rl_ppo_cfg:KnotTyingPPORunnerCfg` | `knot_tying_v7` | 0 |
| `Isaac-Twisted-RL-Simple-v7` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:103` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v7.env_cfg:SimpleEnvCfg` | `versions.v7.agents.rsl_rl_ppo_cfg:KnotTyingPPORunnerCfg` | `knot_tying_v7` | 0 |
| `Isaac-Twisted-RL-Simple-Play-v7` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:112` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v7.env_cfg:SimplePlayEnvCfg` | `versions.v7.agents.rsl_rl_ppo_cfg:KnotTyingPPORunnerCfg` | `knot_tying_v7` | 0 |
| `Isaac-Twisted-RL-Stiff-v7` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:103` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v7.env_cfg:StiffEnvCfg` | `versions.v7.agents.rsl_rl_ppo_cfg:KnotTyingPPORunnerCfg` | `knot_tying_v7` | 0 |
| `Isaac-Twisted-RL-Stiff-Play-v7` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:112` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v7.env_cfg:StiffPlayEnvCfg` | `versions.v7.agents.rsl_rl_ppo_cfg:KnotTyingPPORunnerCfg` | `knot_tying_v7` | 0 |
| `Isaac-Twisted-RL-Fine-v7` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:103` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v7.env_cfg:FineEnvCfg` | `versions.v7.agents.rsl_rl_ppo_cfg:KnotTyingPPORunnerCfg` | `knot_tying_v7` | 0 |
| `Isaac-Twisted-RL-Fine-Play-v7` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:112` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v7.env_cfg:FinePlayEnvCfg` | `versions.v7.agents.rsl_rl_ppo_cfg:KnotTyingPPORunnerCfg` | `knot_tying_v7` | 0 |
| `Isaac-Twisted-RL-Knot-v7` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:103` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v7.env_cfg:KnotEnvCfg` | `versions.v7.agents.rsl_rl_ppo_cfg:KnotTyingPPORunnerCfg` | `knot_tying_v7` | 6 |
| `Isaac-Twisted-RL-Knot-Play-v7` | `src/isaac_twisted_rl/tasks/knot_tying/__init__.py:112` | `isaaclab.envs:ManagerBasedRLEnv` | `versions.v7.env_cfg:KnotPlayEnvCfg` | `versions.v7.agents.rsl_rl_ppo_cfg:KnotTyingPPORunnerCfg` | `knot_tying_v7` | 3 |
| `Isaac-Twisted-RL-RS-Art28-v4` | `src/isaac_twisted_rl/tasks/knot_tying/rope_study/__init__.py:20` | `common.rope_backends.env:RopeStudyEnv` | `rope_study.env_cfg:Art28EnvCfg` | `rope_study.agents.rsl_rl_ppo_cfg:Art28PPORunnerCfg` | `knot_tying_rs_art28` | 0 |
| `Isaac-Twisted-RL-RS-Art28-Play-v4` | `src/isaac_twisted_rl/tasks/knot_tying/rope_study/__init__.py:20` | `common.rope_backends.env:RopeStudyEnv` | `rope_study.env_cfg:Art28PlayEnvCfg` | `rope_study.agents.rsl_rl_ppo_cfg:Art28PPORunnerCfg` | `knot_tying_rs_art28` | 2 |
| `Isaac-Twisted-RL-RS-Art28LowMu-v4` | `src/isaac_twisted_rl/tasks/knot_tying/rope_study/__init__.py:20` | `common.rope_backends.env:RopeStudyEnv` | `rope_study.env_cfg:Art28LowMuEnvCfg` | `rope_study.agents.rsl_rl_ppo_cfg:Art28LowMuPPORunnerCfg` | `knot_tying_rs_art28lowmu` | 0 |
| `Isaac-Twisted-RL-RS-Art28LowMu-Play-v4` | `src/isaac_twisted_rl/tasks/knot_tying/rope_study/__init__.py:20` | `common.rope_backends.env:RopeStudyEnv` | `rope_study.env_cfg:Art28LowMuPlayEnvCfg` | `rope_study.agents.rsl_rl_ppo_cfg:Art28LowMuPPORunnerCfg` | `knot_tying_rs_art28lowmu` | 0 |
| `Isaac-Twisted-RL-RS-Art56-v4` | `src/isaac_twisted_rl/tasks/knot_tying/rope_study/__init__.py:20` | `common.rope_backends.env:RopeStudyEnv` | `rope_study.env_cfg:Art56EnvCfg` | `rope_study.agents.rsl_rl_ppo_cfg:Art56PPORunnerCfg` | `knot_tying_rs_art56` | 0 |
| `Isaac-Twisted-RL-RS-Art56-Play-v4` | `src/isaac_twisted_rl/tasks/knot_tying/rope_study/__init__.py:20` | `common.rope_backends.env:RopeStudyEnv` | `rope_study.env_cfg:Art56PlayEnvCfg` | `rope_study.agents.rsl_rl_ppo_cfg:Art56PPORunnerCfg` | `knot_tying_rs_art56` | 0 |
| `Isaac-Twisted-RL-RS-XpbdC28-v4` | `src/isaac_twisted_rl/tasks/knot_tying/rope_study/__init__.py:20` | `common.rope_backends.env:RopeStudyEnv` | `rope_study.env_cfg:XpbdC28EnvCfg` | `rope_study.agents.rsl_rl_ppo_cfg:XpbdC28PPORunnerCfg` | `knot_tying_rs_xpbdc28` | 4 |
| `Isaac-Twisted-RL-RS-XpbdC28-Play-v4` | `src/isaac_twisted_rl/tasks/knot_tying/rope_study/__init__.py:20` | `common.rope_backends.env:RopeStudyEnv` | `rope_study.env_cfg:XpbdC28PlayEnvCfg` | `rope_study.agents.rsl_rl_ppo_cfg:XpbdC28PPORunnerCfg` | `knot_tying_rs_xpbdc28` | 3 |
| `Isaac-Twisted-RL-RS-XpbdC56-v4` | `src/isaac_twisted_rl/tasks/knot_tying/rope_study/__init__.py:20` | `common.rope_backends.env:RopeStudyEnv` | `rope_study.env_cfg:XpbdC56EnvCfg` | `rope_study.agents.rsl_rl_ppo_cfg:XpbdC56PPORunnerCfg` | `knot_tying_rs_xpbdc56` | 0 |
| `Isaac-Twisted-RL-RS-XpbdC56-Play-v4` | `src/isaac_twisted_rl/tasks/knot_tying/rope_study/__init__.py:20` | `common.rope_backends.env:RopeStudyEnv` | `rope_study.env_cfg:XpbdC56PlayEnvCfg` | `rope_study.agents.rsl_rl_ppo_cfg:XpbdC56PPORunnerCfg` | `knot_tying_rs_xpbdc56` | 0 |
| `Isaac-Twisted-RL-RS-XpbdC28LowMu-v4` | `src/isaac_twisted_rl/tasks/knot_tying/rope_study/__init__.py:20` | `common.rope_backends.env:RopeStudyEnv` | `rope_study.env_cfg:XpbdC28LowMuEnvCfg` | `rope_study.agents.rsl_rl_ppo_cfg:XpbdC28LowMuPPORunnerCfg` | `knot_tying_rs_xpbdc28lowmu` | 2 |
| `Isaac-Twisted-RL-RS-XpbdC28LowMu-Play-v4` | `src/isaac_twisted_rl/tasks/knot_tying/rope_study/__init__.py:20` | `common.rope_backends.env:RopeStudyEnv` | `rope_study.env_cfg:XpbdC28LowMuPlayEnvCfg` | `rope_study.agents.rsl_rl_ppo_cfg:XpbdC28LowMuPPORunnerCfg` | `knot_tying_rs_xpbdc28lowmu` | 0 |
| `Isaac-Twisted-RL-RS-XpbdMS28-v4` | `src/isaac_twisted_rl/tasks/knot_tying/rope_study/__init__.py:20` | `common.rope_backends.env:RopeStudyEnv` | `rope_study.env_cfg:XpbdMS28EnvCfg` | `rope_study.agents.rsl_rl_ppo_cfg:XpbdMS28PPORunnerCfg` | `knot_tying_rs_xpbdms28` | 0 |
| `Isaac-Twisted-RL-RS-XpbdMS28-Play-v4` | `src/isaac_twisted_rl/tasks/knot_tying/rope_study/__init__.py:20` | `common.rope_backends.env:RopeStudyEnv` | `rope_study.env_cfg:XpbdMS28PlayEnvCfg` | `rope_study.agents.rsl_rl_ppo_cfg:XpbdMS28PPORunnerCfg` | `knot_tying_rs_xpbdms28` | 0 |
| `Isaac-Twisted-RL-RS-Fem-v4` | `src/isaac_twisted_rl/tasks/knot_tying/rope_study/__init__.py:20` | `common.rope_backends.env:RopeStudyEnv` | `rope_study.env_cfg:FemEnvCfg` | `rope_study.agents.rsl_rl_ppo_cfg:FemPPORunnerCfg` | `knot_tying_rs_fem` | 1 |
| `Isaac-Twisted-RL-RS-Fem-Play-v4` | `src/isaac_twisted_rl/tasks/knot_tying/rope_study/__init__.py:20` | `common.rope_backends.env:RopeStudyEnv` | `rope_study.env_cfg:FemPlayEnvCfg` | `rope_study.agents.rsl_rl_ppo_cfg:FemPPORunnerCfg` | `knot_tying_rs_fem` | 0 |

## 4. ログのディレクトリ名

### 4.1 名前を決めている箇所

| 何を決めるか | 箇所 |
|---|---|
| ログ根 `logs/rsl_rl/<experiment_name>/` (学習) | `scripts/rsl_rl/train.py:140` |
| ログ根 (再生・評価が読む) | `scripts/rsl_rl/play.py:101`。`eval_chain.py`・`eval_search.py` は `--checkpoint` のパスを直接受ける (`scripts/rsl_rl/eval_chain.py:76`、`scripts/rsl_rl/eval_search.py:559`) |
| run フォルダ `<日時>[_<run_name>]` | `scripts/rsl_rl/train.py:144-149`、`--run_name` の適用は `scripts/rsl_rl/cli_args.py:82-83` |
| `--experiment_name` (定義だけで適用されない) | 定義 `scripts/rsl_rl/cli_args.py:25-27`、適用の関数 `:70-91` に無い。`kt/next_action.md:580` も「効かない」と書く |
| 版の `experiment_name` | v0 `kt/versions/v0/agents/rsl_rl_ppo_cfg.py:43`、v1 `kt/versions/v1/agents/rsl_rl_ppo_cfg.py:53`、v2 `kt/versions/v2/agents/rsl_rl_ppo_cfg.py:140` / `:199` / `:229`、v2-R2 `kt/versions/v2_r2/agents/rsl_rl_ppo_cfg.py:29` / `:36` / `:43`、v3 `kt/versions/v3/agents/rsl_rl_ppo_cfg.py:27` / `:34` / `:41`、v4 `kt/versions/v4/agents/rsl_rl_ppo_cfg.py:52`、v5 `kt/versions/v5/agents/rsl_rl_ppo_cfg.py:25`、v6 `kt/versions/v6/agents/rsl_rl_ppo_cfg.py:21`、v7 `kt/versions/v7/agents/rsl_rl_ppo_cfg.py:18` |
| rope_study の `experiment_name` | `kt/rope_study/agents/rsl_rl_ppo_cfg.py:21` (`f"knot_tying_rs_{name.lower()}"`) |
| rope_study の run 名 | `kt/rope_study/run_train_chain.sh:25` (`--run_name rs_${M}_s42`) |
| ログ名から run を探す | `kt/rope_study/make_eval_jobs.py:43` (`knot_tying_rs_{model.lower()}/*_rs_{model}_s{seed}`) |
| run フォルダの中のファイル | `archive.pt` (`kt/versions/v4/env_cfg.py:119`)、`eval_chain_it{N}.json`・`eval_search_*_it{N}.json` (`kt/README.md:99`、`:103`)、`model_<it>.pt` |
| 命名の規約 | `kt/README.md:186` (「`experiment_name` は `knot_tying_vN`、変種は `_rnn` / `_lstm`」) |

抜き出しのコマンド: `git grep -n -E 'experiment_name' -- 'src/**/agents/*.py'`。

### 4.2 コードの名前と `ls logs/rsl_rl` の突き合わせ

`ls logs/rsl_rl` (2026-10-03 時点) の結果: `eval_search_lane.sh` (ファイル) と、次のフォルダ 16 個。
`knot_tying_rs_art28`、`knot_tying_rs_art28lowmu`、`knot_tying_rs_xpbdc28`、`knot_tying_rs_xpbdc28lowmu`、`knot_tying_rs_xpbdc56`、`knot_tying_rs_xpbdms28`、
`knot_tying_v0`、`knot_tying_v1`、`knot_tying_v2`、`knot_tying_v2_r2`、`knot_tying_v3`、`knot_tying_v4`、`knot_tying_v4_speed`、`knot_tying_v5`、`knot_tying_v6`、`knot_tying_v6_ablation`。

| コードの `experiment_name` (23 個) | どのタスク ID が使うか | `logs/rsl_rl/` に | run の数 |
|---|---|---|---|
| `knot_tying_v0` | `*-v0` (5 プリセット × 学習/Play) | あり | 1 |
| `knot_tying_v1` | `*-v1` | あり | 1 |
| `knot_tying_v2` | `*-v2` | あり | 1 |
| `knot_tying_v2_rnn` | `*-v2-RNN` | **なし** | — |
| `knot_tying_v2_lstm` | `*-v2-LSTM` | **なし** | — |
| `knot_tying_v2_r2` | `*-v2-R2` | あり | 1 |
| `knot_tying_v2_r2_rnn` | `*-v2-R2-RNN` | **なし** | — |
| `knot_tying_v2_r2_lstm` | `*-v2-R2-LSTM` | **なし** | — |
| `knot_tying_v3` | `*-v3` | あり | 3 |
| `knot_tying_v3_rnn` | `*-v3-RNN` | **なし** | — |
| `knot_tying_v3_lstm` | `*-v3-LSTM` | **なし** | — |
| `knot_tying_v4` | `*-v4` | あり | 2 |
| `knot_tying_v5` | `*-v5` | あり | 1 |
| `knot_tying_v6` | `*-v6` | あり | 2 |
| `knot_tying_v7` | `*-v7` | **なし** (v7 は未学習: `README.md:66`) | — |
| `knot_tying_rs_art28` | `RS-Art28[-Play]-v4` | あり | 1 |
| `knot_tying_rs_art28lowmu` | `RS-Art28LowMu[-Play]-v4` | あり | 1 |
| `knot_tying_rs_art56` | `RS-Art56[-Play]-v4` | **なし** | — |
| `knot_tying_rs_xpbdc28` | `RS-XpbdC28[-Play]-v4` | あり | 3 |
| `knot_tying_rs_xpbdc56` | `RS-XpbdC56[-Play]-v4` | あり | 3 |
| `knot_tying_rs_xpbdms28` | `RS-XpbdMS28[-Play]-v4` | あり | 1 |
| `knot_tying_rs_xpbdc28lowmu` | `RS-XpbdC28LowMu[-Play]-v4` | あり | 1 |
| `knot_tying_rs_fem` | `RS-Fem[-Play]-v4` | **なし** | — |

**コードにあってログに無い名前 (9 個):** `knot_tying_v2_rnn`、`knot_tying_v2_lstm`、`knot_tying_v2_r2_rnn`、`knot_tying_v2_r2_lstm`、`knot_tying_v3_rnn`、`knot_tying_v3_lstm`、`knot_tying_v7`、`knot_tying_rs_art56`、`knot_tying_rs_fem`。
なお `kt/README.md:77` はこのうち `_rnn` / `_lstm` と `knot_tying_v7` を「分かれる」フォルダとして列挙している。

**ログにあってコードに無い名前 (2 個 + ファイル 1 個):**

| 名前 | 中身 (`ls` だけ) | 文書での扱い | どうやって付けた名前か |
|---|---|---|---|
| `knot_tying_v4_speed` | run 3 個 (`2026-09-30_22-27-57_A_it16_T12`、`2026-10-01_02-27-56_B_it16_T4`、`2026-10-01_06-27-56_C_it8_T4`) | `kt/versions/v4/README.md:251`、`paper/Discussion.md:42` | 未確認 (`--experiment_name` は効かないので、Hydra の上書きなど別の方法。追跡ファイルに記録なし) |
| `knot_tying_v6_ablation` | run 4 個 (`abl_relA_s42`・`abl_depB_s42`・`abl_toP_s42`・`abl_sw2_s42`) と `README.md` | `kt/README.md:81`、`kt/versions/v6/Result_Report_v6.md:220`。**v7 の参照 run は `abl_depB_s42`** (`kt/versions/v7/README.md:70`、`paper/Report_Search.md:87`) | 同上 (未確認) |
| `eval_search_lane.sh` (ファイル) | — | `paper/Report_Search.md:268` | ログのフォルダではない |

### 4.3 run 名 (過去の実験名として残っているもの)

`ls logs/rsl_rl/*/` の run 名の接尾辞: (なし: v0・v1・v2・v2_r2・v3 の 2 本)、`v3_newbuf_10it`、`3h_compare`、`overnight_s42`、`overnight_s43`、`A_it16_T12`・`B_it16_T4`・`C_it8_T4`、`abl_relA_s42`・`abl_depB_s42`・`abl_toP_s42`・`abl_sw2_s42`、`rs_<M>_s42`・`rs_<M>_s43` (M は大文字小文字のままのモデル名。フォルダ名は小文字なので、同じ名前が 2 つの書き方で並ぶ。例 `knot_tying_rs_xpbdc28/2026-10-03_10-35-27_rs_XpbdC28_s43`)。

文書の中の方策のラベル: `v4Knot` (`kt/rope_study/summarize_eval.py:24`、`outputs/rope_study/eval/v4Knot/`)、`XpbdC28_s43` / `XpbdC56_s43` (同 `:24`)、`v4=…`・`v6s42=…` (`scripts/rsl_rl/summarize_search.py:4`)。

## 5. 名前の層の対応表 (タスク ID ↔ プリセット ↔ rope_study のモデル名 ↔ ログ)

| rope_specs のプリセット (小文字) | タスク ID のプリセット | 版のタスク ID (学習 / Play) | 文書上の学習の用途 | rope_study のモデル | rope_study のタスク ID | ログ名 | 初期アーカイブのファイル |
|---|---|---|---|---|---|---|---|
| `simple` | `Simple` | `Isaac-Twisted-RL-Simple[-Play]-vN[-RNN\|-LSTM]` (30 本) | 比較用 | — | — | `knot_tying_vN` (他のプリセットと共用) | v4 以降は `overhand_20links.pt` を探す (`outputs/archive_seed/` に無い) |
| `stiff` | `Stiff` | `…-Stiff-…` (30 本) | 比較用 | — | — | 同上 | 同上 |
| `twist` | `Twist` | `…-Twist-…` (30 本) | v0〜v2-R2 の学習 (`kt/README.md:48`) | — | — | 同上 (`v0`〜`v2_r2` のログ) | 同上 |
| `fine` | `Fine` | `…-Fine-…` (30 本) | 比較用 | — | — | 同上 | `overhand_48links.pt` を探す (無い) |
| `knot` | `Knot` | `…-Knot-…` (30 本) | v3〜v7 の学習。`Knot-v2-R2` はゲート 1 の台本の既定 (`kt/demos/overhand_script.py:85`) | `Art28` (kind `articulation`)、`Art28LowMu` (摩擦だけ違う) | `Isaac-Twisted-RL-RS-Art28[-Play]-v4`、`…-RS-Art28LowMu[-Play]-v4` | 版: `knot_tying_v3`〜`v7`。RS: `knot_tying_rs_art28`・`knot_tying_rs_art28lowmu` | 版: `outputs/archive_seed/overhand_28links.pt`。RS: `outputs/archive_seed/rope_study/Art28.pt` など |
| `knot56` | (無し) | — | — | `Art56` (kind `articulation_resampled`) | `…-RS-Art56[-Play]-v4` | `knot_tying_rs_art56` (ログ無し) | `rope_study/Art56.pt` |
| (rope_specs に無い) | (無し) | — | — | `XpbdC28`・`XpbdC56`・`XpbdMS28`・`XpbdC28LowMu` (kind `xpbd`) | `…-RS-<M>[-Play]-v4` | `knot_tying_rs_<m 小文字>` | `rope_study/<M>.pt` |
| (rope_specs に無い) | (無し) | — | — | `Fem` (kind `fem`) | `…-RS-Fem[-Play]-v4` | `knot_tying_rs_fem` (ログ無し) | `rope_study/Fem.pt` |

表の根拠: `kt/__init__.py:54`、`kt/common/rope_model/rope_specs.py:579-581`、`kt/rope_study/models.py:97-129`、`kt/rope_study/agents/rsl_rl_ppo_cfg.py:21`、`kt/versions/v4/env_cfg.py:53`、`kt/rope_study/env_cfg.py:43`、`:202`、`ls outputs/archive_seed` (`overhand_28links.pt` と `rope_study/` だけ)。

対応表から分かること:

* **プリセットの層はタスク ID にしか無い。** ログ名・run 名・チェックポイントのどれにもプリセットが現れない (C-2)。
* **`Knot` と `Art28` は「同じ紐」だが同じ環境ではない。** `kt/rope_study/env_cfg.py:14-15` は「`Art28` は v4 の `Knot` と同じ環境になる」と書くが、
  entry_point (`RopeStudyEnv` と `ManagerBasedRLEnv`)、初期アーカイブのファイル (`rope_study/Art28.pt` と `overhand_28links.pt`)、ログ名が違う。
* **表記の規則が層ごとに違う。** タスク ID `v2-R2` ↔ パッケージ `v2_r2` ↔ クラス `V2R2` ↔ ログ `v2_r2`。変種 `-RNN` ↔ ログ `_rnn` ↔ クラス `RNN` (中身は GRU)。
  モデル `XpbdC28` ↔ タスク ID `RS-XpbdC28` ↔ ログ `rs_xpbdc28` ↔ run `rs_XpbdC28_s42`。

## 6. 所見

### 6.1 衝突・揺れの一覧 (重大な順)

| 番号 | 衝突・揺れ | なぜ重いか | 根拠 |
|---|---|---|---|
| C-1 | **「段」の二重定義と、逆向きの説明。** v1〜v3 の「段」は位相クラス (報酬ラダーの段)、v4 以降の「段」は `BfsPotential.stage` = round(3Φ)。値の範囲が同じ 0〜3 なので取り違えやすい。v4 の段 2 は C2 より広い (cover 後を含む)。文書は段を「固結びへの BFS 距離」と書くが、段 = 3 − 距離で向きが逆。`stage_gated.py` は「交差数に当たる」と、さらに別の説明をする。`StateArchive` の引数 `classes` に v4 は段を入れる | MDP の中心の概念。v3 と v4 の比較、アーカイブの中身の解釈を誤る | `kt/common/topology/knot_detect.py:233-234`、`kt/common/topology/knot_graph.py:273-276`、`:548-559`、`kt/common/mdp/stages.py:1`、`kt/README.md:42`、`kt/common/policies/stage_gated.py:6`、`kt/common/mdp/archive.py:269`、`kt/versions/v4/README.md:49` |
| C-2 | **ログ名にプリセットが無く、層ごとに表記の規則も違う。** `experiment_name` は版 (と変種) だけで決まる。同じ版の 5 プリセットが同じ `logs/rsl_rl/knot_tying_vN/` に入り、`--resume` は最新の run を拾う。リンク数が違えば観測の次元が違う (`Knot` で 129 次元、うちリンク位置 84 = 28 × 3) ので、README が変種で避けたはずの「別構造のチェックポイントを読む」事故がプリセットで起きうる。どのプリセットで学習した run かは、ログ名からは分からない | 過去の run の出所が名前から復元できない | `kt/__init__.py:97-108`、`kt/versions/v4/agents/rsl_rl_ppo_cfg.py:52`、`scripts/rsl_rl/train.py:140`、`kt/README.md:75-79`、`kt/versions/v4/README.md:51`、`paper/Discussion.md:71` |
| C-3 | **「Knot」の 4 義。** ロープのプリセット (28 リンク) / 課題の名前 (`knot_tying`、`KnotTying*`、`knot_tying_vN`) / 目標の達成 (`knot`、`knot_tied`) / 結び目一般 (`knot_detect`、`knot_graph`)。`Isaac-Twisted-RL-Knot-v4` の Knot はロープで、`Twist-v4` も課題は同じ固結び。`Knot-v2-R2` (ロープ Knot で v2-R2 の MDP) は v2-R2 の学習 (Twist) とは別物 | タスク ID を読み違える | `kt/__init__.py:54`、`kt/common/rope_model/rope_specs.py:22`、`scripts/rsl_rl/eval_chain.py:16`、`kt/common/beam_search.py:51`、`kt/versions/v3/env_cfg.py:12`、`kt/demos/overhand_script.py:85` |
| C-4 | **「Twist」「Twisted」の多義。** Twist = プリセット / 捩り / R1 の輪。Twisted = 本リポジトリ (README は「Twisted-RL の課題を実装」) / TWISTED-RL (Freund ら) / TWISTED (Sudry ら) / 移植元の MuJoCo 実装 / パッケージ。タスク ID `Isaac-Twisted-RL-Twist-v0` には両方が入る | プロジェクト名と論文名が同じで、引用・比較の文で主語が曖昧になる | `README.md:8`、`:135`、`paper/Report_Paper.md:7`、`kt/common/topology/representation.py:3`、`kt/common/rope_model/rope_specs.py:223`、`kt/common/topology/knot_detect.py:67` |
| C-5 | **`vN` の多重使用と一覧の漏れ。** MDP の版 v0〜v7 のほかに、XPBD の指の接触 v1 / v2、物理試験 v2 / v3。`Report_rope_models.md:483` の「以下、XPBD の学習・評価はすべて v2」は指の接触の版で、MDP は v4。pyproject の版 1.2.0 とも無関係。`versions/__init__.py` の一覧は v6 で止まっている (v7 は `95ac19a` で追加、この docstring の最終更新は `ccae2ae`) | 「v2 の結果」が 2 通りに読める | `kt/rope_study/models.py:65-79`、`kt/rope_study/HANDOFF.md:82`、`:98`、`kt/rope_study/Report_rope_models.md:483`、`kt/versions/__init__.py:6-13`、`pyproject.toml:3` |
| C-6 | **`-Play` の意味が版で変わる。** v0〜v3 は「推論を見る環境」(環境 9、描画を増やす)。v4 以降と rope_study は chain 評価の環境 (18 手、段をつなぐ、終了条件が違う)。同じ接尾辞で中身が評価の定義になる | 評価の数字がどの環境の数字か、ID から分からない | `kt/common/rope_task_cfg.py:145`、`:161`、`kt/versions/v3/env_cfg.py:205-208`、`kt/versions/v4/env_cfg.py:214-223`、`kt/rope_study/__init__.py:4` |
| C-7 | **「発散」の判定が 3 通り。** 終了項 `diverged` (NaN・高さ 5 m、v2-R2 から水平 3 m・速度 20 m/s を追加)、報酬側の既定 (NaN だけ)、`divergence_bench` の spike。版によって終了項の閾値も違う | 版をまたぐ発散率の比較で定義がずれる | `kt/common/mdp/terminations.py:29-48`、`kt/versions/v2_r2/env_cfg.py:48-51`、`:199`、`kt/common/mdp/rewards.py:401-405`、`:454-458`、`kt/demos/divergence_bench.py:26-30` |
| C-8 | **`experiment_name` 周りの食い違い。** `--experiment_name` は定義だけで効かない。`train.py:146` の「experiment name」は run フォルダ名のこと。コードに無いログ名 2 つが実在し、v7 の参照 run は `knot_tying_v6_ablation` の下にある | ログの在りかをコードから辿れない | `scripts/rsl_rl/cli_args.py:25-27`、`:70-91`、`scripts/rsl_rl/train.py:146`、`kt/next_action.md:580`、`kt/versions/v7/README.md:70` |
| C-9 | **rope_study のモデル名の一覧が 4 か所にあり、同じ名前 `MODELS` が別物を指す。** `models.MODELS` (8 個の dict) / `make_eval_jobs.MODELS` (6 個の tuple、評価先) / `summarize_eval.MODELS` (6 個) / `run_physics_suite.sh` の `MODELS` (7 個、Fem 無し)。ログ名は小文字、run 名と初期アーカイブは元の大文字小文字 | モデルを足す・改名するときに漏れる | `kt/rope_study/models.py:97-132`、`kt/rope_study/make_eval_jobs.py:30-31`、`kt/rope_study/summarize_eval.py:22`、`kt/rope_study/run_physics_suite.sh:14`、`kt/rope_study/agents/rsl_rl_ppo_cfg.py:21` |
| C-10 | **「モデル」の多義。** 紐モデル (`RopeModel`) / `common/rope_model/` (剛体鎖ロープの USD と設定) / `RopeStudyEnvCfg.rope_model` (名前の文字列) / XPBD の `"model"` (cosserat・mass_spring) / `model_<it>.pt` / 方策。`rope_model` パッケージは rope_study の「モデル」より前からあり、Art 系だけに使う | パッケージ名から中身を推測できない | `kt/rope_study/models.py:44-62`、`:84`、`kt/rope_study/env_cfg.py:71`、`kt/common/rope_model/rope_cfg.py:62` |
| C-11 | **「archive」と「seed」。** 文書の archive フォルダは今は無い (`git ls-files`・全履歴に `archive/` なし)。しかし archive はすでに状態アーカイブ (`StateArchive`、`mdp/archive.py`、`archive.pt`、`outputs/archive_seed/`、ログのキー `Curriculum/archive_log/*`) の名前。文書の置き場を `archive/` にすると grep と会話で衝突する。seed も乱数のシードと初期アーカイブの両方 | 新しく作る名前の衝突 (予防) | `kt/common/state_archive.py:1`、`kt/common/mdp/archive.py:57`、`:468`、`kt/versions/v4/env_cfg.py:53`、`kt/versions/v3/README.md:113-118`、`kt/demos/divergence_bench.py:15` |
| C-12 | **「chain」の多義。** chain 評価 / `chain_est` (学習ログの見積り) / chain モード (Play の終了条件) / 剛体鎖 / `run_train_chain.sh` (ジョブのキュー) | 「chain の結果」が評価か見積りか分からない | `scripts/rsl_rl/eval_chain.py:6`、`kt/common/mdp/archive.py:402-420`、`kt/common/mdp/stages.py:163-184`、`kt/rope_study/run_train_chain.sh:2` |
| C-13 | **「手」と move。** 文書は「手」、コードは `moves` (`beam_search`・`eval_search` の予算) と `actions` (`actions_per_episode`)。同じ `moves` が `knot_graph` では位相移動 (R1 / R2 / cross) を指す | 予算の単位と位相の移動が同じ語 | `kt/common/beam_search.py:303`、`scripts/rsl_rl/eval_search.py:64`、`kt/common/topology/knot_graph.py:138-140`、`:201`、`kt/common/README.md:10` |
| C-14 | **「辺 / edge」と ρ(g)。** `representation.Edge` は半辺、`GraphEdge` は高レベルの辺、`representation.py` の局所変数 `next_edge` は半辺の next ポインタで、v6 の `next_edge` (次にたどる辺) と同名。v6 の next edge は論文の ρ(g) に「当たる」とされるが、ρ(g) の中身は論文に書かれていない | 同名で別物。論文との対応が過大に読める | `kt/common/topology/representation.py:96`、`:193`、`kt/common/topology/knot_graph.py:282`、`kt/common/mdp/observations.py:214-217`、`paper/explainer/twisted-rl.html:737` |
| C-15 | **プリセットの数と大文字小文字の揺れ。** ROPE_SPECS は 6 種、タスク ID は 5 種、文書は「4 プリセット」「5 種」「5 つ」と書く。v2 のコメントは生成クラスを 4 つしか挙げない。小文字 `knot` と大文字 `Knot` が同じものを指し、`knot56` はタスク ID に無い | 「全プリセット」の範囲が文書ごとに違う | `kt/common/rope_model/rope_specs.py:579-581`、`kt/__init__.py:54`、`kt/common/rope_model/__init__.py:15`、`kt/common/rope_state.py:20`、`kt/README.md:7`、`:47`、`kt/versions/v2/env_cfg.py:286` |
| C-16 | **古い参照。** `paper/Report.md` (改名済み) が 26 行、存在しない `eval_compare.py`、「学習は `Knot` (v3)」で止まったコメント、README のフォルダ構成に `rope_backends/`・`rope_study/`・`beam_search.py`・`state_clone.py` が無い | リンク切れ・古い説明 | commit `dbf9a97`、`README.md:118`、`scripts/rsl_rl/eval_chain.py:8`、`kt/__init__.py:52-53`、`kt/common/rope_task_cfg.py:8`、`kt/README.md:105-152` |
| C-17 | **「ゲート 1」に 2 以降が無い。** 番号付きの関門は 1 つだけで、rope_study では同じ台本の試験名 (`gate1/`) にもなった。`StageGated` (gating) や LSTM のゲートとも語が重なる | 軽い。初出で定義すれば足りる | `kt/next_action.md:465-470`、`kt/rope_study/run_physics_suite.sh:24-38`、`kt/common/policies/stage_gated.py:11` |
| C-18 | **`-RNN` の中身は GRU。** README は明記しているが、ID・クラス名・ログ名は RNN | 軽い | `kt/README.md:69-70`、`kt/versions/v2/agents/rsl_rl_ppo_cfg.py:155`、`:208` |
| C-19 | **「R2」の多義と近似。** R2 除去 / R2 移動 (`MOVE_R2`) / 版の名前 v2-R2。R2 除去は Gauss 符号の上での近似 | 軽い | `kt/common/topology/knot_detect.py:249-290`、`kt/common/topology/knot_graph.py:139`、`kt/__init__.py:40` |
| C-20 | **`LowMu` の名前と中身。** 「低い摩擦」ではなく「実物の範囲の摩擦」(既定が高すぎる)。剛体鎖と XPBD で効かせるパラメータが違う | 軽い | `kt/rope_study/models.py:10`、`:38-40`、`:101-126` |
| C-21 | **表記の揺れ: 紐 / ロープ、固結び / overhand / 三葉 / trefoil。** rope_study の文書は「紐」、それ以外は「ロープ」。固結び (開いた紐) と三葉結び (閉じた結び目) を同一視する書き方がある | 軽い | `kt/rope_study/Report_rope_models.md:166`、`kt/common/rope_task_cfg.py:1`、`kt/common/topology/knot_detect.py:1`、`kt/versions/v1/README.md:17` |
| C-22 | **「表現」の多義。** 紐の表現 (rope_study) と位相の表現 (`representation.py`) | 軽い | `kt/rope_study/models.py:4`、`kt/common/topology/representation.py:1` |
| C-23 | **使われていない登録 ID が多い。** 166 本のうち追跡ファイルで参照されるのは 28 本。`Simple` / `Stiff` / `Fine` の 90 本は 0 本。v4 以降の `Twist` / `Simple` / `Stiff` / `Fine` は初期アーカイブ `overhand_{20,48}links.pt` が `outputs/archive_seed/` に無い (読めなければ警告して飛ばす) | 名前の整理の範囲を決める材料 | §3.3 の「参照」列、`kt/versions/v4/env_cfg.py:53`、`kt/common/mdp/archive.py:241-242`、`ls outputs/archive_seed` |

### 6.2 文書だけで解決できるもの / 識別子を変えないと解決しないもの

| 番号 | 文書だけで解決できる | 識別子の変更が要る (または望ましい) |
|---|---|---|
| C-1 | ほぼ全部。「段」を v4 の stage だけに使い、v1〜v3 は「位相クラス」と書く。定義を「段 = round(3Φ)」に直し、「BFS 距離」の記述 (`kt/README.md:42`、`kt/common/mdp/stages.py:1`) を直す | 任意: `StateArchive(classes=…)` → `keys=…` などに改名 (保存ファイルに引数名は出ないので互換の問題なし)。`topology_class` の docstring「報酬ラダーの段」を直す |
| C-2 | 一部。「Knot 以外で学習するなら `--run_name` か Hydra で分ける」と規約に書けば事故は防げる | 根本的には `experiment_name` にプリセットを入れる (例 `knot_tying_knot_v4`)。ログ名が変わる |
| C-3 | はい。用語集で 4 義を書き分け、プリセットには必ず「ロープ」を付ける | 不要 (変えるとタスク ID が変わる) |
| C-4 | はい。用語集で書き分ける | 不要 |
| C-5 | はい。指の接触・物理試験には vN を使わない。`versions/__init__.py` に v7 を足す | 任意: `XPBD_FINGER_V2` を内容の名前に改名 (内部の定数。Hydra の上書きはフィールド名なので互換の問題なし) |
| C-6 | はい。版ごとの Play の意味を 1 つの表にする | 不要 (変えるとタスク ID 83 本が変わる) |
| C-7 | はい。判定の定義を 1 つの表にし、NaN だけの判定は「NaN」と呼ぶ | 不要 |
| C-8 | 一部。`v4_speed`・`v6_ablation` の出どころと、v7 の参照 run の場所を文書に書く | `--experiment_name` を効かせるか、削除する (`cli_args.py`) |
| C-9 | いいえ | モデル名の一覧を `models.py` に一本化し、評価先の一覧は別名 (例 `EVAL_TARGETS`) にする。モデル名そのものは変えない |
| C-10 | 一部 (用語集) | 望ましい: `common/rope_model/` の改名 (例 `rope_asset/`)、`RopeStudyEnvCfg.rope_model` → `rope_model_name` |
| C-11 | はい。古い文書の置き場に `archive` を使わない。seed は「シード」と「初期アーカイブ」で書き分ける | 不要 |
| C-12 | はい | 任意: `run_train_chain.sh` → 例 `run_train_queue.sh` (出力の名前は未追跡のものだけ) |
| C-13 | はい。「1 手 = 1 env ステップ」と位相移動を用語集で分ける | 任意: `knot_graph` の `moves` を `topo_moves` に (内部変数)。`--max_moves` は CLI なので据え置き |
| C-14 | はい (ρ(g) との対応を「代わり」と書く) | 任意: `representation.py` の局所変数 `next_edge` を `next_half_edge` に (互換の問題なし) |
| C-15 | はい | 不要 |
| C-16 | はい | 不要 |
| C-17 | はい | 不要 |
| C-18 | はい | 不要 (変えるとタスク ID 18 本とログ名が変わる) |
| C-19 | はい | 不要 |
| C-20 | はい | 不要 (モデル名の一部) |
| C-21 | はい (統一の規則を決める) | 不要 |
| C-22 | はい | 不要 |
| C-23 | — | 登録を減らすならタスク ID が消える (外部からの利用は未確認) |

### 6.3 識別子を変える場合の影響 (タスク ID・ログ・チェックポイント)

前提: rsl-rl 3.0.1 のチェックポイント (`model_<it>.pt`) は `model_state_dict`・`optimizer_state_dict`・`iter`・`infos` だけを保存する
(`.venv/lib/python3.11/site-packages/rsl_rl/runners/on_policy_runner.py:289-301`。追跡外のライブラリ)。クラス名・タスク ID・ログ名は入らない。
読み込みに効くのは、方策のネットワークの構造 (属性名と形) と観測の次元だけ。

| 変える識別子 | タスク ID | ログ (`logs/rsl_rl/`) | チェックポイント | ほかに影響する所 |
|---|---|---|---|---|
| プリセット名 (`Twist` / `Knot` …) | 変わる (30 本 / プリセット) | 変わらない (ログ名にプリセットが無い) | 中身は影響なし | 文書のコマンド (ID の参照 28 本の大半)、`outputs/overhand_script/<ID の短縮>/`・`outputs/divergence_bench/<ID の短縮>_*` (`kt/demos/overhand_script.py:716-717`、`kt/demos/divergence_bench.py:328-330`)、`--rope` の値、USD のファイル名 `rope_<preset>.usd` |
| ID の接頭辞 `Isaac-Twisted-RL-` | 全 166 本が変わる | 変わらない | 影響なし | `scripts/list_envs.py:47` の `"Twisted" in task_spec.id`、上の 2 つの短縮 (`replace("Isaac-Twisted-RL-", "")`)、追跡ファイルで ID の接頭辞が出る 200 行 / 61 ファイル (`cnt -F 'Isaac-Twisted-RL-'`) |
| 版の表記 (`v2-R2` など) | 変わる | 変わる (`knot_tying_v2_r2`) | 影響なし | パッケージ名 `versions/v2_r2/`、`_version_package` |
| `experiment_name` | 変わらない | **変わる** (新しい run は別フォルダ。`--resume` / `--load_run` は新しい名前の下しか探さない) | ファイルの場所が変わるだけ | 追跡ファイルの `logs/rsl_rl/…` 78 行 / 32 ファイル (`cnt -F 'logs/rsl_rl/'`)、`kt/rope_study/make_eval_jobs.py:43` の glob |
| rope_study のモデル名 | 変わる (`RS-<M>`) | 変わる (`knot_tying_rs_<m>`、run 名 `rs_<M>_s42`) | 影響なし | 初期アーカイブ `outputs/archive_seed/rope_study/<M>.pt`・`<M>_knots.pt` のファイル名 (`kt/rope_study/env_cfg.py:202`、`kt/rope_study/run_physics_suite.sh:27`)、`outputs/rope_study/{gate1,gate1_fast,physics,eval}/<M>/`、`make_eval_jobs`・`summarize_eval`・`run_physics_suite.sh` の一覧、報告書の表 |
| 変種の接尾辞 `-RNN` | 変わる (18 本) | 変わる (`_rnn`。ただし実在しない) | 影響なし | なし |
| 方策のクラス `StageGatedActorCritic` | 変わらない | 変わらない | クラス名だけなら影響なし。**属性名を変えると `load_state_dict` (strict) が失敗する** | 設定の `class_name` 文字列 (`kt/versions/v4/agents/rsl_rl_ppo_cfg.py:39`) と `register_with_rsl_rl` |
| 観測・終了・報酬の項の名前 (`knot_stage`、`next_edge`、`diverged`、`stage_up` …) | 変わらない | 変わらない | 項の順序と次元が同じなら影響なし (未確認: Isaac Lab が観測を並べる順序の規則は確かめていない) | Hydra の上書きパス (例 `env.observations.policy.next_edge.params.rule`)、TensorBoard のキー (`Episode_Termination/diverged` など)、評価 JSON のキー、`scripts/rsl_rl/eval_search.py:214` |
| `outputs/archive_seed/` のファイル名 | 変わらない | 変わらない | 影響なし | 初期アーカイブを読むパスがコードに直書き (`kt/versions/v4/env_cfg.py:53`、`kt/rope_study/env_cfg.py:43`) |
| パッケージ・モジュール名 (`common/rope_model/` など) | 変わらない (entry point の文字列は要更新) | 変わらない | 影響なし | `gym.register` の文字列 entry point、import |

## 7. 確認できなかったこと

* 各ログの run がどのプリセットで学習したか。`params/env.yaml` を開いていないので未確認 (ログ名からは分からない: C-2)。
* `knot_tying_v4_speed`・`knot_tying_v6_ablation` のログ名をどうやって付けたか。追跡ファイルに記録が無く、`--experiment_name` は効かない。未確認。
* `logs/rsl_rl/knot_tying_v6_ablation/README.md` の中身 (開いていない)。
* `RopeStudyEnvCfg.rope_model` などがログの `params/*.yaml` に書き出されるか (開いていない)。
* Isaac Lab が観測の項を並べる順序の規則 (項の改名で順序が変わるか)。
* 「文書の archive フォルダ」をどこに作る予定か。`refactor/` の他のファイルを読んでいないので未確認。現時点では追跡ファイルにも履歴にも無い。
* `Simple` / `Stiff` / `Fine` のタスク ID を外部 (このリポジトリの外) で使っているか。
* ρ(g) の論文上の正確な中身 (PDF は読んでいない。explainer は「論文本文に書かれていない」と書く: `paper/explainer/twisted-rl.html:737`)。

## 付録 A. 計数のコマンドと結果

`cnt` は §1 の関数。結果は「行 / ファイル」。括弧内はコード (.py・.sh) / md / html の行数。

```text
K1   cnt -wF 'Twist'                       80 / 28  (28 / 46 / 6)
K2   cnt -F  '-Twist-'                     52 / 21  (25 / 27 / 0)
K3   cnt -F  '`twist`'                     41 / 9   (23 / 18 / 0)
K4   cnt -F  'twist_stiffness'             21 / 8   (20 / 1 / 0)
K5   cnt -F  'ツイスト'                    9 / 6    (7 / 2 / 0)
K6   cnt -F  'Twisted-RL'                  210 / 66 (87 / 114 / 7)
K7   cnt -F  'TWISTED-RL'                  192 / 39 (26 / 67 / 99)
K8   cnt -P  'TWISTED(?!-RL)'              146 / 19 (8 / 36 / 102)
K9   cnt -F  'isaac_twisted_rl'            184 / 59 (102 / 77 / 0)
K10  cnt -P  '(?<!isaac_)twisted_rl'       17 / 12  (12 / 5 / 0)
K11  cnt -F  'TwistedRL'                   7 / 6    (1 / 6 / 0)
K12  cnt -wF 'Knot'                        238 / 57 (79 / 132 / 25)
K13  cnt -F  '-Knot-'                      106 / 32 (42 / 59 / 5)
K14  cnt -F  '`knot`'                      38 / 11  (17 / 21 / 0)
K15  cnt -F  'knot_tying'                  272 / 85 (127 / 145 / 0)
K16  cnt -F  'KnotTying'                   111 / 33 (97 / 14 / 0)
K17  cnt -F  '固結び'                      502 / 68 (167 / 251 / 84)
K18  cnt -iF 'overhand'                    185 / 42 (120 / 60 / 5)
K19  cnt -F  '三葉'                        26 / 10  (15 / 11 / 0)
K20  cnt -F  'trefoil'                     52 / 6   (51 / 1 / 0)
K21  cnt -F  '結び目'                      120 / 38 (34 / 41 / 45)
K22  cnt -wF 'Simple'                      12 / 2
K23  cnt -wF 'Stiff'                       14 / 4
K24  cnt -wF 'Fine'                        12 / 2
K25  cnt -F  '`simple`'                    26 / 9
K26  cnt -F  '`stiff`'                     12 / 5
K27  cnt -F  '`fine`'                      33 / 11
K28  cnt -F  'knot56'                      25 / 6
K29  cnt -wF 'Art28'                       123 / 11 (18 / 105 / 0)
K30  cnt -wF 'Art28LowMu'                  68 / 7
K31  cnt -wF 'Art56'                       49 / 8
K32  cnt -wF 'XpbdC28'                     92 / 12
K33  cnt -wF 'XpbdC56'                     47 / 7
K34  cnt -wF 'XpbdMS28'                    41 / 8
K35  cnt -wF 'XpbdC28LowMu'                55 / 7
K36  cnt -wF 'Fem'                         18 / 5
K37  cnt -wF 'art28'                       0 / 0    (小文字の xpbdc28 なども 0。ログ名は name.lower() で作る)
K38  cnt -F  '剛体鎖'                      98 / 14
K39  cnt -F  'LowMu'                       116 / 7  (モデル名を含む)
K40  cnt -wF 'LowMu'                       6 / 2
K41  cnt -F  'LOW_MU'                      9 / 1
K42  cnt -F  '低摩擦'                      13 / 4
K43  cnt -F  '実物の範囲'                  14 / 3
K44  cnt -P  '(?<!手|階)段(?!階|落)'       868 / 85 (311 / 440 / 117)
K45  cnt -P  '段 ?[0-3]'                   208 / 39
K46  cnt -wF 'stage'                       150 / 24
K47  cnt -F  'STAGE_GOAL'                  14 / 5
K48  cnt -F  'BFS 距離'                    44 / 16
K49  cnt -F  'ラダー'                      116 / 33
K50  cnt -F  '位相クラス'                  27 / 11
K51  cnt -F  'topology_class'              47 / 16
K52  cnt -F  'CLASS_'                      42 / 10
K53  cnt -wF 'knot_class'                  24 / 9
K54  cnt -F  'knot_stage'                  12 / 7
K55  cnt -P  '\bC[0-3]\b'                  198 / 28 (論文の記号などを含む)
K56  cnt -iF 'topology class'              0 / 0
K57  cnt -F  'p-data'                      297 / 56
K58  cnt -F  'p_data'                      197 / 17
K59  cnt -F  'short_p_data'                51 / 6
K60  cnt -iF 'p data'                      0 / 0
K61  cnt -F  'アーカイブ'                  278 / 43
K62  cnt -iwF 'archive'                    46 / 16
K63  cnt -F  'state_archive'               17 / 13
K64  cnt -F  'StateArchive'                24 / 9
K65  cnt -F  'archive_seed'                30 / 12
K66  cnt -F  '初期アーカイブ'              54 / 15
K67  cnt -F  'archive.pt'                  9 / 5
K68  cnt -F  'chain 評価'                  114 / 20
K69  cnt -F  'chain eval'                  6 / 3
K70  cnt -F  'eval_chain'                  34 / 17
K71  cnt -F  'chain_est'                   25 / 12
K72  cnt -F  'chain モード'                18 / 11
K73  cnt -F  'train_chain'                 8 / 4
K74  cnt -F  'ゲート 1'                    137 / 31
K75  cnt -F  'ゲート1'                     1 / 1
K76  cnt -F  'gate1'                       20 / 3
K77  cnt -P  '(?<!ag)gate(?!d)'            23 / 5   (aggregate と gated を除く)
K78  cnt -F  'StageGated'                  38 / 18
K79  cnt -F  '発散'                        567 / 69
K80  cnt -F  'diverged'                    225 / 38
K81  cnt -F  'divergence'                  25 / 15
K82  cnt -F  'NaN'                         76 / 33
K83  cnt -F  '1 手'                        305 / 49
K84  cnt -F  '手数'                        110 / 26
K85  cnt -F  'RL ステップ'                 8 / 6
K86  cnt -F  'env ステップ'                27 / 14
K87  cnt -F  '物理ステップ'                108 / 37
K88  cnt -wF 'moves'                       36 / 11
K89  cnt -F  'max_moves'                   15 / 5
K90  cnt -F  'MOVE_'                       63 / 6
K91  cnt -P  '\bv[0-7]\b'                  1534 / 116
K92  cnt -F  'v2-R2'                       208 / 41
K93  cnt -F  'v2_r2'                       30 / 14
K94  cnt -F  'V2R2'                        27 / 6
K95  cnt -F  'バージョン'                  117 / 24
K96  cnt -F  '版'                          173 / 62
K97  cnt -F  '指 v'                        19 / 2
K98  cnt -F  '物理試験 v'                  7 / 2
K99  cnt -F  'XPBD_FINGER_V2'              5 / 3
K100 cnt -F  '1.2.0'                       3 / 2
K101 cnt -F  'experiment_name'             33 / 17
K102 cnt -iF 'experiment'                  40 / 17
K103 cnt -F  '実験'                        41 / 8
K104 cnt -F  'run_name'                    12 / 7
K105 cnt -F  'プリセット'                  139 / 47
K106 cnt -F  'preset'                      49 / 19
K107 cnt -F  '--rope '                     19 / 13
K108 cnt -iF 'backend'                     75 / 15
K109 cnt -F  'rope_backends'               54 / 10
K110 cnt -F  'rope_backend'                66 / 12
K111 cnt -F  'バックエンド'                11 / 8
K112 cnt -F  'モデル'                      371 / 44
K113 cnt -F  '紐モデル'                    40 / 27
K114 cnt -F  'rope_model'                  62 / 25
K115 cnt -F  'RopeModel'                   13 / 2
K116 cnt -F  'model_'                      60 / 22
K117 cnt -F  'ロープモデル'                1 / 1
K118 cnt -F  'next_edge'                   71 / 16
K119 cnt -F  'NextEdge'                    34 / 12
K120 cnt -F  '次にたどる'                  28 / 14
K121 cnt -F  '最短の次の辺'                5 / 4
K122 cnt -F  'ρ(g)'                        14 / 7
K123 cnt -F  '半辺'                        31 / 7
K124 cnt -wF 'Edge'                        12 / 3
K125 cnt -F  'R2 除去'                     143 / 32
K126 cnt -F  'reduce_r2'                   70 / 16
K127 cnt -F  'Reidemeister'                28 / 12
K128 cnt -F  '_r2'                         141 / 29
K129 cnt -F  '-Play'                       75 / 36
K130 cnt -F  'Play 環境'                   20 / 10
K131 cnt -F  '-RNN'                        19 / 12
K132 cnt -F  'GRU'                         47 / 20
K133 cnt -F  '紐'                          417 / 64 (203 / 199 / 15)
K134 cnt -F  'ロープ'                      763 / 86 (469 / 189 / 105)
K135 cnt -wF 'seed'                        195 / 41
K136 cnt -F  'シード'                      49 / 16
K137 cnt -F  '表現'                        91 / 30
K138 cnt -F  'representation'              10 / 6
K139 cnt -F  'paper/Report.md'             26 / 23
K140 cnt -F  'eval_compare'                2 / 2
```

## 付録 B. タスク ID を抜き出すスクリプト

Isaac Sim も isaaclab も import しない。`gymnasium` を記録するだけの偽物に差し替え、登録モジュール 2 つを実行する。
Phase 4 で変更の前後を比べるときは、同じスクリプトの出力 (TSV) を `diff` すればよい。

```python
import importlib.util, inspect, sys, types

ROOT = "/home/itoyama/work/isaac_twisted_rl/src/isaac_twisted_rl/tasks/knot_tying"
records = []
fake_gym = types.ModuleType("gymnasium")

def register(id, entry_point, kwargs=None, disable_env_checker=False):
    frame = inspect.stack()[1]
    records.append((id, entry_point, kwargs or {}, frame.filename, frame.lineno))

fake_gym.register = register
sys.modules["gymnasium"] = fake_gym

def run(path, modname):
    spec = importlib.util.spec_from_file_location(modname, path, submodule_search_locations=[])
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = modname
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)

# rope_study/__init__.py が読む models.py (純粋な dataclass) を先に入れておく
name = "isaac_twisted_rl.tasks.knot_tying.rope_study.models"
spec = importlib.util.spec_from_file_location(name, f"{ROOT}/rope_study/models.py")
models = importlib.util.module_from_spec(spec); sys.modules[name] = models; spec.loader.exec_module(models)

run(f"{ROOT}/__init__.py", "isaac_twisted_rl.tasks.knot_tying")
run(f"{ROOT}/rope_study/__init__.py", "isaac_twisted_rl.tasks.knot_tying.rope_study")

rel = "/home/itoyama/work/isaac_twisted_rl/"
for id_, ep, kw, fn, ln in records:
    p = "isaac_twisted_rl.tasks.knot_tying."
    print(f"{id_}\t{fn.replace(rel, '')}:{ln}\t{ep}\t"
          f"{kw.get('env_cfg_entry_point', '').replace(p, '')}\t{kw.get('rsl_rl_cfg_entry_point', '').replace(p, '')}")
# 実行: .venv/bin/python list_tasks.py > tasks.tsv   (166 行)
```

---

## メインによる根拠の確認 (2026-10-03)

メインの session が、次の 4 か所を自分で開いて確かめた。誤りは見つからなかった。

| 所見 | 確かめた内容 | 結果 |
|---|---|---|
| C-1 段の向き | `knot_graph.py:548-555` の `stage` は `(Φ * STAGE_GOAL).round()`、STAGE_GOAL = 3 (:558)。`topology_class` の Φ = 1 − d / d_max で d_max = 3 (:273-276 と :553 の docstring)。そのため 段 = 3 − d。一方、`common/mdp/stages.py:1` と `knot_tying/README.md:42` は「段 (固結びへの BFS 距離)」と書いている | 一致 |
| C-2 ログ名がプリセットを含まない | `versions/v4/agents/rsl_rl_ppo_cfg.py:52` は `experiment_name = "knot_tying_v4"`。`knot_tying/__init__.py:97-108` は、同じ版の全プリセットに同じ PPO 設定を登録している | 一致 |
| C-8 `--experiment_name` が効かない | `scripts/rsl_rl/cli_args.py` で `experiment_name` が現れるのは定義の :26 だけ。設定へ反映する処理はない | 一致 |
| C-5 versions の一覧に v7 がない | `versions/__init__.py:6-13` の docstring は v0〜v6 だけ。`versions/v7/` は存在する | 一致 |
