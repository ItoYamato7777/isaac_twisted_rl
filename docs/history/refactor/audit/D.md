# D. 規則と失敗の抽出

- 対象: commit `6f41ef1` (branch `refactor/claude-foundation`)。調査日 2026-10-03。
- 読んだもの: 追跡されている Markdown 23 本、コードのコメント (`git grep` で「注意・罠・必ず・しないこと・黙って・例外が出ない」など)、全 34 commit のメッセージと stat、`logs/`・`outputs/` の一覧と小さなテキスト (シェル・README・`INVALID.txt`・短い `.log` の末尾・各 run の `git/*.diff` の見出し)。
- 実行したもの: `git`・`grep`・`ls`・`wc` だけ。Isaac Sim・GPU は使っていない。プロセスには触れていない。書いたのはこのファイルだけ。
- 用語は C.md の「推奨する正式名称」に揃えた (段 (stage)、位相クラス、chain 評価、ゲート 1 (台本の固結び試験)、初期アーカイブ、ロープ `Knot`、紐モデル、ログ名 (experiment_name) と run など)。
- 略記 (パスはリポジトリのルートからの相対):
  `KT/` = `src/isaac_twisted_rl/tasks/knot_tying/`、`SC/` = `src/isaac_twisted_rl/scripts/`、
  `HO` = `KT/rope_study/HANDOFF.md`、`NA` = `KT/next_action.md`、`RR` = `KT/rope_study/Report_rope_models.md`、
  `R3` = `KT/versions/v3/Result_report_v3.md`、`R6` = `KT/versions/v6/Result_Report_v6.md`、`RS` = `paper/Report_Search.md`。
  `logs/`・`outputs/` のパスは git の管理外 (今の時点の中身)。

---

## 0. 要点

**2 回以上起きた失敗** (hook・skill を作る根拠になるもの。詳細は §4.1):

| 所見 | 一言で | 回数 (数え方は §2) |
|---|---|---|
| D-1 | 自分が起動していないプロセスの kill が権限判定で拒否された。1 回目はその後 Bash がすべて拒否された | 2 セッション (HO:198-203、HO:44-48) |
| D-2 | Kit・FEM が固まり、`timeout` の SIGTERM で止まらない。追跡しているシェル 4 本の `timeout` は、どれも `-k` を付けていない | 3 件以上 (HO:198、HO:152、HO:48/62) |
| D-3 | `env.step()` の処理の順序 (終了判定 → 報酬 → reset → 観測) から来る罠。同じ罠が `KT/rope_study/bench_physics.py:378` に残っている疑い | 3 件 + 疑い 1 |
| D-4 | 最後のチェックポイントは `model_99.pt` だが、道具の既定値と文書は `100` のまま。`make_eval_jobs.py` はファイルが無いと黙って飛ばす | 注意書き 2 か所、古い記述 9 か所 |
| D-5 | 結果を無効にして手で退避した。スクリプトは「出力があれば飛ばす」ので、退避を忘れると古い結果が使われる | 5 回 |
| D-6 | 未コミットのコードで学習した。未追跡のファイルの中身は run のフォルダに残らない | 29 run 中 21 run |
| D-7 | 代理の指標 (学習ログ・学習済み方策での計測) を根拠に、結論を事実として書いた。後で撤回した | 3 回 |
| D-8 | 設定が黙って効かない・黙って壊れる | 8 種 (各 1 回) |
| D-9 | 文書と実装の食い違い。文書の整理の commit を 2 回出したが、まだ 10 件以上残っている | commit 2 回 + 残り 11 件 |
| D-10 | GPU を共有したので、時間の計測が当てにならない。FEM は GPU が混んでいると落ちる | 4 件 |
| D-11 | 長い作業が中断された (コンテキスト・セッションの切断・評価の途中停止)。引き継ぎメモは先頭に節を足して伸び、「最新」の印が古いまま残っている | 中断 4 件 |
| D-12 | 一時的な場所 (Claude の scratchpad) を、残る文書やスクリプトから参照している | 4 か所 |

**損害が大きい規則** (§4.2): D-13 共有資産を上書きしない、D-14 `common/mdp/` の項は追加だけ・既存の版を書き換えない、D-15 他のセッションのプロセスと未コミットの作業に触らない、D-16 `__init__.py` の import と `mdp` フォルダの名前、D-17 評価と学習の条件を揃える (Hydra の上書き、`agent.device`)、D-18 報酬の測り方 (退避の後に測る、γ を揃える)。

**今、規則を守らせる仕組みはほぼ無い。** リポジトリに CLAUDE.md・`.claude/settings.json`・hook・CI・pre-commit・ruff の設定は無い (`.claude/` は空。`git ls-files` にも無い)。利用者の `~/.claude/settings.json` には hooks のキーが無く、permissions は `allow` だけ (deny・ask は無い)。その allow に、別のプロジェクト用の `pkill -f "quick_train_10min.py"` と `Read(//home/itoyama/work/IsaacLab/**)` が入っている (§4.3 D-22)。

---

## 1. 明文化された規則の一覧

列: 損害 = 破ったときの損害、判定 = 機械で判定できるか、仕組み = 今それを守らせる仕組み。

### 1.1 起動・import・コードの構造

| # | 規則 | 理由 | 出典 | 損害 | 判定 | 判定の方法 | 仕組み |
|---|---|---|---|---|---|---|---|
| R-01 | 自動で import される `__init__.py` は docstring だけにする (`isaaclab.sim` などを import しない) | `import_packages` が Kit の起動前に実行し、`list_envs` / `train` が丸ごと落ちる | `KT/README.md:214-220`、`KT/__init__.py:28-30`、`KT/common/__init__.py:23-25`、`KT/versions/__init__.py:17`、`KT/versions/v0/__init__.py:8` (v1〜v7 も同じ文)、`KT/common/mdp/__init__.py:7`、`KT/rope_study/__init__.py:9`、ほか 5 本 | 中 (起動するとすぐ落ちるが、Isaac Sim を起動するまで分からない) | できる | ast で import を推移的にたどる試験 (F.md §6.3 `test_init_imports`) | なし。今は守られている (F.md §6.1 I1) |
| R-02 | 環境の設定は `gym.register` の文字列の entry point で参照する | 実際の import を `gym.make()` まで遅らせる | `KT/README.md:221-222`、`KT/__init__.py:95` | 中 | できる | ast: `entry_point` / `env_cfg_entry_point` が文字列か | なし |
| R-03 | `mdp` というフォルダの名前を変えない。`utils` や `.mdp` を含むパッケージ名を作らない | ブラックリストは部分一致。外れたパッケージは**黙って**登録から消える | `KT/README.md:225-227`、`NA:582-583`、F.md §6.2 の注意 | 高 (エラーが出ずにタスクが消える) | できる | F.md `test_blacklist` | なし (I3 は守られている) |
| R-04 | `common/topology/` と `common/state_archive.py` で `isaaclab` / `omni` / `pxr` を import しない | 単体テストが Isaac Sim なしで走らなくなる | `KT/README.md:231-233`、`KT/README.md:130-131`、`NA:76-77` | 中 | できる | `git grep` (F.md §6.3) か ast | なし (I2 は守られている) |
| R-05 | `rope.data.*` を読むのは `rope_state.py` (と `mdp/`) だけにする。body は名前で解決する | 順序が狂うと**例外が出ないまま** p-data が壊れる | `KT/common/rope_state.py:7`、`:14`、`KT/README.md:233`、`NA:168-170`、`KT/common/mdp/events.py:82` | 高 (結果が黙って壊れる) | 一部 | `git grep "rope.data\."` の出現箇所を許可リストと照合 | 起動時の検算 `check_link_order` (`KT/common/env_cfg_base.py:99`) |
| R-06 | `common/mdp/` の項は追加だけにする。削除・改名はしない。役目を終えた項は `.. deprecated::` で残す | 過去の版の設定がその項を参照している | `KT/README.md:180-183`、`NA:574`、`KT/common/__init__.py:17-21` | 高 (過去の版が起動しない、または再現できない) | できる | F.md `test_mdp_names_additive` | なし。commit の履歴では守られている (F.md §6.1 I4)。未コミットの項の削除は捕まらない (§2 F-09) |
| R-07 | 既存の項の**挙動**を変えたいときは、新しい名前の項を足す (バグ修正は例外) | 過去の版の結果が変わる | `KT/README.md:184-185`、`KT/common/__init__.py:16-21` | 高 | 一部 | 純粋な項はゴールデン試験で固定できる。isaaclab に依存する項は Isaac Sim が要る | なし |
| R-08 | 課題は版で管理する。既存の版を書き換えず、新しい版を足す。1 版で変えるのは 1 つ | 過去の版をいつでも再学習・再評価し、比べられるようにする | `KT/README.md:31-33`、`NA:561-563`、`KT/versions/v0/README.md:18-19`、`KT/versions/v7/README.md:46` | 高 | 一部 | 既存の `versions/vK/` を編集したら警告する (hook)。ただし v4 を変えると v5〜v7 と rope_study に波及する (F.md §5.2) | なし |
| R-09 | 新しい版を足すときは、`_VERSIONS` に足す、ログ名を `knot_tying_vN` にする、README に「v(N-1) から何を・なぜ変えたか」の表を書く | ログ名を共有すると、`--resume` が別の版・別の構造のチェックポイントを黙って読む | `KT/README.md:75-79`、`:175-187`、`KT/versions/v2_r2/agents/rsl_rl_ppo_cfg.py:5-8` | 中 | できる | F.md `test_registry_static`。`versions/__init__.py` に v7 が無い (C-5) | なし |
| R-10 | Hydra の上書きで 1 変数ずつ変えた run は、別のフォルダに置き、上書きを README に書く | 通常の run と混ざらないように | `KT/README.md:80-81`、`logs/rsl_rl/knot_tying_v6_ablation/README.md:1-12` | 中 | 一部 | `--experiment_name` が効かないので (`NA:580`)、ラッパーで振り分ける | なし |
| R-11 | `representation.py` の camelCase のメソッド名は移植元のまま残す | 移植元から後でコピーするときの事故を防ぐ | `NA:133-134`、F.md N-3 | 低 | できる | ruff の N 規則をこのファイルだけ除外する | なし |

### 1.2 MDP・報酬・物理

| # | 規則 | 理由 | 出典 | 損害 | 判定 | 判定の方法 | 仕組み |
|---|---|---|---|---|---|---|---|
| R-12 | 成功の判定は、必ずハンドを退避させた後に測る | 「ハンドで吊って交差を見せかける」方策を学習する | `NA:294-296`、`KT/common/README.md:131-134`、`KT/common/mdp/rewards.py:13` | 高 (偽の成功で学習する) | 一部 | 静的な試験で、成功系の項に高さの条件の params があるかを見る | フェーズの順序 (clear → retreat → relax) と高さの条件 (`KT/common/README.md:132-134`) |
| R-13 | ログに `rope_zmax` と `finger_gap` を出す | 吊っているだけの状態を「交差が残った」と取り違えない | `NA:298-299`、`KT/common/mdp/observations.py:314` | 中 | 一部 | 設定の静的な検査 | 実装済み |
| R-14 | 整形報酬の γ を PPO の `gamma` と必ず一致させる | 一致しないと、整形が最適方策を変えてしまう | `KT/common/mdp/rewards.py:124-132`、`KT/versions/v1/agents/rsl_rl_ppo_cfg.py:24`、`:70`、`KT/versions/v2/agents/rsl_rl_ppo_cfg.py:37`、`:121`、`KT/versions/v3/agents/rsl_rl_ppo_cfg.py:8`、`KT/versions/v1/env_cfg.py:31`、`KT/versions/v2/env_cfg.py:32` | 高 (黙って最適方策が変わる) | できる | 試験: 全版の agent の `gamma` と env の `shaping.params["gamma"]` を比べる (isaaclab の import が要るので、環境の設定を読む試験になる) | 定数 `GAMMA` を共有している (`KT/versions/v1/agents/rsl_rl_ppo_cfg.py:70`) |
| R-15 | ログ専用の報酬項 (常に 0 を返す) の重みを 0 にしない | 重みが 0 だと `RewardManager` が項を呼ばない。ログ・アーカイブが黙って空になる | `KT/common/mdp/archive.py:17`、`:209`、`:475`、`KT/common/mdp/topology_log.py:79`、`:214`、`KT/versions/v2_r2/env_cfg.py:181`、`KT/versions/v3/env_cfg.py:121`、`KT/versions/v4/env_cfg.py:115`、`KT/versions/v6/env_cfg.py:75` (コメント 9 か所)、`KT/versions/v2_r2/README.md:122-128` | 中 (ログが黙って欠ける) | できる | 試験か起動時の assert: ログ専用の項の重みが 0 でないこと | なし (コメントだけ) |
| R-16 | `termination_manager.get_term()` は、その手で reset された行 (`reset_buf` / `dones`) で絞ってから読む。報酬には使わない | `_term_dones` は前のエピソードを終わらせた項を持ち続ける | `KT/versions/v2_r2/README.md:165-167`、`KT/common/mdp/terminations.py:74-81`、`SC/rsl_rl/eval_search.py:208`、`KT/common/mdp/archive.py:429` | 高 (数が 5 倍になった: 本当は 3.4%、`get_term` では 17.5%) | 一部 | `git grep "get_term("` の呼び出しが `reset` / `done` / `terminated` との論理積になっているか (§4.1 D-3) | なし (コメントだけ) |
| R-17 | 物性を変えるときは、計測だけでなく学習で確かめる | 学習済みの方策での計測と、ゼロからの学習での効果が一致しなかった | `NA:40-42`、`KT/versions/v5/README.md:188-189` | 高 (誤った結論) | できない | — | なし |
| R-18 | ソルバの反復を 8 以外にするときは、必ず測り直す | 反復と成否の関係は単調ではない (12 が最悪) | `KT/versions/v4/README.md:221` | 中 | できない | — | なし |
| R-19 | ロープ `Knot` で学習するなら、carry などを延ばした尺を使う | 速く運ぶと発散する (carry 0.35 s で 1 手目に 13/32 が発散) | `KT/common/rope_model/rope_specs.py:492-499`、`NA:524-527` | 中 | 一部 | 設定の試験 (`Knot` の版が `KNOT_DURATIONS` を使っているか) | v3 以降の設定で配線済み (`NA:536-537`) |
| R-20 | ほかのプリセットで学習するなら、「離したのに付いてくる」現象が再発しないかを確かめる。`fine` で学習するなら `sim.dt` を詰める | ヤング率を下げたのは `twist` / `knot` だけ。`fine` だけ発散する | `NA:214-216`、`NA:337-338`、`KT/common/rope_task_cfg.py:13` | 中 | できない | — | なし |
| R-21 | 尺を変えるときは、`episode_length_s` の浮動小数の誤差で手数が 1 増えないかに注意する | `ceil(episode_length_s / step_dt)` で 1 手増えることがある (807 ステップで 13 手) | `KT/versions/v5/README.md:201-204` | 中 | できる | 試験: 版ごとに `ceil` で戻した手数が意図した手数と一致するか | なし |
| R-22 | p-data の記号層 (`p_data`) を、毎ステップ全環境で呼ばない | 4096 環境で 94 ms かかる | `KT/common/rope_state.py:288`、`:295`、`NA:107`、`NA:305-306` | 中 (遅くなる) | 一部 | — | なし |
| R-23 | R1 / R2 / cross の操作は、`copy()` した状態に対して呼ぶ | 元の状態を書き換える経路がある | `KT/common/topology/representation.py:352` | 中 | 一部 | — | なし |

### 1.3 実験・評価

| # | 規則 | 理由 | 出典 | 損害 | 判定 | 判定の方法 | 仕組み |
|---|---|---|---|---|---|---|---|
| R-24 | 版の比較は chain 評価で行う。学習ログの段ごとの成功率や `chain_est` では比べない | 学習ログの値は本番の成功率を表さない | `KT/README.md:85-87`、`NA:34-35`、`R6:166-170`、`KT/versions/v7/README.md:62-63`、`HO:403`、`RR:321` | 高 (誤った結論。3 回起きた: §2 F-07) | 一部 | 報告の数値の出典が `eval_chain_it*.json` かを見る (skill の手順) | なし |
| R-25 | 版の比較は 2 シード以上で行う | シード差 (15〜19 ポイント) が標準誤差 (2 ポイント) よりずっと大きい | `KT/README.md:97-98`、`NA:46-47`、`R6:176`、`RS:30` | 高 (誤った結論) | 一部 | 集計スクリプトが、シードが 1 本のときに警告する | なし |
| R-26 | 学習で物理を Hydra で上書きした run は、評価にも同じ上書きを付ける | 付けないと、違う物理で評価してしまう (エラーは出ない) | `KT/README.md:96`、`R6:229`、`R6:306-307`、`logs/rsl_rl/knot_tying_v6_ablation/README.md:14` | 高 (黙って誤った評価になる) | できる | 評価スクリプトが、run の `params/env.yaml` と評価の設定の差を出す | なし |
| R-27 | 別の GPU で回すときは、`--device cuda:N` と `agent.device=cuda:N` を揃える | `--device` は環境だけを移す。PPO は `cuda:0` に残る (`SC/rsl_rl/train.py:121`) | `KT/versions/v4/README.md:285-286`、`R6:204` | 中 (GPU の取り合い・遅くなる) | できる | hook: Bash のコマンドの `--device` と `agent.device` を照合する。または `train.py` で揃える | なし |
| R-28 | Isaac Sim を起動するスクリプトは `python -u` で回す | 出力をファイルへリダイレクトすると、ブロックバッファリングで数分間なにも出ず、ハングしたように見える | `KT/README.md:242-243`、`KT/demos/generate_assets.py:39`、`HO:301`、`KT/rope_study/README.md:31`、`RR:791` | 中 (時間の浪費。止める判断を誤る) | できる | hook でコマンドを照合する。または `.claude/settings.json` の `env` に `PYTHONUNBUFFERED=1` を置く | なし (文書のコマンド例には `-u` が付いている) |
| R-29 | テストは `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` を付けて回す | `/opt/ros/jazzy` の `launch_testing` を読み込んで `ModuleNotFoundError: lark` で落ちる | `KT/README.md:234-239`、`README.md:107`、`HO:260`、`HO:302`、`RR:797`、`tests/*.py` の docstring 10 本 | 低 (すぐ落ちるので気付く) | できる | ラッパーか CLAUDE.md の 1 行。F.md:608 は `uv run python -m pytest` の形を推す (venv の取り違え、F.md F-6) | なし |
| R-30 | 変更したら必ずテストを通す | `tests/` が期待される動作の仕様 | `KT/README.md:234`、`NA:70` | 中 | できる | hook (Stop の前、または commit の前に pytest) | なし (CI も無い: F.md §7) |
| R-31 | lint と format を通す (ruff v0.14.10、line-length 120) | — | `KT/README.md:240-241`、`HO:261`、`HO:303`、`RR:798` | 低 | できる | ruff。ただし `.pre-commit-config.yaml` が存在しない (F.md 要点 3) | なし |
| R-32 | docstring は日本語で書き、「なぜそうしたか」を書く。移植元と挙動を変えたら、理由と検証の結果を残す | — | `KT/README.md:244-245` | 低 | 一部 | 日本語が含まれるかだけは判定できる | なし |
| R-33 | rsl-rl は 3.0.1、isaaclab は 2.3.0 (`.venv`)。`/home/itoyama/work/IsaacLab` の例をそのまま写さない | rsl-rl 5.x 前提で API が違う。`rnn_hidden_size` のように黙って効かない名前もある | `KT/README.md:246-247`、`KT/versions/v2/README.md:203-222`、`HO:300` | 中 | 一部 | permissions で IsaacLab の Read を拒否する。ただし今は利用者の設定で許可されている (§0) | なし |
| R-34 | `rnn_type` は学習の開始ログで `GRU(...)` / `LSTM(...)` を目で確かめる | `"gru"` 以外の文字列は黙って LSTM になる | `KT/versions/v2/README.md:213-217`、`KT/versions/v2/agents/rsl_rl_ppo_cfg.py:160-169`、`:226` | 中 | できる | 試験: `rnn_type in {"gru", "lstm"}` | なし |
| R-35 | 学習を回す前に、ゲート 1 (台本で固結びが結べるか)、`num_envs` のスループット、位相クラスごとの発散率を確かめる | ゲート 1 が通らなければ、報酬をどう設計しても学習できない | `NA:465-482` | 中 | できない | skill の手順にする | なし |
| R-36 | ゲート 1 は必ずまっすぐなロープから始める | 学習の設定のままだと、初期アーカイブがあるときに途中の形から始まる | `KT/demos/overhand_script.py:725`、commit `a9b02b3` | 高 (結果が無効になった: §2 F-12) | できる | コードで固定済み | 実装済み (`a9b02b3`) |
| R-37 | 物理試験のキューを回している間は `xpbd_rod.py` を変えない。変えたら、どの結果が変更前・変更後かを記録する | 変更は、その後に起動したプロセスにだけ効く | `HO:232`、`HO:369-370`、`KT/rope_study/README.md:111-112` | 中 (結果が混ざる) | 一部 | 出力にコードの版 (git HEAD と diff のハッシュ) を書く (§4.1 D-5) | なし |
| R-38 | 無効にした結果は、退避して README を置く。run には `INVALID.txt` を置く | 比較に混ぜない | `HO:132`、`HO:381`、`RR:856-858`、`KT/rope_study/README.md:89` | 中 | 一部 | `make_eval_jobs.py` は `INVALID.txt` のある run を飛ばす (`KT/rope_study/make_eval_jobs.py:44`) | 一部あり (評価の一覧だけ) |
| R-39 | GPU を共有しているときの時間は参考値にとどめる。速度の比較は、GPU が空いてから測り直す | 共有すると 1 物理ステップが 50 ms 前後になる | `HO:156`、`HO:304-305`、`RR:331`、`RS:250` | 中 (誤った結論) | できない | 時間の記録に、そのときの GPU の負荷を残すことはできる | なし |
| R-40 | `next_action.md` の節番号を変えない | コードの docstring が節番号で参照している | `NA:11-12`、commit `32c9f93` のメッセージ | 低 | できる | 試験: コード中の `next_action.md N.N` が見出しに存在するか | なし |
| R-41 | 厳密に再現するときは、基準の commit まで戻る (v0 は README に基準の commit、ほかの版はその版を足した commit) | `common/` を共有しているので、後の版の変更が前の版にも効く | `KT/README.md:169-173`、`KT/versions/v2/README.md:263-265`、`KT/versions/v0/README.md:16` | 高 (再現できない) | 一部 | run の `git/*.diff` を見る (§2 F-09) | rsl_rl が run ごとに `git/*.diff` を保存する。ただし未追跡のファイルの中身は入らない。v0 の基準の commit `1f5693f` はこのリポジトリに無い |

### 1.4 共有の資源・プロセス・他のセッション

| # | 規則 | 理由 | 出典 | 損害 | 判定 | 判定の方法 | 仕組み |
|---|---|---|---|---|---|---|---|
| R-42 | 別のセッションのプロセスを止めない | GPU 2 枚を別のセッションと共有している | `HO:101`、`HO:171`、`HO:294` | 高 (他の実験を壊す) | 一部 | permissions で `kill` / `pkill` を ask にする。ジョブの起動元を記録するラッパー | 自動の権限判定が拒否した (`HO:30`、`HO:201`)。意図した仕組みではない |
| R-43 | 別のセッションの未コミットの作業に触らない | — | `HO:296-297` | 高 | 一部 | 書き込み先のパスと `git status` の差分を照合する hook。git worktree で分ける | なし |
| R-44 | `KT/common/rope_model/data/rope_knot.usd` と `outputs/archive_seed/overhand_28links.pt` を上書きしない。ロープの USD を作り直すときは `--rope <名前> --only rope` で対象を絞る | v4〜v7 が使う。どちらも git の管理外で戻せない | `HO:298-299`、`KT/rope_study/README.md:101-105`、`RR:800`、`RR:845` | 高 (前提の資産が変わる。データの喪失) | できる | PreToolUse hook: 書き込み先のパスを照合する。`generate_assets.py` を `--rope` なしで呼んだら拒否する | なし |
| R-45 | 起動前に `nvidia-smi` で空きメモリを確かめる (学習 1 本 ≈ 7.8 GB、128 環境の試験 ≈ 2.5 GB) | 共有している | `HO:294-295`、`KT/rope_study/README.md:31-32`、`RR:791` | 中 | できる | 起動用のラッパーで確かめる | なし |
| R-46 | 1 GPU に学習は 2 本まで | 相乗りで 1 iteration が約 2 倍になる | `HO:101`、`HO:190`、`HO:396`、`RR:312`、`RR:330`、`R6:178` | 中 | できる | ラッパーで、`nvidia-smi` に出ている学習のプロセスの数を数える | なし |
| R-47 | `pkill -f <パターン>` を使わない。PID を確かめてから `kill <PID>` | パターンを含む自分のシェルまで殺す | `HO:150-151` | 中 | できる | permissions で `Bash(pkill -f*)` を deny | なし (利用者の設定には逆に `pkill -f` の allow がある) |
| R-48 | Kit・FEM が固まったら `timeout -k` か `kill -9` を使う | 固まった FEM には SIGTERM が効かない | `HO:31`、`HO:152` | 中 (GPU を占有し続ける) | できる | 試験・hook: シェルスクリプトの `timeout` に `-k` があるか | なし (追跡しているシェルの `timeout` は 4 か所ともに `-k` なし: `KT/rope_study/run_eval_queue.sh:22`、`KT/rope_study/run_physics_suite.sh:26`、`:32`、`:37`) |
| R-49 | 長い待ちには `sleep` を使わず、Bash の `run_in_background` と `until` のループか、Monitor を使う | — | `HO:155` | 中 (セッションの時間を浪費する) | 一部 | Claude Code が前景の `sleep` を止める | 一部あり (Claude Code の側) |
| R-50 | `cd X && nohup ... &` は cd ごとバックグラウンドに行く。続くコマンドの cwd は変わらない | — | `HO:32` | 低 | 一部 | — | なし |

---

## 2. 実際に起きた失敗・罠の一覧

回数の数え方: 「記録 N 件」は、別の事象として書かれた出典の数。同じ事象を複数の文書が書いているときは 1 件と数える。数えられないものは「1 回 (出典 1 か所)」か「記録なし」と書いた。

| # | 失敗の内容 (何が起き、何で気づき、どう直したか) | 出典 | 発生回数と数え方 | 機械で防げるか | 防ぐ仕組みの候補 | 損害 |
|---|---|---|---|---|---|---|
| F-01 | **kill が権限判定で拒否された。** (a) 2 回目のセッションで、前のセッションが起動して終了時に固まった `generate_assets.py` (PID 1396214) を `kill -9` しようとした。「他の作業への干渉」として拒否され、その後は `date` を含むすべての Bash が拒否された。止めたのは 1 回目のセッション。(b) 4 回目のセッションで、無効になった学習 2 本 (PID 1586101、1655235) と固まった FEM (PID 1635316) を止められず、ユーザーが止めた | `HO:198-211`、`HO:44-48`、`HO:55`、注意書き `HO:30`、`HO:151` | **2 セッション** ((a) と (b))。注意書きは 2 か所 | 一部 | permissions (`kill` を ask)、ラッパー (起動したジョブの PID を記録する)、CLAUDE.md (自分の記録に無い PID は止めず、ユーザーに渡す) | 時間: (a) のセッションの残り (17:58〜18:50 頃) は Read / Edit / Write だけ。テストと lint を回せなかった (`HO:256`)。(b) 無効な学習 2 本が最後まで GPU を使った (`outputs/rope_study/xpbd_world_coords/README.md:19-20`) |
| F-02 | **Kit・FEM が固まり、`timeout` の SIGTERM が効かない。** `generate_assets.py --rope knot56` は USD を書いた後、Kit の終了で固まった。`bench_cantilever_art` も終了で固まった。FEM は GPU が混んでいると CUDA illegal memory access で固まった。直し方: `timeout -k` か `kill -9` を使う (規則にしただけ) | `HO:198-200`、`HO:152`、`HO:31`、`HO:48`、`HO:62`、`RR:559`、`outputs/rope_study/fem_crash_32env/gate1_Fem.log` の末尾 (`Mem free failed with error code 700`)、`outputs/rope_study/logs_suite_fem_gpu1.txt` (開始の 1 行だけ) | **記録 3 件以上**: Kit の終了時 2 件 (generate_assets、bench_cantilever_art)、FEM 1〜2 件 (`HO:48` と `HO:62` が同じプロセスかは未確認)。ほかに傍証: `outputs/rope_study/diag_scripts/small_queue.sh:25` の FEM のスモークテストは `timeout 1800` なのに、23:11:34 に始まり 00:07:42 に終わった (`outputs/rope_study/logs_small_queue.txt` の末尾) | できる | ラッパー (`timeout -k`、PID の記録)、シェルの `timeout` に `-k` を求める試験 | GPU のメモリを占有し続けた (GPU0 約 340 MB・GPU1 約 280 MB: `HO:200`。FEM 3.5 GB: `HO:48`)。F-01 の引き金になった |
| F-03 | **出力のバッファリング。** リダイレクトすると、数分間なにも出ずにハングしたように見える | `KT/README.md:242-243`、`KT/demos/generate_assets.py:39` | **記録なし** (日付のある事象の記録は無い。現象の説明が規則の理由として書かれているだけ) | できる | `.claude/settings.json` の `env` に `PYTHONUNBUFFERED=1`、または hook | 時間 (推定。記録なし) |
| F-04 | **標準出力のログが残らず、最後のチェックポイントも保存されなかった。** v3 の run は iteration 159 で PhysX の "Patch buffer overflow" により止まったが、`model_159.pt` は無く、run のフォルダに標準出力のログも無い | `R3:30-31`、`R3:176` (「標準出力をファイルに残し」) | 1 回 (出典 1 か所) | できる | 学習の起動ラッパーで、標準出力を run のフォルダへ `tee` する | 最後の保存 (150) から 9 iteration ぶんの学習。原因の調査は commit メッセージ頼みになった |
| F-05 | **`model_99.pt` が最後のチェックポイント。** `--max_iterations 100` は iteration 0〜99 なので、最後は `model_99.pt`。v4〜v6 の run は 400 の予定を途中で止めたので `model_100.pt` がある (`R6:205`)。rope_study の道具と文書は `100` のまま。`make_eval_jobs.py` の既定は `--iters 50,100` (`:51`) で、ファイルが無いと**黙って飛ばす** (`:78-80`、`:90-91`) | 注意書き `HO:29`、`HO:56`。古いままの記述: `KT/rope_study/README.md:60`、`:65`、`:70`、`KT/rope_study/eval_robust.py:5`、`KT/rope_study/make_eval_jobs.py:3`、`:51`、`HO:143`、`HO:398`、`RR:321`、`RR:323`。実物: `ls logs/rsl_rl/knot_tying_rs_*/*/` は 10 run すべて `model_99.pt` で終わる | HO で「踏んだもの」として **2 か所**。古い記述は **9 か所** (`make_eval_jobs.py:3` と `:51` を 2 と数え、`RR:321` と `:323` も 2 と数えた) | できる | `make_eval_jobs.py` が、無いチェックポイントを警告し、`last` を最後のチェックポイントに解決する (コードと試験) | 評価が黙って欠ける。文書のコマンドどおりに打つと転移の評価 (`max(iters)`) が 0 本になる |
| F-06 | **GPU の取り合い。** 別のセッションと GPU 2 枚を共有した。時間の計測が当てにならない。FEM は混んでいると落ちる。相乗りで 1 iteration が約 2 倍。見積り 200 s/it に対して、実測は 390〜530 s/it | `HO:304-305`、`HO:156`、`HO:62`、`RR:559`、`R6:178`、`HO:93`、`HO:164-165`、`HO:396` | **記録 4 件** (FEM の固まり `HO:62`、時間が当てにならない `HO:304`、相乗りで 2 倍 `R6:178`、見積りと実測のずれ `HO:396` 対 `HO:93`) | 一部 | 起動ラッパー (空きメモリ、1 GPU に 2 本まで、そのときの GPU の負荷を記録する) | 時間。速度の比較を測り直す必要がある (`RR:331`) |
| F-07 | **評価の数え方の間違い。** (a) `eval_compare.py` は step ごとのログの平均なので、早く終わるエピソードを重く数えた → `eval_chain.py` (各環境の最初の 1 エピソード) で置き換えた。(b) `get_term("diverged")` は前のエピソードの値を残す (本当は 3.4%、`get_term` では 17.5%) → `mdp.diverged_now`。(c) 学習ログの `chain_est` と段ごとの成功率は本番の成功率を表さない。(d) `overhand_script.py --replay_actions` は、v3 以降の reset を「発散」と数える | (a) commit `eccce81`。(b) `KT/versions/v2_r2/README.md:165-167`、commit `2192c33`。(c) `KT/versions/v4/README.md:307`、`R6:166-170`、`R6:273-276`。(d) `KT/versions/v5/README.md:154-156` | **確定 4 件** (a〜d)。ほかに**疑い 1 件**: `KT/rope_study/bench_physics.py:378` の `d = tm.get_term("diverged").clone()` は reset で絞っていない (§4.1 D-3) | 一部 | 試験 (`get_term` の呼び出しの形)、skill (評価の数え方の手順) | 誤った結論。(a) の数え方で出した過去の数値は残っていない (未確認) |
| F-08 | **整形報酬の検証の罠。** `knot_class_shaping` の正しさを、`env.step()` が返った後の位相クラスで確かめようとした。reset の後の値を読んでしまい、最大 0.455 ずれた。終了した環境が無いステップでもずれた (抽出器のキャッシュ)。報酬関数を差し替えて、計算するその瞬間に Φ を捕まえるように直したら、誤差は厳密に 0 になった | `NA:447-463`、結果は `KT/versions/v1/README.md:108` | 1 回 (出典 1 か所)。同じ根 (step の順序) の罠は F-07 (b) と、抽出器のキャッシュ (`NA:351-353`、`KT/common/README.md:119-127`) にもある。合わせて 3 件 (§4.1 D-3) | 一部 | `.claude/rules` (mdp と評価を触るときの注意)、試験 | 時間 (検証のバグを報酬のバグと取り違えるおそれ) |
| F-09 | **再現できない、または再現の材料が欠けている。** (a) 29 run 中 21 run が、未コミットの `.py` の変更の上で学習した (各 run の `git/*.diff` の `git status`)。rsl_rl の diff には、未追跡のファイル (`rope_backends/`、`rope_study/`、`versions/v7/`) の中身が入らない (`logs/rsl_rl/knot_tying_rs_xpbdc28/2026-10-03_00-08-33_rs_XpbdC28_s42/git/isaac_twisted_rl.diff` の `diff --git` は追跡しているファイル 3 本だけ)。(b) 切り分けの run `abl_toP_s42` は、未コミットの報酬項 `stage_time_out` と「当時の v7」で学習した。その後、項はコードから削除され、v7 は別の中身になった。ログは `knot_tying_v7/` から `knot_tying_v6_ablation/` へ移された。(c) v0 の基準の commit `1f5693f` はこのリポジトリに無い。v0 と v2 の run は別のリポジトリのパス (`src/isaac_so_arm101/...`) で記録されている。(d) 同じ状態・同じ手でも、手の後の段が約 3 割変わる (物理の非決定性。解釈) | (a) `logs/rsl_rl/*/*/git/*.diff` (付録 A-1)。(b) `R6:226`、`R6:288`、`KT/versions/v7/README.md:77-81`、`logs/rsl_rl/knot_tying_v6_ablation/README.md:11`、`HO:292`、run の diff (`stages.py` が modified)。(c) `KT/versions/v0/README.md:16`、`git cat-file -t 1f5693f` が失敗、`logs/rsl_rl/knot_tying_v0/2026-09-12_13-51-53/git/*.diff`。(d) `RS:20-21`、`SC/rsl_rl/README.md:167-174` | (a) **21 / 29 run** (付録 A-1 のコマンドで数えた。`eval_chain.py` だけが未追跡の run は除外)。(b)〜(d) 各 1 回 | (a) できる | hook かラッパー: 学習・評価の起動時に作業ツリーが汚れていたら警告し、未追跡の `.py` を含む完全な patch (`git stash create` 相当か `git diff` + 未追跡ファイルの tar) を run のフォルダに保存する | 再現できない run が残る。(b) は規則 R-06 を、commit の外で破ったことになる |
| F-10 | **文書と実装の食い違い。** 整理の commit を 2 回出した (`32c9f93`「重複・矛盾・古い記述を解消」、`ccae2ae`「系譜に揃える」)。それでも残っているもの: (1) 改名した `paper/Report.md` への参照が 24 ファイル 29 行 (F.md 要点 8) (2) `.pre-commit-config.yaml` が無い (F.md 要点 3) (3) 段を「固結びへの BFS 距離」と書くが、向きが逆 (C-1、`KT/README.md:42`) (4) `versions/__init__.py` に v7 が無い (C-5) (5) `--experiment_name` が効かない (`NA:580`、C-8) (6) `paper/Report_Paper.md:403` は推論時の探索 (P2-4) を「未実装」と書くが、`72d18df` で実装済み (`RS:275` が更新を予定) (7) `KT/versions/v7/README.md:8` のログ `knot_tying_v7/` が存在しない (8) F-05 の `model_100` (9) `eval_compare.py` への参照 (`SC/rsl_rl/eval_chain.py:8`、F.md R-13) (10) 「ロープ 5 種」など、プリセットの数の揺れ (C.md 2.1 の preset の行) (11) `KT/versions/v7/README.md:28` が「止めていたのは 0.4 m/s だった」と断定し、同じ文書の `:35` が「確かめられなかった」と書く | 各項目に記載 | 整理の commit **2 回**。残っている食い違い **11 件** (この表で数えた) | 一部 | 試験 (参照先のファイル・節・`_VERSIONS` と `versions/` の一致)、CLAUDE.md (文書には現状を書き、経緯は 1 か所に) | 誤った手順で作業する。誤った結論を引用する |
| F-11 | **版の継承元を間違えた。** v5 が v3 の `KnotTyingV3EnvCfg` を継承していて、v4 の段ごとのエージェントなどが欠けていた。v4 の学習比較を見て気付き、v4 を継承するように作り直した。v3 ベースのときの学習結果の記述は README から消した | commit `881dc8d` (メッセージに「誤って」)、`ccae2ae` | 1 回 (commit 1 か所) | できる | 試験: 版ごとの継承元の表 (F.md §5.2) と実際の基底クラスを比べる | v3 ベースの v5 / v6 の計測と学習が無駄になった (量は未確認) |
| F-12 | **ゲート 1 が途中の形から始まっていた。** `overhand_script.py` は reset に学習の設定を使っていた。初期アーカイブがあると、段 0〜2 の途中の形から始まる。`run_physics_suite.sh` は遅い尺がアーカイブを書いた後に学習の尺を回すので、学習の尺のゲート 1 は全モデルで無効だった。`XpbdC28` の学習の初期アーカイブも、この無効な結果から作られた。まっすぐなロープに固定して直し、やり直した | `HO:105-107`、`HO:133-134`、`RR:371`、`outputs/rope_study/contaminated_start/README.md:1-9`、commit `a9b02b3`、`outputs/rope_study/logs_gate1_redo.txt` (22:39〜23:16) | 1 回 (影響は全モデル) | できる (直した) | 試験はコードで固定済み。手順としては「試験の前に入力の状態を出力に書く」 | 誤った結論 (報告書の中間の表)。やり直しに約 37 分。`XpbdC28` の学習は無効なアーカイブのまま続けた (`HO:133-134`) |
| F-13 | **XPBD を world 座標で解いていた。** float32 の丸めで、原点から遠い環境では床とロープどうしの動摩擦が消えた (4096 環境の約 88%)。低摩擦のモデルが床を滑り続けたことから気付いた。局所座標に直し、学習 2 本を無効にし、物理試験をやり直した | `HO:38-46`、`RR:315`、`RR:720`、`outputs/rope_study/xpbd_world_coords/README.md:1-20`、`logs/rsl_rl/knot_tying_rs_xpbdc{28,56}/2026-10-02_*/INVALID.txt` | 1 回 | 一部 (単体テストを 2 件足した: `HO:41`) | 試験 (原点を遠くに置いた環境で摩擦の停止距離を比べる) | 学習 2 本、XPBD 4 モデルの物理試験、v4 の方策の XPBD 上の評価が無効になった |
| F-14 | **PhysX の接触バッファがあふれ、接触が黙って捨てられた。** v3 の学習中の発散の約半分がこれだった (旧バッファ 4.55% / 手、新バッファ 2.20%)。上限を上げて直した | commit `b345065`、`KT/versions/v5/README.md:65-80`、`R3:187-197` | 1 回 | 一部 | ラッパー: 学習のログを "Patch buffer overflow" で grep して警告する | 誤った結論のおそれ (発散対策の効果と取り違える: `KT/versions/v5/README.md:80`)。v3 の run が止まった (F-04) |
| F-15 | **設定が黙って効かない・黙って壊れる。** (1) `--experiment_name` が効かない (2) `solver_position_iterations` は Hydra から効かない (3) `rnn_type` は `"gru"` 以外が黙って LSTM になる (4) 重み 0 の報酬項は呼ばれない (v2 の `diverged_class*` は一度も計算されなかった) (5) `rope_diameter` の既定値 0.02 が黙って残った (6) `bind_physics_material` は正規表現を受け付けず、取りこぼすとロープが指をすり抜ける (7) body の順序が狂うと p-data だけが壊れる (8) 接触バッファのあふれ (F-14) | (1) `NA:580` (2) `KT/versions/v4/README.md:277` (3) `KT/versions/v2/README.md:213-217` (4) `KT/versions/v2_r2/README.md:122-128` (5) `KT/versions/v2/README.md:271-276`、`KT/common/rope_task_cfg.py:98-106` (6) `NA:355-356`、`KT/common/mdp/events.py:74` (7) `NA:168-170` (8) F-14 | **8 種、各 1 回** (実際に踏んだと書いてあるのは (4)・(5)・(8)。ほかは調べて分かった性質) | 一部 | 試験と起動時の assert (§4.1 D-8) | 誤った学習・誤ったログ |
| F-16 | **発散した手の報酬で学習が壊れた。** ロープが手の途中で吹き飛び、把持距離 97 m → `grasp_shaping` −107.8、value loss 7.2e9 になった。発散の判定を強め、発散した手の報酬を 0 にした | commit `2192c33`、`KT/versions/v2_r2/README.md:132-163` | 1 回 | 一部 | 試験 (発散した手の報酬の上限) | v2-R2 の最初の run は比較に使えない (`KT/versions/v2_r2/README.md:136`) |
| F-17 | **代理の指標を根拠にした結論。** (a) v4 は学習ログの `chain_est` が伸び続けたが、本番は伸びていない (b) v5 のめり込み解消速度 0.4 m/s は、学習済みの方策での計測では良かったが、ゼロからの学習では段 2 → 固結びを学べなくした (c) v6 の切り分けで、学習ログから「原因はめり込み解消速度」と結論し、v7 を作った。chain 評価では確かめられなかった | (a) `KT/versions/v4/README.md:307` (b) `KT/versions/v5/README.md:188-189`、`NA:40-42` (c) `R6:273-276`、`KT/versions/v7/README.md:28` 対 `:35`、commit `95ac19a` のメッセージ (「止めていたと分かった」) | **3 回** ((a)〜(c)) | 一部 | skill (版の比較の手順: chain 評価・2 シード以上・同じ Hydra の上書き)、CLAUDE.md | 誤った結論。v7 は作ったが、確認の学習は保留 (`KT/versions/v7/README.md:73`) |
| F-18 | **結果を無効にして手で退避した。** スクリプトは「出力があれば飛ばす」(`KT/rope_study/run_physics_suite.sh:25`、`:31`、`:36`、`KT/rope_study/run_eval_queue.sh:18`) ので、直した後は古い出力を手で退避する必要があった | `outputs/rope_study/contaminated_start/`、`finger_v1/`、`xpbd_world_coords/`、`fem_crash_32env/` (それぞれ README か log)、片持ち梁 `xpbd_before_elastica_fix.json` (`HO:185`)、`HO:132` (「出力は退避してから」)、`HO:381` | **5 回** (退避先 5 つ) | できる | ラッパー: 出力にコードの版 (HEAD と diff のハッシュ) を書き、版が一致するときだけ飛ばす | 退避を忘れると、古い物理の結果を黙って使う (今回は起きていないことを確認できていない) |
| F-19 | **数値・解析の小さなバグ。** (a) `elastica` の射撃法が大たわみで偽の根に落ちた (b) `Art28` の物理ベンチが NaN の環境で `ValueError` (c) XPBD のデバッグで見つかった 6 件 (λ を積み上げていない、など) | (a) `HO:118`、`HO:184-185`、`HO:239-240` (b) `HO:234-238` (c) `HO:354-359` | (a)・(b) 各 1 回。(c) 6 件 | 一部 | 単体試験 (XPBD は試験あり: `tests/test_rope_backends.py`) | やり直し (片持ち梁の再計算、物理ベンチの再実行) |
| F-20 | **作業・評価の遅れと中断。** (a) 1 回目のセッションがコンテキストの都合で中断 (b) 10:40 にセッションが一度切れて再開 (c) chain 評価を 09:00 に始める予定が 11:14 になった (d) 推論時探索の評価を途中で止めた (10 本中 4 本が完走) (e) seed 43 の評価を途中で止めた | (a) `HO:3` (b) `HO:53` (c) `R6:181` (d) `RS:8-10`、`RS:102-108` (e) `HO:21-22`、`outputs/rope_study/eval_s43_stopped/` | **記録 5 件** ((c) は原因の記述が無い) | 一部 | skill (引き継ぎの手順)、CLAUDE.md (作業の区切り方) | 時間。v7 の評価が無いまま (`RS:8-10`) |
| F-21 | **環境の不整合。** (a) GPU ドライバのパッケージが更新され、カーネルモジュールと合わず `nvidia-smi` が失敗した (b) `uv run pytest` が別のリポジトリ (`isaac_so_arm101`) の venv で走っていた | (a) `R3:170` (b) F.md F-6、F.md:598-608 | 各 1 回 | (b) できる | (b) CLAUDE.md の検証コマンドを `uv run python -m pytest` にする (F.md:608) | (a) 再開の前に再起動が要った (b) 試験の結果を別の依存で出していた |
| F-22 | **Claude Code の作業ファイルを commit した。** `.claude/scheduled_tasks.lock` が最初の commit から入り、各 run の diff に modified / deleted として現れ、最後に削除した | commit `5fd7e3c`、`6f41ef1`、`logs/rsl_rl/knot_tying_v4/2026-10-01_14-08-32_3h_compare/git/*.diff` | 1 回 | できる | `.gitignore` に `.claude/*.lock` (または `.claude/` の中の作業ファイル) | 小さい (run の diff が汚れる) |

---

## 3. Claude Code の作業に固有の失敗

Claude が起こしたと読める根拠: HANDOFF は Claude のセッションの引き継ぎメモ (「1 回目のセッション」「scratchpad」の語)。34 commit 中 20 commit に `Co-Authored-By: Claude` がある (`git log` で数えた)。

| # | 失敗 | 判断の根拠となる文 (引用) | 出典 | 回数 | 防ぐ仕組みの候補 |
|---|---|---|---|---|---|
| CC-1 | 自分が起動していないプロセスを kill しようとした | 「これを `kill -9` しようとしたところ自動の権限判定に「他の作業への干渉」として拒否され、**その後は `date` を含むすべての Bash が拒否された**」。「前のセッションが起動したプロセスの kill は権限判定で拒否される。ユーザーに PID を伝えて止めてもらう」 | `HO:201-203`、`HO:30`、`HO:44-45` | 2 セッション。`HO:44` の PID 1655235 (run の開始 23:42) は 4 回目のセッション (23:40〜、`HO:34`) が起動した可能性があり、「自分が起動していない」と言い切れるかは未確認 | permissions (`kill` を ask)、ラッパー (PID の記録)、CLAUDE.md |
| CC-2 | `pkill -f` で自分のシェルを殺した | 「`pkill -f <パターン>` はパターンを含む自分のシェルまで殺す (コマンドが途中で止まった)」 | `HO:150` | 1 回 | permissions で deny |
| CC-3 | 長い待ちを前景の `sleep` で回した (読み取り) | 「長い待ちは `sleep` ではなく Bash の `run_in_background` + `until` ループか Monitor」。「注意 (3 回目で踏んだもの)」の節にある | `HO:148`、`HO:155` | 1 回 (出典 1 か所。実際に何分待ったかは記録なし) | Claude Code が前景の `sleep` を止める (今は環境の側にある) |
| CC-4 | `cd X && nohup ... &` で cwd を取り違えた | 「`cd X && nohup ... &` は cd ごとバックグラウンドに行くので、続くコマンドの cwd は変わらない」。「注意 (このセッションで踏んだもの)」の節にある | `HO:27`、`HO:32` | 1 回 | ラッパー (絶対パスで起動する) |
| CC-5 | 引き継ぎメモを先頭へ節を足して伸ばした。「最新」「ここから読む」の印が古いまま、3 つ並んでいる | 「## 00000. 最終状態 … ← 最初にここを読む」、「## 00. 3 回目のセッション … ← 最新」、「## 0. 2 回目のセッションの状況 (次のセッションはここから読む)」。節の番号は 0 → 00 → 000 → 0000 → 00000 と増え、全体で 439 行 | `HO:8`、`HO:158`、`HO:194` | 印の食い違い 2 か所 (`HO:158`、`HO:194`) | skill (引き継ぎメモの書式: 「現在の状態」を上書きし、経緯は別のファイルへ) |
| CC-6 | 古い数字・古い計画が文書に残った | `HO:143` の「`--iters 50,100`」と `HO:398` の「iteration 50 と 100」が、`HO:29` の「最後のチェックポイントは `model_99.pt`」と同じ文書に並んでいる。報告書は「§0・§8・§9・§10・§12 は 18:50 時点の記述が残っている」と自分で書いた。`RR:367` は「この節は経過の記録」と断っている | `HO:136`、`HO:143`、`HO:398`、`RR:367`、F-05 | 古い記述 9 か所 (F-05) | 試験 (F-05 の道具側)、skill (報告書を仕上げるときの確認項目) |
| CC-7 | 代理の指標で出した結論を、commit メッセージと README に事実として書き、後で撤回した | commit `95ac19a`:「0.4 m/s が、学習中の段 2 -> 固結び (tuck) を止めていたと分かった」。`R6:275-276`:「**中間報告の「原因はめり込み解消速度」は学習ログだけに基づいた結論で、本番の指標では確かめられなかった。**」 | `95ac19a`、`KT/versions/v7/README.md:28`、`R6:275-276` | 1 回 (F-17 (c)。F-17 は全部で 3 回) | skill (版の比較の手順) |
| CC-8 | 未コミットの報酬項で学習した後、その項を削除した (R-06 を commit の外で破った) | 「報酬項 (`stage_time_out`) ごと削除した」。「当時の v7 … 今の v7 とは別物で、この項はコードから削除済み」 | `KT/versions/v7/README.md:79-81`、`logs/rsl_rl/knot_tying_v6_ablation/README.md:11` | 1 回 | hook (学習の前に commit を求める。§4.1 D-6) |
| CC-9 | 版の継承元を間違えた | commit `881dc8d`:「v5 は誤って v3 の KnotTyingV3EnvCfg を継承していた」 | `881dc8d` | 1 回 | 試験 (継承元の表) |
| CC-10 | 一時的な scratchpad を、残る文書やスクリプトから参照した | 「キューのスクリプト: `/tmp/claude-1002/…/scratchpad/small_queue.sh`」。「scratchpad から写した」。`logs/rsl_rl/eval_search_lane.sh:4` は、ログの出力先が別のセッションの scratchpad | `HO:170`、`HO:128`、`HO:183`、`logs/rsl_rl/eval_search_lane.sh:4` | 4 か所 | 試験 (追跡しているファイルに `/tmp/claude-` が無いこと)、CLAUDE.md (残すものは `outputs/` へ) |
| CC-11 | テストと lint を回せないまま、セッションを終えた (CC-1 の結果) | 「テスト (`pytest`) と lint (`ruff`) は **このセッションでは回せていない**。最初に回すこと」。`bench_cantilever_art.py` は「**未実行・未 lint**」 | `HO:246`、`HO:256` | 1 回 | hook (Stop の前にテスト)。ただし Bash が使えないときは効かない |
| CC-12 | 作業がコンテキストの上限で中断された | 「1 回目のセッション: 〜2026-10-02 17:57 (コンテキストの都合で中断)」。利用者の自動メモリに「長い作業は使用率 50% で区切る」がある (リポジトリの外: `/home/itoyama/.claude/projects/-home-itoyama-work-isaac-twisted-rl/memory/MEMORY.md`) | `HO:3` | 1 回 (この調査 D も API の制限で一度中断したが、リポジトリには記録が無い) | CLAUDE.md か skill (区切り方と引き継ぎの書式) |

---

## 4. 所見

### 4.1 2 回以上起きた失敗 (hook・skill を作る根拠になるもの)

| # | 所見 | 根拠 (回数) | 推奨する置き場所 |
|---|---|---|---|
| **D-1** | 他のセッション・前のセッションのプロセスの kill が拒否された。1 回目は、その後 Bash がすべて拒否された。原因の半分は「止める必要のあるプロセスが固まる」(D-2) こと。利用者の設定は逆に `pkill -f` を allow している | F-01、CC-1 (2 セッション)。`~/.claude/settings.json` の allow | **permissions**: `Bash(kill *)`・`Bash(pkill *)` を ask、`Bash(pkill -f*)` を deny (プロジェクトの `.claude/settings.json`)。**skill** (ジョブの起動と停止): 起動時に PID・セッション・GPU を `outputs/jobs/` に記録し、止めるのは自分の記録にある PID だけにする。それ以外の PID はユーザーに渡す。**CLAUDE.md** に 1 行 |
| **D-2** | Kit・FEM が固まり、SIGTERM で止まらない。規則 (R-48) はあるが、追跡しているシェル 4 本の `timeout` は、どれも `-k` を付けていない | F-02 (3 件以上)、`KT/rope_study/run_eval_queue.sh:22`、`KT/rope_study/run_physics_suite.sh:26`、`:32`、`:37` | **ラッパースクリプト** (Isaac Sim を起動する共通の入口: `timeout -k`、`-u`、PID の記録、標準出力の `tee`)。**テスト**: 追跡している `*.sh` の `timeout` に `-k` があること |
| **D-3** | `env.step()` の処理の順序 (終了判定 → 報酬 → reset → 観測) から来る罠が 3 回起きた: 抽出器のキャッシュ、整形報酬の検証、`get_term`。**同じ形が `KT/rope_study/bench_physics.py:378` に残っている疑いがある。** `random` 試験は reset で絞らずに `get_term("diverged")` を読む。インストールされた Isaac Lab の `TerminationManager.compute()` は、立った行だけを書き換え、`reset()` でも `_term_dones` を消さない (`.venv/lib/python3.11/site-packages/isaaclab/source/isaaclab/isaaclab/managers/termination_manager.py:172-175`、`:126-149`)。一度発散した環境は、別の終了項が立つまで発散として数え直される。その数値 (例: `Art28` の発散 10.2% / 手、`HO:116`) は過大の可能性がある | F-07 (b)、F-08、`NA:351-353`。疑い: `KT/rope_study/bench_physics.py:378` (静的に読んだだけ。実行して確かめてはいない) | **テスト** (静的): `get_term(` の呼び出しが `reset` / `done` / `terminated` との論理積になっていること (例外は許可リストに)。**.claude/rules** (`common/mdp/**` と評価スクリプトに適用する注意: step の順序と `get_term`)。bench_physics の件は、リファクタリングとは別に確かめる |
| **D-4** | チェックポイントの番号の取り違え。`make_eval_jobs.py` は、無いチェックポイントを黙って飛ばす | F-05 (注意 2 か所、古い記述 9 か所) | **コードとテスト**: `make_eval_jobs.py` で無いチェックポイントを警告し、`last` を解決する。文書の 9 か所を直す。hook は要らない |
| **D-5** | 結果を無効にして手で退避した (5 回)。スクリプトは「出力があれば飛ばす」ので、退避を忘れると古い結果を黙って使う | F-18 (5 回)、F-12、F-13、R-37 | **ラッパースクリプト**: 出力にコードの版 (HEAD と、未追跡を含む diff のハッシュ) と入力 (初期アーカイブのハッシュ) を書く。版が一致するときだけ飛ばす。**skill**: 無効にするときの手順 (退避先・README・`INVALID.txt`・報告書への記載) |
| **D-6** | 未コミットのコードで学習した (21 / 29 run)。未追跡のファイルの中身は run に残らない。未コミットの報酬項を学習に使ってから消した (R-06 を commit の外で破った) | F-09 (a)(b)、CC-8 | **hook** (PreToolUse で `train` / 評価の起動を照合): 作業ツリーが汚れていたら警告する。**ラッパー**: 未追跡の `.py` を含む patch を run のフォルダに保存する。**CLAUDE.md**: 長い学習の前に、WIP の commit をユーザーに提案する |
| **D-7** | 代理の指標 (学習ログ・学習済みの方策での計測) で結論を出し、事実として書いた (3 回)。1 シードで比べた | F-17 (3 回)、CC-7、R-24、R-25 | **skill** (版・紐モデルの比較の手順: chain 評価、2 シード以上、学習と同じ Hydra の上書き、標準誤差とシード差を並べて書く、「解釈」と「事実」を分ける)。**CLAUDE.md** に 1 行 |
| **D-8** | 設定が黙って効かない・黙って壊れる (8 種) | F-15 | **テスト**: 重み 0 のログ専用の項が無いこと (R-15)、`rnn_type` の値 (R-34)、`gamma` の一致 (R-14)、`rope_diameter` を宣言した項が埋まっていること。**コード**: `--experiment_name` を効かせるか、引数を消す (C-8) |
| **D-9** | 文書と実装の食い違い。整理の commit を 2 回出しても 11 件残っている | F-10 | **テスト**: リンク・参照先 (ファイル、`next_action.md` の節番号、`_VERSIONS` と `versions/` と一覧)。**CLAUDE.md**: 「文書は現状だけを書く。経緯は `next_action.md` に 1 か所」。版の結果の表は 1 か所から生成する案もある |
| **D-10** | GPU の共有で、時間の計測が当てにならない。FEM が落ちる | F-06 (4 件) | **ラッパースクリプト** (起動前の空きメモリの確認、1 GPU に学習 2 本まで、時間の記録にそのときの GPU の負荷を付ける)。D-2 と同じ入口にまとめる |
| **D-11** | 長い作業の中断 (5 件) と、引き継ぎメモの肥大 (印の食い違い 2 か所) | F-20、CC-5、CC-6、CC-12 | **skill** (引き継ぎ: 書式は「現在の状態・走っているもの・次の作業」を上書きする。経緯は別のファイルへ追記する)。**CLAUDE.md** (区切るタイミング。今は利用者の自動メモリにある) |
| **D-12** | 一時的な scratchpad を、残る文書・スクリプトから参照した (4 か所)。切り分けのスクリプトは scratchpad から写した後、追跡外の `outputs/` に置いたまま、報告書の再現手順が参照している (`RR:838-839`) | CC-10 | **テスト**: 追跡しているファイルに `/tmp/claude-` が無いこと。**CLAUDE.md**: 残すスクリプトは追跡する場所に置く |

### 4.2 損害が大きい規則 (hook・skill を作る根拠になるもの)

| # | 所見 | 根拠 | 推奨する置き場所 |
|---|---|---|---|
| **D-13** | 共有資産 (`rope_knot.usd`、`overhand_28links.pt`、`outputs/archive_seed/`) を上書きしない。git の管理外なので戻せない | R-44 | **hook** (PreToolUse: Write / Edit / Bash の書き込み先を照合する。`generate_assets.py` を `--rope` なしで呼んだら拒否する) |
| **D-14** | `common/mdp/` の項は追加だけ。既存の版を書き換えない | R-06、R-07、R-08。commit の外で破られた (CC-8) | **テスト** (F.md `test_mdp_names_additive`)。commit の外の違反には D-6 の hook。既存の `versions/vK/` を編集したら警告する **hook** (警告だけ。バグ修正は通す) |
| **D-15** | 別のセッションのプロセスと未コミットの作業に触らない | R-42、R-43、F-01 | **permissions** (D-1 と同じ)。**CLAUDE.md**。並行の作業は git worktree で分ける |
| **D-16** | `__init__.py` の import、`mdp` の名前、topology の純粋さ。破るとエラーなしでタスクが消えることがある | R-01、R-03、R-04 | **テスト** (F.md §6.3 の 3 本)。hook は要らない |
| **D-17** | 評価と学習の条件を揃える (Hydra の上書き、`--device` と `agent.device`) | R-26、R-27 | **コード** (評価スクリプトが run の `params/env.yaml` と照合する。`train.py` で `agent.device` を揃える)。それまでは **hook** でコマンドを照合する |
| **D-18** | 報酬の測り方: 退避の後に測る、γ を揃える、`get_term` を絞る | R-12、R-14、R-16 | **テスト** (γ、`get_term`)。**.claude/rules** (`common/mdp/**` を触るときに読み込ませる) |

### 4.3 その他 (hook・skill の根拠にはしない。文書・設定で足りるもの)

| # | 所見 | 根拠 | 推奨する置き場所 |
|---|---|---|---|
| D-19 | `python -u` (R-28)。実際に踏んだ記録は無い (F-03) | R-28、F-03 | **CLAUDE.md** と、`.claude/settings.json` の `env` に `PYTHONUNBUFFERED=1` (hook は作らない) |
| D-20 | テストのコマンド (R-29、R-30)。venv の取り違え (F-21 (b)) | R-29、F.md:608 | **CLAUDE.md** (`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run python -m pytest tests/`) |
| D-21 | lint の基準 (R-31) | F.md F-1 | **pyproject.toml** (F.md F-1)。CLAUDE.md にコマンドを 1 行 |
| D-22 | rsl-rl / isaaclab の版と、IsaacLab の checkout を写さない (R-33)。利用者の設定は IsaacLab の Read を allow している | R-33、`~/.claude/settings.json` | **CLAUDE.md**。必要ならプロジェクトの permissions で IsaacLab の Read を ask にする (利用者の判断) |
| D-23 | Claude Code の作業ファイルを commit した (F-22) | F-22 | **.gitignore** |
| D-24 | 物理・MDP の個別の注意 (R-17〜R-23、R-35) | — | **文書のみ** (`KT/README.md` と各版の README)。手順にするなら、R-35 のゲートを skill (学習を始める前の確認) に入れる |

---

## 5. 確認できなかったこと

- `KT/rope_study/bench_physics.py:378` が実際に発散を過大に数えているか。静的に読んだだけで、実行していない。`random` 試験の 6 手のあいだに、ほかの終了項がどれだけ立つかにもよる。
- F-01 (b) の PID 1655235 を起動したのが 3 回目と 4 回目のどちらのセッションか (`HO:44`、`HO:34`)。
- `HO:48` の FEM (PID 1635316) と `HO:62` の FEM の固まり (`fem_crash_32env`) が同じプロセスか。
- F-03 (出力のバッファリング) が実際に起きた日時と回数。規則の理由としての記述しか無い。
- F-11 で消した、v3 ベースの v5 / v6 の学習と計測の量 (README から削除済み。`git show 881dc8d^` などで追えるが、読んでいない)。
- F-18 で、退避を忘れて古い出力を使った例があるか。
- 長時間ジョブが実際に `timeout` で切られた記録。`HO:99` の「timeout 1800 で切れる」は見込みで、結果は未確認。
- `eval_compare.py` (F-07 (a)) の数え方で出した数値が、今の文書に残っているか。
- `~/.claude/settings.json` の `defaultMode` の値 (キーがあることだけ確認した)。

---

## 付録 A. 使ったコマンド (scratchpad にもリポジトリにもファイルは置いていない)

### A-1 未コミットのコードで学習した run を数える (F-09 (a))

```bash
cd /home/itoyama/work/isaac_twisted_rl
tot=0; code=0
for f in logs/rsl_rl/*/*/git/*.diff; do
  tot=$((tot+1))
  # git status の部分 (--- git diff --- より前) で、変更・削除・追加された .py、または未追跡の src/・tests/ を数える。
  # 学習に関係しない未追跡の eval_chain.py は除く。
  n=$(awk '/^--- git diff/{exit} {print}' $f \
      | grep -E "^\s+(modified|deleted|new file):.*\.py|^\s+src/.*(/|\.py)$|^\s+tests/" \
      | grep -v eval_chain.py | wc -l)
  [ $n -gt 0 ] && code=$((code+1))
done
echo "runs=$tot code_dirty=$code"     # -> runs=29 code_dirty=21
```

21 run の内訳: `knot_tying_rs_*` 10、v0 1、v2 1、v3 2 (`2026-09-28_17-28-57`、`2026-10-01_10-43-15_v3_newbuf_10it`)、`v4_speed` 3、`v6_ablation` 4。汚れていなかった 8 run: v1、v2_r2、v3 `2026-09-30_19-54-58`、v4 の 2 本 (`3h_compare` は `.claude/scheduled_tasks.lock` だけ、`overnight_s42` は未追跡の `eval_chain.py` だけ)、v5、v6 の 2 本。

### A-2 チェックポイントの番号 (F-05)

```bash
for d in logs/rsl_rl/*/*/; do ls "$d" | grep -E '^model_' | sort -t_ -k2 -n | tail -1; done
git grep -n -E "iters 50,(99|100)|model_99|model_100|it ?99|it ?100|iteration 50 と 100" -- 'src/isaac_twisted_rl/tasks/knot_tying/rope_study/*'
```

### A-3 そのほか

```bash
git log --reverse --format='%h %ad %s' --date=format:'%m-%d %H:%M'
git log --format='%B%x00' | tr '\0' '\n' | grep -c 'Co-Authored-By: Claude'     # -> 20
git grep -n -E "注意|罠|NOTE|WARNING|必ず|しないこと|ないこと|黙って|例外が出ない|例外は出ない|地雷" -- '*.py' '*.sh'
git grep -n "get_term("        # D-3
git grep -n "/tmp/claude"      # D-12 (追跡しているファイルは HO:170 だけ)
git cat-file -t 1f5693f        # -> fatal: Not a valid object name (F-09 (c))
grep -n "timeout" src/isaac_twisted_rl/tasks/knot_tying/rope_study/*.sh     # D-2
```

---

## メインによる根拠の確認 (2026-10-03)

メインの session が、次の 5 か所を自分で開いて確かめた。誤りは見つからなかった。

| 所見 | 確かめた内容 | 結果 |
|---|---|---|
| D-1 kill の拒否 | HANDOFF.md:198-203: PID 1396214 (`generate_assets.py --rope knot56`) を `kill -9` しようとして、自動の権限判定に「他の作業への干渉」として拒否された。その後は `date` を含むすべての Bash が拒否された。:44-48: 学習 2 本 (PID 1586101・1655235) と固まった FEM (PID 1635316) を止められなかった | 一致 (2 つの session の記録) |
| D-2 `timeout` に `-k` がない | `git grep -n "timeout " -- '*.sh'` は 4 か所 (run_eval_queue.sh:22、run_physics_suite.sh:26,32,37)。どれも `-k` がない | 一致 |
| D-4 存在しないチェックポイントを黙って飛ばす | make_eval_jobs.py:51 は既定が `--iters 50,100`。:78-80 は `model_{it}.pt` が無ければ `continue` する | 一致 |
| D-3 の新しい疑い | bench_physics.py:378 は `tm.get_term("diverged").clone()` で、reset のマスクを掛けていない (行の確認だけ。数が過大かどうかは未確認のまま) | 行は一致 |
| D-6 未コミットの .py の変更で学習した run | `logs/rsl_rl/*/*/git/*.diff` に `diff --git a/....py` を含む run を数えると 29 run 中 21 run。.py の変更がない 8 run の内訳は次のとおり (下にパス)。7 run は diff に変更がない。1 run (`knot_tying_v4/2026-10-01_14-08-32_3h_compare`) は `.claude/scheduled_tasks.lock` の変更だけ | 一致 |

D-6 の補足 (メインが気づいたこと。未調査):
- 29 run 中 3 run は `git/` に `isaac_so_arm101.diff` を持つ (ほかの 26 run は `isaac_twisted_rl.diff`)。
- rsl-rl のロガーは、読み込んだパッケージのリポジトリの diff を記録する。そのため、この 3 run は別のリポジトリ (isaac_so_arm101) の venv のパッケージで動いた可能性がある。F-6 (venv の shebang) と関係するかもしれない。

.py の変更がない 8 run:
- logs/rsl_rl/knot_tying_v1/2026-09-15_21-49-25
- logs/rsl_rl/knot_tying_v2_r2/2026-09-26_15-57-53
- logs/rsl_rl/knot_tying_v3/2026-09-30_19-54-58
- logs/rsl_rl/knot_tying_v4/2026-10-01_14-08-32_3h_compare (.lock だけ)
- logs/rsl_rl/knot_tying_v4/2026-10-01_19-22-13_overnight_s42
- logs/rsl_rl/knot_tying_v5/2026-10-01_19-22-38_overnight_s42
- logs/rsl_rl/knot_tying_v6/2026-10-01_19-23-03_overnight_s42
- logs/rsl_rl/knot_tying_v6/2026-10-01_19-23-28_overnight_s43
