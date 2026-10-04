# E. 繰り返し現れた手順の抽出

- 対象: HEAD `6f41ef1` (ブランチ `refactor/claude-foundation`) の追跡ファイル、全 34 コミット、`logs/`・`outputs/` (git の管理外) の一覧。調査日 2026-10-03。
- 目的: どの手順を Claude Code の skill にするかの根拠。**skill にしてよいのは 2 回以上実際に行われた手順だけ。**
- 書いたのはこのファイルだけ。Isaac Sim・GPU は使っていない。`logs/`・`outputs/` は一覧と小さなテキスト
  (シェル・`summary.log` の `[final]` 行・`.hydra/*.yaml` と `params/*.yaml` の数個のキー・評価 JSON のキー名) だけを読んだ。学習ログの本文とチェックポイントは開いていない。
- 略記: `kt/` = `src/isaac_twisted_rl/tasks/knot_tying/`、`sc/` = `src/isaac_twisted_rl/scripts/rsl_rl/`。行番号は `6f41ef1` のもの。
- 用語は C.md の推奨名に揃えた: 「chain 評価」「ゲート 1 (台本の固結び試験)」「発散 (終了項 `diverged`)」「版 vN」「ログ名 (experiment_name)」「run」「初期アーカイブ」「状態アーカイブ」「紐モデル」「ロープ `Knot`」「シード」「学習キュー」(`run_train_chain.sh`)。

## 0. 要点

| # | 手順 | 回数 (主な数え方。詳細は §2) | skill の根拠 |
|---|---|---|---|
| 1 | 版の追加 | 新しい版を作ったコミット 6 (v2-R2・v3・v4・v5・v6・v7)。ほかに作り直し 1 (`881dc8d`)、未コミットの定義替え 1 (旧 v7) | **ある** |
| 2 | 学習 → chain 評価 | 学習の run 29 (うち chain 評価まで進んだ run 14)。chain 評価の結果ファイル: `eval_chain_*.json` 15、rope_study の評価 JSON 117 | **ある** |
| 3 | ゲート 1 | 台本の出力フォルダ 76 (出力の時期 4 つ: 09-27・09-28・10-01・10-02〜03)。文書の記録 4 か所 | **ある** |
| 4 | 発散の計測 | 文書の記録 7 か所 (4 つの時期: 09-10・09-27・09-30〜10-01・10-02〜03)。`divergence_bench` の出力 38、物理ベンチの出力 18 | **ある** |
| 5 | 評価キューの実行と集計 | キュー・ループの起動 10 (始めたジョブ 124、終わった 117 = JSON 117)。キューのシェル 7 本 (追跡 4・未追跡 3) | **ある** (ただし rope_study でしか道具化されていない) |
| 6 | 結果報告書の作成 | 結果の報告書 4 本 (v3・v6・rope_study・推論時探索) + 文献の報告 2 本 | **ある** |
| 7 | セッションの引き継ぎ | `next_action.md` の「現在地」: 作成 1 + 更新 4 コミット。`HANDOFF.md`: 8 つの節 (4 セッション) をコミット 1 回で | **ある** |
| 8a | 物理パラメータの調整 | 7 件以上 (ヤング率・240 Hz・`Knot`・接触バッファ・ソルバ反復・めり込み解消速度 ×2・XPBD の指・座標) | **ある** |
| 8b | 初期アーカイブの作成・作り直し | 6 回 (09-28 に 1、rope_study で 5 世代) | **ある** (ゲート 1 の一部として) |
| 8c | 無効な結果の退避 | 7 件 (README 付きの退避 3、`INVALID.txt` 2、run の手での移動 2) | **ある** |
| 8d | 学習の前のスモークテスト | Hydra の起動 4 + 文書の記録 4 以上 | **ある** (#2 の前段として) |
| 8e | 速度の計測 | 3 回 (方法が毎回違う) | ある (弱い) |
| 8f | テスト・lint の実行 | 文書の記録 5 回以上 | ある (skill より hook 向き) |
| 8g | 方策の書き出し | `exported/` 4 個。すべて `play.py` の副作用で、意図して行った記録なし | **ない** |

主な結論 (所見は §5):

* **最も繰り返されたのは「学習 → chain 評価 → 報告書」で、3 回のまとまり (overnight・切り分け・rope_study) がある。** ただし道具化されたのは rope_study だけ。v4〜v6 の評価を流したシェル (`pipeline2.sh`・`eval_queue2.sh`) は別セッションの scratchpad にあり、リポジトリに残っていない (E-5)。
* **道具はいつも「使った後」にコミットされた。** `overhand_script.py`・`divergence_bench.py`・`eval_chain.py`・`eval_search.py`・rope_study の全部 (E-15)。
* **「シード差 15〜19 ポイント」は v6 の 2 シード (42 / 43) の 1 組だけから出ている。** iteration 50 では差は 4.9 ポイント。ほかの設定はすべて 1 シード (E-16)。

---

## 1. 方法

| 情報源 | 何を数えたか | コマンド |
|---|---|---|
| 文書 | 実行の記録がある行 | `git grep -n` (コマンド・run のパス・日付)。主に `kt/next_action.md`、`kt/versions/*/README.md`、`kt/versions/{v3,v6}/Result_*.md`、`kt/rope_study/{HANDOFF,Report_rope_models}.md`、`paper/Report_Search.md` |
| コミット | 手順の結果のコミット | `git log --format='%h %ad %s'`、`git show --stat`、`git log -S` |
| `logs/rsl_rl/` | run のフォルダ、評価の結果ファイル | `ls logs/rsl_rl/*/*/`。各 run の `params/agent.yaml` から `seed`・`max_iterations`・`experiment_name`・`run_name`、`params/env.yaml` から `num_envs` と USD の名前だけを `grep` |
| `outputs/<日付>/<時刻>/.hydra/` | Hydra を使うスクリプト (`train`・`play`・`eval_chain`・`eval_search`・`eval_robust`) の起動 1 回につき 1 フォルダ | 付録 A-1 の `hydra_scan.py`。`config.yaml` から `experiment_name`・`num_envs`・USD 名・`chain:`・`rope_model` だけ、`overrides.yaml` の全行 |
| `outputs/` のその他 | ゲート 1 の `summary.log` の `[final]` 行、キューの進み具合のログ、退避フォルダの README | 付録 A-2 |

**Hydra のフォルダの読み方 (重要)。** `config.yaml` は CLI 引数 (`--seed`・`--num_envs`・`--run_name`) を当てる前の値なので、
`num_envs` は設定の既定値を表す。Play タスクの既定は 9 環境、学習の既定は 1024 / 2048 / 4096。
そこで「`num_envs` = 9」を Play・評価の起動、それ以外を学習の起動とし、学習の起動の時刻が run のフォルダ名
(`<日時>_<run_name>`) と一致するものを「run あり」とした。

| 分類 | 数 |
|---|---|
| Hydra のフォルダ (`ls -d outputs/2026-*/*/`) | 269 |
| うちテンプレートの `reach` (課題と無関係) | 6 |
| 学習の起動 | 68 (うち run のフォルダが残っているもの 27) |
| Play・評価の起動 | 195 |

run のフォルダは 29 個 (`ls -d logs/rsl_rl/*/*/ \| wc -l`)。27 との差 2 は、(1) `knot_tying_v4_speed/2026-09-30_22-27-57_A_it16_T12` の Hydra のフォルダが 1 秒前の `22-27-56`、
(2) `2026-10-03_10-35-27` に 2 本 (`XpbdC28_s43` と `XpbdC56_s43`) を同じ秒に起動して Hydra のフォルダを共有した (中身は `XpbdC56` の設定だけ)。

---

## 2. 手順ごとの記録

### 2.1 版の追加 (versions/vN を作る)

#### 回数

| 情報源 | 数 | 内訳・根拠 |
|---|---|---|
| コミット (新しい版) | **6** | v2-R2 `f9c4e4c` (09-26)、v3 `3b71145` (09-28)、v4 `5115f9f` (09-30)、v5 `db375b3` (10-01)、v6 `09ec904` (10-01)、v7 `95ac19a` (10-02)。`git log --diff-filter=A -- kt/versions/vN/__init__.py` |
| コミット (作り直し) | 1 | `881dc8d`: v5 / v6 を v3 ベースから v4 ベースに作り直し |
| git の前 | 1 (v0〜v2 をまとめて) | 版の制度は 2026-09-24 に導入 (`kt/next_action.md:559-564`)。v0〜v2 は first commit `5fd7e3c` に最初から入っている |
| 未コミットの定義替え | 1 | 旧 v7 = 「時間切れの罰則 `stage_time_out`」で 1 本学習した (`logs/rsl_rl/knot_tying_v6_ablation/README.md:11`、run の `params/agent.yaml` の `experiment_name: knot_tying_v7`)。その後 v7 を「v6 + めり込み解消速度 0.96」に定義し直してコミット。`git log -S stage_time_out` は文書の 2 コミット (`4c58709`・`95ac19a`) だけ → 旧 v7 のコードは一度もコミットされていない |
| 文書 | 9 版 | `kt/README.md:35-45` の版の表 |
| `logs/rsl_rl/` | ログ名 8 + 派生 2 | `knot_tying_v0`〜`v6` と `v2_r2`。派生は `knot_tying_v4_speed`・`knot_tying_v6_ablation`。`knot_tying_v7` は無い (v7 は未学習: `kt/README.md:45`) |

#### 手順の中身 (コミットの中身から)

規約は `kt/README.md:175-187` (5 項目)。6 つの作成コミットで実際に触ったファイル:

| やること | 規約にあるか | 6 回のうち行った回数 | 根拠 |
|---|---|---|---|
| `versions/vN/` に 5 ファイル (`__init__.py`・`env_cfg.py`・`agents/__init__.py`・`agents/rsl_rl_ppo_cfg.py`・`README.md`) | あり (1) | 6 / 6 | 各コミットの `git show --stat` |
| `kt/__init__.py` の `_VERSIONS` に足す | あり (1) | 6 / 6 | 同上。現在値 `kt/__init__.py:40` |
| `kt/README.md` の版の表 | なし | 5 / 6 (v7 は同時刻の別コミット `4c58709`) | 同上 |
| `kt/versions/__init__.py` の一覧 | なし | **3 / 6** (v3・v4・v5)。v6 は後の `ccae2ae` で追記、**v7 は今も無い** (`kt/versions/__init__.py:6-13`。C-5) | `git log -- kt/versions/__init__.py` |
| ルートの `README.md` | なし | 1 / 6 (v4) | 同上 |
| テストを足す | なし | 5 / 6 (v7 は設定だけなので無し) | 同上 |
| `next_action.md` の「現在地」 | なし | 1 / 6 (v5) | 同上 |
| `experiment_name = knot_tying_vN` | あり (4) | 6 / 6 (`kt/versions/v7/agents/rsl_rl_ppo_cfg.py:18` まで。C.md §4.1) | — |

* **スクリプト化: なし。** 雛形も生成スクリプトも無く、毎回手で写している。プリセット 5 種 × 学習 / Play のタスク登録だけは `kt/__init__.py` の二重ループで自動 (C.md §3)。
* **実装の前に走らせた例がある。** v4 の Play・学習の起動が 09-28 17:41〜18:54 (Hydra)、コミットは 09-30 19:56。v6 のスモークテストが 10-01 17:55、コミットは 18:51 (Hydra の `agent.run_name=smoke_v6`)。v3 の学習 run は「v4 の作業中の変更 (未コミット) が乗っていた」(`kt/versions/v3/Result_report_v3.md:33`)。

#### 失敗

* 旧 v7 の番号を、別の中身の v7 に使い直した (上表)。旧 v7 の run は `knot_tying_v6_ablation/` に移されて残り、「当時の v7」と注記するしかなくなった (`logs/rsl_rl/knot_tying_v6_ablation/README.md:11`)。
* v5 は v3 ベースで作った後 (`db375b3`)、同じ日に v4 ベースへ作り直した (`881dc8d`)。v3 ベースの v5 の 3 時間の学習 (Hydra `2026-10-01/14-08-34`、`agent.run_name=3h_compare`) は run のフォルダが残っていない。

#### 成果物の置き場所

`kt/versions/vN/` (追跡)。学習のログ `logs/rsl_rl/knot_tying_vN/` (管理外、`.gitignore:49-50`)。

---

### 2.2 学習 → chain 評価

#### 回数

**学習の run (情報源ごと)**

| 版・用途 | 文書の記録 | `logs/` の run | Hydra の学習の起動 (うち run あり) | 評価の結果ファイル |
|---|---|---|---|---|
| v0 | `kt/versions/v2/README.md:20-21` (970 iteration) | 1 | 版の前の `knot_tying` 9 (2) を v0 / v1 で共有 + `knot_tying_v0` 2 (0) | なし (Play のみ) |
| v1 | `kt/versions/v1/README.md:9` (157 で打ち切り) | 1 | 上の 9 に含む + `knot_tying_v1` 4 (0) | なし |
| v2 (MLP / RNN / LSTM) | `kt/versions/v2/README.md:20`、`kt/versions/v2_r2/README.md:35` | 1 | 2 (1) + RNN 1 (0) + LSTM 1 (0) | なし |
| v2-R2 | `kt/versions/v2_r2/README.md:134-136` (比較に使えない) | 1 | 3 (1) | なし |
| v3 | `kt/versions/v3/Result_report_v3.md:1-31`、`kt/versions/v4/README.md:16-17` | 3 | 10 (3) | なし (TensorBoard と Play) |
| v4 | `kt/versions/v4/README.md:250-264`、`:292-305` | 2 + 速度 3 | 4 (2) + 速度 5 (2) | `eval_chain_*.json` 3、`eval_search_*.json` 2 |
| v5 | `kt/versions/v5/README.md:170-171` | 1 | 4 (1) | `eval_chain` 2、`eval_search` 1 + 途中 1 |
| v6 | `kt/versions/v6/Result_Report_v6.md:29-38` | 2 | 2 (2) + スモーク 1 (0) | `eval_chain` 6、`eval_search` 3 + 途中 2 |
| v6 の切り分け | `kt/versions/v6/Result_Report_v6.md:215-229` | 4 (うち 1 本は旧 v7) | 4 (4) | `eval_chain` 4 |
| v7 | 未学習 (`kt/README.md:45`) | 0 | 0 | (切り分けの `abl_depB` を v7 の代わりに `eval_search` で評価: `paper/Report_Search.md:292-295`) |
| rope_study | `kt/rope_study/Report_rope_models.md:303-317` | 10 (うち無効 2) | 9 (9。1 フォルダを 2 本で共有) | `outputs/rope_study/eval/` の JSON 113 + 退避 4 |
| 計 | — | **29** | **68 (27)** | — |

Hydra の学習の起動のうち run の無い 41 本の内訳: 版の制度の前後の試運転、`smoke_v5_baseline`・`smoke_v6`・`knot_tying_smoke` ×2 (スモークテスト、§2.8d)、`knot_tying_bench` ×5 (速度の計測、§2.8e)、v3 の `agent.resume=True` ×2、run を消したもの (v3 の最初の 1024 環境の run: `kt/versions/v4/README.md:16-17`、v3 ベースの v5)。

**chain 評価 (v4 以降の版の比較) の回数**

| 情報源 | 数 | 根拠 |
|---|---|---|
| 文書の表の行 | 15 (+ 試運転 1) | `kt/versions/v6/Result_Report_v6.md:74-83` (8 行)、`:254-262` (7 行)。試運転 = `3h_compare/model_50.pt` の 64 エピソード (`:85`) |
| `eval_chain_*.json` | **15** | `ls logs/rsl_rl/*/*/eval_chain_*.json` (v4 3、v5 2、v6 s42 3、v6 s43 3、切り分け 4) |
| Hydra の Play の起動 (v4〜v6、10-01 19:24〜10-02 22:01) | 16 | `hydra_scan` (10-01 19:24 の 1 本が試運転。`Result_Report_v6.md:53-54` の「評価スクリプトの試運転」と一致) |

3 つの数は一致する。**15 本の JSON の更新時刻はすべて 10-02 23:04** で、Hydra の起動 (11:14〜22:01) より後。評価の後にまとめて run のフォルダへ置いたと読める (置き方は未確認)。

**まとまり (学習を並べて回し、評価して、報告書に書く単位) は 3 回:**

| まとまり | 学習 | 評価 | 記録 |
|---|---|---|---|
| overnight (10-01 19:22〜10-02 09:00) | v4・v5・v6 s42・v6 s43 の 4 本、GPU 1 枚に 2 本 | chain 評価 it50 / 70 / 100 (10-02 11:14〜) | `Result_Report_v6.md` (`4c58709`) |
| v6 の切り分け (10-02 12:33〜) | 4 本、iteration 70 で停止 | chain 評価 it70 (21:36〜22:01)、比較用に v4・v6 も it70 で 3 本 | 同上の追記 |
| rope_study (10-02 21:39〜10-03 12:40) | 6 紐モデル × seed 42 (+ 無効 2、seed 43 2) | `eval_robust.py` を学習キュー・評価キューで | `Report_rope_models.md` (`b53f070`) |

#### 実際に使われたコマンド

```bash
# overnight の学習 (kt/versions/v6/Result_Report_v6.md:204-206)
uv run train --task Isaac-Twisted-RL-Knot-v6 --headless --device cuda:0 --seed 42 --max_iterations 400 \
    --run_name overnight_s42 agent.device=cuda:0 agent.save_interval=10
# chain 評価 (同 :208-210。kt/README.md:91-93 も同じ形)
uv run python -u sc/eval_chain.py --task Isaac-Twisted-RL-Knot-Play-v6 \
    --num_envs 512 --headless --device cuda:0 --seed 7 \
    --checkpoint logs/rsl_rl/knot_tying_v6/2026-10-01_19-23-03_overnight_s42/model_100.pt agent.device=cuda:0
# 切り分け: 学習にも評価にも同じ Hydra の上書きを付ける (同 :303-310)
... --run_name abl_depB_s42 agent.device=cuda:1 agent.save_interval=10 env.scene.rope.spawn.rigid_props.max_depenetration_velocity=0.96
# rope_study の学習キュー (kt/rope_study/run_train_chain.sh:24-25)
.venv/bin/train --task Isaac-Twisted-RL-RS-$M-v4 --headless --device $DEV --seed 42 --max_iterations 100 \
    --run_name rs_${M}_s42 agent.device=$DEV agent.save_interval=10 > $LOG/train_$M.log 2>&1
```

#### 引数のばらつき (各 run の `params/*.yaml` から)

| 項目 | 値 (run の数) |
|---|---|
| `num_envs` | 1024 (v0・v1)、2048 (v2・v2-R2)、4096 (v3 以降の 25 本) |
| `max_iterations` | 2000 (v0・v2・v2-R2・v3 ×2)、4000 (v1)、10 (`v3_newbuf_10it`)、6000 (`3h_compare`)、100000 (速度 ×3)、400 (overnight ×4)、101 (切り分け ×4)、100 (rope_study ×10) |
| 実際に止めた iteration | v1 157、v3 159 (PhysX の "Patch buffer overflow")、overnight 106〜115 (「時刻で一斉に停止」`Result_Report_v6.md:36-37`)、切り分け 70 (`:220`)、rope_study 100 (完走) |
| `--seed` | 42 が 26 本、43 が 3 本 (v6 s43・`XpbdC28_s43`・`XpbdC56_s43`) |
| `agent.save_interval` | 既定 (50 など) か 10 (overnight・切り分け・rope_study) |
| `--device` / `agent.device` | cuda:0 / cuda:1。**揃えないと PPO が cuda:0 に残る** (`kt/versions/v4/README.md:285-286`) |
| `--run_name` | 無し (v0〜v3)、`3h_compare`、`A_it16_T12` 型、`overnight_s4N`、`abl_<略>_s42`、`rs_<モデル>_s4N` |
| ログ名の付け方 | 3 通り: 設定の `experiment_name` / Hydra の `agent.experiment_name=knot_tying_v4_speed` (Hydra の `overrides.yaml`) / **後から手で移動** (v0・v1 の run は `experiment_name: knot_tying` のまま `knot_tying_v0/`・`v1/` にある。切り分けの run は `knot_tying_v6`・`v7` のまま `knot_tying_v6_ablation/` にある)。C.md §4.2 と §7 の「未確認」はこれで説明がつく |
| 評価 | chain 評価は常に 512 環境・seed 7・決定論的。ただし `sc/eval_chain.py` の docstring と既定は 1024 環境・`model_300.pt` (`sc/eval_chain.py:12-13`、`:30`) で、実際の使い方と違う |
| 標準出力のログ | 残したのは `outputs/v4_train/*.out` 1、`outputs/v5_train/*.out` 3、rope_study の `outputs/rope_study/logs/train_*.log` 8 だけ。v3 の本番 run と overnight の 4 本は残っていない (`Result_report_v3.md:31`) |

#### スクリプトになっている部分 / 手の部分

| 部分 | v 系列 | rope_study |
|---|---|---|
| 学習の起動 | 手 (`nohup` で 1 本ずつ。HANDOFF.md:89 の「nohup」) | `run_train_chain.sh` (先の PID と初期アーカイブを待つ。**seed 42 が直書き** `:24` なので seed 43 は手で起動: `HANDOFF.md:65`) |
| 停止 | 手 (時刻で kill) | 完走 |
| 評価の一覧 | 手 | `make_eval_jobs.py` |
| 評価の実行 | 別セッションの `pipeline2.sh` (`HANDOFF.md:293`)・`eval_queue2.sh` (`:101`、`:171`)。**どちらもリポジトリにも `logs/`・`outputs/` にも無い** (`find` で確認) | `run_eval_queue.sh`・`run_eval_loop.sh` |
| JSON を run のフォルダへ | 手 (`--out`。`eval_chain.py` は `--out` が無いと書かない: `sc/eval_chain.py:141-143`) | キューが `outputs/rope_study/eval/<方策>/<評価>.json` に書く |
| 表 | 手 (報告書に手で写す) | `summarize_eval.py`・`summarize_train.py` |

#### 失敗

| 何が | 根拠 |
|---|---|
| v1: 4000 iteration の設定で 157 で打ち切り | `kt/versions/v1/README.md:9` |
| v2-R2: iteration 1 で value loss 7.2e9、「比較に使えない」。修正 (`2192c33`) の後は 3 iteration の確認だけで、本学習はしていない | `kt/versions/v2_r2/README.md:134-136`、`:178`。09-27 以降の `knot_tying_v2_r2` の run なし |
| v3: 最初の 1024 環境の run を「run ごと置き換え」(26 iteration)。本番は 159 で PhysX のバッファあふれで停止。再開の起動 2 回 (`agent.resume=True`) は run を残していない | `kt/versions/v4/README.md:16-17`、`Result_report_v3.md:16`、Hydra `2026-09-30/19-29-31`・`19-32-49` |
| overnight: 09:00 の停止の後、評価を始めたのは 11:14 | `Result_Report_v6.md:181` |
| rope_study: `XpbdC28`・`XpbdC56` の最初の run は無効 (物理の不具合)。止められず (kill が権限判定で拒否) 走り切った | `HANDOFF.md:44-46`、run の `INVALID.txt` |
| rope_study: seed 43 の評価を途中で止めた (始めたジョブ 6、JSON 0) | `outputs/rope_study/eval_s43_stopped/` の一覧、`HANDOFF.md:21-22` |
| 同じ秒に 2 本起動すると Hydra の出力フォルダが 1 つになる | `outputs/2026-10-03/10-35-27/` (中身は `XpbdC56` だけ。§1) |

#### 成果物の置き場所

| もの | 場所 | git |
|---|---|---|
| チェックポイント・TensorBoard・`params/`・`git/` (差分)・`archive.pt` | `logs/rsl_rl/<ログ名>/<run>/` | 管理外 |
| chain 評価・推論時探索の結果 | 同じ run のフォルダの `eval_chain_it{N}.json`・`eval_search_*_it{N}.json` | 管理外 |
| rope_study の評価・表 | `outputs/rope_study/eval/`、`eval/summary.md`、`train_summary.md` | 管理外 |
| Hydra の起動の記録 | `outputs/<日付>/<時刻>/.hydra/` | 管理外 |
| 結果の数字 | 報告書 (§2.6) | 追跡 |

---

### 2.3 ゲート 1 (台本の固結び試験)

#### 回数

| 情報源 | 数 | 根拠 |
|---|---|---|
| 文書 | 4 つの時期に記録 | (1) 09-27: `kt/next_action.md:484-541` (表 8 条件 `:505-512`) (2) v4 のソルバ反復の振り: `kt/versions/v4/README.md:205-221` (反復 7 値 + 反復 8 と 16 でシード 3、計 10 回、256 環境) (3) v5 の確認: `kt/versions/v5/README.md:152-163` (4 設定 × 種 2) (4) rope_study: `Report_rope_models.md` §7.1.2 / §7.1.5 / §7.1.7 / §7.1.8、`HANDOFF.md:113-115`・`:217-225` |
| コミット | 4 | `686af21` (通過とロープ `Knot`)、`5115f9f` (`--dump_archive`)、`db375b3` (`--replay_actions` の修正)、`a9b02b3` (まっすぐな紐から始める修正と `--dump_knots`) |
| `outputs/` の `summary.log` | **76** | `outputs/overhand_script/` 25 (09-27 18:23〜20:18。うち `Knot-v2-R2` は 09-28 17:19 に初期アーカイブの書き出しで上書き) + `overhand_script/v5_check/` 8 (10-01 06:11〜06:28) + `outputs/rope_study/` 43 (`gate1` 8、`gate1_fast` 8、`contaminated_start` 9、`finger_v1` 4、`xpbd_world_coords` 8、`diag` 5、`fem_crash_32env` 1) |

数の照合: v5 の文書の表 (`v5/README.md:160-163`) は `v5_check` の 8 個の `[final]` 行の和と一致する (例: v3 の設定 272/512 = 142 + 130、発散 42 = 23 + 19)。
v4 の反復の振り (10 回) の出力は `outputs/overhand_script/` に見当たらない (未確認)。`diag` 5 は `--moves 2` の切り分けで、ゲート 1 そのものではない。

#### 実際に使われたコマンド

```bash
# 基本形 (kt/README.md:202-206)
uv run python -u kt/demos/overhand_script.py --task Isaac-Twisted-RL-Knot-v2-R2 --num_envs 128 --jitter 0.25 --headless
#   + --replay_actions (6 次元の行動で再生) / --dump_archive (初期アーカイブ) / --fast (学習の尺)
# rope_study の物理試験 (kt/rope_study/run_physics_suite.sh:26-27、:37-38)
timeout 7200 $PY kt/demos/overhand_script.py --task $T --num_envs $NENV --jitter 0.25 \
    --headless --device $DEV --out $OUT/gate1/$M --dump_archive $SEED/$M.pt --dump_knots $SEED/${M}_knots.pt
```

引数のばらつき: 環境数 32 (09-27 の試行・FEM)、96 (`Twist-v2-R2`)、128 (標準)、256 (v4 の振り・v5 の確認)、64 (切り分け)。揺らし `--jitter 0.25` は全記録で同じ。種は既定 0、v5 の確認だけ 0 / 1、v4 の振りは 0 / 1 / 2。尺は遅い尺 (7.8 s/手) と学習の尺 (`--fast`、3.16 s/手)。`timeout` は 7200 s (スクリプト経由のときだけ)。

#### スクリプトになっている部分 / 手の部分

* rope_study では `run_physics_suite.sh` が「遅い尺 (初期アーカイブを書く) → 物理ベンチ → 学習の尺」を順に回し、済んだステップを飛ばす (`:25`、`:31`、`:36`)。起動は 7 回 (GPU0・GPU1 17:51、v2 22:06、v3 23:51 ×2、FEM 00:25・10:32: `outputs/rope_study/logs_suite_*.txt`)。ほかに手書きの `outputs/rope_study/diag_scripts/gate1_redo.sh` (未追跡) 1 回。
* v 系列 (09-27・10-01) は手で 1 本ずつ。条件ごとに `--out` のフォルダ名を手で付けた (`knot_fast_c06`・`knot_replay_med_xy30` など)。
* 結果の読み取り (`[final] ... overhand=N/128 diverged=M`) は手。rope_study のシェルは `grep` で表示するだけ (`run_physics_suite.sh:29`)。

#### 失敗

| 何が | 根拠 |
|---|---|
| ロープ `Twist` では結べない (長さ不足) → ロープ `Knot` を追加 | `kt/next_action.md:486`、`:516-519`、`686af21` |
| 学習の尺のゲート 1 が全モデルで「途中の形から」始まっていた (初期アーカイブがあると reset がそこから始まる) → 全部やり直し | `HANDOFF.md:105-107`、`outputs/rope_study/contaminated_start/README.md:3-9`、修正 `a9b02b3` |
| XPBD の指の接触 v1 → v2 でやり直し | `outputs/rope_study/finger_v1/README.md:1-5`、`HANDOFF.md:178-182` |
| XPBD を world 座標で解いていた → 修正後に XPBD 4 モデルをやり直し | `outputs/rope_study/xpbd_world_coords/README.md:1-10`、`HANDOFF.md:38-42` |
| FEM 32 環境が GPU 混雑時に CUDA illegal memory access で固まる (`summary.log` に `[final]` 無し) | `HANDOFF.md:62-63`、`outputs/rope_study/fem_crash_32env/` |
| Kit が終了時に固まることがある (`timeout` 待ち) | `HANDOFF.md:152` |
| 未解明: `env.step()` での再生は台本モードより発散が多い | `kt/next_action.md:538-540`、`:51` (次の作業として残ったまま) |

#### 成果物の置き場所

`outputs/overhand_script/<名前>/`、`outputs/rope_study/{gate1,gate1_fast}/<モデル>/` (`summary.log`・`envs.log`・`move*.png`)、初期アーカイブ `outputs/archive_seed/` (すべて管理外)。結果の表は `next_action.md` §4.8.10、v4・v5 の README、`Report_rope_models.md` §7.1 (追跡)。

---

### 2.4 発散の計測

#### 回数

| 時期 | 文書の記録 | 道具 | `outputs/` の跡 |
|---|---|---|---|
| 09-10 | `kt/next_action.md:319-338` (120 / 240 Hz、16 環境 × 10 手、ランダム方策) | 未確認 (記録なし) | 記録なし |
| 09-27 | `kt/versions/v2_r2/README.md:138` 付近 (ランダム方策 512 環境 × 24 手)、`kt/next_action.md:525-527` (ゲート 1 の運搬速度) | 記録なし / `overhand_script.py` | `outputs/overhand_script/div_{A,B,C,D}/` 4 |
| 09-30〜10-01 | `kt/versions/v4/README.md:223-243` (反復ごとの発散)、`kt/versions/v5/README.md:39-163`、`Result_report_v3.md:185-` (接触バッファ) | `demos/divergence_bench.py` (`db375b3` で追加) | `outputs/divergence_bench/` **38** (`.out` 35。09-30 20:34〜10-01 06:44) |
| 10-02〜03 | `Report_rope_models.md` §7.1.3、`HANDOFF.md:116`、`:233-238` | `rope_study/bench_physics.py` の `random` | `outputs/rope_study/physics/` 10 + `finger_v1/physics/` 4 + `xpbd_world_coords/physics/` 4 = **18** |

コミット: `b345065` (計測で見つけたバッファあふれの修正)、`4115db4` (反復 8)、`db375b3` (v5 の対策と道具)。

#### 実際に使われたコマンド

```bash
# kt/versions/v5/README.md:56-60
V5='grasp_limits.release_tip_z=(0.05,0.10);scene.rope.spawn.rigid_props.max_depenetration_velocity=0.4'
uv run python -u kt/demos/divergence_bench.py --task Isaac-Twisted-RL-Knot-v3 \
    --num_envs 1024 --steps 24 --headless --archive_prob 0.0 --cfg "$V5"
... --policy ckpt --checkpoint logs/rsl_rl/knot_tying_v3/2026-09-28_17-28-57/model_150.pt
# rope_study (kt/rope_study/run_physics_suite.sh:32-33)
timeout 7200 $PY kt/rope_study/bench_physics.py --model $M --num_envs $NENV --headless --device $DEV --out $OUT/physics/$M
```

ばらつき: 環境数 16 / 512 / 1024 / 4096 / 128、手数 8 / 10 / 12 / 20 / 24、行動 3 通り (一様乱数 + まっすぐ / 一様乱数 + 初期アーカイブ / 学習済み方策 `model_150.pt`)。出力フォルダ名は `<条件>_<行動>` を手で付けた (`Adep03_ckpt150`・`Ait8_rand_straight` など)。

#### スクリプト / 手

`divergence_bench.py` は 1 条件 1 起動。38 回は手で起動した (キューのシェルは見つからない)。条件 (`--cfg`) と出力名は手。rope_study は `run_physics_suite.sh` に組み込み。

#### 失敗・注意

* **判定が 2 つの道具で違う。** 終了項 `diverged` (NaN・水平 3 m・高さ 5 m・20 m/s) と `divergence_bench` の spike (C-7)。`XpbdC28LowMu` の「発散」はすべて 3 m の漂流だった (`HANDOFF.md:60-61`)。
* `bench_physics` の hang で `Art28` が NaN → `ValueError` で落ち、修正後に再実行 (`HANDOFF.md:233-238`、`outputs/rope_study/logs_small_queue.txt`)。
* 計測と学習で結論が逆になった: めり込み解消速度 0.4 m/s は計測では良かったが、学習では段 2 → 固結びを学べなくした (`kt/next_action.md:40-42`)。

#### 成果物の置き場所

`outputs/divergence_bench/<名前>/` と `<名前>.out`、`outputs/rope_study/physics/<モデル>/{summary.json,shapes.pt}` (管理外)。表は v4・v5 の README、`Report_rope_models.md` (追跡)。

---

### 2.5 評価キューの実行と集計

#### 回数

| 情報源 | 数 | 根拠 |
|---|---|---|
| キュー・ループのログ | **10** (始めたジョブ 124、終わった 117、`FAIL` 0) | `outputs/rope_study/eval/{queue_v4_gpu0,loop_gpu0,loop_cuda{0,1}_{a,b},queue_it50_cuda{0,1}}.txt`、`eval_s43_stopped/queue_s43_cuda{0,1}.txt` の ` start `・` done ` の行数 |
| 評価の JSON | 117 (= 終わったジョブの数) | `outputs/rope_study/eval/` 113 + `xpbd_world_coords/eval/` 4。.log だけで JSON が無いもの 7 (`v4Knot/obs5mm_it100` 1、`eval_s43_stopped` 6) |
| ジョブ一覧 | 6 ファイル | `jobs_v4.txt` (10 行)、`jobs_it50.txt` (40)、`jobs_s43.txt` (32)、`jobs_auto_cuda{0,1}.txt` (66) |
| Hydra の RS の Play の起動 | 126 | 124 + スモーク 1 (`outputs/rope_study/eval_smoke_test/`) + 診断 1 (`diag/lowmu_eval_breakdown.json`) と概ね一致 |
| v 系列の推論時探索 | 計画 10、完了 4、途中 3、未着手 3 (+ 状態の複製の検証 2) | `paper/Report_Search.md:8-10`。`logs/` の `eval_search_*.json` 6、`*_partial.log` 3。Hydra の起動 (10-03 11:33〜13:39) は 11 で、出力 9 との差 2 は未確認 |
| v 系列の chain 評価のキュー | 少なくとも 2 本のシェル (`pipeline2.sh`・`eval_queue2.sh`) | `HANDOFF.md:101`、`:171`、`:293`。**実物は見つからない** |

キューのシェル 7 本 (追跡 4: `kt/rope_study/run_{eval_queue,eval_loop,physics_suite,train_chain}.sh`、未追跡 3: `logs/rsl_rl/eval_search_lane.sh`、`outputs/rope_study/diag_scripts/{gate1_redo,small_queue}.sh`) は同じ型をしている:
`cd` を絶対パスで直書き → 出力があれば飛ばす → `timeout N .venv/bin/python -u ...` → ログへリダイレクト → `grep` で要約 → `$(date +%T)` の時刻。

#### 実際に使われたコマンド

```bash
# 一覧を作る (Report_rope_models.md:823-824)
.venv/bin/python kt/rope_study/make_eval_jobs.py --policies Art28,Art28LowMu,XpbdC28,XpbdC28LowMu,XpbdC56,XpbdMS28 --iters 50,99 \
    --out outputs/rope_study/eval/jobs.txt
# 同じ一覧を 2 GPU で同時に (同 :825-826。run_eval_queue.sh:20 が noclobber で .log を取り合わない)
bash kt/rope_study/run_eval_queue.sh outputs/rope_study/eval/jobs.txt cuda:0 &
# 学習中にチェックポイントが増えるたびに (run_eval_loop.sh:4、止めるときは outputs/rope_study/eval/STOP)
bash kt/rope_study/run_eval_loop.sh cuda:0 600 [make_eval_jobs.py の引数...]
# 表 (Report_rope_models.md:835-836)
.venv/bin/python kt/rope_study/summarize_eval.py --md outputs/rope_study/eval/summary.md
# 推論時探索のレーン (logs/rsl_rl/eval_search_lane.sh:19、:24-26。未追跡)
.venv/bin/python -u sc/eval_search.py --task $task --mode search --slots 16 --beam 4 --problems 128 --max_moves 1024 --max_minutes 240 \
    --num_envs 512 --budgets 16,32,64,128,256,512,1024 --headless --device $DEV --seed 7 --checkpoint $ck agent.device=$DEV
```

ばらつき: `timeout` 5400 s (`run_eval_queue.sh:22`) / 7200 s (`run_physics_suite.sh`) / 3600・1800 s (`small_queue.sh`) / 探索は `--max_minutes 240`。評価の環境数 512 (自分のモデル)・それ以外は一覧が決める。`--iters 50,100` と `50,99` が混在 (最後のチェックポイントが `model_99.pt` の run がある: `HANDOFF.md:29`)。

#### 失敗

| 何が | 根拠 |
|---|---|
| v 系列の評価を流したシェルが別セッションの scratchpad にあり、残っていない | `HANDOFF.md:101`、`:171`、`:293`。`find` で見つからない |
| `eval_search_lane.sh` のログの出力先が別セッションの scratchpad | `logs/rsl_rl/eval_search_lane.sh:4` |
| 推論時探索の評価を途中で止めた (10 本中 4 本だけ完了) | `paper/Report_Search.md:8-10` |
| seed 43 の評価を止めた | `outputs/rope_study/eval_s43_stopped/` |
| 前のセッションのプロセスを kill できない (権限判定)。`pkill -f` が自分のシェルを殺した。その後 Bash 全体が拒否された | `HANDOFF.md:30`、`:150-151`、`:198-203` |
| 走っているキューの最中にソルバを書き換えると、まだ回っていないモデルに効く | `HANDOFF.md:369-370` |
| 再現手順の GPU の割り当てが実際と逆 (手順: GPU0 = 剛体鎖。実際: GPU0 = XPBD) | `Report_rope_models.md:808-809` と `outputs/rope_study/logs_suite_gpu0.txt` (XpbdC28 から始まる) |

#### 成果物の置き場所

`outputs/rope_study/eval/` (JSON・.log・一覧・キューのログ・`summary.md`)、`logs/rsl_rl/<run>/eval_search_*`。すべて管理外。道具 (`make_eval_jobs.py`・`summarize_*.py`・シェル 4 本) は追跡。

---

### 2.6 結果報告書の作成 (Result_*.md、Report_*.md)

#### 回数

| 報告書 | 種類 | コミット | 後からの追記 |
|---|---|---|---|
| `kt/versions/v3/Result_report_v3.md` | 学習の結果 | `fabaa65` (09-30) | `db375b3` (§8 追記: 発散の原因) |
| `kt/versions/v6/Result_Report_v6.md` | 学習と chain 評価の結果 (v4 / v5 / v6) | `4c58709` (10-02) | 同じコミット内に「追記 (2026-10-02)」(切り分け) |
| `kt/rope_study/Report_rope_models.md` | 調査・実装・比較 | `b53f070` (10-03) | — |
| `paper/Report_Search.md` | 推論時探索の評価 (中間) | `72d18df` (10-03) | — |
| `paper/Report_Paper.md` (旧 `Report.md`) | 論文との比較 | `92f9e86` (09-27) | `91f9c79`、`32c9f93`、`ccae2ae`、改名 `dbf9a97` |
| `paper/Discussion.md` | 論文を含めた調査 | `44d188a` (10-01) | — |
| HTML (`paper/explainer/*`、`kt/versions/v3/HTML/*`) | 解説・実装の図解 | `53e6fda`、`28c37dd` (09-28) | — |

結果の報告書は **4 本**、文献の報告書は 2 本 (+ HTML 2 コミット)。

#### 中身の型 (3 本が同じ骨組み)

| 節 | v3 | v6 | 探索 | rope_study |
|---|---|---|---|---|
| 冒頭の箇条 (作成日・対象・読んだもの・「解釈」の断り) | `:3-6` | `:3-9` | `:3-11` | `:3-7` |
| 0. 要旨 | `:10` | `:13` | `:15` | `:9` |
| run の条件 / 評価の条件 | `:20`、`:37` | `:29`、`:59` | `:77` | `:275` |
| 結果 → 結果からわかること | `:49`、`:136` | `:70`、`:133` | `:116`、`:210` | `:337`〜 |
| 注意・未確認 | `:164` | `:174` | `:246` | `:695` |
| 次にやること | `:174` | `:185` | `:258` | `:769` |
| 再現の手順 | なし | `:201` | `:279` | `:789` |
| 追記 | `:185` | `:215` | — | — |

「解釈」と書いた箇所を分ける約束は 4 本とも冒頭にある (上の行番号)。

#### スクリプト / 手

すべて手。数値の表は TensorBoard・JSON から手で写した (rope_study だけは `summarize_eval.py` の `summary.md` から)。

#### 失敗・食い違い

| 何が | 根拠 |
|---|---|
| 版の表の v3 が「smoke test のみ」のまま (結果の報告書と `next_action.md` は 25%) | `kt/README.md:41` と `kt/next_action.md:27` |
| 改名した `paper/Report.md` への参照が残る | C-16、`kt/next_action.md:25`、`:63` |
| 中間版の要旨が残ったまま時間が経つ (rope_study の §0・§10 は 18:50 時点の記述を後で書き直し) | `HANDOFF.md:136` |
| 「次にやること」の進捗を報告書の中で更新した (v6 §6 に「進捗」の注記) | `Result_Report_v6.md:187-188` |

#### 成果物の置き場所

報告書は 3 か所に分かれる: `kt/versions/vN/`、`kt/rope_study/`、`paper/` (すべて追跡。F.md §3.1)。名前の大文字小文字も揃っていない (`Result_report_v3.md` / `Result_Report_v6.md`)。

---

### 2.7 セッションの引き継ぎ

#### 回数

| 情報源 | 数 | 根拠 |
|---|---|---|
| `kt/next_action.md` の「現在地と次の作業」 | 作成 1 + 更新 4 コミット | 作成 `32c9f93` (09-28)、更新 `4115db4`・`db375b3`・`ccae2ae` (v5 の行)、`4c58709` (日付を 10-02 に、v3〜v7 の行、次の作業を書き直し)。`git show -U0` の差分の位置 (16〜52 行) で確認 |
| `kt/rope_study/HANDOFF.md` | **8 つの節** (4 セッション)、コミットは **1 回** (`b53f070`、最後) | 節: §1〜§7 (1 回目)、§0 (2 回目)、§00 (3 回目)、§000 (3 回目の終わり)、§0000・§0000.1b・§0000.1c (4 回目)、§00000 (最終)。`HANDOFF.md:3-6`、`:8`、`:34`、`:53`、`:69`、`:87`、`:158`、`:194` |
| 報告書の「次にやること」 | 4 か所 | `Result_report_v3.md:174`、`Result_Report_v6.md:185`・`:290`、`Report_Search.md:258`、`Report_rope_models.md:769` |
| 退避フォルダ・run の README | 4 | `outputs/rope_study/{contaminated_start,finger_v1,xpbd_world_coords}/README.md`、`logs/rsl_rl/knot_tying_v6_ablation/README.md` |

#### 手順の中身 (HANDOFF.md に毎回書かれたもの)

| 項目 | 書かれた節の数 | 例 |
|---|---|---|
| 走っているもの (何・GPU・ログ・出力) | 4 | `:78-85`、`:89-99`、`:160-168`、`:361-370` |
| 分かったこと・したこと | 5 | `:36-51`、`:103-121`、`:173-185`、`:213-242` |
| 次にやること (優先順) | 5 | `:16-25`、`:138-146`、`:187-192`、`:258-266`、`:372-407` |
| 注意 (踏んだもの) | 3 | `:27-32`、`:148-156`、`:409-418` |
| 別セッションとの取り決め (止めない・触らない) | 3 | `:101`、`:171`、`:289-298` |

新しい節を上に足し、`0` を 1 つずつ増やす形で、「← 最新」の印が古い節 (`:158`) に残っている。

#### スクリプト / 手

すべて手。

#### 失敗・食い違い

| 何が | 根拠 |
|---|---|
| 引き継ぎ先が 4 系統に分かれ、互いに参照していない: `next_action.md` (本線)、`HANDOFF.md` (rope_study)、`Report_Search.md` §6 (探索)、`Result_Report_v6.md` §6 (v6) | 上の表。`next_action.md` は rope_study と推論時探索 (10-03) に触れない (`git grep -n 'rope_study\|Report_Search' kt/next_action.md` が 0 件) |
| 「現在地」の更新が遅れる: v4 の行は v4 の実装 (09-30) から 2 日後の `4c58709` まで無かった | `git show 4115db4`〜`4c58709` の差分 |
| HANDOFF の古い記述: 切り分けの run の場所を `knot_tying_v6/`・`knot_tying_v7/` と書くが、今は `knot_tying_v6_ablation/` | `HANDOFF.md:291-292` |
| 他セッションの scratchpad のスクリプトを指したまま (失われる) | `HANDOFF.md:170` (`small_queue.sh`。後で `outputs/rope_study/diag_scripts/` に写した)、`:293` (`pipeline2.sh`) |
| 2 回目のセッションは Bash が使えず、ファイルの読み書きだけで引き継いだ | `HANDOFF.md:196-211` |

#### 成果物の置き場所

`kt/next_action.md`・`kt/rope_study/HANDOFF.md`・報告書 (追跡)。`outputs/`・`logs/` の README (管理外)。

---

### 2.8 追加: 2 回以上繰り返されたほかの手順

#### 2.8a 物理パラメータの調整

| # | 何を | 計測 | 決定 | 根拠 |
|---|---|---|---|---|
| 1 | ロープ `Twist` のヤング率 1e6 → 1e5、指を全開 | 目視 (台本デモ) | git の前 | `kt/next_action.md:204-216` |
| 2 | 物理 240 Hz | ランダム方策の発散率 | git の前 | `kt/next_action.md:319-338` |
| 3 | ロープ `Knot` (28 リンク) | ゲート 1 | `686af21` | `kt/next_action.md:484-541` |
| 4 | PhysX の接触バッファ | 学習の停止 → 新旧バッファの比較 (`divergence_bench/buf_{old,new}_ckpt_4096`) | `b345065` | `Result_report_v3.md:185-` |
| 5 | ソルバの位置反復 16 → 8 | ゲート 1 の再生 (7 値) + 発散 + 学習 4 時間 × 3 | `4115db4` | `kt/versions/v4/README.md:201-269` |
| 6 | めり込み解消速度 0.4 m/s・離す高さ | `divergence_bench` (`Adep01`〜`Adep05` など) + ゲート 1 の再生 (`v5_check`) | `db375b3` | `kt/versions/v5/README.md:100-163` |
| 7 | めり込み解消速度を 0.96 に戻す | 学習 (切り分け) + chain 評価 | `95ac19a` | `Result_Report_v6.md:215-282` |
| 8 | rope_study: XPBD の指 v2、局所座標、`Art56` の反復 16 / 32、摩擦 `LowMu` | ゲート 1 + 物理ベンチ | `3e0b7c9`・`6092606` | `HANDOFF.md:38-42`、`:108-112`、`Report_rope_models.md:808-814` |

型: 計測 (発散・ゲート 1 の再生) → 値を決める → 学習で確かめる。7 の教訓「物性を変えるときは、計測だけでなく学習で確かめること」(`kt/next_action.md:40-42`) は 6 → 7 の 1 回から出た。
値の変更は Hydra の上書き (`env.scene.rope.spawn.rigid_props.max_depenetration_velocity=...`) か `--cfg` で試し、決まったら版の設定に入れる。**ソルバ反復は Hydra から効かない** ので資産の値を上書きする (`kt/versions/v4/README.md:277-280`)。

#### 2.8b 初期アーカイブの作成・作り直し

| 回 | 時刻 | 出力 | 根拠 |
|---|---|---|---|
| 1 | 09-28 17:19 | `outputs/archive_seed/overhand_28links.pt` (v4〜v7 が読む) | `ls -l`、`kt/versions/v4/README.md:314` |
| 2 | 10-02 17:51〜 (指 v1) | `outputs/archive_seed/rope_study/finger_v1/` 4 + 剛体鎖 3 | `ls`、`finger_v1/README.md` |
| 3 | 10-02 22:06〜 (指 v2・world 座標) | `xpbd_world_coords/` 7 (`_straight`・`_trainseed`・`_fast` を含む) | `ls`、`HANDOFF.md:133-134` |
| 4 | 10-02 22:39 (まっすぐからやり直し) | `gate1_redo.sh` の `*_straight.pt` | `outputs/rope_study/diag_scripts/gate1_redo.sh`、`logs_gate1_redo.txt` |
| 5 | 10-02 23:51〜 (局所座標) | `outputs/archive_seed/rope_study/Xpbd*.pt` 4 | `logs_suite_v3_gpu{0,1}.txt` |
| 6 | 10-03 10:32 (FEM) | `Fem.pt` | `logs_suite_fem_gpu0.txt` |

いずれもゲート 1 の遅い尺の `--dump_archive` (§2.3)。上書きしてはいけないファイル (`overhand_28links.pt`) がある (`HANDOFF.md:298`)。学習が「無効な台本の結果」のアーカイブで始まった例がある (`HANDOFF.md:133-134`、`contaminated_start/README.md:8-9`)。

#### 2.8c 無効な結果の退避

| 回 | 何を | どこへ | 根拠 |
|---|---|---|---|
| 1 | v0・v1 の run (ログ名 `knot_tying`) | `knot_tying_v0/`・`v1/` (フォルダの更新時刻 09-24 12:06) | run の `params/agent.yaml`、`ls -l` |
| 2 | 途中の形から始まったゲート 1 | `outputs/rope_study/contaminated_start/` + README | 同 README |
| 3 | 指 v1 の結果 | `outputs/rope_study/finger_v1/` + README | 同 README |
| 4 | world 座標の XPBD の結果 | `outputs/rope_study/xpbd_world_coords/` + README | 同 README |
| 5・6 | 無効な学習 2 本 | run のフォルダに `INVALID.txt` (`make_eval_jobs.py` が飛ばす) | `logs/rsl_rl/knot_tying_rs_xpbdc{28,56}/*/INVALID.txt` |
| 7 | 切り分けの run 4 本 | `knot_tying_v6_ablation/` + README | 同 README、`kt/README.md:80-81` |

#### 2.8d 学習の前のスモークテスト

Hydra: `smoke_v6` (10-01 17:55)、`smoke_v5_baseline` (18:22)、`knot_tying_smoke` (10-02 12:22、21:38)。文書: v1 の学習前の動作確認 (`kt/versions/v1/README.md` §4)、v2-R2 の 16 環境 × 2 iteration (`kt/versions/v2_r2/README.md:48`)、v3 の smoke test (`kt/README.md:41`)、rope_study の `smoke_test.py` (`HANDOFF.md:344`)、FEM (`HANDOFF.md:47`)、評価のスモーク (`outputs/rope_study/eval_smoke_test/`)。

#### 2.8e 速度の計測

(1) VRAM と環境数 (`kt/next_action.md:473-476`、09-28) (2) v4 の A / B / C を 4 時間ずつ (`knot_tying_v4_speed/` 3 run、`kt/versions/v4/README.md:248-269`) (3) GPU 1 枚あたりの本数 (Hydra `knot_tying_bench` 5 起動、`Result_Report_v6.md:40-49`)。ほかに XPBD 単体の速度 (`HANDOFF.md:347`)。方法は毎回違う。

#### 2.8f テスト・lint の実行

記録: 241 件 (`HANDOFF.md:175`、`:302`)、245 件 (`:121`)、247 件 (`:14`、`:41`、`Report_rope_models.md:796`)。規則は `kt/README.md:234-241`。2 回目のセッションは回せなかった (`HANDOFF.md:256`)。F.md によると lint の基準 (`.pre-commit-config.yaml`) は存在しない (F.md §0 の 3)。

#### 2.8g 方策の書き出し

`exported/policy.{onnx,pt}` は v0・v1・v2・v2-R2 の 4 run にだけある。`sc/play.py:171-174` が再生のたびに書き出す副作用で、時刻は Play の起動と一致する (例: v2 の `exported/` 09-26 15:45 = Hydra `2026-09-26/15-45-16`)。v3 以降は `play.py` を使っておらず、書き出しも無い。**意図して繰り返した手順ではない。**

---

## 3. 研究の 1 サイクルの実際 (版ごと)

想定の順: 仮説 → 実装 → ゲート → 学習 → 評価 → 記録 → 判断。実際の順:

| 版 | 実際の順 (時刻は 2026 年) | 根拠 |
|---|---|---|
| v1 | 仮説 (位相クラスのラダー) → 実装 → 動作確認 → 学習 (09-15 21:49) → 157 iteration で打ち切り → 記録は次の v2 の README で | `kt/versions/v1/README.md:9`、`kt/versions/v2/README.md:17-41` |
| v2 | 実装 → **ゲートを待たずに**学習 (09-24 14:07) → Play で再生 (09-26) → 「偽の C2」と判断 | `kt/versions/v2/README.md:248-249`、Hydra `2026-09-26/15-19-25` |
| v2-R2 | 実装コミット (09-26 15:56) → **1 分後に学習** (15:57) → iteration 1 で壊れる → 修正 (09-27) → 3 iteration の確認だけ → 論文と比較 (`92f9e86`) → ゲート 1 (09-27 18:23〜) → ロープ `Knot` | `f9c4e4c`、run `2026-09-26_15-57-53`、`kt/versions/v2_r2/README.md:134-136`、`:178` |
| v3 | 実装 (09-28 13:25) → 学習 1024 環境 (同時刻、後で削除) → 環境数 4096 (`a1c75cc` 17:28) → 学習 (17:28、49.6 時間、159 で停止) → 報告書 (`fabaa65` 09-30) → 停止の原因 → 発散の計測 → v5 | `Result_report_v3.md:1-31`、`kt/versions/v4/README.md:16-17` |
| v4 | **v3 の学習中に実装** (未コミット) → ゲート 1 の再生で反復を決める → 実装コミット (09-30 19:56) → 速度の比較 3 本 → `4115db4` → 3 時間の比較 → overnight → chain 評価 (10-02 11:14) → 記録 (`4c58709`) | `Result_report_v3.md:33`、`kt/versions/v4/README.md:201-305` |
| v5 | 発散の計測 (09-30 20:34〜10-01 06:44) → ゲート 1 の再生 (10-01 06:11) → 実装 (v3 ベース、13:47) → 3 時間の比較 → **v4 ベースに作り直し** (19:04) → overnight → chain 評価 9.6% → 記録 | §2.4、`881dc8d` |
| v6 | 準備 (`cd92d06` 17:46) → スモーク (17:55) → 実装コミット (18:51) → overnight 2 シード (19:23) → chain 評価 → 報告書 (`4c58709`) | §2.2 |
| v6 の切り分け → v7 | 学習 4 本 (10-02 12:33、iteration 70 で停止) → chain 評価 (21:36〜) → 追記 → **v7 を定義し直してコミット** (23:09) → 確認の学習は保留 | `Result_Report_v6.md:215-288` |
| v7 | 実装のみ。切り分けの run を v7 の代わりに推論時探索で評価 (10-03) | `kt/versions/v7/README.md:70-71`、`paper/Report_Search.md:292-295` |
| rope_study | 調査 → 実装 → 物理試験 (ゲート 1 + 物理ベンチ、17:51) → 不具合の発見と修正を 3 回 → やり直し → 学習 (21:39〜) → 評価キュー → 報告書 → まとめてコミット (10-03 13:09、5 コミット) | `HANDOFF.md:6`、§2.3 |

読み取れること:

* **ゲート 1 は「版ごとの関門」としては 1 回 (09-27) しか置かれていない。** 以後は物理を変えたときの確認 (v4・v5・rope_study) として回った。v2 はゲートを待たずに学習した。
* **実装のコミットは実験の後か途中。** 実装前の起動 (v4・v6)、実装と同時刻の学習 (v2-R2・v3)、作り直し (v5) がある。
* **評価の型は v4 で変わった。** v0〜v3 は TensorBoard と Play の目視、v4 以降は chain 評価 (`eval_chain.py`、10-02 23:09 コミット)。
* **判断の記録は次の版の README か報告書の追記に入る。** 判断だけをまとめた場所は `next_action.md` の「現在地」の 1 行ずつ。

---

## 4. 既存の知見の確認

| 知見 | 出典 (すべて) | 元のデータ | 比べたシード数 | 確認の結果 |
|---|---|---|---|---|
| 「同じ版の 2 シードが 15〜19 ポイント離れる」 | `Result_Report_v6.md:19`、`:277-278`、`:293`、`kt/next_action.md:46`、`kt/README.md:97`、`kt/versions/v6/README.md:162`、`paper/Report_Search.md:30`、`:269` | v6 s42 / s43 の chain 評価: it50 22.5 / 17.6 (差 **4.9**)、it70 35.2 / 16.0 (差 **19.2**)、it100 37.9 / 22.9 (差 **15.0**) (`Result_Report_v6.md:80-83`、`:257-258`) | **2 (v6 だけ)** | 数字は正しい。ただし 1 組 (1 版 × 2 シード) からの値で、it50 の 4.9 ポイントは範囲に入っていない |
| 「512 エピソードの標準誤差は 2 ポイント前後」 | `Result_Report_v6.md:15`、`:264`、`kt/README.md:97` | 二項分布で p = 0.4 なら √(0.4 × 0.6 / 512) ≈ 2.2 ポイント (本書の計算) | — | 妥当 |
| 「版の比較は 2 シード以上で」 | `kt/README.md:97-98` | — | — | **守られたのは v6 だけ。** v4・v5・切り分け 4 本・rope_study 6 モデルはすべて 1 シード (42)。seed 43 の 3 本のうち chain 評価まで済んだのは v6 だけ (`XpbdC28_s43`・`XpbdC56_s43` は評価を止めた) |
| 「v4 は iteration 50 で頭打ち」 | `kt/next_action.md:28`、`Result_Report_v6.md:16` | 39.3 → 41.6 (it50 → it100) | 1 | 1 シードの観察 |
| 「1 GPU に 2 本で +12%」 | `Result_Report_v6.md:44-48` | 2 iteration のベンチ (Hydra `knot_tying_bench` 5 起動、10-01 19:07〜19:09) | — | 1 回の計測 |
| 「物性を変えるときは学習で確かめる」 | `kt/next_action.md:40-42` | v5 → 切り分け (b) | 1 | 1 回の経験から |

シードの使い方のまとめ: 学習のシードは 42 (26 run) と 43 (3 run)。評価のシードは常に 7 (`eval_chain`・`eval_robust`・`eval_search`)。ゲート 1 の種は既定 0、v5 の確認だけ 0 / 1、v4 の反復の振りは 0 / 1 / 2。

---

## 5. 所見

| 番号 | 所見 | skill の根拠 | 根拠 |
|---|---|---|---|
| E-1 | **学習 → chain 評価は skill にする根拠がある** (run 29、chain 評価の JSON 15、まとまり 3)。固定してよい引数: `--headless`、`--device` と `agent.device` を揃える、`agent.save_interval=10`、`--run_name <略>_s<シード>`、評価は 512 環境・seed 7・決定論的、物理を上書きした run は評価にも同じ上書き。ばらつきが大きいのは `max_iterations` と止め方 (時刻・iteration 70・完走) | **ある** | §2.2 |
| E-2 | **版の追加は skill にする根拠がある** (コミット 6 + 作り直し 1)。手順の漏れが毎回出ている: `versions/__init__.py` は 3/6、ルート README 1/6、`next_action.md` 1/6。旧 v7 の番号の使い直しもある | **ある** | §2.1 |
| E-3 | **ゲート 1 は skill にする根拠がある** (出力 76、4 つの時期)。過去の失敗から、(1) まっすぐな紐から始める (`a9b02b3` の修正)、(2) 物理の版ごとに出力を分ける、(3) `[final]` 行を表にする、を手順に含める必要がある | **ある** | §2.3 |
| E-4 | **発散の計測は skill にする根拠がある** (4 時期・文書 7 か所、出力 38 + 18)。ただし道具が 2 つ (`divergence_bench`・`bench_physics`) で判定の定義が違う (C-7)。skill は「どの判定で数えたか」を出力に書かせる必要がある | **ある** | §2.4 |
| E-5 | **評価キューは skill にする根拠があるが、道具は rope_study 専用。** v 系列の評価 (chain 評価 15、探索 11 起動) を流したシェルはリポジトリに無い (`pipeline2.sh`・`eval_queue2.sh` は消えた)。キューのシェル 7 本は同じ型 (絶対パスの `cd`、出力があれば飛ばす、`timeout`、ログ、`grep` の要約) | **ある** | §2.5 |
| E-6 | **結果報告書は skill にする根拠がある** (4 本)。3 本が同じ骨組み (要旨 → 条件 → 結果 → わかること → 注意 → 次 → 再現 → 追記) と「解釈」の断りを持つ。版の表 (`kt/README.md:41`) の更新漏れが起きている | **ある** | §2.6 |
| E-7 | **引き継ぎは skill にする根拠がある** (「現在地」5 コミット、HANDOFF 8 節)。ただし引き継ぎ先が 4 系統に割れ、`next_action.md` は 10-03 の作業 (rope_study・探索) を含まない。HANDOFF の項目 (走っているもの・分かったこと・次・注意・別セッションとの取り決め) は毎回同じ | **ある** | §2.7 |
| E-8 | **物理パラメータの調整は skill にする根拠がある** (7 件以上)。中身は E-4 (計測) + E-3 (ゲート 1 の再生) + E-1 (学習で確認) の組み合わせ | **ある** (E-1・E-3・E-4 を呼ぶ上位の手順として) | §2.8a |
| E-9 | 初期アーカイブの作成は 6 回。ゲート 1 の遅い尺の副産物なので、E-3 の一部にするのがよい | **ある** (E-3 に含める) | §2.8b |
| E-10 | 無効な結果の退避は 7 件。README の型 (何が・なぜ無効・どこを使うか) が毎回同じ | **ある** (小さい) | §2.8c |
| E-11 | スモークテストは 8 件以上。E-1 の前段 | **ある** (E-1 に含める) | §2.8d |
| E-12 | 速度の計測は 3 回だが方法が毎回違う | ある (弱い。手順として固まっていない) | §2.8e |
| E-13 | テスト・lint は 5 回以上の記録。決まった 1 行のコマンドなので skill より hook 向き | ある (hook 向き) | §2.8f |
| E-14 | 方策の書き出しは `play.py` の副作用だけ。意図した繰り返しは無い | **ない** | §2.8g |
| E-15 | **道具は使った後にコミットされた** (`overhand_script.py` 09-27 使用 → 09-28 コミット、`divergence_bench.py` 09-30 → 10-01、`eval_chain.py` 10-01 → 10-02 23:09、rope_study 10-02 → 10-03)。学習の run には rsl_rl が `git/` に差分を残すが、ゲート 1・発散・評価の出力にはコードの版の記録が無い。さらに `eval_chain_*.json` はシードと Hydra の上書きを持たない (キーは `task`・`checkpoint`・`stochastic` と結果だけ)。`eval_robust`・`eval_search` の JSON は `seed`・`hydra` を持つ | — (skill の設計に効く) | `git log` (§2.3〜§2.5)、JSON のキーの一覧 |
| E-16 | **「シード差 15〜19 ポイント」は v6 の 2 シードの 1 組だけが出典。** it50 では 4.9 ポイント。「2 シード以上で比べる」規約 (`kt/README.md:97-98`) を守ったのは v6 だけ | — | §4 |
| E-17 | ログ名の付け方が 3 通り (設定 / Hydra の `agent.experiment_name` / 後から手で移動)。C.md §7 の未確認 2 件 (`knot_tying_v4_speed`・`knot_tying_v6_ablation` の付け方) は、前者が Hydra の上書き (`outputs/2026-09-30/22-2*/.hydra/overrides.yaml`)、後者が手での移動 (run の `params/agent.yaml` は `knot_tying_v6` / `v7`) と分かった | — | §2.2 |
| E-18 | 学習の起動 68 のうち run が残るのは 27。消えた run の記録は Hydra のフォルダ (管理外) にしか無い (v3 の 1024 環境の run、v3 ベースの v5 の 3 時間 run など) | — | §1、§2.2 |
| E-19 | 研究のサイクルは「実装 → ゲート → 学習」の順に固定されていない (§3)。ゲート 1 は 09-27 の 1 回だけが関門で、以後は物理を変えたときの確認。評価の型は v4 で TensorBoard・Play から chain 評価に変わった | — | §3 |

---

## 6. 確認できなかったこと

* `pipeline2.sh`・`eval_queue2.sh` (v 系列の chain 評価を流したシェル) の中身。別セッションの scratchpad にあったと `HANDOFF.md:101`、`:171`、`:293` が書くが、ファイルは見つからない。
* v4 のソルバ反復の振り (ゲート 1 の再生 10 回、`kt/versions/v4/README.md:210-218`) の出力の場所。`outputs/overhand_script/` に該当する名前が無い。
* `eval_chain_*.json` 15 本をいつ・どうやって run のフォルダへ置いたか (更新時刻はすべて 10-02 23:04)。
* 推論時探索の Hydra の起動 11 回と出力 9 個の差 2 の中身。
* 消えた run (v3 の 1024 環境、v3 ベースの v5 の 20 iteration・3 時間など) を誰がいつ消したか。
* 09-10〜09-15 の Hydra の起動 (ログ名 `knot_tying`) のどれが v0 / v1 の何の試行か。
* 09-10 と 09-27 の発散の計測 (`next_action.md:319-338`、`v2_r2/README.md:138`) に使った道具。
* セッションの総数。rope_study の 4 セッションと、名前の出る別セッション (`2f001223-...`、`8f42943d-...`、`7e308401-...`) 以外は記録が無い。
* 学習ログ (TensorBoard) の中身は開いていないので、文書の数値 (成功率など) が実際のログと一致するかは確かめていない。

---

## 付録 A. 使ったコマンド・スクリプト (scratchpad。リポジトリには置いていない)

### A-1 Hydra のフォルダの読み取り (`hydra_scan.py`)

```python
# outputs/<date>/<time>/.hydra の config.yaml から数個のキーだけを正規表現で抜き、logs/rsl_rl の run 名の時刻と突き合わせる
import glob, os, re
ROOT = "/home/itoyama/work/isaac_twisted_rl"
runs = {os.path.basename(p.rstrip("/"))[:19]: p for p in glob.glob(f"{ROOT}/logs/rsl_rl/*/*/")}
for d in sorted(glob.glob(f"{ROOT}/outputs/2026-*/*/")):
    date, tm = d.rstrip("/").split("/")[-2:]
    txt = open(os.path.join(d, ".hydra/config.yaml")).read()
    exp = re.search(r"^  experiment_name: (.*)$", txt, re.M)
    ne = re.search(r"^    num_envs: (\d+)", txt, re.M)
    rope = re.search(r"rope_model/data/rope_(\w+)\.usd", txt)
    chain = set(re.findall(r"^\s+chain: (true|false)", txt, re.M))
    rm = re.search(r"^  rope_model: (\S+)", txt, re.M)
    ovs = [l.strip()[2:] for l in open(os.path.join(d, ".hydra/overrides.yaml")) if l.strip().startswith("- ")]
    print(f"{date}_{tm}", exp and exp.group(1), ne and ne.group(1), rope and rope.group(1), chain, rm and rm.group(1),
          runs.get(f"{date}_{tm}", ""), ";".join(ovs), sep="\t")
# 分類: experiment_name == "reach" はテンプレート、num_envs == 9 は Play・評価、それ以外は学習の起動
```

### A-2 その他

```bash
# run の設定 (シード・iteration・ログ名・環境数・USD)
for d in logs/rsl_rl/*/*/; do grep -m1 -E '^seed:' $d/params/agent.yaml; grep -m1 -E '^max_iterations:' $d/params/agent.yaml; \
  grep -m1 -E '^experiment_name:' $d/params/agent.yaml; grep -m1 -E '^\s+num_envs:' $d/params/env.yaml; done
# ゲート 1 の結果
find outputs/overhand_script outputs/rope_study -name summary.log | xargs grep -H '^\[final\]'
# 評価キューの進み具合
grep -c ' start ' outputs/rope_study/eval/*.txt; grep -c ' done ' outputs/rope_study/eval/*.txt
# 評価 JSON の数
find outputs/rope_study/eval outputs/rope_study/xpbd_world_coords/eval -name '*.json' | wc -l   # 117
ls logs/rsl_rl/*/*/eval_chain_*.json | wc -l                                                   # 15
# 版の作成コミットと、そのとき触ったファイル
git log --diff-filter=A --format='%h %ad %s' -- kt/versions/vN/__init__.py
git show --stat --format='' <commit> -- README.md kt/README.md kt/__init__.py kt/versions/__init__.py kt/next_action.md tests
# 「現在地」の更新
git log --format=%h --follow -- kt/next_action.md; git show -U0 <commit> -- kt/next_action.md | grep '^@@'
# 旧 v7 の報酬項がコミットされたことがあるか
git log --format='%h %s' -S stage_time_out
```

---

## メインによる根拠の確認 (2026-10-03)

この subagent は完了の報告を返した。その後、session の切り替えで「完了の記録なし」と通知されたが、ファイルは最後まで書かれている。
メインの session が、次の 3 か所を自分で確かめた。誤りは見つからなかった。

| 所見 | 確かめた内容 | 結果 |
|---|---|---|
| `eval_chain_*.json` は 15 本で、更新時刻がすべて 10-02 23:04 | `find logs outputs -name "eval_chain_*.json"` は 15 本。更新時刻は 15 本とも 2026-10-02 23:04 | 一致 |
| E-5 v4〜v6 の chain 評価を流したシェルが残っていない | `pipeline2.sh`・`eval_queue2.sh` は HANDOFF.md:101・:171・:293 で言及されるだけ。`find logs outputs` でも 0 本 | 一致 |
| E-17 `knot_tying_v4_speed` は Hydra の上書き | `outputs/2026-10-01/02-27-56/.hydra/overrides.yaml` と `outputs/2026-09-30/22-20-11/.hydra/overrides.yaml` に `agent.experiment_name=knot_tying_v4_speed` がある | 一致 |
