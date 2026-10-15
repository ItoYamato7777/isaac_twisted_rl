# ログ: フォルダの名前・run のフォルダの中身・ログのキーの意味

- 状態: 現行
- 読むべき状況: 学習の run を探すとき、チェックポイントを選ぶとき、TensorBoard のキーを読むとき、`logs/`・`outputs/` の置き場所を決めるとき
- 最終更新: 2026-10-03

`logs/` と `outputs/` は git の管理外で、消えると戻らない。再現に要る情報 (run のフォルダ名・commit・コマンド) は、実験記録 (`docs/experiments/`) に書く。
Claude は `logs/`・`outputs/` を手で編集しない (書くのは学習・評価のスクリプト)。

## 1. 学習のログのフォルダ

`logs/rsl_rl/<ログ名>/<run>`

| 部分 | 決まり方 | 根拠 |
|---|---|---|
| `<ログ名>` | エージェントの設定の `experiment_name`。版ごとに `knot_tying_vN` (v2・v2-R2・v3 の変種は `_rnn`・`_lstm` が付く)。rope study は紐モデルごとに `knot_tying_rs_<紐モデルの小文字>` | `scripts/rsl_rl/train.py:140`、`versions/*/agents/rsl_rl_ppo_cfg.py` (例 `versions/v4/agents/rsl_rl_ppo_cfg.py:52`)、`rope_study/agents/rsl_rl_ppo_cfg.py:21` |
| `<run>` | 学習を始めた時刻 `%Y-%m-%d_%H-%M-%S`。`--run_name` を付けたときだけ `_<run_name>` が続く | `scripts/rsl_rl/train.py:143-149`、`scripts/rsl_rl/cli_args.py:85-86` |

注意:

- **ロープのプリセットはログ名に入らない。** 同じ版の 5 つのプリセットは同じエージェントの設定で登録されるので、同じ `knot_tying_vN/` に入り、フォルダ名で違うのは時刻だけになる (`knot_tying/__init__.py:97-108`)。`--resume` は最新の run を拾うので、別のプリセットの run を読みうる。ロープ `Knot` 以外で学習するときは、`--run_name` にプリセット名を入れる (`docs/glossary.md` の命名規則)。
- **ログ名を変えるには `--experiment_name <名前>` を渡す** (`update_rsl_rl_cfg` が設定に写す。`scripts/rsl_rl/cli_args.py`、試験は `tests/test_cli_args.py`)。Hydra の `agent.experiment_name=<名前>` でも同じ。渡さなければ版の設定の名前のまま。評価・`play` も同じ名前を渡さないと、そのフォルダを探さない。
- コードに無いログ名が 2 つある (refactor の調査 E-17):
  - `knot_tying_v4_speed`: Hydra の上書き `agent.experiment_name` で作った (`outputs/2026-09-30/22-2*/.hydra/overrides.yaml`)。
  - `knot_tying_v6_ablation`: v6 の切り分けの run を後から手で移した。各 run の `params/agent.yaml` のログ名は `knot_tying_v6` か `knot_tying_v7`。v7 の参照 run (`abl_depB`) もここにある。

## 2. run のフォルダの中身

| ファイル | 中身 |
|---|---|
| `model_<it>.pt` | チェックポイント。`agent.save_interval` ごと |
| `params/env.yaml`・`params/agent.yaml` | 学習の時点の環境・エージェントの設定。プリセットや物理の上書きはここで確かめる |
| `git/*.diff` | rsl_rl が保存した、学習を始めたときの作業ツリーの差分。**追跡しているファイルの差分だけで、未追跡のファイルの中身は入らない** (refactor の調査 D-6: 29 run 中 21 run が未コミットの変更の上で学習した) |
| `events.out.tfevents.*` | TensorBoard のログ |
| `archive.pt` | v3 以降の状態アーカイブ |
| `eval_chain_it{N}.json`・`eval_search_*_it{N}.json` | chain 評価・探索ありの評価の結果 (`docs/system/evaluation.md`) |

## 3. チェックポイントの番号

- 最後のチェックポイントの番号は run ごとに `ls` で確かめる。
- v4〜v6 の overnight の run でも、最後のチェックポイントは run ごとに違う (v6 s42 は `model_100.pt`、v5 と v6 s43 は `model_110.pt`: [`docs/experiments/2026-10-01_v4-v7_readme-notes.md`](../experiments/2026-10-01_v4-v7_readme-notes.md))。記録の値 (iteration 100 の chain 評価など) を再現するときは、記録の iteration のものを使う。
- rope study の run の最後は `model_99.pt` で、`model_100.pt` は無い (`docs/history/rope_study_HANDOFF.md:29`)。評価の道具 (`rope_study/make_eval_jobs.py`) は無いチェックポイントを黙って飛ばすので、番号を取り違えると評価が抜け落ちる (refactor の調査 D-4)。

## 4. ログのキーの意味

v3 の本番の run で確かめた意味 (`docs/experiments/2026-09-28_v3_main_detail.md` の §2)。v4 以降で同じかは、キーごとに未確認。

| キー | 実際の意味 |
|---|---|
| `Episode_Termination/*` | **全環境**のうち、直近のエピソードがその項で終わった環境の割合。項の合計はほぼ 1。まっすぐ始まりとアーカイブ始まりが混ざる |
| `Episode_Reward/*` | 終わったエピソードの報酬の合計を `episode_length_s` で割った値 (v3 では 37.95 s。37.95 倍すると 1 エピソードあたりの合計) |
| `Curriculum/archive_log/scratch_success` | その step で終わった「まっすぐ始まり」のエピソードのうち、`knot_tied` で終わった割合。**学習中の確率的な方策**の成績 |
| `Curriculum/archive_log/archive_success` | 同じくアーカイブ始まりのエピソード |
| `Curriculum/topology_log/*` | **1 手あたり・全環境の平均** (アーカイブ始まりを含む)。`div_clsK` は「手を打つ前の段が K で、その手で発散した」環境の割合 (段 K の中での割合ではない) |
| `Curriculum/stage_log/*` (v4 以降) | 段ごとの成功率と `chain_est`。chain 評価とは始め方も手数も違うので、版の比較に使わない (`docs/system/evaluation.md` §2) |
| `Loss/bounds` (v13 以降。`BoundedPPO`) | 1 iteration の mini-batch の罰則 Σ max(\|μ\| − 1, 0)² の平均 (係数を掛ける前。全段) |
| `Loss/value_mean_s{k}`・`Loss/value_over_s{k}` (`BoundedPPO` を使う版。2026-10-15 に追加。v13 の run には無い) | 段 k の rollout の価値 V の平均と、V が 1/γ を**超えた**割合 (`common/policies/value_stats.py`)。`Loss/` の下に出るのは runner が `loss_dict` の全キーをそこへ書くため。段 0 の V は 1 を超えないはずで、膨らむのは段 0 の崩れと同時期に起きた ([2026-10-15_critic-inflation](../experiments/2026-10-15_critic-inflation.md))。崩れの警報にするかは未確認 |

## 5. `outputs/` の中

| フォルダ | 中身 | 注意 |
|---|---|---|
| `outputs/archive_seed/` | v4 以降の学習が読む初期アーカイブ (`overhand_28links.pt` など)、rope study の初期アーカイブ | **上書きしない** (v4〜v7 が使う共有の資産で、戻せない) |
| `outputs/overhand_script/` | ゲート 1 (台本の固結び試験) の出力 | |
| `outputs/divergence_bench/` | 発散の計測の出力 | |
| `outputs/rope_study/` | rope study の物理試験・評価・ログ | |
| `outputs/<日付>/<時刻>/.hydra/` | Hydra が起動ごとに作るフォルダ。消えた run の起動の記録はここにしか無いことがある | |

ロープの USD (`common/rope_model/data/rope_<プリセット>.usd`) も管理外の共有の資産。作り直すときは `generate_assets.py --rope <名前>` で対象を絞る (`--rope` なしでは全プリセットを作り直す)。
