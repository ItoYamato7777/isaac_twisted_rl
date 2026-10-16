# STATUS (現在地)

- 状態: 現行。上書きで更新する (追記しない。手順は skill `handoff`)
- 読むべき状況: セッションの始め。今どこにいて、次に何をするかを知るとき
- 最終更新: 2026-10-16

## 要点

- **段 0 の崩れの原因の最有力は、時間切れの bootstrap の不具合** (v4 以降の全版。因果は未確認。[記録](experiments/2026-10-15_timeout-bootstrap-and-stage2-bottleneck.md))。直す版は **v15** (ユーザー、2026-10-16。寸法 DR より先に直すため v16 から繰り上げた。仕様と実装の手順は [0028](decisions/0028-v15-timeout-bootstrap-fix.md))。**v15 の学習は v14 の結果が出てから** (継承元をそのとき v13・v14 から決める。コードの実装とテストは先に進めてよい)。
- **崩れない run の頭打ち (30〜50%) は段 2 の詰まり**でほぼ説明できる (マルコフ連鎖の模型の上。同じ記録)。
- v14 seed 43 (GPU 1)・seed 44 (GPU 0) を学習中 (どちらも不具合を持つので、it100〜150 の崩れを見守る)。**次の session は v15 のコード実装から** (「次の作業」1。学習は v14 の結果が出て継承元を決めてから)。

## 版の状態

数字は各記録にある。比べ方は [`evaluation.md`](system/evaluation.md)。全記録に効く注意 (摩擦 0.5・不具合) は [実験記録の一覧](experiments/README.md)。

| 版 | 状態 | 記録 |
|---|---|---|
| v0〜v2-R2 | 古い版 (固結びは学べなかった)。v2-R2 は修正の後の本番の学習が未実施 | [一覧](experiments/README.md) の「v0〜v6」 |
| v3 | 本番の run 1 本 (初めて固結びを学べた) | [2026-09-28_v3_main](experiments/2026-09-28_v3_main.md) |
| v4〜v6 | chain 評価 済み (v4・v5 は seed 42 だけ、v6 は 42・43) | [2026-10-01_v4-v6_overnight](experiments/2026-10-01_v4-v6_overnight.md) |
| v7 | chain 評価 済み (seed 43・44 は it100 まで、s43 は延長で it330 まで)。**比べる相手の基準** | [2026-10-04_v7_seeds](experiments/2026-10-04_v7_seeds.md) |
| v8 / v9 / v10 | chain 評価 済み (seed 43・44)。どれも v7 を上回らなかった | [v8](experiments/2026-10-05_v8_symmetry.md)・[v9](experiments/2026-10-05_v9_stage2-weight.md)・[v10](experiments/2026-10-06_v10_path-stages.md) |
| v11 | chain 評価 済み (seed 43・44、it70〜300。s43 は it1000 まで延長)。v7 を上回らず、s44 が段 0 で崩れた | [v11](experiments/2026-10-07_v11_dead-end-stages.md)・[段 2 のループ](experiments/2026-10-09_v11_stage2-archive-loop.md) |
| v12 | chain 評価 済み (seed 43・44、it70〜300)。s43 は横ばい、s44 は段 0 から崩れた | [v12](experiments/2026-10-13_v12_archive-stage-up.md) |
| v13 | chain 評価 済み (seed 43・44、it70〜300)。罰則は張り付きの割合を減らさず、s44 は v12 s44 と同じ型に崩れた | [v13](experiments/2026-10-14_v13_bounds-loss.md) |
| v14 | **seed 43・44 を学習中** (段 0 の作り直し + DR + 特権観測の critic。ζ は ×1.0〜×2、ほかの 5 軸は ×0.5〜×2) | [0027](decisions/0027-v14-s0-reshape-dr-priv-critic.md)・[ゲート](experiments/2026-10-14_v14_physics-gate.md) |
| v15 | **未実装** (継承元 + 段の変化・発散と同じ手では時間切れを立てない。時間切れの bootstrap の不具合の修正。継承元は v14 の結果が出てから v13・v14 から決める。コードの実装・テストは先に進めてよい) | [0028](decisions/0028-v15-timeout-bootstrap-fix.md) |
| v16 | 未着手 (継承元 + ロープの直径・長さの DR。継承元は v15 の結果が出てから決める。2026-10-16 に bootstrap の修正と番号を入れ替えた) | [0029](decisions/0029-v16-rope-size-dr.md) |

## 走っているジョブ

| ジョブ | GPU | pid | job のフォルダ | run のフォルダ | 起動 | 終わりの見込み |
|---|---|---|---|---|---|---|
| v14 s43 (0 -> it300) | 1 | 262619 (`run_job.sh`、timeout 30h) | `outputs/jobs/2026-10-15_15-07-05_v14_s43` | `logs/rsl_rl/knot_tying_v14/2026-10-15_15-07-10_v14_s43` | 10-15 15:07 | 10-16 12:30 頃 (約 4.3 分/iteration) |
| v14 s44 (0 -> it300) | 0 | 344651 (同上) | `outputs/jobs/2026-10-15_16-07-16_v14_s44` | `logs/rsl_rl/knot_tying_v14/2026-10-15_16-07-22_v14_s44` | 10-15 16:07 | 10-16 15:00 頃 (約 4.6 分/iteration) |

- コマンド: `--task Isaac-Twisted-RL-Knot-v14 --device cuda:N --seed S --max_iterations 301 --run_name v14_sS agent.device=cuda:N agent.save_interval=10`。commit `d8dc8f7`。Hydra の上書きは `agent.*` だけ (評価には何も付けない)。この run には `Loss/value_mean_s*` が無い。
- 学習ログは it70〜90 まで健全 (`timeout_s0` 0.01〜0.02、v13 s44 の同じ時期と同じ水準)。
- **他のジョブ (触らない)**: anaconda の python の実機の作業 (pid 560983・616313・895144・2892930・876400)。GPU 0・1 は他のユーザーとも共有している。 別の Claude session の `smoke_v13_in_v14env` (`run_job.sh` pid 2044855、python pid 2044864、GPU 0、`eval_chain.py` v13 のチェックポイントを v14 の環境で 16 環境、timeout 40m。2026-10-16 11:51 起動)。

## 次の作業

1. **v15 の実装** (コードの実装・テスト・ゲートは先に進めてよい。学習は v14 の結果が出て継承元を決めてから)。仕様・外れと言える予想・実装の手順はすべて [0028](decisions/0028-v15-timeout-bootstrap-fix.md)。学習の前に継承元を v13・v14 から決める (ユーザーと。v14 が崩れず v13 より良ければ v14、崩れれば v13 が目安)。決めたら、継承元の終了項の名前と PPO が `Loss/value_mean_s*` を出すかを確かめ、0028 の手順 (時間切れの項 `time_out_unless_terminated` を `common/mdp/terminations.py` に足す -> skill `add-version` -> テスト -> ゲート -> commit -> seed 43 / 44 を it300) に進む。比べる相手は継承元の同じシード。 実装の入口: 0028 の「実装の見通し」1 (`common/mdp/terminations.py` に追加)。最初に確かめること: `time_out` の項が `stage_up` より先に評価される時点で、`stage_changed` の抽出結果が今の手の後の形か (`get_extractor` の更新の仕組みを読む)。GPU は v14 が 2 枚とも使っているので、Isaac Sim を要るゲート・スモークテストは v14 の終わり (10-16 の 12:30〜15:00 頃) か、空きを見てユーザーに確かめてから。
2. **v14 の見守り**: it100〜150 で `timeout_s0` が 0.1 を超えないか (v12 s44 は it100 前後、v13 s44 は it120 前後で超えた)。崩れたら、そのチェックポイントで `diag_stage_moves.py --dump_obs` -> `chain_moves.py sequences` で 6 手目の山を数える。完走は 2 本とも `exit.txt` に `signal:` の行が無いことと `model_300.pt` で確かめる。
3. **v14 の chain 評価**を it70 / 100 / 150 / 200 / 250 / 300 で v13 と並べる (公称の物理と、`env.events.rope_dr.params.enabled=true` の DR 付き)。0027 の予想を確かめる。1 点の差で結論しない (隣のチェックポイントと平均 9 ポイント揺れる)。内訳を並べるなら v13 も測り直す。
4. **頭打ち (段 2) の切り分けの計測** (GPU を使う前にユーザーに確かめる。コマンドの案は[記録](experiments/2026-10-15_timeout-bootstrap-and-stage2-bottleneck.md)の「次の作業」): 段 2 の形ごとの最良の 1 手 (`eval_search.py --mode shapes --shape_moves 1`)、6 手目の山がほかの崩れた run にもあるか、36 手の chain 評価。
5. v16 の実装 ([0029](decisions/0029-v16-rope-size-dr.md) の「実装の順序」に従う。`add-version` -> `physics-check` -> スモークテスト -> commit。v15 の結果が出てから、継承元はそのとき決める)。未確認: `MultiAssetSpawnerCfg` が Articulation で動くか、`replicate_physics=False` での起動時間と学習速度。
6. v7 s44 を it300 まで延ばす (比べる相手を 2 シードにする。GPU が空いてから、順番はユーザーに確かめる): `uv run train --task Isaac-Twisted-RL-Knot-v7 --headless --device cuda:N --seed 44 --resume --load_run 2026-10-04_22-37-47_v7_s44 --checkpoint model_100.pt --max_iterations 201 --run_name v7_s44_to300 agent.device=cuda:N agent.save_interval=10 env.events.reset_to_stage.params.load_paths=[/home/itoyama/work/isaac_twisted_rl/logs/rsl_rl/knot_tying_v7/2026-10-04_22-37-47_v7_s44/archive.pt]` (`run_job.sh --timeout 20h --detach`)。
7. v11 s43 の延長の chain 評価を it400 / 500 / 600 / 700 / 800 で (it1000 は済み)。
8. 探索の設定を 1 つずつ比べる (`--slots 8` / `4`、`--rank random`、`--no_mean_first`。v7 s43 / s44 の `model_100.pt`)。
9. 学習のアルゴリズムの見直しの候補 (効くかは未確認。3 の結果を見てから選ぶ): 離散 2 チャネルをカテゴリカルにする、観測・価値の正規化、探索の結果の蒸留、段が上がった手を次の段の V で bootstrap する。
10. 摩擦を意図どおり (ロープ 2.0、指 4.0 の `max`) にする版。skill `physics-check` から ([記録](experiments/2026-10-14_friction-binding.md))。
11. `run_job.sh` は `bash -c "..."` の中の `--device cuda:N` を読まない。包むときは `--check-gpu N` を付ける (直すかは未定)。
12. その他の候補: std の下限、形ごとの成功率で優先して引く (0025 の「見込みの弱点」)、v8 の切り分け、P2 の残り ([0022](decisions/0022-p2-one-version-each.md))、めり込み解消速度の中間値。

## 保留

| 研究 | 状態・誰の判断待ちか | 再開するときの 1 番目 | 記録 |
|---|---|---|---|
| 推論時探索 | 再開中 (2026-10-05、ユーザーの指示)。v6・v5 の止めた評価を再開するかはユーザーの判断 | 「次の作業」8 | [v7](experiments/2026-10-06_inference_search_v7.md)・[中間](experiments/2026-10-03_inference_search.md) |
| eval_chain.py の提案 #4〜#6 | ユーザーが後回しにした (2026-10-15) | 二項の揺れを実測するなら #6 か `--num_envs` を変える評価 | [記録](experiments/2026-10-15_eval-chain-repro.md) |
| rope study | 打ち切り中 (2026-10-03)。再開はユーザーの判断 | XPBD の段 0 の崩れの切り分け (不具合も候補)。発散の数値の数え方の過大の疑いも確かめる | [要約](experiments/2026-10-02_rope_models.md) |
