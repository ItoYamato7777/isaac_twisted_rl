# STATUS (現在地)

- 状態: 現行。上書きで更新する (追記しない。手順は skill `handoff`)
- 読むべき状況: セッションの始め。今どこにいて、次に何をするかを知るとき
- 最終更新: 2026-10-08

## 要点

- **v11 (行き止まりの形だけを −1。[0024](decisions/0024-v11-dead-end-stages.md)) は iteration 300 で v7 を上回らなかった** (chain 評価。比べる相手の v7 は s43 の延長の 1 シードだけ)。v11 s44 は iteration 150 以降、段 0 で交差を作らない手を繰り返す崩れ方をした ([記録](experiments/2026-10-07_v11_dead-end-stages.md))。
- 次は v7 s44 を iteration 300 まで延ばして比べる相手を 2 シードにすることと、s44 の崩れが v11 の段の定義のせいかの切り分け。v11 を続けるかは、その後にユーザーと決める。
- v10・v9・v8 は v7 を上回らなかった。rope study は打ち切りのまま。

## 版の状態

数字は各記録にある (ここには書かない)。比べ方は [`docs/system/evaluation.md`](system/evaluation.md)。

| 版 | 状態 | 記録 |
|---|---|---|
| v0 | 学習した (交差 1 つは学べた) | [2026-09-15_v0-v1](experiments/2026-09-15_v0-v1.md) |
| v1 | 学習した (失敗: 疎報酬が一度も発火しない) | 同上 |
| v2 | 学習した (偽の C2 に居座った) | [2026-09-24_v2_fake-c2](experiments/2026-09-24_v2_fake-c2.md) |
| v2-R2 | 発散の報酬の修正 (2026-09-27) の後の本番の学習は**未実施**。「本物の C2 が稀で詰まる」は、ランダム方策と数 iteration の観察からの**仮説** | [2026-09-27_v2r2_fix](experiments/2026-09-27_v2r2_fix.md) |
| v3 | 本番の run 1 本 (シード 1 本) | [2026-09-28_v3_main](experiments/2026-09-28_v3_main.md) |
| v4 | chain 評価 済み (seed 42 だけ。2 本目のシードは未実施) | [2026-10-01_v4-v6_overnight](experiments/2026-10-01_v4-v6_overnight.md) |
| v5 | chain 評価 済み (seed 42 だけ) | 同上 |
| v6 | chain 評価 済み (seed 42・43、GPU 1 枚に 2 本) | 同上 |
| v7 | chain 評価 済み (seed 43・44 は iteration 100 まで、GPU 1 枚に 1 本。seed 42 に当たる `abl_depB` は iteration 70 まで) | [2026-10-04_v7_seeds](experiments/2026-10-04_v7_seeds.md) |
| v8 | chain 評価 済み (seed 43・44、iteration 70 / 100、GPU 1 枚に 1 本)。v7 を下回った | [2026-10-05_v8_symmetry](experiments/2026-10-05_v8_symmetry.md)、判断は [0022](decisions/0022-p2-one-version-each.md) |
| v9 | chain 評価 済み (seed 43 は iteration 70 / 100、seed 44 は timeout で iteration 90 まで)。v7 を上回らなかった | [2026-10-05_v9_stage2-weight](experiments/2026-10-05_v9_stage2-weight.md) |
| v10 | chain 評価 済み (seed 43・44、iteration 70 / 90 / 100、GPU 1 枚に 1 本)。v7 を大きく下回った | [2026-10-06_v10_path-stages](experiments/2026-10-06_v10_path-stages.md)、[0023](decisions/0023-v10-path-stages.md) |
| v11 | chain 評価 済み (seed 43・44、iteration 70〜300、GPU 1 枚に 1 本)。iteration 300 で v7 を上回らなかった。s44 が段 0 で崩れた | [2026-10-07_v11_dead-end-stages](experiments/2026-10-07_v11_dead-end-stages.md)、[0024](decisions/0024-v11-dead-end-stages.md) |

## 走っているジョブ

| job (`outputs/jobs/`) | pid (`timeout`) | GPU | run (`logs/rsl_rl/knot_tying_v11/`) | 1 iteration | 終わりの見込み |
|---|---|---|---|---|---|
| `2026-10-08_19-23-57_v11_s43_to1000` | 3371712 | 1 | `2026-10-08_19-24-03_v11_s43_to1000` | 約 4.8 分 | 2026-10-11 3 時ごろ (`model_1000.pt`) |

- v11 s43 を `model_300.pt` から iteration 1000 まで延ばす (ユーザーの指示、2026-10-08)。v7 s43 の延長と同じ形: `uv run train --task Isaac-Twisted-RL-Knot-v11 --headless --device cuda:1 --seed 43 --resume --load_run 2026-10-07_14-11-04_v11_s43 --checkpoint model_300.pt --max_iterations 701 --run_name v11_s43_to1000 agent.device=cuda:1 agent.save_interval=10 env.events.reset_to_stage.params.load_paths=[<元の run の archive.pt (it300 の時点)>]` (`run_job.sh --timeout 72h --detach`)。commit `072e4c7`。

- GPU 0・1 は他のユーザー (ohtsukatakeshi) とも共有している。他の人のジョブがあれば触らない。
- 完走の確認は `exit.txt` に `signal:` の行が無いことと最後のチェックポイント。300 iteration は 1 本約 21〜23 時間 (v11、GPU 1 枚に 1 本)。

## 次の作業

### v11 の比べる相手と切り分け ([記録](experiments/2026-10-07_v11_dead-end-stages.md))

1. **v7 s44 を iteration 300 まで延ばす** (比べる相手を 2 シードにする)。v7 s43 の延長と同じ形: `uv run train --task Isaac-Twisted-RL-Knot-v7 --headless --device cuda:N --seed 44 --resume --load_run 2026-10-04_22-37-47_v7_s44 --checkpoint model_100.pt --max_iterations 201 --run_name v7_s44_to300 agent.device=cuda:N agent.save_interval=10 env.events.reset_to_stage.params.load_paths=[/home/itoyama/work/isaac_twisted_rl/logs/rsl_rl/knot_tying_v7/2026-10-04_22-37-47_v7_s44/archive.pt]` (`run_job.sh --timeout 20h --detach`。再開で 100 から 300 まで)。v11 は 0 から通しで条件がそろわないが、そろえずに比べる (ユーザーの判断、2026-10-07)。終わったら chain 評価 (seed 7、512 環境) を it200 / 300 で。
2. **v11 s44 の段 0 の崩れが段の定義のせいかを分ける** (仮説: 段 0 で行き止まりの形を作ると −1 になり、交差を作る手そのものが避けられた。対立仮説: シードによる偶然)。案: v11 s44 の `model_140.pt` から、v7 の段の定義 (`bfs`) と v11 のままの 2 通りで同じ長さ続け、段 0 の `timeout_s0`・`straight_success` と、`diag_stage_moves.py` の段 0 の「形が変わらない」手の割合を比べる。v7 の定義で続けるには、Hydra で `env.stage_definition=bfs` を渡せるか (または版を足すか) を先に確かめる。ユーザーと方法を決めてから起動する。
3. v11 を続けるか (iteration を延ばす・シードを足す・別の段の定義へ) は 1・2 の後にユーザーと決める。

### その他

4. **探索の設定を詰める** (1 回に 1 つ変える): `--slots 8` / `--slots 4`、`--rank random`、`--no_mean_first`。v7 s43 / s44 の `model_100.pt`、それ以外は推論時探索 v7 の記録の `search` と同じ。
5. **成功する段 2 の形と失敗する形の幾何の違いを見る** (`eval_search_shapes_it100_run_det.json` の `shapes_detail` の `index` でアーカイブの形を引き、`scripts/rsl_rl/view_archive.py` で並べる)。
6. その他の候補 (どれも未着手): v8 の切り分け ([v8 の記録](experiments/2026-10-05_v8_symmetry.md))、P2 の残り ([0022](decisions/0022-p2-one-version-each.md))、めり込み解消速度の中間値 ([overnight の記録](experiments/2026-10-01_v4-v6_overnight.md))。

## 保留・再開した研究

| 研究 | 止めた所 | 再開するときの 1 番目 | 記録 |
|---|---|---|---|
| 推論時探索 (ビーム探索・best-of-N) | 再開中 (2026-10-05 にユーザーの指示)。v7 の rollouts / search / shapes は済み。探索の設定の比較は未着手。v6 s43 の rollouts / search、v6 s42・v5 の search は止めたまま (再開するかはユーザーの判断) | 上の「次の作業」4 | [2026-10-06_inference_search_v7](experiments/2026-10-06_inference_search_v7.md)、[2026-10-03_inference_search](experiments/2026-10-03_inference_search.md) §6 |
| rope study (紐モデルの比較。打ち切り中、再開はユーザーの判断待ち) | 最終報告まで書いた。評価と実装は 2026-10-03 13:00 に打ち切り | XPBD の段 0 の方策の崩れの切り分け (結論を変えうる最大の未解決) | [2026-10-02_rope_models](experiments/2026-10-02_rope_models.md) §12 |

- rope study の発散の数値には、`rope_study/bench_physics.py:378` の数え方による過大の疑いがある (未確認。再開するときに確かめる)。
- 推論時探索の記録の「V8」は、2026-10-05 に作った v8 (対称性のデータ拡張) とは別に書かれたもので、中身の定義は無い。v7 を土台に探索を足す次の版の意味と読める。
