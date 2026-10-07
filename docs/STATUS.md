# STATUS (現在地)

- 状態: 現行。上書きで更新する (追記しない。手順は skill `handoff`)
- 読むべき状況: セッションの始め。今どこにいて、次に何をするかを知るとき
- 最終更新: 2026-10-07 (14:30)

## 要点

- v10 (段を正規の経路に絞る) は chain 評価で v7 を大きく下回った (固結び 0〜1.8%。段 1 -> 段 2 を定義が塞いだ)。[記録](experiments/2026-10-06_v10_path-stages.md)
- **次は v11 の実装** (ユーザーの指示、2026-10-07): v7 の段のまま、表に無くラダーでも 0 の「行き止まりの形」だけを −1 にし、偽の交差は R2 除去後の形で救う。仕様・根拠のデータ・実装の見通しは [0024](decisions/0024-v11-dead-end-stages.md)。seed 43 / 44 を iteration 300 まで学習し、v7 (s44 も 300 まで延ばす) と chain 評価で比べる。
- v9 は v7 を上回らなかった ([記録](experiments/2026-10-05_v9_stage2-weight.md))。対称性の切り分け (v8)・rope study は打ち切りのまま。

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
| v11 | 学習中 (seed 43・44、iteration 300 まで)。継承元 v7、差は表に無い形の段の付け方だけ | [0024](decisions/0024-v11-dead-end-stages.md)、[README](../src/isaac_twisted_rl/tasks/knot_tying/versions/v11/README.md) |

## 走っているジョブ

| job (`outputs/jobs/`) | pid (`timeout`) | GPU | run (`logs/rsl_rl/knot_tying_v11/`) | 1 iteration | 終わりの見込み |
|---|---|---|---|---|---|
| `2026-10-07_14-10-56_v11_s43` | 3195433 | 0 | `2026-10-07_14-11-04_v11_s43` | 約 4 分 | 2026-10-08 10 時ごろ |
| `2026-10-07_14-10-57_v11_s44` | 3195535 | 1 | `2026-10-07_14-11-04_v11_s44` | 約 6 分 | 2026-10-08 21 時ごろ |

- コマンド: `uv run train --task Isaac-Twisted-RL-Knot-v11 --headless --device cuda:N --seed S --max_iterations 301 --run_name v11_sS agent.device=cuda:N agent.save_interval=10` (Hydra の物理の上書きなし。`run_job.sh --timeout 40h --detach`)。commit `8fe0821`。
- スモークテスト (`outputs/jobs/2026-10-07_14-02-13_smoke_v11`) は通った (`stage_definition: bfs_dead_end`、`restore_mismatch` 0)。

- GPU 0・1 には他のユーザー (ohtsukatakeshi) の Isaac Lab のプロセス (PID 673851、各約 2 GB) が載っている。触らない。v9・v10 の学習が v7 より 1.2〜1.6 倍遅かったのは、これとの共有が疑わしい (未確認)。
- 完走の確認は `exit.txt` に `signal:` の行が無いことと最後のチェックポイント。遅い GPU では timeout を長めにする (v9 s44 は `timeout 10h` が足りずに止まった。v10 は 14h で足りた)。

## 次の作業

### v11 (行き止まりの形だけを −1 にする。仕様は [0024](decisions/0024-v11-dead-end-stages.md))

1. 実装は済み (commit `8fe0821`)。`outputs/archive_pdata/*.json` で 0024 の「データ」の表の v11 の列が再現した。
2. v11 seed 43 / 44 の学習 (上の「走っているジョブ」) の完走を確かめる (`exit.txt` に `signal:` が無いこと、`model_300.pt`)。
3. 比べる相手の v7 seed 44 を iteration 300 まで延ばす。**v11 の s43 が終わって GPU 0 が空いてから起動する** (GPU 1 枚に 1 本。ユーザーの判断、2026-10-07)。v7 s43 の延長 (`logs/rsl_rl/knot_tying_v7/2026-10-05_20-39-25_v7_s43_to500`、iteration 330 まで) と同じ形: `--resume --load_run 2026-10-04_22-37-42_v7_s44 --checkpoint model_100.pt --max_iterations 201 --run_name v7_s44_to300 env.events.reset_to_stage.params.load_paths=[<その run の archive.pt>]` (s43 の延長は `--max_iterations 401`。job `outputs/jobs/2026-10-05_20-39-20_v7_s43_to500` の `cmd.txt`)。**v11 は 0 から 300 まで通しで、iteration 100 で再開した v7 と条件がそろわない。これはそろえず、違い (再開での optimizer の状態の引き継ぎ・全 env のリセット・初期アーカイブが run の `archive.pt` だけになること) を記録に書いて比べる** (ユーザーの判断、2026-10-07)。
4. chain 評価 (seed 7、512 環境、Hydra の上書きなし) を iteration 100 / 200 / 300 (できれば 70・90 なども) で、v11 と v7 の 2 シードずつ。学習ログでは `Curriculum/stage_log/success_s1` (v7 並みの 0.55 前後に戻るか) と、段 0 のアーカイブの形 (`view_archive.py --dump_pdata` で iteration 100 / 300 の段 0 の −1 の割合) を見る。

### その他

5. **探索の設定を詰める** (1 回に 1 つ変える): `--slots 8` / `--slots 4`、`--rank random`、`--no_mean_first`。v7 s43 / s44 の `model_100.pt`、それ以外は推論時探索 v7 の記録の `search` と同じ。
6. **成功する段 2 の形と失敗する形の幾何の違いを見る** (`eval_search_shapes_it100_run_det.json` の `shapes_detail` の `index` でアーカイブの形を引き、`scripts/rsl_rl/view_archive.py` で並べる)。
7. その他の候補 (どれも未着手): v8 の切り分け ([v8 の記録](experiments/2026-10-05_v8_symmetry.md))、P2 の残り ([0022](decisions/0022-p2-one-version-each.md))、めり込み解消速度の中間値 ([overnight の記録](experiments/2026-10-01_v4-v6_overnight.md))。v7 s43 の延長 (chain 評価 it200 50.0%・it300 39.3%) の記録は未作成で、v11 の記録でまとめて書く。

## 保留・再開した研究

| 研究 | 止めた所 | 再開するときの 1 番目 | 記録 |
|---|---|---|---|
| 推論時探索 (ビーム探索・best-of-N) | 再開中 (2026-10-05 にユーザーの指示)。v7 の rollouts / search / shapes は済み。探索の設定の比較は未着手。v6 s43 の rollouts / search、v6 s42・v5 の search は止めたまま (再開するかはユーザーの判断) | 上の「次の作業」5 | [2026-10-06_inference_search_v7](experiments/2026-10-06_inference_search_v7.md)、[2026-10-03_inference_search](experiments/2026-10-03_inference_search.md) §6 |
| rope study (紐モデルの比較。打ち切り中、再開はユーザーの判断待ち) | 最終報告まで書いた。評価と実装は 2026-10-03 13:00 に打ち切り | XPBD の段 0 の方策の崩れの切り分け (結論を変えうる最大の未解決) | [2026-10-02_rope_models](experiments/2026-10-02_rope_models.md) §12 |

- rope study の発散の数値には、`rope_study/bench_physics.py:378` の数え方による過大の疑いがある (未確認。再開するときに確かめる)。
- 推論時探索の記録の「V8」は、2026-10-05 に作った v8 (対称性のデータ拡張) とは別に書かれたもので、中身の定義は無い。v7 を土台に探索を足す次の版の意味と読める。
