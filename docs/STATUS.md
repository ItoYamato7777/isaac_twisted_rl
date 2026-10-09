# STATUS (現在地)

- 状態: 現行。上書きで更新する (追記しない。手順は skill `handoff`)
- 読むべき状況: セッションの始め。今どこにいて、次に何をするかを知るとき
- 最終更新: 2026-10-09

## 要点

- v11 は iteration 300 で v7 を上回らなかった ([記録](experiments/2026-10-07_v11_dead-end-stages.md))。v11 s43 の延長で段 2 の成功が落ちた原因は、**到達状態アーカイブが毎手の形を保存するため、方策の抜け出せない形が溜まるループ**だった (方策は悪くなっていない。[切り分け](experiments/2026-10-09_v11_stage2-archive-loop.md))。
- **次は v12 の実装** (ユーザーの指示、2026-10-09): v7 + アーカイブに「段が上がった手の後の形」だけを保存する。仕様・根拠・実装の見通しは [0025](decisions/0025-archive-save-on-stage-up.md)。
- v11 s43 を iteration 1000 まで延ばす学習が GPU 1 で走っている。v10・v9・v8 は v7 を上回らなかった。rope study は打ち切りのまま。

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
| v11 | chain 評価 済み (seed 43・44、iteration 70〜300、GPU 1 枚に 1 本)。iteration 300 で v7 を上回らなかった。s44 が段 0 で崩れた。s43 は iteration 1000 まで延長中 | [2026-10-07_v11_dead-end-stages](experiments/2026-10-07_v11_dead-end-stages.md)、[2026-10-09_v11_stage2-archive-loop](experiments/2026-10-09_v11_stage2-archive-loop.md)、[0024](decisions/0024-v11-dead-end-stages.md) |
| v12 | **未実装**。仕様は決定済み (継承元 v7、差は到達状態アーカイブの保存の規則だけ) | [0025](decisions/0025-archive-save-on-stage-up.md) |

## 走っているジョブ

| job (`outputs/jobs/`) | pid (`timeout`) | GPU | run (`logs/rsl_rl/knot_tying_v11/`) | 1 iteration | 終わりの見込み |
|---|---|---|---|---|---|
| `2026-10-08_19-23-57_v11_s43_to1000` | 3371712 | 1 | `2026-10-08_19-24-03_v11_s43_to1000` | 約 4.2 分 | 2026-10-10 19〜20 時ごろ (`model_1000.pt`。10/9 23:24 に iteration 713) |

- v11 s43 を `model_300.pt` から iteration 1000 まで延ばす (ユーザーの指示、2026-10-08)。v7 s43 の延長と同じ形: `uv run train --task Isaac-Twisted-RL-Knot-v11 --headless --device cuda:1 --seed 43 --resume --load_run 2026-10-07_14-11-04_v11_s43 --checkpoint model_300.pt --max_iterations 701 --run_name v11_s43_to1000 agent.device=cuda:1 agent.save_interval=10 env.events.reset_to_stage.params.load_paths=[<元の run の archive.pt (it300 の時点)>]` (`run_job.sh --timeout 72h --detach`)。commit `072e4c7`。

- GPU 0・1 は他のユーザー (ohtsukatakeshi) とも共有している。他の人のジョブがあれば触らない。
- 完走の確認は `exit.txt` に `signal:` の行が無いことと最後のチェックポイント。300 iteration は 1 本約 21〜23 時間 (v11、GPU 1 枚に 1 本)。

## 次の作業

### v12 (到達状態アーカイブに段が上がった手の後の形だけを保存する。仕様は [0025](decisions/0025-archive-save-on-stage-up.md))

1. skill `add-version` で v12 を作る (継承元 v7)。0025 の「実装の見通し」の順に:
   - `kt/common/state_archive.py` に保存する行を選ぶ純 torch の関数 (`rows_to_save(ok, before, after, rule)`、`rule` は `every_move` / `stage_up`) と、`tests/test_state_archive.py` の単体テスト (0 -> 1・1 -> 2・0 -> 2 は保存、0 -> 0・2 -> 2・2 -> 1・1 -> 0・2 -> 3・−1 は保存しない、`every_move` は今と同じ)。
   - `kt/common/mdp/archive.py` の `record_stage_archive` と `ResetToStage.record` に引数 `save_on` (既定 `every_move`)・`action_name` (既定 `pick_place`)・`history_every` (既定 0) を足す。手の前後の段は `kt/common/mdp/stages.py` の `stage_transition`。既定の値で v4〜v11 の挙動が変わらないこと。`pop_stats` に段ごとの保存の件数 `saved_s{k}` を足す。
   - `versions/v12/` (`__post_init__` で `save_on = "stage_up"`、`history_every = 200` だけ) と `tests/test_v12_cfg.py` (`tests/test_v11_cfg.py` が手本)。
2. テスト・lint の後に commit し、skill `train-eval` のスモークテストで `params/env.yaml` の `save_on: stage_up`、`arc_size_s0` が 128 のまま、`saved_s0` が 0、`restore_mismatch` 0 を確かめる。
3. v12 seed 43 / 44 を 0 から iteration 300 まで (`--max_iterations 301`、`agent.save_interval=10`、timeout 30h 前後、GPU 1 枚に 1 本)。**GPU の割り当てと、v11 s43 の延長 (GPU 1) が終わるのを待つかをユーザーに確かめる。**
4. chain 評価 (seed 7、512 環境、Hydra の上書きなし) を it100 / 200 / 300 で、v7・v11 と並べる。0025 の「外れたら仮説が誤りと言える予想」(段 2 のアーカイブで 1 つの p-data が 20% を超えない、段 2 の成功が途中で半分以下に落ちない、chain 評価が it100 -> it300 で下がらない) を、`archive_history/` の写しと学習ログで確かめる。

### v7・v11 の比べる相手

5. **v7 s44 を iteration 300 まで延ばす** (比べる相手を 2 シードにする)。v7 s43 の延長と同じ形: `uv run train --task Isaac-Twisted-RL-Knot-v7 --headless --device cuda:N --seed 44 --resume --load_run 2026-10-04_22-37-47_v7_s44 --checkpoint model_100.pt --max_iterations 201 --run_name v7_s44_to300 agent.device=cuda:N agent.save_interval=10 env.events.reset_to_stage.params.load_paths=[/home/itoyama/work/isaac_twisted_rl/logs/rsl_rl/knot_tying_v7/2026-10-04_22-37-47_v7_s44/archive.pt]` (`run_job.sh --timeout 20h --detach`)。v12 と GPU を取り合うので、順番はユーザーに確かめる。
6. v11 s43 の延長 (iteration 1000) が終わったら、`exit.txt` と `model_1000.pt` を確かめ、chain 評価を it400 / 500 / 600 / 700 / 800 / 1000 で (`--checkpoint` は延長の run)。段 2 のアーカイブの写し (`outputs/archive_snapshots/`) も足して p-data を数える。
7. v11 s44 の段 0 の崩れ: 同じループ (段 0 の手の後の形が段 0 に保存される) が関わるかを、s44 の `archive.pt` (iteration 300) の段 0 の p-data で見る (`view_archive.py --dump_pdata`)。段の定義のせいかの切り分け (`model_140.pt` から v7 の段の定義で続ける案) は、v12 の結果を見てからユーザーと決める。

### その他

8. **探索の設定を詰める** (1 回に 1 つ変える): `--slots 8` / `--slots 4`、`--rank random`、`--no_mean_first`。v7 s43 / s44 の `model_100.pt`、それ以外は推論時探索 v7 の記録の `search` と同じ。
9. その他の候補 (どれも未着手): 形ごとの成功率で優先して引く (Prioritized Level Replay 型。v12 の後の候補、0025 の「見込みの弱点」)、v8 の切り分け ([v8 の記録](experiments/2026-10-05_v8_symmetry.md))、P2 の残り ([0022](decisions/0022-p2-one-version-each.md))、めり込み解消速度の中間値 ([overnight の記録](experiments/2026-10-01_v4-v6_overnight.md))。

## 保留・再開した研究

| 研究 | 止めた所 | 再開するときの 1 番目 | 記録 |
|---|---|---|---|
| 推論時探索 (ビーム探索・best-of-N) | 再開中 (2026-10-05 にユーザーの指示)。v7 の rollouts / search / shapes は済み。探索の設定の比較は未着手。v6 s43 の rollouts / search、v6 s42・v5 の search は止めたまま (再開するかはユーザーの判断) | 上の「次の作業」8 | [2026-10-06_inference_search_v7](experiments/2026-10-06_inference_search_v7.md)、[2026-10-03_inference_search](experiments/2026-10-03_inference_search.md) §6 |
| rope study (紐モデルの比較。打ち切り中、再開はユーザーの判断待ち) | 最終報告まで書いた。評価と実装は 2026-10-03 13:00 に打ち切り | XPBD の段 0 の方策の崩れの切り分け (結論を変えうる最大の未解決) | [2026-10-02_rope_models](experiments/2026-10-02_rope_models.md) §12 |

- rope study の発散の数値には、`rope_study/bench_physics.py:378` の数え方による過大の疑いがある (未確認。再開するときに確かめる)。
- 推論時探索の記録の「V8」は、2026-10-05 に作った v8 (対称性のデータ拡張) とは別に書かれたもので、中身の定義は無い。v7 を土台に探索を足す次の版の意味と読める。
