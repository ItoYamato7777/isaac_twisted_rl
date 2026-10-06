# STATUS (現在地)

- 状態: 現行。上書きで更新する (追記しない。手順は skill `handoff`)
- 読むべき状況: セッションの始め。今どこにいて、次に何をするかを知るとき
- 最終更新: 2026-10-06

## 要点

- v9 (v7 + 段 2 から始める重み 2 倍) は chain 評価で v7 を上回らなかった。v7 の推論時探索と段 2 の形ごとの切り分けから、段 2 の詰まりは方策が段 2 に作る形 (幾何) の偏りが大きいと読める (推測)。記録: [v9](experiments/2026-10-05_v9_stage2-weight.md)、[推論時探索 v7](experiments/2026-10-06_inference_search_v7.md)。
- v10 (段を固結びへの正規の経路に絞る。[0023](decisions/0023-v10-path-stages.md)) を seed 43 / 44 で学習中 (別のセッションが起動)。終わったら v7 と chain 評価で比べる。
- 対称性の切り分け (v8)・rope study は打ち切りのまま。

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
| v10 | 学習中 (seed 43 / 44)。継承元 v7、差は段の定義だけ (段 3 は v7 と同じ判定) | [versions/v10/README.md](../src/isaac_twisted_rl/tasks/knot_tying/versions/v10/README.md)、[0023](decisions/0023-v10-path-stages.md) |

## 走っているジョブ

2026-10-06 22:45 に確認。どちらもこのリポジトリの別のセッションが起動したもの (止める必要があれば PID をユーザーに伝える)。

| job (`outputs/jobs/`) | PID | GPU | 中身 | run のフォルダ |
|---|---|---|---|---|
| `2026-10-06_18-42-55_v10_s43` | 1896378 | cuda:0 | v10 seed 43 の学習 (`--max_iterations 101`、`agent.save_interval=10`、timeout 14h)。commit `59ebca8`。22:45 で iteration 50、1 iteration 約 240 s、02:10 頃に終わる見込み | `logs/rsl_rl/knot_tying_v10/2026-10-06_18-43-01_v10_s43` |
| `2026-10-06_22-44-31_v10_s44` | 2165042 | cuda:1 | v10 seed 44 の学習 (`--max_iterations 101`、`agent.save_interval=10`、timeout 14h)。commit `d38f82d` (v10 のコードは `7b57c91` から変わっていない)。ユーザーの指示で 22:44 に起動 (予定の 20:00 を取り逃がした) | `logs/rsl_rl/knot_tying_v10/` の `*_v10_s44` |
| `2026-10-06_22-44-31_v10_s44` | 2165042 | cuda:1 | v10 seed 44 の学習 (同じ引数)。commit `d38f82d`。22:44 起動 | `logs/rsl_rl/knot_tying_v10/2026-10-06_22-44-37_v10_s44` |

- GPU 0・1 には他のユーザー (ohtsukatakeshi) の Isaac Lab のプロセス (PID 673851、各約 2 GB) が載っている。触らない。v9 の学習が v7 より 1.4〜1.6 倍遅かったのは、これとの共有が疑わしい (未確認)。
- v10 s43 の iteration 51〜60 あたりは、v9 / v7 の chain 評価 6 本 (2026-10-06 18:41〜19:52) と GPU 0 を共有した (実時間だけが変わる)。
- 完走の確認は `exit.txt` に `signal:` の行が無いことと `model_100.pt`。v9 s44 は `timeout 10h` が足りずに止まったので、遅い GPU では timeout を長めにする (v10 は 14h)。

## 次の作業

1. **v10 の chain 評価** (学習が終わったら。seed 43 / 44、iteration 70 / 100、512 環境、seed 7、Hydra の上書きなし。`knot_strict` も見る)。比べる相手は v7 の既存の評価 JSON (seed 43 / 44 の iteration 70 / 90 / 100。90 は [v9 の記録](experiments/2026-10-05_v9_stage2-weight.md) で測った)。同じ run の中でも iteration で 20 ポイント前後動くので、70 / 90 / 100 を並べて読む。差は v7 のシード差・標準誤差 (約 2 ポイント) を超えるかで読む。仕様は [0023](decisions/0023-v10-path-stages.md)、コマンドは [v7 の記録](experiments/2026-10-04_v7_seeds.md) の「コマンド」のタスクを `Play-v10` に。
2. **v10 に段 2 の形ごとの切り分け** (`eval_search.py --mode shapes`、学習中に到達した形 128 形、64 回、6 手、決定論的) を回し、v7 の形の成功率の分布 (平均 0.27〜0.29) と比べる。コマンドは [推論時探索 v7 の記録](experiments/2026-10-06_inference_search_v7.md) の「コマンド」。
3. **探索の設定を詰める** (1 回に 1 つ変える): `--slots 8` / `--slots 4`、`--rank random`、`--no_mean_first`。v7 s43 / s44 の `model_100.pt`、それ以外は推論時探索 v7 の記録の `search` と同じ。
4. **成功する段 2 の形と失敗する形の幾何の違いを見る** (`eval_search_shapes_it100_run_det.json` の `shapes_detail` の `index` でアーカイブの形を引き、`scripts/rsl_rl/view_archive.py` で並べる)。
5. その他の候補 (どれも未着手): v8 の切り分け ([v8 の記録](experiments/2026-10-05_v8_symmetry.md))、P2 の残り ([0022](decisions/0022-p2-one-version-each.md))、v7 seed 43 の延長 (iteration 330 で止めた。chain 評価は it200 50.0%・it300 39.3%。1 シードで記録は未作成。run は `logs/rsl_rl/knot_tying_v7/2026-10-05_20-39-25_v7_s43_to500`)、めり込み解消速度の中間値 ([overnight の記録](experiments/2026-10-01_v4-v6_overnight.md))。

## 保留・再開した研究

| 研究 | 止めた所 | 再開するときの 1 番目 | 記録 |
|---|---|---|---|
| 推論時探索 (ビーム探索・best-of-N) | 再開中 (2026-10-05 にユーザーの指示)。v7 の rollouts / search / shapes は済み。探索の設定の比較は未着手。v6 s43 の rollouts / search、v6 s42・v5 の search は止めたまま (再開するかはユーザーの判断) | 上の「次の作業」3 | [2026-10-06_inference_search_v7](experiments/2026-10-06_inference_search_v7.md)、[2026-10-03_inference_search](experiments/2026-10-03_inference_search.md) §6 |
| rope study (紐モデルの比較。打ち切り中、再開はユーザーの判断待ち) | 最終報告まで書いた。評価と実装は 2026-10-03 13:00 に打ち切り | XPBD の段 0 の方策の崩れの切り分け (結論を変えうる最大の未解決) | [2026-10-02_rope_models](experiments/2026-10-02_rope_models.md) §12 |

- rope study の発散の数値には、`rope_study/bench_physics.py:378` の数え方による過大の疑いがある (未確認。再開するときに確かめる)。
- 推論時探索の記録の「V8」は、2026-10-05 に作った v8 (対称性のデータ拡張) とは別に書かれたもので、中身の定義は無い。v7 を土台に探索を足す次の版の意味と読める。
