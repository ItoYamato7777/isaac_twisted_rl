# STATUS (現在地)

- 状態: 現行。上書きで更新する (追記しない。手順は skill `handoff`)
- 読むべき状況: セッションの始め。今どこにいて、次に何をするかを知るとき
- 最終更新: 2026-10-07

## 要点

- v9 (v7 + 段 2 から始める重み 2 倍) は chain 評価で v7 を上回らなかった。v7 の推論時探索と段 2 の形ごとの切り分けから、段 2 の詰まりは方策が段 2 に作る形 (幾何) の偏りが大きいと読める (推測)。記録: [v9](experiments/2026-10-05_v9_stage2-weight.md)、[推論時探索 v7](experiments/2026-10-06_inference_search_v7.md)。
- **v10 (段を固結びへの正規の経路に絞る。[0023](decisions/0023-v10-path-stages.md)) は chain 評価で v7 を大きく下回った** (seed 43 / 44、iteration 70 / 90 / 100 のどれでも固結び 0〜1.8%)。段 1 -> 経路の上の段 2 の手をほとんど学べなかった ([記録](experiments/2026-10-06_v10_path-stages.md))。v7 の固結びのうち経路の終点 (`knot_strict`) は 1〜3 割だけだった。
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
| v10 | chain 評価 済み (seed 43・44、iteration 70 / 90 / 100、GPU 1 枚に 1 本)。v7 を大きく下回った | [2026-10-06_v10_path-stages](experiments/2026-10-06_v10_path-stages.md)、[0023](decisions/0023-v10-path-stages.md) |

## 走っているジョブ

2026-10-07 12:30 に確認。このリポジトリのジョブは無い。

- GPU 0・1 には他のユーザー (ohtsukatakeshi) の Isaac Lab のプロセス (PID 673851、各約 2 GB) が載っている。触らない。v9・v10 の学習が v7 より 1.2〜1.6 倍遅かったのは、これとの共有が疑わしい (未確認)。
- 完走の確認は `exit.txt` に `signal:` の行が無いことと `model_100.pt`。遅い GPU では timeout を長めにする (v9 s44 は `timeout 10h` が足りずに止まった。v10 は 14h で足りた)。

## 次の作業

1. **v10 の後をどうするかをユーザーに確かめる** (推奨: v10 をこの定義のまま延ばさない)。段の定義を変える次の版を作るなら、その前に v10 の段 1 から打った手の後の p-data の分布を測る (どの形に外れているか)。案と理由は [v10 の記録](experiments/2026-10-06_v10_path-stages.md) の「次の作業」。
2. v10 の段 2 の形ごとの切り分け (`--mode shapes`) は見送る。v10 の段 2 には chain 評価でほとんど届かないため。
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
