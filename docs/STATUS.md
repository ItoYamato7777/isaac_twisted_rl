# STATUS (現在地)

- 状態: 現行。上書きで更新する (追記しない。手順は skill `handoff`)
- 読むべき状況: セッションの始め。今どこにいて、次に何をするかを知るとき
- 最終更新: 2026-10-06

## 要点

- v8 (= v7 + 対称性のデータ拡張) は iteration 100 で 2 シードとも v7 を大きく下回った (v8 3.7% / 29.9%、v7 53.7% / 47.3%)。原因は未確認 ([記録](experiments/2026-10-05_v8_symmetry.md))。
- 2026-10-05 にユーザーの指示で 2 つを始めた (記録はまだ書いていない):
  - **(A) v9 = v7 + `stage_weights` (1, 1, 2)**。seed 43 は iteration 100 まで学習し終えた (chain 評価はまだ)。seed 44 は学習中。学習ログ (代理の指標) の段 2 の成功は、iteration 90〜99 で v9 s43 が 0.34、v7 が 0.19 / 0.24 (推測。chain 評価で確かめる)。
  - **(B) 推論時探索の評価を v7 で再開**。v7 (iteration 100、seed 43 / 44) の `rollouts` と `search` は終わった (JSON は各 run のフォルダの `eval_search_{rollouts,search}_it100.json`)。ビーム探索は 128 問題すべてを予算 1024 手以内に解いた。段 2 の形ごとの診断 (`--mode shapes`) は、初回が `inference_mode` のバグで失敗し、直して (`8a22558`) smoke test を通した。本番はこれから。
- 対称性の切り分け (v8) は今回やらない。rope study は打ち切りのまま。

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
| v9 | seed 43 は iteration 100 まで学習済み (chain 評価はまだ)、seed 44 は学習中。継承元 v7、差は `stage_weights` だけ | [versions/v9/README.md](../src/isaac_twisted_rl/tasks/knot_tying/versions/v9/README.md) |

## 走っているジョブ

2026-10-06 13:00 に確認。止めてよいのは自分が起動した PID だけ (job の `pid` のファイルの PID に SIGTERM)。

| job (`outputs/jobs/`) | PID | GPU | 中身 | run のフォルダ |
|---|---|---|---|---|
| `2026-10-06_07-23-22_v9_s44` | 1124402 | cuda:1 | v9 seed 44 の学習 (`--max_iterations 101`、`agent.save_interval=10`)。13:00 で iteration 58。**直近の 1 iteration は約 380 s** (v9 s43 の序盤は約 218 s)。遅くなった原因は未確認。残り 43 iteration で、終わるのは 17:30 頃の見込み | `logs/rsl_rl/knot_tying_v9/2026-10-06_07-23-28_v9_s44` |
| `2026-10-06_13-00-47_chain2_shapes_v7` | 1490021 | cuda:1 | v9 s44 の `exit.txt` を待ち、v7 (seed 43 / 44、iteration 100) の `shapes` 4 本を順に回す (run の `archive.pt` の段 2 から 128 形 / 台本の初期アーカイブの段 2 の全 130 形、各 64 回・6 手・決定論的・1024 環境)。失敗したら止まり、JSON が無くても止まる。1 本約 40 分と見込む。コマンドは job の `cmd.txt` | JSON は v7 の各 run のフォルダの `eval_search_shapes_it100_{run,seed}_det.json` |

- GPU 1 には他のユーザー (ohtsukatakeshi) の Isaac Lab のプロセス (PID 673851) も載っている。触らない。
- GPU の割り当てはユーザーの判断 (2026-10-05): v9 は GPU 1 に 1 本ずつ、作業 B の評価は v9 の学習と同じ GPU に載せない。
- v7 seed 43 の延長 (`2026-10-05_20-39-20_v7_s43_to500`) は 2026-10-06 13:47 にユーザーの指示で止めた (`model_330.pt` まで)。run は `logs/rsl_rl/knot_tying_v7/2026-10-05_20-39-25_v7_s43_to500`。chain 評価 (seed 7、512 エピソード、Hydra 上書きなし、`eval_chain_it{200,300}.json`): 固結び it100 53.7% → it200 50.0% → it300 39.3%。1 シードなので、300 の低下は推測 (記録は未作成)。
- 終わった job: `2026-10-05_21-01-23_v9_s43` (`exit: 0`、`model_100.pt` あり。約 8 時間 45 分。v7 は約 6 時間 10 分)、`2026-10-06_0[5-7]-*_{rollouts,search}_v7_s4{3,4}_it100` (`exit: 0`)。`shapes_*_v7_s4{3,4}_it100` の 4 本 (2026-10-06 05:47〜06:02) は `exit: 1` で失敗した (原因と修正は上の「要点」。JSON は無い)。
- `2026-10-06_12-59-49_chain2_shapes_v7` は起動の引用の誤りで何も実行せずに `exit: 0` で終わった空の job (評価の結果ではない。無視する)。
- 実験記録に書く commit: v9 s43 は `8bab06d`、v9 s44 は `c1e6f6d` から学習を始めた (`git.txt`)。どちらも v9 のコードは `7a05d15` から変わっていない。

## 次の作業

1. **v9 の chain 評価** (seed 43 / 44、iteration 70 / 100、512 環境、seed 7、Hydra の上書きなし。コマンドは [v7 の記録](experiments/2026-10-04_v7_seeds.md) の「コマンド」のタスクを `Play-v9`・ログ名を `knot_tying_v9` に)。s44 の学習が終わる 17:30 頃以降。s43 の分は、GPU の割り当てをユーザーが許せばそれより前に回せる (1 本約 7 分)。比べる相手は v7 の既存の評価 JSON (seed 43 / 44、iteration 70 / 100)。差は v7 のシード差 (iteration 100 で約 6 ポイント)・標準誤差 (約 2 ポイント) を超えるかで読む。
2. **`shapes` の結果を読む** (chain2 が終わる 20:00 頃)。形ごとの成功率が 0 / 1 に分かれるか、平均の付近に集まるかを、`summary.knot` の `shape_share`・`hist` と `by_p_data` で見る。台本の初期アーカイブと、学習中に到達した形 (`archive.pt`) を比べる。smoke test (v7 s43、6 形) では形ごとの成功が 1〜34 / 64 とばらけたが、6 形だけで結論には使えない。
3. **v7 の `rollouts` / `search` と `shapes`、v9 の記録を書く** (skill `experiment-record`)。探索ありの成功率は chain 評価の `knot` と別に報告する ([evaluation.md](system/evaluation.md) §4)。
4. **探索の設定を詰める** (1 回に 1 つ変える。[2026-10-03_inference_search](experiments/2026-10-03_inference_search.md) §6 の 3): `--slots` を 16 -> 4 か 8 / `--rank random` (V(s) の順位付けとの比較) / `--no_mean_first`。v7 s43 と s44 の両方で。
5. 結果が出たら `paper/Report_Paper.md` §9.3 の P2-4 の状況と、この STATUS の「推論時探索」の行を更新する。v9 も同じ条件で探索を測れる。
6. その他の候補 (どれも未着手): v8 の切り分け ([v8 の記録](experiments/2026-10-05_v8_symmetry.md))、P2 の残り ([0022](decisions/0022-p2-one-version-each.md))、v7 を iteration 150〜200 まで ([v7 の記録](experiments/2026-10-04_v7_seeds.md)。ユーザーが GPU 0 で iteration 500 まで延長中)、めり込み解消速度の中間値 ([overnight の記録](experiments/2026-10-01_v4-v6_overnight.md))。

## 保留・再開した研究

| 研究 | 止めた所 | 再開するときの 1 番目 | 記録 |
|---|---|---|---|
| 推論時探索 (ビーム探索・best-of-N) | **2026-10-05 にユーザーの指示で再開** (v7 に絞る)。v7 の rollouts / search は終了、shapes は未完、探索の設定の比較は未着手。v6 s43 の rollouts / search、v6 s42・v5 の search は未完了のまま | 上の「次の作業」2〜4 | [2026-10-03_inference_search](experiments/2026-10-03_inference_search.md) §6 |
| rope study (紐モデルの比較。打ち切り中、再開はユーザーの判断待ち) | 最終報告まで書いた。評価と実装は 2026-10-03 13:00 に打ち切り | XPBD の段 0 の方策の崩れの切り分け (結論を変えうる最大の未解決) | [2026-10-02_rope_models](experiments/2026-10-02_rope_models.md) §12 |

- rope study の発散の数値には、`rope_study/bench_physics.py:378` の数え方による過大の疑いがある (未確認。再開するときに確かめる)。
- 推論時探索の記録の「V8」は、2026-10-05 に作った v8 (対称性のデータ拡張) とは別に書かれたもので、中身の定義は無い。v7 を土台に探索を足す次の版の意味と読める。
