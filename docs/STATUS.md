# STATUS (現在地)

- 状態: 現行。上書きで更新する (追記しない。手順は skill `handoff`)
- 読むべき状況: セッションの始め。今どこにいて、次に何をするかを知るとき
- 最終更新: 2026-10-05

## 要点

- v8 (= v7 + PPO の対称性のデータ拡張) を seed 43 / 44 で学習し、chain 評価した。**iteration 100 で 2 シードとも v7 を大きく下回った** (v8 3.7% / 29.9%、v7 53.7% / 47.3%)。原因は未確認 ([記録](experiments/2026-10-05_v8_symmetry.md))。
- 2026-10-05 にユーザーの指示で 2 つを始めた: (A) **v9 = v7 + 学習の初期状態の段の重み (1, 1, 2)** (段 2 -> 固結びの学習量を増やす。v8 の変更は入れない) を seed 43 / 44 で学習し、v7 と chain 評価で比べる。(B) **推論時探索の評価を v7 で再開する** (打ち切っていた v7 の `rollouts` / `search`、段 2 の切り分け、探索の設定)。
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
| v9 | 作った (継承元 v7、差は `stage_weights` だけ)。学習はこれから | [versions/v9/README.md](../src/isaac_twisted_rl/tasks/knot_tying/versions/v9/README.md) |

## 走っているジョブ

2026-10-05 22:05 に確認。止めてよいのは自分が起動した PID だけ (job の `pid` のファイルの PID に SIGTERM)。

| job (`outputs/jobs/`) | PID | GPU | 中身 | run のフォルダ |
|---|---|---|---|---|
| `2026-10-05_21-01-23_v9_s43` | 460925 | cuda:1 | v9 seed 43 の学習 (`--max_iterations 101`、`agent.save_interval=10`)。1 iteration 約 218 s、03:10 頃に終わる見込み | `logs/rsl_rl/knot_tying_v9/2026-10-05_21-01-28_v9_s43` |
| `2026-10-05_22-04-07_chain_B_then_v9_s44` | 527195 | cuda:1 | v9 s43 の `exit.txt` を待ち、v7 (seed 43 / 44、iteration 100) の評価を順に回してから v9 seed 44 の学習を `--detach` で起動する。評価: `shapes` (run の `archive.pt` の段 2 から 128 形、台本の初期アーカイブの段 2 の全 130 形。各 64 回・6 手・決定論的・1024 環境)、`rollouts`、`search` (記録の付録と同じ設定、`--max_minutes 120`)。コマンドは job の `cmd.txt` | 評価の JSON は v7 の各 run のフォルダの `eval_search_*_it100*.json` |
| (自分の起動ではない) `2026-10-05_20-39-20_v7_s43_to500` | 435971 | cuda:0 | v7 seed 43 を iteration 100 から 500 まで延長 (`--resume`)。20:39 起動 | `logs/rsl_rl/knot_tying_v7/2026-10-05_20-39-25_v7_s43_to500` |

- GPU の割り当てはユーザーの判断 (2026-10-05): v9 は GPU 1 に 1 本ずつ (s43 の後に s44)、作業 B の評価は s43 と s44 の間に GPU 1 で回す。GPU 0 は上の延長学習が使っている。
- smoke test (`outputs/jobs/2026-10-05_20-44-19_smoke_v9`、GPU 0 を延長学習と共有) は通った。`params/env.yaml` を v7 s43 と diff し、差がシード・ログのパス・`stage_weights` の 3 番目 (1.0 -> 2.0) だけであることを確かめた。学習ログの `start_frac_s2` は 0.50。

## 次の作業

1. **v9 の学習** (seed 43 / 44、4096 環境、`--max_iterations 101`、`agent.save_interval=10`、GPU 1 枚に 1 本。v7 と同じ条件。コマンドは [v7 の記録](experiments/2026-10-04_v7_seeds.md) の「コマンド」のタスクを v9 に)。仮説: 段 2 から始めるエピソードを 1/3 -> 1/2 に増やすと、段 2 -> 固結びの学習が進み、chain 評価の固結びが v7 (iteration 100 で 53.7% / 47.3%) を上回る (推測)。比べる相手は v7 の既存の run (学習し直さない)。
2. 学習の間に、推論時探索の段 2 の切り分け (段 2 のアーカイブの形ごとに、同じ方策で N 回試した固結びの割合) を `eval_search.py` に新しい `--mode` として足す (既存のモードは変えない)。
3. 学習が終わったら: v9 の chain 評価 (iteration 70 / 100、512 環境、seed 7) と記録。v7 (iteration 100、seed 43 / 44) の `rollouts` / `search` と段 2 の切り分け、探索の設定 (1 ラウンドの環境数 4〜8、V(s) の順位付けとランダムの比較、`--no_mean_first`。1 回に 1 つ) を回して記録 ([2026-10-03_inference_search](experiments/2026-10-03_inference_search.md) §6)。
4. その他の候補 (どれも未着手): v8 の切り分け (鏡像の損失だけ・片方の変換だけ・重要度比の測定。[v8 の記録](experiments/2026-10-05_v8_symmetry.md))、P2 の残り ([0022](decisions/0022-p2-one-version-each.md))、v6 seed 44・v7 を iteration 150〜200 まで ([v7 の記録](experiments/2026-10-04_v7_seeds.md))、めり込み解消速度の中間値 ([overnight の記録](experiments/2026-10-01_v4-v6_overnight.md))。

## 保留・再開した研究

| 研究 | 止めた所 | 再開するときの 1 番目 | 記録 |
|---|---|---|---|
| 推論時探索 (ビーム探索・best-of-N) | **2026-10-05 にユーザーの指示で再開** (v7 に絞る)。v6 s43 の rollouts / search、v6 s42・v5 の search は未完了のまま | v7 (iteration 100、seed 43 / 44) の rollouts / search。上の「次の作業」3 | [2026-10-03_inference_search](experiments/2026-10-03_inference_search.md) §6 |
| rope study (紐モデルの比較。打ち切り中、再開はユーザーの判断待ち) | 最終報告まで書いた。評価と実装は 2026-10-03 13:00 に打ち切り | XPBD の段 0 の方策の崩れの切り分け (結論を変えうる最大の未解決) | [2026-10-02_rope_models](experiments/2026-10-02_rope_models.md) §12 |

- rope study の発散の数値には、`rope_study/bench_physics.py:378` の数え方による過大の疑いがある (未確認。再開するときに確かめる)。
- 推論時探索の記録の「V8」は、2026-10-05 に作った v8 (対称性のデータ拡張) とは別に書かれたもので、中身の定義は無い。v7 を土台に探索を足す次の版の意味と読める。
