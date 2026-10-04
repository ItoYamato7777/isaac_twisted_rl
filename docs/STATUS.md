# STATUS (現在地)

- 状態: 現行。上書きで更新する (追記しない。手順は skill `handoff`)
- 読むべき状況: セッションの始め。今どこにいて、次に何をするかを知るとき
- 最終更新: 2026-10-04

## 要点

- 研究の作業 (rope study・推論時探索の評価) は 2026-10-03 に打ち切った。2026-10-04 の時点で、ユーザーの判断でまだ再開しない。
- リファクタリングは 2026-10-04 に完了し、main に入れた (push はしていない)。記録は [`history/refactor/`](history/refactor/README.md)。
- 版の比較で決着がついていないのは v4 / v6 / v7 の優劣 (各 1〜2 シード)。

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
| v6 | chain 評価 済み (seed 42・43) | 同上 |
| v7 | **未学習**。同じ設定 (v6 + めり込み解消速度 0.96 m/s) の切り分けの run (`abl_depB`、iteration 70) だけがある | 同上の追記 (v6 の切り分け) |

## 走っているジョブ

v7 の学習 2 本 (2026-10-04 22:37 起動、commit `55fa432`、GPU 1 枚に 1 本、1 iteration 約 190 s、終わりの見込み 10-05 04:10 頃)。
止めるときは job の pid に SIGTERM を送る。

| job のフォルダ (`outputs/jobs/`) | pid | GPU | run のフォルダ (`logs/rsl_rl/knot_tying_v7/`) |
|---|---|---|---|
| `2026-10-04_22-37-37_v7_s43` | 876278 | cuda:0 | `*_v7_s43` |
| `2026-10-04_22-37-42_v7_s44` | 876547 | cuda:1 | `*_v7_s44` |

コマンド (s44 は `--device cuda:1 --seed 44 --run_name v7_s44 agent.device=cuda:1`):
`run_job.sh --name v7_s43 --timeout 10h --detach -- uv run train --task Isaac-Twisted-RL-Knot-v7 --headless --device cuda:0 --seed 43 --max_iterations 101 --run_name v7_s43 agent.device=cuda:0 agent.save_interval=10`

終わったら: `exit.txt` に `signal:` が無いことと `model_100.pt` を確かめ、iteration 70 / 100 を chain 評価する (Hydra の上書きなし)。
v7 seed 42 に当たるのは `abl_depB_s42` (v6 + 上書きで params の env.yaml は v7 と同じ。iteration 70 まで)。

## 次の作業

1. 次に何をするかはユーザーが決める。打ち切った 2 つの研究 (下の「保留」) は、ユーザーの指示があるまで再開しない。
2. 研究の候補 (記録にあるもの。どれも未着手):
   - v4 / v6 / v7 の決着: v7 を seed 43 / 44、v6 を seed 44 で学習し、各 3 シードで chain 評価する (GPU 1 枚に 2 本で約 13.5 時間)。出典: [overnight の記録](experiments/2026-10-01_v4-v6_overnight.md) の「追記の後の次の候補」。
   - めり込み解消速度の中間値 (0.6〜0.7 m/s)。学習の前に skill `physics-check` で発散と固結びを測る。出典: 同上。
   - 評価を強くする (確率的な方策での chain 評価、10 iteration ごとの学習曲線、エピソード数・シードを増やす)。出典: 同上。
   - 研究方針の提案 E0〜E7・S1 ([`paper/Discussion.md`](../paper/Discussion.md)、2026-10-01 時点の仮説の文書)。S1 (推論時の候補評価) は推論時探索で一部実施された。

## 保留 (打ち切り中。再開するかはユーザーの判断待ち)

| 研究 | 止めた所 | 再開するときの 1 番目 | 記録 |
|---|---|---|---|
| 推論時探索 (ビーム探索・best-of-N) | 評価の途中 (v7・v6 s43 の rollouts / search、v6 s42・v5 の search が未完了) | 止めた評価を最後まで回す。その後、v7 を v6 の 2 シードと比べて次の版 (v8) の土台を決める (差がシード差に届かなければ v7 の 2 本目のシードを先に回す) | [2026-10-03_inference_search](experiments/2026-10-03_inference_search.md) §6 |
| rope study (紐モデルの比較) | 最終報告まで書いた。評価と実装は 2026-10-03 13:00 に打ち切り | XPBD の段 0 の方策の崩れの切り分け (結論を変えうる最大の未解決) | [2026-10-02_rope_models](experiments/2026-10-02_rope_models.md) §12 |

- rope study の発散の数値には、`rope_study/bench_physics.py:378` の数え方による過大の疑いがある (未確認。再開するときに確かめる)。
- 推論時探索の記録の「V8」は、次の版 (v8) のことと読める (定義は記録に無い)。v8 の中身は未定。
