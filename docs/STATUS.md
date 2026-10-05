# STATUS (現在地)

- 状態: 現行。上書きで更新する (追記しない。手順は skill `handoff`)
- 読むべき状況: セッションの始め。今どこにいて、次に何をするかを知るとき
- 最終更新: 2026-10-05

## 要点

- v8 (= v7 + PPO の対称性のデータ拡張、鏡像・逆順で 4 倍) を seed 43 / 44 で学習中 (GPU 1 枚に 1 本)。比べる相手は v7 の seed 43 / 44 (iteration 70 / 100 の chain 評価)。
- 学習の前の確認 (`scripts/rsl_rl/symmetry_check.py`) で、計算し直した観測は環境と一致した。物理は 1 手あたり数 cm 非対称 (同じ手どうしの約 5〜10 倍) だが、方策が受けている物理の揺れの範囲として進めた。数値は v8 の実験記録に書く (未作成)。
- 研究の作業 (rope study・推論時探索の評価) は 2026-10-03 に打ち切った。ユーザーの指示があるまで再開しない。

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
| v8 | 学習中 (seed 43 / 44)。smoke test (`outputs/jobs/2026-10-05_12-50-39_smoke_v8`) は通った。判断の記録は [0022](decisions/0022-p2-one-version-each.md) | (未作成) |

## 走っているジョブ

自分が起動したもの (2026-10-05 13:59 起動、commit `deae5d7`。止めるときは job の pid に SIGTERM):

| job のフォルダ (`outputs/jobs/`) | pid | GPU | run のフォルダ (`logs/rsl_rl/knot_tying_v8/`) | 終わる見込み |
|---|---|---|---|---|
| `2026-10-05_13-59-03_v8_s43` | 7199 | cuda:1 | `2026-10-05_13-59-12_v8_s43` | 10-05 20:00 頃 (1 iteration 約 3.3 分 × 101) |
| `2026-10-05_13-59-03_v8_s44` | 7263 | cuda:0 | `2026-10-05_13-59-12_v8_s44` | 同上 |

コマンド (s44 は `--device cuda:0 --seed 44 --run_name v8_s44 agent.device=cuda:0`):
`run_job.sh --name v8_s43 --timeout 10h --detach -- uv run train --task Isaac-Twisted-RL-Knot-v8 --headless --device cuda:1 --seed 43 --max_iterations 101 --run_name v8_s43 agent.device=cuda:1 agent.save_interval=10`

- 無効にした run: `logs/rsl_rl/knot_tying_v8/2026-10-05_12-58-46_v8_s43` (job `2026-10-05_12-58-41_v8_s43`)。PC の電源断で iteration 15 で止まった (`exit.txt` 無し、`model_10.pt` まで)。到達状態のアーカイブ (50 iteration ごとに保存) が失われるので `--resume` せず、最初から回し直した。
- 同じ電源断で、自分が起動していない v7 の学習 (PID 1224684、run `logs/rsl_rl/knot_tying_v7/2026-10-05_12-43-46`) も止まった。再起動していない。

## 次の作業

1. 学習が終わったら `exit.txt` (`signal:` が無い) と `model_100.pt` を確かめ、iteration 70 / 100 を chain 評価する (`Isaac-Twisted-RL-Knot-Play-v8`、512 環境、seed 7、Hydra の上書きなし)。v7 s43 / s44 の `eval_chain_it{70,100}.json` と比べる。仮説: 対称性のデータ拡張で段 2 -> 固結びの学習が速くなり、v7 との差が v7 のシード差 (iteration 100 で 6 ポイント) と標準誤差 (約 2 ポイント) を超える。
2. skill `experiment-record` で v8 の記録を書く (電源断で無効にした run も書く)。学習の前の確認 (symmetry_check の結果 `outputs/symcheck4_v8_{policy_v7s43_it100,random}.json`、job は `outputs/jobs/2026-10-05_12-44-1*_symcheck4_*`) もそこに入れる。
3. その他の候補 (どれも未着手):
   - v6 を seed 44 で学習し、v7 と同じシード 43 / 44 で比べる。v7 を iteration 150〜200 まで延ばす。出典: [v7 の記録](experiments/2026-10-04_v7_seeds.md) の「次の作業」。
   - P2 の残り (新しさの探索を位相のキーの訪問回数で、初期形状のランダム化)。順と条件は [0022](decisions/0022-p2-one-version-each.md)。
   - めり込み解消速度の中間値 (0.6〜0.7 m/s)、評価を強くする。出典: [overnight の記録](experiments/2026-10-01_v4-v6_overnight.md) の「追記の後の次の候補」。

## 保留 (打ち切り中。再開するかはユーザーの判断待ち)

| 研究 | 止めた所 | 再開するときの 1 番目 | 記録 |
|---|---|---|---|
| 推論時探索 (ビーム探索・best-of-N) | 評価の途中 (v7・v6 s43 の rollouts / search、v6 s42・v5 の search が未完了) | 止めた評価を最後まで回す。v7 と v6 の比較は [v7 の記録](experiments/2026-10-04_v7_seeds.md) で 2 シードずつになった | [2026-10-03_inference_search](experiments/2026-10-03_inference_search.md) §6 |
| rope study (紐モデルの比較) | 最終報告まで書いた。評価と実装は 2026-10-03 13:00 に打ち切り | XPBD の段 0 の方策の崩れの切り分け (結論を変えうる最大の未解決) | [2026-10-02_rope_models](experiments/2026-10-02_rope_models.md) §12 |

- rope study の発散の数値には、`rope_study/bench_physics.py:378` の数え方による過大の疑いがある (未確認。再開するときに確かめる)。
- 推論時探索の記録の「V8」は、2026-10-05 に作った v8 (対称性のデータ拡張) とは別に書かれたもので、中身の定義は無い。
