# v9 (v7 + 段 2 から始める重みを 2 倍) の学習 (seed 43 / 44) と v7 との chain 評価の比較 (2026-10-05)

- 状態: 凍結
- 読むべき状況: 学習の初期状態の段の重み (`stage_weights`) を変えると chain 評価が変わるかを知るとき、v9 の run のフォルダと評価 JSON を探すとき。

## 目的と仮説

v7 の chain 評価では段 2 以上に届くのが 98〜99% なのに、固結びは 53.7% / 47.3% (iteration 100) に留まり、学習ログの段ごとの成功も段 2 -> 固結びが一番低い (0.19〜0.24。[`2026-10-04_v7_seeds.md`](2026-10-04_v7_seeds.md))。

仮説: 段 2 から始めるエピソードを約 1/3 から約 1/2 に増やせば、段 2 -> 固結びの学習が進み、chain 評価の固結びが v7 を上回る。

同じ (1, 1, 2) は v6 の切り分け (`abl_sw2_s42`) で試して効果が見えなかったが、v6 はめり込み解消速度 0.4 m/s で段 2 の学習が止まっていた版なので、v7 の物理で測り直した (理由は [`versions/v9/README.md`](../../src/isaac_twisted_rl/tasks/knot_tying/versions/v9/README.md))。

## 条件

| 項目 | 値 |
|---|---|
| commit | v9 のコードは `7a05d15`。学習を始めた commit は s43 `8bab06d`、s44 `c1e6f6d` (間の commit は `eval_search.py` と STATUS だけで、v9 のコードは同じ)。未コミットの差分: なし (run の `git/isaac_twisted_rl.diff` は `git status` が clean であることだけを記す 194 バイト、job の `git.diff` は 0 バイト) |
| タスク ID とプリセット | 学習 `Isaac-Twisted-RL-Knot-v9`、評価 `Isaac-Twisted-RL-Knot-Play-v9`。プリセット `Knot` |
| 変えた変数 | v7 との差は `events.reset_to_stage.params["stage_weights"]` だけ ((1, 1, 1) -> (1, 1, 2))。`tests/test_v9_cfg.py` が静的に検査し、smoke test の run の `params/env.yaml` を v7 s43 と diff して、差がシード・ログのパス・`stage_weights` の 3 番目だけであることを確かめた |
| シード | 43、44 (学習)。評価は seed 7 |
| iteration (最後のチェックポイント) | `--max_iterations 101`。**s43 は `model_100.pt`、s44 は `model_90.pt` (iteration 100 に届いていない)**。s44 は job の `timeout 10h` で iteration 99 の途中で止まった (`exit: 124`) |
| num_envs | 学習 4096、評価 512 |
| GPU の配置 | 学習は GPU 1 枚に 1 本 (どちらも cuda:1、s43 の後に s44)。ただし GPU 1 には他のユーザーのプロセス (約 2 GB) が載っていた。学習の時間は s43 8.7 時間、s44 は 99 iteration で 9.9 時間 (v7 は 6.1 時間)。遅くなった原因は未確認 |
| 学習のコマンド | 下の「コマンド」 |
| 評価のコマンド | 同上 (Hydra の上書きなし)。v9 の 4 本は GPU 0 に 4 本並べ、v7 の iteration 90 の 2 本は GPU 0 に 2 本並べた (実時間だけが変わる) |
| run のフォルダ | `logs/rsl_rl/knot_tying_v9/2026-10-05_21-01-28_v9_s43`、`logs/rsl_rl/knot_tying_v9/2026-10-06_07-23-28_v9_s44` |
| job のフォルダ | 学習 `outputs/jobs/2026-10-05_21-01-23_v9_s43`、`outputs/jobs/2026-10-06_07-23-22_v9_s44`。smoke test `outputs/jobs/2026-10-05_20-44-19_smoke_v9`。評価 `outputs/jobs/2026-10-06_18-4*_eval_v9_s4*`、`outputs/jobs/2026-10-06_19-2*_eval_v7_s4*_it90` |
| 評価 JSON | v9: s43 `eval_chain_it{70,100}.json`、s44 `eval_chain_it{70,90}.json`。v7 (比べる相手): 既存の `eval_chain_it{70,100}.json` と、今回測った `eval_chain_it90.json` (各 run のフォルダ) |
| 無効にした run | なし |

比べる相手は v7 の既存の run (`logs/rsl_rl/knot_tying_v7/2026-10-04_22-37-42_v7_s43`、`2026-10-04_22-37-47_v7_s44`。条件は [`2026-10-04_v7_seeds.md`](2026-10-04_v7_seeds.md))。v7 は学習し直していない。

### コマンド

```bash
# 学習 (s44 は --seed 44 --run_name v9_s44)
src/isaac_twisted_rl/scripts/run_job.sh --name v9_s43 --timeout 10h --detach -- \
    uv run train --task Isaac-Twisted-RL-Knot-v9 --headless --device cuda:1 --seed 43 \
    --max_iterations 101 --run_name v9_s43 agent.device=cuda:1 agent.save_interval=10

# chain 評価 (v9 は it 70 / 100 (s44 は 70 / 90)、v7 の iteration 90 は --task Isaac-Twisted-RL-Knot-Play-v7 と v7 の run)
src/isaac_twisted_rl/scripts/run_job.sh --name eval_v9_s43_it100 --timeout 1h --detach -- \
    uv run python src/isaac_twisted_rl/scripts/rsl_rl/eval_chain.py --task Isaac-Twisted-RL-Knot-Play-v9 \
    --num_envs 512 --headless --device cuda:0 --seed 7 \
    --checkpoint logs/rsl_rl/knot_tying_v9/2026-10-05_21-01-28_v9_s43/model_100.pt agent.device=cuda:0 \
    --out logs/rsl_rl/knot_tying_v9/2026-10-05_21-01-28_v9_s43/eval_chain_it100.json
```

## ゲート (学習の前の確認)

- 物理は v7 と同じなので、skill `physics-check` は回していない。
- smoke test (`outputs/jobs/2026-10-05_20-44-19_smoke_v9`、seed 42、2 iteration): 起動・学習・保存まで通った。`Curriculum/stage_log/start_frac_s{0,1,2}` は 0.25 / 0.25 / 0.50。終わりの `TypeError: post_quit()` は既知の Kit の終了処理のもの。
- 学習: s43 は `exit: 0` で `signal:` の行が無く、`model_100.pt` がある。s44 は `exit: 124` (timeout) で `model_90.pt` まで。評価 6 本は `exit: 0`、`signal:` の行なし。

## 結果

### chain 評価 (まっすぐなロープから 18 手以内に固結び、512 エピソード、seed 7、決定論的方策)

| 版 | シード | iteration | 固結び (標準誤差) | 時間切れ | 発散 | 段 2 以上に到達 | 固結びまでの平均手数 |
|---|---|---|---|---|---|---|---|
| **v9** | 43 | 70 | **56.4%** (2.2) | 33.6% | 10.0% | 97.9% | 6.1 |
| v7 | 43 | 70 | 40.6% (2.2) | 42.0% | 17.4% | 95.5% | 9.0 |
| v7 | 43 | 90 | 39.5% (2.2) | 43.0% | 17.6% | 94.3% | 10.3 |
| **v9** | 43 | 100 | **35.0%** (2.1) | 46.7% | 18.4% | 91.0% | 9.2 |
| v7 | 43 | 100 | 53.7% (2.2) | 32.6% | 13.7% | 98.2% | 6.4 |
| **v9** | 44 | 70 | **25.6%** (1.9) | 64.1% | 10.4% | 100% | 10.5 |
| v7 | 44 | 70 | 22.9% (1.9) | 65.6% | 11.5% | 97.9% | 10.5 |
| **v9** | 44 | 90 | **32.6%** (2.1) | 55.1% | 12.3% | 94.7% | 8.4 |
| v7 | 44 | 90 | 33.8% (2.1) | 50.4% | 15.8% | 94.1% | 8.5 |
| v7 | 44 | 100 | 47.3% (2.2) | 41.2% | 11.5% | 99.2% | 8.3 |

標準誤差は `sqrt(p (1 - p) / 512)`。v7 の iteration 70 / 100 の行は既存の評価 JSON の値 (同じ条件)、iteration 90 は今回測った。

| 同じシード・同じ iteration の差 (v9 − v7) | iteration 70 | iteration 90 | iteration 100 |
|---|---|---|---|
| seed 43 | +15.8 | — (v9 は未測定) | −18.7 |
| seed 44 | +2.7 | −1.2 | — (v9 は届いていない) |

### 学習ログ (代理の指標。`Curriculum/stage_log/*` と `Episode_Termination/diverged` の 10 iteration の平均)

| run | iteration | 段 2 から始めた割合 | 段 0 成功 | 段 1 成功 | 段 2 成功 | 段 2 時間切れ | まっすぐなロープからの成功 | 発散で終了 |
|---|---|---|---|---|---|---|---|---|
| v9 s43 | 60〜69 | 0.50 | 0.97 | 0.47 | 0.21 | 0.07 | 0.97 | 2.6% |
| v9 s43 | 90〜99 | 0.50 | 0.98 | 0.55 | 0.34 | 0.09 | 0.99 | 2.2% |
| v9 s44 | 60〜69 | 0.50 | 0.97 | 0.47 | 0.25 | 0.03 | 0.98 | 2.4% |
| v9 s44 | 90〜99 | 0.50 | 0.97 | 0.56 | 0.27 | 0.10 | 0.98 | 2.5% |
| v7 s43 | 60〜69 | 0.34 | 0.97 | 0.52 | 0.15 | 0.02 | 0.97 | 3.1% |
| v7 s43 | 90〜99 | 0.33 | 0.97 | 0.58 | 0.19 | 0.03 | 0.98 | 2.8% |
| v7 s44 | 60〜69 | 0.33 | 0.97 | 0.50 | 0.17 | 0.03 | 0.97 | 2.8% |
| v7 s44 | 90〜99 | 0.33 | 0.97 | 0.57 | 0.24 | 0.08 | 0.98 | 2.6% |

「まっすぐなロープからの成功」は `straight_success`。v9 s44 の 90〜99 は iteration 99 までのログ (チェックポイントは 90 まで)。

## 結論

- 分かった:
  1. **v9 は最後のチェックポイントで v7 を上回らなかった。** seed 43 は iteration 100 で v9 35.0% 対 v7 53.7% (−18.7 ポイント)、seed 44 は iteration 90 どうしで 32.6% 対 33.8% (−1.2 ポイント、標準誤差の範囲)。仮説は今回の 2 シードでは確かめられなかった。
  2. **v9 s43 は iteration 70 -> 100 で 56.4% -> 35.0% に下がった。** 発散は 10.0% -> 18.4%、段 2 以上への到達は 97.9% -> 91.0% に悪化した。iteration 70 の 56.4% はこの比較の中で最も高いが、同じ run の iteration 100 では v7 より低い。
  3. **学習ログの段 2 の成功は v9 が高い** (iteration 90〜99 で 0.34 / 0.27、v7 は 0.19 / 0.24)。段 0 (0.97〜0.98)、まっすぐなロープからの成功 (0.98〜0.99)、発散 (2.2〜2.5%) は v7 と同じ水準で、まっすぐなロープから始める割合の低下 (1/6 -> 1/8) で段 0 は崩れなかった。**この学習ログの差は chain 評価の差にならなかった** ([`2026-10-01_v4-v6_overnight.md`](2026-10-01_v4-v6_overnight.md) §4.5 と同じ)。
- 推測:
  - 同じ run の中で iteration によって固結びが 20 ポイント前後動く (v9 s43 は 70 と 100 で 21 ポイント、v7 s44 は 90 と 100 で 14 ポイント)。v9 と v7 の差はこのチェックポイントごとのばらつきに埋もれていて、`stage_weights` の効果は、あったとしてもこれより小さい。
  - 段 2 の詰まりは学習量よりも、方策が段 2 に着いたときの形 (tuck しにくい形) によるところが大きい。v7 の段 2 の形ごとの診断で、学習中に到達した形は成功率が形で大きく違い、平均 0.27〜0.29 だった ([`2026-10-06_inference_search_v7.md`](2026-10-06_inference_search_v7.md))。段 2 から始める回数を増やしても、その形の分布は変わらない。
- 未確認:
  - **v9 s44 は iteration 100 に届いていない。** iteration 100 どうしの比較は seed 43 の 1 本だけ。
  - シードは 2 本だけ。チェックポイントごとのばらつきを均すための、複数の iteration の平均や、シードの追加はしていない。
  - 学習が v7 より遅かった (6.1 時間に対して 8.7 / 9.9 時間) 原因。GPU 1 の他のユーザーのプロセスとの共有が疑わしいが、確かめていない。同じアルゴリズム・環境数なので iteration あたりの学習は変わらないと見込む。

## 次の作業

1. `stage_weights` をこれ以上詰めない。段 2 -> 固結びへの対策は、段 2 に着いたときの形のほう (段の定義を正規の経路に絞る v10。[0023](../decisions/0023-v10-path-stages.md)) と推論時の再試行 (推論時探索) で見る。
2. 版の比較の読み方: 1 iteration の 1 点では 20 ポイント前後動くので、iteration 70 / 90 / 100 などの複数点を並べて読む (今回 v7 の iteration 90 を足したのはこのため)。
