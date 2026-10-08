# v11 (v7 + 表に無くラダーでも 0 の行き止まりの形だけを段 −1 にする) の学習 (seed 43 / 44、iteration 300) と chain 評価 (2026-10-07)

- 状態: 凍結
- 読むべき状況: v11 の結果と run のフォルダを探すとき、行き止まりの形を −1 にする段の定義を続けるか決めるとき、段 0 の方策が交差を作らなくなる崩れ (v11 s44) を調べるとき。

## 目的と仮説

v7 を長く学習すると、Φ の表に無くラダーでも 0 の形 (行き止まりの形) が段 0 のアーカイブに溜まり、chain 評価が下がった (v7 s43 の延長: it100 53.7% -> it300 39.3%、1 シード)。

仮説: v7 の段のまま行き止まりの形だけを −1 (終了・アーカイブに入れない) にすれば、段 0 のアーカイブが汚れず、長い学習 (it100 -> it300) で chain 評価が v7 ほど下がらない。効果は it100 より後で出ると見込んだ (推測)。

仕様と根拠のデータは [0024](../decisions/0024-v11-dead-end-stages.md)、配線は [`versions/v11/README.md`](../../src/isaac_twisted_rl/tasks/knot_tying/versions/v11/README.md)。

## 条件

| 項目 | 値 |
|---|---|
| commit | `8fe0821` (s43・s44 とも)。未コミットの差分: job の `git.diff` は 0 バイト、run の `git/isaac_twisted_rl.diff` は clean を記す 193 バイト |
| タスク ID とプリセット | 学習 `Isaac-Twisted-RL-Knot-v11`、評価 `Isaac-Twisted-RL-Knot-Play-v11`。プリセット `Knot` |
| 変えた変数 | v7 との差は `stage_definition = "bfs_dead_end"` (`topology.DeadEndStage`) だけ。初期アーカイブ・`next_edge`・`stage_weights`・PPO は v7 のまま (`tests/test_v11_cfg.py` が静的に検査) |
| シード | 43、44 (学習)。評価は seed 7 |
| iteration (最後のチェックポイント) | `--max_iterations 301`、0 から通し。どちらも `model_300.pt`。`exit.txt` は `exit: 0` で `signal:` の行なし |
| num_envs | 学習 4096、評価 512 |
| GPU の配置 | GPU 1 枚に 1 本 (s43 cuda:0、s44 cuda:1)。学習の時間は s43 20.7 時間、s44 22.6 時間 (1 iteration 約 4 分) |
| 学習のコマンド | 下の「コマンド」 |
| 評価のコマンド | 同上 (Hydra の上書きなし) |
| run のフォルダ | `logs/rsl_rl/knot_tying_v11/2026-10-07_14-11-04_v11_s43`、`logs/rsl_rl/knot_tying_v11/2026-10-07_14-11-04_v11_s44` |
| job のフォルダ | smoke test `outputs/jobs/2026-10-07_14-02-13_smoke_v11`。学習 `outputs/jobs/2026-10-07_14-10-56_v11_s43`、`outputs/jobs/2026-10-07_14-10-57_v11_s44`。評価 `outputs/jobs/2026-10-07_20-47-06_eval_v11_s4{3,4}_it70_90_100`、`outputs/jobs/2026-10-08_15-21-45_eval_v11_s4{3,4}_it150_200_250_300`。手ごとの診断 `outputs/jobs/2026-10-08_15-23-12_diag_v11_stage_moves`、`outputs/jobs/2026-10-08_16-53-30_diag_v11_stage_moves_fix` |
| 評価 JSON | 各 run のフォルダの `eval_chain_it{70,90,100,150,200,250,300}.json`。手ごとの診断は `outputs/diag_stage_moves/*.json` |
| アーカイブの写し | `outputs/archive_snapshots/v11_s43_it100.pt` (s43 の iteration 99〜100 の時点)。s44 の写しは取れていない (待ちのコマンドが session の終わりで止まった) |
| 無効にした run | なし。手ごとの診断の最初の出力 (`outputs/diag_stage_moves/*_bad_grasp.json`) は掴む点・置く先の列だけが誤り (下の「結果」) |

比べる相手は v7 の既存の run: `logs/rsl_rl/knot_tying_v7/2026-10-04_22-37-42_v7_s43`・`2026-10-04_22-37-47_v7_s44` (iteration 100 まで、条件は [`2026-10-04_v7_seeds.md`](2026-10-04_v7_seeds.md)) と、v7 s43 の延長 `logs/rsl_rl/knot_tying_v7/2026-10-05_20-39-25_v7_s43_to500` (iteration 330 まで。この記録で初めて結果を書く。下の「v7 s43 の延長の条件」)。**v7 s44 の延長はまだ無い** (STATUS の次の作業)。

### v7 s43 の延長の条件

- commit・job: `outputs/jobs/2026-10-05_20-39-20_v7_s43_to500` (`cmd.txt`)。`exit: 134` で 2026-10-06 13:47 に止まり、最後は `model_330.pt`。
- コマンド: `uv run train --task Isaac-Twisted-RL-Knot-v7 --headless --device cuda:0 --seed 43 --resume --load_run 2026-10-04_22-37-42_v7_s43 --checkpoint model_100.pt --max_iterations 401 --run_name v7_s43_to500 agent.device=cuda:0 agent.save_interval=10 env.events.reset_to_stage.params.load_paths=[logs/rsl_rl/knot_tying_v7/2026-10-04_22-37-42_v7_s43/archive.pt]` (パスは絶対パスで渡した)。
- **v11 とそろっていない条件** (ユーザーの判断で、そろえずに比べる。2026-10-07): v7 は iteration 100 で再開した (optimizer の状態は引き継ぎ、全 env がリセットされ、初期アーカイブは run の `archive.pt` だけで台本の初期アーカイブを読まない)。v11 は 0 から 300 まで通し。
- chain 評価 `eval_chain_it200.json`・`eval_chain_it300.json` (コマンドの記録なし。JSON の `checkpoint` は延長の run の `model_200.pt`・`model_300.pt`)。

### コマンド

```bash
# smoke test
src/isaac_twisted_rl/scripts/run_job.sh --name smoke_v11 --timeout 20m --detach -- \
    uv run train --task Isaac-Twisted-RL-Knot-v11 --headless --device cuda:0 --seed 42 --max_iterations 2 \
    --run_name smoke_v11 agent.device=cuda:0

# 学習 (S = 43 / 44、N = 0 / 1)
src/isaac_twisted_rl/scripts/run_job.sh --name v11_sS --timeout 40h --detach -- \
    uv run train --task Isaac-Twisted-RL-Knot-v11 --headless --device cuda:N --seed S --max_iterations 301 \
    --run_name v11_sS agent.device=cuda:N agent.save_interval=10

# chain 評価 (IT = 70 / 90 / 100 / 150 / 200 / 250 / 300)
uv run python -u src/isaac_twisted_rl/scripts/rsl_rl/eval_chain.py --task Isaac-Twisted-RL-Knot-Play-v11 \
    --num_envs 512 --headless --device cuda:N --seed 7 \
    --checkpoint logs/rsl_rl/knot_tying_v11/<run>/model_IT.pt agent.device=cuda:N \
    --out logs/rsl_rl/knot_tying_v11/<run>/eval_chain_itIT.json

# 手ごとの診断 (commit b1069a5。--stochastic を付けた run もある)
uv run python -u src/isaac_twisted_rl/scripts/rsl_rl/diag_stage_moves.py --task Isaac-Twisted-RL-Knot-Play-v11 \
    --num_envs 512 --headless --device cuda:1 --seed 7 \
    --checkpoint logs/rsl_rl/knot_tying_v11/<run>/model_IT.pt agent.device=cuda:1 \
    --out outputs/diag_stage_moves/v11_sS_itIT_det.json
```

it70〜100 の評価は学習と同じ GPU で並べて走らせた (実時間だけが変わる)。

## ゲート (学習の前の確認)

- 段の関数の単体テスト (`tests/test_dead_end_stages.py`): 0024 の代表の形、Φ の表の全 321 個で v7 と同じ段、R2 除去を先にすると表の段 2 の 16 個・段 1 の 12 個が 1 段下がること。全体のテスト 350 件が通った。
- 0024 の「データ」の表の再現: `DeadEndStage` を `outputs/archive_pdata/{v7_s43_it100,v7_s44_it100,v7_s43_it330,seed}.json` の `raw` に当て、v11 の列 (段 0 の −1 が 9% / 10% / 70%、段 0 -> 1 が 20% / 17% / 9%、段 1・2 は全形 v7 と同じ段、台本の初期アーカイブは変わらない) がすべて再現した。JSON の `reduced` と `reduce_r2` で計算し直した p-data も全形で一致した。
- smoke test: 2 iteration 回り、`params/env.yaml` に `stage_definition: bfs_dead_end`、初期アーカイブは v7 と同じ `overhand_{num_links}links.pt`。`restore_mismatch` 0。

## 結果

### chain 評価 (固結びの割合 `knot`。512 エピソード、標準誤差は約 2 ポイント)

| iteration | v11 s43 | v11 s44 | v7 s43 (it200 以降は延長) | v7 s44 |
|---|---|---|---|---|
| 70 | 39.5% | 16.6% | 40.6% | 22.9% |
| 90 | 44.1% | 34.4% | 39.5% | 33.8% |
| 100 | 47.1% | 30.3% | 53.7% | 47.3% |
| 150 | 32.0% | 50.2% | — | — |
| 200 | 31.6% | 12.1% | 50.0% | — |
| 250 | 25.4% | 11.3% | — | — |
| 300 | 40.0% | 22.9% | 39.3% | — |

v11 のほかの列:

| run | it | `knot_strict` | `diverged` | `time_out` | 段 1 以上 | 段 2 以上 | 固結びまでの手数 |
|---|---|---|---|---|---|---|---|
| v11 s43 | 100 | 5.3% | 17.4% | 35.5% | 100% | 96.7% | 8.7 |
| v11 s43 | 200 | 9.6% | 14.6% | 53.7% | 100% | 96.7% | 10.9 |
| v11 s43 | 300 | 11.5% | 16.6% | 43.4% | 100% | 97.1% | 8.8 |
| v11 s44 | 150 | 16.0% | 11.5% | 38.3% | 99.6% | 97.9% | 7.5 |
| v11 s44 | 200 | 1.2% | 25.2% | 62.7% | **91.8%** | **51.4%** | 10.2 |
| v11 s44 | 300 | 0.6% | 12.5% | 64.6% | 98.6% | 86.9% | 12.6 |

### 学習ログ (代理の指標。10 iteration の平均)

| 指標 | run | it90〜99 | it140〜149 | it190〜199 | it240〜249 | it290〜300 |
|---|---|---|---|---|---|---|
| `success_s0` | v11 s43 | 0.97 | 0.97 | 0.98 | 0.99 | 0.99 |
| | v11 s44 | 0.97 | 0.97 | **0.69** | **0.44** | **0.50** |
| | v7 s43 (延長) | 0.97 | 0.99 | 0.94 | 0.89 | 0.78 |
| `straight_success` | v11 s44 | — | 0.97 | 0.60 | 0.42 | 0.55 |
| | v7 s43 (延長) | — | 0.995 | 0.997 | 0.997 | 0.997 |
| `success_s1` | v11 s43 / s44 | 0.61 / 0.59 | 0.63 / 0.64 | 0.64 / 0.64 | 0.66 / 0.64 | 0.67 / 0.63 |
| | v7 s43 (延長) | 0.58 | 0.60 | 0.64 | 0.67 | 0.68 |
| `success_s2` | v11 s43 / s44 | 0.19 / 0.22 | 0.20 / 0.24 | 0.28 / 0.25 | 0.23 / 0.25 | 0.38 / **0.06** |
| | v7 s43 (延長) | 0.19 | 0.21 | 0.25 | 0.32 | 0.27 |
| `Train/mean_reward` | v11 s43 / s44 | 0.19 / 0.19 | 0.25 / 0.23 | 0.35 / 0.13 | 0.30 / 0.08 | 0.44 / 0.04 |

- iteration 100 までは、v11 の 2 シードとも報酬・終了の内訳・段ごとの成功・エントロピーが v7 とほぼ同じ曲線だった。v10 のような段 1 の詰まりは無い (`success_s1` 0.59〜0.61、v7 0.57〜0.58)。v11 の `success_s1`・`fail_s1` は、段の数え方の違い (偽の交差の形が「段 1 のまま」になる) の分だけ v7 より良く見える。
- v11 s44 の段 0 の崩れ: `timeout_s0` が 0.02 (it140) -> 0.47 (it210) に増え、`fail_s0` は 0.006 -> 0.001 に減った。`Loss/value_function` は 0.37 -> 0.85。段 2 も it270 から崩れた (`success_s2` 0.25 -> 0.04、`timeout_s2` 0.07 -> 0.32)。
- v11 s43 は段 0 が 0.99 のまま。段 2 は it210〜240 に一度下がり (0.17〜0.23、`timeout_s2` 0.3)、it300 で 0.38。
- v7 s43 の延長も段 0 が下がった (0.99 -> 0.75) が、`straight_success` は 0.997 のまま (アーカイブの形から進めない型)。v11 s44 はまっすぐなロープからも下がった (別の型)。

### 段 0 の手ごとの診断 (`diag_stage_moves.py`、決定論的、各環境の最初のエピソード)

| チェックポイント | 段 0 の手の数 | 段が上がった | 形が変わらない | 同じ段で形が変わった | 床に落ちていない | 発散 |
|---|---|---|---|---|---|---|
| v11 s44 it140 | 1172 | 82% | 12% | 3% | 1.4% | 0.9% |
| v11 s44 it200 | 3818 | **16%** | **82%** | 0.2% | 0.1% | 1.2% |
| v11 s44 it300 | 3571 | 24% | 75% | 0.3% | 0.1% | 0.3% |
| v11 s43 it300 | 880 | 88% | 8% | 3% | 0.3% | 0.2% |

確率的な方策 (`--stochastic`) でも s44 it140 は上がった 58%・変わらない 37%、it200 は 17% / 79%。

形が変わらない手と上がった手の中身 (平均。手ごとのばらつきは大きい。リンクは 0〜27、掴めるのは 2〜25):

| チェックポイント | 手の結果 | 掴むリンク | 置く先 (掴んだリンクからの差) | 運んだ距離 | 離した指先の高さ |
|---|---|---|---|---|---|
| s44 it140 | 上がった | 4.5 | 中央の側へ 6.3 | 0.46 m | 9.6 cm |
| s44 it200 | 上がった | 14.4 | 端の側へ 7.9 | 0.43 m | 5.2 cm |
| s44 it200 | 形が変わらない | 7.8 | 端の側へ 4.5、置く位置のずらし (行動 ch2 / ch3) +0.49 / +0.72 | 0.39 m | 7.1 cm |
| s43 it300 | 上がった | 18.4 | 端の側へ 7.3 | 0.41 m | 6.2 cm |

- 形が変わらない手でも、ロープは掴めていて (it200 で 99.8%) 0.4 m 近く運んでいる。動かない手や、ロープを浮かせて段の判定を逃れる手ではない。
- 最初の診断の出力 (`*_bad_grasp.json`) は、エピソードが終わった環境で `processed_actions` が 0 に戻るため、掴む点・置く先の列が誤り。手の結果の割合と行動の生の値 (`act_mean`) は正しい。掴む点を直した出力は s44 it140・it200、s43 it300 の決定論的なものだけ (`outputs/diag_stage_moves/v11_s*_it*_det.json`)。

## 結論

- 分かった:
  - v11 は iteration 300 で v7 を上回らなかった。it300 の chain 評価は v11 s43 40.0%・s44 22.9%、v7 s43 の延長 39.3% (v7 s44 の延長はまだ無く、比べる相手は 1 シード)。
  - v11 の 2 シードの it100 -> it300 は 47.1% -> 40.0%、30.3% -> 22.9%。v11 のシードどうしの差は 14〜23 ポイントあり、1 シードどうしの比べ方では v7 との差を結論にできない。
  - v11 s44 は iteration 150 以降、段 0 (まっすぐなロープを含む) で交差を作らない手を繰り返して時間切れになった (段 0 の手の 82% が形を変えない)。段 1 の方策の手の結果はほぼ変わらなかった。
  - v10 のような段 1 の詰まりは起きなかった (iteration 100 まで学習ログは v7 とほぼ同じ)。
- 推測:
  - s44 の段 0 の崩れは、段 0 の方策の置き方が「中央の側へ高めに離して輪を作る」から「端寄りを掴んで端の側の外へ大きくずらして置く」へずれたもの。v11 では段 0 から行き止まりの形を作ると −1、時間切れは 0 なので、交差を作る手そのものが避けられた可能性がある (段 0 の失敗が 0.006 -> 0.001 に減ったこととは合う)。s43 では起きていないので、シードによる偶然の可能性も同じくらいある。
  - v11 s43 の it150〜250 の落ち込み (32% -> 25%) と it300 の回復は、段 2 の学習ログの落ち込み・回復と時期が合う。
- 未確認:
  - s44 の崩れが v11 の段の定義のせいか (同じ s44 の it140 から v7 の定義で続けて比べる、などで分けられる)。
  - v7 s44 を iteration 300 まで延ばしたときの chain 評価。
  - v11 の段 0 のアーカイブに、行き止まりではないが進めない形が溜まったか (s43 の it100 の写しはあるが、p-data はまだ書き出していない)。

## 次の作業

- v7 s44 を iteration 300 まで延ばし (v7 s43 の延長と同じ形)、v11 と 2 シードずつ比べる。
- s44 の崩れの切り分け (上の「未確認」の 1 つ目)。
- 続けるかの判断は、上の 2 つの後にユーザーと決める。
