---
name: train-eval
description: 版や設定を学習し、chain 評価で比べるときに使う。起動前の確認、スモークテスト、シード 2 本以上の学習、chain 評価、集計 (標準誤差とシード差) までの手順。評価の定義を知りたいだけなら docs/system/evaluation.md を読む。
---

# 学習 → chain 評価 → 集計

評価の定義と比べ方の規則は `docs/system/evaluation.md`、ログの置き場所は `docs/system/logs.md` にある。この skill はコマンドの手順だけを持つ。

## 0. 起動の前に確かめる

1. 仮説・比べる相手 (基準の版)・指標 (chain 評価の `knot`)・シード数が `docs/STATUS.md` の「次の作業」にある。
2. `git status --porcelain --untracked-files=all` が空。空でなければ commit する (未追跡のファイルは run のフォルダの `git/*.diff` に残らない)。
3. `nvidia-smi` で GPU の空きを見る。GPU 2 枚を他のジョブと共有している。自分が起動していないジョブには触らない。
4. 物理・ロープのプリセット・紐モデルを変えたなら、先に skill `physics-check` を通す。

## 1. スモークテスト

数 iteration だけ回し、起動・観測の次元・報酬の配線が落ちないことを確かめる。run のフォルダが 1 つできるので、run 名に `smoke` を入れる。

```bash
src/isaac_twisted_rl/scripts/run_job.sh --name smoke_vN --timeout 20m --detach -- \
    uv run train --task Isaac-Twisted-RL-Knot-vN --headless --device cuda:0 --seed 42 --max_iterations 2 \
    --run_name smoke_vN agent.device=cuda:0
```

v6 (4096 環境) では 1 iteration が約 3 分、2 iteration で約 7 分かかった (2026-10-04、`outputs/jobs/2026-10-04_18-24-56_smoke_runjob_v6`)。
終わりに出る `TypeError: post_quit()` は Kit の終了処理のもので、無視してよい (`docs/experiments/2026-09-28_v4_smoke-test.md`)。

## 2. 学習 (シードごとに 1 本)

固定する引数: `--headless`、`--device` と `agent.device` を同じに、`agent.save_interval=10`、`--run_name <略>_s<シード>`。
ロープ `Knot` 以外のプリセットで学習するときは、ログ名にプリセットが入らないので run 名に入れる。

```bash
src/isaac_twisted_rl/scripts/run_job.sh --name <略>_s42 --timeout <上限> --detach -- \
    uv run train --task Isaac-Twisted-RL-Knot-vN --headless --device cuda:0 --seed 42 \
    --max_iterations 100 --run_name <略>_s42 agent.device=cuda:0 agent.save_interval=10
```

- run_job.sh は起動の前に GPU の空きを確かめ、`outputs/jobs/<時刻>_<略>_s42/` に job.log・コマンド・commit・差分・GPU の状態・pid を残し、終わると `exit.txt` を書く。
- 終わりを待つときは前景の `sleep` を使わない (`run_in_background` で `until [ -f <job>/exit.txt ]; do sleep 10; done`、か Monitor)。
- 起動したら pid (run_job.sh が出す。`timeout` のプロセス)・job のフォルダ・GPU・run のフォルダ・コマンドを `docs/STATUS.md` の「走っているジョブ」に書く。
- 最後まで走ったかは `exit.txt` (`exit: 0` で `signal:` の行が無い) と最後の `model_*.pt` で確かめる。
- 物理などを Hydra で上書きするなら、上書きをすべてメモする (評価にも同じものを付ける)。
- 止めてよいのは自分が起動して記録した PID だけ。job の pid に SIGTERM を送る (timeout が子に伝え、止まらなければ SIGKILL)。`pkill -f` は使わない。

## 3. chain 評価

シードごと・評価する iteration ごとに 1 回。最後のチェックポイントの番号は `ls logs/rsl_rl/<ログ名>/<run>/` で確かめる。

```bash
src/isaac_twisted_rl/scripts/run_job.sh --name eval_<略>_s42_it<it> --timeout <上限> --detach -- \
    uv run python src/isaac_twisted_rl/scripts/rsl_rl/eval_chain.py --task Isaac-Twisted-RL-Knot-Play-vN \
    --num_envs 512 --headless --device cuda:0 --seed 7 \
    --checkpoint logs/rsl_rl/<ログ名>/<run>/model_<it>.pt agent.device=cuda:0 <学習と同じ Hydra の上書き> \
    --out logs/rsl_rl/<ログ名>/<run>/eval_chain_it<it>.json --video
```

JSON にはシードと Hydra の上書きが入らない。評価のコマンドをそのまま実験記録に書く。

### 動画 (`--video`)

chain 評価には `--video` を付ける ([0031](../../../docs/decisions/0031-eval-chain-video.md))。評価したエピソードのうち先頭 8 環境の 1 手の中を撮り、
`video/sim/<ログ名>/<run>/<JSON の名前>/` に mp4 と `manifest.json` を置く (動画は git の管理外)。目的は、報酬のハックや物理的におかしい挙動を目で見つけること。

- **動画は結論の根拠にしない。** 動画で見たことは実験記録に「観察 (推測)」と書き、数で確かめるまで事実にしない。
- GPU の空きが `--video_min_free_mib` (既定 6000 MiB) より少なければ、動画なしで評価だけが走り、JSON の `video` キーに撮らなかった理由が入る。その場合は実験記録に「撮らなかった」と書く。
- 撮影の分、評価が延びる (VRAM が約 5 GB 以上増え、描画の時間が足される)。所要時間は初回の実測で下の欄を埋める。
- **未確認 (0031 のゲート 4)**: `--video` を付けても JSON の値が変わらないことは、512 環境ではまだ確かめていない (`--enable_cameras` で Kit の設定が変わる)。確かめるまでは、動画付きの評価の値と、付けなかった評価の値 (v14 など) を並べるときに「動画付き」と書く。

## 4. 集計

1. シードごとに `knot`・`diverged`・`time_out`・`max_stage_ge{k}` を表にする。
2. 標準誤差 `sqrt(p (1 - p) / 512)` (512 エピソードで 2 ポイント前後) と、シードどうしの差を並べる。
3. 基準の版と比べる。1 シードどうしの 10〜15 ポイントの差は結論に使わない。
4. 学習ログの段ごとの成功率と `chain_est` は比較に使わない。

## 5. 記録

skill `experiment-record` で `docs/experiments/` に書く。否定的な結果も書く。結論は「分かった / 推測 / 未確認」に分ける。
