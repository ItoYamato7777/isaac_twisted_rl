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
uv run train --task Isaac-Twisted-RL-Knot-vN --headless --device cuda:0 --seed 42 --max_iterations 2 \
    --run_name smoke_vN agent.device=cuda:0
```

## 2. 学習 (シードごとに 1 本)

固定する引数: `--headless`、`--device` と `agent.device` を同じに、`agent.save_interval=10`、`--run_name <略>_s<シード>`。
ロープ `Knot` 以外のプリセットで学習するときは、ログ名にプリセットが入らないので run 名に入れる。

```bash
PYTHONUNBUFFERED=1 uv run train --task Isaac-Twisted-RL-Knot-vN --headless --device cuda:0 --seed 42 \
    --max_iterations 100 --run_name <略>_s42 agent.device=cuda:0 agent.save_interval=10 \
    > outputs/jobs/<略>_s42.log 2>&1
```

- Bash の `run_in_background` で起動し、待つときは前景の `sleep` を使わない (Monitor か `run_in_background` の until ループ)。
- 起動したら PID・GPU・run のフォルダ・コマンドを `docs/STATUS.md` の「走っているジョブ」に書く。
- 物理などを Hydra で上書きするなら、上書きをすべてメモする (評価にも同じものを付ける)。
- 止めてよいのは自分が起動して記録した PID だけ。`pkill -f` は使わない。

## 3. chain 評価

シードごと・評価する iteration ごとに 1 回。最後のチェックポイントの番号は `ls logs/rsl_rl/<ログ名>/<run>/` で確かめる。

```bash
uv run python -u src/isaac_twisted_rl/scripts/rsl_rl/eval_chain.py --task Isaac-Twisted-RL-Knot-Play-vN \
    --num_envs 512 --headless --device cuda:0 --seed 7 \
    --checkpoint logs/rsl_rl/<ログ名>/<run>/model_<it>.pt agent.device=cuda:0 <学習と同じ Hydra の上書き> \
    --out logs/rsl_rl/<ログ名>/<run>/eval_chain_it<it>.json
```

JSON にはシードと Hydra の上書きが入らない。評価のコマンドをそのまま実験記録に書く。

## 4. 集計

1. シードごとに `knot`・`diverged`・`time_out`・`max_stage_ge{k}` を表にする。
2. 標準誤差 `sqrt(p (1 - p) / 512)` (512 エピソードで 2 ポイント前後) と、シードどうしの差を並べる。
3. 基準の版と比べる。1 シードどうしの 10〜15 ポイントの差は結論に使わない。
4. 学習ログの段ごとの成功率と `chain_est` は比較に使わない。

## 5. 記録

skill `experiment-record` で `docs/experiments/` に書く。否定的な結果も書く。結論は「分かった / 推測 / 未確認」に分ける。
