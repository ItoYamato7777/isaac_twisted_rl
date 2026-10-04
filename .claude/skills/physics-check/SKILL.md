---
name: physics-check
description: 物理パラメータ・ロープのプリセット・紐モデルを変えたとき、学習の前に、台本で固結びが結べるか (ゲート 1) と 1 手あたりの発散率を測るときに使う。初期アーカイブの作り直しを含む。学習中のログの発散率を読むだけのときは使わない。
---

# 物理の確認 (ゲート 1 と発散の計測)

学習を始める前の関門。物理・プリセット・紐モデルを変えたときだけ行う (MDP だけを変えた版では要らない)。
過去の結果: ゲート 1 は `docs/experiments/2026-09-27_gate1.md`、発散の計測は v5 の記録 (`docs/experiments/` の一覧)。

## 1. ゲート 1: 台本で固結びが結べるか

学習と同じタスクの環境で、台本 (5 手) を揺らしながら回し、固結びになった割合を数える。台本は常にまっすぐなロープから始まる (commit `a9b02b3` で固定した)。

起動は他の Isaac Sim のジョブと同じく `run_job.sh` で包む (下のコマンドは中身だけを書く)。

```bash
src/isaac_twisted_rl/scripts/run_job.sh --name gate1_<名前> --timeout <上限> --detach -- \
    uv run python src/isaac_twisted_rl/tasks/knot_tying/demos/overhand_script.py --task Isaac-Twisted-RL-Knot-vN \
    --num_envs 128 --jitter 0.25 --headless --out outputs/overhand_script/<物理の条件の名前>
```

- 揺らしは `--jitter 0.25` (過去の記録はすべてこの値)。環境は 128 が標準。
- 尺: 既定は遅い尺。`--fast` で学習の尺。`--replay_actions` で 6 次元の行動に写して `env.step()` で再生する (学習の行動空間で結べるかの確認)。
- **出力のフォルダは物理の条件ごとに分ける** (同じフォルダに上書きしない)。
- 結果は最後の `[final] ... overhand=N/128 diverged=M` の行。条件ごとに表にする。

## 2. 初期アーカイブの作り直し (v4 以降の学習が読む)

ゲート 1 の遅い尺で、台本の途中の状態を段ごとに保存する。

```bash
... overhand_script.py --task <ID> --num_envs 128 --jitter 0.25 --headless --out outputs/overhand_script/<名前> \
    --dump_archive outputs/archive_seed/<新しい名前>.pt [--dump_knots outputs/archive_seed/<新しい名前>_knots.pt]
```

- **既存のファイル (`outputs/archive_seed/overhand_28links.pt` など) に上書きしない。** v4〜v7 が使う共有の資産で、git の管理外なので戻せない。新しい名前で作り、設定で指す。
- ロープの USD を作り直すときは `generate_assets.py --rope <名前>` で対象を絞る (`--rope` を付けないと全プリセットを作り直す)。

## 3. 1 手あたりの発散の計測

```bash
src/isaac_twisted_rl/scripts/run_job.sh --name div_<名前> --timeout <上限> --detach -- \
    uv run python src/isaac_twisted_rl/tasks/knot_tying/demos/divergence_bench.py --task Isaac-Twisted-RL-Knot-vN \
    --num_envs 1024 --steps 24 --headless --archive_prob 0.0 --cfg "<Hydra の上書き>"
#   学習済みの方策で打つとき: --policy ckpt --checkpoint logs/rsl_rl/<ログ名>/<run>/model_<it>.pt
```

- **どの判定で数えたかを記録に書く。** 判定は 3 通りある: 終了項 `diverged` (NaN・高さ・水平 3 m・速さ 20 m/s。閾値は版で違う)、NaN だけ、`divergence_bench` の spike。定義は `docs/glossary.md`。
- 行動の与え方 (一様乱数 + まっすぐ / 一様乱数 + 初期アーカイブ / 学習済みの方策) で値が変わる。条件と一緒に書く。
- 計測で効いた対策が、学習では逆効果だったことがある (v5。`docs/decisions/` の v5 の教訓)。計測は学習の代わりにならない。
- 自作の計測で終了項を `get_term` で読むときは、reset した環境で絞る (`env.step()` の順序の罠。`.claude/rules/mdp.md`)。

## 4. 記録

skill `experiment-record` で、条件ごとの表 (ゲート 1 の `[final]`、発散率と判定の種類) を実験記録に書く。出力のフォルダのパスも書く。
