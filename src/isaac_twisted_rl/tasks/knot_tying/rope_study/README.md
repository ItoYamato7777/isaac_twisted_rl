# rope_study: 紐のシミュレーションモデルの比較

同じ MDP (v4) のまま、紐のシミュレーションモデルだけを差し替えて比べるためのタスク群と計測の道具。
何を比べたか・結果・結論は [`docs/experiments/2026-10-02_rope_models.md`](../../../../../docs/experiments/2026-10-02_rope_models.md) (打ち切り中)。
rope study の run の最後のチェックポイントは `model_99.pt` (`model_100.pt` は無い)。

## モデル

| 名前 | 表現 | `scene["rope"]` |
|---|---|---|
| `Art28` | PhysX Articulation、カプセル 28 本 (4 cm)。v4 の `Knot` と同じ | 本物の Articulation (アダプタなし) |
| `Art28LowMu` | 同上、摩擦を実物の範囲 (紐-紐・紐-床 0.5、指 1.0) | 同上 |
| `Art56` | PhysX Articulation、カプセル 56 本 (2 cm)、1 本おきの接触を除外 | `ResampledArticulationRope` |
| `XpbdC28` / `XpbdC56` | Warp の XPBD Cosserat 棒、28 / 56 区間 | `XpbdRope` |
| `XpbdC28LowMu` | `XpbdC28`、摩擦を実物の範囲 | `XpbdRope` |
| `XpbdMS28` | Warp の XPBD 質点-ばね、28 区間 | `XpbdRope` |
| `Fem` | PhysX の FEM 変形体 (起動を確認。学習はしていない。報告書 §7.1.8) | `FemRope` |

一覧は `models.py`、環境設定は `env_cfg.py`、PPO は `agents/rsl_rl_ppo_cfg.py` (v4 と同一)。
紐の差し替えの仕組み (アダプタ・仮想リンク・XPBD のソルバ) は `../common/rope_backends/`。

タスク ID:

    Isaac-Twisted-RL-RS-<モデル>-v4        学習
    Isaac-Twisted-RL-RS-<モデル>-Play-v4   chain 評価

観測・行動は「中心線を弧長で 28 等分した仮想リンク」で全モデル共通なので、どのモデルで学習した方策も
どのモデルの `-Play-v4` でそのまま評価できる。

## 使い方

リポジトリのルートで。Isaac Sim を起動するものは `python -u`。GPU を共有しているときは起動前に
`nvidia-smi` で空きメモリを確かめる (4096 環境の学習 ≈ 7.8 GB、128 環境の試験 ≈ 2.5 GB)。

```bash
# 環境が立ち上がるか (観測の形・終了の内訳・発散)
.venv/bin/python -u src/isaac_twisted_rl/tasks/knot_tying/rope_study/smoke_test.py \
    --task Isaac-Twisted-RL-RS-XpbdC28-v4 --num_envs 16 --steps 3 --headless --device cuda:0

# 物理試験をまとめて (ゲート 1 の遅い尺 -> 物理ベンチ -> ゲート 1 の学習の尺)。済んだステップは飛ばす
bash src/isaac_twisted_rl/tasks/knot_tying/rope_study/run_physics_suite.sh cuda:0 "Art28 XpbdC28" 128

# 物理ベンチだけ (設定の上書きつき)
.venv/bin/python -u src/isaac_twisted_rl/tasks/knot_tying/rope_study/bench_physics.py --model Art56 --num_envs 128 \
    --headless --device cuda:0 --label Art56_it16 --out outputs/rope_study/physics/Art56_it16 \
    --cfg "scene.rope_native.spawn.articulation_props.solver_position_iteration_count=16"

# 片持ち梁 (曲げ剛性が名目値どおりか)
.venv/bin/python src/isaac_twisted_rl/tasks/knot_tying/rope_study/bench_cantilever_xpbd.py --device cpu \
    --out outputs/rope_study/cantilever/xpbd.json                     # XPBD (Kit 不要)
.venv/bin/python -u src/isaac_twisted_rl/tasks/knot_tying/rope_study/bench_cantilever_art.py \
    --headless --device cuda:0 --out outputs/rope_study/cantilever/art.json  # 剛体鎖

# 学習
uv run train --task Isaac-Twisted-RL-RS-XpbdC28-v4 --headless --device cuda:0 --seed 42 --max_iterations 100 \
    --run_name rs_XpbdC28_s42 agent.device=cuda:0 agent.save_interval=10
# 学習を順に (先に走っている PID の終了を待ってから。初期アーカイブができるまで待つ)
bash src/isaac_twisted_rl/tasks/knot_tying/rope_study/run_train_chain.sh cuda:0 <PID or 0> XpbdC28 XpbdMS28

# 評価をまとめて (ジョブ一覧 -> キュー -> 表)
.venv/bin/python src/isaac_twisted_rl/tasks/knot_tying/rope_study/make_eval_jobs.py --policies Art28,XpbdC28 --iters 50,99 \
    --out outputs/rope_study/eval/jobs.txt
bash src/isaac_twisted_rl/tasks/knot_tying/rope_study/run_eval_queue.sh outputs/rope_study/eval/jobs.txt cuda:0
.venv/bin/python src/isaac_twisted_rl/tasks/knot_tying/rope_study/summarize_eval.py --md outputs/rope_study/eval/summary.md
# 学習中にチェックポイントが増えるたびに評価 (2 GPU で同時に回してよい。止めるときは outputs/rope_study/eval/STOP)
bash src/isaac_twisted_rl/tasks/knot_tying/rope_study/run_eval_loop.sh cuda:0 600 --iters 50,99 \
    --extra_policy v4Knot=logs/rsl_rl/knot_tying_v4/2026-10-01_19-22-13_overnight_s42

# 評価を 1 本だけ (自分のモデル / 別のモデル / 摂動)
.venv/bin/python -u src/isaac_twisted_rl/tasks/knot_tying/rope_study/eval_robust.py \
    --task Isaac-Twisted-RL-RS-Art28-Play-v4 --checkpoint logs/rsl_rl/knot_tying_rs_xpbdc28/<run>/model_99.pt \
    --num_envs 512 --seed 7 --headless --device cuda:0 --obs_noise 0.005 \
    env.actions.pick_place.exec_noise_xy=0.005 --out outputs/rope_study/eval/XpbdC28/on_Art28_noise5mm.json
# XPBD の物理パラメータの摂動 (曲げ剛性 x0.5 など。`XpbdRodParams` の項)
    ... --task Isaac-Twisted-RL-RS-XpbdC28-Play-v4 --xpbd bend_stiffness=3.927e-4 --xpbd twist_stiffness=1.9635e-4
```

## ファイル

| ファイル | 内容 |
|---|---|
| `models.py` | モデルの一覧 (純 dataclass。Kit 起動前に import される) |
| `env_cfg.py` | v4 の設定を継承し、モデルごとに紐・摩擦・グリッパー・行動・初期アーカイブを埋める |
| `agents/rsl_rl_ppo_cfg.py` | v4 と同一の PPO。ログは `logs/rsl_rl/knot_tying_rs_<モデル小文字>/` |
| `run_physics_suite.sh` | 物理試験をモデルごとに順に回す |
| `bench_physics.py` | 物理ベンチ (settle / hang / drag / knot / random)。出力 `outputs/rope_study/physics/<モデル>/` |
| `bench_cantilever_xpbd.py` / `bench_cantilever_art.py` | 片持ち梁 (XPBD / 剛体鎖) |
| `elastica.py` | heavy elastica の数値解と、離散鎖の小たわみの解析解 |
| `eval_robust.py` | chain 評価を別モデル・観測ノイズ・Hydra の上書きの下で回す |
| `make_eval_jobs.py` | 評価のジョブ一覧 (自分のモデル・モデル間の転移・物理の摂動・観測ノイズ / 実行誤差) を作る。`INVALID.txt` のある run は飛ばす |
| `run_eval_queue.sh` | ジョブ一覧を 1 本ずつ回す (出力済みは飛ばす。2 GPU で同じ一覧を同時に回せる。失敗したジョブは `.log` を消すと再実行) |
| `run_eval_loop.sh` | 一覧の作成とキューを繰り返す (学習中に増えるチェックポイントを順に評価) |
| `run_train_chain.sh` | 学習を 1 本ずつ順に回す (GPU の学習の枠 1 つぶん) |
| `summarize_eval.py` | 評価結果を表にする (自分のモデル・転移行列・摂動・選定に使う平均と最悪値。最後の iteration と自分のモデルで良い方の iteration の 2 通り) |
| `summarize_train.py` | 学習ログを表にする (1 iteration の時間・手/s・発散・段ごとの成功率) |
| `smoke_test.py` | 環境を立ち上げて乱数の手を数手打つ |
| (移動) `docs/history/rope_study_HANDOFF.md` | 作業の引き継ぎメモ (経緯。凍結) |
| (移動) [`docs/experiments/2026-10-02_rope_models.md`](../../../../../docs/experiments/2026-10-02_rope_models.md) | 報告書 |

## 注意

* 初期アーカイブはモデルごと (`outputs/archive_seed/rope_study/<モデル>.pt`)。`run_physics_suite.sh` の
  ゲート 1 (遅い尺) が作る。無ければ警告して空から始める。v4〜v7 が使う
  `outputs/archive_seed/overhand_28links.pt` とは別のファイル。
* 紐の USD を作り直すときは対象を限定する (`generate_assets.py --rope knot56 --only rope`)。
  `common/rope_model/data/rope_knot.usd` は v4〜v7 が使う。
* XPBD の紐は USD に存在しない (GUI では節点に球のマーカーを描く)。PhysX のグリッパーは紐と接触しない。
* XPBD は環境の原点を引いた局所座標で解く (`adapters.XpbdRope`)。world 座標のままだと、原点が遠い環境で床・紐どうしの
  動摩擦が float32 の丸めで消える (報告書 §7.1.6)。2026-10-02 23:50 より前の XPBD の結果は無効
  (`outputs/rope_study/xpbd_world_coords/`)。
* FEM (`Fem`) は `scene.replicate_physics = False` が要る (PhysX は変形体の物理の複製に対応していない)。
* `xpbd_rod.py` を変えると、そのあとに起動したプロセスにだけ効く。物理試験のキューを回している途中で
  変えたときは、どの結果が変更前/後かを記録する。
