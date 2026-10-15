# STATUS (現在地)

- 状態: 現行。上書きで更新する (追記しない。手順は skill `handoff`)
- 読むべき状況: セッションの始め。今どこにいて、次に何をするかを知るとき
- 最終更新: 2026-10-15

## 要点

- **段 0 の崩れの原因の最有力: 時間切れの bootstrap の不具合** (v4 以降の全版)。6 手目に段が上がると時間切れも立ち、PPO が +1 の手にも γ·V を足す。崩れた方策はまっすぐなロープから 6 手目に段 0 を上げていた ([記録](experiments/2026-10-15_timeout-bootstrap-and-stage2-bottleneck.md)。因果は未確認)。次は、これを直す版の作り方をユーザーと決める (「次の作業」1)。
- **崩れない run の頭打ち (30〜50%) は段 2 の詰まり**: 段 2 の 1 手の固結び 8〜18%・段 1 への下降 25〜47% で、ほぼ説明できる (マルコフ連鎖の模型の上。同じ記録)。
- v14 seed 43 (GPU 1)・seed 44 (GPU 0) を学習中。どちらも不具合を持つので、it100〜150 の崩れを見守る。

## 版の状態

数字は各記録にある (ここには書かない)。比べ方は [`docs/system/evaluation.md`](system/evaluation.md)。v4 以降の全版が時間切れの bootstrap の不具合を持つ ([記録](experiments/2026-10-15_timeout-bootstrap-and-stage2-bottleneck.md))。

| 版 | 状態 | 記録 |
|---|---|---|
| v0 | 学習した (交差 1 つは学べた) | [2026-09-15_v0-v1](experiments/2026-09-15_v0-v1.md) |
| v1 | 学習した (失敗: 疎報酬が一度も発火しない) | 同上 |
| v2 | 学習した (偽の C2 に居座った) | [2026-09-24_v2_fake-c2](experiments/2026-09-24_v2_fake-c2.md) |
| v2-R2 | 発散の報酬の修正 (2026-09-27) の後の本番の学習は**未実施**。「本物の C2 が稀で詰まる」は、ランダム方策と数 iteration の観察からの**仮説** | [2026-09-27_v2r2_fix](experiments/2026-09-27_v2r2_fix.md) |
| v3 | 本番の run 1 本 (シード 1 本) | [2026-09-28_v3_main](experiments/2026-09-28_v3_main.md) |
| v4 | chain 評価 済み (seed 42 だけ) | [2026-10-01_v4-v6_overnight](experiments/2026-10-01_v4-v6_overnight.md) |
| v5 | chain 評価 済み (seed 42 だけ) | 同上 |
| v6 | chain 評価 済み (seed 42・43) | 同上 |
| v7 | chain 評価 済み (seed 43・44 は it100 まで、s43 は延長の it330 まで。seed 42 に当たる `abl_depB` は it70 まで) | [2026-10-04_v7_seeds](experiments/2026-10-04_v7_seeds.md) |
| v8 | chain 評価 済み (seed 43・44)。v7 を下回った | [2026-10-05_v8_symmetry](experiments/2026-10-05_v8_symmetry.md)、[0022](decisions/0022-p2-one-version-each.md) |
| v9 | chain 評価 済み (seed 43・44)。v7 を上回らなかった | [2026-10-05_v9_stage2-weight](experiments/2026-10-05_v9_stage2-weight.md) |
| v10 | chain 評価 済み (seed 43・44)。v7 を大きく下回った | [2026-10-06_v10_path-stages](experiments/2026-10-06_v10_path-stages.md)、[0023](decisions/0023-v10-path-stages.md) |
| v11 | chain 評価 済み (seed 43・44、it70〜300。s43 は it1000 まで延長し、chain 評価は it1000 だけ)。v7 を上回らなかった。s44 が段 0 で崩れた | [2026-10-07_v11_dead-end-stages](experiments/2026-10-07_v11_dead-end-stages.md)、[2026-10-09_v11_stage2-archive-loop](experiments/2026-10-09_v11_stage2-archive-loop.md)、[0024](decisions/0024-v11-dead-end-stages.md) |
| v12 | chain 評価 済み (seed 43・44、it70〜300)。s43 は横ばい、s44 は it70 の後に段 0 から崩れた | [2026-10-13_v12_archive-stage-up](experiments/2026-10-13_v12_archive-stage-up.md)、[0025](decisions/0025-archive-save-on-stage-up.md) |
| v13 | chain 評価 済み (seed 43・44、it70〜300)。罰則は張り付きの割合を減らさず、s44 は v12 s44 と同じ型に崩れた | [2026-10-14_v13_bounds-loss](experiments/2026-10-14_v13_bounds-loss.md)、[0026](decisions/0026-v13-action-bounds-loss.md) |
| v14 | **seed 43・44 を学習中** (段 0 の作り直し + DR + 特権観測の critic。DR の ζ は ×1.0〜×2、ほかの 5 軸は ×0.5〜×2) | [0027](decisions/0027-v14-s0-reshape-dr-priv-critic.md)、[ゲート](experiments/2026-10-14_v14_physics-gate.md)、[versions/v14](../src/isaac_twisted_rl/tasks/knot_tying/versions/v14/README.md) |
| v15 | 未着手 (v14 + ロープの直径・長さの DR)。[0028](decisions/0028-v15-rope-size-dr.md) で決定済み。v14 の結果と、不具合を直す版との順番を決めてから | 「次の作業」5 |

## 走っているジョブ

| ジョブ | GPU | pid | job のフォルダ | run のフォルダ | 起動 | 終わりの見込み |
|---|---|---|---|---|---|---|
| v14 s43 (0 -> it300) | 1 | 262619 (`run_job.sh` の timeout 30h) | `outputs/jobs/2026-10-15_15-07-05_v14_s43` | `logs/rsl_rl/knot_tying_v14/2026-10-15_15-07-10_v14_s43` | 2026-10-15 15:07 | 2026-10-16 12:30 頃 (it100 が 22:15、約 4.3 分/iteration) |
| v14 s44 (0 -> it300) | 0 | 344651 (同上) | `outputs/jobs/2026-10-15_16-07-16_v14_s44` | `logs/rsl_rl/knot_tying_v14/2026-10-15_16-07-22_v14_s44` | 2026-10-15 16:07 | 2026-10-16 15:00 頃 (it80 が 22:00、約 4.6 分/iteration) |

- コマンドは v13 と同じ形: `--task Isaac-Twisted-RL-Knot-v14 --device cuda:N --seed S --max_iterations 301 --run_name v14_sS agent.device=cuda:N agent.save_interval=10`。commit `d8dc8f7`、Hydra の上書きは `agent.*` だけ (評価には何も付けない)。v14 の run には `Loss/value_mean_s*` が無い (その commit より前)。
- 2026-10-15 の学習ログ (it70〜90 まで) は健全: `timeout_s0` 0.01〜0.02、`straight_success` 0.96〜0.98。v13 s44 の同じ時期と同じ水準。
- **他のジョブ (自分が起動していない。触らない)**: anaconda の python の実機の作業 (pid 560983 `ensenso_ROPE/tools/run_online.py`、616313 `hiro_knot run --real`、895144・2892930 `knot_policy_real.py --real`、876400 jupyter)。GPU 0・1 は他のユーザー (ohtsukatakeshi) とも共有している。

## 次の作業

1. **時間切れの bootstrap を直す版を決める** (ユーザーの判断)。案: v13 を継承し、時間切れの項を `time_out & ~(段の上昇 | 段の下降 | 発散)` に置き換える (1 版 1 変数)。critic-inflation の記録の案 B (V を 1 で頭打ち)・C (critic の出力を [−1, 1]) は 6 手目に上げる得 (目標 1 + γ·1) を残すので勧めない。案 A (時間切れを終端にする) は得を消すが、時間切れの bootstrap 自体もやめる。版の番号 (v15 は 0028 のロープ寸法の DR に予約済み) と、v14 の結果を待つかもユーザーと決める。決めたら `docs/decisions/` に書き、skill `add-version`。外れと言える予想: 2 シードとも段 0 の V が 1/γ を超えない・段 0 を 6 手目に上げる山が出ない・`timeout_s0` が 0.1 を超えない。
2. **v14 の見守り**: it100〜150 で `timeout_s0` が 0.1 を超えないか (v12 s44 は it100 前後、v13 s44 は it120 前後で超えた)。崩れたら、そのチェックポイントで `diag_stage_moves.py --dump_obs` を回し、`chain_moves.py sequences` で 6 手目の山を数える。完走は 2 本とも `exit.txt` に `signal:` の行が無いことと `model_300.pt` で確かめる。
3. v14 の chain 評価 (公称の物理、Play の既定) を it70 / 100 / 150 / 200 / 250 / 300 で v13 と並べる。DR を付けた評価 (`env.events.rope_dr.params.enabled=true`) を v14 と v13 に。0027 の予想 (段 0 のアーカイブに交差の無い形が溜まり `timeout_s0` が 0.1 を超えない・公称で v13 を 5 ポイント以上下回らない・DR を付けた評価の下がり幅が v13 より小さい) を確かめる。チェックポイント 1 つの値は隣と平均 9 ポイント揺れる ([記録](experiments/2026-10-15_timeout-bootstrap-and-stage2-bottleneck.md))。iteration を並べて、1 点の差で結論しない。v13 と内訳 (`outcome_by_last_stage`・`action_saturation`) を並べるなら v13 も測り直す (決定論的なので既存のキーは記録と同じ値)。
4. **頭打ち (段 2) の切り分けの計測** (GPU。使う前にユーザーに確かめる。コマンドの案は [記録](experiments/2026-10-15_timeout-bootstrap-and-stage2-bottleneck.md) の「次の作業」):
   - 段 2 の形ごとの最良の 1 手 (`eval_search.py --mode shapes --shape_moves 1`、`--deterministic` なし)。最良が高ければ選び方 (平均の行動・std・離散チャネル) の問題、低ければ形 (段 1 が段 2 のどの形でも +1 をもらう報酬) の問題。
   - 6 手目の山がほかの崩れた run にもあるか (v13 s44 の it150・it300、v11 s44 の it200、比べる相手の v13 s43 の it300 で `--dump_obs`)。
   - 36 手の chain 評価 (v12 s43 it300、模型の予測 56%)。Play の手数の上書きが効くかを先に確かめる。
5. v15 (v14 + ロープの直径・長さの DR、[0028](decisions/0028-v15-rope-size-dr.md)。判断 1〜4 は決定済み): 長さ ×0.8/1.0/1.25 × 直径 ×0.4/0.63/1.0 の USD を全部作り、ゲートで通らない組を外す。critic に寸法 2 次元。実装は 0028 の「実装の順序」(skill `add-version` -> `physics-check` -> スモークテスト -> commit)。`MultiAssetSpawnerCfg` が Articulation で動くか、`replicate_physics=False` で 4096 env の起動時間と学習速度は未確認。細い組 (直径 ×0.4) はゲートで最初に見る。**IsaacLab の checkout の例をそのまま写さない。** 既存の USD と `outputs/archive_seed/` は上書きしない。不具合を直す版 (1) との順番はユーザーと決める。
6. v7 s44 を iteration 300 まで延ばす (比べる相手を 2 シードにする)。`uv run train --task Isaac-Twisted-RL-Knot-v7 --headless --device cuda:N --seed 44 --resume --load_run 2026-10-04_22-37-47_v7_s44 --checkpoint model_100.pt --max_iterations 201 --run_name v7_s44_to300 agent.device=cuda:N agent.save_interval=10 env.events.reset_to_stage.params.load_paths=[/home/itoyama/work/isaac_twisted_rl/logs/rsl_rl/knot_tying_v7/2026-10-04_22-37-47_v7_s44/archive.pt]` (`run_job.sh --timeout 20h --detach`)。GPU が空いてから、順番はユーザーに確かめる。
7. v11 s43 の延長の chain 評価を it400 / 500 / 600 / 700 / 800 で (it1000 は済み)。
8. 探索の設定を詰める (1 回に 1 つ変える): `--slots 8` / `--slots 4`、`--rank random`、`--no_mean_first`。v7 s43 / s44 の `model_100.pt`、それ以外は推論時探索 v7 の記録の `search` と同じ。
9. 学習のアルゴリズムの見直し (候補。どれも効くかは未確認): 離散 2 チャネル (`grasp_link_id`・`delta_link`) をカテゴリカルにする方策の頭、観測・価値の正規化 (`actor_obs_normalization=False` が v4 から続く)、探索の結果を方策へ蒸留する、段が上がった手を終端にせず次の段の critic の V で bootstrap する (段 1 に段 2 の結びやすさを伝える)。SAC 系は rsl-rl 3.0.1 に無く作り直しが大きい。どれを選ぶかは 4 の結果を見てから。
10. 摩擦を意図どおり (ロープ 2.0、指 4.0 の `max`) にする版 (実際は 0.5)。物理を変えるので skill `physics-check` から。[記録](experiments/2026-10-14_friction-binding.md)
11. `run_job.sh` は `bash -c "..."` の文字列の中の `--device cuda:N` を読まず、GPU 0 の空きを確かめる。`bash -c` で包むときは `--check-gpu N` を付ける。直すかは未定。
12. その他の候補 (未着手): std の下限、形ごとの成功率で優先して引く (Prioritized Level Replay 型、0025 の「見込みの弱点」)、v8 の切り分け、P2 の残り ([0022](decisions/0022-p2-one-version-each.md))、めり込み解消速度の中間値。

## 保留・再開した研究

| 研究 | 止めた所 | 再開するときの 1 番目 | 記録 |
|---|---|---|---|
| 推論時探索 (ビーム探索・best-of-N) | 再開中 (2026-10-05 にユーザーの指示)。v7 の rollouts / search / shapes は済み。探索の設定の比較は未着手。v6 s43 の rollouts / search、v6 s42・v5 の search は止めたまま (再開するかはユーザーの判断) | 「次の作業」8 | [2026-10-06_inference_search_v7](experiments/2026-10-06_inference_search_v7.md)、[2026-10-03_inference_search](experiments/2026-10-03_inference_search.md) §6 |
| eval_chain.py の提案 #4〜#6 (環境ごとの記録・diag の統合・複数チェックポイントの同時評価・初期状態の揺らぎ) | ユーザーが後回しにした (2026-10-15) | 二項の 2 ポイントを実測するなら #6 か、`--num_envs`・環境の間隔を変える評価。手の列は `diag_stage_moves.py --dump_obs` + `chain_moves.py sequences` で一部代わりになる | [2026-10-15_eval-chain-repro](experiments/2026-10-15_eval-chain-repro.md) |
| rope study (紐モデルの比較。打ち切り中、再開はユーザーの判断待ち) | 最終報告まで書いた。評価と実装は 2026-10-03 13:00 に打ち切り | XPBD の段 0 の方策の崩れの切り分け (結論を変えうる最大の未解決。時間切れの bootstrap の不具合も候補) | [2026-10-02_rope_models](experiments/2026-10-02_rope_models.md) §12 |

- rope study の発散の数値には、`rope_study/bench_physics.py:378` の数え方による過大の疑いがある (未確認。再開するときに確かめる)。
- 推論時探索の記録の「V8」は、2026-10-05 に作った v8 (対称性のデータ拡張) とは別に書かれたもので、中身の定義は無い。
