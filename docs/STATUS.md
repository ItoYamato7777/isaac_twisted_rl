# STATUS (現在地)

- 状態: 現行。上書きで更新する (追記しない。手順は skill `handoff`)
- 読むべき状況: セッションの始め。今どこにいて、次に何をするかを知るとき
- 最終更新: 2026-10-14

## 要点

- v7・v11・v12 のどれも chain 評価の山が it70〜100 にあり、その後に下がる。v12 (アーカイブのループを断った版) でも s44 が it70 の 58.8% から崩れた ([v12 の記録](experiments/2026-10-13_v12_archive-stage-up.md))。共通の原因の最有力の候補は、**方策の平均が行動の範囲 [−1, 1] の外へ出て張り付くこと** (相関のみ。[調査](experiments/2026-10-14_action-saturation.md))。
- **次は v13 の実装** (ユーザーの指示、2026-10-14): v12 + PPO の損失に平均の範囲外の罰則。仕様・根拠・実装の見通しは [0026](decisions/0026-v13-action-bounds-loss.md)。
- 走っているジョブは無い。v10・v9・v8 は v7 を上回らなかった。rope study は打ち切りのまま。

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
| v9 | chain 評価 済み (seed 43 は iteration 70 / 100、seed 44 は timeout で iteration 90 まで)。v7 を上回らなかった | [2026-10-05_v9_stage2-weight](experiments/2026-10-05_v9_stage2-weight.md) |
| v10 | chain 評価 済み (seed 43・44、iteration 70 / 90 / 100、GPU 1 枚に 1 本)。v7 を大きく下回った | [2026-10-06_v10_path-stages](experiments/2026-10-06_v10_path-stages.md)、[0023](decisions/0023-v10-path-stages.md) |
| v11 | chain 評価 済み (seed 43・44、iteration 70〜300、GPU 1 枚に 1 本)。iteration 300 で v7 を上回らなかった。s44 が段 0 で崩れた。s43 は iteration 1000 まで延ばした (chain 評価は it1000 だけ。値は [張り付きの調査](experiments/2026-10-14_action-saturation.md)) | [2026-10-07_v11_dead-end-stages](experiments/2026-10-07_v11_dead-end-stages.md)、[2026-10-09_v11_stage2-archive-loop](experiments/2026-10-09_v11_stage2-archive-loop.md)、[0024](decisions/0024-v11-dead-end-stages.md) |
| v12 | chain 評価 済み (seed 43・44、iteration 70〜300、GPU 1 枚に 1 本)。s43 は it100 と it300 がほぼ同じ、s44 は it70 の後に崩れた。アーカイブの p-data の偏り (0025 の予想 1) は未確認 | [2026-10-13_v12_archive-stage-up](experiments/2026-10-13_v12_archive-stage-up.md)、[0025](decisions/0025-archive-save-on-stage-up.md) |
| v13 | 実装した (2026-10-14、`BoundedPPO`・`bounds_loss_coef` 0.01)。学習は未実施 | [0026](decisions/0026-v13-action-bounds-loss.md)、[versions/v13](../src/isaac_twisted_rl/tasks/knot_tying/versions/v13/README.md) |

## 走っているジョブ

- なし (2026-10-14 21 時)。GPU 0・1 は他のユーザー (ohtsukatakeshi) とも共有している。他の人のジョブがあれば触らない。
- 完走の確認は `exit.txt` に `signal:` の行が無いことと最後のチェックポイント。300 iteration は 1 本約 16〜18 時間 (v12、GPU 1 枚に 1 本。s44 は 2026-10-13 15:17 -> 10-14 07:11)。

## 次の作業

### v13 (v12 + 方策の平均の範囲外の罰則。仕様は [0026](decisions/0026-v13-action-bounds-loss.md))

1. skill `add-version` で v13 を作る (継承元 v12)。0026 の「実装の見通し」の順に:
   - `kt/common/policies/bounded_ppo.py` に `BoundedPPO(PPO)` (引数 `bounds_loss_coef`、既定 0)。`update` の間だけ `policy.update_distribution` を包み、勾配を取る呼び出しで平均を「順方向は恒等・逆方向で `coef · ∂penalty/∂μ` を足す」関数に通す。`loss_dict["bounds"]` を足す。`on_policy_runner` に登録する。
   - 先に `tests/test_bounded_ppo.py` (罰則を直接書いた損失と勾配が一致する、|μ| ≤ 1 では変わらない、coef 0 で PPO と同じ、inference mode では何もしない)。
   - `versions/v13/` (env は v12 のまま、agent の `algorithm` を `class_name = "BoundedPPO"`・`bounds_loss_coef = 0.01` に) と `tests/test_v13_cfg.py` (`tests/test_v12_cfg.py` が手本)。
2. テスト・lint の後に commit し、skill `train-eval` のスモークテストで `params/agent.yaml` の `class_name: BoundedPPO`・`bounds_loss_coef`、TensorBoard の `Loss/bounds` を確かめる。
3. **学習の前に、係数 (推奨 0.01) と GPU の割り当てをユーザーに確かめる。** v13 seed 43 / 44 を 0 から iteration 300 まで (`--max_iterations 301`、`agent.save_interval=10`、timeout 30h、GPU 1 枚に 1 本)。
4. chain 評価 (seed 7、512 環境、Hydra の上書きなし) を it70 / 100 / 150 / 200 / 250 / 300 で、v12 と並べる。`action_saturation.py --obs outputs/diag_stage_moves/obs/*.pt --run v13_sS=<run>` で張り付きを数える。0026 の「外れたら仮説が誤りと言える予想」(段 0 の置き方のチャネルの張り付きが 20% を超えない、chain 評価が it70〜100 の後に下がらない、s44 の `timeout_s0` が 0.1 を超えない) を確かめる。

### v12 の残り

5. 0025 の予想 1 (段 2 のアーカイブで 1 つの p-data が 20% を超えない) を `archive_history/` の写しで数える。コマンドは [v12 の記録](experiments/2026-10-13_v12_archive-stage-up.md) の「次の作業」。

### v7・v11 の比べる相手

6. v7 s44 を iteration 300 まで延ばす (比べる相手を 2 シードにする)。`uv run train --task Isaac-Twisted-RL-Knot-v7 --headless --device cuda:N --seed 44 --resume --load_run 2026-10-04_22-37-47_v7_s44 --checkpoint model_100.pt --max_iterations 201 --run_name v7_s44_to300 agent.device=cuda:N agent.save_interval=10 env.events.reset_to_stage.params.load_paths=[/home/itoyama/work/isaac_twisted_rl/logs/rsl_rl/knot_tying_v7/2026-10-04_22-37-47_v7_s44/archive.pt]` (`run_job.sh --timeout 20h --detach`)。v13 と GPU を取り合うので、順番はユーザーに確かめる。
7. v11 s43 の延長の chain 評価を it400 / 500 / 600 / 700 / 800 で (it1000 は済み)。
8. v11 s44 の段 0 の崩れ: v12 s44 は段 0 のアーカイブが変わらないまま同じ型に崩れたので、アーカイブの汚れは原因ではない ([v12 の記録](experiments/2026-10-13_v12_archive-stage-up.md))。v13 の結果で張り付きが原因かを見てから、残りの切り分けをユーザーと決める。

### その他

9. **探索の設定を詰める** (1 回に 1 つ変える): `--slots 8` / `--slots 4`、`--rank random`、`--no_mean_first`。v7 s43 / s44 の `model_100.pt`、それ以外は推論時探索 v7 の記録の `search` と同じ。
10. その他の候補 (どれも未着手): std の下限 (0026 で v13 から外した。段 1 の最小の std がどの run も 0.03〜0.06)、形ごとの成功率で優先して引く (Prioritized Level Replay 型、0025 の「見込みの弱点」)、v8 の切り分け ([v8 の記録](experiments/2026-10-05_v8_symmetry.md))、P2 の残り ([0022](decisions/0022-p2-one-version-each.md))、めり込み解消速度の中間値 ([overnight の記録](experiments/2026-10-01_v4-v6_overnight.md))。

## 保留・再開した研究

| 研究 | 止めた所 | 再開するときの 1 番目 | 記録 |
|---|---|---|---|
| 推論時探索 (ビーム探索・best-of-N) | 再開中 (2026-10-05 にユーザーの指示)。v7 の rollouts / search / shapes は済み。探索の設定の比較は未着手。v6 s43 の rollouts / search、v6 s42・v5 の search は止めたまま (再開するかはユーザーの判断) | 上の「次の作業」9 | [2026-10-06_inference_search_v7](experiments/2026-10-06_inference_search_v7.md)、[2026-10-03_inference_search](experiments/2026-10-03_inference_search.md) §6 |
| rope study (紐モデルの比較。打ち切り中、再開はユーザーの判断待ち) | 最終報告まで書いた。評価と実装は 2026-10-03 13:00 に打ち切り | XPBD の段 0 の方策の崩れの切り分け (結論を変えうる最大の未解決) | [2026-10-02_rope_models](experiments/2026-10-02_rope_models.md) §12 |

- rope study の発散の数値には、`rope_study/bench_physics.py:378` の数え方による過大の疑いがある (未確認。再開するときに確かめる)。
- 推論時探索の記録の「V8」は、2026-10-05 に作った v8 (対称性のデータ拡張) とは別に書かれたもので、中身の定義は無い。v7 を土台に探索を足す次の版の意味と読める。
