# STATUS (現在地)

- 状態: 現行。上書きで更新する (追記しない。手順は skill `handoff`)
- 読むべき状況: セッションの始め。今どこにいて、次に何をするかを知るとき
- 最終更新: 2026-10-20 (v17 の学習の前の確認を記録し、学習と s44 の chain 評価を起動した)

## 要点

- **v16 (v15 + ロープの長さの DR) の学習と chain 評価が済んだ** ([記録](experiments/2026-10-18_v16_length-dr.md))。崩れず、最後の点の公称の `knot` は v15 と同じ。中盤 (it150〜250) は両シードとも v15 より低い。寸法を振った評価と公称の差は向きが一定しない。
- **v17 (v15 + 待ちのフェーズの尺を詰める) を実装した** (継承元はユーザーが v15 に決めた。理由は [0030](decisions/0030-v17-shorter-wait-phases.md))。Play は 18 手の上限が浮動小数の誤差で 19 になるので 17.5 手ぶんにした。v17 の方策を実機で使うには hiro_knot の尺の定数の更新が要る ([README](../src/isaac_twisted_rl/tasks/knot_tying/versions/v17/README.md))。学習の前の確認は通った ([記録](experiments/2026-10-19_v17_gate.md)。1 手の時間は v15 の 0.766 倍、発散 3.76%)。seed 43・44 を学習し、s44 (済み) の chain 評価を実行中。
- **長さの DR (v16) の効果を判定した** ([記録](experiments/2026-10-19_v15-sized-eval_length-dr-effect.md)): 寸法を振ったロープでの `knot` は v15 と同じ。公称からの落ちは v15 の 1 シード (s43) だけが大きく、v16 は落ちない。長さの DR を後の版に入れるかはユーザーと決める。

## 版の状態

数字は各記録にある。比べ方は [`evaluation.md`](system/evaluation.md)。全記録に効く注意 (摩擦 0.5・不具合) は [実験記録の一覧](experiments/README.md)。

| 版 | 状態 | 記録 |
|---|---|---|
| v0〜v2-R2 | 古い版 (固結びは学べなかった)。v2-R2 は修正の後の本番の学習が未実施 | [一覧](experiments/README.md) の「v0〜v6」 |
| v3 | 本番の run 1 本 (初めて固結びを学べた) | [2026-09-28_v3_main](experiments/2026-09-28_v3_main.md) |
| v4〜v6 | chain 評価 済み (v4・v5 は seed 42 だけ、v6 は 42・43) | [2026-10-01_v4-v6_overnight](experiments/2026-10-01_v4-v6_overnight.md) |
| v7 | chain 評価 済み (seed 43・44 は it100 まで、s43 は延長で it330 まで)。**比べる相手の基準** | [2026-10-04_v7_seeds](experiments/2026-10-04_v7_seeds.md) |
| v8 / v9 / v10 | chain 評価 済み (seed 43・44)。どれも v7 を上回らなかった | [v8](experiments/2026-10-05_v8_symmetry.md)・[v9](experiments/2026-10-05_v9_stage2-weight.md)・[v10](experiments/2026-10-06_v10_path-stages.md) |
| v11 | chain 評価 済み (seed 43・44、it70〜300。s43 は it1000 まで延長)。v7 を上回らず、s44 が段 0 で崩れた | [v11](experiments/2026-10-07_v11_dead-end-stages.md)・[段 2 のループ](experiments/2026-10-09_v11_stage2-archive-loop.md) |
| v12 | chain 評価 済み (seed 43・44、it70〜300)。s43 は横ばい、s44 は段 0 から崩れた | [v12](experiments/2026-10-13_v12_archive-stage-up.md) |
| v13 | chain 評価 済み (seed 43・44、it70〜300)。罰則は張り付きの割合を減らさず、s44 は v12 s44 と同じ型に崩れた | [v13](experiments/2026-10-14_v13_bounds-loss.md) |
| v14 | chain 評価 済み (seed 43・44、it70〜300、公称と DR 付き)。2 シードとも崩れず、it100 以降 v13 を大きく上回った。記録のある it300 の値で最も高い | [v14](experiments/2026-10-15_v14_s0-reshape-dr-priv-critic.md)・[0027](decisions/0027-v14-s0-reshape-dr-priv-critic.md)・[ゲート](experiments/2026-10-14_v14_physics-gate.md) |
| v15 | chain 評価 済み (seed 43・44、it70〜300、公称)。崩れず V は膨らまない。it200〜300 の 2 シード平均は v14 より高いが s43 に集中、it70・100 は低い。DR 付きの評価は未実施 | [記録](experiments/2026-10-16_v15_timeout-bootstrap-fix.md)・[ゲート](experiments/2026-10-16_v15_timeout-gate.md)・[0028](decisions/0028-v15-timeout-bootstrap-fix.md) |
| v16 | chain 評価 済み (seed 43・44、it70〜299、公称と寸法を振った評価)。崩れず、最後の点は v15 と同じ、中盤は v15 より低い。寸法を振ったロープでは v15 と同じ成績で、寸法による差は v15 (1 シード) より小さい (v15 + ロープの長さの DR ×0.8/1.0/1.1) | [記録](experiments/2026-10-18_v16_length-dr.md)・[DR の効果](experiments/2026-10-19_v15-sized-eval_length-dr-effect.md)・[0029](decisions/0029-v16-rope-size-dr.md)・[ゲート (長さ)](experiments/2026-10-17_v16_length-gate.md)・[ゲート (9 組)](experiments/2026-10-17_v16_size-gate.md)・[spawn](experiments/2026-10-17_v16_rope-size-spawn.md)・[README](../src/isaac_twisted_rl/tasks/knot_tying/versions/v16/README.md) |
| v17 | **学習中** (v15 + 待ちのフェーズの尺を詰める。1 手 759 → 591 ステップ、bench で v15 の 0.766 倍の時間。物理は変えない。s44 は学習済みで chain 評価中、s43 は学習中) | [0030](decisions/0030-v17-shorter-wait-phases.md)・[計測](experiments/2026-10-16_physics-speed.md)・[ゲート](experiments/2026-10-19_v17_gate.md)・[README](../src/isaac_twisted_rl/tasks/knot_tying/versions/v17/README.md) |

## 走っているジョブ

- **v17 の学習** (2026-10-20 01:11 起動、commit `3d8561e`、`--max_iterations 301`、Hydra の上書きなし = v15 と同じコマンド): s43 は GPU 0 で実行中 (`outputs/jobs/2026-10-20_01-11-30_v17_s43`、pid 2641527)。s44 は 18:35 に終了 (exit 0、`signal:` なし、`model_300.pt` まで)。run は `logs/rsl_rl/knot_tying_v17/2026-10-20_01-11-*_v17_s{43,44}`。
- **v17 s44 の chain 評価** (GPU 1、`outputs/jobs/2026-10-20_18-50-35_eval_v17_s44`、pid 3663637): it70・100・150・200・250・300 を順に `--video` 付きで (`--task Isaac-Twisted-RL-Knot-Play-v17 --num_envs 512 --seed 7`、Hydra の上書きなし)。出力は run の `eval_chain_it*.json`。
- 学習の前の確認は記録済み ([記録](experiments/2026-10-19_v17_gate.md))。
- 使わない run: `logs/rsl_rl/knot_tying_v16/2026-10-18_01-45-40_v16_s{43,44}` (`model_0.pt` だけ。`_invalid/` に退避するかはユーザーに確かめる)。
- `.gitignore` の未コミットの変更 (動画の `manifest.json` も追跡しない) はユーザーのもの (触らない)。

## 次の作業

1. **v17 の実装** (継承元 v15。仕様・手順・予想は [0030](decisions/0030-v17-shorter-wait-phases.md)、skill `add-version`)。
   - 済み (GPU 不要): `versions/v17/`・登録と一覧・`tests/test_v17_cfg.py`・README の hiro_knot 側の更新の要否 (要る: `constants.DURATIONS` の直書きと `check_env_yaml`)。
   - 済み (GPU): 学習の前の確認 ([記録](experiments/2026-10-19_v17_gate.md))。
   - **次**: s44 の chain 評価 (実行中) と s43 の学習の終わりを待ち、s43 も it70〜300 を `--video` 付きで評価する。v15 の同じシード (`logs/rsl_rl/knot_tying_v15/2026-10-16_17-07-55_v15_s{43,44}` の `eval_chain_it*.json`) と比べて記録する (0030 の予想 1・3・4。予想 4 は学習ログの `restore_mismatch`・`save_stage_mismatch`・`Episode_Termination/diverged`)。
   - 長さの DR を v17 の後の版に入れるか (ユーザーの判断)。材料は [DR の効果の記録](experiments/2026-10-19_v15-sized-eval_length-dr-effect.md) と [v16 の記録](experiments/2026-10-18_v16_length-dr.md) (利点: 寸法による成績の差が小さい (1 シードの根拠)。コスト: 中盤の公称の `knot` が低い・学習が約 5% 遅い)。組ごとの内訳が要るなら下の 3。
   - v16 の動画 (24 個、`video/sim/knot_tying_v16/`) は未確認 (ユーザーが判断する)。
2. **chain 評価の動画 (実装済み。評価は `--video` 付きの 1 回だけ。動画なしの 2 回目は回さない)**。0031 のゲート 4 は v15 s44 の it300 で通った (動画ありなしで 23 キー一致)。**動画は撮るが、中身は Claude が見ない・判断しない (ユーザーが判断する。`CLAUDE.md`・0031 の決定 3)**。v15 s44 の it300 の動画はあるが未確認 (`video/sim/knot_tying_v15/2026-10-16_17-07-55_v15_s44/eval_chain_it300_video/`)。ルート直下の `knot_*_mock.mp4` と `README.md` の変更はユーザーのもの (触らない)。
3. DR 付きの評価・寸法を振った評価の内訳を見るなら、`eval_chain.py` に env ごとの倍率と寸法の記録を足す (既定の評価とビット単位で一致することを確かめる)。
4. 段 0・段 1 から 1 手で固結びに達したエピソード (v14 s44 it100 で 2%・8%) が正しい判定かを、その手の前後の形で確かめる (優先度は低い)。
5. **頭打ち (段 2) の切り分けの計測** (GPU を使う前にユーザーに確かめる。コマンドの案は[記録](experiments/2026-10-15_timeout-bootstrap-and-stage2-bottleneck.md)の「次の作業」): 段 2 の形ごとの最良の 1 手 (`eval_search.py --mode shapes --shape_moves 1`)、6 手目の山がほかの崩れた run にもあるか、36 手の chain 評価。
6. 速さの次の候補 (v17 の結果を見てから): 反復 6 (Hydra の上書きで学習できる。v17 とは別の学習で確かめる)、環境数 (2048・8192) を変えたときの速さ (GPU が空いたとき)。v17 の実装は 1。
7. v7 s44 を it300 まで延ばす (比べる相手を 2 シードにする。GPU が空いてから、順番はユーザーに確かめる): `uv run train --task Isaac-Twisted-RL-Knot-v7 --headless --device cuda:N --seed 44 --resume --load_run 2026-10-04_22-37-47_v7_s44 --checkpoint model_100.pt --max_iterations 201 --run_name v7_s44_to300 agent.device=cuda:N agent.save_interval=10 env.events.reset_to_stage.params.load_paths=[/home/itoyama/work/isaac_twisted_rl/logs/rsl_rl/knot_tying_v7/2026-10-04_22-37-47_v7_s44/archive.pt]` (`run_job.sh --timeout 20h --detach`)。
8. v11 s43 の延長の chain 評価を it400 / 500 / 600 / 700 / 800 で (it1000 は済み)。
9. 探索の設定を 1 つずつ比べる (`--slots 8` / `4`、`--rank random`、`--no_mean_first`。v7 s43 / s44 の `model_100.pt`)。
10. 学習のアルゴリズムの見直しの候補 (効くかは未確認。5 の結果を見てから選ぶ): 離散 2 チャネルをカテゴリカルにする、観測・価値の正規化、探索の結果の蒸留、段が上がった手を次の段の V で bootstrap する。
11. 摩擦を意図どおり (ロープ 2.0、指 4.0 の `max`) にする版。skill `physics-check` から ([記録](experiments/2026-10-14_friction-binding.md))。
12. `run_job.sh` は `bash -c "..."` の中の `--device cuda:N` を読まない。包むときは `--check-gpu N` を付ける (直すかは未定)。
13. その他の候補: std の下限、形ごとの成功率で優先して引く (0025 の「見込みの弱点」)、v8 の切り分け、P2 の残り ([0022](decisions/0022-p2-one-version-each.md))、めり込み解消速度の中間値。

## 保留

| 研究 | 状態・誰の判断待ちか | 再開するときの 1 番目 | 記録 |
|---|---|---|---|
| 推論時探索 | 再開中 (2026-10-05、ユーザーの指示)。v6・v5 の止めた評価を再開するかはユーザーの判断 | 「次の作業」9 | [v7](experiments/2026-10-06_inference_search_v7.md)・[中間](experiments/2026-10-03_inference_search.md) |
| eval_chain.py の提案 #4〜#6 | ユーザーが後回しにした (2026-10-15) | 二項の揺れを実測するなら #6 か `--num_envs` を変える評価 | [記録](experiments/2026-10-15_eval-chain-repro.md) |
| 評価の初期状態を交差なしの様々な形へ広げる | ユーザーの指示待ち (まっすぐな状態の成功率が高くなるか、直す内容が尽きたとき。Claude は先に始めない)。決定は [0032](decisions/0032-sim-only-sim2real-goal.md) | ユーザーの指示を受けたら 0032 の「広げるときに決めること」から | [0032](decisions/0032-sim-only-sim2real-goal.md) |
| rope study | 打ち切り中 (2026-10-03)。再開はユーザーの判断 | XPBD の段 0 の崩れの切り分け (不具合も候補)。発散の数値の数え方の過大の疑いも確かめる | [要約](experiments/2026-10-02_rope_models.md) |
