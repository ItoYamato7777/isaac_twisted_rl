# STATUS (現在地)

- 状態: 現行。上書きで更新する (追記しない。手順は skill `handoff`)
- 読むべき状況: セッションの始め。今どこにいて、次に何をするかを知るとき
- 最終更新: 2026-10-17 (v16 を実装し、寸法の組ごとのゲートを測った。細い組が通らず、ユーザーの判断待ち)

## 要点

- **v16 (v15 + ロープの直径・長さの DR) を実装した (commit `606887f` まで、未学習)**。spawn・env ごとの寸法・アーカイブの割り戻しは動く ([spawn の記録](experiments/2026-10-17_v16_rope-size-spawn.md))。
- **寸法の組ごとのゲートで、台本が固結びを結べるのは直径 ×1.0 の 3 組だけ**。直径 ×0.4 (実機の 8 mm) は掴めず、直径 ×0.63 は発散が多い。寸法を混ぜると公称の組の発散も 2.8% -> 7.3% (アーカイブの書き戻しと推測) ([ゲートの記録](experiments/2026-10-17_v16_size-gate.md))。
- **次: 寸法の組をどうするか・アーカイブの書き戻しをどう直すかをユーザーが決める** (0029 の判断 1: 実機に近い組が残らなければ相談)。学習はその後。

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
| v16 | **実装済み・未学習。ゲートで細い組が通らず、組の決め直しがユーザーの判断待ち** (v15 + ロープの直径・長さの DR。継承元は v15: ユーザー、2026-10-17) | [0029](decisions/0029-v16-rope-size-dr.md)・[spawn](experiments/2026-10-17_v16_rope-size-spawn.md)・[ゲート](experiments/2026-10-17_v16_size-gate.md)・[README](../src/isaac_twisted_rl/tasks/knot_tying/versions/v16/README.md) |
| v17 | **実装の予約のみ** (継承元 + 待ちのフェーズの尺を詰める。1 手 759 → 591 ステップで 1.28 倍の見込み。物理は変えない。継承元は未確認: v16 と同じ v15 にするかはユーザーに確かめる) | [0030](decisions/0030-v17-shorter-wait-phases.md)・[計測](experiments/2026-10-16_physics-speed.md) |

## 走っているジョブ

なし (自分のジョブ)。v16 のゲートの job はすべて終わった (`outputs/jobs/2026-10-17_*`)。

- **他のジョブ (触らない)**: anaconda の python の実機の作業 (pid 560983・616313・895144・2892930・876400)。GPU 0・1 は他のユーザーとも共有している。

## 次の作業

1. **v16 の寸法の組とアーカイブの書き戻しを決める (ユーザーの判断待ち)**。数値は [ゲートの記録](experiments/2026-10-17_v16_size-gate.md)。案:
   - 組: (A) 長さだけを振る (×0.8/1.0/1.25、直径 ×1.0。3 組とも台本が結べる)、(B) 直径の範囲を ×0.8〜1.0 などに狭めて USD を足し、測り直す、(C) 細い紐はリンク数を増やす別の版にする (観測・行動の範囲が変わる)。直径 ×0.4 は指のゲインを下げても結べなかった (2/128)。
   - アーカイブ: 寸法の間で書き戻すと関節角が書き戻し先の角度制限を超える (推測)。(a) 寸法の組ごとにアーカイブを分ける、(b) 書き戻し先の関節の制限に切り詰める、(c) 角度制限が同じ組だけにする。初期アーカイブ (公称) も ×0.8 の組 (±47°) の制限を超えうる。
   - 決まったら: `versions/v16/env_cfg.py` の `ROPE_SIZE_SCALES` と書き戻しを直す -> 残す組だけでゲート (発散・把持) を測り直す -> スモークテスト (`uv run train --task Isaac-Twisted-RL-Knot-v16 ... --max_iterations 2`) -> 学習 (seed 43 / 44、it300。GPU を使う前にユーザーに確かめる) -> chain 評価 (公称の寸法、v15 の同じ seed と比べる。寸法を振った評価は `env.scene.rope.spawn.scales=...`)。
   - 学習のログで `PhysX error` (aggregate pairs のあふれ) が出ていないかを確かめる (replicate なしで上限を上げてある。[spawn の記録](experiments/2026-10-17_v16_rope-size-spawn.md))。v16 は起動が約 20 s 長く、ランダムな手で 1 手の時間は v15 の 77% の速さ。

2. **chain 評価の動画 (実装済み。評価は `--video` 付きの 1 回だけ。動画なしの 2 回目は回さない)**。0031 のゲート 4 は v15 s44 の it300 で通った (動画ありなしで 23 キー一致)。**動画は撮るが、中身は Claude が見ない・判断しない (ユーザーが判断する。`CLAUDE.md`・0031 の決定 3)**。v15 s44 の it300 の動画はあるが未確認 (`video/sim/knot_tying_v15/2026-10-16_17-07-55_v15_s44/eval_chain_it300_video/`)。ルート直下の `knot_*_mock.mp4` と `README.md` の変更はユーザーのもの (触らない)。
3. DR 付きの評価の内訳を見るなら、`eval_chain.py` に env ごとの倍率の記録を足す (既定の評価とビット単位で一致することを確かめる)。
4. 段 0・段 1 から 1 手で固結びに達したエピソード (v14 s44 it100 で 2%・8%) が正しい判定かを、その手の前後の形で確かめる (優先度は低い)。
5. **頭打ち (段 2) の切り分けの計測** (GPU を使う前にユーザーに確かめる。コマンドの案は[記録](experiments/2026-10-15_timeout-bootstrap-and-stage2-bottleneck.md)の「次の作業」): 段 2 の形ごとの最良の 1 手 (`eval_search.py --mode shapes --shape_moves 1`)、6 手目の山がほかの崩れた run にもあるか、36 手の chain 評価。
6. **v17 の実装 (予約。待ちのフェーズの尺を詰めて学習を 1.28 倍に速くする版)**。仕様・予想・手順は [0030](decisions/0030-v17-shorter-wait-phases.md)、根拠の計測は [記録](experiments/2026-10-16_physics-speed.md)。コードの実装・テストは先に進めてよい (版は `durations` を上書きするだけ。継承元は未確認 (v16 と同じ v15 にするかをユーザーに確かめる))。GPU は v15 の学習が終わったので空いているが、共有なので `nvidia-smi` を見てから。最初に確かめること: `bench_step_time.py --task Isaac-Twisted-RL-Knot-v17` の `meta` が `decimation` 591・`max_episode_length` 6 (Play は 18) か (積の浮動小数の誤差で 1 手増える事故)。速さの次の候補: 反復 6 (Hydra の上書きで学習できる。D4 とは別の学習で確かめる)、環境数 (2048・8192) を変えたときの速さ (GPU が空いたとき)。
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
| rope study | 打ち切り中 (2026-10-03)。再開はユーザーの判断 | XPBD の段 0 の崩れの切り分け (不具合も候補)。発散の数値の数え方の過大の疑いも確かめる | [要約](experiments/2026-10-02_rope_models.md) |
