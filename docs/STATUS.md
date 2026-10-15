# STATUS (現在地)

- 状態: 現行。上書きで更新する (追記しない。手順は skill `handoff`)
- 読むべき状況: セッションの始め。今どこにいて、次に何をするかを知るとき
- 最終更新: 2026-10-15

## 要点

- **v14 seed 43 を GPU 1、seed 44 を GPU 0 で学習中** (s43: 2026-10-15 15:07 起動、s44: 16:07 起動。どちらも約 17 時間)。DR の減衰比 ζ の下限は ×1.0 に決めた (ユーザー、commit `d8dc8f7`)。
- v14 の結果が出るまで v15 の実装は後回し (ユーザーの判断、2026-10-15)。その間は GPU を使わない調べもの。決定 [0028](decisions/0028-v15-rope-size-dr.md) は決まった (長さ ×0.8/1.0/1.25 × 直径 ×0.4/0.63/1.0 の 3 × 3、実機は直径 8 mm)。
- **次は eval_chain.py の改善** (ユーザーの指示、2026-10-15): #0 再現性の確認 (同じ初期状態・決定論的方策で 512 エピソードが独立か) -> #1 段ごとの内訳・#2 評価中の行動の張り付き・#3 再現の情報と物理の照合。「次の作業」1〜4。v13 は chain 評価の it300 を待って実験記録を書く (「次の作業」5)。

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
| v12 | chain 評価 済み (seed 43・44、iteration 70〜300、GPU 1 枚に 1 本)。s43 は it100 と it300 がほぼ同じ、s44 は it70 の後に崩れた。段 2 のアーカイブは Φ の表にある 1 つの形に 3〜4 割寄った (0025 の予想 1 は外れ) | [2026-10-13_v12_archive-stage-up](experiments/2026-10-13_v12_archive-stage-up.md)、[0025](decisions/0025-archive-save-on-stage-up.md) |
| v13 | 学習は 2 本とも完走 (seed 43・44、0 -> iteration 300、GPU 1 枚に 1 本)。chain 評価は it70〜250 が済み、it300 を評価中。実験記録は未作成 (「次の作業」5) | [0026](decisions/0026-v13-action-bounds-loss.md)、[versions/v13](../src/isaac_twisted_rl/tasks/knot_tying/versions/v13/README.md) |
| v14 | 実装・ゲート・スモークテスト済み (2026-10-14)。**seed 43 (GPU 1、2026-10-15 15:07 起動)・seed 44 (GPU 0、16:07 起動) を学習中**。DR の ζ の範囲は ×1.0〜×2 (他の 5 軸は ×0.5〜×2) | [0027](decisions/0027-v14-s0-reshape-dr-priv-critic.md)、[ゲート](experiments/2026-10-14_v14_physics-gate.md)、[versions/v14](../src/isaac_twisted_rl/tasks/knot_tying/versions/v14/README.md) |
| v15 | 未着手 (v14 + ロープの直径・長さの DR)。決定 [0028](decisions/0028-v15-rope-size-dr.md) は決定済み。v14 の結果を見てから実装する | 「次の作業」8 |

## 走っているジョブ

| ジョブ | GPU | pid | job のフォルダ | run のフォルダ | 起動 |
|---|---|---|---|---|---|
| v14 s43 (0 -> it300) | 1 | 262619 (`run_job.sh` の timeout 30h) | `outputs/jobs/2026-10-15_15-07-05_v14_s43` | `logs/rsl_rl/knot_tying_v14/2026-10-15_15-07-10_v14_s43` | 2026-10-15 15:07 |
| v14 s44 (0 -> it300) | 0 | 344651 (`run_job.sh` の timeout 30h) | `outputs/jobs/2026-10-15_16-07-16_v14_s44` | `logs/rsl_rl/knot_tying_v14/` の `*_v14_s44` | 2026-10-15 16:07 |
| v13 s43 の chain 評価 (it70〜300) | 0 | 341125 (timeout 4h) | `outputs/jobs/2026-10-15_16-05-07_eval_v13_s43_it70-300` | `logs/rsl_rl/knot_tying_v13/2026-10-14_21-40-13_v13_s43` の `eval_chain_it*.json` | 2026-10-15 16:05 |
| v13 s44 の chain 評価 (it70〜300) | 0 | 341429 (timeout 4h) | `outputs/jobs/2026-10-15_16-05-12_eval_v13_s44_it70-300` | `logs/rsl_rl/knot_tying_v13/2026-10-14_21-40-15_v13_s44` の `eval_chain_it*.json` | 2026-10-15 16:05 |

- v13 s43 は完走した (`outputs/jobs/2026-10-14_21-40-08_v13_s43/exit.txt` は `exit: 0`、`signal:` の行なし、`model_300.pt` あり、2026-10-15 16:01)。
- v13 s44 (pid 3065345) は完走した。`outputs/jobs/2026-10-14_21-40-10_v13_s44/exit.txt` は `exit: 0`、`signal:` の行なし、`logs/rsl_rl/knot_tying_v13/2026-10-14_21-40-15_v13_s44/model_300.pt` がある。job.log の終わりに Kit の終了処理の `post_quit()` の TypeError が出ているが、学習の後の終了の処理でのもの。
- v14 の終わりの見込み: s43 は 2026-10-16 08:00〜09:00、s44 は 09:00〜10:00 (約 17 時間。GPU 0 は評価のジョブと共有するので s44 は遅れうる)。起動のコマンドは v13 と同じで、`--task Isaac-Twisted-RL-Knot-v14 --device cuda:1 --seed 43`・`agent.device=cuda:1`・`--run_name v14_s43`。commit `d8dc8f7`、Hydra の上書きは `agent.*` だけ (評価には何も付けない)。スモークテストは省いた (前日の v14 のスモークテストの後に変えたのは `DR_RANGES` の ζ の数値だけ。単体テストは通した)。
- コマンド (v13): `run_job.sh --name v13_sS --timeout 30h --detach -- uv run train --task Isaac-Twisted-RL-Knot-v13 --headless --device cuda:N --seed S --max_iterations 301 --run_name v13_sS agent.device=cuda:N agent.save_interval=10` (Hydra の上書きは `agent.*` だけ。評価には何も付けない)。commit `a709b49`、`bounds_loss_coef` 0.01 (ユーザーが確かめた)。
- v13 の chain 評価: 評価のコマンドは v12 と同じ (`eval_chain.py --task Isaac-Twisted-RL-Knot-Play-v13 --num_envs 512 --headless --device cuda:0 --seed 7 --checkpoint <run>/model_IT.pt agent.device=cuda:0 --out <run>/eval_chain_itIT.json`、IT = 70 / 100 / 150 / 200 / 250 / 300、Hydra の上書きなし)。2 本とも GPU 0 で v14 s44 と共有し、1 チェックポイントに約 23 分。it300 は 2026-10-15 18:30〜18:45 に終わる見込み。完走は `exit.txt` に `signal:` の行が無いことと `eval_chain_it300.json` で確かめる。
- **他のジョブ (自分が起動していない。触らない)**: `hiro_knot sim2real replay` (anaconda の python、別のセッションの実機移植の作業)。`dump_real_deploy_check.py` は別セッションが commit した (`ecc6254`)。
- GPU 0・1 は他のユーザー (ohtsukatakeshi) とも共有している。他の人のジョブがあれば触らない。

## 次の作業

### eval_chain.py の改善 (ユーザーの指示、2026-10-15。#0 -> #1〜#3 の順。環境ごとの記録・diag の統合・複数チェックポイントの同時評価・初期状態の揺らぎは、#0 の結果を見てからユーザーと決める)

背景 (コードから読めたこと。結果への影響は未確認): Play は全環境が `reset_scene_to_default` の同じまっすぐなロープから始まり (位置・向きの乱数は無い。`random_yaw` はアーカイブからの復元だけ)、段の重みは (1, 0, 0)、方策は決定論的。`common/mdp/` にアーカイブ以外の乱数は見当たらない。512 エピソードは独立な標本ではなく、結末が分かれるのは GPU の物理の細かな違いが接触で増幅されるためだけかもしれない。そうなら `--seed` は決定論的な評価に効かず、`docs/system/evaluation.md` §2 の「512 エピソードの標準誤差は 2 ポイント前後」の前提が崩れる。

1. **#0 再現性の確認 (コードは変えない)**:
   - GPU なしで先に: `outputs/diag_stage_moves/obs/v12_*_det.pt` から、各環境の最初の手の actor の観測が環境の間で同じか (最大の差) を見る。
   - 評価: v13 s44 の `model_70.pt` (記録では knot 65.4%) を seed 7 で 2 回・seed 8 で 1 回。v12 s43 の `model_300.pt` を seed 7 で 1 回 (記録の 39.8% と比べる。GPU の混み方が違う)。標準の条件 (512 環境、Hydra の上書きなし)。`--out` は `outputs/eval_repro/` に (run のフォルダの `eval_chain_it*.json` を上書きしない)。起動の前に `nvidia-smi` (v14 の 2 本が GPU 0・1 にいる)。
   - 見ること: 同じ seed の 2 回が一致するか、seed で変わるか、ばらつきの大きさ (二項の ±2 ポイントと比べて)。skill `experiment-record` で記録し、比べ方の規則 (何ポイントの差を信じるか) が変わるなら `evaluation.md` §2 を直す。
2. **#1 段ごとの内訳** (新しいキーを足すだけ): 結末 (固結び / 時間切れ / 発散 / 段の下降) × 終わったときの段の表、各段に初めて着いた手、段ごとに使った手数、条件付きの成功率 P(段 2 以上 | 段 1 以上)・P(固結び | 段 2 以上)、`knot` の Wilson 95% 区間。v13 s44 のような崩れ (段 0 の時間切れ) を、`diag_stage_moves.py` を別に回さずに見るため。
3. **#2 評価の rollout での行動の張り付き**: 段 × チャネルごとに、クリップ前の行動で \|a\| > 1 の割合と、Σ max(\|a\| − 1, 0)² の平均 (BoundedPPO の罰則と同じ式)。各環境の最初のエピソードの手だけを数える。`action_saturation.py` は v12 の rollout の観測を使い回すので分布がずれる、という制約を無くすため (v13 の予想 1 の確かめ方)。
4. **#3 再現の情報と物理の照合**: JSON に seed・Hydra の上書き・`num_envs`・`steps`・commit (と未コミットの差分の有無)・日時を入れる。チェックポイントの run の `params/env.yaml` と評価の環境の設定を比べ、Play が意図して変える項 (`play_post_init` の中身・`scene.num_envs`・`seed`・`log_dir` など) 以外に違いがあれば止める (`--allow_param_diff` で続ける)。学習の Hydra の上書きを評価に付け忘れると、違う物理のままエラーが出ずに測ってしまうため。
   - 1〜4 の共通の約束: 集計は isaaclab を import しない純粋な関数に分け (`tests/conftest.py` の経路で読める置き場所)、単体テストを書く。**既存のキーと既定の挙動を変えない** (`summarize_search.py` と過去の `eval_chain_it*.json` が前提にしている)。`diag_stage_moves.py` は変えない。修正の後、#0 と同じチェックポイント・条件で既存のキーの値が修正の前と同じ (#0 で決定論的と分かった場合) か、#0 のばらつきの範囲に入ることを確かめる。`docs/system/evaluation.md` の出力の表・§1.1 と `scripts/rsl_rl/README.md` を更新する。テスト・lint の後に commit。

### v13 の結果 (仕様は [0026](decisions/0026-v13-action-bounds-loss.md))

5. **v13 の実験記録を書く** (skill `experiment-record`)。chain 評価の it300 が終わってから (上の「走っているジョブ」)、v12 の同じ seed と並べる。材料: 学習ログの表と固定した観測での罰則は `outputs/v13_analysis/` (README.txt に作り方)、段 0 のチャネルごとの張り付きは `outputs/action_saturation/v13.json`。0026 の予想 1 (段 0 の置き方のチャネルが 20% を超えない) は固定した観測で 2 シードとも外れ、予想 3 (s44 の `timeout_s0` が 0.1 を超えない) は学習ログで外れた。予想 2 (it300 が it70・it100 の高い方から 5 ポイント以上は下がらない) は chain 評価で判定する。0026 の取り決めでは、予想 1 が外れたので張り付きが崩れの原因かの結論は保留 (係数の見直し)。評価中の張り付きは #2 ができたら数え直せる。

### v14 の学習と評価 (仕様は [0027](decisions/0027-v14-s0-reshape-dr-priv-critic.md)、ゲートは [2026-10-14_v14_physics-gate](experiments/2026-10-14_v14_physics-gate.md))

6. **v14 の学習の経過を確かめる**: 最初の数 iteration のログ (`Loss/bounds`・`arc_size_s0` の増加・`timeout_s0`・`diverged_s0`) が正常か。s43 / s44 とも `exit.txt` に `signal:` の行が無いことと `model_300.pt` で完走を確かめる。GPU 0 は評価のジョブと共有している。
7. v14 の chain 評価 (公称の物理、Play の既定) を it70 / 100 / 150 / 200 / 250 / 300 で v13 と並べる。DR を付けた評価 (`env.events.rope_dr.params.enabled=true`) を v14 と v13 に。0027 の予想 (段 0 のアーカイブに交差の無い形が溜まり `timeout_s0` が 0.1 を超えない・公称で v13 を 5 ポイント以上下回らない・DR を付けた評価の下がり幅が v13 より小さい) を確かめる。悪いときだけ切り分けの run (DR を切る・特権観測を切る・段 0 の保存を `stage_up` に戻す)。

### v15 (v14 + ロープの直径・長さの DR。v14 の結果を見てから。ユーザーの判断、2026-10-15)

8. **判断 1〜4 は決まった** ([0028](decisions/0028-v15-rope-size-dr.md)、ユーザー 2026-10-15): 長さ ×0.8/1.0/1.25 × 直径 ×0.4/0.63/1.0 の 3 × 3 の USD を全部作り、ゲートで通らない組を外す。観測は正規化しない。critic に寸法 2 次元 (205 次元)。指のゲインは固定。実機の紐は直径 8 mm・全長 1.02 m (長さ ×0.895、直径 ×0.4)。v14 の結果が悪ければ、継承元を v13 にする・範囲を狭めることも考える。
9. 実装 (0028 の「実装の順序」)。skill `add-version` -> `physics-check` -> スモークテスト -> commit。Isaac Lab 2.3.0 の `MultiAssetSpawnerCfg` が Articulation で動くか、`replicate_physics=False` で 4096 env の起動時間と学習速度がどうなるかは未確認 (Isaac Sim 上の小さなスクリプトで先に確かめる)。アーカイブの復元は寸法の倍率で割り戻す案 (0028)。細い組 (直径 ×0.4) はリンクの慣性が小さく暴走した前例があるので、ゲートで最初に見る。**IsaacLab の checkout の例をそのまま写さない。** 既存の USD (`common/rope_model/data/`) と `outputs/archive_seed/` は上書きしない。

### v7・v11 の比べる相手

10. v7 s44 を iteration 300 まで延ばす (比べる相手を 2 シードにする)。`uv run train --task Isaac-Twisted-RL-Knot-v7 --headless --device cuda:N --seed 44 --resume --load_run 2026-10-04_22-37-47_v7_s44 --checkpoint model_100.pt --max_iterations 201 --run_name v7_s44_to300 agent.device=cuda:N agent.save_interval=10 env.events.reset_to_stage.params.load_paths=[/home/itoyama/work/isaac_twisted_rl/logs/rsl_rl/knot_tying_v7/2026-10-04_22-37-47_v7_s44/archive.pt]` (`run_job.sh --timeout 20h --detach`)。v13 の 2 本は終わったが、GPU 0 は v14 s44 が使っている。起動は GPU が空いてからで、順番はユーザーに確かめる。
11. v11 s43 の延長の chain 評価を it400 / 500 / 600 / 700 / 800 で (it1000 は済み)。
12. v11 s44 の段 0 の崩れ: v12 s44 は段 0 のアーカイブが変わらないまま同じ型に崩れたので、アーカイブの汚れは原因ではない ([v12 の記録](experiments/2026-10-13_v12_archive-stage-up.md))。v13 の結果で張り付きが原因かを見てから、残りの切り分けをユーザーと決める。

### その他

13. **探索の設定を詰める** (1 回に 1 つ変える): `--slots 8` / `--slots 4`、`--rank random`、`--no_mean_first`。v7 s43 / s44 の `model_100.pt`、それ以外は推論時探索 v7 の記録の `search` と同じ。
14. 摩擦を意図どおり (ロープ 2.0、指 4.0 の `max`) にする版 (USD の材質を `Material` の prim に、指の bind を scene の読み込みの前に)。物理を変えるので skill `physics-check` から。[記録](experiments/2026-10-14_friction-binding.md)
15. 学習のアルゴリズムの見直し (2026-10-15 にユーザーと話した。決定ではなく候補): 離散 2 チャネル (`grasp_link_id`・`delta_link`) をカテゴリカルにする方策の頭 (0002: 今は全チャネルを連続で受けて環境側で丸める。張り付きの調査で grasp・Δlink は崩れない run でも 76〜92% が範囲の外)、観測の正規化 (`actor_obs_normalization=False` が v4 から続く)・価値の正規化、探索の結果を方策へ蒸留する (state_clone と推論時探索がある)、SAC 系 (off-policy。0006 は PPO を使う限りの決定。rsl-rl 3.0.1 に SAC は無く、段ごとの方策・アーカイブ・BoundedPPO の作り直しが要る)。「Sim は PPO、実機は SAC」は、PPO の critic が V(s) で Q が無いこと、実機ではアーカイブからの reset ができないことから、そのままでは勧めない。どれも効くかは未確認 (推測)。
16. その他の候補 (どれも未着手): std の下限 (0026 で v13 から外した。段 1 の最小の std がどの run も 0.03〜0.06)、形ごとの成功率で優先して引く (Prioritized Level Replay 型、0025 の「見込みの弱点」)、v8 の切り分け ([v8 の記録](experiments/2026-10-05_v8_symmetry.md))、P2 の残り ([0022](decisions/0022-p2-one-version-each.md))、めり込み解消速度の中間値 ([overnight の記録](experiments/2026-10-01_v4-v6_overnight.md))。

## 保留・再開した研究

| 研究 | 止めた所 | 再開するときの 1 番目 | 記録 |
|---|---|---|---|
| 推論時探索 (ビーム探索・best-of-N) | 再開中 (2026-10-05 にユーザーの指示)。v7 の rollouts / search / shapes は済み。探索の設定の比較は未着手。v6 s43 の rollouts / search、v6 s42・v5 の search は止めたまま (再開するかはユーザーの判断) | 上の「次の作業」13 | [2026-10-06_inference_search_v7](experiments/2026-10-06_inference_search_v7.md)、[2026-10-03_inference_search](experiments/2026-10-03_inference_search.md) §6 |
| rope study (紐モデルの比較。打ち切り中、再開はユーザーの判断待ち) | 最終報告まで書いた。評価と実装は 2026-10-03 13:00 に打ち切り | XPBD の段 0 の方策の崩れの切り分け (結論を変えうる最大の未解決) | [2026-10-02_rope_models](experiments/2026-10-02_rope_models.md) §12 |

- rope study の発散の数値には、`rope_study/bench_physics.py:378` の数え方による過大の疑いがある (未確認。再開するときに確かめる)。
- 推論時探索の記録の「V8」は、2026-10-05 に作った v8 (対称性のデータ拡張) とは別に書かれたもので、中身の定義は無い。v7 を土台に探索を足す次の版の意味と読める。
