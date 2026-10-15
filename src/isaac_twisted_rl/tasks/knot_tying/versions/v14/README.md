# v14: v13 + 段 0 の形を変えた手の保存 + ロープの物理の DR + critic の特権観測

```bash
uv run train --task Isaac-Twisted-RL-Knot-v14 --headless           # 段ごとの MLP + BoundedPPO + critic の特権観測
uv run play  --task Isaac-Twisted-RL-Knot-Play-v14 --num_envs 16   # 公称の物理で、まっすぐなロープから段をつないで評価
```

ログ: `logs/rsl_rl/knot_tying_v14/`。**学習には `Knot` プリセットを使う** (v3〜v13 と同じ)。継承元は **v13**。
仕様と理由の正本は [`docs/decisions/0027-v14-s0-reshape-dr-priv-critic.md`](../../../../../../docs/decisions/0027-v14-s0-reshape-dr-priv-critic.md)。ここには配線と、切り方と、見るべき指標を書く。
初期アーカイブは v13 と同じ (`overhand_{num_links}links.pt`)。

## なぜ作ったか

- v12・v13 の段 0 のアーカイブには学習中に何も入らない。段 0 から始まる形は、まっすぐなロープ (確率 0.5) と、台本の 128 形 (交差 0 個・関節角の平均絶対値 約 0.08 rad でほぼまっすぐ) だけ。交差が無くても様々な形から段 0 を練習させる。
- 物理の値の違い (剛性・質量・摩擦など) に強い方策にする。critic には引いた物理を教え、値の見積りの分散を下げる (actor は物理を知らないまま)。

1 版 1 変数から外れる (ユーザーが了承済み)。3 つとも Hydra の上書きで個別に切れるので、結果が悪いときだけ切り分けの run を回す。

## v13 から何を・なぜ変えたか

| 項目 | v13 | v14 | 切るとき |
|---|---|---|---|
| アーカイブの保存の規則 (`record_stage_archive.params.save_on`) | `stage_up` | **`stage_up_s0_reshape`**: `stage_up` の行に、段 0 の形を変えた手の後の形を足す | `env.rewards.record_stage_archive.params.save_on=stage_up` |
| 段 0 の形を変えた手の下限 (`s0_min_change`) | — | **0.02 m** (直径 1 本) | — |
| ロープの物理 (`events.rope_dr`、`mdp.RandomizeRopePhysics`) | 公称のまま | **reset のたびに env ごとに 6 軸を引く (対数一様。減衰比 ζ は ×1.0〜×2、ほかの 5 軸は ×0.5〜×2)** | `env.events.rope_dr.params.enabled=false` |
| critic の観測 (`agent.obs_groups.critic`) | `[policy]` (197) | **`[rope_params, policy]`** (6 + 197 = 203) | `agent.obs_groups.critic=[policy]` |
| Play | v13 | v13 + `rope_dr.params.enabled = False` (公称の物理) | DR を付けた評価は `env.events.rope_dr.params.enabled=true` |
| それ以外 (actor の観測・報酬・終了・行動・PPO・方策の構造・初期アーカイブ) | | v13 のまま | |

### 段 0 の形を変えた手

次をすべて満たす手 (`ResetToStage.record`):

- 床に落ちていて発散していない (`stage_up` と同じ `ok`)。
- 手の前後とも段 0、かつ手の前後とも交差 0 個 (R2 除去の前)。交差はあるが段 0 の形 (行き止まり) は入れない。v7 の段 0 の汚れはこの形だった。
- 形の変化 ≥ `s0_min_change`。変化は、手の前後のリンクの xy を 2 次元の剛体の位置合わせ (重心と回転) をした後のずれの RMS (`state_archive.rigid_aligned_rms`)。ロープ全体を引きずって回しただけの手は形を変えていない扱いになる。
- その env のこのエピソードで、まだ段 0 に保存していない (1 エピソード 1 回)。段 0 の方策の出力が段 0 のアーカイブに戻るので、v12 が断ったループを、入る量を絞って抑える。

### ロープの物理の DR

| 軸 | 掛ける先 | 公称値 (`knot`) |
|---|---|---|
| `bend_stiffness` | 曲げの DOF (`JointNN:1`・`:2`) のばね定数 | 0.019635 N·m/rad |
| `twist_stiffness` | 捩りの DOF (`JointNN:0`) のばね定数 | 0.0098175 N·m/rad |
| `damping_ratio` | 減衰 = 公称 × sqrt(s_k · s_m) × s_ζ (臨界減衰に対する比を独立に振る) | 曲げ 5.066e-4、捩り 1.405e-4 (ζ = 1) |
| `mass` | リンクの質量と慣性 (同じ倍率) | 0.010053 kg × 28 |
| `rope_friction` | ロープの 28 形状の静止・動摩擦 | **0.5** |
| `finger_friction` | 指の 2 形状の静止・動摩擦 | **0.5** |

- 摩擦の公称値はロープ・指とも 0.5 (PhysX の既定の材質)。USD と `rope_specs.py` の 2.0、`bind_grasp_material` の 4.0 は PhysX に届いていない ([2026-10-14_friction-binding](../../../../../../docs/experiments/2026-10-14_friction-binding.md))。倍率は起動時に PhysX から読んだ値に掛ける。
- 床は全 env で 1 枚なので振らない。ロープと床の摩擦は average 合成で (μ_rope + 2.0) / 2 (0.625〜1.5)。ロープとロープは μ_rope、ロープと指は (μ_rope + μ_finger) / 2。
- 握る力 (指のゲイン 1508 N/m) は公称の質量に合わせたまま固定。
- アーカイブには物理パラメータを残さない。ある物理で保存した形を別の物理の env に書き戻す (`fail_s{k}`・`restore_mismatch` で見張る)。
- 1 つの範囲を 1 点にすると、その軸は振らない (例: `env.events.rope_dr.params.ranges.mass=[1.0,1.0]`)。指の範囲が 1 点なら、指の材質には触れない。

### critic の特権観測

`rope_params` グループ (`mdp.rope_physics_params`) は、引いた倍率の log2 を範囲の中点と半幅で [−1, 1] に写した 6 次元。公称の物理 (DR を切ったとき) は、ζ 以外の 5 軸では 0、ζ では −1 (範囲の端)。
`StageGatedActorCritic` は critic の観測でも最後の 4 列 (段の one-hot) で頭を選ぶので、特権観測は前に置く。critic の幅が変わるので、v12・v13 のチェックポイントからは再開できない。
`eval_search.py` は状態を別の env に写すので、写し先の env の物理で V(s) を計算する (Play の既定では DR を切るので公称の物理)。

## 見るべき指標

版の比較は chain 評価で行う ([`docs/system/evaluation.md`](../../../../../../docs/system/evaluation.md))。比べる相手は v13 の同じ seed。学習ログは代理の指標。

* chain 評価 (公称の物理、Play の既定) の `knot` を iteration 70 / 100 / 150 / 200 / 250 / 300 で v13 と並べる。
* DR を付けた chain 評価 (`env.events.rope_dr.params.enabled=true`) を v14 と v13 に。v14 の方が公称からの下がり幅が小さいか。
* `Curriculum/stage_log/s0_reshape_saved`・`s0_change_mean`・`s0_change_frac`・`saved_s0`・`arc_size_s0`: 段 0 のアーカイブに交差の無い形が入っているか。
* `Curriculum/stage_log/timeout_s0`: 段 0 のループが起きていないか (0027 の予想では 0.1 を超えない)。
* `Curriculum/stage_log/restore_mismatch`・`save_stage_mismatch` (0 のはず)、`diverged_s{k}` (DR の端で発散が増えないか)。
