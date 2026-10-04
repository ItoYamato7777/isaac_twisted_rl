# 0020 v4 以降はソルバの位置反復を 8、1 iteration を 4 手にする (環境数は 4096 のまま)

- 状態: 有効 (v4〜v7。v5〜v7 は v4 の設定を継承する)
- 読むべき状況: 学習を速くしたいとき、PhysX のソルバの位置反復・`num_steps_per_env`・環境数・チェックポイントやアーカイブの保存間隔を変えたくなったとき
- 最終更新: 2026-10-04

## 決定

- ロープとグリッパーの PhysX ソルバの位置反復を **16 → 8** にする (`KnotTyingV4EnvCfg.solver_position_iterations`)。
- PPO の 1 iteration で各環境が打つ手の数 `num_steps_per_env` を **12 → 4** にする。環境数は 4096 のまま減らさない。
- 総手数を v3 の既定と揃えるため `max_iterations` を 2000 → 6000 (= 2000 × 12 / 4) にし、アーカイブを書く間隔をチェックポイント (50 iteration) と揃えて 600 手 → 200 手にする。
- 8 以外の反復に変えるときは、ゲート 1 の台本の再生と発散率を必ず測り直す。

## 理由

(数値の正本は [`docs/experiments/2026-09-30_v4_speed.md`](../experiments/2026-09-30_v4_speed.md)。条件・シードもそこにある。)

- v3 のままの設定 (4096 環境 × 12 手、反復 16) では 1 iteration が約 20 分、既定の 2000 iteration で約 27 日かかった。
- 時間のほぼ全部が PhysX の物理ステップで、1 物理ステップ ≈ 34 ms の固定費 + 0.019 ms × 環境数。固定費の正体はソルバの位置反復。PhysX はシーン内の最大値で解くので、グリッパーだけ下げても効かない。
- 固定費が大きいので、環境数を減らすとスループットだけが落ちる。そのため環境数ではなく 1 iteration の手数を減らした (1024 環境 × 6 手でも 1 iteration は約 4 分になるが、スループットが半分になる)。
- v3 の 1024 環境と 4096 環境の run は同じ iteration 数でほぼ同じ所にいた。序盤は PPO の更新回数 (1 iteration に 20 回) が律速だった。
- v4 のエピソードは段が変われば終わり (最大 6 手)、報酬 (+1 / −1) は段を変えた手そのものに付く。4 手で打ち切られたエピソードは GAE が次の状態の価値で bootstrap する。
- 反復 8 は、ゲート 1 の台本の再生 (256 環境、seed 0 / 1 / 2) で 3 シードとも反復 16 より固結びが多く、発散が少なかった。学習済みの v3 方策でも発散が少なかった。
- 4 時間ずつの学習の比較 (seed 42、各 1 本) で、反復 8・4 手 (C) の `chain_est` は変更前 (A) の約 7 倍で、発散で終わるエピソードも 3 条件で最も少なかった。
- 合わせて 1 iteration は約 20 分 → 約 3 分になり、同じ時間で回る更新は約 6 倍、集まる手は約 1.8 倍になった。

## 検討して採らなかった案

| 案 | 採らなかった理由 |
|---|---|
| 環境数を減らす | 固定費が大きく、1 iteration はあまり短くならずスループットだけが落ちる |
| 反復 4 / 6 (さらに速い) | ランダムな手の 1 手あたりの発散が 6 以下で急に増える (8 は増え方が緩やかな最後の値) |
| 反復 10 / 12 / 32 | 台本の再生で反復 8 より固結びが少なく発散が多い (12 が最悪。反復と成否の関係は単調ではない) |
| 自己接触を切る | 結べなくなる |
| PhysX のバッファを既定に戻す / グリッパーだけ反復を下げる | 時間が変わらない |
| 1 iteration の手数だけを減らす (反復 16 のまま。B) | 速くなるが、反復 8 を足した C のほうがさらに速く進んだ |

注意 (未確認の解釈): ランダムな手では反復が少ないほど発散が増える。反復を減らすと接触が柔らかくなり、滑らかな手では安定するが乱暴な手ではめり込みやすい、と考えている。学習全体ではこの増加は表に出なかった (C の発散が最小)。

## 今の実装

(`kt/` = `src/isaac_twisted_rl/tasks/knot_tying/`)

| 何 | 場所 | 確認 |
|---|---|---|
| `solver_position_iterations: int = 8` と実測の表の docstring | `kt/versions/v4/env_cfg.py:192-205` | 一致 |
| プリセット適用でロープ・グリッパーの両方に反復を書き込む | `kt/common/rope_task_cfg.py:78-81`、既定 `None` は `kt/common/env_cfg_base.py:156` | 一致 |
| `NUM_STEPS_PER_ENV = 4` と理由の docstring | `kt/versions/v4/env_cfg.py:60-66` | 一致 |
| `num_steps_per_env`・`save_interval`・`max_iterations = 2000 * 12 // NUM_STEPS_PER_ENV` | `kt/versions/v4/agents/rsl_rl_ppo_cfg.py:55-59` | 一致 |
| アーカイブを書く間隔 `_SAVE_EVERY = SAVE_INTERVAL * NUM_STEPS_PER_ENV` (= 200 手) | `kt/versions/v4/env_cfg.py:70-74` | 一致 |
| v5 は反復 8・1 iteration 4 手を v4 のまま使う | `kt/versions/v5/env_cfg.py:57-60`、`kt/versions/v5/agents/rsl_rl_ppo_cfg.py:5` | 一致 |
| 変更前に戻す上書き (Hydra からは `solver_position_iterations` が効かないので資産の値を直接上書き) | `kt/versions/v4/README.md` の「学習速度」 | — |

既知の食い違い: `kt/versions/v4/README.md` の「段ごとのアーカイブ」は保存間隔を「600 手 (= 50 iteration)」と書くが、今の実装は 200 手 (`refactor/audit/B_2.md` の B-114)。

## 出典

- `kt/versions/v4/README.md:148-288` (commit 62b251b の時点の「学習速度 (2026-09-30)」。今は [`docs/experiments/2026-09-30_v4_speed.md`](../experiments/2026-09-30_v4_speed.md) に転記)
- `docs/history/next_action.md` §4.8.9 の `num_envs` のスループットの実測の追記 (頭書きが付く前の行番号で `:477-480`)
- `kt/versions/v4/env_cfg.py:17-18`、`:60-74`、`:192-205`
