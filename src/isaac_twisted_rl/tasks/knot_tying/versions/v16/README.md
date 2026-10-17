# v16: v15 + ロープの長さを env ごとに振る DR

```bash
uv run train --task Isaac-Twisted-RL-Knot-v16 --headless           # v15 + 寸法の DR
uv run play  --task Isaac-Twisted-RL-Knot-Play-v16 --num_envs 16   # 公称の寸法 (v15 と同じロープ) で chain 評価
```

ログ: `logs/rsl_rl/knot_tying_v16/`。**学習には `Knot` プリセットを使う** (寸法の組は `knot` からしか作れない。ほかのプリセットの v16 のタスクは起動時に例外になる)。継承元は **v15** (ユーザー、2026-10-17)。
仕様と理由の正本は [`docs/decisions/0029-v16-rope-size-dr.md`](../../../../../../docs/decisions/0029-v16-rope-size-dr.md)。ここには配線と、見るべき指標を書く。
初期アーカイブは v15 と同じ (`outputs/archive_seed/overhand_28links.pt`、公称の寸法。復元のときに env の寸法に伸ばす)。

## なぜ作ったか

実機の紐 (直径 8 mm・全長 1.02 m) は sim の公称 (直径 20 mm・全長 1.14 m) と寸法が違う (直径 ×0.4、長さ ×0.895)。寸法を振って学習し、方策が寸法の違いを吸収できるかを見る (ユーザーの指示、2026-10-14)。

0029 は直径も振る格子 (長さ ×0.8/1.0/1.25 × 直径 ×0.4/0.63/1.0) だったが、物理のゲートで直径 ×0.4・×0.63 の台本が結べず (直径 ×0.4 は掴めない)、**長さだけを振る**ことにした (ユーザー、2026-10-17。[ゲートの記録](../../../../../../docs/experiments/2026-10-17_v16_size-gate.md))。実機の直径は入らない。長さ ×1.25 は、角度制限を ±60° に揃えると台本・発散とも悪く (台本 25/128、1 手あたりの発散 14.7%)、×1.1 に替えた (ユーザー、2026-10-17)。実機の長さ (×0.895) は範囲の中。

## v15 から何を・なぜ変えたか

| 項目 | v15 | v16 |
|---|---|---|
| ロープ | `knot` 1 種 | **長さの 3 組** (×0.8 `knot_l080` / ×1.0 `knot` / ×1.1 `knot_l110`、直径 ×1.0) の USD を、env `i` に組 `i % 3` で割り当てる (`ROPE_SIZE_SCALES`)。公称の組は `knot` の USD そのもの。**関節の角度制限は 3 組とも公称** (曲げ ±60°・捩り ±90°。`KEEP_JOINT_LIMITS`) |
| critic の特権観測 `rope_params` | 物理の倍率 6 次元 | 6 + **寸法の倍率 2 次元** (`mdp.rope_size_params`。log2 を学習の範囲 (長さ ×0.8〜×1.1) で [−1, 1] に。長さ ×0.8 / 1.0 / 1.1 は −1 / +0.40 / +1。直径は範囲が 1 点なので 0 の定数)。critic の入力は 203 -> 205 次元 (0029 の判断 3 の形のまま) |
| 起動時の検算 | | `events.rope_sizes` (`mdp.check_rope_sizes`): 全 env の USD の半径・円柱の長さが割り当てと一致するか |
| Play | v15 | v15 と同じ + 寸法を公称の 1 組に (`env.scene.rope.spawn.scales`) |
| それ以外 (actor の観測 197 次元・行動・報酬・物理の DR 6 軸・段 0 の保存の規則・時間切れの項・PPO・方策の構造) | | v15 のまま |

- PhysX は実行中にカプセルの形を変えられないので、寸法は起動時に決まり、run の間変わらない (reset で引き直す物理の DR とは違う)。
- 長さの組は同じ材料・同じリンク数 28・同じ半径で、リンク間隔だけを変える (`rope_specs.make_size_variant(..., keep_joint_limits=True)`)。質量・慣性・関節のばね (E I / 間隔)・減衰はリンク間隔から導き直す。
- **関節の角度制限は 3 組で揃える** (曲げ ±60°・捩り ±90°。ユーザー、2026-10-17)。アーカイブは形を関節角で持ち、寸法の違う env の間で書き戻すので、組ごとに制限が違うと保存した関節角が書き戻し先の制限を超えて、リセットの直後に飛ぶ (導出した制限は ×0.8 で ±47.2° で、学習のアーカイブの形の半分以上がそれを超えた)。揃えると書き戻した形は保存した形と相似になる。代わりに最小曲げ半径がリンク間隔に比例する (×0.8 で 32 mm = 直径の 1.6 倍、公称 40 mm、×1.1 で 44 mm = 2.2 倍)。
- 指のゲインは公称 (`knot`) のまま (0029 の判断 4)。
- 観測はメートルのまま (長さで割らない。0029 の判断 2)。actor には寸法が直接は見えず、リンクの位置から読み取るしかない。

### 寸法を env ごとに使う箇所 (`mdp.get_rope_sizes` から読む)

ロープの spawn 設定 (`RopeSizeSpawnerCfg`) が寸法の唯一の出どころ。v15 までの設定は spawn 設定が `UsdFileCfg` なので、どの箇所も 1 つの数のまま動く。

| 箇所 | 何を env ごとにするか |
|---|---|
| 行動 (`PickPlaceAction`) | 置き先の範囲 `drop_xy_rel_frac × 全長`、`clear` の距離 `clear_diameters × 直径` (v16 は `clear` の尺 0 だが、次の手の始点に使う)、診断の半径。v16 は `release_tip_z` (床からの高さ [m]) を使うので `drop_z_diameters` は効かない |
| 床に落ちた判定 (報酬・終了・段・アーカイブ) | `settled_height_diameters × 直径` (`mdp.per_env_diameter`) |
| アーカイブ | 保存は root の重心からの xy をリンク間隔の倍率で、高さを直径の倍率で割って公称の寸法で持つ。復元は書き込み先の env の倍率を掛ける (`state_archive.rescale_rel_pose`) |
| 接触の設定 | 寸法ごとの USD の spawn 設定が持つ (`contact_offset`・`max_depenetration_velocity` は半径に比例) |

寸法に依らない (変えていない) もの: 位相の判定、物理の DR (公称値は env ごとに起動時の PhysX から読むので、寸法ごとの値に倍率が掛かる)、離すときの指先の高さ `release_tip_z`、段 0 の形の変化の下限 `s0_min_change` (0.02 m)、発散の判定。

## 寸法を振った評価

```bash
... eval_chain.py --task Isaac-Twisted-RL-Knot-Play-v16 ... \
    env.scene.rope.spawn.scales="[[0.8,1.0],[1.0,1.0],[1.1,1.0]]"
```

`rope_size_scales` (env の設定のフィールド) は Hydra の上書きでは効かない (プリセットの適用で spawn 設定に写した後なので)。spawn 設定の `scales` を直接書く。組を 1 つにすれば、その寸法だけの評価になる。

## 制約

- 寸法の違う env の間では状態を写せない (推論時探索 `eval_search.py`。`EnvStateCloner` が例外にする)。探索は `scales` を 1 組にして回す。
- アーカイブの復元は、root 以外のリンクの高さを関節角とリンク間隔で決めるので、長さの違う env へ書き戻すと交差の上下のずれが直径と合わない (×0.8 へ書き戻すと上の紐が約 4 mm めり込み、×1.1 では約 2 mm 浮く。`state_archive.rescale_rel_pose` の docstring)。`Curriculum/stage_log/restore_mismatch` に出る。
- `scene.replicate_physics=False` (env ごとに違う USD を PhysX に読ませるため)。起動の時間と 1 手の時間への影響は実験記録にある。

## 見るべき指標

- 公称の寸法での chain 評価を v15 の同じ seed と比べる (寸法を振ったことで公称の成績が落ちていないか)。
- 寸法を振った chain 評価 (上の上書き)。組ごとの内訳は、`scales` を 1 組ずつにした評価で見る。
- `Curriculum/stage_log/restore_mismatch`: v15 より増えたら、寸法の違う env の間の書き戻しで段が変わっている。
- `Episode_Termination/diverged`: v15 と比べる。寸法を混ぜたことで増えていないか (ゲートでは寸法ごとの発散を `divergence_bench.py` の組ごとの内訳で見た)。
- 学習の job のログの `PhysX error` (aggregate pairs のあふれ)。0 であること。
