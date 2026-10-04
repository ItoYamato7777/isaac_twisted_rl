# 0001 キネマティックハンドと「1 手 = 掴む→運ぶ→離す 全部」

- 状態: 有効 (v0〜v7 のすべてに適用)
- 読むべき状況: ロボットの形態 (アームを付けるか) や、行動の粒度 (1 RL ステップに何をさせるか) を変えたくなったとき
- 最終更新: 2026-10-04

## 決定

2026-09-10 に人間と合意した決定事項 (旧 `next_action.md` §4.0)。

| 項目 | 決定 | 理由 |
|---|---|---|
| 最初に学習させる課題 | 交差を 1 つ作って**残す** (v0) | 本命タスクの最小版。作った env / action / reward がそのまま先に使える |
| ロボットの形態 | **キネマティックハンドのまま** (アームを付けない) | IK の失敗・可達領域外・自己衝突・特異点が無い。学習が「腕を動かす」ではなく「紐を操作する」ことだけに集中する |
| 行動の粒度 | **1 アクション = 掴む→運ぶ→離す 全部** (Options Framework 的) | 1 エピソード 3〜5 アクション。Step 5 の台本 (`demos/topology_debug.py`) と同じ粒度 |
| アクション空間 | ロープ相対 + `Δlink` | [0002](0002-action-space-rope-relative.md) |

1 手 = 6 次元の pick&place 1 回 = 1 env ステップ = `decimation` 回の物理ステップ。

## 理由

表の「理由」の列のとおり。粒度については、Isaac Lab の `ManagerBasedRLEnv.step()` が「`process_action` を 1 回 → `decimation` 回の物理ステップ → termination / reward / observation を 1 回」の構造なので、`decimation` を 1 手ぶんの物理ステップ数にすれば、報酬・観測・終了判定が手の終わりに 1 回だけ評価される形がそのまま作れる (`kt/common/mdp/actions.py` の docstring)。

1 エピソードの手数は版ごとに変わった (v0 は 5 手で合意どおり。以降の版は課題に合わせて増やした。各 `versions/vN/README.md`)。

## 今の実装

(`kt/` = `src/isaac_twisted_rl/tasks/knot_tying/`)

| 何 | 場所 | 確認 |
|---|---|---|
| キネマティックハンド (アームなし)、`write_root_pose_to_sim` で直接駆動 | `kt/common/env_cfg_base.py:19-24` (docstring)、`kt/common/mdp/actions.py:582` | 一致 |
| `decimation` = 1 手ぶんの物理ステップ数 | `kt/common/env_cfg_base.py:213-216`、`kt/common/mdp/phases.py` の `PhaseDurations.decimation` | 一致 |
| 1 エピソードの手数 | v0 5 手 (`kt/versions/v0/env_cfg.py:116`)、v1 8 手、v2 6 手、v3 12 手、v4 6 手 (`kt/versions/v1/env_cfg.py:136`、`v2/env_cfg.py:261`、`v3/env_cfg.py:196`、`v4/env_cfg.py:184`) | v0 は合意の「3〜5 アクション」の範囲。v1 以降はこの範囲の外 (各版の README に理由) |

## 出典

`docs/history/next_action.md` の行番号は、頭書き (33 行) を足す前のもの (今の行番号は 33 を足す)。

- `docs/history/next_action.md:222-229` (旧 §4.0)
- `kt/common/mdp/actions.py:3-26` (docstring。旧 §4.0 を参照していた)
