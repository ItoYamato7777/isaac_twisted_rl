# 0005 成功の判定はハンドを退避させた後、ロープが床に落ちているときだけ数える

- 状態: 有効 (v0〜v7 のすべての成功・段の判定に適用)
- 読むべき状況: 成功・段の変化を判定する報酬や終了項を足す・変えるとき。フェーズ列の最後 (`relax`) を変えるとき。ログで「交差が残った」を読むとき
- 最終更新: 2026-10-04

## 決定

- **最重要: 成功判定は必ずハンドを退避させた後に測る。** 報酬の測定タイミングは、「離すと付いてくる」問題 ([0004](0004-release-follow-physics.md)) が解決したこととは別に、明示的に押さえる。
- 加えて、ロープの最高点 `rope_zmax` が床付近 (ロープ直径の 3 倍以下) にあることを成功・段の変化の条件にする (二重の保険)。
- ログには `rope_zmax` と `finger_gap` も出す。この 2 つを一緒に見ないと、「ハンドが吊っているだけ」を「交差が残った」と取り違える。
- p-data は離散量なので、一致 / 不一致だけでは勾配が出ない。v0 では密報酬と併用した (重みは v0 README)。

## 理由

退避させずに測ると、「ハンドで吊って交差を見せかける」方策を学習する。

ハンドに吊られていればロープの最高点は 0.2 m 以上になり、床に落ちて交差していれば直径の数倍に収まる (把持高さ `GRASP_PALM_Z` は指先が床上 2 mm に来る高さ)。この差で「吊っているだけ」を弾ける (`kt/common/mdp/rewards.py:86-92` の docstring)。

## 今の実装

(`kt/` = `src/isaac_twisted_rl/tasks/knot_tying/`)

| 何 | 場所 | 確認 |
|---|---|---|
| フェーズ列の最後が `relax` (横へ抜けて待機高さ 0.30 m へ上がった後、指を全開で静止) なので、報酬計算の時点でハンドは離れている | `kt/common/mdp/rewards.py:9-23` (docstring)、`kt/common/mdp/phases.py` のフェーズ表 | 一致。ただし v2 以降の尺 (`SHORT_DURATIONS`、`KNOT_DURATIONS`) は `clear = 0.0` (1 物理ステップで通過) |
| `settled_height_diameters = 3.0` の条件 (v0 `crossing_achieved`、v1 以降 `overhand_achieved` など) | `kt/common/mdp/rewards.py:77-98`、`:252-276` | 一致 |
| v4 以降: 段の上がり下がりは、床に落ちていて (最高点が直径の 3 倍以下) 発散していない手だけ数える。吊ったままの手は「変わらない」扱い | `kt/common/mdp/stages.py:92-112` (`_outcome`) | 一致 |
| `rope_zmax` / `finger_gap` | `kt/common/mdp/observations.py:310-330` | 一致 |

## 出典

`docs/history/next_action.md` の行番号は、頭書き (33 行) を足す前のもの (今の行番号は 33 を足す)。

- `docs/history/next_action.md:289-299` (旧 §4.2)
- `kt/versions/v4/README.md:77-84` (「1 遷移 = 1 エピソード」の床に落ちている条件)
