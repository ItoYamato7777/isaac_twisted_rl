# 0014 `Knot` で学習するための 3 点: 置き先 ±0.30 L・1 手 3.16 s の尺・運搬高さ 0.20 m 固定

- 状態: 有効 (v3〜v7。共有層の既定値は v0〜v2-R2 のまま)
- 読むべき状況: 置き先の範囲・1 手の尺 (特に carry・descend)・運搬高さを変えたくなったとき。運搬高さを行動に入れる案が出たとき
- 最終更新: 2026-10-04

## 決定

ゲート 1 (2026-09-27) が「学習に使うなら v3 で配線すること」とした 3 点 (`Knot` プリセット・`drop_xy_rel_frac` 0.30・3.2 s/手の尺) を、v3 で配線した。`Knot` については [0013](0013-rope-knot.md)。

| 項目 | v2-R2 まで | v3 以降 |
|---|---|---|
| 置き先の範囲 `drop_xy_rel_frac` | ±0.15 L | **±0.30 L** |
| 1 手の尺 | 2.44 s (`SHORT_DURATIONS`) | **3.16 s** (`KNOT_DURATIONS`: descend / lift / lower 0.3 s, carry 0.6 s。759 物理ステップ) |
| 運搬高さ | 0.20 m 固定 (`CARRY_PALM_Z`) | 0.20 m 固定のまま (行動に入れない) |

- **共有層の既定値 (`GraspLimits` / `SHORT_DURATIONS`) は変えない。** v0〜v2-R2 の挙動を変えないため ([0012](0012-version-is-config.md))。v3 では `KnotTyingEnvCfg.grasp_limits` と `KNOT_DURATIONS` を通して配線した。

## 理由

### 置き先 ±0.15 L は足りない

- ゲート 1 の前の予想 (旧 §4.8.9): 失敗したら、まず `GraspLimits.drop_xy_rel_frac` を 0.15 → 0.30 に広げること (`drop_xy_max` = 0.123 m は、最小ループ周長 0.251 m の半分しかない)。予想どおり 0.30 が必要だった。
- cover の置き先は、128 環境中 79〜97 で target リンク ±8 本 + ±0.15 L の外にあり、クリップされて輪ごと崩れた。
- 6 次元の行動空間でも結べる (66%)。条件は `drop_xy_rel_frac` 0.30 と 3.2 s/手の尺。1 手目 (たるみ) の置き先は ±0.30 L でも届かずクリップされるが、クリップ後の形でも後続は成立する (届く範囲に台本を直すと 49/128 に下がった)。

| 6 次元行動で再生 (`env.step`、`Knot`、128 環境) | 固結び |
|---|---|
| 7.8 s/手, `drop_xy` ±0.15 L | 36 (28%) |
| 3.2 s/手, `drop_xy` ±0.15 L | 12 (9%) |
| 7.8 s/手, `drop_xy` ±0.30 L | 54 (42%), 発散 19 |
| 3.2 s/手, `drop_xy` ±0.30 L | **84 (66%)**, 発散 9, クリップ 0 (1 手目を除く) |

### `Knot` は学習の短い尺で発散する

原因は運搬速度 (carry 0.35 s で 0.5 m 以上動かし、1.14 m の本体を引きずる)。carry 0.6 s にすると、1 手目の発散は 13/32 → 0/32 になった。ソルバ反復 32・指ゲイン 2 倍は効かなかった。descend が速いと、結び目の上へ指を降ろす手でも発散する (tuck の descend/close で 18/128)。

### 運搬高さ 0.20 m 固定は障害ではない (むしろ良い)

被せる手・通す手を低く運ぶ台本 (60%) より、全部 0.20 m の台本 (95%) のほうが成功した。`paper/Report_Paper.md` §5.6 / P3-1 の「z_max を行動に入れる」は、少なくとも固結びには不要。

## 未確認

同じ尺でも、`env.step()` で再生すると台本モードより発散が多い (3.2 s/手で 9 vs 3、7.8 s/手で 19 vs 0)。遅い尺のほうが再生の成功率が低い理由も分かっていない。`PickPlaceAction` の経路 (approach の yaw 補間、終了判定の速度閾値など) を疑っているが、未確認。

## 検討して採らなかった案

| 案 | 採らなかった理由 |
|---|---|
| 共有層の既定値を 0.30 L・長い尺に変える | v0〜v2-R2 の挙動が変わる |
| 運搬高さを行動に入れる (Report §5.6 / P3-1) | 上の実測で、固結びには不要 |
| 発散をソルバ反復 32・指ゲイン 2 倍で抑える | 効かなかった |

## 今の実装

(`kt/` = `src/isaac_twisted_rl/tasks/knot_tying/`)

| 何 | 場所 | 確認 |
|---|---|---|
| `KNOT_DURATIONS` (approach 0.0, descend 0.3, lift 0.3, carry 0.6, lower 0.3, clear 0.0, relax 0.5、合計 3.1625 s = 759 物理ステップ) | `kt/versions/v3/env_cfg.py:56-71`、`:186` | 一致 (ゲート 1 の「3.2 s/手」と v3 README の「3.16 s」はこの値) |
| `grasp_limits = GraspLimits(drop_xy_rel_frac=0.30)` | `kt/versions/v3/env_cfg.py:188-194` | 一致 |
| 共有層の既定 `drop_xy_rel_frac = 0.15`、`SHORT_DURATIONS` | `kt/common/mdp/phases.py` の `GraspLimits`・`SHORT_DURATIONS` | 一致 (変えていない) |
| 運搬高さ `CARRY_PALM_Z = 0.20` | `kt/common/mdp/phases.py:46`、`kt/common/mdp/actions.py:518-521` | 一致 |
| v4〜v7 は v3 の尺と置き先の範囲のまま | `kt/versions/v4/README.md` の変更表 (「ロープ・尺・置き先の範囲・発散の扱い・PPO 本体・γ は v3 のまま」) | 一致 |

## 出典

`docs/history/next_action.md` の行番号は、頭書き (33 行) を足す前のもの (今の行番号は 33 を足す)。

- `docs/history/next_action.md:467-470` (旧 §4.8.9 の 1 つ目)、`:486-488`、`:509-512`、`:520-527`、`:533-540` (旧 §4.8.10 の「分かったこと」2・3・4・7)
- `kt/versions/v3/README.md:31-45` (変更表)
- 結果の表の全体は [`docs/experiments/2026-09-27_gate1.md`](../experiments/2026-09-27_gate1.md)
