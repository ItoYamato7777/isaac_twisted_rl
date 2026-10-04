# 0003 物理は 240 Hz で回し、残った発散はその場で reset する

- 状態: 有効 (v0〜v7 のすべてに適用)
- 読むべき状況: `sim.dt` を変えたくなったとき (学習を速くするために 120 Hz へ落とす案など)、ロープの接触設定を変えるとき、発散した環境の扱いを変えるとき
- 最終更新: 2026-10-04

## 決定

- 物理は **240 Hz** (`sim.dt = 1/240 s`) で回す。120 Hz へ落とす案は実測で却下した。
- `make_rope_physx_cfg` (シーン全体の接触設定) は併用しない。
- 残った発散は、終了項 `diverged` がその場で reset し、報酬 `failure` (`action_failed`) が罰を与える経路で吸収する。

## 理由

`twist` / 16 環境 × 10 手 / ランダム方策での発散率の実測:

| 設定 | 発散率 |
|---|---|
| 120 Hz | 11.2% |
| **240 Hz** | **1.2%** ← 採用 |
| 240 Hz + `make_rope_physx_cfg` | 3.8% |

- 120 Hz では、把持中の接触でロープの速度が 60〜175 m/s (正常時は 1 m/s 程度) まで跳ね上がり NaN になった。
- 原因は、`rope_cfg.py` の `max_depenetration_velocity` が **240 Hz を前提に**スケールされていること (`_SIM_HZ_REF = 240.0`)。120 Hz では 1 ステップの押し戻し量が倍になり、「半径の 40% 以上は押し戻さない」という前提が崩れる。
- `make_rope_physx_cfg` を併用すると却って悪化するのは、シーン全体の接触設定を変えてグリッパーの押し付け力にも影響するため。
- エピソードが NaN で落ちるのは RL では致命的なので、発散した環境はその場で終了させて reset に回す (`kt/common/mdp/terminations.py` の docstring)。

プリセット別 (240 Hz, ランダム方策 20 手): `twist` / `simple` / `stiff` は発散 0、`fine` のみ 20 手あたり 1〜2 回。**`fine` で本格的に学習を回すなら、さらに `sim.dt` を詰める必要がある。**

この節の数値は v0 の時点 (2026-09-10 前後) の `twist` での計測。`Knot` と v3 以降の発散は別に測ってある (v3 の本番の 1 手 5〜6%、v4 の 18 手のあいだ 17% など。`docs/experiments/` と v5 README)。

## 検討して採らなかった案

| 案 | 採らなかった理由 |
|---|---|
| 120 Hz (学習を速くする) | 発散 11.2% (240 Hz の約 9 倍)。上の理由 |
| 240 Hz + `make_rope_physx_cfg` | 発散 3.8%。グリッパーの押し付けにも影響する |

## 今の実装

(`kt/` = `src/isaac_twisted_rl/tasks/knot_tying/`)

| 何 | 場所 | 確認 |
|---|---|---|
| `self.sim.dt = 1.0 / 240.0` と、上の表のコメント | `kt/common/env_cfg_base.py:192-211` | 一致 |
| `_SIM_HZ_REF = 240.0`、`max_depenetration_velocity = 0.4 × 半径 × 240` (`Knot` で 0.96 m/s) | `kt/common/rope_model/rope_cfg.py:34`、`:48-59` | 一致 |
| 終了項 `diverged` (NaN・高さ 5 m 超。v2-R2 以降は水平距離・リンク速度も) | `kt/common/env_cfg_base.py:110-112`、`kt/common/mdp/terminations.py:29-65` | 一致 |
| 報酬 `failure` (`action_failed`) | `kt/common/mdp/rewards.py:387`、v0 の重み −0.2 (`kt/versions/v0/env_cfg.py:99`) | 一致 |
| `fine` は 240 Hz のまま (「`sim.dt` を詰める必要がある」の注記) | `kt/common/rope_task_cfg.py:131-140` | 一致 (未対応のまま) |

## 出典

`docs/history/next_action.md` の行番号は、頭書き (33 行) を足す前のもの (今の行番号は 33 を足す)。

- `docs/history/next_action.md:319-338` (旧 §4.5)
- `kt/common/README.md:41-45` (「物理は 240 Hz」)
