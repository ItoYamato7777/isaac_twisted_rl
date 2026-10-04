# 0007 密報酬は位相クラスのラダーをポテンシャル整形で登らせる (v1〜v3)

- 状態: 有効 (v1・v2・v2-R2・v3 の設定。v3 はラダーと BFS 距離の max。v4 以降の報酬は [0008](0008-transition-reward-and-termination.md) に置き換わった)
- 読むべき状況: v1〜v3 の報酬を読む・直すとき。整形報酬の項を足すとき。`gamma` を変えるとき
- 最終更新: 2026-10-04

## 決定

- 密報酬は交差数ではなく**位相クラス** (0: 交差なし / 1: 交差 1 つ / 2: 絡み 2 交差 / 3: 固結び) を段にし、ポテンシャル整形 `F = γΦ(s') − Φ(s)` で登らせる (`knot_class_shaping`)。
- **`gamma` は報酬整形と PPO で一致させる。** 値を二重に管理しないよう、`env_cfg.py` の `GAMMA` を `agents/rsl_rl_ppo_cfg.py` が import する。

| 位相クラス | Φ (v1〜v2-R2) | 位相 |
|---|---|---|
| 0 | 0.00 | 交差なし / 行き止まり (独立したツイストだけ、など) |
| 1 | 0.35 | 交差 1 つ (v0 の目標がここ) |
| 2 | 0.70 | 絡んだ 2 交差・同符号 |
| 3 | 1.00 | 固結び |

v3 は Φ = max(BFS 距離の値, R2 除去後のラダーの class / 3) にした ([0010](0010-stage-granularity-bfs.md))。切り分け用の `potential="class"` の値は (0, 1/3, 2/3, 1)。

## 理由

### 交差数ではなく位相クラス

`crossing_progress` (Δ交差数、v0 の密報酬) には「R1 を 3 回ひねれば交差数 3」という抜け道がある。独立した R1 ツイストを 3 回作れば交差数は 3 になるが、固結びには一歩も近づかない。三葉結びを途中まで描くと段が 0 → 1 → 2 → 3 と順に上がることを確認してあり (`tests/test_knot_detect.py::test_partial_trefoil_walks_up_the_ladder`)、人間が結ぶ手順にそのまま対応する。固結びの判定 (段 3) の理由は `kt/common/topology/knot_detect.py` の docstring にある。

### 素朴なボーナスではなくポテンシャル整形

`if class == 2: +0.70` のように段そのものに報酬を付けると、**2 段目に居座って毎手ボーナスを稼ぐ**のが最適方策になってしまう。ポテンシャル形式なら、段に留まるだけでは `γ·0.70 − 0.70 ≈ −0.014` (γ = 0.98) にしかならず、登らないと得にならない。しかも Ng et al. (1999) により、**最適方策を変えないことが保証されている**。この保証は γ の一致が前提になっている。

### 注意 (後の版で分かったこと)

- 「ポテンシャル整形は最適方策を変えない」という保証自体は正しいが、この保証は状態ボーナスで決まる目的関数に対するものであり、「固結びを作る」だけを目的にした問題とは一致しない (`paper/Report_Paper.md` §5.5)。v2 / v2-R2 は整形に加えて段ごとの疎報酬を**毎手**払い、C1 や偽の C2 に居座られた ([0008](0008-transition-reward-and-termination.md))。
- 整形報酬を検証するときの罠: [0015](0015-shaping-verification-pitfall.md)。

## 今の実装

(`kt/` = `src/isaac_twisted_rl/tasks/knot_tying/`)

| 何 | 場所 | 確認 |
|---|---|---|
| `knot_class_shaping` | `kt/common/mdp/rewards.py:100` | 一致 |
| v1 の配線 `potentials = (0.0, 0.35, 0.70, 1.0)`、`gamma = GAMMA` | `kt/versions/v1/env_cfg.py:96-100`、`GAMMA = 0.98` は `:25` | 一致 |
| PPO 側が `GAMMA` を import | `kt/versions/v1/agents/rsl_rl_ppo_cfg.py:45`、`:70` | 一致 |
| 位相クラスの定数 `CLASS_TRIVIAL` 〜 `CLASS_OVERHAND` | `kt/common/topology/knot_detect.py` | 一致 |
| v3 の切り分け用の値 `_LADDER_POTENTIALS = (0, 1/3, 2/3, 1)` | `kt/versions/v3/env_cfg.py:52-54` | 一致 |

## 出典

`docs/history/next_action.md` の行番号は、頭書き (33 行) を足す前のもの (今の行番号は 33 を足す)。

- `docs/history/next_action.md:436-441` (旧 §4.8.4)
- `kt/versions/v1/README.md:14-61` (§1 の変更表・「位相クラスのラダー」・「なぜ素朴なボーナスではなくポテンシャル整形なのか」)
- `paper/Report_Paper.md:187-198` (§5.5)
