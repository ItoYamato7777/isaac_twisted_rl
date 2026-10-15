# 0022 探索と汎化の改善 (Report_Paper の P2) は 1 版に 1 つずつ入れ、v8 は対称性のデータ拡張にする

- 状態: 有効 (v8 以降、P2 の案を版にするとき)
- 読むべき状況: `paper/Report_Paper.md` §7 の P2 (対称性・RND・初期形状のランダム化・推論時探索) を版に入れたくなったとき。rsl_rl の `rnd_cfg` を使いたくなったとき
- 最終更新: 2026-10-05

## 決定

- P2 の 4 案は 1 つの版にまとめない。1 版に 1 つずつ入れ、chain 評価 (シード 2 本以上) で前の版と比べる。
- v8 は **対称性のデータ拡張 (P2-1)** だけを入れる (v7 + 鏡像・逆順の 4 倍)。
- **RND (P2-2) は rsl_rl の `rnd_cfg` のままでは使わない。** 新しさで探索させるなら、環境側の報酬項として、
  手の後 (reset の前) の位相のキー (`knot_graph.batch_keys`) の訪問回数から作る。
- 初期形状のランダム化 (P2-3) は、chain 評価の失敗が段 0 に寄るまで入れない。
- 推論時探索 (P2-4) は版にしない (評価の道具。[0011](0011-success-rate-without-search.md))。

## 理由

### まとめない理由

- 版の差は chain 評価で測るが、シード差が 6〜19 ポイントある (v7 は iteration 100 で 6 ポイント、v6 は 15 ポイント。
  [`docs/experiments/2026-10-04_v7_seeds.md`](../experiments/2026-10-04_v7_seeds.md))。4 つを入れた版が良くても悪くても、どれが効いたか切り分けられず、
  切り分けの学習を 1 つずつ回すことになる (1 比較 = GPU 2 枚 × 約 6.5 時間)。
- RND は報酬を変え、対称性は同じ報酬のまま更新のデータを増やす。性質の違う変更を混ぜると、効いたときも理由が読めない。

### 対称性を先にする理由

- v7 の詰まりは段 2 -> 固結び (学習ログの成功 0.19〜0.24。段 0 は 0.97)。成功が稀な遷移ほど、1 回の成功を 4 倍に使える効果が大きいと見込む (推測)。
- 物理を 1 手も増やさず、報酬・環境を変えない。rsl_rl 3.0.1 の PPO に実装がある。

### RND を rsl_rl のまま使わない理由

- rsl_rl は内発報酬を `PPO.process_env_step` に渡された観測で計算する。Isaac Lab の `env.step()` は
  「終了判定 -> 報酬 -> reset -> 観測」の順なので、終了した環境の観測は **reset の後の、次のエピソードの初期状態** になる。
- v4 以降は段が変わると必ずエピソードが終わる ([0008](0008-transition-reward-and-termination.md))。「新しい位相に着いた」手の新しさは、
  次のエピソードの初期状態の新しさとして数えられ、新しい位相に着いたことには払われない。
- 観測 (197 次元) の新しさは、物理の非決定性 (同じ状態・同じ手で手の後の段が約 3 割変わる。
  [`docs/experiments/2026-10-03_inference_search_detail.md`](../experiments/2026-10-03_inference_search_detail.md) §3.1) で水増しされる。位相のキーは離散なので数えられる。

### 初期形状のランダム化を後にする理由

- 段 0 は確率 0.5 で、学習中に到達した段 0 の形状 (アーカイブ) から始まる (`kt/common/mdp/archive.py` の `ResetToStage`)。初期形状のランダム化の役目をある程度果たしている。
- 段 0 の成功は 0.97 で詰まりではない。chain 評価もまっすぐなロープから始まる。

## 検討して採らなかった案

| 案 | 採らなかった理由 |
|---|---|
| 4 案をまとめて v8 にする | 上の「まとめない理由」 |
| `rnd_cfg` をそのまま有効にする | 上の「RND を rsl_rl のまま使わない理由」 |

## 今の実装

| 何 | 場所 | 確認 |
|---|---|---|
| 対称変換 (純 torch) | `kt/common/topology/symmetry.py` | `tests/test_symmetry.py` |
| rsl_rl に渡す関数 | `kt/common/mdp/symmetry.py` の `augment_mirror_reverse` | — |
| v8 の配線 | `kt/versions/v8/agents/rsl_rl_ppo_cfg.py` | — |
| 内発報酬の計算に使う観測 | `.venv/.../rsl_rl/algorithms/ppo.py` の `process_env_step` (`self.rnd.get_intrinsic_reward(obs)`) | rsl_rl 3.0.1 |

## 出典

- `paper/Report_Paper.md` §7 P2、§9.3
