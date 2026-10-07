# v11: v7 + 表に無くラダーでも 0 の「行き止まりの形」だけを段 −1 にする

```bash
uv run train --task Isaac-Twisted-RL-Knot-v11 --headless           # 段ごとの MLP
uv run play  --task Isaac-Twisted-RL-Knot-Play-v11 --num_envs 16   # まっすぐなロープから段をつないで評価 (chain モード)
```

ログ: `logs/rsl_rl/knot_tying_v11/`。**学習には `Knot` プリセットを使う** (v3〜v10 と同じ)。継承元は **v7** (v8・v9・v10 ではない)。
仕様と理由の正本は [`docs/decisions/0024-v11-dead-end-stages.md`](../../../../../../docs/decisions/0024-v11-dead-end-stages.md)。ここには配線と、見るべき指標を書く。
初期アーカイブは v7 と同じファイルなので、作り直しは要らない。

## なぜ作ったか

v7 の段は「Φ の表に無い形はラダーの段」で、ラダーが 0 の形は段 0 になる。v7 seed 43 を長く学習すると、方策の失敗で作った
この形 (交差 4 個・R2 で消えない・ラダー 0) が段 0 のアーカイブに溜まった (段 0 のうち it100 で 9〜10%、it330 で 70%)。
v10 は段を正規の経路に絞ってこれを除いたが、段 1 -> 段 2 の判定まで塞いで固結びをほぼ学べなかった。v11 は段 1 -> 段 2 の判定を
v7 のまま残し、行き止まりの形だけを外す。効果は長い学習 (iteration 100 -> 300) で出ると見込む (推測)。

## v7 から何を・なぜ変えたか

| 項目 | v7 | v11 | 理由 |
|---|---|---|---|
| 段の定義 (`stage_definition`) | `bfs` (`BfsPotential.stage` = round(3Φ)。属性は持たない) | **`bfs_dead_end`** (`topology.DeadEndStage`) | 下の表 |
| それ以外 (初期アーカイブ・`next_edge` の表・観測の形・報酬の値・アーカイブの容量・`stage_weights` 1:1:1・`straight_prob`・行動・物理・PPO・Play) | | v7 のまま | 1 版 1 変数 |

v7 との差がこれだけであることは `tests/test_v11_cfg.py` が静的に検査する。段の定義そのものは `tests/test_dead_end_stages.py`。

### 段の定義 `bfs_dead_end` (`common/topology/dead_end_stages.py`)

「表」は Φ の表 (`knot_graph.potential_table()` の 321 個の p-data)、生 = R2 除去の前。上から順に決める:

| 順 | 条件 | 段 |
|---|---|---|
| 1 | 生の p-data が表にある | v7 と同じ |
| 2 | 生は無いが、R2 除去後の p-data が表にある | R2 除去後の形の段 (偽の交差を許す) |
| 3 | どちらも無く、R2 除去後のラダーの段が 1 以上 | ラダーの段 (v7 と同じ) |
| 4 | どちらも無く、ラダーの段が 0 | **−1 (行き止まり)** |

- R2 除去は 1 が外れたときだけ使う。先に使うと、表の段 2 の 16 個 (台本の cover 後を含む) と段 1 の 12 個が 1 つ下の段になる。
- 表の全 p-data で v7 と同じ段。v7 と段が変わるのは表に無い形だけで、v7 のアーカイブの段 1・2 の形は 100% v7 と同じ段になる
  (0024 の「データ」。`outputs/archive_pdata/*.json` で数え直して再現した)。

段を引く項は、すべて `common/mdp/observations.py` の `stage_function(env)` を通る。v11 で何が変わるか:

| 項 | v11 での挙動 |
|---|---|
| 報酬 `stage_transition_reward`・終了 `stage_up` / `stage_down` | 段 0 から行き止まりの形を作った手は、v7 では「段が同じ」で続いたが、v11 では段 −1 に下がるので −1 で終了。2 で段 1 に上がる形 (v7 では段 0) を作った段 0 の手は +1 で終了。段 1・2 から行き止まりの形へ行く手は v7 でも「段が下がった」 |
| 観測 `knot_stage` (段の one-hot) | −1 は段 0 にする (`clamp_min(0)`)。v7 でも行き止まりの形は段 0 なので、chain 評価の観測は v7 と同じ |
| 観測 `next_edge` | v7 と同じ絞らない表。行き止まりの形はもともと表に無く、辺なし |
| アーカイブの保存 `record_stage_archive` | 段 −1 の形は捨てる (`StateArchive.add`)。2 の形は R2 除去後の形の段で保存する |
| Play (chain) | 固結びの判定は v7 と同じ (段 >= 3) |

## 見るべき指標

版の比較は chain 評価で行う ([`docs/system/evaluation.md`](../../../../../../docs/system/evaluation.md))。比べる相手は v7 の seed 43 / 44 を
iteration 300 まで延ばした run (0024 の「学習の長さ」)。学習ログは代理の指標で、版の良し悪しの結論には使わない。

* chain 評価の `knot` を iteration 100 / 200 / 300 で並べる。v7 s43 の延長は it100 53.7% -> it300 39.3% に下がった (1 シード)。
* 段 0 のアーカイブの形 (`view_archive.py --dump_pdata`): 行き止まりの形が入らないので、`raw`・`reduced` がどちらも表に無くラダー 0 の形は 0 のはず。
* `Curriculum/stage_log/success_s1`: v10 で 0.01〜0.02 に落ちた。v11 は段 1 -> 段 2 を v7 のままにしたので、v7 並み (0.55 前後) のはず。
* `Curriculum/stage_log/fail_s0`: 段 0 から行き止まりの形を作った手が −1 になるので、v7 より上がる見込み。
* `Curriculum/stage_log/restore_mismatch`: 0 に近いはず (初期アーカイブの形は v11 でも段が変わらない)。
