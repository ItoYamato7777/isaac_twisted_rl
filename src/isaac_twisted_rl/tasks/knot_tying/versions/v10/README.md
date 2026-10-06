# v10: v7 + 段を固結びへの正規の経路の上の p-data だけに絞る

```bash
uv run train --task Isaac-Twisted-RL-Knot-v10 --headless           # 段ごとの MLP
uv run play  --task Isaac-Twisted-RL-Knot-Play-v10 --num_envs 16   # まっすぐなロープから段をつないで評価 (chain モード)
```

ログ: `logs/rsl_rl/knot_tying_v10/`。**学習には `Knot` プリセットを使う** (v3〜v9 と同じ)。継承元は **v7** (v8・v9 ではない)。
仕様と理由の正本は [`docs/decisions/0023-v10-path-stages.md`](../../../../../../docs/decisions/0023-v10-path-stages.md)。ここには配線と、見るべき指標を書く。

**学習の前に初期アーカイブを作る** (1 回だけ。下の「初期アーカイブ」)。

## なぜ作ったか

v7 の段は Φ を丸めたもので、Φ の表に無い形は段 0 になる。v7 seed 43 を iteration 330 まで延ばすと、方策の失敗で作った形
(交差 4 個・表に無い・たどる辺なし) が段 0 のアーカイブに溜まり、chain 評価は it100 53.7% -> it300 39.3% に下がった (1 シード)。
段の形を正規の経路の上に限れば、アーカイブに入る形も限られる、という見込みで作った (このループの説明は推測。確かめるのがこの版の目的)。

## v7 から何を・なぜ変えたか

| 項目 | v7 | v10 | 理由 |
|---|---|---|---|
| 段の定義 (`stage_definition`) | `bfs` (`BfsPotential.stage` = round(3Φ)。属性は持たない) | **`path`** (`topology.PathStage`) | 段 0〜2 を正規の経路の上の 7 個の p-data に限る |
| 初期アーカイブ (`events.reset_to_stage.params["load_paths"]`) | `overhand_{num_links}links.pt` | **`overhand_{num_links}links_v10.pt`** | 上のファイルのうち、新しい段の定義で段が変わらない形だけ (段の定義を変えた結果) |
| それ以外 (観測の形・報酬の値・アーカイブの容量・`stage_weights` 1:1:1・`straight_prob`・行動・物理・PPO・Play) | | v7 のまま | 1 版 1 変数 |

v7 との差がこの 2 つだけであることは `tests/test_v10_cfg.py` が静的に検査する。段の定義そのもの (表と経路) は
`tests/test_path_stages.py`。

### 段の定義 `path` (`common/topology/path_stages.py`)

| 段 | 生の p-data (R2 除去の前) |
|---|---|
| 0 | `` (交差なし) |
| 1 | `1O2-_2U1-`、`1U2+_2O1+` |
| 2 | `1O3-_2U4-_3U1-_4O2-`、`1O3+_2U4+_3U1+_4O2+` (論文 Fig.1)、`1U3-_2O4-_3O1-_4U2-`、`1U3+_2O4+_3O1+_4U2+` (台本) |
| 3 | `BfsPotential.stage` >= 3 (**v7 と同じ判定**) |
| −1 | 上のどれでもない形 (**道から外れた形**) |

段を引く項は、すべて `common/mdp/observations.py` の `stage_function(env)` を通る (env の設定の `stage_definition`。
無ければ `bfs`)。v10 で何が変わるか:

| 項 | v10 での挙動 |
|---|---|
| 報酬 `stage_transition_reward`・終了 `stage_up` / `stage_down` | 段が上がれば +1。**道から外れた手は段 −1 で、手の前の段 (0〜2) より下がるので −1 で終了** (`stage_down`)。段 2 から、経路の終点ではないもう 1 つの固結びに着いた手も段 3 なので +1 |
| 観測 `knot_stage` (段の one-hot) | −1 は段 0 にする。学習では外れたら終わるので、効くのは chain 評価だけ (外れている間は段 0 の方策が打つ) |
| 観測 `next_edge` | 候補を「子が正規の経路の上にある辺」に絞る (段 0 で 2 本、段 1 で各 2 本、段 2 で各 1 本 = 経路の終点へ)。経路の上に無い形では辺なし (全部 0)。次元は v7 と同じ |
| アーカイブの保存 `record_stage_archive` | 段 0〜2 の形だけ (段 −1 と段 3 は `StateArchive.add` が捨てる) |
| アーカイブの復元の検算 `restore_mismatch` | 新しい段の定義で計算し直した段と比べる |
| ログ `NextEdgeStats` | 段の比較に新しい定義を使う (外れた手は `down`) |
| Play (chain) | 固結びの判定は v7 と同じ (`after >= STAGE_GOAL`)。外れても 18 手まで続ける |

### 初期アーカイブ

ファイルには段のラベルと関節角しか無く、p-data は形を物理に書き込まないと分からない。v4〜v9 の初期アーカイブを v10 の Play 環境に
書き込み、計算し直した段がラベルと一致した形だけを新しいファイルに書く (物理は進めないので、残る形は元とまったく同じ):

```bash
uv run python src/isaac_twisted_rl/scripts/rsl_rl/view_archive.py --headless \
    --archive outputs/archive_seed/overhand_28links.pt --task Isaac-Twisted-RL-Knot-Play-v10 \
    --write_filtered outputs/archive_seed/overhand_28links_v10.pt
```

0023 の見込みでは、段 0 の 128 形・段 1 の 264 形 (すべて `1U2+_2O1+`)・段 2 の 18 形 (`1U3+_2O4+_3O1+_4U2+`) が残り、段 2 の 112 形が外れる。
経路 A・A' (負の符号) の形は初期アーカイブに無い。ファイルが無いと `ResetToStage` は警告して飛ばし、まっすぐなロープだけから始まる。

## 見るべき指標

版の比較は chain 評価で行う ([`docs/system/evaluation.md`](../../../../../../docs/system/evaluation.md))。v7 の既存の run
(seed 43 / 44、iteration 70 / 100 の `eval_chain_it{N}.json`) と比べる。学習ログは代理の指標で、版の良し悪しの結論には使わない。

* chain 評価の `knot` (v7 と同じ判定)。差は v7 のシード差 (iteration 100 で約 6 ポイント) と標準誤差 (約 2 ポイント) を超えるかで読む。
* chain 評価の `knot_strict` (経路の終点の固結びに着いた割合。`eval_chain.py` が全部の版で出す)。v7 も同じ列で測り直せる。
* `Curriculum/stage_log/restore_mismatch`: 初期アーカイブと学習中に保存した形が新しい定義で復元できているか (0 に近いはず)。
* `Curriculum/stage_log/fail_s{k}`: 段 k から道を外れた (または下がった) 割合。v7 の段では「同じ段のまま」だった手の一部が −1 になるので、
  v7 より高くなる見込み。`success_s{k}` との兼ね合いで見る。
* `Curriculum/next_edge_log/invalid`: たどる辺が無い手の割合。v7 の延長の run では 8% -> 44% に増えた。v10 では、学習中の観測は
  経路の上の形だけなので、0 に近いはず。
* `Curriculum/stage_log/success_s0`・`straight_success`: R1 の向き (符号) によっては道から外れるので、段 0 の成功が v7 (0.97) から下がりうる。
