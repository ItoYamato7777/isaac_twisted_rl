# v6: v5 + 次にたどる位相の辺のリンクマスク (観測)

```bash
uv run train --task Isaac-Twisted-RL-Knot-v6 --headless           # 段ごとの MLP (学習はこれ)
uv run play  --task Isaac-Twisted-RL-Knot-Play-v6 --num_envs 16   # まっすぐなロープから段をつないで評価 (chain モード)
```

ログ: `logs/rsl_rl/knot_tying_v6/`。**学習には `Knot` プリセットを使う** (v3〜v5 と同じ)。

土台の v5 は、v4 (段ごとのエージェント `StageGatedActorCritic`・1 遷移 = 1 エピソード・段ごとのアーカイブ・
観測末尾の段の one-hot) に pick&place の発散対策を足した版 (`../v5/README.md`)。GRU / LSTM 版は v4 / v5 と同じく無い。

## なぜ作ったか

TWISTED-RL は下位の方策に「次にたどる高レベルの辺」を固定長のベクトル ρ(g) として渡している
(公開コード `exploration/preprocessing/preprocessor.py` の `preprocess_high_level_action`): under 区間と
over 区間のリンクの 0/1 マスク、`[left, over_first, over_before_under, sign]`、移動の種類。

v5 の観測 (`Knot` で 133 次元) にはリンク位置・交差特徴 (点の番号のみ)・段があるが、「区間 k がどの
リンクか」は無い。方策は、輪や先端がどのリンクかをリンク位置から自力で見つける必要があった。
v6 はこれを観測に足す。**変えた変数はこれ 1 つだけ** (報酬・終了・アーカイブ・PPO・行動は v5 のまま)。

## v5 から何を・なぜ変えたか

| 項目 | v5 | v6 | 理由 |
|---|---|---|---|
| 観測 `policy` (critic も同じ) | 133 次元 (末尾 4 が段の one-hot) | **197 次元**: 段の one-hot の **直前** に `next_edge` (64) | 上の「なぜ作ったか」。段の one-hot を末尾に残すのは、`StageGatedActorCritic` が末尾 4 列で段のエージェントを選ぶため (`next_edge` を最後に置くと、マスクの末尾を段と読んで振り分けが壊れる)。v2-R2 の項の位置は v5 と同じ |
| ログ | | `next_edge_stats` (報酬項・常に 0) と `next_edge_log` (カリキュラム項) | 方策がマスクを使っているかを見るため (下の「見るべき指標」)。報酬の値は変わらない |
| 報酬・終了・アーカイブ・PPO・行動 | | v5 のまま | 1 版 1 変数 |

観測の次元が変わるので、**v5 のチェックポイントからは `--resume` で暖機開始できない**
(`load_state_dict` は strict で、入力層の形が合わない)。`Twist` (20 リンク) では 105 + 48 + 4 = 157 次元。

`env_cfg.py` の `ObservationsCfg` は v5 (= v4) のものではなく v2-R2 のものを継承し、`next_edge` -> `knot_stage` の
順に宣言している。v4 の `PolicyCfg` を継承して `next_edge` を足すと、継承した `knot_stage` より後ろに並ぶため
(dataclass のフィールドは上書きしても基底の位置に残る)。

### `next_edge` の中身 (`common/mdp/observations.py` の `NextEdgeGoal`)

`[over マスク N, under マスク N, 種類 one-hot 4 (R1 / R2 / cross / 複合), sign, left, obu, 有効]` = 2N + 8。

* sign / left / obu は ±1、該当しないものは 0。TWISTED-RL は −1 を「該当なし」に使っているが、値の −1 と
  衝突するので採らなかった。
* R1 は over と under に同じマスクを入れる (TWISTED-RL・Yan2020 と同じ)。
* 有効フラグが 0 (表に無い状態: 交差 5 個以上・固結びに届かない・固結びそのもの) のときは全部 0。

### 辺の候補 (`common/topology/knot_graph.py` の `shortest_edges`)

* **最短の単純な辺**: R1 / R2 / cross のうち、子の BFS 距離 (`distance_to_overhand`) が親より 1 小さいもの。
* **複合の辺「端の置き直し」**: 頭の端 (点 1) か尻尾の端 (点 2n) の交差を 1 つ外し、**同じ端で** cross を
  1 つ打つ。子の距離が親より小さいものだけ。グラフは交差を増やす移動しか持たないが、pick&place で端を
  掴んで持ち上げると、その端に乗った交差がいったん消えてから置き直される。ゲート 1 の台本の tuck
  (cover 後 -> 固結び) はこの形で、グラフの単純な辺では表せない。

**BFS の距離と報酬は変えていない。** 複合の辺は観測の候補にだけ入る。

### 区間 -> リンク (`common/topology/next_edge.py`)

ポリラインの節点は N+1 個なので、セグメント k = リンク k。点 p が乗っているリンクを L(p)、L(0) = 0、
L(2n+1) = N−1 とし、区間 `(a, b)` のリンク範囲を `[L(a), L(b)]` (両端を含む) とする (TWISTED-RL の
`get_link_segments` と同じ)。複合の辺は、外した交差の 2 点を飛ばして区間を数え直す。

実行時は p-data 文字列を作らず、`batch_keys` の int64 キーで GPU 上の表を `searchsorted` で引く
(`BfsPotential` と同じ方式)。

### 辺を保持する期間

選んだ辺は環境ごとに保持し、**reset のときと、R2 除去前の p-data のキーが変わったときだけ選び直す**。
同じ位相のまま形を整える手のあいだに目標が揺れないようにするため (TWISTED-RL の 1 遷移 = 1 目標と同じ)。
**リンクマスクは毎手、今の形状から作り直す** (区間の境界は位相が同じでも動くため)。

reset では「選び直す」印を付けるだけにしている。`_reset_idx` は `event (アーカイブ復元) ->
observation.reset -> action.reset (invalidate_extractor)` の順で、観測の計算はその後なので、
アーカイブから始めた env も復元後の形状で選び直される。

## 台本との照合 (ゲート 1 の台本 `demos/overhand_script.py`)

台本 (`--task Isaac-Twisted-RL-Knot-v2-R2 --num_envs 128 --jitter 0.25`) の p-data の遷移を辺の候補で分類し、
辺の選び方の規則ごとに台本の次の p-data との一致率を数えた。分類の件数・規則ごとの一致率・外れの内訳は
実験記録 [`docs/experiments/2026-10-01_v6_script-edge-match.md`](../../../../../../docs/experiments/2026-10-01_v6_script-edge-match.md) にある。
台本の遷移は、最短の単純な辺と最短の複合の辺 (tuck) でほぼ全部表せた。「台本は短い側 (尻尾) の端を使う」という
仮説 (`short_end`) は、実際の形状では逆だった。

**既定は `long_end` にした** (`env_cfg.py` の `NEXT_EDGE_RULE`)。`tail_end` のほうが 1.6 ポイント高いが、
これは台本が尻尾の端から操作するように作られていることに合わせただけで、ロープの形とは関係がない。
`long_end` は「掴んで別の区間の上下へ運ぶだけの自由な長さがある端を使う」という物理的な意味を持つ。

## 見るべき指標

* 主指標は v5 (= v4) と同じ **`Curriculum/stage_log/success_s{k}` と `chain_est`**、Play (chain モード) では
  `Episode_Termination/stage_up` (まっすぐなロープから固結びに達した割合)。v5 の同じ時刻の値と比べる。
* `Curriculum/next_edge_log/*` (1 手あたり・全環境平均。`reached` 〜 `invalid` の 6 つは足すと 1):

| key | 意味 |
|---|---|
| `reached` | 手を打つ前に選んだ辺の子 (複合なら最終の子) に着いた |
| `other_up` | 選んだ辺とは別の手で段が上がった |
| `same` / `down` | 段が変わらなかった / 下がった |
| `diverged` | 辺はあったが手の途中で発散した |
| `invalid` | 有効な辺が無かった (交差 5 個以上・届かない・固結び) |
| `kind_r1` 〜 `kind_end` | 選んだ辺の種類の割合 |

`reached / (reached + other_up)` が学習とともに上がれば、方策はマスクを使って目標の辺をたどっている。

**学習ログの段ごとの成功率と `chain_est` は、本番の成功率を表さない** (下の「学習の結果」)。版の比較は
`scripts/rsl_rl/eval_chain.py` の chain 評価で行うこと。

## 学習の結果 (2026-10-01〜02)

v4 / v5 と同じ条件で 2 シード (seed 42 / 43) 学習した結果と chain 評価は
[`docs/experiments/2026-10-01_v4-v6_overnight.md`](../../../../../../docs/experiments/2026-10-01_v4-v6_overnight.md) (数値の正本)。次にたどる辺の観測は
v5 に対して大きく効いたが、v4 との差はシード差の範囲だった。run のフォルダ名は
[`docs/experiments/2026-10-01_v4-v7_readme-notes.md`](../../../../../../docs/experiments/2026-10-01_v4-v7_readme-notes.md)。

### v6 の切り分け (2026-10-02)

v6 に 1 つずつ変更を入れた 4 本 (seed 42、iteration 70 で停止。`logs/rsl_rl/knot_tying_v6_ablation/`) の結果は、同じ報告書の
末尾の追記にある。めり込み解消速度を戻すと学習ログでは段 2 を v4 並みに学べるが、chain 評価では v6 との差が出なかった。
v7 は [`../v7/README.md`](../v7/README.md)。

## 規則を変えて比べる

```bash
uv run train --task Isaac-Twisted-RL-Knot-v6 --headless env.observations.policy.next_edge.params.rule=first
```

規則は `topology.NEXT_EDGE_RULES` (`short_end` / `long_end` / `tail_end` / `first` / `random`)。
