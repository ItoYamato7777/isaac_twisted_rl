# v6: v5 (v4 ベース) + 次にたどる位相の辺のリンクマスク (観測)

```bash
uv run train --task Isaac-Twisted-RL-Knot-v6 --headless           # 段ごとの MLP (学習はこれ)
uv run play  --task Isaac-Twisted-RL-Knot-Play-v6 --num_envs 16   # まっすぐなロープから段をつないで評価 (chain モード)
```

ログ: `logs/rsl_rl/knot_tying_v6/`。**学習には `Knot` プリセットを使う** (v3〜v5 と同じ)。

土台の v5 は、v4 (段ごとのエージェント `StageGatedActorCritic`・1 遷移 = 1 エピソード・段ごとのアーカイブ・
観測末尾の段の one-hot) に pick&place の発散対策を足した版 (`../v5/README.md`)。v5 を v3 ベースから v4 ベースに
作り直したのに合わせて (2026-10-01)、v6 も観測の並び・報酬・ログ・Play をそれに追従させた。GRU / LSTM 版は
v5 と同じく無い。

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

### どの辺をたどっているか

台本のログ (`outputs/overhand_script/knot_carry020/envs.log`、128 環境) から、p-data が変わった遷移を
すべて分類した:

| 分類 | 件数 |
|---|---|
| (a) 最短の単純な辺 | 279 |
| (b) 最短の複合の辺 (tuck) | 99 |
| (c) 最短ではないグラフの辺 | 0 |
| (d) どれでもない (cover が崩れた失敗) | 4 |

他の `Knot` の台本ログでも、固結びに近づいた遷移 3449 件のうち (a) 2760、(b) 640、(d) 49 (1.4%)。
(d) の大半は、余分なねじれを外しながらの R2 cover (21 件) と、1 手で交差が 2 つ増える手。外す交差は
1 つで足りる。

### 辺の選び方と台本の一致率 (実際の形状)

同じ台本 (`--task Isaac-Twisted-RL-Knot-v2-R2 --num_envs 128 --jitter 0.25`) を回し直して各手の
交差リストを保存し、台本の次の p-data が選んだ辺の子と一致する割合を数えた (遷移 378 件、固結び 123/128):

| 規則 | 一致率 | 中身 |
|---|---|---|
| `short_end` | 34.7% | 端の区間のリンク数が少ない候補を優先 |
| **`long_end` (既定)** | **93.4%** | 端の区間のリンク数が多い候補を優先 |
| `tail_end` | 95.0% | 尻尾の端 (リンク番号の大きい側) を使う候補を優先 |
| `first` | 33.9% | 列挙順の最初 |
| `random` | 31.2% | 候補から一様に |

同点の崩し方はどれも「種類 R1 > R2 > 複合 > cross、次に列挙順」。辺を選ぶ時点を「遷移の直前」にしても
「そのキーに初めて着いた時点で選んで保持」(v6 の挙動) にしても、一致率は同じだった。

`short_end` は「台本は短い側 (尻尾) の端を使う」という仮説で作ったが、**実際の形状では逆だった**。
台本の輪は頭側のリンク 5 付近で本体と交差するので、輪の状態の端の区間は頭側 6 本・尻尾側 11 本
(中央値) で、台本は長い側 (尻尾) の端を使っている。cover 後の状態でも、尻尾の置き直しの区間
(点 5 から端まで) は頭の端の区間 (3 本) より長い。

**既定は `long_end` にした** (`env_cfg.py` の `NEXT_EDGE_RULE`)。`tail_end` のほうが 1.6 ポイント高いが、
これは台本が尻尾の端から操作するように作られていることに合わせただけで、ロープの形とは関係がない。
`long_end` は「掴んで別の区間の上下へ運ぶだけの自由な長さがある端を使う」という物理的な意味を持つ。

`long_end` の外れ 25 件 (6.6%) の内訳:

| 件数 | 遷移 | `long_end` が選んだ辺 |
|---|---|---|
| 18 | 輪 -> 本物の C2 | R2 cover。どちらも最短の辺で、決定的な規則では片方しか選べない |
| 6 | 本物の C2 -> 固結び | 頭の端の `cross(0, 2)`。C2 の端の区間は頭 3 本・尻尾 4 本 (中央値) と近く、頭の端のほうが長い環境がある。選んだ辺も固結び (逆の走査方向) に着く |
| 1 | cover 後 -> 輪 | 尻尾の置き直し。cover が崩れた失敗 |

## 動作確認 (2026-10-01、`Knot`、RTX 3090 1 枚、**v3 ベースだったときの v6**)

この節は v5 / v6 を v4 ベースに作り直す前の確認 (観測は 129 + 64 = 193 次元、報酬・終了は v3) で、
今の v6 (197 次元、段ごとのエージェント) ではまだ回していない。

| 項目 | 結果 |
|---|---|
| 観測の次元 | 193 (`next_edge` 64 が末尾) |
| smoke test の iteration 時間 (4096 env、3 iteration) | v6: 128 / 170 / 174 s、v5: 170 / 170 / 177 s (同じ GPU で続けて実行)。増えていない |
| 有効フラグが 1 の割合 | 学習の手で 97.6〜99.8% (`next_edge_log/invalid`)、乱数の手 512 env x 8 手で 96〜100% |
| NaN | 観測は全部有限 (乱数の手 512 env x 8 手) |
| マスク | 有効な env では over / under とも連続した区間 |
| 保持と選び直し | reset (アーカイブからを含む) の直後も含め、保持しているキーが毎手、今の状態のキーと一致 |

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

## 規則を変えて比べる

```bash
uv run train --task Isaac-Twisted-RL-Knot-v6 --headless env.observations.policy.next_edge.params.rule=first
```

規則は `topology.NEXT_EDGE_RULES` (`short_end` / `long_end` / `tail_end` / `first` / `random`)。
