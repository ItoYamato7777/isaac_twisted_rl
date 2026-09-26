# v2-R2: 報酬ラダーを R2 除去後の位相で判定する

```bash
uv run train --task Isaac-Twisted-RL-Twist-v2-R2 --headless          # MLP
uv run train --task Isaac-Twisted-RL-Twist-v2-R2-RNN --headless      # GRU
uv run train --task Isaac-Twisted-RL-Twist-v2-R2-LSTM --headless     # LSTM
uv run play  --task Isaac-Twisted-RL-Twist-Play-v2-R2 --num_envs 1   # 毎手 p-data を表示
```

ログ: `logs/rsl_rl/knot_tying_v2_r2{,_rnn,_lstm}/`

## 何が問題だったか (v2)

固結びは、交差の無い紐から **R1 -> cross -> cross** の 3 手で作れる:

| 段 | 操作 | 位相 (先端側の上下) |
|---|---|---|
| C1 | R1 で輪を作る | 交差 1 つ |
| C2 | 先端を輪の中へ **くぐらせる** | 2 交差・交互 (OU / UO) |
| C3 | 先端を輪の反対側から外へ | 固結び (OUO / UOU) |

v2 の C2 判定 `is_interleaved_two` は **部分列判定** で、「交差のうち 2 つを
取り出して交互・同符号になる組があるか」しか見ていなかった。そのため
**先端で輪を上 (または下) から丸ごと跨いだだけ** の 3 交差状態が C2 と誤判定される:

```
1O4+_2O5-_3O6+_4U1+_5U2-_6U3+     <- 交差 A (1,4) と C (3,6) が交互・同符号の組
  R2 で B-C (または A-B) を消すと
1O2+_2U1+                         <- 実体は輪 1 つ (C1)
```

### 実測 (v2 の `model_700.pt` を v2-R2 の Play 環境で再生, 9 環境 x 約 11 エピソード)

| 手 | p-data (R2 除去前) | 判定 (v2) | R2 除去後 |
|---|---|---|---|
| 1〜2 | `1O2+_2U1+` | class 1 | class 1 |
| 3〜6 | `1O4+_2O5-_3O6+_4U1+_5U2-_6U3+` | **class 2** | **class 1** `1O2+_2U1+` |

全環境・全エピソードで同じ動きだった (決定論的方策)。v2 の方策は
**輪を作り、3 手目で先端を輪の下に丸ごと通して、そのまま居座る** ことを
学習しており、これが `success_c2` (+1.5) と整形報酬 (0.35 -> 0.70) を
毎手稼いでいた。Play で「先端が輪に入っていない」と見えたのはこのため。

ランダム方策 (学習開始直後) でも `raw_c2` 0.029 に対して `r2_c2` は 0.000 で、
**v2 の C2 発火は最初から全て偽物だった** (16 環境 x 2 iteration の smoke test)。

## v2 から何を・なぜ変えたか

| 項目 | v2 | v2-R2 | 理由 |
|---|---|---|---|
| ラダーの判定対象 | 交差リストそのまま | **R2 で消せる交差の組を除いた後** (`topology.reduce_r2`) | 上記の偽 C2 を消す |
| `shaping` | `reduce_r2` なし | `reduce_r2=True` (前後とも) | Φ(s), Φ(s') を同じ判定器で測らないとポテンシャル整形にならない |
| `success_c1` | 生の交差数 ≥ 1 | R2 除去後の交差数 ≥ 1 | 先端を輪の上に置いただけ (OO, 実体は交差 0) に +0.5 を払わない |
| `success_c2` / `success` | 部分列判定 | R2 除去後に部分列判定 | 本題。固結びは交代結び目なので R2 除去で消えない |
| `offladder` | 生の交差数 ≥ 2 かつ class 0 | R2 除去後で判定 | OO (実体は交差 0) を「行き止まり」として罰しない |
| 観測 `knot_class` | 生の段 | R2 除去後の段 (次元は同じ 4) | 報酬と観測で段がずれると方策が区別できない |
| `diverged_class0..3` (重み 0) | あり | **削除** | 重み 0 の項は `RewardManager` が呼ばない (下記) |
| `topology_stats` + カリキュラム項 | なし | **追加** | 偽 C2 の率などを TensorBoard に出す |

重み・γ・エピソード長・フェーズの尺・PPO 設定は **v2 と完全に同一**
(比較のため、変えるのは判定器の 1 変数だけ)。

### 各状態の段 (単体テスト `tests/test_r2_reduce.py` で全通り検証済み)

| 状態 (先端側の上下) | v2 | v2-R2 |
|---|---|---|
| 輪 1 つ (O / U) | 1 | 1 |
| 先端を輪の中へくぐらせた (OU / UO) | 2 | 2 |
| 先端を輪の中へ上から置いただけ (OO / UU) | 0 + `offladder` 罰 | 0 (交差 0) |
| 輪を上/下から跨いだだけ (OOO, OOU, OUU, UOO, UUO, UUU) | **2** | **1** |
| 固結び (OUO / UOU) | 3 | 3 |

## R2 除去の判定

交差 X = (p, q) と Y = (r, s) (p, q, r, s は紐の始点からの交点番号) が

* 片方の紐上で隣り合い (|p − r| = 1)、もう片方でも隣り合い (|q − s| = 1)
  (p↔s, q↔r の対応も見る)、
* その隣り合った 2 点で **同じ紐が両方とも上**

なら二角形を成すので、R2 で 2 つとも消す。消せなくなるまで繰り返す。
上下が交互の組は決して消さないので、固結び (交代結び目) は必ず残る。
実装はテンソル層のみ (`common/topology/knot_detect.py`)。詳細は同ファイルのコメント。

## ログ (TensorBoard: `Curriculum/topology_log/*`)

| key | 意味 |
|---|---|
| `raw_c2` / `r2_c2` / `fake_c2` | R2 除去前 / 後の class 2 率、除去で降格した率 |
| `raw_c3` / `r2_c3` / `fake_c3` | 同じく class 3 |
| `r2_removed` | R2 で消えた交差数の平均 |
| `raw_count` / `r2_count` | 除去前 / 後の交差数の平均 |
| `r2_cls0..3` | R2 除去後のクラス分布 |
| `div_cls0..3` | 手を始めた時点のクラス別の発散率 (v2 の `diverged_class*` の代わり) |

**学習がうまく行っているかは `r2_c2` → `r2_c3` の立ち上がりで見る。**
`fake_c2` が再び増えるなら、判定の新しい抜け道を疑うこと。

`Episode_Reward/topology_stats` は常に 0 と表示されるが正しい (統計を積むだけの項)。

### Play 時の p-data 表示

`-Play-` 付きのタスクでは、先頭 9 環境 (`--num_envs` がそれより少なければ全環境)
について毎手次の形で標準出力に出す:

```
[topology] env=0 step=3 raw(class=2) 1O4+_2O5-_3O6+_4U1+_5U2-_6U3+ | R2後(class=1) 1O2+_2U1+  <- R2で降格
```

v2 のチェックポイントも `--checkpoint` で直接読める (観測次元が同じため):

```bash
uv run play --task Isaac-Twisted-RL-Twist-Play-v2-R2 --num_envs 1 \
    --checkpoint logs/rsl_rl/knot_tying_v2/2026-09-24_14-07-21/model_700.pt
```

## 副次的に見つかった v2 のバグ: 重み 0 の報酬項は呼ばれない

Isaac Lab の `RewardManager.compute()` は重み 0 の項を **スキップする**
(`isaaclab/managers/reward_manager.py` の "skip if weight is zero")。
v2 の `diverged_class0..3` は重み 0 で登録していたので一度も計算されておらず、
TensorBoard にも出ていない。v2 の設定は比較のため変えず、`rewards.diverged_by_class`
の docstring に警告を追記した。v2-R2 では `div_cls*` として出している。

## この版で解決しないこと (次の課題)

本物の C2 (OU / UO) には **先端をどこかで紐の下にくぐらせる** 交差が要る。
pick&place は「掴んだ部分を上に置く」動作なので、先端側を動かすだけでは
くぐらせられない。**輪の側の紐を掴んで先端の上に被せる** 必要があり、
ランダム探索でここへ辿り着く確率は低い (smoke test の `r2_c2` は 0)。
v2-R2 で偽の C2 という局所解は消えるが、C2 が学習できるかは別問題なので、
`r2_c2` が 0 のまま張り付くなら、C1 からの初期化 + 台本デモによる C2 の
足場作り (カリキュラム) を検討すること。
