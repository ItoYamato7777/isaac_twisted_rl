# v2-R2: 報酬ラダーを R2 除去後の位相で判定する

```bash
uv run train --task Isaac-Twisted-RL-Twist-v2-R2 --headless          # MLP
uv run train --task Isaac-Twisted-RL-Twist-v2-R2-RNN --headless      # GRU
uv run train --task Isaac-Twisted-RL-Twist-v2-R2-LSTM --headless     # LSTM
uv run play  --task Isaac-Twisted-RL-Twist-Play-v2-R2 --num_envs 1   # 毎手 p-data を表示
```

ログ: `logs/rsl_rl/knot_tying_v2_r2{,_rnn,_lstm}/`

> **その後:** この版を 3 本の論文と比べた分析が [`paper/Report_Paper.md`](../../../../../../paper/Report_Paper.md) にある。
> 「詰まっている原因は報酬の細部ではなく定式化」という結論を受けて [v3](../v3/README.md) を作った。

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

### 実測

v2 の方策は、この偽の C2 (R2 除去後は輪 1 つ) に居座って `success_c2` と整形報酬を毎手稼いでいた。
v2 の `model_700.pt` を v2-R2 の Play 環境で再生した実測 (p-data の表・環境数・ランダム方策での `raw_c2` / `r2_c2`) は
実験記録 [`docs/experiments/2026-09-24_v2_fake-c2.md`](../../../../../../docs/experiments/2026-09-24_v2_fake-c2.md) にある。

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

## 発散した手の報酬 (2026-09-27 修正)

### 何が起きたか

最初の run は iteration 1 で `grasp_shaping` の罰が価値関数を壊し、**比較に使えない**
(run のフォルダと値は実験記録 [`docs/experiments/2026-09-27_v2r2_fix.md`](../../../../../../docs/experiments/2026-09-27_v2r2_fix.md))。

原因の連鎖 (ランダム方策 512 環境 x 24 手で実測):

1. 手の途中 (`close` 終了時) にロープが「有限だが巨大な座標」へ吹き飛び、
   把持距離が **97 m** と記録される (`nan_to_num` は NaN / Inf しか消さない)。
2. 手の終わりにはロープが NaN になり `diverged` で終了するが、同じ手の報酬は
   計算されるので `grasp_shaping` が 97 m ぶんの罰 (−0.05 x 97) を払う。
3. 発散した手では位相・高さも出鱈目なので、`shaping` や `success_*` も
   意味の無い値を払い得る。

正常な手の終わりは、xy が env 原点から最大 1.52 m、リンク速度が最大 3.3 m/s、
把持距離が最大 0.25 m だった。

### 何を変えたか (重み・正常な手の報酬は v2 と同一)

| 項目 | 変更 |
|---|---|
| 終了 `diverged` | NaN / 高さ 5 m に加え、**水平距離 3.0 m / リンク速度 20 m/s** を超えたら発散 |
| `grasp_shaping` | 距離を **直径の 15 倍 (0.30 m) で頭打ち**。発散した手は 0 |
| `shaping` | 発散した手は `Φ(s') = 0` (終端) として **−Φ(s)** を払う。0 にはしない (登った段を失う損失を残す) |
| `success_c1/c2`, `success`, `offladder`, `held_up` | 発散した手は 0 |
| `failure` | NaN だけでなく上の `diverged` 判定も失敗に数える (発散の罰はここに一本化) |
| 観測 `rope_link_pos` | 保険として `[-3, 3]` m にクリップ (次元は同じ) |
| ログ `div_cls*` | `diverged` と同じ基準で数える |

「発散した手」の判定は `mdp.diverged_now` で、終了項 `diverged` の関数と params を
そのまま呼び直す (閾値は `env_cfg.TerminationsCfg` の 1 か所だけ)。

**`termination_manager.get_term("diverged")` は使えない。** `_term_dones` は
「直近のエピソードを終わらせた項」を持ち続けるので、前のエピソードが発散で
終わった環境でも True を返す (実測: 本当の発散 3.4% に対して 17.5% が True)。

`common/mdp/` の各項には「既定値なら従来どおり」の引数 (`diverged_term`、
`max_distance_diameters`、`max_xy_distance`、`max_link_speed`) を足しただけなので、
v0〜v2 の挙動は変わらない。

### 確認

修正の後の確認 (ランダム方策 512 環境 x 24 手、PPO 3 iteration x 1024 環境) は
実験記録 [`docs/experiments/2026-09-27_v2r2_fix.md`](../../../../../../docs/experiments/2026-09-27_v2r2_fix.md) にある。

## この版で解決しないこと (済み: [v3](../v3/README.md))

本物の C2 (OU / UO) には **先端をどこかで紐の下にくぐらせる** 交差が要る。
pick&place は「掴んだ部分を上に置く」動作なので、先端側を動かすだけでは
くぐらせられない。**輪の側の紐を掴んで先端の上に被せる** 必要があり、
ランダム探索でここへ辿り着く確率は低い (1 手あたり約 0.3%)。
v2-R2 で偽の C2 という局所解は消えるが、C2 を学習できるかは別の問題になる。

→ この課題への対応が v3。台本でも cover (輪の側を被せる手) で固結びが結べることを確かめ
([`docs/experiments/2026-09-27_gate1.md`](../../../../../../docs/experiments/2026-09-27_gate1.md)。元は `next_action.md` §4.8.10)、到達した C1 / C2 からエピソードを始める
アーカイブを入れた。
