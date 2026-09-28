# v3: 論文の中核 (到達状態からのリセット・成功で終了・BFS 距離) を取り込む

```bash
uv run train --task Isaac-Twisted-RL-Knot-v3 --headless          # MLP (学習はこれ)
uv run train --task Isaac-Twisted-RL-Knot-v3-RNN --headless      # GRU
uv run train --task Isaac-Twisted-RL-Knot-v3-LSTM --headless     # LSTM
uv run play  --task Isaac-Twisted-RL-Knot-Play-v3 --num_envs 1   # まっすぐなロープから評価 (p-data を毎手表示)
```

ログ: `logs/rsl_rl/knot_tying_v3{,_rnn,_lstm}/`

**学習には `Knot` プリセットを使う。** `Twist` (20 リンク) は長さが足りず、台本でも
固結びがほぼ結べない (ゲート 1、`next_action.md` §4.8.10)。

## なぜ作り直したか

[`paper/Report.md`](../../../../../../paper/Report.md) の結論は、v1〜v2-R2 が詰まっているのは
報酬の細部ではなく **定式化** のせい、というもの。TWISTED の SAC+HER ベースラインと
同じ壁 (1 交差は解けるが 2 交差でほぼ 0) に当たっている。論文 (TWISTED /
TWISTED-RL) はこの壁を次の 3 つで越えている。v3 はこれを Report §7 P1 として取り込む:

| # | 論文の仕組み | v2-R2 の問題 | v3 |
|---|---|---|---|
| P1-1 | 到達した形状からエピソードを始める (TWISTED-RL §IV-D、Go-Explore) | 毎回まっすぐなロープから始めるので、C2 を練習できるのは偶然 C1 に着いた後の残りの手だけ (§5.3) | 到達した C1 / C2 の形状を保存し、reset の半分をそこから始める |
| P1-2 | 目標に達したら +1 で終了 | 段のボーナスを **毎手** 払うので、C1 に居座るのが局所解。判定の抜け道も毎手悪用される (§5.5) | 固結びで終了。ボーナスは段を初めて上がった手にだけ払う |
| P1-3 | p-data グラフの BFS で計画する | 4 クラスのラダーでは、ゲート 1 の cover の手 (R2 除去後は C1 のまま) が評価されない | ポテンシャルを `max(BFS 距離, ラダー)` にする |
| P1-4 | — | 8 スロット打ち切り (§6.3)、C1 の定義の不一致 (§6.4) | 修正 |

加えて、ゲート 1 が「学習に使うなら v3 で配線すること」とした 3 点を入れる。

## v2-R2 から何を・なぜ変えたか

| 項目 | v2-R2 | v3 | 理由 |
|---|---|---|---|
| ロープ (学習用) | `Twist` 20 リンク | **`Knot` 28 リンク** | ゲート 1: `Twist` では台本でも結べない |
| 1 手の尺 | 2.44 s (`SHORT_DURATIONS`) | **3.16 s** (`KNOT_DURATIONS`: descend / lift / lower 0.3 s, carry 0.6 s。759 物理ステップ) | ゲート 1: `Knot` を速く運ぶと 1 手目で 13/32 が発散 |
| 置き先の範囲 `drop_xy_rel_frac` | ±0.15 L | **±0.30 L** | ゲート 1: cover の置き先が届かず、6 次元行動の再生が 9% -> 66% |
| 手数 | 6 | **12** | 成功で終了するので長くしても居座れない |
| 初期状態 | 常にまっすぐ | **確率 0.5 でアーカイブ (C1:C2 = 1:2)** | P1-1 |
| 終了 | 時間切れ・発散 | + **`knot_tied` (固結び)** | P1-2 |
| 疎報酬 | `success_c1/c2/success` を毎手 | **`ladder_bonus`**: 段を初めて上がった手に +0.5 / +1.5 / +3.0 | P1-2 |
| 整形 `shaping` | ラダー (0, 0.35, 0.70, 1) | **`max(BFS 距離, ラダー)`** | P1-3 |
| `offladder` | 交差 ≥ 2 かつ class 0 に −0.05 | **外す** | BFS では独立したねじれ 2 つも輪 1 つと同じ d = 2 で、行き止まりではなかった |
| 段の判定 | 項ごとに `count >= 1` / `count == 1` が混在 | すべて R2 除去後の `topology_class` | P1-4 (§6.4) |
| 観測・PPO・γ・発散の扱い | | v2-R2 のまま | |

共有層の変更 (過去バージョンへの影響):

* **§6.3 の修正は全バージョンに効く** (バグ修正のため)。`has_knot_subsequence` は
  8 スロットに打ち切る前に gap 1 の交差を後ろへ回す。gap 1 の交差はどの部分列でも
  条件を満たさないので結果は変わらず、交差 9 個以上での見落としだけが消える
  (`tests/test_knot_detect.py`)。
* それ以外は追加のみ。既定値なら従来と同じ挙動 (`knot_class_shaping(potential=...)`、
  `KnotTyingEnvCfg.grasp_limits`、`_apply_rope` が終了項の `rope_diameter` も埋める、など)。

## P1-1 到達状態アーカイブ

`common/state_archive.py` (純 torch) と `common/mdp/archive.py` (Isaac Lab との接続)。

* **保存** (`record_archive`、報酬項・常に 0): 毎手の終わりに、床に落ちていて
  発散していない形状を、R2 除去後の class ごとのリングバッファ (各 4096 件) へ入れる。
  class 1 と 2 だけ (固結びから始めても学ぶことが無い)。root の姿勢 (xy はロープ
  重心からの相対) と関節角を持つ。
* **復元** (`reset_from_archive`、event): 基底の `reset_all` の後に、確率 0.5 で
  C1 : C2 = 1 : 2 の重みで 1 件選び、重心を env 原点に置いて z 軸まわりに一様乱数で
  回して書き込む (回しても p-data は変わらない)。直後に FK で位相を計算し直し、
  保存時の class と一致したかを `restore_mismatch` に数える。
* アーカイブは学習の最初は空で、ランダムな手が C1 / C2 に着いた時点から埋まる
  (v2-R2 の実測で本物の C2 は 1 手あたり約 0.3%)。

## P1-2 成功で終了 / 遷移ベースの疎報酬

`LadderProgressBonus` は env ごとに「このエピソードで到達した最高段」を持ち、それを
超えた手にだけ `bonuses[best+1..after]` の合計を払う。アーカイブから C2 で始めた
エピソードは、固結びの +3.0 しかもらえない。段を落として登り直しても 2 度目は払わない。

整形項は成功の手でも `Φ(s')` を実際の値 (1.0) のまま払う (終端を 0 にすると、固結びを
作った手に `−Φ(s)` の罰が乗るため)。

## P1-3 BFS 距離のポテンシャル

`common/topology/knot_graph.py`。移植元の R1 / R2 / cross (`representation.py` に移植) で
自明な状態から **交差 4 個まで** BFS し (8541 状態、約 20 s、起動時に 1 回)、各 p-data から
「R2 除去後に固結びを含む状態」までの最短手数 d を表にする。
`Φ = max(1 − d/3, R2 除去後のラダーの class / 3)`。

| 状態 | ラダー | BFS |
|---|---|---|
| 自明 / 輪 1 つ / 本物の C2 / 固結び | 0 / 1/3 / 2/3 / 1 | 同じ |
| **cover 後** (`1U4+_2O5+_3O6-_4O1+_5U2+_6U3-` など、R2 除去後は C1) | 1/3 | **2/3** |
| 輪の上に先端を置いただけ・独立したねじれ 2 つ (R2 除去後は class 0) | 0 | **1/3** |
| 偽の C2 (輪を跨いだだけ) | 1/3 | 届かない -> ラダーの 1/3 |

### 実装中に分かったこと (Report の想定との違い)

* 最初は Report どおり **R2 除去後の p-data を交差 3 個まで** BFS したが、固結びに届く
  p-data は 13 個 (自明・輪 4・C2 4・固結び 4) だけで、**経路上ではラダーと完全に同じ値**
  だった。cover 後の状態は R2 除去前の p-data で見て「tuck 1 手で、R2 除去すると固結びに
  なる 4 交差の状態へ行ける」ので、除去前の p-data で 4 交差まで見て初めて拾える。
* 輪 1 つの 4 通りはどれも d = 2。Report §5.4 の「次の手が打てる C1 / 打てない C1」の
  差は p-data には現れない (幾何の差)。
* Report が挙げた「掴んだ側が上になる」制約を辺に入れる案は採らなかった。
  この制約はどちらの区間を掴むかを決めるだけで、位相の辺は減らない
  (先端が下になる cross も cover で実行できる)。
* BFS 単独ではなく max を取るのは、交差 4 個で打ち切るため 4 交差の途中状態や
  尻尾に余分な輪が付いた状態が「届かない」(0) になり、ラダーより下がって
  「ねじれを足すと値が上がる」逆転が生じるため。

## ログ (TensorBoard)

| key | 意味 |
|---|---|
| `Curriculum/archive_log/scratch_success` | **主指標。** まっすぐなロープから始めたエピソードが固結びで終わった割合 |
| `Curriculum/archive_log/archive_success` | アーカイブから始めたエピソードの成功率 (途中から始まるので高めに出る) |
| `Curriculum/archive_log/scratch_episodes` | 前回のログから終わった、まっすぐ始まりのエピソード数 |
| `Curriculum/archive_log/archive_frac` | reset のうちアーカイブから始めた割合 (アーカイブが埋まれば `prob` = 0.5 に近づく) |
| `Curriculum/archive_log/arc_size_c1` / `arc_size_c2` | アーカイブの件数 (上限 4096) |
| `Curriculum/archive_log/restore_mismatch` | 復元した形状の位相が保存時と違った割合。**0 に近くないなら復元が壊れている** |
| `Episode_Termination/knot_tied` | reset のうち固結びで終わった割合 (開始の種別は混ざる) |
| `Episode_Reward/ladder_bonus` | 段を上がったボーナス |
| `Curriculum/topology_log/*` | v2-R2 と同じ (`r2_c2` / `r2_c3` / `fake_c2` など)。`r2_trunc` を追加 (部分列探索の打ち切りが効いた率) |

`Episode_Reward/record_archive` と `Episode_Reward/topology_stats` は常に 0 と表示されるが正しい。

## 切り分け (Hydra の CLI 上書き)

v3 は 4 項目を同時に入れたので、効果の切り分けは CLI で 1 つずつ外して比べる:

```bash
# P1-1 を外す (常にまっすぐなロープから)
uv run train --task Isaac-Twisted-RL-Knot-v3 --headless env.events.reset_from_archive.params.prob=0.0
# P1-2 の終了を外す (時間切れまで続ける)
uv run train --task Isaac-Twisted-RL-Knot-v3 --headless env.terminations.knot_tied=null
# P1-3 を外す (4 クラスのラダー。値は BFS と同じ k/3)
uv run train --task Isaac-Twisted-RL-Knot-v3 --headless env.rewards.shaping.params.potential=class
```

v2-R2 と比べるときは `Isaac-Twisted-RL-Knot-v2-R2` を使う (ロープを揃える)。ただし
v2-R2 は尺と `drop_xy_rel_frac` が旧値なので、ゲート 1 の結果からは v2-R2 の条件では
ほぼ結べないと予想される。

## smoke test (2026-09-28、`Knot`、64 環境 x 3 iteration、ランダムに近い初期方策)

| iter | `arc_size_c1` / `c2` | `archive_frac` | `restore_mismatch` | `knot_tied` | `diverged` | `value_function` |
|---|---|---|---|---|---|---|
| 0 | 141 / 3 | 0.44 | 0.000 | 0.000 | 0.26 | 0.13 |
| 1 | 417 / 10 | 0.52 | 0.000 | 0.003 | 0.47 | 0.09 |
| 2 | 672 / 22 | 0.45 | 0.000 | 0.031 | 0.46 | 0.15 |

* アーカイブは最初の iteration から埋まり、復元した形状の位相は全件一致した。
  固結びはアーカイブ始まりのエピソードでだけ出ている (`archive_success` 0.075、
  `scratch_success` 0)。
* **発散が多い。** 1 手あたり約 5% (`div_cls0` 3%、`div_cls1` 2%) で、手数を 12 に
  延ばしたぶんエピソードの約半分が発散で終わる。ゲート 1 の台本の再生 (9/128 を 5 手)
  より多いのは、ランダムな手が結び目の上へ指を降ろすなど乱暴なため。本番の学習で
  下がらなければ、尺をさらに延ばすか `actions_per_episode` を戻すことを検討する。
* 1 手あたり約 25 s (64 環境)。v2-R2 (約 19 s、`Twist`) に尺の延長 (x1.3) を掛けた値で、
  物理ステップ数どおり。BFS の表の作成 (約 20 s) は最初の 1 回だけ。
* 3 つの切り分けの上書き (上記) を同時に指定した run と、Play (`prob = 0`、まっすぐな
  ロープから 12 手) も動くことを確かめた。

## この版で解決しないこと

* **本番の学習は未実施。** 上の smoke test で配線が動くことだけを確かめた。
* ゲート 1 の台本の途中状態 (輪 / cover 後 / C2) でアーカイブを初期化すれば、学習の
  最初からアーカイブを使える (Report P0-2 / next_action.md §4.8.10 の 6)。未実装。
* 推論時の探索 (Report P2-4)、対称性の拡張 (P2-1)、遷移スキルへの分解 (P3-4) は範囲外。
* アーカイブはプロセス内だけで持つので、`--resume` すると空から始まる。
