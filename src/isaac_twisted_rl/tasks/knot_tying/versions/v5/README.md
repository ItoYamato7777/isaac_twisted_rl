# v5: v4 (段ごとのエージェント・1 遷移 = 1 エピソード) + pick&place の発散対策

```bash
uv run train --task Isaac-Twisted-RL-Knot-v5 --headless           # 学習 (段 0 / 1 / 2 の 3 エージェント)
uv run play  --task Isaac-Twisted-RL-Knot-Play-v5 --num_envs 16   # まっすぐなロープから段をつないで評価 (chain モード)
```

ログ: `logs/rsl_rl/knot_tying_v5/`。**学習には `Knot` プリセットを使う** (v3 / v4 と同じ)。
初期アーカイブ (`outputs/archive_seed/overhand_28links.pt`) は v4 と同じものを読む (作り方は `../v4/README.md`)。

## なぜ作ったか

v3 の最初の学習 ([`docs/experiments/2026-09-28_v3_main.md`](../../../../../../docs/experiments/2026-09-28_v3_main.md)) では、1 手あたり 5〜6% のロープが発散し、エピソードの
37% が発散で終わっていた。v5 は、発散の原因をフェーズ単位で突き止めて、効いた対策だけを v4 に足した版。
観測・報酬・終了・段ごとのアーカイブ・PPO (段ごとのエージェント `StageGatedActorCritic`)・ソルバ反復 8・
1 iteration 4 手・Play (chain モード) は v4 のまま。v3 -> v4 の変更は `../v4/README.md`。

## v4 から何を・なぜ変えたか

| 項目 | v4 | v5 | 理由 |
|---|---|---|---|
| 離すときの指先の高さ (`drop_z` チャネル) | 床上 2 mm〜62 mm | **床上 50〜100 mm** (`GraspLimits.release_tip_z`) | 低く離す手ほど発散する (実験記録 [`docs/experiments/2026-10-01_v5_divergence.md`](../../../../../../docs/experiments/2026-10-01_v5_divergence.md) の「残りの発散の内訳」)。床すれすれで離すと、掴んだ紐を床や下の紐へ押し込む |
| ロープのめり込み解消速度の上限 | 0.96 m/s (`rope_cfg.py` の既定) | **0.4 m/s** (`KnotTyingEnvCfg.rope_max_depenetration_velocity`) | 指や紐どうしが食い込んだときに紐が弾き出されて暴れるのを抑える。0.3 以下は把持が崩れ始める。**学習では段 2 -> 固結びを学べなくした** (下の「学習の結果」) |
| それ以外 | | v4 のまま | |

観測の次元は v4 と同じ (133) なので、v4 のチェックポイントから `--resume` で暖機開始できる
(ただし `drop_z` チャネルの意味は変わっている)。

共有層の変更はすべて **追加** (既定値なら v0〜v4 の挙動は変わらない):
`GraspLimits.release_tip_z` / `grasp_clearance` / `grasp_obstacle_min_z`、`KnotTyingEnvCfg.rope_max_depenetration_velocity`、
`PickPlaceActionCfg.diag` (診断値の記録)、`common/mdp/finger_geometry.py` (新規、純 torch)、
`demos/divergence_bench.py` (新規)、`demos/overhand_script.py` の `--replay_actions` が `release_tip_z` を逆変換するようにした。

## 見るべき指標

v4 と同じ (`../v4/README.md` の「ログ」)。`Curriculum/stage_log/success_s{k}` と `chain_est`、
`diverged_s{k}`、Play での `Episode_Termination/stage_up` (固結びに達した割合)。v4 の同じ時刻の値と比べる。

> 注 (2026-10-04): その後の比較で、学習ログの `chain_est` と段ごとの成功率は本番の成功率を表さないと分かった。版の比較は
> chain 評価で行う (`../v6/README.md` の「見るべき指標」、[`docs/system/evaluation.md`](../../../../../../docs/system/evaluation.md))。

## 計測の方法 (`demos/divergence_bench.py`)

発散は行動と物理だけで決まるので、v3 の環境 (`Knot-v3`、ソルバ反復 16) を基準にして、対策を `--cfg` で足したり外したりして
1 手ごとに記録した。計測の方法・コマンド・結果の表 (接触バッファのあふれ、フェーズ別の内訳、試した対策、ゲート 1 の台本の再生) は
実験記録 [`docs/experiments/2026-10-01_v5_divergence.md`](../../../../../../docs/experiments/2026-10-01_v5_divergence.md) にある。

## 分かったこと

要点 (数値・条件・シードは実験記録):

* v3 の学習中の発散の約半分は PhysX の接触バッファのあふれで、b345065 で解消済み。**b345065 より前の v3 の学習ログと
  比べるときは、この差を発散対策の効果と取り違えないこと**。
* 残りの発散は、乱数の手では離す手 (lower / release / settle_out)、学習済み方策では指を降ろして閉じる手 (descend / close) に多い。
* 離す高さを 5〜10 cm にする対策 (A) とめり込み解消速度 0.4 m/s を採った。掴む高さの対策 (C)・carry の延長・contact offset・
  ソルバ反復 8 は採らなかった (理由は実験記録の「試した対策」の表と下の 2 節)。
* ゲート 1 の台本の再生でも、v5 の設定で固結びは v3 より多くできた。

### 対策の採否の理由

#### 掴む高さの対策 (C) を入れなかった理由

開いた指の真下に紐があると、指を降ろしたときにその紐を押し潰して発散する (実験記録の §2)。そこで指の真下の紐
(`finger_footprint_top`) の上で指を止める対策を 3 通り試したが、どれも把持を大きく壊した:

* 掴むリンクの高さに合わせる: 浮いたリンクで把持が 99.6% -> 7.1%。指先がリンクの下端のすぐ下にしか
  来ないので、閉じる途中でリンクが下へ抜ける。
* 別の紐に乗った紐だけを避ける: `model_150.pt` の手の 58% がこれに当たり、その把持がほぼ全部失敗した
  (全体で 41.8%)。1 cm まで押し込むのを許しても 51.6%。結び目の近くでは、掴みたいリンクの横を別の紐が
  跨いでいるのが普通で、指をその紐より下へ入れないと掴めない。

めり込み解消速度を下げると、同じ手の発散 (descend / close) が 258 -> 34 手に減り、把持も壊れないので、
こちらで置き換えた。

#### めり込み解消速度をいくつにするか

下げるほど発散は減るが、0.25 m/s で把持が落ち始め、0.2 m/s では半分が失敗する。指は close フェーズで
片側 0.15 m/s 前後で閉じるので、ロープのめり込み解消がそれより遅いと、指がロープに食い込んだまま
摩擦で保持できなくなる (解釈)。0.3 m/s は把持の統計では問題が出ないが、ゲート 1 の台本の再生では固結びが
減った (実験記録の §4)。余裕を取って **0.4 m/s** にした。

## 学習の結果 (2026-10-01〜02)

v4 / v6 と同じ条件 (seed 42、4096 環境、GPU 1 枚に 2 本) で学習した。発散対策は学習中も効いていたが、**段 2 -> 固結び (tuck) を
学べず、chain 評価は v4 の 1/4 以下だった**。原因は切り分けで、離す高さではなくめり込み解消速度 0.4 m/s と分かった
(ただし chain 評価では、0.96 m/s に戻しても v6 との差は出なかった)。数値の正本は
[`docs/experiments/2026-10-01_v4-v6_overnight.md`](../../../../../../docs/experiments/2026-10-01_v4-v6_overnight.md)、run のフォルダ名は
[`docs/experiments/2026-10-01_v4-v7_readme-notes.md`](../../../../../../docs/experiments/2026-10-01_v4-v7_readme-notes.md)。

**学習済みの方策での計測と、ゼロからの学習での効果は一致しなかった。** 物性を変えるときは学習で確かめること
([`docs/decisions/0021-measure-physics-changes-by-training.md`](../../../../../../docs/decisions/0021-measure-physics-changes-by-training.md))。

## 残っている課題

* **めり込み解消速度と学習の両立** (上の「学習の結果」)。0.4 m/s は発散を減らすが学習を止め、0.96 m/s は学べるが
  発散が戻る。中間値 (0.6〜0.7 m/s) は未計測。
* **運搬中の発散** が残りの大半になった (乱数の手で v5 の発散の 49%)。引きずる紐の自分自身との擦れが
  原因と考えられるが、未確認。contact offset を上げると 3 割減るが、遅くなり接触バッファも増える。
* 学習済み方策での残り (1 手 0.44%) は descend / carry / relax に分かれていて、決まった原因は見えていない。
* 離す高さを変えたので、v4 のチェックポイントからの暖機開始は、`drop_z` チャネルの意味が変わる
  (観測の次元と方策の形は同じなので `--resume` 自体はできる)。v3 のチェックポイントは観測の次元
  (129 / 133) も方策の形も違うので読めない。
* `KnotTyingEnvCfg.__post_init__` は `episode_length_s = 手数 x decimation x dt` で、IsaacLab は
  `ceil(episode_length_s / step_dt)` で手数に戻す。浮動小数の誤差で 1 手増えることがある (尺を
  descend 0.5 s にすると 807 ステップで 13 手になった)。v3 の 759 ステップはちょうど 12 手なので
  影響はないが、尺を変えるときは注意すること。
