# v4〜v7 の版の README にあった学習の結果の補足 (2026-10-01)

- 状態: 凍結
- 読むべき状況: v4 / v5 / v6 の overnight と v6 の切り分け (= v7 の設定) の run のフォルダ名・最後のチェックポイント・評価 JSON を探すとき。結果の数値の正本は [`2026-10-01_v4-v6_overnight.md`](2026-10-01_v4-v6_overnight.md) で、ここにはそこに無いものだけを置く

## 要約

- 結論: v4〜v6 の overnight の比較と v6 の切り分けの、run のフォルダ名・最後のチェックポイント・評価 JSON の一覧 (再現の索引)。結論は [overnight の記録](2026-10-01_v4-v6_overnight.md) のとおり。
- 有効か: 有効 (索引)。3 時間の run と一部の run の学習のコマンドは記録なし。

> 転記元 (commit 62b251b の時点): `src/isaac_twisted_rl/tasks/knot_tying/versions/v4/README.md` の「学習の結果」(`:290-310`)、`v5/README.md` の「学習の結果」(`:168-189`)、`v6/README.md` の「学習の結果」「v6 の切り分け」(`:147-183`)、`v7/README.md` の「なぜ作ったか」の表 (`:17-38`) と「結果」(`:67-75`)。
> これらの節の数値・条件は、下に転記したものを除いて [`2026-10-01_v4-v6_overnight.md`](2026-10-01_v4-v6_overnight.md) にあることを確かめてから、版の README からはリンクに置き換えた。

## 目的と仮説

[`2026-10-01_v4-v6_overnight.md`](2026-10-01_v4-v6_overnight.md) と同じ (v4 / v5 / v6 の比較と、v6 の切り分け)。

## 条件

共通の条件は [`2026-10-01_v4-v6_overnight_detail.md`](2026-10-01_v4-v6_overnight_detail.md) の §1 と追記の「条件」(4096 環境、GPU 1 枚に 2 本、10 iteration ごとに保存、chain 評価は 512 エピソード・seed 7・決定論的方策)。

### run のフォルダ (ls で確認)

| run | タスク ID (学習) | commit | 未コミットの差分 | シード | 最後のチェックポイント | 評価 JSON | 学習の上書き |
|---|---|---|---|---|---|---|---|
| `logs/rsl_rl/knot_tying_v4/2026-10-01_14-08-32_3h_compare` | `Isaac-Twisted-RL-Knot-v4` | 記録なし | あり (`git/isaac_twisted_rl.diff`) | 記録なし | `model_50.pt` | なし (64 エピソードの試運転。値は overnight の §3.1 の参考) | 記録なし |
| `logs/rsl_rl/knot_tying_v4/2026-10-01_19-22-13_overnight_s42` | `Isaac-Twisted-RL-Knot-v4` | `881dc8d` | あり | 42 | `model_100.pt` (106 iteration 学習した) | `eval_chain_it{50,70,100}.json` | `agent.save_interval=10` |
| `logs/rsl_rl/knot_tying_v5/2026-10-01_19-22-38_overnight_s42` | `Isaac-Twisted-RL-Knot-v5` | `881dc8d` | あり | 42 | `model_110.pt` (115 iteration 学習した) | `eval_chain_it{50,100}.json` | `agent.save_interval=10` |
| `logs/rsl_rl/knot_tying_v6/2026-10-01_19-23-03_overnight_s42` | `Isaac-Twisted-RL-Knot-v6` | `881dc8d` | あり | 42 | `model_100.pt` (107 iteration 学習した) | `eval_chain_it{50,70,100}.json` | `agent.save_interval=10` |
| `logs/rsl_rl/knot_tying_v6/2026-10-01_19-23-28_overnight_s43` | `Isaac-Twisted-RL-Knot-v6` | `881dc8d` | あり | 43 | `model_110.pt` (115 iteration 学習した) | `eval_chain_it{50,70,100}.json` | `agent.save_interval=10` |
| `logs/rsl_rl/knot_tying_v6_ablation/2026-10-02_12-33-10_abl_relA_s42` | `Isaac-Twisted-RL-Knot-v6` | `881dc8d` (ログの `README.md`) | あり | 42 | `model_70.pt` | `eval_chain_it70.json` | `env.actions.pick_place.limits.release_tip_z=null` |
| `logs/rsl_rl/knot_tying_v6_ablation/2026-10-02_12-33-35_abl_depB_s42` (**= v7 の設定**) | `Isaac-Twisted-RL-Knot-v6` | `881dc8d` (同) | あり | 42 | `model_70.pt` | `eval_chain_it70.json` | `env.scene.rope.spawn.rigid_props.max_depenetration_velocity=0.96` |
| `logs/rsl_rl/knot_tying_v6_ablation/2026-10-02_12-34-00_abl_toP_s42` (**当時の v7**) | `Isaac-Twisted-RL-Knot-v7` (当時。報酬項 `stage_time_out`) | 記録なし (`stage_time_out` はどのコミットにも無い: `docs/history/refactor/audit/B_2.md` の B-102) | あり | 42 | `model_70.pt` | `eval_chain_it70.json` | — |
| `logs/rsl_rl/knot_tying_v6_ablation/2026-10-02_12-34-26_abl_sw2_s42` | `Isaac-Twisted-RL-Knot-v6` | `881dc8d` (同) | あり | 42 | `model_70.pt` | `eval_chain_it70.json` | `'env.events.reset_to_stage.params.stage_weights=[1.0,1.0,2.0]'` |

- 学習した iteration 数は overnight の §1 の「到達 iteration」の値。チェックポイントは 10 iteration ごとなので、最後のファイルはそれより手前になる。
- 学習と評価のコマンドの例は overnight の「付録: 再現の手順」(v6 s42) と「再現の手順 (追記の run)」(depB)。他の run のコマンドは記録なし。
- 切り分けの上書きの正本は `logs/rsl_rl/knot_tying_v6_ablation/README.md` (追跡外)。relA / depB を Play・chain 評価で読むときは、学習と同じ上書きを付ける。
- **v7 の結果は v7 のタスク ID からは辿れない。** v7 と同じ設定の run は、v7 のコミット (`95ac19a`、2026-10-02 23:09) より前に v6 + Hydra の上書きで走った `abl_depB_s42` だけ (`docs/history/refactor/audit/B_1.md` の B-13)。`logs/rsl_rl/knot_tying_v7/` は無い。

## ゲート (学習の前の確認)

記録なし (各版の発散とゲート 1 の再生は [`2026-10-01_v5_divergence.md`](2026-10-01_v5_divergence.md))。

## 結果 (overnight の報告書に無いものの転記)

### v4 README の「学習の結果」から

| iteration | 固結び | 時間切れ | 発散 | 固結びまでの平均手数 |
|---|---|---|---|---|
| 70 | 29.7% | 53.1% | 17.2% | 9.7 |

(iteration 50 / 100 の行と、iteration 70 の固結び・時間切れ・発散は overnight の §3.1 と追記の chain 評価の表にある。iteration 70 の平均手数 9.7 は報告書に無い。)

* 失敗は時間切れと発散が半々。18 手のあいだに 17% が発散する (学習中の「発散で終わったエピソード」は 3〜4% だが、
  学習のエピソードは最大 6 手で、段が変わればすぐ終わるので小さく見える)。
* iteration 100 の時点で、v6 (37.9% / 22.9%) との差はシード差の範囲。v5 (9.6%) よりは明らかに良い。

### v5 README の「学習の結果」から

* 上の「試した対策」の表 ([`2026-10-01_v5_divergence.md`](2026-10-01_v5_divergence.md) の §3) は学習済みの v3 方策 (`model_150.pt`) で測ったもので、0.4 m/s は固結びを増やしていた。
  **学習済みの方策での計測と、ゼロからの学習での効果は一致しなかった。** 物性を変えるときは学習で確かめること。
  (判断としての扱いは [`docs/decisions/0021-measure-physics-changes-by-training.md`](../decisions/0021-measure-physics-changes-by-training.md)。)

### v7 README の「なぜ作ったか」「結果」から

* 段 0 の時間切れも、学習前 (iteration 0〜9) から v6 29% に対して v7 5% と違う。0.4 m/s は学習の
  問題ではなく、物理の段階で交差を作りにくくしている (理由は未確認)。
* **v7 としての学習はまだ無い** (2026-10-02 時点)。計画していた確認の学習 (v7 を 3 シード + v4 の 2 本目) は、chain 評価で v6 を上回る根拠が無かったので保留した。

## 結論

- 分かった: 上の表と、[`2026-10-01_v4-v6_overnight.md`](2026-10-01_v4-v6_overnight.md) の結論のとおり。
- 推測: 記録なし (解釈は overnight の報告書の「解釈」の箇所)。
- 未確認: 3 時間の run と、v6 s42 と depB 以外の run の学習のコマンド。

## 次の作業

overnight の報告書の「追記の後の次の候補」(シードを増やす、めり込み解消速度の中間値 0.6〜0.7 m/s、評価を強くする)。推論時探索の評価は [`2026-10-03_inference_search.md`](2026-10-03_inference_search.md)。
