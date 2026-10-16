# 実験記録の一覧

- 状態: 現行 (記録を足すたびに 1 行足す)
- 読むべき状況: 過去の学習・評価・計測の結果を探すとき。数値の正本は各記録で、この一覧には数値を書かない
- 最終更新: 2026-10-16

各記録は凍結 (本文を書き換えない)。後で分かったことは、冒頭の引用ブロックの注記か新しい記録で足す。書き方は skill `experiment-record`。
評価の定義は [`docs/system/evaluation.md`](../system/evaluation.md)、ログのフォルダ名・run 名・チェックポイントの意味は [`docs/system/logs.md`](../system/logs.md)。
run のフォルダは `logs/rsl_rl/` の下 (git の管理外)。日時付きのフォルダ名は各記録の「条件」にある。

## 読み方

- 各記録の冒頭に「要約」(結論と、今も有効か) がある。まず要約だけを読み、数値・条件が要るときに本文へ進む。
- 大きい 4 つ (v3 本番・overnight・rope study・推論時探索) は、要約のファイル (`<名前>.md`) と全文 (`<名前>_detail.md`) に分けてある。節番号・行番号の参照は全文のほうを指す。

## 全記録に効く注意

- **摩擦**: v0 以降のすべての学習・評価・ゲート 1 は、ロープも指も摩擦 0.5 の物理 (意図の 2.0・4.0 ではない)。[摩擦の記録](2026-10-14_friction-binding.md)
- **時間切れの bootstrap の不具合**: v4 以降の全版 (v14 を含む) にある。版どうしの比較は全版が同じ不具合を持つ条件の上の比較で、段 0 の崩れの解釈 (v11〜v13 の記録の推測) はこれで置き換わりうる。[時間切れの bootstrap の記録](2026-10-15_timeout-bootstrap-and-stage2-bottleneck.md)
- **chain 評価の揺れ**: chain 評価は決定論的で、測り直しの揺れは 0 ([記録](2026-10-15_eval-chain-repro.md))。一方、隣り合うチェックポイントの値は平均 9 ポイント動く ([時間切れの bootstrap の記録](2026-10-15_timeout-bootstrap-and-stage2-bottleneck.md))。1 点の差で結論しない。

## v7 以降 (今の比較の土台)

| 日付 | 題 | 状態 | 要点 | run のフォルダ | 記録 |
|---|---|---|---|---|---|
| 2026-10-04 | v7 の確認の学習 (seed 43 / 44) と v6・v4 との比較 | 凍結 | iteration 100 の chain 評価で v7 の 2 シードとも v6・v4 を上回った。iteration 70 では差が見えない。v6 とシードと GPU の配置が揃っていない | `knot_tying_v7/` | [2026-10-04_v7_seeds.md](2026-10-04_v7_seeds.md) |
| 2026-10-05 | v8 (v7 + 対称性のデータ拡張) の学習 (seed 43 / 44) と v7 との chain 評価の比較 | 凍結 | v8 は v7 を大きく下回った (iteration 100 で 2 シードとも v7 の低いほうより下)。原因は未確認。拡張の重要度比が方策の非対称で成り立たない可能性 (推測) | `knot_tying_v8/` | [2026-10-05_v8_symmetry.md](2026-10-05_v8_symmetry.md) |
| 2026-10-05 | v9 (v7 + 段 2 から始める重みを 2 倍) の学習 (seed 43 / 44) と v7 との chain 評価の比較 | 凍結 | v9 は最後のチェックポイントで v7 を上回らなかった。学習ログの段 2 の成功の差は chain 評価に出なかった。seed 44 は timeout で iteration 90 まで | `knot_tying_v9/` | [2026-10-05_v9_stage2-weight.md](2026-10-05_v9_stage2-weight.md) |
| 2026-10-06 | v7 の推論時探索 (best-of-N・ビーム探索) と段 2 の形ごとの切り分け | 凍結 | 探索を付けると v7 はほぼ確実に固結びに届く。学習中に段 2 に着いた形は成功率が形で大きく違い (分散の約半分)、台本の形よりずっと難しい。同じ p-data でも違うので幾何が効く | `knot_tying_v7/` (`eval_search_*_it100*`) | [2026-10-06_inference_search_v7.md](2026-10-06_inference_search_v7.md) |
| 2026-10-06 | v10 (v7 + 段を固結びへの正規の経路の上の p-data だけに絞る) の学習 (seed 43 / 44) と v7 との chain 評価の比較 | 凍結 | v10 は v7 を大きく下回った (2 シード・3 つの iteration のどれでも固結びがほぼ 0)。段 1 -> 経路の上の段 2 の手をほとんど学べなかった。v7 の固結びの大半は経路の終点ではない | `knot_tying_v10/`、`knot_tying_v7/` (`eval_chain_it100_strict.json`) | [2026-10-06_v10_path-stages.md](2026-10-06_v10_path-stages.md) |
| 2026-10-07 | v11 (v7 + 表に無くラダーでも 0 の行き止まりの形だけを段 −1) の学習 (seed 43 / 44、iteration 300) と chain 評価 | 凍結 | iteration 300 で v7 を上回らなかった (比べる相手の v7 は延長の 1 シードだけ)。s44 は iteration 150 以降、段 0 で交差を作らない手を繰り返す崩れ方をした | `knot_tying_v11/` | [2026-10-07_v11_dead-end-stages.md](2026-10-07_v11_dead-end-stages.md) |
| 2026-10-09 | v11 s43 の延長で段 2 の成功が落ちた原因の切り分け (段 2 のアーカイブの自己ループ) | 凍結 | 段 2 のアーカイブが方策の抜け出せない 1 つの形に 4 割占められていた。方策は悪くなっておらず、始まりの分布が崩れていた。v12 (0025) の根拠 | `knot_tying_v11/` | [2026-10-09_v11_stage2-archive-loop.md](2026-10-09_v11_stage2-archive-loop.md) |
| 2026-10-13 | v12 (v7 + アーカイブに段が上がった手の後の形だけを保存) の学習 (seed 43 / 44、iteration 300) と chain 評価 | 凍結 | 段 2 の時間切れのループは起きなかった。s43 は it100 と it300 がほぼ同じ、s44 は it70 の後に段 0・1 から崩れた (段 0 のアーカイブは変わらないまま)。段 2 のアーカイブは Φ の表にある 1 つの形に 3〜4 割寄った (0025 の予想 1 は外れ) | `knot_tying_v12/` | [2026-10-13_v12_archive-stage-up.md](2026-10-13_v12_archive-stage-up.md) |
| 2026-10-14 | chain 評価が it70〜100 の後に下がる原因の調査 (方策の平均が行動の範囲の外へ出る) | 凍結 | v7・v11・v12 の全 run で方策の平均が範囲の外へ出る割合が増え続けた。大きく崩れた run では段 0 の置き方のチャネルまで張り付いた。相関のみで因果は未確認。v13 (0026) の根拠 | `knot_tying_v7/`・`v11/`・`v12/` (出力は `outputs/action_saturation/`) | [2026-10-14_action-saturation.md](2026-10-14_action-saturation.md) |
| 2026-10-14 | v13 (v12 + PPO の損失に方策の平均の範囲外の罰則、係数 0.01) の学習 (seed 43 / 44、iteration 300) と chain 評価 | 凍結 | 罰則は範囲の外への出方の大きさを抑えたが、範囲の外に出る割合はあまり減らなかった (0026 の予想 1 は外れ、結論は保留)。s44 は v12 s44 と同じ型に段 0 から崩れ、chain 評価は 2 シードとも it70 の後に下がった | `knot_tying_v13/` (張り付きは `outputs/action_saturation/v13.json`) | [2026-10-14_v13_bounds-loss.md](2026-10-14_v13_bounds-loss.md) |
| 2026-10-14 | ロープと指の実際の摩擦の確認 (PhysX の読み戻し・滑りの試験・指の試験) | 凍結 | ロープの摩擦は USD の 2.0 でなく PhysX の既定の 0.5 (bind の先が Scope)。指も 0.5 と見られる (推測)。v0 以降の学習はすべてこの物理 | `outputs/friction_check/` | [2026-10-14_friction-binding.md](2026-10-14_friction-binding.md) |
| 2026-10-14 | v14 の学習の前のゲート (DR の読み戻し・ゲート 1・発散の計測・スモークテスト) | 凍結 | 配線は動く。DR 全体でゲート 1 は 81%、発散は公称の 1.6 倍。減衰比の小さい側が発散で公称の 2 倍を超える (範囲を狭める案) | `outputs/overhand_script/v14_*`・`outputs/divergence_bench/v14_*`・`knot_tying_v14/` | [2026-10-14_v14_physics-gate.md](2026-10-14_v14_physics-gate.md) |
| 2026-10-15 | 段 0 の critic の価値 V の膨らみと段 0 の崩れ (v7・v11・v12・v13 の事後調査) | 凍結 | 崩れた 4 本の run は段 0 の V の平均が 1/γ を超えて 2〜15 まで膨らみ、崩れなかった 3 本は最大 1.05。V と段 0 の時間切れの増加は 10〜40 iteration で重なる (因果は未確認)。V が膨らむ原因は不明 | `knot_tying_v7/`・`v11/`・`v12/`・`v13/` (出力は `outputs/critic_stage_values/`) | [2026-10-15_critic-inflation.md](2026-10-15_critic-inflation.md) |
| 2026-10-15 | chain 評価が頭打ちになる原因の事後調査 (時間切れの bootstrap の不具合と段 2 の詰まり) | 凍結 | 6 手目に段が上がると時間切れも立ち、PPO が +1 の手にも γ·V を足す不具合がある (v4 以降の全版)。崩れた方策はまっすぐなロープから 6 手目に段 0 を上げていた (因果は未確認)。崩れない run の頭打ちは、段 2 の 1 手の成功の低さと下降の多さでほぼ説明できる (マルコフ連鎖の模型の上)。チェックポイントの間の揺れは二項の見積りの約 4 倍 | `knot_tying_v7/`〜`v13/` (出力は `outputs/chain_moves/`) | [2026-10-15_timeout-bootstrap-and-stage2-bottleneck.md](2026-10-15_timeout-bootstrap-and-stage2-bottleneck.md) |
| 2026-10-15 | chain 評価の再現性 (測り直し・`--seed`・GPU で値が変わるか) | 凍結 | chain 評価は決定論的。測り直し・`--seed`・GPU を変えても全キーがビット単位で同じ。512 の結末は同じ初期状態から物理の増幅で分かれる。揺らぎの組を引き直したときの揺れは未実測 | `outputs/eval_repro/` | [2026-10-15_eval-chain-repro.md](2026-10-15_eval-chain-repro.md) |
| 2026-10-15 | v14 (v13 + 段 0 の形を変えた手の保存・ロープの物理の DR・critic の特権観測) の学習 (seed 43 / 44、iteration 300) と chain 評価 (公称と DR 付き) | 凍結 | 2 シードとも段 0 で崩れず、公称の chain 評価で it100 以降に v13 を大きく上回り、it300 まで下がらなかった (0027 の予想 1・2 は当たり)。DR 付きの評価での下がり幅は v14 の方が大きい (予想 3 は外れ)。3 つの変更のどれが効いたかは未確認 | `knot_tying_v14/` (v13 の方策を v14 の Play で測った値は `outputs/eval_v14env/`) | [2026-10-15_v14_s0-reshape-dr-priv-critic.md](2026-10-15_v14_s0-reshape-dr-priv-critic.md) |
| 2026-10-16 | v15 の学習の前のゲート (terminated と truncated の重なり) と、v14 の段 0 の V | 凍結 | v14 では両方が立つ手が 124 / 6144 手、v15 では 0。純粋な時間切れは残る。v14 の段 0 の V は固定した観測で 1/γ をやや超える点があるが膨らみは無い。2 iteration のスモークテストは別には回していない | (学習の run は `knot_tying_v15/`) | [2026-10-16_v15_timeout-gate.md](2026-10-16_v15_timeout-gate.md) |
| 2026-10-16 | 学習の速さのために v14 の物理の設定 (ソルバの反復・dt・待ちのフェーズの尺) を軽くできるかの計測 | 凍結 | 時間のほぼ全部が PhysX。v14 の方策を軽い物理で chain 評価すると、待ちの尺を詰めた D4 は質がほぼ落ちず、反復 6 は少し、dt・反復 4・組み合わせは大きく落ちた。ゼロからの学習での確認は未実施 | `outputs/bench_step_time/`・`outputs/eval_physics_speed/`・`outputs/overhand_script/speed/` | [2026-10-16_physics-speed.md](2026-10-16_physics-speed.md) |

## v0〜v6 と、打ち切り中の研究

古い版の記録も、版の README・`docs/decisions/`・skill から設計の根拠として引かれているので、ここに置く (経緯の `docs/history/` には移さない)。

| 日付 | 題 | 状態 | 要点 | run のフォルダ | 記録 |
|---|---|---|---|---|---|
| 2026-09-15 | v0 (交差 1 つ) と v1 (固結び) の学習の比較 | 凍結 | v1 は疎報酬を一度も得られず、発散とエントロピーが増えて失敗した。到達できる中間目標が無いことが主因 | `knot_tying_v0/`、`knot_tying_v1/` | [2026-09-15_v0-v1.md](2026-09-15_v0-v1.md) |
| 2026-09-24 | v2 の方策が偽の C2 に居座っていたことの実測 | 凍結 | v2 の C2 判定の抜け道 (R2 除去後は輪 1 つ) に方策が居座っていた。v2-R2 の動機 | `knot_tying_v2/` | [2026-09-24_v2_fake-c2.md](2026-09-24_v2_fake-c2.md) |
| 2026-09-27 | ゲート 1: 台本で固結びが結べるか | 凍結 | 台本で固結びは結べる。ただし長いロープ (`Knot`)・置き先の範囲の拡大・長い尺が要る。`env.step` での再生は台本より発散が多い (未解明) | なし (台本の出力は `outputs/overhand_script/`。どれが表のどの行かは記録なし) | [2026-09-27_gate1.md](2026-09-27_gate1.md) |
| 2026-09-27 | v2-R2: 発散した手の報酬で学習が壊れた問題と修正の確認 | 凍結 | 発散した手の罰が価値関数を壊した。修正の後は短い確認で安定。最初の run は比較に使えない | `knot_tying_v2_r2/` (壊れた最初の run だけ) | [2026-09-27_v2r2_fix.md](2026-09-27_v2r2_fix.md) |
| 2026-09-28 | v3 本番の学習 | 凍結 | まっすぐなロープから固結びを初めて学習できた。次のボトルネックは発散 | `knot_tying_v3/` | [2026-09-28_v3_main.md](2026-09-28_v3_main.md) |
| 2026-09-28 | v4 の初期アーカイブと smoke test | 凍結 | 台本の途中状態を段ごとの初期アーカイブにした。v4 の配線 (アーカイブ・切り分けの上書き・Play・ONNX) が動いた | 記録なし (初期アーカイブは `outputs/archive_seed/`) | [2026-09-28_v4_smoke-test.md](2026-09-28_v4_smoke-test.md) |
| 2026-09-30 | v4 の学習速度の調査 | 凍結 | 1 iteration の時間はほぼ PhysX のソルバの反復の固定費。反復と 1 iteration の手数を減らして速くし、学習も速く進んだ | `knot_tying_v4_speed/` | [2026-09-30_v4_speed.md](2026-09-30_v4_speed.md) |
| 2026-10-01 | v5 の発散の計測: フェーズ別の内訳と対策の比較 | 凍結 | 発散の半分は接触バッファのあふれ。離す高さとめり込み解消速度の対策を採った。**計測では効いたが、学習では逆効果だった** (v4〜v6 の比較) | なし (計測の出力は `outputs/divergence_bench/` と思われる。未確認) | [2026-10-01_v5_divergence.md](2026-10-01_v5_divergence.md) |
| 2026-10-01 | v6 の「次にたどる辺」とゲート 1 の台本の照合 | 凍結 | 台本の遷移は最短の辺でほぼ表せる。台本との一致で辺の選び方 `long_end` を既定にした | なし (`outputs/overhand_script/knot_carry020/`) | [2026-10-01_v6_script-edge-match.md](2026-10-01_v6_script-edge-match.md) |
| 2026-10-01 | v4 / v5 / v6 の学習比較 (overnight) と v6 の切り分け | 凍結 | chain 評価では v4 と v6 が並び、v5 は大きく劣る。シード差が大きい。v6 の切り分け (= v7 の根拠) はどれも chain 評価で v6 を上回らなかった | `knot_tying_v4/`、`knot_tying_v5/`、`knot_tying_v6/`、`knot_tying_v6_ablation/` | [2026-10-01_v4-v6_overnight.md](2026-10-01_v4-v6_overnight.md) |
| 2026-10-01 | v4〜v7 の版の README にあった学習の結果の補足 | 凍結 | 上の比較の run のフォルダ名・最後のチェックポイント・評価 JSON の一覧と、報告書に無い値。v7 の結果は v6 + 上書きの run | 同上 | [2026-10-01_v4-v7_readme-notes.md](2026-10-01_v4-v7_readme-notes.md) |
| 2026-10-02 | 紐のシミュレーションモデルの比較 (rope study) | 打ち切り中 | 実機の検証へ持っていく最有力は剛体鎖。XPBD は物理では最良だが学習の途中で方策が崩れる。発散の数値は過大の疑い (未確認) | `knot_tying_rs_*/` (無効な run は `INVALID.txt`) | [2026-10-02_rope_models.md](2026-10-02_rope_models.md) |
| 2026-10-03 | 推論時探索の評価 (中間) | 打ち切り中 | 探索を入れても版の順位は変わらない (1 シードずつ)。今の設定のビーム探索は best-of-N をはっきりとは上回らない。評価は途中で停止 | `knot_tying_v4/`、`knot_tying_v5/`、`knot_tying_v6/`、`knot_tying_v6_ablation/` (`eval_search_*`) | [2026-10-03_inference_search.md](2026-10-03_inference_search.md) |

記録から抜き出した設計の判断: [`0020`](../decisions/0020-v4-solver-iterations-and-steps-per-iteration.md) (v4 のソルバの反復 8・1 iteration 4 手)、[`0021`](../decisions/0021-measure-physics-changes-by-training.md) (物性の変更は学習と chain 評価で確かめる)。
