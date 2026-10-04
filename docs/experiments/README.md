# 実験記録の一覧

- 状態: 現行 (記録を足すたびに 1 行足す)
- 読むべき状況: 過去の学習・評価・計測の結果を探すとき。数値の正本は各記録で、この一覧には数値を書かない
- 最終更新: 2026-10-04

各記録は凍結 (本文を書き換えない)。後で分かったことは、冒頭の引用ブロックの注記か新しい記録で足す。書き方は skill `experiment-record`。
評価の定義は [`docs/system/evaluation.md`](../system/evaluation.md)、ログのフォルダ名・run 名・チェックポイントの意味は [`docs/system/logs.md`](../system/logs.md)。
run のフォルダは `logs/rsl_rl/` の下 (git の管理外)。日時付きのフォルダ名は各記録の「条件」にある。

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
