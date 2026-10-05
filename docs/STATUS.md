# STATUS (現在地)

- 状態: 現行。上書きで更新する (追記しない。手順は skill `handoff`)
- 読むべき状況: セッションの始め。今どこにいて、次に何をするかを知るとき
- 最終更新: 2026-10-05

## 要点

- v7 を seed 43 / 44 で学習した。iteration 100 の chain 評価で 2 シードとも v6 (2 シード)・v4 を上回った。ただしシードと GPU の配置が v6 と揃っていない ([記録](experiments/2026-10-04_v7_seeds.md))。
- 次の候補は、v6 seed 44 で揃えて比べることと、v7 を iteration 150〜200 まで延ばすこと。どれを回すかはユーザーが決める。
- 研究の作業 (rope study・推論時探索の評価) は 2026-10-03 に打ち切った。ユーザーの指示があるまで再開しない。

## 版の状態

数字は各記録にある (ここには書かない)。比べ方は [`docs/system/evaluation.md`](system/evaluation.md)。

| 版 | 状態 | 記録 |
|---|---|---|
| v0 | 学習した (交差 1 つは学べた) | [2026-09-15_v0-v1](experiments/2026-09-15_v0-v1.md) |
| v1 | 学習した (失敗: 疎報酬が一度も発火しない) | 同上 |
| v2 | 学習した (偽の C2 に居座った) | [2026-09-24_v2_fake-c2](experiments/2026-09-24_v2_fake-c2.md) |
| v2-R2 | 発散の報酬の修正 (2026-09-27) の後の本番の学習は**未実施**。「本物の C2 が稀で詰まる」は、ランダム方策と数 iteration の観察からの**仮説** | [2026-09-27_v2r2_fix](experiments/2026-09-27_v2r2_fix.md) |
| v3 | 本番の run 1 本 (シード 1 本) | [2026-09-28_v3_main](experiments/2026-09-28_v3_main.md) |
| v4 | chain 評価 済み (seed 42 だけ。2 本目のシードは未実施) | [2026-10-01_v4-v6_overnight](experiments/2026-10-01_v4-v6_overnight.md) |
| v5 | chain 評価 済み (seed 42 だけ) | 同上 |
| v6 | chain 評価 済み (seed 42・43、GPU 1 枚に 2 本) | 同上 |
| v7 | chain 評価 済み (seed 43・44 は iteration 100 まで、GPU 1 枚に 1 本。seed 42 に当たる `abl_depB` は iteration 70 まで) | [2026-10-04_v7_seeds](experiments/2026-10-04_v7_seeds.md) |

## 走っているジョブ

なし (2026-10-05 に `nvidia-smi` で確認)。

## 次の作業

1. 次に何をするかはユーザーが決める。候補 (出典: [v7 の記録](experiments/2026-10-04_v7_seeds.md) の「次の作業」):
   - v6 を seed 44 で学習し (できれば GPU 1 枚に 1 本)、v7 と同じシード 43 / 44 で chain 評価を比べる。仮説: v7 の優位 (iteration 100) はシードと GPU の配置を揃えても残る。約 6 時間 + 評価 15 分。
   - v7 を `--resume` で iteration 150〜200 まで続け、10〜20 iteration ごとに chain 評価して頭打ちの位置を見る。
   - 上の 2 つで差が保たれれば、v7 を次の版 (v8) の土台にする。
2. その他の候補 (どれも未着手):
   - めり込み解消速度の中間値 (0.6〜0.7 m/s)。学習の前に skill `physics-check` で発散と固結びを測る。出典: [overnight の記録](experiments/2026-10-01_v4-v6_overnight.md) の「追記の後の次の候補」。
   - 評価を強くする (確率的な方策での chain 評価、10 iteration ごとの学習曲線、エピソード数・シードを増やす)。出典: 同上。
   - 研究方針の提案 E0〜E7・S1 ([`paper/Discussion.md`](../paper/Discussion.md)、2026-10-01 時点の仮説の文書)。S1 (推論時の候補評価) は推論時探索で一部実施された。

## 保留 (打ち切り中。再開するかはユーザーの判断待ち)

| 研究 | 止めた所 | 再開するときの 1 番目 | 記録 |
|---|---|---|---|
| 推論時探索 (ビーム探索・best-of-N) | 評価の途中 (v7・v6 s43 の rollouts / search、v6 s42・v5 の search が未完了) | 止めた評価を最後まで回す。v7 と v6 の比較は [v7 の記録](experiments/2026-10-04_v7_seeds.md) で 2 シードずつになった | [2026-10-03_inference_search](experiments/2026-10-03_inference_search.md) §6 |
| rope study (紐モデルの比較) | 最終報告まで書いた。評価と実装は 2026-10-03 13:00 に打ち切り | XPBD の段 0 の方策の崩れの切り分け (結論を変えうる最大の未解決) | [2026-10-02_rope_models](experiments/2026-10-02_rope_models.md) §12 |

- rope study の発散の数値には、`rope_study/bench_physics.py:378` の数え方による過大の疑いがある (未確認。再開するときに確かめる)。
- 推論時探索の記録の「V8」は、次の版 (v8) のことと読める (定義は記録に無い)。v8 の中身は未定。
