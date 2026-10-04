# 0011 成功率は方策を 1 回展開して測り、論文の「探索込みの成功率」と分けて報告する

- 状態: 有効 (v4 以降の chain 評価と、推論時探索の評価の両方に適用)
- 読むべき状況: 成功率を論文 (TWISTED / TWISTED-RL / Yan2020) の数字と比べるとき。評価に探索 (ビーム探索・best-of-N) を混ぜたくなったとき
- 最終更新: 2026-10-04

## 決定

- 版どうしの比較に使う成功率 (chain 評価) は、学習済み方策を**探索なしで 1 回展開**した結果にする (定義は [`docs/system/evaluation.md`](../system/evaluation.md) §1)。
- 論文と比べるときは、学習済み方策に同じ枠組みの探索を組み合わせた数字を、chain 評価とは**別に**報告する (`eval_search.py`。[`docs/system/evaluation.md`](../system/evaluation.md) §4)。

## 理由

TWISTED / TWISTED-RL / Yan2020 の成功率は、どれも**シミュレータを完全な世界モデルとして使い、最大 30〜60 分探索した結果**である。候補の手を実際にシミュレートし、うまくいったものだけを採用し、失敗したら到達木の別の状態からやり直す。TWISTED-RL の方策の成功率が低くても、探索がそれを補っている。

このリポジトリが目指しているのは「方策を 1 回展開して結ぶ」ことで、論文より難しい目標になっている (v2-R2 の時点では 6 手)。論文と比べるなら、学習済み方策に同じ枠組みの探索を組み合わせるべき (`paper/Report_Paper.md` §7 P2-3)。Isaac Lab の並列環境は、この「候補を N 本同時に試す」処理に向いている。

## 今の実装

| 何 | 場所 | 確認 |
|---|---|---|
| chain 評価 (探索なし) | `src/isaac_twisted_rl/scripts/rsl_rl/eval_chain.py` | 定義は [`docs/system/evaluation.md`](../system/evaluation.md) §1 |
| 探索ありの評価 (ビーム探索・best-of-N) | `src/isaac_twisted_rl/scripts/rsl_rl/eval_search.py` | 結果は [`docs/experiments/2026-10-03_inference_search.md`](../experiments/2026-10-03_inference_search.md) (打ち切り中) |

## 出典

- `paper/Report_Paper.md:163-167` (§5.2)
