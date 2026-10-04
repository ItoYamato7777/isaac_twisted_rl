# 0006 移植元の構成 (記号プランナ + 単一スキル + SAC/HER) をそのまま使わない

- 状態: 有効 (on-policy の PPO を使う限り)
- 読むべき状況: 移植元 TWISTED-RL の仕組み (HER・replay buffer のカリキュラム・記号 BFS プランナ) を取り込みたくなったとき。学習器を SAC など off-policy に替える案が出たとき
- 最終更新: 2026-10-04

## 決定

固結び (v1 以降) では、移植元の構成 (記号プランナ + 単一の Reidemeister スキル) を採らない。HER と、replay buffer を次段の初期状態プールにするカリキュラムは、PPO には載せない。代わりに次を取り込む。

| 移植元の仕組み | 代わりに取り込んだもの |
|---|---|
| HER | v1 では報酬ラダーを HER の代替と位置づけた ([0007](0007-topology-class-ladder-shaping.md)) |
| replay buffer を次段の初期状態プールにするカリキュラム (G0→G1→G2) | 初期状態プールの考え方自体は PPO でも使えることが後で分かり、v3 の到達状態アーカイブとして取り込んだ ([0009](0009-reached-state-archive.md)) |

## 理由

移植元 TWISTED-RL は、**結び目そのものを一度も報酬にしていない**。報酬は `{+1 目標位相に一致 / -1 それ以外へ変化 / 0 不変}` の疎報酬で、**位相が変わった瞬間にエピソードが終了**する。固結びは、記号 BFS プランナが単一の移動スキルを 3 回連鎖させて組み立てる。

これを支えているのが **HER** と、**replay buffer を次段の初期状態プールにするカリキュラム** (G0→G1→G2)。どちらも **off-policy (SAC) が前提**で、本リポジトリの on-policy PPO には載らない。

その後の経緯: v1 の学習は失敗した。v4 で、論文の「1 回の位相遷移 = 1 エピソード」と +1 / 0 / −1 の報酬、交差数ごとのエージェント (C 版) を PPO のまま取り込んだ ([0008](0008-transition-reward-and-termination.md))。グラフを使った高レベル計画 (次に狙う遷移の選択・再計画) は未実装 (旧 §4.7)。

## 今の実装

(`kt/` = `src/isaac_twisted_rl/tasks/knot_tying/`)

| 何 | 場所 | 確認 |
|---|---|---|
| 学習器は RSL-RL の PPO | `kt/versions/*/agents/rsl_rl_ppo_cfg.py` | 一致 |
| 移植元の R1 / R2 / cross は移植済みで、p-data グラフの BFS 距離を報酬のポテンシャル・段に使う (計画には使わない) | `kt/common/topology/representation.py`、`kt/common/topology/knot_graph.py` | 一致 |

## 出典

`docs/history/next_action.md` の行番号は、頭書き (33 行) を足す前のもの (今の行番号は 33 を足す)。

- `docs/history/next_action.md:380-396` (旧 §4.8 冒頭・§4.8.1)、`:369-376` (旧 §4.7)
- `docs/history/v1_diff_to_TwistedRL.md` (v1 時点のコード比較)
