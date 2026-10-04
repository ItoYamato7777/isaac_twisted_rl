# 文書の地図

- 状態: 現行
- 読むべき状況: どの文書を読めばよいかを決めるとき。新しい文書を置く場所を決めるとき
- 最終更新: 2026-10-04

## 読む順番

1. `CLAUDE.md` (リポジトリのルート。Claude Code では自動で読み込まれる): 規則とコマンド
2. [`STATUS.md`](STATUS.md): 現在地・版の状態・次の作業・保留
3. 作業に応じて下の表から。用語が分からなければ [`glossary.md`](glossary.md)

## 置き場所の規則

1 つの情報の正本は 1 か所に置き、ほかの場所からはリンクする。

| 情報の種類 | 置き場所 |
|---|---|
| 現在地・次の作業・保留 | [`STATUS.md`](STATUS.md) (上書きで更新。skill `handoff`) |
| 実験の結果と、再現のメタデータ | [`experiments/`](experiments/README.md) (1 実験 1 ファイル。書いたら凍結。skill `experiment-record`) |
| 版をまたぐ設計の判断と理由 | [`decisions/`](decisions/README.md) (1 判断 1 ファイル) |
| 横断的な仕様 (評価・ログ) | [`system/`](system/) |
| 1 つのパッケージ・版に閉じた仕様と理由 | コードの隣 (パッケージの README・docstring) |
| 経緯 (通常は読まない) | `history/` |
| 論文と、論文に関わる文書 | `paper/` (リポジトリのルート) |

## 文書の一覧

### 現行 (仕様・規則)

| 文書 | 読むべき状況 |
|---|---|
| [`STATUS.md`](STATUS.md) | セッションの始め |
| [`glossary.md`](glossary.md) | 用語の意味、過去の名前 (タスク ID・ログ名・旧 v7・移動したファイル)、命名規則 |
| [`workflow.md`](workflow.md) | 研究の 1 サイクル (段階・終了の条件・残す記録・飛ばしてよい段階)、セッションと長時間ジョブの規約。コマンドは skill にある |
| [`system/evaluation.md`](system/evaluation.md) | 版を比べるとき。chain 評価のコマンドを組むとき |
| [`system/logs.md`](system/logs.md) | run を探す・チェックポイントを選ぶ・ログのキーを読むとき |
| [`decisions/README.md`](decisions/README.md) | 設計の理由を知りたいとき (行動空間・240 Hz・報酬・「版 = 設定」など) |
| [`knot_tying/README.md`](../src/isaac_twisted_rl/tasks/knot_tying/README.md) | 版の系譜・タスク ID の規則・ロープのプリセット・フォルダ構成 |
| [`knot_tying/common/README.md`](../src/isaac_twisted_rl/tasks/knot_tying/common/README.md) | 全版に共通の仕組み (1 手 = 1 RL ステップ、フェーズ、行動、終了判定) |
| `knot_tying/versions/vN/README.md` | その版の課題・観測・報酬と理由 |
| [`scripts/rsl_rl/README.md`](../src/isaac_twisted_rl/scripts/rsl_rl/README.md) | 学習・評価のスクリプトの使い方、推論時探索のアルゴリズム |
| [`knot_tying/rope_study/README.md`](../src/isaac_twisted_rl/tasks/knot_tying/rope_study/README.md) | 紐モデルの比較 (打ち切り中) の道具の使い方 |
| `.claude/skills/*/SKILL.md` | 手順: `train-eval`・`add-version`・`physics-check`・`experiment-record`・`handoff` |
| `.claude/rules/*.md` | `common/mdp/`・`versions/` を変えるときの規則 (そのファイルを扱うときに自動で読み込まれる) |

### 実験記録

一覧と要点は [`experiments/README.md`](experiments/README.md)。打ち切り中の 2 つの研究の報告書もここにある
([`2026-10-02_rope_models.md`](experiments/2026-10-02_rope_models.md)、[`2026-10-03_inference_search.md`](experiments/2026-10-03_inference_search.md))。

### 論文と提案 (`paper/`)

| 文書 | 状態 | 読むべき状況 |
|---|---|---|
| [`paper/Report_Paper.md`](../paper/Report_Paper.md) | 凍結 (2026-09-26 時点) | 3 本の論文と実装の違い。v3 の設計の根拠 |
| [`paper/Discussion.md`](../paper/Discussion.md) | 2026-10-01 時点の提案 (仮説) | 次の研究の候補 (E0〜E7・S1) を選ぶとき |
| [`paper/explainer/index.html`](../paper/explainer/index.html) | 参考資料 | 3 本の論文の図解 (前提知識・比較表・用語集) |
| `paper/*.pdf` | 参考資料 | 論文の本体 (TWISTED-RL・TWISTED・Yan2020) |

### 経緯 (`history/`。通常は読まない)

| 文書 | 中身 |
|---|---|
| [`history/next_action.md`](history/next_action.md) | v0〜v3 の設計と経緯の記録 (旧 `knot_tying/next_action.md`)。冒頭に旧節番号と新しい場所の対応表 |
| [`history/rope_study_HANDOFF.md`](history/rope_study_HANDOFF.md) | rope study の 4 つのセッションの引き継ぎメモ |
| [`history/v1_diff_to_TwistedRL.md`](history/v1_diff_to_TwistedRL.md) | v1 の時点で移植元のコードと比べたスナップショット |
| [`history/v3_html/`](history/v3_html/v3-implementation.html) | v3 の実装の図解と、論文との比較 (2026-09-28 時点) |
| [`history/refactor/`](history/refactor/README.md) | 2026-10-03〜04 のリファクタリングの記録 (依頼・調査・提案・最終報告・確認の道具)。文書や試験が引く所見の番号 (A-3・B-101・D-6 など) の出典 |
