# skills の description の区別の試験 (Phase 3、2026-10-04)

BRIEF §4 Phase 3 の手順。proposal §4.4 の表の依頼を、各 skill の description だけを読んで振り分けた (メインが判定)。

| 依頼 | 期待 | description からの判定 | 結果 |
|---|---|---|---|
| v6 を seed 42 と 43 で 100 iteration 学習して | train-eval | train-eval (「版や設定を学習し」) | OK |
| v7 と v6 を比べて | train-eval | train-eval (「chain 評価で比べる」) | OK |
| この run の chain 評価を測って | train-eval | train-eval | OK |
| chain 評価の定義を教えて | なし | なし (train-eval が「定義を知りたいだけなら evaluation.md」と除外) | OK |
| TensorBoard のログを要約して | なし (agent log-analyst) | なし。log-analyst の description に合う | OK |
| v8 を作って | add-version | add-version | OK |
| v7 をもとに観測を変えた版を足して | add-version | add-version | OK |
| 新しい版を登録して | add-version | add-version | OK |
| v6 の報酬のバグを直して | なし | なし (add-version が「既存の版のバグ修正」を除外) | OK |
| v4 の仕様を教えて | なし | なし (「版の仕様を調べるだけ」を除外) | OK |
| ソルバの反復を変えたので学習の前に確かめて | physics-check | physics-check | OK |
| ゲート 1 を回して | physics-check | physics-check | OK |
| 新しいプリセットの発散率を測って | physics-check | physics-check | OK |
| 学習中の発散率のログを見て | なし | なし (「学習中のログの発散率を読むだけ」を除外) | OK |
| 発散の定義は | なし | なし (どの description も定義の説明を対象にしていない) | OK |
| 今の結果を実験記録にまとめて | experiment-record | experiment-record | OK |
| この run は無効なので退避して | experiment-record | experiment-record (「無効にして退避する」) | OK |
| 切り分けの報告書を書いて | experiment-record | experiment-record (「結果を docs/experiments/ に」)。「報告書」の語は無いが「結果を…書く」に合う | OK (弱い) |
| STATUS を更新して | handoff | handoff (experiment-record は「STATUS の更新だけなら handoff」と除外) | OK |
| README の誤記を直して | なし | なし | OK |
| 今日はここまで。引き継ぎを書いて | handoff | handoff | OK |
| 次のセッション用にまとめて | handoff | handoff | OK |
| HANDOFF.md の昔の記録を探して | なし | 修正前は「引き継ぐ」の語で handoff に寄りえた。description に「過去の引き継ぎメモ (docs/history/) を探すときは使わない」を足した | OK (修正後) |
| 実験記録を書いて | experiment-record (handoff ではない) | experiment-record。handoff は「結果の記録は experiment-record」と除外 | OK |

限界: 判定はメインが description を読んで行ったもので、実際に Claude Code が自動で呼ぶかは確かめていない (新しい session でないと skill の一覧に載らない。Phase 5 で使ってみて確かめる)。
