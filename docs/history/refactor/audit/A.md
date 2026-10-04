# A. 文書の棚卸し

対象: `/home/itoyama/work/isaac_twisted_rl` の HEAD `6f41ef1` (ブランチ `refactor/claude-foundation`)。
追跡されている Markdown 23 本 (`git ls-files '*.md'`)、`paper/` の全ファイル (PDF 3 本・HTML 4 本・md 3 本)、
`versions/v3/HTML/` の HTML 2 本、`.py` のモジュール docstring (`ast` で抜き出し)。
`logs/`・`outputs/` は `ls` だけ。Isaac Sim・GPU・pytest は動かしていない。`refactor/` は `audit/C.md` だけを読んだ。

表記:

* パスはリポジトリのルートからの相対。`kt/` = `src/isaac_twisted_rl/tasks/knot_tying/`。根拠は `path:line`。
* 用語は C.md の「推奨する正式名称」に揃える: ロープ `Knot` (28 リンクのプリセット)、段 (v4 以降の `BfsPotential.stage`)、
  位相クラス (v1〜v3 の報酬ラダーの段)、chain 評価、Play タスク、ログ名 (experiment_name)、初期アーカイブ、
  固結び (overhand knot)、紐モデル (`RopeModel`)、紐のバックエンド、版 vN (MDP の版)、1 手。
* 情報の分類は依頼の 8 分類 (Spec = Current specification、Rationale = Design rationale、Result = Verified result、
  Hypo = Hypothesis、Open = Open question、Repro = Reproducibility metadata、Hist = Historical context、
  Obs = Obsolete information)。割合は全文を読んだうえでの**目安** (行数を数えた値ではない)。

---

## 0. 要点 (重大な順。詳細は §8)

| 番号 | 一言で | 根拠 (代表) |
|---|---|---|
| A-1 | **「現在地」が 8 か所に分かれ、食い違う。** v3 の結果は `kt/README.md` では「smoke test のみ」、ルートの README と next_action では「25%」。最新の「現在地」(next_action、10-02) は 10-03 の rope_study と推論時探索を含まない | `kt/README.md:41`、`README.md:62`、`kt/next_action.md:16`、`kt/versions/v3/README.md:164` |
| A-2 | **v2-R2 の「結果」は検証されていない。** v2-R2 の run は 1 本 (2 iteration、文書自身が「比較に使えない」)。それでも 3 つの入口文書が「本物の C2 が稀で詰まる」を結果欄に書く | `kt/versions/v2_r2/README.md:134-136`、`paper/Report_Paper.md:50`、`README.md:61`、`kt/README.md:40` |
| A-3 | **リンク切れは 11 本 (md) + 4 本 (HTML)、原因はすべて `paper/Report.md` → `paper/Report_Paper.md` の改名 (`dbf9a97`)。** その結果 `Report_Paper.md` へのリンクは 0 本 | §5.1 |
| A-4 | **ルートの README から辿れない文書が 32 本中 14 本。** rope_study の 3 本、Discussion、Report_Paper、論文の図解 HTML、v3 の HTML、PDF | §6.2 |
| A-5 | **経緯と現在が混ざり、最新が下に埋もれる文書がある。** next_action (名前と中身が違う、追記 5)、Result_Report_v6 (追記 12、本文の仮説を後の引用ブロックで否定)、HANDOFF (逆時系列 + 「最新」の印が 3 か所) | `kt/next_action.md:37-42`、`kt/versions/v6/Result_Report_v6.md:149-152`、`kt/rope_study/HANDOFF.md:8`、`:158`、`:194` |
| A-6 | **next_action.md の節番号はコードから参照されている** (48 行 / 16 ファイルの .py)。分割すると参照が切れる | `kt/next_action.md:11-12`、§6.4 |
| A-7 | **結果の数字が最大 12 か所に重複し、細部で食い違う** (v4 の段 2 成功 26% / 27%、v4 の学習中の発散 2.9% / 3〜4%、v3 の 1 手の発散 3% / 5〜6%、発散の判定) | §7.2 |
| A-8 | **古くなった「未実装」「次の候補」が残る** (推論時探索は実装済みなのに「未実装」、v3 の「本番の学習は未実施」など) | `paper/Report_Paper.md:403`、`kt/versions/v4/README.md:342-343`、`kt/versions/v3/README.md:164-166` |
| A-9 | **再現の手掛かりが追跡外に置かれている** (切り分けの上書きは `logs/` の README、評価の表は `outputs/`、探索のレーンは `logs/` の .sh)。HANDOFF のログのパスは実在しない | `kt/README.md:81`、`paper/Report_Search.md:268`、`kt/rope_study/HANDOFF.md:292` |
| A-10 | **開発の約束とコマンドが 6 か所に書かれ、少しずつ違う。`CLAUDE.md` は無い** | §7.1 の「開発の約束」 |

ほかに A-11〜A-15 (§8)。

---

## 1. 方法

| 調べたこと | 方法 |
|---|---|
| 対象の一覧・行数・最終更新 | `git ls-files '*.md' '*.html' '*.pdf'`、`wc -l`、`git log -1 -- <file>` |
| 文書の中身 | 23 本の Markdown を全文読んだ (Report_rope_models・HANDOFF・Report_Paper・Discussion は節ごとに読み、Discussion は §1・§12 と見出しの全部) |
| HTML | `<title>` と本文の冒頭を抜き出して何の文書かを判断し、相対 `href` の存在を検査 |
| PDF | 中身は読まない。ファイル名・大きさ (`ls -la`)・参照元 (`git grep`) |
| リンク | 付録 B.1 のスクリプト (相対リンク・アンカー・行アンカー)。補助に B.2 (バッククォートのファイル名) |
| 地図 | B.1 の結果から「リンクの辺」を作り、`README.md` から幅優先で辿った |
| docstring | 付録 B.3 のスクリプト (`ast.get_docstring`)。15 行以上は 61 本 |
| 重複 | 数字・見出しを `git grep -n` で数え、読んだ箇所と突き合わせた |

---

## 2. 文書ごとの表 (追跡されている Markdown 23 本)

「重複・矛盾」の列は主なものだけ。数字の重複の全体は §7。

| ファイル | 行数 | 現在の役割 (誰が・いつ読む) | 情報の種類 (多い順、目安) | 他の文書との重複・矛盾 | AI にとっての問題 | 推奨する移動先・分割先 |
|---|---|---|---|---|---|---|
| `README.md` | 143 | GitHub の入口。初めての人がインストールと版の一覧を見る (英語) | Spec 55 / Result 20 / Repro 10 / Hist 10 / Obs 5 | 版の結果表 `:56-66` = `kt/README.md:35-45` = `kt/next_action.md:18-32`。v2-R2 の状態 `:61` は未検証 (A-2) | 文書の表 `:110-118` に rope_study・Report_Search・scripts の README が無い。`:118` はリンク切れ。英語なので他文書と用語が揃わない (`Twisted-RL` の意味は C-4) | 概要とインストールだけ残す。版の表 → `docs/STATUS.md`、文書の表 → `docs/README.md` へのリンク 1 本 |
| `kt/README.md` | 247 | パッケージの入口。開発者が使い方・版の一覧・規約・開発の約束を見る | Spec 45 / Rationale 20 / Result 15 / 手順・規約 (Spec) 15 / Obs 5 | 版の表 `:41` (v3「smoke test のみ」) が `README.md:62`・`kt/next_action.md:27` (25%) と矛盾。開発の約束 `:229-247` は 6 か所に重複 (§7.1) | 「現在地」と仕様と約束が 1 本に混在。木 `:105-152`・タスク ID `:55-64`・ログ名 `:75-77` が古い (A-11)。`:27` はリンク切れ。`-Play-` の説明 `:52-53` は v4 以降と合わない (C-6) | 版の表 → `STATUS.md`。開発の約束 → `CLAUDE.md` (要点) + `docs/workflow.md` (詳細)。「版 = 設定」`:154-173` → `docs/decisions/`。木・タスク ID → `docs/system/layout.md`。Kit 前 import の注意 `:214-227` → `docs/system/` と `tasks/__init__` の docstring |
| `kt/common/README.md` | 196 | 全版に共通の仕組みの説明。環境の挙動を知りたいとき | Spec 70 / Rationale 20 / Result 5 / Hist 5 | 行動の説明 `:79-110` ≒ `kt/next_action.md:231-287`。240 Hz の実測 `:41-45` ≒ `kt/next_action.md:319-338` | 表が v3 で止まっている: フェーズの尺 `:52-66` に v4〜v7、終了判定 `:142-150` に v4 の `stage_up` / `stage_down` が無い | ほぼそのまま `docs/system/env-mechanics.md` (またはコードの隣に残す)。尺の変遷 `:68-77` → `docs/decisions/` |
| `kt/next_action.md` | 583 | 設計判断と経緯の記録。冒頭に「現在地と次の作業」。コードの docstring が節番号で参照する | Hist 40 / Rationale 25 / Result 15 / Open 10 / Spec 5 / Obs 5 | 現在地 `:16-52` は `README.md:56-66` ほか 7 か所と重複。行動の根拠 `:247-287` ≒ `common/README.md:94-99`。gate 1 `:484-549` ≒ `v3/README.md:35-37`、`Report_rope_models.md` §2.2 | 名前 (next action) と中身 (経緯の記録) が違う。現在地は 10-02 時点で rope_study・推論時探索を含まない。日付の追記 `:37-42`。地図 `:56-70` が v3 で止まる。`:532` の「(未実装)」は v4 で実装済み。`:25`・`:63` はリンク切れ。節番号を変えられない (A-6) | 現在地 → `STATUS.md`。§4.0・§4.1・§4.8.1〜4.8.4・§6 の判断 → `docs/decisions/` (1 判断 1 ファイル)。§1〜§3・§4.6・§6 の経緯 → `docs/history/`。§4.8.10 (gate 1) → `docs/experiments/`。旧節番号 → 新しい場所の対応表を残す |
| `kt/versions/v0/README.md` | 80 | v0 の仕様 (観測・報酬・PPO) | Spec 75 / Rationale 15 / Repro 10 | 結果は `v2/README.md:17-33` にしか無い | 結果を別の版の文書で読む必要がある | 仕様 → `docs/system/versions/v0.md` (またはコードの隣)。基準コミット `:16` → Repro として残す |
| `kt/versions/v1/README.md` | 115 | v1 の仕様と、学習前の確認 | Spec 45 / Rationale 30 / Result 15 / Hist 10 | 位相クラスのラダー `:35-50` ≒ `kt/next_action.md:436-441`、`knot_detect.py` の docstring | 「段」を位相クラスの意味で使う (C-1) | 仕様 → `system/versions/`。ポテンシャル整形の理由 `:52-61` → `decisions/` |
| `kt/versions/v1/diff_to_TwistedRL.md` | 287 | v1 時点で移植元 (MuJoCo 版) のコードと比べたスナップショット | Hist 60 / Rationale 25 / Spec (移植元の構造) 15 | `paper/Report_Paper.md` §2〜§4 と論点が重なる | 冒頭 `:7-12` に「v1 時点」と注記はあるが、v2〜v7 の後では読む価値の判断が付かない。`:8` はリンク切れ。移植元のパス (`exploration/...`) はリポジトリの外 | `docs/history/2026-09_port_vs_twisted_rl.md`。移植元の構造の要点は `docs/glossary.md` |
| `kt/versions/v2/README.md` | 276 | v2 の仕様と、v1 の失敗の分析 | Rationale 30 / Result 30 / Spec 25 / Hist・Obs 15 | v1 の分析 `:17-57` と PPO の変更理由 `:64-78` が `v2/agents/rsl_rl_ppo_cfg.py` の docstring (68 行) と重複 | `:201` の「結果はこの節に追記する」は未履行 (GRU / LSTM の学習ログは無い: `paper/Discussion.md:45`)。§5 `:237-244` の v3 構想は採られなかった。`:244` はリンク切れ | 仕様 → `system/versions/`。v1 の失敗の分析 → `experiments/2026-09-15_v1.md`。rsl-rl 3.0.1 の注意 `:203-222` → `docs/system/` か `workflow.md` (版に依らない知識) |
| `kt/versions/v2_r2/README.md` | 191 | v2-R2 の仕様と、偽の C2・発散の報酬の修正 | Spec 40 / Result 35 / Rationale 15 / Hist 10 | 位相クラスの表 `:68-76` | 修正後の学習が無いのに「次の課題」`:181-187` が結果のように読める (A-2)。`:12`・`:50` はリンク切れ | 仕様 → `system/versions/`。修正の確認 `:130-179` → `experiments/2026-09-27_v2r2_fix.md` |
| `kt/versions/v3/README.md` | 168 | v3 の仕様 (アーカイブ・成功で終了・BFS) | Spec 45 / Rationale 25 / Result (smoke) 15 / Obs 10 / Open 5 | BFS の説明 `:80-107` ≒ `kt/common/topology/knot_graph.py` の docstring | `:164`「本番の学習は未実施」、`:165-166`「初期アーカイブは未実装」は古い (A-8)。`:17` はリンク切れ | 仕様 → `system/versions/`。「実装中に分かったこと」`:94-107` → `decisions/` |
| `kt/versions/v3/Result_report_v3.md` | 198 | v3 の本番 run (`2026-09-28_17-28-57`) の結果報告 | Result 55 / Open 15 / Hypo 15 / Repro 10 / 追記 5 | §8 の追記 `:185-198` = `v5/README.md:65-80` | §7 の次にやること `:174-181` は v4・v5 でほぼ消化済み。`:140` の `paper/Report.md` は古い名前 | `docs/experiments/2026-09-28_v3_main.md` (そのまま)。§7 は Hist 扱いと明記 |
| `kt/versions/v4/README.md` | 351 | v4 の仕様、学習速度の調査、結果 | Result 40 / Spec 35 / Rationale 15 / Open 5 / Repro 5 | 結果 `:290-310` = `Result_Report_v6.md` §3.1。論文の数字 `:36` = `paper/Report_Paper.md:100` | 長い。仕様・速度の調査・結果・今後が 1 本。`:342-343` (推論時探索) と `:349-350` (発散の調査) は済んでいる。`:349`「v3 で約 3%」は v3 報告と食い違う (§7.2) | 仕様 → `system/versions/v4.md`。速度の調査 `:148-288` → `experiments/2026-09-30_v4_speed.md`、その判断 (反復 8・4 手) → `decisions/`。結果 → `experiments/` |
| `kt/versions/v5/README.md` | 204 | v5 の仕様と、発散の計測 | Result 55 / Spec 20 / Rationale 15 / Open 10 | `:65-80` = `Result_report_v3.md:185-198`。結果表 `:173-178` = `Result_Report_v6.md` §3 | 「計測では効いたが学習では逆効果」という最重要の教訓 `:189` が版の README の中にある | 計測 → `experiments/2026-10-01_v5_divergence.md`。教訓 → `decisions/` と `workflow.md` |
| `kt/versions/v6/README.md` | 191 | v6 の仕様 (次にたどる辺) と台本との照合 | Spec 45 / Result 35 / Rationale 15 / 重複 5 | 切り分けの表 `:174-180` = `Result_Report_v6.md:233-262` = `v7/README.md:19-26` | 結果の数字が版の README・報告・v7 に三重 | 仕様 → `system/versions/v6.md`。照合 `:76-125` → `experiments/` |
| `kt/versions/v6/Result_Report_v6.md` | 311 | v4 / v5 / v6 の overnight 比較と、v6 の切り分け (追記) | Result 55 / Open 15 / Hypo 15 / Repro 10 / Hist 5 | v4〜v6 の数字が 10 文書に重複 (§7.2) | 「追記」12 か所。要旨 7 `:21-25` と §4.2 の引用ブロック `:149-152` で本文の仮説を後から否定。最新の結論 (切り分け) は末尾 `:215-311`。`:63` の `eval_compare.py` は存在しない | `experiments/2026-10-01_v4-v6_overnight.md` と `experiments/2026-10-02_v6_ablation.md` に分割。次の候補 `:290-297` → `STATUS.md` |
| `kt/versions/v7/README.md` | 81 | v7 の仕様と、まだ学習していないこと | Result 50 (重複) / Spec 25 / Open 15 / Hist 10 | 切り分けの表 `:19-26` は `Result_Report_v6.md:235-241` の写し | 「v7」が過去には「v6 + 時間切れの罰則」を指した `:77-81`、`Result_Report_v6.md:226`、`HANDOFF.md:171` | 仕様だけ → `system/versions/v7.md`。表は experiments への参照に置き換え |
| `kt/rope_study/README.md` | 112 | rope_study の使い方 (コマンド集) | Spec (手順) 80 / Result (注意) 20 | ファイル表 `:77-97` ≒ `HANDOFF.md:317-340` ≒ `Report_rope_models.md:182-211` | どこからもリンクされていない (§6) | コードの隣に残す。`docs/README.md` からリンク |
| `kt/rope_study/HANDOFF.md` | 439 | rope_study のセッション間の引き継ぎメモ | Hist 45 / Open 20 / Repro 15 / Result 10 / Obs 10 | 次にやること `:16-25` = `Report_rope_models.md:769-787`。参考文献 `:420-439` = `Report_rope_models.md:861-880` | 逆時系列で、節番号が `00000`→`0000`→`000`→`00`→`0`→`1`。「最初にここを読む」`:8`、「← 最新」`:158`、「次のセッションはここから読む」`:194` が並ぶ。`:5`「中間版」と `:10`「完成」が矛盾。PID・別セッションの scratchpad のパス `:170` など今は無意味な情報。`:292` のログのパスは実在しない | 全体 → `docs/history/2026-10-02_rope_study_sessions.md` (凍結)。未解決 → `STATUS.md`。踏んだ注意 (`kill`・Kit の固まり・`pkill`) `:27-32`、`:148-156` → `workflow.md` |
| `kt/rope_study/Report_rope_models.md` | 880 | 紐モデルの比較の最終報告 | Result 40 / Spec (実装) 20 / Rationale (文献) 15 / Hypo 10 / Open 10 / Repro 5 | §2.2 `:75-84` は v0〜v5 の経緯の要約。§5.2 は HANDOFF §3.2 と重複。参考文献は HANDOFF §7 と重複 | 長い。冒頭に要旨 (§0) はあるが、どの節が「今も有効な仕様」かが分からない。`:323`「(it100)」と `:601`「最後は model_99」が不一致。数値の表 `:7` は追跡外 (`outputs/`) | `experiments/2026-10-02_rope_models.md` (本体)。§10 の選定 → `decisions/rope-model-choice.md`。§5 の実装 → `common/rope_backends/` の README か docstring。§12 → `STATUS.md` |
| `src/isaac_twisted_rl/scripts/rsl_rl/README.md` | 190 | 学習・評価スクリプトの使い方と、ビーム探索のアルゴリズム | Spec 60 / Rationale 20 / Result 15 / 手順 5 | 複製の検証 `:154-174` = `paper/Report_Search.md:118-134`。作ったもの `:26-28` = `Report_Search.md:37-49` | 良い形 (コードの隣の仕様)。段の説明 `:92` が C-1 の「交差数に当たる」 | そのまま残す。`docs/README.md` からリンク |
| `paper/Report_Paper.md` | 466 | 2026-09-26 時点の v2-R2 と 3 論文の比較 (+ 10-01 の実装状況の追記) | Hist 45 / Rationale (論文の要点) 30 / Open 10 / Obs 10 / Result 5 | 論文の数字 `:98-101` = `v4/README.md:36`、`stage_gated.py` の docstring | リンク 0 本 (全部 `Report.md` を指す)。§9 `:364-425` は 10-01 時点の状況で、`:403` (P2-4 未実装)・`:378`・`:415-421` (v5 未コミット) が古い。行アンカー 9 本のうち 6 本が別の行を指す (§5.2) | `docs/history/2026-09-26_paper_comparison.md` (凍結)。§2 (論文の要点) → `docs/system/papers.md` か glossary。§9 → `STATUS.md` |
| `paper/Report_Search.md` | 304 | 推論時探索の中間報告 (2026-10-03、途中で停止) | Result 50 / Open 15 / Spec 15 / Hypo 10 / Repro 10 | §1・§3.1 = `scripts/rsl_rl/README.md`。chain 評価の数字 `:140-144` = 他 10 文書 | 「V8」`:9`、`:230`、`:269` はどこにも定義が無い。再現に追跡外の `logs/rsl_rl/eval_search_lane.sh` `:268` を使う | `experiments/2026-10-03_inference_search.md`。§6 → `STATUS.md` |
| `paper/Discussion.md` | 1078 | 研究方針の提案 (2026-10-01)。E0〜E7 の実験計画 | Hypo 60 / Rationale (論文の事実) 20 / Spec (前提) 10 / Open 10 | 前提の表 `:62-76` は v4 の仕様の写し。論文の要約は `Report_Paper.md` §2 と重複 | 最長。どこからも参照されない (リンクも言及も 0)。`:13`・`:38` はリンク切れ。`:1067`「1 run しか無い現状」は古い (E0・S1 は eval_chain・eval_search で一部実現)。主張のラベル `:21-29` は良い慣行 | 叩き台に「提案」の置き場が無い。選択肢: (a) `docs/history/2026-10-01_research_directions.md` で凍結し、E0〜E7 のうち生きているものを `STATUS.md` の候補へ、(b) `docs/proposals/` を新設 |

---

## 3. `paper/` の Markdown 以外と、`versions/v3/HTML/`

| ファイル | 大きさ | 何の文書か (冒頭で判断) | どこから参照されているか | 問題 | 推奨 |
|---|---|---|---|---|---|
| `paper/Learning_Topological_Motion_Primitives_for_Knot-Plannning.pdf` | 2,390,694 B | Yan2020 の論文 | リンク: `paper/Discussion.md:37`。HTML: `paper/explainer/index.html:658`、`paper/explainer/yan2020.html:452` | ファイル名の綴り `Plannning` (n が 3 つ) | 置き場は `paper/` のまま (または `docs/papers/`)。改名するなら参照 3 か所も |
| `paper/Twisted.pdf` | 558,395 B | Sudry ら TWISTED (CoRL 2023) | リンク: `paper/Discussion.md:35`。HTML: `index.html:658`、`twisted.html:452` | 名前から論文が分かりにくい (C-4) | 同上 |
| `paper/Twisted-RL.pdf` | 6,342,180 B | Freund ら TWISTED-RL (arXiv 2602.14526) | リンク: `paper/Discussion.md:36`。HTML: `index.html:658`、`twisted-rl.html:452`。言及: `README.md:135` | 同上 | 同上 |
| `paper/explainer/index.html` | 658 行 / 56,697 B | 「紐結び 3 論文の図解」の入口 (前提知識・比較表・用語集) | Markdown からのリンクは 0。`v3/HTML/*.html:167`、`:189` の `href` | `:323` の `href="../Report.md"` が切れている | 論文の解説 (Rationale の参考資料)。`docs/README.md` からリンク |
| `paper/explainer/twisted.html` | 915 行 | TWISTED の図解 | `paper/Discussion.md:35` | — | 同上 |
| `paper/explainer/twisted-rl.html` | 984 行 | TWISTED-RL の図解 | `paper/Discussion.md:36`。コードの docstring 4 か所 (`kt/common/mdp/stages.py:3`、`:145`、`kt/common/mdp/archive.py:234`、`kt/common/policies/stage_gated.py:5`)、`kt/versions/v4/README.md:33` | `:787` の `href="../Report.md"` が切れている | 同上。コードが節番号 (§02・§06・§07) で参照するので節番号は保つ |
| `paper/explainer/yan2020.html` | 993 行 | Yan2020 の図解 | `paper/Discussion.md:37` | — | 同上 |
| `kt/versions/v3/HTML/v3-implementation.html` | 894 行 / 91,410 B | v3 の実装の図解 (コミット `3b71145` 時点) | Markdown からのリンクは 0。`v3-vs-papers.html:167`、`:501` | 「現在の最新版 v3」`:193`、「本番の学習は未実施」`:197`、`:863` が古い。`:241` の `paper/Report.md` が切れている | Hist (凍結したスナップショット)。`docs/history/` へ。冒頭に「2026-09-28 時点」と明記 |
| `kt/versions/v3/HTML/v3-vs-papers.html` | 509 行 / 58,180 B | v3 と 3 論文の比較と考察 (予想) | `kt/versions/v3/Result_report_v3.md:126` (言及)、`paper/Report_Paper.md:400` (言及) | `:173` の `paper/Report.md` が切れている。予想の答え合わせは `Result_report_v3.md` §4 | 同上 |

---

## 4. 主要な docstring (モジュール docstring、15 行以上の上位 20 本)

`ast.get_docstring` で抜き出した (付録 B.3)。15 行以上は 61 本 / docstring のあるモジュール 123 本 / 追跡されている .py 124 本。
列の「参照」は docstring 中の `next_action` / `README` / `Report` / `.md` / `explainer` の数、「実測」は数値と単位を含む行の数。

| 順 | ファイル | 行数 | 参照 | 実測 | 中身 (情報の種類) | 文書との重複・問題 | 推奨 |
|---|---|---|---|---|---|---|---|
| 1 | `kt/common/rope_model/generate_rope_usd.py` | 70 | 0 | 0 | Spec (関節の構成・単位変換) + Rationale (捩りを解放する理由) | `:7` の `assets/generate_rope_usd.py` は追跡ファイルに無い。`:15` の `knot_tying/generate_assets.py` は実際は `demos/generate_assets.py`。`:24-33` は「4 本」で `knot`・`knot56` が無い | コードに残す。古い 3 か所を直す |
| 2 | `kt/versions/v2/agents/rsl_rl_ppo_cfg.py` | 68 | 3 | 4 | Rationale + Result (v1 の TensorBoard の値) + Obs | `kt/versions/v2/README.md:17-78` と重複。`:18`「段別の学習は v3 で…実装する」は採られなかった (`kt/versions/v2/README.md:242`) | 理由を数行に縮め、数字は experiments へ |
| 3 | `kt/demos/overhand_script.py` | 68 | 9 | 9 | Spec (使い方・台本) + Rationale | 台本の説明は `kt/next_action.md:494-499`、`:542-549` と重複。`:3` に `paper/Report.md` | コードに残す (台本の仕様はここが正) |
| 4 | `kt/common/topology/intersections.py` | 68 | 0 | 1 | Spec (数式) + Rationale (移植元との差) | `kt/next_action.md:125-134` と重複 | コードに残す (正はここ) |
| 5 | `kt/common/topology/knot_graph.py` | 64 | 7 | 4 | Spec + Rationale + Result (2026-09-28 実測) + Hist (「最初は…」) | `kt/versions/v3/README.md:80-107` と重複。`:7` に `paper/Report.md` | 仕様と理由は残す。経緯は decisions へ |
| 6 | `kt/demos/rope_catch.py` | 58 | 0 | 0 | Spec (使い方) + Rationale (摩擦だけで掴む) | `kt/common/README.md:107-110` と重複 | コードに残す |
| 7 | `kt/demos/topology_debug.py` | 56 | 2 | 0 | Rationale + Hist (Step 5) | `kt/next_action.md:174-216` と重複 | コードに残す |
| 8 | `kt/common/topology/knot_detect.py` | 53 | 0 | 5 | Rationale + Result (528 状態の全列挙) | `kt/next_action.md:398-431`、`kt/versions/v1/README.md:48-50` と重複。`:91` に `paper/Report.md` | 正はここ。next_action 側を削る |
| 9 | `kt/common/mdp/actions.py` | 52 | 8 | 3 | Spec + Rationale | `kt/common/README.md` §3 と重複 | 残す |
| 10 | `kt/common/rope_state.py` | 49 | 0 | 0 | Spec + Result (body の順序、日付 1) | `kt/next_action.md:155-170` と重複 | 残す |
| 11 | `kt/common/parallel_gripper/generate_gripper_usd.py` | 49 | 0 | 0 | Spec + Rationale | — | 残す |
| 12 | `kt/common/topology/state_2_topology.py` | 47 | 0 | 8 | Rationale (テンソル層と記号層を分ける理由) + Result (94 ms / 1.6 ms) | 94 ms / 1.6 ms は 5 か所以上に重複 (`kt/common/README.md:114`、`kt/next_action.md:100-107`、`:306`、`kt/versions/v1/diff_to_TwistedRL.md:224`) | 正はここ |
| 13 | `kt/common/rope_model/rope_specs.py` | 42 | 0 | 0 | Spec (プリセット一覧) | プリセットの数が文書で 4 / 5 / 6 と揺れる (C-15) | 正はここ |
| 14 | `kt/common/policies/stage_gated.py` | 42 | 1 | 0 | Spec + Rationale + 論文の数字 | 段を「交差数に当たる」と書く (C-1) | 残す。段の定義を直す |
| 15 | `kt/common/topology/representation.py` | 41 | 0 | 0 | Spec | — | 残す |
| 16 | `kt/demos/generate_assets.py` | 40 | 0 | 0 | Spec (使い方) + Rationale (USD をコミットしない理由) | — | 残す |
| 17 | `kt/common/mdp/topology_log.py` | 40 | 0 | 3 | Spec + Rationale + Hist (v2-R2 で追加) | `kt/versions/v2_r2/README.md:90-104` と重複 | 残す |
| 18 | `kt/common/topology/next_edge.py` | 39 | 1 | 3 | Spec + Rationale | `kt/versions/v6/README.md:38-74` と重複 | 残す |
| 19 | `kt/common/topology/__init__.py` | 39 | 0 | 0 | Spec (パッケージの地図) | `kt/next_action.md:79-89` の木と重複 | 正はここ |
| 20 | `kt/demos/multi_rope_catch.py` | 37 | 0 | 0 | Spec + Rationale | `rope_catch.py` と同じ理由を重複して書く | 残す |

読み取り:

* 上位 20 本は大半が「Spec + Rationale」で、コードの隣に置く価値がある。経緯や実測が多いのは 2・5・8・12。
* 文書との重複は「docstring が正、文書が写し」の形が多い (4・8・10・12・19)。文書側 (主に next_action) を削るほうが安全。
* `paper/Report.md` を docstring で参照する .py が 10 本ある (`git grep -c 'paper/Report\.md' -- '*.py'`: `kt/common/mdp/{archive,rewards,terminations}.py`、`kt/common/state_archive.py`、`kt/common/topology/{knot_detect,knot_graph}.py`、`kt/demos/overhand_script.py`、`kt/versions/v3/{__init__,env_cfg}.py`、`kt/versions/v4/agents/rsl_rl_ppo_cfg.py`。ほかに `tests/test_knot_detect.py`)。
* 開発の約束が「docstring には何をするかより、なぜそうしたかを書く」(`kt/README.md:244-245`) としているので、理由と経緯が docstring に溜まる。経緯 (Hist) をどこまで docstring に置くかの方針が要る (A-13)。

---

## 5. リンクの検査

### 5.1 壊れた相対リンク (Markdown)

付録 B.1 のスクリプトで、追跡されている Markdown 23 本の `[..](..)` (外部 URL を除く) 139 本を調べた。
結果: 実在 118、行アンカー 9、見出しアンカー 1 (存在する)、**切れ 11**。11 本すべてが `paper/Report.md` (改名済み) を指す。

| file:line | リンク先 | 正しいと思われる先 |
|---|---|---|
| `README.md:118` | `paper/Report.md` | `paper/Report_Paper.md` |
| `paper/Discussion.md:13` | `Report.md` | `Report_Paper.md` |
| `paper/Discussion.md:38` | `Report.md` | `Report_Paper.md` |
| `kt/README.md:27` | `../../../../paper/Report.md` | `../../../../paper/Report_Paper.md` |
| `kt/next_action.md:25` | `../../../../paper/Report.md` | `../../../../paper/Report_Paper.md` |
| `kt/next_action.md:63` | `../../../../paper/Report.md` | `../../../../paper/Report_Paper.md` |
| `kt/versions/v1/diff_to_TwistedRL.md:8` | `../../../../../../paper/Report.md` | `../../../../../../paper/Report_Paper.md` |
| `kt/versions/v2/README.md:244` | `../../../../../../paper/Report.md` | 同上 |
| `kt/versions/v2_r2/README.md:12` | `../../../../../../paper/Report.md` | 同上 |
| `kt/versions/v2_r2/README.md:50` | `../../../../../../paper/Report.md` | 同上 |
| `kt/versions/v3/README.md:17` | `../../../../../../paper/Report.md` | 同上 |

原因: commit `dbf9a97` (2026-10-03 10:41「paperフォルダ内の Report.md を Report_Paper.md に変更」) が参照を直していない
(`git log --diff-filter=R --summary`: `rename paper/{Report.md => Report_Paper.md} (100%)`)。

### 5.2 行アンカー (`#L<n>`) の検査

`paper/Report_Paper.md` だけが行アンカーを使う。ファイルの範囲には収まるが、2026-09-26 以降の変更で **9 本中 6 本が別の行** を指す。

| file:line | リンク先 | 指したいもの (リンクの文) | 今の行 |
|---|---|---|---|
| `paper/Report_Paper.md:114` | `common/mdp/phases.py#L45` | 運搬高さ 0.20 m | `CARRY_PALM_Z` は `:46` (1 行ずれ) |
| `paper/Report_Paper.md:115` | `common/env_cfg_base.py#L102` | `reset_all` | `:102` (一致) |
| `paper/Report_Paper.md:228` | `common/mdp/actions.py#L532` | `_record_grasp_distance` | `:677` |
| `paper/Report_Paper.md:228` | `common/mdp/rewards.py#L231` | `grasp_distance_penalty` | `:354` |
| `paper/Report_Paper.md:229` | `common/mdp/terminations.py#L28` | `rope_diverged` | `:29` (1 行ずれ) |
| `paper/Report_Paper.md:252` | `common/topology/knot_detect.py#L150` | `num_slots = min(...)` | 該当行は無い (`MAX_CROSSINGS_CONSIDERED` は `:80`、使う関数は `:152-159`) |
| `paper/Report_Paper.md:268` | `common/mdp/rewards.py#L80` | `crossing_achieved` | `:77` |
| `paper/Report_Paper.md:269` | `common/topology/knot_detect.py#L202` | `topology_class` | `:233` |
| `paper/Report_Paper.md:275` | `common/mdp/actions.py#L255` | 行動のクリップ | `:320` (`actions.clamp(-1.0, 1.0)`) |

この文書は「2026-09-26 時点のスナップショット」(`paper/Report_Paper.md:10`) なので、行アンカーはコミット固定の permalink にするか、関数名に置き換えるのがよい。

### 5.3 HTML の相対 `href` (補助)

| file:line | href | 正しいと思われる先 |
|---|---|---|
| `paper/explainer/index.html:323` | `../Report.md` | `../Report_Paper.md` |
| `paper/explainer/twisted-rl.html:787` | `../Report.md` | `../Report_Paper.md` |
| `kt/versions/v3/HTML/v3-implementation.html:241` | `../../../../../../../paper/Report.md` | `.../paper/Report_Paper.md` |
| `kt/versions/v3/HTML/v3-vs-papers.html:173` | `../../../../../../../paper/Report.md` | 同上 |

### 5.4 リンクでないファイル参照 (補助、付録 B.2)

バッククォートの中のファイル名で、basename が追跡ファイルに 1 つも無いもの (外部ライブラリ・移植元のパスを除いた主なもの):

| 参照 | 箇所 | 状態 |
|---|---|---|
| `paper/Report.md` (改名済み) | md: `README.md:118`、`paper/Discussion.md:38`、`kt/README.md:27`、`kt/next_action.md:25`、`:63`、`kt/versions/v1/diff_to_TwistedRL.md:8`、`kt/versions/v2/README.md:244`、`kt/versions/v2_r2/README.md:12`、`kt/versions/v3/README.md:17`、`kt/versions/v3/Result_report_v3.md:140`。ほかに .py 11 本 (§4) | C.md の C-16 と同じ (C.md は 26 行 / 23 ファイルと数えた) |
| `eval_compare.py` | `kt/versions/v6/Result_Report_v6.md:63` | 追跡ファイルに無い |
| `logs/rsl_rl/eval_search_lane.sh` | `paper/Report_Search.md:268` | 追跡外 (`ls logs/rsl_rl` で実在) |
| `outputs/rope_study/eval/summary.md`、`outputs/rope_study/train_summary.md` | `kt/rope_study/Report_rope_models.md:7`、`:601`、`:857`、`kt/rope_study/HANDOFF.md:14` | 追跡外。報告書の数値の出どころ |
| `outputs/rope_study/diag_scripts/*.py`、`gate1_redo.sh` | `kt/rope_study/Report_rope_models.md:839`、`kt/rope_study/HANDOFF.md:97`、`:128` | 追跡外。再現手順 §13 が依存 |
| 別セッションの scratchpad (`/tmp/claude-1002/.../7e308401-.../small_queue.sh`)、`eval_queue2.sh`、`pipeline2.sh` | `kt/rope_study/HANDOFF.md:170`、`:101`、`:171`、`:293` | 消える場所。今は意味を持たない |

---

## 6. 文書の地図の現状

### 6.1 入口と導線

| 入口 | リンクしている文書 | 足りないもの |
|---|---|---|
| `README.md` (文書の表 `:110-118`) | `kt/README.md`、`kt/common/README.md`、`kt/next_action.md`、(切れ) `paper/Report.md` | rope_study の 3 本、`scripts/rsl_rl/README.md`、`paper/Report_Search.md`、`paper/Discussion.md`、図解 HTML |
| `kt/README.md` (文書の表 `:21-27` + 本文 `:37-45`、`:102-103`) | `common/README.md`、`next_action.md`、`versions/v0〜v7/README.md`、`versions/v6/Result_Report_v6.md`、`scripts/rsl_rl/README.md`、`paper/Report_Search.md`、(切れ) `paper/Report.md` | rope_study、Discussion、Result_report_v3 (next_action 経由のみ)、diff_to_TwistedRL (木の中に名前だけ `:137`) |
| `kt/next_action.md` (§0 地図 `:56-70` + 現在地の表 `:18-32`) | `README.md`、`common/README.md`、`versions/v0〜v2_r2/README.md`、`v3/Result_report_v3.md`、`v4`・`v5`・`v7/README.md`、`v6/Result_Report_v6.md`、`v1/diff_to_TwistedRL.md`、(切れ) `paper/Report.md` | 地図は v3 で止まる。v4 以降・rope_study・推論時探索・Discussion が無い |
| `kt/rope_study/README.md` | `Report_rope_models.md` | 自分自身がどこからもリンクされていない |

`CLAUDE.md` はリポジトリに無い (`git ls-files` と `ls` で確認)。`.claude/` は空。

### 6.2 どこからもリンクされていない文書 (Markdown のリンクの辺、`README.md` から幅優先)

`README.md` から辿れるのは 32 本 (md 23 + HTML 6 + PDF 3) 中 18 本。辿れない 14 本:

| 文書 | 被リンク (md) | 言及 (リンクでない) |
|---|---|---|
| `paper/Discussion.md` | 0 | 0 (`git grep -F 'Discussion.md'` で 0 件) |
| `paper/Report_Paper.md` | 0 (11 本が旧名 `Report.md` を指す) | 4 行 / 4 ファイル: `paper/Report_Search.md:275`、`src/isaac_twisted_rl/scripts/rsl_rl/README.md:16`、`src/isaac_twisted_rl/scripts/rsl_rl/eval_search.py:9`、`kt/common/beam_search.py:3` |
| `kt/rope_study/README.md` | 0 | HANDOFF・Report_rope_models の中だけ (`kt/rope_study/HANDOFF.md:251`、`kt/rope_study/Report_rope_models.md:210`) |
| `kt/rope_study/HANDOFF.md` | 0 | `kt/rope_study/README.md:96` (表の中) |
| `kt/rope_study/Report_rope_models.md` | 1 (`kt/rope_study/README.md:4`、ただしその README が孤立) | .py 7 行 / 6 ファイル (`kt/common/rope_backends/adapters.py:264`、`kt/common/rope_backends/xpbd_rod.py:145`、`:154`、`kt/demos/overhand_script.py:728`、`kt/rope_study/__init__.py:6`、`tests/test_rope_backends.py:219`、`:292`) |
| `paper/explainer/*.html` 4 本 | Discussion からのみ (`paper/Discussion.md:35-37`) | .py 4 本 (`twisted-rl.html` の節番号) |
| `kt/versions/v3/HTML/*.html` 2 本 | 0 | `kt/versions/v3/Result_report_v3.md:126`、`paper/Report_Paper.md:400` |
| PDF 3 本 | Discussion からのみ | `README.md:135` ("papers in `paper/`") |

深さ: `README.md` → 深さ 1 が 3 本 (`kt/README.md`、`kt/common/README.md`、`kt/next_action.md`)、深さ 2 が 15 本。

### 6.3 被リンクの数 (多い順)

| 被リンク | 文書 |
|---|---|
| 8 | `kt/next_action.md` |
| 7 | `kt/versions/v2/README.md`、`kt/versions/v3/README.md` |
| 6 | `kt/common/README.md` |
| 5 | `kt/README.md`、`kt/versions/v2_r2/README.md` |
| 4 | `kt/versions/v1/README.md`、`kt/versions/v1/diff_to_TwistedRL.md` |
| 3 | `kt/versions/v4/README.md`、`kt/versions/v6/Result_Report_v6.md` |
| 2 | `kt/versions/v0`・`v5`・`v7/README.md`、`paper/Report_Search.md`、`scripts/rsl_rl/README.md` |
| 1 | `kt/versions/v6/README.md`、`kt/versions/v3/Result_report_v3.md`、`kt/rope_study/Report_rope_models.md` |
| 0 | §6.2 の表 |

### 6.4 コードからの文書参照 (移動の制約)

| 参照先 | 数 | 内訳 |
|---|---|---|
| `next_action.md` | .py で 48 行 / 16 ファイル (`git grep -c -F next_action -- '*.py'`) | 節番号つき: §4.8.10 が 15、§3(c) が 9、§4.8.9 と §4.1 が各 7、§4.5 が 6、§4.2 が 5、§4.0 が 4、§4.3 が 3、§3(b) が 2、§4.8.8 と §4.8.2 が各 1 (md を含む集計) |
| `paper/Report.md` (旧名) | .py 11 本 | §4 の読み取りの 3 つ目 |
| `explainer/twisted-rl.html` の節 | .py 4 か所 | §3 の表 |

`kt/next_action.md:11-12` は「節番号は変えないこと」と書く。next_action を分割するなら、(a) 旧節番号 → 新しいファイルの対応表を next_action.md に残す、(b) docstring の参照を一括で書き換える、のどちらかが要る。

---

## 7. 重複の地図

### 7.1 話題ごとの重複

| 話題 | 書かれている場所 | 数 | 正にすべき場所 (提案) |
|---|---|---|---|
| 版ごとの結果 (状態の一覧) | `README.md:56-66`、`kt/README.md:35-45`、`kt/next_action.md:18-32`、`paper/Report_Paper.md:368-378` (§9.0)、`kt/versions/v5/README.md:173-178`、`kt/versions/v6/README.md:155-159`、`kt/versions/v6/Result_Report_v6.md:15-25`、`paper/Report_Search.md:136-144` | 8 | `docs/STATUS.md` の 1 表 (数字は experiments への参照) |
| 次にやること | `kt/next_action.md:44-52`、`kt/versions/v6/Result_Report_v6.md:185-197`、`:290-297`、`kt/versions/v7/README.md:73-75`、`kt/versions/v5/README.md:191-204`、`kt/versions/v4/README.md:338-351`、`kt/versions/v3/README.md:162-168`、`kt/versions/v3/Result_report_v3.md:174-181`、`paper/Report_Search.md:258-275`、`kt/rope_study/HANDOFF.md:16-25`、`kt/rope_study/Report_rope_models.md:769-787`、`paper/Discussion.md:1061-1078`、`paper/Report_Paper.md:296-360` | 13 | `docs/STATUS.md` (上書き) |
| 開発の約束・コマンド | `kt/README.md:229-247`、`README.md:104-108`、`kt/next_action.md:553-555` (移したという注記)、`kt/versions/v2/README.md:203-207`、`kt/rope_study/HANDOFF.md:300-305`、`kt/rope_study/README.md:31-32`、`kt/rope_study/Report_rope_models.md:791-798` | 7 | `CLAUDE.md` (要点) + `docs/workflow.md` |
| ディレクトリ構成・ファイル表 | `kt/README.md:105-152`、`kt/next_action.md:79-89`、`kt/rope_study/README.md:77-97`、`kt/rope_study/HANDOFF.md:317-340`、`kt/rope_study/Report_rope_models.md:182-211`、`src/isaac_twisted_rl/scripts/rsl_rl/README.md:3-8`、`paper/Report_Search.md:39-46`、`kt/common/topology/__init__.py` の docstring | 8 | パッケージごとの README / `__init__` docstring。全体の木は `docs/system/layout.md` の 1 か所 |
| タスク ID の一覧 | `kt/README.md:55-64`、`kt/rope_study/README.md:21-24`、`kt/rope_study/Report_rope_models.md:196` | 3 | `docs/system/` (C.md §3 の機械的な一覧を元に) |
| ログ名の一覧 | `kt/README.md:75-81`、各版の README の「ログ:」行 9 本 (`kt/versions/v0/README.md:8` ほか)、`kt/rope_study/README.md:83` | 11 | `docs/system/` の 1 表 |
| 行動空間 | `kt/common/README.md:79-110`、`kt/next_action.md:231-287`、`kt/versions/v1/diff_to_TwistedRL.md:233-244`、`paper/Report_Paper.md:114`、`paper/Discussion.md:70`、`kt/common/mdp/actions.py` の docstring | 6 | `actions.py` の docstring と `common/README.md` |
| フェーズの尺 | `kt/common/README.md:52-77`、`kt/next_action.md:361-367`、`kt/versions/v2/README.md:68`、`kt/versions/v3/README.md:36`、`kt/common/mdp/phases.py` の docstring | 5 | `phases.py` |
| 観測の次元 (75 / 105 / 129 / 133 / 197) | `kt/versions/v0/README.md:35`、`kt/versions/v1/README.md:26`、`kt/versions/v4/README.md:51`、`kt/versions/v6/README.md:27`、`paper/Discussion.md:71`、`kt/rope_study/Report_rope_models.md:281` | 6 | 各版の仕様 |
| 発散の判定 | `kt/common/README.md:147`、`kt/versions/v2_r2/README.md:154`、`kt/versions/v5/README.md:47`、`kt/rope_study/Report_rope_models.md:287` | 4 | `terminations.py` + glossary (C-7) |
| ゲート 1 の結果 | `kt/next_action.md:484-549`、`kt/README.md:50-52`、`kt/versions/v3/README.md:35-37`、`kt/versions/v4/README.md:205-221`、`kt/versions/v5/README.md:152-166`、`kt/rope_study/Report_rope_models.md:365-552` | 6 | `docs/experiments/` (gate 1 の回ごと) |
| 学習速度 (34 ms + 0.019 ms × 環境数 など) | `kt/next_action.md:477-480`、`kt/versions/v4/README.md:148-199`、`kt/versions/v3/Result_report_v3.md:116-120`、`kt/rope_study/Report_rope_models.md:94` (§2.3 の 7) | 4 | `experiments/2026-09-30_v4_speed.md` |
| v6 の切り分けの表 | `kt/versions/v6/Result_Report_v6.md:233-262`、`kt/versions/v6/README.md:174-180`、`kt/versions/v7/README.md:19-26` | 3 | experiments の 1 か所 |
| 論文の数字 (TWISTED-RL-C 3-Medium 44% など) | `paper/Report_Paper.md:98-101`、`kt/versions/v4/README.md:36`、`kt/common/policies/stage_gated.py` の docstring、`paper/explainer/twisted-rl.html` | 4 | 図解 HTML か `docs/system/papers.md` |
| 参考文献 (紐モデル) | `kt/rope_study/HANDOFF.md:420-439`、`kt/rope_study/Report_rope_models.md:861-880` | 2 | 報告書 |
| p-data の速さ (94 ms / 1.6 ms) | `kt/common/README.md:114-115`、`kt/next_action.md:100-107`、`:306`、`kt/versions/v1/diff_to_TwistedRL.md:224`、`kt/common/topology/knot_detect.py` / `state_2_topology.py` の docstring | 6 | `state_2_topology.py` |

開発の約束の食い違い (同じことを違うコマンドで書く):

| 項目 | 書き方 A | 書き方 B |
|---|---|---|
| テスト | `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/` (`kt/README.md:236`、`README.md:107`) | `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/` (`kt/rope_study/HANDOFF.md:302`、`kt/rope_study/Report_rope_models.md:797`) |
| lint | `ruff check <path> && ruff format <path>` (`kt/README.md:240-241`) | `uvx ruff@0.14.10 check --line-length 120 <path>` (`kt/rope_study/HANDOFF.md:303`、`kt/rope_study/Report_rope_models.md:798`) |
| 学習 | `uv run train ...` (`kt/README.md:10-18` ほか) | `.venv/bin/train ...` (`kt/rope_study/Report_rope_models.md:817`) |
| テストの件数 | 116 件 (`kt/versions/v1/README.md:105`、09-15)、140 passed (`paper/Report_Paper.md:51`、09-26)、241 件 (`kt/rope_study/HANDOFF.md:302`)、247 件 (`kt/rope_study/HANDOFF.md:14`、`kt/rope_study/Report_rope_models.md:796`) | 今の件数は未確認 (pytest を回していない。`def test_` は 177 個、parametrize を含まない) |

### 7.2 結果の数字の重複と食い違い

同じ数字が書かれている場所 (読んで確かめたもの):

| 数字 | 場所 | 数 |
|---|---|---|
| v4 の chain 評価 41.6% (iteration 100) | `README.md:63`、`kt/README.md:42`、`kt/next_action.md:28`、`kt/versions/v4/README.md:303`、`kt/versions/v5/README.md:175`、`kt/versions/v6/README.md:159`、`kt/versions/v6/Result_Report_v6.md:15`、`:77`、`paper/Report_Search.md:140`、`kt/rope_study/Report_rope_models.md:613`、`:662`、`kt/rope_study/HANDOFF.md:395` | 12 (10 文書) |
| v5 の chain 評価 9.6% | `README.md:64`、`kt/README.md:43`、`kt/next_action.md:29`、`kt/versions/v4/README.md:310`、`kt/versions/v5/README.md:175`、`kt/versions/v6/README.md:159`、`kt/versions/v6/Result_Report_v6.md:15`、`:79`、`paper/Report_Search.md:141` | 9 |
| v6 の chain 評価 37.9% / 22.9% | `README.md:65`、`kt/README.md:44`、`kt/next_action.md:30`、`kt/versions/v4/README.md:310`、`kt/versions/v5/README.md:175`、`kt/versions/v6/README.md:159`、`kt/versions/v6/Result_Report_v6.md:15`、`:81`、`:83`、`paper/Report_Search.md:142-143` | 10 |
| v7 相当の run の chain 評価 31.2% (iteration 70) | `README.md:66`、`kt/README.md:45`、`kt/versions/v6/README.md:178`、`kt/versions/v6/Result_Report_v6.md:22`、`:259`、`:268`、`kt/versions/v7/README.md:35`、`:71`、`paper/Report_Search.md:144` | 9 |
| v3 の `scratch_success` 25% | `README.md:62`、`kt/next_action.md:27`、`kt/versions/v3/Result_report_v3.md:12`、`kt/versions/v4/README.md:182` | 4 |

食い違い:

| 何 | A | B | 判断 |
|---|---|---|---|
| v3 の学習の状態 | 「smoke test のみ」`kt/README.md:41`、「本番の学習は未実施」`kt/versions/v3/README.md:164`、`kt/versions/v3/HTML/v3-implementation.html:197` | 「25% (iteration 150〜159)」`README.md:62`、`kt/next_action.md:27`、`kt/versions/v3/Result_report_v3.md:12` | B が正 (run `logs/rsl_rl/knot_tying_v3/2026-09-28_17-28-57` が実在)。A は古い |
| v2-R2 の学習の状態 | 「本物の C2 が稀で詰まる」(結果欄) `README.md:61`、`kt/README.md:40`、`kt/next_action.md:24` | run は 1 本・2 iteration・「比較に使えない」`kt/versions/v2_r2/README.md:134-136`、`paper/Report_Paper.md:50`。値は iteration 0〜1 のもの `kt/versions/v3/Result_report_v3.md:77`。修正版で 300〜500 iteration 回す提案 `paper/Report_Paper.md:357` の記録は無い | A は Hypo。`ls logs/rsl_rl/knot_tying_v2_r2` は `2026-09-26_15-57-53` の 1 本だけ |
| v4 の段 2 の学習成功 (最後) | 27% `kt/versions/v6/Result_Report_v6.md:17`、`:142` | 26% `kt/versions/v5/README.md:177`、0.26 `kt/versions/v6/Result_Report_v6.md:99`、`:111` | 表は 0.26。27% は「80〜89」の値 (`:98`) か丸めの違い (未確認) |
| v4 の学習中の「発散で終わったエピソード」 | 3〜4% `kt/versions/v4/README.md:308` | 2.9% (最後の 10 iteration) `kt/versions/v5/README.md:176`、`kt/versions/v6/Result_Report_v6.md:121` | 区間の取り方の違いと思われる (未確認) |
| v3 の 1 手あたりの発散 | 約 3% `kt/versions/v4/README.md:349` | 5〜6% `kt/versions/v5/README.md:13`、iteration 50 で 2.9%・159 で 6.0% `kt/versions/v3/Result_report_v3.md:14` | v4 README は iteration 50 前後の値を使っている。最新の値と合わない |
| 発散の判定 | NaN・水平 3 m・速さ 20 m/s `kt/rope_study/Report_rope_models.md:287` | + 高さ 5 m `kt/common/README.md:147`、`kt/versions/v5/README.md:47` | Report_rope_models が 1 条件を落としている (C-7 と関連) |
| rope_study の報告の状態 | 「中間版」`kt/rope_study/HANDOFF.md:5` | 「完成」`kt/rope_study/HANDOFF.md:10`、「最終版」`kt/rope_study/Report_rope_models.md:3` | 後者が新しい |
| rope_study の評価のチェックポイント | 「(it100)」`kt/rope_study/Report_rope_models.md:323` | 最後は `model_99.pt` `kt/rope_study/Report_rope_models.md:601`、`kt/rope_study/HANDOFF.md:29` | 後者が正 |
| v6 の切り分けのログの場所 | `logs/rsl_rl/knot_tying_v6/*_abl_{relA,depB,sw2}_s42`、`knot_tying_v7/*_abl_toP_s42` `kt/rope_study/HANDOFF.md:292` | `logs/rsl_rl/knot_tying_v6_ablation/` `kt/versions/v6/Result_Report_v6.md:220`、`kt/versions/v7/README.md:70` | `ls logs/rsl_rl` で B が実在、`knot_tying_v7/` は無い |
| 「v7」の中身 | v6 + 時間切れの罰則 (当時) `kt/versions/v6/Result_Report_v6.md:226`、「別セッションは v7 の学習を終え」`kt/rope_study/HANDOFF.md:171` | v6 + めり込み解消速度 0.96 m/s、未学習 `kt/versions/v7/README.md:1`、`:69` | 版の中身が途中で入れ替わった。HANDOFF の「v7」は旧義 |
| 推論時探索の実装状況 | 「❌ 未実装」`paper/Report_Paper.md:403`、「今後の候補」`kt/versions/v4/README.md:342-343` | 実装済み・中間結果あり `paper/Report_Search.md:5-10`、commit `72d18df` | B が正 |
| ロープのプリセットの数 | 「5 種」`kt/README.md:7`、`kt/next_action.md:69` / 「4 プリセット」`kt/next_action.md:358` / 「4 本」`kt/common/rope_model/generate_rope_usd.py:24` | `ROPE_SPECS` は 6 種 (C-15) | C.md C-15 と同じ |

---

## 8. 所見 (重大な順)

| 番号 | 所見 | なぜ重いか (AI にとって) | 根拠 | 提案の方向 |
|---|---|---|---|---|
| A-1 | **「現在地」が 8 か所に分かれ、互いに食い違う。** 版の状態の表が `README.md`・`kt/README.md`・`next_action.md` に三重にあり、v3 は「smoke test のみ」と「25%」が並ぶ。最も新しく見える `next_action.md` の現在地 (2026-10-02) は、10-03 の rope_study と推論時探索を含まない。次にやることは 13 か所 | 冒頭で読む文書によって、今どの版が基準か・何が済んだかの答えが変わる | `kt/README.md:41`、`README.md:62`、`kt/next_action.md:16-52`、`kt/versions/v3/README.md:164`、§7.1 の「次にやること」 | `docs/STATUS.md` を 1 本にし、上書きで更新。他の文書は STATUS へのリンクだけを持つ |
| A-2 | **v2-R2 の「結果」は検証されていない。** 修正後の v2-R2 の学習ログは無く、残る run は 2 iteration で文書自身が「比較に使えない」とする。それでも入口の 3 文書が「本物の C2 が稀で詰まる」を結果として書き、v3 の動機になっている | Hypothesis が Verified result として読まれる。「v2-R2 は駄目だった」を前提に推論してしまう | `kt/versions/v2_r2/README.md:134-136`、`:181-187`、`paper/Report_Paper.md:50`、`:153-161`、`:357`、`kt/versions/v3/Result_report_v3.md:77`、`README.md:61`、`kt/README.md:40`、`kt/next_action.md:24`、`ls logs/rsl_rl/knot_tying_v2_r2` | STATUS と実験記録で「仮説 (ランダム方策と 2 iteration の観察)」と明記 |
| A-3 | **リンク切れ 11 本 (md) + 4 本 (HTML)。原因は 1 つ (`paper/Report.md` の改名)。** `Report_Paper.md` への被リンクは 0。.py の docstring 11 本も旧名を指す | 参照を辿ると存在しないファイルに着く。改名の影響を検査する仕組みが無い | §5.1、§5.3、commit `dbf9a97` | 参照を直す。付録 B.1 を CI か pre-commit に入れる |
| A-4 | **入口から辿れない文書が 32 本中 14 本。** rope_study の 3 本 (README・HANDOFF・最終報告)、Discussion、Report_Paper、図解 HTML 6 本、PDF 3 本。ルートの README の文書の表のリンクは 4 本 (うち 1 本は切れ) | 紐モデルの比較の結論 (Report_rope_models §10) や研究計画 (Discussion) を、入口から読み始めた AI は見つけられない | §6.2、`README.md:110-118`、`kt/README.md:21-27`、`kt/next_action.md:56-70` | `docs/README.md` (地図) に全文書を「状態 (現行 / 凍結)・対象読者」付きで載せる |
| A-5 | **経緯・現在・結果が 1 文書に混ざり、日付の追記で最新が下に埋もれる。** `next_action.md` は名前が「次の作業」だが中身の 7 割は経緯で、追記が冒頭に溜まる。`Result_Report_v6.md` は「追記」12 か所、本文の仮説 (§4.2) を後の引用ブロックで否定し、最新の結論が末尾。`HANDOFF.md` は逆時系列で節番号が `00000`〜`0`、「最初にここを読む」「← 最新」「次のセッションはここから読む」が 3 か所 | どこまで読めば最新かが分からない。古い節の「次にやること」を今の指示と取り違える | `kt/next_action.md:1-4`、`:37-42`、`kt/versions/v6/Result_Report_v6.md:21-25`、`:149-152`、`:187-188`、`kt/rope_study/HANDOFF.md:5`、`:8`、`:10`、`:158`、`:194` | 実験記録は「日付つき・凍結」、現在地は STATUS に上書き、判断は decisions に 1 判断 1 ファイル |
| A-6 | **`next_action.md` の節番号はコードから参照されている** (.py で 48 行 / 16 ファイル。§4.8.10 だけで 15)。文書自身が「節番号は変えないこと」と書く | 分割・改名すると docstring の参照がすべて宙に浮く (A-3 と同じ事故) | `kt/next_action.md:11-12`、§6.4 | 移行時に「旧節番号 → 新しい場所」の対応表を残すか、docstring を一括で書き換える。順番を決めてから分割する |
| A-7 | **結果の数字が最大 12 か所に重複し、細部で食い違う。** v4 の段 2 成功 (26% / 27%)、v4 の学習中の発散 (2.9% / 3〜4%)、v3 の 1 手の発散 (約 3% / 5〜6%)、発散の判定 (高さ 5 m の有無)、rope_study の評価の iteration (100 / 99) | 重複が多いほど更新漏れが起き、どの値が正かを AI が判定できない | §7.2 | 数字は experiments の 1 か所だけに書き、他はリンク |
| A-8 | **古くなった「未実装」「未実施」「次の候補」が残っている。** 推論時探索 (実装済み) を「未実装」、v3 の本番学習 (実施済み) を「未実施」、初期アーカイブ (v4 で実装) を「未実装」、v5 を「未コミット」、v3 での段別タスクの構想 (採られなかった) を「v3 で実装する」 | 済んだ作業を再度提案・実装してしまう | `paper/Report_Paper.md:378`、`:403`、`:415-421`、`kt/versions/v4/README.md:342-343`、`:349-350`、`kt/versions/v3/README.md:164-166`、`kt/next_action.md:532`、`kt/versions/v2/agents/rsl_rl_ppo_cfg.py:18`、`kt/versions/v2/README.md:201`、`paper/Discussion.md:1067` | 凍結する文書の冒頭に「この時点のスナップショット」と書き、現行の状態は STATUS へ |
| A-9 | **再現の手掛かりが追跡外 (消えうる場所) にある。** 切り分けの run の上書きは `logs/` の README、推論時探索のレーンは `logs/` の .sh、rope_study の評価表と切り分けのスクリプトは `outputs/`、別セッションの scratchpad。HANDOFF のログのパスは実在しない | git だけでは結果を再現・検証できない。HANDOFF のパスを辿ると空振りする | `kt/README.md:81`、`kt/versions/v6/Result_Report_v6.md:5`、`:302`、`paper/Report_Search.md:268`、`kt/rope_study/Report_rope_models.md:7`、`:839`、`kt/rope_study/HANDOFF.md:97`、`:170`、`:292`、`ls logs/rsl_rl` | 再現に要るもの (上書きの一覧・小さいスクリプト) は `docs/experiments/` か `scripts/` に追跡する |
| A-10 | **開発の約束とコマンドが 7 か所に書かれ、少しずつ違う。`CLAUDE.md` は無い。** テスト (`uv run pytest` / `.venv/bin/python -m pytest`)、lint (`ruff check` / `uvx ruff@0.14.10 check --line-length 120`)、学習 (`uv run train` / `.venv/bin/train`)、テストの件数 (116 / 140 / 241 / 247) | どれが正規の手順か決まらない。AI が毎回探し直す | §7.1 の食い違いの表、`kt/README.md:229-247` | `CLAUDE.md` に要点 (禁止事項・テスト・lint・`python -u`・GPU 共有の注意)、`docs/workflow.md` に詳細 |
| A-11 | **構成・一覧の記述が古い。** `kt/README.md` の木に `rope_backends/`・`rope_study/`・`beam_search.py`・`mdp/state_clone.py`・demos 3 本 (`multi_rope_catch`・`spawn_rope`・`parallel_gripper`) が無い。タスク ID の一覧に `RS-*` が無い。ログ名の一覧は実在しない `_rnn`・`_lstm`・`v7` を挙げ、実在する `v4_speed`・`v6_ablation`・`rs_*` が無い。`common/README.md` の表は v3 で止まる | 構成の説明を信じると、ファイルの在りかを誤る | `kt/README.md:55-64`、`:75-81`、`:105-152`、`kt/common/README.md:52-66`、`:142-150`、`git ls-files kt/`、`ls logs/rsl_rl` | 木は 1 か所 (`docs/system/layout.md`) にし、パッケージの詳細は各 `__init__` docstring / README に任せる (C.md C-16 と同じ) |
| A-12 | **長い文書に「誰が・いつ読むか」「現行か凍結か」の宣言が無い。** Discussion 1078 行、Report_rope_models 880 行、next_action 583 行、Report_Paper 466 行、HANDOFF 439 行。宣言があるのは Report_Paper `:10` と diff_to_TwistedRL `:7` だけ | 冒頭で読むべきかが判断できず、文脈を浪費する | 各文書の冒頭 (§2 の表) | 全文書の冒頭に 3 行の頭書き (状態・対象読者・最終更新) を付ける。`docs/README.md` にも同じ列 |
| A-13 | **docstring に経緯と実測が溜まり、文書と二重になっている。** 15 行以上の docstring は 61 本。v2 の PPO 設定 (68 行) は v2 README の写し、`knot_graph.py` (64 行) は v3 README と同じ経緯を持つ。開発の約束が「なぜそうしたかを書く」を求める | どちらを直せばよいか分からず、片方だけ古くなる (例: `kt/versions/v2/agents/rsl_rl_ppo_cfg.py:18`) | §4、`kt/README.md:244-245` | docstring = 仕様と「今も有効な理由」、経緯と実測値 = experiments / history、と線を引く |
| A-14 | **HTML の図解はスナップショットで、古い記述とリンク切れを含む。** v3 の図解は「現在の最新版 v3」「本番の学習は未実施」。HTML 4 本に `Report.md` への切れたリンク。PDF のファイル名に綴りの誤り (`Plannning`) | 図解を現状の説明と取り違える | `kt/versions/v3/HTML/v3-implementation.html:193`、`:197`、`:863`、§5.3、`paper/Learning_Topological_Motion_Primitives_for_Knot-Plannning.pdf` | v3 の HTML は history へ。論文の図解は参考資料として地図に載せる |
| A-15 | **用語・名前の小さな問題。** 「V8」が定義なしに使われる。ルートの README だけ英語。`next_action.md` という名前が中身と合わない (A-5)。文書の置き場に `archive/` を使うと状態アーカイブと衝突する (C.md C-11) | 新しい語が説明なしに増える | `paper/Report_Search.md:9`、`:230`、`:269`、`README.md:1-143` | glossary に版の命名規則を書く。古い文書の置き場は `docs/history/` (叩き台どおり) |

---

## 9. 確認できなかったこと

* 修正後の v2-R2 の学習が過去に存在したか (削除された可能性)。`logs/rsl_rl/knot_tying_v2_r2/` には 1 本しか無く、文書にも記録が無い。未確認。
* `kt/versions/v6/Result_Report_v6.md:17` の「27%」が何の区間の値か。未確認。
* `kt/versions/v4/README.md:308` の「3〜4%」の区間。未確認。
* 現在のテストの件数と合否 (pytest を回していない。`def test_` は 177 個)。
* `logs/rsl_rl/knot_tying_v6_ablation/README.md`、`logs/rsl_rl/eval_search_lane.sh`、`outputs/rope_study/eval/summary.md` の中身 (開いていない)。文書の数字がそれらと一致するかは未確認。
* `paper/Discussion.md` の §2〜§11 (約 900 行) は見出しと一部しか読んでいない。§1 と §12 は全部読んだ。本文の中の数字と他文書との食い違いは未確認。
* HTML の本文 (図解の中身) と論文の記述が一致するか。PDF は読んでいない。
* GitHub の見出しアンカーの生成規則は近似で実装した (`paper/Report_Paper.md:22` の 1 本だけが対象で、存在すると判定)。
* 行アンカーの「指したいもの」は、リンクの文と周辺の記述から推定した (§5.2)。

---

## 付録 A. 使ったコマンド (主なもの)

```bash
cd /home/itoyama/work/isaac_twisted_rl
git ls-files '*.md' | while read f; do echo "$(wc -l < "$f") $f"; done        # 23 本と行数
ls -la paper paper/explainer src/isaac_twisted_rl/tasks/knot_tying/versions/v3/HTML
for f in $(git ls-files '*.md' '*.html'); do git log -1 --format='%h %ad' --date=short -- $f; done
git log --diff-filter=R --summary --format='%h %s' | grep -B1 rename          # Report.md の改名
git grep -n -F 'Report_Paper' ; git grep -n -F 'Discussion.md'                  # 言及の数
git grep -c -F 'next_action' -- '*.py'                                         # 48 行 / 16 ファイル
git grep -h -o -E 'next_action\.md[^0-9§]{0,3}§?[0-9]+(\.[0-9]+)*(\([a-z]\))?' -- '*.py' '*.md' | sort | uniq -c
git grep -c 'paper/Report\.md' -- '*.py'
git grep -c -E '^\s*def test_' -- tests                                        # 177
ls logs/rsl_rl logs/rsl_rl/*/ ; ls outputs                                      # 一覧だけ
```

## 付録 B. スクリプト

### B.1 相対リンクの検査 (`check_links.py`)

実行: `python3 check_links.py /home/itoyama/work/isaac_twisted_rl` (出力は TSV: 種別, file:line, リンク先, 状態, 補足)。
地図 (§6) は、この出力の `OK` / `LINE` / `ANCHOR` の行から「元の文書 → 先の文書」の辺を作り、`README.md` から幅優先で辿った。

```python
"""追跡されている Markdown の相対リンク [..](..) を検査する。

使い方: python3 check_links.py <repo_root>
出力: TSV (種別, file:line, リンク先, 状態, 補足)
"""
import re, subprocess, sys, unicodedata
from pathlib import Path

root = Path(sys.argv[1]).resolve()
md_files = subprocess.run(["git", "-C", str(root), "ls-files", "*.md"],
                          capture_output=True, text=True, check=True).stdout.split()

LINK = re.compile(r"(?<!!)\[([^\]]*)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
FENCE = re.compile(r"^\s*(```|~~~)")


def slug(h: str) -> str:
    """GitHub の見出しアンカー (近似): 小文字化、記号を除去、空白を - に。"""
    h = h.strip().lower()
    h = re.sub(r"`", "", h)
    out = []
    for ch in h:
        cat = unicodedata.category(ch)
        if ch in "-_ " or cat[0] in "LN" or cat in ("Mn", "Mc"):
            out.append(ch)
    return "".join(out).replace(" ", "-")


def anchors(path: Path):
    res, in_fence = set(), False
    for line in path.read_text(encoding="utf-8").splitlines():
        if FENCE.match(line):
            in_fence = not in_fence
            continue
        if not in_fence:
            m = re.match(r"^(#{1,6})\s+(.*?)\s*#*\s*$", line)
            if m:
                res.add(slug(m.group(2)))
    return res


rows = []
for rel in md_files:
    src = root / rel
    in_fence = False
    for i, line in enumerate(src.read_text(encoding="utf-8").splitlines(), 1):
        if FENCE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        for m in LINK.finditer(line):
            target = m.group(2)
            if re.match(r"^[a-z]+://|^mailto:", target):
                continue
            path_part, _, frag = target.partition("#")
            dest = src if path_part == "" else (src.parent / path_part).resolve()
            loc = f"{rel}:{i}"
            if not dest.exists():
                rows.append(("BROKEN", loc, target, "missing", ""))
                continue
            try:
                dest_rel = dest.relative_to(root)
            except ValueError:
                rows.append(("OUTSIDE", loc, target, "outside repo", str(dest)))
                continue
            tracked = subprocess.run(["git", "-C", str(root), "ls-files", "--error-unmatch", str(dest_rel)],
                                     capture_output=True).returncode == 0 or dest.is_dir()
            if frag:
                if re.fullmatch(r"L\d+(-L?\d+)?", frag):
                    n = int(re.match(r"L(\d+)", frag).group(1))
                    lines = dest.read_text(encoding="utf-8").splitlines()
                    txt = lines[n - 1].strip()[:80] if n <= len(lines) else "<beyond EOF>"
                    rows.append(("LINE", loc, target, "ok" if n <= len(lines) else "beyond EOF", txt))
                    continue
                if dest.suffix == ".md":
                    ok = frag.lower() in anchors(dest)
                    rows.append(("ANCHOR", loc, target, "ok" if ok else "anchor missing", ""))
                    continue
            rows.append(("OK" if tracked else "UNTRACKED", loc, target, "ok", str(dest_rel)))

for r in rows:
    print("\t".join(r))
```

HTML の相対 `href` は次の数行で調べた:

```python
import re, subprocess
from pathlib import Path
root = Path(".").resolve()
for rel in subprocess.run(["git", "ls-files", "*.html"], capture_output=True, text=True).stdout.split():
    s = (root / rel).read_text(encoding="utf-8")
    hrefs = set(re.findall(r'(?:href|src)="([^"#:]+)(?:#[^"]*)?"', s))
    bad = [h for h in hrefs if not (root / rel).parent.joinpath(h).resolve().exists()]
    print(rel, "relative hrefs:", len(hrefs), "broken:", bad)
```

### B.2 バッククォートのファイル参照の検査 (`check_code_paths.py`)

```python
"""Markdown のバッククォート内のファイル参照 (.md/.py/.sh/.html) が、追跡ファイルに実在するかを調べる。

リンクでない参照 (例 `paper/Report.md`) の切れを拾うための補助。basename が追跡ファイルに 1 つも無いものを出す。
使い方: python3 check_code_paths.py <repo_root>
"""
import re, subprocess, sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
tracked = subprocess.run(["git", "-C", str(root), "ls-files"], capture_output=True, text=True).stdout.split()
basenames = {Path(p).name for p in tracked}
CODE = re.compile(r"`([^`\s]+?\.(?:md|py|sh|html))(?::[\d\-]+)?`")
for rel in [p for p in tracked if p.endswith(".md")]:
    for i, line in enumerate((root / rel).read_text(encoding="utf-8").splitlines(), 1):
        for m in CODE.finditer(line):
            ref = re.sub(r"[{}]", "", m.group(1).split("/")[-1])
            if "*" in ref or "<" in ref or "," in m.group(1):
                continue
            if ref not in basenames:
                print(f"{rel}:{i}\t{m.group(1)}")
```

### B.3 モジュール docstring の抜き出し (`docstrings.py`)

```python
"""追跡されている .py のモジュール docstring を ast で抜き出し、行数の多い順に並べる。

列: 行数 / ファイル / 日付 (2026-MM-DD) の数 / 版 (v0〜v7・v2-R2) の言及数 / 文書への参照数 / 実測値らしき数値の行数 /
    クラス・関数の docstring の行数の合計 / 1 行目
使い方: python3 docstrings.py <repo_root> [min_lines]
"""
import ast, re, subprocess, sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
min_lines = int(sys.argv[2]) if len(sys.argv) > 2 else 15
files = subprocess.run(["git", "-C", str(root), "ls-files", "*.py"], capture_output=True, text=True).stdout.split()
rows = []
for rel in files:
    src = (root / rel).read_text(encoding="utf-8")
    try:
        mod = ast.parse(src)
    except SyntaxError:
        continue
    doc = ast.get_docstring(mod, clean=False)
    if not doc:
        continue
    lines = doc.strip("\n").splitlines()
    total_doc = sum(len((ast.get_docstring(nd, clean=False) or "").splitlines())
                    for nd in ast.walk(mod) if isinstance(nd, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)))
    dates = len(re.findall(r"20\d\d-\d\d-\d\d", doc))
    vers = len(re.findall(r"\bv[0-7](?:-R2)?\b", doc))
    refs = len(re.findall(r"next_action|README|Report|\.md\b|explainer", doc))
    meas = sum(1 for l in lines if re.search(r"\d+(\.\d+)?\s?(%|ms|m/s|件|手|環境|s\b)", l))
    rows.append((len(lines), rel, dates, vers, refs, meas, total_doc, lines[0].strip()[:60] if lines else ""))
rows.sort(reverse=True)
for r in rows:
    if r[0] >= min_lines:
        print("\t".join(map(str, r)))
print("count>=%d:" % min_lines, sum(1 for r in rows if r[0] >= min_lines),
      "/ modules with docstring:", len(rows), "/ py files:", len(files))
```

---

## メインによる根拠の確認 (2026-10-03)

メインの session が、次の 5 か所を自分で開いて確かめた。誤りは見つからなかった。

| 結論 | 確かめた内容 | 結果 |
|---|---|---|
| v3 の結果の食い違い | `kt/README.md:41` は「smoke test」、`v3/README.md:164` は「本番の学習は未実施」。`README.md:62` は「25% (iteration 150)」、`next_action.md:27` は「25% (iteration 150〜159)」 | 一致 |
| v2-R2 の未検証の結果が結果欄にある | `v2_r2/README.md:134-136` は「この run は比較に使えない」。`README.md:61` の結果欄は「stuck on the rare real intermediate state」 | 一致 |
| v4 の段 2 の成功率が 27% と 0.26 | `Result_Report_v6.md:17` は「v4 が 6% -> 27%」。:99 の「最後の 10」の行の最初の列は `0.96 / 0.56 / 0.26` | 一致 (丸めの差の範囲。食い違いとして扱うかは Phase 2 で判断する) |
| 発散の判定から高さ 5 m が落ちている | `Report_rope_models.md:287` は「NaN、水平に 3 m 以上、リンクの速さ 20 m/s 以上」。`common/mdp/terminations.py:31` は `max_height: float = 5.0` | 一致 |
| HANDOFF.md:292 のログのパスは実在しない | :292 は `logs/rsl_rl/knot_tying_v6/*_abl_*` と `knot_tying_v7/*_abl_toP_s42` を指す。どちらにも abl の run は 0 本。実体は `logs/rsl_rl/knot_tying_v6_ablation/` の 4 run (relA・depB・toP・sw2、2026-10-02 12:33〜12:34)。そこに git 管理外の README.md もある | 一致 |
