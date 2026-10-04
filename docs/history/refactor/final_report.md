# 最終報告: Claude Code の基盤づくりとリファクタリング (Phase 0〜5)

- 期間: 2026-10-03〜2026-10-04。ブランチ `refactor/claude-foundation` (基点 `6f41ef1`、最新 `0252f55`、コミット 21 本、push していない)
- この報告の構成は BRIEF §10 の 11 項目の順。詳細の正本は各項目に書いたファイル (多くは git 管理外の `refactor/` の中)
- 「分かった / 推測 / 未確認」を分けて書く。検証のコマンドと出力は §11

## 目次

0. 要点
1. 新しい文書の構成と、各文書の責任範囲
2. Claude 基盤の一覧
3. 用語集と互換表
4. 定義した作業の型
5. 削除した情報と、history へ移した情報
6. 修正したリンク
7. コードの変更 (Phase 4) と、挙動が変わらないことの確認
8. cold-start テストの前後の比較
9. Claude が最初に読むべき文書の順番
10. まだ人間の判断が必要な点と、今後の改善の候補
11. 実行した検証のコマンドとその結果
12. 承認済みの提案からの逸脱 (BRIEF §0 の【方針】の逸脱の報告)

---

## 0. 要点

- cold-start テスト: 正答率は上がっていない (上限なしは前後とも 100% で天井。上限付きは前後とも 92.9%)。読んだ量は減った (上限なしで読んだ行が約 −50%、トークンが約 −38%)。未公開の 5 問は前後とも 5/5 (§8)。
- 規則のうち機械で判定できるものは、試験 (I1〜I6・T5〜T7)・hook (3 本)・permissions に置いた。CLAUDE.md は 55 行 (§2)。
- 学習・評価の挙動・タスク ID・ログとの互換は変えていない。タスク ID 166 件の一覧が前後で同じ。P4-9 の前後で 166 タスクの環境設定と PPO 設定のダンプがバイト単位で同じ (§7)。
- 再現の項目 (commit・seed・iteration・チェックポイント・パス・単位付きの数値) 2,259 組は、移動・要約の後もすべて追跡ファイルに残っている (§5)。
- 人の判断が要る点: Q5-4 (`refactor/` の後始末)、Q5-5 (見つかった文書の誤りを直すか)。どちらも回答をまだ受け取っていない (§10.1)。
- 記録のない逸脱が 1 つ見つかった: 承認された proposal の `docs/workflow.md` (研究サイクルの段階表) を作っていない (§12)。

---

## 1. 新しい文書の構成と、各文書の責任範囲

```
CLAUDE.md                     # 毎セッション自動で読み込まれる。規則・コマンド・導線 (55 行)
.claude/                      # §2
docs/
  README.md                   # 文書の地図: 読む順番・置き場所の規則・全文書の一覧 (67 行)
  STATUS.md                   # 現在地・版の状態・走っているジョブ・次の作業・保留。上書きで更新 (50 行)
  glossary.md                 # 用語の正式名称・命名規則・互換名の表 (227 行)
  system/evaluation.md        # chain 評価の定義・標準の条件・比べ方 (80 行)
  system/logs.md              # ログのフォルダ名・run 名・チェックポイント・ログのキー (67 行)
  decisions/                  # 版をまたぐ設計の判断。1 判断 1 ファイル (0001〜0015・0020・0021 の 17 本と README)
  experiments/                # 実験記録。1 実験 1 ファイル、書いたら凍結 (13 本と一覧の README)
  history/                    # 経緯。通常は読まない (4 本と v3 の HTML 2 本)
src/.../knot_tying/README.md  # パッケージの仕様 (版の系譜・タスク ID の規則・フォルダ構成)。状態と結果は持たない
src/.../versions/vN/README.md # その版の仕様と理由だけ。結果は experiments へ移してリンク
src/.../common/README.md      # 全版に共通の仕組みの仕様
paper/                        # 論文と論文に関わる文書 (動かしていない)
```

| 情報の種類 (BRIEF §5) | 正本の置き場所 |
|---|---|
| Current specification | 横断的なもの (評価・ログ) は `docs/system/`。パッケージ・版に閉じたものはコードの隣の README・docstring |
| Design rationale | 版をまたぐものは `docs/decisions/`。局所的なものは docstring・版の README |
| Verified result・Reproducibility metadata | `docs/experiments/` (再現のメタデータの表を持つ。欠けた値は「記録なし」) |
| Hypothesis・Open question | `docs/STATUS.md` の「次の作業」「保留」。仮説と明記する |
| Historical context | `docs/history/` |
| 用語・過去の名前 | `docs/glossary.md` (過去の名前は互換表にだけ置く) |

proposal (Phase 2) からの主な変更: `docs/system/` は 2 本だけにした (版の仕様はコードの隣に残す)。`docs/workflow.md` は作っていない (§12)。

---

## 2. Claude 基盤の一覧

根拠の番号は Phase 1 の所見 (`refactor/audit/{A..G}.md`)。設計の詳細は `refactor/proposal.md` §4。

| 種類 | ファイル | 目的 | 根拠 | 効いていることの確かめ |
|---|---|---|---|---|
| CLAUDE.md | `CLAUDE.md` (55 行) | 毎セッション要る、機械で判定できない規則と、入口への導線 | A-1、A-4、G-2、D-6・D-7、D-1・CC-2、BRIEF §2 | Phase 5 の新 HEAD の回答役 6 体 (14 問 4・未公開 2) がすべて「文脈に入っていた」と申告 (入ったのは本体の checkout の CLAUDE.md で、HEAD は同じ。`coldstart_runs/.../phase5_*.md` の 7 行目) |
| 試験 | `tests/test_invariants.py` I1〜I4 | `__init__.py` の import・純粋な層の import・`mdp` の名前・mdp の項の追加のみ | D-16、D-14、F-4 | わざと壊したコピーで 4 件とも落ちる (分かった) |
| 試験 | 同 I5 | 追跡している `*.sh` の `timeout` に `-k` | D-2 | 変更を戻すとちょうど 3 ファイルで落ちる (分かった) |
| 試験 | 同 I6 | 版をまたぐ import は公開名で、import 先の最上位に定義がある | F-12 | 変更前のコードで F §5.2 の 9 か所ちょうどで落ちる (分かった) |
| 試験 | `tests/test_docs.py` T5〜T7 | 版の一覧の整合・全 Markdown の相対リンク・scratchpad のパスの参照 | C-5、E-2、A-3、D-9、CC-10 | 281 passed の中に含まれる |
| 試験 | `tests/test_cli_args.py`、`tests/conftest.py` | `--experiment_name` の挙動 (P4-5)、試験の import の経路 (P4-8) | C-8、F-5 | 変更前のコードで「渡すと上書き」の 1 件だけ落ちる (分かった) |
| hook H1 | `.claude/hooks/session_start.sh` (SessionStart) | git status・STATUS の要点・GPU のジョブを冒頭に出す | D-11、D-15、CC-1、E-7 | 新しい session の冒頭の文脈に出力が入った (分かった。この session でも入った) |
| hook H2・H4 | `.claude/hooks/guard_bash.py` (PreToolUse Bash) | H2: 未コミットの変更があるときの学習・評価の起動に確認を求める。H4: `generate_assets.py` を `--rope` なしで、`overhand_script.py --dump_archive` をパスなしで呼ぶと拒否 (共有の資産を上書きするため)。`run_job.sh`・`bash -c` などの包みを外して照合する | D-6 (29 run 中 21 run)、D-13、R-44 | `refactor/phase3/test_guard.py` 24/24。実際の session で拒否が出た (分かった) |
| hook H3 | `.claude/hooks/ruff_after_edit.py` (PostToolUse Edit\|Write) | 編集した `*.py` に ruff format と check、残った指摘を返す | A-10、E-13、F §0 | `refactor/phase4/test_ruff_hook.py` 6/6。実際の Write で効いた (分かった) |
| permissions | `.claude/settings.json` | kill・pkill・killall は確認、`pkill -f` は拒否。`logs/`・`outputs/`・`.venv/`・USD・`paper/*.pdf` の Edit を拒否。旧実装の `knot_tying` の Read を拒否 | D-1、CC-2、D-13、D-22、BRIEF §2 | kill で確認の画面が出た (ユーザーの回答で分かった) |
| env | 同 `env.PYTHONUNBUFFERED=1` | 出力のバッファリングでログが空に見える失敗を防ぐ | D-19 | `echo $PYTHONUNBUFFERED` → `1` (分かった) |
| rules | `.claude/rules/mdp.md` (16 行)、`versions.md` (10 行) | `common/mdp/`・`versions/` を扱うときだけ読み込まれる規則 | D-14、D-3、D-18、CC-8・CC-9 | `versions/v7/env_cfg.py` を Read した時点で versions.md が文脈に入った (分かった)。mdp.md は同じ仕組みなので効くと推測 |
| skills | `train-eval`・`add-version`・`physics-check`・`experiment-record`・`handoff` | 2 回以上繰り返された手順 | E-1〜E-11、D-4〜D-7、D-11 | description で依頼 25 件を区別できるかを読んで確かめた (`refactor/phase3/skill_description_check.md`、全件 OK・1 件は弱い)。自動で呼ばれるかは未確認 |
| subagent | `.claude/agents/log-analyst.md` | 学習ログ・評価 JSON を読んで表を返す (メインの文脈から隔離) | E-1、E-15、B-14 | 未確認 (proposal の確かめ方「v6 seed 43 の数字を出させて比べる」を行っていない) |
| スクリプト | `src/isaac_twisted_rl/scripts/run_job.sh` | Isaac Sim のジョブの起動: バッファ無効・`timeout -k`・ログ・PID・GPU・commit・差分の記録・受けた signal の記録・`--detach` | D-1、D-2、D-5、D-6、D-10、CC-4 | `refactor/phase4/test_run_job.sh` 19/19。Isaac Sim で 2 iteration の学習を完走 (§7) |
| 設定 | `ruff.toml`、`.gitignore` | ruff の版の固定と規則 (E4/E7/E9/F + I)、Claude Code の作業ファイルを追跡しない | F-1、D-23 | ruff check が全体で通る |

作らなかったもの (proposal §4): 文書の整合の監査役の subagent、長時間ジョブの skill (run_job.sh と CLAUDE.md で足りる)、Stop で全テストを回す hook、既存の版を編集したら警告する hook。

---

## 3. 用語集と互換表

`docs/glossary.md` (227 行)。

- §1 用語の表 (47 語): 課題・ロープ・紐モデル (8 個)、MDP・位相 (段 / 位相クラス)、発散の 3 つの判定、評価・ログ・版。用語ごとに正式名称・コード上の識別子・注意。
- §2 命名規則: 紐モデル (表現 + リンク数 + 摩擦の変種)、版 (番号を使い直さない)、run 名 (`<略>_s<シード>`)、実験記録・設計判断のファイル名。
- §3 互換名の表 (51 行): タスク ID の規則 (166 本、うち `-Play` 83 本、`-RNN` 30 本。全一覧は持たずスクリプトで出す)、ログのフォルダ名、run 名と方策のラベル、使い直された名前 (旧 v7)、改名・移動したファイル (`paper/Report.md` → `Report_Paper.md` ほか、62b251b の 9 件)。

コードの識別子・タスク ID・ログ名は変えていない (Q2-2 (a))。glossary に既知の誤り 2 件 (`:107`・`:206`、§10.2)。

---

## 4. 定義した作業の型

| 型 | 置き場所 | 中身 |
|---|---|---|
| 研究の 1 サイクル | skill `train-eval` (§0 起動の前の確認 → §1 スモークテスト → §2 学習 → §3 chain 評価 → §4 集計 → §5 記録)、`physics-check` (物理を変えたときのゲート)、CLAUDE.md の「実験の約束」 | 仮説・比べる相手・指標・シード数を STATUS に書いてから始める。学習の前に commit。シード 2 本以上を chain 評価で比べる。否定的な結果も記録。結論は「分かった / 推測 / 未確認」 |
| 実験記録のテンプレート | skill `experiment-record` §2 | 目的と仮説・条件の表 (commit と未コミットの差分・タスク ID とプリセット・変えた変数・シード・iteration・num_envs・学習と評価のコマンド・run のフォルダ・評価 JSON・無効にした run)・ゲート・結果・結論 (3 区分)・次の作業。無効にするときの手順 |
| セッション | CLAUDE.md「セッションの始めと終わり」、hook H1、skill `handoff` | 始め: H1 の出力 (git・STATUS の要点・GPU のジョブ) を確かめる。終わり: STATUS を上書き (追記しない)、結果は実験記録、コミットできる状態か確かめる |
| 長時間ジョブ | CLAUDE.md「長時間のジョブと GPU」、`run_job.sh`、permissions | `run_job.sh --detach` で起動。記録は `outputs/jobs/<時刻>_<名前>/`。exit 0 は完了を意味しない (`exit.txt` の `signal:` の行とチェックポイントで確かめる)。自分が起動していないプロセスは kill せず PID を伝える。起動前に `nvidia-smi` |

不足: proposal §5.1 の「段階・終了の条件・残す記録・飛ばしてよい条件」の表を 1 枚にした人間向けの文書 (`docs/workflow.md`) が無い。中身は skill 4 本に分かれて入っている (§12)。

---

## 5. 削除した情報と、history へ移した情報

### 5.1 消えた項目の照合 (BRIEF §4 Phase 3 の【必須】)

- 移動・分割の前に、追跡している Markdown・HTML 29 本から再現の項目を機械的に抜き出した (`refactor/phase3/baseline_items.json`、2,259 組。commit の hash・seed・iteration・`model_*.pt`・`logs/` などのパス・単位付きの数値・%)。
- 今の HEAD で照合: `python3 refactor/phase3/items.py check` → `check: 2259 pairs, missing 0`。**消えた項目は 0 件。**

### 5.2 history へ移したもの (git mv、本文は変えず頭書きとリンクだけ直した)

| 旧 | 新 | 残した理由 |
|---|---|---|
| `knot_tying/next_action.md` | `docs/history/next_action.md` | コードの docstring と凍結した文書が節番号で参照する。冒頭に旧節番号 → 新しい場所の対応表 |
| `knot_tying/rope_study/HANDOFF.md` | `docs/history/rope_study_HANDOFF.md` | rope study の 4 session の経緯。有効な run の一覧と未解決は実験記録と STATUS へ抜き出した |
| `versions/v1/diff_to_TwistedRL.md` | `docs/history/v1_diff_to_TwistedRL.md` | v1 時点の移植元との比較のスナップショット |
| `versions/v3/HTML/*.html` (2 本) | `docs/history/v3_html/` | 2026-09-28 時点の図解 |

### 5.3 experiments へ移したもの (git mv、本文は変えず頭書きを足した)

`Result_report_v3.md` → `2026-09-28_v3_main.md`、`Result_Report_v6.md` → `2026-10-01_v4-v6_overnight.md`、`Report_rope_models.md` → `2026-10-02_rope_models.md`、`paper/Report_Search.md` → `2026-10-03_inference_search.md`。

### 5.4 版の README から外したもの

結果の表と分析 (v2 の v0/v1 比較、v2-R2 の修正の確認、v4 の速度の調査・結果・smoke test、v5 の発散の計測、v6 の台本との照合、v7 の切り分けの表の写し) は、新しい実験記録 9 本へ移してリンクにした。値は 5.1 の照合で残っていることを確かめた。

### 5.5 削除したもの (どこにも移していない)

ファイルの削除は 0 件 (`git diff --diff-filter=D 6f41ef1..HEAD` が空)。本文から消したのは、守られなかった約束の文 (v2 README の「結果はこの節に追記する」を、記録が無い旨の注記に置き換え) と、古い文書の地図 (next_action §0。`docs/README.md` が代わる) などで、再現の項目は含まない。

### 5.6 残した経緯と理由

- 凍結した報告書 (experiments へ移した 4 本) は書き換えていない。要約を重ねると数値と条件が落ちるため (BRIEF の最重要の原則 3)。今と違う点は冒頭の注記で示す。
- `common/README.md` の尺の変更の経緯と、v1・v3 の README の局所の理由は、コードの隣に残した (proposal §1 の「局所の理由はコードの隣」を優先。§12)。

---

## 6. 修正したリンク

- `paper/Report.md` (実体は `Report_Paper.md`、`dbf9a97` で改名) への切れたリンク: Markdown 11 か所 (ルート README・kt/README・Discussion・next_action・diff_to_TwistedRL・v2・v2-R2・v3 の README)、`paper/explainer/` の HTML 2 ファイルの `href="../Report.md"` 2 か所。docstring・コメントの旧名の参照も直した。
- 移動 (62b251b) で切れるリンク: `refactor/phase3/fix_links.py` で history・paper の 7 本を機械的に書き換え、残りは手で直した。
- 今の状態: `tests/test_docs.py` (T6) が全 Markdown の相対リンクの実在を検査し、通る (§11)。`paper/Report.md` の文字列が残るのは互換表・history・凍結した文書の中の旧名の言及だけ。
- 残っている注意: 追跡ファイル 15 本・18 行が、git 管理外の `refactor/` のファイルを根拠として名指ししている (`git grep -n "refactor/"`。ブランチ名を除く)。どれもリンクではなくコード表記の中のパスなので、T6 には掛からない (`_strip_code` が外す: `tests/test_docs.py:61-63`)。他の checkout では辿れない (§10.1 の Q5-4)。

---

## 7. コードの変更 (Phase 4) と、挙動が変わらないことの確認

| # | 変更 | コミット | 確かめたこと |
|---|---|---|---|
| P4-2 | ruff の設定を `ruff.toml` に (0.16.10 に固定、E4/E7/E9/F + I、起動スクリプト 19 本は E402・I を無視) | e937577 | コードの差分なし。指摘 253 → 16 件 |
| P4-3 | 全体の整形 (改行だけ 6 本、残り 27 本) | 453b33c・cbef9be | 33 本とも AST が完全に一致 (`phase4/ast_equal.py`)。pytest 275 passed。タスクの一覧 166 件が同じ |
| H3 | 編集後の ruff の hook | e437754 | 試験 6/6 |
| P4-4 | 安全な自動修正 (UP033・RUF022・PLR0402・I001) 15 本 | 34ce7a5 | `phase4/ast_equiv.py` で 15 本とも同じ (比較器が本当の違いを検出することも 3 例で確認)。タスクの一覧が同じ |
| P4-1 | `run_job.sh` | cbb6029・325ec11・2b16cad | 試験 19/19。Isaac Sim のスモークテスト 2 回 (1 回目は session の終わりの SIGTERM で学習前に止まり exit 0 → `--detach` と signal の記録を足した。2 回目は v6 の 2 iteration を完走し `model_0.pt`・`model_1.pt`・`params/`・`git/` ができた: `outputs/jobs/2026-10-04_18-24-56_smoke_runjob_v6`) |
| P4-6 | rope study のシェルの `timeout` に `-k`、試験 I5 | d605441 | シェルの差分は `-k` だけ |
| P4-5 | `--experiment_name` を効かせる (help の文のとおり) | 9ca24b0 | 渡さないときのログ名は版の名前のまま (`tests/test_cli_args.py`)。**挙動の変更** (引数を渡したときだけ。承認 Q3-2) |
| P4-8 | 試験の import の経路を `tests/conftest.py` に | 6f1fa9a | `pytest --collect-only` の一覧が前後で同じ 280 件 |
| P4-9 | 版をまたぐ private な名前の import を公開名に | 0252f55 | 試験 I6。166 タスクの `parse_env_cfg(...).to_dict()` と PPO 設定のダンプを Isaac Sim で前後に取り、`cmp` でバイト単位に同じ (各 3,505,448 bytes)。v5 の名前空間から `_play_post_init` が消えた (参照なし) |

行わなかったもの: P4-7 (`make_eval_jobs.py` のチェックポイントの警告) と P4-10 (`bench_physics.py:378` の発散の数え方の疑い) は rope study を再開するときに回した。P4-8 の後半 (`test_knot_detect` の紐の生成関数の共通化) は不要と判断した。大きなファイルの分割・重複の削減 (F-8〜F-11) は Isaac Sim で確かめる手段が要るので見送った。

---

## 8. cold-start テストの前後の比較

正本: `refactor/phase5/coldstart.md` (基準値は `refactor/audit/G.md`)。採点は正答を持つ別の subagent が行い、メインが各 3 か所以上を確かめた。

| 試行 | 正答率 (前 → 後) | 読んだ行 (2 本の平均) | トークン (平均) | tool_uses (平均) |
|---|---|---|---|---|
| 14 問・上限なし | 100% → 100% | 約 5,610 → 約 2,785 (約 −50%) | 約 302,600 → 約 187,100 (約 −38%) | 60.5 → 41 |
| 14 問・上限付き (15 回) | 92.9% → 92.9% | 約 1,700 → 約 1,200 (約 −30%) | 約 131,200 → 約 114,000 (約 −13%) | 18.5 → 18 |
| 未公開 5 問 | 100% → 100% | 約 1,880 → 約 1,520 (約 −19%) | 同程度 (新は 1 本だけ取得) | 31 → 29 (新は 1 本) |

- 後の読んだ行には CLAUDE.md の 55 行を含む。読んだファイル数: 上限なしで 42・41 → 28・31。
- 達成の基準: 「正答率が上がる」は示せない (天井と同率)。「読んだ量が減る」は達成。「未公開でも同等以上」は達成。
- 間違えた問の分析: 上限付きの後の 2 本は、どちらも問 5 で部分正答。原因は `docs/glossary.md:107` の古い記述 (「`versions/__init__.py` の docstring の一覧は v6 で止まっている」。Phase 3 で v7 を足したのに glossary を直し忘れた)。**リファクタリングの側の誤り。** 基準値の部分正答 (問 8・13、答えの無い問の扱い) は後では起きなかった。
- 未公開の問で差が小さかった理由は、H1〜H4 がコードに答えのある問で、文書の導線の効きが小さいためとみられる (推測)。
- 測定の条件の違い: 旧 HEAD の未公開の回答役にも、今のブランチ名と最新のコミットのメッセージが見えた (git status のスナップショット)。新 r1 の usage は 429 の中断で取れなかった。上限なし r1 の grep 1 回が auto mode の判定器に拒否された (プロジェクトの設定ではない)。

---

## 9. Claude が最初に読むべき文書の順番

1. `CLAUDE.md` (自動で読み込まれる) と、SessionStart hook の出力 (git・STATUS の要点・GPU のジョブ)
2. `docs/STATUS.md`: 現在地と次の作業
3. `docs/README.md`: 作業に応じて読む文書を選ぶ
4. 作業ごと: 手順は skill (`train-eval`・`add-version`・`physics-check`・`experiment-record`・`handoff`)。版の比較は `docs/system/evaluation.md`、run の場所は `docs/system/logs.md`、用語は `docs/glossary.md`、理由は `docs/decisions/`、過去の結果は `docs/experiments/README.md`
5. `docs/history/` は通常は読まない

---

## 10. まだ人間の判断が必要な点と、今後の改善の候補

### 10.1 判断していただきたい点

**Q5-4 作業ディレクトリ `refactor/` の後始末** (BRIEF §4「Phase 5 で私が決める」)
- (a) 推奨: 再現に要る道具と記録 (`phase3/`・`phase4/` のスクリプト、`audit/`、`phase5/coldstart.md`、`proposal.md`、`claude-code-spec.md`、この報告、BRIEF、PROGRESS) を `docs/history/refactor/` へ移して追跡し、追跡ファイルの 18 行の参照をそこへ向け直す。理由: 実験記録の「記録なし」の根拠 (B-101・B-117 など) と、Phase 4 の挙動の確認の道具が他の checkout からも辿れる。大きな JSON (`env_cfgs_*.json` 各約 3.5 MB、`baseline_items.json`) は追跡に入れるかを別に決める (入れないなら 18 行の参照に影響しない)。
- (b) `refactor/` を削除し、18 行の参照を消す (根拠の一部が辿れなくなる)。
- (c) 今のまま git 管理外に残す (他の checkout では 18 行が切れたまま)。

**Q5-5 見つかった文書の誤りを直すか**
- (a) 推奨: 直す (文書と docstring だけ。挙動は変えない。T6 などの試験を回してからコミット)。測定は終わっているので、質問への最適化にはならない。対象は 10.2 の一覧。
- (b) 直さずに、この報告の改善の候補として残す。

**その他**
- 打ち切った 2 研究 (rope study・推論時探索) を再開するか (STATUS の「保留」)。再開するなら P4-7・P4-10 と rope study の `MODELS` の一本化 (C-9) を最初に行う。
- `refactor/claude-foundation` を main に入れるか (push・マージは行っていない)。
- `docs/workflow.md` を作るか (§12)。

### 10.2 見つかった文書の誤り (Q5-5 の対象。まだ直していない)

| # | 場所 | 誤り | 出典 |
|---|---|---|---|
| 1 | `docs/glossary.md:107` | 「`versions/__init__.py` の一覧は v6 で止まっている」は古い (`:14` に v7) | 回答役 limited r1・r2、full_r2。メインが確認 |
| 2 | `docs/glossary.md:206` | 根拠 `v7/README.md:77-81` は存在しない行 (ファイルは 72 行) | full_r2。メインが確認 |
| 3 | `docs/decisions/0012-version-is-config.md:55` | I4 の検査を `tests/test_invariants.py:193` とするが、I4 は `:245` | full_r2。メインが確認 |
| 4 | `docs/system/evaluation.md:57` | シード差の出典 `overnight.md:264`・`:277` は誤り (記述は `:283-284`) | full_r2。メインが確認 |
| 5 | `docs/STATUS.md:9` | 「いまはリファクタリングの Phase 3」は古い (Phase 5) | この session でメインが発見 |
| 6 | `CLAUDE.md:31` と `docs/system/evaluation.md:40`・skill `train-eval:32` | 「学習と同じ `--device`」と「`--device` と `agent.device` を揃える」の書き方の差。記録の実例は学習 cuda:1 → 評価 cuda:0 | full_r2 (メインは未確認) |
| 7 | `docs/system/logs.md:41`、`docs/experiments/2026-10-01_v4-v7_readme-notes.md:25`、`docs/system/evaluation.md:41` | 「`model_100.pt` まである」が最後のチェックポイントと読める (v5・v6 s43 は `model_110.pt`) | full_r2・点検役 (メインは未確認) |
| 8 | `docs/system/logs.md:17` | 行番号の参照が古い | 点検役 (メインは未確認) |
| 9 | `kt/common/__init__.py:21` | 無くなった README の節を指す | 点検役 (メインは未確認) |
| 10 | `versions/v4/README.md:107` | 保存間隔「600 手 (= 50 iteration)」(実装と同じ README の `:166` は 200 手) | full_r2・decisions 0020 に注記済み (メインは未確認) |
| 11 | `versions/v5/README.md` | 「`chain_est` が主指標」(decisions 0020 と README に注記済み) | Phase 3 |
| 12 | `versions/v3/README.md:166` | proposal §2.3 で消すとした古い「未実装」(A-8) が残る | この session でメインが発見 (今も古いかは未確認) |
| 13 | `phases.py:12`・`:29`、`rewards.py:7`、`v0/env_cfg.py:77,79` の docstring | `docs/history/next_action.md` の本文に無い文言を引く | Phase 3 |
| 14 | `rope_specs.py:62-87`、`env_cfg_base.py:129` の docstring | 「ロープ↔床の実効摩擦は 2.5」(値はどちらも 2.0)、「8192 環境では p-data がボトルネック」(実測では PhysX が大半)。物理の解釈が要る | decisions の担当の報告 (メインは未確認) |
| 15 | `docs/experiments/2026-10-02_rope_models.md:820-821` | テストと lint のコマンドが今と違う (凍結した記録の当時の手順。直さない) | full_r2 |

### 10.3 今後の改善の候補 (今回は根拠が足りず見送ったもの)

- 正答率の天井: 14 問では差が出ない。次に測るなら、文書の導線の効きを測れる問 (コードから答えにくい手順・現在地の問) を未公開で足す。
- `log-analyst` の効果の確認 (既知の run の数字を出させて記録と比べる)。使われる頻度が低ければ消す候補。
- skill が自動で呼ばれるかの確認 (今回は description を読んで判定しただけ)。
- 大きなファイルの分割・重複の削減 (F-8〜F-11): Isaac Sim で設定のダンプを比べる道具 (`phase4/dump_env_cfgs.py`) ができたので、次は前後の比較で確かめられる。
- 文書の古さの検出: 文書の中の `file:line` の参照が実在する行を指すかの試験 (10.2 の 2・3・4・8 はこれで見つかる)。誤検知 (凍結した記録の当時の行番号) の扱いを決めてから。

---

## 11. 実行した検証のコマンドとその結果

2026-10-04、HEAD `0252f55`、作業ツリーはクリーン (`git status --short` が空)、GPU のジョブなし。

| コマンド | 結果 |
|---|---|
| `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run python -m pytest tests/ -q` | `281 passed, 1 warning in 50.12s` (warning は torch.onnx の DeprecationWarning。T6 のリンク検査を含む) |
| `uvx ruff@0.16.10 check .` | `All checks passed!` |
| `uvx ruff@0.16.10 format --check .` | `130 files already formatted` |
| `python3 refactor/phase3/items.py check` | `check: 2259 pairs, missing 0` |
| `uv run python refactor/phase4/dump_tasks.py \| diff - refactor/phase4/tasks_before.json` | 差分なし (166 件。基準は 0a54bac で取得) |
| `bash refactor/phase4/test_run_job.sh` | `19 passed, 0 failed` |
| `python3 refactor/phase3/test_guard.py` | `24/24 as expected` |

Phase 3〜5 の途中で行った検証 (コマンドと出力の詳細は `refactor/PROGRESS.md` の各表):

| 検証 | 結果 |
|---|---|
| 試験 I1〜I4 を壊したコピーで回す | 4 件とも落ちた |
| `phase4/ast_equal.py` (P4-3 の 33 本)、`ast_equiv.py` (P4-4 の 15 本) | すべて同じ |
| `pytest --collect-only -q` の前後 (P4-8) | 同じ 280 件 |
| `phase4/test_ruff_hook.py` | 6/6 |
| Isaac Sim: `run_job.sh` のスモークテスト 1 回目 (`outputs/jobs/2026-10-04_17-59-16_smoke_runjob_v6`) | **失敗**: 11 s で exit 0、学習の前に止まった (session の終わりの SIGTERM)。対策を入れて 2 回目へ |
| Isaac Sim: 同 2 回目 `--detach` (`..._18-24-56_smoke_runjob_v6`) | exit 0、`signal:` の行なし、2 iteration 完走、チェックポイント 2 つ。終わりの `TypeError: post_quit()` は既知 |
| Isaac Sim: P4-9 の設定のダンプ前後 (`..._19-15-43_p49_cfg_before2`・`..._19-15-55_p49_cfg_after`) | バイト単位で同じ。1 回目の前 (`..._19-14-55_p49_cfg_before`) は読んだコードを出力から判別できず、印を足して再実行した (自分の計測の誤り) |
| SessionStart・rules の `paths`・kill の確認・guard の拒否・`env` | 新しい session で効いた (§2) |

---

## 12. 承認済みの提案からの逸脱

| 逸脱 | 理由 | 記録 |
|---|---|---|
| **`docs/workflow.md` (研究サイクルの段階表) を作っていない** (proposal §2.1・§5、Q2 の (a) で承認) | 作らない判断の記録が無い。作業の型は skill 4 本と CLAUDE.md に入っているが、人間が全体を見渡す 1 枚が無い。原因は未確認 (Phase 3 の計画の表 1〜14 に workflow.md の行が無く、計画に落とすときに抜けたとみられる: 推測) | この報告で初めて記録 |
| ruff の設定を `pyproject.toml` ではなく `ruff.toml` に置いた | `pyproject.toml` を変えると `uv run` が editable install を作り直して `.venv` を書き換える (Q2-8 (a) に反する) | PROGRESS「計画の小さな変更」 |
| CLAUDE.md は約 90 行の案に対して 55 行 | 各行が「消すと間違えるか」に通るものだけにした | — |
| v1・v3 の README の局所の理由を decisions へ移さず残した | proposal §1 の「局所の理由はコードの隣」と §2.3 の表が食い違っていたため、§1 を優先 | PROGRESS (2026-10-03) |
| H2 と H4 を 1 本の `guard_bash.py` にし、H4 に `overhand_script.py --dump_archive` (パスなし) を足した | 同じ照合の仕組みを使う。旧 kt/README のデモのコマンドが共有の初期アーカイブを上書きする形だったため | PROGRESS (Phase 3 の 13) |
| 試験 T8 を I5 として足し、proposal に無い I6 を足した | P4-9 の不変条件を固定するため | PROGRESS (Phase 4 の 6・9) |
| Phase 1 で F を C と並列に走らせた (BRIEF の「C を先に」) | F は用語の揺れの影響を受けにくい | PROGRESS (Phase 1) |
| 上限付きの cold-start の上限を 15 回とし、抜け道を塞ぐ条件を足した | ユーザーの回答「a」に値の指定が無く、選択肢の文の例を使った | PROGRESS (G-5) |
| Q4-3 の「前後 2 回」に対して前を 2 回走らせた (計 3 回、各 6 s) | 1 回目の出力では読んだコードを判別できなかった | PROGRESS (Phase 4) |
| decisions の番号が 0016〜0019 を飛ばしている | 担当を分けたときに番号の範囲を 0001〜0019 と 0020〜 に割り当てた名残 | — |
