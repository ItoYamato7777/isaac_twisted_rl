# Phase 2 設計の提案

- 対象: HEAD `6f41ef1` (ブランチ `refactor/claude-foundation`)。Phase 1 の調査 (`refactor/audit/A〜G`) と `refactor/claude-code-spec.md` を根拠にする。
- この文書は提案で、まだ何も変えていない。承認された項目だけを Phase 3・4 で実施する。
- 根拠の番号は監査のファイルの所見の番号 (A-1、B-12、B-104、C-2、D-6、CC-8、E-1、F-4、N-1 など)。G は cold-start テスト。
- 各提案の「確信」は 高 / 中 / 低。「変わる条件」は、何が分かれば判断を変えるか。
- 用語は C.md の推奨名に揃える (段 = v4 以降の `BfsPotential.stage`、位相クラス = v1〜v3 の報酬ラダーの段、ロープ `Knot` = 28 リンクのプリセット、固結び = overhand knot、紐モデル = rope study の `RopeModel`、chain 評価、1 手)。

## 目次

0. 要点
1. 情報の置き場所の表
2. 文書の構成案
3. 用語の正式名称と命名規則
4. Claude 基盤の設計
5. 作業の型
6. コード構造の変更案 (Phase 4 の候補)
7. 判断していただきたい点
8. 自己点検 (BRIEF §11)

---

## 0. 要点

1. **現在地を 1 か所にする。** 今は 8 か所に分かれて食い違う (A-1、B-103)。`docs/STATUS.md` を上書きで保ち、他の文書は数字と状態を持たずにリンクする。
2. **結果を 1 か所にする。** 同じ数字が最大 12 か所にあり、細部が食い違う (A-7、B-105)。結果は `docs/experiments/` だけに書く。版の README は仕様と理由だけにする (B-107)。
3. **経緯を通常の読み物から外す。** `next_action.md` (583 行、約 510 行が経緯) と `HANDOFF.md` (439 行) を `docs/history/` へ移す。ただし先に、今も有効な理由をコードの参照ごと `docs/decisions/` へ抜き出す (B-104、A-6)。
4. **規則は強制できる所に置く。** 4 つの不変条件はテストにする (F-4、D-16)。kill と保護されたパスは permissions (D-1、D-13)。未コミットのコードでの学習は hook で確認を求める (D-6: 29 run 中 21 run)。CLAUDE.md には、機械で判定できない規則と入口への導線だけを置く。
5. **skill は 5 つ** (学習と chain 評価・版の追加・物理の確認・実験記録・引き継ぎ)。どれも 2 回以上繰り返された手順を根拠にする (E-1〜E-7)。
6. **識別子は変えない (推奨)。** C.md の衝突 23 件のうち、文書だけで解決できないのは C-2・C-8・C-9 などの一部。どれも互換の費用が利益より大きいか、打ち切った研究のもの。
7. **判断していただきたい点は §7 の 10 個** (言語・改名の範囲・rope study の文書・ruff・next_action の分け方・結果の報告書の置き場所・Discussion の扱い・venv・D-3 の疑い・cold-start の天井)。

---

## 1. 情報の置き場所の表

BRIEF §5 の 8 分類を、このリポジトリに合わせて具体化した。**1 つの情報の正本は 1 か所**にし、他の場所はリンクだけを持つ。

| 分類 | 正本の置き場所 | このリポジトリでの具体例 | 根拠 |
|---|---|---|---|
| 現在の仕様 (局所) | コードの隣。モジュールの docstring か、パッケージの README | 版ごとの MDP は `versions/vN/README.md`。共通の仕組みとロープのプリセットは `kt/common/README.md`。スクリプトの使い方は `scripts/rsl_rl/README.md` (今も良い形: A §2) | A-11、B-107、B-114 |
| 現在の仕様 (横断) | `docs/system/` | chain 評価の定義 (今は 7 か所: B-111)、ログのフォルダ名・run 名・チェックポイント・ログのキーの意味 (C-2、C-8、B-113、D-4) | B-111、B-113、C-6、C-12 |
| 設計の理由 (局所) | docstring (「なぜそうしたか」。経緯は書かない) | 観測の式・行動の座標系の理由など、1 つのモジュールで閉じるもの | A-13、BRIEF §9 |
| 設計の理由 (横断) | `docs/decisions/NNNN-<題>.md` (1 判断 1 ファイル。理由・却下した案・今も有効か) | 「版 = 設定」、行動空間 (ロープ相対 + Δlink)、240 Hz、段の定式化と到達状態アーカイブ、ロープ `Knot` の選定など (§2.4 の表) | B-12、B-104、A-5 |
| 検証済みの結果 | `docs/experiments/YYYY-MM-DD_<題>.md` (凍結。条件・seed・iteration・評価の方法・commit を併記) | v3 本番、v4〜v6 の overnight、v6 の切り分け、ゲート 1、紐モデルの比較、推論時探索 | A-7、B-102、B-105 |
| 仮説 | 「仮説」と明記して、検証の計画と同じ場所に置く。進行中の計画は `docs/STATUS.md` の「次の作業」、終わった実験の解釈は実験記録の「結論」の「推測」欄 | v2-R2 の「本物の C2 が稀で詰まる」は今は結果の欄にある。仮説として書き直す (A-2、B-110) | A-2、B-115、D-7 |
| 未解決の課題 | `docs/STATUS.md` の「次の作業」と「保留」 | 打ち切り中の rope study と推論時探索の課題は「保留」の欄 (B-16) | B-106、B-16 |
| 再現のメタデータ | 実験記録の中 (表)。`logs/` は管理外なので、run のフォルダ名・commit・未コミットの差分の有無・コマンド・Hydra の上書きを必ず書く | rope study の有効な run のフォルダ名は今 HANDOFF の 1 行だけにある (B-2)。切り分けの上書きは `logs/` の README にある (A-9) | B-1、B-2、A-9、D-6 |
| 経緯 | `docs/history/` (通常は読まない。本文は書き換えない) | `next_action.md` の経緯の節、`HANDOFF.md`、`diff_to_TwistedRL.md`、v3 の HTML | A-5、B-4、B-118 |
| 廃止された情報 | 削除の候補 (承認が要る)。再現に要るものは `docs/history/` | 「未実装」と書かれた実装済みの機能 (A-8)、存在しないファイルへの参照 (B-112) | A-8、B-106、B-112 |

付け足す約束 (3 つ):
- **凍結した文書** (実験記録・history) は本文を書き換えない。足すのは冒頭の頭書き (状態・読むべき状況・最新の結論の場所) と、リンクの修正だけにする。理由: 要約や書き直しで数値や条件が抜け落ちる (BRIEF §11)。
- **全文書の冒頭に頭書きを付ける** (状態 = 現行 / 凍結、読むべき状況、最終更新)。宣言があるのは今 2 本だけ (A-12)。
- **段落ごとの「事実 / 解釈 / 仮説」のラベル**は `paper/Discussion.md:21-29` の慣行を採る (B-22)。実験記録のテンプレート (§5.2) の「結論」欄で使う。

確信: 高 (分類と置き場所)。変わる条件: Phase 3 で抜き出してみて、1 判断 1 ファイルにすると 20 本を超えるほど細かくなる場合は、話題ごとにまとめる (例: `decisions/` を「行動」「報酬」「物理」の 3 本に)。

---

## 2. 文書の構成案

### 2.1 構成

BRIEF の叩き台を、このリポジトリの実態に合わせて削った。**新しく作るファイルは、置く中身が今あるものだけ**にした。

```
CLAUDE.md                     # 毎セッション読む。約 80〜120 行 (§4.2)
.claude/                      # §4
docs/
  README.md                   # 文書の地図。全文書を「状態・読むべき状況」付きで載せる (A-4、A-12)
  STATUS.md                   # 現在地・版の状態表・走っているジョブ・次の作業・保留。上書きで更新
  glossary.md                 # 用語と互換名の表 (C.md の推奨名、§3)
  workflow.md                 # 研究サイクルと記録の型 (§5)。手順のコマンドは skills に置き、ここには書かない
  system/
    evaluation.md             # chain 評価の定義・比べ方・Play タスクの意味 (B-111、C-6、C-12、D-7)
    logs.md                   # ログのフォルダ名・run 名・チェックポイント・ログのキーの意味 (C-2、C-8、B-113、D-4、E-17)
  decisions/                  # 横断的な設計の理由。1 判断 1 ファイル (§2.4)
  experiments/
    README.md                 # 実験の一覧 (日付・題・状態・要点・run のフォルダ・リンク)
    YYYY-MM-DD_<題>.md        # 実験記録 (凍結)
  history/                    # 経緯。通常は読まない ("archive" は状態アーカイブと衝突するので使わない: C-11)
src/.../knot_tying/README.md  # パッケージの仕様 (版の系譜・タスク ID の規則・フォルダ構成)。状態と結果は持たない
src/.../versions/vN/README.md # その版の仕様と理由だけ。結果は experiments へ
src/.../common/README.md      # 共通の仕組みの仕様 (ロープのプリセットを含む)
paper/                        # 論文と、論文に関わる文書 (PDF・図解 HTML・Report_Paper・Discussion)
```

叩き台から変えた点:

| 変えた点 | 理由 | 別案と、採らなかった理由 |
|---|---|---|
| `docs/system/` は 2 本だけ | 横断的な仕様で、置き場が無いのは評価とログの 2 つだけ。版・共通の仕組み・スクリプトの仕様は、コードの隣にすでに README がある | 版の仕様を `docs/system/versions/vN.md` へ移す (A §2 の案)。採らない: 版はパッケージごとに閉じていて、README は `env_cfg.py` の隣にある方が探しやすい。移すと版の README への被リンク (v2・v3 は各 7 本: A §6.3) が全部切れる |
| `docs/system/layout.md` は作らない | フォルダ構成の木は `kt/README.md` の 1 か所に直して残す (A-11 の古さは直す) | 木を docs へ移す (A-11 の案)。採らない: 木はパッケージの説明の一部で、2 か所目を作る理由がない |
| `workflow.md` は残すが、コマンドを書かない | 研究サイクルの段階・終了の条件・残す記録は、人間も読む。コマンドは skill に置いて、二重にしない | workflow.md を作らず skill だけにする。採らない: skill は Claude が呼んだときしか読まれず、人間が全体を見渡せない |
| `paper/` を残す | `Discussion.md` は `explainer/*.html` と PDF を相対リンクで指す (A §3)。動かすと切れる (N-10) | `paper/` を `docs/papers/` へ移す。採らない: 探しやすさが変わらず、リンクが切れるだけ |

確信: 中。変わる条件: Phase 5 の cold-start テストで、回答役が版の README を見つけられずに迷う (読む量が減らない) なら、`docs/README.md` の導線を足す。置き場所は変えない。

### 2.2 モジュール固有の仕様の置き場所

**提案: コードの隣 (README か docstring) に置く。`docs/system/` には、複数のパッケージをまたぐものだけを置く。**

- 理由 1: サブディレクトリの README や docstring は、Claude がそのコードを開いたときに一緒に目に入る。`docs/system/` に置くと、仕様を変えたときに docs 側の更新を忘れやすい (D-9: 整理の commit を 2 回出しても 11 件残った)。
- 理由 2: 今の版の README・`common/README.md`・`scripts/rsl_rl/README.md` は、すでにこの形で読まれている (cold-start の基準値の回答役は 2 本とも、版の README を 5 本 (v2-R2・v3・v4・v6・v7。v3 だけ一部) 読んで答えた。`coldstart_runs/.../baseline_r{1,2}.md` の §3 の読んだファイルの一覧)。
- 別案: すべての仕様を `docs/system/` に集める。採らない理由: 移動でリンクと人間の記憶が古くなる (BRIEF §4 Phase 2)。探しやすさは `docs/README.md` の地図で補える。
- 確信: 中。変わる条件: 同じ仕様を 2 つ以上のパッケージの README が説明している例が Phase 3 で多く見つかれば、その仕様だけを `docs/system/` へ上げる。

### 2.3 既存の文書の行き先

追跡されている Markdown 23 本と、Markdown 以外の文書。行数は HEAD `6f41ef1`。動かすものは `git mv` で履歴を保つ。

| 文書 | 行数 | 行き先 | 中身の振り分け | 根拠 |
|---|---|---|---|---|
| `README.md` (英語) | 143 | 残す | 概要・インストール・クイックスタートだけにする。版の結果表 `:56-66` を消して `docs/STATUS.md` へのリンクに。文書の表 `:110-118` を `docs/README.md` へのリンク 1 本に | A-1、A-3、A-4 |
| `kt/README.md` | 247 | 残す | 版の系譜 (各版が何を変えたか、1 行ずつ)・タスク ID の規則・フォルダ構成 (最新に直す)。版の状態と結果 `:29-82` → STATUS と experiments。chain 評価 `:83-104` → `docs/system/evaluation.md`。「版 = 設定」`:154-173` → decisions。追加の規約 `:175-188` → skill `add-version` と rules。Kit の注意 `:214-227` と開発の約束 `:229-247` → テスト・CLAUDE.md・rules (§4) | A-1、A-10、A-11、B-108 |
| `kt/common/README.md` | 196 | 残す | 仕様として最新に直す (表が v3 で止まる: A §2、B-114)。尺の変遷 `:68-77` → decisions。240 Hz の実測 `:41-45` は decisions の 240 Hz の判断へ | A §2、B-114 |
| `kt/next_action.md` | 583 | `docs/history/next_action.md` | 先に §2.4 の表のとおり抜き出す。節番号は変えずに移す | A-5、A-6、B-104 |
| `kt/versions/v0/README.md` | 80 | 残す | 仕様のまま。基準コミット `1f5693f` は存在しない (B-101) ことを注記する | B-101 |
| `kt/versions/v1/README.md` | 115 | 残す | 「段」→「位相クラス」と書き直す (C-1)。ポテンシャル整形の理由 `:52-61` → decisions | C-1 |
| `kt/versions/v1/diff_to_TwistedRL.md` | 287 | `docs/history/` | 全体が経緯 (本人が v1 時点と書く) | B-118 |
| `kt/versions/v2/README.md` | 276 | 残す | v1 の失敗の分析 `:17-57` → `experiments/2026-09-15_v0-v1.md`。rsl-rl 3.0.1 の注意 `:203-222` → `docs/system/` ではなく CLAUDE.md か rules (版に依らない知識。§4)。未履行の「結果はここに追記」`:201` と採られなかった v3 構想 `:237-244` は廃止 | A §2、B-109 |
| `kt/versions/v2_r2/README.md` | 191 | 残す | 修正の確認 `:130-179` → `experiments/2026-09-27_v2r2_fix.md`。「次の課題」`:181-187` は仮説と明記して STATUS へ | A-2、B-110 |
| `kt/versions/v3/README.md` | 168 | 残す | 古い「未実施」「未実装」`:164-166` を消す (A-8)。「実装中に分かったこと」`:94-107` → decisions | A-8 |
| `kt/versions/v3/Result_report_v3.md` | 198 | `docs/experiments/2026-09-28_v3_main.md` | 本文は変えない。頭書きに「§7 の次の作業は v4・v5 で消化済み」 | A §2、§1 の約束 |
| `kt/versions/v4/README.md` | 351 | 残す (約 150 行に) | 速度の調査 `:148-288` → `experiments/2026-09-30_v4_speed.md` (判断は decisions)。結果 `:290-310` と smoke test `:312-336` → experiments。課題 `:338-351` → STATUS (済んだものは消す) | B-107、A-8 |
| `kt/versions/v5/README.md` | 204 | 残す | 発散の計測 → `experiments/2026-10-01_v5_divergence.md`。教訓「計測では効いたが学習では逆効果」`:189` → workflow.md と decisions | A §2 |
| `kt/versions/v6/README.md` | 191 | 残す | 台本との照合 `:76-125` と結果の表 → experiments | B-105 |
| `kt/versions/v6/Result_Report_v6.md` | 311 | `docs/experiments/2026-10-01_v4-v6_overnight.md` | 本文は変えない (分割しない)。頭書きに「最新の結論は §5 以降の切り分け (`:215-311`)。§4.2 の仮説は `:149-152` で否定された」 | A-5、§1 の約束 |
| `kt/versions/v7/README.md` | 81 | 残す | 切り分けの表 `:19-26` (写し) を消して experiments へのリンクに。旧 v7 の注記 `:77-81` は残す (再現に要る) | B-105、B-6 |
| `kt/rope_study/README.md` | 112 | 残す | 道具の使い方として残し、誤ったコマンド (`model_100.pt`・`--iters 50,100`) を直す (B-3) | B-3、D-4 |
| `kt/rope_study/HANDOFF.md` | 439 | `docs/history/` | 先に、有効な run のフォルダ名の全一覧 (B-2) を実験記録へ、未解決 (`:16-25`) を STATUS の「保留」へ、踏んだ注意 (`:27-32`、`:148-156`) を workflow.md と CLAUDE.md へ移す | B-2、B-4、D-11 |
| `kt/rope_study/Report_rope_models.md` | 880 | `docs/experiments/2026-10-03_rope_models.md` | 本文は変えない。頭書きに「打ち切り中」「§7.1.2 は無効な経過の記録 (B-17)」「発散の数値は D-3 の疑いあり (未確認)」 | B-17、D-3、§7 の Q2-3 |
| `scripts/rsl_rl/README.md` | 190 | 残す | 段の説明 `:92` (「交差数に当たる」) を直す (C-1) | A §2、C-1 |
| `paper/Report_Paper.md` | 466 | 残す (`paper/`) | 頭書きに「2026-09-26 時点の論文の比較。§9 は 2026-10-01 時点で古い」。11 本の切れたリンクの行き先になる (A-3) | A-3、B-11 |
| `paper/Report_Search.md` | 304 | `docs/experiments/2026-10-03_inference_search.md` | 本文は変えない。頭書きに「打ち切り中」「V8 の定義はない (A-15)」。§6 → STATUS の「保留」 | B-16、A-15 |
| `paper/Discussion.md` | 1078 | 残す (`paper/`) | 頭書きに「2026-10-01 時点の研究方針の提案 (仮説の文書)。§1.6 は v4 時点」。生きている候補 (E0〜E7・S1) は STATUS の「候補」から指す (B-15) | B-14、B-18、§7 の Q2-7 |
| `paper/*.pdf`・`paper/explainer/*.html` | — | 残す | 動かさない (N-10)。HTML の切れたリンク 4 本を直す。`docs/README.md` から参考資料として指す | A §3、N-10 |
| `kt/versions/v3/HTML/*.html` | — | `docs/history/` | 2026-09-28 時点のスナップショット (A-14) | A-14 |

新しく作る実験記録 (今は版の README の中にある結果を抜き出すもの): v0〜v1 の比較、v2-R2 の修正の確認、v4 の速度の調査、v5 の発散の計測、v6 の台本との照合、ゲート 1 (2026-09-27)。実験記録は Phase 1 の B_2 §3.1 の 17 件の表をもとに、再現のメタデータの欄を埋める。欠けている値は「記録なし」と書く (B-102)。

### 2.4 `next_action.md` の分け方

コードの docstring が節番号で 48 行・16 ファイルから参照している (A-6)。そのため、次の順で行う (B-104)。

1. 今も有効な理由を、下の表の行き先へ抜き出す。
2. コードの参照を新しい行き先に書き換える (コメントと docstring だけ。Phase 3 で許される範囲)。
3. 残りを節番号を変えずに `docs/history/next_action.md` へ `git mv` する。冒頭に「旧節番号 → 新しい場所」の対応表を置く。
4. スクリプトで、旧い参照 (`next_action.md §`) が `docs/history/` の対応表にあるものだけになったことを確かめる。

| 節 | 行 | 中身 | 行き先 | 節番号つきの参照の数 (A §6.4。md を含む) |
|---|---|---|---|---|
| 現在地と次の作業 | 16-55 | 2026-10-02 の現在地 | STATUS (新しい内容で書き直す) | — |
| §0 文書の地図 | 56-73 | v3 で止まった地図 | 廃止 (`docs/README.md` が代わる) | — |
| §1〜§2 | 74-173 | topology・rope_state の作成の経緯と実測 | history。§2 の body の順序の実測は `rope_state.py` の docstring にあるか確かめ、無ければ decisions | — |
| §3 (a)〜(c) | 174-219 | 位相抽出の検証・たるみ・物理パラメータで「付いてくる」を直した | decisions (物理パラメータ)。§3(c) は 9 行から参照される | 11 |
| §4.0〜§4.3 | 220-313 | 合意した決定事項・行動空間 (ロープ相対 + Δlink)・最初の報酬・mdp の実装 | decisions (行動空間 1 本、報酬の設計 1 本) | 19 |
| §4.4〜§4.5 | 314-339 | 退化時の扱い・240 Hz | decisions (240 Hz) | 6 |
| §4.6〜§4.7 | 340-379 | 実装中の判明事項・高レベル状態遷移グラフの構想 | history (構想は v3 で実現済み) | — |
| §4.8.1〜§4.8.4 | 380-442 | 移植元の構成を使わない理由・固結びの判定・交差数 3 を使わない理由・位相クラスのラダー | decisions (固結びの判定 1 本、報酬ラダー 1 本) | 1 |
| §4.8.8 | 447-464 | 整形報酬の検証の罠 | `.claude/rules/mdp.md` (D-3、D-18) と decisions | 1 |
| §4.8.9 | 465-483 | 学習の前の確認 (ゲート) | skill `physics-check` と workflow.md | 7 |
| §4.8.10 | 484-552 | ゲート 1 の結果と探索の経緯 | `experiments/2026-09-27_gate1.md` (シード・commit・出力のパスは「記録なし」と書く: B-117) | 15 |
| §5 作業上の約束 | 553-558 | 約束 | CLAUDE.md・テスト (§4) | — |
| §6 バージョン制への移行 | 559-583 | 「版 = 設定」の理由・実装の注意 | decisions (`kt/README.md:154-173` と合わせて 1 本) | — |

decisions の候補 (Phase 3 で抜き出しながら確定する。近いものはまとめる):
行動空間 (ロープ相対 + Δlink)、240 Hz、物理パラメータ (「付いてくる」問題)、固結びの判定 (テンソル層・交差数 3 を使わない)、報酬の設計 (位相クラスのラダー + ポテンシャル整形 → v3/v4 の段と遷移で 1 回だけ払う報酬: B-12)、「版 = 設定」、到達状態アーカイブと段の粒度 (B-12)、v4 の速度の判断 (反復 8・4 手)、ロープ `Knot` の選定 (`Report_rope_models.md` §10)、整形報酬の検証の注意。計 10 本前後。

確信: 中。変わる条件: 抜き出してみて、docstring がすでに理由を持っている節は、decisions を作らずに docstring への参照で済ませる (二重にしない)。

### 2.5 情報を失わないことの確かめ方

BRIEF §4 Phase 3 の【必須】の手順をそのまま行う。加えて次の 2 点。
- 抜き出しのスクリプトは `refactor/` に置き、一覧 (commit・seed・iteration・`model_*.pt`・パス・単位付きの数値・%) を移行の前に保存する。
- `next_action.md` の節番号の参照 (48 行) と、`Report.md` の旧名の参照 (29 行: F §0 の 8) は、移行の後に `git grep` で 0 件 (または対応表だけ) になることを確かめる。

---

## 3. 用語の正式名称と命名規則

### 3.1 `docs/glossary.md` の中身

C.md §2 の表 (推奨する正式名称) を、そのまま glossary の元にする。主な決定は次のとおり。どれも**文書だけで直せる** (C §6.2)。

| 用語 | 正式名称と使い分け | 根拠 |
|---|---|---|
| 段 / 位相クラス | 「段」は v4 以降の `BfsPotential.stage` (= round(3Φ)、0 = 自明・3 = 固結び) だけに使う。v1〜v3 の報酬ラダーの段は「位相クラス」。「段 = 固結びへの BFS 距離」という記述 (`kt/README.md:42`、`kt/common/mdp/stages.py:1`) は向きが逆なので直す | C-1、G (cold-start の 1 問は 2 本ともこの古い記述に気づかなかった) |
| Knot | ロープのプリセットには必ず「ロープ」を付ける (ロープ `Knot` = 28 リンク)。課題は「紐結び」(`knot_tying`)、目標は「固結び」(overhand knot)、評価のキー `knot` は「固結びに達した」 | C-3 |
| Twist / Twisted | ロープ `Twist`、捩り、R1 の輪を書き分ける。論文は TWISTED-RL (Freund ら) と TWISTED (Sudry ら)。このリポジトリは isaac_twisted_rl | C-4 |
| 版 vN | MDP の版だけに使う。XPBD の指の接触は「指の接触 (方式 1 / 2)」、物理試験の回は日付で呼ぶ。版の番号は使い直さない (旧 v7: E-2) | C-5、E-2 |
| Play タスク | 版ごとの意味の表を置く (v0〜v3 = 推論を見る環境、v4 以降と rope study = chain 評価の環境) | C-6 |
| 発散 | 判定を 3 つに分けて呼ぶ: 終了項 `diverged` (版ごとの閾値の表付き)、NaN、`divergence_bench` の spike | C-7 |
| chain | 「chain 評価」は `eval_chain.py` の評価だけ。`chain_est` は「学習ログの見積り」。剛体鎖・ジョブのキューには chain を使わない | C-12 |
| 1 手 / 位相移動 | 1 手 = 1 env ステップ (1 回の把持と置き)。`knot_graph` の R1 / R2 / cross は「位相移動」 | C-13 |
| next edge / ρ(g) | v6 の観測 `next_edge` は論文の ρ(g) の「代わり」と書く (ρ(g) の中身は論文に無い) | C-14 |
| archive / seed | 「状態アーカイブ」と「初期アーカイブ」。乱数は「シード」。文書の置き場に archive を使わない | C-11 |
| モデル | 「紐モデル」は rope study の `RopeModel`。`common/rope_model/` は「剛体鎖のロープの資産と設定」。`model_<it>.pt` は「チェックポイント」 | C-10 |
| 紐 / ロープ | 新しい文書は「ロープ」に統一する。凍結した rope study の文書は書き換えず、glossary に同義と書く | C-21 |
| その他 | ゲート 1 (台本の固結び試験)、`-RNN` = GRU、R2 除去 / R2 移動 / v2-R2、`LowMu` = 実物の範囲の摩擦 | C-17〜C-20 |

互換名の表 (glossary の後半。BRIEF §9 の「過去の名前は互換表にだけ置く」):
- タスク ID の規則と全一覧への参照 (C §3 の 166 本。一覧はスクリプトで出せるので glossary には規則だけを書き、全一覧は持たない)。
- ログのフォルダ名 ↔ 版・変種・紐モデル (C §4.2。コードに無い `knot_tying_v4_speed`・`knot_tying_v6_ablation` の出どころ: E-17)。
- 過去の実験名 (run 名) と、旧 v7 (= v6 + 時間切れの罰則) のような使い直された名前 (B-6、C-5)。
- 改名されたファイル (`paper/Report.md` → `paper/Report_Paper.md`、`dbf9a97`)。

### 3.2 これからの命名規則

新しく名前を付けるときの規則だけを決める。既存の名前は変えない (§3.3)。

| 対象 | 規則 | 根拠 |
|---|---|---|
| 紐モデル | 軸の順に連結: 表現 (`Art` / `XpbdC` / `XpbdMS` / `Fem`) + リンク数 + 摩擦の変種 (`LowMu`)。剛性などの軸を足すときは、末尾に変種として足す。既存の 8 個はこの規則にほぼ合う | C-9、C-20、BRIEF §4 Phase 2 の 3 |
| 版 | 整数を 1 つずつ増やす。番号は使い直さない。派生 (`-R2` のような) は作らず、次の番号にする | C-5、E-2 |
| run 名 | `--run_name <略>_s<シード>`。ロープ `Knot` 以外のプリセットで学習するときは、略にプリセット名を入れる (ログ名にプリセットが無いため: C-2) | E-1、C-2 |
| 実験記録 | `docs/experiments/YYYY-MM-DD_<題>.md` (日付は学習を始めた日) | §2 |
| 設計判断 | `docs/decisions/NNNN-<題>.md` | §2 |

### 3.3 コードの識別子を変える案

**推奨: この作業では識別子を変えない (§7 の Q2-2)。** 下の表は、変える場合の費用を判断の材料として示す。参照の数は HEAD `6f41ef1` で数えた (C §6.3 の数、または下のコマンド)。

| 対象 | 変える案 | 参照の数 | ログ・チェックポイント・タスク ID への影響 | 後方互換の方法 | 文書だけで足りない理由 | 推奨 |
|---|---|---|---|---|---|---|
| `experiment_name` にプリセットを入れる (C-2) | `knot_tying_vN` → 例 `knot_tying_knot_vN` | 追跡ファイルの `logs/rsl_rl/` が 78 行 / 32 ファイル (C §6.3) | **ログのフォルダが変わる。** 新しい run は別フォルダ、`--resume` は新しい名前の下しか探さない。チェックポイントの中身は影響なし | 既存のフォルダは動かさない。新旧の対応を glossary に書く | プリセット違いの run がフォルダ名で区別できず、`--resume` が別のプリセットの最新 run を拾いうる | しない。代わりに run 名の規則 (§3.2) と `docs/system/logs.md` で防ぐ。学習するのは今ほぼロープ `Knot` だけ (C-23: Simple/Stiff/Fine の ID は参照 0 本) |
| `--experiment_name` (C-8) | 効くようにする、または引数を消す | `cli_args.py` の定義 1 か所 | 引数を渡したときだけログ名が変わる (今は黙って無視される) | — | 文書に「別フォルダに置く」とあるのに手段が無い (B-108) | Phase 4 の候補 (§6 の P4-6)。挙動の変更なので承認が要る |
| rope study の `MODELS` (C-9) | 一覧を `models.py` に一本化し、評価先の一覧を別名にする | 同名の定義 4 か所 (`models.py:97`、`make_eval_jobs.py:30`、`summarize_eval.py:22`、`run_physics_suite.sh:14`) | なし (名前の値は変えない) | — | 同じ名前で中身が違う | しない。rope study は打ち切り中。再開するときに行う |
| `common/rope_model/` (C-10) | 例 `rope_asset/` | import の行 13 行 / 8 ファイル (`git grep -E 'common\.rope_model\|from \.+rope_model\|rope_model\.' -- '*.py'`) + `gym.register` の文字列 | なし | 旧名の再 export | 紐モデルと名前が衝突する | しない。glossary で書き分ければ足りる |
| `StateArchive(classes=…)` (C-1) | `keys=` | `classes=` が 19 行 (`git grep -n 'classes=' -- '*.py'`。全部がこの引数とは限らない) | なし (保存ファイルに引数名は出ない: C §6.2) | 旧い引数名を受ける | v4 は段を入れるのに名前が「クラス」 | しない。docstring で足りる |
| `XPBD_FINGER_V2` (C-5)・`representation.py` の `next_edge` (C-14、11 行)・`knot_graph` の `moves` (C-13)・`run_train_chain.sh` (C-12、参照 7 行) | 中身の名前に変える | 各 5〜11 行 | なし | — | 同名で別物 | しない。glossary で足りる。利益が小さく、移植元との対応 (N-3) を崩すものもある |
| タスク ID・`-Play`・`-RNN` (C-3、C-6、C-18) | — | 166 本 / 83 本 / 30 本 (2026-10-04 訂正: C.md と初版は 18 本としていたが、v2・v2-R2・v3 x 5 プリセット x 学習 / Play で 30 本) | タスク ID が変わる | — | — | しない (BRIEF の最重要の原則 1) |

確信: 高 (タスク ID とログ名を変えない)。中 (内部の識別子も変えない)。変わる条件: rope study を再開すると決まれば、C-9 の一本化は再開の最初の作業として価値が出る。

---

## 4. Claude 基盤の設計

### 4.1 一覧

今、規則を守らせる仕組みはほぼ無い (D §0: CLAUDE.md・`.claude/settings.json`・hook・CI・pre-commit・ruff の設定のどれも無い)。
BRIEF §6 の順 (テスト → hook → permissions → CLAUDE.md → rules → skills → agents) で、上の段で強制できるものは上に置いた。

| # | 置き場所 | ファイル | 目的 | 根拠 | 効果の確かめ方 |
|---|---|---|---|---|---|
| T1〜T7 | テスト | `tests/test_invariants.py`、`tests/test_docs.py` | 4 つの不変条件・版の一覧・リンク・一時パスを機械で検査する (§4.8) | F-4、D-16、D-9、C-5、E-2、A-3、CC-10 | 今の HEAD で通り、わざと壊したコピーで落ちる |
| H1 | hook (SessionStart) | `.claude/hooks/session_start.sh` | git status・STATUS の要点・走っているジョブを冒頭に出す | D-11、D-15、CC-1、E-7 | 想定の JSON を渡して出力を見る。新しい session で文脈に入ることを確かめる |
| H2 | hook (PreToolUse Bash) | `.claude/hooks/guard_launch.sh` | 学習・評価の起動時に、未コミットの変更があれば確認を求める | D-6 (29 run 中 21 run)、CC-8、B-1 | 汚れた作業ツリーと綺麗な作業ツリーで、起動コマンドと無関係なコマンドを渡す |
| H3 | hook (PostToolUse Edit/Write) | `.claude/hooks/ruff_after_edit.sh` | `*.py` の編集の後に整形と lint (方式は §7 の Q2-4 で決まる) | A-10、E-13、F §0 の 3、CC-11 | 整形の要る `.py`・要らない `.py`・`.md` を渡す |
| H4 | hook (PreToolUse Bash) | `guard_launch.sh` に同居 | `generate_assets.py` を `--rope` なしで呼んだら止める (全プリセットの USD を上書きする) | D-13、R-44 (損害が大: 管理外の資産で戻せない。破られた記録は無い) | `--rope` あり・なしのコマンドを渡す |
| P | permissions | `.claude/settings.json` | kill は確認、`pkill -f` は拒否。`logs/`・`outputs/`・`.venv/`・USD・PDF の編集を拒否。旧実装の読み込みを拒否。`PYTHONUNBUFFERED=1` | D-1、CC-2、D-13、D-19、D-22、BRIEF §2 | 該当する操作をしてみて、拒否・確認が出ることを確かめる |
| M | CLAUDE.md | `CLAUDE.md` (ルート) | 毎セッション要る、機械で判定できない規則と入口への導線 (§4.2) | §4.2 の表の各行 | Phase 5 の cold-start テスト。各行が「消すと間違えるか」の試験に通るか見直す |
| R1 | rules | `.claude/rules/mdp.md` (paths: `**/common/mdp/**`) | 項は追加だけ・`env.step()` の順序の罠・報酬の測り方 | D-14、D-3 (3 回 + 疑い 1)、D-18、CC-8 | `common/mdp/` のファイルを開いた session で、規則が文脈に入ることを確かめる (InstructionsLoaded) |
| R2 | rules | `.claude/rules/versions.md` (paths: `**/knot_tying/versions/**`) | 既存の版の挙動を変えない・継承の波及・版の追加は skill へ | D-14、F §0 の 9、CC-9、E-2 | 同上 |
| S1〜S5 | skills | `.claude/skills/<名前>/SKILL.md` | 学習と chain 評価・版の追加・物理の確認・実験記録・引き継ぎ (§4.4) | E-1〜E-10、D-4〜D-7、D-11 | description の区別の試験 (§4.4 の表)。Phase 5 で実際に使う |
| A1 | subagent | `.claude/agents/log-analyst.md` | 学習ログ・評価 JSON を読んで数値の表を返す | E-1、E-15、B-14、D-7 | 既知の run (v6 seed 43) の数字を出させ、`Result_Report_v6.md` の値と比べる |
| G | .gitignore | `.gitignore` | Claude Code の作業ファイルをコミットしない | D-23 (`6f41ef1` で消した `scheduled_tasks.lock`) | `git status` に出ないこと |

### 4.2 CLAUDE.md の案

約 90 行を目安にする (BRIEF の目安 150 行、spec §1 の 200 行未満)。`@` import は使わない (spec §11 の 1)。下は骨組みで、各行の根拠は後の表に書く。

```markdown
# isaac_twisted_rl

Isaac Lab 上で、ロープを固結び (overhand knot) にする強化学習の研究コード。
現在地は `docs/STATUS.md`、文書の地図は `docs/README.md`、用語は `docs/glossary.md`。

## セッションの始めと終わり
- 始め: 冒頭に出る git status・STATUS の要点・走っているジョブを確かめる (SessionStart hook)。
- 終わり: skill `handoff` で `docs/STATUS.md` を上書きする。追記で伸ばさない。結果は実験記録へ。

## 検証のコマンド
- テスト: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run python -m pytest tests/ -q` (約 50 s、Isaac Sim 不要)
  - `uv run pytest` は使わない (`.venv/bin/pytest` は別リポジトリの python を指す)
- lint: (Q2-4 の決定に従って 1 行)

## 壊してはいけない制約 (tests/test_invariants.py が検査する)
- `tasks/` の下の `__init__.py` で isaaclab.sim などを import しない。Kit の起動前に実行される。
- フォルダ名 `mdp` を変えない。import の除外がこの名前の部分一致で決まる。
- `common/topology/` と `common/state_archive.py` は isaaclab・omni・pxr を import しない。Isaac Sim なしで単体テストを回すため。
- `common/mdp/` の項は追加だけ。削除・改名しない。過去の版の設定が参照している。

## 実験の約束
- 版どうしの比較は chain 評価 (定義: `docs/system/evaluation.md`) で、シード 2 本以上。学習ログの値や `chain_est` で比べない。
- 学習の前にコードを commit する。未追跡のファイルは run のフォルダに残らない。
- 学習ログや代理の指標だけで出した結論は「推測」と書く。事実として commit メッセージや README に書かない。
- 評価は学習と同じ Hydra の上書き・同じ `--device` と `agent.device` で行う。
- rope study の run の最後のチェックポイントは `model_99.pt` (`model_100.pt` は無い)。

## 長時間のジョブと GPU
- GPU 2 枚を他のジョブと共有している。起動の前に `nvidia-smi` を見る。
- Isaac Sim を使うスクリプトは `python -u` で起動し、`timeout -k` を付け、ログをファイルに書く。
- 待つときは前景の `sleep` を使わず、`run_in_background` か Monitor を使う。
- 自分が起動していないプロセスを kill しない。PID をユーザーに伝える。`pkill -f` は使わない (自分のシェルも殺す)。
- 残すスクリプトとログは追跡する場所か `outputs/` に置く。scratchpad (`/tmp/claude-*`) を文書から参照しない。

## 触らないもの
- `/home/itoyama/work/IsaacLab/` の下の旧実装の knot_tying (rsl-rl 5.x 前提の移植前のコード) は参照しない。
- `logs/`・`outputs/` は学習・評価のスクリプトが書く。手で編集しない (git の管理外で戻せない)。
- ロープの USD と `outputs/archive_seed/` を上書きしない。USD を作り直すときは `generate_assets.py --rope <名前>` で絞る。
```

| 節 | 行 | 根拠 | 機械で強制できるか (上の段との関係) |
|---|---|---|---|
| 冒頭 | 導線 3 本 | A-1、A-4、G-2 (1 問あたり約 400 行を読んでいる) | — |
| 始めと終わり | 2 行 | D-11、CC-5、E-7 | 始めは H1 が出す。終わりは機械で判定できない |
| 検証のコマンド | 3 行 | A-10 (コマンドが 7 か所で違う)、F §0 の 2 (venv の取り違え)、D-20 | — (正しいコマンドを知らせるだけ) |
| 制約 4 行 | 4 行 | D-16、D-14、F-4 | T1〜T4 が強制する。CLAUDE.md には、書く前に避けるための 1 行の理由だけを置く |
| 実験の約束 | 5 行 | D-7 (3 回)、CC-7、D-6 (21/29)、D-17、D-4、E-16 | 2 行目は H2 が確認を求める。他は機械で判定できない |
| 長時間のジョブ | 5 行 | D-1・CC-1 (2 セッション)、CC-2、D-2 (3 件以上)、D-10 (4 件)、CC-3、CC-10 (4 か所)、D-19 | kill は P が確認を出す。`timeout -k` と PID の記録は Phase 4 のラッパー (§6 の P4-1) ができたら「ラッパーで起動する」の 1 行に置き換える |
| 触らないもの | 3 行 | BRIEF §2【必須】、D-13、R-44 | P と H4 が強制する。CLAUDE.md は理由を 1 行で |

入れないもの: 経緯、版ごとの仕様、コマンドの詳細 (skill へ)、session の区切り方 (利用者の好みなので auto memory のまま: BRIEF §6 の 8)。

確信: 中。変わる条件: Phase 5 で回答役が CLAUDE.md の導線を使わずに探し回るなら、導線の書き方を変える。逆に、ある行を消しても答えが変わらない質問しか無ければ、その行は消す候補にする。

### 4.3 パスを限定した規則 (`.claude/rules/`)

サブディレクトリの CLAUDE.md ではなく `.claude/rules/` に置く。理由: 置き場所が 1 つのフォルダにまとまり、一覧しやすい。読み込みの時期はどちらも「そのファイルを扱うとき」で同じ (spec §1・§2)。

| ファイル | `paths` | 中身 | 根拠 |
|---|---|---|---|
| `mdp.md` | `src/isaac_twisted_rl/tasks/knot_tying/common/mdp/**` | (1) 項は追加だけ。役目を終えた項は `.. deprecated::` で残す (T4 が検査)。(2) `env.step()` の順序 (終了判定 → 報酬 → reset → 観測) から来る罠: 報酬と終了の値は reset の前のもの、キャッシュの更新の時期、`get_term` は reset した環境で絞る。(3) 整形報酬は退避の後に測り、γ を学習と揃える | D-14、R-06、CC-8、D-3 (3 回 + 疑い 1)、D-18、next_action §4.8.8 |
| `versions.md` | `src/isaac_twisted_rl/tasks/knot_tying/versions/**` | (1) 既存の版の挙動を変えない。直すなら新しい版を作る (バグ修正は例外として記録する)。(2) v4 の `env_cfg.py` を変えると v5・v6・v7・rope study に波及する (継承の連鎖)。(3) 新しい版は skill `add-version` で作る | D-14、F §0 の 9・§5.2、CC-9、E-2 |

作らない rules: `common/topology/` の import の禁止 (T2 が強制し、CLAUDE.md に理由が 1 行ある。3 か所目を作らない)。rope study 用 (打ち切り中)。

確信: 中。変わる条件: `paths` の glob が想定のファイルで効かない場合は、サブディレクトリの CLAUDE.md に切り替える (Phase 3 で InstructionsLoaded hook か `/memory` で確かめる)。

### 4.4 skills

どれも E.md で 2 回以上繰り返された手順を根拠にする (BRIEF §7 の条件)。本文にはコマンドと確認の項目を書く。description は用途を先頭に書く (spec §3)。

| skill | description の案 | 中身 | 根拠 |
|---|---|---|---|
| `train-eval` | 版や設定を学習し、chain 評価で比べるときに使う。学習の起動 (シード 2 本以上)、スモークテスト、chain 評価、集計と標準誤差まで | スモークテスト (数 iteration) → 学習 (`--headless`、`--device` と `agent.device` を揃える、`agent.save_interval=10`、`--run_name <略>_s<シード>`) → chain 評価 (512 環境・seed 7・決定論的・学習と同じ Hydra の上書き・最後のチェックポイントの名前を確かめる) → 集計 (標準誤差とシード差を並べる) → skill `experiment-record` へ | E-1 (run 29、chain 評価の JSON 15、まとまり 3)、E-5、E-11、D-4、D-7、D-17、E-16 |
| `add-version` | MDP の新しい版 vN を `versions/` に追加するときに使う | 作るファイル (`env_cfg.py`・`agents/`・`__init__.py`・README)、登録 (`_VERSIONS`・`versions/__init__.py` の一覧・タスク ID)、継承元の確かめ方、T5 を回す、STATUS と `kt/README.md` の系譜の 1 行 | E-2 (版の追加 6 + 作り直し 1。`versions/__init__.py` の更新漏れ 3/6、ルート README 1/6)、CC-9 |
| `physics-check` | 物理パラメータ・ロープのプリセット・紐モデルを変えたとき、学習の前に、台本で固結びが結べるか (ゲート 1) と 1 手あたりの発散を測るときに使う | ゲート 1 (まっすぐなロープから始める、物理の版ごとに出力を分ける、`[final]` 行を表にする)、初期アーカイブの作り直し、発散の計測 (どの判定で数えたかを出力に書く) | E-3 (出力 76、4 つの時期)、E-4 (4 つの時期)、E-8 (7 件以上)、E-9、C-7、D-3 |
| `experiment-record` | 学習・評価・計測の結果を `docs/experiments/` に実験記録として書くとき、また結果を無効にして退避するときに使う | §5.2 のテンプレート。再現のメタデータの確認 (commit・未コミットの差分・run のフォルダ名・コマンド・Hydra の上書き)。無効にするときの手順 (退避先・実験記録への記載。`logs/` には手で書かない)。`docs/experiments/README.md` の一覧への 1 行 | E-6 (報告書 4)、E-10 (退避 7 件)、B-1、B-2、B-102、D-5 (5 回) |
| `handoff` | セッションを終える・区切るとき、`docs/STATUS.md` を上書きで最新にして次のセッションへ引き継ぐときに使う | STATUS の書式 (現在地・走っているジョブと PID・次の作業・保留)。上書きで書き、経緯は書かない。コミットできる状態か (テスト・lint) を確かめる | E-7 (「現在地」5 コミット、HANDOFF 8 節)、D-11、CC-5、CC-12 |

description の区別の試験 (BRIEF §4 Phase 3。Phase 3 で description を書いた後に行う):

| skill | 呼ばれるべき依頼 (3) | 呼ばれるべきでない依頼 (2) |
|---|---|---|
| `train-eval` | 「v6 を seed 42 と 43 で 100 iteration 学習して」「v7 と v6 を比べて」「この run の chain 評価を測って」 | 「chain 評価の定義を教えて」(evaluation.md)、「TensorBoard のログを要約して」(log-analyst) |
| `add-version` | 「v8 を作って」「v7 をもとに観測を変えた版を足して」「新しい版を登録して」 | 「v6 の報酬のバグを直して」、「v4 の仕様を教えて」 |
| `physics-check` | 「ソルバの反復を変えたので学習の前に確かめて」「ゲート 1 を回して」「新しいプリセットの発散率を測って」 | 「学習中の発散率のログを見て」、「発散の定義は」 |
| `experiment-record` | 「今の結果を実験記録にまとめて」「この run は無効なので退避して」「切り分けの報告書を書いて」 | 「STATUS を更新して」(handoff)、「README の誤記を直して」 |
| `handoff` | 「今日はここまで。引き継ぎを書いて」「STATUS を更新して」「次のセッション用にまとめて」 | 「HANDOFF.md の昔の記録を探して」、「実験記録を書いて」 |

作らない skill: 長時間ジョブの起動と監視 (規則は 5 行で CLAUDE.md に入る。起動の本体は Phase 4 のラッパー)、速度の計測 (E-12: 3 回とも方法が違い、手順が固まっていない)、方策の書き出し (E-14: 意図した繰り返しが無い)、評価キュー (E-5: `train-eval` の中に入れる)、テストと lint (E-13: hook 向き)。

確信: 中。変わる条件: Phase 3 で書いてみて、`physics-check` のゲート 1 と発散の計測が別々に呼ばれる場面が多いと分かれば 2 つに分ける。

### 4.5 subagent

| agent | 中身 | 根拠 | 確信 |
|---|---|---|---|
| `log-analyst` | 学習ログ (TensorBoard の events)・評価 JSON・run の `params/*.yaml` を読み、指定された指標の表を返す。ログのキーの意味 (B-113)、チェックポイントの番号 (D-4)、`chain_est` を比較に使わないこと (D-7) を定義に書く。tools は Read・Grep・Glob・Bash (読むだけ)。書き込みは scratchpad だけ | E-1 (chain 評価の JSON 15、rope study の評価 JSON 117)、E-15 (JSON のキーが道具ごとに違う)、B-14 (TensorBoard から抜き出した数値のスクリプトが残っていない) | 中。変わる条件: 実際の分析が数回に 1 回しか無いなら、general-purpose への依頼文の型 (skill の中) で足りる |

作らない: 文書とコードの整合性の監査役 (機械的な部分はテスト T5〜T7 が受け持つ。整理は D-9 で 2 回あっただけ)。

### 4.6 hooks

どの hook も、設定は `.claude/settings.json`、本体は `.claude/hooks/` のシェルスクリプトに置く。拒否は `exit 2` か JSON の `permissionDecision` で行う (`exit 1` では止まらない: spec §5)。有効にする前に、想定する入力の JSON を渡して「止めるべき操作」と「通すべき操作」の両方を試す (BRIEF §4 Phase 3)。

| hook | イベントと matcher | 動き | 誤検知したときの影響 | 止める方法 |
|---|---|---|---|---|
| H1 | SessionStart、`startup\|resume\|clear\|compact` | 標準出力に次を出す (計 40 行まで): `git status -sb` (15 行まで)、`docs/STATUS.md` の「要点」の節、`nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader`、`ps` で train・eval・play・kit を含むプロセス。どのコマンドが失敗しても `exit 0` | 誤検知は無い (表示だけ)。費用は毎回 40 行ほどの文脈 (Phase 5 の読んだ量に含めて測る) | settings.json から外す |
| H2 | PreToolUse、`Bash` | コマンドが学習・評価の起動 (`train`・`scripts/rsl_rl/train.py`・`eval_chain.py`・`eval_search.py`・`eval_robust.py`。`timeout`・`nohup`・環境変数の代入の後ろも見る) で、かつ `git status --porcelain --untracked-files=all` に `.py`・`.sh`・`.yaml` があれば、`permissionDecision: "ask"` と理由 (変更のファイル数・未追跡のファイル名) を返す。それ以外は何もしない | 起動でないコマンド (例 `grep train.py`) で確認が出る。影響は確認の 1 回だけで、作業は止まらない | settings.json から外す。誤検知のパターンはスクリプトの照合を直す |
| H3 | PostToolUse、`Edit\|Write` | 編集したファイルが `*.py` なら、Q2-4 で決まった方式で ruff を回し、残った指摘を `additionalContext` で返す (PostToolUse は操作を止められない) | 整形でファイルが書き換わる (Q2-4 の方式による)。誤った指摘は文脈を少し使うだけ | settings.json から外す |
| H4 | PreToolUse、`Bash` (H2 と同じスクリプト) | コマンドに `generate_assets.py` があり `--rope` が無ければ `permissionDecision: "deny"` と理由 | `--rope` を別の書き方 (`--rope=fine`) にしたときに誤って止める。照合を `--rope` の前方一致にして避ける | 同上 |

採らない hook: 既存の版を編集したら警告する hook (D-14 の案。バグ修正で正当に編集するので誤検知が多い。T4 と rules R2 で足りる)、Stop で全テストを回す hook (CC-11 の 1 回だけ。1 回 50 s かかる)、`python -u` の照合 (D-19: 踏んだ記録が無い。`env` で足りる)。

### 4.7 `.claude/settings.json` の permissions と env

```json
{
  "permissions": {
    "ask":  ["Bash(kill *)", "Bash(pkill *)", "Bash(killall *)"],
    "deny": ["Bash(pkill -f *)",
             "Edit(/logs/**)", "Edit(/outputs/**)", "Edit(/.venv/**)",
             "Edit(**/*.usd)", "Edit(/paper/*.pdf)",
             "Read(//home/itoyama/work/IsaacLab/**/knot_tying/**)"]
  },
  "env": { "PYTHONUNBUFFERED": "1" }
}
```

- kill を ask にし、allow には入れない (BRIEF §6 の 3)。`pkill -f` は自分のシェルまで殺したので deny (CC-2)。利用者の設定の allow `pkill -f "quick_train_10min.py"` より、プロジェクトの deny が勝つ (spec §7)。このプロジェクトの中だけで効く。
- `Edit(...)` の deny は Write と、Bash の `sed`・`tee`・リダイレクトにも効く。学習・評価のスクリプト (Python の子プロセス) の書き込みには効かない (spec §7、§11 の 4)。そのため学習は妨げない。
- 旧実装の Read の deny は、利用者の設定の allow `Read(//home/itoyama/work/IsaacLab/**)` に勝つ (spec §7)。IsaacLab 本体のソースの正当な参照は止めないよう、`knot_tying` の下に限る。旧実装の正確なパスは BRIEF の【必須】に従って開いていない。`grep -r` と Python には効かない (spec §7)。
- `env` が Bash の子プロセスに渡ることは [未確認]。Phase 3 で `echo $PYTHONUNBUFFERED` で確かめ、渡らなければ CLAUDE.md の 1 行 (`python -u`) だけにする。
- 根拠: D-1、CC-1、CC-2、D-13、R-44、D-19、D-22、BRIEF §2。

### 4.8 テスト (Phase 3 で足す)

BRIEF §3 で承認なしに足してよい「不変条件を確かめるテスト」にあたる。どれも静的な検査で、Isaac Sim は要らない。F.md §6.3 と付録 A のスクリプトを元にする。

| # | 検査 | 今の HEAD での状態 | 根拠 |
|---|---|---|---|
| T1 | `tasks/` の下の `__init__.py` が、isaaclab.sim など Kit の起動後でないと読めないものを最上位で import しない | 違反 0 (F §0 の 1) | D-16、R-01 |
| T2 | `common/topology/` と `common/state_archive.py` が isaaclab・omni・pxr を import しない | 違反 0 | D-16、R-04 |
| T3 | `mdp` のフォルダ名と、`import_packages` の除外の部分一致が合っている | 違反 0 | D-16、R-03 |
| T4 | `common/mdp/` の公開名が、基準の一覧 (6f41ef1 の時点) から消えていない | 違反 0 (I4) | D-14、R-06 |
| T5 | `versions/` のフォルダ・`_VERSIONS`・`versions/__init__.py` の一覧が一致する | **違反あり**: 一覧に v7 が無い (C-5)。docstring を直してから足す | C-5、E-2 |
| T6 | 全 Markdown の相対リンクが実在するファイルを指す | **違反あり**: 11 本 (A-3)。リンクを直してから足す | A-3、D-9、BRIEF Phase 5 |
| T7 | 追跡しているファイル (`docs/history/` を除く) に `/tmp/claude-` が無い | **違反あり** (CC-10 の 4 か所のうち追跡ファイルのもの)。移動・修正の後に足す | CC-10、D-12 |

Phase 4 に回すもの: 追跡している `*.sh` の `timeout` に `-k` があること (D-2。今は 4 本とも違反で、直すのはシェルの変更)。

### 4.9 その他

- `.gitignore` に `.claude/*.lock`、`.claude/worktrees/`、`.claude/settings.local.json` を足す (D-23)。`.git/info/exclude` にあるものはリポジトリで共有されないため。
- auto memory には何も足さない。リポジトリが持つべき事実はリポジトリに置く (BRIEF §6 の 8)。

---

## 5. 作業の型

`docs/workflow.md` に段階・終了の条件・残す記録を書き、コマンドは skill に置く (§2.1)。

### 5.1 研究の 1 サイクル

実際の順序は版ごとに揺れていた (E-19: ゲート 1 が関門だったのは 09-27 の 1 回だけで、以後は物理を変えたときの確認。評価の型は v4 で TensorBoard・Play から chain 評価に変わった)。そのため、どの段階を飛ばしてよいかを条件で書く。

| 段階 | やること | 終了の条件 | 残す記録 | skill | 根拠 |
|---|---|---|---|---|---|
| 1. 仮説 | 何を変えると何がどれだけ良くなるかを書く。比べる相手 (基準の版) と指標 (chain 評価の何 %) を決める | 仮説・基準・指標・必要なシード数が STATUS の「次の作業」にある | STATUS | — | D-7、R-23 |
| 2. 実装 | 新しい版か設定の変更。既存の版の挙動は変えない | テストと lint が通る。**commit がある** (未追跡のファイルを含む) | commit | `add-version` | D-6 (21/29)、CC-8、D-14 |
| 3. ゲート (学習の前) | スモークテスト (数 iteration)。物理・プリセット・紐モデルを変えたときだけ、ゲート 1 と発散の計測 | スモークテストが落ちない。ゲート 1 の `[final]` の表と発散率が出ている | 実験記録の「ゲート」欄 | `physics-check`、`train-eval` | E-3、E-4、E-11、E-19 |
| 4. 学習 | シード 2 本以上。GPU の空きを確かめてから起動 | 最後のチェックポイント (`model_<最後>.pt`) がある。run のフォルダ名が分かっている | STATUS の「走っているジョブ」(PID・GPU・run のフォルダ) | `train-eval` | E-1、D-10、E-16 |
| 5. 評価 | chain 評価 (512 環境・seed 7・決定論的・学習と同じ上書き) | シードごとの評価 JSON がある | 評価 JSON のパス | `train-eval` | E-1、D-17、B-111 |
| 6. 実験記録 | §5.2 のテンプレートで書く。否定的な結果も書く | 再現のメタデータに空欄が無い (欠けているものは「記録なし」) | `docs/experiments/` と一覧の 1 行 | `experiment-record` | B-1、B-102、E-6 |
| 7. 判断 | 標準誤差とシード差を並べて、仮説を採るか捨てるかを決める。「分かった / 推測 / 未確認」を分ける | 結論が実験記録の「結論」と STATUS にある | STATUS。設計を変えたなら decisions | `handoff` | D-7、E-16 |

既存の知見の書き方 (E-16 による注意): 「シード差は 15〜19 ポイント」は v6 の 2 シード (42 / 43) の 1 組だけが出典で、iteration 50 では 4.9 ポイント。workflow.md には「シード差の推定はこの 1 組だけ。だから 2 シード以上で比べる」と出典ごと書き、一般則のようには書かない。

### 5.2 実験記録のテンプレート

```markdown
# <題> (YYYY-MM-DD)

状態: 凍結 / 読むべき状況: <一文>

## 目的と仮説
## 条件
| 項目 | 値 |
|---|---|
| commit | <hash> (未コミットの差分: なし / あり → パッチのパス) |
| タスク ID とプリセット | |
| 変えた変数 | |
| シード | |
| iteration (最後のチェックポイント) | |
| num_envs | |
| 学習のコマンド | (Hydra の上書きを全部) |
| 評価のコマンド | |
| run のフォルダ | logs/rsl_rl/<ログ名>/<日時>_<run 名> (シードごと) |
| 評価 JSON | |
| 無効にした run | (あれば。退避先と理由) |
## ゲート (学習の前の確認)
## 結果
(表。指標・シードごとの値・標準誤差)
## 結論
- 分かった:
- 推測:
- 未確認:
## 次の作業
(STATUS へ写すもの)
```

E-15 のとおり、評価 JSON にシードと Hydra の上書きが入っていない道具がある (`eval_chain_*.json`)。そのため、コマンドを記録の側に必ず書く。

### 5.3 セッションの規約

| 時 | やること | 仕組み | 根拠 |
|---|---|---|---|
| 始め | git status・STATUS の要点・走っているジョブを確かめる | H1 が自動で出す。CLAUDE.md に 1 行 | D-11、D-15、CC-1 |
| 途中 | 長い待ちは `run_in_background` か Monitor。並行の作業は git worktree で分ける | CLAUDE.md | CC-3、D-15 |
| 終わり | STATUS を上書き (走っているジョブと PID を含む)。結果は実験記録へ。テストと lint が通る状態にする | skill `handoff` | E-7、CC-5、CC-11 |
| 引き継ぎメモ | HANDOFF.md のように先頭へ追記して伸ばす形をやめる。「上書きの STATUS」と「凍結の実験記録」に分ける | skill `handoff`、`experiment-record` | CC-5、B-4、BRIEF §8 の 3 |

### 5.4 長時間ジョブの規約

| 項目 | 規約 | 仕組み | 根拠 |
|---|---|---|---|
| 起動 | Phase 4 のラッパー (P4-1) で起動する。それまでは `python -u`・`timeout -k`・ログをファイルへ・絶対パス | ラッパー、CLAUDE.md | D-2、D-19、CC-4 |
| 記録 | PID・GPU・commit・未コミットの差分・コマンドを `outputs/jobs/` に (ラッパーが書く)。STATUS の「走っているジョブ」に PID と run のフォルダ | ラッパー、skill `handoff` | D-1、D-6 |
| 止め方 | 自分の記録にある PID だけを止める。それ以外は PID をユーザーに伝える。`pkill -f` は使わない | permissions (kill は確認、`pkill -f` は拒否) | D-1、CC-1、CC-2 |
| GPU の共有 | 起動の前に `nvidia-smi`。時間を測るときは、そのときの GPU の負荷を記録に書く | ラッパー (空きメモリの確認)、CLAUDE.md | D-10 |

---

## 6. コード構造の変更案 (Phase 4 の候補)

F.md §9 の順序を基にし、D・C の所見から足した。リスクの基準は F.md と同じ (低 = 試験と静的検査で挙動が同じと言い切れる、中 = Isaac Sim なしでは確かめきれない部分がある、高 = 学習・評価の結果が変わりうる)。

| # | 候補 | リスク | 挙動が変わらないことの確かめ方 | 工数 | 根拠 |
|---|---|---|---|---|---|
| P4-1 | 長時間ジョブのラッパー (新しいスクリプト): `python -u`、`timeout -k`、ログの `tee`、PID・GPU・commit・未追跡を含む差分のパッチ・コマンドの記録、起動前の GPU の空きの確認 | 低 (既存のコードは変えない) | GPU の要らない短い Python で試す。その後、承認を得て Isaac Sim のスモークテストを 1 回 | 0.5 日 | D-1、D-2、D-5、D-6、D-10、CC-4 |
| P4-2 | ruff の設定を `pyproject.toml` に置く (F-1)。規則は E4/E7/E9/F + I から。起動スクリプトの E402 は per-file-ignores。版を固定 | 低 | コードの差分が無いこと。設定の前後で `ruff check --statistics` を比べる | 1〜2 h | F-1、A-10 |
| P4-3 | 全体を整形する専用のコミット (F-2)。末尾の改行だけの 8 本は別のコミット | 低 | 全ファイルの `ast.dump` が前後で一致 (docstring の空白を除く)。pytest が 268 passed のまま | 1 h | F-2 |
| P4-4 | 安全な自動修正だけ (F-3: UP033・RUF022・PLR0402。I001 は起動スクリプトを除く。UP018 は除く) | 低 | 対象の行の AST の差分を確かめる。pytest | 1 h | F-3、N-5 |
| P4-5 | `--experiment_name` を効くようにする (または消す) | 低 (引数を渡したときだけ変わる) | 引数なしのときのログのパスが前後で同じことを、`cli_args` の単体の試験で確かめる | 1 h | C-8、B-108 |
| P4-6 | rope study のシェル 4 本の `timeout` に `-k` を付け、T8 (§4.8) を足す | 低 | シェルの差分が `-k` だけ。T8 が通る | 0.5 h | D-2 |
| P4-7 | `make_eval_jobs.py` で無いチェックポイントを警告し、`last` を解決する | 低〜中 | 既存の run の一覧で、生成されるジョブが前後で同じ (足りないものに警告が出るだけ) | 1 h | D-4 |
| P4-8 | 試験の共通化 (`tests/conftest.py` の `sys.path`、F-5) | 低〜中 | `pytest --collect-only -q` の一覧が前後で同じ (268 件) | 2〜3 h | F-5 |
| P4-9 | 版の間の private な名前の import を、明示的な公開名 (別名の追加) にする (F-12) | 低〜中 | import のグラフを静的な試験で固定。設定の値は Isaac Sim で `parse_env_cfg` を比べるまで未確認 | 2 h | F-12、F §0 の 9 |
| P4-10 | `bench_physics.py:378` の `get_term` の疑い (D-3) を確かめる。確かなら直す | **中** (計測の出力が変わる) | Isaac Sim で直す前後の発散率を比べる (GPU の承認が要る) | 0.5 日 | D-3、§7 の Q2-9 |

勧める順序: P4-2 → P4-3 → (H3 を有効にする) → P4-4 → P4-1 → P4-6 → P4-5 → P4-8 → P4-9。P4-7・P4-10 は rope study を再開するときに。

Isaac Sim で確かめる手段ができるまで見送るもの: 純粋な層とスクリプト層の重複の削減 (F-8、F-9)、置き場所の整理 (F-10)、大きなファイルの分割 (F-11)。どれも Isaac Sim なしでは挙動が同じだと言い切れない (F §9 のリスク「中」)。

Phase 3 で行うもの (コメント・docstring・Markdown だけなので Phase 4 ではない): 古い参照の修正 (F-7: `paper/Report.md` の 29 行、`eval_compare.py`、`cli_args.py` の docstring)、`versions/__init__.py` の一覧に v7 を足す (C-5。docstring)。

### やらない方がよいもの

| # | やらないこと | 理由 | 根拠 |
|---|---|---|---|
| N-1 | `mdp` フォルダの改名、mdp の項の削除・改名 | import の除外の部分一致と、過去の版の設定の参照で壊れる | F §10 |
| N-2 | `versions/v0`〜`v2` の重複の共通化 | 過去の実験の設定を書き換えることになる | F §10 |
| N-3 | `representation.py` の camelCase の改名・分割 | 移植元との対応を保つ方針 | F §10 |
| N-4 | `PickPlaceAction` の分割 | Isaac Sim なしで試験できない中核のクラス | F §10 |
| N-5・N-6 | `ruff check --fix` の一括適用、SIM118・B008 などの手での修正 | 起動スクリプトの import 順や Warp のカーネルを壊す。挙動が変わる | F §8.2 |
| N-7・N-8 | デモの把持の定数・AppLauncher の定型の統合 | 意図して値が違う。Kit の起動と import の順序に意味がある | F §10 |
| N-9 | `__init__.py` への import の追加、`topology/__init__.py` の再 export の削減 | 不変条件 I1 に反する、または試験が壊れる | F §10 |
| N-10 | `paper/` の PDF・HTML の移動・削除 | 相対リンクが切れる | F §10 |
| N-11 | Isaac Lab のテンプレート由来のスクリプトの大きな書き換え | upstream との比較が難しくなる | F §10 |
| 追加 | タスク ID・ログ名の変更 (§3.3)、凍結した報告書の分割や書き直し (§1 の約束) | 互換が壊れる。数値や条件が抜け落ちる | BRIEF の最重要の原則 1・3 |

---

## 7. 判断していただきたい点

各問とも、最初の選択肢が推奨。

### Q2-1 文書の言語
- **(a) 推奨: 今のままにする。** 文書・docstring・CLAUDE.md・skills は日本語、ルートの README だけ英語。
- (b) ルートの README も日本語にする。
- (c) Claude 向けのもの (CLAUDE.md・rules・skills・agents) を英語にする。
- 推奨の理由: 追跡している Markdown 23 本のうち 22 本と docstring が日本語で、利用者の言語でもある。訳すと数値や条件が抜け落ちる危険がある (BRIEF §11)。(c) は Claude 向けと人間向けで言語が割れ、用語の対応表がもう 1 つ要る。

### Q2-2 コードの識別子の改名の範囲
- **(a) 推奨: 改名しない。** 衝突は glossary と docstring で解く (§3.3)。
- (b) 互換の問題が無い内部の名前だけ変える (`StateArchive(classes=)`、`XPBD_FINGER_V2`、`representation.py` の `next_edge`、`knot_graph` の `moves`、`run_train_chain.sh`)。
- (c) (b) に加えて、`experiment_name` にプリセットを入れる (新しい run のログのフォルダが変わる)。
- 推奨の理由: C.md の 23 件のうち、識別子を変えないと解決しないものは少なく (C §6.2)、どれも利益より費用 (参照の書き換え・人間の記憶・過去のセッションの記録) が大きい。プリセット違いの事故は run 名の規則で防げる。

### Q2-3 打ち切った 2 研究 (rope study・推論時探索) の文書をどこまで再構成するか
- **(a) 推奨: 最小限。** 報告書 2 本は本文を変えずに `docs/experiments/` へ移し、頭書きを付ける。HANDOFF.md は、有効な run のフォルダ名の一覧 (B-2) と未解決の課題を移したうえで `docs/history/` へ。`rope_study/README.md` は道具の説明として残し、誤ったコマンド (B-3) だけ直す。
- (b) 全部再構成する (報告書を仕様・判断・結果に分け、用語を書き直す)。
- (c) 触らない (`docs/README.md` からリンクするだけ)。
- 推奨の理由: 再開するか未定なので、再構成の費用を今かけない。ただし (c) では、有効な run のフォルダ名が HANDOFF の 1 行にしか無いまま (B-2) で、誤ったコマンドも残る (B-3)。

### Q2-4 ruff の扱い
- **(a) 推奨: 設定を `pyproject.toml` に書き (P4-2)、全体を整形する専用のコミットを作り (P4-3)、そのあと H3 で編集したファイルを整形・lint する。**
- (b) 設定を書き、H3 で編集したファイルだけを整形する (全体の整形はしない)。
- (c) 設定を書き、H3 は check だけ (整形しない)。
- (d) 今は何もしない。
- 推奨の理由: (b) では、1 行直しただけのファイルで触っていない行まで整形され、差分が膨らむ (BRIEF §2 の懸念)。(a) は 1 回の整形の費用だけで、以後は差分が綺麗になる。整形で挙動が変わらないことは AST の一致で機械的に確かめられる (F-2)。規則は最初は小さい集合 (E4/E7/E9/F + I) にする (F-1)。ruff は `uvx ruff@<固定の版>` で呼び、uv の環境には入れない (uv.lock を変えない)。

### Q2-5 `next_action.md` の分け方
- **(a) 推奨: 今も有効な理由を decisions へ抜き出し、コードの参照 (48 行・16 ファイルのコメントと docstring) を書き換えてから、残りを `docs/history/` へ移す** (§2.4)。
- (b) 丸ごと `docs/history/` へ移し、コードの参照を `docs/history/next_action.md §X` に書き換えるだけにする。
- (c) 場所を変えず、冒頭の「現在地」だけを STATUS へ移す。
- 推奨の理由: (b) ではコードが「通常は読まない」経緯の文書を指し続け、今も有効な理由が経緯に埋もれたまま (B-104)。(c) は A-5 (名前と中身が違う、最新が下に埋もれる) を解かない。

### Q2-6 既存の結果の報告書の置き場所
- **(a) 推奨: `git mv` で `docs/experiments/` に集める** (`Result_report_v3.md`、`Result_Report_v6.md`、`Report_rope_models.md`、`Report_Search.md`)。本文は変えない。
- (b) 置き場所は変えず、`docs/experiments/README.md` の一覧から指すだけにする。
- 推奨の理由: 結果の正本が 1 つのフォルダにあれば、新しい session は一覧 1 本を読めば足りる。リンクとコードの参照 (`Report_rope_models.md` は .py から 7 行: A §6.2) は機械的に直せて、T6 で確かめられる。(b) は移動の費用が無い代わりに、結果の置き場所の規則が「新しいものは docs、古いものは各所」の 2 つになる。

### Q2-7 `paper/Discussion.md` (1078 行の研究方針の提案) の扱い
- **(a) 推奨: `paper/` に残し、冒頭に「2026-10-01 時点の提案 (仮説の文書)」と書く。** 生きている候補は STATUS から指す。
- (b) `docs/history/` へ移して凍結する。
- (c) `docs/proposals/` を新しく作る。
- 推奨の理由: PDF と図解 HTML への相対リンクを持つ (N-10)。中身は仮説で、まだ一部しか実施されていない (B-15) ので、経緯ではない。(c) は 1 本のためにフォルダを作ることになる。

### Q2-8 `.venv` の扱い
- **(a) 推奨: 触らない。** 文書のテストのコマンドを `uv run python -m pytest` に揃える (このリポジトリの venv の python で動く: PROGRESS 0-4)。
- (b) `.venv` を作り直す。
- 推奨の理由: `.venv/bin` の 74 本中 57 本の shebang が別のリポジトリ (`isaac_so_arm101`) の venv を指すが、`python -m` なら問題が出ない。作り直すと Isaac Lab などの版が変わりうり、学習の環境に響く。作り直すかは利用者の判断 (F-6)。

### Q2-9 rope study の発散の数値の疑い (D-3)
- **(a) 推奨: 今は確かめず、報告書の頭書きに「`bench_physics.py:378` の数え方により過大の可能性 (未確認)」と書く。** 確かめるのは rope study を再開するとき (P4-10)。
- (b) Phase 4 で Isaac Sim を使って確かめる (GPU の承認が要る)。
- (c) 何もしない。
- 推奨の理由: 打ち切り中の研究の数値で、今の作業の判断には使わない。ただし (c) では、再開したときに疑いを知らないまま数値を使う危険がある。

### Q2-10 cold-start テストの天井 (上限を付けても 13/14: `audit/G.md` の G-4)
- **(a) 推奨: これ以上の基準値は取らない。** Phase 5 では、基準値と同じ 2 つの条件 (上限なし・上限 15 回) で測る。差を見る主な指標は、読んだ量 (行・tokens、CLAUDE.md と SessionStart の出力を含む)、食い違いの指摘 (1・6・7 問)、未公開の質問の成績にする。
- (b) 上限をさらに絞った試行 (例: 8 回) を足す。費用は回答 2 本と採点 1 本。
- (c) 難しい質問を足す。
- 推奨の理由: 上限 15 回でも、答えは質問の語での grep と少数のハブの文書で見つかった (G-4)。上限をさらに絞ると、文書の良し悪しより grep の当たり外れを測ることになる。(c) は質問に合わせて文書を作る危険が増える (BRIEF §11)。未公開の質問 (Phase 5) が、この偏りへの本来の歯止めになる。

### 進め方についての確認 (選択ではなく報告)
- Phase 3 は §2〜§4 のうち承認された項目を、BRIEF §4 Phase 3 の順 (glossary → system/decisions/experiments/history → STATUS → CLAUDE.md と rules → skills/agents → hooks と settings) で行う。permissions と hooks (§4.6・§4.7) は、この提案の承認とは別に、有効にする前にもう一度確認をいただく (BRIEF §3)。
- H3 (ruff の hook) は、Q2-4 が (a) なら P4-3 (全体の整形) の後に有効にする。
- Phase 5 の cold-start テストは、基準値と同じ 2 つの条件 (上限なし・上限 15 回) で測る (依頼文は `audit/G_prompts.md`)。Phase 5 の開始時に、未公開の質問 3〜5 問をお願いする。

---

## 8. 自己点検 (BRIEF §11)

| 失敗 | この提案での確かめ | 結果 |
|---|---|---|
| 根拠のない skill・規則・文書を作る | §4.1 の全項目と §4.4 の全 skill に、D・E の番号を付けた。根拠が「損害が大きい」だけのもの (H4) は、そう明記した。根拠の無い候補 (長時間ジョブの skill、文書の監査役、Stop の hook、topology の rules、`docs/system/layout.md`) は作らないと書いた | 付けられないものは無い |
| 数値・条件を落とす | この段階では文書を書き換えていない。Phase 3 では §2.5 と BRIEF の抜き出しと照合を行う。凍結した文書は本文を書き換えない (§1) | — |
| subagent の報告を確かめずに使う | 使った所見は Phase 1 で根拠を確かめた監査のファイルから引いた (各ファイルの末尾の「メインによる根拠の確認」)。新しく使った数 (§3.3 の参照の数、R-44・`generate_assets.py` の引数) は自分で `git grep` と grep を走らせた | — |
| 前提が違うのに計画どおりに進める | HEAD が 6f41ef1 のままであることを確かめた。cold-start の天井 (G-1) は Phase 1 のゲートで報告し、上限付きの試行 (G-5) を足した | — |
| 実行していない検証を「確認済み」と書く | `env` が子プロセスに渡ること、rules の `paths` が効くことは [未確認] と書き、Phase 3 で確かめる手順を書いた | — |
| 承認されていない判断を承認済みと思い込む | §7 の 10 問は未承認として並べた。PROGRESS.md の「承認済みの判断」に無いものは使っていない | — |
| 文書を cold-start の質問に合わせて最適化する | 提案は監査の所見 (A〜F) から組み立て、質問の正答 (`coldstart_answers`) は設計に使っていない。段の記述の修正 (C-1) は質問 1 と重なるが、C-1 は質問より前に見つかった所見 | 未公開の質問で Phase 5 に確かめる |
