# F. コード構造の監査

- 対象: commit `6f41ef1` (branch `refactor/claude-foundation`)。`git ls-files` の 167 ファイル (`.py` 124、`.md` 23、その他 20)。
- 調査日: 2026-10-03。挙動を変える変更はしていない。書いたのはこのファイルだけ。
- 実行したもの: `ast` で静的に解析するスクリプト (`python3`、リポジトリのコードは import しない)、pytest を 1 回、`uvx ruff` (check / format --check / format --diff)。Isaac Sim・GPU は使っていない。
- 略記: `KT/` = `src/isaac_twisted_rl/tasks/knot_tying/`、`SC/` = `src/isaac_twisted_rl/scripts/`。行番号はすべて `6f41ef1` のもの。

## 0. 要点

| # | 結論 | 主な根拠 |
|---|---|---|
| 1 | README の 4 つの不変条件は、今すべて守られている (静的検査で違反 0)。ただし、守る仕組み (テスト・lint) はない | §6 のスクリプトの出力 `RESULT I1 violations=0 I2 violations=0` |
| 2 | テストは 268 passed (48.6 s)。ただし `uv run pytest` は別リポジトリの venv で走った。`.venv/bin/pytest` の shebang が `isaac_so_arm101` を指している | `.venv/bin/pytest:1`、pytest の警告に出たパス `/home/itoyama/work/isaac_so_arm101/.venv/.../torch/onnx/utils.py:185` |
| 3 | lint の基準が決まっていない。README は「`.pre-commit-config.yaml` の ruff v0.14.10」と書くが、そのファイルは存在せず、git の履歴にもない。`uvx ruff` (0.16.10、既定の 413 規則) で 247 件。0.14.10 (既定の 59 規則) では 84 件 (全部 E402) | `KT/README.md:240`、`git log --all -- .pre-commit-config.yaml` (出力なし)、§8 |
| 4 | `--fix` で直せる 177 件のうち、そのまま当てると挙動が変わりうるものがある。I001 (起動スクリプト 17 本の import 順)、UP018 (Warp カーネル内の `float(0.0)`)。手で直す方では、SIM118 を直すと壊れる (`InteractiveScene` に `__contains__` がない) | §8.2 |
| 5 | 未整形は 34 ファイル (py 30 + md 4)。うち 8 本は末尾の改行だけ | §8.4 |
| 6 | 重複の主なもの: chain 評価のループ (`eval_chain` と `eval_robust` で 77 行中 61 行が同じ)、デモ間のハンド操作の関数、リポジトリの根のパス計算 (9 箇所)、テストの `sys.path` 設定 (10 箇所) | §5 |
| 7 | 置き場所: `demos/` に「デモでないもの」がある (資産の生成、v4 以降の学習が読む初期アーカイブの生成、計測)。`rope_study/` には、タスク定義・計測の道具・シェル (絶対パスを直書き)・報告書が混在している。`topology_debug.py` の既定の出力先は src の中 (`KT/outputs/`) | §3 |
| 8 | 古い参照: 改名済みの `paper/Report.md` を指す行が 24 ファイルに 29 行ある。存在しないファイル名の参照 (`eval_compare.py` など) もある。`__init__` の docstring の一覧も欠けている | §2、§3 |
| 9 | バージョンは凍結されていない。v4 の `env_cfg.py` を変えると、v5・v6・v7・rope_study に波及する。private な名前 (`_play_post_init` など) をまたいで import している | §5.2 |
| 10 | モジュール docstring が欠けているのは 1 件だけ (`SC/rsl_rl/cli_args.py`) | §4 |

---

## 1. 大きなファイル (`.py` の行数の上位 15 件)

コマンド: `git ls-files '*.py' | xargs wc -l | sort -rn | head -16`。構造は `ast` で取った (付録 A-3)。

| 順位 | 行数 | ファイル | 役割 (docstring から) | 分割の候補 (切れ目) |
|---|---|---|---|---|
| 1 | 1093 | `KT/common/rope_backends/xpbd_rod.py` | Warp で書いた XPBD の紐ソルバ (rope_study 用。isaaclab 非依存) | 3 分割できる: `:50` 設定 (`XpbdRodParams` `:54-185`)、`:188` Warp カーネル (`:192-787`)、`:790` ソルバ本体 (`XpbdRod` `:793-1093`)。分けるなら、試験のローダ `tests/test_rope_backends.py:23-35` も直す |
| 2 | 893 | `KT/demos/overhand_script.py` | ゲート 1 (台本で固結び)。初期アーカイブも生成する (`--dump_archive`) | `:163` 尺と `ScriptParams` (`:184-252`)、`:255` 1 手の指示 (`MovePlan`/`ScriptRunner` `:257-485`)、`:488` 台本 (`:489-600`)、`:603` 行動への写像 (`:604-649`)、`:652` 出力 (`:653-712`)、`main` (`:715-888`) |
| 3 | 739 | `KT/demos/topology_debug.py` | 台本で操作しながら、p-data の抽出を目で確かめるデモ | `main` が 350 行 (`:385-734`) で、内側に関数を持つ (`write_gripper_state` `:468`、`rope_tangent_yaw` `:488`)。それらを外に出す (§5 D-3/D-4) |
| 4 | 739 | `KT/common/mdp/actions.py` | `PickPlaceAction` (pick&place の 1 手を 1 行動にする ActionTerm) | クラス 1 つで 629 行 (`:77-705`)。実行 (`:311-562`)、行動開始時の位相の控え (`:220-268`, `:687-705`)、診断 (`:281-308`, `:640-675`) が混在。**分割は勧めない** (N-4) |
| 5 | 630 | `tests/test_knot_detect.py` | `knot_detect` の判定・ラダー・特徴量の試験 | `:44-167` の紐の生成関数を、他の試験 3 本が import している (§5 D-12)。共通モジュールへ出す |
| 6 | 624 | `KT/common/topology/representation.py` | 半辺構造 `AbstractState` と p-data (twisted_rl からの移植) | `AbstractState` `:156-624`。**分割しない** (N-3) |
| 7 | 613 | `tests/test_knot_tying_topology.py` | 点列から p-data へ: ゴールデン照合・不変性・バッチ | `:51` ゴールデンデータ (`:54-170`) をデータとして分けられる。効果は小さい |
| 8 | 612 | `KT/common/rope_model/rope_specs.py` | ロープのプリセット (dataclass) と CLI の補助 | `RopeSpec` `:169-326`、`_make_physical_spec` `:329-407`、`:410` プリセット定義 (`:410-585`)、CLI (`:588-612`)。デモが `sys.path` 経由でこのファイル単体を import している (D-15) ので、分けると影響が広い |
| 9 | 610 | `SC/rsl_rl/eval_search.py` | 推論時探索の評価 (clone_test / rollouts / search) | `:226` clone_test、`:338` rollouts、`:419` search、`:546` main。モードごとの関数を、Kit の起動後に import するモジュールへ移せる |
| 10 | 571 | `KT/common/topology/knot_graph.py` | 位相グラフ・BFS 距離・最短の次の辺 | `:93` p-data の解析、`:127` 列挙、`:279` 最短の次の辺 (v6、`:282-462`)、`:465` キー、`:513` 実行時の表。`:279-462` を別モジュールにできる (試験あり) |
| 11 | 539 | `KT/common/beam_search.py` | ビーム探索の純 torch 部分 | `:190` `NodeStore`、`:272` 探索本体。分ける必要は薄い |
| 12 | 490 | `KT/demos/multi_rope_catch.py` | 全プリセットを並べて、掴み比べるデモ | `RopeLane` `:140-334`、`main` `:359-485`。D-3/D-4 の関数を共通化する |
| 13 | 489 | `KT/common/mdp/archive.py` | アーカイブからのリセット (v3) と、段ごとの版 (v4) | `:228` で v3 (`:57-225`) と v4 (`:228-489`) に分かれる。v4 の部分を別ファイルにして、`mdp` の名前は再 export で保つ |
| 14 | 469 | `KT/common/mdp/rewards.py` | 報酬項 14 個 | 小さい関数の集まり。分割は不要 |
| 15 | 451 | `KT/demos/divergence_bench.py` | 1 手あたりの発散の計測 | `summarize` `:236-324`、`_run` `:354-446`。距離の関数は D-5 (centerline と重複) |

---

## 2. 役割が曖昧なモジュール

| # | モジュール | 何が曖昧か | 根拠 |
|---|---|---|---|
| R-1 | `KT/demos/` | docstring は「RL 環境からは import されない独立したデモ」。実際には、学習の前に必須の資産の生成 (`generate_assets.py`)、v4 以降の学習が読む初期アーカイブの生成元 (`overhand_script.py`)、計測 (`divergence_bench.py`) を含む。docstring の一覧には後の 2 本がない | `KT/demos/__init__.py:1-9`、`KT/versions/v4/env_cfg.py:53` (`SEED_ARCHIVE`) ← `KT/demos/overhand_script.py:117`、`KT/rope_study/run_physics_suite.sh:26` |
| R-2 | `KT/common/mdp/observations.py` | 観測項に加えて、位相の抽出器を env ごとにキャッシュする共有の仕組みを持つ (`get_extractor`/`invalidate_extractor`)。mdp の他の 8 モジュールが使う | `KT/common/mdp/observations.py:39,58`。使う側: `rewards.py:37`, `terminations.py:23`, `archive.py:50`, `stages.py:47`, `topology_log.py:53`, `state_clone.py:40`, `events.py:86`, `actions.py:497,689` |
| R-3 | `KT/common/mdp/actions.py` | 1 つのクラスが、行動の実行・行動開始時の位相の控え (報酬などが読む)・診断を兼ねる | `:220-268`, `:281-308`, `:640-705`。位相の控えを読む側: `rewards.py` (7 箇所)、`stages.py` (2)、`topology_log.py` (1) |
| R-4 | `KT/common/mdp/state_clone.py` | MDP 項ではない。推論時探索のための状態の複製。使うのは eval_search だけ | `SC/rsl_rl/eval_search.py:105` |
| R-5 | `KT/common/mdp/finger_geometry.py` | 純 torch の幾何が `mdp/` にある。試験から単体で読むため、import を 2 通りに書いている | `KT/common/mdp/finger_geometry.py:11-14` |
| R-6 | `KT/common/beam_search.py` | 評価用の探索アルゴリズムが、タスクの共有層にある。使うのは eval_search だけ | `SC/rsl_rl/eval_search.py:103` |
| R-7 | `KT/common/rope_backends/` | 「バックエンド」の中に ActionTerm と環境クラスがある | `KT/common/rope_backends/noisy_action.py:17`、`KT/common/rope_backends/env.py:27` |
| R-8 | `KT/rope_study/` | 中身は 5 種類: タスクの登録、設定 (`env_cfg.py`、`agents/`、`models.py`)、計測 (`bench_*.py`、`elastica.py`)、評価のジョブ (`eval_robust.py`、`make_eval_jobs.py`、`summarize_*.py`)、シェル 4 本、文書 3 本 | `KT/rope_study/__init__.py:18-28`、§3 |
| R-9 | `KT/common/rope_model/rope_specs.py` | プリセットの定義に、CLI の補助と `__main__` が混在 | `:598` (`add_rope_arg`)、`:610` |
| R-10 | `SC/rsl_rl/` | Isaac Lab のテンプレート (train / play / cli_args) と、このリポジトリ固有の eval_chain / eval_search / summarize_search が同居。`__init__.py` がない (namespace package) | `SC/rsl_rl/train.py:1-4` (Isaac Lab の著作権表示)、`find src -type d` の結果 |
| R-11 | `src/isaac_twisted_rl/__init__.py` | docstring がテンプレートのまま ("project/extension template") | `src/isaac_twisted_rl/__init__.py:1-3` |
| R-12 | `__init__` の docstring の一覧が古い | `common/__init__.py` の一覧に `policies/`、`rope_backends/`、`state_archive.py`、`beam_search.py` がない。`versions/__init__.py` に v7 がない。`rope_model/__init__.py` は存在しない `assets/` と `demo_spawn_rope.py` を挙げ、「4 プリセット」と書く (実際は 6: `rope_specs.py:579-581`)。`demos/__init__.py:3` は「ロープ 4 種」と書く | `KT/common/__init__.py:5-11`、`KT/versions/__init__.py:6-13`、`KT/common/rope_model/__init__.py:3,12,15` |
| R-13 | 存在しないファイルへの参照 | `eval_compare.py` (`SC/rsl_rl/eval_chain.py:8`)、`knot_tying/generate_assets.py` (`KT/common/parallel_gripper/generate_gripper_usd.py:8`、`KT/common/rope_model/generate_rope_usd.py:15`。実物は `demos/` の下)、`rope_model/demo_spawn_rope.py` (`KT/demos/spawn_rope.py:12`) | `git ls-files` に無い |

---

## 3. 置き場所が不自然なもの

### 3.1 指定された確認項目

| 項目 | 結果 | 根拠 |
|---|---|---|
| `knot_tying/outputs/` | git では追跡していない。ディスク上には空の `KT/outputs/topology_debug/` がある (`.gitignore` で無視)。作ったのは `topology_debug.py` の既定の出力先。ほかのスクリプトは、リポジトリの根の `outputs/` に書く | `KT/demos/topology_debug.py:75`、`.gitignore:33`、`git check-ignore -v`。根に書く例: `KT/demos/overhand_script.py:717`、`KT/demos/divergence_bench.py:330`、`KT/rope_study/bench_physics.py:412` |
| `rope_study/` の混在 | シェル 4 本・文書 3 本 (README 112 行、HANDOFF 439 行、Report 880 行)・Python 15 本 (agents を含む) が同じ場所にある。シェルはどれも `ROOT=/home/itoyama/work/isaac_twisted_rl` を直書きし、`.venv/bin/python` を直接呼ぶ | `KT/rope_study/run_eval_loop.sh:13`、`run_eval_queue.sh:12`、`run_physics_suite.sh:16`、`run_train_chain.sh:13` |
| `versions/vN/` の中の HTML | `v3/HTML/` に 2 本 (91 KB と 58 KB) だけある。コードへの相対リンク (`../env_cfg.py` など) と、`paper/Report.md` へのリンク (改名で切れている) を持つ。動かすとリンクが切れる | `KT/versions/v3/HTML/v3-implementation.html:241`、`v3-vs-papers.html:173` |
| `paper/` | 中身は 3 種類: 第三者の論文 PDF 3 本 (計 9.3 MB)、論文の解説 HTML 4 本、自分の報告 MD 3 本 (`Discussion.md` 1078 行、`Report_Paper.md` 466 行、`Report_Search.md` 304 行)。`Report_Search.md` は eval_search の中間レポート (commit `72d18df`)。報告書の置き場所は 3 通りに分かれている (`paper/`、`rope_study/`、`versions/vN/`) | `git show --stat 72d18df`、`b53f070` |
| 報告書の名前 | `v3/Result_report_v3.md` と `v6/Result_Report_v6.md` で、大文字小文字が揃っていない | `git ls-files` |

### 3.2 追跡している、Python でも Markdown でもないファイル (20 件、全部)

コマンド: `git ls-files | grep -vE '\.(py|md)$'`

| ファイル | 大きさ | 判定 |
|---|---|---|
| `.gitignore` | 52 行 | 妥当。`**/outputs/*` と `logs/` を無視する |
| `CITATION.cff` | 20 行 | テンプレート由来。著者はテンプレートの作者 (Louis Le Lay, Muammer Bay)、`date-released: 2025-04-18`。変えるかは利用者の判断 |
| `LICENSE` | 28 行 | テンプレート由来の BSD-3。帰属の表示は残す必要がある |
| `pyproject.toml` | 41 行 | `authors` がテンプレートの作者。`[tool.ruff]` がない (F-1) |
| `uv.lock` | 2896 行 | 妥当 |
| `paper/Twisted-RL.pdf` / `paper/Twisted.pdf` / `paper/Learning_Topological_Motion_Primitives_for_Knot-Plannning.pdf` | 6.3 MB / 0.56 MB / 2.4 MB | 第三者の論文。再配布してよいかは未確認 |
| `paper/explainer/{index,twisted,twisted-rl,yan2020}.html` | 57〜81 KB | 論文の解説。`index.html:323` と `twisted-rl.html:787` から `../Report.md` へのリンクが切れている |
| `KT/versions/v3/HTML/v3-implementation.html`, `v3-vs-papers.html` | 91 KB, 58 KB | §3.1 |
| `KT/rope_study/run_eval_loop.sh`, `run_eval_queue.sh`, `run_physics_suite.sh`, `run_train_chain.sh` | 27〜41 行 | Python のパッケージの中にある。絶対パスを直書き (§3.1) |
| `KT/common/parallel_gripper/data/.gitkeep`, `KT/common/rope_model/data/.gitkeep` | 0 B | 生成した USD の置き場所 (USD 自体は無視)。妥当 |

### 3.3 追跡していないが、構造に関係するもの (一覧を見ただけ。中は開いていない)

| もの | 何が問題か | 根拠 |
|---|---|---|
| `outputs/archive_seed/` | v4 以降の学習と rope_study の設定が、ここ (git の管理外) のファイルを読む | `KT/versions/v4/env_cfg.py:53`、`KT/rope_study/env_cfg.py:43` |
| `logs/rsl_rl/eval_search_lane.sh` | ログ置き場 (git の管理外) にシェルスクリプトがある | `ls logs/rsl_rl` |
| 容量 | `logs/` は 3.7 GB、`outputs/` は 234 MB | `du -sh logs outputs` |

---

## 4. モジュール docstring の欠落

結果: **124 本のうち 1 本だけ**。

| ファイル | 行数 | 冒頭 |
|---|---|---|
| `SC/rsl_rl/cli_args.py` | 91 | Isaac Lab の著作権コメントだけで、docstring がない |

使ったスクリプト (`python3 docstrings.py`。リポジトリのコードは import しない):

```python
"""List tracked .py files lacking a module docstring (ast.get_docstring is None)."""
import ast, subprocess
ROOT = "/home/itoyama/work/isaac_twisted_rl"
files = subprocess.check_output(["git", "ls-files", "*.py"], cwd=ROOT, text=True).split()
missing = []
for f in files:
    src = open(f"{ROOT}/{f}", encoding="utf-8").read()
    if ast.get_docstring(ast.parse(src)) is None:
        first = next((l for l in src.splitlines() if l.strip()), "")
        missing.append((f, len(src.splitlines()), first[:70]))
print(f"{len(missing)} / {len(files)} files lack a module docstring")
for f, n, first in missing:
    print(f"{f}\t{n}\t{first}")
```

補足: docstring はあっても、中身が古いものがある (R-11〜R-13)。機械的な判定では見つからない。

---

## 5. 重複した実装

検出方法: 関数とクラスの本体 (docstring は除く) を `ast.dump` で比べて完全一致を探した。ほぼ一致は、`ast.unparse` の結果を `difflib` で比べた (しきい値 0.85 と 0.72)。スクリプトは付録 A-4。ブロック単位の比較は、行の空白を除いて `difflib` で行った。

### 5.1 重複の一覧

| # | 内容 | 場所 | 一致度 | 備考 |
|---|---|---|---|---|
| D-1 | 四元数の積 | `KT/common/rope_backends/centerline.py:111-123` = `KT/common/state_archive.py:156-168` | 完全一致 | どちらも純 torch。`state_archive` は試験から単体で読まれるので (`tests/test_state_archive.py:17-20`)、相対 import で共有すると試験が壊れる |
| D-2 | `Log` クラス | `KT/demos/divergence_bench.py:220-228` = `KT/demos/overhand_script.py:704-712` | 完全一致 | |
| D-3 | `rope_tangent_yaw` | `KT/demos/rope_catch.py:275-284` = `KT/demos/topology_debug.py:488-497`、`KT/demos/multi_rope_catch.py:224-233` | 完全一致 / 0.96 | どれも `main` の中の関数 |
| D-4 | ハンドの状態の書き込み | `KT/demos/rope_catch.py:251-269`、`topology_debug.py:468-482`、`multi_rope_catch.py:197-215`、`overhand_script.py:315-324` ~ `KT/common/mdp/actions.py:564-586` | 0.84〜0.93 | |
| D-5 | 線分どうしの距離 | `KT/common/rope_backends/centerline.py:162-182` ~ `KT/demos/divergence_bench.py:95-116` | 0.99 | `centerline.py:166` の docstring に「同じ式」と書いてある |
| D-6 | chain 評価のループ | `SC/rsl_rl/eval_chain.py:68-144` ~ `KT/rope_study/eval_robust.py:87-194` | 77 行中 61 行が同じ | `_STAGE_DIMS = 4` が 3 箇所にある (`eval_chain.py:64`、`eval_search.py:108`、`eval_robust.py:68`) |
| D-7 | リポジトリの根のパス計算 | `KT/demos/divergence_bench.py:47`、`overhand_script.py:82`、`KT/rope_study/bench_cantilever_art.py:38`、`bench_physics.py:32`、`env_cfg.py:42`、`make_eval_jobs.py:29`、`summarize_eval.py:20`、`summarize_train.py:17`、`KT/versions/v4/env_cfg.py:51` | 9 箇所 | 遡る階層が 4・5・6 とファイルの深さごとに違う。動かすと値がずれる |
| D-8 | AppLauncher の定型 | 19 本 (`git grep -n 'AppLauncher('`) | — | Isaac Lab の慣用。import の順序に意味がある (N-8) |
| D-9 | `main` | `SC/random_agent.py:39-63` ~ `SC/zero_agent.py:39-63` | 0.98 | テンプレート由来 |
| D-10 | v0〜v2 の設定 | v1 の `PolicyCfg` (`KT/versions/v1/env_cfg.py:40-64`) = v2 の `PolicyCfg` (`KT/versions/v2/env_cfg.py:66-86`)。v0 と v1 の `env_cfg.py` は 121 行中 87 行が同じ | 完全一致 / 0.64 (ファイル全体) | 方針として凍結した実験 (N-2) |
| D-11 | 観測の式の写し | `KT/common/mdp/observations.py:122-187` ~ `tests/test_knot_detect.py:392-423` | 0.92 | 意図的な写し (`tests/test_knot_detect.py:379-389`。本体が isaaclab に依存するため)。本体を変えると写しとずれる |
| D-12 | 試験の準備 | `sys.path.insert` が 10 本の試験にそれぞれある。conftest はない。`test_knot_detect` の補助関数を、他の 3 本が import している (`tests/test_knot_graph.py:33`、`tests/test_next_edge.py:34`、`tests/test_r2_reduce.py:32`)。`_trefoil` は `tests/test_knot_detect.py:47-67` ~ `tests/test_knot_tying_topology.py:159-169` (0.95) | — | |
| D-13 | 把持の定数 | `KT/common/rope_task_cfg.py:36-43`、`KT/common/mdp/phases.py:36-60` と、デモ 4 本 (`rope_catch.py:119-150`、`multi_rope_catch.py:94-119`、`topology_debug.py:149-187`、`overhand_script.py:167-180`) | 値が一部違う | 例: `OPEN_RATIO` が 0.8 (`rope_catch.py:131`) と 1.0 (`phases.py`)。`CARRY_PALM_Z` が 0.15 (`multi_rope_catch.py:103`) と 0.20。統合すると挙動が変わる (N-7) |
| D-14 | プリセット名の一覧 | `KT/__init__.py:54`、`KT/common/rope_task_cfg.py:127`、`KT/common/rope_model/rope_specs.py:579-581` | 今は一致 (rope_specs だけ `knot56` を余分に持つ) | 3 箇所を手で揃えている |
| D-15 | `rope_specs` の 2 通りの読み込み | デモは `sys.path` に足して、`rope_specs` という名前で読む (`KT/demos/generate_assets.py:54`、`multi_rope_catch.py:52`、`rope_catch.py:73`、`spawn_rope.py:29`、`topology_debug.py:72`)。ライブラリ側も相対 import が失敗したら同じことをする (`KT/common/rope_model/rope_cfg.py:25-31`、`generate_rope_usd.py:80-86`) | — | 同じファイルが 2 つのモジュール名で読まれうる。実害が出るかは未確認 |
| D-16 | 表の整形の補助 | `pct`/`_pct`: `KT/rope_study/summarize_eval.py:38`、`SC/rsl_rl/summarize_search.py:31`、`KT/demos/divergence_bench.py:231` | 似ている | 小さい |

### 5.2 バージョン間の継承 (設定のコピーではなく、import の連鎖)

| 版 | 継承元 (`env_cfg.py`) | private な名前の import | 根拠 |
|---|---|---|---|
| v0 / v1 / v2 | `common` だけ (互いにコピー) | — | `KT/versions/v0/env_cfg.py:24-26` ほか |
| v2_r2 | v2 | — | `KT/versions/v2_r2/env_cfg.py:30-32` |
| v3 | v2_r2, v2 | `_DIVERGED_TERM`, `_PLAY_PRINT_ENVS` | `KT/versions/v3/env_cfg.py:38-42` |
| v4 | v3, v2_r2 | `_DIVERGED_TERM`, `_PLAY_PRINT_ENVS` | `KT/versions/v4/env_cfg.py:38-44` |
| v5 | v4 | `_play_post_init` | `KT/versions/v5/env_cfg.py:28-30` |
| v6 | v5, v2_r2 | `_DIVERGED_TERM`, `_play_post_init` | `KT/versions/v6/env_cfg.py:32-37` |
| v7 | v6, v5 | `_play_post_init` | `KT/versions/v7/env_cfg.py:22-23` |
| rope_study | v4 | `_play_post_init` | `KT/rope_study/env_cfg.py:39` |

PPO の設定も同じように連鎖している: v2_r2←v2、v3←v2_r2、v4←v3、v5←v4、v6←v5、v7←v6、rope_study←v4 (各 `agents/rsl_rl_ppo_cfg.py` の import)。**v3 や v4 を書き換えると、後の版がすべて変わる。** 環境設定のクラス名は、`globals().update(make_preset_cfgs(...))` で動的に作っている (`KT/versions/v0/env_cfg.py:121` ほか、rope_study は `KT/rope_study/env_cfg.py:232`)。そのため、静的な解析では参照を追えない。

---

## 6. 不変条件の現状

### 6.1 まとめ

| 不変条件 | 現状 | 根拠 |
|---|---|---|
| I1: `import_packages` が実行する `__init__.py` で isaaclab.sim などを import しない | **守られている**。自動で import される 29 本 (`tasks/__init__` を含む) のうち、何かを import するのは 4 本だけ。どれも、モジュールの最上位で推移的にたどっても isaaclab・omni・pxr に届かない | 付録 A-1 の出力 |
| I2: `common/topology/` と `common/state_archive.py` は isaaclab・omni・pxr を import しない | **守られている** (関数の中の import も含めて 0 件)。外部の依存は torch と標準ライブラリだけ | 付録 A-1、下の grep |
| I3: `mdp` というフォルダ名 (ブラックリストの部分一致で除外) | **守られている**。`mdp` を含むディレクトリは `KT/common/mdp` だけ。ブラックリストは `["utils", ".mdp"]` | `src/isaac_twisted_rl/tasks/__init__.py:15`。部分一致の判定は `.venv/lib/python3.11/site-packages/isaaclab/source/isaaclab_tasks/isaaclab_tasks/utils/importer.py:73` |
| I4: `common/mdp/` の項は追加だけ | **守られている**。全 34 コミットで、`common/mdp/*.py` に定義された def/class の名前は 58 個。消えたものは 0。コード中の `mdp.X` の参照 (40 種) はすべて解決する (自前 37、isaaclab 3)。isaaclab の `envs.mdp` との名前の衝突もない | 付録 A-2 の出力 |

### 6.2 各 `__init__.py` の import (31 本)

| ファイル | 最上位の import | 推移的に届く外部パッケージ |
|---|---|---|
| `src/isaac_twisted_rl/__init__.py` | なし | — |
| `src/isaac_twisted_rl/tasks/__init__.py` | `isaaclab_tasks.utils.import_packages` (`:12`) | 仕組みそのもの |
| `KT/__init__.py` | `gymnasium` (`:36`) | gymnasium |
| `KT/common/topology/__init__.py` | 下位の 7 モジュール (`:42-105`) | torch と標準ライブラリ |
| `KT/rope_study/__init__.py` | `gymnasium` (`:12`)、`.models` (`:14`) | gymnasium、dataclasses |
| `KT/common/mdp/__init__.py` | `isaaclab.envs.mdp` の star import (`:12`) と下位の 9 モジュール | **ブラックリストで除外されるので対象外** |
| 残りの 25 本 (`common/`、`parallel_gripper/`、`policies/`、`rope_backends/`、`rope_model/`、`demos/`、`rope_study/agents/`、`versions/` と `versions/v*/`、`versions/v*/agents/`) | なし (docstring だけ) | — |

注意点:
- `import_packages` が import するのはパッケージだけ (`importer.py:79-81` の `info.ispkg`)。普通のモジュール (デモなど) は自動では import されない。
- ブラックリストは部分一致なので、名前に `utils` を含むパッケージや、`.mdp` で始まるパッケージ (例: `mdp_v2`) を新しく作ると、**黙って登録から外れる** (`importer.py:73`)。
- README は「Kit 起動前に実行される」と書く (`KT/README.md:214-217`)。しかし、リポジトリの中のスクリプトはどれも、AppLauncher の後で `isaac_twisted_rl.tasks` を import している (`SC/list_envs.py:21`→`:29`、`SC/rsl_rl/train.py:48`→`:96`、`SC/rsl_rl/play.py:51`→`:77`)。README の言う「Kit 起動前」がどの経路のことかは未確認。不変条件そのものは守られているので、守り続ける前提で扱う。

### 6.3 確かめ方 (Phase 3 のテストの材料)

```bash
# I1 / I2: ast による静的検査。リポジトリのコードは実行しない (付録 A-1)
python3 invariants.py            # 最後の行が "RESULT I1 violations=0  I2 violations=0"

# I2 の簡易版: 出力が空なら OK (終了コード 1)
git grep -nE '^\s*(import|from)\s+(isaaclab|omni|pxr|isaacsim|carb)' -- \
  'src/isaac_twisted_rl/tasks/knot_tying/common/topology/*.py' \
  'src/isaac_twisted_rl/tasks/knot_tying/common/state_archive.py'

# I3
find src -type d -name '*mdp*' -not -path '*/__pycache__*'   # -> KT/common/mdp だけ
grep -n '_BLACKLIST_PKGS' src/isaac_twisted_rl/tasks/__init__.py

# I4: 付録 A-2 (git の全コミットについて、mdp の def/class 名が減っていないか。mdp.X の参照が解決するか)
python3 mdp_history.py && python3 mdp_refs.py
```

pytest にするときの案 (どれも Isaac Sim なしで動く):

| 試験 | 中身 |
|---|---|
| `test_init_imports` | 自動で import される `__init__.py` について、最上位の import を推移的にたどり、許可した集合 (`gymnasium`, `torch`, `dataclasses` と標準ライブラリ) に収まることを確かめる。`tasks/__init__.py` の `isaaclab_tasks.utils` だけは例外にする |
| `test_pure_layers` | `topology/**`、`state_archive.py` (README の約束) に加えて、試験が単体で読んでいる純粋なモジュール (`beam_search.py`、`mdp/finger_geometry.py`、`mdp/phases.py`、`rope_backends/centerline.py`、`rope_study/models.py`、`rope_study/elastica.py`) も、isaaclab・omni・pxr を import しないことを確かめる |
| `test_blacklist` | `mdp` という名前のディレクトリが `common/mdp` だけであること。ブラックリストに `".mdp"` があること。ほかのパッケージの名前が `utils` も `.mdp` も含まないこと |
| `test_mdp_names_additive` | `common/mdp/*.py` の def/class の名前を、記録した一覧 (今は 58 個) と比べる。**減っていたら失敗**。増えたら一覧を更新する。star import で漏れる付随の名前 (`observations.py`、`rewards.py`、`terminations.py` には `__all__` がない) は対象にしない |
| `test_registry_static` | `_VERSIONS` (`KT/__init__.py:40`) と `versions/` のディレクトリが対応していること。`_ROPE_PRESETS` (`:54`) と `_PRESET_DOCS` (`KT/common/rope_task_cfg.py:127`) のキーが一致すること。`_AGENT_VARIANTS` の Runner クラスが各 `agents/rsl_rl_ppo_cfg.py` にあること。登録される ID の数は、静的に数えると 150 (`KT/__init__.py`) + 16 (rope_study、8 モデル x 2) = 166 (実行して数えてはいない) |

---

## 7. テストの現状

実行したコマンド (1 回だけ): `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 timeout -k 10 600 uv run pytest tests/ -q`

結果: **268 passed, 1 warning、48.64 s** (wall 50.0 s)。終了コード 0。skip は 0。警告は `tests/test_stage_gated_policy.py::test_export_jit_and_onnx` の torch.onnx の DeprecationWarning。

**注意 (F-6):** 警告に出たパスは `/home/itoyama/work/isaac_so_arm101/.venv/lib/python3.11/site-packages/torch/onnx/utils.py:185` だった。`.venv/bin/pytest` の 1 行目は `#!/home/itoyama/work/isaac_so_arm101/.venv/bin/python3` で、`.venv/bin` の 74 本のうち 57 本が同じ shebang を持つ。一方で `.venv/bin/python` は uv の cpython を指し、`train`/`play`/`list_envs` の shebang はこのリポジトリの `.venv` を指す。つまり **`uv run pytest` は別リポジトリの venv の torch と rsl_rl で走った**。このリポジトリの venv で回した結果 (例: `uv run python -m pytest`) は未確認。

| ファイル | 試験関数 (parametrize) | 何を検査しているか |
|---|---|---|
| `tests/test_beam_search.py` | 17 (1) | `beam_search` の純関数 (classify、frontier の選択、スロットの割り当て、再開の重み、成功率曲線・best-of-N) と、`NodeStore`・`BeamSearch` のラウンドの進め方 |
| `tests/test_finger_geometry.py` | 7 (0) | 指の下の紐を探す幾何 (交差・並走・ヨー・NaN・床の上の紐) |
| `tests/test_knot_detect.py` | 35 (7) | 固結びの判定 (gap と符号の両方の条件)、棄却、ラダーの段、行の順序・バッチに依らないこと、交差の特徴量の正規化、報酬ラダー、8 スロットでの打ち切り |
| `tests/test_knot_graph.py` | 19 (3) | 位相グラフの列挙数 (交差 3 個で 528、4 個で 7968)、距離の表、`batch_keys` の一致、`BfsPotential` がラダーより下がらないこと |
| `tests/test_knot_tying_topology.py` | 37 (7) | 点列から p-data への変換: twisted_rl とのゴールデン照合、平行移動・回転・拡大での不変性、バッチ、交点の順序、ポリラインの前処理、半辺構造 |
| `tests/test_next_edge.py` | 17 (2) | 最短の次の辺の本数と台本が通る辺、リンクマスクの境界、表に無い状態、キーで引いた行の一致 |
| `tests/test_r2_reduce.py` | 10 (4) | R2 除去 (O/U の全組み合わせ)、v2 の誤判定の回帰 |
| `tests/test_rope_backends.py` | 13 (2) | centerline (仮想リンク・再標本化・枠・めり込み) と XPBD (Warp を CPU で: 静止・把持・自己接触・開いた指・瞬間移動・摩擦・アダプタの座標)、heavy elastica |
| `tests/test_stage_gated_policy.py` | 9 (0) | 段ごとの方策のヘッドの選択・段ごとの std・勾配が他の段に流れないこと・JIT/ONNX への書き出し・rsl_rl への登録 |
| `tests/test_state_archive.py` | 13 (0) | アーカイブのリングバッファ・重み付きの抽出・重心の回転・保存と読み込み |

試験関数は静的に数えて計 177 個。parametrize を展開すると 268 件。

試験がないもの: isaaclab に依存するコード全部。具体的には、`mdp/` の項 (`finger_geometry`・`phases` を除く)、`env_cfg`、gym への登録、`rope_state.py`、`rope_backends/adapters.py`・`env.py`、`state_clone.py`、`scripts/`、`demos/`、`rope_study/` (`elastica.py` を除く)。CI の設定もない (`git ls-files` に無い)。

---

## 8. lint の現状

### 8.1 前提

| 項目 | 事実 | 根拠 |
|---|---|---|
| ruff の設定ファイル | なし。`--show-settings` に設定ファイルの行がない | `uvx ruff check --show-settings ...` |
| README の約束 | 「`.pre-commit-config.yaml` の ruff v0.14.10、line-length 120」と書くが、そのファイルは存在せず、git の履歴にもない | `KT/README.md:240-241`、`git log --all -- .pre-commit-config.yaml` (出力なし) |
| `uvx ruff` の版 | 0.16.10。既定で有効な規則は 413 個 (E402 を含まない) | `uvx ruff --version`、`--show-settings` の `linter.rules.enabled` |
| 参考: ruff 0.14.10 | 既定で有効な規則は 59 個。check は **84 件で、全部 E402** (`SC/` の 7 本: train 21、play 18、eval_search 18、eval_chain 11、zero_agent 6、random_agent 6、list_envs 4)。format は 30 ファイルが未整形 | `uvx ruff@0.14.10 check --line-length 120 --statistics .` |
| 参考: E501 | `--select E501` で 71 件。日本語は 1 文字を幅 2 で数える | `uvx ruff check --line-length 120 --select E501 --statistics .` |

**ruff の版によって規則の集合が大きく違う。** 0.16.10 の RUF100 (128 件) のうち 113 件は「無効な規則 E402 の noqa」だが、0.14.10 では E402 が有効なので、その noqa は必要になる。

### 8.2 規則ごとの件数 (`uvx ruff check --line-length 120 --statistics .`。計 247 件、`[*]` の 177 件は `--fix` で直せる)

| 規則 | 件数 | `--fix` | 分類 | 根拠・場所 |
|---|---|---|---|---|
| RUF100 unused-noqa | 128 | ○ | 挙動は変わらない。ただし**規則の集合を決める前に消してはいけない** (113 件は E402 の noqa。ほかに E731 が 5、N802 が 4、F401/F403 が 4、A002 が 1、C901 が 1) | §8.1 |
| I001 unsorted-imports | 32 | ○ | **挙動が変わりうる** (起動スクリプト 17 本)。そのほかは低リスク (ライブラリ 14 本、試験 1 本) | 例: `SC/rsl_rl/eval_chain.py:47` を直すと、`import isaaclab_tasks` (副作用のある import) が torch や `isaaclab.envs` より前に移る (`--diff` で確認)。今の並びは Isaac Lab 流の区切り (rsl_rl / isaaclab / isaaclab_rl / isaaclab_tasks) |
| B008 function-call-in-default-argument | 34 | × | **変えない** (Isaac Lab の慣用 `SceneEntityCfg("rope")`)。マネージャが解決するのは `params` に入った `SceneEntityCfg` だけ | `.venv/.../isaaclab/managers/manager_base.py:394`。場所: `mdp/observations.py` 10、`rewards.py` 10、`stages.py` 5、`archive.py` 4、`terminations.py` 3、`topology_log.py` 2 |
| RUF046 unnecessary-cast-to-int | 17 | × | Python の float なら挙動は変わらない (`round()`・`math.ceil`・`math.floor` は int を返す)。1 箇所ずつ型を確かめる | 例: `KT/common/mdp/phases.py:188`、`KT/common/rope_backends/xpbd_rod.py:856,857,865` |
| PLR0402 manual-from-import | 9 | ○ | 低い (束縛されるオブジェクトは同じ) | `SC/rsl_rl/train.py:16,53`、`KT/common/policies/stage_gated.py:48,167` ほか |
| UP033 lru-cache-with-maxsize-none | 4 | ○ | 変わらない (`functools.cache` は同じもの) | `KT/common/topology/knot_graph.py:191,219,239,377` |
| RUF007 zip-instead-of-pairwise | 4 | × | 変わらない | `KT/demos/topology_debug.py:628`、試験 3 箇所 |
| UP018 native-literals | 2 | ○ | **変えない**。Warp のカーネルの中の `float(0.0)` は、後で値を変える変数を宣言する書き方。リテラルに直したときに動くかは未確認 | `KT/common/rope_backends/xpbd_rod.py:400,503` |
| RUF022 unsorted-dunder-all | 2 | ○ | 変わらない (`__all__` の集合は同じ) | `KT/common/topology/__init__.py:107`、`KT/versions/v4/env_cfg.py:46` |
| SIM102 collapsible-if | 2 | × | 1 件は Warp の関数の中 (`xpbd_rod.py:693`)。もう 1 件は `representation.py:526` (移植のまま残す、N-3) | |
| B023 function-uses-loop-variable | 2 | × | 実害なし (lambda をその場で呼んでいる) | `KT/rope_study/summarize_eval.py:130` |
| B006 mutable-argument-default | 2 | × | **変わりうる** (rsl_rl に渡す引数の型と、設定の記録が変わる) | `KT/common/policies/stage_gated.py:111-112` |
| C408 / RUF059 | 2 / 2 | × | 試験だけ。変わらない | `tests/test_beam_search.py:179`、`tests/test_rope_backends.py:161,168`、`tests/test_next_edge.py:172` |
| DTZ005 call-datetime-now-without-tzinfo | 1 | × | **変わる** (学習の run のディレクトリ名に時刻を使っている) | `SC/rsl_rl/train.py:144` |
| ISC004 | 1 | × | 意図的 (表の見出しを 2 行に分けている) | `KT/rope_study/summarize_train.py:77-78` |
| SIM118 in-dict-keys | 1 | × | **直すと壊れる**。`scene` は `InteractiveScene` で、`keys()` と `__getitem__` しかなく `__contains__` がない | `KT/common/rope_backends/env.py:73`、`.venv/.../isaaclab/scene/interactive_scene.py:656,675` |
| RUF012 mutable-class-default | 1 | × | **変わりうる** (`configclass` が既定値を独自に扱う) | `KT/common/rope_backends/backend_cfg.py:33` |
| PLR1722 sys-exit-alias | 1 | × | 低い (テンプレートのコード) | `SC/rsl_rl/train.py:71` |

### 8.3 ファイルごとの件数 (`--output-format concise`。多い順、56 本)

| 件数 | ファイル | 内訳 |
|---|---|---|
| 16 | `KT/demos/topology_debug.py` | RUF100 11, RUF046 3, I001, RUF007 |
| 15 | `KT/rope_study/eval_robust.py` | RUF100 13, I001, PLR0402 |
| 12 | `KT/demos/overhand_script.py` | RUF100 10, I001, RUF046 |
| 12 | `KT/rope_study/bench_physics.py` | RUF100 10, I001, RUF046 |
| 11 | `KT/common/mdp/observations.py` | B008 10, I001 |
| 11 | `KT/common/mdp/rewards.py` | B008 10, I001 |
| 10 | `KT/demos/divergence_bench.py` | RUF100 9, I001 |
| 10 | `KT/rope_study/bench_cantilever_art.py` | RUF100 8, I001, RUF046 |
| 9 | `KT/demos/multi_rope_catch.py` | RUF100 7, I001, RUF046 |
| 9 | `KT/demos/rope_catch.py` | RUF100 7, I001, RUF046 |
| 8 | `KT/common/rope_backends/xpbd_rod.py` | RUF046 3, RUF100 2, UP018 2, SIM102 |
| 6 | `KT/common/mdp/stages.py` | B008 5, I001 |
| 6 | `KT/common/topology/representation.py` | RUF100 5, SIM102 |
| 6 | `KT/demos/parallel_gripper.py` / `KT/demos/spawn_rope.py` / `KT/rope_study/smoke_test.py` | 各 RUF100 5, I001 |
| 6 | `tests/test_rope_backends.py` | C408, I001, RUF007, RUF046, RUF059, RUF100 |
| 5 | `SC/rsl_rl/train.py` | PLR0402 2, DTZ005, I001, PLR1722 |
| 5 | `KT/common/mdp/archive.py` | B008 4, I001 |
| 5 | `KT/common/mdp/__init__.py` | RUF100 4, I001 |
| 5 | `KT/common/policies/stage_gated.py` | B006 2, PLR0402 2, I001 |
| 5 | `tests/test_next_edge.py` | RUF100 4, RUF059 |
| 4 | `KT/common/mdp/terminations.py` | B008 3, I001 |
| 4 | `KT/common/topology/knot_graph.py` | UP033 4 |
| 4 | `KT/demos/generate_assets.py` | RUF100 4 |
| 4 | `tests/test_r2_reduce.py` | RUF100 3, RUF007 |
| 3 | `SC/rsl_rl/eval_chain.py` | I001, PLR0402, RUF100 |
| 3 | `KT/common/mdp/topology_log.py` | B008 2, I001 |
| 3 | `KT/rope_study/summarize_eval.py` | B023 2, RUF100 |
| 3 | `tests/test_beam_search.py` | C408, RUF007, RUF100 |
| 3 | `tests/test_knot_graph.py` | RUF100 3 |
| 3 | `tests/test_stage_gated_policy.py` | RUF100 2, PLR0402 |
| 2 | `SC/rsl_rl/eval_search.py` / `SC/rsl_rl/play.py` | 各 I001, PLR0402 |
| 2 | `KT/rope_study/bench_cantilever_xpbd.py` | RUF046, RUF100 |
| 2 | `tests/test_finger_geometry.py` / `tests/test_knot_detect.py` | 各 RUF100 2 |
| 1 | `SC/random_agent.py`、`SC/zero_agent.py`、`KT/common/env_cfg_base.py`、`KT/common/mdp/actions.py`、`KT/common/mdp/state_clone.py`、`KT/common/rope_backends/noisy_action.py`、`KT/common/rope_state.py`、`KT/versions/v5/env_cfg.py` | 各 I001 |
| 1 | `KT/common/mdp/phases.py`、`KT/common/rope_backends/centerline.py`、`KT/common/rope_model/rope_specs.py`、`KT/rope_study/elastica.py` | 各 RUF046 |
| 1 | `KT/common/topology/__init__.py`、`KT/versions/v4/env_cfg.py` | 各 RUF022 |
| 1 | `KT/common/rope_backends/backend_cfg.py` / `KT/common/rope_backends/env.py` / `KT/rope_study/summarize_train.py` | RUF012 / SIM118 / ISC004 |
| 1 | `tests/test_knot_tying_topology.py`、`tests/test_state_archive.py` | 各 RUF100 |

### 8.4 未整形のファイル (`uvx ruff format --check --line-length 120 .`。34 本、全部)

「行数」は `uvx ruff format --diff --line-length 120 <file> | wc -l`。0.16.10 は Markdown の中の Python のコードブロックも整形する (md 4 本)。

| 行数 | ファイル | 中身 |
|---|---|---|
| 203 | `KT/common/rope_backends/xpbd_rod.py` | +104 / −20 (長い式の折り返し) |
| 144 | `KT/demos/multi_rope_catch.py` | +27 / −40 |
| 110 | `KT/demos/divergence_bench.py` | +28 / −12 |
| 97 | `KT/demos/rope_catch.py` | +15 / −18 |
| 52 | `KT/demos/overhand_script.py` | +19 / −10 |
| 48 | `KT/next_action.md` | md のコードブロック |
| 37 | `KT/rope_study/make_eval_jobs.py` | +9 / −3 |
| 37 | `KT/rope_study/bench_cantilever_xpbd.py` | +16 / −4 |
| 36 | `tests/test_rope_backends.py` | +7 / −5 |
| 33 | `KT/common/rope_model/rope_specs.py` | +2 / −6 |
| 32 | `KT/common/rope_model/generate_rope_usd.py` | +3 / −9 |
| 32 | `KT/common/rope_backends/adapters.py` | +13 / −2 |
| 31 | `KT/common/parallel_gripper/generate_gripper_usd.py` | +7 / −5 |
| 29 | `KT/rope_study/env_cfg.py` | +10 / −2 |
| 28 | `KT/common/README.md` | md のコードブロック |
| 27 | `KT/rope_study/bench_physics.py` | +8 / −2 |
| 26 | `tests/test_r2_reduce.py` | +7 / −2 |
| 24 | `KT/demos/spawn_rope.py` | +3 / −4 |
| 20 | `KT/versions/v0/README.md` | md のコードブロック (コメントの桁揃えが崩れる) |
| 19 | `KT/rope_study/summarize_eval.py` | +6 / −2 |
| 19 | `KT/common/rope_backends/centerline.py` | +4 / −2 |
| 19 | `SC/rsl_rl/train.py` / `SC/rsl_rl/play.py` | 各 +2 / −2 (末尾の改行と `# isort: skip` の前の空白) |
| 16 | `KT/rope_study/agents/rsl_rl_ppo_cfg.py` | +5 / −1 |
| 14 | `tests/test_knot_detect.py` / `KT/common/rope_backends/env.py` / `KT/common/mdp/actions.py` | 各 +3 / −1 |
| 13 | `KT/versions/v1/diff_to_TwistedRL.md` | md のコードブロック |
| 10 | `src/isaac_twisted_rl/tasks/__init__.py`、`SC/zero_agent.py`、`SC/rsl_rl/cli_args.py`、`SC/random_agent.py`、`SC/list_envs.py` | 末尾の改行だけ |
| 9 | `src/isaac_twisted_rl/__init__.py` | 末尾の改行だけ |

---

## 9. 所見 (変更の候補)

工数は、1 人が Claude Code と作業するときの目安。リスクの基準: **低** = 試験と静的検査で挙動が同じだと言い切れる。**中** = Isaac Sim なしでは確かめきれない部分がある。**高** = 学習・評価の結果が変わりうる。

| # | 候補 | リスク | 挙動が変わらないことの確かめ方 | 工数 |
|---|---|---|---|---|
| **F-1** | ruff の設定を `pyproject.toml` に置く。決めること: 版の固定 (`required-version`)、規則の集合 (まずは 0.14 系の既定と同じ E4/E7/E9/F + I から)、line-length 120、Isaac Lab 流の isort の区切り、起動スクリプトだけ E402 を無視 (per-file-ignores)、`mdp/` は B008 を無視、`*.md` を format から外すか。README の `.pre-commit-config.yaml` の記述も実態に合わせる (`KT/README.md:240`)。**これを最初にやる** (F-2、F-3 の件数がこれで決まる) | 低 | コードの差分がないこと (`git diff --stat` が `pyproject.toml` と README だけ)。設定の前後で `uvx ruff check --statistics` を比べる | 1〜2 h |
| **F-2** | `ruff format` を py 30 本に当てる (md 4 本は F-1 の決定に従う)。末尾の改行だけの 8 本は単独のコミットにする | 低 | 整形の前後で、全ファイルの `ast.dump` が一致すること (docstring の空白の正規化だけは除いて比べるスクリプトを書く)。pytest が 268 passed のまま。Warp のカーネルのキャッシュが作り直されるだけで済むかは未確認 | 1 h |
| **F-3** | 安全な自動修正だけを当てる: UP033 (4)、RUF022 (2)、PLR0402 (9)。RUF100 は F-1 の後で数え直してから。I001 はライブラリと試験 (15 本) だけにして、起動スクリプト 17 本は除く。UP018 は除く (N-5) | 低 | 対象の行の AST の差分だけを目で確かめる。pytest。`git diff` が対象の規則の行だけであること | 1 h |
| **F-4** | §6.3 の不変条件の試験を足す (I1〜I4 と登録の整合)。静的な検査だけで、Isaac Sim は要らない | 低 | 今の HEAD で通ること。わざと壊したコピー (scratchpad の中) で落ちること (例: topology に `import isaaclab` を足す、mdp の関数を 1 つ消す) | 0.5 日 |
| **F-5** | 試験の共通化: `tests/conftest.py` で `sys.path` を設定する (10 箇所)。`test_knot_detect` の紐の生成関数を共通モジュールへ出す (D-12) | 低〜中 | `pytest --collect-only -q` の一覧が前後で同じ (268 件)。全部通ること。`topology` などのモジュール名が変わらないこと | 2〜3 h |
| **F-6** | 試験の実行方法を、このリポジトリの venv で走る形 (`uv run python -m pytest`) にする。README の記述を直す (`KT/README.md:236-238`、`README.md:107`)。venv を作り直すかは利用者の判断 | 低 | `head -1 .venv/bin/pytest`。`python -c 'import torch; print(torch.__file__)'` が `isaac_twisted_rl/.venv` を指すこと。両方で回して、件数と結果が同じか比べる | 0.5 h (venv の作り直しは別) |
| **F-7** | 古い参照と docstring を直す: `paper/Report.md` (24 ファイル 29 行、md・HTML・docstring)、R-11〜R-13、`cli_args.py` の docstring。**docstring・コメント・md・HTML だけ** | 低 | docstring を除いた AST が前後で一致すること。`git grep -E '(^|[^_A-Za-z])Report\.md'` が 0 件 | 1〜2 h |
| **F-8** | 純粋な層の重複を減らす: D-5 (divergence_bench が centerline の関数を使う)、D-11 (観測の式の純 torch の核を純粋なモジュールに出し、observations と試験の両方から使う。`mdp` の名前は残す)。D-1 は、`state_archive` が単体で読まれる制約があるので後回し | 中 | 前後の関数をランダムな入力で比べる試験 (`torch.equal`)。`mdp` の名前の試験 (F-4)。observations の配線は Isaac Sim でのスモークテストが要る (未確認のまま残る) | 0.5 日 |
| **F-9** | スクリプト層の重複を減らす: D-6 (chain 評価のループと `_STAGE_DIMS`)、D-7 (根のパスを 1 箇所で計算)、D-2〜D-4 (デモの補助関数) | 中 | D-7: 前後で計算されるパスを、純 Python のスクリプトで比べる (Isaac Sim 不要)。D-6: 同じチェックポイント・同じ seed で eval_chain の JSON が一致すること (GPU が要る)。デモは目で確かめるしかない | 1 日 + GPU での確認 |
| **F-10** | 置き場所の整理: (a) `demos/` から道具 (`generate_assets`、`overhand_script`、`divergence_bench`) を分ける。(b) rope_study のシェルを Python のパッケージの外へ出し、`ROOT` を相対にする。(c) `topology_debug.py:75` の既定の出力先をリポジトリの根の `outputs/` にする。(d) `state_clone`・`beam_search` を評価用のまとまりにする | 中 | 旧パスの `git grep` が 0 件 (README・シェル・docstring)。`_REPO_ROOT` の階層の数が変わるので、パスを静的に比べる。スクリプトの起動確認は Isaac Sim が要る (未確認)。(c) は出力先が変わる、つまり意図した挙動の変更 | 0.5〜1 日 |
| **F-11** | 大きなファイルを分ける。対象は、純粋で試験のあるものから: `knot_graph.py` (`:279-462` を別モジュールへ、`topology/__init__` の再 export は保つ)、`xpbd_rod.py` (設定・カーネル・ソルバ。試験のローダも直す)、`mdp/archive.py` (v3 と v4。名前は再 export で保つ) | 中 | pytest。`topology` と `mdp` の公開名の一覧が前後で同じ (F-4)。archive は Isaac Sim で import して確かめる必要がある (未確認)。Warp のカーネルは CPU の試験 (`tests/test_rope_backends.py`) で確かめる | 1 本あたり 0.5 日 |
| **F-12** | バージョン間の private な名前の import (§5.2) を、明示的な公開名 (別名を追加するだけ) に置き換えるか、依存の形を試験で固定する。継承の連鎖そのものは変えない | 低〜中 | 静的な試験で import のグラフを固定する。設定の値は、Isaac Sim で `parse_env_cfg` の結果を前後で比べないと確かめられない (未確認) | 2 h |

勧める順序: F-1 → F-6 → F-4 → F-2 → F-3 → F-7 → F-5 →(Isaac Sim で確かめる手段ができてから) F-8〜F-12。

---

## 10. やらない方がよいもの

| # | やらない方がよいこと | 理由 | 根拠 |
|---|---|---|---|
| N-1 | `mdp` フォルダの改名、mdp の項の削除・改名 | ブラックリストの部分一致と、過去の版の設定が項を参照していることの両方で壊れる | `KT/README.md:180,225-227`、`src/isaac_twisted_rl/tasks/__init__.py:15` |
| N-2 | `versions/v0`〜`v2` の重複 (D-10) の共通化 | 過去の実験の設定を書き換えることになる。README の方針は「版は設定」で、基準コミットで再現する | `KT/README.md:154-173` |
| N-3 | `representation.py` の camelCase の改名・分割 | 移植元との対応を保つ方針 | `KT/common/topology/representation.py:36-41` |
| N-4 | `PickPlaceAction` の分割 | 中核のクラスで、Isaac Sim なしでは試験できない。サブクラスが内部に依存している | `KT/common/mdp/actions.py:77-705`、`KT/common/rope_backends/noisy_action.py:17-40` |
| N-5 | `ruff check --fix` をまとめて当てる | I001 は起動スクリプトの import 順を変える。UP018 は Warp の可変変数の宣言を消す。RUF100 は規則の集合を決める前に noqa を消す | §8.2 |
| N-6 | 手で直す lint の SIM118・B008・B006・RUF012・DTZ005 を直す | 壊れる (SIM118)、または挙動が変わる (他) | §8.2 |
| N-7 | デモと共有層の把持の定数 (D-13) の統合 | デモごとに意図して値が違う | `KT/demos/rope_catch.py:131`、`KT/demos/multi_rope_catch.py:103`、`KT/common/mdp/phases.py:36-60` |
| N-8 | AppLauncher の定型 (D-8) の共通化 | Kit の起動と import の順序に意味がある | `SC/rsl_rl/train.py:48`→`:75-96` |
| N-9 | `__init__.py` に import を足すこと、`topology/__init__.py` の再 export を減らすこと | 不変条件 I1 に反する。または、試験の `import topology as tp` が壊れる | §6、`tests/test_knot_detect.py:41` ほか |
| N-10 | `paper/` の PDF・HTML と `versions/v3/HTML/` の移動・削除 | 相対リンクが切れる。PDF を残すかは利用者の判断 (再配布の条件は未確認) | §3.1 |
| N-11 | Isaac Lab のテンプレート由来のファイル (`SC/rsl_rl/train.py`、`play.py`、`cli_args.py`、`SC/list_envs.py`、`zero_agent.py`、`random_agent.py`) の大きな書き換え | upstream との比較が難しくなる。整形 (F-2) と docstring (F-7) までに留める | 各ファイルの 1〜4 行目の著作権表示 |

---

## 11. 確認できなかったこと (未確認)

- このリポジトリの venv で試験を回したときの結果 (F-6)。
- README の言う「Kit 起動前に `__init__` が実行される」経路 (§6.2 の注意)。リポジトリ内のスクリプトでは、AppLauncher の後で import している。
- I001・UP018 を当てたとき、実際に壊れるか (順序や書き方に意味がありうることだけ確かめた)。
- `ruff format` の後、Warp のカーネルが再コンパイルされるだけで済むか。
- `rope_specs` が 2 つのモジュール名で読まれること (D-15) の実害。
- isaaclab に依存するコード (mdp の項、env_cfg、scripts、demos、rope_study) が今動くか。Isaac Sim を起動していないので、gym の ID が 166 個という数も静的に数えた値。
- `paper/` の PDF の再配布の条件。
- `uv_build` の wheel に `.sh`・`.md`・`.html` が含まれるか。

---

## 付録 A: 使ったスクリプト (scratchpad に置いた。リポジトリには置いていない)

### A-1 不変条件 I1〜I3 (`invariants.py`)

```python
"""Static (ast only) check of README invariants I1-I3. Nothing from the repo is imported or executed."""
import ast, os, subprocess
ROOT = "/home/itoyama/work/isaac_twisted_rl"; SRC = os.path.join(ROOT, "src")
FORBID = ("isaaclab", "omni", "pxr", "isaacsim", "carb"); BLACK = ["utils", ".mdp"]

def modname(p):
    r = os.path.relpath(p, SRC)[:-3].replace(os.sep, ".")
    return r[: -len(".__init__")] if r.endswith(".__init__") else r

def modfile(m):
    b = os.path.join(SRC, *m.split("."))
    for c in (b + ".py", os.path.join(b, "__init__.py")):
        if os.path.isfile(c):
            return c

def imports(path, include_functions=False):
    t = ast.parse(open(path, encoding="utf-8").read()); mod, ispkg = modname(path), path.endswith("__init__.py")
    skip = set()
    if not include_functions:
        for fn in ast.walk(t):
            if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                skip |= {id(n) for n in ast.walk(fn)}
    for n in ast.walk(t):
        if id(n) in skip:
            continue
        if isinstance(n, ast.Import):
            for a in n.names:
                yield n.lineno, a.name
        elif isinstance(n, ast.ImportFrom):
            if n.level:
                parts = mod.split(".") if ispkg else mod.split(".")[:-1]
                parts = parts[: len(parts) - (n.level - 1)]
                base = ".".join(parts + ([n.module] if n.module else []))
            else:
                base = n.module
            yield n.lineno, base
            for a in n.names:
                if modfile(base + "." + a.name):
                    yield n.lineno, base + "." + a.name

def reach(path):  # module-level transitive closure inside the repo
    seen, ext, stack = set(), [], [path]
    while stack:
        p = stack.pop()
        if p in seen:
            continue
        seen.add(p)
        for ln, name in imports(p):
            f = modfile(name)
            (stack.append(f) if f else ext.append((os.path.relpath(p, ROOT), ln, name)))
    return seen, ext

files = subprocess.check_output(["git", "ls-files", "*.py"], cwd=ROOT, text=True).split()
bad1 = bad2 = 0
for f in files:  # I1
    if f.startswith("src/isaac_twisted_rl/tasks/") and f.endswith("__init__.py"):
        if any(b in modname(os.path.join(ROOT, f)) for b in BLACK):
            print("SKIP(blacklisted)", f); continue
        seen, ext = reach(os.path.join(ROOT, f))
        forb = [e for e in ext if e[2].split(".")[0].startswith(FORBID) and e[2] != "isaaclab_tasks.utils"]
        bad1 += len(forb); print("NG" if forb else "OK", f, sorted({e[2].split(".")[0] for e in ext}), forb or "")
for f in files:  # I2
    if "/common/topology/" in f or f.endswith("/common/state_archive.py"):
        forb = [(l, n) for l, n in imports(os.path.join(ROOT, f), True) if n.split(".")[0].startswith(FORBID)]
        bad2 += len(forb); print("NG" if forb else "OK", f, forb or "")
for dp, dn, _ in os.walk(SRC):  # I3
    for d in dn:
        if "mdp" in d and d != "__pycache__":
            print("mdp dir:", os.path.relpath(os.path.join(dp, d), ROOT))
print(f"RESULT I1 violations={bad1}  I2 violations={bad2}")
```

出力 (要約): 自動で import される 29 本はすべて OK (外部の依存は `KT/__init__.py` が `gymnasium`、`topology/__init__.py` が torch と標準ライブラリ、`rope_study/__init__.py` が `gymnasium` と `dataclasses`、`tasks/__init__.py` が `isaaclab_tasks`)。`common/mdp/__init__.py` は SKIP。I2 の 9 本は OK。`mdp dir: src/isaac_twisted_rl/tasks/knot_tying/common/mdp`。`RESULT I1 violations=0  I2 violations=0`。

### A-2 不変条件 I4 (`mdp_history.py` と `mdp_refs.py`)

```python
# mdp_history.py: 全コミットについて、*/mdp/*.py の最上位の def/class 名を集め、HEAD に無いものを出す
import ast, subprocess
R = "/home/itoyama/work/isaac_twisted_rl"
git = lambda *a: subprocess.check_output(["git", *a], cwd=R, text=True)
def names_at(c):
    out = {}
    for p in git("ls-tree", "-r", "--name-only", c).split():
        if "/mdp/" in p and p.endswith(".py") and not p.endswith("__init__.py"):
            for n in ast.parse(git("show", f"{c}:{p}")).body:
                if isinstance(n, (ast.FunctionDef, ast.ClassDef)):
                    out[n.name] = p
    return out
head, seen = names_at("HEAD"), {}
for c in git("rev-list", "--reverse", "HEAD").split():
    for k, p in names_at(c).items():
        seen.setdefault(k, (c[:7], p))
gone = {k: v for k, v in seen.items() if k not in head}
print(f"ever: {len(seen)}, HEAD: {len(head)}, missing in HEAD: {len(gone)}", gone)
```

出力: `names ever defined: 58, in HEAD: 58, missing in HEAD: 0`

`mdp_refs.py` は、追跡している全 `.py` の `mdp.<name>` (ast.Attribute) を集め、`common/mdp/*.py` と `.venv` の `isaaclab/envs/mdp/**/*.py` の最上位の定義と照らし合わせる。出力: `defined in common/mdp: 92 names; referenced as mdp.X: 40 distinct names (37 from common/mdp, 3 from isaaclab.envs.mdp, 0 unresolved)`。自前の def/class と isaaclab の名前の衝突は 0。

### A-3 ファイルの構造 (`outline.py`)

`ast.parse` して、最上位の def/class の行の範囲と、`# ----` などの区切りコメントの行を出す。§1 の切れ目は、この出力から取った。

### A-4 重複の検出 (`dups.py`)

```python
# 関数・クラスの本体 (docstring を除く) で、完全一致 (ast.dump) と、ほぼ一致 (ast.unparse の difflib.ratio >= THR) を探す
import ast, difflib, subprocess, collections, sys
R = "/home/itoyama/work/isaac_twisted_rl"; THR = float(sys.argv[1]); MIN = int(sys.argv[2])
items = []
for f in subprocess.check_output(["git", "ls-files", "*.py"], cwd=R, text=True).split():
    for n in ast.walk(ast.parse(open(f"{R}/{f}").read())):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.end_lineno - n.lineno + 1 >= MIN:
            b = n.body[1:] if (n.body and isinstance(n.body[0], ast.Expr) and isinstance(getattr(n.body[0], "value", None), ast.Constant)
                               and isinstance(n.body[0].value.value, str)) else n.body
            if b:
                m = ast.Module(body=b, type_ignores=[]); items.append((f, n.lineno, n.end_lineno, n.name, ast.dump(m), ast.unparse(m)))
ex = collections.defaultdict(list)
for it in items:
    ex[it[4]].append(it)
for g in ex.values():
    if len(g) > 1:
        print("EXACT", [(f, a, b, nm) for f, a, b, nm, *_ in g])
for i, a in enumerate(items):
    for b in items[i + 1:]:
        if a[0] != b[0] and a[4] != b[4] and min(len(a[5]), len(b[5])) / max(len(a[5]), len(b[5])) >= THR:
            r = difflib.SequenceMatcher(None, a[5], b[5], autojunk=False).ratio()
            if r >= THR:
                print(f"{r:.2f}", a[:4], b[:4])
```

`python3 dups.py 0.85 6` と `python3 dups.py 0.72 10` で回した。

### A-5 lint で使ったコマンド

```bash
uvx ruff --version                                                        # ruff 0.16.10
uvx ruff check --line-length 120 --statistics .                           # 247 件 (177 件 fixable)
uvx ruff check --line-length 120 --output-format concise .                # ファイルごとの件数
uvx ruff format --check --line-length 120 .                               # 34 files would be reformatted, 113 already formatted
uvx ruff format --diff --line-length 120 <file> | wc -l                   # 整形の差分の大きさ
uvx ruff check --line-length 120 --select I001 --diff <file>              # I001 の差分を見るだけ (書き込まない)
uvx ruff check --show-settings --line-length 120 <file>                   # 設定ファイルの有無と、有効な規則
uvx ruff@0.14.10 check --line-length 120 --statistics .                   # 参考: 84 件 (E402 だけ)
uvx ruff@0.14.10 format --check --line-length 120 .                       # 参考: 30 files would be reformatted
```

---

## メインによる根拠の確認 (2026-10-03)

メインの session が、次の 6 か所を自分で開いて確かめた。誤りは見つからなかった。

| 主張 | 確かめた内容 | 結果 |
|---|---|---|
| `.venv/bin` の 74 本中 57 本の shebang が so_arm101 の python | `.venv/bin/pytest:1` = `#!/home/itoyama/work/isaac_so_arm101/.venv/bin/python3`。各ファイルの 1 行目の shebang を数えると 57 本 | 一致 |
| KT/README.md:240 の `.pre-commit-config.yaml` は存在しない | README.md:240 の記述を確認。ファイルはなく、`git log --all -- .pre-commit-config.yaml` も 0 件 | 一致 |
| SIM118 を直すと壊れる (rope_backends/env.py:73) | env.py:73 は `if "rope" in scene.keys():`。インストールされた InteractiveScene (`.venv/.../isaaclab/source/isaaclab/isaaclab/scene/interactive_scene.py`) には `keys` (:656) と `__getitem__` (:675) があり、`__contains__` と `__iter__` はない | 一致 (静的な確認。実行はしていない) |
| v4/env_cfg.py:53 が overhand_script.py:117 の出力を読む | 両方の行が `outputs/archive_seed/overhand_{num_links}links.pt` を指す | 一致 |
| topology_debug.py:75 は既定で src の中に書く | `_DEFAULT_OUT` は `knot_tying/outputs/topology_debug` になる | 一致 |
| `paper/Report.md` を指す行は 24 ファイルに 29 行 | `git grep -n "Report\.md" \| grep -v Report_` が 29 行・24 ファイル | 一致 |

追加で確かめたこと (F-6 に関係する):
- このリポジトリの venv でテストを回すと、`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run python -m pytest tests/ -q` で **268 passed, 1 warning, 48.67 s**。
- 警告のパスは `/home/itoyama/work/isaac_twisted_rl/.venv/...` で、このリポジトリの venv で動いた。
- `uv run pytest` (shebang の経路) と結果は同じ。CLAUDE.md に書く検証のコマンドは `uv run python -m pytest` の形にする候補になる。
