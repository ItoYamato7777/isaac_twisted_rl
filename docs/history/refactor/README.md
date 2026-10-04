# リファクタリングの記録 (2026-10-03〜04、ブランチ `refactor/claude-foundation`)

- 状態: 凍結 (経緯。通常は読まない)
- 読むべき状況: 今の文書の構成・Claude 基盤 (CLAUDE.md・`.claude/`)・試験の根拠を辿るとき。実験記録の「記録なし」の根拠 (B-101 など) を確かめるとき。cold-start テストをやり直すとき

作業の途中は `refactor/` (git 管理外) に置き、Phase 5 の後にここへ移した。本文の中の `refactor/...` というパスは、今の `docs/history/refactor/...` と読み替える。

## 中身

| ファイル | 中身 |
|---|---|
| [`final_report.md`](final_report.md) | 最終報告 (BRIEF §10 の 11 項目)。最初に読むならこれ |
| [`BRIEF.md`](BRIEF.md) | 依頼の全文 |
| [`PROGRESS.md`](PROGRESS.md) | 各 Phase の計画・状態・承認済みの判断 |
| [`claude-code-spec.md`](claude-code-spec.md) | 設計の前提にした Claude Code の仕様 (出典付き、2026-10-03 時点) |
| [`proposal.md`](proposal.md) | Phase 2 の設計の提案 (承認済み) |
| `audit/A.md`〜`G.md` | Phase 1 の調査 (A 文書・B 情報の分類・C 用語・D 規則と失敗・E 繰り返した手順・F コード構造・G cold-start の基準値)。他の文書が引く所見の番号 (A-3、B-101、C-1、D-6 など) の出典 |
| `audit/G_questions.md`、`G_prompts.md` | cold-start テストの 14 問と、回答役・採点役への依頼文 |
| [`phase5/coldstart.md`](phase5/coldstart.md) | cold-start テストの前後の比較 |
| `phase3/` | 再現の項目の照合 (`items.py`、`baseline_items.json`)、リンクの書き換え、guard の hook の試験、skill の description の試験 |
| `phase4/` | 挙動が変わらないことの確認の道具: タスク ID の一覧 (`dump_tasks.py`・`tasks_*.json`)、AST の比較、環境設定のダンプ (`dump_env_cfgs.py`)、`run_job.sh`・ruff の hook の試験 |

## 道具の使い方 (リポジトリのルートで)

```bash
python3 docs/history/refactor/phase3/items.py check            # 再現の項目 2,259 組が文書に残っているか (この folder は照合の外)
uv run python docs/history/refactor/phase4/dump_tasks.py | diff - docs/history/refactor/phase4/tasks_before.json
bash docs/history/refactor/phase4/test_run_job.sh
python3 docs/history/refactor/phase3/test_guard.py
```

`test_run_job.sh`・`test_guard.py` はリポジトリの絶対パス (`/home/itoyama/work/isaac_twisted_rl`) を使う。ruff の対象から外してある (`ruff.toml`)。

## 追跡していないもの

- `phase4/env_cfgs_{before,before_run1,after}.json` (各 3,505,448 bytes、`.gitignore`)。P4-9 の前後の 166 タスクの設定のダンプで、3 本とも sha256 `83146bc681e7a23db3f0d03f11c527d04f57c58cc9ff077800b4f5b11de37e95`。作り直しは `dump_env_cfgs.py` (Isaac Sim が要る)。
- cold-start テストの正答・採点 (`/home/itoyama/coldstart_answers/isaac_twisted_rl/`) と回答の全文 (`/home/itoyama/coldstart_runs/isaac_twisted_rl/`)。リポジトリの外。

## cold-start テストをやり直すときの注意

この folder には質問・正答の要点 (`audit/G.md`、`phase5/coldstart.md`)・調査の結論がある。回答役には、この folder を読ませない (作業中は `refactor/` を git 管理外にして worktree から外していた)。
