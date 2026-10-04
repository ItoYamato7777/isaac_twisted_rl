---
name: handoff
description: セッションを終える・区切るとき、docs/STATUS.md を上書きで最新にして次のセッションへ引き継ぐときに使う。走っているジョブと PID、次の作業、保留を書き、コミットできる状態かを確かめる。結果の記録は skill experiment-record。過去の引き継ぎメモ (docs/history/) を探すときは使わない。
---

# 引き継ぎ (STATUS の上書き)

`docs/STATUS.md` は「今」だけを持つ。上書きで更新し、追記で伸ばさない。経緯は書かない (結果は `docs/experiments/`、判断は `docs/decisions/`)。

## 1. 先に片付ける

1. 結果が出たなら、skill `experiment-record` で実験記録を書く。
2. テスト: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run python -m pytest tests/ -q`
3. `git status` を見て、コミットできる単位にまとめる (コミットはユーザーの方針に従う)。scratchpad (`/tmp/claude-*`) にしか無いスクリプトやログで、後で要るものは、追跡する場所か `outputs/` に移す。

## 2. 走っているジョブを確かめる

```bash
nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader
ps -eo pid,etime,args | grep -E 'train|eval_|play|overhand_script|divergence_bench|bench_physics' | grep -v grep
```

自分が起動したジョブは PID・GPU・run のフォルダ・ログのパス・終わる見込みを書く。自分が起動していないジョブは「他のジョブ」として PID だけを書き、触らない。

## 3. `docs/STATUS.md` を上書きする

節の形は変えない。各節は箇条書きで短く。

| 節 | 書くこと |
|---|---|
| 要点 | 3 行まで。今どこにいて、次に何をするか (SessionStart hook がこの節を表示する) |
| 版の状態 | 版ごとに 1 行 (状態と、結果の記録へのリンク。数値は書かず記録にリンク) |
| 走っているジョブ | §2 の内容。無ければ「なし」 |
| 次の作業 | 番号つき。1 番目は、次のセッションが最初にやること。仮説と比べる相手があれば併記 |
| 保留 | 打ち切り中・判断待ちのもの。誰の判断待ちか |
| 最終更新 | 日付 |

- 済んだ作業は STATUS から消す (結果は実験記録にある)。
- 「最新」「ここから読む」のような印を足さない。STATUS 全体が常に最新。
