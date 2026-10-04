---
name: add-version
description: MDP の新しい版 vN を versions/ に追加するときに使う。作るファイル、登録する箇所、継承元の確かめ方、更新漏れしやすい一覧。既存の版のバグ修正や、版の仕様を調べるだけのときは使わない。
---

# 版の追加

版は「設定」で、既存の版の挙動は変えない (理由は `docs/decisions/` の「版 = 設定」)。直したい挙動があるときも、新しい版を作る。

## 1. 番号と継承元を決める

- 番号は今ある最大の番号 + 1。**番号を使い直さない** (旧 v7 = 「v6 + 時間切れの罰則」の run が残り、今の v7 と食い違っている)。
- 継承元の版を明示して決める。`versions/vK/env_cfg.py` のクラスを継承する。v5 は誤って v3 のクラスを継承し、作り直した (commit `881dc8d`)。
- v4 の `env_cfg.py` を変えると v5・v6・v7・rope study に波及する。継承元のファイルは変えない。

## 2. 作るファイル (`src/isaac_twisted_rl/tasks/knot_tying/versions/vN/`)

| ファイル | 中身 |
|---|---|
| `__init__.py` | docstring だけ (Kit の起動前に import されるので、ここで isaaclab などを import しない。`tests/test_invariants.py` が検査する) |
| `env_cfg.py` | 継承元のクラスを継承し、変える項だけを書く |
| `agents/__init__.py` | docstring だけ |
| `agents/rsl_rl_ppo_cfg.py` | Runner の設定。`experiment_name = "knot_tying_vN"` |
| `README.md` | その版の仕様と理由だけ (結果は書かない。結果は `docs/experiments/`) |

新しい報酬・観測・終了の項が要るときは `common/mdp/` に**追加**する (既存の項は削除・改名しない)。

## 3. 書き換える箇所

| 箇所 | やること | 漏れた回数 (過去 6 回の版の追加で) |
|---|---|---|
| `knot_tying/__init__.py` の `_VERSIONS` | `"vN"` を足す (タスク ID はこれで自動で登録される) | 0 / 6 |
| `knot_tying/versions/__init__.py` の docstring の一覧 | vN の 1 行を足す | 3 / 6 |
| `knot_tying/README.md` の版の系譜 | 何を変えた版かを 1 行 | 1 / 6 |
| `docs/STATUS.md` | 版の状態に vN を足す | — |
| テスト | 新しい純粋な関数を足したなら単体テスト | 1 / 6 |

## 4. 確かめる

1. `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run python -m pytest tests/ -q` (版の一覧の整合と、`__init__.py` の import を検査する)。
2. commit する。
3. skill `train-eval` のスモークテストで起動を確かめる (Isaac Sim を使う)。
