# 0012 「版 = 設定」であって「版 = コードのコピー」ではない

- 状態: 有効 (v0〜v7 と、これから足す版のすべて)
- 読むべき状況: 新しい版を足すとき (手順は skill `add-version`)。`common/` の項を消す・改名する・挙動を変えたくなったとき。過去の版の結果を厳密に再現したいとき
- 最終更新: 2026-10-04

## 決定

- タスクを**版**に分ける (v0 = 交差 1 つ、v1 = 固結び、以降も同様)。以降の改良は既存の版を書き換えず、新しい版を足していく。
- `versions/vN/` に置くのは **env 設定・PPO 設定・README だけ**。物理・位相判定・MDP 項の実装は `common/` に共有で置く。版の間の差は **「どの項を、どの重みで配線するか」**だけで表現する。
- この性質を保つため、**共有層の項は追加のみとし、削除・改名しない**。役目を終えた項は docstring に `.. deprecated::` を書いて残す (`first_crossing_feature` が既にこの形)。
- 既存の項の**挙動**を変えたいときは、新しい名前の項を足して新しい版だけが使う。**バグ修正は例外** (全版に効かせる)。
- 厳密な再現が要るときは、git のコミットを遡る (v0 は README に基準コミットを書いてある。他の版は、その版を追加したコミットが基準)。

## 理由

### きっかけ (2026-09-24)

`main` と `feat/overhand-knot` で、同じ `Isaac-Twisted-RL-Twist-v0` が別の実装を指してしまっていた。これを解消するため、タスクを版の制度にした。

### この方針にできた理由

旧 §4.8 (v1) の変更が、**共有 MDP 層への純粋な追加だった**ため。`git diff main..HEAD` で確認すると、`mdp/rewards.py` と `mdp/observations.py` はどちらも「84 行追加・0 行削除」で、`mdp/terminations.py` / `mdp/phases.py` / `mdp/events.py` / `rope_state.py` はバイト一致だった。v0 の項がすべて残っているので、**v0 を共有コードから再構成しても main と挙動が一致する**。したがってコードを複製・凍結する必要がなく、版の間の差は「どの項をどの重みで配線するか」だけで表現できる。実際 `crossing_progress` (v0) と `knot_class_shaping` (v1) は `common/mdp/rewards.py` に共存している。

### 利点とトレードオフ

- **利点:** 物理まわりのバグ (発散対策、把持ゲイン) を直すと全版に効く。コピーしていると v0 だけ古いバグが残る。また v1 の README が「v0 のこの項をこう差し替えた」と書けるのは、両者が同じ語彙で書かれているから。
- **トレードオフ:** `common/` を変えると v0 の挙動も変わりうる。これは意図的な選択で、「過去の版の完全凍結」より「共通基盤の改善が全版に効く」ことを優先している。

v4・v5 は物理の値 (ソルバの位置反復・めり込み解消速度) も設定で変える。どちらも `KnotTyingEnvCfg` の上書き項 (既定 `None` なら従来の挙動) として追加した (`kt/common/env_cfg_base.py` の `solver_position_iterations`・`rope_max_depenetration_velocity`)。

### 実装時に踏んだ注意点

- `make_preset_cfgs` の `__post_init__` で `super(type(self), self)` を使うと、Play 側から呼ばれたときに無限再帰する。基底クラスを閉包で固定して直接呼ぶこと。
- `--experiment_name` フラグは `cli_args.update_rsl_rl_cfg` が適用していないので効かない。cfg の側で `knot_tying_vN` を指定する。
- `import_packages` の注意 (`mdp` というフォルダ名を変えない、など) は `kt/README.md` の「構造上の制約」にある。

## 検討して採らなかった案

| 案 | 採らなかった理由 |
|---|---|
| 版ごとにコードを複製・凍結する (「版 = コードのコピー」) | コピーしていると v0 だけ古いバグが残る。共通基盤の改善を全版に効かせることを優先した |

## 今の実装

(`kt/` = `src/isaac_twisted_rl/tasks/knot_tying/`)

| 何 | 場所 | 確認 |
|---|---|---|
| 版の一覧 `_VERSIONS = ("v0", ..., "v7")` | `kt/__init__.py:40` | 一致 |
| `crossing_progress` (v0) と `knot_class_shaping` (v1) の共存 | `kt/common/mdp/rewards.py:61`、`:100` | 一致 |
| `first_crossing_feature` の `.. deprecated::` | `kt/common/mdp/observations.py:97-100` | 一致 |
| 閉包で基底を固定した `__post_init__` | `kt/common/rope_task_cfg.py:196-207` | 一致 |
| `--experiment_name` は `update_rsl_rl_cfg` で適用されない | `src/isaac_twisted_rl/scripts/rsl_rl/cli_args.py:58-90` (定義は `:26`)、ログ名は `agent_cfg.experiment_name` から (`train.py:140`) | 一致 (ログ名の決まり方は [`docs/system/logs.md`](../system/logs.md)) |
| 追加のみの規約 | `kt/README.md` の「構造上の制約」(静的な検査は `tests/test_invariants.py:193` の I4) | README に同じ規約がある。既存の版の挙動を変えない規則は `.claude/rules/versions.md` |

## 出典

`docs/history/next_action.md` の行番号は、頭書き (33 行) を足す前のもの (今の行番号は 33 を足す)。

- `docs/history/next_action.md:559-583` (旧 §6)
- `kt/README.md:154-173` (commit 62b251b の「設計方針: 『バージョン = 設定』であって『バージョン = コードのコピー』ではない」。commit 55480a7 で README から外れ、この文書へのリンクになった)、`:175-187` (同 commit の「バージョンを追加するときの規約」)
