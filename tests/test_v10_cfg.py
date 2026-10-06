"""v10 と v7 の設定の差が「段の定義」と「初期アーカイブのファイル」だけであることを、静的に検査する。

v10 は 1 版 1 変数 (`docs/decisions/0012-version-is-config.md`) で、v7 との比較の結論は「段の定義だけが違う」
ことに依っている (初期アーカイブは、段の定義を変えた結果として v7 のファイルの部分集合にしたもの)。
env の設定は Isaac Sim なしでは import できないので、`ast` でソースを読んで次を確かめる:

- v10 の env の設定クラスは v7 のクラスだけを継承し、本体はフィールド `stage_definition = "path"` と、
  `__post_init__` で `load_paths` を 1 か所書き換えるだけ。`stage_weights` は書かない (v7 の 1:1:1)。
- `stage_definition` の値は段の定義の名前 (`topology.STAGE_DEFINITIONS`) のどれか。
- 初期アーカイブのファイルは v4〜v9 のファイル名に `_v10` を付けたもの。
- Play の設定 (chain モード) とプリセットの生成は v7 と同じ関数を使う。
- v10 の PPO の設定は v7 のクラスを継承し、ログ名だけを変える。

段の定義の中身は `tests/test_path_stages.py`、組み立てた設定の差 (`params/env.yaml`) は smoke test の run で確かめる。
"""

from __future__ import annotations

import ast

from test_v9_cfg import (
    _body_without_docstring,
    _classes,
    _imported_as,
    _module_assign,
    _parse,
    _preset_call,
    _stage_weight_writes,
)
from topology import path_stages as ps


def test_v10_env_cfg_only_changes_stage_definition():
    tree = _parse("v10/env_cfg.py")
    (cls,) = _classes(tree)
    assert cls.name == "KnotTyingV10EnvCfg"
    assert [ast.unparse(b) for b in cls.bases] == ["KnotTyingV7EnvCfg"]
    assert _imported_as(tree, "KnotTyingV7EnvCfg") == ("v7.env_cfg", "KnotTyingV7EnvCfg")

    body = [
        b for b in _body_without_docstring(cls) if not (isinstance(b, ast.Expr) and isinstance(b.value, ast.Constant))
    ]
    assert [type(b) for b in body] == [ast.AnnAssign, ast.FunctionDef]
    field, post_init = body
    assert ast.unparse(field.target) == "stage_definition"
    assert ast.literal_eval(field.value) == "path"
    assert "path" in ps.STAGE_DEFINITIONS
    assert post_init.name == "__post_init__"
    assert [ast.unparse(s) for s in _body_without_docstring(post_init)] == [
        "super().__post_init__()",
        "self.events.reset_to_stage.params['load_paths'] = (SEED_ARCHIVE,)",
    ]
    assert _stage_weight_writes(tree) == []


def test_v10_seed_archive_is_v4_name_with_suffix():
    v4 = ast.unparse(_module_assign(_parse("v4/env_cfg.py"), "SEED_ARCHIVE"))
    v10 = ast.unparse(_module_assign(_parse("v10/env_cfg.py"), "SEED_ARCHIVE"))
    assert v10 == v4.replace("overhand_{num_links}links.pt", "overhand_{num_links}links_v10.pt")
    assert v10 != v4


def test_v10_presets_and_play_match_v7():
    v7, v10 = _parse("v7/env_cfg.py"), _parse("v10/env_cfg.py")
    c7, c10 = _preset_call(v7), _preset_call(v10)
    assert ast.unparse(c10.args[0]) == "KnotTyingV10EnvCfg"
    assert [(kw.arg, ast.unparse(kw.value)) for kw in c7.keywords] == [
        (kw.arg, ast.unparse(kw.value)) for kw in c10.keywords
    ]
    assert _imported_as(v7, "_play_post_init") == _imported_as(v10, "_play_post_init")


def test_v10_agent_cfg_matches_v7_except_log_name():
    tree = _parse("v10/agents/rsl_rl_ppo_cfg.py")
    (cls,) = _classes(tree)
    assert [ast.unparse(b) for b in cls.bases] == ["_V7MLP"]
    assert _imported_as(tree, "_V7MLP") == ("v7.agents.rsl_rl_ppo_cfg", "KnotTyingPPORunnerCfg")
    assert [ast.unparse(b) for b in _body_without_docstring(cls)] == ["experiment_name = 'knot_tying_v10'"]


def test_earlier_versions_do_not_set_stage_definition():
    """v4〜v9 は `stage_definition` を持たない (= `bfs`。`topology.stage_definition_of` の既定)。"""
    for ver in ("v4", "v5", "v6", "v7", "v8", "v9"):
        src = _parse(f"{ver}/env_cfg.py")
        assert "stage_definition" not in ast.unparse(src), ver
