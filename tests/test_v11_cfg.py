"""v11 と v7 の設定の差が「段の定義」だけであることを、静的に検査する。

v11 は 1 版 1 変数 (`docs/decisions/0012-version-is-config.md`) で、v7 との比較の結論は「表に無い形の段の付け方だけが
違う」ことに依っている (`docs/decisions/0024-v11-dead-end-stages.md`)。env の設定は Isaac Sim なしでは import
できないので、`ast` でソースを読んで次を確かめる:

- v11 の env の設定クラスは v7 のクラスだけを継承し、本体はフィールド `stage_definition = "bfs_dead_end"` だけ。
  `__post_init__` を持たない (初期アーカイブ・`stage_weights` は v7 のまま)。
- `stage_definition` の値は段の定義の名前 (`topology.STAGE_DEFINITIONS`) のどれか。
- Play の設定 (chain モード) とプリセットの生成は v7 と同じ関数を使う。
- v11 の PPO の設定は v7 のクラスを継承し、ログ名だけを変える。

段の定義の中身は `tests/test_dead_end_stages.py`、組み立てた設定の差 (`params/env.yaml`) は smoke test の run で確かめる。
"""

from __future__ import annotations

import ast

from test_v9_cfg import _body_without_docstring, _classes, _imported_as, _parse, _preset_call, _stage_weight_writes
from topology import path_stages as ps


def test_v11_env_cfg_only_changes_stage_definition():
    tree = _parse("v11/env_cfg.py")
    (cls,) = _classes(tree)
    assert cls.name == "KnotTyingV11EnvCfg"
    assert [ast.unparse(b) for b in cls.bases] == ["KnotTyingV7EnvCfg"]
    assert _imported_as(tree, "KnotTyingV7EnvCfg") == ("v7.env_cfg", "KnotTyingV7EnvCfg")

    body = [
        b for b in _body_without_docstring(cls) if not (isinstance(b, ast.Expr) and isinstance(b.value, ast.Constant))
    ]
    assert [type(b) for b in body] == [ast.AnnAssign]
    (field,) = body
    assert ast.unparse(field.target) == "stage_definition"
    assert ast.literal_eval(field.value) == "bfs_dead_end"
    assert "bfs_dead_end" in ps.STAGE_DEFINITIONS
    assert _stage_weight_writes(tree) == []
    assert "load_paths" not in ast.unparse(tree)


def test_v11_presets_and_play_match_v7():
    v7, v11 = _parse("v7/env_cfg.py"), _parse("v11/env_cfg.py")
    c7, c11 = _preset_call(v7), _preset_call(v11)
    assert ast.unparse(c11.args[0]) == "KnotTyingV11EnvCfg"
    assert [(kw.arg, ast.unparse(kw.value)) for kw in c7.keywords] == [
        (kw.arg, ast.unparse(kw.value)) for kw in c11.keywords
    ]
    assert _imported_as(v7, "_play_post_init") == _imported_as(v11, "_play_post_init")


def test_v11_agent_cfg_matches_v7_except_log_name():
    tree = _parse("v11/agents/rsl_rl_ppo_cfg.py")
    (cls,) = _classes(tree)
    assert [ast.unparse(b) for b in cls.bases] == ["_V7MLP"]
    assert _imported_as(tree, "_V7MLP") == ("v7.agents.rsl_rl_ppo_cfg", "KnotTyingPPORunnerCfg")
    assert [ast.unparse(b) for b in _body_without_docstring(cls)] == ["experiment_name = 'knot_tying_v11'"]
