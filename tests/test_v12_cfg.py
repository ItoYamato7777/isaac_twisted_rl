"""v12 と v7 の設定の差が「アーカイブの保存の規則 (と記録の写し)」だけであることを、静的に検査する。

v12 は 1 版 1 変数 (`docs/decisions/0012-version-is-config.md`) で、v7 との比較の結論は「どの手の後の形を
保存するかだけが違う」ことに依っている (`docs/decisions/0025-archive-save-on-stage-up.md`)。env の設定は
Isaac Sim なしでは import できないので、`ast` でソースを読んで次を確かめる:

- v12 の env の設定クラスは v7 のクラスだけを継承し、本体は `__post_init__` で `record_stage_archive` の
  `save_on` と `history_every` を書くだけ。値は `stage_up` と 200。
- Play は v7 と同じ関数を呼んでから `history_every` を 0 にするだけ。
- v12 の PPO の設定は v7 のクラスを継承し、ログ名だけを変える。

保存する行の選び方は `tests/test_state_archive.py`、組み立てた設定 (`params/env.yaml`) は smoke test の run で確かめる。
"""

from __future__ import annotations

import ast

import state_archive as sa
from test_v9_cfg import _body_without_docstring, _classes, _imported_as, _module_assign, _parse, _preset_call


def test_v12_env_cfg_only_changes_save_rule():
    tree = _parse("v12/env_cfg.py")
    (cls,) = _classes(tree)
    assert cls.name == "KnotTyingV12EnvCfg"
    assert [ast.unparse(b) for b in cls.bases] == ["KnotTyingV7EnvCfg"]
    assert _imported_as(tree, "KnotTyingV7EnvCfg") == ("v7.env_cfg", "KnotTyingV7EnvCfg")

    body = _body_without_docstring(cls)
    assert [type(b) for b in body] == [ast.FunctionDef], "クラスの本体は __post_init__ だけ (フィールドを上書きしない)"
    (post_init,) = body
    assert post_init.name == "__post_init__"
    assert [ast.unparse(s) for s in _body_without_docstring(post_init)] == [
        "super().__post_init__()",
        "self.rewards.record_stage_archive.params['save_on'] = SAVE_ON",
        "self.rewards.record_stage_archive.params['history_every'] = HISTORY_EVERY",
    ]
    assert ast.literal_eval(_module_assign(tree, "SAVE_ON")) == "stage_up"
    assert ast.literal_eval(_module_assign(tree, "SAVE_ON")) in sa.SAVE_RULES
    assert ast.literal_eval(_module_assign(tree, "HISTORY_EVERY")) == 200


def test_v7_does_not_set_save_rule():
    # v7 は v4 の既定 (`save_on` も `history_every` も params に無い = 項の既定の every_move / 0) を継承する。
    for ver in ("v4", "v5", "v6", "v7"):
        src = ast.unparse(_parse(f"{ver}/env_cfg.py"))
        assert "save_on" not in src and "history_every" not in src, ver


def test_v12_play_extends_v7_play_only_by_history():
    v7, v12 = _parse("v7/env_cfg.py"), _parse("v12/env_cfg.py")
    assert _imported_as(v12, "_v7_play_post_init") == _imported_as(v7, "_play_post_init")
    fn = next(n for n in v12.body if isinstance(n, ast.FunctionDef) and n.name == "play_post_init")
    assert [ast.unparse(s) for s in _body_without_docstring(fn)] == [
        "_v7_play_post_init(cfg)",
        "cfg.rewards.record_stage_archive.params['history_every'] = 0",
    ]

    c7, c12 = _preset_call(v7), _preset_call(v12)
    assert ast.unparse(c12.args[0]) == "KnotTyingV12EnvCfg"
    assert [kw.arg for kw in c7.keywords] == [kw.arg for kw in c12.keywords] == ["play_post_init"]
    assert ast.unparse(c12.keywords[0].value) == "play_post_init"


def test_v12_agent_cfg_matches_v7_except_log_name():
    tree = _parse("v12/agents/rsl_rl_ppo_cfg.py")
    (cls,) = _classes(tree)
    assert [ast.unparse(b) for b in cls.bases] == ["_V7MLP"]
    assert _imported_as(tree, "_V7MLP") == ("v7.agents.rsl_rl_ppo_cfg", "KnotTyingPPORunnerCfg")
    assert [ast.unparse(b) for b in _body_without_docstring(cls)] == ["experiment_name = 'knot_tying_v12'"]
