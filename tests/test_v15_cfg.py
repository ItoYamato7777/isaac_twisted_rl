"""v15 と v14 の設定の差が「終了項 `time_out` の関数の差し替え」だけであることを、静的に検査する。

v15 は 1 版 1 変数 (`docs/decisions/0028-v15-timeout-bootstrap-fix.md`) で、v14 との比較の結論は「時間切れの
bootstrap の不具合を直したかだけが違う」ことに依っている。設定は Isaac Sim なしでは import できないので、`ast` で
ソースを読んで次を確かめる:

- env の設定クラスは v14 のクラスだけを継承し、本体は `__post_init__` で `terminations.time_out` を
  `mdp.time_out_unless_terminated` (`time_out=True`) に差し替えることだけ。除く項は v4 の終了項の残り全部。
- Play は v14 の `play_post_init` を呼んでから `time_out` を `mdp.time_out` に戻すだけ (chain 評価の定義を揃える)。
- agent の設定は v14 のクラスを継承し、`experiment_name` だけを書く。
- v2〜v14 は `time_out_unless_terminated` を使わない (既存の版の挙動を変えていない)。

終了項の関数そのものは Isaac Sim が要るので、`terminated & truncated` の数と組み立てた設定 (`params/env.yaml`) は
ゲートの run で確かめる。
"""

from __future__ import annotations

import ast

from test_v9_cfg import VERSIONS, _body_without_docstring, _classes, _imported_as, _module_assign, _parse, _preset_call

_TERMINATIONS = VERSIONS.parent / "common" / "mdp" / "terminations.py"


def test_v15_env_cfg_only_replaces_time_out():
    tree = _parse("v15/env_cfg.py")
    (cls,) = _classes(tree)
    assert cls.name == "KnotTyingV15EnvCfg"
    assert [ast.unparse(b) for b in cls.bases] == ["KnotTyingV14EnvCfg"]
    assert _imported_as(tree, "KnotTyingV14EnvCfg") == ("v14.env_cfg", "KnotTyingV14EnvCfg")

    (post,) = _body_without_docstring(cls)
    assert isinstance(post, ast.FunctionDef) and post.name == "__post_init__"
    assert [ast.unparse(s) for s in _body_without_docstring(post)] == [
        "super().__post_init__()",
        "self.terminations.time_out = DoneTerm(func=mdp.time_out_unless_terminated, time_out=True, "
        "params={'exclude_terms': EXCLUDE_TERMS})",
    ]
    assert _imported_as(tree, "DoneTerm") == ("isaaclab.managers", "TerminationTermCfg")


def test_v15_exclude_terms_are_the_other_v4_terminations():
    # v4 の終了項は time_out・diverged (v0 から)・stage_up・stage_down (v4 で追加。v3 の knot_tied は None)。
    tree = _parse("v15/env_cfg.py")
    assert ast.literal_eval(_module_assign(tree, "EXCLUDE_TERMS")) == ("diverged", "stage_up", "stage_down")
    v4 = next(c for c in _classes(_parse("v4/env_cfg.py")) if c.name == "TerminationsCfg")
    fields = [ast.unparse(s.targets[0]) for s in _body_without_docstring(v4) if isinstance(s, ast.Assign)]
    assert fields == ["knot_tied", "stage_up", "stage_down"]
    # v5〜v14 は terminations を差し替えない (v14 の terminations は v4 の TerminationsCfg)。
    for ver in ("v5", "v6", "v7", "v12", "v13", "v14"):
        src = ast.unparse(_parse(f"{ver}/env_cfg.py"))
        assert "terminations:" not in src and "terminations.time_out" not in src, ver


def test_v15_play_restores_v14_time_out():
    tree = _parse("v15/env_cfg.py")
    assert _imported_as(tree, "_v14_play_post_init") == ("v14.env_cfg", "play_post_init")
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "play_post_init")
    assert [ast.unparse(s) for s in _body_without_docstring(fn)] == [
        "_v14_play_post_init(cfg)",
        "cfg.terminations.time_out = DoneTerm(func=mdp.time_out, time_out=True)",
    ]
    call = _preset_call(tree)
    assert ast.unparse(call.args[0]) == "KnotTyingV15EnvCfg"
    assert [(kw.arg, ast.unparse(kw.value)) for kw in call.keywords] == [("play_post_init", "play_post_init")]


def test_v15_agent_cfg_only_sets_log_name():
    tree = _parse("v15/agents/rsl_rl_ppo_cfg.py")
    (runner,) = _classes(tree)
    assert runner.name == "KnotTyingPPORunnerCfg"
    assert [ast.unparse(b) for b in runner.bases] == ["_V14MLP"]
    assert _imported_as(tree, "_V14MLP") == ("v14.agents.rsl_rl_ppo_cfg", "KnotTyingPPORunnerCfg")
    assert [ast.unparse(b) for b in _body_without_docstring(runner)] == ["experiment_name = 'knot_tying_v15'"]


def test_time_out_unless_terminated_signature():
    # 既定の除く項が v15 の EXCLUDE_TERMS と同じ (Hydra で params を消しても挙動が変わらない)。
    tree = ast.parse(_TERMINATIONS.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "time_out_unless_terminated")
    assert [a.arg for a in fn.args.args] == ["env", "exclude_terms"]
    assert ast.literal_eval(fn.args.defaults[0]) == ("diverged", "stage_up", "stage_down")


def test_older_versions_do_not_use_v15_terms():
    for path in sorted(VERSIONS.glob("v*/**/*.py")):
        if "v15" in path.parts:
            continue
        assert "time_out_unless_terminated" not in path.read_text(), path
