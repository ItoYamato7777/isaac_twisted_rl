"""v13 と v12 の設定の差が「PPO の損失の平均の範囲外の罰則」だけであることを、静的に検査する。

v13 は 1 版 1 変数 (`docs/decisions/0012-version-is-config.md`) で、v12 との比較の結論は「PPO の損失に罰則が
あるかだけが違う」ことに依っている (`docs/decisions/0026-v13-action-bounds-loss.md`)。設定は Isaac Sim なしでは
import できないので、`ast` でソースを読んで次を確かめる:

- v13 の env の設定クラスは v12 のクラスだけを継承し、本体は docstring だけ。Play は v12 の `play_post_init`。
- v13 の PPO の設定は v12 のクラスを継承し、ログ名と `__post_init__` だけを書く。`__post_init__` は v12 の
  algorithm のフィールドを全部引き継ぎ、クラスを `BoundedPpoAlgorithmCfg` に替えて `bounds_loss_coef` を足すだけ。
- `BoundedPpoAlgorithmCfg` の `class_name` は `BoundedPPO`、フィールドの名前は `BoundedPPO.__init__` の引数と同じ。
  係数は 0.01。モジュールの読み込みで `register_with_rsl_rl()` を呼ぶ。

`BoundedPPO` の損失と勾配は `tests/test_bounded_ppo.py`、組み立てた設定 (`params/agent.yaml`) は smoke test の run で確かめる。
"""

from __future__ import annotations

import ast
import inspect

from policies.bounded_ppo import BoundedPPO
from test_v9_cfg import _body_without_docstring, _classes, _imported_as, _module_assign, _parse, _preset_call


def test_v13_env_cfg_is_v12():
    tree = _parse("v13/env_cfg.py")
    (cls,) = _classes(tree)
    assert cls.name == "KnotTyingV13EnvCfg"
    assert [ast.unparse(b) for b in cls.bases] == ["KnotTyingV12EnvCfg"]
    assert _imported_as(tree, "KnotTyingV12EnvCfg") == ("v12.env_cfg", "KnotTyingV12EnvCfg")
    assert _body_without_docstring(cls) == [], "クラスの本体は docstring だけ"

    assert _imported_as(tree, "play_post_init") == ("v12.env_cfg", "play_post_init")
    assert not [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    call = _preset_call(tree)
    assert ast.unparse(call.args[0]) == "KnotTyingV13EnvCfg"
    assert [(kw.arg, ast.unparse(kw.value)) for kw in call.keywords] == [("play_post_init", "play_post_init")]


def test_v13_agent_cfg_only_adds_bounds_loss():
    tree = _parse("v13/agents/rsl_rl_ppo_cfg.py")
    alg_cls, runner = _classes(tree)

    assert runner.name == "KnotTyingPPORunnerCfg"
    assert [ast.unparse(b) for b in runner.bases] == ["_V12MLP"]
    assert _imported_as(tree, "_V12MLP") == ("v12.agents.rsl_rl_ppo_cfg", "KnotTyingPPORunnerCfg")
    body = _body_without_docstring(runner)
    assert [ast.unparse(b) for b in body[:1]] == ["experiment_name = 'knot_tying_v13'"]
    assert [type(b) for b in body[1:]] == [ast.FunctionDef] and body[1].name == "__post_init__"
    assert [ast.unparse(s) for s in _body_without_docstring(body[1])] == [
        "super().__post_init__()",
        "values = {f.name: getattr(self.algorithm, f.name) for f in dataclasses.fields(self.algorithm)}",
        "del values['class_name']",
        "self.algorithm = BoundedPpoAlgorithmCfg(**values, bounds_loss_coef=BOUNDS_LOSS_COEF)",
    ]

    assert alg_cls.name == "BoundedPpoAlgorithmCfg"
    assert [ast.unparse(b) for b in alg_cls.bases] == ["RslRlPpoAlgorithmCfg"]
    fields = {
        s.target.id: ast.literal_eval(s.value) for s in _body_without_docstring(alg_cls) if isinstance(s, ast.AnnAssign)
    }
    assert fields == {"class_name": "BoundedPPO", "bounds_loss_coef": 0.0}
    assert ast.literal_eval(_module_assign(tree, "BOUNDS_LOSS_COEF")) == 0.01


def test_v13_alg_cfg_matches_bounded_ppo():
    tree = _parse("v13/agents/rsl_rl_ppo_cfg.py")
    assert "bounds_loss_coef" in inspect.signature(BoundedPPO.__init__).parameters
    # モジュールの最上位で登録する (runner が eval(class_name) で引く前に必ず読まれる)。
    assert _imported_as(tree, "register_with_rsl_rl") == ("common.policies.bounded_ppo", "register_with_rsl_rl")
    calls = [ast.unparse(n) for n in tree.body if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)]
    assert calls == ["register_with_rsl_rl()"]


def test_v12_agent_cfg_has_plain_ppo():
    # v13 が継承する v12 の algorithm は v2 の `_algorithm` (`RslRlPpoAlgorithmCfg`、class_name は既定の "PPO")。
    for ver in ("v2_r2", "v3", "v4", "v5", "v6", "v7", "v12"):
        tree = _parse(f"{ver}/agents/rsl_rl_ppo_cfg.py")
        targets = [ast.unparse(t) for n in ast.walk(tree) if isinstance(n, ast.Assign) for t in n.targets]
        assert not [t for t in targets if "algorithm" in t], ver
        src = ast.unparse(tree)
        assert "bounds_loss" not in src and "BoundedPPO" not in src, ver
