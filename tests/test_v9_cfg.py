"""v9 と v7 の設定の差が `reset_to_stage` の `stage_weights` だけであることを、静的に検査する。

v9 は 1 版 1 変数 (`docs/decisions/0012-version-is-config.md`) で、v7 との比較の結論は
「段 2 の重みだけが違う」ことに依っている。env の設定は Isaac Sim なしでは import できないので、
`ast` でソースを読んで次を確かめる:

- v9 の env の設定クラスは v7 のクラスだけを継承し、本体は `__post_init__` で `stage_weights` を
  1 か所書き換えるだけ。その値は (1, 1, 2)。
- v7 が継承する `stage_weights` は (1, 1, 1)。v4 の既定から、v5〜v7 が書き換えていない。
- Play の設定 (chain モード) とプリセットの生成は v7 と同じ関数を使う。
- v9 の PPO の設定は v7 のクラスを継承し、ログ名だけを変える (v8 の `symmetry_cfg` を持ち込まない)。

実際に組み立てた設定の差 (`params/env.yaml`) は、smoke test の run で確かめる (実験記録に書く)。
"""

from __future__ import annotations

import ast
from pathlib import Path

VERSIONS = Path(__file__).resolve().parents[1] / "src" / "isaac_twisted_rl" / "tasks" / "knot_tying" / "versions"


def _parse(rel: str) -> ast.Module:
    path = VERSIONS / rel
    return ast.parse(path.read_text(), filename=str(path))


def _classes(tree: ast.Module) -> list[ast.ClassDef]:
    return [n for n in tree.body if isinstance(n, ast.ClassDef)]


def _body_without_docstring(node: ast.ClassDef | ast.FunctionDef) -> list[ast.stmt]:
    body = list(node.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        body = body[1:]
    return body


def _module_assign(tree: ast.Module, name: str) -> ast.expr:
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets):
            return node.value
    raise AssertionError(f"{name} が最上位に無い")


def _preset_call(tree: ast.Module) -> ast.Call:
    """`globals().update(make_preset_cfgs(...))` の `make_preset_cfgs(...)`。"""
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "make_preset_cfgs":
            return node
    raise AssertionError("make_preset_cfgs の呼び出しが無い")


def _imported_as(tree: ast.Module, local: str) -> tuple[str | None, str]:
    """`from M import N as local` の (M, N)。"""
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if (alias.asname or alias.name) == local:
                    return node.module, alias.name
    raise AssertionError(f"{local} の import が無い")


def _stage_weight_writes(tree: ast.Module) -> list[ast.AST]:
    """`stage_weights` を書く箇所 (辞書の添字への代入と、辞書リテラルのキー)。"""
    hits: list[ast.AST] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if (
                    isinstance(t, ast.Subscript)
                    and isinstance(t.slice, ast.Constant)
                    and t.slice.value == "stage_weights"
                ):
                    hits.append(node)
        elif isinstance(node, ast.Dict):
            hits += [k for k in node.keys if isinstance(k, ast.Constant) and k.value == "stage_weights"]
    return hits


def test_v9_env_cfg_only_changes_stage_weights():
    tree = _parse("v9/env_cfg.py")
    (cls,) = _classes(tree)
    assert cls.name == "KnotTyingV9EnvCfg"
    assert [ast.unparse(b) for b in cls.bases] == ["KnotTyingV7EnvCfg"]
    assert _imported_as(tree, "KnotTyingV7EnvCfg") == ("v7.env_cfg", "KnotTyingV7EnvCfg")

    body = _body_without_docstring(cls)
    assert [type(b) for b in body] == [ast.FunctionDef], "クラスの本体は __post_init__ だけ (フィールドを上書きしない)"
    (post_init,) = body
    assert post_init.name == "__post_init__"
    stmts = _body_without_docstring(post_init)
    assert [ast.unparse(s) for s in stmts] == [
        "super().__post_init__()",
        "self.events.reset_to_stage.params['stage_weights'] = STAGE_WEIGHTS",
    ]
    assert ast.literal_eval(_module_assign(tree, "STAGE_WEIGHTS")) == (1.0, 1.0, 2.0)


def test_v7_inherits_uniform_stage_weights():
    # v4 の既定 (学習) が (1, 1, 1) で、Play の chain モードが (1, 0, 0) に書き換える。
    v4 = _parse("v4/env_cfg.py")
    reset = next(
        n
        for n in ast.walk(v4)
        if isinstance(n, ast.Assign) and any(getattr(t, "id", None) == "reset_to_stage" for t in n.targets)
    )
    params = next(kw.value for kw in reset.value.keywords if kw.arg == "params")
    weights = dict(zip([ast.literal_eval(k) for k in params.keys if k is not None], params.values, strict=False))
    assert ast.literal_eval(weights["stage_weights"]) == (1.0, 1.0, 1.0)
    assert len(_stage_weight_writes(v4)) == 2  # 既定と Play の 2 か所

    # v5〜v7 は stage_weights を書かない (v7 は v4 の既定をそのまま継承する)。
    for ver in ("v5", "v6", "v7"):
        assert _stage_weight_writes(_parse(f"{ver}/env_cfg.py")) == [], ver


def test_v9_presets_and_play_match_v7():
    v7, v9 = _parse("v7/env_cfg.py"), _parse("v9/env_cfg.py")
    c7, c9 = _preset_call(v7), _preset_call(v9)
    assert ast.unparse(c7.args[0]) == "KnotTyingV7EnvCfg"
    assert ast.unparse(c9.args[0]) == "KnotTyingV9EnvCfg"
    assert [(kw.arg, ast.unparse(kw.value)) for kw in c7.keywords] == [
        (kw.arg, ast.unparse(kw.value)) for kw in c9.keywords
    ]
    assert _imported_as(v7, "_play_post_init") == _imported_as(v9, "_play_post_init")


def test_v9_agent_cfg_matches_v7_except_log_name():
    tree = _parse("v9/agents/rsl_rl_ppo_cfg.py")
    (cls,) = _classes(tree)
    assert [ast.unparse(b) for b in cls.bases] == ["_V7MLP"]
    assert _imported_as(tree, "_V7MLP") == ("v7.agents.rsl_rl_ppo_cfg", "KnotTyingPPORunnerCfg")
    body = _body_without_docstring(cls)
    assert [ast.unparse(b) for b in body] == ["experiment_name = 'knot_tying_v9'"]
