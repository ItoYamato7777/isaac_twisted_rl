"""v14 と v13 の設定の差が「保存の規則・ロープの物理の DR・critic の特権観測」の 3 つだけであることを、静的に検査する。

v14 は 1 版 1 変数から外れる (ユーザーが了承済み) が、3 つとも Hydra で個別に切れることが前提になっている
(`docs/decisions/0027-v14-s0-reshape-dr-priv-critic.md`)。設定は Isaac Sim なしでは import できないので、
`ast` でソースを読んで次を確かめる:

- env の設定クラスは v13 のクラスだけを継承し、本体は `observations`・`events` の差し替えと、`__post_init__` で
  `record_stage_archive` の `save_on`・`s0_min_change` を書くことだけ。
- `EventCfg` は v4 (= v13) の `EventCfg` に `rope_dr` だけを足す。`ObservationsCfg` は v6 (= v13) の
  `ObservationsCfg` に `rope_params` だけを足す。
- Play は v13 の `play_post_init` を呼んでから DR を切るだけ。
- agent の設定は v13 のクラスを継承し、`experiment_name` と `obs_groups` だけを書く。critic の観測は特権観測が先。
- v2〜v13 に `rope_dr`・`rope_params`・`stage_up_s0_reshape` が無い (既存の版の挙動を変えていない)。

保存の規則は `tests/test_state_archive.py`、DR の値は `tests/test_domain_rand.py`、critic の幅は
`tests/test_stage_gated_policy.py`、組み立てた設定 (`params/env.yaml`・`params/agent.yaml`) は smoke test の run で確かめる。
"""

from __future__ import annotations

import ast
import re

import domain_rand as dr
import state_archive as sa
from test_v9_cfg import VERSIONS, _body_without_docstring, _classes, _imported_as, _module_assign, _parse, _preset_call


def _class(tree: ast.Module, name: str) -> ast.ClassDef:
    return next(c for c in _classes(tree) if c.name == name)


def test_v14_env_cfg_inherits_v13():
    tree = _parse("v14/env_cfg.py")
    assert [c.name for c in _classes(tree)] == ["EventCfg", "ObservationsCfg", "KnotTyingV14EnvCfg"]
    cls = _class(tree, "KnotTyingV14EnvCfg")
    assert [ast.unparse(b) for b in cls.bases] == ["KnotTyingV13EnvCfg"]
    assert _imported_as(tree, "KnotTyingV13EnvCfg") == ("v13.env_cfg", "KnotTyingV13EnvCfg")

    body = _body_without_docstring(cls)
    assert [ast.unparse(b) for b in body[:2]] == [
        "observations: ObservationsCfg = ObservationsCfg()",
        "events: EventCfg = EventCfg()",
    ]
    assert [type(b) for b in body[2:]] == [ast.FunctionDef] and body[2].name == "__post_init__"
    assert [ast.unparse(s) for s in _body_without_docstring(body[2])] == [
        "super().__post_init__()",
        "self.rewards.record_stage_archive.params['save_on'] = SAVE_ON",
        "self.rewards.record_stage_archive.params['s0_min_change'] = S0_MIN_CHANGE",
    ]
    assert ast.literal_eval(_module_assign(tree, "SAVE_ON")) == "stage_up_s0_reshape"
    assert ast.literal_eval(_module_assign(tree, "SAVE_ON")) in sa.SAVE_RULES
    assert ast.literal_eval(_module_assign(tree, "S0_MIN_CHANGE")) == 0.02


def test_v14_events_add_only_rope_dr():
    tree = _parse("v14/env_cfg.py")
    ev = _class(tree, "EventCfg")
    assert [ast.unparse(b) for b in ev.bases] == ["V4EventCfg"]
    assert _imported_as(tree, "V4EventCfg") == ("v4.env_cfg", "EventCfg")
    (assign,) = _body_without_docstring(ev)
    assert ast.unparse(assign) == (
        "rope_dr = EventTerm(func=mdp.RandomizeRopePhysics, mode='reset', params={'enabled': True, 'ranges': DR_RANGES})"
    )
    ranges = ast.literal_eval(_module_assign(tree, "DR_RANGES"))
    assert tuple(ranges) == dr.AXES
    # 減衰比だけ下限を ×1.0 に上げた (ゲートで ζ の小さい側の発散が公称の 2.4 倍だったため)。
    assert ranges.pop("damping_ratio") == (1.0, 2.0)
    assert all(v == (0.5, 2.0) for v in ranges.values())
    # v5〜v13 は events を差し替えない (v13 の events は v4 の EventCfg)。
    for ver in ("v5", "v6", "v7", "v12", "v13"):
        src = ast.unparse(_parse(f"{ver}/env_cfg.py"))
        assert "events:" not in src, ver


def test_v14_observations_add_only_rope_params():
    tree = _parse("v14/env_cfg.py")
    obs = _class(tree, "ObservationsCfg")
    assert [ast.unparse(b) for b in obs.bases] == ["V6ObservationsCfg"]
    assert _imported_as(tree, "V6ObservationsCfg") == ("v6.env_cfg", "ObservationsCfg")
    group, field = _body_without_docstring(obs)
    assert isinstance(group, ast.ClassDef) and group.name == "RopeParamsCfg"
    assert ast.unparse(field) == "rope_params: RopeParamsCfg = RopeParamsCfg()"
    term, post = _body_without_docstring(group)
    assert ast.unparse(term) == (
        "rope_physics = ObsTerm(func=mdp.rope_physics_params, params={'event_term': 'rope_dr'})"
    )
    assert [ast.unparse(s) for s in post.body] == ["self.enable_corruption = False", "self.concatenate_terms = True"]
    # v7〜v13 は observations を差し替えない (v13 の observations は v6 の ObservationsCfg)。
    for ver in ("v7", "v12", "v13"):
        assert "observations:" not in ast.unparse(_parse(f"{ver}/env_cfg.py")), ver


def test_v14_play_extends_v13_play_only_by_disabling_dr():
    tree = _parse("v14/env_cfg.py")
    assert _imported_as(tree, "_v13_play_post_init") == ("v13.env_cfg", "play_post_init")
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "play_post_init")
    assert [ast.unparse(s) for s in _body_without_docstring(fn)] == [
        "_v13_play_post_init(cfg)",
        "cfg.events.rope_dr.params['enabled'] = False",
    ]
    call = _preset_call(tree)
    assert ast.unparse(call.args[0]) == "KnotTyingV14EnvCfg"
    assert [(kw.arg, ast.unparse(kw.value)) for kw in call.keywords] == [("play_post_init", "play_post_init")]


def test_v14_agent_cfg_only_sets_log_name_and_critic_obs():
    tree = _parse("v14/agents/rsl_rl_ppo_cfg.py")
    (runner,) = _classes(tree)
    assert runner.name == "KnotTyingPPORunnerCfg"
    assert [ast.unparse(b) for b in runner.bases] == ["_V13MLP"]
    assert _imported_as(tree, "_V13MLP") == ("v13.agents.rsl_rl_ppo_cfg", "KnotTyingPPORunnerCfg")
    assert [ast.unparse(b) for b in _body_without_docstring(runner)] == [
        "experiment_name = 'knot_tying_v14'",
        "obs_groups = {'policy': ['policy'], 'critic': ['rope_params', 'policy']}",
    ]


def test_older_versions_do_not_use_v14_terms():
    # v14 より後の版 (v15〜) は v14 を継ぐので名前を使ってよい (v16 は `rope_params` に寸法の項を足す)。
    for path in sorted(VERSIONS.glob("v*/**/*.py")):
        ver = path.relative_to(VERSIONS).parts[0]
        num = re.match(r"v(\d+)", ver)
        if num and int(num.group(1)) >= 14:
            continue
        src = path.read_text()
        for word in ("rope_dr", "rope_params", "RandomizeRopePhysics", "stage_up_s0_reshape", "s0_min_change"):
            assert word not in src, (path, word)
