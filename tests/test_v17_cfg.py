"""v17 (待ちのフェーズの尺を詰める) の設定を、Isaac Sim なしで検査する。

- v17 の設定は v15 のクラスを継承し、`durations` だけを差し替える (`ast` で読む)。尺の値は 0030 の表 (D4)。
- 尺から決まる 1 手の物理ステップ数 (`decimation`) が 591、フェーズごとの内訳が 0030 の表と合う。
- 手数の上限 (Isaac Lab の `ceil(episode_length_s / (dt * decimation))`) が、学習 6・Play 18 になる。Play は
  v4 の式 (`18 * decimation * dt`) のままだと 591 で 19 になるので、v17 の `play_post_init` が 17.5 手ぶんにする。
- PPO の設定は v15 のクラスを継承し、ログ名だけを変える。

組み立てた設定 (`params/env.yaml` の `durations`・`actions.pick_place.durations`・`decimation`) と、起動した環境の
`max_episode_length` は、`bench_step_time.py` の `meta` とスモークテストで確かめる。
"""

from __future__ import annotations

import ast
import math

import phases
from test_v9_cfg import _body_without_docstring, _classes, _imported_as, _parse, _preset_call

SIM_DT = 1.0 / 240.0
"""基底の `__post_init__` の `sim.dt`。"""

D4 = {"close": 0.2, "hold": 0.1, "release": 0.15, "settle_out": 0.15, "relax": 0.25}
"""0030 の表で詰めたフェーズ。ほかは v3 の `KNOT_DURATIONS` のまま。"""


def _durations_kwargs(rel: str, name: str) -> dict[str, float]:
    """`name = phases.PhaseDurations(...)` のキーワード引数。"""
    tree = _parse(rel)
    for node in tree.body:
        if isinstance(node, ast.Assign) and [ast.unparse(t) for t in node.targets] == [name]:
            call = node.value
            assert ast.unparse(call.func) == "phases.PhaseDurations"
            assert not call.args
            return {kw.arg: ast.literal_eval(kw.value) for kw in call.keywords}
    raise AssertionError(f"{rel} に {name} が無い")


def _max_episode_length(moves: float, decimation: int) -> int:
    """Isaac Lab の `ManagerBasedRLEnv.max_episode_length` と同じ式 (`episode_length_s` は版の設定と同じ積の順)。"""
    episode_length_s = moves * decimation * SIM_DT
    return math.ceil(episode_length_s / (SIM_DT * decimation))


def test_v17_durations_are_d4():
    v17 = _durations_kwargs("v17/env_cfg.py", "V17_DURATIONS")
    # 全項目を明示する (`PhaseDurations` の既定値に依らない)。
    assert set(v17) == set(phases.PHASE_NAMES)
    v3 = phases.PhaseDurations(**_durations_kwargs("v3/env_cfg.py", "KNOT_DURATIONS"))
    for name in phases.PHASE_NAMES:
        want = D4.get(name, getattr(v3, name))
        assert v17[name] == want, name


def test_v17_step_counts():
    v17 = phases.PhaseDurations(**_durations_kwargs("v17/env_cfg.py", "V17_DURATIONS"))
    v3 = phases.PhaseDurations(**_durations_kwargs("v3/env_cfg.py", "KNOT_DURATIONS"))
    assert v3.decimation(SIM_DT) == 759
    assert v17.decimation(SIM_DT) == 591
    steps = dict(zip(phases.PHASE_NAMES, v17.step_counts(SIM_DT)))
    assert steps == {
        "approach": 1,
        "descend": 72,
        "close": 48,
        "hold": 24,
        "lift": 72,
        "carry": 144,
        "lower": 72,
        "release": 36,
        "settle_out": 36,
        "clear": 1,
        "retreat": 25,
        "relax": 60,
    }


def test_episode_length_moves():
    # 学習 (基底の `actions_per_episode * decimation * dt`) は 591 でも 6 ちょうど。
    assert _max_episode_length(6, 591) == 6
    # Play を v4 の式のままにすると 19 手になる (759 では 18)。これが v17 の play_post_init の理由。
    assert _max_episode_length(18, 759) == 18
    assert _max_episode_length(18, 591) == 19
    assert _max_episode_length(17.5, 591) == 18


def test_v17_env_cfg_only_changes_durations():
    tree = _parse("v17/env_cfg.py")
    (cfg,) = _classes(tree)
    assert cfg.name == "KnotTyingV17EnvCfg"
    assert [ast.unparse(b) for b in cfg.bases] == ["KnotTyingV15EnvCfg"]
    assert _imported_as(tree, "KnotTyingV15EnvCfg") == ("v15.env_cfg", "KnotTyingV15EnvCfg")
    assert [ast.unparse(s) for s in _body_without_docstring(cfg)] == [
        "durations: phases.PhaseDurations = V17_DURATIONS"
    ]
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "play_post_init")
    assert _imported_as(tree, "_v15_play_post_init") == ("v15.env_cfg", "play_post_init")
    assert [ast.unparse(s) for s in _body_without_docstring(fn)] == [
        "_v15_play_post_init(cfg)",
        "cfg.episode_length_s = (cfg.actions_per_episode - 0.5) * cfg.decimation * cfg.sim.dt",
    ]
    call = _preset_call(tree)
    assert ast.unparse(call.args[0]) == "KnotTyingV17EnvCfg"
    assert [ast.unparse(k.value) for k in call.keywords if k.arg == "play_post_init"] == ["play_post_init"]


def test_v17_agent_cfg_only_sets_log_name():
    tree = _parse("v17/agents/rsl_rl_ppo_cfg.py")
    (runner,) = _classes(tree)
    assert [ast.unparse(b) for b in runner.bases] == ["_V15MLP"]
    assert _imported_as(tree, "_V15MLP") == ("v15.agents.rsl_rl_ppo_cfg", "KnotTyingPPORunnerCfg")
    assert [ast.unparse(b) for b in _body_without_docstring(runner)] == ["experiment_name = 'knot_tying_v17'"]
