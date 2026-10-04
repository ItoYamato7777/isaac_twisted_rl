"""登録されたタスク ID と、その entry_point・kwargs (設定の entry point の文字列) を JSON で出す。

Kit なしで動く: knot_tying/__init__.py は gym.register を文字列で行い、isaaclab を import しない (不変条件 I1)。
Phase 4 の各変更の前後で出力を diff し、タスク ID と設定の参照先が変わっていないことを確かめる。
"""
import json
import sys
import types

import gymnasium as gym

# isaaclab_tasks は Kit の外では import できない。tasks/__init__.py が使う import_packages だけを空にし、
# knot_tying を直接 import する (登録そのものは本物)。
_stub = types.ModuleType("isaaclab_tasks.utils")
_stub.import_packages = lambda *a, **k: None
sys.modules["isaaclab_tasks"] = types.ModuleType("isaaclab_tasks")
sys.modules["isaaclab_tasks.utils"] = _stub

import isaac_twisted_rl.tasks.knot_tying  # noqa: E402, F401
import isaac_twisted_rl.tasks.knot_tying.rope_study  # noqa: E402, F401

out = {}
for tid, spec in sorted(gym.registry.items()):
    if not tid.startswith("Isaac-Twisted-RL"):
        continue
    out[tid] = {"entry_point": str(spec.entry_point), "kwargs": {k: str(v) for k, v in sorted(spec.kwargs.items())},
                "disable_env_checker": spec.disable_env_checker}
json.dump(out, sys.stdout, indent=1, ensure_ascii=False, sort_keys=True)
print()
