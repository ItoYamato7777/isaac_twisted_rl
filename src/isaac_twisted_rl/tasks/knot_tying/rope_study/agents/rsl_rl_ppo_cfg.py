"""紐モデルの比較 (`rope_study`) の PPO 設定。**v4 と同一** で、ログのフォルダだけをモデルごとに分ける。

方策 (段ごとの MLP `StageGatedActorCritic`)・学習率・エポック・1 iteration 4 手・行動の初期 std 0.5 は
v4 のまま (`../../versions/v4/agents/rsl_rl_ppo_cfg.py`)。モデル間で学習アルゴリズムを変えないことで、
差を紐の表現に帰属させる。ログは `logs/rsl_rl/knot_tying_rs_<モデル>/`。
"""

from isaaclab.utils import configclass

from ...versions.v4.agents.rsl_rl_ppo_cfg import KnotTyingPPORunnerCfg as _V4
from ..models import MODELS


def _make() -> dict[str, type]:
    out = {}
    for name in MODELS:
        cls = configclass(
            type(
                f"{name}PPORunnerCfg",
                (_V4,),
                {
                    "__module__": __name__,
                    "__doc__": f"{name} の PPO 設定 (v4 と同一)",
                    "experiment_name": f"knot_tying_rs_{name.lower()}",
                },
            )
        )
        out[f"{name}PPORunnerCfg"] = cls
    return out


globals().update(_make())
