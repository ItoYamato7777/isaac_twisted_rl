"""紐モデルの比較 (rope study): 同じ MDP (v4) のまま紐のシミュレーションモデルだけを差し替えたタスク群。

    Isaac-Twisted-RL-RS-<モデル>-v4        学習
    Isaac-Twisted-RL-RS-<モデル>-Play-v4   chain 評価 (まっすぐな紐から 18 手)

モデルの一覧は `models.py`、何を比べたかと結果は `Report_rope_models.md`、使い方は `README.md`。

`../__init__.py` の `_VERSIONS` には足さない (v0〜v7 は「MDP の版」で、ここは「紐の版」なので軸が違う)。
この __init__ は Kit 起動前に import されるので、isaaclab を import しないこと (`models.py` は純粋な dataclass)。
"""

import gymnasium as gym

from .models import MODEL_NAMES

_ENV_ENTRY = "isaac_twisted_rl.tasks.knot_tying.common.rope_backends.env:RopeStudyEnv"

for _name in MODEL_NAMES:
    for _play in ("", "-Play"):
        gym.register(
            id=f"Isaac-Twisted-RL-RS-{_name}{_play}-v4",
            entry_point=_ENV_ENTRY,
            kwargs={
                "env_cfg_entry_point": f"{__name__}.env_cfg:{_name}{'Play' if _play else ''}EnvCfg",
                "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:{_name}PPORunnerCfg",
            },
            disable_env_checker=True,
        )
