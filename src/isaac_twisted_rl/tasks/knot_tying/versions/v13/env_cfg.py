"""v13: v12 + PPO の損失に「方策の平均が行動の範囲 [−1, 1] の外へ出た分」の罰則を足す版。

    uv run train --task Isaac-Twisted-RL-Knot-v13 --headless
    uv run play  --task Isaac-Twisted-RL-Knot-Play-v13 --num_envs 16

v12 から何を・なぜ変えたかは `README.md`、仕様は `docs/decisions/0026-v13-action-bounds-loss.md`。

v13 で変わるのは PPO の損失だけ (`agents/rsl_rl_ppo_cfg.py`)。環境の設定 (観測・報酬・終了・アーカイブの保存の規則
`stage_up`・物理) は v12 のままなので、v12 のクラスを本体なしで継承し、Play も v12 の `play_post_init` を使う。
クラスを分けるのは、タスク ID ごとに設定のクラスを持つ (`knot_tying/__init__.py` の登録) ためだけ。
v12 との差が無いことは `tests/test_v13_cfg.py` が静的に検査する。
"""

from __future__ import annotations

from isaaclab.utils import configclass

from ...common.rope_task_cfg import make_preset_cfgs
from ..v12.env_cfg import KnotTyingV12EnvCfg, play_post_init

__all__ = ["KnotTyingV13EnvCfg", "play_post_init"]


@configclass
class KnotTyingV13EnvCfg(KnotTyingV12EnvCfg):
    """v13 の環境設定。v12 と同じ。"""


globals().update(make_preset_cfgs(KnotTyingV13EnvCfg, __name__, play_post_init=play_post_init))
