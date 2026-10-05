"""v8: 環境は v7 と同じ。差は PPO の対称性のデータ拡張だけ (`agents/rsl_rl_ppo_cfg.py`)。

    uv run train --task Isaac-Twisted-RL-Knot-v8 --headless
    uv run play  --task Isaac-Twisted-RL-Knot-Play-v8 --num_envs 16

v7 から何を・なぜ変えたかは `README.md`。

環境の設定クラスを別に持つのは、タスク ID (`Isaac-Twisted-RL-*-v8`) を v7 と分けて登録するため
(`knot_tying/__init__.py` は `versions/v8/env_cfg.py` のプリセットを引く)。観測・報酬・終了・
アーカイブ・行動・物理・Play (chain モード) はすべて v7 のまま。

データ拡張は観測の項を名前で変換する (`common/topology/symmetry.py`)。観測の項を足す・変える
版を v8 から派生させるときは、その項の変換を決めること (決めていない項があると学習の最初の
更新で例外になる)。
"""

from __future__ import annotations

from isaaclab.utils import configclass

from ...common.rope_task_cfg import make_preset_cfgs
from ..v5.env_cfg import play_post_init as _play_post_init
from ..v7.env_cfg import KnotTyingV7EnvCfg

__all__ = ["KnotTyingV8EnvCfg"]


@configclass
class KnotTyingV8EnvCfg(KnotTyingV7EnvCfg):
    """v8 の環境設定。v7 と同じ。"""


globals().update(make_preset_cfgs(KnotTyingV8EnvCfg, __name__, play_post_init=_play_post_init))
