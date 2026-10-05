"""v9: v7 の学習の初期状態で、段 2 を選ぶ重みを 2 倍にした版。

    uv run train --task Isaac-Twisted-RL-Knot-v9 --headless
    uv run play  --task Isaac-Twisted-RL-Knot-Play-v9 --num_envs 16

v7 から何を・なぜ変えたかは `README.md`。ここは「何を配線したか」だけを書く。

v7 の設定クラスを **継承して、`reset_to_stage` の `stage_weights` だけ** を変える (1 版 1 変数)。
観測・報酬・終了・アーカイブの中身・PPO・行動・物理・Play (chain モード) は v7 のまま。
v8 (対称性のデータ拡張) は継承しない。

v7 との差が `stage_weights` だけであることは `tests/test_v9_cfg.py` が静的に検査する。
"""

from __future__ import annotations

from isaaclab.utils import configclass

from ...common.rope_task_cfg import make_preset_cfgs
from ..v5.env_cfg import play_post_init as _play_post_init
from ..v7.env_cfg import KnotTyingV7EnvCfg

__all__ = ["STAGE_WEIGHTS", "KnotTyingV9EnvCfg"]

STAGE_WEIGHTS = (1.0, 1.0, 2.0)
"""学習の初期状態で段 0 / 1 / 2 を選ぶ重み。v4〜v8 は (1, 1, 1)。

段 2 から始めるエピソードは約 1/3 から 1/2 になる。段 0 を選んだうちの半分
(`straight_prob = 0.5`) がまっすぐなロープなので、まっすぐなロープから始める割合は
約 1/6 から 1/8 に下がる。どの段も空でないときの割合で、空の段は選ばれない (段 0 は空でも選ぶ。
`common/mdp/archive.py` の `ResetToStage`)。
"""


@configclass
class KnotTyingV9EnvCfg(KnotTyingV7EnvCfg):
    """v9 の環境設定。v7 との差は `events.reset_to_stage.params["stage_weights"]` だけ。"""

    def __post_init__(self):
        super().__post_init__()
        # Play はこの後に `_play_post_init` が (1, 0, 0) に上書きする (プリセットの __post_init__ が
        # 版の __post_init__ の後に Play の設定を掛ける。`common/rope_task_cfg.py` の `make_preset_cfgs`)。
        self.events.reset_to_stage.params["stage_weights"] = STAGE_WEIGHTS


globals().update(make_preset_cfgs(KnotTyingV9EnvCfg, __name__, play_post_init=_play_post_init))
