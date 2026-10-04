"""v7: v6 のロープのめり込み解消速度を v4 の既定に戻した版。

    uv run train --task Isaac-Twisted-RL-Knot-v7 --headless
    uv run play  --task Isaac-Twisted-RL-Knot-Play-v7 --num_envs 16

v6 から何を・なぜ変えたかは `README.md`。ここは「何を配線したか」だけを書く。

v6 の設定クラスを **継承して、ロープの物性 1 つだけ** を変える (1 版 1 変数)。観測 (`next_edge` + 段の
one-hot)・報酬・終了・段ごとのアーカイブ・PPO・行動 (離す高さ 5〜10 cm を含む)・Play (chain モード) は v6 のまま:

1. ロープのめり込み解消速度の上限を v5 / v6 の 0.4 m/s から、ロープの既定 (`rope_cfg.py` が半径から
   決める値。`Knot` で 0.96 m/s、v4 と同じ) に戻す (`rope_max_depenetration_velocity = None`)。
   v6 の切り分け (`../v6/Result_Report_v6.md` の追記) で、0.4 m/s が段 2 -> 固結びの学習を
   止めていた原因だと分かった。
"""

from __future__ import annotations

from isaaclab.utils import configclass

from ...common.rope_task_cfg import make_preset_cfgs
from ..v5.env_cfg import play_post_init as _play_post_init
from ..v6.env_cfg import KnotTyingV6EnvCfg

__all__ = ["KnotTyingV7EnvCfg"]


@configclass
class KnotTyingV7EnvCfg(KnotTyingV6EnvCfg):
    """v7 の環境設定。v6 との差は `rope_max_depenetration_velocity` だけ。"""

    rope_max_depenetration_velocity: float | None = None
    """`None` = 上書きしない (ロープの既定。`Knot` で 0.96 m/s)。v5 / v6 は 0.4 m/s。

    v6 の切り分け (seed 42、学習の iteration 34〜43、v6 に 1 つずつ入れた):

    | | 段 2 -> 固結びの成功 | 段 2 の時間切れ | 発散で終了 |
    |---|---|---|---|
    | v6 (0.4 m/s) | 0.04 | 0.30 | 1.6% |
    | v6 + 0.96 m/s (= v7) | **0.14** | **0.03** | 3.7% |
    | v6 + 離す高さを v4 に戻す | 0.06 | 0.13 | 3.0% |
    | v4 (0.96 m/s、低く離す) | 0.15 | 0.03 | 5.5% |

    離す高さ (`grasp_limits.release_tip_z`、5〜10 cm) は v6 のまま残す。学習を妨げず、v4 からの
    発散の減少分のおよそ半分を保つため。
    """


globals().update(make_preset_cfgs(KnotTyingV7EnvCfg, __name__, play_post_init=_play_post_init))
