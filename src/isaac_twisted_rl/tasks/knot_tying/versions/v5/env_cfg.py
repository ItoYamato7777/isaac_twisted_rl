"""v5: v4 (段ごとのエージェント・1 遷移 = 1 エピソード) に pick&place の発散対策を足した版。

    uv run train --task Isaac-Twisted-RL-Knot-v5 --headless
    uv run play  --task Isaac-Twisted-RL-Knot-Play-v5 --num_envs 16

v4 から何を・なぜ変えたかは `README.md` (計測の表もそこにある)。ここは「何を配線したか」だけを書く。

v4 の設定クラスを **継承して差分だけ** を上書きする。観測 (末尾の段の one-hot)・報酬 (+1 / 0 / −1)・
終了 (段が変われば終わる)・段ごとのアーカイブ・ログ・PPO (`StageGatedActorCritic`)・ソルバ反復 8・
1 iteration 4 手は v4 のまま:

1. 離すときの指先の高さを床から 5〜10 cm にする (`grasp_limits.release_tip_z`)。`drop_z` チャネルは
   この範囲に写る。v3 / v4 は床上 2 mm〜6.2 cm で、低く離す手が掴んだ紐を床や下の紐へ押し込んで発散していた。
2. ロープのめり込み解消速度の上限を 0.96 -> 0.4 m/s にする (`rope_max_depenetration_velocity`)。
   指や紐どうしが食い込んだときに、紐が一気に弾き出されて暴れるのを抑える。0.3 以下は把持が
   崩れ始める (指が閉じる速さに紐の押し戻しが追いつかない) ので、余裕を取って 0.4 にした。

Play は v4 と同じ chain モード (`_play_post_init`): まっすぐなロープから始め、固結び・発散・18 手で
だけ終わる。段が上がると次の段のエージェントに切り替わる。
"""

from __future__ import annotations

from isaaclab.utils import configclass

from ...common.mdp import phases
from ...common.rope_task_cfg import make_preset_cfgs
from ..v4.env_cfg import (
    GAMMA,
    CurriculumCfg,
    EventCfg,
    KnotTyingV4EnvCfg,
    ObservationsCfg,
    RewardsCfg,
    TerminationsCfg,
    _play_post_init,
)

__all__ = [
    "GAMMA",
    "RELEASE_TIP_Z",
    "ROPE_MAX_DEPENETRATION_VELOCITY",
    "CurriculumCfg",
    "EventCfg",
    "KnotTyingV5EnvCfg",
    "ObservationsCfg",
    "RewardsCfg",
    "TerminationsCfg",
]

RELEASE_TIP_Z = (0.05, 0.10)
"""離すときの指先の高さの範囲 [m, 床から]。`drop_z` チャネルの [-1, 1] がこの範囲に写る。"""

ROPE_MAX_DEPENETRATION_VELOCITY = 0.4
"""ロープのめり込み解消速度の上限 [m/s]。v4 まで (`rope_cfg.py` の既定) は `Knot` で 0.96 m/s。

0.3 のほうが発散は少ないが、台本の固結びの再生が 59% -> 50% に下がり、0.25 で把持の成功率が
下がり始め、0.2 では把持の半分が失敗する。README の「めり込み解消速度をいくつにするか」。
"""


@configclass
class KnotTyingV5EnvCfg(KnotTyingV4EnvCfg):
    """v5 の環境設定。v4 との差は `grasp_limits` と `rope_max_depenetration_velocity` だけ。

    観測・報酬・終了・イベント・カリキュラム・`actions_per_episode`・`solver_position_iterations`
    は v4 から継承する (ここで再定義しないこと。v4 を変えれば v5 も追従する)。
    """

    grasp_limits: phases.GraspLimits = phases.GraspLimits(drop_xy_rel_frac=0.30, release_tip_z=RELEASE_TIP_Z)
    """v3 / v4 と同じ置き先の範囲 (±0.30 L) に、離す高さの範囲を足す。"""

    rope_max_depenetration_velocity: float | None = ROPE_MAX_DEPENETRATION_VELOCITY


globals().update(make_preset_cfgs(KnotTyingV5EnvCfg, __name__, play_post_init=_play_post_init))
