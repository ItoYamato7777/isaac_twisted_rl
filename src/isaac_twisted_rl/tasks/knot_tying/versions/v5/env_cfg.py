"""v5: v3 に pick&place の発散対策だけを足した版。

    uv run train --task Isaac-Twisted-RL-Knot-v5 --headless
    uv run play  --task Isaac-Twisted-RL-Knot-Play-v5 --num_envs 1

v3 から何を・なぜ変えたかは `README.md` (計測の表もそこにある)。ここは「何を配線したか」だけを書く。

v3 の設定クラスを **継承して差分だけ** を上書きする。報酬・終了・観測・アーカイブ・PPO は v3 のまま
(v3 の学習ログと並べて、発散対策の効果だけを比べるため):

1. 離すときの指先の高さを床から 5〜10 cm にする (`grasp_limits.release_tip_z`)。`drop_z` チャネルは
   この範囲に写る。v3 は床上 2 mm〜6.2 cm で、低く離す手が掴んだ紐を床や下の紐へ押し込んで発散していた。
2. ロープのめり込み解消速度の上限を 0.96 -> 0.4 m/s にする (`rope_max_depenetration_velocity`)。
   指や紐どうしが食い込んだときに、紐が一気に弾き出されて暴れるのを抑える。0.3 以下は把持が
   崩れ始める (指が閉じる速さに紐の押し戻しが追いつかない) ので、余裕を取って 0.4 にした。
3. v4 (4115db4) と同じ学習速度の改善を足す: PhysX のソルバの位置反復を 16 -> 8
   (`solver_position_iterations`)、1 iteration の手数を 12 -> 4 (`NUM_STEPS_PER_ENV`、
   `agents/` が読む)。README の「試した対策」の表では乱数の手でこの変更単体は不採用だったが、
   学習速度の改善幅 (約 2 倍) を優先して採用した。根拠は `../v4/README.md` の「学習速度」。
"""

from __future__ import annotations

from isaaclab.utils import configclass

from ...common.mdp import phases
from ...common.rope_task_cfg import make_preset_cfgs
from ..v3.env_cfg import GAMMA, KnotTyingV3EnvCfg
from ..v3.env_cfg import _play_post_init as _v3_play_post_init

__all__ = [
    "GAMMA",
    "NUM_STEPS_PER_ENV",
    "RELEASE_TIP_Z",
    "ROPE_MAX_DEPENETRATION_VELOCITY",
    "SAVE_INTERVAL",
    "KnotTyingV5EnvCfg",
]

NUM_STEPS_PER_ENV = 4
"""PPO の 1 iteration で各環境が打つ手の数 (`agents/rsl_rl_ppo_cfg.py` の `num_steps_per_env`)。

v3 から引き継いだ 12 のままだと 1 iteration が長い。v4 (4115db4) と同じ理由で 4 に下げる
(`../v4/env_cfg.py` の同名の定数)。
"""

SAVE_INTERVAL = 50
"""チェックポイントを保存する間隔 [iteration] (`agents/` の `save_interval`)。v3 と同じ値。"""

RELEASE_TIP_Z = (0.05, 0.10)
"""離すときの指先の高さの範囲 [m, 床から]。`drop_z` チャネルの [-1, 1] がこの範囲に写る。"""

ROPE_MAX_DEPENETRATION_VELOCITY = 0.4
"""ロープのめり込み解消速度の上限 [m/s]。v3 まで (`rope_cfg.py` の既定) は `Knot` で 0.96 m/s。

0.3 のほうが発散は少ないが、台本の固結びの再生が 59% -> 50% に下がり、0.25 で把持の成功率が
下がり始め、0.2 では把持の半分が失敗する。README の「めり込み解消速度をいくつにするか」。
"""


@configclass
class KnotTyingV5EnvCfg(KnotTyingV3EnvCfg):
    """v5 の環境設定。v3 との差は `grasp_limits`・`rope_max_depenetration_velocity`・
    `solver_position_iterations`。"""

    grasp_limits: phases.GraspLimits = phases.GraspLimits(drop_xy_rel_frac=0.30, release_tip_z=RELEASE_TIP_Z)
    """v3 と同じ置き先の範囲 (±0.30 L) に、離す高さの範囲を足す。"""

    rope_max_depenetration_velocity: float | None = ROPE_MAX_DEPENETRATION_VELOCITY

    solver_position_iterations: int | None = 8
    """ロープとグリッパーの PhysX ソルバの位置反復 (v3 まではロープ・グリッパーとも 16)。
    v4 (4115db4) と同じ値。計測と採否の議論は `../v4/README.md` の「学習速度」。
    """


globals().update(make_preset_cfgs(KnotTyingV5EnvCfg, __name__, play_post_init=_v3_play_post_init))
