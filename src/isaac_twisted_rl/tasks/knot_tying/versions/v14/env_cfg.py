"""v14: v13 + 段 0 の保存の規則 + ロープの物理の DR + critic の特権観測。

    uv run train --task Isaac-Twisted-RL-Knot-v14 --headless
    uv run play  --task Isaac-Twisted-RL-Knot-Play-v14 --num_envs 16

v13 から何を・なぜ変えたかは `README.md`、仕様は `docs/decisions/0027-v14-s0-reshape-dr-priv-critic.md`。
ここは「何を配線したか」だけを書く。v13 の設定クラスを継承して、次の 3 つを足す (どれも Hydra で個別に切れる):

1. 保存の規則 `save_on = "stage_up_s0_reshape"`: `stage_up` に、段 0 で交差なしのまま形を変えた手 (1 エピソード 1 回)
   の後の形を足す (`state_archive.rows_to_save`)。形の変化の下限は `s0_min_change` [m]。
   切るとき: `env.rewards.record_stage_archive.params.save_on=stage_up`。
2. ロープの物理の DR (`events.rope_dr`、`mdp.RandomizeRopePhysics`): reset のたびに env ごとに 6 軸の倍率を引く。
   切るとき: `env.events.rope_dr.params.enabled=false`。
3. critic の特権観測 (`observations.rope_params`、`mdp.rope_physics_params`)。critic に渡すのは agent の `obs_groups`
   (`agents/rsl_rl_ppo_cfg.py`)。切るとき: `agent.obs_groups.critic=[policy]` (観測のグループは残るが、方策は読まない)。

`rope_dr` は `EventCfg` の宣言の順で `reset_to_stage` の後ろに並ぶ (継承したフィールドは基底の位置を保ち、足した
フィールドは後ろに付く)。物理パラメータの書き込みは形を変えず、2 つの間に物理ステップは無いので、順序は結果に効かない。

Play (chain 評価) は v13 と同じ設定に、DR を切る設定 (`enabled = False`、公称の物理) を足す。v13 と同じ物理で比べるため。
DR を付けた評価 (頑健性) は `env.events.rope_dr.params.enabled=true` の上書きで行う。
"""

from __future__ import annotations

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass

from ...common import mdp
from ...common.rope_task_cfg import make_preset_cfgs
from ..v4.env_cfg import EventCfg as V4EventCfg
from ..v6.env_cfg import ObservationsCfg as V6ObservationsCfg
from ..v13.env_cfg import KnotTyingV13EnvCfg
from ..v13.env_cfg import play_post_init as _v13_play_post_init

__all__ = ["DR_RANGES", "S0_MIN_CHANGE", "SAVE_ON", "KnotTyingV14EnvCfg", "play_post_init"]

SAVE_ON = "stage_up_s0_reshape"
"""どの手の後の形を保存するか (`state_archive.SAVE_RULES`)。v12・v13 は `stage_up`。"""

S0_MIN_CHANGE = 0.02
"""段 0 の手が形を変えたと見なす変化の下限 [m] (剛体の位置合わせの後のリンクの xy のずれの RMS)。ロープの直径 1 本。"""

DR_RANGES = {
    "bend_stiffness": (0.5, 2.0),
    "twist_stiffness": (0.5, 2.0),
    "damping_ratio": (0.5, 2.0),
    "mass": (0.5, 2.0),
    "rope_friction": (0.5, 2.0),
    "finger_friction": (0.5, 2.0),
}
"""公称値に掛ける倍率の範囲 (対数一様)。どの軸も ×0.5〜×2。

摩擦の公称値はロープ・指とも 0.5 (PhysX の既定の材質。`docs/experiments/2026-10-14_friction-binding.md`) なので、
μ は 0.25〜1.0。公称の物理が分布の真ん中にあり、公称の物理での chain 評価で v13 と比べられる。
"""


@configclass
class EventCfg(V4EventCfg):
    """イベント。v4 (= v7〜v13) の並びの後ろ (`reset_to_stage` の後) に、ロープの物理の DR を足す。"""

    rope_dr = EventTerm(func=mdp.RandomizeRopePhysics, mode="reset", params={"enabled": True, "ranges": DR_RANGES})


@configclass
class ObservationsCfg(V6ObservationsCfg):
    """観測。v6 (= v7〜v13) の `policy` に、critic の特権観測のグループ `rope_params` を足す。"""

    @configclass
    class RopeParamsCfg(ObsGroup):
        """引いた物理の倍率を [−1, 1] に正規化した 6 次元 (`domain_rand.AXES` の順)。"""

        rope_physics = ObsTerm(func=mdp.rope_physics_params, params={"event_term": "rope_dr"})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    rope_params: RopeParamsCfg = RopeParamsCfg()


@configclass
class KnotTyingV14EnvCfg(KnotTyingV13EnvCfg):
    """v14 の環境設定。v13 との差は観測のグループ・DR のイベント・保存の規則だけ。"""

    observations: ObservationsCfg = ObservationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        super().__post_init__()
        self.rewards.record_stage_archive.params["save_on"] = SAVE_ON
        self.rewards.record_stage_archive.params["s0_min_change"] = S0_MIN_CHANGE


def play_post_init(cfg: KnotTyingV14EnvCfg) -> None:
    """Play (chain モード): v13 と同じ設定に、DR を切る設定だけを足す (公称の物理で評価する)。"""
    _v13_play_post_init(cfg)
    cfg.events.rope_dr.params["enabled"] = False


globals().update(make_preset_cfgs(KnotTyingV14EnvCfg, __name__, play_post_init=play_post_init))
