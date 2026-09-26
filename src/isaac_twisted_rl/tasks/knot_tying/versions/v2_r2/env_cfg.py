"""v2-R2: v2 の報酬ラダーを **R2 除去後の位相** で判定する。それ以外は v2 と同一。

    uv run train --task Isaac-Twisted-RL-Twist-v2-R2 --headless
    uv run play  --task Isaac-Twisted-RL-Twist-Play-v2-R2 --num_envs 1

v2 から何を・なぜ変えたかは `README.md`。ここは「何を配線したか」だけを書く。

v2 の設定クラスを **継承して差分だけ** を上書きする。物理・行動・エピソード長・
フェーズの尺・γ・観測の並びは v2 のまま (v2 と効果を比べるため、変えるのは
判定器の 1 変数だけにする)。
"""

from __future__ import annotations

from dataclasses import MISSING

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from ...common import mdp
from ...common.rope_task_cfg import make_preset_cfgs
from ..v2.env_cfg import GAMMA, KnotTyingV2EnvCfg
from ..v2.env_cfg import ObservationsCfg as V2ObservationsCfg
from ..v2.env_cfg import RewardsCfg as V2RewardsCfg

__all__ = ["GAMMA", "KnotTyingV2R2EnvCfg"]

_PLAY_PRINT_ENVS = 9
"""Play で p-data を表示する環境数の上限 (`--num_envs` がこれより少なければそちら)。"""


@configclass
class ObservationsCfg(V2ObservationsCfg):
    """観測。v2 と同じ 105 次元・同じ並び。`knot_class` の中身だけ R2 除去後の段にする。

    報酬が R2 除去後の段で払われるのに、観測だけ v2 の段 (偽の C2 を含む) を
    見せると、方策は「観測上は class 2 なのに報酬が来ない」状態を区別できない。
    次元を変えないので v2 のチェックポイントもそのまま読める。
    """

    @configclass
    class PolicyCfg(V2ObservationsCfg.PolicyCfg):
        knot_class = ObsTerm(func=mdp.knot_class_one_hot, params={"reduce_r2": True})

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg(V2RewardsCfg):
    """報酬。v2 の各項に `reduce_r2=True` を渡し、ログ用の項を差し替える。

    重みは v2 と同一 (比較のため)。変わるのは「どの状態がどの段か」だけ:

    | 状態 (先端側の上下) | v2 | v2-R2 |
    |---|---|---|
    | 輪 1 つ (R1) | class 1 | class 1 |
    | 先端を輪の中へ、1 回くぐらせて (OU / UO) | class 2 | class 2 |
    | 先端を輪の中へ上から置いただけ (OO) | class 0 + `offladder` 罰 | class 0 (交差 0 に戻る。罰なし) |
    | 先端で輪を上から跨いだだけ (OOO など 6 通り) | **class 2** | **class 1** |
    | 固結び (OUO / UOU) | class 3 | class 3 |
    """

    shaping = RewTerm(
        func=mdp.knot_class_shaping,
        weight=1.0,
        params={"gamma": GAMMA, "potentials": (0.0, 0.35, 0.70, 1.0), "reduce_r2": True},
    )

    # 最下段も R2 除去後の交差数で測る。先端を輪の上へ置いただけ (OO) は
    # 交差 0 と同じなので、v2 のように +0.5 を払わない。
    success_c1 = RewTerm(
        func=mdp.crossing_achieved,
        weight=0.5,
        params={"min_crossings": 1, "settled_height_diameters": 3.0, "rope_diameter": MISSING, "reduce_r2": True},
    )

    # **v2-R2 の本題。** 掛け渡し (偽の C2) をここで弾く。
    success_c2 = RewTerm(
        func=mdp.interleaved2_achieved,
        weight=1.5,
        params={"settled_height_diameters": 3.0, "rope_diameter": MISSING, "reduce_r2": True},
    )

    # 固結びは交代結び目なので R2 除去で消えることは無い。判定器を揃えるために渡す。
    success = RewTerm(
        func=mdp.overhand_achieved,
        weight=3.0,
        params={"settled_height_diameters": 3.0, "rope_diameter": MISSING, "reduce_r2": True},
    )

    offladder = RewTerm(func=mdp.offladder_penalty, weight=-0.05, params={"reduce_r2": True})

    # v2 のログ専用項 (重み 0) は `RewardManager` に呼ばれないので外す
    # (`common/mdp/topology_log.py` の docstring)。代わりに下の topology_stats。
    diverged_class0 = None
    diverged_class1 = None
    diverged_class2 = None
    diverged_class3 = None

    # ログ専用: 常に 0 を返すが、毎手呼ばれて統計を積む。**重みを 0 にしないこと**
    # (0 だと呼ばれない)。値は `curriculum.topology_log` が TensorBoard へ出す。
    topology_stats = RewTerm(func=mdp.TopologyStats, weight=1.0, params={"print_envs": 0})


@configclass
class CurriculumCfg:
    """カリキュラムは使わない。`Curriculum/topology_log/*` へのログ出力のためだけの項。"""

    topology_log = CurrTerm(func=mdp.log_topology_stats, params={"reward_term": "topology_stats"})


@configclass
class KnotTyingV2R2EnvCfg(KnotTyingV2EnvCfg):
    """v2-R2 の環境設定。v2 との差は観測・報酬・カリキュラム (ログ) だけ。"""

    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()


def _enable_p_data_print(cfg: KnotTyingV2R2EnvCfg) -> None:
    """Play のときだけ、毎手 p-data (R2 除去前 / 後) を表示する。"""
    cfg.rewards.topology_stats.params["print_envs"] = _PLAY_PRINT_ENVS


globals().update(make_preset_cfgs(KnotTyingV2R2EnvCfg, __name__, play_post_init=_enable_p_data_print))
