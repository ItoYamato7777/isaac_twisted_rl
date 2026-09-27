"""v2-R2: v2 の報酬ラダーを **R2 除去後の位相** で判定する。

    uv run train --task Isaac-Twisted-RL-Twist-v2-R2 --headless
    uv run play  --task Isaac-Twisted-RL-Twist-Play-v2-R2 --num_envs 1

v2 から何を・なぜ変えたかは `README.md`。ここは「何を配線したか」だけを書く。

v2 の設定クラスを **継承して差分だけ** を上書きする。物理・行動・エピソード長・
フェーズの尺・γ・観測の並び・報酬の重みは v2 のまま。変えたのは 2 点:

1. 報酬ラダーの判定器 (R2 除去)。本題。
2. **発散した手の扱い** (2026-09-27 修正)。v2 の設定のままだと、吹き飛んだ
   ロープまでの把持距離がそのまま罰になり学習が壊れた (README の
   「発散した手の報酬」)。正常な手の報酬は変わらないので、1. の比較は保たれる。
"""

from __future__ import annotations

from dataclasses import MISSING

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from ...common import mdp
from ...common.env_cfg_base import TerminationsCfg as BaseTerminationsCfg
from ...common.rope_task_cfg import make_preset_cfgs
from ..v2.env_cfg import GAMMA, KnotTyingV2EnvCfg
from ..v2.env_cfg import ObservationsCfg as V2ObservationsCfg
from ..v2.env_cfg import RewardsCfg as V2RewardsCfg

__all__ = ["GAMMA", "KnotTyingV2R2EnvCfg"]

_PLAY_PRINT_ENVS = 9
"""Play で p-data を表示する環境数の上限 (`--num_envs` がこれより少なければそちら)。"""

# ---------------------------------------------------------------- 発散の扱い (2026-09-27 修正)
# 詳細は README の「発散した手の報酬」。閾値はランダム方策の実測
# (512 環境 x 24 手) で、正常な手の終わりが xy 最大 1.52 m / 速度最大 3.3 m/s、
# 発散時が数十 m 以上 / 60〜175 m/s だったことから、正常側に 2 倍以上の余裕を取った。

_DIVERGED_TERM = "diverged"
"""発散の判定に使う終了項の名前。報酬・ログはこの項の判定をそのまま読む
(`mdp.diverged_now`)。閾値を 1 か所 (`TerminationsCfg.diverged`) に集めるため。"""

_MAX_XY_DISTANCE = 3.0
"""env 原点からのロープ節点の水平距離の上限 [m] (= `env_spacing`)。全長 0.82 m の約 3.7 倍。"""

_MAX_LINK_SPEED = 20.0
"""手の終わり (relax 後) のリンク速度の上限 [m/s]。"""

_GRASP_DIST_MAX_DIAMETERS = 15.0
"""把持距離の罰の上限 (ロープ直径比)。twist で 0.30 m。正常な空振りは最大 0.25 m。"""

_OBS_POS_CLIP = (-_MAX_XY_DISTANCE, _MAX_XY_DISTANCE)
"""`rope_link_pos` 観測の値域 [m]。発散判定を通った状態では到達しない保険。"""


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

        # 保険: 発散判定を通った状態なら必ず範囲内だが、判定をすり抜けた
        # 巨大な座標が critic に入ると価値関数が一発で壊れる (README 参照)。
        # 次元は変わらないので v2 のチェックポイントもそのまま読める。
        rope_link_pos = ObsTerm(func=mdp.rope_link_positions, clip=_OBS_POS_CLIP)

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

    # 発散した手 (`diverged_term`) は Φ(s') = 0 の終端として扱う (−Φ(s) を払う)。
    shaping = RewTerm(
        func=mdp.knot_class_shaping,
        weight=1.0,
        params={
            "gamma": GAMMA,
            "potentials": (0.0, 0.35, 0.70, 1.0),
            "reduce_r2": True,
            "diverged_term": _DIVERGED_TERM,
        },
    )

    # 最下段も R2 除去後の交差数で測る。先端を輪の上へ置いただけ (OO) は
    # 交差 0 と同じなので、v2 のように +0.5 を払わない。
    success_c1 = RewTerm(
        func=mdp.crossing_achieved,
        weight=0.5,
        params={
            "min_crossings": 1,
            "settled_height_diameters": 3.0,
            "rope_diameter": MISSING,
            "reduce_r2": True,
            "diverged_term": _DIVERGED_TERM,
        },
    )

    # **v2-R2 の本題。** 掛け渡し (偽の C2) をここで弾く。
    success_c2 = RewTerm(
        func=mdp.interleaved2_achieved,
        weight=1.5,
        params={
            "settled_height_diameters": 3.0,
            "rope_diameter": MISSING,
            "reduce_r2": True,
            "diverged_term": _DIVERGED_TERM,
        },
    )

    # 固結びは交代結び目なので R2 除去で消えることは無い。判定器を揃えるために渡す。
    success = RewTerm(
        func=mdp.overhand_achieved,
        weight=3.0,
        params={
            "settled_height_diameters": 3.0,
            "rope_diameter": MISSING,
            "reduce_r2": True,
            "diverged_term": _DIVERGED_TERM,
        },
    )

    offladder = RewTerm(
        func=mdp.offladder_penalty, weight=-0.05, params={"reduce_r2": True, "diverged_term": _DIVERGED_TERM}
    )

    # ---- 発散した手の扱い (2026-09-27 修正。重みは v2 と同一)
    # v2 の設定のままだと、`close` の瞬間に吹き飛んだロープまでの距離
    # (実測 97 m〜) がそのまま罰になり、最初の run の critic を壊した。
    # 上限を掛け、発散した手は 0 にする (発散の罰は `failure` に一本化)。
    grasp_shaping = RewTerm(
        func=mdp.grasp_distance_penalty,
        weight=-0.05,
        params={
            "max_distance_diameters": _GRASP_DIST_MAX_DIAMETERS,
            "rope_diameter": MISSING,
            "diverged_term": _DIVERGED_TERM,
        },
    )

    # NaN だけでなく、終了判定と同じ基準 (有限のまま吹き飛んだロープ) も失敗に数える。
    failure = RewTerm(func=mdp.action_failed, weight=-0.1, params={"diverged_term": _DIVERGED_TERM})

    held_up = RewTerm(
        func=mdp.held_up_penalty,
        weight=-0.1,
        params={"height_diameters": 5.0, "rope_diameter": MISSING, "diverged_term": _DIVERGED_TERM},
    )

    # v2 のログ専用項 (重み 0) は `RewardManager` に呼ばれないので外す
    # (`common/mdp/topology_log.py` の docstring)。代わりに下の topology_stats。
    diverged_class0 = None
    diverged_class1 = None
    diverged_class2 = None
    diverged_class3 = None

    # ログ専用: 常に 0 を返すが、毎手呼ばれて統計を積む。**重みを 0 にしないこと**
    # (0 だと呼ばれない)。値は `curriculum.topology_log` が TensorBoard へ出す。
    topology_stats = RewTerm(
        func=mdp.TopologyStats, weight=1.0, params={"print_envs": 0, "diverged_term": _DIVERGED_TERM}
    )


@configclass
class TerminationsCfg(BaseTerminationsCfg):
    """終了判定。v2 までの「NaN / 高さ 5 m 超」に、水平距離と速度の上限を足す。

    高さだけでは、床の上を水平に吹き飛んだ有限の座標を見逃す。報酬項は
    この項の判定 (`mdp.diverged_now`) で「発散した手」を見分けるので、
    閾値はここだけに書く。
    """

    diverged = DoneTerm(
        func=mdp.rope_diverged,
        params={"max_xy_distance": _MAX_XY_DISTANCE, "max_link_speed": _MAX_LINK_SPEED},
    )


@configclass
class CurriculumCfg:
    """カリキュラムは使わない。`Curriculum/topology_log/*` へのログ出力のためだけの項。"""

    topology_log = CurrTerm(func=mdp.log_topology_stats, params={"reward_term": "topology_stats"})


@configclass
class KnotTyingV2R2EnvCfg(KnotTyingV2EnvCfg):
    """v2-R2 の環境設定。v2 との差は観測・報酬・終了判定・カリキュラム (ログ)。"""

    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()


def _enable_p_data_print(cfg: KnotTyingV2R2EnvCfg) -> None:
    """Play のときだけ、毎手 p-data (R2 除去前 / 後) を表示する。"""
    cfg.rewards.topology_stats.params["print_envs"] = _PLAY_PRINT_ENVS


globals().update(make_preset_cfgs(KnotTyingV2R2EnvCfg, __name__, play_post_init=_enable_p_data_print))
