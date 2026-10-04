"""v6: v5 に「次にたどる位相の辺」のリンクマスクを観測として足した版。

    uv run train --task Isaac-Twisted-RL-Knot-v6 --headless
    uv run play  --task Isaac-Twisted-RL-Knot-Play-v6 --num_envs 16

v5 から何を・なぜ変えたかは `README.md`。ここは「何を配線したか」だけを書く。

v5 (= v4 の段ごとのエージェント・1 遷移 = 1 エピソード + 発散対策) の設定クラスを **継承して、
観測だけ** を変える (1 版 1 変数)。報酬・終了・段ごとのアーカイブ・PPO・行動・Play (chain モード)
は v5 のまま:

1. 観測 `policy` の **段の one-hot の直前** に `next_edge` (`mdp.NextEdgeGoal`、`Knot` で 64 次元) を
   足す。段の one-hot は末尾に残す (`StageGatedActorCritic` が末尾 4 列で段のエージェントを選ぶため)。
   critic も `policy` グループを見る (`../v2/agents/rsl_rl_ppo_cfg.py` の `_OBS_GROUPS`) ので両方に入る。
2. ログ用に、常に 0 を返す報酬項 `next_edge_stats` (`mdp.NextEdgeStats`) と、それを
   TensorBoard へ流すカリキュラム項 `next_edge_log` を足す。報酬の値は変わらない
   (`topology_stats` と同じ仕組み)。

辺の選び方は `NEXT_EDGE_RULE`。比較は Hydra の上書きで行う:
`env.observations.policy.next_edge.params.rule=first` (または `random`)。
"""

from __future__ import annotations

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from ...common import mdp
from ...common.rope_task_cfg import make_preset_cfgs
from ..v2_r2.env_cfg import DIVERGED_TERM as _DIVERGED_TERM
from ..v2_r2.env_cfg import ObservationsCfg as V2R2ObservationsCfg
from ..v5.env_cfg import CurriculumCfg as V5CurriculumCfg
from ..v5.env_cfg import KnotTyingV5EnvCfg
from ..v5.env_cfg import RewardsCfg as V5RewardsCfg
from ..v5.env_cfg import play_post_init as _v5_play_post_init

__all__ = ["NEXT_EDGE_RULE", "KnotTyingV6EnvCfg"]

NEXT_EDGE_RULE = "long_end"
"""次にたどる辺の選び方 (`topology.NEXT_EDGE_RULES`)。

端の区間を使う候補のうち、その端の区間のリンク数が **多い** ものを優先する (同点なら
R1 > R2 > 複合 > cross、さらに同点なら列挙順)。ゲート 1 の台本 (`demos/overhand_script.py`) を
実際の形状で照合すると、台本の次の p-data と 93.4% 一致した (README の「台本との照合」)。
当初案の `short_end` (短い端を優先) は 34.7% だった。台本の輪は頭側で本体と交差し、
台本は長い側 (尻尾) の端を使うため。
"""


@configclass
class ObservationsCfg(V2R2ObservationsCfg):
    """観測。v2-R2 の並び + `next_edge` + 段の one-hot (末尾)。

    v5 (= v4) の `ObservationsCfg.PolicyCfg` を継承して `next_edge` を足すと、継承した `knot_stage`
    より **後ろ** に並んでしまう (dataclass は上書きしても基底の位置を保つ)。`StageGatedActorCritic`
    は末尾 4 列を段として読むので、v2-R2 から継承し直して `next_edge` -> `knot_stage` の順に宣言する。
    """

    @configclass
    class PolicyCfg(V2R2ObservationsCfg.PolicyCfg):
        # v2-R2 の項の後ろ、段の one-hot の前に置く: v5 と入力層の先頭が同じ並びになる。
        next_edge = ObsTerm(func=mdp.NextEdgeGoal, params={"rule": NEXT_EDGE_RULE, "seed": 0})
        # **最後の項であること** (v4 / v5 と同じ)。StageGatedActorCritic が末尾 4 列の argmax で振り分ける。
        knot_stage = ObsTerm(func=mdp.knot_stage_one_hot)

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg(V5RewardsCfg):
    """報酬。v5 (= v4 の +1 / 0 / −1) と同じ値。ログ用の `next_edge_stats` (常に 0) だけを足す。"""

    # 手の前に選んだ辺と手の後の位相を比べて積む。**重みを 0 にしないこと** (0 だと呼ばれない)。
    next_edge_stats = RewTerm(func=mdp.NextEdgeStats, weight=1.0, params={"diverged_term": _DIVERGED_TERM})


@configclass
class CurriculumCfg(V5CurriculumCfg):
    """ログ専用。v5 (= v4) の `stage_log` に、`Curriculum/next_edge_log/*` を出すための項を足す。"""

    next_edge_log = CurrTerm(func=mdp.log_next_edge_stats, params={"reward_term": "next_edge_stats"})


@configclass
class KnotTyingV6EnvCfg(KnotTyingV5EnvCfg):
    """v6 の環境設定。v5 との差は観測 (`next_edge`) と、ログ用の 2 項だけ。"""

    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()


globals().update(make_preset_cfgs(KnotTyingV6EnvCfg, __name__, play_post_init=_v5_play_post_init))
