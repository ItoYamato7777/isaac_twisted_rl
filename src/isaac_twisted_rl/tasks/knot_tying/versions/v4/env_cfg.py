"""v4: 段ごとのエージェント (TWISTED-RL-C) と 1 遷移 = 1 エピソード。

    uv run train --task Isaac-Twisted-RL-Knot-v4 --headless
    uv run play  --task Isaac-Twisted-RL-Knot-Play-v4 --num_envs 16

v3 から何を・なぜ変えたかは `README.md`。ここは「何を配線したか」だけを書く。

v3 の設定クラスを **継承して差分だけ** を上書きする (ロープ `Knot`・1 手の尺・置き先の範囲・
発散の扱いは v3 のまま):

1. 観測の **最後に** 段の one-hot (4) を足す (`observations.policy.knot_stage`)。
   方策 `StageGatedActorCritic` がこの列で段 0 / 1 / 2 のエージェントを選ぶ (`agents/`)。
2. 1 エピソード = 1 回の段の遷移。段が上がる・下がると終了 (`terminations.stage_up/down`)、
   最大 6 手。報酬は +1 / 0 / −1 (`rewards.transition`)。v3 の整形・段のボーナスは外す。
3. リセットは段ごとのアーカイブから (`events.reset_to_stage`)。ゲート 1 の台本の途中状態を
   ファイルから読み、学習中のアーカイブも `{log_dir}/archive.pt` へ書く。

Play は評価用の chain モード: まっすぐなロープから始め、固結び・発散・18 手でだけ終わる。
段が上がると観測の段が変わり、次の段のエージェントに切り替わる。
"""

from __future__ import annotations

import pathlib
from dataclasses import MISSING

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from ...common import mdp
from ...common.rope_task_cfg import make_preset_cfgs
from ..v2_r2.env_cfg import _DIVERGED_TERM, _PLAY_PRINT_ENVS
from ..v2_r2.env_cfg import ObservationsCfg as V2R2ObservationsCfg
from ..v3.env_cfg import GAMMA, KnotTyingV3EnvCfg
from ..v3.env_cfg import CurriculumCfg as V3CurriculumCfg
from ..v3.env_cfg import EventCfg as V3EventCfg
from ..v3.env_cfg import RewardsCfg as V3RewardsCfg
from ..v3.env_cfg import TerminationsCfg as V3TerminationsCfg

__all__ = ["GAMMA", "KnotTyingV4EnvCfg", "SEED_ARCHIVE"]

_STAGE_TERM = "reset_to_stage"
"""段ごとのアーカイブを持つ event 項の名前。保存 (`record_stage_archive`) とログがこれを引く。"""

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[6]

SEED_ARCHIVE = str(_REPO_ROOT / "outputs" / "archive_seed" / "overhand_{num_links}links.pt")
"""ゲート 1 の台本の途中状態 (輪 / cover 後 / C2) を段ごとに保存したファイル。

`demos/overhand_script.py --dump_archive` が既定でここへ書く。`{num_links}` はロープの
リンク数 (`Knot` は 28)。無ければ警告して空のアーカイブから始める。
"""

_SAVE_EVERY = 600
"""アーカイブをファイルへ書く間隔 [手] = 50 iteration x 12 手 (チェックポイントの間隔と同じ)。"""

_SETTLE = {"settled_height_diameters": 3.0, "rope_diameter": MISSING, "diverged_term": _DIVERGED_TERM}
"""段が変わったとみなす条件 (床に落ちていて発散していない)。報酬・終了・保存で揃える。"""


@configclass
class ObservationsCfg(V2R2ObservationsCfg):
    """観測。v3 (= v2-R2) の 129 次元 (`Knot`) の **後ろに** 段の one-hot (4) を足す。"""

    @configclass
    class PolicyCfg(V2R2ObservationsCfg.PolicyCfg):
        # **最後の項であること。** StageGatedActorCritic が末尾 4 列の argmax で振り分ける。
        knot_stage = ObsTerm(func=mdp.knot_stage_one_hot)

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg(V3RewardsCfg):
    """報酬。段の遷移の +1 / 0 / −1 と、v3 の小さな整形・罰則。

    | 項 | v3 | v4 |
    |---|---|---|
    | `transition` | — | 段が上がる +1 / 下がる −1 / 発散 −1 / それ以外 0 |
    | `shaping` | `max(BFS 距離, ラダー)` の整形 | 外す (段の中で Φ は一定) |
    | `ladder_bonus` | 段を初めて上がった手に +0.5 / +1.5 / +3.0 | 外す (上がれば終了するので `transition` と同じ) |
    | `record_archive` | class ごとのアーカイブへ保存 | `record_stage_archive` (段ごと) に置き換え |
    | `grasp_shaping` / `failure` / `held_up` / `topology_stats` | | v3 のまま |
    """

    shaping = None
    ladder_bonus = None
    record_archive = None

    transition = RewTerm(
        func=mdp.stage_transition_reward,
        weight=1.0,
        params={"success": 1.0, "failure": -1.0, "diverged": -1.0, "strict_topology": False, **_SETTLE},
    )

    # 常に 0 を返す。**重みを 0 にしないこと** (0 だと呼ばれない)。
    record_stage_archive = RewTerm(
        func=mdp.record_stage_archive,
        weight=1.0,
        params={"event_term": _STAGE_TERM, "save_every": _SAVE_EVERY, "save_name": "archive.pt", **_SETTLE},
    )


@configclass
class TerminationsCfg(V3TerminationsCfg):
    """終了判定。v3 の固結びでの終了を、段が変わったら終わる 2 項に置き換える (時間切れ・発散はそのまま)。

    `stage_up` を `time_out` より後ろに置く。最後の手で上がったとき、ログ (`ResetToStage`) が
    時間切れではなく成功として数えるため (同じ手で立った項は cfg の後ろが残る)。
    """

    knot_tied = None

    stage_up = DoneTerm(
        func=mdp.stage_changed,
        params={"direction": "up", "chain": False, "strict_topology": False, **_SETTLE},
    )
    stage_down = DoneTerm(
        func=mdp.stage_changed,
        params={"direction": "down", "chain": False, "strict_topology": False, **_SETTLE},
    )


@configclass
class EventCfg(V3EventCfg):
    """イベント。v3 のアーカイブ (class キー) を、段ごとのアーカイブに置き換える (`reset_all` の後ろ)。"""

    reset_from_archive = None

    reset_to_stage = EventTerm(
        func=mdp.ResetToStage,
        mode="reset",
        params={
            "stages": (0, 1, 2),
            "stage_weights": (1.0, 1.0, 1.0),
            "straight_prob": 0.5,
            # 既定の環境数 4096 (`env_cfg_base.py`) では、段 1 は 1 手で数千件入る。v3 の 4096 件だと
            # 数手で入れ替わり、初期アーカイブ (台本の途中状態) もすぐ消えるので 4 倍にした。
            "capacity": 16384,
            "random_yaw": True,
            "load_paths": (SEED_ARCHIVE,),
            "outcome_terms": {"up": "stage_up", "down": "stage_down", "diverged": "diverged", "time_out": "time_out"},
        },
    )


@configclass
class CurriculumCfg(V3CurriculumCfg):
    """ログ専用。v3 の `archive_log` を、段ごとの成功率 `Curriculum/stage_log/*` に置き換える。"""

    archive_log = None
    stage_log = CurrTerm(func=mdp.log_stage_stats, params={"event_term": _STAGE_TERM})


@configclass
class KnotTyingV4EnvCfg(KnotTyingV3EnvCfg):
    """v4 の環境設定。学習は `Knot` プリセット (`Isaac-Twisted-RL-Knot-v4`) で回す。"""

    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    actions_per_episode: int = 6
    """1 遷移 (= 1 エピソード) の手数の上限 M。

    TWISTED-RL の移植元コードは 7 手、3-Easy の成功例は 1 遷移あたり平均 2.5 手。ゲート 1 の
    台本は段 0→1 に 2 手 (slack, loop)、1→2 に 2 手 (tail, cover)、2→3 に 1 手 (tuck)。
    把持のやり直しを含めても届き、発散 (1 手あたり約 3%) に巻き込まれにくい長さにした。
    """


_PLAY_ACTIONS = 18
"""Play (chain モード) の手数の上限 = 3 遷移 x 6 手。"""


def _play_post_init(cfg: KnotTyingV4EnvCfg) -> None:
    """Play は chain モード: まっすぐなロープから段をつなぎ、固結び・発散・18 手でだけ終わる。"""
    cfg.events.reset_to_stage.params["stage_weights"] = (1.0, 0.0, 0.0)
    cfg.events.reset_to_stage.params["straight_prob"] = 1.0
    cfg.rewards.record_stage_archive.params["save_every"] = 0
    cfg.terminations.stage_up.params["chain"] = True
    cfg.terminations.stage_down.params["chain"] = True
    # episode_length_s は基底の __post_init__ で手数から計算済みなので作り直す。
    cfg.actions_per_episode = _PLAY_ACTIONS
    cfg.episode_length_s = _PLAY_ACTIONS * cfg.decimation * cfg.sim.dt
    cfg.rewards.topology_stats.params["print_envs"] = _PLAY_PRINT_ENVS


globals().update(make_preset_cfgs(KnotTyingV4EnvCfg, __name__, play_post_init=_play_post_init))
