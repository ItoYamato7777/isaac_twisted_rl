"""v3: 論文の中核 (到達状態からのリセット・成功で終了・BFS 距離) を取り込む。

    uv run train --task Isaac-Twisted-RL-Knot-v3 --headless
    uv run play  --task Isaac-Twisted-RL-Knot-Play-v3 --num_envs 1

v2-R2 から何を・なぜ変えたかは `README.md`。ここは「何を配線したか」だけを書く。

v2-R2 の設定クラスを **継承して差分だけ** を上書きする (観測・発散の扱い・
PPO は v2-R2 のまま)。`paper/Report.md` §7 P1 の 4 項目:

1. P1-1 到達状態アーカイブからのリセット (`events.reset_from_archive`)
2. P1-2 成功で終了 (`terminations.knot_tied`) と、段を初めて上がった手にだけ払う
   疎報酬 (`rewards.ladder_bonus`)。手数を 6 -> 12
3. P1-3 ポテンシャルを `max(BFS 距離, ラダー)` にする (`rewards.shaping`)
4. P1-4 段の判定を R2 除去後の `topology_class` 1 本に揃える (§6.4)。
   §6.3 の 8 スロット打ち切りは共有層のバグ修正として全バージョンに入れた

加えて、ゲート 1 (`next_action.md` §4.8.10) が「学習に使うなら v3 で配線すること」
とした 3 点 (`Knot` プリセット・`drop_xy_rel_frac` 0.30・3.2 s/手の尺) を入れる。

切り分けは Hydra の CLI 上書きで行う (`README.md` の「切り分け」)。
"""

from __future__ import annotations

from dataclasses import MISSING

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from ...common import mdp
from ...common.env_cfg_base import EventCfg as BaseEventCfg
from ...common.mdp import phases
from ...common.rope_task_cfg import make_preset_cfgs
from ..v2.env_cfg import GAMMA
from ..v2_r2.env_cfg import _DIVERGED_TERM, _PLAY_PRINT_ENVS, KnotTyingV2R2EnvCfg
from ..v2_r2.env_cfg import CurriculumCfg as V2R2CurriculumCfg
from ..v2_r2.env_cfg import RewardsCfg as V2R2RewardsCfg
from ..v2_r2.env_cfg import TerminationsCfg as V2R2TerminationsCfg

__all__ = ["GAMMA", "KNOT_DURATIONS", "KnotTyingV3EnvCfg"]

_ARCHIVE_TERM = "reset_from_archive"
"""アーカイブ本体を持つ event 項の名前。保存 (`record_archive`) とログがこれを引く。"""

_SUCCESS_TERM = "knot_tied"
"""成功で終了する終了項の名前。アーカイブ項が開始の種別ごとの成功率を数えるのに使う。"""

_LADDER_POTENTIALS = (0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0)
"""`potential="class"` (切り分け用) のときの段ごとの Φ。BFS 距離の値と揃えてある
(`topology/knot_graph.py`)。v2-R2 は (0, 0.35, 0.70, 1.0) でほぼ同じ。"""

KNOT_DURATIONS = phases.PhaseDurations(
    approach=0.0,
    descend=0.3,
    lift=0.3,
    carry=0.6,
    lower=0.3,
    clear=0.0,
    relax=0.5,
)
"""v3 の 1 手の尺 (合計 3.17 s)。v2 の `SHORT_DURATIONS` (2.44 s) の移動系を延ばしたもの。

ゲート 1 (`next_action.md` §4.8.10) の実測: `Knot` (1.14 m) を v2 の尺 (carry 0.35 s) で
運ぶと 1 手目で 13/32 が発散した。運搬速度が原因で、carry 0.6 s で 0/32 になった。
descend が速いと結び目の上へ指を降ろす手でも発散する。この尺 (descend / lift / lower
0.3 s、carry 0.6 s) で、台本を 6 次元の行動に写した再生が 66% 成功した。
"""


@configclass
class RewardsCfg(V2R2RewardsCfg):
    """報酬。v2-R2 の状態ボーナスを外し、遷移ベースの疎報酬と BFS ポテンシャルにする。

    | 項 | v2-R2 | v3 |
    |---|---|---|
    | `shaping` | 4 クラスのラダー (0, 0.35, 0.70, 1) | `max(BFS 距離, ラダー)` |
    | `success_c1/c2/success` | 条件を満たす間 **毎手** +0.5 / +1.5 / +3.0 | 外す |
    | `ladder_bonus` | — | 段を **初めて** 上がった手に同じ値 (+0.5 / +1.5 / +3.0) |
    | `offladder` | 交差 ≥ 2 かつ class 0 に −0.05 | 外す (BFS では行き止まりでない) |
    | `record_archive` | — | アーカイブへの保存 (常に 0) |
    | `grasp_shaping` / `failure` / `held_up` / `topology_stats` | | v2-R2 のまま |
    """

    # P1-3: Φ = max(BFS 距離の値, R2 除去後のラダー)。切り分けは potential=class。
    shaping = RewTerm(
        func=mdp.knot_class_shaping,
        weight=1.0,
        params={
            "gamma": GAMMA,
            "potentials": _LADDER_POTENTIALS,
            "reduce_r2": True,
            "diverged_term": _DIVERGED_TERM,
            "potential": "bfs",
        },
    )

    # P1-2: 状態に払う (毎手) ボーナスは外し、段を初めて上がった手にだけ払う。
    success_c1 = None
    success_c2 = None
    success = None
    ladder_bonus = RewTerm(
        func=mdp.LadderProgressBonus,
        weight=1.0,
        params={
            "bonuses": (0.0, 0.5, 1.5, 3.0),
            "reduce_r2": True,
            "settled_height_diameters": 3.0,
            "rope_diameter": MISSING,
            "diverged_term": _DIVERGED_TERM,
        },
    )

    # 「交差 ≥ 2 で class 0 = 行き止まり」という前提が BFS で崩れた (独立したねじれ 2 つは
    # 輪 1 つと同じ d = 2。`tests/test_knot_graph.py`)。整形報酬と逆向きの信号になるので外す。
    offladder = None

    # P1-1: 手の終わりの形状をアーカイブへ入れる。常に 0 を返す。**重みを 0 にしないこと**。
    record_archive = RewTerm(
        func=mdp.record_archive,
        weight=1.0,
        params={
            "event_term": _ARCHIVE_TERM,
            "settled_height_diameters": 3.0,
            "rope_diameter": MISSING,
            "diverged_term": _DIVERGED_TERM,
        },
    )


@configclass
class TerminationsCfg(V2R2TerminationsCfg):
    """終了判定。v2-R2 の時間切れ・発散に、成功 (固結び) を足す。"""

    # P1-2: 固結びで終了 (time_out=False なので PPO は bootstrap しない)。
    knot_tied = DoneTerm(
        func=mdp.ladder_class_reached,
        params={
            "min_class": 3,
            "reduce_r2": True,
            "settled_height_diameters": 3.0,
            "rope_diameter": MISSING,
            "diverged_term": _DIVERGED_TERM,
        },
    )


@configclass
class EventCfg(BaseEventCfg):
    """イベント。基底の `reset_all` の **後ろ** に、アーカイブからの復元を足す (cfg の並び順 = 適用順)。"""

    # P1-1: 確率 prob で到達済みの C1 / C2 から始める。Play では 0 にする。
    reset_from_archive = EventTerm(
        func=mdp.ResetFromArchive,
        mode="reset",
        params={
            "prob": 0.5,
            "capacity": 4096,
            "archive_classes": (1, 2),
            "class_weights": (1.0, 2.0),
            "random_yaw": True,
            "success_term": _SUCCESS_TERM,
        },
    )


@configclass
class CurriculumCfg(V2R2CurriculumCfg):
    """カリキュラムは使わない。ログ出力のためだけの項 (`Curriculum/archive_log/*` を足す)。"""

    archive_log = CurrTerm(func=mdp.log_archive_stats, params={"event_term": _ARCHIVE_TERM})


@configclass
class KnotTyingV3EnvCfg(KnotTyingV2R2EnvCfg):
    """v3 の環境設定。学習は `Knot` プリセット (`Isaac-Twisted-RL-Knot-v3`) で回す。"""

    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    durations: phases.PhaseDurations = KNOT_DURATIONS

    grasp_limits: phases.GraspLimits = phases.GraspLimits(drop_xy_rel_frac=0.30)
    """置き先の範囲を ±0.15 L -> ±0.30 L に広げる。

    ゲート 1 の実測: 台本の cover (輪の片側を掴んで尻尾に被せる) の置き先は、128 環境中
    79〜97 で「target リンク ±8 本 + ±0.15 L」の外にあり、クリップされて輪ごと崩れた。
    0.30 にすると 6 次元の行動での再生が 9% -> 66% になった (3.2 s/手)。
    """

    actions_per_episode: int = 12
    """1 エピソードの手数の上限 (v2 / v2-R2 は 6)。

    固結びで終了するので、長くしても「結んだ後に居座る手」は生じない。ゲート 1 の台本は
    5 手、TWISTED-RL の成功例は 1 遷移あたり平均 2.5 手 (3 遷移で約 7.5 手) なので、
    まっすぐなロープから始めても把持の失敗やり直しを含めて届く長さにした。
    """


def _play_post_init(cfg: KnotTyingV3EnvCfg) -> None:
    """Play では常にまっすぐなロープから始め (実力の評価)、毎手 p-data を表示する。"""
    cfg.events.reset_from_archive.params["prob"] = 0.0
    cfg.rewards.topology_stats.params["print_envs"] = _PLAY_PRINT_ENVS


globals().update(make_preset_cfgs(KnotTyingV3EnvCfg, __name__, play_post_init=_play_post_init))
