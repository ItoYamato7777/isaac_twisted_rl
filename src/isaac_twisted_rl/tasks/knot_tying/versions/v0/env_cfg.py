"""v0: 交差を 1 つ作って残す (基準バージョン)。

    uv run train --task Isaac-Twisted-RL-Twist-v0 --headless

紐結びの最小版。ロープの一部を持ち上げてロープ本体の上をまたがせて置くと
「交差」ができる。これを位相不変量として検出し、報酬に変える。

**このバージョンは基準点として凍結する意図で置いてある。** 改良はここを
書き換えるのではなく `../v1/` 以降を足すこと (`../../README.md` の規約)。

シーン・物理・行動・終了判定は `common/env_cfg_base.py` と共通。
このファイルが定義するのは **観測・報酬・エピソード長**だけ。
"""

from __future__ import annotations

from dataclasses import MISSING

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from ...common import mdp
from ...common.env_cfg_base import KnotTyingEnvCfg
from ...common.rope_task_cfg import make_preset_cfgs

GAMMA = 0.95
"""割引率。`agents/rsl_rl_ppo_cfg.py` が import して使う。

v0 は 1 エピソードが 5 手しかないので、lift の 0.98 (実質 50 ステップ相当を
見る) は長すぎる。v0 の報酬はポテンシャル整形を使っていないため、v1 と違って
PPO 側と一致させる制約は無いが、定義を 1 箇所に集める方針は揃えてある。
"""


@configclass
class ObservationsCfg:
    """観測。テンソル層のみ (p-data 文字列は使わない)。"""

    @configclass
    class PolicyCfg(ObsGroup):
        """方策への観測。

        `rope_link_pos` がロープの形、`crossings` / `writhe` /
        `first_crossing` が位相。幾何と位相の両方を渡すのは、位相だけでは
        「次にどこを掴めば交差を作れるか」が決まらないため。

        目標が「交差 1 つ」なので、交差の特徴も先頭 1 つぶんで足りる。
        (v1 は固結び = 交差 3 つが目標なので `crossing_features` に拡張した。)
        """

        rope_link_pos = ObsTerm(func=mdp.rope_link_positions)
        crossings = ObsTerm(func=mdp.crossing_count)
        writhe = ObsTerm(func=mdp.writhe)
        first_crossing = ObsTerm(func=mdp.first_crossing_feature)
        hand_pos = ObsTerm(func=mdp.hand_position)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            # ノイズは入れない。位相量 (交差数・符号) は離散で、ノイズを
            # 乗せると意味が壊れる (1.2 個の交差、という状態は無い)。
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """報酬 (`docs/history/next_action.md` §4.2。今も有効な理由は `docs/decisions/0005-measure-after-retreat.md`)。

    **重みは `__post_init__` で `step_dt` で割られる**。Isaac Lab の
    `RewardManager.compute()` は各項に `dt` (= 1 env ステップの秒数) を
    掛ける仕様で、報酬を「毎秒の割合」として扱う設計になっている。
    ところがこの環境は 1 env ステップが 1 手と極端に長いため、
    4.2 が指定する「1 手あたり +1.0」がそのまま書くと数倍になってしまう。

    そこで ここには **4.2 が指定する「1 手あたりの値」をそのまま書き**、
    `KnotTyingEnvCfg.__post_init__` が `step_dt` で割って辻褄を合わせる。
    こうすることで、フェーズの尺を変えて 1 手の長さが変わっても
    報酬のスケールは変わらない。
    """

    # 密報酬: 交差が増えたか。p-data は離散なので一致判定だけでは勾配が出ない。
    progress = RewTerm(func=mdp.crossing_progress, weight=0.3)

    # 疎報酬: 交差が残ったか。**ハンド退避後に測る**のが最重要 (rewards.py 参照)。
    success = RewTerm(
        func=mdp.crossing_achieved,
        weight=1.0,
        params={"min_crossings": 1, "settled_height_diameters": 3.0, "rope_diameter": MISSING},
    )

    # 整形: 狙ったリンクを実際に掴めたか。
    grasp_shaping = RewTerm(func=mdp.grasp_distance_penalty, weight=-0.05)

    # 罰則: 把持失敗 / NaN。
    failure = RewTerm(func=mdp.action_failed, weight=-0.2)

    # 罰則: 手の終わりにロープが浮いたまま (「吊っているだけ」の検出も兼ねる)。
    held_up = RewTerm(
        func=mdp.held_up_penalty,
        weight=-0.1,
        params={"height_diameters": 5.0, "rope_diameter": MISSING},
    )


@configclass
class KnotTyingV0EnvCfg(KnotTyingEnvCfg):
    """v0 の環境設定。ロープのプリセット別の派生は下の `make_preset_cfgs`。"""

    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()

    actions_per_episode: int = 5
    """1 エピソードで打てる手の数 (`docs/decisions/0001-kinematic-hand-and-one-move.md` の「3〜5 アクション」)。"""


# ロープのプリセット 5 種 x (通常 / Play) の設定クラスを生成する。
globals().update(make_preset_cfgs(KnotTyingV0EnvCfg, __name__))
