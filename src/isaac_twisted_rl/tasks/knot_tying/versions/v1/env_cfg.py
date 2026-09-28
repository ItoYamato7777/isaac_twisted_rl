"""v1: 固結び (overhand knot) を作る。

    uv run train --task Isaac-Twisted-RL-Twist-v1 --headless

v0 (交差を 1 つ作る) からの変更点と、その理由は `README.md` を参照。
ここは「何を配線したか」だけを書く。

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

GAMMA = 0.98
"""割引率。**報酬整形と PPO で同じ値を使う**ための唯一の定義。

ポテンシャル整形 `F = γΦ(s') − Φ(s)` が最適方策を変えないという保証
(Ng et al. 1999) は、この γ が方策の割引率と一致して初めて成り立つ。
`agents/rsl_rl_ppo_cfg.py` はここを import して使う。片方だけ書き換えて
ずれる事故を防ぐため、数値をそこに直接書かないこと。
"""


@configclass
class ObservationsCfg:
    """観測。テンソル層のみ (p-data 文字列は使わない)。"""

    @configclass
    class PolicyCfg(ObsGroup):
        """方策への観測。

        `rope_link_pos` がロープの形、`crossings` / `writhe` /
        `crossing_feats` / `knot_class` が位相。幾何と位相の両方を渡すのは、
        位相だけでは「次にどこを掴めば結べるか」が決まらないため。

        v0 との差は `crossing_feats` (交差 5 個ぶん) と `knot_class`
        (報酬ラダーの段) の 2 項。固結びは交差 3 つなので、v0 の
        「先頭 1 交差だけ」では状態が識別できない。
        """

        rope_link_pos = ObsTerm(func=mdp.rope_link_positions)
        crossings = ObsTerm(func=mdp.crossing_count)
        writhe = ObsTerm(func=mdp.writhe)
        crossing_feats = ObsTerm(func=mdp.crossing_features, params={"max_crossings": 5})
        knot_class = ObsTerm(func=mdp.knot_class_one_hot)
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
    """報酬 (固結び / overhand knot 版)。

    **重みは `__post_init__` で `step_dt` で割られる**。Isaac Lab の
    `RewardManager.compute()` は各項に `dt` (= 1 env ステップの秒数) を
    掛ける仕様で、報酬を「毎秒の割合」として扱う設計になっている。
    ところがこの環境は 1 env ステップが 1 手 = 約 2.9 秒と極端に長いため、
    「1 手あたり +1.0」をそのまま書くと +2.9 になってしまう。

    そこで ここには **「1 手あたりの値」をそのまま書き**、
    `KnotTyingEnvCfg.__post_init__` が `step_dt` で割って辻褄を合わせる。

    ## 交差数を数える報酬から、位相クラスのラダーへ (v0 からの変更)

    v0 は `crossing_progress` (Δ交差数) を密報酬にしていたが、
    **独立した R1 ツイストを 3 回作れば交差数 3 になる**という抜け道が
    あり、固結びには一歩も近づかない。そこで交差数ではなく
    **位相クラス** (`common/topology/knot_detect.py`) を段とするラダーに変え、
    ポテンシャル整形 `F = γΦ(s') − Φ(s)` で登らせる。

    ラダーが「結ぶ動作の途中経過」に対応していることは、三葉結びを
    途中まで描くと段が 0 -> 1 -> 2 -> 3 と順に上がることで確認してある
    (`tests/test_knot_detect.py::test_partial_trefoil_walks_up_the_ladder`)。
    """

    # 密報酬: 位相ラダーをポテンシャル整形で登らせる。
    # `gamma` はモジュール定数 `GAMMA` から取る (PPO 側と共有)。
    shaping = RewTerm(
        func=mdp.knot_class_shaping,
        weight=1.0,
        params={"gamma": GAMMA, "potentials": (0.0, 0.35, 0.70, 1.0)},
    )

    # 疎報酬: 固結びができ、かつ床に落ちているか。
    # **ハンド退避後に測る**のが最重要 (rewards.py の docstring 参照)。
    # 重みが v0 (交差 1 つ) の 3 倍なのは、目標がそれだけ稀になるため。
    success = RewTerm(
        func=mdp.overhand_achieved,
        weight=3.0,
        params={"settled_height_diameters": 3.0, "rope_diameter": MISSING},
    )

    # 罰則: 交差はあるがラダーに乗っていない (独立ツイストなど行き止まり)。
    offladder = RewTerm(func=mdp.offladder_penalty, weight=-0.05)

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
class KnotTyingV1EnvCfg(KnotTyingEnvCfg):
    """v1 の環境設定。ロープのプリセット別の派生は下の `make_preset_cfgs`。"""

    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()

    actions_per_episode: int = 8
    """1 エピソードで打てる手の数 (v0 は 5)。

    固結びの位相的な下限は 3 遷移 (移植元のレシピ
    `Reide1 -> cross -> cross`)。ただし物理的には 1 遷移あたり約 2 手
    かかる — Step 5 の台本も **1 交差を作るのに 2 手** 使っている
    (1 手目でたるみを作り、2 手目で渡す)。3 x 2 = 6 手に、把持失敗や
    発散からの復帰ぶんを足して 8 手とした。
    """


# ロープのプリセット 5 種 x (通常 / Play) の設定クラスを生成する。
# 生成されるのは TwistEnvCfg / SimpleEnvCfg / StiffEnvCfg / FineEnvCfg / KnotEnvCfg と
# それぞれの *PlayEnvCfg。`__init__.py` が文字列 entry point で引く。
globals().update(make_preset_cfgs(KnotTyingV1EnvCfg, __name__))
