"""紐結び RL 環境の**バージョン非依存**な基底設定。

シーン (ロープ + キネマティックハンド + 地面)、物理の刻み、
「1 手 = 1 RL ステップ」を成立させる `decimation` の計算、
行動・終了判定・イベントを定義する。

**観測と報酬はここには無い。** それらは課題そのもので、バージョンごとに
変わるため `versions/vN/env_cfg.py` が定義する
(設計の意図は `../README.md`)。

## 1 手 = 1 RL ステップ

`decimation` を「pick&place 1 手ぶんの物理ステップ数」に設定するのがこの
環境の要。`__post_init__` が `PhaseDurations` の合計から自動計算するので、
フェーズの尺を変えれば decimation も追随する (手で足した値がずれる事故を防ぐ)。

詳細は `mdp/actions.py` の docstring を参照。

## ロボットの形態

`next_action.md` 4.0 の決定どおり **キネマティックハンドのまま**。アームを
付けないので IK 失敗・可達領域外・自己衝突・特異点が無く、学習が
「腕を動かす」ではなく「紐を操作する」ことだけに集中できる。
ハンドは `write_root_pose_to_sim` で直接駆動する (`mdp/actions.py`)。
"""

from __future__ import annotations

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from . import mdp

from .mdp import phases
from .rope_model.rope_specs import GROUND_FRICTION

##
# シーン定義
##


@configclass
class KnotTyingSceneCfg(InteractiveSceneCfg):
    """ロープ 1 本 + キネマティックハンド 1 つ + 地面。

    ロープとグリッパーはプリセットごとに寸法・ゲインが変わるので、
    派生クラス (`rope_task_cfg.py`) が埋める。
    """

    rope: ArticulationCfg = MISSING
    gripper: ArticulationCfg = MISSING

    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(
            physics_material=sim_utils.RigidBodyMaterialCfg(
                # ロープが床の上で滑りすぎると、掴んで運ぶ操作がロープ全体の
                # 平行移動になってしまい交差が作れない (next_action.md 3(b))。
                static_friction=GROUND_FRICTION,
                dynamic_friction=GROUND_FRICTION,
                restitution=0.0,
            ),
        ),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2000.0),
    )


##
# MDP 設定
##


@configclass
class ActionsCfg:
    """行動空間。`next_action.md` 4.1 のロープ相対 + Δlink。"""

    pick_place: mdp.PickPlaceActionCfg = MISSING


@configclass
class EventCfg:
    """イベント。"""

    # 把持は接触摩擦だけで成立するので、この bind を取りこぼすと
    # 「エラーは出ないがロープが指をすり抜ける」(mdp/events.py 参照)。
    grasp_material = EventTerm(func=mdp.bind_grasp_material, mode="startup")

    # body 順序が狂うと例外なく p-data が壊れるので、起動時に物理的に検算する。
    link_order = EventTerm(func=mdp.check_rope_link_order, mode="startup")

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")


@configclass
class TerminationsCfg:
    """終了判定。"""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # NaN / 吹き飛びはその場で reset (next_action.md 4.5)。
    diverged = DoneTerm(func=mdp.rope_diverged)


##
# 環境設定
##


@configclass
class KnotTyingEnvCfg(ManagerBasedRLEnvCfg):
    """紐結び環境の基底。プリセットごとの派生は `rope_task_cfg.py`。"""

    scene: KnotTyingSceneCfg = KnotTyingSceneCfg(num_envs=1024, env_spacing=3.0)
    actions: ActionsCfg = ActionsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    # -- バージョンごとに変わるもの。`versions/vN/env_cfg.py` が埋める。
    observations: object = MISSING
    """観測の構成。課題が変われば渡すべき特徴も変わるのでバージョン側で定義する。"""

    rewards: object = MISSING
    """報酬の構成。**ここがバージョン間の主な差**。`versions/vN/env_cfg.py` 参照。"""

    durations: phases.PhaseDurations = phases.PhaseDurations()
    """1 手を構成するフェーズの尺。decimation はこの合計から決まる。"""

    grasp_limits: phases.GraspLimits = phases.GraspLimits()
    """行動のチャネルを物理量へ変換するときの範囲 (`PickPlaceActionCfg.limits` へ渡る)。

    v3 で追加した (既定値は従来の `GraspLimits()` なので v0〜v2-R2 は変わらない)。
    `actions.pick_place` はプリセット適用 (`rope_task_cfg._apply_rope`) で作られるので、
    バージョン側の設定からはここを通して変える (`durations` と同じ経路)。
    """

    actions_per_episode: int = MISSING
    """1 エピソードで打てる手の数。

    課題の難しさに直結するのでバージョンごとに決める
    (v0 は 5 手、v1 は 8 手)。根拠は各 `versions/vN/README.md`。
    """

    _reward_scale_applied: float = 1.0
    """報酬の重みに既に適用済みの `step_dt`。二重スケールを防ぐための内部状態。"""

    def __post_init__(self):
        # 物理は 240 Hz (デモと同じ)。**120 Hz へ落とす案は実測で却下した**。
        #
        # `next_action.md` 4.5 の「発散頻度を実測すること」の結果 (twist,
        # 16 環境 x 10 手, ランダム方策):
        #
        #     120 Hz          発散 11.2%
        #     240 Hz          発散  1.2%   <- これを採用
        #     240 Hz + rope_physx  発散  3.8%
        #
        # 120 Hz だと把持中の接触でロープの速度が 60〜175 m/s (正常時は
        # 1 m/s 程度) まで跳ね上がり NaN になる。原因は `rope_cfg.py` の
        # `max_depenetration_velocity` が **240 Hz を前提に**
        # 「1 ステップで半径の 40% 以上は押し戻さない」を満たすよう
        # スケールされていること (`_SIM_HZ_REF = 240.0`)。120 Hz では
        # 1 ステップの押し戻し量が倍になり、この前提が崩れる。
        #
        # `make_rope_physx_cfg` の接触設定を併用すると却って悪化するのは、
        # あれがシーン全体の設定を変えてグリッパーの押し付けにも影響する
        # ため (`rope_cfg.py` の warning に明記されている)。
        self.sim.dt = 1.0 / 240.0

        # ここが本環境の要。1 手ぶんの物理ステップ数を decimation にすることで
        # 「env.step() 1 回 = pick&place 1 手」になる (mdp/actions.py 参照)。
        self.decimation = self.durations.decimation(self.sim.dt)
        self.episode_length_s = self.actions_per_episode * self.decimation * self.sim.dt

        # 描画は 1 手に 1 回で十分 (GUI での観察用)。毎ステップ描くと極端に遅い。
        self.sim.render_interval = self.decimation

        self.viewer.eye = (1.2, 1.2, 0.9)
        self.viewer.lookat = (0.0, 0.0, 0.05)

        # ロープは 20〜48 body の自己接触ありなので接触ペアが多い。
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.bounce_threshold_velocity = 0.05

        self._rescale_reward_weights()

    def _rescale_reward_weights(self):
        """報酬の重みを `step_dt` で割る (`RewardsCfg` の docstring 参照)。

        `RewardManager.compute()` が各項に `dt` を掛けるので、ここで割って
        おくことで「1 手あたり」で書いた重みがそのまま実効値になる。

        `sim.dt` や `decimation` を変える派生クラス (`FineEnvCfg`) は、
        **変更後にこれを呼び直すこと**。二重に割らないよう、適用済みの
        `step_dt` を覚えておいて差分だけを補正する。
        """
        step_dt = self.decimation * self.sim.dt
        applied = getattr(self, "_reward_scale_applied", 1.0)
        factor = applied / step_dt
        if factor == 1.0:
            return
        for _term_name in self.rewards.__dict__:
            _term = getattr(self.rewards, _term_name)
            if hasattr(_term, "weight"):
                _term.weight = _term.weight * factor
        self._reward_scale_applied = step_dt
