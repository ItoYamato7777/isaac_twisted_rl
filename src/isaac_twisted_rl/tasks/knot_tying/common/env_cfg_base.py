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

`docs/decisions/0001-kinematic-hand-and-one-move.md` の決定どおり **キネマティックハンドのまま**。アームを
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
                # 平行移動になってしまい交差が作れない (docs/decisions/0002-action-space-rope-relative.md)。
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
    """行動空間。`docs/decisions/0002-action-space-rope-relative.md` のロープ相対 + Δlink。"""

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

    # NaN / 吹き飛びはその場で reset (docs/decisions/0003-physics-240hz.md)。
    diverged = DoneTerm(func=mdp.rope_diverged)


##
# 環境設定
##


@configclass
class KnotTyingEnvCfg(ManagerBasedRLEnvCfg):
    """紐結び環境の基底。プリセットごとの派生は `rope_task_cfg.py`。"""

    scene: KnotTyingSceneCfg = KnotTyingSceneCfg(num_envs=4096, env_spacing=3.0)
    """既定の並列環境数。

    RTX 3090 (24 GB) で実測: 4096 env のシーン構築時ピーク VRAM は約 10 GB
    (2 GPU 共有環境でも安全マージンを残せる)。ロープの物理は GPU 並列なので
    `num_envs` を増やすのが学習速度に最も効く (`docs/history/next_action.md` §4.8.9)。
    ただし 1 iteration の時間のほぼ全部は PhysX の物理ステップで (観測・報酬・p-data は誤差程度)、
    1 物理ステップ ≈ 34 ms の固定費 + 0.019 ms x 環境数なので、環境数を増やす効果は 4096 でほぼ
    頭打ちになる (8192 でも +20% 程度の見込み。`docs/experiments/2026-09-30_v4_speed.md`、
    `docs/history/next_action.md` §4.8.9)。そのため 4096 に留めた。
    """
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

    solver_position_iterations: int | None = None
    """ロープとグリッパーの PhysX ソルバの位置反復回数の上書き。`None` なら各資産の既定値のまま。

    既定値はロープ `min(32, max(16, リンク数 // 2))` (`rope_cfg.py`)、グリッパー 16。v4 で追加
    (`None` の既定では v0〜v3 は変わらない)。`actions.pick_place` と同じく、プリセット適用
    (`rope_task_cfg._apply_rope`) がロープとグリッパーの `articulation_props` に書き込む。

    **反復回数が物理ステップの時間のほぼ全部を決める** (2026-09-30 実測、RTX 3090、`Knot`、
    v4、ランダム方策): 1 物理ステップ ≈ 34 ms (環境数によらない固定費) + 0.019 ms x 環境数
    (反復 16)。4096 環境のスループットは反復 16 / 8 / 4 で 47 / 87 / 144 手/s。PhysX は
    シーン内の最大値で解くので、グリッパーだけ下げても速くならない。

    Hydra の CLI からは効かない (`__post_init__` の後に上書きされるため)。CLI では
    `env.scene.rope.spawn.articulation_props.solver_position_iteration_count=...` と
    `env.scene.gripper.spawn.articulation_props.solver_position_iteration_count=...` を直接指定する。
    """

    rope_max_depenetration_velocity: float | None = None
    """ロープのめり込み解消速度の上限 [m/s] の上書き。`None` なら `rope_cfg.py` の既定 (半径から
    スケールした値。`Knot` で 0.96 m/s) のまま。v5 で追加 (`None` の既定では v0〜v4 は変わらない)。

    `solver_position_iterations` と同じく、プリセット適用 (`rope_task_cfg._apply_rope`) がロープの
    `rigid_props` に書き込む。根拠は `versions/v5/README.md` (下げると発散が減る)。
    """

    rope_size_scales: tuple[tuple[float, float], ...] | None = None
    """env ごとに割り当てるロープの寸法 (長さの倍率, 直径の倍率) の並び。`None` ならプリセットのロープ 1 種のまま。v16 で追加
    (`None` の既定では v0〜v15 は変わらない)。

    値があると、プリセット適用 (`rope_task_cfg._apply_rope`) がロープの spawn を `RopeSizeSpawnerCfg` にし、
    `scene.replicate_physics` を偽にする (`rope_task_cfg.apply_rope_sizes`)。`solver_position_iterations` と同じく
    Hydra の CLI からは効かない。CLI (評価で寸法を変えるとき) は `env.scene.rope.spawn.scales=...` を直接指定する。
    仕様は `docs/decisions/0029-v16-rope-size-dr.md`。
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
        # `docs/decisions/0003-physics-240hz.md` の発散頻度の実測 (twist,
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

        # `gpu_max_rigid_patch_count` / `gpu_max_rigid_contact_count` は
        # IsaacLab の既定 (163840 / 8388608) のままだと 4096 env で
        # 学習序盤 (~150 iter, 方策がまだ緩い結び目しか作れない段階) で
        # 早くも "Patch buffer overflow" で落ちる (2026-09-30 実測,
        # `logs/rsl_rl/knot_tying_v3/2026-09-28_17-28-57`)。結び目が
        # 締まるほど 1 env あたりの自己接触パッチ数は増えるため、既定値は
        # env 数を増やした構成 (num_envs=4096) には全く足りない。
        # エラーメッセージが要求した実測下限 (188392) に対して
        # 十分な安全マージンを載せて 4 倍の 2**19 (524288) に設定する。
        # VRAM 予算: 4096 env で従来の実測ピークが約 10 GB (24 GB 中) なので、
        # このバッファ増分 (数百 MB 程度) を載せても余裕がある。
        self.sim.physx.gpu_max_rigid_patch_count = 2**19
        self.sim.physx.gpu_max_rigid_contact_count = 2**24

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
