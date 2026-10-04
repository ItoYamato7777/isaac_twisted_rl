"""紐モデルの比較 (`rope_study`) の環境設定。**MDP は v4 のまま、紐の表現だけを差し替える。**

    uv run train --task Isaac-Twisted-RL-RS-XpbdC28-v4 --headless
    uv run python -u src/isaac_twisted_rl/scripts/rsl_rl/eval_chain.py --task Isaac-Twisted-RL-RS-XpbdC28-Play-v4 ...

v4 (段ごとのエージェント・1 遷移 = 1 エピソード・+1/0/−1) を選んだのは、2026-10-02 時点の chain 評価で
いちばん良い版だから (`../versions/v6/Result_Report_v6.md`)。v5 の発散対策 (離す高さ・めり込み解消速度) は
PhysX の剛体鎖に固有の手当てで、他の表現に持ち込むと比較が歪むので入れない。

観測・行動・報酬・終了・アーカイブ・PPO は v4 の設定クラスを継承してそのまま使う。v4 と違うのは:

1. 紐 (`scene.rope` / `scene.rope_native` / `scene.rope_fem` / XPBD) とその摩擦を `models.MODELS` から選ぶ。
2. 環境クラスが `common/rope_backends/env.RopeStudyEnv` (アダプタを `scene["rope"]` に差し込む)。
3. 観測・行動は 28 本の仮想リンク (弧長で等分) で表す。28 リンクの剛体鎖 (`Art28`) では実リンクと
   一致するので、`Art28` は v4 の `Knot` と同じ環境になる。
4. グリッパー (指のゲイン) は **すべてのモデルで `Knot` と同じ**。v0〜v7 は指のゲインを紐のリンク質量から
   決めていた (`RopeSpec.grasp_finger_stiffness`) が、ロボットは紐の表現によって変わらないので固定する。
5. 初期アーカイブは、モデルごとに同じ台本 (`demos/overhand_script.py --dump_archive`) で作ったもの
   (`outputs/archive_seed/rope_study/<モデル>.pt`)。無ければ警告して空から始める。
"""

from __future__ import annotations

import pathlib

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, DeformableObjectCfg
from isaaclab.envs import mdp as isaac_mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from ..common.env_cfg_base import KnotTyingSceneCfg
from ..common.parallel_gripper.gripper_cfg import PARALLEL_GRIPPER_CFG
from ..common.rope_backends.backend_cfg import RopeBackendCfg
from ..common.rope_backends.noisy_action import NoisyPickPlaceActionCfg
from ..common.rope_model.rope_cfg import make_rope_cfg
from ..common.rope_model.rope_specs import get_spec
from ..versions.v4.env_cfg import KnotTyingV4EnvCfg, _play_post_init
from .models import MODELS, RopeModel

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[5]
SEED_DIR = _REPO_ROOT / "outputs" / "archive_seed" / "rope_study"
"""モデルごとの初期アーカイブの置き場所 (`overhand_script.py --dump_archive <ここ>/<モデル>.pt`)。"""

_ROPE_INIT_Z = 0.06
_HOME_PALM_Z = 0.30
_KNOT = get_spec("knot")
FINGER_STIFFNESS = _KNOT.grasp_finger_stiffness
"""指の PD ゲイン [N/m]。全モデル共通で `Knot` と同じ (1507 N/m)。"""

ROPE_LENGTH = _KNOT.rope_length
ROPE_DIAMETER = _KNOT.diameter


@configclass
class RopeStudySceneCfg(KnotTyingSceneCfg):
    """紐の資産を表現ごとに別の名前で持てるシーン。使わないものは None (InteractiveScene が飛ばす)。"""

    rope: ArticulationCfg | None = None
    rope_native: ArticulationCfg | None = None
    rope_fem: DeformableObjectCfg | None = None


@configclass
class RopeStudyEnvCfg(KnotTyingV4EnvCfg):
    """v4 の設定に、紐のバックエンドを足したもの。"""

    scene: RopeStudySceneCfg = RopeStudySceneCfg(num_envs=4096, env_spacing=3.0)
    rope_backend: RopeBackendCfg = RopeBackendCfg()
    rope_model: str = ""


def _fem_cfg(model: RopeModel) -> DeformableObjectCfg:
    """FEM の紐 (円柱メッシュ -> PhysX が六面体の格子から四面体の計算用メッシュを作る)。

    格子の解像度は「最も長い辺の分割数」なので、全長 1.14 m を 1 cm 角で刻むには 114 になる。
    断面は 2 x 2 区画しかなく、線形四面体は曲げでロックする (実際より硬くなる) ので、ヤング率は
    片持ち梁の試験で EI = 7.85e-4 N m^2 に合うよう較正する (`fem` の `youngs_modulus`)。
    """
    p = {
        "hex_resolution": 114,
        "youngs_modulus": 1.0e5,
        "poissons_ratio": 0.45,
        "solver_iterations": 16,
        "self_collision": True,
        "damping": 0.005,
    }
    p.update(model.fem)
    return DeformableObjectCfg(
        prim_path="{ENV_REGEX_NS}/RopeFem",
        spawn=sim_utils.MeshCylinderCfg(
            radius=ROPE_DIAMETER / 2.0,
            height=ROPE_LENGTH,
            axis="X",
            deformable_props=sim_utils.DeformableBodyPropertiesCfg(
                rest_offset=0.0,
                contact_offset=0.004,
                self_collision=p["self_collision"],
                self_collision_filter_distance=0.005,
                solver_position_iteration_count=p["solver_iterations"],
                simulation_hexahedral_resolution=p["hex_resolution"],
                vertex_velocity_damping=0.0,
            ),
            physics_material=sim_utils.DeformableBodyMaterialCfg(
                youngs_modulus=p["youngs_modulus"],
                poissons_ratio=p["poissons_ratio"],
                dynamic_friction=model.rope_friction if model.rope_friction is not None else 2.0,
                elasticity_damping=p["damping"],
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=_KNOT.total_mass),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.5, 0.1)),
        ),
        init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.0, 0.0, _ROPE_INIT_Z)),
    )


def _apply_model(cfg: RopeStudyEnvCfg, model: RopeModel) -> None:
    """モデルに従って紐・グリッパー・行動・バックエンド・初期アーカイブを埋める (`rope_task_cfg._apply_rope` の対応物)。"""
    cfg.rope_model = model.name
    bcfg = RopeBackendCfg(kind=model.kind, radius=ROPE_DIAMETER / 2.0, init_height=_ROPE_INIT_Z)
    rope_entity = "rope"

    # -- 紐
    if model.kind in ("articulation", "articulation_resampled"):
        rope_cfg = make_rope_cfg(model.preset)
        rope_cfg.init_state.pos = (0.0, 0.0, _ROPE_INIT_Z)
        if cfg.solver_position_iterations is not None:
            rope_cfg.spawn.articulation_props.solver_position_iteration_count = cfg.solver_position_iterations
        if cfg.rope_max_depenetration_velocity is not None:
            rope_cfg.spawn.rigid_props.max_depenetration_velocity = cfg.rope_max_depenetration_velocity
        if model.kind == "articulation":
            rope_cfg.prim_path = "{ENV_REGEX_NS}/Rope"
            cfg.scene.rope = rope_cfg
        else:
            rope_cfg.prim_path = "{ENV_REGEX_NS}/RopeNative"
            cfg.scene.rope_native = rope_cfg
            bcfg.native_name = rope_entity = "rope_native"
        if model.rope_friction is not None:
            cfg.events.rope_material = EventTerm(
                func=isaac_mdp.randomize_rigid_body_material,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg(rope_entity),
                    "static_friction_range": (model.rope_friction, model.rope_friction),
                    "dynamic_friction_range": (model.rope_friction, model.rope_friction),
                    "restitution_range": (0.0, 0.0),
                    "num_buckets": 1,
                },
            )
    elif model.kind == "xpbd":
        bcfg.xpbd_params = dict(model.xpbd)
        bcfg.xpbd_params.setdefault("finger_stiffness", FINGER_STIFFNESS)
    elif model.kind == "fem":
        cfg.scene.rope_fem = _fem_cfg(model)
        # PhysX は変形体の物理の複製 (replicate_physics) に対応していない ("Replication of this type is not
        # supported")。複製すると env_0 以外の変形体が物理に入らず、節点の書き込みで device-side assert になった
        cfg.scene.replicate_physics = False
        bcfg.native_name = "rope_fem"
    else:
        raise ValueError(model.kind)
    cfg.rope_backend = bcfg

    # -- 床の摩擦
    cfg.scene.ground.spawn.physics_material.static_friction = model.ground_friction
    cfg.scene.ground.spawn.physics_material.dynamic_friction = model.ground_friction
    # -- 指の摩擦材 (PhysX の紐にだけ効く。XPBD は xpbd_params の friction_finger)
    cfg.events.grasp_material.params["friction"] = model.finger_friction

    # -- グリッパー (全モデル共通。`rope_task_cfg._apply_rope` と同じ設定、指のゲインは Knot の値で固定)
    gripper_cfg = PARALLEL_GRIPPER_CFG.copy()
    gripper_cfg.prim_path = "{ENV_REGEX_NS}/Gripper"
    gripper_cfg.init_state.pos = (0.0, 0.0, _HOME_PALM_Z)
    gripper_cfg.spawn.articulation_props.fix_root_link = False
    gripper_cfg.spawn.rigid_props.disable_gravity = True
    gripper_cfg.spawn.articulation_props.solver_position_iteration_count = cfg.solver_position_iterations or 16
    gripper_cfg.spawn.articulation_props.solver_velocity_iteration_count = 1
    finger = gripper_cfg.actuators["finger_actuator"]
    finger.stiffness = FINGER_STIFFNESS
    finger.damping = 0.04 * FINGER_STIFFNESS
    finger.effort_limit_sim = 100.0
    cfg.scene.gripper = gripper_cfg

    # -- 行動 (寸法は全モデル共通 = Knot)
    # 実行誤差 (`exec_noise_xy` / `exec_noise_yaw`) は既定 0。頑健性の評価で Hydra から上書きする
    cfg.actions.pick_place = NoisyPickPlaceActionCfg(
        asset_name="gripper",
        rope_asset_name="rope",
        finger_joint_names=["left_finger_joint", "right_finger_joint"],
        rope_length=ROPE_LENGTH,
        rope_diameter=ROPE_DIAMETER,
        durations=cfg.durations,
        limits=cfg.grasp_limits,
    )
    for group in (cfg.rewards, cfg.terminations):
        for term in group.__dict__.values():
            if hasattr(term, "params") and "rope_diameter" in term.params:
                term.params["rope_diameter"] = ROPE_DIAMETER

    # -- 初期アーカイブ (モデルごと)
    stage_term = cfg.events.reset_to_stage
    stage_term.params["load_paths"] = (str(SEED_DIR / f"{model.name}.pt"),)
    stage_term.params["capacity"] = model.archive_capacity


def _make_cfgs() -> dict[str, type]:
    out: dict[str, type] = {}
    for name, model in MODELS.items():

        def _post_init(self, _model=model):
            RopeStudyEnvCfg.__post_init__(self)
            _apply_model(self, _model)

        env_cls = configclass(
            type(
                f"{name}EnvCfg",
                (RopeStudyEnvCfg,),
                {"__doc__": model.description, "__module__": __name__, "__post_init__": _post_init},
            )
        )

        def _play_init(self, _base=env_cls):
            _base.__post_init__(self)
            self.scene.num_envs = 9
            self.sim.render_interval = 8
            _play_post_init(self)

        play_cls = configclass(
            type(
                f"{name}PlayEnvCfg",
                (env_cls,),
                {"__doc__": f"{name} の chain 評価用", "__module__": __name__, "__post_init__": _play_init},
            )
        )
        out[f"{name}EnvCfg"] = env_cls
        out[f"{name}PlayEnvCfg"] = play_cls
    return out


globals().update(_make_cfgs())
