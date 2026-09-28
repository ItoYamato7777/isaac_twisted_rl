"""イベント項 (startup / reset)。

## 摩擦材の bind がマルチ env で必要になる理由

`demos/rope_catch.py` / `demos/topology_debug.py` は把持を **接触摩擦だけ** で
成立させている。固定ジョイントで拘束しないので「ロープのどのリンクでも
掴める」という性質が得られ、それが `grasp_link_id` を自由に選ばせる前提に
なっている。この摩擦把持は指のプリムに高摩擦材 (μ=4.0,
`friction_combine_mode="max"`) を bind して初めて成立する。

デモは 1 環境なので `/World/Gripper/left_finger` へ直接 bind すれば済んだ。
マルチ env では `/World/envs/env_*/Gripper/left_finger` が env の数だけ
存在する。`sim_utils.bind_physics_material` は **正規表現を受け付けず**、
存在しないパスには `ValueError` を投げるので、`find_matching_prim_paths` で
実際のパスへ展開してから 1 つずつ bind する。

これを忘れると「エラーは出ないがロープが指をすり抜ける」という、
最も気付きにくい壊れ方をする。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

_GRASP_MATERIAL_PATH = "/World/PhysicsMaterials/GraspMaterial"


def bind_grasp_material(
    env: ManagerBasedRLEnv,
    env_ids,
    friction: float = 4.0,
    finger_body_names: tuple[str, ...] = ("left_finger", "right_finger"),
    gripper_asset_name: str = "gripper",
):
    """全 env の指プリムに高摩擦材を bind する (`mode="startup"` で 1 回だけ)。

    Args:
        friction: 静止・動摩擦係数。デモの `GRASP_FRICTION` と同じ 4.0。
        finger_body_names: 指のリンク名。
        gripper_asset_name: シーン中のグリッパー資産名。
    """
    from isaacsim.core.utils.stage import get_current_stage

    stage = get_current_stage()
    if not stage.GetPrimAtPath(_GRASP_MATERIAL_PATH).IsValid():
        material_cfg = sim_utils.RigidBodyMaterialCfg(
            static_friction=friction,
            dynamic_friction=friction,
            restitution=0.0,
            # ロープ側の摩擦 (2.0) と平均されないよう max を取る。指の高摩擦を
            # 効かせるのが目的なので、combine_mode を既定 (average) のままだと
            # 実効摩擦が下がって滑る。
            friction_combine_mode="max",
            restitution_combine_mode="min",
        )
        material_cfg.func(_GRASP_MATERIAL_PATH, material_cfg)

    gripper_prim = env.scene[gripper_asset_name].cfg.prim_path
    bound = 0
    for finger in finger_body_names:
        pattern = f"{gripper_prim}/{finger}"
        for path in sim_utils.find_matching_prim_paths(pattern):
            sim_utils.bind_physics_material(path, _GRASP_MATERIAL_PATH)
            bound += 1
    if bound == 0:
        raise RuntimeError(
            f"把持用の摩擦材を bind できなかった (pattern='{gripper_prim}/<finger>')。"
            " 指のリンク名か prim_path を確認すること。"
            " これを取りこぼすとロープが指をすり抜けるが、例外は出ない"
        )
    print(f"[knot_tying] 把持用の摩擦材 (mu={friction}) を {bound} 個の指プリムに bind した")


def check_rope_link_order(env: ManagerBasedRLEnv, env_ids, rope_asset_name: str = "rope"):
    """ロープの body 順序を物理的に検算する (`mode="startup"`)。

    順序が狂うと **例外が出ないまま p-data だけが壊れる** ので、
    起動時に 1 回だけ隣接リンク距離の比を見ておく (`rope_state.py` の
    `check_link_order` の docstring 参照)。安い保険。
    """
    from .observations import get_extractor

    rope = env.scene[rope_asset_name]
    extractor = get_extractor(env, rope_asset_name)
    ratio = extractor.check_link_order(rope)
    print(f"[knot_tying] check_link_order: 隣接リンク距離の max/median = {ratio:.4f} (1.0 に近いほど健全)")
