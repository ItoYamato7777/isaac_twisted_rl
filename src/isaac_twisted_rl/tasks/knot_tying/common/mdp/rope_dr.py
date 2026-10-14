"""ロープの物理のドメインランダム化 (DR) のイベント項と、引いた値の観測項。v14 で追加。

| 項 | 種類 | 役割 |
|---|---|---|
| `RandomizeRopePhysics` | event (`mode="reset"`) | reset のたびに env ごとに 6 軸の倍率を引き、剛性・減衰・質量・摩擦を書く。引いた倍率を持つ |
| `rope_physics_params` | 観測 | 引いた倍率を [−1, 1] に正規化した `(N, 6)`。critic の特権観測に使う (actor には入れない) |

値の引き方・正規化・減衰の式は `../domain_rand.py` (純 torch)。仕様と範囲の理由は
`docs/decisions/0027-v14-s0-reshape-dr-priv-critic.md`。

## 自前の項にする理由

Isaac Lab の randomize 系の関数は、引いた値を残さず (critic に渡せない)、要素ごと (関節ごと・形状ごと) に
独立に引く。ここでは env ごとに 1 組の倍率を引いて全リンクに同じに掛け、キャッシュする。剛性と減衰は
`randomize_actuator_gains` も使えない (ロープは actuator を持たず `actuators={}`、ばねは USD の joint drive)。

## 公称値

公称値は起動時に PhysX から読んだ値 (`default_joint_stiffness`・`get_masses`・`get_material_properties`)。
倍率はこれに掛ける。摩擦の公称値はロープ・指とも 0.5 (PhysX の既定の材質) で、USD や `rope_specs.py` に書いた
2.0 / 4.0 ではない (`docs/experiments/2026-10-14_friction-binding.md`)。

## 何を書くか

* 曲げ (`JointNN:1`・`:2` = rotY / rotZ) と捩り (`JointNN:0` = rotX) のばね定数と減衰。DOF の名前の規則は
  2026-10-14 に PhysX から読んで確かめた (`:0` の公称のばね定数が捩りの値 0.0098)。
* リンクの質量と慣性 (同じ倍率)。
* ロープの全形状の静止・動摩擦。指 (`finger_body_names`) の形状の静止・動摩擦。指の範囲が 1 点
  (`(1.0, 1.0)`) なら指の材質には触れない。
* 床は全 env で 1 枚なので振らない。ロープと床の摩擦は average 合成で `(μ_rope + 2.0) / 2` になる。

`enabled=False` なら何も書かない (起動時の物理のまま。v13 と同じ) で、倍率は 1 のまま。

## reset の順序

`reset_to_stage` の後ろに置く (cfg の並び順 = 適用順)。物理パラメータの書き込みは形 (姿勢・関節角) を
変えず、この 2 つの間に物理ステップは無いので、順序は結果に効かない。

## アーカイブとの関係

アーカイブには物理パラメータを残さない。ある物理で保存した形を、別の物理の env へ書き戻す。剛いロープは
最初の手の間にまっすぐへ戻ろうとし、摩擦が低いと滑る。これは chain 評価で別の物理に出会うのと同じ扱いと見る。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg

from .. import domain_rand as dr

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

_I_BEND, _I_TWIST, _I_ZETA, _I_MASS, _I_ROPE_MU, _I_FINGER_MU = range(len(dr.AXES))


class RandomizeRopePhysics(ManagerTermBase):
    """reset のたびに、env ごとにロープの物理 (6 軸) を引き直して書く event 項。

    params (すべて `__call__` の引数):
        enabled: 偽なら何も書かない (公称の物理。Play の既定)。
        ranges: 軸 (`domain_rand.AXES`) -> 公称値に掛ける倍率の範囲 `(lo, hi)`。対数一様に引く。
        rope_cfg: ロープ。
        gripper_cfg: グリッパー。
        finger_body_names: 摩擦を振る指のリンク。
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        params = cfg.params
        self.ranges = _ranges(params.get("ranges", dr.NOMINAL_RANGES))
        rope: Articulation = env.scene[params.get("rope_cfg", SceneEntityCfg("rope")).name]
        gripper: Articulation = env.scene[params.get("gripper_cfg", SceneEntityCfg("gripper")).name]

        # -- 曲げと捩りの DOF。D6 の回転の DOF は rotX, rotY, rotZ の順に `<関節>:0`, `:1`, `:2` と並ぶ。
        suffix = [name.rsplit(":", 1)[-1] for name in rope.joint_names]
        if sorted(set(suffix)) != ["0", "1", "2"] or suffix.count("0") * 3 != len(suffix):
            raise RuntimeError(f"ロープの DOF の名前が想定 (<関節>:0/1/2) と違う: {rope.joint_names[:6]} ...")
        self._twist = torch.tensor([s == "0" for s in suffix], device=env.device)

        self._k0 = rope.data.default_joint_stiffness.clone()
        self._c0 = rope.data.default_joint_damping.clone()
        self._mass0 = rope.root_physx_view.get_masses().clone()
        self._inertia0 = rope.root_physx_view.get_inertias().clone()
        self._rope_mat0 = rope.root_physx_view.get_material_properties().clone()
        self._grip_mat0 = gripper.root_physx_view.get_material_properties().clone()
        self._finger_shapes = _shape_indices(gripper, params.get("finger_body_names", ("left_finger", "right_finger")))

        self.scales = torch.ones(env.num_envs, len(dr.AXES), device=env.device)
        """引いた倍率 `(N, 6)` (`domain_rand.AXES` の順)。`enabled=False` なら 1 のまま。"""

        print(
            f"[INFO] RandomizeRopePhysics: 捩り {int(self._twist.sum())} DOF・曲げ {int((~self._twist).sum())} DOF、"
            f"公称の摩擦 ロープ {float(self._rope_mat0[0, 0, 0]):.3g}・指 {float(self._grip_mat0[0, self._finger_shapes[0], 0]):.3g}、"
            f"範囲 {self.ranges}"
        )

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor,
        enabled: bool = True,
        ranges: Mapping[str, tuple[float, float]] = dr.NOMINAL_RANGES,
        rope_cfg: SceneEntityCfg = SceneEntityCfg("rope"),
        gripper_cfg: SceneEntityCfg = SceneEntityCfg("gripper"),
        finger_body_names: tuple[str, ...] = ("left_finger", "right_finger"),
    ) -> None:
        if not enabled:
            return
        if env_ids is None:
            env_ids = torch.arange(env.num_envs, device=env.device)
        ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
        ids_cpu = ids.cpu()
        s = dr.sample_scales(ids.numel(), self.ranges, device=env.device)
        self.scales[ids] = s
        rope: Articulation = env.scene[rope_cfg.name]

        # -- 剛性と減衰 (DOF ごとに曲げか捩りの倍率を選ぶ)
        k_scale = torch.where(self._twist.unsqueeze(0), s[:, _I_TWIST : _I_TWIST + 1], s[:, _I_BEND : _I_BEND + 1])
        c_scale = dr.damping_scale(k_scale, s[:, _I_MASS : _I_MASS + 1], s[:, _I_ZETA : _I_ZETA + 1])
        rope.write_joint_stiffness_to_sim(self._k0[ids] * k_scale, env_ids=ids)
        rope.write_joint_damping_to_sim(self._c0[ids] * c_scale, env_ids=ids)

        # -- 質量と慣性 (PhysX の CPU のテンソル。全 env の配列に書いて、ids の分だけを送る)
        s_cpu = s.cpu()
        m = s_cpu[:, _I_MASS]
        masses = rope.root_physx_view.get_masses()
        masses[ids_cpu] = self._mass0[ids_cpu] * m[:, None]
        rope.root_physx_view.set_masses(masses, ids_cpu)
        inertias = rope.root_physx_view.get_inertias()
        inertias[ids_cpu] = self._inertia0[ids_cpu] * m[:, None, None]
        rope.root_physx_view.set_inertias(inertias, ids_cpu)

        # -- 摩擦 (静止 = 動。反発は公称のまま)
        mats = rope.root_physx_view.get_material_properties()
        mats[ids_cpu, :, 0:2] = self._rope_mat0[ids_cpu, :, 0:2] * s_cpu[:, _I_ROPE_MU, None, None]
        rope.root_physx_view.set_material_properties(mats, ids_cpu)
        if tuple(self.ranges["finger_friction"]) != (1.0, 1.0):
            gripper: Articulation = env.scene[gripper_cfg.name]
            gm = gripper.root_physx_view.get_material_properties()
            sel = (ids_cpu[:, None], self._finger_shapes[None, :])
            gm[sel[0], sel[1], 0:2] = self._grip_mat0[sel[0], sel[1], 0:2] * s_cpu[:, _I_FINGER_MU, None, None]
            gripper.root_physx_view.set_material_properties(gm, ids_cpu)


def _ranges(ranges: Mapping[str, tuple[float, float]]) -> dict[str, tuple[float, float]]:
    """全軸がそろった `{軸: (lo, hi)}` (Hydra の上書きはリストで来るので tuple にする)。"""
    unknown = set(ranges) - set(dr.AXES)
    if unknown:
        raise ValueError(f"未知の軸: {sorted(unknown)} (軸は {dr.AXES})")
    out = dict(dr.NOMINAL_RANGES)
    out.update({k: (float(v[0]), float(v[1])) for k, v in ranges.items()})
    return out


def _shape_indices(asset: Articulation, body_names: tuple[str, ...]) -> torch.Tensor:
    """`get_material_properties` の形状の列のうち、`body_names` のリンクに属する列 (CPU の long)。

    Articulation は形状とリンクの対応を持たないので、リンクごとの rigid body view で形状の数を数える
    (Isaac Lab の `randomize_rigid_body_material` と同じ方法)。
    """
    counts, names = [], []
    for path in asset.root_physx_view.link_paths[0]:
        counts.append(asset._physics_sim_view.create_rigid_body_view(path).max_shapes)
        names.append(path.rsplit("/", 1)[-1])
    if sum(counts) != asset.root_physx_view.max_shapes:
        raise RuntimeError(f"形状の数が合わない: リンクごと {counts}、全体 {asset.root_physx_view.max_shapes}")
    cols = []
    for name in body_names:
        if name not in names:
            raise RuntimeError(f"リンク {name} が無い ({names})")
        i = names.index(name)
        start = sum(counts[:i])
        cols.extend(range(start, start + counts[i]))
    return torch.tensor(cols, dtype=torch.long)


def _get_dr_term(env: ManagerBasedRLEnv, event_term: str) -> RandomizeRopePhysics:
    term = env.event_manager.get_term_cfg(event_term).func
    if not isinstance(term, RandomizeRopePhysics):
        raise TypeError(f"event term '{event_term}' is not RandomizeRopePhysics (got {type(term).__name__})")
    return term


def rope_physics_params(env: ManagerBasedRLEnv, event_term: str = "rope_dr") -> torch.Tensor:
    """観測: 引いた物理の倍率を [−1, 1] に正規化した `(N, 6)` (`domain_rand.normalize`)。振らない軸は 0。"""
    term = _get_dr_term(env, event_term)
    return dr.normalize(term.scales, term.ranges)
