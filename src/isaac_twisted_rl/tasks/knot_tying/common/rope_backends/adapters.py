"""紐のバックエンドを「Articulation と同じ読み書き口」に見せるアダプタ。紐モデルの比較 (`rope_study/`) で追加。

既存の MDP 項 (`mdp/`) は `env.scene["rope"]` から次のものしか読まない (2026-10-02 に grep で確認):

    data.body_pos_w / data.body_lin_vel_w     リンク中心の位置・速度 (観測・位相・発散判定・行動)
    data.root_link_pose_w / data.joint_pos    アーカイブへ保存する形状
    body_names / num_bodies / num_joints / find_bodies / device
    write_root_pose_to_sim / write_root_velocity_to_sim / write_joint_state_to_sim   アーカイブからの復元

アダプタはこの口を **仮想リンク** (中心線を弧長で `num_virtual_links` 等分した区間の中点。`centerline.py`)
で返す。したがって観測・行動・報酬・位相の判定は、紐の表現 (リンク数・剛体鎖か棒か FEM か) によらず
同じ意味になる。28 リンクの剛体鎖では仮想リンク = 実リンクなので、アダプタを通さない (v4 と完全に同じ)。

「関節」(`joint_pos`) はアーカイブ用の形状の入れ物として使う:

* 剛体鎖 (`ResampledArticulationRope`): 本物の Articulation の root 姿勢と関節角をそのまま。
* XPBD / FEM: 根元 (節点 0) を原点にした節点の座標 (+ Cosserat 棒なら区間の向き) を平らに並べたもの。
  root の向きは単位 quaternion として保存する。アーカイブは復元時に root を重心まわりに yaw で回す
  (`state_archive.place_at_origin`) ので、書き込み時に「root の姿勢 x 局所形状」で world の形状を作れば、
  剛体鎖と同じ規約でランダムな yaw 回転が効く。
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from .. import topology as tp
from . import centerline as cl

_LINK_FORMAT = "Link{:02d}"


class _AdapterData:
    """`Articulation.data` の代わり。読まれたときに計算し、物理ステップか書き込みのたびに捨てる。"""

    def __init__(self, adapter: RopeAdapter):
        self._a = adapter

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._a._cached("pos", self._a._virtual_pos)

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._a._cached("vel", self._a._virtual_vel)

    @property
    def root_link_pose_w(self) -> torch.Tensor:
        return self._a._root_pose()

    @property
    def joint_pos(self) -> torch.Tensor:
        return self._a._local_state()


class RopeAdapter:
    """アダプタの共通部分。派生クラスは節点列 (`nodes_w`)・その速度・形状の保存と復元を実装する。"""

    is_rope_adapter = True

    def __init__(self, num_envs: int, num_virtual_links: int, device: str, radius: float, env_origins: torch.Tensor):
        self.num_envs = num_envs
        self.num_instances = num_envs
        self.num_virtual_links = num_virtual_links
        self.device = device
        self.radius = radius
        self.env_origins = env_origins
        self.body_names = [_LINK_FORMAT.format(k) for k in range(num_virtual_links)]
        self.data = _AdapterData(self)
        self._cache: dict[str, torch.Tensor] = {}
        self._frac = cl.virtual_link_fractions(num_virtual_links, device)

    # ------------------------------------------------------------ Articulation 互換
    @property
    def num_bodies(self) -> int:
        return self.num_virtual_links

    @property
    def num_joints(self) -> int:
        return self.state_dim

    def find_bodies(self, names, preserve_order: bool = False):
        if isinstance(names, str):
            names = [names]
        ids = [self.body_names.index(n) for n in names]
        return ids, list(names)

    def write_root_velocity_to_sim(self, root_velocity: torch.Tensor, env_ids=None):
        """速度は形状の書き込み (`write_joint_state_to_sim`) で 0 にするので、ここでは何もしない。"""

    # ------------------------------------------------------------ 共通の計算
    def invalidate(self):
        self._cache.clear()

    def _cached(self, key, fn):
        if key not in self._cache:
            self._cache[key] = fn()
        return self._cache[key]

    def _virtual_pos(self) -> torch.Tensor:
        nodes = self.nodes_w()
        idx, t = cl.interp_weights(nodes, self._frac)
        self._cache["interp"] = (idx, t)
        return cl.gather_interp(nodes, idx, t)

    def _virtual_vel(self) -> torch.Tensor:
        if "interp" not in self._cache:
            self._virtual_pos()
        idx, t = self._cache["interp"]
        return cl.gather_interp(self.node_vel_w(), idx, t)

    # ------------------------------------------------------------ 派生クラスが実装する
    @property
    def state_dim(self) -> int:
        raise NotImplementedError

    def nodes_w(self) -> torch.Tensor:
        """中心線の節点 `(E, M, 3)` (world)。表現本来の分解能。"""
        raise NotImplementedError

    def node_vel_w(self) -> torch.Tensor:
        raise NotImplementedError

    def _root_pose(self) -> torch.Tensor:
        raise NotImplementedError

    def _local_state(self) -> torch.Tensor:
        raise NotImplementedError

    def write_root_pose_to_sim(self, root_pose: torch.Tensor, env_ids=None):
        raise NotImplementedError

    def write_joint_state_to_sim(self, joint_pos: torch.Tensor, joint_vel: torch.Tensor, env_ids=None):
        raise NotImplementedError

    def reset(self, env_ids: Sequence[int] | None = None):
        """シーンの reset (`InteractiveScene.reset`) と一緒に呼ばれる。外部ソルバはここで初期姿勢へ戻す。"""

    def update(self, dt: float):
        """物理ステップの後 (`InteractiveScene.update`) に呼ばれる。外部ソルバはここで 1 ステップ進める。"""
        self.invalidate()

    def bind(self, env) -> None:
        """物理が動き出した後 (`load_managers` の直前) に 1 回だけ呼ばれる。グリッパーなどを掴む。

        アダプタ自体はシーンを作った直後 (物理の開始前) に `scene["rope"]` へ入れる。v4 のアーカイブ
        (`ResetToStage`) は物理の開始時に初期化されて `scene["rope"].num_joints` を読むので、
        `load_managers` で入れるのでは間に合わない (`env.RopeStudyEnv` の docstring)。
        """


def _env_ids(env_ids, n: int, device) -> torch.Tensor:
    if env_ids is None or (isinstance(env_ids, slice) and env_ids == slice(None)):
        return torch.arange(n, device=device)
    return torch.as_tensor(env_ids, device=device, dtype=torch.long)


# ---------------------------------------------------------------- 剛体鎖 (リンク数が仮想リンクと違うとき)


class ResampledArticulationRope(RopeAdapter):
    """本物の Articulation (例: 56 リンク) を、仮想リンク (28 本) で見せる。書き込みは素通し。"""

    def __init__(self, articulation, num_envs: int, device: str, num_virtual_links: int, radius: float, env_origins: torch.Tensor):
        super().__init__(num_envs, num_virtual_links, device, radius, env_origins)
        self.native = articulation
        self._ids = None

    @property
    def _body_ids(self) -> torch.Tensor:
        # 名前による解決は Articulation の初期化 (物理の開始) の後でないとできない
        if self._ids is None:
            from ..rope_state import resolve_rope_bodies

            self._ids = torch.tensor(resolve_rope_bodies(self.native), dtype=torch.long, device=self.device)
        return self._ids

    @property
    def state_dim(self) -> int:
        return self.native.num_joints

    def nodes_w(self) -> torch.Tensor:
        """カプセルの軸の端点 (= 関節の位置) の列 `(E, L + 1, 3)`。リンクの向きから正確に出す。

        `topology.polyline_from_link_centers` (隣り合う中心の中点) は、曲がった鎖では関節の位置から
        ずれる (ずれ = 半リンク長 x 曲がり角の半分程度)。仮想リンクの中心を弧長で正しく取るため、
        カプセルの軸 (局所 x 軸、`generate_rope_usd.py`) の向きから端点を作る。
        """
        c = self.native.data.body_pos_w[:, self._body_ids]
        q = self.native.data.body_quat_w[:, self._body_ids]
        ex = torch.zeros_like(c)
        ex[..., 0] = 1.0
        half = 0.5 * (c[:, 1:] - c[:, :-1]).norm(dim=-1).mean(dim=1, keepdim=True).unsqueeze(-1)
        axis = cl.quat_rotate(q, ex) * half
        joints = 0.5 * ((c[:, :-1] + axis[:, :-1]) + (c[:, 1:] - axis[:, 1:]))
        return torch.cat([c[:, :1] - axis[:, :1], joints, c[:, -1:] + axis[:, -1:]], dim=1)

    def node_vel_w(self) -> torch.Tensor:
        return tp.polyline_from_link_centers(self.native.data.body_lin_vel_w[:, self._body_ids])

    def _root_pose(self) -> torch.Tensor:
        return self.native.data.root_link_pose_w

    def _local_state(self) -> torch.Tensor:
        return self.native.data.joint_pos

    def write_root_pose_to_sim(self, root_pose, env_ids=None):
        self.native.write_root_pose_to_sim(root_pose, env_ids=env_ids)
        self.invalidate()

    def write_root_velocity_to_sim(self, root_velocity, env_ids=None):
        self.native.write_root_velocity_to_sim(root_velocity, env_ids=env_ids)

    def write_joint_state_to_sim(self, joint_pos, joint_vel, env_ids=None):
        self.native.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self.invalidate()


# ---------------------------------------------------------------- 形状を「根元 + 局所形状」で持つ表現の共通部分


class _NodalRope(RopeAdapter):
    """節点 (と向き) を world で持つ表現。root = 節点 0 の位置 + 単位 quaternion。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._root = torch.zeros(self.num_envs, 7, device=self.device)
        self._root[:, 3] = 1.0

    def _root_pose(self) -> torch.Tensor:
        pose = torch.zeros(self.num_envs, 7, device=self.device)
        pose[:, :3] = self.nodes_w()[:, 0]
        pose[:, 3] = 1.0
        return pose

    def write_root_pose_to_sim(self, root_pose, env_ids=None):
        """root を置き換える。形状 (局所) はそのままで、world の形状を作り直す。"""
        ids = _env_ids(env_ids, self.num_envs, self.device)
        local = self._local_state()[ids]
        self._root[ids] = root_pose.to(self._root.dtype)
        self._apply_local(ids, local)
        self.invalidate()

    def write_joint_state_to_sim(self, joint_pos, joint_vel, env_ids=None):
        ids = _env_ids(env_ids, self.num_envs, self.device)
        self._apply_local(ids, joint_pos)
        self.invalidate()

    def _apply_local(self, ids: torch.Tensor, local: torch.Tensor):
        raise NotImplementedError


# ---------------------------------------------------------------- XPBD (Warp)


class XpbdRope(_NodalRope):
    """`xpbd_rod.XpbdRod` を環境につなぐ。手と指は PhysX のグリッパー (紐とは接触しない) から読む。

    **ソルバの状態は環境の原点を引いた座標 (環境ごとの局所座標) で持つ。** world 座標のまま解くと、原点が
    遠い環境 (4096 環境・間隔 3 m で最大 ±94 m) では float32 の刻み (94 m で 7.6e-6 m) が小ステップ 1 回の
    床の動摩擦の補正 (μ g h^2 = 3.3e-7 m (μ 0.5)、1.3e-6 m (μ 2.0)) より大きくなり、補正が丸めで消えて
    紐が床の上を一定の速さで滑り続けた (2026-10-02、`rope_study/Report_rope_models.md` §7.1.6)。
    world との変換は、読み出し (`nodes_w`)・書き込み (`_apply_local`, `reset`)・手 (`_read_hand`) だけで行う。
    """

    def __init__(self, num_envs: int, device: str, params, num_virtual_links: int, init_height: float, env_origins):
        from .xpbd_rod import XpbdRod

        super().__init__(num_envs, num_virtual_links, device, params.radius, env_origins)
        self.env = None
        self.gripper = None
        self.rod = XpbdRod(num_envs, params, device=str(device))
        self._org = env_origins.to(device=self.rod.torch_device, dtype=torch.float32).clone()
        self.init_height = init_height
        self._resync = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self._markers = None
        self.reset(None)

    def bind(self, env, gripper_name: str = "gripper") -> None:
        self.env = env
        self.gripper = env.scene[gripper_name]
        self._finger_ids, _ = self.gripper.find_joints(["left_finger_joint", "right_finger_joint"], preserve_order=True)
        self._read_hand()
        self.rod.sync_hand()

    @property
    def state_dim(self) -> int:
        return self.rod.num_state

    def nodes_w(self) -> torch.Tensor:
        return self.rod.nodes() + self._org.unsqueeze(1)

    def node_vel_w(self) -> torch.Tensor:
        return self.rod.node_vel()

    def _local_state(self) -> torch.Tensor:
        nodes = self.rod.nodes()
        local = nodes - nodes[:, :1]
        return torch.cat([local.reshape(self.num_envs, -1), self.rod.quat_wxyz().reshape(self.num_envs, -1)], dim=-1)

    def _apply_local(self, ids, local):
        P, N = self.rod.P, self.rod.N
        n = ids.numel()
        nodes_l = local[:, : 3 * P].reshape(n, P, 3)
        quats_l = local[:, 3 * P :].reshape(n, N, 4)
        root = self._root[ids]
        q_root = root[:, 3:7].unsqueeze(1)
        # root (world) から原点を先に引き、局所座標の中で形状を組む (world の大きい値を経由すると丸めが入る)
        root_l = (root[:, :3] - self._org[ids]).unsqueeze(1)
        nodes = root_l + cl.quat_rotate(q_root.expand(-1, P, -1), nodes_l)
        quats = cl.quat_mul(q_root.expand(-1, N, -1), quats_l)
        self.rod.set_state(ids, nodes, quats)

    def reset(self, env_ids=None):
        ids = _env_ids(env_ids, self.num_envs, self.device)
        nodes = self.rod.straight_nodes(torch.zeros_like(self._org[ids]), self.init_height)
        self.rod.set_state(ids, nodes)
        self._resync[ids] = True
        self.invalidate()

    def _read_hand(self):
        root = self.gripper.data.root_pose_w
        yaw = 2.0 * torch.atan2(root[:, 6], root[:, 3])
        hand = torch.cat([root[:, :3] - self._org, yaw.unsqueeze(-1)], dim=-1)
        fing = self.gripper.data.joint_pos[:, self._finger_ids]
        self.rod.set_hand(hand, fing)

    def update(self, dt: float):
        if self.gripper is None:  # bind の前 (ManagerBasedEnv.__init__ の初回の scene.update)
            self.invalidate()
            return
        self._read_hand()
        if bool(self._resync.any()):
            self.rod.sync_hand(self._resync.nonzero(as_tuple=False).squeeze(-1))
            self._resync[:] = False
        self.rod.step(dt)
        self.invalidate()
        self._draw()

    def _draw(self):
        """GUI があるときだけ、節点に球を描く (Play で紐を見るため。XPBD の紐は USD に存在しない)。"""
        if not self.env.sim.has_gui():
            return
        if self._markers is None:
            import isaaclab.sim as sim_utils
            from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

            cfg = VisualizationMarkersCfg(
                prim_path="/Visuals/XpbdRope",
                markers={
                    "node": sim_utils.SphereCfg(
                        radius=self.radius, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.2, 0.75))
                    )
                },
            )
            self._markers = VisualizationMarkers(cfg)
        nodes = self.nodes_w()
        # 区間の中点も足して、隙間の無い紐に見せる
        mids = 0.5 * (nodes[:, 1:] + nodes[:, :-1])
        pts = torch.cat([nodes, mids], dim=1).reshape(-1, 3)
        self._markers.visualize(translations=pts)


# ---------------------------------------------------------------- PhysX の FEM 変形体


class FemRope(_NodalRope):
    """Isaac Lab の `DeformableObject` (PhysX の FEM 四面体) の紐を、節点を軸方向の区画で平均した中心線で見せる。

    シミュレーション用メッシュの頂点を、静止形状での軸方向の座標 (x) で `num_bins` 個の区画に分け、
    区画ごとの頂点の平均を中心線の節点にする (区画の数は既定で仮想リンクの 2 倍)。両端の区画は
    メッシュの端面を含むので、節点を区画の中心ではなく端面まで外挿する。
    """

    def __init__(self, deformable, num_envs: int, device: str, num_virtual_links: int, radius: float, env_origins: torch.Tensor, num_bins: int = 0):
        super().__init__(num_envs, num_virtual_links, device, radius, env_origins)
        self.native = deformable
        self._num_bins_req = num_bins
        self._avg = None

    def _ensure_map(self):
        """静止形状の頂点を軸方向の区画に分ける (FEM の資産の初期化 = 物理の開始の後でないとできない)。"""
        if self._avg is not None:
            return
        num_bins = self._num_bins_req
        num_virtual_links = self.num_virtual_links
        rest = self.native.data.default_nodal_state_w[0, :, :3] - self.env_origins[0]
        x = rest[:, 0]
        x0, x1 = float(x.min()), float(x.max())
        self.num_bins = num_bins if num_bins > 0 else 2 * num_virtual_links
        b = ((x - x0) / max(x1 - x0, 1e-9) * self.num_bins).floor().clamp(0, self.num_bins - 1).long()
        onehot = torch.zeros(self.num_bins, x.numel(), device=self.device)
        onehot[b, torch.arange(x.numel(), device=self.device)] = 1.0
        counts = onehot.sum(dim=1, keepdim=True)
        if bool((counts == 0).any()):
            raise RuntimeError(f"FEM rope: 頂点の無い区画がある (num_bins={self.num_bins} が細かすぎる)")
        self._avg = onehot / counts  # (B, V)
        self._num_vertices = x.numel()

    @property
    def state_dim(self) -> int:
        return 3 * self.native.max_sim_vertices_per_body

    def _bin_means(self, field: torch.Tensor) -> torch.Tensor:
        self._ensure_map()
        return torch.einsum("bv,evd->ebd", self._avg, field)

    def nodes_w(self) -> torch.Tensor:
        c = self._bin_means(self.native.data.nodal_pos_w)
        # 区画の中心 -> 端面まで半区画ぶん外挿
        head = c[:, :1] + 0.5 * (c[:, :1] - c[:, 1:2])
        tail = c[:, -1:] + 0.5 * (c[:, -1:] - c[:, -2:-1])
        return torch.cat([head, c, tail], dim=1)

    def node_vel_w(self) -> torch.Tensor:
        c = self._bin_means(self.native.data.nodal_vel_w)
        return torch.cat([c[:, :1], c, c[:, -1:]], dim=1)

    def _local_state(self) -> torch.Tensor:
        pos = self.native.data.nodal_pos_w
        return (pos - self.nodes_w()[:, :1]).reshape(self.num_envs, -1)

    def _apply_local(self, ids, local):
        self._ensure_map()
        n = ids.numel()
        verts_l = local.reshape(n, self._num_vertices, 3)
        root = self._root[ids]
        q = root[:, 3:7].unsqueeze(1).expand(-1, self._num_vertices, -1)
        verts = root[:, :3].unsqueeze(1) + cl.quat_rotate(q, verts_l)
        state = torch.cat([verts, torch.zeros_like(verts)], dim=-1)
        self.native.write_nodal_state_to_sim(state, env_ids=ids)

    def update(self, dt: float):
        self.invalidate()
