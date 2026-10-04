"""紐のバックエンド層 (`common/rope_backends/centerline.py` と `xpbd_rod.py`) の単体テスト。紐モデルの比較で追加。

Isaac Sim を起動せずに走る (centerline は純 torch、XPBD は Warp の CPU 実行)::

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/test_rope_backends.py
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import types

import pytest
import torch

_COMMON = pathlib.Path(__file__).resolve().parents[1] / "src" / "isaac_twisted_rl" / "tasks" / "knot_tying" / "common"

# `rope_backends` を isaaclab 抜きで読む (パッケージの __init__ は docstring だけだが、親の
# `isaac_twisted_rl.tasks` を経由すると isaaclab を import してしまうので、直接ファイルから読む)。
_pkg = types.ModuleType("rope_backends")
_pkg.__path__ = [str(_COMMON / "rope_backends")]
sys.modules.setdefault("rope_backends", _pkg)


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"rope_backends.{name}", _COMMON / "rope_backends" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"rope_backends.{name}"] = mod
    spec.loader.exec_module(mod)
    return mod


cl = _load("centerline")
from topology import polyline_from_link_centers  # noqa: E402

# ---------------------------------------------------------------- centerline


def _wiggly_chain(num_links: int, spacing: float, seed: int = 0) -> torch.Tensor:
    """等間隔のリンク中心 (曲がった鎖) `(2, num_links, 3)`。"""
    g = torch.Generator().manual_seed(seed)
    ang = torch.cumsum(torch.randn(2, num_links, generator=g) * 0.4, dim=1)
    step = torch.stack([torch.cos(ang), torch.sin(ang), 0.1 * torch.randn(2, num_links, generator=g)], -1)
    step = step / step.norm(dim=-1, keepdim=True) * spacing
    return torch.cumsum(step, dim=1)


def test_virtual_links_equal_real_links_for_uniform_chain():
    """カプセルの軸の端点 (関節) の列から取った仮想リンクの中心は、剛体鎖のリンク中心と一致する。

    曲がった鎖の関節の位置を、リンク中心と「軸の向き」から作る (アダプタ `ResampledArticulationRope`
    と同じ作り方)。中点近似 (`polyline_from_link_centers`) では 1 cm 弱ずれるので、それとも比べる。
    """
    links = _wiggly_chain(28, 0.04)
    axis = links[:, 1:] - links[:, :-1]
    axis = torch.cat([axis[:, :1], 0.5 * (axis[:, 1:] + axis[:, :-1]), axis[:, -1:]], dim=1)
    # 各リンクの軸を、両側の関節が隣と一致するように組み直す (連続な剛体鎖)
    a = axis / axis.norm(dim=-1, keepdim=True) * 0.02
    joints = [links[:, 0] - a[:, 0]]
    centers = []
    for k in range(28):
        c = joints[-1] + a[:, k]
        centers.append(c)
        joints.append(c + a[:, k])
    centers = torch.stack(centers, 1)
    nodes = torch.stack(joints, 1)
    virt = cl.virtual_link_centers(nodes, 28)
    assert torch.allclose(virt, centers, atol=1e-5)
    approx = cl.virtual_link_centers(polyline_from_link_centers(centers), 28)
    assert (approx - centers).norm(dim=-1).max().item() < 0.25 * 0.04


def test_virtual_links_exact_for_straight_chain():
    links = torch.zeros(1, 28, 3)
    links[0, :, 0] = torch.arange(28) * 0.04
    virt = cl.virtual_link_centers(polyline_from_link_centers(links), 28)
    assert torch.allclose(virt, links, atol=1e-6)


def test_resample_preserves_length_and_endpoints():
    nodes = polyline_from_link_centers(_wiggly_chain(56, 0.02, seed=3))
    res = cl.resample(nodes, 200)
    assert torch.allclose(res[:, 0], nodes[:, 0], atol=1e-6)
    assert torch.allclose(res[:, -1], nodes[:, -1], atol=1e-6)
    l0 = cl.arc_lengths(nodes)[:, -1]
    l1 = cl.arc_lengths(res)[:, -1]
    assert torch.all(l1 <= l0 + 1e-5) and torch.all(l1 > 0.98 * l0)


def test_parallel_transport_frames_follow_tangent():
    nodes = polyline_from_link_centers(_wiggly_chain(28, 0.04, seed=1))
    q = cl.parallel_transport_frames(nodes)
    ez = torch.tensor([0.0, 0.0, 1.0]).expand(q.shape[0], q.shape[1], 3)
    d3 = cl.quat_rotate(q, ez)
    tan = nodes[:, 1:] - nodes[:, :-1]
    tan = tan / tan.norm(dim=-1, keepdim=True)
    assert torch.allclose(d3, tan, atol=1e-4)
    assert torch.allclose(q.norm(dim=-1), torch.ones(q.shape[:2]), atol=1e-5)


def test_self_penetration_and_strain():
    links = torch.zeros(1, 28, 3)
    links[0, :, 0] = torch.arange(28) * 0.04
    nodes = polyline_from_link_centers(links)
    assert cl.self_penetration(nodes, 0.01, 0.0314).item() == pytest.approx(0.0, abs=1e-6)
    assert cl.strain(nodes, 0.04).item() == pytest.approx(0.0, abs=1e-5)
    # 折り返して 1.5 cm まで近づけるとめり込み 5 mm
    folded = torch.cat([nodes[:, :15], nodes[:, :15].flip(1) + torch.tensor([0.0, 0.015, 0.0])], dim=1)
    assert cl.self_penetration(folded, 0.01, 0.0314).item() == pytest.approx(0.005, abs=1e-4)


# ---------------------------------------------------------------- XPBD (Warp CPU)

wp = pytest.importorskip("warp")


@pytest.fixture(scope="module")
def xr():
    wp.init()
    return _load("xpbd_rod")


def _rod(xr, model="cosserat", n=2, **kw):
    p = xr.XpbdRodParams(model=model, **kw)
    rod = xr.XpbdRod(n, p, device="cpu")
    origins = torch.zeros(n, 3)
    origins[:, 0] = torch.arange(n) * 3.0
    rod.set_state(torch.arange(n), rod.straight_nodes(origins, 0.06))
    return rod, origins


def _move(rod, target, fingers, seconds, dt=1.0 / 240.0):
    steps = int(round(seconds / dt))
    h0, f0 = rod.hand_now.clone(), rod.fing_now.clone()
    for i in range(steps):
        u = (i + 1) / steps
        s = u * u * (3 - 2 * u)
        rod.set_hand(h0 + (target - h0) * s, f0 + (fingers - f0) * s)
        rod.step(dt)


@pytest.mark.parametrize("model", ["cosserat", "mass_spring"])
def test_rest_on_ground_is_still_and_inextensible(xr, model):
    rod, _ = _rod(xr, model)
    hand = torch.zeros(2, 4)
    hand[:, 2] = 0.3
    rod.set_hand(hand, torch.tensor([[0.03, -0.03]]).repeat(2, 1))
    rod.sync_hand()
    for _ in range(120):
        rod.step(1.0 / 240.0)
    x = rod.nodes()
    assert torch.allclose(x[..., 2], torch.full_like(x[..., 2], 0.01), atol=1e-4)
    assert rod.node_vel().norm(dim=-1).max().item() < 1e-3
    assert cl.strain(x, rod.p.segment_length).max().item() < 1e-4


RIGID_FINGER = dict(
    finger_rigid_outside=True, finger_x_faces=True, finger_x_face_margin=0.001, hand_teleport_distance=0.02
)
"""2026-10-02 に足した指の接触 (`XpbdRodParams.finger_rigid_outside` ほか)。"""


@pytest.mark.parametrize("model,finger", [("cosserat", {}), ("mass_spring", {}), ("cosserat", RIGID_FINGER)])
def test_grasp_and_lift(xr, model, finger):
    """中央を掴んで 0.2 m まで持ち上げると、掴んだ点が指と一緒に上がる (摩擦把持)。"""
    rod, origins = _rod(xr, model, **finger)
    for _ in range(60):
        rod.step(1.0 / 240.0)
    x = rod.nodes()
    g = 0.5 * (x[:, 14] + x[:, 15])
    above = torch.stack([g[:, 0], g[:, 1], torch.full_like(g[:, 0], 0.3), torch.zeros_like(g[:, 0])], -1)
    open_ = torch.tensor([[0.03, -0.03]]).repeat(2, 1)
    rod.set_hand(above, open_)
    rod.sync_hand()
    grasp = above.clone()
    grasp[:, 2] = 0.077
    _move(rod, grasp, open_, 0.3)
    _move(rod, grasp, torch.zeros(2, 2), 0.3)
    _move(rod, grasp, torch.zeros(2, 2), 0.2)
    lift = grasp.clone()
    lift[:, 2] = 0.2
    _move(rod, lift, torch.zeros(2, 2), 0.3)
    x = rod.nodes()
    gz = 0.5 * (x[:, 14, 2] + x[:, 15, 2])
    assert torch.all(gz > 0.1), gz
    assert torch.isfinite(x).all()
    assert cl.strain(x, rod.p.segment_length).max().item() < 0.02


def test_self_contact_prevents_pass_through(xr):
    """2 本の区間を上下に重ねて落としても、すり抜けずに直径ぶん離れて止まる。"""
    rod, _ = _rod(xr, "cosserat", n=1)
    # U 字に折って、後半を前半の上 3 cm に置く
    x = rod.nodes()[0].clone()
    half = rod.P // 2
    bent = x.clone()
    bent[half:, 0] = x[half - 1, 0] - (x[half:, 0] - x[half - 1, 0])
    bent[half:, 1] = 0.0
    bent[:half, 2] = 0.01
    bent[half:, 2] = 0.04
    bent[half:, 0] += 0.01  # 折り返しの位置を少しずらす
    rod.set_state(torch.tensor([0]), bent.unsqueeze(0))
    hand = torch.zeros(1, 4)
    hand[:, 2] = 1.0
    rod.set_hand(hand, torch.zeros(1, 2))
    rod.sync_hand()
    for _ in range(240):
        rod.step(1.0 / 240.0)
    x = rod.nodes()
    assert cl.self_penetration(x, 0.01, 0.0314).item() < 0.003
    assert torch.isfinite(x).all()


def test_open_finger_does_not_carry_rope(xr):
    """開いた指を床の紐の上に降ろし、横へ動かしてから上げても、紐は指に付いてこない (剛の外側の面)。

    台本の遅い尺で XPBD の紐だけ固結びが 2% だった原因 (`rope_study/Report_rope_models.md` §7.1.2)。
    旧来の compliance だけの指では、指の中に入った紐が押し込みの力と μ = 4 の摩擦で捕まって持ち上がった。
    """
    rod, _ = _rod(xr, "cosserat", n=1, **RIGID_FINGER)
    for _ in range(60):
        rod.step(1.0 / 240.0)
    # 右の指 (開いた中心 = 手の y - 0.036) を紐の真上に置く
    hand = torch.tensor([[0.0, 0.036, 0.3, 0.0]])
    open_ = torch.tensor([[0.03, -0.03]])
    rod.set_hand(hand, open_)
    rod.sync_hand()
    low = hand.clone()
    low[:, 2] = 0.077
    _move(rod, low, open_, 0.3)
    side = low.clone()
    side[:, 0] += 0.08
    _move(rod, side, open_, 0.4)
    up = side.clone()
    up[:, 2] = 0.3
    _move(rod, up, open_, 0.5)
    for _ in range(60):
        rod.step(1.0 / 240.0)
    x = rod.nodes()
    assert x[..., 2].max().item() < 0.03, x[..., 2].max()
    assert torch.isfinite(x).all()


def test_teleported_finger_does_not_fling_rope(xr):
    """1 物理ステップで瞬間移動した指が紐に深く重なっても、紐を弾き飛ばさない (押し戻しの速さの上限)。"""
    rod, _ = _rod(xr, "cosserat", n=1, **RIGID_FINGER)
    for _ in range(60):
        rod.step(1.0 / 240.0)
    open_ = torch.tensor([[0.03, -0.03]])
    rod.set_hand(torch.tensor([[0.0, 0.0331, 0.3, 0.0]]), open_)
    rod.sync_hand()
    rod.step(1.0 / 240.0)
    # 右の指の中心が紐の線から 2.9 mm (内側) の位置に、指先を床すれすれまで瞬間移動
    rod.set_hand(torch.tensor([[0.0, 0.0331, 0.077, 0.0]]), open_)
    vmax = 0.0
    for _ in range(24):
        rod.step(1.0 / 240.0)
        vmax = max(vmax, rod.node_vel().norm(dim=-1).max().item())
    assert vmax < 3.0, vmax
    assert torch.isfinite(rod.nodes()).all()


def _slide(rod_nodes, rod_vel, step, seconds=1.0, v0=0.26):
    """床の上の紐に横 (+y) の初速 v0 を与えて `seconds` 回し、重心の移動と最後の速さを返す。"""
    rod_vel()[..., 1] = v0
    c0 = rod_nodes().mean(1).clone()
    for _ in range(int(seconds * 240)):
        step()
    return (rod_nodes().mean(1) - c0)[:, :2].norm(dim=-1), rod_vel().mean(1)[:, :2].norm(dim=-1)


def test_ground_friction_stops_sliding_rope(xr):
    """低摩擦 (μ 0.5) の床で横へ滑らせた紐は、理論の停止距離 v^2 / (2 μ g) = 7 mm 程度で止まる。"""
    rod, _ = _rod(xr, "cosserat", friction_ground=0.5, friction_self=0.5, friction_finger=1.0, **RIGID_FINGER)
    hand = torch.zeros(2, 4)
    hand[:, 2] = 5.0
    rod.set_hand(hand, torch.zeros(2, 2))
    rod.sync_hand()
    for _ in range(60):
        rod.step(1.0 / 240.0)
    disp, v = _slide(rod.nodes, rod.node_vel, lambda: rod.step(1.0 / 240.0))
    assert disp.max().item() < 0.02, disp
    assert v.max().item() < 1e-3, v


def test_xpbd_adapter_solves_in_env_local_coordinates(xr):
    """環境の原点が遠くても (4096 環境の端は ±94 m)、アダプタは局所座標で解くので床の摩擦が効く。

    world 座標のまま解くと、float32 の刻みが小ステップ 1 回の摩擦の補正より大きくなり、紐が 0.26 m/s で
    滑り続けた (`rope_study/Report_rope_models.md` §7.1.6)。
    """
    parent = types.ModuleType("rs_common")
    parent.__path__ = [str(_COMMON)]
    sub = types.ModuleType("rs_common.rope_backends")
    sub.__path__ = [str(_COMMON / "rope_backends")]
    sys.modules.setdefault("rs_common", parent)
    sys.modules.setdefault("rs_common.rope_backends", sub)
    sys.modules.setdefault("rs_common.rope_backends.xpbd_rod", xr)  # Warp のカーネルを作り直さない
    spec = importlib.util.spec_from_file_location(
        "rs_common.rope_backends.adapters", _COMMON / "rope_backends" / "adapters.py"
    )
    ad = importlib.util.module_from_spec(spec)
    sys.modules["rs_common.rope_backends.adapters"] = ad
    spec.loader.exec_module(ad)

    origins = torch.tensor([[0.0, 0.0, 0.0], [90.0, -90.0, 0.0]])
    params = xr.XpbdRodParams(friction_ground=0.5, friction_self=0.5, friction_finger=1.0, **RIGID_FINGER)
    rope = ad.XpbdRope(2, "cpu", params, 28, 0.0101, origins)
    root = torch.zeros(2, 7)
    root[:, :3] = origins + torch.tensor([0.0, 0.0, 5.0])
    root[:, 3] = 1.0
    gripper = types.SimpleNamespace(
        data=types.SimpleNamespace(root_pose_w=root, joint_pos=torch.zeros(2, 2)),
        find_joints=lambda names, preserve_order=True: ([0, 1], names),
    )
    rope.bind(types.SimpleNamespace(scene={"gripper": gripper}, sim=types.SimpleNamespace(has_gui=lambda: False)))
    for _ in range(60):
        rope.update(1.0 / 240.0)
    disp, v = _slide(rope.rod.nodes, rope.rod.node_vel, lambda: rope.update(1.0 / 240.0))
    assert disp.max().item() < 0.02, disp
    assert v.max().item() < 1e-3, v
    # 読み出しは world (原点を足した値)、仮想リンクの中心も world
    assert torch.allclose(rope.nodes_w().mean(1)[:, :2], origins[:, :2], atol=0.03)
    assert torch.allclose(rope.data.body_pos_w.mean(1)[:, :2], origins[:, :2], atol=0.03)


def test_heavy_elastica_large_deflection_is_monotonic():
    """heavy elastica の先端の下がりは EI に対して単調で、大たわみ (w L^3 / EI > 100) でも真下を越えない。

    以前は射撃法が θ'(1) の符号だけで判定していて、大たわみで偽の根に落ち、`effective_ei` が異なる下がりに
    同じ値を返した (2026-10-02)。
    """
    spec = importlib.util.spec_from_file_location("rope_study_elastica", _COMMON.parent / "rope_study" / "elastica.py")
    el = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(el)
    ei, w, length = 7.854e-4, 0.2513 * 9.81, 0.16
    drops = [el.heavy_cantilever(length, ei * r, w)["tip_drop"] for r in (1.0, 0.5, 0.2, 0.1, 0.05, 0.01)]
    assert all(b > a for a, b in zip(drops, drops[1:])), drops
    assert drops[-1] < length
    # 小たわみの極限は w L^4 / (8 EI) に近い
    small = el.heavy_cantilever(0.02, ei, w)
    assert abs(small["tip_drop"] / small["small_deflection_drop"] - 1.0) < 0.01
    # 逆問題が往復で戻る
    d = el.heavy_cantilever(length, ei * 0.3, w)["tip_drop"]
    assert abs(el.effective_ei(length, w, d, ei) / (ei * 0.3) - 1.0) < 0.01
