"""`common/mdp/finger_geometry.py` (指の真下の紐を探す幾何) のテスト。Isaac Sim 無しで動く。"""

from __future__ import annotations

import math

import finger_geometry as fg
import phases
import torch

R = 0.01
"""`knot` のロープ半径 [m]。"""


def _straight(num_links: int, y: float = 0.0, z: float = R, spacing: float = 0.04) -> torch.Tensor:
    """x 軸に沿ったまっすぐなロープのリンク中心 `(num_links, 3)`。中央のリンクが x = 0。"""
    x = (torch.arange(num_links, dtype=torch.float32) - num_links // 2) * spacing
    return torch.stack([x, torch.full_like(x, y), torch.full_like(x, z)], dim=-1)


def test_sample_points_interleave_centers_and_midpoints():
    links = _straight(5).unsqueeze(0)
    pts, coord = fg.rope_sample_points(links)
    assert pts.shape == (1, 9, 3)
    assert torch.allclose(pts[0, 0::2], links[0])
    assert torch.allclose(pts[0, 1], 0.5 * (links[0, 0] + links[0, 1]))
    assert coord.tolist() == [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]


def test_no_rope_under_fingers_on_a_single_straight_rope():
    """掴むリンク自身とその前後 (指の間を通る) は、指の下に入らない。"""
    pts, _ = fg.rope_sample_points(_straight(9).unsqueeze(0))
    top = fg.finger_footprint_top(pts, torch.zeros(1, 2), torch.zeros(1), R)
    assert float(top) == 0.0


def test_crossing_rope_under_the_fingers_is_found():
    """掴むリンクの下を直交して通る紐 (交差の上側を掴む手) は、指の下に入る。"""
    upper = _straight(9, z=3 * R)  # 上側 (掴む側)。x 軸沿い
    lower = _straight(9)[:, [1, 0, 2]]  # 下側。y 軸沿いに原点を通る
    pts_u, _ = fg.rope_sample_points(upper.unsqueeze(0))
    pts_l, _ = fg.rope_sample_points(lower.unsqueeze(0))
    pts = torch.cat([pts_u, pts_l], dim=1)
    top = fg.finger_footprint_top(pts, torch.zeros(1, 2), torch.zeros(1), R)
    assert math.isclose(float(top), 2 * R, abs_tol=1e-6)  # 下の紐の上端 (床の上なので 2r)


def test_parallel_rope_next_to_the_fingers_is_found_only_within_reach():
    """横に並んだ紐は、指の外側 + 半径より近ければ入り、遠ければ入らない。"""
    y_hi = phases.FINGER_OPEN_Y[1]
    own, _ = fg.rope_sample_points(_straight(9).unsqueeze(0))
    for dy, expect in ((0.5 * R, 2 * R), (1.5 * R, 0.0)):
        # 2 本は別々に点列にする (1 本の折れ線につなぐと、端どうしの中点が間に入る)
        other, _ = fg.rope_sample_points(_straight(9, y=y_hi + dy).unsqueeze(0))
        pts = torch.cat([own, other], dim=1)
        top = fg.finger_footprint_top(pts, torch.zeros(1, 2), torch.zeros(1), R)
        assert math.isclose(float(top), expect, abs_tol=1e-6)


def test_footprint_follows_the_hand_yaw():
    """手を 90° 回すと、指の占める場所も回る。"""
    y_mid = sum(phases.FINGER_OPEN_Y) / 2
    blob = torch.tensor([[[0.0, y_mid, R]]])  # 手のローカル y 側に 1 点
    yaw0 = fg.finger_footprint_top(blob, torch.zeros(1, 2), torch.zeros(1), R)
    yaw90 = fg.finger_footprint_top(blob, torch.zeros(1, 2), torch.full((1,), math.pi / 2), R)
    assert float(yaw0) > 0.0
    assert float(yaw90) == 0.0


def test_nan_rope_does_not_poison_the_result():
    pts = torch.full((1, 3, 3), float("nan"))
    top = fg.finger_footprint_top(pts, torch.zeros(1, 2), torch.zeros(1), R)
    assert float(top) == 0.0


def test_min_z_ignores_rope_lying_on_the_floor():
    """`min_z` より低い (床の上の) 紐は数えず、別の紐に乗った紐だけを数える。"""
    own, _ = fg.rope_sample_points(_straight(9).unsqueeze(0))
    floor = fg.rope_sample_points(_straight(9)[:, [1, 0, 2]].unsqueeze(0))[0]  # 床の上を y 軸沿いに
    stacked = fg.rope_sample_points(_straight(9, z=3 * R)[:, [1, 0, 2]].unsqueeze(0))[0]  # 紐 1 本ぶん浮いている
    args = (torch.zeros(1, 2), torch.zeros(1), R)
    assert float(fg.finger_footprint_top(torch.cat([own, floor], 1), *args, min_z=1.5 * R)) == 0.0
    top = fg.finger_footprint_top(torch.cat([own, stacked], 1), *args, min_z=1.5 * R)
    assert math.isclose(float(top), 4 * R, abs_tol=1e-6)
