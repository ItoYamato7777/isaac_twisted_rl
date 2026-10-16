"""`common/curved_shapes.py` (評価用の曲がった初期形状の生成) の単体テスト。Isaac Sim は不要。"""

import math

import curved_shapes as cs
import pytest
import torch


def test_zero_turns_is_straight():
    xy = cs.chain_nodes(torch.zeros(1, 27), spacing=0.02)
    assert torch.allclose(xy[0, :, 1], torch.zeros(28), atol=1e-7)
    assert torch.allclose(xy[0, :, 0], torch.arange(28) * 0.02, atol=1e-6)


def test_link_spacing_is_kept_when_bent():
    """曲がっても、隣のリンク中心の間隔は直線のときの 1 区間に近い (関節はリンク端にある)。"""
    gen = torch.Generator().manual_seed(0)
    xy = cs.chain_nodes(cs.sample_turning_angles(64, 27, gen), spacing=0.02)
    gaps = (xy[:, 1:] - xy[:, :-1]).norm(dim=-1)
    assert float(gaps.max()) <= 0.02 + 1e-7
    assert float(gaps.min()) >= 0.02 * math.cos(cs.MAX_TURN / 2) - 1e-7


def test_constant_turn_makes_circular_arc():
    xy = cs.chain_nodes(torch.full((1, 27), 0.1), spacing=0.02)
    assert float(cs.total_turning(xy)[0]) == pytest.approx(0.1 * 26, rel=1e-4)


def test_turns_are_clamped_and_reproducible():
    a = cs.sample_turning_angles(32, 27, torch.Generator().manual_seed(3))
    b = cs.sample_turning_angles(32, 27, torch.Generator().manual_seed(3))
    assert torch.equal(a, b)
    assert float(a.abs().max()) <= cs.MAX_TURN + 1e-7


def test_sign_of_turn_is_counter_clockwise():
    """正の曲がりは反時計回り (x 軸から y 軸の向き)。関節角の符号の規約の確認が実測で要る前提。"""
    xy = cs.chain_nodes(torch.full((1, 5), 0.3), spacing=0.02)
    assert float(xy[0, -1, 1]) > 0.0


def test_min_clearance_detects_touching_far_links():
    xy = torch.zeros(1, 6, 2)
    xy[0, :, 0] = torch.arange(6) * 1.0
    xy[0, 5] = torch.tensor([0.0, 0.05])  # リンク 5 がリンク 0 のすぐ隣に来る
    assert float(cs.min_clearance(xy)[0]) == pytest.approx(0.05, rel=1e-4)
    assert float(cs.min_clearance(xy, skip=6)[0]) == float("inf")
