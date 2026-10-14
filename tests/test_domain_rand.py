"""ロープの物理の DR の値の引き方と正規化 (`common/domain_rand.py`、v14) を Isaac Sim なしで検査する。

書き込み (`mdp/rope_dr.py`) は Isaac Sim が要るので、スモークテストで読み戻して確かめる。
"""

from __future__ import annotations

import math

import domain_rand as dr
import pytest
import torch

RANGES = {name: (0.5, 2.0) for name in dr.AXES}


def test_axes_order_is_fixed():
    # critic の特権観測の列の順。変えると学習済みの critic と食い違う。
    assert dr.AXES == ("bend_stiffness", "twist_stiffness", "damping_ratio", "mass", "rope_friction", "finger_friction")


def test_sample_within_range_and_log_uniform():
    g = torch.Generator().manual_seed(0)
    s = dr.sample_scales(200_000, RANGES, generator=g)
    assert s.shape == (200_000, len(dr.AXES))
    assert float(s.min()) >= 0.5 and float(s.max()) <= 2.0
    # 対数一様: log2 が [−1, 1] に一様なので、半分が 1 より下、平均 0、分散 1/3。
    lg = torch.log2(s)
    assert abs(float((s < 1.0).float().mean()) - 0.5) < 0.01
    assert abs(float(lg.mean())) < 0.01
    assert abs(float(lg.var()) - 1.0 / 3.0) < 0.01


def test_point_range_is_constant():
    ranges = dict(RANGES, mass=(1.0, 1.0), rope_friction=(0.7, 0.7))
    s = dr.sample_scales(1000, ranges, generator=torch.Generator().manual_seed(1))
    i_mass, i_mu = dr.AXES.index("mass"), dr.AXES.index("rope_friction")
    assert torch.all(s[:, i_mass] == 1.0)
    assert torch.allclose(s[:, i_mu], torch.full((1000,), 0.7))


def test_bad_range_raises():
    with pytest.raises(ValueError):
        dr.sample_scales(1, dict(RANGES, mass=(2.0, 0.5)))
    with pytest.raises(ValueError):
        dr.sample_scales(1, dict(RANGES, mass=(0.0, 1.0)))


def test_normalize_maps_ends_and_nominal():
    s = torch.tensor([[0.5] * 6, [1.0] * 6, [2.0] * 6, [math.sqrt(2.0)] * 6])
    n = dr.normalize(s, RANGES)
    assert torch.allclose(n[0], torch.full((6,), -1.0))
    assert torch.allclose(n[1], torch.zeros(6), atol=1e-6)
    assert torch.allclose(n[2], torch.ones(6))
    assert torch.allclose(n[3], torch.full((6,), 0.5), atol=1e-6)


def test_normalize_point_range_is_zero_and_stays_in_bounds():
    ranges = dict(RANGES, mass=(1.0, 1.0))
    s = dr.sample_scales(5000, ranges, generator=torch.Generator().manual_seed(2))
    n = dr.normalize(s, ranges)
    assert torch.all(n[:, dr.AXES.index("mass")] == 0.0)
    assert float(n.abs().max()) <= 1.0
    # 公称 (倍率 1) は、範囲が 1 点でも 0。
    assert torch.all(dr.normalize(torch.ones(1, 6), dr.NOMINAL_RANGES) == 0.0)


def test_damping_scale_keeps_damping_ratio():
    # 臨界減衰 c = 2 ζ sqrt(k I)、I ∝ m。剛性と質量を振っても ζ の倍率だけが残る。
    k, m, z = torch.tensor([2.0, 0.5, 1.0]), torch.tensor([0.5, 2.0, 1.0]), torch.tensor([1.0, 1.0, 1.5])
    c = dr.damping_scale(k, m, z)
    zeta = c / torch.sqrt(k * m)
    assert torch.allclose(zeta, z)
    assert torch.allclose(dr.damping_scale(torch.ones(1), torch.ones(1), torch.ones(1)), torch.ones(1))
