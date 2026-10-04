"""R2 (ライデマイスター II) 除去 `knot_detect.reduce_r2` の単体テスト。v2-R2 で追加。

Isaac Sim を起動せずに走る (`topology` パッケージは isaaclab を import しない)::

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/test_r2_reduce.py

## 検証の骨子

固結びの手順 R1 -> cross -> cross を 2D の経路で作り、各交差で「先端側が上か
下か」(O / U) を全通り振る。上下が交互 (OU / UO、OUO / UOU) のものだけが本物の
中間状態・固結びで、それ以外は R2 で消せる。v2 の判定 (部分列) はこのうち
「先端で輪を上から跨いだだけ」(OOO など) を class 2 と誤判定していた。
"""

from __future__ import annotations

import itertools
import math

import pytest
import topology as tp
import torch
from test_knot_detect import _random_ropes, _trefoil, _trefoil_with_tail_loop
from topology import knot_detect as kd

# ---------------------------------------------------------------- 手順どおりのロープ


def _circle(cx: float, cy: float, r: float, a0: float, a1: float, n: int = 40) -> list[tuple[float, float]]:
    return [
        (
            cx + r * math.cos(math.radians(a0 + (a1 - a0) * k / n)),
            cy + r * math.sin(math.radians(a0 + (a1 - a0) * k / n)),
        )
        for k in range(n + 1)
    ]


# 始点 -> 右へ -> 反時計回りの輪 -> 真下へ降りて始点側の直線を x=0 で横切る (交差 A = R1)
_LOOP = [(-3.0, 0.0), (1.0, 0.0)] + _circle(1.0, 1.0, 1.0, -90, 180)[1:] + [(0.0, -1.0)]
_STAGES = {
    # C1: 輪だけ
    1: _LOOP + [(0.0, -1.5)],
    # C2: 先端を下から輪の中へ (x=0.5 で輪の紐を横切る -> 交差 B)
    2: _LOOP + [(0.0, -1.5), (0.5, -1.5), (0.5, 0.6)],
    # C3: 輪の右側から外へ抜く (-> 交差 C)
    3: _LOOP + [(0.0, -1.5), (0.5, -1.5), (0.5, 0.6), (1.5, 0.6), (2.6, 0.6)],
}


def _dense(wps: list[tuple[float, float]], n: int = 400) -> torch.Tensor:
    w = torch.tensor(wps, dtype=torch.float64)
    seg = (w[1:] - w[:-1]).norm(dim=-1)
    cum = torch.cat([torch.zeros(1, dtype=torch.float64), seg.cumsum(0)])
    s = torch.linspace(0, float(cum[-1]), n, dtype=torch.float64)
    idx = torch.searchsorted(cum, s, right=True).clamp(1, len(w) - 1) - 1
    t = ((s - cum[idx]) / seg[idx]).unsqueeze(-1)
    return w[idx] * (1 - t) + w[idx + 1] * t


def _staged_rope(stage: int, tip_over: tuple[bool, ...]) -> torch.Tensor:
    """`stage` 段目のロープ。`tip_over[k]` は k 番目の交差 (A, B, C) で先端側が上か。"""
    xy = _dense(_STAGES[stage])
    pts = torch.cat([xy, torch.zeros(len(xy), 1, dtype=torch.float64)], dim=-1)
    flat = tp.segment_intersections(pts.unsqueeze(0))
    items = sorted(flat.to_list(0), key=lambda it: it[1])  # 先端側セグメント順 = A, B, C
    assert len(items) == stage
    s = torch.arange(len(xy), dtype=torch.float64)
    z = torch.zeros(len(xy), dtype=torch.float64)
    for (i, j, _, _), over in zip(items, tip_over):
        z += 0.05 * torch.exp(-(((s - (j if over else i)) / 4.0) ** 2))
    pts[:, 2] = z
    return pts


def _classes(points: torch.Tensor) -> tuple[int, int]:
    batch = tp.segment_intersections(points.unsqueeze(0))
    return int(kd.topology_class(batch)[0]), int(kd.topology_class(kd.reduce_r2(batch))[0])


def _is_alternating(tip_over: tuple[bool, ...]) -> bool:
    return all(a != b for a, b in zip(tip_over, tip_over[1:]))


# ---------------------------------------------------------------- 手順 R1 -> cross -> cross


@pytest.mark.parametrize("tip_over", [(True,), (False,)])
def test_r1_loop_is_class_one(tip_over):
    assert _classes(_staged_rope(1, tip_over)) == (kd.CLASS_SINGLE, kd.CLASS_SINGLE)


@pytest.mark.parametrize("tip_over", list(itertools.product([True, False], repeat=2)))
def test_tip_into_loop(tip_over):
    """先端を輪の中へ。くぐらせた (OU / UO) ときだけ class 2。上から置いた (OO) は交差 0 に戻る。"""
    raw, reduced = _classes(_staged_rope(2, tip_over))
    if _is_alternating(tip_over):
        assert (raw, reduced) == (kd.CLASS_INTERLEAVED2, kd.CLASS_INTERLEAVED2)
    else:
        assert (raw, reduced) == (kd.CLASS_TRIVIAL, kd.CLASS_TRIVIAL)
        batch = tp.segment_intersections(_staged_rope(2, tip_over).unsqueeze(0))
        assert int(kd.reduce_r2(batch).count[0]) == 0


@pytest.mark.parametrize("tip_over", list(itertools.product([True, False], repeat=3)))
def test_tip_through_loop(tip_over):
    """輪を通して抜く。交互 (OUO / UOU) だけが固結び。それ以外は R2 後に輪 1 つ (class 1)。

    v2 の判定では非交互の 6 通りが全て class 2 (偽の C2) になっていた。
    """
    raw, reduced = _classes(_staged_rope(3, tip_over))
    if _is_alternating(tip_over):
        assert (raw, reduced) == (kd.CLASS_OVERHAND, kd.CLASS_OVERHAND)
    else:
        assert raw == kd.CLASS_INTERLEAVED2, "前提: v2 の判定はこれを class 2 と誤る"
        assert reduced == kd.CLASS_SINGLE


def test_drape_over_loop_p_data():
    """先端で輪を上から跨いだだけの p-data が、R2 後に輪 1 つの p-data になる。"""
    batch = tp.segment_intersections(_staged_rope(3, (True, True, True)).unsqueeze(0))
    assert tp.p_data_from_batch(batch, short=True)[0] == "1U4+_2U5-_3U6+_4O1+_5O2-_6O3+"
    assert tp.p_data_from_batch(kd.reduce_r2(batch), short=True)[0] == "1U2+_2O1+"


# ---------------------------------------------------------------- 性質


def test_trefoil_untouched():
    """交代結び目 (固結び) の交差は 1 つも消えない。"""
    batch = tp.segment_intersections(_trefoil().unsqueeze(0))
    reduced = kd.reduce_r2(batch)
    assert torch.equal(reduced.count, batch.count)
    assert tp.p_data_from_batch(reduced) == tp.p_data_from_batch(batch)


def test_trefoil_with_tail_loop_still_overhand():
    batch = tp.segment_intersections(_trefoil_with_tail_loop().unsqueeze(0))
    assert bool(kd.is_overhand(kd.reduce_r2(batch))[0])


@pytest.mark.parametrize("frac", [0.60, 0.70, 0.78, 1.0])
def test_partial_trefoil_ladder_unchanged(frac: float):
    """結びかけ (交互) の各段は R2 除去で変わらない。"""
    batch = tp.segment_intersections(_trefoil(frac=frac).unsqueeze(0))
    assert torch.equal(kd.topology_class(kd.reduce_r2(batch)), kd.topology_class(batch))


def test_random_ropes_properties():
    """交差数は増えない・べき等・番号は 1..2n の置換・固結びの誤検出は増えない。"""
    batch = tp.segment_intersections(_random_ropes(600, scale=0.6, seed=1))
    assert int((batch.count >= 2).sum()) > 50, "前提: 交差 2 つ以上のロープが十分ある"
    reduced = kd.reduce_r2(batch)
    assert bool((reduced.count <= batch.count).all())
    assert bool(((batch.count - reduced.count) % 2 == 0).all()), "R2 は 2 つずつ消す"
    again = kd.reduce_r2(reduced)
    assert torch.equal(again.count, reduced.count)
    for env in range(reduced.num_envs):
        n = int(reduced.count[env])
        pts = sorted(reduced.order[env, :n].flatten().tolist())
        assert pts == list(range(1, 2 * n + 1))
    assert not bool((kd.is_overhand(reduced) & ~kd.is_overhand(batch)).any())


def test_batched_matches_single():
    """バッチで回しても env ごとに回しても同じ結果 (env 間で干渉しない)。"""
    shapes = [
        _staged_rope(3, (True, True, True)),
        _staged_rope(3, (True, False, True)),
        _staged_rope(2, (True, True)),
        _staged_rope(1, (True,)),
    ]
    batch = tp.segment_intersections(torch.stack(shapes))
    reduced = kd.reduce_r2(batch)
    for env, pts in enumerate(shapes):
        single = kd.reduce_r2(tp.segment_intersections(pts.unsqueeze(0)))
        assert tp.p_data_from_batch(single)[0] == tp.p_data_from_batch(reduced)[env]


def test_empty_batch():
    batch = tp.segment_intersections(torch.zeros(3, 5, 3, dtype=torch.float64))
    reduced = kd.reduce_r2(batch)
    assert reduced.count.tolist() == [0, 0, 0]
    assert kd.topology_class(reduced).tolist() == [0, 0, 0]
