"""位相グラフと BFS 距離 `topology/knot_graph.py` の単体テスト。v3 で追加。

Isaac Sim を起動せずに走る (グラフの列挙に約 20 s かかる。プロセス内で 1 回だけ)::

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/test_knot_graph.py

## 検証の骨子

* 移植した R1 / R2 / cross で自明な状態から BFS し、交差 3 個の状態が 528 個
  (移植元で数えた値。`knot_detect.py` の docstring) になること。交差 4 個は 7968 個。
* 距離の表 (`knot_graph.py` の docstring の表)。とくにゲート 1 の cover 後の状態が
  d = 1 (R2 除去後のラダーでは C1 と同じ) になること。
* テンソル層のキー `batch_keys` が、記号層の p-data から作ったキーと一致すること。
* `BfsPotential` がラダーより下がらないこと。
"""

from __future__ import annotations

import collections

import pytest
import topology as tp
import torch
from test_knot_detect import (
    _batch,
    _independent_twists,
    _random_ropes,
    _straight,
    _trefoil,
    _trefoil_with_tail_loop,
)
from topology import knot_graph as kg

_R1 = ("1O2+_2U1+", "1U2+_2O1+", "1O2-_2U1-", "1U2-_2O1-")
_C2 = ("1O3+_2U4+_3U1+_4O2+", "1U3+_2O4+_3O1+_4U2+", "1O3-_2U4-_3U1-_4O2-", "1U3-_2O4-_3O1-_4U2-")
_COVER = "1U4+_2O5+_3O6-_4O1+_5U2+_6U3-"
"""ゲート 1 の cover の直後に最も多かった状態 (`docs/experiments/2026-09-27_gate1.md` の「結論」の 5)。R2 除去後は C1。"""
_FAKE_C2 = "1U4+_2U5-_3U6+_4O1+_5O2-_6O3+"
"""輪を上から跨いだだけの偽の C2 (v2-R2 README)。"""


def _crossings(p_data: str) -> int:
    return len(p_data.split("_")) // 2 if p_data else 0


def _from_p_data(p_data: list[str] | tuple[str, ...]) -> tp.IntersectionBatch:
    return tp.IntersectionBatch.from_lists([kg.p_data_to_lists(pd) for pd in p_data])


# ---------------------------------------------------------------- グラフの列挙


def test_state_counts():
    p_data, _ = kg.enumerate_graph()
    counts = collections.Counter(_crossings(pd) for pd in p_data)
    assert dict(counts) == {0: 1, 1: 4, 2: 40, 3: 528, 4: 7968}
    # 状態と p-data は 1 対 1 (面の構造だけ違う状態は無い)。
    assert len(set(p_data)) == len(p_data)


def test_goals_are_r2_reduced_overhands():
    p_data, _ = kg.enumerate_graph()
    goals = [pd for pd, g in zip(p_data, kg.is_goal(p_data)) if g]
    assert len(goals) == 172
    assert set(kg.OVERHAND_P_DATA) <= set(goals)


def test_distance_table():
    dist = kg.distance_to_overhand()
    assert dist[""] == 3
    assert all(dist[pd] == 2 for pd in _R1)
    assert all(dist[pd] == 1 for pd in _C2)
    assert all(dist[pd] == 0 for pd in kg.OVERHAND_P_DATA)
    assert dist[_COVER] == 1
    assert _FAKE_C2 not in dist


def test_potential_table_values():
    table = kg.potential_table()
    assert table[""] == 0.0
    assert all(table[pd] == pytest.approx(1 / 3) for pd in _R1)
    assert all(table[pd] == pytest.approx(2 / 3) for pd in _C2)
    assert table[_COVER] == pytest.approx(2 / 3)
    assert all(table[pd] == 1.0 for pd in kg.OVERHAND_P_DATA)


def test_generate_next_does_not_mutate():
    start = tp.AbstractState()
    list(kg.generate_next(start))
    assert start.pts == 0 and start.short_p_data == ""


# ---------------------------------------------------------------- p-data <-> 交差リスト / キー


def test_p_data_to_lists_roundtrip():
    """全状態について、p-data -> 交差リスト -> 記号層の p-data で元に戻る。"""
    p_data, _ = kg.enumerate_graph()
    assert tp.p_data_from_batch(_from_p_data(p_data), short=True) == list(p_data)


def test_p_data_key_is_unique_over_graph():
    p_data, _ = kg.enumerate_graph()
    keys = [kg.p_data_key(pd) for pd in p_data]
    assert len(set(keys)) == len(keys)


def test_batch_keys_match_p_data_key_over_graph():
    p_data, _ = kg.enumerate_graph()
    assert kg.batch_keys(_from_p_data(p_data)).tolist() == [kg.p_data_key(pd) for pd in p_data]


def test_p_data_key_rejects_large_or_malformed():
    with pytest.raises(ValueError):
        kg.p_data_key("_".join(f"{2 * k + 1}O{2 * k + 2}+_{2 * k + 2}U{2 * k + 1}+" for k in range(5)))
    with pytest.raises(ValueError):
        kg.p_data_key("1X2+_2U1+")


@pytest.mark.parametrize("seed", range(5))
def test_batch_keys_match_p_data_on_random_ropes(seed: int):
    """実際の交差検出の出力でも、テンソル層のキーと記号層の p-data のキーが一致する。"""
    batch = tp.segment_intersections(_random_ropes(400, seed=seed, scale=0.35))
    keys = kg.batch_keys(batch).tolist()
    p_data = tp.p_data_from_batch(batch, short=True)
    checked = 0
    for key, pd, c in zip(keys, p_data, batch.count.tolist()):
        if c <= kg.MAX_GRAPH_CROSSINGS:
            assert key == kg.p_data_key(pd), pd
            checked += 1
        else:
            assert key == -1
    assert checked > 100


def test_batch_keys_straight_rope():
    assert kg.batch_keys(_batch(_straight())).tolist() == [0]


# ---------------------------------------------------------------- ポテンシャル


def test_bfs_potential_raises_cover_state():
    """cover 後の状態は、ラダーでは C1 (1/3) だが BFS では 2/3。"""
    pot = kg.BfsPotential()
    batch = _from_p_data([_COVER])
    assert int(tp.topology_class(tp.reduce_r2(batch)).item()) == tp.CLASS_SINGLE
    assert pot(batch).item() == pytest.approx(2 / 3)


def test_bfs_potential_keeps_ladder_for_unreachable():
    """偽の C2 は BFS では届かないが、ラダーの値 (R2 除去後 C1 = 1/3) より下げない。"""
    pot = kg.BfsPotential()
    batch = _from_p_data([_FAKE_C2])
    assert pot.bfs_only(batch).item() == 0.0
    assert pot(batch).item() == pytest.approx(1 / 3)


def test_bfs_potential_on_shapes():
    pot = kg.BfsPotential()
    assert pot(_batch(_straight())).item() == 0.0
    assert pot(_batch(_trefoil())).item() == 1.0
    # 交差 7 (表の範囲外) でも固結びを含めばラダーの 1。
    tail = _batch(_trefoil_with_tail_loop())
    assert int(tail.count.item()) > kg.MAX_GRAPH_CROSSINGS
    assert pot(tail).item() == 1.0


def test_independent_twists_are_not_dead_ends():
    """独立したねじれ 2 つは、ラダーでは class 0 (`offladder` の「行き止まり」) だが BFS では d = 2。

    片方のねじれを残したまま、もう片方から R1 -> cross -> cross ではなく cross 2 回で
    固結びを作れば、交差 4 個 (固結び + 余分なねじれ) のゴールに届く。交差 3 個までの
    グラフでは見えなかった。v3 が `offladder` を外した根拠。
    """
    batch = _batch(_independent_twists())
    assert int(tp.topology_class(tp.reduce_r2(batch)).item()) == tp.CLASS_TRIVIAL
    assert kg.BfsPotential()(batch).item() == pytest.approx(1 / 3)


@pytest.mark.parametrize("seed", range(5))
def test_bfs_potential_never_below_ladder(seed: int):
    batch = tp.segment_intersections(_random_ropes(400, seed=seed, scale=0.35))
    reduced = tp.reduce_r2(batch)
    pot = kg.BfsPotential()(batch, reduced)
    ladder = tp.topology_class(reduced).float() / 3.0
    assert (pot >= ladder).all()
    assert torch.equal(pot, kg.BfsPotential()(batch))  # reduced を省略しても同じ


# ---------------------------------------------------------------- 段 (v4)


def test_stage_values():
    """v4 のエージェントを分ける段。固結びへの経路上の交差数 (0 -> 1 -> 2 -> 3) に当たる。"""
    pot = kg.BfsPotential()
    cases = {"": 0, **{pd: 1 for pd in _R1}, **{pd: 2 for pd in _C2}, _COVER: 2, _FAKE_C2: 1}
    cases.update({pd: 3 for pd in kg.OVERHAND_P_DATA})
    stage = pot.stage(_from_p_data(list(cases)))
    assert stage.dtype == torch.int64
    assert stage.tolist() == list(cases.values())


def test_stage_on_shapes():
    pot = kg.BfsPotential()
    assert pot.stage(_batch(_straight())).item() == 0
    assert pot.stage(_batch(_independent_twists())).item() == 1
    assert pot.stage(_batch(_trefoil())).item() == tp.STAGE_GOAL
    assert pot.stage(_batch(_trefoil_with_tail_loop())).item() == tp.STAGE_GOAL


@pytest.mark.parametrize("seed", range(3))
def test_stage_matches_potential(seed: int):
    batch = tp.segment_intersections(_random_ropes(400, seed=seed, scale=0.35))
    pot = kg.BfsPotential()
    stage = pot.stage(batch)
    assert ((stage >= 0) & (stage <= tp.STAGE_GOAL)).all()
    assert torch.allclose(stage.float() / tp.STAGE_GOAL, pot(batch))
