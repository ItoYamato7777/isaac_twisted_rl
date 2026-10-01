"""最短の次の辺 (`knot_graph.shortest_edges`) とリンクマスク (`next_edge.py`) の単体テスト。v6 で追加。

Isaac Sim を起動せずに走る (グラフの列挙に約 20 s。`test_knot_graph.py` と同じプロセスなら共有)::

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/test_next_edge.py

## 検証の骨子

* 辺の本数の表 (`shortest_edges` の docstring) と、台本 (`demos/overhand_script.py`) が通る辺が
  候補に入っていること: 輪からの R2 cover、本物の C2 からの `cross(4, 2)`、cover 後の状態からの
  「尻尾の端の置き直し」(子が固結び)。
* 合成の交差リストで、マスクの境界が `[L(a), L(b)]` (L(0) = 0、L(2n+1) = N−1) になること。
* 表に無い状態 (交差 5 個以上・届かない・ゴール) では有効フラグが 0 で、ベクトルが全部 0 になること。
* 実行時のキー (`batch_keys`) で引いた行が `p_data_key` の行と一致すること。
"""

from __future__ import annotations

import collections
import pathlib
import sys

import pytest
import torch

_COMMON = pathlib.Path(__file__).resolve().parents[1] / "src" / "isaac_twisted_rl" / "tasks" / "knot_tying" / "common"
sys.path.insert(0, str(_COMMON))

import topology as tp  # noqa: E402
from topology import knot_graph as kg  # noqa: E402
from topology import next_edge as ne  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from test_knot_detect import _random_ropes  # noqa: E402

N = 28
"""`Knot` プリセットのリンク数。"""

_LOOP = "1U2+_2O1+"
_C2 = "1U3+_2O4+_3O1+_4U2+"
_COVER = "1U4+_2O5+_3O6-_4O1+_5U2+_6U3-"
"""台本の cover の直後に最も多い状態。R2 除去後は C1、d = 1。"""
_OVERHAND = "1U4+_2O5+_3U6+_4O1+_5U2+_6O3+"
_FAKE_C2 = "1U4+_2U5-_3U6+_4O1+_5O2-_6O3+"
"""輪を上から跨いだだけの偽の C2 (ゴールに届かない)。"""


def _edges(pd: str) -> tuple[kg.GraphEdge, ...]:
    return kg.shortest_edges()[pd]


def _crossings(pd: str) -> int:
    return len(pd.split("_")) // 2 if pd else 0


def _batch_from(p_data: str, links: list[int]) -> tp.IntersectionBatch:
    """p-data と「点 p が乗っているリンク」(`links[p - 1]`) から合成の交差リストを作る。"""
    items, orders = [], []
    for i, j in kg.p_data_to_lists(p_data)[1]:
        over_i = kg._parse(p_data)[i - 1][1]
        positive = kg._parse(p_data)[i - 1][3]
        items.append((links[i - 1], links[j - 1], 1 if over_i else -1, 1 if positive else -1))
        orders.append((i, j))
    return tp.IntersectionBatch.from_lists([(items, orders)])


def _features(batch: tp.IntersectionBatch, rule: str = "short_end") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    table = ne.NextEdgeTable()
    row, found, _ = table.lookup(batch)
    links = ne.point_links(batch, N)
    cand = table.select(row, found, links, batch.count, rule)
    return table.features(row, cand, links, N), cand, row


def _mask(lo: int, hi: int) -> list[float]:
    return [1.0 if lo <= k <= hi else 0.0 for k in range(N)]


# ---------------------------------------------------------------- 列挙


def test_generate_moves_matches_generate_next():
    """`generate_moves` の子の並びが `generate_next` (v3) と同じ。"""
    for pd in ("", _LOOP, _C2, _COVER):
        state = kg._enumerate_moves(kg.MAX_GRAPH_CROSSINGS)[0][kg.enumerate_graph()[0].index(pd)]
        assert [s.short_p_data for s in kg.generate_next(state)] == [
            s.short_p_data for _, _, s in kg.generate_moves(state)
        ]


def test_shortest_edge_counts():
    """最短の単純な辺の本数。全状態と、自明な状態から最短の辺だけでたどれる状態の両方。"""
    dist = kg.distance_to_overhand()
    table = kg.shortest_edges()
    simple = {pd: [e for e in es if e.kind != kg.MOVE_END_CROSS] for pd, es in table.items()}
    assert set(table) == {pd for pd, d in dist.items() if d > 0}
    assert all(e.child in dist and dist[e.child] == dist[pd] - 1 for pd, es in simple.items() for e in es)

    def hist(pds) -> dict[int, dict[int, int]]:
        out = collections.defaultdict(collections.Counter)
        for pd in pds:
            out[dist[pd]][len(simple[pd])] += 1
        return {d: dict(c) for d, c in out.items()}

    assert hist(table) == {3: {8: 1}, 2: {4: 32, 6: 8}, 1: {1: 16, 2: 76, 5: 16}}

    seen, frontier = {""}, [""]
    while frontier:
        frontier = [e.child for pd in frontier for e in simple.get(pd, ()) if e.child not in seen]
        seen.update(frontier)
    assert hist(pd for pd in seen if pd in table) == {3: {8: 1}, 2: {4: 4, 6: 4}, 1: {1: 16, 2: 12, 5: 8}}


def test_composite_children_are_closer():
    dist = kg.distance_to_overhand()
    for pd, edges in kg.shortest_edges().items():
        for e in edges:
            if e.kind == kg.MOVE_END_CROSS:
                assert dist[e.child] < dist[pd]
                assert 1 in e.removed or 2 * _crossings(pd) in e.removed  # 頭か尻尾の端の交差


def test_script_edges_are_candidates():
    """台本が通る辺がすべて候補にある (`demos/overhand_script.py` の 5 手)。"""
    loop = {(e.kind, e.params): e.child for e in _edges(_LOOP)}
    assert loop[(kg.MOVE_R2, (1, 2, -1, 1))] == _COVER  # cover
    assert loop[(kg.MOVE_CROSS, (1, 2, 1))] == _C2  # 本物の C2
    assert len(loop) == 4

    c2 = {(e.kind, e.params): e.child for e in _edges(_C2)}
    assert c2[(kg.MOVE_CROSS, (4, 2, 1))] == _OVERHAND  # 尻尾の端
    assert c2[(kg.MOVE_CROSS, (0, 2, 1))] in kg.OVERHAND_P_DATA  # 頭の端

    cover = _edges(_COVER)
    simple = [e for e in cover if e.kind != kg.MOVE_END_CROSS]
    assert [(e.kind, e.params) for e in simple] == [(kg.MOVE_CROSS, (0, 2, 1))]
    assert _crossings(simple[0].child) == 4
    (tuck,) = [e for e in cover if e.kind == kg.MOVE_END_CROSS]
    assert tuck.removed == (3, 6)  # 尻尾の端 (点 6) の交差を外す -> 本物の C2
    assert tuck.params == (4, 2, 1)  # 外した後の C2 で cross(over=4, under=2)
    assert tuck.child == _OVERHAND
    assert tuck.over == (5, 7) and tuck.under == (2, 4)  # 今の点番号: 点 5 -> 尻尾の端、点 2 -> 点 4


def test_remove_crossing():
    assert kg.remove_crossing(_COVER, 6) == (_C2, (3, 6))
    assert kg.remove_crossing(_LOOP, 1) == ("", (1, 2))
    assert kg.remove_crossing(_C2, 1) == ("1O2+_2U1+", (1, 3))


# ---------------------------------------------------------------- 区間 -> リンク


def test_point_links():
    batch = _batch_from(_COVER, [8, 12, 16, 19, 22, 25])
    assert ne.point_links(batch, N).tolist() == [[0, 8, 12, 16, 19, 22, 25, N - 1, N - 1, N - 1]]
    # 細分したポリラインのセグメント番号はリンク番号 x subdivide
    sub = _batch_from(_COVER, [2 * k + 1 for k in (8, 12, 16, 19, 22, 25)])
    assert ne.point_links(sub, N, subdivide=2).tolist() == ne.point_links(batch, N).tolist()
    trivial = tp.IntersectionBatch.from_lists([([], [])])
    assert ne.point_links(trivial, N).tolist() == [[0] + [N - 1] * 9]


def test_feature_dim():
    assert ne.feature_dim(N) == 64
    feat, _, _ = _features(_batch_from(_LOOP, [10, 20]))
    assert feat.shape == (1, 64)


def test_mask_loop_tail_shorter():
    """輪 (点 1 がリンク 10、点 2 がリンク 20)。尻尾側の端の区間が短いので R2 cover を選ぶ。"""
    feat, cand, row = _features(_batch_from(_LOOP, [10, 20]))
    edge = _edges(_LOOP)[int(cand)]
    assert (edge.kind, edge.params) == (kg.MOVE_R2, (1, 2, -1, 1))
    f = feat[0].tolist()
    assert f[:N] == _mask(10, 20)  # over = 区間 1 = 輪
    assert f[N : 2 * N] == _mask(20, N - 1)  # under = 区間 2 = 尻尾の端
    assert f[2 * N : 2 * N + 4] == [0.0, 1.0, 0.0, 0.0]  # R2
    assert f[2 * N + 4 :] == [0.0, -1.0, 1.0, 1.0]  # sign (R2 は 0), left, obu, 有効


def test_mask_loop_head_shorter():
    """輪を頭側に寄せると、頭の端を使う候補 (R2 > cross なので R2(0, 1)) を選ぶ。"""
    feat, cand, _ = _features(_batch_from(_LOOP, [4, 12]))
    edge = _edges(_LOOP)[int(cand)]
    assert (edge.kind, edge.params) == (kg.MOVE_R2, (0, 1, 1, 1))
    assert feat[0, :N].tolist() == _mask(0, 4)
    assert feat[0, N : 2 * N].tolist() == _mask(4, 12)


def test_mask_cover_tuck_composite():
    """cover 後の状態。尻尾の端の置き直し (複合) は、外した点 6 を飛ばして区間を数える。"""
    feat, cand, _ = _features(_batch_from(_COVER, [8, 12, 16, 19, 22, 25]))
    edge = _edges(_COVER)[int(cand)]
    assert edge.kind == kg.MOVE_END_CROSS and edge.child == _OVERHAND
    f = feat[0].tolist()
    assert f[:N] == _mask(22, N - 1)  # over = 点 5 から尻尾の端 (点 6 のリンク 25 を飛ばす)
    assert f[N : 2 * N] == _mask(12, 19)  # under = 点 2 から点 4 (点 3 のリンク 16 を飛ばす)
    assert f[2 * N : 2 * N + 4] == [0.0, 0.0, 0.0, 1.0]
    assert f[2 * N + 4 :] == [1.0, 0.0, 0.0, 1.0]


def test_mask_cover_head_shorter_picks_head_cross():
    """頭の端の区間 (0..L(1)) が尻尾の置き直しの区間より短ければ、頭の端の cross を選ぶ。"""
    feat, cand, _ = _features(_batch_from(_COVER, [3, 12, 16, 19, 20, 25]))
    edge = _edges(_COVER)[int(cand)]
    assert (edge.kind, edge.params) == (kg.MOVE_CROSS, (0, 2, 1))
    assert feat[0, :N].tolist() == _mask(0, 3)
    assert feat[0, N : 2 * N].tolist() == _mask(12, 16)


@pytest.mark.parametrize(
    ("p_data", "links", "rule", "expected"),
    [
        # 輪: 頭の端 0..4 (5 本) < 尻尾の端 12..27 (16 本)
        (_LOOP, [4, 12], "long_end", (kg.MOVE_R2, (1, 2, -1, 1))),
        (_LOOP, [4, 12], "tail_end", (kg.MOVE_R2, (1, 2, -1, 1))),
        # 輪: 頭の端 0..10 (11 本) > 尻尾の端 20..27 (8 本)
        (_LOOP, [10, 20], "long_end", (kg.MOVE_R2, (0, 1, 1, 1))),
        (_LOOP, [10, 20], "tail_end", (kg.MOVE_R2, (1, 2, -1, 1))),
        # cover 後: 頭の cross は 0..8 (9 本)、尻尾の置き直しは 22..27 (6 本)
        (_COVER, [8, 12, 16, 19, 22, 25], "long_end", (kg.MOVE_CROSS, (0, 2, 1))),
        (_COVER, [8, 12, 16, 19, 22, 25], "tail_end", (kg.MOVE_END_CROSS, (4, 2, 1))),
    ],
)
def test_long_end_and_tail_end(p_data: str, links: list[int], rule: str, expected: tuple):
    _, cand, _ = _features(_batch_from(p_data, links), rule)
    edge = _edges(p_data)[int(cand)]
    assert (edge.kind, edge.params) == expected


def test_trivial_state_picks_r1_over_whole_rope():
    feat, cand, _ = _features(tp.IntersectionBatch.from_lists([([], [])]))
    edge = _edges("")[int(cand)]
    assert edge.kind == kg.MOVE_R1 and edge.child == _LOOP
    assert feat[0, :N].tolist() == [1.0] * N
    assert feat[0, N : 2 * N].tolist() == [1.0] * N  # R1 は over と under に同じマスク


def test_rules_first_and_random():
    batch = tp.IntersectionBatch.from_lists([([], [])] * 64)
    table = ne.NextEdgeTable()
    row, found, _ = table.lookup(batch)
    links = ne.point_links(batch, N)
    assert table.select(row, found, links, batch.count, "first").tolist() == [0] * 64
    gen = torch.Generator().manual_seed(0)
    picks = table.select(row, found, links, batch.count, "random", generator=gen)
    assert set(picks.tolist()) == set(range(len(_edges(""))))  # 8 本すべて出る (64 回で)
    with pytest.raises(ValueError):
        table.select(row, found, links, batch.count, "nope")


# ---------------------------------------------------------------- 表に無い状態


def test_invalid_states_are_zero():
    five = "_".join(f"{2 * k + 1}U{2 * k + 2}+_{2 * k + 2}O{2 * k + 1}+" for k in range(5))
    batches = [
        _batch_from(five, list(range(2, 22, 2))),  # 交差 5 個
        _batch_from(_FAKE_C2, [5, 8, 11, 14, 17, 20]),  # ゴールに届かない
        _batch_from(_OVERHAND, [5, 8, 11, 14, 17, 20]),  # ゴールそのもの
    ]
    for batch in batches:
        feat, cand, _ = _features(batch)
        assert cand.tolist() == [-1]
        assert feat.abs().sum().item() == 0.0
    feat, cand, _ = _features(_batch_from(_LOOP, [10, 20]))
    assert feat[0, -1].item() == 1.0


# ---------------------------------------------------------------- キー


def test_table_rows_match_p_data_key():
    """表の全 p-data について、`batch_keys` で引いた行が `p_data_key` の行で、候補の子も一致する。"""
    table = ne.NextEdgeTable()
    p_data = list(kg.shortest_edges())
    batch = tp.IntersectionBatch.from_lists([kg.p_data_to_lists(pd) for pd in p_data])
    row, found, keys = table.lookup(batch)
    assert found.all()
    assert keys.tolist() == [kg.p_data_key(pd) for pd in p_data]
    assert table.keys[row].tolist() == keys.tolist()
    for r, pd in zip(row.tolist(), p_data):
        edges = kg.shortest_edges()[pd]
        assert table.child_key[r, : len(edges)].tolist() == [kg.p_data_key(e.child) for e in edges]
        assert (table.kind[r, len(edges) :] == -1).all()


@pytest.mark.parametrize("seed", range(3))
def test_random_ropes_features_are_well_formed(seed: int):
    """実際の交差検出の出力で: 有効 = 表にある、マスクは連続した 1 の並び、NaN なし。"""
    batch = tp.segment_intersections(_random_ropes(400, seed=seed, scale=0.35))
    num_links = 20  # `_random_ropes` は 21 節点 = 20 セグメント
    table = ne.NextEdgeTable()
    row, found, _ = table.lookup(batch)
    links = ne.point_links(batch, num_links)
    cand = table.select(row, found, links, batch.count)
    feat = table.features(row, cand, links, num_links)
    assert torch.isfinite(feat).all()
    assert torch.equal(feat[:, -1] == 1, found)
    assert found.sum() > 50
    dist = kg.distance_to_overhand()
    p_data = tp.p_data_from_batch(batch, short=True)
    for b in range(batch.num_envs):
        assert bool(found[b]) == (dist.get(p_data[b], 0) > 0 and int(batch.count[b]) <= kg.MAX_GRAPH_CROSSINGS)
        if not found[b]:
            continue
        for part in (feat[b, :num_links], feat[b, num_links : 2 * num_links]):
            on = part.nonzero().flatten()
            assert on.numel() >= 1 and on[-1] - on[0] + 1 == on.numel()  # 連続
