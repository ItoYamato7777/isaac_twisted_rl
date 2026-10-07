"""v11 の段の定義 (`topology/dead_end_stages.py`) と、その差し替えの口の単体テスト。

Isaac Sim を起動せずに走る (グラフの列挙に約 20 s。`test_knot_graph.py` と同じプロセスなら共有)::

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run python -m pytest tests/test_dead_end_stages.py

## 検証の骨子 (`docs/decisions/0024-v11-dead-end-stages.md`)

* 0024 の「実装の見通し」の代表の p-data 5 つが、段の決め方の 1〜4 のそれぞれに当たる値になる。
* Φ の表の全 p-data で v7 (`BfsPotential.stage`) と同じ段。
* R2 除去を先に使うと、表の段 2 の 16 個と段 1 の 12 個が 1 つ下の段になる (1 を先にする理由)。
* `bfs_dead_end` の `next_edge` の表は `bfs` (v7) の表そのもの。
"""

from __future__ import annotations

import pytest
import topology as tp
from topology import dead_end_stages as ds
from topology import knot_graph as kg
from topology import next_edge as ne
from topology import path_stages as ps

DEAD_END_KNOT = "1O6+_2U5+_3O8-_4U7-_5O2+_6U1+_7O4-_8U3-"
"""v7 s43 it330 の段 0 のアーカイブの 613 形を占める 2 つの片方 (交差 4 個・R2 で消えない・ラダー 0)。"""

DEAD_END_MIRROR = "1U6+_2O5+_3U8-_4O7-_5U2+_6O1+_7U4-_8O3-"
"""`DEAD_END_KNOT` の鏡像 (613 形のもう片方)。"""

COVER = "1U4+_2O5+_3O6-_4O1+_5U2+_6U3-"
"""台本の cover 後。表の段 2 で、R2 除去すると段 1 の `1U2+_2O1+`。"""


def _batch(p_data: list[str]) -> tp.IntersectionBatch:
    return tp.IntersectionBatch.from_lists([kg.p_data_to_lists(pd) for pd in p_data])


def _stage(p_data: list[str]) -> list[int]:
    return ds.DeadEndStage().stage(_batch(p_data)).tolist()


def _v7(p_data: list[str]) -> list[int]:
    return kg.get_bfs_potential("cpu").stage(_batch(p_data)).tolist()


@pytest.mark.parametrize(
    ("p_data", "v11", "v7"),
    [
        (DEAD_END_KNOT, ds.DEAD_END, 0),  # 4: 表に無い・ラダー 0
        (DEAD_END_MIRROR, ds.DEAD_END, 0),  # 4: その鏡像
        ("1U6-_2U5+_3O8-_4O7+_5O2+_6O1-_7U4+_8U3-", 0, 0),  # 2: R2 除去で交差なし
        ("1O4-_2O5+_3O6-_4U1-_5U2+_6U3-", 1, 1),  # 2: R2 除去で `1O2-_2U1-` (v7 もラダー 1 で段 1)
        ("1U6-_2O5-_3U8+_4U7-_5U2-_6O1-_7O4-_8O3+", 1, 0),  # 2: R2 除去で表の段 1 の `1U4-_2O3-_3U2-_4O1-` (v7 は 0)
        (COVER, 2, 2),  # 1: 生が表にある (R2 除去を先にしない)
        ("1U6-_2O5-_3O8-_4U7-_5U2-_6O1-_7O4-_8U3-", 2, 2),  # 3: 表に無い・ラダー 2
    ],
)
def test_representative_shapes(p_data: str, v11: int, v7: int):
    assert _stage([p_data]) == [v11]
    assert _v7([p_data]) == [v7]


def test_representative_shapes_take_the_intended_branch():
    """代表の形が、段の決め方の 1〜4 のどれを通るか (表にあるか・ラダー) を直接確かめる。"""
    table = set(kg.potential_table())

    def reduced_p_data(pd: str) -> str:
        return tp.p_data_from_batch(tp.reduce_r2(_batch([pd])), short=True)[0]

    assert COVER in table
    assert reduced_p_data(COVER) == "1U2+_2O1+"
    for pd, reduced in [
        ("1U6-_2U5+_3O8-_4O7+_5O2+_6O1-_7U4+_8U3-", ""),
        ("1O4-_2O5+_3O6-_4U1-_5U2+_6U3-", "1O2-_2U1-"),
    ]:
        assert pd not in table
        assert reduced_p_data(pd) == reduced and reduced in table
    for pd, ladder in [(DEAD_END_KNOT, 0), ("1U6-_2O5-_3O8-_4U7-_5U2-_6O1-_7O4-_8U3-", 2)]:
        assert pd not in table and reduced_p_data(pd) not in table
        assert tp.topology_class(tp.reduce_r2(_batch([pd]))).tolist() == [ladder]


def test_whole_table_matches_v7():
    """Φ の表の全 p-data (321 個) で v7 と同じ段 (変わるのは表に無い形だけ)。"""
    table = sorted(kg.potential_table())
    assert len(table) == 321
    assert _stage(table) == _v7(table)


def test_r2_first_would_lower_table_shapes():
    """R2 除去後の形で先に引くと、表の段 2 の 16 個と段 1 の 12 個が 1 つ下の段になる (1 を先にする理由)。"""
    table = sorted(kg.potential_table())
    batch = _batch(table)
    bfs = kg.get_bfs_potential("cpu")
    reduced = tp.reduce_r2(batch)
    v7 = bfs.stage(batch, reduced).tolist()
    r2_first = bfs.stage(reduced, reduced).tolist()
    lowered = {}
    for a, b in zip(v7, r2_first):
        assert b <= a
        if b < a:
            assert b == a - 1
            lowered[a] = lowered.get(a, 0) + 1
    assert lowered == {1: 12, 2: 16}
    assert r2_first[table.index(COVER)] == 1


def test_reduced_batch_is_optional():
    """アーカイブの復元の検算 (`archive.py`) は R2 除去後の batch を渡さない。渡したときと同じ値になる。"""
    p_data = [DEAD_END_KNOT, COVER, "1O4-_2O5+_3O6-_4U1-_5U2+_6U3-", "", "1O2-_2U1-"]
    batch = _batch(p_data)
    st = ds.DeadEndStage()
    assert st.stage(batch).tolist() == st.stage(batch, tp.reduce_r2(batch)).tolist()


def test_more_than_four_crossings_use_ladder():
    """交差 5 個以上は表を引かない。輪 5 つはラダー 0 なので −1 (v7 は段 0)。"""
    pd = "1O2-_2U1-_3O4-_4U3-_5O6-_6U5-_7O8-_8U7-_9O10-_10U9-"
    assert _stage([pd]) == [ds.DEAD_END]
    assert _v7([pd]) == [0]


def test_dead_end_is_below_every_stage():
    """学習では手の前の段 (0〜2) より下なので `stage_down` で終わる。v10 の `OFF_PATH` と同じ値。"""
    assert ds.DEAD_END == ps.OFF_PATH == -1


# ---------------------------------------------------------------- 差し替えの口


def test_stage_function_selection():
    assert "bfs_dead_end" in ps.STAGE_DEFINITIONS
    st = ps.get_stage_function("bfs_dead_end", "cpu")
    assert isinstance(st, ds.DeadEndStage)
    assert st is ps.get_stage_function("bfs_dead_end", "cpu")
    assert isinstance(ps.get_stage_function("path", "cpu"), ps.PathStage)


def test_next_edge_table_is_v7_table():
    assert ne.get_next_edge_table("cpu", "bfs_dead_end") is ne.get_next_edge_table("cpu", "bfs")
    assert ne.get_next_edge_table("cpu", "path") is not ne.get_next_edge_table("cpu", "bfs")
