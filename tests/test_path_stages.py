"""v10 の段の定義 (`topology/path_stages.py`) と、段を差し替える口の単体テスト。

Isaac Sim を起動せずに走る (グラフの列挙に約 20 s。`test_knot_graph.py` と同じプロセスなら共有)::

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run python -m pytest tests/test_path_stages.py

## 検証の骨子

* 段 0〜2 の表と経路の終点が、`AbstractState` の手で自明な状態から経路 A / A' / B / B' をたどって作れる
  (`docs/decisions/0023-v10-path-stages.md` の「正規の手」の表)。
* 段の値: 表の 7 個は表の段、固結び (BFS の段 >= 3) は段 3、それ以外は −1。段 2 の 4 個のどれからも、1 手で
  経路の終点 1 個ともう 1 つの固結び 1 個へ行け、どちらも段 3 になる (段 3 を v7 と同じ判定にした理由)。
* `next_edge` の表 (`path`) の候補は、子が経路の上にある辺だけ。`bfs` (既定) の表は今までと同じ。
* 既定 (`stage_definition` を持たない設定) は `BfsPotential` のままで、`common/mdp/` の段を引く箇所は
  すべて `stage_function` を通る (v4〜v9 の挙動が変わらないことの静的な検査)。
"""

from __future__ import annotations

import ast
import pathlib
import types

import pytest
import topology as tp
import torch
from topology import knot_graph as kg
from topology import next_edge as ne
from topology import path_stages as ps
from topology.representation import AbstractState

MDP = (
    pathlib.Path(__file__).resolve().parents[1] / "src" / "isaac_twisted_rl" / "tasks" / "knot_tying" / "common" / "mdp"
)

ROUTES = {
    # 経路: (R1 の (idx, sign, left)、C2 の (over, under, sign)、C3 の (over, under, sign))、(段 1, 段 2, 段 3)
    "A": (((0, -1, 1), (2, 1, -1), (2, 4, -1)), ("1O2-_2U1-", "1O3-_2U4-_3U1-_4O2-", "1O4-_2U5-_3O6-_4U1-_5O2-_6U3-")),
    "A'": (((0, -1, 1), (1, 0, -1), (0, 2, -1)), ("1O2-_2U1-", "1U3-_2O4-_3O1-_4U2-", "1O4-_2U5-_3O6-_4U1-_5O2-_6U3-")),
    "B": (((0, 1, 1), (0, 1, 1), (2, 0, 1)), ("1U2+_2O1+", "1O3+_2U4+_3U1+_4O2+", "1U4+_2O5+_3U6+_4O1+_5U2+_6O3+")),
    "B'": (((0, 1, 1), (1, 2, 1), (4, 2, 1)), ("1U2+_2O1+", "1U3+_2O4+_3O1+_4U2+", "1U4+_2O5+_3U6+_4O1+_5U2+_6O3+")),
}
"""0023 の「正規の手」の表。R1 はリポジトリの引数の順 (idx, sign, left) で、論文の (index, left, sign) と違う。"""

OTHER_KNOTS = ("1O4+_2U5+_3O6+_4U1+_5O2+_6U3+", "1U4-_2O5-_3U6-_4O1-_5U2-_6O3-")
"""段 2 から 1 手で行ける、経路の終点ではない固結び。"""


def _batch(p_data: list[str]) -> tp.IntersectionBatch:
    return tp.IntersectionBatch.from_lists([kg.p_data_to_lists(pd) for pd in p_data])


def _stage(p_data: list[str]) -> list[int]:
    return ps.PathStage().stage(_batch(p_data)).tolist()


# ---------------------------------------------------------------- 表と経路


@pytest.mark.parametrize("route", sorted(ROUTES))
def test_routes_reproduce_table(route: str):
    (r1, c2, c3), expected = ROUTES[route]
    state = AbstractState()
    assert state.Reide1(*r1)
    got = [state.short_p_data]
    for move in (c2, c3):
        assert state.cross(*move)
        got.append(state.short_p_data)
    assert tuple(got) == expected
    assert got[0] in ps.PATH_STAGE_P_DATA[1]
    assert got[1] in ps.PATH_STAGE_P_DATA[2]
    assert got[2] in ps.STRICT_GOAL_P_DATA


def test_table_is_covered_by_routes():
    stage1 = {r[1][0] for r in ROUTES.values()}
    stage2 = {r[1][1] for r in ROUTES.values()}
    goals = {r[1][2] for r in ROUTES.values()}
    assert stage1 == set(ps.PATH_STAGE_P_DATA[1])
    assert stage2 == set(ps.PATH_STAGE_P_DATA[2])
    assert goals == set(ps.STRICT_GOAL_P_DATA)
    assert ps.PATH_STAGE_P_DATA[0] == ("",)


# ---------------------------------------------------------------- 段の値


def test_path_p_data_get_their_stage():
    for stage, p_data in ps.PATH_STAGE_P_DATA.items():
        assert _stage(list(p_data)) == [stage] * len(p_data)


def test_all_overhand_knots_are_stage3_and_only_two_are_strict():
    knots = list(ps.STRICT_GOAL_P_DATA) + list(OTHER_KNOTS)
    assert _stage(knots) == [tp.STAGE_GOAL] * 4
    strict = ps.strict_goal_mask(tp.batch_keys(_batch(knots))).tolist()
    assert strict == [True, True, False, False]


def test_off_path_shapes():
    off = [
        "1O2+_2U1+",  # 段 0 の R1 のうち経路に乗らない向き
        "1U2-_2O1-",
        "1U4+_2O5+_3O6-_4O1+_5U2+_6U3-",  # 台本の cover 後 (R2 除去すると段 1 と同じ)
        "1U4+_2U5-_3U6+_4O1+_5O2-_6O3+",  # 偽の C2
    ]
    assert _stage(off) == [ps.OFF_PATH] * len(off)


def test_whole_bfs_table_against_v7_stage():
    """Φ の表の全 p-data: 段 3 は v7 の段 3 と一致し、段 0〜2 は表の 7 個だけ、残りは −1。"""
    table = sorted(kg.potential_table())
    batch = _batch(table)
    v7 = kg.get_bfs_potential("cpu").stage(batch).tolist()
    v10 = ps.PathStage().stage(batch).tolist()
    on_path = {pd: s for s, pds in ps.PATH_STAGE_P_DATA.items() for pd in pds}
    for pd, a, b in zip(table, v7, v10):
        if a >= tp.STAGE_GOAL:
            assert b == tp.STAGE_GOAL, pd
        elif pd in on_path:
            assert b == on_path[pd] == a, pd  # 経路の上の形は v7 と同じ段
        else:
            assert b == ps.OFF_PATH, pd


def test_more_than_four_crossings_are_off_path_unless_knot():
    """交差 5 個以上はキーが −1 になり、表を引かずに −1 (固結びを含めば段 3、ラダーで v7 と同じ)。"""
    pd = "1O2-_2U1-_3O4-_4U3-_5O6-_6U5-_7O8-_8U7-_9O10-_10U9-"  # 輪 5 つ
    assert _stage([pd]) == [ps.OFF_PATH]


def test_exits_from_stage2():
    """段 2 の 4 個のどれからも、1 手の出口のうち固結びはちょうど 2 個 (経路の終点 1 + もう 1 つ)。"""
    states, moves = kg._enumerate_moves(kg.MAX_GRAPH_CROSSINGS)
    index = {s.short_p_data: i for i, s in enumerate(states)}
    for pd in ps.PATH_STAGE_P_DATA[2]:
        children = [states[j].short_p_data for _, _, j in moves[index[pd]]]
        stages = _stage(children)
        knots = [c for c, s in zip(children, stages) if s == tp.STAGE_GOAL]
        assert len(knots) == 2, pd
        assert sum(c in ps.STRICT_GOAL_P_DATA for c in knots) == 1, pd
        assert sum(c in OTHER_KNOTS for c in knots) == 1, pd
        assert all(s in (ps.OFF_PATH, tp.STAGE_GOAL) for s in stages), pd  # 1 手は交差を増やすので段 0〜2 には無い


# ---------------------------------------------------------------- 差し替えの口


def test_stage_function_selection():
    assert ps.get_stage_function("bfs", "cpu") is kg.get_bfs_potential("cpu")
    assert isinstance(ps.get_stage_function("path", "cpu"), ps.PathStage)
    assert ps.get_stage_function("path", "cpu") is ps.get_stage_function("path", "cpu")
    with pytest.raises(ValueError):
        ps.get_stage_function("phi", "cpu")


def test_default_definition_is_bfs():
    assert ps.stage_definition_of(types.SimpleNamespace()) == "bfs"
    assert ps.stage_definition_of(types.SimpleNamespace(stage_definition="path")) == "path"


def test_mdp_stage_calls_go_through_stage_function():
    """`common/mdp/` で段を引く箇所は `stage_function` か `get_next_edge_table(..., 定義)` を通る。

    `BfsPotential.stage` を直接呼ぶと、v10 の段の定義が効かない箇所が黙って残る。`rewards.py` の
    `get_bfs_potential` は Φ の整形 (v3) で段ではないので対象外。
    """
    problems = []
    for path in sorted(MDP.glob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            src = ast.unparse(node.func)
            if src.endswith(".stage") and "stage_function" not in src and src not in ("pot.stage",):
                problems.append(f"{path.name}:{node.lineno}: {src}")
            if src.endswith("get_next_edge_table") and len(node.args) + len(node.keywords) < 2:
                problems.append(f"{path.name}:{node.lineno}: 段の定義を渡していない get_next_edge_table")
    assert not problems, "\n".join(problems)
    stages_src = (MDP / "stages.py").read_text()
    assert "pot = stage_function(env)" in stages_src


# ---------------------------------------------------------------- next_edge


def test_next_edge_default_table_is_unchanged():
    assert ne.get_next_edge_table("cpu") is ne.get_next_edge_table("cpu", "bfs")
    a, b = ne.get_next_edge_table("cpu"), ne.NextEdgeTable("cpu")
    for name in ("keys", "kind", "over_a", "over_b", "under_a", "under_b", "sign", "left", "obu", "child_key"):
        assert torch.equal(getattr(a, name), getattr(b, name)), name
    with pytest.raises(ValueError):
        ne.get_next_edge_table("cpu", "phi")


def _names() -> dict[int, str]:
    """キー -> p-data (グラフに現れる全部の状態。固結びは `shortest_edges` の行に無いので列挙から引く)。"""
    states, _ = kg._enumerate_moves(kg.MAX_GRAPH_CROSSINGS)
    return {kg.p_data_key(st.short_p_data): st.short_p_data for st in states}


def test_next_edge_path_candidates():
    """候補は子が経路の上 (段 2 は経路の終点) にある辺だけ。経路の上に無い行は候補なし。"""
    table = ne.get_next_edge_table("cpu", "path")
    children = ps.path_children()
    names = _names()
    expected_counts = {"": 2, "1O2-_2U1-": 2, "1U2+_2O1+": 2, **{pd: 1 for pd in ps.PATH_STAGE_P_DATA[2]}}
    for r, key in enumerate(table.keys.tolist()):
        pd = names[key]
        exists = (table.kind[r] >= 0).tolist()
        got = {names[k] for k, e in zip(table.child_key[r].tolist(), exists) if e}
        if pd in children:
            assert got <= children[pd] and len(got) == expected_counts[pd], pd
        else:
            assert not any(exists), pd


def test_next_edge_path_select_points_to_next_stage():
    """経路の上の 7 形では、選んだ辺の子が 1 つ上の段。外れた形では辺なし (−1)。"""
    table = ne.get_next_edge_table("cpu", "path")
    p_data = [pd for s in (0, 1, 2) for pd in ps.PATH_STAGE_P_DATA[s]] + ["1O2+_2U1+"]
    batch = _batch(p_data)
    row, found, _ = table.lookup(batch)
    cand = table.select(row, found, ne.point_links(batch, 28), batch.count, "long_end")
    assert (cand[:-1] >= 0).all() and int(cand[-1]) == -1
    names = _names()
    child = [names[k] for k in table.chosen_child_key(row, cand)[:-1].tolist()]
    assert _stage(child) == [s + 1 for s in _stage(p_data[:-1])]
