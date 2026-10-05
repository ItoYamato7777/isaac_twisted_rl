"""対称変換 (鏡像・逆順。`topology/symmetry.py`) の単体テスト。v8 で追加。

Isaac Sim を起動せずに走る (グラフの列挙に約 20 s。`test_knot_graph.py` と同じプロセスなら共有)::

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run python -m pytest tests/test_symmetry.py

## 検証の骨子

* p-data の変換 (鏡像は符号の反転、逆順は点番号の反転) が、変換したロープから直接計算した p-data と一致する。
* 段・class・固結びの判定が、交差 4 個までの全状態で鏡像・逆順に対して不変 (データ拡張で段の
  one-hot をそのまま使う前提)。
* `next_edge` の解析的な変換が、変換したロープの候補のどれかの特徴と一致する (表の辺の全単射)。
* 観測・行動の変換が対合 (2 回で元に戻る) で、鏡像と逆順が可換。拡張の先頭が元のデータ。
* 行動の量子化 (`PickPlaceAction.process_actions` と同じ式) で、逆順が掴むリンク N−1−l・Δ の符号の反転、
  yaw の +π になる。
"""

from __future__ import annotations

import math

import pytest
import topology as tp
import torch
from test_knot_detect import _random_ropes
from topology import knot_graph as kg
from topology import next_edge as ne
from topology import symmetry as sym
from topology.state_2_topology import polyline_to_p_data

N = 28
"""`Knot` プリセットのリンク数。"""

_V7_LAYOUT: list[sym.ObsTermLayout] = [
    ("rope_link_pos", 3 * N, {}),
    ("crossings", 1, {}),
    ("writhe", 1, {}),
    ("crossing_feats", 30, {"max_crossings": 5}),
    ("knot_class", 4, {"reduce_r2": True}),
    ("hand_pos", 3, {}),
    ("actions", 6, {}),
    ("next_edge", ne.feature_dim(N), {"rule": "long_end", "seed": 0}),
    ("knot_stage", 4, {}),
]
"""v6 / v7 の `policy` グループの並び (`Knot` で 197 次元)。"""


# ---------------------------------------------------------------- p-data の変換


def _fmt(items: list[tuple[int, bool, int, bool]]) -> str:
    return "_".join(f"{i}{'O' if o else 'U'}{j}{'+' if s else '-'}" for i, o, j, s in items)


def _mirror_pd(pd: str) -> str:
    return _fmt([(i, o, j, not s) for i, o, j, s in kg._parse(pd)])


def _reverse_pd(pd: str) -> str:
    items = kg._parse(pd)
    m = len(items) + 1
    return _fmt(sorted((m - i, o, m - j, s) for i, o, j, s in items))


def _link_centers(num: int, seed: int) -> torch.Tensor:
    """リンク中心 `(num, N, 3)` のランダムなロープ (float32、観測と同じ型)。"""
    return _random_ropes(num, num_nodes=N, scale=0.4, seed=seed).float()


def _batch(link_pos: torch.Tensor) -> tp.IntersectionBatch:
    return tp.segment_intersections(tp.polyline_from_link_centers(link_pos))


def test_p_data_transforms_match_transformed_ropes():
    ropes = _link_centers(300, seed=1)
    checked = 0
    for mirror, reverse, f in ((True, False, _mirror_pd), (False, True, _reverse_pd)):
        moved = sym.transform_link_pos(ropes, mirror, reverse)
        for r, m in zip(ropes, moved):
            pd = polyline_to_p_data(tp.polyline_from_link_centers(r), short=True)
            if pd == "":
                continue
            assert polyline_to_p_data(tp.polyline_from_link_centers(m), short=True) == f(pd)
            checked += 1
    assert checked > 100


# ---------------------------------------------------------------- 段・class の不変


def _pd_batch(pds: list[str]) -> tp.IntersectionBatch:
    lists = []
    for pd in pds:
        items, orders = kg.p_data_to_lists(pd)
        lists.append((items, orders))
    return tp.IntersectionBatch.from_lists(lists)


def test_stage_and_class_invariant_on_whole_graph():
    """交差 4 個までの全状態で、段 (R2 除去前後の BFS) と R2 除去後の class が不変。"""
    pds = list(kg.enumerate_graph()[0])
    bfs = tp.get_bfs_potential("cpu")

    def stage_and_class(xs: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        b = _pd_batch(xs)
        reduced = tp.reduce_r2(b)
        return bfs.stage(b, reduced), tp.topology_class(reduced)

    stage, cls = stage_and_class(pds)
    for f in (_mirror_pd, _reverse_pd):
        s2, c2 = stage_and_class([f(pd) if pd else pd for pd in pds])
        assert torch.equal(stage, s2)
        assert torch.equal(cls, c2)
    assert int(stage.max()) == kg.STAGE_GOAL  # 固結びを含む


def test_count_and_writhe_on_random_ropes():
    ropes = _link_centers(400, seed=2)
    b = _batch(ropes)
    bm = _batch(sym.transform_link_pos(ropes, True, False))
    br = _batch(sym.transform_link_pos(ropes, False, True))
    assert torch.equal(b.count, bm.count) and torch.equal(b.count, br.count)
    assert torch.equal(bm.writhe, -b.writhe)
    assert torch.equal(br.writhe, b.writhe)
    assert int(b.count.max()) >= 3


# ---------------------------------------------------------------- 次にたどる辺


def test_next_edge_transform_matches_a_candidate():
    """変換した特徴が、変換したロープの表の候補のどれかの特徴と完全に一致する。"""
    table = ne.NextEdgeTable()
    ropes = _link_centers(600, seed=3)
    checked = 0
    for mirror, reverse in ((True, False), (False, True), (True, True)):
        moved = sym.transform_link_pos(ropes, mirror, reverse)
        b, bt = _batch(ropes), _batch(moved)
        row, found, _ = table.lookup(b)
        row_t, found_t, _ = table.lookup(bt)
        assert torch.equal(found, found_t)
        links = ne.point_links(b, N)
        links_t = ne.point_links(bt, N)
        for c in range(table.num_cands):
            cand = torch.full_like(row, c)
            exists = found & (table.kind[row, c] >= 0)
            feat = sym.transform_next_edge(table.features(row, cand, links, N), N, mirror, reverse)
            cands_t = torch.stack(
                [table.features(row_t, torch.full_like(row_t, k), links_t, N) for k in range(table.num_cands)], dim=1
            )
            ok_t = table.kind[row_t] >= 0  # (B, C)
            match = ((cands_t == feat.unsqueeze(1)).all(dim=-1) & ok_t).any(dim=1)
            assert bool(match[exists].all()), (mirror, reverse, c)
            checked += int(exists.sum())
    assert checked > 300


def test_next_edge_zero_stays_zero():
    feat = torch.zeros(5, ne.feature_dim(N))
    for m, r in sym.TRANSFORMS["mirror_reverse"]:
        assert torch.equal(sym.transform_next_edge(feat, N, m, r), feat)


# ---------------------------------------------------------------- 観測の拡張


def _fake_obs(num: int, seed: int) -> torch.Tensor:
    """v7 の並びの観測を、環境と同じ純関数で作る (段は BFS、辺は long_end)。"""
    g = torch.Generator().manual_seed(seed)
    link_pos = _link_centers(num, seed)
    b = _batch(link_pos)
    reduced = tp.reduce_r2(b)
    table = ne.NextEdgeTable()
    row, found, _ = table.lookup(b)
    links = ne.point_links(b, N)
    cand = table.select(row, found, links, b.count, "long_end")
    stage = tp.get_bfs_potential("cpu").stage(b, reduced).clamp(max=3)
    parts = [
        link_pos.reshape(num, -1),
        b.count.float().unsqueeze(-1),
        b.writhe.float().unsqueeze(-1),
        sym.crossing_features_from_batch(b, N + 1, 5),
        tp.class_one_hot(reduced),
        torch.rand(num, 3, generator=g),
        torch.rand(num, 6, generator=g) * 3.0 - 1.5,
        table.features(row, cand, links, N),
        torch.nn.functional.one_hot(stage, 4).float(),
    ]
    return torch.cat(parts, dim=-1)


def test_augmented_obs_shape_and_identity_block():
    obs = _fake_obs(64, seed=4)
    assert obs.shape[1] == 197
    for name, k in (("mirror_reverse", 4), ("mirror", 2), ("reverse", 2)):
        aug = sym.augment_policy_obs(obs, _V7_LAYOUT, name)
        assert aug.shape == (k * 64, 197)
        assert torch.equal(aug[:64], obs)
        # 段の one-hot は変えない
        assert torch.equal(aug[:, -4:], obs[:, -4:].repeat(k, 1))


def test_obs_transforms_are_involutions_and_commute():
    obs = _fake_obs(128, seed=5)
    t = sym._transform_policy_obs

    def close(a: torch.Tensor, b: torch.Tensor) -> bool:
        # 直前の行動の yaw (ch5) は clip した値の上で周期 2 (±1 が同じ角度) で比べる。
        # 逆順は clip してからずらすので、[−1, 1] の外の値は clip した値に戻る (環境も clip する)。
        a5, b5 = a.clone(), b.clone()
        i5 = sum(d for name, d, _ in _V7_LAYOUT[:6]) + 5
        d = torch.remainder(a5[:, i5].clamp(-1, 1) - b5[:, i5].clamp(-1, 1) + 1.0, 2.0) - 1.0
        a5[:, i5] = b5[:, i5] = 0.0
        return torch.allclose(a5, b5, atol=1e-6) and bool((d.abs() < 1e-6).all())

    for m, r in ((True, False), (False, True)):
        assert close(t(t(obs, _V7_LAYOUT, m, r), _V7_LAYOUT, m, r), obs)
    mr = t(t(obs, _V7_LAYOUT, True, False), _V7_LAYOUT, False, True)
    rm = t(t(obs, _V7_LAYOUT, False, True), _V7_LAYOUT, True, False)
    assert close(mr, rm)
    assert close(mr, t(obs, _V7_LAYOUT, True, True))


def test_recomputed_terms_follow_the_topology():
    obs = _fake_obs(256, seed=6)
    dims = [d for _, d, _ in _V7_LAYOUT]
    names = [n for n, _, _ in _V7_LAYOUT]
    split = dict(zip(names, obs.split(dims, dim=-1)))
    for m, r in sym.TRANSFORMS["mirror_reverse"][1:]:
        got = dict(zip(names, sym._transform_policy_obs(obs, _V7_LAYOUT, m, r).split(dims, dim=-1)))
        assert torch.equal(got["crossings"], split["crossings"])
        assert torch.equal(got["knot_class"], split["knot_class"])
        assert torch.equal(got["writhe"], -split["writhe"] if m else split["writhe"])
        # 交差の valid の数は変わらない。sign の総和は鏡像で反転する。over (番号の小さいセグメントが上か)
        # の総和は逆順で反転する (逆順では番号の小さい側が、交差のもう一方の紐になる)。
        f, g = split["crossing_feats"].reshape(-1, 5, 6), got["crossing_feats"].reshape(-1, 5, 6)
        assert torch.equal(f[..., 0].sum(-1), g[..., 0].sum(-1))
        assert torch.equal(g[..., 1].sum(-1), -f[..., 1].sum(-1) if r else f[..., 1].sum(-1))
        assert torch.equal(g[..., 2].sum(-1), -f[..., 2].sum(-1) if m else f[..., 2].sum(-1))


def test_unknown_term_raises():
    obs = torch.zeros(2, 3 * N + 1)
    with pytest.raises(ValueError, match="foo"):
        sym._transform_policy_obs(obs, [("rope_link_pos", 3 * N, {}), ("foo", 1, {})], True, False)


# ---------------------------------------------------------------- 行動


def _quantize(a: torch.Tensor, margin: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """`PickPlaceAction.process_actions` と同じ式で (掴むリンク, Δ, yaw) を作る。"""
    act = a.clamp(-1.0, 1.0)
    lo, hi = margin, N - 1 - margin
    grasp = ((act[:, 0] + 1.0) * 0.5 * (hi - lo) + lo).round().long().clamp(lo, hi)
    delta = (act[:, 1] * 8).round().long()
    return grasp, delta, act[:, 5] * math.pi


@pytest.mark.parametrize("margin", [0, 2])
def test_reverse_action_maps_to_the_same_physical_move(margin: int):
    g = torch.Generator().manual_seed(7)
    a = torch.rand(4096, 6, generator=g) * 3.0 - 1.5
    grasp, delta, yaw = _quantize(a, margin)
    grasp_r, delta_r, yaw_r = _quantize(sym.transform_actions(a, False, True), margin)
    assert torch.equal(grasp_r, N - 1 - grasp)
    assert torch.equal(delta_r, -delta)
    dyaw = torch.remainder(yaw_r - yaw, 2 * math.pi) - math.pi  # 差が π なら 0
    assert bool((dyaw.abs() < 1e-5).all())
    # 置き先 xy・高さはそのまま
    assert torch.equal(sym.transform_actions(a, False, True)[:, 2:5], a[:, 2:5])


def test_mirror_action():
    a = torch.rand(100, 6) * 2.0 - 1.0
    m = sym.transform_actions(a, True, False)
    assert torch.equal(m[:, [0, 1, 2, 4]], a[:, [0, 1, 2, 4]])
    assert torch.equal(m[:, [3, 5]], -a[:, [3, 5]])


def test_augmented_actions():
    a = torch.rand(10, 6) * 2.0 - 1.0
    aug = sym.augment_actions(a, "mirror_reverse")
    assert aug.shape == (40, 6)
    assert torch.equal(aug[:10], a)
