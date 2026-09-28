# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""`knot_tying/topology/knot_detect.py` (固結びの判定) の単体テスト。

`test_knot_tying_topology.py` と同じく **Isaac Sim を起動せずに走る**:

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/test_knot_detect.py

## このテストが守っているもの

判定器の設計根拠は `knot_detect.py` の docstring にある 2 点:

1. 固結び ⟺ 「交差 3 つ・partner gap が全て 3・同符号」。
   移植元の BFS で到達可能な 3 交差状態 528 個を全列挙して確認した性質。
   **gap 条件だけでも符号条件だけでも足りない**ので、どちらを落としても
   落ちるテストを個別に置いてある
   (`test_mixed_sign_three_crossings_rejected` /
   `test_independent_twists_are_not_overhand`)。

2. 実機では余分な交差が乗るので、厳密な `count == 3` ではなく
   **部分列判定**にする。`test_trefoil_with_tail_loop_still_detected` が
   その回帰テストで、尻尾にループを 1 つ足すと交差数が 7 に増え
   点番号が全て振り直される状況を再現している。厳密判定に戻すと落ちる。
"""

from __future__ import annotations

import math
import pathlib
import sys

import pytest
import torch

_KNOT_TYING_DIR = pathlib.Path(__file__).resolve().parents[1] / "src" / "isaac_twisted_rl" / "tasks" / "knot_tying" / "common"
sys.path.insert(0, str(_KNOT_TYING_DIR))

import topology as tp  # noqa: E402
from topology import knot_detect as kd  # noqa: E402

# ---------------------------------------------------------------- ポリライン生成


def _trefoil(num_nodes: int = 120, margin: float = 0.15, frac: float = 1.0) -> torch.Tensor:
    """三葉結び。`frac < 1` で途中まで (= 結びかけ) を作れる。

    `frac` を 0.6 -> 1.0 と動かすと位相クラスが 0 -> 1 -> 2 -> 3 と
    順に上がる。報酬ラダーが「結ぶ動作の途中経過」と対応していることの
    根拠になっている (`test_partial_trefoil_walks_up_the_ladder`)。
    """
    t0 = margin
    t1 = t0 + (2.0 * math.pi - margin - t0) * frac
    ts = [t0 + (t1 - t0) * k / (num_nodes - 1) for k in range(num_nodes)]
    return torch.tensor(
        [
            [
                math.sin(t) + 2.0 * math.sin(2.0 * t),
                math.cos(t) - 2.0 * math.cos(2.0 * t),
                -math.sin(3.0 * t),
            ]
            for t in ts
        ],
        dtype=torch.float64,
    )


def _transform(
    points: torch.Tensor,
    *,
    mirror_z: bool = False,
    reverse: bool = False,
    yaw: float = 0.0,
):
    """キラリティ (z 反転) と走査方向 (節点逆順) と z 軸回転を与える。

    固結びの p-data はこの 3 つで 4 通りに分かれるが、判定はすべて True を
    返さなければならない。
    """
    out = points.clone()
    if mirror_z:
        out[:, 2] = -out[:, 2]
    if reverse:
        out = out.flip(0)
    if yaw:
        c, s = math.cos(yaw), math.sin(yaw)
        x = out[:, 0] * c - out[:, 1] * s
        y = out[:, 0] * s + out[:, 1] * c
        out[:, 0], out[:, 1] = x, y
    return out


def _trefoil_with_tail_loop(num_nodes: int = 120, loop_nodes: int = 26) -> torch.Tensor:
    """三葉結びの尻尾に小さな R1 ループを足す。

    これが厳密判定 (`count == 3`) を壊す実機相当のケース。交差数が 7 に
    増え、交差点の番号が全て振り直される。
    """
    base = _trefoil(num_nodes)
    tail = base[-1]
    loop = [
        [
            tail[0] + 0.9 * math.sin(i / (loop_nodes - 1) * 2.0 * math.pi),
            tail[1] + 0.9 * (1.0 - math.cos(i / (loop_nodes - 1) * 2.0 * math.pi)),
            tail[2] + 0.35 * math.sin(2.0 * i / (loop_nodes - 1) * 2.0 * math.pi),
        ]
        for i in range(loop_nodes)
    ]
    return torch.cat([base, torch.tensor(loop, dtype=torch.float64)])


def _unknotted_coil(num_nodes: int = 180) -> torch.Tensor:
    """交差はあるが結ばれていない螺旋。同符号でも固結びではない。"""
    pts = []
    for i in range(num_nodes):
        t = i / (num_nodes - 1) * 3.0 * 2.0 * math.pi
        pts.append(
            [
                t * 0.25 + 0.45 * math.sin(t),
                0.45 * math.cos(t) - 0.45,
                0.12 * math.sin(t * 0.5),
            ]
        )
    return torch.tensor(pts, dtype=torch.float64)


def _independent_twists(num_nodes: int = 30) -> torch.Tensor:
    """独立した R1 ループ 2 つ。同符号だが gap は `{1, 1}` で行き止まり。"""
    pts = []
    for offset in (-0.9, 0.9):
        rng = range(num_nodes) if offset < 0 else range(1, num_nodes)
        for i in rng:
            a = i / (num_nodes - 1) * 2.0 * math.pi
            pts.append(
                [
                    0.35 * math.sin(a) + offset,
                    0.35 * (1.0 - math.cos(a)),
                    0.05 * math.sin(2.0 * a),
                ]
            )
    return torch.tensor(pts, dtype=torch.float64)


def _straight(num_nodes: int = 21) -> torch.Tensor:
    x = torch.arange(num_nodes, dtype=torch.float64) * 0.04
    return torch.stack([x, torch.zeros_like(x), torch.zeros_like(x)], dim=-1)


def _batch(points: torch.Tensor):
    return tp.segment_intersections(points.unsqueeze(0))


def _random_ropes(num: int, num_nodes: int = 21, scale: float = 0.25, seed: int = 0) -> torch.Tensor:
    """ランダムウォークのロープ。誤検出率の確認用。"""
    torch.manual_seed(seed)
    out = []
    for _ in range(num):
        p = torch.zeros(num_nodes, 3, dtype=torch.float64)
        d = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
        for i in range(1, num_nodes):
            d = d + torch.randn(3, dtype=torch.float64) * scale
            d = d / d.norm()
            p[i] = p[i - 1] + d * 0.04
        out.append(p)
    return torch.stack(out)


# ---------------------------------------------------------------- 固結びの検出


@pytest.mark.parametrize("num_nodes", [21, 25, 31, 40, 60, 120])
def test_overhand_detected_on_trefoil(num_nodes: int):
    """離散化解像度に依らず固結びと判定される。

    21 節点は `twist` プリセット (20 リンク) のポリライン節点数そのもの。
    """
    batch = _batch(_trefoil(num_nodes))
    assert bool(kd.is_overhand(batch)[0])
    assert int(kd.topology_class(batch)[0]) == kd.CLASS_OVERHAND


@pytest.mark.parametrize("mirror_z", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
def test_overhand_both_chiralities_and_directions(mirror_z: bool, reverse: bool):
    """キラリティ 2 通り × 走査方向 2 通り = p-data 4 通りすべてで True。"""
    points = _transform(_trefoil(), mirror_z=mirror_z, reverse=reverse)
    assert bool(kd.is_overhand(_batch(points))[0])


@pytest.mark.parametrize("yaw", [0.0, 0.7, 1.9, 3.5])
def test_overhand_invariant_to_rotation(yaw: float):
    """z 軸回りの回転で判定が変わらない (位相量なので当然だが回帰として置く)。"""
    assert bool(kd.is_overhand(_batch(_transform(_trefoil(), yaw=yaw)))[0])


def test_trefoil_with_tail_loop_still_detected():
    """**余分な交差があっても中の固結びを見つける** (設計の要)。

    厳密な `count == 3` に戻すとこのテストが落ちる。実機のたるんだロープ
    ではこの状況が常態なので、落ちたら学習は成立しない。
    """
    batch = _batch(_trefoil_with_tail_loop())
    assert int(batch.count[0]) > 3, "前提: 尻尾のループで交差数が増えていること"
    assert bool(kd.is_overhand(batch)[0])
    assert int(kd.topology_class(batch)[0]) == kd.CLASS_OVERHAND


# ---------------------------------------------------------------- 棄却されるもの


def test_straight_rope_is_trivial_class():
    batch = _batch(_straight())
    assert int(batch.count[0]) == 0
    assert int(kd.topology_class(batch)[0]) == kd.CLASS_TRIVIAL
    assert not bool(kd.is_overhand(batch)[0])


def test_unknotted_coil_rejected():
    """交差 3 つでも結ばれていなければ False。"""
    batch = _batch(_unknotted_coil())
    assert int(batch.count[0]) == 3
    assert not bool(kd.is_overhand(batch)[0])


def test_independent_twists_are_not_overhand():
    """独立ツイストは同符号でも行き止まり (gap 条件が効いていることの確認)。"""
    batch = _batch(_independent_twists())
    gaps = kd.partner_gaps(batch)[0][: int(batch.count[0])]
    assert bool(kd.sign_consistent(batch)[0]), "前提: 符号は揃っていること"
    assert set(gaps.tolist()) == {1}, "前提: gap が 1 (隣り合う点) であること"
    assert not bool(kd.is_overhand(batch)[0])
    assert int(kd.topology_class(batch)[0]) != kd.CLASS_INTERLEAVED2


def test_mixed_sign_three_crossings_rejected():
    """gap が全て 3 でも符号が揃っていなければ False (符号条件の分離)。"""
    batch = _batch(_trefoil())
    assert bool(kd.is_overhand(batch)[0]), "前提: 素の三葉結びは True"
    # 1 つだけ符号を反転させる。gap (order) は触らない。
    batch.data[0, 0, 3] = -batch.data[0, 0, 3]
    assert not bool(kd.sign_consistent(batch)[0])
    assert not bool(kd.is_overhand(batch)[0])


def test_no_false_positive_on_random_ropes():
    """ランダムロープ 600 本で固結びの誤検出が 0 件。"""
    batch = tp.segment_intersections(_random_ropes(600))
    assert int(kd.is_overhand(batch).sum()) == 0


# ---------------------------------------------------------------- ラダーの段


def test_partial_trefoil_walks_up_the_ladder():
    """結びかけを進めると位相クラスが 0 -> 1 -> 2 -> 3 と単調に上がる。

    報酬ラダーが「結ぶ動作の途中経過」と対応していることの根拠。
    どこかの段が飛ばされるとラダーとして機能しない。
    """
    classes = [int(kd.topology_class(_batch(_trefoil(frac=f)))[0]) for f in (0.60, 0.70, 0.78, 1.0)]
    assert classes == [
        kd.CLASS_TRIVIAL,
        kd.CLASS_SINGLE,
        kd.CLASS_INTERLEAVED2,
        kd.CLASS_OVERHAND,
    ]


def test_single_crossing_is_class_one():
    batch = _batch(_trefoil(frac=0.70))
    assert int(batch.count[0]) == 1
    assert int(kd.topology_class(batch)[0]) == kd.CLASS_SINGLE


def test_interleaved_two_is_class_two():
    """ループに端をくぐらせかけた状態が 2 段目になる。"""
    batch = _batch(_trefoil(frac=0.78))
    assert int(batch.count[0]) == 2
    assert set(kd.partner_gaps(batch)[0][:2].tolist()) == {2}
    assert bool(kd.is_interleaved_two(batch)[0])
    assert int(kd.topology_class(batch)[0]) == kd.CLASS_INTERLEAVED2


def test_overhand_takes_priority_over_interleaved_two():
    """固結びは `is_interleaved_two` も満たすので、段の優先順位が要る。"""
    batch = _batch(_trefoil())
    assert bool(kd.is_interleaved_two(batch)[0]), "前提: 2 段目の条件も満たす"
    assert int(kd.topology_class(batch)[0]) == kd.CLASS_OVERHAND


def test_class_one_hot_shape_and_sum():
    points = torch.stack([_trefoil(120), _trefoil(120, frac=0.78), _straight(120), _unknotted_coil(120)])
    one_hot = kd.class_one_hot(tp.segment_intersections(points))
    assert one_hot.shape == (4, kd.NUM_CLASSES)
    assert torch.all(one_hot.sum(dim=-1) == 1.0)
    assert one_hot.dtype == torch.float32


# ---------------------------------------------------------------- 実装の性質


def test_detector_ignores_row_order():
    """`batch.data` / `order` の行順に依存しない。

    行順は `(seg_i, seg_j)` の辞書順であって走査順ではないため、
    セグメントがずれると同じ結び目でも行の並びが変わりうる。
    """
    batch = _batch(_trefoil())
    assert bool(kd.is_overhand(batch)[0])
    perm = torch.tensor([2, 0, 1])
    batch.data[0, :3] = batch.data[0, perm]
    batch.order[0, :3] = batch.order[0, perm]
    assert bool(kd.is_overhand(batch)[0])


def test_detector_is_batched():
    """異種の env を並べても、1 本ずつ流した結果と一致する。"""
    cases = [
        _trefoil(120),
        _straight(120),
        _unknotted_coil(120),
        _trefoil(120, frac=0.78),
        _independent_twists(60),
    ]
    per_env = [int(kd.topology_class(_batch(c))[0]) for c in cases]
    # 節点数を揃えてから stack する (バッチ演算の前提)。
    resampled = torch.stack([tp.resample_polyline(c, 160) for c in cases])
    batched = kd.topology_class(tp.segment_intersections(resampled)).tolist()
    assert batched == per_env


def test_partner_gaps_zero_on_padding():
    """パディング行の gap は 0 (無効値 PAD が漏れない)。"""
    points = torch.stack([_trefoil(160), _straight(160)])
    batch = tp.segment_intersections(points)
    gaps = kd.partner_gaps(batch)
    assert torch.all(gaps[~batch.valid_mask] == 0)
    assert torch.all(gaps[batch.valid_mask] > 0)


def test_sign_consistent_is_false_without_crossings():
    """交差 0 個は「符号が揃っている」に含めない (段 0 に落とすため)。"""
    assert not bool(kd.sign_consistent(_batch(_straight()))[0])


def test_float32_matches_float64():
    """Isaac Lab の既定 dtype でも判定が変わらない。"""
    points = _trefoil()
    assert bool(kd.is_overhand(_batch(points))[0]) == bool(kd.is_overhand(_batch(points.float()))[0])


def test_respects_max_crossings_considered():
    """交差数の上限を超えたぶんは見ない (組み合わせ爆発の防止)。

    上限を 2 に絞れば 3 個組は作れないので、固結びでも False になる。
    """
    batch = _batch(_trefoil())
    original = kd.MAX_CROSSINGS_CONSIDERED
    try:
        kd.MAX_CROSSINGS_CONSIDERED = 2
        assert not bool(kd.is_overhand(batch)[0])
    finally:
        kd.MAX_CROSSINGS_CONSIDERED = original
    assert bool(kd.is_overhand(batch)[0])


def test_knot_detect_imports_no_isaaclab():
    """`topology/` の不変条件を機械的に守る。

    ここが崩れるとこのテスト自体が Isaac Sim 無しで走らなくなり、
    開発サイクルが桁で遅くなる (`next_action.md` 5 節)。
    """
    for forbidden in ("isaaclab", "omni", "pxr"):
        assert forbidden not in sys.modules


# ---------------------------------------------------------------- 観測の特徴量
#
# `mdp/observations.py` の `crossing_features` は `isaaclab` に依存するため
# (`SceneEntityCfg` / `ManagerBasedRLEnv`) このテストからは import できない。
# そこで**本体と同じ式**をここに写して、テンソル層の性質だけを検証する。
# 実際の配線は in-sim のスモークテスト S1 で確認する。
#
# ここで押さえたいのは 1 点だけ: **正規化の分母が定数であること**。
# 旧 `first_crossing_feature` は `batch.data.shape[1]` (バッチごとに変わる)
# で割っており、同じロープでも「バッチに他に何が入っているか」で値が変わって
# いた (`test_old_normalization_was_batch_dependent` が実際にそれを示す)。


def _crossing_features_core(batch, num_nodes: int, num_envs: int, max_crossings: int = 5) -> torch.Tensor:
    """`observations.crossing_features` のテンソル部分と同じ式。"""
    device = batch.data.device
    out = torch.zeros(num_envs, max_crossings, 6, device=device)
    num_slots = min(batch.max_crossings, max_crossings)
    if num_slots == 0:
        return out.reshape(num_envs, -1)
    denom = float(max(2 * num_nodes, 1))
    valid = batch.valid_mask[:, :num_slots]
    order = batch.order[:, :num_slots].float()
    data = batch.data[:, :num_slots].float()
    gaps = kd.partner_gaps(batch)[:, :num_slots].float()
    feat = torch.stack(
        [
            valid.float(),
            data[..., 2],
            data[..., 3],
            order[..., 0] / denom,
            order[..., 1] / denom,
            gaps / denom,
        ],
        dim=-1,
    )
    feat = torch.where(valid.unsqueeze(-1), feat, torch.zeros_like(feat))
    key = torch.where(
        valid,
        batch.order[:, :num_slots, 0],
        torch.full_like(batch.order[:, :num_slots, 0], 1 << 30),
    )
    feat = feat.gather(1, key.argsort(dim=-1).unsqueeze(-1).expand_as(feat))
    out[:, :num_slots] = feat
    return out.reshape(num_envs, -1)


def _feature_cases(num_nodes: int = 160) -> torch.Tensor:
    cases = [
        _trefoil(120),
        _straight(120),
        _unknotted_coil(120),
        _trefoil(120, frac=0.78),
        _independent_twists(60),
        _trefoil_with_tail_loop(),
    ]
    return torch.stack([tp.resample_polyline(c, num_nodes) for c in cases])


def test_crossing_features_shape_and_finite():
    points = _feature_cases()
    feats = _crossing_features_core(tp.segment_intersections(points), 160, points.shape[0])
    assert feats.shape == (points.shape[0], 5 * 6)
    assert bool(torch.isfinite(feats).all())


def test_crossing_features_bounded():
    """全成分が `[-1, 1]` に収まる。**旧実装のバグに対する回帰テスト**。"""
    points = _feature_cases()
    feats = _crossing_features_core(tp.segment_intersections(points), 160, points.shape[0])
    assert bool((feats.abs() <= 1.0).all())


def test_crossing_features_are_batch_independent():
    """同じロープなら、バッチに何が入っていても同じ値になる。

    旧実装が満たしていなかった性質 (下のテストが対比)。
    """
    num_nodes = 160
    solo = tp.resample_polyline(_trefoil(120), num_nodes).unsqueeze(0)
    solo_feat = _crossing_features_core(tp.segment_intersections(solo), num_nodes, 1)[0]
    batched = _feature_cases(num_nodes)
    batch_feat = _crossing_features_core(tp.segment_intersections(batched), num_nodes, batched.shape[0])[0]
    assert torch.allclose(solo_feat, batch_feat)


def test_old_normalization_was_batch_dependent():
    """旧 `first_crossing_feature` の分母がバッチ依存だったことを固定する。

    このテストは「直したバグを記録する」ためのもの。新実装は
    `test_crossing_features_are_batch_independent` を満たす。
    """

    def old_first_crossing(batch, num_envs):
        out = torch.zeros(num_envs, 4)
        if batch.max_crossings == 0:
            return out
        valid = batch.valid_mask[:, 0]
        data = batch.data[:, 0, :].float()
        order = batch.order[:, 0, :].float()
        denom = float(max(batch.data.shape[1], 1))  # <- バッチごとに変わる
        feat = torch.stack([data[:, 2], data[:, 3], order[:, 0] / denom, order[:, 1] / denom], dim=-1)
        return torch.where(valid.unsqueeze(-1), feat, out)

    num_nodes = 160
    solo = tp.resample_polyline(_trefoil(120), num_nodes).unsqueeze(0)
    solo_feat = old_first_crossing(tp.segment_intersections(solo), 1)[0]
    batched = _feature_cases(num_nodes)
    batch_feat = old_first_crossing(tp.segment_intersections(batched), batched.shape[0])[0]
    assert not torch.allclose(solo_feat, batch_feat), "旧実装はバッチ依存だったはず"
    assert float(solo_feat.abs().max()) > 1.0, "旧実装は 1 を超える値を出しうる"


def test_crossing_features_sorted_by_traversal():
    """有効な交差が走査順 (point_i 昇順) に並ぶ。"""
    num_nodes = 160
    points = _feature_cases(num_nodes)
    feats = _crossing_features_core(tp.segment_intersections(points), num_nodes, points.shape[0])
    for row in feats:
        blocks = row.reshape(5, 6)
        valid = blocks[:, 0] > 0
        point_i = (blocks[valid, 3] * 2 * num_nodes).round().long().tolist()
        assert point_i == sorted(point_i)
        # 無効ブロックは末尾に寄せられ、全成分 0。
        assert bool((blocks[~valid] == 0).all())


def test_crossing_features_truncates_to_k():
    """交差が K を超えても形状は変わらず、先頭 K 個ぶんが埋まる。"""
    num_nodes = 200
    points = tp.resample_polyline(_trefoil_with_tail_loop(), num_nodes).unsqueeze(0)
    batch = tp.segment_intersections(points)
    assert int(batch.count[0]) > 5, "前提: 交差が K=5 を超えていること"
    feats = _crossing_features_core(batch, num_nodes, 1, max_crossings=5)
    assert feats.shape == (1, 30)
    assert int((feats.reshape(5, 6)[:, 0] > 0).sum()) == 5


# ---------------------------------------------------------------- 報酬ラダー
#
# `mdp/rewards.py` の `knot_class_shaping` は `isaaclab` に依存するので、
# ここでは**ポテンシャル整形の数式そのもの**だけを検証する。
# 位相クラスの判定 (上のテスト群) と、この整形の性質が両方正しければ、
# 報酬項は正しく動く。実際の配線は in-sim のスモークテスト S2 で確認する。

_GAMMA = 0.98
_POTENTIALS = (0.0, 0.35, 0.70, 1.0)


def _shaping(before: int, after: int, gamma: float = _GAMMA) -> float:
    """`F = γ·Φ(s') − Φ(s)`。`rewards.knot_class_shaping` と同じ式。"""
    return gamma * _POTENTIALS[after] - _POTENTIALS[before]


def _episode_total(classes: list[int], start: int = 0, gamma: float = _GAMMA) -> float:
    prev, total = start, 0.0
    for cls in classes:
        total += _shaping(prev, cls, gamma)
        prev = cls
    return total


@pytest.mark.parametrize("cls", range(kd.NUM_CLASSES))
def test_shaping_never_rewards_standing_still(cls: int):
    """同じ段に留まるだけでは報酬が増えない。

    **報酬ハッキング対策の中心**。段そのものにボーナスを付ける設計だと
    「2 段目に居座って毎手稼ぐ」が最適方策になってしまう。
    """
    assert _shaping(cls, cls) <= 0.0


def test_shaping_rewards_climbing():
    """段を上がると正、下がると負。"""
    for lo in range(kd.NUM_CLASSES - 1):
        for hi in range(lo + 1, kd.NUM_CLASSES):
            assert _shaping(lo, hi) > 0.0
            assert _shaping(hi, lo) < 0.0


def test_climbing_beats_oscillating():
    """1<->2 を往復するより、固結びまで登る方が総報酬が高い。"""
    oscillate = [1, 2, 1, 2, 1, 2, 1, 2]
    climb = [1, 1, 2, 2, 2, 3, 3, 3]
    assert _episode_total(climb) > _episode_total(oscillate)


def test_shaping_is_a_true_potential():
    """γ=1 なら閉路 (元の段へ戻る) の総和がちょうど 0。

    ポテンシャル整形であることの定義そのもの。これが成り立つ限り
    Ng et al. (1999) により最適方策は変わらない。
    """
    loop = [1, 2, 3, 3, 2, 1, 0]
    assert _episode_total(loop, start=0, gamma=1.0) == pytest.approx(0.0, abs=1e-9)


def test_making_and_breaking_a_knot_is_not_profitable():
    """固結びを作って壊すのを繰り返しても稼げない。"""
    assert _episode_total([3, 0, 3, 0], gamma=1.0) == pytest.approx(0.0, abs=1e-9)
    assert _episode_total([3, 0, 3, 0]) < 0.0


def test_potential_table_matches_class_count():
    """ポテンシャル表の長さが位相クラス数と一致する。"""
    assert len(_POTENTIALS) == kd.NUM_CLASSES


# ---------------------------------------------------------------- 8 スロット打ち切り (Report §6.3)


def _trefoil_with_r1_loops(num_loops: int, at_head: bool):
    """三葉結び (上下交互・同符号) に、独立した R1 ループを `num_loops` 個足した交差リスト。

    `paper/Report.md` 付録 A-2 の再現。セグメント番号は点番号をそのまま使うので、
    スロットの辞書順 = 紐の始点側からの順になる (実機の `segment_intersections` と同じ)。
    """
    items, orders = [], []
    loops = [(2 * t + 1, 2 * t + 2) for t in range(num_loops)]
    base = 2 * num_loops if at_head else 0
    trefoil = [(base + 1, base + 4, 1), (base + 2, base + 5, -1), (base + 3, base + 6, 1)]
    if not at_head:
        loops = [(6 + p, 6 + q) for p, q in loops]
    for p, q in loops:
        items.append((p, q, 1, 1))
        orders.append((p, q))
    for p, q, over in trefoil:
        items.append((p, q, over, 1))
        orders.append((p, q))
    ranked = sorted(zip(items, orders))
    return [x for x, _ in ranked], [o for _, o in ranked]


@pytest.mark.parametrize("num_loops", [5, 6, 7, 9])
@pytest.mark.parametrize("at_head", [True, False])
def test_overhand_found_regardless_of_r1_loop_side(num_loops: int, at_head: bool):
    """R1 ループが頭側にあっても尻尾側にあっても固結びを見つける (以前は頭側 6 個で失敗した)。"""
    batch = tp.IntersectionBatch.from_lists([_trefoil_with_r1_loops(num_loops, at_head)])
    assert int(batch.count.item()) == num_loops + 3
    assert bool(kd.is_overhand(batch).item())
    assert int(kd.topology_class(tp.reduce_r2(batch)).item()) == kd.CLASS_OVERHAND
    assert not bool(kd.subsequence_truncated(batch).item())


def test_subsequence_truncated_counts_only_useful_crossings():
    """gap ≥ 2 の交差が上限を超えたときだけ True。"""
    many = [(p, p + 2, 1 if p % 2 else -1, 1) for p in range(1, 20, 4)]  # gap 2 の交差を多数
    items = [(p, q, o, s) for p, q, o, s in many] + [(p + 1, p + 3, 1, 1) for p, *_ in many]
    orders = [(p, q) for p, q, *_ in items]
    batch = tp.IntersectionBatch.from_lists([(sorted(items), sorted(orders))])
    useful = int((kd.partner_gaps(batch) >= 2).sum().item())
    assert bool(kd.subsequence_truncated(batch).item()) == (useful > kd.MAX_CROSSINGS_CONSIDERED)
