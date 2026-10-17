"""v16 (寸法の DR) の純粋な部分と設定の形を、Isaac Sim なしで検査する。

- 寸法の組のプリセット (`rope_specs.make_size_variant`): 公称の組は `knot` そのもの、寸法と材料の導出、角度制限。
- env への割り当てと critic の特権観測の正規化 (`domain_rand.size_assignment`・`normalize_size`)。
- アーカイブの寸法の割り戻し (`state_archive.rescale_rel_pose`): 保存と復元で元に戻り、関節角が同じ鎖の root の
  重心からの位置がリンク間隔の倍率で伸びる。
- v16 の設定は v15 のクラスを継承し、寸法の割り当て・特権観測 1 項・起動時の検算だけを足す (`ast` で読む)。
- v15 までの版は寸法の DR の名前を使わない (既存の版の挙動を変えていない)。

spawn と env ごとの寸法の読み出し (`mdp/rope_size.py`) は Isaac Sim が要るので、起動時の検算
(`mdp.check_rope_sizes`) とスモークテストで確かめる。
"""

from __future__ import annotations

import ast
import math
import sys

import domain_rand as dr
import pytest
import state_archive as sa
import torch
from test_v9_cfg import VERSIONS, _body_without_docstring, _classes, _imported_as, _parse, _preset_call

sys.path.insert(0, str(VERSIONS.parent / "common" / "rope_model"))
import rope_specs as rs  # noqa: E402

GRID = [(ls, ds) for ls in rs.SIZE_LENGTH_SCALES for ds in rs.SIZE_DIAMETER_SCALES]


def test_grid_levels_are_the_user_decision():
    # 0029 の「寸法の範囲」(ユーザー、2026-10-15)。
    assert rs.SIZE_LENGTH_SCALES == (0.8, 1.0, 1.25)
    assert rs.SIZE_DIAMETER_SCALES == (0.4, 0.63, 1.0)
    assert len(rs.SIZE_SCALES) == 9
    assert set(rs.SIZE_SCALES.values()) == set(GRID)


def test_nominal_size_is_the_knot_preset_itself():
    assert rs.size_variant_name("knot", 1.0, 1.0) == "knot"
    assert rs.SIZE_SCALES["knot"] == (1.0, 1.0)
    # make_size_variant で作り直しても knot と同じ値になる (導出が knot の作り方と一致している)。
    again = rs.make_size_variant(rs.get_spec("knot"), 1.0, 1.0)
    knot = rs.get_spec("knot")
    for field in ("num_links", "capsule_radius", "capsule_half_length", "link_mass", "bend_stiffness", "bend_damping"):
        assert getattr(again, field) == pytest.approx(getattr(knot, field), rel=1e-12), field
    assert again.bend_limit_deg == pytest.approx(knot.bend_limit_deg)
    assert again.twist_stiffness == pytest.approx(knot.twist_stiffness)
    assert again.twist_limit_deg == knot.twist_limit_deg


@pytest.mark.parametrize(("ls", "ds"), GRID)
def test_size_variant_dimensions_and_material(ls, ds):
    knot = rs.get_spec("knot")
    spec = rs.get_spec(rs.size_variant_name("knot", ls, ds))
    assert spec.num_links == knot.num_links == 28
    assert spec.capsule_radius == pytest.approx(knot.capsule_radius * ds)
    assert spec.capsule_half_length == pytest.approx(knot.capsule_half_length * ls)
    assert spec.diameter == pytest.approx(knot.diameter * ds)
    # 同じ材料: 曲げのばね = E I / リンク間隔 (I ∝ r^4)、捩り = 0.5 × 曲げ、質量 = 密度 × カプセルの体積。
    assert spec.bend_stiffness == pytest.approx(knot.bend_stiffness * ds**4 / ls)
    assert spec.twist_stiffness == pytest.approx(rs.TORSION_TO_BENDING_RATIO * spec.bend_stiffness)
    r, seg = spec.capsule_radius, spec.link_spacing
    assert spec.link_mass == pytest.approx(rs.ROPE_DENSITY * (math.pi * r**2 * seg + 4.0 / 3.0 * math.pi * r**3))
    # USD のファイル名は組ごとに別 (公称の組だけ knot の USD)。
    assert spec.usd_path.endswith(f"rope_{spec.name}.usd")


def test_bend_limit_is_removed_only_where_it_saturates():
    for ls, ds in GRID:
        spec = rs.get_spec(rs.size_variant_name("knot", ls, ds))
        sin_half = spec.link_spacing / (2.0 * rs.MIN_BEND_RADIUS_IN_DIAMETERS * spec.diameter)
        if sin_half >= 1.0:
            assert spec.bend_limit_deg is None, spec.name
        else:
            assert spec.bend_limit_deg == pytest.approx(math.degrees(2.0 * math.asin(sin_half))), spec.name
            assert spec.bend_limit_deg < 180.0
    # 直径 ×0.4 の 3 組が制限なし (最小曲げ半径 16 mm < リンク間隔の半分)。
    free = {n for n in rs.SIZE_SCALES if rs.get_spec(n).bend_limit_deg is None}
    assert free == {"knot_l080_d040", "knot_l100_d040", "knot_l125_d040"}


@pytest.mark.parametrize("ls", rs.SIZE_LENGTH_SCALES)
def test_length_variants_keep_nominal_joint_limits(ls):
    # v16 の学習の組 (ユーザー、2026-10-17): 長さだけを振り、曲げ ±60°・捩り ±90° を全組で揃える。
    knot = rs.get_spec("knot")
    name = rs.size_variant_name("knot", ls, 1.0, keep_joint_limits=True)
    assert name == ("knot" if ls == 1.0 else f"knot_l{round(ls * 100):03d}")
    assert rs.LENGTH_VARIANT_SCALES[name] == (ls, 1.0)
    spec = rs.get_spec(name)
    assert spec.bend_limit_deg == knot.bend_limit_deg
    assert spec.twist_limit_deg == knot.twist_limit_deg == 90.0
    # ばね・減衰・質量は角度制限を導出する組と同じ (変えたのは角度制限だけ)。
    derived = rs.get_spec(rs.size_variant_name("knot", ls, 1.0))
    for field in (
        "capsule_radius",
        "capsule_half_length",
        "link_mass",
        "bend_stiffness",
        "bend_damping",
        "twist_stiffness",
    ):
        assert getattr(spec, field) == getattr(derived, field), field
    # 1 本おきのリンクの最も近い点はリンク間隔ぶん離れ、接触の届く範囲 (2 r + 2 contact_offset) より遠い。
    assert spec.link_spacing > 2.0 * spec.capsule_radius + 2.0 * spec.contact_offset


def test_kept_limits_only_for_length_only():
    assert set(rs.LENGTH_VARIANT_SCALES) == {"knot_l080", "knot", "knot_l125"}
    with pytest.raises(ValueError):
        rs.size_variant_name("knot", 0.8, 0.4, keep_joint_limits=True)


def test_size_variant_only_from_knot():
    with pytest.raises(ValueError):
        rs.make_size_variant(rs.get_spec("twist"), 0.8, 0.4)


def test_existing_presets_are_unchanged():
    # 寸法の DR を足しても既存のプリセットの値は変わらない (USD を作り直さない前提)。
    knot = rs.get_spec("knot")
    assert (knot.capsule_radius, knot.capsule_half_length, knot.bend_limit_deg is not None) == (0.01, 0.02, True)
    assert knot.rope_length == pytest.approx(1.14)
    fine = rs.get_spec("fine")
    assert fine.capsule_half_length == pytest.approx(rs.HALF_LENGTH_TO_RADIUS * fine.capsule_radius)


def test_size_assignment_matches_spawn_rule():
    idx = dr.size_assignment(4096, 9)
    assert idx.tolist()[:11] == [0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1]
    counts = torch.bincount(idx, minlength=9).tolist()
    assert counts == [456] + [455] * 8
    assert torch.all(dr.size_assignment(5, 1) == 0)
    with pytest.raises(ValueError):
        dr.size_assignment(4, 0)


def test_normalize_size_maps_grid_range_to_unit_interval():
    lo = torch.tensor([min(rs.SIZE_LENGTH_SCALES), min(rs.SIZE_DIAMETER_SCALES)])
    hi = torch.tensor([max(rs.SIZE_LENGTH_SCALES), max(rs.SIZE_DIAMETER_SCALES)])
    s = torch.tensor([[0.8, 0.4], [1.25, 1.0], [1.0, 1.0], [1.0, 0.63]])
    out = dr.normalize_size(s, lo, hi)
    assert torch.allclose(out[0], torch.tensor([-1.0, -1.0]))
    assert torch.allclose(out[1], torch.tensor([1.0, 1.0]))
    # 公称: 長さは log2 の範囲の中点からわずかに下 (0.8 × 1.25 = 1)、直径は上端。
    assert out[2, 0] == pytest.approx(0.0, abs=1e-6)
    assert out[2, 1] == pytest.approx(1.0)
    # 直径 ×0.63 は log2 でほぼ中点 (0.4 と 1.0 の幾何平均は 0.632)。
    assert out[3, 1] == pytest.approx((math.log2(0.63) - 0.5 * math.log2(0.4)) / (-0.5 * math.log2(0.4)), abs=1e-6)
    # 範囲が 1 点の軸は 0。
    assert torch.all(dr.normalize_size(s, torch.tensor([1.0, 0.4]), torch.tensor([1.0, 1.0]))[:, 0] == 0)


def test_rescale_rel_pose_roundtrip_and_orientation():
    g = torch.Generator().manual_seed(0)
    rel = torch.randn(6, 7, generator=g)
    rel[:, 3:7] = torch.nn.functional.normalize(rel[:, 3:7], dim=-1)
    ls = torch.tensor([0.8, 1.0, 1.25, 0.8, 1.0, 1.25])
    ds = torch.tensor([0.4, 0.63, 1.0, 1.0, 0.4, 0.63])
    nominal = sa.rescale_rel_pose(rel, 1.0 / ls, 1.0 / ds)
    back = sa.rescale_rel_pose(nominal, ls, ds)
    assert torch.allclose(back, rel, atol=1e-6)
    assert torch.equal(nominal[:, 3:7], rel[:, 3:7])
    assert torch.allclose(nominal[:, 0], rel[:, 0] / ls)
    assert torch.allclose(nominal[:, 2], rel[:, 2] / ds)
    # 倍率 1 なら変わらない (v15 までのアーカイブをそのまま読める)。
    ones = torch.ones(6)
    assert torch.equal(sa.rescale_rel_pose(rel, ones, ones), rel)


def test_root_offset_scales_with_link_spacing():
    # 関節角が同じ平面の鎖では、root の重心からの位置はリンク間隔に比例する (アーカイブを長さの倍率で伸ばす根拠)。
    angles = torch.linspace(-0.6, 0.9, 27)
    heading = torch.cat([torch.zeros(1), torch.cumsum(angles, 0)])

    def chain(spacing):
        step = spacing * torch.stack([torch.cos(heading[:-1]), torch.sin(heading[:-1])], dim=-1)
        pts = torch.cat([torch.zeros(1, 2), torch.cumsum(step, 0)])
        return pts[0] - pts.mean(dim=0)

    for ls in rs.SIZE_LENGTH_SCALES:
        assert torch.allclose(chain(0.04 * ls), chain(0.04) * ls, atol=1e-7)


# ---------------------------------------------------------------- 設定の形 (ast)


def test_v16_env_cfg_adds_only_size_dr():
    tree = _parse("v16/env_cfg.py")
    classes = {c.name: c for c in _classes(tree)}
    assert set(classes) == {"EventCfg", "ObservationsCfg", "KnotTyingV16EnvCfg"}
    cfg = classes["KnotTyingV16EnvCfg"]
    assert [ast.unparse(b) for b in cfg.bases] == ["KnotTyingV15EnvCfg"]
    assert _imported_as(tree, "KnotTyingV15EnvCfg") == ("v15.env_cfg", "KnotTyingV15EnvCfg")
    assert [ast.unparse(s) for s in _body_without_docstring(cfg)] == [
        "observations: ObservationsCfg = ObservationsCfg()",
        "events: EventCfg = EventCfg()",
        "rope_size_scales: tuple[tuple[float, float], ...] | None = ROPE_SIZE_SCALES",
        "rope_size_keep_joint_limits: bool = KEEP_JOINT_LIMITS",
    ]
    ev = classes["EventCfg"]
    assert [ast.unparse(b) for b in ev.bases] == ["V14EventCfg"]
    assert [ast.unparse(s) for s in _body_without_docstring(ev)] == [
        "rope_sizes = EventTerm(func=mdp.check_rope_sizes, mode='startup')"
    ]
    obs = classes["ObservationsCfg"]
    assert [ast.unparse(b) for b in obs.bases] == ["V14ObservationsCfg"]
    inner = next(n for n in obs.body if isinstance(n, ast.ClassDef))
    assert [ast.unparse(b) for b in inner.bases] == ["V14ObservationsCfg.RopeParamsCfg"]
    assert [ast.unparse(s) for s in _body_without_docstring(inner)] == [
        "rope_size = ObsTerm(func=mdp.rope_size_params)"
    ]


def test_v16_size_grid_and_play():
    tree = _parse("v16/env_cfg.py")
    src = ast.unparse(tree)
    # 長さの 3 組・直径 ×1.0・角度制限は公称のまま (ユーザー、2026-10-17)。
    assert "ROPE_SIZE_SCALES = tuple(((ls, 1.0) for ls in SIZE_LENGTH_SCALES))" in src
    assert "KEEP_JOINT_LIMITS = True" in src
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "play_post_init")
    assert [ast.unparse(s) for s in _body_without_docstring(fn)] == [
        "_v15_play_post_init(cfg)",
        "cfg.scene.rope.spawn.scales = NOMINAL_SIZE",
    ]
    call = _preset_call(tree)
    assert ast.unparse(call.args[0]) == "KnotTyingV16EnvCfg"


def test_v16_agent_cfg_only_sets_log_name():
    tree = _parse("v16/agents/rsl_rl_ppo_cfg.py")
    (runner,) = _classes(tree)
    assert [ast.unparse(b) for b in runner.bases] == ["_V15MLP"]
    assert _imported_as(tree, "_V15MLP") == ("v15.agents.rsl_rl_ppo_cfg", "KnotTyingPPORunnerCfg")
    assert [ast.unparse(b) for b in _body_without_docstring(runner)] == ["experiment_name = 'knot_tying_v16'"]
    # critic の並びは v14 から継ぐ (特権観測を前、policy を後ろ)。
    v14 = _classes(_parse("v14/agents/rsl_rl_ppo_cfg.py"))[0]
    assert "obs_groups = {'policy': ['policy'], 'critic': ['rope_params', 'policy']}" in ast.unparse(v14)


def test_older_versions_do_not_use_size_dr():
    names = (
        "rope_size_scales",
        "rope_size_keep_joint_limits",
        "check_rope_sizes",
        "rope_size_params",
        "RopeSizeSpawnerCfg",
        "apply_rope_sizes",
    )
    for path in sorted(VERSIONS.glob("v*/**/*.py")):
        if "v16" in path.parts:
            continue
        text = path.read_text()
        for name in names:
            assert name not in text, (path, name)
