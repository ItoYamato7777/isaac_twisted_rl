"""剛体鎖 (PhysX Articulation) の片持ち梁の試験。曲げ剛性が名目値どおりかを heavy elastica と比べる (Kit が要る)。

    .venv/bin/python -u src/isaac_twisted_rl/tasks/knot_tying/rope_study/bench_cantilever_art.py \
        --headless --device cuda:0 --out outputs/rope_study/cantilever/art.json

`bench_cantilever_xpbd.py` (XPBD、Kit 不要) と同じ試験を、v3〜v7 の紐 (`knot` = `Art28`) と
`knot56` (= `Art56`) と同じ作り方の剛体鎖で行う。根元のリンク (Link00) を `fix_root_link=True` で
水平に固定し、残りのリンクを自重で垂らして `--seconds` 後の先端 (最後のリンクの軸の端) の下がりを、
`elastica.heavy_cantilever` (連続体) と `elastica.discrete_chain_drop` (関節ごとにばね EI / l を持つ
離散鎖の小たわみ) と比べる。

| 鎖 | リンク長 | 自由なリンク | 自由長 |
|---|---|---|---|
| `knot` (Art28) | 4 cm | 2 / 4 | 8 / 16 cm |
| `knot56` (Art56) | 2 cm | 4 / 8 | 8 / 16 cm |

自由長 = 固定したリンクの先の関節から先端まで (XPBD の試験の「区間 0 を固定」と同じ数え方)。
すべての鎖を 1 つのステージに並べて同時に回す。床は置かず、接触は切る (曲げだけを見る)。
ソルバの位置反復を `--iterations` で振る (v4 は 8)。曲げの角度制限 (最小曲げ半径 4 cm: 4 cm リンクで
±60°、2 cm リンクで ±29°) に当たったかも出す (自由長 16 cm では根元の関節が制限に当たりうる)。

出力: `--out` の JSON。USD は `outputs/rope_study/cantilever/usd/` に作る (`common/rope_model/data/` には書かない)。
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import pathlib
import sys

from isaaclab.app import AppLauncher

_HERE = pathlib.Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[4]

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--seconds", type=float, default=3.0)
parser.add_argument("--iterations", type=str, default="8,16,32", help="ソルバの位置反復 (カンマ区切り)。")
parser.add_argument("--out", type=str, default=str(_REPO_ROOT / "outputs" / "rope_study" / "cantilever" / "art.json"))
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg  # noqa: E402

from isaac_twisted_rl.tasks.knot_tying.common.rope_backends import centerline as cl  # noqa: E402
from isaac_twisted_rl.tasks.knot_tying.common.rope_model import rope_specs as rs  # noqa: E402
from isaac_twisted_rl.tasks.knot_tying.common.rope_model.generate_rope_usd import build_stage  # noqa: E402
from isaac_twisted_rl.tasks.knot_tying.common.rope_model.rope_cfg import MAX_ANGULAR_VELOCITY_DEG  # noqa: E402

sys.path.insert(0, str(_HERE))
from elastica import discrete_chain_drop, effective_ei, heavy_cantilever  # noqa: E402

DT = 1.0 / 240.0
CASES = (("knot", 2), ("knot", 4), ("knot56", 4), ("knot56", 8))
"""(プリセット, 自由なリンクの数)。"""


def _chain_spec(base: str, n_free: int) -> rs.RopeSpec:
    """プリセットと同じ材料・リンクで、本数だけを「固定するリンク 1 本 + 自由なリンク n_free 本」にした鎖。"""
    spec = rs.get_spec(base)
    return dataclasses.replace(spec, name=f"cant_{base}_{n_free}", num_links=n_free + 1)


def _make_cfg(spec: rs.RopeSpec, usd_path: str, prim_path: str, pos, iterations: int) -> ArticulationCfg:
    """`rope_cfg.make_rope_cfg` と同じ剛体・ソルバの設定で、根元を固定し接触を切ったもの。"""
    return ArticulationCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                disable_gravity=False,
                max_linear_velocity=100.0,
                max_angular_velocity=MAX_ANGULAR_VELOCITY_DEG,
                enable_gyroscopic_forces=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=iterations,
                solver_velocity_iteration_count=1,
                fix_root_link=True,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=pos),
        actuators={},
    )


def _axis_end(c: torch.Tensor, q: torch.Tensor, half: float) -> torch.Tensor:
    """リンク中心 `c` (3,) と向き `q` (wxyz) から、カプセルの軸 (局所 x) の +側の端。"""
    ex = torch.tensor([half, 0.0, 0.0], device=c.device, dtype=c.dtype)
    return c + cl.quat_rotate(q.view(1, 4), ex.view(1, 3)).view(3)


def main():
    usd_dir = _REPO_ROOT / "outputs" / "rope_study" / "cantilever" / "usd"
    usd_dir.mkdir(parents=True, exist_ok=True)
    iterations = [int(v) for v in args_cli.iterations.split(",") if v]

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=DT, device=args_cli.device))
    runs = []
    row = 0
    for base, n_free in CASES:
        spec = _chain_spec(base, n_free)
        usd_path = str(usd_dir / f"rope_{spec.name}.usd")
        build_stage(spec).Export(usd_path)
        for it in iterations:
            prim = f"/World/Cant_{base}_{n_free}_it{it}"
            cfg = _make_cfg(spec, usd_path, prim, (0.0, 0.3 * row, 1.0), it)
            runs.append({"spec": spec, "base": base, "n_free": n_free, "iterations": it, "art": Articulation(cfg)})
            row += 1

    sim.reset()
    for r in runs:
        names = [f"Link{k:02d}" for k in range(r["spec"].num_links)]
        r["ids"], _ = r["art"].find_bodies(names, preserve_order=True)
        r["root0"] = r["art"].data.body_pos_w[0, r["ids"][0]].clone()
    steps = int(round(args_cli.seconds / DT))
    for _ in range(steps):
        sim.step(render=False)
        for r in runs:
            r["art"].update(DT)

    rows = []
    for r in runs:
        spec, art = r["spec"], r["art"]
        seg = spec.link_spacing
        half = 0.5 * seg
        ei = spec.bend_stiffness * seg
        w = spec.link_mass / seg * 9.81
        free_len = r["n_free"] * seg
        c = art.data.body_pos_w[0, r["ids"]]
        q = art.data.body_quat_w[0, r["ids"]]
        joint0 = _axis_end(c[0], q[0], half)
        tip = _axis_end(c[-1], q[-1], half)
        drop = float(joint0[2] - tip[2])
        ref = heavy_cantilever(free_len, ei, w)
        jmax = float(art.data.joint_pos[0].abs().max()) if art.num_joints > 0 else 0.0
        out = {
            "preset": r["base"],
            "seg_len": seg,
            "free_len": free_len,
            "iterations": r["iterations"],
            "tip_drop": drop,
            "ref_tip_drop": ref["tip_drop"],
            "rel_err": drop / ref["tip_drop"] - 1.0,
            "effective_EI_ratio": effective_ei(free_len, w, drop, ei) / ei,
            "discrete_small_deflection": discrete_chain_drop(free_len, seg, ei, w),
            "continuum_small_deflection": ref["small_deflection_drop"],
            "max_joint_angle_deg": math.degrees(jmax),
            "bend_limit_deg": spec.bend_limit_deg,
            "tip_speed": float(art.data.body_lin_vel_w[0, r["ids"][-1]].norm()),
            "root_drift": float((c[0] - r["root0"]).norm()),
        }
        rows.append(out)
        print(
            f"{r['base']:7s} seg={seg * 100:.0f}cm L={free_len * 100:.0f}cm it={r['iterations']:2d} "
            f"drop={drop * 1000:6.1f} mm ref={ref['tip_drop'] * 1000:6.1f} mm err={out['rel_err'] * 100:+6.1f}% "
            f"EI_eff/EI={out['effective_EI_ratio']:.2f} (small-defl: discrete {out['discrete_small_deflection'] * 1000:.1f} / "
            f"continuum {out['continuum_small_deflection'] * 1000:.1f} mm) joint_max={out['max_joint_angle_deg']:.1f}deg "
            f"(limit {spec.bend_limit_deg:.0f}) v_tip={out['tip_speed']:.3f}",
            flush=True,
        )
    os.makedirs(os.path.dirname(os.path.abspath(args_cli.out)), exist_ok=True)
    with open(args_cli.out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"[cantilever] wrote {args_cli.out}")


if __name__ == "__main__":
    main()
    simulation_app.close()
