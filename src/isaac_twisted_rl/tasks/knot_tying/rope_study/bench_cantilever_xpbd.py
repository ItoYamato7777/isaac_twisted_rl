"""XPBD の紐の片持ち梁の試験 (Isaac Sim 不要、Warp)。曲げ剛性が名目値どおりかを heavy elastica と比べる。

    uv run python src/isaac_twisted_rl/tasks/knot_tying/rope_study/bench_cantilever_xpbd.py [--device cpu]

節点 0 と 1 を固定 (区間 0 が水平な固定端)、残りの区間を自由にして 3 s 落ち着かせ、先端の下がりを
`elastica.heavy_cantilever` と比べる。区間長 4 / 2 / 1 cm、自由長 8 / 16 cm、Cosserat 棒と質点-ばね。
剛体鎖 (PhysX) の同じ試験は `bench_cantilever_art.py` (Kit が要る)。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pathlib
import sys
import types

import torch

_HERE = pathlib.Path(__file__).resolve().parent
_RB = _HERE.parent / "common" / "rope_backends"
_pkg = types.ModuleType("rope_backends")
_pkg.__path__ = [str(_RB)]
sys.modules.setdefault("rope_backends", _pkg)
for _name in ("centerline", "xpbd_rod"):
    _spec = importlib.util.spec_from_file_location(f"rope_backends.{_name}", _RB / f"{_name}.py")
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules[f"rope_backends.{_name}"] = _mod
    _spec.loader.exec_module(_mod)
xr = sys.modules["rope_backends.xpbd_rod"]
sys.path.insert(0, str(_HERE))
from elastica import discrete_chain_drop, effective_ei, heavy_cantilever  # noqa: E402

EI, RHO = 7.854e-4, 0.2513


def run(model: str, seg_len: float, free_len: float, device: str, seconds: float = 3.0, substeps: int = 16, **kw):

    n_free = int(round(free_len / seg_len))
    n_seg = n_free + 1
    p = xr.XpbdRodParams(
        model=model, num_segments=n_seg, length=n_seg * seg_len, radius=0.01, linear_density=RHO,
        bend_stiffness=EI, twist_stiffness=0.5 * EI, substeps=substeps, **kw,
    )
    rod = xr.XpbdRod(1, p, device=device)
    nodes = torch.zeros(1, p.num_nodes, 3, device=device)
    nodes[0, :, 0] = torch.arange(p.num_nodes, device=device) * seg_len
    nodes[0, :, 2] = 1.0
    rod.set_state(torch.tensor([0], device=device), nodes)
    rod.inv_m[0] = 0.0
    rod.inv_m[1] = 0.0
    hand = torch.zeros(1, 4, device=device)
    hand[:, 2] = 5.0
    rod.set_hand(hand, torch.zeros(1, 2, device=device))
    rod.sync_hand()
    for _ in range(int(seconds * 240)):
        rod.step(1.0 / 240.0)
    x = rod.nodes()[0]
    drop = float(1.0 - x[-1, 2])
    ref = heavy_cantilever(free_len, EI, RHO * 9.81)
    return {
        "model": model, "seg_len": seg_len, "free_len": free_len, "tip_drop": drop, "ref_tip_drop": ref["tip_drop"],
        "rel_err": drop / ref["tip_drop"] - 1.0, "effective_EI_ratio": effective_ei(free_len, RHO * 9.81, drop) / EI,
        "discrete_small_deflection": discrete_chain_drop(free_len, seg_len, EI, RHO * 9.81),
        "continuum_small_deflection": ref["small_deflection_drop"],
        "tip_speed": float(rod.node_vel()[0, -1].norm()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    import warp as wp

    wp.init()
    rows = []
    for model in ("cosserat", "mass_spring"):
        for seg in (0.04, 0.02, 0.01, 0.005):
            for L in (0.08, 0.16):
                r = run(model, seg, L, a.device)
                rows.append(r)
                print(
                    f"{model:11s} seg={seg * 100:.0f}cm L={L * 100:.0f}cm drop={r['tip_drop'] * 1000:6.1f} mm "
                    f"ref={r['ref_tip_drop'] * 1000:6.1f} mm err={r['rel_err'] * 100:+6.1f}% EI_eff/EI={r['effective_EI_ratio']:.2f} "
                    f"(small-defl: discrete {r['discrete_small_deflection'] * 1000:.1f} / continuum {r['continuum_small_deflection'] * 1000:.1f} mm) "
                    f"v_tip={r['tip_speed']:.3f}",
                    flush=True,
                )
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
