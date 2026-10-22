"""8 の字結び・もやい結びを、Isaac Sim なしで p-data のグラフの上に載せて測る (0033 の §3・§4)。

    uv run --with snappy==3.2 --with topoly==1.1.0 python \
        src/isaac_twisted_rl/scripts/knot_transfer/measure_abstract.py --out outputs/knot_transfer/abstract.json

測るもの:

* `curves`: 3D の曲線 (固結び・8 の字結び・もやい結びと、その鏡像・逆向き) から `state_2_topology` で作った p-data、
  閉じ方ごとの型 (SnapPy の HFK)、Topoly (3D の座標から直接、2 点を無限遠へ伸ばす閉じ方を 100 回)、今の判定
  (`is_overhand`・R2 除去後の固結び = `knot_graph.is_goal`・段)、ロープ `Knot` (直径 2 cm・リンク 4 cm) で
  作れるかの目安 (非隣接の紐の最小の間隔を直径にしたときの長さと、4 cm のリンクに取り直しても p-data が残るか)。
* `search`: その p-data を目標にした、交差を間引いた図だけを広げる BFS (`knotlib.pruned_search`) の最短手数と経路。
* `graph_le4`: 交差 4 個までの全グラフで、今のゴール (`is_goal` = 172 個) と、閉じた結び目の型 (両方の閉じ方で 3_1)
  の食い違いを数える。
* `twisted_rl_sets`: TWISTED-RL の評価のゴール (`~/work/twisted_rl/system_flow/metrics/states.py`) の型。
* `key_bits`: 今の表のキー (`p_data_key`) が入る交差数の上限。
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
from pathlib import Path

import knotlib as K
import numpy as np
import torch

ROPE_DIAMETER = 0.02
LINK_LENGTH = 0.04
ROPE_LENGTH = 1.14
"""ロープ `Knot` (`docs/decisions/0013-rope-knot.md`)。"""


def _current_judgement(pd: str) -> dict:
    """今の判定 (固結び専用) を p-data に当てる。"""
    batch = K.tp.IntersectionBatch.from_lists([K.kg.p_data_to_lists(pd)])
    red = K.tp.reduce_r2(batch)
    n = len(pd.split("_")) // 2 if pd else 0
    out = {
        "is_overhand_raw": bool(K.tp.is_overhand(batch)[0]),
        "r2_goal": bool(K.kg.is_goal([pd])[0]),
        "r2_removed": int(K.tp.r2_removed_count(batch, red)[0]),
        "ladder_class": int(K.tp.topology_class(red)[0]),
    }
    if n <= K.kg.MAX_GRAPH_CROSSINGS:
        out["stage_bfs"] = int(K.kg.get_bfs_potential("cpu").stage(batch, red)[0])
        out["in_bfs_table"] = pd in K.kg.distance_to_overhand()
    else:
        out["stage_bfs"] = None
        out["in_bfs_table"] = False
    return out


def _rope_fit(curve: np.ndarray, pd: str) -> dict:
    """直径 2 cm のロープで作れるか: 非隣接の紐の間隔を直径に合わせて縮めたときの長さと、4 cm のリンクでの p-data。"""
    skip = 3.0 * ROPE_DIAMETER
    scale = 1.0
    for _ in range(30):  # 間隔は縮尺に比例するが、`skip` は弧長なので数回で収束させる
        c = curve * scale
        clear = K.min_clearance(c, skip)
        scale *= ROPE_DIAMETER / clear
    c = curve * scale
    length = K.arc_length(c)
    links = max(2, round(length / LINK_LENGTH))
    coarse = K.resample(c, links + 1)
    return {
        "length_at_clearance_m": round(length, 3),
        "links_at_4cm": links,
        "fits_knot_rope_length": length <= ROPE_LENGTH,
        "p_data_kept_at_4cm": K.p_data_of(coarse) == pd,
        "p_data_at_4cm": K.p_data_of(coarse),
    }


def _topoly(curve: np.ndarray) -> dict:
    import topoly

    res = topoly.alexander(curve.tolist(), closure=topoly.Closure.TWO_POINTS, tries=100)
    return {str(k): round(float(v), 3) for k, v in (res.items() if isinstance(res, dict) else [(res, 1.0)])}


def measure_curves(with_topoly: bool) -> dict:
    base = {
        "overhand": K.overhand_curve(),
        "figure8": K.figure8_curve(),
        "bowline": K.bowline_curve(),
        "bowline_collar_front": K.bowline_curve("collar_front"),
    }
    out = {}
    for name, c in base.items():
        for tag, cc in (("", c), ("_mirror", K.mirror(c)), ("_reverse", K.reverse(c))):
            pd = K.p_data_of(cc)
            over, under, same = K.robust_type(pd)
            row = {
                "p_data": pd,
                "crossings": len(K.crossings_of(pd)),
                "ends_same_face": same,
                "type_over_closure": over,
                "type_under_closure": under,
                "current": _current_judgement(pd),
                "rope": _rope_fit(cc, pd),
            }
            if with_topoly:
                row["topoly_two_points"] = _topoly(cc)
            out[name + tag] = row
            print(name + tag, row["crossings"], over, under, flush=True)
    return out


def measure_search(curves: dict) -> dict:
    out = {}
    for name, row in curves.items():
        if name.endswith(("_mirror", "_reverse")) and not name.startswith("bowline"):
            continue
        r = K.pruned_search(row["p_data"])
        out[name] = r
        print("search", name, r["distance"], r["moves"], r["states_visited"], flush=True)
    return out


def measure_graph_le4() -> dict:
    """交差 4 個までの全 p-data で、今のゴールと閉じた結び目の型を突き合わせる。"""
    dist = K.kg.distance_to_overhand()
    p_data = sorted(set(K.kg.enumerate_graph()[0]))
    goal = dict(zip(p_data, K.kg.is_goal(p_data)))
    conf: collections.Counter = collections.Counter()
    examples: dict[str, list[str]] = collections.defaultdict(list)
    for pd in p_data:
        over, under, _ = K.robust_type(pd)
        t = "robust_3_1" if over == under == "3_1" else ("one_closure_3_1" if "3_1" in (over, under) else "no_3_1")
        key = f"r2_goal={goal[pd]}|{t}"
        conf[key] += 1
        if len(examples[key]) < 5:
            examples[key].append(f"{pd} ({over}/{under}, d={dist.get(pd)})")
    return {"confusion": dict(conf), "examples": dict(examples), "num_p_data": len(p_data)}


def _states_py(path: Path) -> dict[str, list[str]]:
    src = path.read_text()
    out = {}
    for name in ("EASY_STATES", "MEDIUM_STATES", "HARD_STATES", "EVAL_3_RANDOM_BINS_STATES", "EVAL_4_EASY_800"):
        m = re.search(name + r" = \[(.*?)\n\]", src, flags=re.S)
        items = re.findall(r"'([^']*)'", m.group(1))
        out[name] = [s.replace("\\n", "_").replace(" ", "").replace(":", "") for s in items]
    return out


def measure_twisted_rl(path: Path) -> dict:
    out = {}
    for name, pds in _states_py(path).items():
        rows = []
        for pd in pds:
            over, under, same = K.robust_type(pd)
            rows.append(
                {
                    "p_data": pd,
                    "type": over if over == under else f"{over}|{under}",
                    "ends_same_face": same,
                    "r2_goal": bool(K.kg.is_goal([pd])[0]),
                }
            )
        out[name] = {"types": dict(collections.Counter(r["type"] for r in rows)), "rows": rows}
        print(name, out[name]["types"], flush=True)
    return out


def fig1_paths() -> dict:
    """TWISTED-RL 図 1 の手の列 (R1 → cross → cross と R1 → cross → cross → cross) で、着ける p-data の型を数える。"""
    states, moves = K.kg._enumerate_moves(K.kg.MAX_GRAPH_CROSSINGS)
    out = {}
    for seq in (("R1", "cross", "cross"), ("R1", "cross", "cross", "cross"), ("R1", "R1", "cross", "cross")):
        frontier = {0}
        for name in seq:
            kind = K.kg.MOVE_NAMES.index(name)
            frontier = {j for i in frontier for k, _, j in moves[i] if k == kind}
        tally = collections.Counter()
        for i in frontier:
            over, under, _ = K.robust_type(states[i].short_p_data)
            tally[over if over == under else f"{over}|{under}"] += 1
        out[" -> ".join(seq)] = dict(tally.most_common())
        print("fig1", seq, out[" -> ".join(seq)], flush=True)
    return out


def key_bits() -> dict:
    """`p_data_key` = 交差数 3 bit + 点ごとに (相手 4 bit + 上 1 + 符号 1)。int64 (符号ありで 63 bit) に入る上限。"""
    rows = {c: 3 + 6 * 2 * c for c in range(1, 9)}
    return {
        "bits_by_crossings": rows,
        "max_crossings_in_int64": max(c for c, b in rows.items() if b <= 63),
        "partner_field_limit_points": 15,
        "count_field_limit": 7,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--twisted_rl_states", default=str(Path.home() / "work/twisted_rl/system_flow/metrics/states.py"))
    ap.add_argument("--no_topoly", action="store_true")
    args = ap.parse_args()
    torch.set_default_dtype(torch.float64)
    curves = measure_curves(not args.no_topoly)
    out = {
        "curves": curves,
        "search": measure_search(curves),
        "graph_le4": measure_graph_le4(),
        "twisted_rl_sets": measure_twisted_rl(Path(args.twisted_rl_states)),
        "key_bits": key_bits(),
        "fig1_move_sequences": fig1_paths(),
        "graph_sizes_known": K.GRAPH_SIZES_KNOWN,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
