"""R1 / R2 / cross のグラフを交差 `--max_crossings` 個まで全列挙し、交差数ごとの状態の数と、閉じた結び目の型の内訳を数える。

`docs/decisions/0033-beyond-overhand-audit.md` §3 の「その交差数でのグラフの大きさ」。交差 4 個までは
`tests/test_knot_graph.py` が数を固定している (1 / 4 / 40 / 528 / 7968)。5 個は列挙に時間とメモリが要るので、
`run_job.sh` を使わず (Isaac Sim を使わない) 背景で回す::

    uv run --with snappy==3.2 python src/isaac_twisted_rl/scripts/knot_transfer/graph_size.py \
        --max_crossings 5 --out outputs/knot_transfer/graph_size_5.json

型は `knotlib.robust_type` (両端が別の面なら上を通す・下を通す両方の閉じ方)。
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import resource
import time

import knotlib as K


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_crossings", type=int, default=5)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no_types", action="store_true", help="型を数えない (列挙の大きさだけ)")
    args = ap.parse_args()

    t0 = time.time()
    states, dist, moves = K.forward_distances(args.max_crossings)
    t_enum = time.time() - t0
    by_c: collections.Counter = collections.Counter()
    p_by_c: dict[int, set] = collections.defaultdict(set)
    for s in states:
        by_c[s.pts // 2] += 1
        p_by_c[s.pts // 2].add(s.short_p_data)
    out = {
        "max_crossings": args.max_crossings,
        "enumerate_seconds": round(t_enum, 1),
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        "states_by_crossings": {str(c): by_c[c] for c in sorted(by_c)},
        "p_data_by_crossings": {str(c): len(p_by_c[c]) for c in sorted(p_by_c)},
        "edges": sum(len(m) for m in moves),
    }
    print(json.dumps(out), flush=True)
    if not args.no_types:
        t1 = time.time()
        first: dict[str, int] = {}
        for s, d in zip(states, dist):
            pd = s.short_p_data
            if pd not in first or d < first[pd]:
                first[pd] = d
        tally: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
        min_dist: dict[str, int] = {}
        for pd, d in first.items():
            over, under, same = K.robust_type(pd)
            c = len(pd.split("_")) // 2 if pd else 0
            key = over if over == under else f"{over}|{under}"
            tally[str(c)][key] += 1
            for t, cond in ((f"robust:{over}", over == under), (f"weak:{over}", True), (f"weak:{under}", True)):
                if cond and (t not in min_dist or d < min_dist[t]):
                    min_dist[t] = d
        out["types_by_crossings"] = {c: dict(v.most_common()) for c, v in sorted(tally.items())}
        out["min_moves_from_trivial"] = dict(sorted(min_dist.items()))
        out["type_seconds"] = round(time.time() - t1, 1)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
