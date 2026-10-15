"""TensorBoard のスカラーを、run x iteration の表にする (Isaac Sim 不要)。

    uv run python src/isaac_twisted_rl/scripts/rsl_rl/tb_table.py \
        --run v12_s44=logs/rsl_rl/knot_tying_v12/<run> --run v13_s44=<run> \
        --tags Curriculum/stage_log/timeout_s0,Loss/value_function --its 70,100,150,200,300 --window=-4,0

## なぜ作ったか

学習ログの比較は代理の指標で、版の良し悪しの結論には使わない (`docs/system/evaluation.md`) が、崩れの時期や
学習の進み具合を並べるときに、run と iteration を揃えた表が要る。1 iteration の値は揺れるので、各 iteration の
前後の窓の平均を出す。

* `--its`: 表の列にする iteration。
* `--window LO,HI`: 窓の幅。iteration `it` について `it+LO` から `it+HI` までの平均 (両端を含む)。
  既定は `-5,5`。学習の途中の様子を「その iteration までの直近」で見たいときは `-4,0`。
* run が無いタグ (v12 以前の `Loss/bounds` など) は `-` を出す。run の最後の iteration より先の列も `-`。
* `--out`: 表の値を JSON にも書く。
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def window_mean(series: dict[int, float], it: int, lo: int, hi: int) -> float | None:
    """`it+lo`〜`it+hi` に値のある点の平均。1 つも無ければ None。"""
    vals = [series[s] for s in range(it + lo, it + hi + 1) if s in series]
    return float(np.mean(vals)) if vals else None


def main() -> None:
    parser = argparse.ArgumentParser(description="TensorBoard scalars as a run x iteration table.")
    parser.add_argument("--run", action="append", required=True, help="name=<run dir>")
    parser.add_argument("--tags", required=True, help="Comma-separated scalar tags.")
    parser.add_argument("--its", required=True, help="Comma-separated iterations (columns).")
    parser.add_argument("--window", default="-5,5", help="LO,HI offsets around each iteration (use --window=-4,0).")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    tags = args.tags.split(",")
    its = [int(x) for x in args.its.split(",")]
    lo, hi = (int(x) for x in args.window.split(","))
    runs = dict(spec.split("=", 1) for spec in args.run)
    data: dict[str, dict[str, dict[int, float]]] = {}
    for name, path in runs.items():
        ea = EventAccumulator(path, size_guidance={"scalars": 0})
        ea.Reload()
        have = set(ea.Tags()["scalars"])
        data[name] = {t: {e.step: e.value for e in ea.Scalars(t)} for t in tags if t in have}

    table: dict[str, dict[str, dict[int, float | None]]] = {}
    for t in tags:
        print(f"\n## {t}  (mean over it{lo:+d}..it{hi:+d})")
        print(f"{'run':16}" + "".join(f"{i:>9}" for i in its))
        table[t] = {}
        for name in runs:
            series = data[name].get(t)
            row = {i: (window_mean(series, i, lo, hi) if series else None) for i in its}
            table[t][name] = row
            print(f"{name:16}" + "".join(f"{v:9.3f}" if v is not None else f"{'-':>9}" for v in row.values()))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"runs": runs, "window": [lo, hi], "table": table}, f, indent=1)


if __name__ == "__main__":
    main()
