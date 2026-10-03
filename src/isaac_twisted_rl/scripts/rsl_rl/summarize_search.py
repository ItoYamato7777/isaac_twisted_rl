"""`eval_search.py` / `eval_chain.py` の JSON を読み、版ごとの比較表 (Markdown) を出す。Isaac Sim は要らない。

    uv run python src/isaac_twisted_rl/scripts/rsl_rl/summarize_search.py \
        v4=logs/rsl_rl/knot_tying_v4/<run>/model_100.pt v6s42=logs/rsl_rl/knot_tying_v6/<run>/model_100.pt ...

引数は `ラベル=チェックポイント`。チェックポイントのフォルダから次を読む (無いものは空欄):

* `eval_chain_it{N}.json` — (a) 決定論的方策の chain 評価 (`eval_chain.py`)
* `eval_search_rollouts_it{N}.json` — (a) 確率的な方策の chain 評価と (b) best-of-N
* `eval_search_search_it{N}.json` — (c) ビーム探索

`--tag` を付けると `eval_search_*_it{N}_{tag}.json` を読む。予算の列は (b) と (c) で共通のもの。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re


def _load(path: str) -> dict | None:
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def _pct(x: float | None) -> str:
    return "—" if x is None else f"{100.0 * x:.1f}%"


def _num(x: float | None, fmt: str = "{:.1f}") -> str:
    return "—" if x is None else fmt.format(x)


def _binom_se(p: float | None, n: int | None) -> str:
    if p is None or not n:
        return "—"
    return f"±{100.0 * math.sqrt(max(p * (1 - p), 0.0) / n):.1f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="label=checkpoint")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    suffix = f"_{args.tag}" if args.tag else ""

    rows = []
    for spec in args.runs:
        label, ckpt = spec.split("=", 1)
        it = re.search(r"model_(\d+)\.pt$", ckpt).group(1)
        d = os.path.dirname(ckpt)
        rows.append(
            {
                "label": label,
                "chain": _load(os.path.join(d, f"eval_chain_it{it}.json")),
                "roll": _load(os.path.join(d, f"eval_search_rollouts_it{it}{suffix}.json")),
                "search": _load(os.path.join(d, f"eval_search_search_it{it}{suffix}.json")),
            }
        )

    print("### (a) chain 評価 (まっすぐなロープから 18 手、各環境の最初の 1 エピソード)\n")
    print("| run | 決定論的: 固結び | 発散 | 確率的: 固結び | 発散 | 時間切れ | エピソード |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        c, ro = r["chain"], (r["roll"] or {}).get("chain")
        print(
            f"| {r['label']} | {_pct(c and c['knot'])} | {_pct(c and c['diverged'])} | "
            f"{_pct(ro and ro['knot'])} {_binom_se(ro and ro['knot'], ro and ro['episodes'])} | "
            f"{_pct(ro and ro['diverged'])} | {_pct(ro and ro['time_out'])} | {ro['episodes'] if ro else '—'} |"
        )

    budgets = None
    for r in rows:
        for key in ("search", "roll"):
            src = r[key]
            if src:
                b = src["summary"]["budgets"] if key == "search" else src["best_of_n"]["budgets"]
                budgets = b if budgets is None else [x for x in budgets if x in b]
    budgets = budgets or []

    print("\n### (b) best-of-N と (c) ビーム探索: 予算 (問題あたりのシミュレートした手数) ごとの成功率\n")
    head = " | ".join(str(b) for b in budgets)
    print(f"| run | 方式 | {head} | 問題数 |")
    print("|---|---|" + "---|" * len(budgets) + "---|")
    for r in rows:
        if r["roll"]:
            bon = r["roll"]["best_of_n"]
            vals = dict(zip(bon["budgets"], bon["success"]))
            cells = " | ".join(_pct(vals.get(b)) for b in budgets)
            print(f"| {r['label']} | (b) best-of-N | {cells} | ({r['roll']['pool']['episodes']} エピソードから計算) |")
        if r["search"]:
            sm = r["search"]["summary"]
            vals = dict(zip(sm["budgets"], sm["success"]))
            cells = " | ".join(_pct(vals.get(b)) for b in budgets)
            print(f"| {r['label']} | (c) ビーム探索 | {cells} | {sm['problems']} |")

    print("\n### (c) ビーム探索の中身\n")
    print(
        "| run | 解けた | 解けた時点の手数 (中央値) | 解の手数 (中央値 / 平均) | 候補: 発散 | 段が下がる | "
        "吊ったまま | 再開 (平均) | 複製の観測の不一致 | 1 問題の処理速度 [手/分] |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        s = r["search"]
        if not s:
            continue
        sm, cf, th = s["summary"], s["summary"]["candidate_frac"], s["throughput"]
        print(
            f"| {r['label']} | {_pct(sm['solved'])} {_binom_se(sm['solved'], sm['problems'])} | "
            f"{_num(sm['moves_at_solve_median'], '{:.0f}')} | "
            f"{_num(sm['depth_at_solve_median'], '{:.0f}')} / {_num(sm['depth_at_solve_mean'])} | "
            f"{_pct(cf['diverged'])} | {_pct(cf['down'])} | {_pct(cf['unsettled'])} | "
            f"{_num(sm['restarts_mean'])} | {_pct(s['clone_check']['obs_mismatch_frac'])} | "
            f"{_num(th['moves_per_minute_per_problem'], '{:.0f}')} |"
        )

    print("\n### (b) best-of-N の中身\n")
    print("| run | 1 本の固結び率 | 発散で終わる | 発散で失った手の割合 | 最初の固結びまでの手数の期待値 |")
    print("|---|---|---|---|---|")
    for r in rows:
        ro = r["roll"]
        if not ro:
            continue
        p = ro["pool"]
        print(
            f"| {r['label']} | {_pct(p['knot'])} | {_pct(p['diverged'])} | {_pct(p['moves_lost_to_divergence_frac'])} | "
            f"{_num(ro['best_of_n']['expected_moves_to_knot'])} |"
        )


if __name__ == "__main__":
    main()
