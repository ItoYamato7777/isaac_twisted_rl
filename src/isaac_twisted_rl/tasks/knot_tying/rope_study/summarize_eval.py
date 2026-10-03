"""紐モデルの比較の評価結果 (`outputs/rope_study/eval/<方策>/<評価>.json`) を表にする (Isaac Sim 不要)。

    .venv/bin/python src/isaac_twisted_rl/tasks/knot_tying/rope_study/summarize_eval.py [--md out.md]

表 1: 自分のモデルでの chain 評価 (固結び率 ± 標準誤差、発散、時間切れ、段 2 以上、手数)
表 2: 転移行列 (行 = 方策、列 = 評価したモデル、値 = 固結び率)、平均と最悪値
表 3: 摂動 (観測ノイズ・実行誤差) の下での固結び率
表 4: 選定に使う量 (報告書 §6.4): 他のモデルでの平均・最悪、全ての「学習と違う条件」(他のモデル・物理の摂動・
      観測ノイズ・実行誤差) での平均・最悪。平均の標準誤差は各評価の二項の標準誤差から (独立とみなす)
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), *[".."] * 5))
EVAL = os.path.join(REPO, "outputs", "rope_study", "eval")
MODELS = ("Art28", "Art28LowMu", "XpbdC28", "XpbdC28LowMu", "XpbdC56", "XpbdMS28")
"""転移の列 (学習に使うモデル)。これ以外の `on_<名前>` は物理の摂動 (`make_eval_jobs.PERTURB_TARGETS`)。"""
HOME = {"v4Knot": "Art28", "XpbdC28_s43": "XpbdC28", "XpbdC56_s43": "XpbdC56"}
"""方策の名前がモデルの名前でないものの「自分のモデル」。"""


def load() -> dict[str, dict[str, dict]]:
    out: dict[str, dict[str, dict]] = {}
    for path in sorted(glob.glob(os.path.join(EVAL, "*", "*.json"))):
        pol = os.path.basename(os.path.dirname(path))
        tag = os.path.splitext(os.path.basename(path))[0]
        with open(path) as f:
            out.setdefault(pol, {})[tag] = json.load(f)
    return out


def pct(d: dict | None, key: str = "knot") -> str:
    if d is None or d.get(key) is None:
        return "—"
    return f"{100 * d[key]:.1f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", default="")
    a = ap.parse_args()
    res = load()
    lines = []
    lines.append("### 自分のモデルでの chain 評価\n")
    lines.append("| 方策 | iteration | 固結び | 発散 | 時間切れ | 段 2 以上 | 固結びまでの手数 | 環境数 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for pol, r in res.items():
        for tag in sorted(t for t in r if t.startswith("self_it")):
            d = r[tag]
            it = tag.split("_it")[-1]
            se = f" ± {100 * d['knot_se']:.1f}" if d.get("knot_se") is not None else ""
            steps = f"{d['mean_actions_to_knot']:.1f}" if d.get("mean_actions_to_knot") else "—"
            lines.append(
                f"| {pol} | {it} | {pct(d)}{se} | {pct(d, 'diverged')} | {pct(d, 'time_out')} | "
                f"{pct(d, 'max_stage_ge2')} | {steps} | {d.get('episodes')} |"
            )
    targets = sorted({t.split("_it")[0][3:] for r in res.values() for t in r if t.startswith("on_")})
    model_cols = [m for m in MODELS if m in targets or any(HOME.get(p, p) == m for p in res)]
    pert_cols = [t for t in targets if t not in MODELS]
    cols = model_cols + pert_cols

    def it_of(tag: str) -> int:
        return int(tag.split("_it")[-1])

    def self_its(r: dict) -> list[int]:
        return sorted(it_of(t) for t in r if t.startswith("self_it"))

    def best_it(r: dict) -> int | None:
        """自分のモデルでの固結び率が最も高い iteration (同じなら後の方)。他のモデルの結果は使わない。"""
        its = self_its(r)
        return max(its, key=lambda i: (r[f"self_it{i}"]["knot"], i)) if its else None

    def agg(ds: list[dict]) -> str:
        if not ds:
            return "—"
        m = sum(d["knot"] for d in ds) / len(ds)
        se = math.sqrt(sum(d.get("knot_se", 0.0) ** 2 for d in ds)) / len(ds)
        return f"{100 * m:.1f} ± {100 * se:.1f}% ({len(ds)})"

    def worst(ds: list[dict]) -> str:
        if not ds:
            return "—"
        d = min(ds, key=lambda d: d["knot"])
        return f"{100 * d['knot']:.1f} ± {100 * d.get('knot_se', 0.0):.1f}%"

    def tables(title: str, pick) -> None:
        """pick(r) -> iteration。その iteration の転移行列と選定の表。"""
        lines.append(f"\n### 転移行列 ({title}。固結び率。行 = 方策、列 = 評価したモデル。括弧は自分のモデル)\n")
        lines.append("| 方策 | it | " + " | ".join(cols) + " |")
        lines.append("|---|---|" + "---|" * len(cols))
        sel_rows = []
        for pol, r in res.items():
            it = pick(r)
            if it is None:
                continue
            home = HOME.get(pol, pol)
            cells = []
            for c in cols:
                if c == home:
                    cells.append(f"({pct(r.get(f'self_it{it}'))})")
                else:
                    cells.append(pct(r.get(f"on_{c}_it{it}")))
            lines.append(f"| {pol} | {it} | " + " | ".join(cells) + " |")
            others = [d for c in model_cols if c != home and (d := r.get(f"on_{c}_it{it}")) is not None]
            allc = others + [d for c in pert_cols if (d := r.get(f"on_{c}_it{it}")) is not None]
            allc += [r[t] for t in r if t.startswith(("obs", "exec")) and it_of(t) == it]
            s_ = r.get(f"self_it{it}")
            se = f" ± {100 * s_['knot_se']:.1f}" if s_ and s_.get("knot_se") is not None else ""
            sel_rows.append(f"| {pol} | {it} | {pct(s_)}{se} | {agg(others)} | {worst(others)} | {agg(allc)} | {worst(allc)} |")
        lines.append(f"\n### 選定に使う量 ({title}。固結び率。平均 ± 標準誤差 (評価の数))\n")
        lines.append("| 方策 | it | 自分のモデル | 他のモデル: 平均 | 他のモデル: 最悪 | 学習と違う全条件: 平均 | 全条件: 最悪 |")
        lines.append("|---|---|---|---|---|---|---|")
        lines.extend(sel_rows)

    if targets:
        tables("各方策の最後の iteration", lambda r: self_its(r)[-1] if self_its(r) else None)
        tables("各方策の自分のモデルで良い方の iteration", best_it)
    lines.append("\n### 摂動 (自分のモデル)\n")
    lines.append("| 方策 | iteration | なし | 観測ノイズ 5 mm | 10 mm | 実行誤差 5 mm | 10 mm |")
    lines.append("|---|---|---|---|---|---|---|")
    for pol, r in res.items():
        its = sorted({int(t.split("_it")[-1]) for t in r if t.startswith(("obs", "exec"))})
        for it in its:
            g = lambda k: r.get(f"{k}_it{it}")  # noqa: E731
            lines.append(
                f"| {pol} | {it} | {pct(g('self'))} | {pct(g('obs5mm'))} | {pct(g('obs10mm'))} | "
                f"{pct(g('exec5mm'))} | {pct(g('exec10mm'))} |"
            )
    text = "\n".join(lines)
    print(text)
    if a.md:
        with open(a.md, "w") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
