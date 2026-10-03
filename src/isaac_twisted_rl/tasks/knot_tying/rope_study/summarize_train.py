"""紐モデルの比較の学習ログ (`outputs/rope_study/logs/train_<モデル>.log`) を表にする (Isaac Sim 不要)。

    .venv/bin/python src/isaac_twisted_rl/tasks/knot_tying/rope_study/summarize_train.py [--md out.md] [--logs 他のログ ...]

iteration ごとに rsl-rl が出す値 (1 iteration の時間、手/s、エピソードの終わり方、段ごとの成功率、`chain_est`) を読み、
iteration の区間ごとの平均を出す。`chain_est` (段ごとの成功率の積) は本番の chain 評価の成功率を表さない
(v6 レポート §4.5) ので、学習の進み方の目安にだけ使う。
"""

from __future__ import annotations

import argparse
import glob
import os
import re

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), *[".."] * 5))
LOGS = os.path.join(REPO, "outputs", "rope_study", "logs")
_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_KEYS = {
    "Iteration time": "iter_s",
    "Episode_Termination/diverged": "diverged",
    "Episode_Termination/stage_up": "stage_up",
    "Curriculum/stage_log/success_s0": "succ_s0",
    "Curriculum/stage_log/success_s1": "succ_s1",
    "Curriculum/stage_log/success_s2": "succ_s2",
    "Curriculum/stage_log/chain_est": "chain_est",
}


def parse(path: str) -> list[dict]:
    rows: list[dict] = []
    cur: dict | None = None
    with open(path, errors="replace") as f:
        for raw in f:
            line = _ANSI.sub("", raw).strip()
            m = re.match(r"Learning iteration (\d+)/(\d+)", line)
            if m:
                cur = {"it": int(m.group(1))}
                rows.append(cur)
                continue
            if cur is None:
                continue
            m = re.match(r"Computation: (\d+) steps/s", line)
            if m:
                cur["steps_s"] = float(m.group(1))
                continue
            if ":" in line:
                key, _, val = line.partition(":")
                key = key.strip()
                if key in _KEYS:
                    try:
                        cur[_KEYS[key]] = float(val.strip().rstrip("s"))
                    except ValueError:
                        pass
    return rows


def mean(rows: list[dict], key: str, lo: int, hi: int) -> float | None:
    vals = [r[key] for r in rows if lo <= r["it"] < hi and key in r]
    return sum(vals) / len(vals) if vals else None


def fmt(v: float | None, pct: bool = True) -> str:
    if v is None:
        return "—"
    return f"{100 * v:.1f}%" if pct else f"{v:.0f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", default="")
    ap.add_argument("--logs", nargs="*", default=None)
    a = ap.parse_args()
    paths = a.logs or sorted(glob.glob(os.path.join(LOGS, "train_*.log")))
    lines = [
        "| モデル | 済んだ iteration | 1 iteration の秒 (中央値) | 手/s | 発散 (it 0-9 / 40-49 / 90-99) | "
        "段の成功 s0 / s1 / s2 (it 90-99) | chain_est (it 40-49 / 90-99) |",
        "|---|---|---|---|---|---|---|",
    ]
    for path in paths:
        rows = parse(path)
        if not rows:
            continue
        name = os.path.basename(path)[len("train_") : -len(".log")]
        its = sorted(r["iter_s"] for r in rows if "iter_s" in r)
        med = its[len(its) // 2] if its else None
        sps = sorted(r["steps_s"] for r in rows if "steps_s" in r)
        div = " / ".join(fmt(mean(rows, "diverged", lo, lo + 10)) for lo in (0, 40, 90))
        succ = " / ".join(fmt(mean(rows, f"succ_s{k}", 90, 100)) for k in range(3))
        chain = " / ".join(fmt(mean(rows, "chain_est", lo, lo + 10)) for lo in (40, 90))
        lines.append(
            f"| {name} | {rows[-1]['it'] + 1} | {fmt(med, False)} | {fmt(sps[len(sps) // 2] if sps else None, False)} | "
            f"{div} | {succ} | {chain} |"
        )
    text = "\n".join(lines)
    print(text)
    if a.md:
        with open(a.md, "w") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
