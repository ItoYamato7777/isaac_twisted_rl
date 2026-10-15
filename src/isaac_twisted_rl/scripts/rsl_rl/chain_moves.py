"""chain モードの手の列を復元して数える・段ごとの 1 手の結果から chain 評価を真似る (Isaac Sim 不要)。

    # 手の列の復元 (`diag_stage_moves.py --dump_obs` の出力)。--eval で同じチェックポイントの chain 評価と照合する
    uv run python src/isaac_twisted_rl/scripts/rsl_rl/chain_moves.py sequences \
        --dump v12_s44_it150=outputs/diag_stage_moves/obs/v12_s44_it150_det.pt \
        --eval v12_s44_it150=logs/rsl_rl/knot_tying_v12/<run>/eval_chain_it150.json --out <json>

    # マルコフ連鎖 (`diag_stage_moves.py` の JSON の段ごとの結果の割合) と反実仮想
    uv run python src/isaac_twisted_rl/scripts/rsl_rl/chain_moves.py markov \
        --diag v12_s43_it300=outputs/diag_stage_moves/v12_s43_it300_det.json \
        --eval v12_s43_it300=<eval_chain_it300.json> --out <json>

    # チェックポイントの間の揺れ (run ごとに iteration の順の eval JSON をカンマで並べる)
    uv run python src/isaac_twisted_rl/scripts/rsl_rl/chain_moves.py spread \
        --series v12_s43=<it70 の JSON>,<it100 の JSON>,... --out <json>

## なぜ作ったか

chain 評価の JSON は 512 エピソードの結末の割合だけで、「どの段で何手使ったか」「崩れた方策が何手目に段を
上げるか」は分からない。`diag_stage_moves.py --dump_obs` の保存は手ごとの段と結果を持つが、環境の番号を
持たないので、並びから環境ごとの列を戻す (`common/move_chain.py` の docstring)。

`markov` は、段ごとの 1 手の結果の割合だけで 18 手の固結びの割合が再現できるかを見て、再現できるなら、
どの段のどの確率を変えるとどれだけ上がるかを同じ模型で見積もる。模型の上の値は推測で、chain 評価ではない。

`spread` は、同じ run の隣り合うチェックポイントの `knot` の差を、二項の見積り (sqrt(2 p (1 − p) / n)) と並べる。
chain 評価は決定論的なので (`docs/experiments/2026-10-15_eval-chain-repro.md`)、この差は方策の変化による。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from pathlib import Path

# `isaac_twisted_rl.tasks` を通すと `import_packages` が全タスクを import するので、`common/` を sys.path に足して
# isaaclab を import しない `move_chain` だけを読む (`critic_stage_values.py` と同じ)。
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tasks" / "knot_tying" / "common"))

from move_chain import (  # noqa: E402
    OUTCOMES,
    episode_result,
    first_up_move,
    markov_absorb,
    position_histogram,
    split_episodes,
    transition_probs,
    visit_table,
)

BUDGETS = (18, 36, 54, 3000)
"""マルコフ連鎖の手数。18 は chain 評価、3000 は手数の上限が無いときの近似。"""


def _named(items: list[str] | None) -> dict[str, str]:
    out = {}
    for item in items or []:
        name, _, path = item.partition("=")
        if not path:
            raise SystemExit(f"name=path の形で渡す: {item}")
        out[name] = path
    return out


def _eval_summary(path: str) -> dict:
    d = json.load(open(path))
    return {k: d[k] for k in ("knot", "diverged", "time_out", "episodes")}


def cmd_sequences(args: argparse.Namespace) -> dict:
    import torch

    evals = _named(args.eval)
    result = {}
    for name, path in _named(args.dump).items():
        d = torch.load(path, map_location="cpu", weights_only=False)
        if tuple(d.get("outcomes") or OUTCOMES) != OUTCOMES:
            raise SystemExit(f"{path}: 結果の名前の並びが違う {d.get('outcomes')}")
        seqs = split_episodes(
            d["before"].tolist(), d["after"].tolist(), d["outcome"].tolist(), args.num_envs, args.max_moves
        )
        n = len(seqs)
        res = [episode_result(s) for s in seqs]
        first_up: dict[str, int] = {}
        for s in seqs:
            k = str(first_up_move(s))
            first_up[k] = first_up.get(k, 0) + 1
        entry = {
            "dump": path,
            "checkpoint": d.get("checkpoint"),
            "rows": len(d["before"]),
            "reconstructed": {r: res.count(r) / n for r in ("knot", "diverged", "time_out")},
            "first_up_move_stage0": dict(
                sorted(first_up.items(), key=lambda kv: (kv[0] == "None", int(kv[0]) if kv[0] != "None" else 0))
            ),
            "visits": {str(k): visit_table(seqs, k) for k in (0, 1, 2)},
            "up_position": {str(k): position_histogram(seqs, k, "up") for k in (0, 1, 2)},
            "down_position": {str(k): position_histogram(seqs, k, "down") for k in (1, 2)},
        }
        if name in evals:
            entry["eval"] = _eval_summary(evals[name])
        result[name] = entry
        r = entry["reconstructed"]
        print(
            f"== {name}: 行 {entry['rows']}  復元 knot {r['knot']:.3f} 発散 {r['diverged']:.3f} 時間切れ {r['time_out']:.3f}",
            end="",
        )
        if "eval" in entry:
            e = entry["eval"]
            print(f"  (chain 評価 {e['knot']:.3f} / {e['diverged']:.3f} / {e['time_out']:.3f})", end="")
        print(f"\n  段 0 を初めて上げた手: {entry['first_up_move_stage0']}")
        for k in ("0", "1", "2"):
            parts = [
                f"{kind}: 訪問 {v['visits']} 手 {v['moves']} up {v['up']:.2f} down {v['down']:.2f} stay {v['stay']:.2f} "
                f"訪問あたり上昇 {v['up_per_visit']:.2f}"
                for kind, v in entry["visits"][k].items()
            ]
            print(f"  段 {k}: " + " | ".join(parts))
            print(
                f"    上昇の位置 {entry['up_position'][k]}"
                + (f"  下降の位置 {entry['down_position'][k]}" if k != "0" else "")
            )
    return result


def _counterfactuals(p: dict[int, dict[str, float]]) -> dict[str, dict[int, dict[str, float]]]:
    """模型の上で確率を書き換える。どれも 1 手の確率の和を 1 に保つ。"""

    def edit(fn):
        q = {s: dict(v) for s, v in p.items()}
        fn(q)
        return q

    def move(q, s, frm, to, frac=1.0):
        x = q[s][frm] * frac
        q[s][frm] -= x
        q[s][to] += x

    def double_up2(q):
        q[2]["stay"] -= q[2]["up"]
        q[2]["up"] *= 2

    def stage1_like_it100(q):
        q[1]["stay"] += q[1]["up"] + q[1]["down"] - 0.55
        q[1]["up"], q[1]["down"] = 0.45, 0.10

    def stage2_script(q):
        q[2]["up"], q[2]["down"], q[2]["stay"] = 0.9, 0.05, 0.05 - q[2]["div"]

    return {
        "as_is": p,
        "no_divergence": edit(lambda q: [move(q, s, "div", "stay") for s in range(3)]),
        "stage2_half_down_to_stay": edit(lambda q: move(q, 2, "down", "stay", 0.5)),
        "stage2_all_down_to_stay": edit(lambda q: move(q, 2, "down", "stay")),
        "stage2_up_x2_from_stay": edit(double_up2),
        "stage2_half_down_to_up": edit(lambda q: move(q, 2, "down", "up", 0.5)),
        "stage1_up0.45_down0.10": edit(stage1_like_it100),
        "stage2_up0.9_down0.05": edit(stage2_script),
    }


def cmd_markov(args: argparse.Namespace) -> dict:
    evals = _named(args.eval)
    result = {}
    for name, path in _named(args.diag).items():
        d = json.load(open(path))
        p = transition_probs({int(s): d["by_stage_before"][str(s)]["outcome"] for s in (0, 1, 2)})
        entry = {
            "diag": path,
            "checkpoint": d.get("checkpoint"),
            "probs": {str(s): v for s, v in p.items()},
            "cases": {},
        }
        for case, q in _counterfactuals(p).items():
            entry["cases"][case] = {str(b): markov_absorb(q, b) for b in BUDGETS}
        if name in evals:
            entry["eval"] = _eval_summary(evals[name])
        result[name] = entry
        as_is = entry["cases"]["as_is"]["18"]
        real = entry.get("eval", {})
        print(
            f"== {name}: 模型 18 手 knot {as_is['knot']:.3f} 発散 {as_is['diverged']:.3f}"
            + (f"  (chain 評価 {real['knot']:.3f} / {real['diverged']:.3f})" if real else "")
        )
        for s in ("0", "1", "2"):
            v = entry["probs"][s]
            print(f"  段 {s}: up {v['up']:.3f} down {v['down']:.3f} stay {v['stay']:.3f} div {v['div']:.3f}")
        for case, by_b in entry["cases"].items():
            print(f"  {case:28s} " + "  ".join(f"{b}手 {by_b[str(b)]['knot']:.3f}" for b in BUDGETS))
    return result


def cmd_spread(args: argparse.Namespace) -> dict:
    series = {name: paths.split(",") for name, paths in _named(args.series).items()}
    vals_all, diffs = [], []
    result = {"series": {}}
    for name, paths in series.items():
        ks = [json.load(open(p))["knot"] for p in paths]
        vals_all += ks
        diffs += [abs(b - a) for a, b in zip(ks, ks[1:])]
        result["series"][name] = {"paths": paths, "knot": ks}
        print(f"{name:10s} " + " ".join(f"{k * 100:.1f}" for k in ks))
    p = statistics.mean(vals_all)
    sd_diff = math.sqrt(2 * p * (1 - p) / args.episodes)
    result |= {
        "points": len(vals_all),
        "mean_knot": p,
        "adjacent_abs_diff_mean": statistics.mean(diffs),
        "adjacent_abs_diff_median": statistics.median(diffs),
        "pairs": len(diffs),
        "binomial_abs_diff_expected": math.sqrt(2 / math.pi) * sd_diff,
    }
    print(
        f"{len(vals_all)} 点 (平均 {p * 100:.1f}%)。隣の差の絶対値: 平均 {result['adjacent_abs_diff_mean'] * 100:.1f}・"
        f"中央値 {result['adjacent_abs_diff_median'] * 100:.1f} ポイント ({len(diffs)} 組)。"
        f"二項の見積りの期待値 {result['binomial_abs_diff_expected'] * 100:.1f} ポイント"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Chain-mode move sequences and Markov model of chain evaluation.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sequences")
    s.add_argument("--dump", action="append", required=True, help="name=<diag_stage_moves --dump_obs の .pt>")
    s.add_argument("--eval", action="append", help="name=<同じチェックポイントの eval_chain_it*.json> (照合用)")
    s.add_argument("--num_envs", type=int, default=512)
    s.add_argument("--max_moves", type=int, default=18)
    m = sub.add_parser("markov")
    m.add_argument("--diag", action="append", required=True, help="name=<diag_stage_moves の JSON>")
    m.add_argument("--eval", action="append", help="name=<同じチェックポイントの eval_chain_it*.json>")
    p = sub.add_parser("spread")
    p.add_argument("--series", action="append", required=True, help="name=<iteration の順の eval JSON をカンマで>")
    p.add_argument("--episodes", type=int, default=512)
    for q in (s, m, p):
        q.add_argument("--out", type=str, default=None)
    args = parser.parse_args()
    out = {"sequences": cmd_sequences, "markov": cmd_markov, "spread": cmd_spread}[args.cmd](args)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"argv": sys.argv, **out}, f, indent=1, ensure_ascii=False)
        print(f"-> {args.out}")


if __name__ == "__main__":
    main()
