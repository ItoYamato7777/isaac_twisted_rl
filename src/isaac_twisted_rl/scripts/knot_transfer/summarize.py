"""`measure_abstract.py`・`measure_logs.py`・`graph_size.py` の JSON から、0033 の表 (Markdown) を出す。

uv run python src/isaac_twisted_rl/scripts/knot_transfer/summarize.py --dir outputs/knot_transfer > outputs/knot_transfer/summary.md
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import knotlib as K


def _stage_along(path_p_data: list[str]) -> list[str]:
    pot = K.kg.get_bfs_potential("cpu")
    out = []
    for pd in path_p_data:
        b = K.tp.IntersectionBatch.from_lists([K.kg.p_data_to_lists(pd)])
        r = K.tp.reduce_r2(b)
        n = len(K.crossings_of(pd))
        st = int(pot.stage(b, r)[0]) if n <= K.kg.MAX_GRAPH_CROSSINGS else int(K.tp.topology_class(r)[0])
        out.append(f"{n}:{st}")
    return out


def curves_table(a: dict) -> str:
    rows = [
        "| 曲線 | 交差 | 両端が同じ面 | 閉じ方 上 / 下 の型 (SnapPy) | Topoly (3D, 2 点の閉じ方 100 回) | `is_overhand` (生) | R2 除去後の固結び (今のゴール) | 段 (v17) | 4 cm のリンクで p-data が残る |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, r in a["curves"].items():
        c = r["current"]
        top = ", ".join(
            f"{k} {v:.2f}" for k, v in sorted(r.get("topoly_two_points", {}).items(), key=lambda kv: -kv[1])
        )
        rows.append(
            f"| {name} | {r['crossings']} | {r['ends_same_face']} | {r['type_over_closure']} / {r['type_under_closure']} | {top} "
            f"| {c['is_overhand_raw']} | {c['r2_goal']} | {c['stage_bfs'] if c['stage_bfs'] is not None else '表の外 (ラダー ' + str(c['ladder_class']) + ')'} "
            f"| {r['rope']['p_data_kept_at_4cm']} |"
        )
    return "\n".join(rows)


def search_table(a: dict) -> str:
    rows = [
        "| 目標 | 交差 | 最短手数 | 手の列 | 途中の最大交差 | 広げた状態の数 | 経路の上の (交差:今の段) |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, r in a["search"].items():
        rows.append(
            f"| {name} | {r['crossings']} | {r['distance']} | {' → '.join(r['moves'])} | {r['crossings']} | {r['states_visited']} "
            f"| {' → '.join(_stage_along(r['path_p_data']))} |"
        )
    return "\n".join(rows)


def graph_table(a: dict, g4: dict) -> str:
    rows = ["| 交差 | 状態の数 | 両方の閉じ方で 3_1 | 両方で 4_1 | 片方だけ 3_1 か 4_1 |", "|---|---|---|---|---|"]
    for c, n in g4["states_by_crossings"].items():
        t = g4["types_by_crossings"].get(c, {})
        r3 = t.get("3_1", 0)
        r4 = t.get("4_1", 0)
        one = sum(v for k, v in t.items() if "|" in k and ("3_1" in k or "4_1" in k))
        rows.append(f"| {c} | {n} | {r3} | {r4} | {one} |")
    conf = a["graph_le4"]["confusion"]
    rows.append("")
    rows.append("| 今のゴール (R2 除去後に固結びを含む) | 両方の閉じ方で 3_1 | 片方だけ 3_1 | 3_1 でない |")
    rows.append("|---|---|---|---|")
    for g in ("True", "False"):
        rows.append(
            f"| {g} | {conf.get(f'r2_goal={g}|robust_3_1', 0)} | {conf.get(f'r2_goal={g}|one_closure_3_1', 0)} | {conf.get(f'r2_goal={g}|no_3_1', 0)} |"
        )
    rows.append("")
    rows.append(
        "最短手数 (自明な状態から、交差 4 個までのグラフ): "
        + ", ".join(f"{k} = {v}" for k, v in g4["min_moves_from_trivial"].items())
    )
    return "\n".join(rows)


def twisted_table(a: dict) -> str:
    rows = ["| 評価の集合 (TWISTED-RL) | 個数 | 閉じた型の内訳 (`上|下` は閉じ方で違う) |", "|---|---|---|"]
    for name, r in a["twisted_rl_sets"].items():
        rows.append(f"| {name} | {len(r['rows'])} | {', '.join(f'{k}: {v}' for k, v in r['types'].items())} |")
    return "\n".join(rows)


def archive_table(logs: list[dict]) -> str:
    rows = [
        "| アーカイブ (`outputs/archive_pdata/`) | 段 2 の形 | 両方の閉じ方で 3_1 | 片方だけ 3_1 |",
        "|---|---|---|---|",
    ]
    seen = set()
    for lg in logs:
        for name, r in lg["archives"].items():
            if name in seen:
                continue
            seen.add(name)
            t = r["tally"]
            s2 = sum(v for k, v in t.items() if k.startswith("stage2"))
            rows.append(
                f"| {name} | {s2} | {sum(v for k, v in t.items() if k.startswith('stage2|robust_3_1'))} "
                f"| {sum(v for k, v in t.items() if k.startswith('stage2|one_closure_3_1'))} |"
            )
    return "\n".join(rows)


def dump_tables(logs: list[dict]) -> str:
    out = []
    rows = [
        "| run (chain 評価の手の前の形) | 形 | 段 0〜2 で両方の閉じ方 3_1 | 段 0〜2 で両方の閉じ方 4_1 | Topoly と一致 (無作為: 自前で 3_1 / 4_1 / 0_1) | Topoly と一致 (段 0〜2 の 3_1 / 4_1) |",
        "|---|---|---|---|---|---|",
    ]
    for lg in logs:
        for name, r in lg["dumps"].items():
            t = r["type_by_stage_before"]
            k3 = sum(v for k, v in t.items() if k.endswith("|3_1") and not k.startswith("stage3"))
            k4 = sum(v for k, v in t.items() if k.endswith("|4_1") and not k.startswith("stage3"))
            tv = r["topoly_vs_ours"]

            def agree(x: str) -> str:
                tot = sum(v for k, v in tv.items() if k.startswith(f"ours={x}|"))
                ok = tv.get(f"ours={x}|topoly={x}", 0)
                return f"{ok}/{tot}"

            mv = r.get("topoly_on_knots_below_stage3", {})
            m3 = f"{mv.get('ours=3_1|topoly=3_1', 0)}/{sum(v for k, v in mv.items() if k.startswith('ours=3_1|'))}"
            m4 = f"{mv.get('ours=4_1|topoly=4_1', 0)}/{sum(v for k, v in mv.items() if k.startswith('ours=4_1|'))}"
            rows.append(
                f"| {name} | {r['rows']} | {k3} | {k4} | {agree('3_1')} / {agree('4_1')} / {agree('0_1')} | {m3} / {m4} |"
            )
    out.append("\n".join(rows))
    classes = [
        "instructed",
        "other_shortest",
        "graph_up1",
        "end_reset",
        "added_multi",
        "removed_r2pair",
        "removed_other",
        "same",
        "other",
    ]
    rows = [
        "| run | 段 | 固結びで終わった手 | 発散 | 手の後の形がある手 | " + " | ".join(classes) + " |",
        "|---|---|---|---|---|" + "---|" * len(classes),
    ]
    for lg in logs:
        for name, r in lg["dumps"].items():
            for st, c in r["transitions_by_stage_before"].items():
                if st == "3":
                    continue
                n = sum(c.get(k, 0) for k in classes)
                rows.append(
                    f"| {name} | {st} | {c.get('end_knot', 0)} | {c.get('end_diverged', 0)} | {n} | "
                    + " | ".join(f"{100 * c.get(k, 0) / max(n, 1):.0f}%" for k in classes)
                    + " |"
                )
    out.append("\n".join(rows))
    tot = collections.Counter()
    r2 = collections.Counter()
    edits = collections.defaultdict(collections.Counter)
    for lg in logs:
        for r in lg["dumps"].values():
            tot.update(r["deviation_explained_by_r2"])
            for st, c in r["r2_pairs_on_changed_moves"].items():
                r2.update({f"{st}|{k}": v for k, v in c.items()})
            for cls, e in r["edit_by_class"].items():
                edits[cls].update(e)
    out.append(
        "R2 除去で説明できるか (指示した辺のある手のうち、指示した辺にも最短の辺にも着かなかった手): "
        + json.dumps(dict(tot), ensure_ascii=False)
    )
    out.append("形が変わった手の R2 の組の有無 (段|前|後): " + json.dumps(dict(sorted(r2.items())), ensure_ascii=False))
    out.append(
        "分類ごとの (消えた交差, 増えた交差) の上位: "
        + json.dumps({k: dict(v.most_common(4)) for k, v in edits.items()}, ensure_ascii=False)
    )
    return "\n\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="outputs/knot_transfer")
    args = ap.parse_args()
    d = Path(args.dir)
    a = json.load(open(d / "abstract.json"))
    g4 = json.load(open(d / "graph_size_4.json"))
    logs = [json.load(open(p)) for p in sorted(d.glob("logs_v*.json"))]
    for title, body in (
        ("曲線", curves_table(a)),
        ("最短経路 (交差を間引いた図だけを広げる BFS)", search_table(a)),
        ("交差 4 個までのグラフ", graph_table(a, g4)),
        ("TWISTED-RL の評価のゴール", twisted_table(a)),
        ("アーカイブ", archive_table(logs)),
        ("chain 評価の手 (diag_stage_moves の保存)", dump_tables(logs)),
    ):
        print(f"## {title}\n\n{body}\n")


if __name__ == "__main__":
    main()
