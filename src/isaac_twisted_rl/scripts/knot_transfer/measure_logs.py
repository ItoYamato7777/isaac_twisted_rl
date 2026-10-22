"""既存のログから、今の固結びの判定と外部の不変量のずれ (0033 §4) と、実際の遷移の表 (§5) を数える。GPU を使わない。

    uv run --with snappy==3.2 --with topoly==1.1.0 python \
        src/isaac_twisted_rl/scripts/knot_transfer/measure_logs.py --out outputs/knot_transfer/logs.json

入力 (どれも既存のファイル。新しく Isaac Sim を回さない):

* `outputs/diag_stage_moves/obs/*.pt` (`diag_stage_moves.py --dump_obs`): chain モードの各手の **手の前の観測**。
  観測の先頭 84 列がリンク中心 (28 × 3)、129〜192 列が `next_edge` (指示した辺)。手の後の形は、同じ環境の次の
  手の観測 (`move_chain.split_episodes` で環境ごとの列に戻す)。固結び・発散で終わった手の後の形は無い。
* `outputs/archive_pdata/*.json` (`view_archive.py --dump_pdata`): アーカイブの形の生の p-data と段。

観測から作り直した交差数・段・指示した辺が、記録された値と全行で一致することを最初に確かめる (`checks`)。

## 遷移の分類 (手の前 → 手の後の p-data)

| 分類 | 意味 |
|---|---|
| `instructed` | 指示した辺の行き先 (複合の辺なら最終の子) に着いた |
| `other_shortest` | 指示しなかった最短の辺の行き先に着いた |
| `graph_up1` | R1 / R2 / cross 1 回の子だが、最短の辺ではない |
| `end_reset` | 端の交差を 1 つ外した形か、外してから同じグラフの手を 1 回打った形 (台本の tuck と同じ型) |
| `same` | p-data が変わらない |
| `removed_r2pair` / `removed_other` | 手の後が手の前の交差の部分集合 (交差が消えた)。消えた交差が R2 除去で消える組だけなら前者 |
| `added_multi` | 手の前が手の後の部分集合で、2 個以上増えた (グラフの手 2 回以上を 1 手で) |
| `other` | どれでもない (交差の並びが入れ替わった。R3 や紐の滑りなど) |
"""

from __future__ import annotations

import argparse
import collections
import functools
import glob
import itertools
import json
import os
import random
import sys
from pathlib import Path

import knotlib as K
import torch

sys.path.insert(0, str(K._COMMON))
from move_chain import OUTCOMES, ends_episode  # noqa: E402

tp, kg = K.tp, K.kg
NUM_LINKS = 28
EDGE_COLS = slice(129, 193)


@functools.cache
def _children(pd: str) -> frozenset[str]:
    """グラフの手 1 回 (R1 / R2 / cross) の子の p-data。"""
    st = K.state_from_p_data(pd)
    return frozenset(child.short_p_data for _, _, child in kg.generate_moves(st))


@functools.cache
def _subcodes(pd: str, max_crossings: int = 9) -> dict[str, int]:
    """交差を間引いた p-data -> 間引いた交差の数 (交差が多すぎる形は空)。"""
    cr = K.crossings_of(pd)
    if len(cr) > max_crossings:
        return {}
    out = {}
    for r in range(len(cr) + 1):
        for keep in itertools.combinations(cr, r):
            sub = K.subcode(pd, {p for pq in keep for p in pq})
            out.setdefault(sub, len(cr) - r)
    return out


@functools.cache
def _end_reset(pd: str) -> frozenset[str]:
    if not pd:
        return frozenset()
    n = len(pd.split("_"))
    out = set()
    for end in (1, n):
        mid, _ = kg.remove_crossing(pd, end)
        out.add(mid)
        out |= _children(mid)
    return frozenset(out)


def _r2_reduced(pd: str) -> str:
    batch = tp.IntersectionBatch.from_lists([kg.p_data_to_lists(pd)])
    return tp.p_data_from_batch(tp.reduce_r2(batch), short=True)[0]


def edit(before: str, after: str) -> tuple[int, int] | None:
    """最大の共通の部分図を残したときの (消えた交差の数, 増えた交差の数)。交差が多すぎる形は None。"""
    sb, sa = _subcodes(before), _subcodes(after)
    if not sb or not sa:
        return None
    common = set(sb) & set(sa)
    best = min(common, key=lambda pd: sb[pd] + sa[pd])
    return sb[best], sa[best]


def has_r2_pair(pd: str) -> bool:
    return _r2_reduced(pd) != pd


def classify(before: str, after: str, instructed: str | None, shortest: frozenset[str]) -> str:
    if after == before:
        return "same"
    if instructed is not None and after == instructed:
        return "instructed"
    if after in shortest:
        return "other_shortest"
    if after in _children(before):
        return "graph_up1"
    if after in _end_reset(before):
        return "end_reset"
    subs = _subcodes(before)
    if after in subs:
        return (
            "removed_r2pair"
            if _r2_reduced(before) == after or _r2_reduced(before) == _r2_reduced(after)
            else "removed_other"
        )
    if before in _subcodes(after):
        return "added_multi"
    return "other"


def _decode_edges(batch: tp.IntersectionBatch, ne: torch.Tensor) -> tuple[list[str | None], list[frozenset[str]]]:
    """観測の `next_edge` を、表の候補の特徴と照合して、指示した辺の子の p-data に戻す。"""
    tab = tp.get_next_edge_table("cpu", "bfs")
    row, found, _ = tab.lookup(batch)
    links = tp.point_links(batch, NUM_LINKS, 1)
    n = ne.shape[0]
    best = torch.full((n,), -1)
    for c in range(tab.num_cands):
        cand = torch.full((n,), c)
        f = tab.features(row, cand, links, NUM_LINKS)
        eq = ((f - ne).abs().max(1).values < 1e-5) & (tab.kind[row, c] >= 0) & found
        best = torch.where((best < 0) & eq, cand, best)
    key_to_pd = {kg.p_data_key(pd): pd for pd in kg.distance_to_overhand()}
    instructed, shortest = [], []
    for i in range(n):
        if not bool(found[i]):
            instructed.append(None)
            shortest.append(frozenset())
            continue
        kids = [int(k) for k in tab.child_key[row[i]].tolist() if k >= 0]
        shortest.append(frozenset(key_to_pd[k] for k in kids))
        instructed.append(key_to_pd[int(tab.child_key[row[i], best[i]])] if best[i] >= 0 else None)
    return instructed, shortest


def _sequences(d: dict, num_envs: int, max_moves: int) -> list[list[int]]:
    """行の番号を、環境ごとの手の列に戻す (`move_chain.split_episodes` と同じ並びの前提)。"""
    after, outcome = d["after"].tolist(), d["outcome"].tolist()
    alive = list(range(num_envs))
    seqs: list[list[int]] = [[] for _ in range(num_envs)]
    i = 0
    for _ in range(max_moves):
        if not alive:
            break
        nxt = []
        for env in alive:
            seqs[env].append(i)
            if not ends_episode(after[i], OUTCOMES[outcome[i]]):
                nxt.append(env)
            i += 1
        alive = nxt
    if i != len(after):
        raise ValueError(f"行の数が合わない: {i} != {len(after)}")
    return seqs


def measure_dump(path: str, num_envs: int, max_moves: int, topoly_sample: int, rng: random.Random) -> dict:
    d = torch.load(path, weights_only=False)
    ob = d["actor_obs"].double()
    poly = tp.polyline_from_link_centers(ob[:, : 3 * NUM_LINKS].reshape(-1, NUM_LINKS, 3))
    batch = tp.segment_intersections(poly)
    red = tp.reduce_r2(batch)
    stage = kg.get_bfs_potential("cpu").stage(batch, red)
    pds = tp.p_data_from_batch(batch, short=True)
    instructed, shortest = _decode_edges(batch, ob[:, EDGE_COLS])
    checks = {
        "count_matches_obs": float((batch.count.double() == ob[:, 84]).double().mean()),
        "stage_matches_dump": float((stage == d["before"]).double().mean()),
        "edge_decoded_when_valid": float(
            sum(1 for i, x in enumerate(instructed) if x is not None and ob[i, 192] > 0.5)
            / max(1, int((ob[:, 192] > 0.5).sum()))
        ),
    }
    seqs = _sequences(d, num_envs, max_moves)
    outcome = d["outcome"].tolist()
    after_stage = d["after"].tolist()
    table: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    explain_r2: collections.Counter = collections.Counter()
    examples: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    edits: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    r2_after: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for seq in seqs:
        for a, b in zip(seq, seq[1:] + [None]):
            before_stage = int(d["before"][a])
            if b is None:  # 固結び・発散で終わった手 (時間切れの最後の手も後の形が無い)
                o = OUTCOMES[outcome[a]]
                cls = (
                    "end_knot"
                    if o == "up" and after_stage[a] == 3
                    else ("end_diverged" if o == "diverged" else "end_timeout")
                )
                table[str(before_stage)][cls] += 1
                continue
            if OUTCOMES[outcome[a]] == "not_settled":
                table[str(before_stage)]["not_settled"] += 1
                continue
            cls = classify(pds[a], pds[b], instructed[a], shortest[a])
            table[str(before_stage)][cls] += 1
            e = edit(pds[a], pds[b])
            edits[cls]["none" if e is None else f"-{e[0]}+{e[1]}"] += 1
            if cls != "same":
                r2_after[str(before_stage)][
                    f"before_r2pair={has_r2_pair(pds[a])}|after_r2pair={has_r2_pair(pds[b])}"
                ] += 1
            if cls in ("other", "added_multi", "removed_other", "graph_up1"):
                examples[cls][f"{pds[a]} -> {pds[b]}"] += 1
            # R2 除去で「指示した辺の子」「グラフの子」に戻せるか
            if cls not in ("same", "instructed", "other_shortest") and instructed[a] is not None:
                r = _r2_reduced(pds[b])
                if r == instructed[a]:
                    explain_r2["r2_reduced_is_instructed"] += 1
                elif r in shortest[a]:
                    explain_r2["r2_reduced_is_other_shortest"] += 1
                elif r == pds[a]:
                    explain_r2["r2_reduced_is_before"] += 1
                else:
                    explain_r2["not_explained"] += 1
    # 外部の不変量 (Topoly、3D の座標から) と、自前の p-data の閉じ方の型 (SnapPy) の照合
    types = collections.Counter()
    knotted_examples: collections.Counter = collections.Counter()
    by_type: collections.Counter = collections.Counter()
    ours: list[str] = []
    for i, pd in enumerate(pds):
        over, under, _ = K.robust_type(pd)
        ours.append(over if over == under else "ambiguous")
        t = "robust_3_1" if over == under == "3_1" else ("one_closure_3_1" if "3_1" in (over, under) else "no_3_1")
        types[f"stage{int(d['before'][i])}|{t}"] += 1
        by_type[f"stage{int(d['before'][i])}|{ours[-1]}"] += 1
        if t == "robust_3_1" and int(d["before"][i]) < 3:
            knotted_examples[pd] += 1
    # 外部の不変量 (Topoly、3D の座標から直接) と、自前の p-data の閉じ方の型 (SnapPy) の照合。
    # 自前で自明でないと出た形を全部 (上限まで) と、自明と出た形の無作為な一部を照合する。
    import topoly

    nontrivial = [i for i, t in enumerate(ours) if t != "0_1"]
    trivial = [i for i, t in enumerate(ours) if t == "0_1" and len(pds[i].split("_")) >= 6]
    rng.shuffle(nontrivial)
    rng.shuffle(trivial)
    topo = collections.Counter()
    for i in nontrivial[:topoly_sample] + trivial[: topoly_sample // 3]:
        res = topoly.alexander(poly[i].tolist(), closure=topoly.Closure.TWO_POINTS, tries=20)
        top = max(res, key=res.get) if isinstance(res, dict) else str(res)
        topo[f"ours={ours[i]}|topoly={top}"] += 1
    # 段 0〜2 (今の判定では固結びでない) なのに、自前の閉じ方で 3_1 / 4_1 と出た形を、Topoly で全部 (上限まで) 照合する
    missed = collections.Counter()
    cand = [i for i, t in enumerate(ours) if t in ("3_1", "4_1") and int(d["before"][i]) < 3]
    rng.shuffle(cand)
    for i in cand[:topoly_sample]:
        res = topoly.alexander(poly[i].tolist(), closure=topoly.Closure.TWO_POINTS, tries=50)
        top = max(res, key=res.get) if isinstance(res, dict) else str(res)
        missed[f"ours={ours[i]}|topoly={top}"] += 1
    return {
        "rows": len(pds),
        "checks": checks,
        "transitions_by_stage_before": {k: dict(v.most_common()) for k, v in sorted(table.items())},
        "deviation_explained_by_r2": dict(explain_r2),
        "examples": {k: dict(v.most_common(8)) for k, v in examples.items()},
        "edit_by_class": {k: dict(v.most_common(10)) for k, v in edits.items()},
        "r2_pairs_on_changed_moves": {k: dict(v) for k, v in sorted(r2_after.items())},
        "closure_types_of_before_shapes": dict(sorted(types.items())),
        "topoly_vs_ours": dict(topo.most_common()),
        "robust_3_1_below_stage3": dict(knotted_examples.most_common(10)),
        "type_by_stage_before": dict(sorted(by_type.items())),
        "topoly_on_knots_below_stage3": dict(missed.most_common()),
    }


def measure_archives(paths: list[str]) -> dict:
    out = {}
    for p in paths:
        shapes = json.load(open(p))["shapes"]
        tally = collections.Counter()
        cache = {}
        for s in shapes:
            pd = s["raw"]
            if pd not in cache:
                over, under, _ = K.robust_type(pd)
                cache[pd] = (
                    "robust_3_1"
                    if over == under == "3_1"
                    else ("one_closure_3_1" if "3_1" in (over, under) else "no_3_1")
                )
            tally[f"stage{s['stage']}|{cache[pd]}|r2_goal={kg.is_goal([pd])[0]}"] += 1
        out[Path(p).stem] = {"shapes": len(shapes), "tally": dict(sorted(tally.items()))}
        print(Path(p).stem, out[Path(p).stem]["tally"], flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--dumps", default="outputs/diag_stage_moves/obs/*.pt")
    ap.add_argument("--archives", default="outputs/archive_pdata/*.json")
    ap.add_argument("--num_envs", type=int, default=512)
    ap.add_argument("--max_moves", type=int, default=18)
    ap.add_argument("--topoly_sample", type=int, default=150)
    args = ap.parse_args()
    rng = random.Random(0)
    out = {"dumps": {}, "archives": measure_archives(sorted(glob.glob(args.archives)))}
    for p in sorted(glob.glob(args.dumps)):
        r = measure_dump(p, args.num_envs, args.max_moves, args.topoly_sample, rng)
        out["dumps"][Path(p).stem] = r
        print(Path(p).stem, r["checks"], r["transitions_by_stage_before"], r["deviation_explained_by_r2"], flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
