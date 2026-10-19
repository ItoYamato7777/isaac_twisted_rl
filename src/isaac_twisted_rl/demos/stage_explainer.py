"""段 (`BfsPotential.stage`) の説明スライドに載せる表と図を、トポロジーの関数を実行して作る。

    uv run python src/isaac_twisted_rl/demos/stage_explainer.py
    uv run python src/isaac_twisted_rl/demos/stage_explainer.py --dump outputs/dump_real_deploy/sim_v14_s43_it300_dump.npz

出力 (`--out`、既定は `outputs/stage_explainer/`):
    tables.json          列挙の数、表 (交差数 x d)、段の内訳、例の値、ダンプの照合
    quiz_questions.json  スライドに載せない p-data 10 個 (読み手のテスト用。答えは入れない)
    quiz_answers.json    その答え (段・ラダー・d・R2 除去後の p-data)
    fig_*.png            結び目の図 (下を通る交差は線を切る)

## なぜ値を手で書かず、このスクリプトで出すのか

スライドの数字と例が、コードの今の挙動と食い違わないようにするため。同じコミットで走らせれば同じ値が出る
(BFS の列挙は決定論、クイズの選び方は `--seed` で固定)。Isaac Sim は使わず CPU で約 30 s。

## 実物の形の出どころ

到達状態アーカイブ (`outputs/archive_seed/`) は根の姿勢と関節角で持つので、形に戻すには Isaac Sim の順運動学が要る。
そこで、`dump_real_deploy_check.py` が書いた手ごとのリンク中心 `link_pos_w` を使う。Sim の抽出器
(`rope_state.RopeTopologyExtractor`、細分なし) と同じく `polyline_from_link_centers` -> `segment_intersections` で段を
計算し直し、ダンプの観測の末尾 4 列 (Sim が計算した段の one-hot) と一致するかを `tables.json` の `dump.agreement` に書く。

`tasks/` の外に置くのは、`tasks/__init__.py` の `import_packages` (Kit の起動前に配下を全部 import する) に拾わせないため。
トポロジーは `tests/conftest.py` と同じく `common/` を `sys.path` に足して読む (`isaac_twisted_rl.tasks` を import すると
isaaclab が要る)。
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import random
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

_REPO = pathlib.Path(__file__).resolve().parents[3]
_COMMON = _REPO / "src" / "isaac_twisted_rl" / "tasks" / "knot_tying" / "common"
sys.path.insert(0, str(_COMMON))

import topology as tp  # noqa: E402
from topology import knot_graph as kg  # noqa: E402

# 図の色。段の色は Play の動画のグリッパの色 (`scripts/rsl_rl/stage_colors.py`: 段 0 青・段 1 黄・段 2 赤) に揃える。
BG = "#FCFBF8"
ROPE = "#3D3A5C"
START = "#D9822B"
LABEL_BG = "#FFFFFF"

# `tests/test_knot_graph.py` と同じ名前の例。スライドとテストで同じ形を指す。
R1 = ("1O2+_2U1+", "1U2+_2O1+", "1O2-_2U1-", "1U2-_2O1-")
C2 = ("1O3+_2U4+_3U1+_4O2+", "1U3+_2O4+_3O1+_4U2+", "1O3-_2U4-_3U1-_4O2-", "1U3-_2O4-_3O1-_4U2-")
COVER = "1U4+_2O5+_3O6-_4O1+_5U2+_6U3-"
FAKE_C2 = "1U4+_2U5-_3U6+_4O1+_5O2-_6O3+"
TWISTS2 = "1U2+_2O1+_3U4+_4O3+"


def _crossings(p_data: str) -> int:
    return len(p_data.split("_")) // 2 if p_data else 0


def _batch(p_data: list[str]) -> tp.IntersectionBatch:
    return tp.IntersectionBatch.from_lists([kg.p_data_to_lists(pd) for pd in p_data])


def describe(p_data: list[str], pot: kg.BfsPotential) -> list[dict]:
    """p-data ごとの、段を決める量 (R2 除去後の p-data・ラダー・BFS の d と Φ・段)。"""
    dist = kg.distance_to_overhand()
    raw = _batch(p_data)
    red = tp.reduce_r2(raw)
    ladder = tp.topology_class(red).tolist()
    bfs3 = (pot.bfs_only(raw) * 3).round().long().tolist()
    stage = pot.stage(raw, red).tolist()
    reduced = tp.p_data_from_batch(red, short=True)
    out = []
    for pd, rd, la, b, s in zip(p_data, reduced, ladder, bfs3, stage):
        out.append(
            {
                "p_data": pd,
                "crossings": _crossings(pd),
                "reduced": rd,
                "reduced_crossings": _crossings(rd),
                "ladder": la,
                "d": dist.get(pd),
                "bfs_x3": b,
                "stage": s,
            }
        )
    return out


def _two_crossing_kind(row: dict) -> str:
    """交差 2 個の状態の分け方。スライドの「36 個の中身」に使う。"""
    if row["ladder"] == tp.CLASS_INTERLEAVED2:
        return "本物の C2"
    if row["reduced_crossings"] == 0:
        return "R2 で交差 0 に戻る"
    items, _ = kg.p_data_to_lists(row["p_data"])
    gaps = sorted(abs(i - j) for i, j, _, _ in items)
    if gaps == [1, 1]:
        return "独立したねじれ 2 つ"
    return f"その他 (gap {gaps})"


def enumeration_tables(pot: kg.BfsPotential) -> dict:
    """BFS の列挙・表・段の内訳。"""
    p_data, _ = kg.enumerate_graph()
    dist = kg.distance_to_overhand()
    rows = describe(list(p_data), pot)

    by_cd = collections.Counter((_crossings(pd), d) for pd, d in dist.items())
    detail = collections.Counter()
    for r in rows:
        if r["bfs_x3"] > r["ladder"]:
            rel = "bfs>ladder"
        elif r["bfs_x3"] < r["ladder"]:
            rel = "bfs<ladder"
        else:
            rel = "equal"
        detail[(r["crossings"], r["stage"], r["ladder"], r["bfs_x3"], rel)] += 1
    two = collections.Counter(_two_crossing_kind(r) for r in rows if r["crossings"] == 2)
    two_examples: dict[str, list[str]] = collections.defaultdict(list)
    for r in rows:
        if r["crossings"] == 2:
            two_examples[_two_crossing_kind(r)].append(r["p_data"])
    return {
        "states_by_crossings": dict(sorted(collections.Counter(r["crossings"] for r in rows).items())),
        "goals": sum(1 for r in rows if r["d"] == 0),
        "table_size": len(dist),
        "table_by_crossings_and_d": [{"crossings": c, "d": d, "count": n} for (c, d), n in sorted(by_cd.items())],
        "stage_over_enumerated": dict(sorted(collections.Counter(r["stage"] for r in rows).items())),
        "stage_detail": [
            {"crossings": c, "stage": s, "ladder": la, "bfs_x3": b, "relation": rel, "count": n}
            for (c, s, la, b, rel), n in sorted(detail.items())
        ],
        "two_crossings": dict(two),
        "two_crossings_examples": {k: v[:4] for k, v in two_examples.items()},
    }, rows


def named_examples(pot: kg.BfsPotential) -> list[dict]:
    """スライドに載せる名前付きの例。最後の 2 つは「BFS だけだと段が下がる」の例 (交差 5 個は表の外)。"""
    names = [
        ("自明", ""),
        ("輪 1 つ", R1[1]),
        ("独立したねじれ 2 つ", TWISTS2),
        ("輪の上に先端を置いただけ (R2 で交差 0)", "1U3+_2U4-_3O1+_4O2-"),
        ("本物の C2", C2[1]),
        ("偽の C2 (輪を跨いだだけ)", FAKE_C2),
        ("cover 後 (ゲート 1)", COVER),
        ("固結び", kg.OVERHAND_P_DATA[0]),
        ("本物の C2 + 尻尾のねじれ 1 つ", "1U3+_2O4+_3O1+_4U2+_5U6+_6O5+"),
        ("本物の C2 + 尻尾のねじれ 2 つ", "1U3+_2O4+_3O1+_4U2+_5U6+_6O5+_7U8+_8O7+"),
        ("本物の C2 + 尻尾のねじれ 3 つ", "1U3+_2O4+_3O1+_4U2+_5U6+_6O5+_7U8+_8O7+_9U10+_10O9+"),
    ]
    rows = describe([pd for _, pd in names], pot)
    for (name, _), r in zip(names, rows):
        r["name"] = name
    return rows


def make_quiz(rows: list[dict], exclude: set[str], seed: int) -> list[dict]:
    """スライドに無い p-data 10 個。段と紛らわしさの種類で層別して選ぶ。"""
    rng = random.Random(seed)
    pools: dict[str, list[dict]] = collections.defaultdict(list)
    for r in rows:
        if r["p_data"] in exclude:
            continue
        c, s, la, b = r["crossings"], r["stage"], r["ladder"], r["bfs_x3"]
        if c == 1:
            pools["輪 1 つ"].append(r)
        elif c == 2 and la == 0 and r["reduced_crossings"] == 0:
            pools["交差 2・R2 で 0"].append(r)
        elif c == 2 and la == 0:
            pools["交差 2・R2 で残る"].append(r)
        elif c == 2 and la == 2:
            pools["本物の C2"].append(r)
        elif c == 3 and s == 2 and b > la:
            pools["交差 3・BFS が持ち上げる"].append(r)
        elif c == 3 and s == 1 and b < la:
            pools["交差 3・表に無くラダー 1"].append(r)
        elif c == 3 and s == 0:
            pools["交差 3・段 0"].append(r)
        elif c == 4 and s == 3:
            pools["交差 4・ゴール"].append(r)
        elif c == 4 and s == 2 and b < la:
            pools["交差 4・表に無くラダー 2"].append(r)
        elif c == 4 and s == 0:
            pools["交差 4・段 0"].append(r)
    picks = [rng.choice(sorted(pool, key=lambda r: r["p_data"])) | {"kind": k} for k, pool in sorted(pools.items())]
    # 表の外 (交差 5 個以上) の 2 問。列挙に無いので、尻尾にねじれを足した p-data を作る。
    beyond = {
        "交差 5・固結び + 尻尾のねじれ 2 つ": with_tail_twists(rng.choice(kg.OVERHAND_P_DATA), 2, rng),
        "交差 5・ねじれ 5 つ": with_tail_twists("", 5, rng),
    }
    for kind, row in zip(beyond, describe(list(beyond.values()), kg.get_bfs_potential("cpu"))):
        picks.append(row | {"kind": kind})
    rng.shuffle(picks)
    return picks


def with_tail_twists(p_data: str, num: int, rng: random.Random) -> str:
    """p-data の尻尾 (番号の大きい側) に、独立したねじれ (R1、gap 1) を `num` 個足す。上下と符号は乱数。"""
    tokens = p_data.split("_") if p_data else []
    for _ in range(num):
        a = len(tokens) + 1
        first = rng.choice("OU")
        second = "U" if first == "O" else "O"
        sign = rng.choice("+-")
        tokens += [f"{a}{first}{a + 1}{sign}", f"{a + 1}{second}{a}{sign}"]
    return "_".join(tokens)


# ---------------------------------------------------------------- 図


def _seg_intersections(strands: list[np.ndarray]) -> list[tuple]:
    """全ストランドの線分どうしの xy の交差 `(over_strand, over_s, under_strand, under_s, x, y)`。s は弧長。

    図を描くためだけの計算 (隣の線分は除く)。上下は交点で線形補間した z で決める (`intersections.py` と同じ規約)。
    """
    segs = []
    for k, p in enumerate(strands):
        cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1))])
        for i in range(len(p) - 1):
            segs.append((k, i, p[i], p[i + 1], cum[i]))
    out = []
    for a in range(len(segs)):
        ka, ia, A, A1, ca = segs[a]
        for b in range(a + 1, len(segs)):
            kb, ib, Cc, C1, cb = segs[b]
            if ka == kb and abs(ia - ib) < 2:
                continue
            d1, d2 = (A1 - A)[:2], (C1 - Cc)[:2]
            den = d1[0] * d2[1] - d1[1] * d2[0]
            if abs(den) < 1e-12:
                continue
            w = (Cc - A)[:2]
            al = (w[0] * d2[1] - w[1] * d2[0]) / den
            be = (w[0] * d1[1] - w[1] * d1[0]) / den
            if not (0 <= al <= 1 and 0 <= be <= 1):
                continue
            za = A[2] + al * (A1[2] - A[2])
            zb = Cc[2] + be * (C1[2] - Cc[2])
            sa = ca + al * np.linalg.norm(d1)
            sb = cb + be * np.linalg.norm(d2)
            x, y = A[:2] + al * d1
            if za >= zb:
                out.append((ka, sa, kb, sb, x, y))
            else:
                out.append((kb, sb, ka, sa, x, y))
    return out


def _piece(p: np.ndarray, s0: float, s1: float, n: int = 24) -> np.ndarray:
    """ストランド p の弧長 [s0, s1] の部分の xy。"""
    cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1))])
    s = np.linspace(max(s0, 0.0), min(s1, cum[-1]), n)
    return np.stack([np.interp(s, cum, p[:, 0]), np.interp(s, cum, p[:, 1])], axis=1)


def draw_strands(ax, strands: list[np.ndarray], lw: float, gap: float, color: str = ROPE) -> list[tuple]:
    """結び目の図: ストランドを描き、各交差で上の線を背景色の縁取り付きで描き直す (下の線が切れて見える)。"""
    for p in strands:
        ax.plot(p[:, 0], p[:, 1], color=color, lw=lw, solid_capstyle="round", solid_joinstyle="round", zorder=1)
    span = max(np.ptp(np.concatenate(strands)[:, 0]), np.ptp(np.concatenate(strands)[:, 1]))
    half = 0.08 * span
    crossings = _seg_intersections(strands)
    for ko, so, _, _, _, _ in crossings:
        q = _piece(strands[ko], so - half, so + half)
        ax.plot(q[:, 0], q[:, 1], color=BG, lw=lw + gap, solid_capstyle="butt", zorder=2)
        ax.plot(q[:, 0], q[:, 1], color=color, lw=lw, solid_capstyle="round", zorder=3)
    ax.set_aspect("equal")
    ax.axis("off")
    return crossings


def _new_fig(w: float, h: float):
    fig = plt.figure(figsize=(w, h), dpi=150, facecolor=BG)
    return fig


def draw_rope(path: pathlib.Path, nodes: np.ndarray, labels: bool, size: float = 6.0) -> str:
    """ロープ (ポリライン `(N, 3)`) を描く。`labels` なら交差に p-data の点番号 (例 `1,4`) を付ける。p-data を返す。

    始点 (p-data の番号を数え始める端) に橙の点を打つ。
    """
    pts = torch.as_tensor(nodes, dtype=torch.float64)
    batch = tp.segment_intersections(pts.unsqueeze(0))
    p_data = tp.p_data_from_batch(batch, short=True)[0]
    pos = tp.crossing_positions(pts.unsqueeze(0), batch)[0].numpy()
    order = batch.order_list(0)

    fig = _new_fig(size, size)
    ax = fig.add_axes([0.03, 0.03, 0.94, 0.94])
    ax.set_facecolor(BG)
    off = np.array([nodes[:, 0].mean(), nodes[:, 1].mean(), 0.0])
    xy = nodes - off
    draw_strands(ax, [xy], lw=7, gap=12)
    ax.plot(xy[0, 0], xy[0, 1], "o", color=START, ms=14 if labels else 10, zorder=5)
    span = max(np.ptp(xy[:, 0]), np.ptp(xy[:, 1]))
    cx, cy = (xy[:, 0].max() + xy[:, 0].min()) / 2, (xy[:, 1].max() + xy[:, 1].min()) / 2
    half = 0.5 * span * 1.15
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    if labels:
        # ラベルは斜め 4 方向の候補のうち、ほかの交差と置いたラベルから一番遠い所に置く (どの交差のラベルか紛れないように)。
        per_pt = 2 * half / (size * 72 * 0.94)  # 1 pt あたりのデータの長さ
        cand = [np.array(c) for c in ((38, 26), (38, -26), (-38, 26), (-38, -26))]
        xs = pos - off[:2]
        placed: list[np.ndarray] = []
        for k, ((a, b), p) in enumerate(zip(order, xs)):
            others = [q for m, q in enumerate(xs) if m != k] + placed

            def clearance(c: np.ndarray, p=p, others=others) -> float:
                center = p + c * per_pt
                return min((np.linalg.norm(center - q) for q in others), default=np.inf)

            best = max(cand, key=clearance)
            placed.append(p + best * per_pt)
            ax.annotate(
                f"{a},{b}",
                tuple(p),
                xytext=(best[0] - 20, best[1] - 9),
                textcoords="offset points",
                fontsize=17,
                color=ROPE,
                bbox={"boxstyle": "round,pad=0.25", "fc": LABEL_BG, "ec": ROPE, "lw": 1},
                zorder=7,
            )
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return p_data


# ---------------------------------------------------------------- 合成の形 (`tests/test_r2_reduce.py` と同じ道筋)


def _circle(cx: float, cy: float, r: float, a0: float, a1: float, n: int = 40) -> list[tuple[float, float]]:
    return [
        (cx + r * np.cos(np.radians(a0 + (a1 - a0) * k / n)), cy + r * np.sin(np.radians(a0 + (a1 - a0) * k / n)))
        for k in range(n + 1)
    ]


# 始点 -> 右へ -> 反時計回りの輪 -> 真下へ降りて始点側の直線を x=0 で横切る (交差 A)。
# `tests/test_r2_reduce.py` の `_LOOP`・`_STAGES` と同じ点列 (そちらがこの形の位相を試験で固定している)。
_LOOP = [(-3.0, 0.0), (1.0, 0.0)] + _circle(1.0, 1.0, 1.0, -90, 180)[1:] + [(0.0, -1.0)]
_PATH = {
    0: [(-3.0, 0.0), (2.6, 0.0)],
    1: _LOOP + [(0.0, -1.5)],
    2: _LOOP + [(0.0, -1.5), (0.5, -1.5), (0.5, 0.6)],
    3: _LOOP + [(0.0, -1.5), (0.5, -1.5), (0.5, 0.6), (1.5, 0.6), (2.6, 0.6)],
}

SYNTHETIC = {
    # 名前: (道筋の段, 交差 A・B・C で先端側が上か, 期待する p-data)
    "straight": (0, (), ""),
    "loop": (1, (True,), "1U2+_2O1+"),
    "tip_on_loop": (2, (True, True), None),
    "c2": (2, (True, False), None),
    "fake_c2": (3, (True, True, True), FAKE_C2),
    "cover": (3, (True, False, False), COVER),
    "overhand": (3, (True, False, True), None),
}
"""説明に使う合成の形。上下の組で、同じ道筋が偽の C2・cover 後と同じ p-data・固結びに分かれる。"""


def staged_rope(path_stage: int, tip_over: tuple[bool, ...]) -> np.ndarray:
    """`tests/test_r2_reduce.py` の `_staged_rope` と同じ作り方 (先端側の上下は交差の近くの z の盛り上がりで決める)。"""
    w = np.array(_PATH[path_stage], dtype=np.float64)
    seg = np.linalg.norm(np.diff(w, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    s = np.linspace(0, cum[-1], 400)
    xy = np.stack([np.interp(s, cum, w[:, 0]), np.interp(s, cum, w[:, 1])], axis=1)
    pts = np.concatenate([xy, np.zeros((len(xy), 1))], axis=1)
    flat = tp.segment_intersections(torch.as_tensor(pts).unsqueeze(0))
    items = sorted(flat.to_list(0), key=lambda it: it[1])  # 先端側のセグメントの順 = A, B, C
    assert len(items) == len(tip_over), (path_stage, tip_over, items)
    idx = np.arange(len(xy), dtype=np.float64)
    for (i, j, _, _), over in zip(items, tip_over):
        pts[:, 2] += 0.05 * np.exp(-(((idx - (j if over else i)) / 4.0) ** 2))
    return pts


def draw_reidemeister(path: pathlib.Path) -> None:
    """R1・R2・R3 の前後の模式図 (2 行 x 3 列)。z は上下の関係だけに使う。"""
    t = np.linspace(0, 1, 200)

    def line(x0, y0, x1, y1, z):
        return np.stack([x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, np.full_like(t, z)], axis=1)

    # R1: まっすぐ -> 輪 1 つ (交差 1)。輪は合成の形の `loop` と同じ道筋を縮めたもの。
    r1_before = [line(-1, -0.3, 1, -0.3, 0)]
    loop = staged_rope(1, (True,))
    loop[:, 0] = (loop[:, 0] + 0.2) / 2.9
    loop[:, 1] = (loop[:, 1] - 0.25) / 2.9
    r1_after = [loop]
    # R2: 並んだ 2 本 -> 左の紐を右の紐の上へ押し出す (交差 2)
    y = -1 + 2 * t
    bump = np.exp(-((y / 0.45) ** 2))
    right = np.stack([np.full_like(y, 0.35), y, np.zeros_like(y)], axis=1)
    r2_before = [np.stack([-0.45 + 0.15 * bump, y, np.ones_like(y)], axis=1), right]
    r2_after = [np.stack([-0.45 + 1.25 * bump, y, np.ones_like(y)], axis=1), right]
    # R3: 1 本 (一番上) を、下の 2 本の交差の反対側へ滑らせる (交差数は 3 のまま)
    a = line(-1, -0.8, 1, 0.8, 0)
    b = line(-1, 0.8, 1, -0.8, 0.5)
    r3_before = [a, b, line(-1, 0.45, 1, 0.45, 1)]
    r3_after = [a, b, line(-1, -0.45, 1, -0.45, 1)]

    fig = _new_fig(12, 7.2)
    for col, (before, after) in enumerate([(r1_before, r1_after), (r2_before, r2_after), (r3_before, r3_after)]):
        for row, strands in enumerate([before, after]):
            ax = fig.add_axes([0.02 + col * 0.33, 0.52 - row * 0.5, 0.29, 0.44])
            ax.set_facecolor(BG)
            draw_strands(ax, strands, lw=6, gap=9)
            ax.set_xlim(-1.15, 1.15)
            ax.set_ylim(-1.15, 1.2)
    fig.savefig(path, facecolor=BG)
    plt.close(fig)


def dump_shapes(dump: pathlib.Path, pot: kg.BfsPotential) -> tuple[dict, dict[str, np.ndarray]]:
    """ダンプの手ごとの形で段を計算し直し、Sim の観測の段と照らす。図に使う代表の形を選ぶ。"""
    d = np.load(dump, allow_pickle=True)
    link = d["link_pos_w"] - d["env_origins"][:, :, None, :]  # (手, 環境, リンク, 3)
    moves, envs = link.shape[:2]
    nodes = tp.polyline_from_link_centers(torch.as_tensor(link.reshape(moves * envs, -1, 3), dtype=torch.float32))
    batch = tp.segment_intersections(nodes)
    red = tp.reduce_r2(batch)
    stage = pot.stage(batch, red).numpy()
    ladder = tp.topology_class(red).numpy()
    bfs3 = (pot.bfs_only(batch) * 3).round().long().numpy()
    obs_stage = d["obs"].reshape(moves * envs, -1)[:, -4:].argmax(axis=1)
    p_data = tp.p_data_from_batch(batch, short=True)
    count = batch.count.numpy()

    summary = {
        "file": str(dump.relative_to(_REPO)) if dump.is_relative_to(_REPO) else str(dump),
        "checkpoint": str(d["checkpoint"]),
        "shapes": int(moves * envs),
        "agreement": int((stage == obs_stage).sum()),
        "stage_counts": {int(k): int(v) for k, v in zip(*np.unique(stage, return_counts=True))},
        "kinds": {},
    }

    def first(mask: np.ndarray, prefer: np.ndarray | None = None) -> int | None:
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            return None
        if prefer is not None:
            idx = idx[np.argsort(prefer[idx], kind="stable")]
        return int(idx[0])

    kinds = {
        "stage0_straight": (stage == 0) & (count == 0),
        "stage1_loop": (stage == 1) & (count == 1),
        "stage1_r2_zero": (stage == 1) & (count == 2) & (ladder == 0) & (red.count.numpy() == 0),
        "stage1_fake_c2": (stage == 1) & (bfs3 < ladder),
        "stage2_c2": (stage == 2) & (count == 2) & (ladder == 2),
        "stage2_cover": (stage == 2) & (bfs3 > ladder),
        "stage3_trefoil": (stage == 3) & (count == 3),
        "stage0_dead_end": (stage == 0) & (count > 0),
    }
    picked: dict[str, np.ndarray] = {}
    for name, mask in kinds.items():
        i = first(mask, prefer=count)
        summary["kinds"][name] = {"count": int(mask.sum())}
        if i is not None:
            m, e = divmod(i, envs)
            summary["kinds"][name].update({"move": m, "env": e, "p_data": p_data[i], "stage": int(stage[i])})
            picked[name] = nodes[i].numpy()
    return summary, picked


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dump", default="outputs/dump_real_deploy/sim_v14_s43_it300_dump.npz")
    parser.add_argument("--out", default="outputs/stage_explainer")
    parser.add_argument("--seed", type=int, default=0, help="クイズの p-data を選ぶ乱数のシード。")
    args = parser.parse_args()
    out = _REPO / args.out
    out.mkdir(parents=True, exist_ok=True)

    pot = kg.BfsPotential()
    tables, rows = enumeration_tables(pot)
    examples = named_examples(pot)
    tables["examples"] = examples

    shown = {r["p_data"] for r in examples} | set(R1) | set(C2) | set(kg.OVERHAND_P_DATA)
    for v in tables["two_crossings_examples"].values():
        shown |= set(v)

    # 合成の形: 図を描き、p-data と段を計算して、期待した p-data (テストで固定した値) と照らす。
    synthetic = []
    for name, (path_stage, tip_over, expected) in SYNTHETIC.items():
        nodes = staged_rope(path_stage, tip_over)
        pd = draw_rope(out / f"fig_syn_{name}.png", nodes, labels=True)
        if expected is not None and pd != expected:
            raise RuntimeError(f"合成の形 {name} の p-data が期待と違う: {pd} != {expected}")
        synthetic.append({"name": name, "tip_over": list(tip_over), **describe([pd], pot)[0]})
        shown.add(pd)
    tables["synthetic"] = synthetic

    dump = _REPO / args.dump
    if dump.exists():
        summary, picked = dump_shapes(dump, pot)
        tables["dump"] = summary
        for name, nodes in picked.items():
            shown.add(draw_rope(out / f"fig_real_{name}.png", nodes, labels=False, size=4.0))
    else:
        print(f"[WARN] ダンプが無いので実物の形の図を飛ばす: {dump}")

    quiz = make_quiz(rows, shown, args.seed)
    (out / "quiz_questions.json").write_text(
        json.dumps([{"id": i + 1, "p_data": q["p_data"]} for i, q in enumerate(quiz)], ensure_ascii=False, indent=1)
    )
    (out / "quiz_answers.json").write_text(
        json.dumps([{"id": i + 1, **q} for i, q in enumerate(quiz)], ensure_ascii=False, indent=1)
    )
    (out / "tables.json").write_text(json.dumps(tables, ensure_ascii=False, indent=1))
    draw_reidemeister(out / "fig_reidemeister.png")
    print(json.dumps({k: v for k, v in tables.items() if k not in ("stage_detail", "examples")}, ensure_ascii=False))
    print(f"[INFO] 書いた: {out}")


if __name__ == "__main__":
    main()
