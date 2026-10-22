"""固結び以外の結び目への持ち越しを抽象層だけで測るための部品 (Isaac Sim 不要)。

`docs/decisions/0033-beyond-overhand-audit.md` の §3〜§5 の計測が使う。`common/topology/` (isaaclab を import
しない純粋層) を `sys.path` から直接読み、外部の不変量の道具 (SnapPy・Topoly) は関数の中で import する
(プロジェクトの依存に無いので、使うスクリプトは `uv run --with snappy==3.2 --with topoly==1.1.0` で起動する)。

## p-data -> 閉じた結び目 (PD 符号)

開いたロープの図には結び目の型が無い。型を言うには両端をつなぐ (閉じる) 必要があり、つなぎ方で型が変わりうる。

* 両端が同じ面にあるとき: その面の中で交差なしにつなげるので、閉じ方は 1 通り (`closure="face"`)。
* 両端が別の面にあるとき: 面の双対グラフの最短路でつなぎ、つなぐ弧を全部の上 (`"over"`) か全部の下
  (`"under"`) に通す。物理では、端をどちら側から抜いても同じロープなので、**2 つの閉じ方で型が揃うときだけ
  型が決まる** と扱う (`robust_type`)。

PD 符号の向きの規約は `AbstractState` の半辺の「左の面」と、p-data の符号 (上の接線 × 下の接線の z 成分) の
組で決まる。規約が食い違えば PD は平面に描けない図になるので、毎回 `pd_is_planar` (面の数 = 交差 + 2) で確かめる。
全体の鏡像は型の判定に影響しない (3_1 と鏡像を区別しない)。

## 型の判定

SnapPy (spherogram) の knot Floer homology (HFK) で判定する。HFK は自明な結び目・三葉結び・8 の字結びを
検出する (自明: Ozsváth–Szabó の種数の検出、三葉・8 の字: Ghiggini 2008) ので、この 3 つは近似ではなく
厳密に分かれる。それ以外は `other(g=種数,r=階数)` とし、双曲なら SnapPy の `identify` の名前を付ける。
"""

from __future__ import annotations

import functools
import itertools
import math
import re
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch

_COMMON = Path(__file__).resolve().parents[2] / "tasks" / "knot_tying" / "common"
if str(_COMMON) not in sys.path:
    sys.path.insert(0, str(_COMMON))

import topology as tp  # noqa: E402
from topology import knot_graph as kg  # noqa: E402
from topology.representation import AbstractState  # noqa: E402

# ---------------------------------------------------------------- p-data <-> AbstractState


def state_from_p_data(short_p_data: str) -> AbstractState:
    """short p-data -> 面まで作った `AbstractState` (実機の抽出と同じ経路 `intersections_to_topology`)。"""
    items, orders = kg.p_data_to_lists(short_p_data)
    return tp.intersections_to_topology(items, orders, update_edges=True, update_faces=True)


def subcode(short_p_data: str, keep: set[int]) -> str:
    """交差の点の組のうち、`keep` (点の番号の集合) に入る交差だけを残して番号を詰めた p-data。"""
    pts = kg._parse(short_p_data)
    renum = {}
    for i, *_ in pts:
        if i in keep:
            renum[i] = len(renum) + 1
    toks = [f"{renum[i]}{'O' if o else 'U'}{renum[j]}{'+' if s else '-'}" for i, o, j, s in pts if i in keep]
    return "_".join(toks)


def crossings_of(short_p_data: str) -> list[tuple[int, int]]:
    """交差 = 点の組 `(p, q)` (p < q) の一覧。"""
    return sorted({(min(i, j), max(i, j)) for i, _, j, _ in kg._parse(short_p_data)})


# ---------------------------------------------------------------- 閉じた図の PD 符号


def _end_faces(st: AbstractState) -> tuple[int, int]:
    n = st.pts
    return st.edges[0].face, st.edges[2 * n].face


def _dual_path(st: AbstractState, src: int, dst: int) -> list[tuple[int, bool]]:
    """面 src から面 dst への、区間を最少回数だけ横切る道 `[(区間 i, 左の面から右の面へ渡るか)]`。"""
    n = st.pts
    adj: dict[int, list[tuple[int, int, bool]]] = {}
    for i in range(n + 1):
        left, right = st.edges[2 * i].face, st.edges[2 * i + 1].face
        if left == right:
            continue
        adj.setdefault(left, []).append((right, i, True))
        adj.setdefault(right, []).append((left, i, False))
    prev: dict[int, tuple[int, int, bool]] = {src: (-1, -1, False)}
    q = deque([src])
    while q:
        f = q.popleft()
        if f == dst:
            break
        for g, i, l2r in adj.get(f, []):
            if g not in prev:
                prev[g] = (f, i, l2r)
                q.append(g)
    if dst not in prev:
        raise RuntimeError("双対グラフで両端の面がつながらない")
    path = []
    f = dst
    while f != src:
        pf, i, l2r = prev[f]
        path.append((i, l2r))
        f = pf
    return path[::-1]


def closed_pd(short_p_data: str, closure: str = "face") -> list[list[int]]:
    """開いた図の両端をつないだ結び目の PD 符号 (KnotTheory の `X[i,j,k,l]`: 下の入り・反時計回り)。

    Args:
        closure: `"face"` (両端が同じ面のときだけ。別の面なら ValueError)、`"over"` / `"under"`
            (双対グラフの最短路で、つなぐ弧を全部の上 / 下に通す)。
    """
    st = state_from_p_data(short_p_data)
    n = st.pts
    if n == 0:
        return []
    f_head, f_tail = _end_faces(st)
    if closure == "face":
        if f_head != f_tail:
            raise ValueError("両端が別の面にある (closure='over' / 'under' を使う)")
        path: list[tuple[int, bool]] = []
    else:
        path = [] if f_head == f_tail else _dual_path(st, f_tail, f_head)
    # 閉じた曲線に沿った出来事の列: ("pt", 点の番号) か ("new", k, 区間の上か)
    on_seg = {i: k for k, (i, _) in enumerate(path)}
    events: list[tuple] = []
    for i in range(n + 1):
        if i in on_seg:
            events.append(("rope", on_seg[i]))
        if i + 1 <= n:
            events.append(("pt", i + 1))
    for k in range(len(path)):
        events.append(("close", k))
    m = len(events)
    # 出来事 e の入りの弧 = e, 出の弧 = e + 1 (mod m)。弧の番号は 1 始まり。
    inc = {e: e + 1 for e in range(m)}
    out = {e: (e + 1) % m + 1 for e in range(m)}
    pos = {ev: e for e, ev in enumerate(events)}
    pd = []
    for p in range(1, n + 1):
        pt = st.points[p]
        if pt.over is None:  # p は上
            continue
        u, o = pos[("pt", p)], pos[("pt", pt.over)]
        pd.append(_x(inc[u], out[u], inc[o], out[o], pt.sign))
    for k, (_, l2r) in enumerate(path):
        r, c = pos[("rope", k)], pos[("close", k)]
        # 区間を東向きに置くと左の面が北。左 -> 右へ渡る弧は南向き。符号 = (上 x 下)·z。
        close_over = closure == "over"
        if close_over:
            sign = 1 if l2r else -1
            pd.append(_x(inc[r], out[r], inc[c], out[c], sign))
        else:
            sign = -1 if l2r else 1
            pd.append(_x(inc[c], out[c], inc[r], out[r], sign))
    return pd


def _x(u_in: int, u_out: int, o_in: int, o_out: int, sign: int) -> list[int]:
    """1 つの交差の PD。符号 +1 (上の向きから反時計回りに 90° が下の向き) なら [下入, 上出, 下出, 上入]。"""
    return [u_in, o_out, u_out, o_in] if sign > 0 else [u_in, o_in, u_out, o_out]


def pd_is_planar(pd: list[list[int]]) -> bool:
    """PD の回転系が球面に描けるか (面の数 = 交差 + 2)。交差 0 は真。"""
    c = len(pd)
    if c == 0:
        return True
    where: dict[int, list[tuple[int, int]]] = {}
    for x, quad in enumerate(pd):
        for j, a in enumerate(quad):
            where.setdefault(a, []).append((x, j))
    if any(len(v) != 2 for v in where.values()):
        return False
    seen = set()
    faces = 0
    for x in range(c):
        for j in range(4):
            if (x, j) in seen:
                continue
            faces += 1
            cur = (x, j)
            while cur not in seen:
                seen.add(cur)
                a = pd[cur[0]][cur[1]]
                y, k = where[a][1] if where[a][0] == cur else where[a][0]
                cur = (y, (k + 1) % 4)
    return faces == c + 2


# ---------------------------------------------------------------- 型の判定 (SnapPy)


@functools.cache
def _type_of_pd(pd_key: tuple) -> str:
    import snappy  # noqa: F401 (spherogram の exterior() は snappy の import を前提にする)
    import spherogram

    if len(pd_key) == 0:
        return "0_1"
    link = spherogram.Link([list(q) for q in pd_key])
    link.simplify("global")
    if len(link.crossings) == 0:
        return "0_1"
    h = link.knot_floer_homology()
    rank, genus, tau = h["total_rank"], h["seifert_genus"], h["tau"]
    if rank == 1:
        return "0_1"
    if rank == 3 and genus == 1 and abs(tau) == 1:
        return "3_1"
    if rank == 5 and genus == 1 and tau == 0 and h["fibered"]:
        return "4_1"
    name = f"other(g={genus},r={rank})"
    try:
        ids = [str(x).split("(")[0] for x in link.exterior().identify()]
        knots = [x for x in ids if re.fullmatch(r"\d+_\d+", x)]
        if knots:
            name = f"{knots[0]}(g={genus},r={rank})"
    except Exception:  # noqa: BLE001 (双曲でない・表に無いときは名前を付けない)
        pass
    return name


def knot_type(short_p_data: str, closure: str = "face") -> str:
    """閉じた結び目の型 (`0_1` / `3_1` / `4_1` / `other(...)`)。PD が平面に描けなければ RuntimeError。"""
    pd = closed_pd(short_p_data, closure)
    if not pd_is_planar(pd):
        raise RuntimeError(f"PD が平面に描けない (規約の不一致): {short_p_data} {closure}")
    return _type_of_pd(tuple(tuple(q) for q in pd))


def robust_type(short_p_data: str) -> tuple[str, str, bool]:
    """`(上を通す閉じ方の型, 下を通す閉じ方の型, 両端が同じ面か)`。同じ面なら 2 つは同じ値。"""
    st = state_from_p_data(short_p_data)
    if st.pts == 0:
        return "0_1", "0_1", True
    f_head, f_tail = _end_faces(st)
    if f_head == f_tail:
        t = knot_type(short_p_data, "face")
        return t, t, True
    return knot_type(short_p_data, "over"), knot_type(short_p_data, "under"), False


# ---------------------------------------------------------------- 3D の曲線


def _catmull_rom(points: np.ndarray, samples_per_seg: int = 40) -> np.ndarray:
    """制御点を通る滑らかな曲線 (一様 Catmull-Rom)。端は制御点を複製して伸ばす。"""
    p = np.vstack([points[:1], points, points[-1:]])
    out = []
    for i in range(1, len(p) - 2):
        p0, p1, p2, p3 = p[i - 1], p[i], p[i + 1], p[i + 2]
        for t in np.linspace(0, 1, samples_per_seg, endpoint=False):
            t2, t3 = t * t, t * t * t
            out.append(
                0.5
                * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3)
            )
    out.append(p[-2])
    return np.array(out)


def _open_parametric(fn, tails: float, n: int = 1200) -> np.ndarray:
    """閉じた結び目の媒介変数表示を、xy の原点から最も遠い点で切り、両端を外向きに伸ばした開いた曲線。"""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    c = fn(t)
    t0 = t[np.argmax(np.hypot(c[:, 0], c[:, 1]))]
    gap = 0.12
    s = np.linspace(t0 + gap, t0 + 2 * np.pi - gap, n)
    body = fn(s)
    r0 = fn(np.array([t0]))[0]
    radial = np.array([r0[0], r0[1], 0.0]) / np.hypot(r0[0], r0[1])
    tang = np.array([-radial[1], radial[0], 0.0])
    k = np.linspace(0, 1, 60)[1:, None]
    head = body[0] + k[::-1] * tails * (radial + 0.6 * tang)
    tail = body[-1] + k * tails * (radial - 0.6 * tang)
    return np.vstack([head, body, tail])


def overhand_curve(tails: float = 4.0) -> np.ndarray:
    """固結び (開いた三葉結び)。両端は外の面。"""
    return _open_parametric(
        lambda t: np.stack([np.sin(t) + 2 * np.sin(2 * t), np.cos(t) - 2 * np.cos(2 * t), -np.sin(3 * t)], 1), tails
    )


def figure8_curve(tails: float = 4.0) -> np.ndarray:
    """8 の字結び (開いた 4_1)。両端は外の面。"""
    return _open_parametric(
        lambda t: np.stack(
            [(2 + np.cos(2 * t)) * np.cos(3 * t), (2 + np.cos(2 * t)) * np.sin(3 * t), np.sin(4 * t)], 1
        ),
        tails,
    )


# もやい結び: 立ち側 (頭の端) を上から下ろし、小さな輪 (穴) を作り (輪の側が立ち側の上)、大きな輪を経た先端が
# 穴を下から上へ通り、立ち側の後ろ (下) を回って、穴を上から下へ戻る。z = +1 は上、−1 は下を通す制御点。
_BOWLINE = [
    (0, 14, 0),
    (0, 9, 0),
    (0, 5.5, 0),
    (0, 3.6, -1),
    (0, 3.0, 0),  # 立ち側。C1 で輪の戻りの下
    (-1.6, 1.0, 0),
    (-1.7, -2.0, 0),
    (0, -3.2, 0),
    (1.8, -2.4, 0),  # 小さな輪 (穴)
    (2.6, 0.0, 0),
    (1.6, 2.6, 0),
    (0, 3.6, 1),
    (-1.5, 4.4, 0),  # 輪の戻りが C1 で立ち側の上
    (-3.5, 4.0, 0),
    (-4.8, 0, 0),
    (-5.2, -6, 0),
    (0, -12, 0),
    (5.2, -6, 0),
    (4.6, -2.0, 0),  # 大きな輪
    (2.3, -1.6, -1),
    (1.0, 0.2, 0),
    (0.8, 3.0, 1),
    (1.0, 5.0, 0),
    (1.3, 7.0, 0),  # 穴を下から上へ
    (0, 8.0, -1),
    (-1.3, 7.0, 0),
    (-1.2, 5.6, 0),
    (-1.0, 4.25, 1),
    (-0.9, 2.0, 1),  # 立ち側の後ろ -> 穴へ上から
    (-0.4, 0.0, 0),
    (-0.4, -3.1, -1),
    (-0.5, -5.5, 0),
    (-0.6, -8.0, 0),  # 穴の下へ抜けて大きな輪の中
]


def bowline_curve(variant: str = "standard") -> np.ndarray:
    """もやい結び。`variant="collar_front"` は先端を立ち側の前 (上) に回した、結べていない形 (比較用)。"""
    pts = np.array(_BOWLINE, dtype=float)
    # 交差させる制御点が相手の紐の節点と xy で重なると、1 つの交差を節点の両側で 2 回数える (`segment_intersections`
    # は内分比 [0, 1] の両端を含む)。立ち側 (制御点 0〜4) だけを少しずらして重なりを避ける。
    pts[:5, 0] += 0.07
    if variant == "collar_front":
        pts[24, 2] = 1.0
    elif variant != "standard":
        raise ValueError(variant)
    return _catmull_rom(pts)


def mirror(curve: np.ndarray) -> np.ndarray:
    """z を反転した鏡像。"""
    return curve * np.array([1.0, 1.0, -1.0])


def reverse(curve: np.ndarray) -> np.ndarray:
    """頭と尻尾を入れ替える (p-data は始点側から番号を振るので別の文字列になる)。"""
    return curve[::-1].copy()


def p_data_of(curve: np.ndarray) -> str:
    return tp.polyline_to_p_data(torch.tensor(curve, dtype=torch.float64), short=True)


def resample(curve: np.ndarray, num_nodes: int) -> np.ndarray:
    return tp.resample_polyline(torch.tensor(curve, dtype=torch.float64), num_nodes).numpy()


def min_clearance(curve: np.ndarray, skip_arc: float) -> float:
    """弧長で `skip_arc` 以上離れた点どうしの最小距離 (ロープの太さを入れたときに、その直径まで縮められる目安)。"""
    seg = np.linalg.norm(np.diff(curve, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    d = np.linalg.norm(curve[:, None] - curve[None], axis=-1)
    far = np.abs(s[:, None] - s[None]) >= skip_arc
    return float(d[far].min())


def arc_length(curve: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(curve, axis=0), axis=1).sum())


# ---------------------------------------------------------------- グラフ


def forward_distances(max_crossings: int) -> tuple[list[AbstractState], list[int], list[list[tuple[int, tuple, int]]]]:
    """自明な状態からの BFS の段数 (交差を増やす手 R1 / R2 / cross だけ。`knot_graph._enumerate_moves` を再利用)。"""
    states, moves = kg._enumerate_moves(max_crossings)
    dist = [-1] * len(states)
    dist[0] = 0
    q = deque([0])
    while q:
        i = q.popleft()
        for _, _, j in moves[i]:
            if dist[j] < 0:
                dist[j] = dist[i] + 1
                q.append(j)
    return states, dist, moves


def pruned_search(target: str) -> dict:
    """目標の p-data の「交差を間引いた図」だけを広げる BFS。

    交差を増やす手は既にある交差の上下・符号・紐の上の順を変えないので、自明な状態から `target` への道の途中の
    状態は必ず `target` の交差の部分集合の図になる。部分集合は 2^c 個しかないので、交差 6〜8 個の目標でも
    全列挙せずに最短手数が厳密に出る。
    """
    cr = crossings_of(target)
    allowed = set()
    for r in range(len(cr) + 1):
        for sub in itertools.combinations(cr, r):
            allowed.add(subcode(target, {p for pq in sub for p in pq}))
    start = AbstractState()
    key = kg._state_key
    seen = {key(start): 0}
    q = deque([start])
    best = None
    parent: dict = {key(start): None}
    hit_key = None
    while q:
        s = q.popleft()
        d = seen[key(s)]
        if best is not None and d >= best:
            continue
        for kind, params, child in kg.generate_moves(s):
            pd = child.short_p_data
            if pd not in allowed:
                continue
            k = key(child)
            if k in seen:
                continue
            seen[k] = d + 1
            parent[k] = (key(s), kg.MOVE_NAMES[kind], s.short_p_data)
            if pd == target and best is None:
                best = d + 1
                hit_key = k
            q.append(child)
    path = []
    k = hit_key
    while k is not None and parent.get(k) is not None:
        pk, name, pd_before = parent[k]
        path.append((pd_before, name))
        k = pk
    return {
        "crossings": len(cr),
        "subsets": len(allowed),
        "states_visited": len(seen),
        "distance": best,
        "moves": [name for _, name in path[::-1]],
        "path_p_data": [pd for pd, _ in path[::-1]] + ([target] if best else []),
    }


GRAPH_SIZES_KNOWN = {0: 1, 1: 4, 2: 40, 3: 528, 4: 7968}
"""交差 c ちょうどの状態の数 (`tests/test_knot_graph.py` が固定)。"""


def log_growth(sizes: dict[int, int]) -> float:
    """交差 1 つ増えるごとの状態の数の比の幾何平均 (最後の 2 段)。"""
    ks = sorted(sizes)
    return math.sqrt((sizes[ks[-1]] / sizes[ks[-2]]) * (sizes[ks[-2]] / sizes[ks[-3]]))
