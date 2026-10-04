"""高レベルの位相グラフ (R1 / R2 / cross) と、固結びまでの BFS 距離。v3 で追加。

TWISTED / TWISTED-RL の高レベル計画は、p-data を節点、Reidemeister 移動
(R1 / R2 / cross) を辺とするグラフを BFS する。ここでは同じグラフを
**交差 4 個以下** について全列挙し、各 p-data から「R2 除去後に固結びを含む
状態」までの最短手数 `d` を表にする。v3 の整形報酬はこれをポテンシャルに使う
(`paper/Report_Paper.md` §7 P1-3):

    Φ(s) = max( 1 − d(s) / d(自明),  R2 除去後のラダーの段 / 3 )

## なぜ「交差 4 個・R2 除去前の p-data・ゴールは R2 除去後の固結び」なのか

ゲート 1 (`docs/experiments/2026-09-27_gate1.md` の「結論」の 5。理由は `docs/decisions/0010-stage-granularity-bfs.md`) で、台本の cover の手の直後
(`1U4+_2O5+_3O6-_4O1+_5U2+_6U3-`) は R2 除去すると輪 1 つ (C1) と同じになるのに、
そこから tuck 1 手で約 7 割が固結びになることが分かった。R2 除去後のラダーでは
「cover を打っても何も起きていない」ように見える。

最初は R2 除去後の p-data を交差 3 個まで BFS したが、それだと固結びに届く
p-data は 13 個 (自明・輪 4・C2 4・固結び 4) だけで、**経路上ではラダーと完全に
同じ値** になり何も足さなかった (2026-09-28)。cover 後の状態は、R2 除去前の
p-data で見て「1 手 (tuck) で、R2 除去すると固結びになる 4 交差の状態へ行ける」
ので、**除去前の p-data で、4 交差まで** 見ないと拾えない。

## 列挙の結果 (2026-09-28 実測。`tests/test_knot_graph.py` で固定)

交差 0 / 1 / 2 / 3 / 4 の状態は 1 / 4 / 40 / 528 / 7968 個 (列挙 約 20 s。
状態と p-data は 1 対 1)。ゴール (R2 除去後に固結びを含む) は 172 個。

| 状態の例 | ラダー (R2 除去後) | d | BFS の Φ |
|---|---|---|---|
| 自明 | 0 | 3 | 0 |
| 輪 1 つ (4 通り) | 1/3 | 2 | 1/3 |
| 輪の上に先端を置いただけ (OO など、2 交差 36 通り) | 0 | 2 | **1/3** |
| 本物の C2 (4 通り) | 2/3 | 1 | 2/3 |
| cover 後 (3 交差・R2 後 C1、16 通り) | 1/3 | 1 | **2/3** |
| 偽の C2 (輪を跨いだだけ) | 1/3 | 届かない | (ラダーの 1/3) |
| 固結び | 1 | 0 | 1 |

輪 1 つの 4 通りはどれも d = 2 で、Report §5.4 の「次の手が打てる C1 / 打てない C1」の
差は p-data には現れない (差は幾何のほうにある)。

## ラダーとの max を取る理由

探索は交差 4 個で打ち切るので、4 交差の状態は「ゴールか、届かないか」の
どちらかになる (5 交差を経由する道は数えない)。また尻尾に余分な輪が付いた
状態など、表の範囲外の状態も多い。BFS の値だけを使うと、これらがラダーより
低く (0 に) なり、「ねじれを 1 つ足すと 0、2 つ足すとラダーの値」のような逆転が
生じる。**BFS は加点にだけ使い、ラダーより下げない** (`BfsPotential`)。

## 実行時の引き方 (テンソル層のみ)

p-data 文字列を作ると 4096 環境で 94 ms かかる (`state_2_topology.py`) ので、
`IntersectionBatch` から直接 int64 のキーを作って表を引く (`batch_keys`)。
キーは交差数 (3 bit) と、紐の始点側から数えた各点の
「相手の点の番号 (4 bit)・上か (1 bit)・符号が + か (1 bit)」を詰めたもの (最大 51 bit)。
p-data 文字列からも同じキーを作れる (`p_data_key`)。両者の一致は
`tests/test_knot_graph.py` でランダムなロープについて検証している。

## pick&place の制約を辺に入れない理由

Report §7 P1-3 は「掴んだ側が上になる」制約を辺の条件に入れる案を挙げていた。
しかしこの制約は **どちらの区間を掴むか** を決めるだけで、位相の辺は減らない。
先端が下になる cross も「輪の側を掴んで先端の上に被せる」手 (cover) で実行できる
(Report §5.4、ゲート 1)。したがって p-data 上の距離はこの制約では変わらない。
"""

from __future__ import annotations

import functools
import re
from collections import deque
from collections.abc import Iterator
from typing import NamedTuple

import torch

from .intersections import IntersectionBatch
from .knot_detect import CLASS_OVERHAND, reduce_r2, topology_class
from .representation import AbstractState

MAX_GRAPH_CROSSINGS = 4
"""表を作る交差数の上限。キーのビット幅 (相手の点番号 4 bit) もこの上限に合わせてある。"""

OVERHAND_P_DATA = (
    "1U4+_2O5+_3U6+_4O1+_5U2+_6O3+",
    "1O4+_2U5+_3O6+_4U1+_5O2+_6U3+",
    "1U4-_2O5-_3U6-_4O1-_5U2-_6O3-",
    "1O4-_2U5-_3O6-_4U1-_5O2-_6U3-",
)
"""固結びの p-data 4 通り (キラリティ 2 x 走査方向 2)。`knot_detect.py` の docstring と同じ。"""


# ---------------------------------------------------------------- p-data <-> 交差リスト

_P_DATA_TOKEN = re.compile(r"^(\d+)([OU])(\d+)([+-])$")


def _parse(short_p_data: str) -> list[tuple[int, bool, int, bool]]:
    """short p-data -> 各点の `(番号, 上か, 相手の番号, 符号が + か)`。形式が壊れていれば ValueError。"""
    if short_p_data == "":
        return []
    out = []
    for i, tok in enumerate(short_p_data.split("_")):
        m = _P_DATA_TOKEN.match(tok)
        if m is None or int(m.group(1)) != i + 1:
            raise ValueError(f"p-data を解釈できない: {short_p_data!r}")
        out.append((i + 1, m.group(2) == "O", int(m.group(3)), m.group(4) == "+"))
    if len(out) % 2:
        raise ValueError(f"p-data の点の数が奇数: {short_p_data!r}")
    return out


def p_data_to_lists(short_p_data: str) -> tuple[list[tuple[int, int, int, int]], list[tuple[int, int]]]:
    """short p-data -> `IntersectionBatch.from_lists` に渡せる `(交差, 交点番号)`。

    セグメント番号は位相に効かないので、交点番号をそのまま入れる (スロットの並びが
    紐の始点側からの順になる)。
    """
    items, orders = [], []
    for i, over, j, positive in _parse(short_p_data):
        if i < j:
            items.append((i, j, 1 if over else -1, 1 if positive else -1))
            orders.append((i, j))
    return items, orders


# ---------------------------------------------------------------- グラフの列挙


def _state_key(state: AbstractState) -> tuple:
    """`AbstractState.__eq__` と同じ基準 (点の上下・符号と、半辺の next / prev) のキー。"""
    return (
        tuple((p.over, p.under, p.sign) for p in state.points),
        tuple((e.next, e.prev) for e in state.edges),
    )


MOVE_R1 = 0
MOVE_R2 = 1
MOVE_CROSS = 2
MOVE_END_CROSS = 3
"""移動の種類の番号。v6 の観測の one-hot はこの順 (R1 / R2 / cross / 複合「端の置き直し」)。"""

MOVE_NAMES = ("R1", "R2", "cross", "end_cross")


def generate_moves(
    state: AbstractState,
) -> Iterator[tuple[int, tuple[int, ...], AbstractState]]:
    """1 回の移動で到達できる状態を、移動の種類とパラメータ付きで返す。v6 で追加。

    `generate_next` と **同じ順序・同じ範囲** (`generate_next` はこれの子の状態だけを返す)。
    パラメータは `AbstractState` の引数の順:

    | 種類 | パラメータ |
    |---|---|
    | `MOVE_CROSS` | `(over_idx, under_idx, sign)` |
    | `MOVE_R1` | `(idx, sign, left)` |
    | `MOVE_R2` | `(over_idx, under_idx, left, over_before_under)` |

    区間 idx は点 idx と点 idx+1 の間 (点 0 と点 pts+1 は両端のダミー)。
    """
    n = state.pts
    for over_idx in range(n + 1):
        for under_idx in range(n + 1):
            for sign in (-1, 1):
                new = state.copy()
                if new.cross(over_idx, under_idx, sign):
                    yield MOVE_CROSS, (over_idx, under_idx, sign), new
    for idx in range(n + 1):
        for sign in (1, -1):
            for left in (1, -1):
                new = state.copy()
                if new.Reide1(idx, sign, left):
                    yield MOVE_R1, (idx, sign, left), new
    for over_idx in range(n + 1):
        for under_idx in range(n + 1):
            for left in (-1, 1):
                for obu in (-1, 1):
                    new = state.copy()
                    if new.Reide2(over_idx, under_idx, left=left, over_before_under=obu):
                        yield MOVE_R2, (over_idx, under_idx, left, obu), new


def generate_next(state: AbstractState) -> Iterator[AbstractState]:
    """1 回の移動で到達できる状態をすべて返す。移植元 `BFS.generate_next` と同じ順序・範囲。"""
    for _, _, new in generate_moves(state):
        yield new


@functools.lru_cache(maxsize=None)
def _enumerate_moves(max_crossings: int) -> tuple[list[AbstractState], list[list[tuple[int, tuple[int, ...], int]]]]:
    """自明な状態から BFS し、状態と「移動の種類・パラメータ・子の番号」を全列挙する。

    `enumerate_graph` (v3) と v6 の辺の表 (`shortest_edges`) が共有する。列挙は約 20 s
    かかるので、両者が別々に BFS しないよう 1 回にまとめた。
    """
    start = AbstractState()
    states = [start]
    index = {_state_key(start): 0}
    moves: list[list[tuple[int, tuple[int, ...], int]]] = [[]]
    head = 0
    while head < len(states):
        for kind, params, new in generate_moves(states[head]):
            if new.pts > 2 * max_crossings:  # pts は交点の数 (交差 1 つで 2 点)
                continue
            key = _state_key(new)
            j = index.get(key)
            if j is None:
                j = len(states)
                index[key] = j
                states.append(new)
                moves.append([])
            moves[head].append((kind, params, j))
        head += 1
    return states, moves


@functools.lru_cache(maxsize=None)
def enumerate_graph(max_crossings: int = MAX_GRAPH_CROSSINGS) -> tuple[tuple[str, ...], tuple[tuple[int, ...], ...]]:
    """自明な状態から BFS し、交差 `max_crossings` 個以下の状態と辺を全列挙する。

    Returns:
        `(p_data, children)`。`p_data[i]` は状態 i の short p-data
        (自明な状態は空文字列)、`children[i]` は状態 i から 1 手で行ける状態の番号。
        同じ p-data でも半辺の繋がり (面の構造) が違えば別の状態として数える
        (移植元の `AbstractState.__eq__` と同じ。実測では交差 4 個まで 1 対 1)。
    """
    states, moves = _enumerate_moves(max_crossings)
    return tuple(s.short_p_data for s in states), tuple(tuple(j for _, _, j in ms) for ms in moves)


def is_goal(p_data: tuple[str, ...] | list[str]) -> list[bool]:
    """R2 除去後に固結びを含むか (= v3 の成功判定 `knot_tied` と同じ基準)。"""
    batch = IntersectionBatch.from_lists([p_data_to_lists(pd) for pd in p_data])
    return (topology_class(reduce_r2(batch)) == CLASS_OVERHAND).tolist()


@functools.lru_cache(maxsize=None)
def distance_to_overhand(max_crossings: int = MAX_GRAPH_CROSSINGS) -> dict[str, int]:
    """p-data ごとの、ゴール (R2 除去後に固結びを含む状態) までの最短手数。届かない p-data は含めない。

    同じ p-data の状態が複数ある (面の構造だけ違う) ときは最小値を取る。
    実機のロープから得られるのは p-data だけで、面の構造は区別できないため。
    """
    p_data, children = enumerate_graph(max_crossings)
    parents: list[list[int]] = [[] for _ in p_data]
    for i, cs in enumerate(children):
        for j in cs:
            parents[j].append(i)

    dist = [-1] * len(p_data)
    queue: deque[int] = deque()
    for i, goal in enumerate(is_goal(p_data)):
        if goal:
            dist[i] = 0
            queue.append(i)
    while queue:
        j = queue.popleft()
        for i in parents[j]:
            if dist[i] < 0:
                dist[i] = dist[j] + 1
                queue.append(i)

    out: dict[str, int] = {}
    for pd, d in zip(p_data, dist):
        if d >= 0 and (pd not in out or d < out[pd]):
            out[pd] = d
    return out


def potential_table(max_crossings: int = MAX_GRAPH_CROSSINGS) -> dict[str, float]:
    """p-data -> Φ = 1 − d / d(自明な状態)。ゴールに届かない p-data は含めない。"""
    dist = distance_to_overhand(max_crossings)
    d_max = dist[""]
    return {pd: 1.0 - d / d_max for pd, d in dist.items()}


# ---------------------------------------------------------------- 最短の次の辺 (v6)


class GraphEdge(NamedTuple):
    """高レベルの辺 1 本。v6 で「次にたどる辺」として方策に渡す単位。

    区間は **今の状態の点番号** の組 `(a, b)` で持つ (点 a と点 b の間。0 と 2n+1 は両端の
    ダミー)。単純な辺では区間 idx が `(idx, idx + 1)`。複合の辺では、外した交差の 2 点を
    除いた点列で区間を数えるので `(a, b)` が隣り合わないことがある。こう持てば、
    実行時は単純な辺も複合の辺も「点 a のリンクから点 b のリンクまで」の同じ式でマスクに
    できる (`next_edge.py`)。

    Attributes:
        kind: `MOVE_R1` / `MOVE_R2` / `MOVE_CROSS` / `MOVE_END_CROSS`。
        over: 上を通る区間 `(a, b)`。R1 は輪を作る区間 (over と under に同じものを入れる)。
        under: 下を通る区間 `(a, b)`。
        sign: 新しい交差の符号 ±1。R2 は 0 (2 つの交差の符号は逆で、指定しない)。
        left: R1 / R2 の `left` ±1。cross は 0。
        obu: R2 の `over_before_under` ±1。それ以外は 0。
        removed: 複合の辺で外す交差の 2 点 `(p, q)` (今の状態の番号)。単純な辺は `(0, 0)`。
        child: 辺をたどった先の short p-data (複合なら最終の子)。
        params: 最後の 1 手の生のパラメータ (`generate_moves` の形。複合なら外した後の状態での番号)。
    """

    kind: int
    over: tuple[int, int]
    under: tuple[int, int]
    sign: int
    left: int
    obu: int
    removed: tuple[int, int]
    child: str
    params: tuple[int, ...]


def remove_crossing(short_p_data: str, point: int) -> tuple[str, tuple[int, int]]:
    """点 `point` を含む交差を外し、残りの点を番号を詰めて並べ直した p-data を返す。v6 で追加。

    端の点 (1 か 2n) の交差を外すのは、その端を交差の手前まで引き戻すこと (cross の逆) で、
    位相の上では常にできる。pick&place で端を掴んで持ち上げると、その端に乗っていた
    交差がいったん消えてから置き直されるので、台本の tuck はこの形になる
    (`shortest_edges` の docstring)。

    Returns:
        `(新しい p-data, 外した 2 点 (p, q) p < q)`。
    """
    points = _parse(short_p_data)
    partner = points[point - 1][2]
    removed = (min(point, partner), max(point, partner))
    renum = {}
    for i, *_ in points:
        if i not in removed:
            renum[i] = len(renum) + 1
    toks = []
    for i, over, j, positive in points:
        if i in removed:
            continue
        toks.append(f"{renum[i]}{'O' if over else 'U'}{renum[j]}{'+' if positive else '-'}")
    return "_".join(toks), removed


def _segment(points: list[int], idx: int) -> tuple[int, int]:
    """点列 `points` (両端のダミー 0 と 2n+1 を含む) の区間 idx を、元の点番号の組にする。"""
    return points[idx], points[idx + 1]


def _simple_edge(kind: int, params: tuple[int, ...], child: str, points: list[int]) -> GraphEdge:
    if kind == MOVE_R1:
        idx, sign, left = params
        seg = _segment(points, idx)
        return GraphEdge(kind, seg, seg, sign, left, 0, (0, 0), child, params)
    if kind == MOVE_R2:
        over, under, left, obu = params
        return GraphEdge(
            kind,
            _segment(points, over),
            _segment(points, under),
            0,
            left,
            obu,
            (0, 0),
            child,
            params,
        )
    over, under, sign = params
    return GraphEdge(
        kind,
        _segment(points, over),
        _segment(points, under),
        sign,
        0,
        0,
        (0, 0),
        child,
        params,
    )


@functools.lru_cache(maxsize=None)
def shortest_edges(
    max_crossings: int = MAX_GRAPH_CROSSINGS,
) -> dict[str, tuple[GraphEdge, ...]]:
    """p-data ごとの「最短の次の辺」の候補。v6 の観測 (次にたどる辺のリンクマスク) が引く。

    候補は 2 種類 (並びは単純な辺が `generate_moves` の列挙順、その後に複合の辺):

    * **単純な辺**: R1 / R2 / cross のうち、子の `distance_to_overhand` が親より 1 小さいもの。
    * **複合の辺「端の置き直し」** (`MOVE_END_CROSS`): 頭の端 (点 1) か尻尾の端 (点 2n) の
      交差を 1 つ外し (`remove_crossing`)、**同じ端で** cross を 1 つ打つ。子の距離が親より
      小さいものだけ。同じ端に限るのは、これが「端を掴んで持ち上げ、置き直す」1 回の
      pick&place に当たるため (尻尾の交差を外して頭の端で cross するのは 2 手)。外した後の状態は列挙済みのグラフから p-data で引く (交差 4 個までは
      状態と p-data が 1 対 1)。グラフに無ければ候補にしない。

    ## なぜ複合の辺が要るのか

    グラフは交差を増やす移動しか持たない。台本 (`demos/overhand_script.py`) の cover 後の
    状態 `1U4+_2O5+_3O6-_4O1+_5U2+_6U3-` から、グラフの最短の辺は頭の端の cross 1 本だけで、
    子は交差 4 個の状態になる。しかし台本の tuck は、尻尾の端を持ち上げて最も端に近い
    交差 (3, 6) を外し (本物の C2 になる)、`cross(over=4, under=2)` で置き直す手で、
    交差 3 個の固結びに着く。これを候補に入れないと、方策に渡す目標が台本の手と食い違う。

    **BFS の距離 (`distance_to_overhand` / `BfsPotential`) は変えない。** 報酬は v5 と同じに
    保ち、変えるのは観測だけにするため (1 版 1 変数)。

    ## 実測 (2026-10-01)

    全状態で数えた最短の単純な辺の本数は d = 3: 8 本 (1 状態)、d = 2: 4 本 (32 状態) /
    6 本 (8 状態)、d = 1: 1 本 (16 状態) / 2 本 (76 状態) / 5 本 (16 状態)。
    自明な状態から最短の辺だけでたどれる状態に限ると d = 2 は 4 本 / 6 本が 4 状態ずつ、
    d = 1 は 1 本 16 / 2 本 12 / 5 本 8 状態。

    Returns:
        `{p-data: 候補のタプル}`。d ≥ 1 の p-data だけを含む (ゴール・届かない状態・
        交差が上限を超える状態は含まない)。
    """
    states, moves = _enumerate_moves(max_crossings)
    dist = distance_to_overhand(max_crossings)
    p_data = [s.short_p_data for s in states]
    index = {pd: i for i, pd in enumerate(p_data)}

    out: dict[str, tuple[GraphEdge, ...]] = {}
    for i, pd in enumerate(p_data):
        d = dist.get(pd)
        if d is None or d == 0 or pd in out:
            continue
        n_pts = states[i].pts
        points = list(range(n_pts + 2))
        edges = [
            _simple_edge(kind, params, p_data[j], points)
            for kind, params, j in moves[i]
            if dist.get(p_data[j], -1) == d - 1
        ]
        # 外す交差 -> その交差が乗っている端 (外した後の状態での端の区間の番号)。
        # 頭と尻尾が同じ交差に乗っている (点 1 と点 2n が組) ときは、どちらの端で置き直してもよい。
        removals: dict[tuple[int, int], set[int]] = {}
        for end, end_seg in ((1, 0), (n_pts, n_pts - 2)):
            if n_pts:
                removals.setdefault(remove_crossing(pd, end)[1], set()).add(end_seg)
        for removed, end_segs in removals.items():
            mid_pd, _ = remove_crossing(pd, removed[0])
            if mid_pd not in index:
                continue
            kept = [0] + [p for p in range(1, n_pts + 1) if p not in removed] + [n_pts + 1]
            for kind, params, j in moves[index[mid_pd]]:
                if kind != MOVE_CROSS or dist.get(p_data[j], d) >= d:
                    continue
                over, under, sign = params
                if over not in end_segs and under not in end_segs:
                    continue  # 外した端とは別の端で cross するのは「置き直し」ではない (2 手になる)
                edges.append(
                    GraphEdge(
                        MOVE_END_CROSS,
                        _segment(kept, over),
                        _segment(kept, under),
                        sign,
                        0,
                        0,
                        removed,
                        p_data[j],
                        params,
                    )
                )
        out[pd] = tuple(edges)
    return out


# ---------------------------------------------------------------- キー

_KEY_COUNT_BITS = 3
_KEY_PARTNER_BITS = 4
_KEY_POINT_BITS = _KEY_PARTNER_BITS + 2


def p_data_key(short_p_data: str) -> int:
    """short p-data (`1U2+_2O1+`) -> int64 のキー。`batch_keys` と同じ値になる。"""
    points = _parse(short_p_data)
    count = len(points) // 2
    if count > MAX_GRAPH_CROSSINGS:
        raise ValueError(f"p-data の交差数が範囲外: {short_p_data!r}")
    key = count
    for i, over, j, positive in points:
        code = j | (int(over) << _KEY_PARTNER_BITS) | (int(positive) << (_KEY_PARTNER_BITS + 1))
        key |= code << (_KEY_COUNT_BITS + _KEY_POINT_BITS * (i - 1))
    return key


def batch_keys(batch: IntersectionBatch, max_crossings: int = MAX_GRAPH_CROSSINGS) -> torch.Tensor:
    """`IntersectionBatch` -> キー `(num_envs,)` int64。交差数が `max_crossings` を超える env は −1。

    交差 m は点 `p = order[m, 0]` (seg_i 側) と `q = order[m, 1]` (seg_j 側) を持ち、
    `data[m, 2] == 1` なら seg_i 側 (点 p) が上、`data[m, 3]` が符号。
    """
    count = batch.count
    num_slots = min(batch.max_crossings, max_crossings)
    if num_slots == 0:
        return torch.where(count == 0, torch.zeros_like(count), torch.full_like(count, -1))

    valid = batch.valid_mask[:, :num_slots]
    ones = torch.ones_like(batch.order[:, :num_slots, 0])
    p = torch.where(valid, batch.order[:, :num_slots, 0], ones)
    q = torch.where(valid, batch.order[:, :num_slots, 1], ones)
    p_over = (batch.data[:, :num_slots, 2] == 1).long()
    positive = (batch.data[:, :num_slots, 3] > 0).long()

    flags = positive << (_KEY_PARTNER_BITS + 1)
    code_p = q | (p_over << _KEY_PARTNER_BITS) | flags
    code_q = p | ((1 - p_over) << _KEY_PARTNER_BITS) | flags
    shift_p = _KEY_COUNT_BITS + _KEY_POINT_BITS * (p - 1)
    shift_q = _KEY_COUNT_BITS + _KEY_POINT_BITS * (q - 1)
    contrib = (code_p << shift_p) + (code_q << shift_q)
    key = count + torch.where(valid, contrib, torch.zeros_like(contrib)).sum(dim=-1)
    return torch.where(count <= max_crossings, key, torch.full_like(key, -1))


# ---------------------------------------------------------------- 実行時の表


class BfsPotential:
    """`Φ = max(BFS 距離の値, R2 除去後のラダーの段 / 3)` を GPU 上で引く。

    BFS の表は **R2 除去前** の p-data で引き、表に無い状態 (届かない・交差 5 個以上)
    はラダーの値になる (モジュール docstring の「ラダーとの max を取る理由」)。
    """

    def __init__(self, device: torch.device | str = "cpu"):
        table = potential_table()
        items = sorted((p_data_key(pd), phi) for pd, phi in table.items())
        self.keys = torch.tensor([k for k, _ in items], dtype=torch.int64, device=device)
        self.values = torch.tensor([v for _, v in items], dtype=torch.float32, device=device)

    def bfs_only(self, raw_batch: IntersectionBatch) -> torch.Tensor:
        """BFS の表の値だけ `(num_envs,)`。表に無ければ 0。ログ・テスト用。"""
        keys = batch_keys(raw_batch)
        idx = torch.searchsorted(self.keys, keys).clamp(max=self.keys.numel() - 1)
        found = (self.keys[idx] == keys) & (keys >= 0)
        return torch.where(found, self.values[idx], torch.zeros_like(self.values[idx]))

    def __call__(self, raw_batch: IntersectionBatch, reduced_batch: IntersectionBatch | None = None) -> torch.Tensor:
        """`(num_envs,)` float32 のポテンシャル。

        Args:
            raw_batch: R2 除去 **前** の交差 (表を引く)。
            reduced_batch: R2 除去後の交差 (ラダー側)。省略すると `raw_batch` から計算する。
        """
        if reduced_batch is None:
            reduced_batch = reduce_r2(raw_batch)
        ladder = topology_class(reduced_batch).float() / 3.0
        return torch.maximum(self.bfs_only(raw_batch), ladder)

    def stage(self, raw_batch: IntersectionBatch, reduced_batch: IntersectionBatch | None = None) -> torch.Tensor:
        """段 `(num_envs,)` int64 = `round(3 Φ)`。0 = 自明 〜 `STAGE_GOAL` = 固結び。v4 で追加。

        v4 が「交差数ごとのエージェント」(TWISTED-RL-C) を分ける単位。固結びへの正規の経路
        (交差 0 → 1 → 2 → 3) に対応する: 輪 1 つは 1、本物の C2 と cover 後は 2。
        生の交差数を使わないのは、cover 後 (生 3・R2 除去後 1) や偽の C2 (生 3) が
        固結びまでの近さと合わないため。Φ は d_max = 3 なのでちょうど k/3 の値を取る。
        """
        return (self(raw_batch, reduced_batch) * STAGE_GOAL).round().long()


STAGE_GOAL = 3
"""`BfsPotential.stage` の最上段 (固結び)。v4 のエージェントは段 0〜2 の 3 つ。"""


_POTENTIAL_CACHE: dict[str, BfsPotential] = {}


def get_bfs_potential(device: torch.device | str) -> BfsPotential:
    """デバイスごとに 1 つだけ作って使い回す (BFS の列挙は `lru_cache` 済み)。"""
    key = str(device)
    if key not in _POTENTIAL_CACHE:
        _POTENTIAL_CACHE[key] = BfsPotential(device)
    return _POTENTIAL_CACHE[key]
