"""高レベルの位相グラフ (R1 / R2 / cross) と、固結びまでの BFS 距離。v3 で追加。

TWISTED / TWISTED-RL の高レベル計画は、p-data を節点、Reidemeister 移動
(R1 / R2 / cross) を辺とするグラフを BFS する。ここでは同じグラフを
**交差 4 個以下** について全列挙し、各 p-data から「R2 除去後に固結びを含む
状態」までの最短手数 `d` を表にする。v3 の整形報酬はこれをポテンシャルに使う
(`paper/Report.md` §7 P1-3):

    Φ(s) = max( 1 − d(s) / d(自明),  R2 除去後のラダーの段 / 3 )

## なぜ「交差 4 個・R2 除去前の p-data・ゴールは R2 除去後の固結び」なのか

ゲート 1 (`next_action.md` §4.8.10 の 5) で、台本の cover の手の直後
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


def generate_next(state: AbstractState) -> Iterator[AbstractState]:
    """1 回の移動で到達できる状態をすべて返す。移植元 `BFS.generate_next` と同じ順序・範囲。"""
    n = state.pts
    for over_idx in range(n + 1):
        for under_idx in range(n + 1):
            for sign in (-1, 1):
                new = state.copy()
                if new.cross(over_idx, under_idx, sign):
                    yield new
    for idx in range(n + 1):
        for sign in (1, -1):
            for left in (1, -1):
                new = state.copy()
                if new.Reide1(idx, sign, left):
                    yield new
    for over_idx in range(n + 1):
        for under_idx in range(n + 1):
            for left in (-1, 1):
                for obu in (-1, 1):
                    new = state.copy()
                    if new.Reide2(over_idx, under_idx, left=left, over_before_under=obu):
                        yield new


@functools.lru_cache(maxsize=None)
def enumerate_graph(max_crossings: int = MAX_GRAPH_CROSSINGS) -> tuple[tuple[str, ...], tuple[tuple[int, ...], ...]]:
    """自明な状態から BFS し、交差 `max_crossings` 個以下の状態と辺を全列挙する。

    Returns:
        `(p_data, children)`。`p_data[i]` は状態 i の short p-data
        (自明な状態は空文字列)、`children[i]` は状態 i から 1 手で行ける状態の番号。
        同じ p-data でも半辺の繋がり (面の構造) が違えば別の状態として数える
        (移植元の `AbstractState.__eq__` と同じ。実測では交差 4 個まで 1 対 1)。
    """
    start = AbstractState()
    states = [start]
    index = {_state_key(start): 0}
    children: list[list[int]] = [[]]
    head = 0
    while head < len(states):
        for new in generate_next(states[head]):
            if new.pts > 2 * max_crossings:  # pts は交点の数 (交差 1 つで 2 点)
                continue
            key = _state_key(new)
            j = index.get(key)
            if j is None:
                j = len(states)
                index[key] = j
                states.append(new)
                children.append([])
            children[head].append(j)
        head += 1
    return tuple(s.short_p_data for s in states), tuple(tuple(c) for c in children)


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


_POTENTIAL_CACHE: dict[str, BfsPotential] = {}


def get_bfs_potential(device: torch.device | str) -> BfsPotential:
    """デバイスごとに 1 つだけ作って使い回す (BFS の列挙は `lru_cache` 済み)。"""
    key = str(device)
    if key not in _POTENTIAL_CACHE:
        _POTENTIAL_CACHE[key] = BfsPotential(device)
    return _POTENTIAL_CACHE[key]
