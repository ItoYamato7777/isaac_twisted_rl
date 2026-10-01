"""「次にたどる高レベルの辺」を、リンクのマスクとして GPU 上で作る (テンソル層)。v6 で追加。

TWISTED-RL は下位の方策に、次にたどる辺を固定長のベクトル ρ(g) で渡している
(`twisted_rl/exploration/preprocessing/preprocessor.py` の `preprocess_high_level_action`):
under 区間と over 区間のリンクの 0/1 マスク、`[left, over_first, over_before_under, sign]`、
移動の種類。v5 までの観測には「区間 k がどのリンクか」が無く、方策は輪や先端の位置を
リンク位置から自力で見つける必要があった。v6 はこれを観測に足す。

## 区間 -> リンク

`polyline_from_link_centers` の節点は N+1 個なので、**ポリラインのセグメント k = リンク k**。
`IntersectionBatch` の `seg_i` / `seg_j` はそのままリンク番号になる (細分したときは
`subdivide` で割る)。点 p (1..2n) が乗っているリンクを L(p) とし、L(0) = 0、L(2n+1) = N−1 と
置いて、区間 `(a, b)` のリンク範囲を `[L(a), L(b)]` (両端を含む) とする。TWISTED-RL の
`get_link_segments` と同じ取り方 (交差の乗ったリンクは両側の区間に入る)。

辺は `knot_graph.GraphEdge` の点番号の組で持っているので、単純な辺も複合の辺
(外した交差の 2 点を飛ばした区間) も同じ式でマスクになる。

## 実行時の引き方

p-data 文字列は作らない (4096 環境で 94 ms)。`BfsPotential` と同じく `batch_keys` の
int64 キーを、`p_data_key` で並べた表から `searchsorted` で引く。表に無い状態
(交差 5 個以上・ゴールに届かない・ゴールそのもの) は有効フラグ 0 になる。

## 辺の選び方 (`RULES`)

候補が複数あるとき (自明な状態で 8 本、輪で 4 本など) にどれを渡すか:

* `short_end` (既定): 端の区間を使う候補のうち、その端の区間のリンク数が少ないものを優先し、
  同点なら種類の優先順位 R1 > R2 > 複合 > cross、それでも同点なら列挙順。端を使わない候補は
  最良の端の候補と同点として扱う。台本 (`demos/overhand_script.py`) が短い側 (尻尾) の端を
  使い、輪からは cover (R2) を多く選ぶことに合わせた仮説。
* `first`: 列挙順の最初 (比較用)。
* `random`: 候補から一様に選ぶ (比較用)。
"""

from __future__ import annotations

import torch

from .intersections import IntersectionBatch
from .knot_graph import (
    MAX_GRAPH_CROSSINGS,
    MOVE_CROSS,
    MOVE_END_CROSS,
    MOVE_R1,
    MOVE_R2,
    batch_keys,
    p_data_key,
)
from .knot_graph import shortest_edges as _shortest_edges

RULES = ("short_end", "first", "random")
"""`NextEdgeTable.select` の選択規則。"""

NUM_MOVE_KINDS = 4
"""種類の one-hot の幅 (R1 / R2 / cross / 複合)。`knot_graph.MOVE_*` の番号順。"""

_KIND_PRIORITY = {MOVE_R1: 0, MOVE_R2: 1, MOVE_END_CROSS: 2, MOVE_CROSS: 3}
"""`short_end` の同点を崩す種類の優先順位 (小さいほど優先)。"""


def feature_dim(num_links: int) -> int:
    """観測ベクトルの次元 `2N + 8` = over マスク N + under マスク N + 種類 4 + sign/left/obu + 有効。"""
    return 2 * num_links + NUM_MOVE_KINDS + 4


def point_links(batch: IntersectionBatch, num_links: int, subdivide: int = 1) -> torch.Tensor:
    """各点が乗っているリンク L(p) `(num_envs, 2 * MAX_GRAPH_CROSSINGS + 2)` int64。

    列 p が点 p。L(0) = 0、点 2n+1 以降は N−1 (尻尾の端のダミー)。交差が表の上限を超える
    env の値は意味を持たない (表に無いので使われない)。
    """
    num_envs = batch.num_envs
    num_points = 2 * MAX_GRAPH_CROSSINGS + 2
    device = batch.count.device
    # 最後の 1 列はゴミ捨て場 (無効な交差・上限を超える点番号をここへ書く)。
    out = torch.full((num_envs, num_points + 1), num_links - 1, dtype=torch.int64, device=device)
    out[:, 0] = 0
    if batch.max_crossings > 0:
        valid = batch.valid_mask
        pts = torch.cat([batch.order[..., 0], batch.order[..., 1]], dim=1)  # (B, 2M)
        links = torch.cat([batch.data[..., 0], batch.data[..., 1]], dim=1) // subdivide
        ok = torch.cat([valid, valid], dim=1) & (pts >= 1) & (pts < num_points)
        idx = torch.where(ok, pts, torch.full_like(pts, num_points))
        out.scatter_(1, idx, torch.where(ok, links, torch.zeros_like(links)))
    return out[:, :num_points]


class NextEdgeTable:
    """p-data ごとの最短の次の辺 (`knot_graph.shortest_edges`) を GPU 上の表にしたもの。

    表の行は `p_data_key` の昇順、列は候補 (`shortest_edges` の並び、足りない分は種類 −1)。
    """

    def __init__(
        self,
        device: torch.device | str = "cpu",
        max_crossings: int = MAX_GRAPH_CROSSINGS,
    ):
        table = _shortest_edges(max_crossings)
        items = sorted((p_data_key(pd), edges) for pd, edges in table.items())
        num_rows = len(items)
        num_cands = max(len(edges) for _, edges in items)
        self.num_cands = num_cands

        def full(value: int) -> list[list[int]]:
            return [[value] * num_cands for _ in range(num_rows)]

        kind, over_a, over_b, under_a, under_b = (
            full(-1),
            full(0),
            full(0),
            full(0),
            full(0),
        )
        sign, left, obu, child = full(0), full(0), full(0), full(-1)
        for r, (_, edges) in enumerate(items):
            for c, e in enumerate(edges):
                kind[r][c] = e.kind
                over_a[r][c], over_b[r][c] = e.over
                under_a[r][c], under_b[r][c] = e.under
                sign[r][c], left[r][c], obu[r][c] = e.sign, e.left, e.obu
                child[r][c] = p_data_key(e.child)

        def t(x: list) -> torch.Tensor:
            return torch.tensor(x, dtype=torch.int64, device=device)

        self.keys = t([k for k, _ in items])
        self.kind = t(kind)
        self.over_a, self.over_b = t(over_a), t(over_b)
        self.under_a, self.under_b = t(under_a), t(under_b)
        self.sign, self.left, self.obu = t(sign), t(left), t(obu)
        self.child_key = t(child)
        """候補の子 (複合なら最終の子) のキー。ログ「選んだ辺の子に着いたか」に使う。"""
        self.priority = torch.tensor(
            [_KIND_PRIORITY.get(k, 0) for k in range(NUM_MOVE_KINDS)],
            dtype=torch.int64,
            device=device,
        )

    # ------------------------------------------------------------------ 引く
    def lookup(self, batch: IntersectionBatch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """`(row, found, keys)`。`row` は表の行 (見つからなければ 0)、`keys` は `batch_keys`。"""
        keys = batch_keys(batch)
        row = torch.searchsorted(self.keys, keys).clamp(max=self.keys.numel() - 1)
        found = (self.keys[row] == keys) & (keys >= 0)
        return torch.where(found, row, torch.zeros_like(row)), found, keys

    def select(
        self,
        row: torch.Tensor,
        found: torch.Tensor,
        links: torch.Tensor,
        count: torch.Tensor,
        rule: str = "short_end",
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """候補の番号 `(num_envs,)` int64 を選ぶ。表に無い env は −1。

        Args:
            row / found: `lookup` の戻り値。
            links: `point_links` (今の形状での L(p))。`short_end` が端の区間のリンク数に使う。
            count: 交差数 `(num_envs,)` (尻尾の端のダミー点 2n+1 の番号を決める)。
            rule: `RULES` のどれか。
        """
        if rule not in RULES:
            raise ValueError(f"unknown rule {rule!r} (expected one of {RULES})")
        kind = self.kind[row]  # (B, C)
        exists = kind >= 0
        num_envs, num_cands = kind.shape
        order = torch.arange(num_cands, device=kind.device).expand(num_envs, -1)
        if rule == "first":
            score = order.clone()
        elif rule == "random":
            noise = torch.rand(kind.shape, device=kind.device, generator=generator)
            score = noise.argsort(dim=1).argsort(dim=1)  # 乱数の順位 (候補ごとに相異なる)
        else:
            end_len = self._end_lengths(row, links, count)  # (B, C)。端を使わなければ大きな値
            no_end = end_len >= _NO_END
            best = torch.where(exists & ~no_end, end_len, torch.full_like(end_len, _NO_END)).min(dim=1).values
            end_len = torch.where(no_end, best.unsqueeze(1), end_len)  # 端を使わない候補は最良と同点
            prio = self.priority[kind.clamp_min(0)]
            score = (end_len * NUM_MOVE_KINDS + prio) * num_cands + order
        big = torch.iinfo(torch.int64).max
        score = torch.where(exists, score, torch.full_like(score, big))
        cand = score.argmin(dim=1)
        return torch.where(found & exists.any(dim=1), cand, torch.full_like(cand, -1))

    def _end_lengths(self, row: torch.Tensor, links: torch.Tensor, count: torch.Tensor) -> torch.Tensor:
        """候補が使う端の区間のリンク数 `(B, C)`。両端を使えば小さいほう、端を使わなければ `_NO_END`。"""
        tail = (2 * count + 1).unsqueeze(1)
        lengths = []
        for a, b in (
            (self.over_a[row], self.over_b[row]),
            (self.under_a[row], self.under_b[row]),
        ):
            n = links.gather(1, b) - links.gather(1, a) + 1
            is_end = (a == 0) | (b == tail)
            lengths.append(torch.where(is_end, n, torch.full_like(n, _NO_END)))
        return torch.minimum(*lengths)

    # ------------------------------------------------------------------ 特徴
    def features(self, row: torch.Tensor, cand: torch.Tensor, links: torch.Tensor, num_links: int) -> torch.Tensor:
        """観測ベクトル `(num_envs, 2N + 8)` float32。

        並びは `[over マスク N, under マスク N, 種類 one-hot 4 (R1/R2/cross/複合), sign, left, obu, 有効]`。
        sign / left / obu は ±1、該当しなければ 0 (TWISTED-RL は −1 を「該当なし」に使うが、
        値の −1 と衝突するので採らない)。`cand < 0` (表に無い・候補なし) の env は全部 0。
        マスクは `links` (今の形状の L(p)) から毎回作るので、区間の境界が動けば追従する。
        """
        valid = cand >= 0
        c = cand.clamp_min(0).unsqueeze(1)

        def take(x: torch.Tensor) -> torch.Tensor:
            return x[row].gather(1, c)  # (B, 1)

        ar = torch.arange(num_links, device=links.device).unsqueeze(0)

        def mask(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            lo, hi = links.gather(1, take(a)), links.gather(1, take(b))
            return ((ar >= lo) & (ar <= hi)).float()

        kind = take(self.kind).clamp_min(0).squeeze(1)
        out = torch.cat(
            [
                mask(self.over_a, self.over_b),
                mask(self.under_a, self.under_b),
                torch.nn.functional.one_hot(kind, NUM_MOVE_KINDS).float(),
                take(self.sign).float(),
                take(self.left).float(),
                take(self.obu).float(),
                torch.ones_like(kind, dtype=torch.float32).unsqueeze(1),
            ],
            dim=1,
        )
        return torch.where(valid.unsqueeze(1), out, torch.zeros_like(out))

    def chosen_child_key(self, row: torch.Tensor, cand: torch.Tensor) -> torch.Tensor:
        """選んだ辺の子のキー `(num_envs,)`。`cand < 0` は −1。"""
        key = self.child_key[row].gather(1, cand.clamp_min(0).unsqueeze(1)).squeeze(1)
        return torch.where(cand >= 0, key, torch.full_like(key, -1))


_NO_END = 1 << 20
"""`_end_lengths` で「端を使わない」を表す値 (リンク数より十分大きい)。"""


_TABLE_CACHE: dict[str, NextEdgeTable] = {}


def get_next_edge_table(device: torch.device | str) -> NextEdgeTable:
    """デバイスごとに 1 つだけ作って使い回す (グラフの列挙は `lru_cache` 済み)。"""
    key = str(device)
    if key not in _TABLE_CACHE:
        _TABLE_CACHE[key] = NextEdgeTable(device)
    return _TABLE_CACHE[key]
