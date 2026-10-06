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

* `short_end`: 端の区間を使う候補のうち、その端の区間のリンク数が少ないものを優先し、
  同点なら種類の優先順位 R1 > R2 > 複合 > cross、それでも同点なら列挙順。端を使わない候補は
  最良の端の候補と同点として扱う。「台本 (`demos/overhand_script.py`) は短い側の端を使う」という
  仮説から作ったが、実際の形状で照合すると台本と 35% しか一致しなかった (台本の輪は頭側の
  リンク 5 付近で本体と交差し、頭の端の区間のほうが短い。v6 の README「台本との照合」)。
* `long_end` (既定): `short_end` の逆で、端の区間のリンク数が **多い** ものを優先する (同点の崩し方は同じ)。
  台本と 93% 一致した。長い端ほど、掴んで別の区間の上下へ運ぶだけの自由な長さがある。
* `tail_end`: 尻尾の端 (リンク番号の大きい側。台本の「先端」) を使う候補を優先する (同点の崩し方は同じ)。
* `first`: 列挙順の最初 (比較用)。
* `random`: 候補から一様に選ぶ (比較用)。

## 段の定義との関係 (v10)

v10 (段の定義 `path`、`path_stages.py`) では、候補を「子が正規の経路の上にある辺」に絞り、経路の上に
無い状態には辺を渡さない (`get_next_edge_table(device, "path")`)。表の形 (観測の次元) は変わらない。
"""

from __future__ import annotations

from collections.abc import Collection, Mapping

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
from .path_stages import STAGE_DEFINITIONS, path_children

RULES = ("short_end", "long_end", "tail_end", "first", "random")
"""`NextEdgeTable.select` の選択規則。"""

NUM_MOVE_KINDS = 4
"""種類の one-hot の幅 (R1 / R2 / cross / 複合)。`knot_graph.MOVE_*` の番号順。"""

_KIND_PRIORITY = {MOVE_R1: 0, MOVE_R2: 1, MOVE_END_CROSS: 2, MOVE_CROSS: 3}
"""端の区間で決まらないときに同点を崩す種類の優先順位 (小さいほど優先)。"""


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

    Args:
        allowed_children: p-data -> たどってよい子の p-data。与えると、候補を子がこの中にあるものに絞り、
            ここに無い p-data の行は候補なしにする (行は残すので `lookup` は見つかり、`select` が −1 を返す)。
            None (既定) は絞らない (v6〜v9)。
    """

    def __init__(
        self,
        device: torch.device | str = "cpu",
        max_crossings: int = MAX_GRAPH_CROSSINGS,
        allowed_children: Mapping[str, Collection[str]] | None = None,
    ):
        table = _shortest_edges(max_crossings)
        if allowed_children is not None:
            table = {pd: [e for e in edges if e.child in allowed_children.get(pd, ())] for pd, edges in table.items()}
        items = sorted((p_data_key(pd), edges) for pd, edges in table.items())
        num_rows = len(items)
        num_cands = max(max(len(edges) for _, edges in items), 1)
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
        rule: str = "long_end",
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """候補の番号 `(num_envs,)` int64 を選ぶ。表に無い env は −1。

        Args:
            row / found: `lookup` の戻り値。
            links: `point_links` (今の形状での L(p))。端の区間で選ぶ規則がリンク数に使う。
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
            shortest, longest, uses_tail, uses_end = self._end_info(row, links, count)
            if rule == "short_end":
                end_key = shortest
            elif rule == "long_end":
                end_key = _NO_END - longest
            else:  # tail_end
                end_key = (~uses_tail).long()
            # 端を使わない候補は、端を使う最良の候補と同点にする。
            best = torch.where(exists & uses_end, end_key, torch.full_like(end_key, _NO_END)).min(dim=1).values
            end_key = torch.where(uses_end, end_key, best.unsqueeze(1))
            prio = self.priority[kind.clamp_min(0)]
            score = (end_key * NUM_MOVE_KINDS + prio) * num_cands + order
        big = torch.iinfo(torch.int64).max
        score = torch.where(exists, score, torch.full_like(score, big))
        cand = score.argmin(dim=1)
        return torch.where(found & exists.any(dim=1), cand, torch.full_like(cand, -1))

    def _end_info(
        self, row: torch.Tensor, links: torch.Tensor, count: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """候補が使う端の区間 `(B, C)` の `(最短のリンク数, 最長のリンク数, 尻尾の端を使うか, 端を使うか)`。

        端を使わない候補の最短は `_NO_END`、最長は 0。交差 0 では区間 0 が頭と尻尾の両方の端。
        """
        tail = (2 * count + 1).unsqueeze(1)
        shortest = longest = uses_tail = uses_end = None
        for a, b in ((self.over_a[row], self.over_b[row]), (self.under_a[row], self.under_b[row])):
            n = links.gather(1, b) - links.gather(1, a) + 1
            is_tail = b == tail
            is_end = (a == 0) | is_tail
            lo = torch.where(is_end, n, torch.full_like(n, _NO_END))
            hi = torch.where(is_end, n, torch.zeros_like(n))
            if shortest is None:
                shortest, longest, uses_tail, uses_end = lo, hi, is_tail, is_end
            else:
                shortest, longest = torch.minimum(shortest, lo), torch.maximum(longest, hi)
                uses_tail, uses_end = uses_tail | is_tail, uses_end | is_end
        return shortest, longest, uses_tail, uses_end

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


_TABLE_CACHE: dict[tuple[str, str], NextEdgeTable] = {}


def get_next_edge_table(device: torch.device | str, stage_definition: str = "bfs") -> NextEdgeTable:
    """段の定義 (`path_stages.STAGE_DEFINITIONS`) とデバイスごとに 1 つだけ作って使い回す。

    `bfs` (既定、v6〜v9) は絞らない表、`path` (v10) は候補を正規の経路の上に絞った表。
    """
    if stage_definition not in STAGE_DEFINITIONS:
        raise ValueError(f"stage_definition は {STAGE_DEFINITIONS} のどれか (got {stage_definition!r})")
    key = (str(device), stage_definition)
    if key not in _TABLE_CACHE:
        allowed = path_children() if stage_definition == "path" else None
        _TABLE_CACHE[key] = NextEdgeTable(device, allowed_children=allowed)
    return _TABLE_CACHE[key]
