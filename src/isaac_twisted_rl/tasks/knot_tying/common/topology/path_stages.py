"""固結びへの正規の経路の上の p-data だけを段 0〜2 に数える段の定義 (v10)。

仕様と理由は `docs/decisions/0023-v10-path-stages.md`。v4〜v9 の段 (`BfsPotential.stage` = round(3Φ)) は
Φ の表 321 個の p-data を段 0〜3 に振り分け、表に無い形を段 0 にする。そのため方策の失敗で作った形
(交差 4 個・表に無い・たどる辺なし) が段 0 のアーカイブに溜まっていった (0023 の「理由」)。ここでは
段 0〜2 を下の表の 7 個の p-data に限り、それ以外を **道から外れた形** (`OFF_PATH` = −1) にする。

| 段 | p-data |
|---|---|
| 0 | `` (交差なし) |
| 1 | `1O2-_2U1-`、`1U2+_2O1+` |
| 2 | `1O3-_2U4-_3U1-_4O2-`、`1O3+_2U4+_3U1+_4O2+` (論文 Fig.1)、`1U3-_2O4-_3O1-_4U2-`、`1U3+_2O4+_3O1+_4U2+` (台本) |
| 3 | `BfsPotential.stage` >= 3 (v7 と同じ判定) |

* **段 3 は v7 と同じ判定にする。** 段 2 の 4 個のどれからも、1 手で行ける固結びが 2 つある (経路の終点の
  `STRICT_GOAL_P_DATA` の 1 つと、もう 1 つ)。終点だけを段 3 にすると、もう 1 つの固結びが学習では −1、
  chain 評価 (v7 と同じ判定) では成功になり食い違う。段 3 は終了の段でアーカイブに入らないので、
  広げても 0023 の理由 (アーカイブの汚れ) とは食い違わない (ユーザーの判断、2026-10-06)。
  経路の終点に着いたかは `strict_goal_mask` で別に数える (chain 評価の `knot_strict`)。
* **生の p-data (R2 除去の前) で引く。** 紐が重なっただけの偽の交差も道から外れる。台本の段 2 の多く
  (`1U4+_2O5+_3O6-_4O1+_5U2+_6U3-`、R2 除去すると段 1 と同じ) も外れる。
* 段 0〜2 の表は、`AbstractState` の手で自明な状態から経路 A / A' / B / B' をたどって作れることを
  `tests/test_path_stages.py` が確かめる。
"""

from __future__ import annotations

from collections.abc import Mapping

import torch

from .intersections import IntersectionBatch
from .knot_graph import STAGE_GOAL, BfsPotential, batch_keys, get_bfs_potential, p_data_key

OFF_PATH = -1
"""道から外れた形の段。学習では「段が下がった」と同じく −1 で終了する (段 0〜2 から見て下がるため)。"""

PATH_STAGE_P_DATA: dict[int, tuple[str, ...]] = {
    0: ("",),
    1: ("1O2-_2U1-", "1U2+_2O1+"),
    2: (
        "1O3-_2U4-_3U1-_4O2-",
        "1O3+_2U4+_3U1+_4O2+",
        "1U3-_2O4-_3O1-_4U2-",
        "1U3+_2O4+_3O1+_4U2+",
    ),
}
"""段 0〜2 に入る生の p-data。"""

STRICT_GOAL_P_DATA: tuple[str, ...] = (
    "1O4-_2U5-_3O6-_4U1-_5O2-_6U3-",
    "1U4+_2O5+_3U6+_4O1+_5U2+_6O3+",
)
"""正規の経路の終点の固結び (厳密な段 3)。段 3 の判定には使わない (評価の別の列)。"""

STAGE_DEFINITIONS = ("bfs", "path")
"""env の設定の `stage_definition` が取る値。`bfs` = v4〜v9 (`BfsPotential.stage`)、`path` = v10 (`PathStage`)。"""


def path_children() -> dict[str, frozenset[str]]:
    """経路の上の p-data -> 次にたどってよい子の p-data。`next_edge` の候補をこれに絞る。

    段 k (0〜1) の子は段 k+1 の表の全部、段 2 の子は経路の終点 (`STRICT_GOAL_P_DATA`) だけ。段 2 の
    もう 1 つの固結びは報酬では成功だが、たどる辺としては渡さない (手掛かりは経路の終点を指す)。
    """
    out: dict[str, frozenset[str]] = {}
    for stage, p_data in PATH_STAGE_P_DATA.items():
        nxt = PATH_STAGE_P_DATA.get(stage + 1, STRICT_GOAL_P_DATA)
        for pd in p_data:
            out[pd] = frozenset(nxt)
    return out


def _sorted_keys(p_data: Mapping[str, int], device: torch.device | str) -> tuple[torch.Tensor, torch.Tensor]:
    items = sorted((p_data_key(pd), v) for pd, v in p_data.items())
    keys = torch.tensor([k for k, _ in items], dtype=torch.int64, device=device)
    values = torch.tensor([v for _, v in items], dtype=torch.int64, device=device)
    return keys, values


def _lookup(table_keys: torch.Tensor, keys: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """`(idx, found)`。`keys` が −1 (交差が表の上限を超える) の env は見つからない。"""
    idx = torch.searchsorted(table_keys, keys).clamp(max=table_keys.numel() - 1)
    return idx, (table_keys[idx] == keys) & (keys >= 0)


class PathStage:
    """v10 の段 `(num_envs,)` int64 を GPU 上で引く。`BfsPotential.stage` と同じ引数で呼べる。"""

    def __init__(self, device: torch.device | str = "cpu"):
        self._bfs = get_bfs_potential(device)
        self.keys, self.values = _sorted_keys({pd: s for s, pds in PATH_STAGE_P_DATA.items() for pd in pds}, device)

    def stage(self, raw_batch: IntersectionBatch, reduced_batch: IntersectionBatch | None = None) -> torch.Tensor:
        """段 0〜`STAGE_GOAL`、道から外れた形は `OFF_PATH`。

        Args:
            raw_batch: R2 除去 **前** の交差 (段 0〜2 の表を引く)。
            reduced_batch: R2 除去後の交差 (段 3 の判定の `BfsPotential` に渡す)。省略すると計算する。
        """
        goal = self._bfs.stage(raw_batch, reduced_batch) >= STAGE_GOAL
        idx, found = _lookup(self.keys, batch_keys(raw_batch))
        on_path = torch.where(found, self.values[idx], torch.full_like(idx, OFF_PATH))
        return torch.where(goal, torch.full_like(on_path, STAGE_GOAL), on_path)


_STRICT_CACHE: dict[str, torch.Tensor] = {}


def strict_goal_mask(keys: torch.Tensor) -> torch.Tensor:
    """`batch_keys` のキー `(N,)` が経路の終点の固結び (`STRICT_GOAL_P_DATA`) か `(N,)` bool。"""
    dev = str(keys.device)
    if dev not in _STRICT_CACHE:
        _STRICT_CACHE[dev] = _sorted_keys({pd: 0 for pd in STRICT_GOAL_P_DATA}, keys.device)[0]
    return _lookup(_STRICT_CACHE[dev], keys)[1]


_PATH_CACHE: dict[str, PathStage] = {}


def get_stage_function(definition: str, device: torch.device | str) -> BfsPotential | PathStage:
    """段の定義の名前 (`STAGE_DEFINITIONS`) -> `.stage(raw_batch, reduced_batch)` を持つもの。デバイスごとに使い回す。"""
    if definition == "bfs":
        return get_bfs_potential(device)
    if definition == "path":
        key = str(device)
        if key not in _PATH_CACHE:
            _PATH_CACHE[key] = PathStage(device)
        return _PATH_CACHE[key]
    raise ValueError(f"stage_definition は {STAGE_DEFINITIONS} のどれか (got {definition!r})")


def stage_definition_of(cfg: object) -> str:
    """env の設定の段の定義。属性が無ければ `bfs` (v4〜v9 の設定はこの属性を持たない)。"""
    return getattr(cfg, "stage_definition", "bfs")
