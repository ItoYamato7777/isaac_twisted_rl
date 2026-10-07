"""v7 の段のまま、表に無く R2 除去後のラダーでも 0 の「行き止まりの形」だけを −1 にする段の定義 (v11)。

仕様と理由は `docs/decisions/0024-v11-dead-end-stages.md`。v4〜v10 の `bfs` (`BfsPotential.stage` = round(3Φ)) は
Φ の表 (321 個の p-data) に無い形をラダーの段にし、ラダーが 0 の形は段 0 になる。v7 を長く学習すると、方策の失敗で
作ったこの形 (例: 交差 4 個・R2 で消えない・ラダー 0 の `1O6+_2U5+_3O8-_4U7-_5O2+_6U1+_7O4-_8U3-`) が段 0 の
アーカイブに溜まった (0024 の「理由」)。ここでは段を次の順に決める (生 = R2 除去の前):

| 順 | 条件 | 段 |
|---|---|---|
| 1 | 生の p-data が表にある | v7 と同じ (`BfsPotential.stage(raw, reduced)`) |
| 2 | 生は無いが、R2 除去後の p-data が表にある | R2 除去後の形の段 (`BfsPotential.stage(reduced, reduced)`) |
| 3 | どちらも無く、R2 除去後のラダーの段が 1 以上 | ラダーの段 (v7 と同じ) |
| 4 | どちらも無く、ラダーの段が 0 | `DEAD_END` (−1) |

* **R2 除去は 1 が外れたときだけ使う。** 表の段 2 のうち 16 個 (台本が通る cover 後 `1U4+_2O5+_3O6-_4O1+_5U2+_6U3-`
  を含む) と段 1 のうち 12 個は、R2 除去すると 1 つ下の段の形になる。先に R2 除去すると、これらが落ちる。
* **2 は偽の交差 (紐が重なっただけの交差) を救う。** v7 のアーカイブの段 0 の 31〜36%・段 1 の 9% がこの形で、
  v7 では段 0、ここでは R2 除去後の形の段 (多くは 1) になる。
* **3 の表に無い形は外さない。** 段 2 の表に無い形 (交差 4 個・ラダー 2) は v7 の段 2 の形ごとの評価で行き止まり
  ではなかった (0024)。ラダー 1 は R2 除去後の交差 1 個で、交差 1 個の形は全部表にあるので 3 には来ない。
* 交差 5 個以上の形はキーが −1 で表を引かず、3 か 4 になる (v7 ではラダーの段)。
"""

from __future__ import annotations

import torch

from .intersections import IntersectionBatch
from .knot_detect import reduce_r2, topology_class
from .knot_graph import BfsPotential, batch_keys, get_bfs_potential

DEAD_END = -1
"""行き止まりの形の段。v10 の `OFF_PATH` と同じく、学習では手の前の段 (0〜2) より下がるので −1 で終了する。"""


def _in_table(bfs: BfsPotential, batch: IntersectionBatch) -> torch.Tensor:
    """`(num_envs,)` bool。p-data が Φ の表にあるか (交差が表の上限を超える env は無い)。"""
    keys = batch_keys(batch)
    idx = torch.searchsorted(bfs.keys, keys).clamp(max=bfs.keys.numel() - 1)
    return (bfs.keys[idx] == keys) & (keys >= 0)


class DeadEndStage:
    """v11 の段 `(num_envs,)` int64 を GPU 上で引く。`BfsPotential.stage` と同じ引数で呼べる。"""

    def __init__(self, device: torch.device | str = "cpu"):
        self._bfs = get_bfs_potential(device)

    def stage(self, raw_batch: IntersectionBatch, reduced_batch: IntersectionBatch | None = None) -> torch.Tensor:
        """段 0〜`STAGE_GOAL`、行き止まりの形は `DEAD_END`。

        Args:
            raw_batch: R2 除去 **前** の交差。
            reduced_batch: R2 除去後の交差。省略すると計算する (アーカイブの復元の検算はこれを渡さない)。
        """
        if reduced_batch is None:
            reduced_batch = reduce_r2(raw_batch)
        v7 = self._bfs.stage(raw_batch, reduced_batch)
        reduced = self._bfs.stage(reduced_batch, reduced_batch)
        ladder = topology_class(reduced_batch)
        off_table = torch.where(ladder > 0, ladder, torch.full_like(ladder, DEAD_END))
        out = torch.where(_in_table(self._bfs, reduced_batch), reduced, off_table)
        return torch.where(_in_table(self._bfs, raw_batch), v7, out)
