"""`view_archive.py` が使う、アーカイブから見せる形を選ぶ関数 (torch だけ。Isaac Sim なしで試験する)。

## なぜこの形か

* **段ごとに同じ数ずつ選ぶ。** 段 0 / 1 / 2 の形を並べて比べるのが目的なので、段の件数の比 (どれも
  容量 16384 で満杯になることが多い) ではなく、段ごとに `per_stage` 件にそろえる。
* **環境の番号は段の順に詰める** (段 0 の形が env 0 〜、次に段 1 …)。Isaac Lab の格子は番号の順に
  並ぶので、同じ段の形が近くに集まる。
* **選ぶのは乱数 (シード固定)**。リングバッファの並びは古い順・新しい順が混ざっていて、先頭から取ると
  学習の特定の時期の形に偏る。
"""

from __future__ import annotations

import torch


def pick_shapes(sizes: dict[int, int], stages: list[int], per_stage: int, seed: int) -> list[tuple[int, int]]:
    """環境の順に `(段, アーカイブの中の番号)` を返す。

    Args:
        sizes: 段 -> アーカイブの件数。
        stages: 見せる段 (この順に並べる)。アーカイブに無い・空の段は飛ばす。
        per_stage: 段ごとの件数。件数が足りない段は全部 (番号の順)。
        seed: 選ぶ乱数のシード。
    """
    gen = torch.Generator().manual_seed(int(seed))
    out: list[tuple[int, int]] = []
    for s in stages:
        n = int(sizes.get(s, 0))
        if n == 0:
            continue
        if n <= per_stage:
            idx = list(range(n))
        else:
            idx = torch.randperm(n, generator=gen)[:per_stage].sort().values.tolist()
        out.extend((s, i) for i in idx)
    return out
