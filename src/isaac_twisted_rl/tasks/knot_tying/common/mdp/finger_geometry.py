"""指とロープの幾何 (v5 の発散対策と、発散の計測で使う)。

**純 torch** (isaaclab を import しない) なので、`tests/test_finger_geometry.py` から
Isaac Sim 無しで検証できる。`PickPlaceAction` (`actions.py`) が使う。
"""

from __future__ import annotations

import torch

try:
    from . import phases
except ImportError:  # テストから単体で import された場合
    import phases


def rope_sample_points(link_pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """リンク中心と、隣り合う中心の中点を並べた点列 `(N, 2L-1, 3)` と、各点のリンク座標 `(2L-1,)`。

    カプセル 1 本は中心から軸方向に ±(半長 + 半径) 伸びているので、中心だけで障害物を
    探すと隙間を見逃す。中点を足すと点の間隔がリンク間隔の半分 (knot で 2 cm = 直径) になり、
    各点を「半径 r の球」とみなせる。
    """
    mid = 0.5 * (link_pos[:, 1:] + link_pos[:, :-1])
    n, num_links, _ = link_pos.shape
    pts = torch.empty(n, 2 * num_links - 1, 3, device=link_pos.device, dtype=link_pos.dtype)
    pts[:, 0::2] = link_pos
    pts[:, 1::2] = mid
    coord = torch.arange(2 * num_links - 1, device=link_pos.device, dtype=link_pos.dtype) * 0.5
    return pts, coord


def finger_footprint_top(
    points: torch.Tensor, center_xy: torch.Tensor, yaw: torch.Tensor, radius: float, min_z: float = -float("inf")
) -> torch.Tensor:
    """全開の指 2 本の真下にある紐の最高点 `(N,)` [m] (紐が無ければ 0 = 床)。

    `points` は `rope_sample_points` の点列 (z は env 原点基準)。手を `center_xy` の真上で
    `yaw` に向けて降ろしたとき、指の直方体 (ローカル |x| ≤ `FINGER_HALF_X`、|y| が
    `FINGER_OPEN_Y` の範囲) を半径ぶん広げた領域に入る点の `z + radius` の最大値を返す。
    掴むリンク自身は指の間 (|y| ≈ 0) にあるので領域に入らない。
    中心の高さが `min_z` 以下の点は数えない (床の上の紐を除くときに使う)。
    """
    d = points[..., :2] - center_xy.unsqueeze(1)
    c, s = torch.cos(yaw).unsqueeze(1), torch.sin(yaw).unsqueeze(1)
    x_loc = c * d[..., 0] + s * d[..., 1]
    y_loc = (-s * d[..., 0] + c * d[..., 1]).abs()
    y_lo, y_hi = phases.FINGER_OPEN_Y
    inside = (x_loc.abs() <= phases.FINGER_HALF_X + radius) & (y_loc >= y_lo - radius) & (y_loc <= y_hi + radius)
    inside &= points[..., 2] > min_z
    top = torch.where(inside, points[..., 2] + radius, torch.zeros_like(points[..., 2]))
    return torch.nan_to_num(top, nan=0.0).amax(dim=1).clamp_min(0.0)
