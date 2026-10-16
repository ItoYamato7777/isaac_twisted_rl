"""評価用の「曲がった・ぐにゃぐにゃだが自己交差のない」紐の初期形状を作る (純 torch。isaaclab を import しない)。

chain 評価はまっすぐなロープから始める (`docs/system/evaluation.md`)。方策が初期形状の違いに頑健かを測るため、
交差が 0 の (= 段 0 の) いろいろな形から始める評価を足した。形は **関節角の列** で決める。剛体鎖のロープは
関節の曲がり (平面の中の回転) を並べれば、その角度どおりの折れ線になるためである。

* 関節ごとの曲がり角は、相関のある乱数 (AR(1)) で作る。相関が高く振れ幅が小さいと緩やかな弧に、
  相関が低く振れ幅が大きいと細かく蛇行した形になる。試料ごとに (相関, 振れ幅) を引き直すので、
  1 つの集合に緩やかな形とぐにゃぐにゃの形が混ざる。
* 自己交差と、交差しなくても隣でないリンクが近すぎる形 (接触して物理が荒れる) は棄却する。
  棄却の判定は、床に落ち着いた後の形でもう一度する (`demos/make_curved_archive.py`)。
"""

from __future__ import annotations

import math

import torch

__all__ = [
    "MAX_TURN",
    "chain_nodes",
    "min_clearance",
    "sample_turning_angles",
    "total_turning",
]


MAX_TURN = 0.45
"""1 関節の曲がりの上限 [rad]。隣のリンクが重なるほどの急な折れと、ジョイントの可動域の外を避ける。"""


def sample_turning_angles(
    n: int,
    num_joints: int,
    generator: torch.Generator,
    max_turn: float = MAX_TURN,
    sigma_range: tuple[float, float] = (0.03, 0.30),
    rho_range: tuple[float, float] = (0.2, 0.95),
) -> torch.Tensor:
    """関節ごとの曲がり角 `(n, num_joints)` [rad]。1 行が 1 つの形。

    角度は AR(1): `a[k+1] = rho * a[k] + sigma * eps` (eps は標準正規)。`rho`・`sigma` は行ごとに一様に引く。
    隣のリンクが重なって見える急な折れを避けるため、1 関節の曲がりを `max_turn` [rad] で打ち切る
    (剛体鎖のジョイントの可動域の中に収める目的もある)。
    """
    cpu = torch.device("cpu")
    sigma = torch.empty(n, 1).uniform_(*sigma_range, generator=generator)
    rho = torch.empty(n, 1).uniform_(*rho_range, generator=generator)
    eps = torch.randn(n, num_joints, generator=generator, device=cpu)
    # 定常分布の標準偏差で始める (先頭の関節だけ曲がりが小さくなるのを避ける)
    a = torch.empty(n, num_joints)
    a[:, 0:1] = sigma / torch.sqrt(1.0 - rho**2) * eps[:, 0:1]
    for k in range(1, num_joints):
        a[:, k : k + 1] = rho * a[:, k - 1 : k] + sigma * eps[:, k : k + 1]
        a[:, k : k + 1] = a[:, k : k + 1].clamp(-max_turn, max_turn)
    a[:, 0:1] = a[:, 0:1].clamp(-max_turn, max_turn)
    return a


def chain_nodes(turns: torch.Tensor, spacing: float, heading0: float = 0.0) -> torch.Tensor:
    """曲がり角 `(n, J)` から、リンク中心の xy `(n, J+1, 2)` を作る。

    リンクは長さ `spacing` で、隣と継ぐ関節はリンク中心から軸方向に `spacing / 2` の位置にある
    (`rope_model/generate_rope_usd.py` の `add_d6_joint`)。リンク k の向きは `heading0 + sum(turns[:k])`。
    """
    n, num_joints = turns.shape
    heading = heading0 + torch.cat([torch.zeros(n, 1), turns.cumsum(dim=1)], dim=1)  # (n, J+1)
    u = torch.stack([torch.cos(heading), torch.sin(heading)], dim=-1)  # (n, J+1, 2)
    step = 0.5 * spacing * (u[:, :-1] + u[:, 1:])  # (n, J, 2)
    return torch.cat([torch.zeros(n, 1, 2), step.cumsum(dim=1)], dim=1)


def min_clearance(xy: torch.Tensor, skip: int = 3) -> torch.Tensor:
    """リンク番号が `skip` 以上離れたリンク中心どうしの最小距離 `(n,)`。

    `skip` 未満の近いリンクは、曲がりの上限 (`max_turn`) で近さが決まるので数えない。
    """
    n, m, _ = xy.shape
    d = torch.cdist(xy, xy)  # (n, m, m)
    idx = torch.arange(m)
    far = (idx[None, :] - idx[:, None]).abs() >= skip
    d = torch.where(far, d, torch.full_like(d, float("inf")))
    return d.flatten(1).amin(dim=1)


def total_turning(xy: torch.Tensor) -> torch.Tensor:
    """リンク中心の折れ線の全曲がり量 `sum |Δ 向き|` `(n,)` [rad]。まっすぐからの遠さの目安 (形の分類の指標)。"""
    seg = xy[:, 1:] - xy[:, :-1]
    heading = torch.atan2(seg[..., 1], seg[..., 0])
    d = heading[:, 1:] - heading[:, :-1]
    d = torch.remainder(d + math.pi, 2.0 * math.pi) - math.pi
    return d.abs().sum(dim=1)
