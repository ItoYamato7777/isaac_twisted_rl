"""紐の中心線 (節点列) の処理。すべてバッチ対応の純 torch (isaaclab / warp を import しない)。

紐モデルの比較 (`rope_study/`) では、紐の表現 (28 リンクの剛体鎖、56 リンクの剛体鎖、XPBD の棒、
FEM の四面体メッシュ) ごとに節点の数が違う。方策の観測・行動・位相の判定を **表現に依らない形** に
揃えるため、どの表現も中心線を弧長で等分した「仮想リンク」(既定 28 本) に写してから MDP 項へ渡す。

* 剛体鎖のリンク中心は、カプセルの軸をつないだ中心線の上で弧長 (k + 0.5) x リンク間隔の点そのもの
  なので、28 リンクの鎖の仮想リンクは実リンクの中心と一致する。ただし `Art28` はアダプタを通さず
  実リンクをそのまま使う (v4 と完全に同じ環境にするため)。中心線を関節の位置ではなく隣り合う中心の
  中点 (`topology.polyline_from_link_centers`) で近似すると、曲がった箇所で 1 cm 弱ずれる。
* 実機でも、カメラで推定した紐の中心線を同じ本数の点に再標本化すれば同じ観測になる。

節点列の約束: `(B, M, 3)`。端から端へ順に並び、隣り合う節点を直線で結んだ折れ線が中心線。
剛体鎖ではカプセルの継ぎ目と両端 (`topology.polyline_from_link_centers` と同じ作り方)。
"""

from __future__ import annotations

import math

import torch

_EPS = 1e-9


# ---------------------------------------------------------------- 弧長と再標本化


def arc_lengths(nodes: torch.Tensor) -> torch.Tensor:
    """各節点までの累積弧長 `(B, M)`。先頭は 0。"""
    seg = (nodes[:, 1:] - nodes[:, :-1]).norm(dim=-1)
    return torch.cat([torch.zeros_like(seg[:, :1]), seg.cumsum(dim=1)], dim=1)


def interp_weights(nodes: torch.Tensor, s_query: torch.Tensor, normalized: bool = True):
    """弧長 `s_query` の点を、どの区間のどの比率で取るか。

    Args:
        nodes: `(B, M, 3)`。
        s_query: `(Q,)` または `(B, Q)`。`normalized=True` なら全長に対する比 (0〜1)。
        normalized: `s_query` を全長比として解釈するか。

    Returns:
        `(idx, t)`。`idx` `(B, Q)` は区間の始点の節点番号 (0〜M-2)、`t` `(B, Q)` は区間内の比率 (0〜1)。
        点は `nodes[idx] * (1 - t) + nodes[idx + 1] * t`。
    """
    cum = arc_lengths(nodes)  # (B, M)
    total = cum[:, -1:].clamp_min(_EPS)
    if s_query.dim() == 1:
        s_query = s_query.unsqueeze(0).expand(nodes.shape[0], -1)
    s = s_query * total if normalized else s_query
    s = torch.minimum(torch.maximum(s, torch.zeros_like(s)), total.expand_as(s))
    idx = torch.searchsorted(cum.contiguous(), s.contiguous(), right=True) - 1
    idx = idx.clamp(0, nodes.shape[1] - 2)
    s0 = cum.gather(1, idx)
    s1 = cum.gather(1, idx + 1)
    t = ((s - s0) / (s1 - s0).clamp_min(_EPS)).clamp(0.0, 1.0)
    return idx, t


def gather_interp(values: torch.Tensor, idx: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """節点ごとの量 `values` `(B, M, D)` を `interp_weights` の結果で補間する `(B, Q, D)`。"""
    d = values.shape[-1]
    a = values.gather(1, idx.unsqueeze(-1).expand(-1, -1, d))
    b = values.gather(1, (idx + 1).unsqueeze(-1).expand(-1, -1, d))
    return a + (b - a) * t.unsqueeze(-1)


def virtual_link_fractions(num_links: int, device=None) -> torch.Tensor:
    """仮想リンク k の中心の全長比 `(num_links,)` = `(k + 0.5) / num_links`。"""
    return (torch.arange(num_links, device=device, dtype=torch.float32) + 0.5) / num_links


def virtual_link_centers(nodes: torch.Tensor, num_links: int) -> torch.Tensor:
    """中心線を弧長で `num_links` 等分した各区間の中点 `(B, num_links, 3)`。

    節点列がカプセルの軸の端点 (関節の位置) の列なら、等間隔の剛体鎖ではリンク中心をそのまま返す。
    """
    frac = virtual_link_fractions(num_links, nodes.device)
    idx, t = interp_weights(nodes, frac)
    return gather_interp(nodes, idx, t)


def resample(nodes: torch.Tensor, num_points: int) -> torch.Tensor:
    """中心線を両端を含めて `num_points` 点に等弧長で再標本化する `(B, num_points, 3)`。"""
    frac = torch.linspace(0.0, 1.0, num_points, device=nodes.device)
    idx, t = interp_weights(nodes, frac)
    return gather_interp(nodes, idx, t)


# ---------------------------------------------------------------- 枠 (Cosserat 棒の初期化用)


def _quat_from_two_vectors(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """単位ベクトル a を b へ回す最小回転の quaternion `(..., 4)` (w, x, y, z)。"""
    dot = (a * b).sum(-1, keepdim=True)
    cross = torch.cross(a, b, dim=-1)
    w = 1.0 + dot
    q = torch.cat([w, cross], dim=-1)
    # a と b が逆向きのときは a に垂直な任意の軸まわりに 180 度
    anti = (w.squeeze(-1) < 1e-6)
    if bool(anti.any()):
        axis = torch.cross(a, torch.tensor([1.0, 0.0, 0.0], device=a.device).expand_as(a), dim=-1)
        small = axis.norm(dim=-1, keepdim=True) < 1e-6
        axis = torch.where(small, torch.cross(a, torch.tensor([0.0, 1.0, 0.0], device=a.device).expand_as(a), dim=-1), axis)
        q_anti = torch.cat([torch.zeros_like(w), axis], dim=-1)
        q = torch.where(anti.unsqueeze(-1), q_anti, q)
    return q / q.norm(dim=-1, keepdim=True).clamp_min(_EPS)


def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """`q1 ⊗ q2` (w, x, y, z)。"""
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dim=-1,
    )


def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """quaternion `q` (w, x, y, z) でベクトル `v` を回す。"""
    w = q[..., :1]
    u = q[..., 1:]
    t = 2.0 * torch.cross(u, v, dim=-1)
    return v + w * t + torch.cross(u, t, dim=-1)


def quat_conj(q: torch.Tensor) -> torch.Tensor:
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


def parallel_transport_frames(nodes: torch.Tensor) -> torch.Tensor:
    """各区間の向き (局所 z 軸 = 接線) を、捩れの無い平行移動で決めた quaternion `(B, M-1, 4)`。

    区間 0 は世界の z 軸を接線へ最小回転で回した向き。以降は直前の区間の向きを、接線の変化ぶんだけ
    最小回転で回す (= 捩れ 0 の Bishop 枠)。Cosserat 棒を中心線から復元するとき
    (アーカイブの形状を別の表現へ移すとき) に使う。
    """
    tan = nodes[:, 1:] - nodes[:, :-1]
    tan = tan / tan.norm(dim=-1, keepdim=True).clamp_min(_EPS)
    ez = torch.zeros_like(tan[:, 0])
    ez[:, 2] = 1.0
    q = _quat_from_two_vectors(ez, tan[:, 0])
    out = [q]
    for k in range(1, tan.shape[1]):
        dq = _quat_from_two_vectors(tan[:, k - 1], tan[:, k])
        q = quat_mul(dq, q)
        q = q / q.norm(dim=-1, keepdim=True).clamp_min(_EPS)
        out.append(q)
    return torch.stack(out, dim=1)


# ---------------------------------------------------------------- 計測 (物理の妥当性の評価用)


def segment_pair_distances(nodes: torch.Tensor, min_gap: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """区間どうしの最短距離 `(B, P)` と、その組の区間番号 `(P,)`, `(P,)`。区間番号が `min_gap` 以上離れた組だけ。

    区間 k は節点 k と k+1 を結ぶ線分。Ericson "Real-Time Collision Detection" 5.1.9 (端点でクランプ)。
    `demos/divergence_bench.py` の `_segment_distances` と同じ式を、組ごとの値を返す形にしたもの。
    """
    p0, p1 = nodes[:, :-1], nodes[:, 1:]
    n = p0.shape[1]
    i, j = torch.triu_indices(n, n, offset=min_gap, device=nodes.device)
    a0, a1, b0, b1 = p0[:, i], p1[:, i], p0[:, j], p1[:, j]
    d1, d2, r = a1 - a0, b1 - b0, a0 - b0
    aa, ee, f = (d1 * d1).sum(-1), (d2 * d2).sum(-1), (d2 * r).sum(-1)
    c, bb = (d1 * r).sum(-1), (d1 * d2).sum(-1)
    denom = (aa * ee - bb * bb).clamp_min(1e-12)
    s = ((bb * f - c * ee) / denom).clamp(0.0, 1.0)
    t = (bb * s + f) / ee.clamp_min(1e-12)
    s = torch.where(t < 0.0, (-c / aa.clamp_min(1e-12)).clamp(0.0, 1.0), s)
    s = torch.where(t > 1.0, ((bb - c) / aa.clamp_min(1e-12)).clamp(0.0, 1.0), s)
    t = t.clamp(0.0, 1.0)
    dist = ((a0 + d1 * s.unsqueeze(-1)) - (b0 + d2 * t.unsqueeze(-1))).norm(dim=-1)
    return torch.nan_to_num(dist, nan=1.0e9), i, j


def self_penetration(nodes: torch.Tensor, radius: float, exclude_arc: float) -> torch.Tensor:
    """紐どうしのめり込みの最大深さ `(B,)` [m]。弧長で `exclude_arc` 以内の区間の組は見ない。

    めり込み = `2 r − (区間の最短距離)`。区間の長さが表現ごとに違うので、除外する近傍は区間の数ではなく
    弧長で決める (隣接・1 本おきの組は曲げれば必ず近づくため)。

    区間の長さの平均は有限の区間だけで取る。発散して NaN を含む環境が 1 つでもあると、平均が NaN になって
    全体が止まった (剛体鎖の hang 試験、2026-10-02)。NaN の環境のめり込みは 0 になる (発散は別に数える)。
    """
    seg = (nodes[:, 1:] - nodes[:, :-1]).norm(dim=-1)
    finite = torch.isfinite(seg)
    if not bool(finite.any()):
        return torch.zeros(nodes.shape[0], device=nodes.device)
    seg_len = seg[finite].mean().clamp_min(_EPS)
    min_gap = max(2, int(math.ceil(exclude_arc / float(seg_len))))
    if nodes.shape[1] - 1 <= min_gap:
        return torch.zeros(nodes.shape[0], device=nodes.device)
    dist, _, _ = segment_pair_distances(nodes, min_gap)
    return (2.0 * radius - dist).clamp_min(0.0).amax(dim=-1)


def strain(nodes: torch.Tensor, rest_segment_length: float) -> torch.Tensor:
    """区間の伸びの最大値 `(B,)` (無次元、0 なら伸びなし)。"""
    seg = (nodes[:, 1:] - nodes[:, :-1]).norm(dim=-1)
    return (seg / rest_segment_length - 1.0).abs().amax(dim=-1)


def turning_angles(nodes: torch.Tensor) -> torch.Tensor:
    """内部節点での曲がり角 `(B, M-2)` [rad]。"""
    e0 = nodes[:, 1:-1] - nodes[:, :-2]
    e1 = nodes[:, 2:] - nodes[:, 1:-1]
    cos = (e0 * e1).sum(-1) / (e0.norm(dim=-1) * e1.norm(dim=-1)).clamp_min(_EPS)
    return torch.arccos(cos.clamp(-1.0, 1.0))
