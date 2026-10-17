"""ロープの物理のドメインランダム化 (DR) の、値の引き方と正規化。v14 で追加。

Isaac Lab への書き込み (剛性・減衰・質量・摩擦) は `mdp/rope_dr.py` の `RandomizeRopePhysics`。
このモジュールは **純 torch** (isaaclab を import しない) で、`tests/test_domain_rand.py` から Isaac Sim なしで
検証できる。仕様と範囲の理由は `docs/decisions/0027-v14-s0-reshape-dr-priv-critic.md`。

## 軸と値

どの軸も「公称値 (プリセット `knot` の USD の値) に掛ける倍率」で持ち、範囲 `(lo, hi)` から **対数一様** に引く。
倍率は 1 を中心に上下対称に振ることが多く (×0.5〜×2)、対数一様なら ×0.5 側と ×2 側が同じ確率で出る。

| 軸 | 掛ける先 |
|---|---|
| `bend_stiffness` | 曲げの DOF (rotY / rotZ) のばね定数 |
| `twist_stiffness` | 捩りの DOF (rotX) のばね定数 |
| `damping_ratio` | 減衰。剛性と質量の振りとは別に、臨界減衰に対する比 ζ を振る (`damping_scale`) |
| `mass` | リンクの質量と慣性 (同じ倍率) |
| `rope_friction` | ロープの全形状の静止・動摩擦 |
| `finger_friction` | グリッパーの指の形状の静止・動摩擦 |

## 減衰を ζ で振る理由

USD の減衰は臨界減衰 (ζ = 1) で入っている (`rope_specs.py`)。臨界減衰は `c = 2 ζ sqrt(k I)` で、慣性 I は質量と同じ
倍率で振るので、剛性や質量だけを振ると減衰の比が勝手に変わる。剛性・質量と減衰の比を独立な軸にするため、
減衰の倍率を `sqrt(s_k · s_m) · s_ζ` にする。

## 正規化 (critic の特権観測)

`normalize` は log2 の倍率を範囲の中点と半幅で [−1, 1] に写す。範囲が 1 点 (lo == hi、その軸を振らない) なら 0。
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch

AXES = ("bend_stiffness", "twist_stiffness", "damping_ratio", "mass", "rope_friction", "finger_friction")
"""軸の並び。`sample_scales` / `normalize` の列と、critic の特権観測の列の順。"""

NOMINAL_RANGES: dict[str, tuple[float, float]] = {name: (1.0, 1.0) for name in AXES}
"""どの軸も振らない範囲 (公称の物理)。"""


def _log_bounds(
    ranges: Mapping[str, tuple[float, float]], device: torch.device | str
) -> tuple[torch.Tensor, torch.Tensor]:
    lo, hi = [], []
    for name in AXES:
        a, b = (float(v) for v in ranges[name])
        if not (0.0 < a <= b):
            raise ValueError(f"{name}: 範囲は 0 < lo <= hi (got {(a, b)})")
        lo.append(math.log(a))
        hi.append(math.log(b))
    return torch.tensor(lo, device=device), torch.tensor(hi, device=device)


def sample_scales(
    n: int,
    ranges: Mapping[str, tuple[float, float]],
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """`(n, len(AXES))` の倍率を、軸ごとに範囲から対数一様に引く。lo == hi の軸は定数。"""
    lo, hi = _log_bounds(ranges, device)
    u = torch.rand(n, len(AXES), device=device, generator=generator)
    return torch.exp(lo + u * (hi - lo))


def normalize(scales: torch.Tensor, ranges: Mapping[str, tuple[float, float]]) -> torch.Tensor:
    """倍率 `(n, len(AXES))` を [−1, 1] に写す (log の範囲の中点が 0)。1 点の範囲の軸は 0。"""
    lo, hi = _log_bounds(ranges, scales.device)
    half = 0.5 * (hi - lo)
    mid = 0.5 * (hi + lo)
    out = (torch.log(scales) - mid) / torch.where(half > 0, half, torch.ones_like(half))
    return torch.where(half > 0, out, torch.zeros_like(out)).clamp(-1.0, 1.0)


def damping_scale(stiffness_scale: torch.Tensor, mass_scale: torch.Tensor, ratio_scale: torch.Tensor) -> torch.Tensor:
    """減衰の倍率 `sqrt(s_k · s_m) · s_ζ` (臨界減衰に対する比を `ratio_scale` 倍にする)。"""
    return torch.sqrt(stiffness_scale * mass_scale) * ratio_scale


# ---------------------------------------------------------------- 寸法の DR (v16)
#
# 物理の 6 軸と違い、寸法は reset で引き直さない (PhysX は実行中にカプセルの形を変えられない)。起動時に env ごとに
# 1 組を割り当て、run の間変わらない。Isaac Lab への接続 (spawn と、env ごとの寸法を読む項) は
# `rope_model/rope_cfg.py` の `RopeSizeSpawnerCfg` と `mdp/rope_size.py`。仕様は `docs/decisions/0029-v16-rope-size-dr.md`。

SIZE_AXES = ("rope_length", "rope_diameter")
"""寸法の軸の並び。critic の特権観測の `rope_size` の列の順 (長さの倍率、直径の倍率)。"""


def size_assignment(num_envs: int, num_sizes: int, device: torch.device | str = "cpu") -> torch.Tensor:
    """env ごとの寸法の組の番号 `(num_envs,)`。env `i` は組 `i % num_sizes`。

    Isaac Lab 2.3.0 の `spawn_multi_asset` (`random_choice=False`) が env の prim に USD を割り当てる規則と同じ。
    実際に spawn された寸法と合っているかは、起動時に USD の半径で確かめる (`mdp.check_rope_sizes`)。
    """
    if num_sizes < 1:
        raise ValueError(f"寸法の組が無い (num_sizes={num_sizes})")
    return torch.arange(num_envs, device=device) % num_sizes


def normalize_size(scales: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    """寸法の倍率 `(n, 2)` を、軸ごとの log2 の範囲 `[log2 lo, log2 hi]` で [−1, 1] に写す。

    範囲は寸法の格子の最小と最大 (`rope_specs.SIZE_LENGTH_SCALES`・`SIZE_DIAMETER_SCALES`) で、使う組の数に依らない
    (公称だけで評価しても、学習と同じ値が critic に入る)。範囲が 1 点の軸は 0。`normalize` と同じ写し方。
    """
    llo, lhi = torch.log2(lo), torch.log2(hi)
    half = 0.5 * (lhi - llo)
    mid = 0.5 * (lhi + llo)
    out = (torch.log2(scales) - mid) / torch.where(half > 0, half, torch.ones_like(half))
    return torch.where(half > 0, out, torch.zeros_like(out)).clamp(-1.0, 1.0)
