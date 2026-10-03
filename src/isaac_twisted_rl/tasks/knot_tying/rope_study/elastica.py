"""重い片持ち梁 (heavy elastica) の数値解。紐モデルの曲げ剛性が理論どおりかを確かめる基準 (純 numpy)。

一端を水平に固定し、自重 (線荷重 w) で大きくたわむ棒の釣り合い:

    EI θ''(s) = w (L − s) cos θ(s),   θ(0) = 0 (固定端は水平),   θ'(L) = 0 (自由端にモーメントなし)

θ は水平からの角度 (上向き正)、s は固定端からの弧長。θ'(0) を二分法で決める (shooting)。
たわみは無次元量 `w L^3 / EI` だけで決まる (小たわみなら先端の下がり = w L^4 / (8 EI))。

紐モデルの比較 (`rope_study/`) では、同じ EI と w を与えた各モデルの片持ち梁の先端の下がりを、
この解と比べる (`bench_cantilever_xpbd.py` / `bench_cantilever_art.py`)。合わなければ、そのモデルの曲げ剛性は名目値と違う
(離散化・ソルバ・減衰の影響で「実効の」EI が変わっている) ということになる。
"""

from __future__ import annotations

import math

import numpy as np


def _shoot(k0: float, alpha: float, n: int = 4000) -> tuple[float, np.ndarray]:
    """無次元化 (s を L で割る) した方程式 θ'' = alpha (1 − s) cos θ を θ(0)=0, θ'(0)=k0 で積分 (RK4)。"""
    h = 1.0 / n
    th, dth = 0.0, k0
    out = np.empty(n + 1)
    out[0] = th

    def f(s, th, dth):
        return dth, alpha * (1.0 - s) * math.cos(th)

    for i in range(n):
        s = i * h
        a1, b1 = f(s, th, dth)
        a2, b2 = f(s + h / 2, th + h / 2 * a1, dth + h / 2 * b1)
        a3, b3 = f(s + h / 2, th + h / 2 * a2, dth + h / 2 * b2)
        a4, b4 = f(s + h, th + h * a3, dth + h * b3)
        th += h / 6 * (a1 + 2 * a2 + 2 * a3 + a4)
        dth += h / 6 * (b1 + 2 * b2 + 2 * b3 + b4)
        out[i + 1] = th
    return dth, out


def heavy_cantilever(length: float, ei: float, w: float) -> dict:
    """先端の位置 (固定端からの水平距離・下がり) と、弧長に沿った形を返す。

    Args:
        length: 自由長 L [m]。
        ei: 曲げ剛性 [N m^2]。
        w: 線荷重 (線密度 x g) [N/m]。
    """
    alpha = w * length**3 / ei
    lo, hi = -2.0 * alpha - 10.0, 0.0  # θ'(0) は負 (下へ曲がる)。|θ'(0)| <= alpha (小たわみで alpha / 2)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        dth_end, th = _shoot(mid, alpha)
        # 物理的な解は θ ∈ [−90°, 0] で単調。大たわみ (alpha が 100 を超える) では、曲げが足りないと θ が水平より
        # 上へ振れ、曲げすぎると真下を越えて、どちらも cos θ の符号が変わって振り子のように戻る。θ'(1) の符号だけで
        # 判定すると偽の根に落ちた (2026-10-02) ので、範囲を越えたかで先に判定する
        if th.max() > 1e-12:
            too_little = True
        elif th.min() < -0.5 * math.pi:
            too_little = False
        else:
            too_little = dth_end > 0.0
        if too_little:
            hi = mid
        else:
            lo = mid
    _, th = _shoot(hi, alpha)  # 根の「曲げすぎでない」側 (真下を越えた解は使わない)
    s = np.linspace(0.0, 1.0, th.size)
    ds = s[1] - s[0]
    x = np.concatenate([[0.0], np.cumsum(0.5 * (np.cos(th[1:]) + np.cos(th[:-1])) * ds)]) * length
    y = np.concatenate([[0.0], np.cumsum(0.5 * (np.sin(th[1:]) + np.sin(th[:-1])) * ds)]) * length
    return {
        "alpha": alpha,
        "tip_x": float(x[-1]),
        "tip_drop": float(-y[-1]),
        "tip_angle_deg": float(math.degrees(th[-1])),
        "small_deflection_drop": w * length**4 / (8.0 * ei),
        "s": s * length,
        "x": x,
        "y": y,
    }


def discrete_chain_drop(length: float, seg: float, ei: float, w: float) -> float:
    """関節ごとにばね EI / seg を持つ離散鎖 (剛体鎖・XPBD の棒と同じ離散化) の小たわみの先端の下がり [m]。

    固定端の関節から先端まで、区間の質量を両端の節点に半分ずつ配る (固定端の節点の分は数えない)。
    連続体 (`heavy_cantilever`) との差が離散化の誤差で、区間長 / 自由長にほぼ比例する。
    """
    n = int(round(length / seg))
    m = w * seg
    xs = [k * seg for k in range(n + 1)]
    ms = [0.0] + [m] * (n - 1) + [0.5 * m]
    drop = 0.0
    for j in range(n):
        moment = sum(ms[i] * (xs[i] - xs[j]) for i in range(j + 1, n + 1))
        drop += moment * seg / ei * (xs[n] - xs[j])
    return drop


def effective_ei(length: float, w: float, tip_drop: float, ei_guess: float = 7.854e-4) -> float:
    """先端の下がりが `tip_drop` になる EI (逆問題。単調なので二分法)。探索は名目値の 1/100〜100 倍。

    下がりが自由長に近い (ほぼ真下に垂れる) と EI はほとんど決まらないので、範囲の端の値になる。
    """
    lo, hi = ei_guess / 100.0, ei_guess * 100.0
    for _ in range(100):
        mid = math.sqrt(lo * hi)
        if heavy_cantilever(length, mid, w)["tip_drop"] > tip_drop:
            lo = mid
        else:
            hi = mid
    return math.sqrt(lo * hi)


if __name__ == "__main__":
    ei, w = 7.854e-4, 0.2513 * 9.81
    for L in (0.08, 0.12, 0.16, 0.24):
        r = heavy_cantilever(L, ei, w)
        print(
            f"L={L:.2f} alpha={r['alpha']:.2f} tip_drop={r['tip_drop'] * 1000:.1f} mm "
            f"(small-deflection {r['small_deflection_drop'] * 1000:.1f} mm) tip_angle={r['tip_angle_deg']:.1f} deg"
        )
