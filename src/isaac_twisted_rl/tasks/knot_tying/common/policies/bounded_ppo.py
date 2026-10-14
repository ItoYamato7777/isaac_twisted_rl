"""PPO の損失に「方策の平均が行動の範囲 [−1, 1] の外へ出た分」の罰則を足す `BoundedPPO`。v13 で追加。

    loss = PPO の損失 + bounds_loss_coef · mean_batch( Σ_チャネル max(|μ| − 1, 0)² )

μ は actor の出力 (クリップ前の平均)。仕様と理由は `docs/decisions/0026-v13-action-bounds-loss.md`。
要点: 環境は行動を [−1, 1] でクリップし (`PickPlaceAction.process_actions`)、PPO の log prob はクリップ前の
行動で計算する。平均が範囲の外にあると標本の多くが同じ端の行動になり、平均を範囲の中へ戻す勾配が無い。
罰則は範囲の中では 0 なので、範囲の中の学習を変えず、クリップ後の行動 (決定論的な評価の行動) も変えない。
境目の 1.0 はクリップの値と同じ。

## `PPO.update` を写さない理由と、勾配が同じになる理由

rsl-rl 3.0.1 の `PPO.update` は損失を 1 行で組み、項を足す口が無い。写すと約 250 行になり、rsl-rl の
更新に追従できない。そこで `update` の間だけ `policy.update_distribution` を包み、勾配を取る呼び出しで
分布の平均を `_AddBoundsGrad` (順方向は恒等、逆方向で `coef · ∂penalty/∂μ` を足す) に通して `Normal` を
作り直す。`update` は mini-batch ごとに `policy.act(obs_batch)` を 1 回だけ呼び、損失はその平均を通る
(log prob) ので、これは損失に `coef · penalty` を足したのと同じ勾配になる (`tests/test_bounded_ppo.py` が
直接の式と比べる)。順方向は恒等なので、surrogate・KL (学習率の適応)・記録する平均は PPO と同じ。

- rollout の `act` は `torch.inference_mode()` の中で呼ばれる (`on_policy_runner.py:100`)。
  `torch.is_grad_enabled()` が偽の呼び出しでは何もしない。
- 罰則の値 (係数を掛ける前) は mini-batch ごとに記録し、`update` の返す `loss_dict["bounds"]` に
  mini-batch の平均を入れる (TensorBoard の `Loss/bounds`。他の `Loss/*` も係数を掛ける前の値)。
  `bounds_loss_coef = 0` でも記録する (勾配は PPO とビット単位で同じ)。
- 対称性のデータ拡張 (`symmetry_cfg`) とは組み合わせない前提 (v13 は使わない)。拡張した行も罰則に入る。

## rsl_rl への登録

`OnPolicyRunner` は `eval(alg_cfg["class_name"])` で algorithm を作る (`on_policy_runner.py:424`)。
`register_with_rsl_rl()` でそのモジュールの名前空間へ登録する (`stage_gated.py` と同じ)。

このモジュールは isaaclab を import しない (Isaac Sim なしで単体テストできる)。
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from dataclasses import dataclass

import torch
from rsl_rl.algorithms import PPO
from torch.distributions import Normal

BOUND = 1.0
"""罰則の境目。環境が行動をクリップする値と同じ。"""


def bounds_penalty(mean: torch.Tensor) -> torch.Tensor:
    """`(..., A)` の平均 -> スカラーの罰則。チャネルで和、それ以外の次元で平均 (surrogate・entropy と同じ取り方)。"""
    return (mean.abs() - BOUND).clamp(min=0.0).square().sum(dim=-1).mean()


class _AddBoundsGrad(torch.autograd.Function):
    """順方向は恒等。逆方向で `coef · ∂bounds_penalty/∂mean` を上流の勾配に足す。"""

    @staticmethod
    def forward(ctx, mean: torch.Tensor, coef: float) -> torch.Tensor:
        ctx.save_for_backward(mean)
        ctx.coef = coef
        return mean.clone()

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        (mean,) = ctx.saved_tensors
        rows = mean.numel() // mean.shape[-1]
        excess = (mean.abs() - BOUND).clamp(min=0.0)
        return grad_out + (ctx.coef * 2.0 / rows) * excess * mean.sign(), None


@dataclass
class BoundsStats:
    """`bounds_loss_on` の間に記録した罰則 (係数を掛ける前) の和と回数。"""

    total: float = 0.0
    count: int = 0


@contextlib.contextmanager
def bounds_loss_on(policy, coef: float) -> Iterator[BoundsStats]:
    """この中で、勾配を取る `policy.update_distribution` の呼び出しに罰則の勾配を足す。

    包むのはインスタンスの属性で、抜けるとき (例外でも) に消してクラスのメソッドに戻す。
    """
    stats = BoundsStats()
    original = policy.update_distribution

    def update_distribution(*args, **kwargs):
        original(*args, **kwargs)
        if not torch.is_grad_enabled():
            return
        dist = policy.distribution
        stats.total += bounds_penalty(dist.mean.detach()).item()
        stats.count += 1
        if coef > 0.0:
            policy.distribution = Normal(_AddBoundsGrad.apply(dist.mean, coef), dist.stddev)

    policy.update_distribution = update_distribution
    try:
        yield stats
    finally:
        del policy.update_distribution


class BoundedPPO(PPO):
    """`PPO` + 平均の範囲外の罰則。

    Args (PPO に加えて):
        bounds_loss_coef: 罰則の係数。0 なら更新は PPO と同じ (罰則の値だけを記録する)。v13 は 0.01。
    """

    def __init__(self, policy, bounds_loss_coef: float = 0.0, **kwargs):
        if bounds_loss_coef < 0.0:
            raise ValueError(f"bounds_loss_coef={bounds_loss_coef} は 0 以上")
        super().__init__(policy, **kwargs)
        self.bounds_loss_coef = bounds_loss_coef

    def update(self):
        with bounds_loss_on(self.policy, self.bounds_loss_coef) as stats:
            loss_dict = super().update()
        loss_dict["bounds"] = stats.total / max(stats.count, 1)
        return loss_dict


def register_with_rsl_rl() -> None:
    """`OnPolicyRunner` の `eval(class_name)` から `"BoundedPPO"` を引けるようにする。"""
    from rsl_rl.runners import on_policy_runner

    on_policy_runner.BoundedPPO = BoundedPPO
