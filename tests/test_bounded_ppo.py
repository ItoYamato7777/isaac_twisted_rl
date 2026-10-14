"""PPO の損失に平均の範囲外の罰則を足す `common/policies/bounded_ppo.py` の単体テスト。v13 で追加。

Isaac Sim を起動せずに走る (torch と rsl_rl だけ)::

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run python -m pytest tests/test_bounded_ppo.py

## 検証の骨子

`BoundedPPO` は `PPO.update` を写さず、`policy.update_distribution` を包んで勾配だけを足す
(`docs/decisions/0026-v13-action-bounds-loss.md`)。そのため「損失に罰則の項を足したのと同じ」ことを
直接の式と比べて確かめる:

* 包んだ平均の勾配が、`loss + coef · penalty` を直接書いたときの勾配と一致する (損失の関数の単位と、
  `update` 全体の単位の両方。後者は optimizer を SGD にして、PPO との更新の差が `−lr · coef · ∂penalty/∂θ` になること)。
* 平均が範囲の中 (|μ| ≤ 1) なら、更新は PPO とビット単位で同じ。
* `bounds_loss_coef = 0` なら、平均が範囲の外でも更新は PPO とビット単位で同じ (罰則の値は記録する)。
* inference mode (rollout の `act`) では分布を作り直さず、罰則も数えない。
"""

from __future__ import annotations

import copy

import pytest
import torch
from policies.bounded_ppo import BoundedPPO, bounds_loss_on, bounds_penalty, register_with_rsl_rl
from policies.stage_gated import StageGatedActorCritic
from rsl_rl.algorithms import PPO
from tensordict import TensorDict
from torch import nn

OBS, ACT, STAGE_DIMS = 12, 6, 4
GROUPS = {"policy": ["policy"], "critic": ["policy"]}
NUM_ENVS, STEPS = 16, 4
COEF = 0.5  # 罰則の勾配が PPO の勾配に埋もれず、差が目に見える大きさにする


def _obs(n: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, OBS, generator=g)
    stages = torch.randint(0, 3, (n,), generator=g)
    x[:, -STAGE_DIMS:] = 0.0
    x[torch.arange(n), OBS - STAGE_DIMS + stages] = 1.0
    return x


def _td(x: torch.Tensor) -> TensorDict:
    return TensorDict({"policy": x}, batch_size=[x.shape[0]])


def _policy(outside: bool) -> StageGatedActorCritic:
    """`outside` なら平均の一部のチャネルが範囲の外 (|μ| > 1)、そうでなければ全部が範囲の中。"""
    torch.manual_seed(0)
    ac = StageGatedActorCritic(
        {"policy": _obs(4, 0)},
        GROUPS,
        ACT,
        num_stages=3,
        stage_obs_dims=STAGE_DIMS,
        actor_hidden_dims=[16, 16],
        critic_hidden_dims=[16],
        init_noise_std=0.5,
    )
    with torch.no_grad():
        for head in ac.actor.heads:
            last = [m for m in head.modules() if isinstance(m, nn.Linear)][-1]
            if outside:
                last.weight.mul_(3.0)
                last.bias[: ACT // 2] = torch.tensor([2.0, -2.5, 1.5])  # 前半は外、後半は 0 のあたり
            else:
                last.weight.mul_(0.01)
                last.bias.zero_()
    return ac


def _loss(ac: StageGatedActorCritic, obs: TensorDict, actions: torch.Tensor, adv: torch.Tensor) -> torch.Tensor:
    """PPO の損失の actor の部分と同じ形 (log prob と entropy)。"""
    ac.act(obs)
    return -(ac.get_actions_log_prob(actions) * adv).mean() - 0.01 * ac.entropy.mean()


def _grads(ac: nn.Module) -> list[torch.Tensor]:
    return [p.grad.clone() if p.grad is not None else torch.zeros_like(p) for p in ac.parameters()]


def test_penalty_definition():
    mu = torch.tensor([[0.5, -1.0, 1.5], [-3.0, 2.0, 0.0]])
    # 行ごとにチャネルの和 (0 + 0 + 0.25, 4 + 1 + 0)、行の平均。
    assert torch.isclose(bounds_penalty(mu), torch.tensor((0.25 + 5.0) / 2))
    assert bounds_penalty(mu.view(1, 2, 3)) == bounds_penalty(mu)


def test_gradient_matches_explicit_penalty():
    ac = _policy(outside=True)
    obs = _td(_obs(32, 1))
    g = torch.Generator().manual_seed(2)
    actions, adv = torch.randn(32, ACT, generator=g), torch.randn(32, generator=g)

    wrapped = copy.deepcopy(ac)
    with bounds_loss_on(wrapped, COEF) as stats:
        _loss(wrapped, obs, actions, adv).backward()

    explicit = copy.deepcopy(ac)
    loss = _loss(explicit, obs, actions, adv)
    penalty = bounds_penalty(explicit.action_mean)
    (loss + COEF * penalty).backward()

    plain = copy.deepcopy(ac)
    _loss(plain, obs, actions, adv).backward()

    assert penalty > 0.1
    assert stats.count == 1 and stats.total == pytest.approx(penalty.item(), rel=1e-6)
    for gw, ge, gp in zip(_grads(wrapped), _grads(explicit), _grads(plain)):
        assert torch.allclose(gw, ge, atol=1e-6)
    assert any(not torch.allclose(gw, gp, atol=1e-4) for gw, gp in zip(_grads(wrapped), _grads(plain)))
    # 順方向は恒等: 平均そのものと、それを使う量 (log prob・KL の材料) は変わらない。
    assert torch.equal(wrapped.action_mean, plain.action_mean)


def test_wrapper_is_removed_after_context():
    ac = _policy(outside=True)
    with bounds_loss_on(ac, COEF):
        assert "update_distribution" in vars(ac)
    assert "update_distribution" not in vars(ac)
    with pytest.raises(RuntimeError), bounds_loss_on(ac, COEF):
        raise RuntimeError
    assert "update_distribution" not in vars(ac)


def test_inference_mode_is_untouched():
    ac = _policy(outside=True)
    obs = _td(_obs(32, 1))
    with bounds_loss_on(ac, COEF) as stats:
        with torch.inference_mode():
            ac.act(obs)
            dist = ac.distribution
            assert torch.equal(dist.mean, ac.actor(obs["policy"]))
        with torch.no_grad():
            ac.act(obs)
    assert stats.count == 0 and stats.total == 0.0
    assert dist.mean.grad_fn is None


def _filled(cls, ac: StageGatedActorCritic, **kwargs) -> PPO:
    """rollout を 1 回分ためた algorithm。データは乱数の種で決まり、`cls` によらない。"""
    alg = cls(
        copy.deepcopy(ac),
        num_learning_epochs=1,
        num_mini_batches=1,
        learning_rate=1e-3,
        schedule="fixed",
        desired_kl=None,
        max_grad_norm=1e9,
        device="cpu",
        **kwargs,
    )
    alg.init_storage("rl", NUM_ENVS, STEPS, _td(_obs(NUM_ENVS, 0)), [ACT])
    torch.manual_seed(3)
    g = torch.Generator().manual_seed(4)
    for t in range(STEPS):
        obs = _td(_obs(NUM_ENVS, 10 + t))
        with torch.inference_mode():
            alg.act(obs)
            alg.process_env_step(obs, torch.randn(NUM_ENVS, generator=g), torch.zeros(NUM_ENVS), {})
    with torch.inference_mode():
        alg.compute_returns(_td(_obs(NUM_ENVS, 99)))
    return alg


def _update(alg: PPO) -> dict:
    torch.manual_seed(5)  # mini-batch の並べ替え
    return alg.update()


def _params(alg: PPO) -> list[torch.Tensor]:
    return [p.detach().clone() for p in alg.policy.parameters()]


def test_update_equals_ppo_plus_penalty_gradient():
    ac = _policy(outside=True)
    lr = 1e-2
    ppo, bounded = _filled(PPO, ac), _filled(BoundedPPO, ac, bounds_loss_coef=COEF)
    for alg in (ppo, bounded):
        alg.optimizer = torch.optim.SGD(alg.policy.parameters(), lr=lr)

    # 罰則の勾配を、更新の前の方策で、rollout の観測の全部 (= 1 つの mini-batch) について直接計算する。
    ref = copy.deepcopy(bounded.policy)
    ref.act(bounded.storage.observations.flatten(0, 1))
    penalty = bounds_penalty(ref.action_mean)
    penalty.backward()
    pen_grads = _grads(ref)

    _update(ppo)
    loss_dict = _update(bounded)
    assert loss_dict["bounds"] == pytest.approx(penalty.item(), rel=1e-5)
    assert set(loss_dict) == {"value_function", "surrogate", "entropy", "bounds"}
    for p_ppo, p_b, g in zip(_params(ppo), _params(bounded), pen_grads):
        assert torch.allclose(p_b, p_ppo - lr * COEF * g, atol=1e-6)
    assert "update_distribution" not in vars(bounded.policy)


def test_update_unchanged_inside_bounds():
    ac = _policy(outside=False)
    ppo, bounded = _filled(PPO, ac), _filled(BoundedPPO, ac, bounds_loss_coef=COEF)
    assert bounded.storage.mu.abs().max() < 1.0
    _update(ppo)
    loss_dict = _update(bounded)
    assert loss_dict["bounds"] == 0.0
    for a, b in zip(_params(ppo), _params(bounded)):
        assert torch.equal(a, b)


def test_coef_zero_equals_ppo():
    ac = _policy(outside=True)
    ppo, bounded = _filled(PPO, ac), _filled(BoundedPPO, ac, bounds_loss_coef=0.0)
    _update(ppo)
    loss_dict = _update(bounded)
    assert loss_dict["bounds"] > 0.1  # 係数 0 でも値は記録する
    for a, b in zip(_params(ppo), _params(bounded)):
        assert torch.equal(a, b)


def test_rejects_negative_coef():
    with pytest.raises(ValueError):
        BoundedPPO(_policy(outside=False), bounds_loss_coef=-0.1)


def test_register_with_rsl_rl():
    from rsl_rl.runners import on_policy_runner

    register_with_rsl_rl()
    # OnPolicyRunner._construct_algorithm と同じ解決のしかた。
    assert eval("BoundedPPO", vars(on_policy_runner)) is BoundedPPO
