"""段ごとのエージェント `common/policies/stage_gated.py` の単体テスト。v4 で追加。

Isaac Sim を起動せずに走る (torch と rsl_rl だけ)::

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/test_stage_gated_policy.py

## 検証の骨子

* 観測の末尾の段の one-hot で actor / critic / std のヘッドが選ばれること。段 3 (固結び) は最後のヘッド。
* 選ばれなかった段のパラメータには勾配が流れないこと (= K 個の独立したエージェント)。
* Isaac Lab の export と同じく `torch.jit.script` / ONNX にできること。
* `num_stages=1` (G 版) は rsl_rl の `ActorCritic` と同じ形になること。
* `register_with_rsl_rl` の後、`OnPolicyRunner` と同じ `eval` で引けること。
"""

from __future__ import annotations

import copy
import io

import pytest
import torch
from policies.stage_gated import GatedMLP, StageGatedActorCritic, register_with_rsl_rl
from rsl_rl.modules import ActorCritic

OBS, ACT, STAGE_DIMS = 12, 6, 4
GROUPS = {"policy": ["policy"], "critic": ["policy"]}
STAGES = torch.tensor([0, 1, 2, 3, 0, 1, 2, 2])


def _obs(stages: torch.Tensor = STAGES, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(stages.numel(), OBS, generator=g)
    x[:, -STAGE_DIMS:] = 0.0
    x[torch.arange(stages.numel()), OBS - STAGE_DIMS + stages] = 1.0
    return x


def _policy(num_stages: int = 3, **kwargs) -> StageGatedActorCritic:
    torch.manual_seed(0)
    return StageGatedActorCritic(
        {"policy": _obs()},
        GROUPS,
        ACT,
        num_stages=num_stages,
        stage_obs_dims=STAGE_DIMS,
        actor_hidden_dims=[16, 16],
        critic_hidden_dims=[16],
        **kwargs,
    )


def test_heads_are_selected_by_stage_one_hot():
    ac = _policy()
    x = _obs()
    mean = ac.act_inference({"policy": x})
    value = ac.evaluate({"policy": x})
    for i, s in enumerate(STAGES.tolist()):
        head = min(s, 2)  # 段 3 (固結び) は最後のヘッド
        assert torch.allclose(mean[i], ac.actor.heads[head](x[i : i + 1])[0], atol=1e-6)
        assert torch.allclose(value[i], ac.critic.heads[head](x[i : i + 1])[0], atol=1e-6)


def test_std_is_per_stage():
    ac = _policy(init_noise_std=0.5)
    with torch.no_grad():
        ac.std[1] += 1.0
    ac.update_distribution(_obs())
    expected = torch.tensor([0.5, 1.5, 0.5, 0.5, 0.5, 1.5, 0.5, 0.5])
    assert torch.allclose(ac.action_std[:, 0], expected)
    assert ac.act({"policy": _obs()}).shape == (STAGES.numel(), ACT)


def test_log_std_type():
    ac = _policy(init_noise_std=0.5, noise_std_type="log")
    ac.update_distribution(_obs())
    assert torch.allclose(ac.action_std, torch.full((STAGES.numel(), ACT), 0.5))


def test_gradients_reach_only_selected_stage():
    ac = _policy()
    x = _obs(torch.tensor([1, 1, 1]))
    ac.act({"policy": x})
    loss = ac.evaluate({"policy": x}).sum() + ac.get_actions_log_prob(torch.zeros(3, ACT)).sum()
    loss.backward()

    def touched(module) -> bool:
        return any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())

    assert [touched(h) for h in ac.actor.heads] == [False, True, False]
    assert [touched(h) for h in ac.critic.heads] == [False, True, False]
    assert ac.std.grad[0].abs().sum() == 0 and ac.std.grad[2].abs().sum() == 0
    assert ac.std.grad[1].abs().sum() > 0


def test_export_jit_and_onnx():
    ac = _policy()
    actor = copy.deepcopy(ac.actor)
    scripted = torch.jit.script(actor)
    x = _obs()
    assert torch.allclose(scripted(x), ac.act_inference({"policy": x}), atol=1e-6)
    torch.onnx.export(copy.deepcopy(ac.actor), x[:1], io.BytesIO())
    # Isaac Lab の ONNX exporter は入力次元を actor[0].in_features で読む。
    assert ac.actor[0].in_features == OBS


def test_single_stage_matches_actor_critic_shapes():
    g = _policy(num_stages=1)
    torch.manual_seed(0)
    base = ActorCritic({"policy": _obs()}, GROUPS, ACT, actor_hidden_dims=[16, 16], critic_hidden_dims=[16])
    assert g.std.shape == (1, ACT)
    assert [p.shape for p in g.actor.heads[0].parameters()] == [p.shape for p in base.actor.parameters()]
    x = _obs()
    assert torch.allclose(g.act_inference({"policy": x}), g.actor.heads[0](x))


def test_rejects_normalization_and_bad_dims():
    with pytest.raises(ValueError):
        _policy(actor_obs_normalization=True)
    with pytest.raises(ValueError):
        StageGatedActorCritic({"policy": _obs()}, GROUPS, ACT, num_stages=5, stage_obs_dims=STAGE_DIMS)


def test_gated_mlp_handles_extra_batch_dims():
    heads = [torch.nn.Linear(OBS, 2) for _ in range(3)]
    gated = GatedMLP(heads, STAGE_DIMS)
    x = _obs().view(2, 4, OBS)
    out = gated(x)
    assert out.shape == (2, 4, 2)
    assert torch.allclose(out.view(-1, 2), gated(x.view(-1, OBS)))


def test_register_with_rsl_rl():
    from rsl_rl.runners import on_policy_runner

    register_with_rsl_rl()
    # OnPolicyRunner._construct_algorithm と同じ解決のしかた。
    assert eval("StageGatedActorCritic", vars(on_policy_runner)) is StageGatedActorCritic


# ---------------------------------------------------------------- v14: critic に特権観測を前に足す


PRIV = 6


def _wide_critic_policy() -> StageGatedActorCritic:
    torch.manual_seed(0)
    obs = {"policy": _obs(), "rope_params": torch.randn(STAGES.numel(), PRIV)}
    return StageGatedActorCritic(
        obs,
        {"policy": ["policy"], "critic": ["rope_params", "policy"]},
        ACT,
        num_stages=3,
        stage_obs_dims=STAGE_DIMS,
        actor_hidden_dims=[16, 16],
        critic_hidden_dims=[16],
    )


def test_wide_critic_routes_by_trailing_stage_columns():
    # critic の観測は [特権 6, policy]。段の one-hot は末尾に残るので、critic も actor と同じ段の頭を選ぶ。
    p = _wide_critic_policy()
    x = _obs()
    priv = torch.randn(STAGES.numel(), PRIV)
    critic_in = torch.cat([priv, x], dim=-1)
    assert p.critic.heads[0][0].in_features == OBS + PRIV
    assert p.actor.heads[0][0].in_features == OBS
    assert p.critic.stage_index(critic_in).tolist() == STAGES.clamp(max=2).tolist()
    v = p.evaluate({"policy": x, "rope_params": priv})
    for i, s in enumerate(STAGES.clamp(max=2).tolist()):
        assert torch.allclose(v[i], p.critic.heads[s](critic_in[i]))


def test_export_policy_reads_critic_width():
    import importlib.util
    import pathlib

    path = pathlib.Path(__file__).resolve().parents[1] / "src/isaac_twisted_rl/scripts/rsl_rl/export_policy.py"
    spec = importlib.util.spec_from_file_location("export_policy", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    for policy in (_policy(), _wide_critic_policy()):
        state = policy.state_dict()
        obs, groups = mod._obs_shapes(state)
        rebuilt = StageGatedActorCritic(
            obs,
            groups,
            ACT,
            num_stages=3,
            stage_obs_dims=STAGE_DIMS,
            actor_hidden_dims=[16, 16],
            critic_hidden_dims=[16],
        )
        rebuilt.load_state_dict(state)  # strict
        x = _obs()
        assert torch.allclose(rebuilt.act_inference({"policy": x}), policy.act_inference({"policy": x}))
