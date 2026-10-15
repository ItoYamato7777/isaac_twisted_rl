"""`scripts/rsl_rl/critic_stage_values.py` と `tb_table.py` の単体テスト。Isaac Sim なしで回る。

## 検証の骨子

* チェックポイントから作った critic の出力が、学習側 (`StageGatedActorCritic`) の critic と同じになる。
* critic の入力の次元が固定した観測と違う (v14 の特権観測) ときは、黙って測らず例外にする。
* 段ごとの統計が `value_stats.stage_value_stats` と同じ数え方 (1/γ を**超えた**割合) になる。
* `tb_table.window_mean` の窓の端 (両端を含む・欠けた点は飛ばす・点が無ければ None)。
"""

from __future__ import annotations

import pytest
import torch
from policies.stage_gated import StageGatedActorCritic
from policies.value_stats import break_even_value, stage_value_stats

from isaac_twisted_rl.scripts.rsl_rl.critic_stage_values import load_critic, stage0_by_outcome, stage_stats
from isaac_twisted_rl.scripts.rsl_rl.tb_table import window_mean

OBS, ACT, STAGE_DIMS = 12, 6, 4
GROUPS = {"policy": ["policy"], "critic": ["policy"]}


def _obs(n: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, OBS, generator=g)
    stages = torch.randint(0, 4, (n,), generator=g)  # 段 3 (固結び) も入れる
    x[:, -STAGE_DIMS:] = 0.0
    x[torch.arange(n), OBS - STAGE_DIMS + stages] = 1.0
    return x


def _checkpoint(tmp_path, critic_obs_dim: int = OBS):
    torch.manual_seed(0)
    ac = StageGatedActorCritic(
        {"policy": _obs(4)},
        GROUPS,
        ACT,
        num_stages=3,
        stage_obs_dims=STAGE_DIMS,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
    )
    sd = ac.state_dict()
    if critic_obs_dim != OBS:  # 入力の次元だけが違う critic (特権観測の版に当たる)
        sd = {k: v for k, v in sd.items() if not k.startswith("critic.")}
        wide = StageGatedActorCritic(
            {"policy": torch.randn(4, critic_obs_dim)},
            {"policy": ["policy"], "critic": ["policy"]},
            ACT,
            num_stages=3,
            stage_obs_dims=STAGE_DIMS,
            critic_hidden_dims=[512, 256, 128],
        )
        sd |= {k: v for k, v in wide.state_dict().items() if k.startswith("critic.")}
    path = tmp_path / "model_10.pt"
    torch.save({"model_state_dict": sd}, path)
    return ac, str(path)


def test_loaded_critic_matches_training_critic(tmp_path):
    ac, path = _checkpoint(tmp_path)
    critic = load_critic(path, OBS)
    x = _obs(32, seed=1)
    with torch.no_grad():
        assert torch.allclose(critic(x), ac.critic(x), atol=1e-6)
    assert critic.num_heads == 3


def test_load_critic_rejects_different_input_dim(tmp_path):
    _, path = _checkpoint(tmp_path, critic_obs_dim=OBS + 8)
    with pytest.raises(ValueError, match="入力は"):
        load_critic(path, OBS)


def test_stage_stats_use_value_stats_counting(tmp_path):
    ac, path = _checkpoint(tmp_path)
    critic = load_critic(path, OBS)
    x = _obs(256, seed=2)
    thr = break_even_value(0.95)
    out = stage_stats(critic, x, thr)
    with torch.no_grad():
        v = critic(x).squeeze(-1)
    expected = stage_value_stats(v, critic.stage_index(x), 3, thr)
    assert set(out) == {"0", "1", "2"}
    for k in "012":
        assert out[k]["mean"] == pytest.approx(expected[f"value_mean_s{k}"], rel=1e-6)
        assert out[k]["over"] == pytest.approx(expected[f"value_over_s{k}"], rel=1e-6)
        assert out[k]["max"] >= out[k]["p95"]
    assert sum(out[k]["n"] for k in out) == 256  # 段 3 は最後のヘッドに入る


def test_stage0_by_outcome_groups_stage0_only(tmp_path):
    _, path = _checkpoint(tmp_path)
    critic = load_critic(path, OBS)
    x = _obs(64, seed=3)
    before = critic.stage_index(x)
    outcome = torch.arange(64) % 3
    out = stage0_by_outcome(critic, x, before, outcome, ("up", "down", "same_shape", "unused"))
    assert "unused" not in out
    assert sum(r["n"] for r in out.values()) == int((before == 0).sum())
    with torch.no_grad():
        v = critic(x).squeeze(-1)
    sel = (before == 0) & (outcome == 0)
    assert out["up"]["mean"] == pytest.approx(float(v[sel].mean()), rel=1e-6)


def test_window_mean_edges():
    series = {10: 1.0, 11: 2.0, 12: 3.0, 14: 10.0}
    assert window_mean(series, 11, -1, 1) == pytest.approx(2.0)  # 両端を含む
    assert window_mean(series, 12, -2, 2) == pytest.approx((1.0 + 2.0 + 3.0 + 10.0) / 4)  # 欠けた 13 は飛ばす
    assert window_mean(series, 12, -4, 0) == pytest.approx(2.0)
    assert window_mean(series, 20, -2, 2) is None
