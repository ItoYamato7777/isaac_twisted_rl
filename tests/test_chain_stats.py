"""`common/chain_stats.py` (chain 評価の段ごとの内訳と行動の張り付き) の試験。Isaac Sim は要らない。"""

import math

import pytest
import torch
from chain_stats import action_saturation, chain_breakdown, wilson_interval


def _episodes():
    """4 環境・最大 4 手の小さな例。

    env 0: 段 0, 1, 2 と進み、3 手目で固結び (長さ 3)
    env 1: 段 0, 0, 0, 0 で時間切れ (長さ 4)
    env 2: 段 0, 1 で 2 手目に発散 (長さ 2)。その後の手は 2 つ目のエピソードなので数えない
    env 3: 段 0, 1 で 2 手目に固結び (段 2 を飛ばした。長さ 2)
    """
    stage = torch.tensor(
        [
            [0, 0, 0, 0],
            [1, 0, 1, 1],
            [2, 0, 0, 0],
            [0, 0, 1, 0],
        ]
    )
    length = torch.tensor([3, 4, 2, 2])
    active = torch.arange(4).unsqueeze(1) < length.unsqueeze(0)
    counted = torch.ones(4, dtype=torch.bool)
    outcomes = {
        "time_out": torch.tensor([False, True, False, False]),
        "diverged": torch.tensor([False, False, True, False]),
        "stage_up": torch.tensor([True, False, False, True]),
        "stage_down": torch.zeros(4, dtype=torch.bool),
    }
    max_stage = torch.tensor([3, 0, 1, 3])
    return stage, active, counted, outcomes, max_stage, length


def test_outcome_by_last_stage_sums_to_one_and_uses_last_counted_move():
    out = chain_breakdown(*_episodes())
    table = out["outcome_by_last_stage"]
    assert set(table) == {"knot", "time_out", "diverged", "stage_down"}
    assert table["knot"] == {"0": 0.0, "1": 0.25, "2": 0.25, "3": 0.0}
    assert table["time_out"]["0"] == 0.25
    # env 2 の 3 手目 (段 0) は 2 つ目のエピソードの手なので、最後の段は 1
    assert table["diverged"] == {"0": 0.0, "1": 0.25, "2": 0.0, "3": 0.0}
    assert math.isclose(sum(v for row in table.values() for v in row.values()), 1.0)


def test_exclusive_outcome_priority():
    """終了項が同じ手に 2 つ立ったら、固結び > 発散 > 段の下降 > その他 > 時間切れ の順で 1 つに数える。"""
    stage, active, counted, outcomes, max_stage, length = _episodes()
    outcomes["time_out"] = torch.ones(4, dtype=torch.bool)
    table = chain_breakdown(stage, active, counted, outcomes, max_stage, length)["outcome_by_last_stage"]
    assert sum(table["knot"].values()) == 0.5
    assert sum(table["diverged"].values()) == 0.25
    assert sum(table["time_out"].values()) == 0.25


def test_stage_reach_and_moves():
    out = chain_breakdown(*_episodes())
    reach = out["stage_reach"]
    # 段 1 以上: env 0 (1 手の後), 2 (1 手の後), 3 (1 手の後)
    assert reach["1"]["frac"] == 0.75
    assert reach["1"]["mean_moves"] == 1.0
    # 段 2 以上: env 0 は 2 手の後に観測で、env 3 は段 2 を飛ばして固結びの手 (2 手目) で着いた
    assert reach["2"]["frac"] == 0.5
    assert reach["2"]["mean_moves"] == 2.0
    # 段 3 (固結び) は固結びの手数 (mean_actions_to_knot と同じ)
    assert reach["3"]["frac"] == 0.5
    assert reach["3"]["mean_moves"] == 2.5


def test_moves_at_stage():
    out = chain_breakdown(*_episodes())
    mas = out["moves_at_stage"]
    # 段 0 の手: env 0 が 1, env 1 が 4, env 2 が 1, env 3 が 1 -> 平均 7 / 4
    assert mas["0"] == {"mean": 1.75, "visited_frac": 1.0, "mean_if_visited": 1.75}
    # 段 1 の手: env 0, 2, 3 が 1 ずつ
    assert mas["1"] == {"mean": 0.75, "visited_frac": 0.75, "mean_if_visited": 1.0}
    assert mas["2"] == {"mean": 0.25, "visited_frac": 0.25, "mean_if_visited": 1.0}
    assert mas["3"] == {"mean": 0.0, "visited_frac": 0.0, "mean_if_visited": None}


def test_conditional_rates_and_wilson():
    out = chain_breakdown(*_episodes())
    assert out["p_ge2_given_ge1"] == pytest.approx(2 / 3)
    assert out["p_knot_given_ge2"] == 1.0
    lo, hi = out["knot_ci95"]
    assert (lo, hi) == pytest.approx(wilson_interval(2, 4))


def test_uncounted_envs_are_excluded():
    stage, active, counted, outcomes, max_stage, length = _episodes()
    counted = torch.tensor([True, True, True, False])
    outcomes = {k: v & counted for k, v in outcomes.items()}
    out = chain_breakdown(stage, active, counted, outcomes, max_stage, length)
    assert out["stage_reach"]["3"]["frac"] == pytest.approx(1 / 3)
    assert out["moves_at_stage"]["0"]["mean"] == 2.0


def test_zero_denominators_give_none():
    stage = torch.zeros(2, 1, dtype=torch.long)
    active = torch.ones(2, 1, dtype=torch.bool)
    counted = torch.ones(1, dtype=torch.bool)
    outcomes = {"time_out": torch.tensor([True]), "stage_up": torch.tensor([False])}
    out = chain_breakdown(stage, active, counted, outcomes, torch.tensor([0]), torch.tensor([2]))
    assert out["p_ge2_given_ge1"] is None
    assert out["p_knot_given_ge2"] is None
    assert out["stage_reach"]["1"] == {"frac": 0.0, "mean_moves": None}
    assert out["outcome_by_last_stage"]["time_out"]["0"] == 1.0


def test_wilson_interval_known_values():
    lo, hi = wilson_interval(335, 512)
    p = 335 / 512
    assert lo < p < hi
    # 正規近似の ±1.96 SE とほぼ同じ (p が 0・1 から離れ、n が大きいとき)
    se = math.sqrt(p * (1 - p) / 512)
    assert lo == pytest.approx(p - 1.959964 * se, abs=2e-3)
    assert hi == pytest.approx(p + 1.959964 * se, abs=2e-3)
    assert wilson_interval(0, 10)[0] == 0.0
    assert wilson_interval(10, 10)[1] == pytest.approx(1.0)
    assert wilson_interval(0, 0) is None


def test_action_saturation_by_stage_and_channel():
    # 2 手 x 2 環境 x 2 チャネル
    stage = torch.tensor([[0, 1], [0, 0]])
    active = torch.tensor([[True, True], [True, False]])
    actions = torch.tensor(
        [
            [[1.5, 0.0], [-2.0, 0.5]],
            [[0.5, -1.25], [9.0, 9.0]],  # env 1 の 2 手目は数えない
        ]
    )
    out = action_saturation(stage, active, actions, channels=("a", "b"))
    assert out["channels"] == ["a", "b"]
    s0 = out["by_stage"]["0"]
    assert s0["moves"] == 2
    assert s0["sat_frac"] == [0.5, 0.5]
    assert s0["sat_any"] == 1.0
    # 罰則 Σ max(|a| - 1, 0)^2 の手の平均: (0.25 + 0.0625) / 2
    assert s0["penalty"] == pytest.approx(0.15625)
    s1 = out["by_stage"]["1"]
    assert s1 == {"moves": 1, "sat_frac": [1.0, 0.0], "sat_any": 1.0, "penalty": 1.0}
    assert out["all"]["moves"] == 3
    assert out["all"]["penalty"] == pytest.approx((0.25 + 0.0625 + 1.0) / 3)
    assert "2" not in out["by_stage"]


def test_action_saturation_penalty_matches_bounded_ppo():
    """罰則は BoundedPPO の `bounds_penalty` (手の平均) と同じ式。"""
    from policies.bounded_ppo import bounds_penalty

    g = torch.Generator().manual_seed(0)
    actions = torch.randn(3, 5, 6, generator=g) * 1.5
    stage = torch.zeros(3, 5, dtype=torch.long)
    active = torch.ones(3, 5, dtype=torch.bool)
    out = action_saturation(stage, active, actions)
    assert out["all"]["penalty"] == pytest.approx(float(bounds_penalty(actions.reshape(-1, 6))))
    assert out["channels"] == ["grasp", "delta_link", "drop_x", "drop_y", "drop_z", "yaw"]


def test_action_saturation_unknown_channel_count_uses_indices():
    out = action_saturation(
        torch.zeros(1, 1, dtype=torch.long), torch.ones(1, 1, dtype=torch.bool), torch.zeros(1, 1, 3)
    )
    assert out["channels"] == ["ch0", "ch1", "ch2"]
