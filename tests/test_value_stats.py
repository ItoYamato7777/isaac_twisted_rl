"""段ごとの価値 (critic の出力) の統計 `common/policies/value_stats.py` の単体テスト。

Isaac Sim を起動せずに走る (torch だけ)::

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run python -m pytest tests/test_value_stats.py

## 検証の骨子

段 0 の価値が 1/γ を超えると、時間切れの bootstrap (γ·V) が段を上げる報酬 (+1) を上回る
(`docs/experiments/2026-10-15_critic-inflation.md`)。その早期警報として、段ごとの V の平均と
「V が閾値を超えた割合」を数える。数え方 (段の振り分け・閾値の境目・空の段の扱い) を直接の式と比べる。
"""

from __future__ import annotations

import pytest
import torch
from policies.value_stats import break_even_value, stage_value_stats


def test_break_even_value_is_inverse_gamma():
    # 時間切れの報酬は γ·V、段を上げる報酬は +1。γ·V > 1 となる V が危険域。
    assert break_even_value(0.95) == pytest.approx(1 / 0.95)
    with pytest.raises(ValueError):
        break_even_value(0.0)
    with pytest.raises(ValueError):
        break_even_value(1.5)


def test_stats_by_stage_match_direct_formula():
    values = torch.tensor([0.5, 1.0, 1.5, 2.0, -1.0, 0.0, 3.0])
    stage = torch.tensor([0, 0, 0, 1, 1, 2, 2])
    out = stage_value_stats(values, stage, num_stages=3, threshold=1.2)
    assert out["value_mean_s0"] == pytest.approx((0.5 + 1.0 + 1.5) / 3)
    assert out["value_over_s0"] == pytest.approx(1 / 3)  # 1.5 だけが 1.2 を超える
    assert out["value_mean_s1"] == pytest.approx(0.5)
    assert out["value_over_s1"] == pytest.approx(1 / 2)
    assert out["value_mean_s2"] == pytest.approx(1.5)
    assert out["value_over_s2"] == pytest.approx(1 / 2)
    assert set(out) == {f"value_{k}_s{s}" for k in ("mean", "over") for s in range(3)}


def test_threshold_is_strict_and_values_may_have_trailing_dim():
    # 境目ちょうどは「超えた」に数えない。(N, 1) の形 (rsl-rl の storage.values) も同じ結果。
    values = torch.tensor([1.0, 1.0, 1.0 + 1e-3])
    stage = torch.tensor([0, 0, 0])
    flat = stage_value_stats(values, stage, 1, threshold=1.0)
    col = stage_value_stats(values.unsqueeze(-1), stage, 1, threshold=1.0)
    assert flat == col
    assert flat["value_over_s0"] == pytest.approx(1 / 3)


def test_stage_beyond_num_stages_goes_to_last_stage_and_empty_stage_is_omitted():
    # 段 3 (固結び) の観測は最後の段のヘッドが使う (`GatedMLP.stage_index` と同じ)。
    values = torch.tensor([1.0, 3.0, 5.0])
    stage = torch.tensor([0, 3, 2])
    out = stage_value_stats(values, stage, num_stages=3, threshold=2.0)
    assert out["value_mean_s2"] == pytest.approx(4.0)
    assert out["value_over_s2"] == 1.0
    assert "value_mean_s1" not in out and "value_over_s1" not in out


def test_extra_batch_dims_and_no_grad_leak():
    values = torch.randn(4, 5, 1, requires_grad=True)
    stage = torch.randint(0, 3, (4, 5))
    out = stage_value_stats(values, stage, 3, threshold=0.0)
    assert all(isinstance(v, float) for v in out.values())
    flat = stage_value_stats(values.detach().reshape(-1), stage.reshape(-1), 3, threshold=0.0)
    assert out == flat
