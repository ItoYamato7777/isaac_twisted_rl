"""Play のグリッパの色分け `scripts/rsl_rl/stage_colors.py` の単体テスト。

Isaac Sim を起動せずに走る (torch と rsl_rl だけ)。色を USD に書く部分は GUI で確かめる。

## 検証の骨子

* 色の番号は、方策が振り分けに使う `GatedMLP.stage_index` と同じ (段 3 = 固結びは最後のヘッドの色)。
* 固結びに達した環境だけが `KNOT_HOLD_STEPS` 手のあいだ到達の色になり、その後は方策の色に戻る。
* 段ごとの方策でなければ、何もしない (例外にしない)。
"""

from __future__ import annotations

import torch
from policies.stage_gated import GatedMLP
from torch import nn

from isaac_twisted_rl.scripts.rsl_rl.stage_colors import (
    KNOT_HOLD_STEPS,
    KNOT_INDEX,
    STAGE_COLOR_NAMES,
    STAGE_COLORS,
    GripperStageColorizer,
    color_indices,
    update_hold,
)


def test_color_follows_gated_head():
    """色の番号 = 方策が選ぶヘッドの番号。段 3 (固結び) は最後のヘッド (段 2) の色。"""
    actor = GatedMLP([nn.Linear(6, 2) for _ in range(3)], stage_dims=4)
    stages = torch.tensor([0, 1, 2, 3])
    obs = torch.cat([torch.randn(4, 2), torch.nn.functional.one_hot(stages, 4).float()], dim=-1)
    idx = color_indices(actor.stage_index(obs), torch.zeros(4, dtype=torch.long))
    assert idx.tolist() == [0, 1, 2, 2]


def test_knot_color_is_held_then_released():
    """固結びに達した環境だけ、続く KNOT_HOLD_STEPS 手が到達の色。"""
    policy_idx = torch.tensor([0, 1])
    hold = torch.zeros(2, dtype=torch.long)
    hold = update_hold(hold, torch.tensor([True, False]))
    shown = []
    for _ in range(KNOT_HOLD_STEPS + 1):
        shown.append(color_indices(policy_idx, hold).tolist())
        hold = update_hold(hold, torch.tensor([False, False]))
    assert shown == [[KNOT_INDEX, 1]] * KNOT_HOLD_STEPS + [[0, 1]]


def test_knot_again_restarts_hold():
    """到達の色を残している間にまた固結びに達したら、残す手数を数え直す。"""
    hold = torch.tensor([1])
    assert update_hold(hold, torch.tensor([True])).tolist() == [KNOT_HOLD_STEPS]


def test_palette_sizes_agree():
    assert len(STAGE_COLORS) == len(STAGE_COLOR_NAMES) == KNOT_INDEX


def test_disabled_without_gated_policy():
    """段ごとの方策でなければ無効になり、呼ばれても何もしない (pxr も要らない)。"""
    policy_nn = nn.Module()
    policy_nn.actor = nn.Linear(4, 2)
    colorizer = GripperStageColorizer(env=None, policy_nn=policy_nn)
    assert not colorizer.enabled
    colorizer.before_step({"policy": torch.zeros(1, 4)})
    colorizer.after_step(torch.zeros(1))
