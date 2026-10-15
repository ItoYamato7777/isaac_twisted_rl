"""段ごとの価値 (critic の出力) の統計。崩れの早期警報として、学習ログ (`Loss/value_*`) に出す。

## なぜ数えるか

段のエピソードは最大 6 手で、時間切れは `time_out=True` なので PPO は時間切れの手の報酬に `γ·V(s')` を足す
(bootstrap。`versions/v4/README.md`)。一方、段 0 で正の報酬は段を上げたときの +1 だけで、ほかは負なので、
段 0 の V は 1 を超えないはずである。V が `1/γ` を超えると、段を上げずに時間切れを待つ方が得になる。
v11・v12・v13 の s44 と v11 s43 の延長で段 0 が崩れた run は、どれも V が `1/γ` を超えて膨らんだ後、
段 0 の時間切れが増えた (`docs/experiments/2026-10-15_critic-inflation.md`。相関と時間の順序まで。因果は未確認)。
崩れない run は、段 0 の V が 0.9〜1.0 のままだった。

学習ログの段ごとの成功率は、崩れてからしか動かない。V の統計は崩れる前に動くので、警報に使える。

## 出す値

段 k ごとに、rollout (1 iteration 分) の `storage.values` から:

* `value_mean_s{k}`: V の平均。
* `value_over_s{k}`: V が閾値 (`break_even_value(γ)` = 1/γ) を**超えた**割合 (境目ちょうどは数えない)。

サンプルが無い段は出さない。段は観測の末尾の one-hot で決まり、範囲外 (固結び) は最後の段に入れる
(`stage_gated.GatedMLP.stage_index` と同じ)。値は更新の前の rollout の価値で、学習の更新のパラメータや
乱数には触れない (`tests/test_bounded_ppo.py`)。

このモジュールは isaaclab を import しない。
"""

from __future__ import annotations

import torch


def break_even_value(gamma: float) -> float:
    """時間切れの報酬 `γ·V` が、段を上げる報酬 +1 と並ぶ価値 `1/γ`。"""
    if not 0.0 < gamma <= 1.0:
        raise ValueError(f"gamma={gamma} は (0, 1] の範囲")
    return 1.0 / gamma


def stage_value_stats(values: torch.Tensor, stage: torch.Tensor, num_stages: int, threshold: float) -> dict[str, float]:
    """段ごとの V の平均と、V が `threshold` を超えた割合。

    Args:
        values: `(...)` か `(..., 1)` の価値。
        stage: `(...)` の段 (整数)。`num_stages` 以上は最後の段に入れる。
        num_stages: 段の数 (`StageGatedActorCritic.num_stages`)。
        threshold: 「超えた」の境目 (`break_even_value(γ)`)。
    """
    with torch.no_grad():
        v = values.detach().reshape(-1).double()
        s = stage.reshape(-1).clamp(max=num_stages - 1)
        out: dict[str, float] = {}
        for k in range(num_stages):
            sel = s == k
            if not bool(sel.any()):
                continue
            vk = v[sel]
            out[f"value_mean_s{k}"] = float(vk.mean())
            out[f"value_over_s{k}"] = float((vk > threshold).double().mean())
    return out
