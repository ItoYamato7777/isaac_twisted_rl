"""chain 評価 (`scripts/rsl_rl/eval_chain.py`) の段ごとの内訳と、評価中の行動の張り付きの集計。

isaaclab を import しない (torch だけ)。`eval_chain.py` は毎手の記録 (手の前の段・数える手か・クリップ前の
行動) と各環境の結末を渡すだけにし、集計はここで単体テストする (`tests/test_chain_stats.py`)。

記録の形 (T = 手の数、n = 環境の数):

* `stage` `(T, n)`: 手を打つ前の観測の段 (段の one-hot の argmax。方策が使うヘッドと同じ)。
* `active` `(T, n)`: その手が各環境の最初のエピソードの手か (`eval_chain.py` の `~counted`)。
* `counted` `(n,)`: 最初のエピソードが終わった環境。内訳はこの環境だけで数える。
* `outcomes`: 終了項の名前 -> `(n,)`。最初のエピソードを終えた手に立った項。
* `max_stage` / `length` `(n,)`: `eval_chain.py` の既存のキー (`max_stage_ge{k}`・`mean_actions_to_knot`) と同じ値。

手の後の段は、エピソードが続くときだけ次の手の前の観測として見える (終わった手の後の観測は次のエピソードの
もの)。固結び (`stage_up`) は段 3 に着いた手として数える (`max_stage` も同じ扱い)。それ以外の結末で最後の手の
後に段が上がっても見えないのは、既存の `max_stage_ge{k}` と同じ。

観測の段は床に落ちているか (`settled`) を見ないが、`stage_up` は床に落ちて発散していない手でだけ立つ。そのため
固結びの形のまま吊っている手の後は、観測の段が 3 でもエピソードが続く (段 3 で打つ手がある)。
"""

from __future__ import annotations

import math

import torch

ACTION_CHANNELS = ("grasp", "delta_link", "drop_x", "drop_y", "drop_z", "yaw")
"""`PickPlaceAction` の 6 チャネル (`common/mdp/actions.py`)。チャネル数が違えば `ch{i}` にする。"""

_OUTCOME_PRIORITY = ("stage_up", "diverged", "stage_down")
"""同じ手に終了項が 2 つ立ったときに 1 つに決める順。ここに無い項はこの後、`time_out` は最後。"""

_Z95 = 1.959963984540054


def _label(name: str) -> str:
    """既存のキーと同じ名前にする (`stage_up` は `knot`)。"""
    return "knot" if name == "stage_up" else name


def _ratio(num: float, den: float) -> float | None:
    return float(num) / float(den) if den > 0 else None


def wilson_interval(k: int, n: int, z: float = _Z95) -> tuple[float, float] | None:
    """二項の割合 k / n の Wilson の信頼区間 (既定は 95%)。n = 0 なら None。

    固結びの割合が 0・1 に近い run (v10 は 0 に近い) でも、正規近似と違って [0, 1] からはみ出さない。
    chain 評価は決定論的なので (`docs/system/evaluation.md` §2)、この区間は測り直しの揺れではなく、
    初期状態の揺らぎの組を引き直したときの揺れを二項で見積もったもの。
    """
    if n <= 0:
        return None
    p = k / n
    den = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, center - half), min(1.0, center + half)


def exclusive_outcomes(outcomes: dict[str, torch.Tensor], counted: torch.Tensor) -> dict[str, torch.Tensor]:
    """終了項ごとの `(n,)` を、各環境がちょうど 1 つに入る結末に分ける (名前は `_label`)。

    既存のキー (`time_out`・`diverged` など) は項ごとに独立に数えるので、同じ手に 2 つ立つと和が 1 を超えうる。
    表は和が 1 になるように、固結び > 発散 > 段の下降 > その他の項 > 時間切れ の順で 1 つに決める。
    """
    names = [k for k in _OUTCOME_PRIORITY if k in outcomes]
    names += [k for k in outcomes if k not in names and k != "time_out"]
    names += [k for k in outcomes if k == "time_out"]
    taken = ~counted.clone()
    out = {}
    for k in names:
        mask = outcomes[k] & ~taken
        out[_label(k)] = mask
        taken |= mask
    other = ~taken
    if bool(other.any()):
        out["other"] = other
    return out


def chain_breakdown(
    stage: torch.Tensor,
    active: torch.Tensor,
    counted: torch.Tensor,
    outcomes: dict[str, torch.Tensor],
    max_stage: torch.Tensor,
    length: torch.Tensor,
    num_stages: int = 4,
) -> dict:
    """結末 x 最後の手の段の表、各段に着いた手、段ごとの手数、条件付きの成功率、`knot` の Wilson 区間。

    v13 s44 のような崩れ (段 0 で時間切れ) を、`diag_stage_moves.py` を別に回さずに JSON で見るため。
    割合の分母は数えた環境の数 (`counted`)。
    """
    stage = stage.long()
    act = active & counted.unsqueeze(0)
    m = int(counted.sum())
    t_idx = torch.arange(stage.shape[0], device=stage.device).unsqueeze(1).expand_as(stage)

    # 最後の手 = 最初のエピソードの最後に数えた手
    last_t = torch.where(act, t_idx, torch.full_like(t_idx, -1)).max(dim=0).values
    last_stage = stage.gather(0, last_t.clamp(min=0).unsqueeze(0)).squeeze(0)
    excl = exclusive_outcomes(outcomes, counted)
    table = {
        name: {str(s): float((mask & (last_stage == s)).sum()) / max(m, 1) for s in range(num_stages)}
        for name, mask in excl.items()
    }

    # 段 k に初めて着いた手数: 観測で見えた手の前の段 (t 手の後) か、見えずに max_stage が k 以上 (固結びの手)
    reach = {}
    big = stage.shape[0] + 1
    for k in range(1, num_stages):
        seen = act & (stage >= k)
        first = torch.where(seen, t_idx, torch.full_like(t_idx, big)).min(dim=0).values
        reached = counted & (max_stage >= k)
        moves = torch.where(first < big, first, length)
        reach[str(k)] = {
            "frac": float(reached.sum()) / max(m, 1),
            "mean_moves": float(moves[reached].float().mean()) if bool(reached.any()) else None,
        }

    moves_at = {}
    for s in range(num_stages):
        cnt = (act & (stage == s)).sum(dim=0)[counted]
        visited = cnt > 0
        moves_at[str(s)] = {
            "mean": float(cnt.float().mean()) if m > 0 else None,
            "visited_frac": float(visited.float().mean()) if m > 0 else None,
            "mean_if_visited": float(cnt[visited].float().mean()) if bool(visited.any()) else None,
        }

    ge1 = int((counted & (max_stage >= 1)).sum())
    ge2 = int((counted & (max_stage >= 2)).sum())
    n_knot = int((counted & outcomes["stage_up"]).sum()) if "stage_up" in outcomes else 0
    ci = wilson_interval(n_knot, m)
    return {
        "knot_ci95": list(ci) if ci is not None else None,
        "p_ge2_given_ge1": _ratio(ge2, ge1),
        "p_knot_given_ge2": _ratio(n_knot, ge2),
        "outcome_by_last_stage": table,
        "stage_reach": reach,
        "moves_at_stage": moves_at,
    }


def action_saturation(
    stage: torch.Tensor,
    active: torch.Tensor,
    actions: torch.Tensor,
    channels: tuple[str, ...] | None = None,
    num_stages: int = 4,
) -> dict:
    """段 x チャネルごとに、クリップ前の行動が [−1, 1] の外に出た手の割合と、BoundedPPO の罰則の手の平均。

    `action_saturation.py` は v12 の rollout の観測 (`diag_stage_moves.py --dump_obs`) を別のチェックポイントに
    使い回すので、そのチェックポイント自身が評価で訪れる状態の分布とずれる。ここでは評価の rollout そのものの
    手を数える。罰則は `policies.bounded_ppo.bounds_penalty` と同じ Σ_チャネル max(|a| − 1, 0)²。
    決定論的な評価では行動は方策の平均なので、学習の罰則と同じ量になる (`--stochastic` ではサンプルの値)。

    数えるのは `active` の手 (各環境の最初のエピソードの手)。手の無い段はキーを出さない。
    """
    num_act = actions.shape[-1]
    if channels is None:
        channels = ACTION_CHANNELS if num_act == len(ACTION_CHANNELS) else tuple(f"ch{i}" for i in range(num_act))
    a = actions.reshape(-1, num_act).double()  # 割合を JSON に float32 の丸めの誤差なしで書くため
    st = stage.reshape(-1).long()
    keep = active.reshape(-1)

    def _stats(mask: torch.Tensor) -> dict:
        x = a[mask]
        out = x.abs() > 1.0
        return {
            "moves": int(x.shape[0]),
            "sat_frac": [float(v) for v in out.float().mean(dim=0)],
            "sat_any": float(out.any(dim=1).float().mean()),
            "penalty": float((x.abs() - 1.0).clamp(min=0.0).square().sum(dim=-1).mean()),
        }

    by_stage = {str(s): _stats(keep & (st == s)) for s in range(num_stages) if bool((keep & (st == s)).any())}
    return {
        "channels": list(channels),
        "by_stage": by_stage,
        "all": _stats(keep) if bool(keep.any()) else None,
    }
