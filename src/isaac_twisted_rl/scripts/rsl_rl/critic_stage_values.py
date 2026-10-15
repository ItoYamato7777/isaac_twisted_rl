"""チェックポイントごとに、固定した観測に対する critic の価値 V を段ごとに数える (Isaac Sim 不要)。

    uv run python src/isaac_twisted_rl/scripts/rsl_rl/critic_stage_values.py \
        --obs outputs/diag_stage_moves/obs/*.pt \
        --run v12_s44=logs/rsl_rl/knot_tying_v12/<run> --run v11_s43=<run>+<延長の run> \
        --out outputs/critic_stage_values/<名前>.json [--by_outcome]

## なぜ数えるか

段 0 の V は 1 を超えないはずで (正の報酬は段を上げたときの +1 だけ)、V が 1/γ を超えると、時間切れの
bootstrap (γ·V) が +1 を上回る。段 0 が崩れた run は、どれも V がこの値を超えて膨らんでいた
(`docs/experiments/2026-10-15_critic-inflation.md`)。学習中は `Loss/value_mean_s*`・`Loss/value_over_s*`
(`common/policies/value_stats.py`、BoundedPPO を使う v13 以降) で見る。このスクリプトは、それが無い版
(v7・v11・v12) のチェックポイントを後から数える。

観測は `diag_stage_moves.py --dump_obs` が chain モードで保存したもの (actor の入力そのもの) を固定して使う。
どの run のチェックポイントにも同じ観測を入れるので、run どうし・iteration どうしを比べられるが、各 run が
実際に訪れる状態での V ではない。critic の入力が actor と同じ観測の版 (v12 まで・v13) にだけ使える。
v14 の critic は特権観測 (`rope_params`) を見るので入力の次元が違い、止める。

## 出す指標 (段ごと。`stages`)

* `n`: 観測の数、`mean`・`p95`・`max`: V の平均・95 パーセンタイル・最大。
* `over`: V が閾値 (`--gamma` の 1/γ。既定 0.95) を超えた割合。

`--by_outcome`: 観測の保存に入っている手の結果 (`up`・`same_shape` など) ごとの段 0 の V の平均も出す
(その観測を作った run の方策での結果。結果の違う状態で V が違うかを見る)。
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path

import torch
from rsl_rl.networks import MLP

# `isaac_twisted_rl.tasks` を通すと `import_packages` が全タスクを import するので、`tests/conftest.py` と同じく
# `common/` を sys.path に足して、isaaclab を import しない `policies` だけを読む。
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tasks" / "knot_tying" / "common"))
from policies.stage_gated import GatedMLP  # noqa: E402
from policies.value_stats import break_even_value  # noqa: E402

HIDDEN = [512, 256, 128]
STAGE_DIMS = 4


def load_critic(path: str, num_obs: int) -> GatedMLP:
    """チェックポイントから critic (段ごとの MLP) を作る。入力の次元が観測と違えば例外にする。"""
    sd = torch.load(path, map_location="cpu", weights_only=False)["model_state_dict"]
    heads = {k.split(".")[2] for k in sd if k.startswith("critic.heads.")}
    critic_obs = sd["critic.heads.0.0.weight"].shape[1]
    if critic_obs != num_obs:
        raise ValueError(
            f"{path}: critic の入力は {critic_obs} 次元で、固定した観測 ({num_obs} 次元) と違う。"
            "特権観測を使う版 (v14 以降) の critic は、この観測では測れない"
        )
    critic = GatedMLP([MLP(num_obs, 1, HIDDEN, "elu") for _ in range(len(heads))], STAGE_DIMS)
    critic.load_state_dict({k[len("critic.") :]: v for k, v in sd.items() if k.startswith("critic.")})
    return critic.eval()


def stage_stats(critic: GatedMLP, obs: torch.Tensor, threshold: float) -> dict[str, dict[str, float]]:
    with torch.no_grad():
        v = critic(obs).squeeze(-1).double()
    head = critic.stage_index(obs)
    out = {}
    for h in range(critic.num_heads):
        vk = v[head == h]
        if vk.numel() == 0:
            continue
        out[str(h)] = {
            "n": int(vk.numel()),
            "mean": float(vk.mean()),
            "p95": float(vk.quantile(0.95)),
            "max": float(vk.max()),
            "over": float((vk > threshold).double().mean()),
        }
    return out


def stage0_by_outcome(
    critic: GatedMLP, obs: torch.Tensor, before: torch.Tensor, outcome: torch.Tensor, names: tuple[str, ...]
) -> dict[str, dict[str, float]]:
    sel = before == 0
    with torch.no_grad():
        v = critic(obs[sel]).squeeze(-1).double()
    oc = outcome[sel]
    return {
        names[i]: {"n": int((oc == i).sum()), "mean": float(v[oc == i].mean())}
        for i in range(len(names))
        if bool((oc == i).any())
    }


def checkpoints(spec: str) -> list[tuple[int, str]]:
    """`<run>[+<run>...]` の `model_N.pt` を N の順に (後ろの run を優先)。"""
    found: dict[int, str] = {}
    for run in spec.split("+"):
        for path in glob.glob(os.path.join(run, "model_*.pt")):
            found[int(re.search(r"model_(\d+)\.pt$", path).group(1))] = path
    return sorted(found.items())


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-stage critic values on fixed observations per checkpoint.")
    parser.add_argument("--obs", nargs="+", required=True, help="diag_stage_moves.py --dump_obs files.")
    parser.add_argument("--run", action="append", required=True, help="name=<run dir>[+<run dir>...]")
    parser.add_argument("--every", type=int, default=10, help="Use checkpoints whose iteration is a multiple of this.")
    parser.add_argument("--gamma", type=float, default=0.95, help="Threshold is 1/gamma (PPO gamma of the run).")
    parser.add_argument("--by_outcome", action="store_true", help="Also group stage-0 V by the saved move outcome.")
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    dumps = [torch.load(p, map_location="cpu", weights_only=False) for p in args.obs]
    obs = torch.cat([d["actor_obs"] for d in dumps])
    threshold = break_even_value(args.gamma)
    result: dict = {"obs_files": args.obs, "num_obs": int(obs.shape[0]), "threshold": threshold, "runs": {}}
    if args.by_outcome:
        names = dumps[0]["outcomes"]
        if any(d["outcomes"] != names for d in dumps):
            raise ValueError("観測の保存の `outcomes` の並びが揃っていない")
        before = torch.cat([d["before"] for d in dumps])
        outcome = torch.cat([d["outcome"] for d in dumps])
        result["obs_outcome_counts"] = {n: int(((before == 0) & (outcome == i)).sum()) for i, n in enumerate(names)}
    for spec in args.run:
        name, dirs = spec.split("=", 1)
        rows = {}
        for it, path in checkpoints(dirs):
            if it % args.every != 0:
                continue
            critic = load_critic(path, obs.shape[-1])
            rows[it] = {"stages": stage_stats(critic, obs, threshold)}
            if args.by_outcome:
                rows[it]["stage0_by_outcome"] = stage0_by_outcome(critic, obs, before, outcome, names)
        result["runs"][name] = rows
        print(f"[INFO] {name}: {len(rows)} checkpoints")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)


if __name__ == "__main__":
    main()
