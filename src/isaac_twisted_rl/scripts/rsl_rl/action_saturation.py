"""段ごとの方策の平均が行動の範囲 [−1, 1] の外へ出ているかを、チェックポイントごとに数える (Isaac Sim 不要)。

    uv run python src/isaac_twisted_rl/scripts/rsl_rl/action_saturation.py \
        --obs outputs/diag_stage_moves/obs/*.pt \
        --run v12_s44=logs/rsl_rl/knot_tying_v12/<run> --run v7_s43=<run>+<延長の run> --out <json>

## なぜ数えるか

`StageGatedActorCritic` の actor の出力は線形で、環境 (`PickPlaceAction.process_actions`) が [−1, 1] にクリップする。
PPO の確率 (log prob) はクリップ前の行動で計算するので、平均が範囲の外へ出ると、標本の多くが同じ端の行動になり、
その次元の行動の違いが報酬に現れなくなる。その次元の勾配は雑音だけになり、平均を範囲の中へ戻す力が無い。

観測は `diag_stage_moves.py --dump_obs` が chain モードで保存したもの (actor の入力そのもの) を固定して使い、
同じ観測に対する各チェックポイントの平均と std を比べる。観測の段 (末尾の one-hot) で使うヘッドが決まる。

## 出す指標 (段ごと)

* `sat`: (観測, 次元) のうち |平均| > 1 の割合。
* `inside`: 標本がクリップされない確率 P(|a| < 1) (a ~ N(平均, std)) の平均。小さいほど行動が端に張り付く。
* `excess`: max(|平均| − 1, 0) の平均。
* `std`: その段の std の平均と最小。
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
# `common/` を sys.path に足して、isaaclab を import しない `policies.stage_gated` だけを読む。
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tasks" / "knot_tying" / "common"))
from policies.stage_gated import GatedMLP  # noqa: E402

HIDDEN = [512, 256, 128]
STAGE_DIMS = 4


def load_actor(path: str, num_obs: int, num_actions: int) -> tuple[GatedMLP, torch.Tensor]:
    """チェックポイントから actor (段ごとの MLP) と std `(段, 行動)` を作る。"""
    sd = torch.load(path, map_location="cpu", weights_only=False)["model_state_dict"]
    num_heads = len({k.split(".")[2] for k in sd if k.startswith("actor.heads.")})
    actor = GatedMLP([MLP(num_obs, num_actions, HIDDEN, "elu") for _ in range(num_heads)], STAGE_DIMS)
    actor.load_state_dict({k[len("actor.") :]: v for k, v in sd.items() if k.startswith("actor.")})
    std = sd["std"] if "std" in sd else torch.exp(sd["log_std"])
    return actor.eval(), std


def stats(actor: GatedMLP, std: torch.Tensor, obs: torch.Tensor) -> dict[str, dict[str, float]]:
    with torch.no_grad():
        mean = actor(obs)
    head = actor.stage_index(obs)
    out = {}
    for h in range(std.shape[0]):
        m = mean[head == h]
        if m.shape[0] == 0:
            continue
        s = std[h].abs().clamp_min(1e-6).expand_as(m)
        dist = torch.distributions.Normal(m, s)
        inside = dist.cdf(torch.ones_like(m)) - dist.cdf(-torch.ones_like(m))
        out[str(h)] = {
            "n": int(m.shape[0]),
            "sat": float((m.abs() > 1).float().mean()),
            "sat_dim": [round(float(x), 3) for x in (m.abs() > 1).float().mean(0)],
            "inside": float(inside.mean()),
            "excess": float((m.abs() - 1).clamp_min(0).mean()),
            "std_mean": float(std[h].mean()),
            "std_min": float(std[h].min()),
        }
    return out


def checkpoints(spec: str) -> list[tuple[int, str]]:
    """`<run>[+<run>...]` の `model_N.pt` を N の順に (後ろの run を優先)。"""
    found: dict[int, str] = {}
    for run in spec.split("+"):
        for path in glob.glob(os.path.join(run, "model_*.pt")):
            found[int(re.search(r"model_(\d+)\.pt$", path).group(1))] = path
    return sorted(found.items())


def main() -> None:
    parser = argparse.ArgumentParser(description="Count policy means outside the action bounds per checkpoint.")
    parser.add_argument("--obs", nargs="+", required=True, help="diag_stage_moves.py --dump_obs files.")
    parser.add_argument("--run", action="append", required=True, help="name=<run dir>[+<run dir>...]")
    parser.add_argument("--every", type=int, default=10, help="Use checkpoints whose iteration is a multiple of this.")
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    obs = torch.cat([torch.load(p, map_location="cpu", weights_only=False)["actor_obs"] for p in args.obs])
    num_obs = obs.shape[-1]
    result: dict = {"obs_files": args.obs, "num_obs": int(obs.shape[0]), "runs": {}}
    for spec in args.run:
        name, dirs = spec.split("=", 1)
        rows = {}
        for it, path in checkpoints(dirs):
            if it % args.every != 0:
                continue
            sd = torch.load(path, map_location="cpu", weights_only=False)["model_state_dict"]
            num_actions = sd["std"].shape[-1] if "std" in sd else sd["log_std"].shape[-1]
            actor, std = load_actor(path, num_obs, num_actions)
            rows[it] = stats(actor, std, obs)
        result["runs"][name] = rows
        print(f"[INFO] {name}: {len(rows)} checkpoints")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)


if __name__ == "__main__":
    main()
