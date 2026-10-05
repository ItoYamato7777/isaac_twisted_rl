# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""v8 の対称性のデータ拡張が前提にしていることを、Play 環境で確かめる (学習の前に 1 回)。

    uv run python -u src/isaac_twisted_rl/scripts/rsl_rl/symmetry_check.py --task Isaac-Twisted-RL-Knot-Play-v8 \
        --num_envs 512 --moves 6 --headless --seed 7 \
        --checkpoint logs/rsl_rl/knot_tying_v7/<run>/model_100.pt --out symmetry_check.json

## 確かめること

1. **観測の照合**: 環境の観測の `rope_link_pos` から、データ拡張と同じ関数で位相の項 (`crossings`・`writhe`・
   `crossing_feats`・`knot_class`) を計算し直し、環境が出した値と比べる (`common/topology/symmetry.py`
   は環境の観測の項の計算を写しているので、ずれていないかを見る)。
2. **物理の対称性**: 環境を 4 つずつの組にし、全部まっすぐなロープ (env 原点を通る x 軸上。自分自身の鏡像) から
   始める。組の先頭に手 a、2 番目に同じ手 a、3 番目に鏡像の手 M a、4 番目に逆順の手 R a を打ち、それを
   `--moves` 手続ける。手の後に、先頭と比べた段の一致率と、ロープの形のずれ (変換を戻したリンク位置の
   平均距離) を出す。**同じ手どうし (2 番目) のずれが物理の非決定性の基準** で、鏡像・逆順のずれがこれと同程度
   なら、データ拡張の変換は物理と合っている。グリッパーの形が左右・前後で非対称なら、鏡像・逆順のほうが
   大きくずれる。

`--checkpoint` を渡すと先頭の手を方策 (決定論的) で選び、渡さなければ一様乱数の手にする。組のどれかが
終わった (固結び・発散・時間切れ) 手からは、その組を比較から外す (reset 後の観測は比べられないため)。
終わった手の終了の種類が先頭と一致したかは別に数える。
"""

import argparse
import sys

from isaaclab.app import AppLauncher

from isaac_twisted_rl.scripts.rsl_rl import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Check the symmetry assumptions of the v8 data augmentation.")
parser.add_argument("--num_envs", type=int, default=512, help="Number of environments (multiple of 4).")
parser.add_argument("--task", type=str, default=None, help="Name of the task (use the -Play- variant).")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--moves", type=int, default=6, help="Moves per group.")
parser.add_argument("--out", type=str, default=None, help="Write the summary as JSON here.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import json
import os

import gymnasium as gym
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
import isaac_twisted_rl.tasks  # noqa: F401
from isaac_twisted_rl.tasks.knot_tying.common.mdp.symmetry import policy_obs_layout
from isaac_twisted_rl.tasks.knot_tying.common.topology import symmetry as sym
from isaaclab_tasks.utils.hydra import hydra_task_config

_STAGE_DIMS = 4
"""観測の末尾にある段の one-hot の次元 (段 0〜3)。"""

_ROLES = ("same", "mirror", "reverse")
"""組の 2〜4 番目に打つ手 (先頭の手と同じ・鏡像・逆順)。"""

_TOL = 1e-4
"""観測の照合で一致とみなす最大の差。"""


def _split(pobs: torch.Tensor, layout) -> dict[str, torch.Tensor]:
    return dict(zip([n for n, _, _ in layout], pobs.split([d for _, d, _ in layout], dim=-1)))


def _quantiles(x: torch.Tensor) -> dict[str, float | None]:
    if x.numel() == 0:
        return {"median": None, "p90": None, "n": 0}
    x = x.float().cpu()
    return {"median": float(x.median()), "p90": float(x.quantile(0.9)), "n": int(x.numel())}


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    if args_cli.num_envs % 4:
        raise ValueError("--num_envs は 4 の倍数にすること")
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if args_cli.checkpoint:
        env_cfg.log_dir = os.path.dirname(retrieve_file_path(args_cli.checkpoint))

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    unwrapped = env.unwrapped
    dev = unwrapped.device
    n = unwrapped.num_envs
    groups = n // 4
    leader = torch.arange(groups, device=dev) * 4

    policy = None
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        runner.load(resume_path)
        policy = runner.get_inference_policy(device=dev)

    layout = policy_obs_layout(env, "policy")
    tm = unwrapped.termination_manager
    term_names = list(tm.active_terms)
    gen = torch.Generator(device=dev)
    gen.manual_seed(int(agent_cfg.seed or 0))

    obs = env.get_observations()
    pobs = obs["policy"]
    link0 = _split(pobs, layout)["rope_link_pos"].reshape(n, -1, 3)
    init = {
        "max_abs_y": float(link0[..., 1].abs().max()),
        "max_spread_between_envs": float((link0 - link0[:1]).abs().max()),
    }

    obs_check = {name: [0, 0] for name, _, _ in layout if name in sym._RECOMPUTED_TERMS}
    alive = torch.ones(groups, dtype=torch.bool, device=dev)
    per_move = []

    def check_obs(pobs: torch.Tensor) -> None:
        parts = _split(pobs, layout)
        recomputed = sym._recompute_terms(parts["rope_link_pos"].reshape(n, -1, 3), layout)
        for name, value in recomputed.items():
            bad = (value.to(parts[name].dtype) - parts[name]).abs().amax(dim=-1) > _TOL
            obs_check[name][0] += int(bad.sum())
            obs_check[name][1] += n

    with torch.inference_mode():
        check_obs(pobs)
        for move in range(args_cli.moves):
            if policy is not None:
                act = policy(obs)
            else:
                act = torch.rand(n, unwrapped.action_manager.total_action_dim, device=dev, generator=gen) * 2.0 - 1.0
            a = act[leader]
            actions = torch.empty_like(act)
            actions[leader] = a
            actions[leader + 1] = a
            actions[leader + 2] = sym.transform_actions(a, True, False)
            actions[leader + 3] = sym.transform_actions(a, False, True)

            prev_stage = obs["policy"][:, -_STAGE_DIMS:].argmax(dim=-1)
            obs, _, dones, _ = env.step(actions)
            pobs = obs["policy"]
            check_obs(pobs)
            done = dones.bool().reshape(groups, 4)
            fired = {k: tm.get_term(k).reshape(groups, 4) & done for k in term_names}

            row = {"move": move + 1, "groups_alive_before": int(alive.sum())}
            # 終わった組: 終了の種類が先頭と一致したか
            ended = alive & done.any(dim=1)
            for j, role in enumerate(_ROLES, start=1):
                same_end = torch.ones(groups, dtype=torch.bool, device=dev)
                for k in term_names:
                    same_end &= fired[k][:, 0] == fired[k][:, j]
                same_end &= done[:, 0] == done[:, j]
                row[f"{role}_same_end"] = float(same_end[ended].float().mean()) if bool(ended.any()) else None
            row["groups_ended"] = int(ended.sum())

            # 続いている組: 段とロープの形
            alive = alive & ~done.any(dim=1)
            row["groups_compared"] = int(alive.sum())
            parts = _split(pobs, layout)
            link = parts["rope_link_pos"].reshape(groups, 4, -1, 3)
            stage = pobs[:, -_STAGE_DIMS:].argmax(dim=-1).reshape(groups, 4)
            ref = link[:, 0]
            back = {
                "same": link[:, 1],
                "mirror": sym.transform_link_pos(link[:, 2], True, False),
                "reverse": sym.transform_link_pos(link[:, 3], False, True),
            }
            for j, role in enumerate(_ROLES, start=1):
                dist = (back[role] - ref).norm(dim=-1).mean(dim=-1)  # (G,) リンクの平均距離 [m]
                row[f"{role}_dist_m"] = _quantiles(dist[alive])
                row[f"{role}_stage_agree"] = (
                    float((stage[:, j] == stage[:, 0])[alive].float().mean()) if bool(alive.any()) else None
                )
            moved = (stage[:, 0] != prev_stage[leader]) & alive
            row["leader_stage_changed"] = int(moved.sum())
            per_move.append(row)
            if not bool(alive.any()):
                break

    summary = {
        "task": args_cli.task,
        "checkpoint": args_cli.checkpoint,
        "num_envs": n,
        "groups": groups,
        "initial_state": init,
        "obs_mismatch": {k: {"bad": v[0], "rows": v[1], "rate": v[0] / max(v[1], 1)} for k, v in obs_check.items()},
        "per_move": per_move,
    }
    print("\n=== symmetry check ===")
    print(json.dumps(summary, indent=2))
    if args_cli.out:
        with open(args_cli.out, "w") as f:
            json.dump(summary, f, indent=2)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
