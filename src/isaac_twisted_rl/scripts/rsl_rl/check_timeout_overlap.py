# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""学習用のタスクで方策を走らせ、同じ手で `terminated` と `truncated` が両方立った数を数える (v15 のゲート)。

両方が立った手は、rsl-rl 3.0.1 の PPO が `terminated` を見ずに γ·V(s) を bootstrap する
(`docs/decisions/0028-v15-timeout-bootstrap-fix.md`)。v14 では 6 手目に段が変わる・発散する手で 0 より大きく、
v15 (`mdp.time_out_unless_terminated`) では 0 になるはず。学習と同じく方策の std で行動を引く
(`--deterministic` で平均)。起動方法と引数は `diag_stage_moves.py` と同じで、タスクは **Play でない** 方::

    uv run python -u src/isaac_twisted_rl/scripts/rsl_rl/check_timeout_overlap.py --task Isaac-Twisted-RL-Knot-v15 \
        --num_envs 256 --headless --device cuda:1 --seed 7 \
        --checkpoint logs/rsl_rl/knot_tying_v12/<run>/model_150.pt \
        agent.device=cuda:1 agent.obs_groups.critic=[policy] --out <json>

`agent.obs_groups.critic=[policy]` は、critic が actor と同じ観測 (197 次元) のチェックポイント (v13 まで) を
v14 以降のタスクで読むため。数えるのは終了だけで critic の値は使わない。

数える値 (`env.step` の後の `termination_manager.terminated` / `time_outs`):

* `terminated` / `truncated` / `both`: 手の数 (env × 手)。
* `both_by_term`: `both` の手を、その手で立った `time_out` 以外の項 (`termination_manager.get_term`。この手で
  終わった env では、この手で立った項のうち cfg の並びの最後の 1 つ) ごとに。
* `truncated_only`: 純粋な時間切れ (段も変わらず発散もしない最後の手) の数。v15 でも残るはず。
"""

import argparse
import sys

from isaaclab.app import AppLauncher

from isaac_twisted_rl.scripts.rsl_rl import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Count steps where terminated and truncated are both set.")
parser.add_argument("--num_envs", type=int, default=256, help="Number of environments.")
parser.add_argument("--task", type=str, default=None, help="Name of the training task (not -Play-).")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--steps", type=int, default=24, help="env.step() calls (4 episodes of 6 moves by default).")
parser.add_argument("--deterministic", action="store_true", default=False, help="Use the policy mean.")
parser.add_argument("--out", type=str, default=None, help="Write the counts as JSON here.")
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
from isaaclab_tasks.utils.hydra import hydra_task_config


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    resume_path = retrieve_file_path(args_cli.checkpoint)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    runner.load(resume_path)
    alg_policy = runner.alg.policy
    alg_policy.eval()
    if args_cli.deterministic:
        policy = runner.get_inference_policy(device=env.unwrapped.device)
    else:
        policy = lambda o: alg_policy.act(o)  # noqa: E731

    unwrapped = env.unwrapped
    tm = unwrapped.termination_manager
    names = [n for n in tm.active_terms if n != "time_out"]
    time_out_func = tm.get_term_cfg("time_out").func
    counts = {"terminated": 0, "truncated": 0, "both": 0, "truncated_only": 0}
    both_by_term = dict.fromkeys(names, 0)
    per_step = []

    obs = env.get_observations()
    with torch.inference_mode():
        for step in range(args_cli.steps):
            obs, _, _, extras = env.step(policy(obs))
            terminated = tm.terminated.clone()
            truncated = tm.time_outs.clone()
            assert torch.equal(truncated, extras["time_outs"].bool()), "rsl-rl に渡る time_outs が truncated と違う"
            both = terminated & truncated
            counts["terminated"] += int(terminated.sum())
            counts["truncated"] += int(truncated.sum())
            counts["both"] += int(both.sum())
            counts["truncated_only"] += int((truncated & ~terminated).sum())
            for name in names:
                both_by_term[name] += int((both & tm.get_term(name)).sum())
            per_step.append({"step": step + 1, "both": int(both.sum()), "truncated": int(truncated.sum())})

    result = {
        "task": args_cli.task,
        "checkpoint": resume_path,
        "num_envs": unwrapped.num_envs,
        "steps": args_cli.steps,
        "deterministic": args_cli.deterministic,
        "time_out_func": getattr(time_out_func, "__name__", repr(time_out_func)),
        **counts,
        "both_by_term": both_by_term,
        "per_step": per_step,
    }
    print(json.dumps({k: v for k, v in result.items() if k != "per_step"}, indent=1))
    if args_cli.out:
        os.makedirs(os.path.dirname(os.path.abspath(args_cli.out)), exist_ok=True)
        with open(args_cli.out, "w") as f:
            json.dump(result, f, indent=1)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
