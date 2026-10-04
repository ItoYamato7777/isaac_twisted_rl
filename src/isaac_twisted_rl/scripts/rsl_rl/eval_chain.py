# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""v4 以降の chain モード評価: チェックポイントを Play 環境で走らせ、エピソード単位で集計する。

以前の比較 (`eval_compare.py`。このファイルはリポジトリに残っていない) は step ごとの `extras["log"]` を平均するので、早く終わるエピソードほど
重く数えられる。ここでは **各環境の最初の 1 エピソードだけ** を数える (まっすぐなロープから、
固結び・発散・18 手のどれかで終わる)。`--steps` は Play の手数の上限 (18) 以上にすること。

    uv run python -u src/isaac_twisted_rl/scripts/rsl_rl/eval_chain.py --task Isaac-Twisted-RL-Knot-Play-v4 \
        --num_envs 1024 --headless --checkpoint logs/rsl_rl/knot_tying_v4/<run>/model_300.pt --out result.json

出力 (JSON と標準出力):
* `knot`: 固結びに達した割合 (`stage_up`。Play では固結びでだけ立つ)
* `diverged` / `time_out`: 発散 / 18 手で終わった割合
* `max_stage_ge{k}`: エピソード中に段 k 以上に達した割合 (観測の段の one-hot から)
* `mean_actions_to_knot`: 固結びに達したエピソードの手数の平均
"""

import argparse
import sys

from isaaclab.app import AppLauncher

import isaac_twisted_rl.scripts.rsl_rl.cli_args as cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Evaluate a checkpoint per episode in chain mode.")
parser.add_argument("--num_envs", type=int, default=1024, help="Number of environments (= episodes counted).")
parser.add_argument("--task", type=str, default=None, help="Name of the task (use the -Play- variant).")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--steps", type=int, default=19, help="env.step() calls (>= the Play episode length).")
parser.add_argument("--stochastic", action="store_true", default=False, help="Sample actions with the policy std.")
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
from isaaclab_tasks.utils.hydra import hydra_task_config

_STAGE_DIMS = 4
"""観測の末尾にある段の one-hot の次元 (段 0〜3)。"""


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    resume_path = retrieve_file_path(args_cli.checkpoint)
    env_cfg.log_dir = os.path.dirname(resume_path)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    runner.load(resume_path)
    if args_cli.stochastic:
        alg_policy = runner.alg.policy
        alg_policy.eval()
        policy = lambda o: alg_policy.act(o)  # noqa: E731
    else:
        policy = runner.get_inference_policy(device=env.unwrapped.device)

    unwrapped = env.unwrapped
    tm = unwrapped.termination_manager
    n = unwrapped.num_envs
    dev = unwrapped.device
    names = list(tm.active_terms)

    counted = torch.zeros(n, dtype=torch.bool, device=dev)
    outcome = {k: torch.zeros(n, dtype=torch.bool, device=dev) for k in names}
    max_stage = torch.zeros(n, dtype=torch.long, device=dev)
    ep_max_stage = torch.zeros(n, dtype=torch.long, device=dev)
    length = torch.zeros(n, dtype=torch.long, device=dev)
    ep_len = torch.zeros(n, dtype=torch.long, device=dev)

    obs = env.get_observations()
    with torch.inference_mode():
        for _ in range(args_cli.steps):
            pobs = obs["policy"] if not torch.is_tensor(obs) else obs
            stage = pobs[:, -_STAGE_DIMS:].argmax(dim=-1)
            ep_max_stage = torch.maximum(ep_max_stage, stage)
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            ep_len += 1
            done = dones.bool() & ~counted
            for k in names:
                outcome[k] |= done & tm.get_term(k)
            # 固結び (stage_up) で終わった手は観測が次のエピソードに替わっているので 3 を入れる。
            final_stage = torch.where(tm.get_term("stage_up"), torch.full_like(ep_max_stage, 3), ep_max_stage)
            max_stage = torch.where(done, final_stage, max_stage)
            length = torch.where(done, ep_len, length)
            counted |= done
            ep_max_stage = torch.where(dones.bool(), torch.zeros_like(ep_max_stage), ep_max_stage)
            ep_len = torch.where(dones.bool(), torch.zeros_like(ep_len), ep_len)
            if bool(counted.all()):
                break

    m = int(counted.sum())
    knot = outcome["stage_up"] & counted
    summary = {
        "task": args_cli.task,
        "checkpoint": resume_path,
        "stochastic": args_cli.stochastic,
        "episodes": m,
        "uncounted": n - m,
        "knot": float(knot.sum()) / max(m, 1),
        **{k: float((outcome[k] & counted).sum()) / max(m, 1) for k in names if k != "stage_up"},
        **{f"max_stage_ge{s}": float(((max_stage >= s) & counted).sum()) / max(m, 1) for s in (1, 2, 3)},
        "mean_actions_to_knot": float(length[knot].float().mean()) if bool(knot.any()) else None,
    }
    print("\n=== chain evaluation ===")
    for k, v in summary.items():
        print(f"{k}: {v}")
    if args_cli.out:
        with open(args_cli.out, "w") as f:
            json.dump(summary, f, indent=2)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
