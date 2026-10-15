# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
# ruff: noqa: E402, I001
# (AppLauncher で Kit を起動してから isaaclab などを import するので、import の位置と順序に意味がある。
#  ruff.toml の per-file-ignores の eval_chain.py などと同じ扱い。既存の ruff.toml は変えずにここで外す)

"""chain モードで方策を走らせ、手ごとの観測と、それを組み立てる元の量を npz に書き出す (実機移植の検算用)。

実機の実行スクリプト (`~/work/hiro_knot/knot_policy_core.py`) は観測 197 次元を isaaclab なしで組み立て直す。
その組み立てが sim と数値で一致するかを、Isaac Sim なしのテストで確かめるための正解データを作る。
起動方法と引数は `eval_chain.py` と同じ (Play 環境、Hydra の上書きはそのまま後ろに付ける)::

    uv run python -u src/isaac_twisted_rl/scripts/rsl_rl/dump_real_deploy_check.py --task Isaac-Twisted-RL-Knot-Play-v11 \
        --num_envs 64 --headless --device cuda:0 --seed 7 \
        --checkpoint logs/rsl_rl/knot_tying_v11/<run>/model_300.pt agent.device=cuda:0 --out <npz>

手ごと (各 `env.step` の前) に書くもの (先頭の次元は `[手, 環境]`):

* `obs` (197): 方策の観測。`link_pos_w` (28, 3): ロープのリンク中心 (world、名前で解決した紐順)。
  `env_origins` (3)。`hand_root_w` (3) / `hand_yaw`: ハンドの root (sim の実測)。`last_action` (6): 観測の `actions` 項の元
* `mean_action` (6): 決定論の平均の行動。`applied_action` (6): 実際に打った行動 (前半の環境は平均、後半は確率的に引いた行動。
  形のばらつきを増やして交差の多い形も含めるため)
* `next_edge_key` / `next_edge_cand`: `NextEdgeGoal` の内部状態 (観測を計算した直後)
* 手の後: `done`、`PickPlaceAction` が確定した量 (`grasp_id` / `target_id` / `grasp_xy` / `grasp_yaw` / `place_pos` /
  `clear_pos` / `end_yaw`、world)。reset で消えない内部の値を読む

さらに先頭 `--traj_envs` 個の環境について、物理ステップごとのハンドの指令 (`_write_hand` の位置・yaw と `_set_fingers` の開度) を
`traj_pos` (手, 物理ステップ, 環境, 3) / `traj_yaw` / `traj_ratio` に書く。実機側の軌道の生成と比べるため。
"""

import argparse
import sys

from isaaclab.app import AppLauncher

from isaac_twisted_rl.scripts.rsl_rl import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Dump per-move observations for the real-robot port check.")
parser.add_argument("--num_envs", type=int, default=64, help="Number of environments.")
parser.add_argument("--task", type=str, default=None, help="Name of the task (use the -Play- variant).")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--steps", type=int, default=18, help="env.step() calls.")
parser.add_argument(
    "--traj_envs", type=int, default=4, help="Record per-physics-step hand commands of the first N envs."
)
parser.add_argument("--out", type=str, required=True, help="Write the npz here.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os

import gymnasium as gym
import numpy as np
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
import isaac_twisted_rl.tasks  # noqa: F401
from isaac_twisted_rl.tasks.knot_tying.common.mdp.observations import get_next_edge_goal
from isaaclab_tasks.utils.hydra import hydra_task_config


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
    alg_policy = runner.alg.policy
    alg_policy.eval()

    unwrapped = env.unwrapped
    term = unwrapped.action_manager.get_term("pick_place")
    gripper = unwrapped.scene["gripper"]
    n = unwrapped.num_envs
    dev = unwrapped.device
    stochastic = torch.arange(n, device=dev) >= n // 2
    k = min(args_cli.traj_envs, n)

    # -- 物理ステップごとのハンドの指令を先頭 k 環境ぶん控える (実装は変えず、呼ばれた値を写すだけ)
    traj: dict[str, list] = {"pos": [], "yaw": [], "ratio": []}
    write_hand, set_fingers = term._write_hand, term._set_fingers

    def _write_hand(pos, yaw, vel, yaw_rate):
        traj["pos"].append(pos[:k].detach().cpu().numpy().copy())
        traj["yaw"].append(yaw[:k].detach().cpu().numpy().copy())
        write_hand(pos, yaw, vel, yaw_rate)

    def _set_fingers(ratio):
        traj["ratio"].append(ratio[:k].detach().cpu().numpy().copy())
        set_fingers(ratio)

    term._write_hand, term._set_fingers = _write_hand, _set_fingers

    rows: dict[str, list] = {}

    def put(name, value):
        rows.setdefault(name, []).append(value.detach().cpu().numpy().copy())

    traj_rows: dict[str, list] = {"traj_pos": [], "traj_yaw": [], "traj_ratio": []}
    obs = env.get_observations()
    with torch.inference_mode():
        for _ in range(args_cli.steps):
            pobs = obs["policy"]
            goal = get_next_edge_goal(unwrapped)
            put("obs", pobs)
            put("link_pos_w", term._rope_link_positions())
            put("env_origins", unwrapped.scene.env_origins)
            put("hand_root_w", gripper.data.root_pos_w)
            put("hand_yaw", 2.0 * torch.atan2(gripper.data.root_quat_w[:, 3], gripper.data.root_quat_w[:, 0]))
            put("last_action", unwrapped.action_manager.action)
            put("next_edge_key", goal.key)
            put("next_edge_cand", goal.cand)
            mean = alg_policy.act_inference(obs)
            sample = alg_policy.act(obs)
            actions = torch.where(stochastic.unsqueeze(-1), sample, mean)
            put("mean_action", mean)
            put("applied_action", actions)
            for key in traj:
                traj[key].clear()
            obs, _, dones, _ = env.step(actions)
            put("done", dones.bool())
            put("grasp_id", term._grasp_id)
            put("target_id", term._target_id)
            put("grasp_xy", term._grasp_xy)
            put("grasp_yaw", term._grasp_yaw)
            put("place_pos", term._place_pos)
            put("clear_pos", term._clear_pos)
            put("end_yaw", term._end_yaw)
            traj_rows["traj_pos"].append(np.stack(traj["pos"]))
            traj_rows["traj_yaw"].append(np.stack(traj["yaw"]))
            traj_rows["traj_ratio"].append(np.stack(traj["ratio"]))

    out = {name: np.stack(v) for name, v in rows.items()}
    out.update({name: np.stack(v) for name, v in traj_rows.items()})
    out["stochastic"] = stochastic.cpu().numpy()
    out["physics_dt"] = np.array(unwrapped.sim.get_physics_dt())
    out["phase_steps"] = term._phase_steps.cpu().numpy()
    out["checkpoint"] = np.array(resume_path)
    os.makedirs(os.path.dirname(os.path.abspath(args_cli.out)), exist_ok=True)
    np.savez_compressed(args_cli.out, **out)
    print(f"[INFO] wrote {args_cli.out}: " + ", ".join(f"{k}{v.shape}" for k, v in out.items()))
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
