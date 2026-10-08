# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""chain モードで方策を走らせ、手の前の段ごとに「どんな手を打ち、何が起きたか」を集計する。

chain 評価 (`eval_chain.py`) はエピソードの結末しか出さない。方策が段 0 で時間切れになるとき
(v11 s44 の iteration 150 以降)、形を変えていないのか・変えたが同じ段に戻るのか・ロープが床に
落ちていない (段の判定が数えない) のかは、手ごとの記録が無いと分からないので、このスクリプトで数える。
起動方法と引数は `eval_chain.py` と同じ (Play 環境、Hydra の上書きはそのまま後ろに付ける)::

    uv run python -u src/isaac_twisted_rl/scripts/rsl_rl/diag_stage_moves.py --task Isaac-Twisted-RL-Knot-Play-v11 \
        --num_envs 512 --headless --device cuda:0 --seed 7 \
        --checkpoint logs/rsl_rl/knot_tying_v11/<run>/model_200.pt agent.device=cuda:0 --out <json>

各環境の最初の 1 エピソードの手だけを数える (`eval_chain.py` と同じ)。手の結果の分類 (`outcome`):

* `up` / `down`: 段が上がった / 下がった (段の定義は版のもの。v11 の行き止まりは −1 なので `down`)
* `same_shape`: 段も p-data も変わらなかった (形を変えていない手)
* `same_stage`: 段は同じで p-data は変わった
* `not_settled`: 手の後のロープの最高点が `settled_height_diameters` 直径より高い (学習では up / down に数えない)
* `diverged`: 発散で終わった

段と p-data は `stages.last_stage_transition` (終了判定が計算した reset の前の値) から読む。`not_settled` の
判定に使うロープの最高点は手の後の観測の時点のもので、reset された環境 (固結び・発散) では使わない。
"""

import argparse
import sys

from isaaclab.app import AppLauncher

from isaac_twisted_rl.scripts.rsl_rl import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Per-move diagnostics of a stage-gated policy in chain mode.")
parser.add_argument("--num_envs", type=int, default=512, help="Number of environments (= episodes counted).")
parser.add_argument("--task", type=str, default=None, help="Name of the task (use the -Play- variant).")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--steps", type=int, default=19, help="env.step() calls (>= the Play episode length).")
parser.add_argument("--stochastic", action="store_true", default=False, help="Sample actions with the policy std.")
parser.add_argument("--top", type=int, default=8, help="Number of most frequent p-data after a move to list.")
parser.add_argument("--out", type=str, default=None, help="Write the summary as JSON here.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import collections
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
from isaac_twisted_rl.tasks.knot_tying.common.mdp.observations import rope_zmax
from isaac_twisted_rl.tasks.knot_tying.common.mdp.stages import last_stage_transition
from isaac_twisted_rl.tasks.knot_tying.common.topology import knot_graph as kg
from isaaclab_tasks.utils.hydra import hydra_task_config

OUTCOMES = ("up", "down", "same_shape", "same_stage", "not_settled", "diverged")


def _key_names() -> dict[int, str]:
    """BFS のグラフに現れる p-data のキー -> short p-data。表の外 (交差 5 個以上) は引けない。"""
    states, _ = kg._enumerate_moves(kg.MAX_GRAPH_CROSSINGS)
    return {kg.p_data_key(st.short_p_data): st.short_p_data for st in states}


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.actions.pick_place.diag = True
    settle = env_cfg.terminations.stage_up.params
    settled_z = settle["settled_height_diameters"] * settle["rope_diameter"]

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
    term = unwrapped.action_manager.get_term("pick_place")
    tm = unwrapped.termination_manager
    n = unwrapped.num_envs
    dev = unwrapped.device
    counted = torch.zeros(n, dtype=torch.bool, device=dev)
    rows: list[dict[str, torch.Tensor]] = []

    obs = env.get_observations()
    with torch.inference_mode():
        for _ in range(args_cli.steps):
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            done = dones.bool()
            tr = last_stage_transition(unwrapped)
            if tr is None:
                raise RuntimeError("stage_transition のキャッシュが無い (段の終了項 stage_up が配線されていない?)")
            diag = term.diagnostics
            diverged = done & tm.get_term("diverged")
            not_settled = ~done & (rope_zmax(unwrapped) > settled_z)
            outcome = torch.full((n,), OUTCOMES.index("same_stage"), dtype=torch.long, device=dev)
            outcome = torch.where(tr["key_after"] == tr["key_before"], OUTCOMES.index("same_shape"), outcome)
            outcome = torch.where(tr["after"] < tr["before"], OUTCOMES.index("down"), outcome)
            outcome = torch.where(tr["after"] > tr["before"], OUTCOMES.index("up"), outcome)
            outcome = torch.where(not_settled, OUTCOMES.index("not_settled"), outcome)
            outcome = torch.where(diverged, OUTCOMES.index("diverged"), outcome)
            act = actions.clamp(-1.0, 1.0)
            proc = term.processed_actions
            rows.append(
                {
                    "keep": ~counted,
                    "before": tr["before"],
                    "after": tr["after"],
                    "key_after": tr["key_after"],
                    "outcome": outcome,
                    "act": act,
                    "delta": proc[:, 1] - proc[:, 0],
                    "grasp": proc[:, 0],
                    "carry": diag["carry_dist"],
                    "held": diag["held"].float(),
                    "release_z": diag["release_tip_z"],
                }
            )
            counted |= done
            if bool(counted.all()):
                break

    cat = {k: torch.cat([r[k] for r in rows]) for k in rows[0] if k != "keep"}
    keep = torch.cat([r["keep"] for r in rows])
    cat = {k: v[keep].cpu() for k, v in cat.items()}
    names = _key_names()
    summary: dict = {"task": args_cli.task, "checkpoint": resume_path, "stochastic": args_cli.stochastic}
    summary["episodes"] = int(counted.sum())
    summary["by_stage_before"] = {}
    for s in sorted(set(cat["before"].tolist())):
        sel = cat["before"] == s
        m = int(sel.sum())
        oc = cat["outcome"][sel]
        entry = {"moves": m, "outcome": {o: float((oc == i).sum()) / m for i, o in enumerate(OUTCOMES)}}
        per = {}
        for i, o in enumerate(OUTCOMES):
            msk = sel & (cat["outcome"] == i)
            if not bool(msk.any()):
                continue
            per[o] = {
                "act_mean": [round(float(x), 3) for x in cat["act"][msk].mean(0)],
                "act_std": [round(float(x), 3) for x in cat["act"][msk].std(0, unbiased=False)],
                "abs_delta_mean": float(cat["delta"][msk].abs().float().mean()),
                "grasp_mean": float(cat["grasp"][msk].float().mean()),
                "carry_mean": float(cat["carry"][msk].mean()),
                "held": float(cat["held"][msk].mean()),
                "release_z_mean": float(cat["release_z"][msk].mean()),
            }
        entry["by_outcome"] = per
        after = collections.Counter()
        for k, a in zip(cat["key_after"][sel].tolist(), cat["after"][sel].tolist()):
            after[(names.get(k, f"key {k}"), a)] += 1
        entry["top_after"] = [[pd, st, c] for (pd, st), c in after.most_common(args_cli.top)]
        summary["by_stage_before"][str(s)] = entry

    print("\n=== per-move diagnostics ===")
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    if args_cli.out:
        os.makedirs(os.path.dirname(os.path.abspath(args_cli.out)), exist_ok=True)
        with open(args_cli.out, "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
