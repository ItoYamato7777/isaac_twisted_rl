"""ロープの物理の DR (`mdp.RandomizeRopePhysics`、v14) が書いた値を PhysX から読み戻し、引いた倍率と照らす。

学習の前のゲート (`docs/decisions/0027-v14-s0-reshape-dr-priv-critic.md`)。Isaac Sim が要るので単体テストでは見られない
部分 (書き込みの経路・DOF の振り分け・形状の列・観測) を確かめる::

    src/isaac_twisted_rl/scripts/run_job.sh --name rope_dr_check --timeout 30m --detach -- \\
        uv run python src/isaac_twisted_rl/tasks/knot_tying/demos/rope_dr_check.py --headless --device cuda:0

確かめること (reset を 2 回して、2 回とも):

* 曲げ・捩りのばね定数と減衰、リンクの質量と慣性、ロープと指の摩擦が、起動時の公称値 × 引いた倍率と一致する。
  減衰は `domain_rand.damping_scale`。読み戻しは PhysX の tensor API (`get_dof_stiffnesses` など)。
* 観測のグループ `rope_params` が `domain_rand.normalize(倍率)` と一致し、`policy` の後に段の one-hot がある。
* 1 手打って、発散 (NaN) せずに回る。
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("--task", type=str, default="Isaac-Twisted-RL-Knot-v14")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------- 本体
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import isaac_twisted_rl.tasks  # noqa: E402, F401
from isaac_twisted_rl.tasks.knot_tying.common import domain_rand as dr  # noqa: E402


def _rel_err(got: torch.Tensor, want: torch.Tensor) -> float:
    return float(((got - want).abs() / want.abs().clamp_min(1e-12)).max())


def _check(u, term) -> bool:
    rope, grip = u.scene["rope"], u.scene["gripper"]
    view = rope.root_physx_view
    s = term.scales.cpu()
    twist = term._twist.cpu()
    k_scale = torch.where(twist[None, :], s[:, 1:2], s[:, 0:1])
    c_scale = dr.damping_scale(k_scale, s[:, 3:4], s[:, 2:3])
    errs = {
        "stiffness": _rel_err(view.get_dof_stiffnesses(), term._k0.cpu() * k_scale),
        "damping": _rel_err(view.get_dof_dampings(), term._c0.cpu() * c_scale),
        "mass": _rel_err(view.get_masses(), term._mass0 * s[:, 3:4]),
        "inertia": _rel_err(view.get_inertias()[..., [0, 4, 8]], term._inertia0[..., [0, 4, 8]] * s[:, 3:4, None]),
        "rope_mu": _rel_err(view.get_material_properties()[..., 0:2], term._rope_mat0[..., 0:2] * s[:, 4, None, None]),
        "finger_mu": _rel_err(
            grip.root_physx_view.get_material_properties()[:, term._finger_shapes, 0:2],
            term._grip_mat0[:, term._finger_shapes, 0:2] * s[:, 5, None, None],
        ),
    }
    obs = u.observation_manager.compute()
    errs["obs_rope_params"] = float(
        (obs["rope_params"].cpu() - dr.normalize(term.scales, term.ranges).cpu()).abs().max()
    )
    print(f"[check] 倍率の範囲 min {s.min(0).values.tolist()} max {s.max(0).values.tolist()}")
    print(f"[check] 観測の幅 policy {obs['policy'].shape[-1]}、rope_params {obs['rope_params'].shape[-1]}")
    ok = True
    for name, err in errs.items():
        good = err < 1e-4
        ok &= good
        print(f"[check] {name:16s} 最大の相対誤差 {err:.2e} {'OK' if good else 'NG'}")
    return ok


def main() -> None:
    cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    cfg.seed = args_cli.seed
    env = gym.make(args_cli.task, cfg=cfg)
    u = env.unwrapped
    term = u.event_manager.get_term_cfg("rope_dr").func
    print(f"[check] 捩りの DOF: {[n for n, t in zip(u.scene['rope'].joint_names, term._twist.tolist()) if t][:3]} ...")
    results = []
    with torch.inference_mode():
        for i in range(2):
            env.reset()
            print(f"[check] reset {i}")
            results.append(_check(u, term))
        actions = torch.rand(u.num_envs, 6, device=u.device) * 2 - 1
        env.step(actions)
        nan = ~torch.isfinite(u.scene["rope"].data.body_pos_w).all(dim=-1).all(dim=-1)
        print(f"[check] 1 手の後に NaN の env: {int(nan.sum())} / {u.num_envs}")
    print(f"[final] {'OK' if all(results) else 'NG'}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
