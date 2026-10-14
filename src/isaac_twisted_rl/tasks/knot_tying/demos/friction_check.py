"""ロープと指の実際の摩擦を、PhysX の読み戻しと物理の試験で確かめる。2026-10-14 に作った。

USD と `rope_specs.py` はロープの摩擦を 2.0、`mdp.bind_grasp_material` は指の摩擦を 4.0 (`max` 合成) と書くが、
tensor API の `get_material_properties` はどちらも 0.5 (PhysX の既定の材質) を返す。どちらが正しいかを次の 3 つで確かめる
(結果は `docs/experiments/2026-10-14_friction-binding.md`)::

    src/isaac_twisted_rl/scripts/run_job.sh --name friction_check --timeout 30m --detach -- \\
        uv run python src/isaac_twisted_rl/tasks/knot_tying/demos/friction_check.py --headless --device cuda:0

1. **USD の bind**: ロープのリンクと指の衝突形状について、`material:binding:physics` の先と、
   `UsdShade.MaterialBindingAPI.ComputeBoundMaterial("physics")` が材質を返すかを書く。
2. **軸方向の滑り** (ロープ↔床): まっすぐなロープを床に落ち着かせ、軸の方向に 1 m/s を与えて止まるまでの距離 d を測る。
   等減速なら μ_eff = v² / (2 g d)。ロープと床は average 合成 (床は 2.0) なので、ロープの μ が分かる。
   何も書かない env と、tensor API で μ を書いた env を並べる。軸の方向に滑らせるのは、横に押すとカプセルが転がるため。
   この試験は決定論的 (同じ条件の env が 4 桁まで一致する)。
3. **指の摩擦** (ロープ↔指): 同じ手 (固定の乱数の行動) を、指の μ だけを tensor API で変えた env の組で打ち、
   1 手目の後のロープの形の組の平均が、何も書かない組からどれだけ離れるかを測る。掴んで運ぶ手は混沌的で、
   同じ条件の env でも数 cm ずれるので、何も書かない組を 2 つ置いてその距離を雑音の目安にする。
"""

from __future__ import annotations

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("--task", type=str, default="Isaac-Twisted-RL-Knot-Play-v13")
parser.add_argument("--slide_envs", type=int, default=4, help="滑りの試験の 1 条件あたりの env 数。")
parser.add_argument("--finger_envs", type=int, default=24, help="指の試験の 1 条件あたりの env 数。")
parser.add_argument("--seed", type=int, default=3)
parser.add_argument("--out", type=str, default="")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------- 本体
import gymnasium as gym  # noqa: E402
import omni.usd  # noqa: E402
import torch  # noqa: E402
from pxr import UsdPhysics, UsdShade  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import isaac_twisted_rl.tasks  # noqa: E402, F401

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 5))

SLIDE_MU = (None, 0.1, 0.5, 1.0, 2.0)
"""滑りの試験の条件。None は何も書かない (USD のまま)。"""

FINGER_MU = (None, None, 0.05, 0.5, 2.0, 4.0)
"""指の試験の条件。None を 2 つ置き、その間の距離を雑音の目安にする。"""


class Log:
    def __init__(self, path: str):
        self._f = open(path, "w")

    def __call__(self, msg: str) -> None:
        print(msg, flush=True)
        self._f.write(msg + "\n")
        self._f.flush()


def _make_env(num_envs: int):
    cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=num_envs)
    cfg.seed = args_cli.seed
    env = gym.make(args_cli.task, cfg=cfg)
    env.reset()
    return env, env.unwrapped


def _check_bindings(log: Log) -> None:
    stage = omni.usd.get_context().get_stage()
    for path in ("/World/envs/env_0/Rope/Link00/geom", "/World/envs/env_0/Gripper/left_finger/geom"):
        prim = stage.GetPrimAtPath(path)
        api = UsdShade.MaterialBindingAPI(prim)
        targets = api.GetDirectBindingRel("physics").GetTargets()
        target = stage.GetPrimAtPath(targets[0]) if targets else None
        mu = UsdPhysics.MaterialAPI(target).GetStaticFrictionAttr().Get() if target else None
        bound, _ = api.ComputeBoundMaterial("physics")
        log(
            f"[bind] {path}: 先 {[str(t) for t in targets]} (型 {target.GetTypeName() if target else None}、μ {mu})"
            f" -> ComputeBoundMaterial {'あり ' + str(bound.GetPath()) if bound else 'なし'}"
        )


def _groups(n_per: int, k: int) -> list[torch.Tensor]:
    return [torch.arange(g * n_per, (g + 1) * n_per) for g in range(k)]


def _slide(log: Log, env, u) -> None:
    """`env.reset()` の後のまっすぐなロープで測る (指の試験で書いた指の材質は、床の上の滑りに関わらない)。"""
    k = args_cli.slide_envs
    env.reset()
    rope = u.scene["rope"]
    log(
        f"[slide] 読み戻し (何も書かない env): ロープ μ {rope.root_physx_view.get_material_properties()[0, 0].tolist()}"
    )
    mats = rope.root_physx_view.get_material_properties().clone()
    groups = _groups(k, len(SLIDE_MU))
    for ids, mu in zip(groups, SLIDE_MU):
        if mu is not None:
            mats[ids, :, 0:2] = mu
    rope.root_physx_view.set_material_properties(mats, torch.arange(mats.shape[0]))
    n = k * len(SLIDE_MU)
    dt = u.sim.get_physics_dt()

    def run(steps: int) -> None:
        for _ in range(steps):
            u.scene.write_data_to_sim()
            u.sim.step(render=False)
            u.scene.update(dt)

    run(240)  # 1 s 落ち着かせる
    p = rope.data.body_pos_w[0, :, :2]
    axis = (p[-1] - p[0]) / (p[-1] - p[0]).norm()
    c0 = rope.data.body_pos_w[:, :, :2].mean(1).clone()
    vel = torch.zeros(mats.shape[0], 6, device=u.device)  # 試験に使わない env も同じに動かす
    vel[:, 0:2] = axis
    rope.write_root_velocity_to_sim(vel)
    rope.write_joint_velocity_to_sim(torch.zeros_like(rope.data.joint_vel))
    run(480)
    d = (rope.data.body_pos_w[:, :, :2].mean(1) - c0).norm(dim=-1).cpu()[:n]
    log("[slide] 軸方向に 1 m/s。μ_eff = 1 / (2 g d)。ロープ↔床は average 合成 (床 2.0) なら μ_eff = (μ_rope + 2) / 2")
    for ids, mu in zip(groups, SLIDE_MU):
        x = d[ids]
        name = "書かない (USD のまま)" if mu is None else f"tensor API で μ={mu}"
        log(
            f"[slide] {name:24s} d = {x.median():.5f} m (min {x.min():.5f}, max {x.max():.5f})"
            f"  μ_eff = {1.0 / (2 * 9.81 * float(x.median())):.3f}"
        )


def _finger(log: Log, env, u) -> None:
    """reset の直後 (ロープの材質に何も書いていない) に 1 手を打つ。"""
    k = args_cli.finger_envs
    rope, grip = u.scene["rope"], u.scene["gripper"]
    log(
        f"[finger] 読み戻し (何も書かない env): 形状ごとの μ {grip.root_physx_view.get_material_properties()[0, :, 0].tolist()}"
    )
    names = [p.rsplit("/", 1)[-1] for p in grip.root_physx_view.link_paths[0]]
    fingers = [i for i, n in enumerate(names) if n.endswith("finger")]  # 1 リンク 1 形状 (起動時に確かめる)
    assert grip.root_physx_view.max_shapes == len(names)
    mats = grip.root_physx_view.get_material_properties().clone()
    groups = _groups(k, len(FINGER_MU))
    for ids, mu in zip(groups, FINGER_MU):
        if mu is not None:
            mats[ids[:, None], torch.tensor(fingers)[None, :], 0:2] = mu
    grip.root_physx_view.set_material_properties(mats, torch.arange(mats.shape[0]))
    gen = torch.Generator().manual_seed(args_cli.seed)
    action = (torch.rand(1, 6, generator=gen) * 2 - 1).repeat(mats.shape[0], 1).to(u.device)
    env.step(action)
    pos = (rope.data.body_pos_w - u.scene.env_origins[:, None, :]).cpu()
    ref = pos[groups[0]].mean(0)
    log(
        f"[finger] 1 手目 (行動 {[round(v, 3) for v in action[0].tolist()]}) の後。組の平均の形の、組 0 (書かない) からの距離"
    )
    for g, (ids, mu) in enumerate(zip(groups, FINGER_MU)):
        p = pos[ids]
        within = (p - p.mean(0)).norm(dim=-1).mean()
        dist = (p.mean(0) - ref).norm(dim=-1).mean()
        name = "書かない" if mu is None else f"指 μ={mu}"
        log(f"[finger] 組 {g} {name:12s} 組 0 からの距離 {dist:.4f} m  組の中のばらつき {within:.4f} m")


def main() -> None:
    out_dir = args_cli.out or os.path.join(_REPO_ROOT, "outputs", "friction_check")
    os.makedirs(out_dir, exist_ok=True)
    log = Log(os.path.join(out_dir, "summary.log"))
    log(f"=== friction_check task={args_cli.task} seed={args_cli.seed}")
    env, u = _make_env(max(args_cli.slide_envs * len(SLIDE_MU), args_cli.finger_envs * len(FINGER_MU)))
    _check_bindings(log)
    # env.step() の中で作られたテンソルは inference tensor なので、後の reset も同じ mode で回す。
    with torch.inference_mode():
        _finger(log, env, u)
        _slide(log, env, u)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
