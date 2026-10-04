"""紐モデルの物理の妥当性と安定性を、全モデル共通の台本で測るベンチマーク。紐モデルの比較で追加。

    uv run python -u src/isaac_twisted_rl/tasks/knot_tying/rope_study/bench_physics.py \
        --model XpbdC28 --num_envs 128 --headless --device cuda:0

学習と同じ環境 (`Isaac-Twisted-RL-RS-<モデル>-v4`) を立ち上げ、ハンドだけを台本で動かす
(`demos/overhand_script.py` と同じやり方: グリッパーの姿勢と指の目標を書き込み、`sim.step()` と
`scene.update()` を回す)。紐の状態は `scene["rope"]` の仮想リンクと、アダプタが返す表現本来の
中心線 (`nodes_w`) から測る。全環境が同じ台本を、パラメータを ±`--jitter` 揺らして打つ。

| 試験 | 何を見るか |
|---|---|
| `settle` | 床に置いたまま 1 s。残留速度 (過剰な振動)、伸び、床へのめり込み |
| `hang` | 中央を掴んで 0.4 m まで上げ 1 s 保持。把持の保持率、垂れ下がった形 (両半分の開き)、伸び、振動、紐どうしのめり込み |
| `drag` | 端から 3 本目を掴み、0.2 m に上げて横へ 0.4 m 運んで降ろす。紐のどこまでが付いてくるか (摩擦と曲げ剛性で決まる) |
| `knot` | ゲート 1 の台本で結べた固結び (`overhand_script.py --dump_knots`、`<モデル>_knots.pt`) から始め、先端側を 0.3 m に持ち上げて降ろす。結び目が残るか (自己接触のすり抜け)、めり込み |
| `random` | 学習と同じ行動 (N(0, 0.5)) を `env.step` で 6 手。発散率、めり込み、把持の成否、段の遷移、1 手の時間 |

出力: `<out>/summary.json` (試験ごとの集計) と `<out>/shapes.pt` (試験ごとの中心線。モデル間の比較用)。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

from isaaclab.app import AppLauncher

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), *[".."] * 5))

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--tests", type=str, default="settle,hang,drag,knot,random")
parser.add_argument("--jitter", type=float, default=0.1, help="台本のパラメータの揺らし幅 (比)。")
parser.add_argument("--random_moves", type=int, default=6)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--cfg", type=str, default="", help="環境設定の上書き 'a.b=値;...' (値は Python リテラル)。")
parser.add_argument("--label", type=str, default="")
parser.add_argument("--out", type=str, default="")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import ast  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaac_twisted_rl.tasks  # noqa: E402, F401
from isaac_twisted_rl.tasks.knot_tying.common import state_archive as sa  # noqa: E402
from isaac_twisted_rl.tasks.knot_tying.common import topology as tp  # noqa: E402
from isaac_twisted_rl.tasks.knot_tying.common.mdp import phases  # noqa: E402
from isaac_twisted_rl.tasks.knot_tying.common.mdp.observations import get_extractor, invalidate_extractor  # noqa: E402
from isaac_twisted_rl.tasks.knot_tying.common.rope_backends import centerline as cl  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

RADIUS = 0.01
EXCLUDE_ARC = math.pi * RADIUS


def _set_cfg(cfg, spec: str):
    for item in filter(None, (s.strip() for s in spec.split(";"))):
        key, val = item.split("=", 1)
        obj = cfg
        parts = key.strip().split(".")
        for p in parts[:-1]:
            obj = obj[p] if isinstance(obj, dict) else getattr(obj, p)
        v = ast.literal_eval(val.strip())
        if isinstance(obj, dict):
            obj[parts[-1]] = v
        else:
            setattr(obj, parts[-1], v)


class Driver:
    """ハンドを台本で動かす (`overhand_script.ScriptRunner` の最小版)。"""

    def __init__(self, env):
        self.env = env
        self.u = env.unwrapped
        self.scene = self.u.scene
        self.sim = self.u.sim
        self.dt = self.sim.get_physics_dt()
        self.rope = self.scene["rope"]
        self.gripper = self.scene["gripper"]
        self.n = self.u.num_envs
        self.dev = self.u.device
        self.origins = self.scene.env_origins
        self.finger_ids, _ = self.gripper.find_joints(["left_finger_joint", "right_finger_joint"], preserve_order=True)
        lim = self.gripper.data.joint_pos_limits[:, self.finger_ids, :]
        self.open_value = torch.where(lim[..., 0].abs() > lim[..., 1].abs(), lim[..., 0], lim[..., 1])
        self.pos = self.gripper.data.root_pos_w.clone()
        self.yaw = torch.zeros(self.n, device=self.dev)
        self.ratio = torch.ones(self.n, device=self.dev)
        self.vmax = torch.zeros(self.n, device=self.dev)

    # -- 紐の状態
    def links(self) -> torch.Tensor:
        """仮想リンクの中心 (env 原点相対) `(N, 28, 3)`。"""
        return self.rope.data.body_pos_w - self.origins.unsqueeze(1)

    def nodes(self) -> torch.Tensor:
        """表現本来の中心線 (env 原点相対)。剛体鎖 (アダプタなし) はリンク中心から作る。"""
        if hasattr(self.rope, "nodes_w"):
            nodes = self.rope.nodes_w()
        else:
            nodes = tp.polyline_from_link_centers(self.rope.data.body_pos_w[:, get_extractor(self.u).body_index])
        return nodes - self.origins.unsqueeze(1)

    def speed(self) -> torch.Tensor:
        return torch.nan_to_num(self.rope.data.body_lin_vel_w.norm(dim=-1), nan=1e6).amax(dim=1)

    # -- ハンド
    def write(self, pos, yaw, vel, yaw_rate, ratio):
        pose = torch.zeros(self.n, 7, device=self.dev)
        pose[:, :3] = pos
        pose[:, 3] = torch.cos(0.5 * yaw)
        pose[:, 6] = torch.sin(0.5 * yaw)
        self.gripper.write_root_pose_to_sim(pose)
        v = torch.zeros(self.n, 6, device=self.dev)
        v[:, :3] = vel
        v[:, 5] = yaw_rate
        self.gripper.write_root_velocity_to_sim(v)
        self.gripper.set_joint_position_target(ratio.unsqueeze(-1) * self.open_value, joint_ids=self.finger_ids)

    def physics_step(self):
        self.scene.write_data_to_sim()
        self.sim.step(render=False)
        self.scene.update(self.dt)
        self.vmax = torch.maximum(self.vmax, self.speed())

    def segment(self, pos1, yaw1=None, r1=None, duration=0.3):
        pos0, yaw0, r0 = self.pos.clone(), self.yaw.clone(), self.ratio.clone()
        yaw1 = yaw0 if yaw1 is None else yaw1
        r1 = r0 if r1 is None else (r1 if torch.is_tensor(r1) else torch.full_like(r0, r1))
        steps = max(1, int(round(duration / self.dt)))
        dyaw = torch.remainder(yaw1 - yaw0 + math.pi, 2 * math.pi) - math.pi
        for i in range(steps):
            u = (i + 1) / steps
            s = u * u * (3 - 2 * u)
            ds = 6 * u * (1 - u) / (steps * self.dt)
            pos = pos0 + (pos1 - pos0) * s
            vel = (pos1 - pos0) * ds
            self.write(pos, yaw0 + dyaw * s, vel, dyaw * ds, r0 + (r1 - r0) * s)
            self.physics_step()
        self.pos, self.yaw, self.ratio = pos1.clone(), yaw0 + dyaw, r1.clone()

    def hold(self, duration):
        self.segment(self.pos.clone(), duration=duration)

    def teleport(self, pos, yaw):
        self.pos, self.yaw = pos.clone(), yaw.clone()
        self.write(pos, yaw, torch.zeros_like(pos), torch.zeros_like(yaw), self.ratio)
        self.physics_step()

    def tangent_yaw(self, links, k):
        d = links[:, min(k + 1, links.shape[1] - 1)] - links[:, max(k - 1, 0)]
        return torch.atan2(d[:, 1], d[:, 0])

    def world(self, xy, z):
        out = torch.zeros(self.n, 3, device=self.dev)
        out[:, :2] = xy + self.origins[:, :2]
        out[:, 2] = (z if torch.is_tensor(z) else torch.full((self.n,), float(z), device=self.dev)) + self.origins[:, 2]
        return out

    def pick(self, k: int, lift_z: float, jitter_xy=None):
        """仮想リンク k の真上へ瞬間移動して掴み、`lift_z` (手のひらの高さ) まで上げる。"""
        links = self.links()
        g = links[:, k, :2] if jitter_xy is None else links[:, k, :2] + jitter_xy
        yaw = self.tangent_yaw(links, k)
        self.ratio = torch.ones(self.n, device=self.dev)
        self.teleport(self.world(g, phases.HOME_PALM_Z), yaw)
        self.segment(self.world(g, phases.GRASP_PALM_Z), duration=0.3)
        self.segment(self.pos, r1=phases.CLOSE_RATIO, duration=0.3)
        self.hold(0.2)
        self.segment(self.world(g, lift_z), duration=0.4)
        return g, yaw

    def release_and_leave(self, place_xy, place_z):
        self.segment(self.world(place_xy, place_z), duration=0.4)
        self.segment(self.pos, r1=phases.OPEN_RATIO, duration=0.25)
        self.hold(0.3)
        self.segment(self.world(place_xy, phases.HOME_PALM_Z), duration=0.2)
        self.hold(0.5)

    def park(self):
        """ハンドを紐から遠くへ (env 原点の 1 m 横、高さ 0.3 m) 退避させる。"""
        xy = torch.zeros(self.n, 2, device=self.dev)
        xy[:, 1] = 1.0
        self.ratio = torch.ones(self.n, device=self.dev)
        self.teleport(self.world(xy, phases.HOME_PALM_Z), torch.zeros(self.n, device=self.dev))

    def reset(self):
        self.env.reset()
        invalidate_extractor(self.u)
        self.park()


# ---------------------------------------------------------------- 計測の小道具


def penetration(nodes):
    return cl.self_penetration(nodes, RADIUS, EXCLUDE_ARC)


def seg_len(nodes):
    return (nodes[:, 1:] - nodes[:, :-1]).norm(dim=-1)


def q(x: torch.Tensor, ps=(0.5, 0.9, 0.99)) -> dict:
    x = torch.nan_to_num(x.float().flatten(), nan=float("inf"))
    finite = x[torch.isfinite(x)]
    out = {
        "mean": float(finite.mean()) if finite.numel() else None,
        "max": float(finite.max()) if finite.numel() else None,
    }
    for p in ps:
        out[f"p{int(p * 100)}"] = float(finite.quantile(p)) if finite.numel() else None
    out["nonfinite"] = int((~torch.isfinite(x)).sum())
    return out


def stage_of(u):
    invalidate_extractor(u)
    ex = get_extractor(u)
    return tp.get_bfs_potential(u.device).stage(ex.batch, ex.reduced_batch)


# ---------------------------------------------------------------- 試験


def t_settle(D: Driver, res, shapes):
    D.reset()
    nodes0 = D.nodes().clone()
    D.vmax.zero_()
    D.hold(1.0)
    nodes = D.nodes()
    l0 = seg_len(nodes0).mean()
    res["settle"] = {
        "residual_speed": q(D.speed()),
        "max_speed_during": q(D.vmax),
        "strain": q((seg_len(nodes) / l0 - 1).abs().amax(dim=1)),
        "floor_penetration": q((RADIUS - nodes[..., 2]).clamp_min(0).amax(dim=1)),
        "zmax": q(nodes[..., 2].amax(dim=1)),
    }
    shapes["settle"] = nodes.cpu()


def t_hang(D: Driver, res, shapes, jitter):
    D.reset()
    D.hold(0.5)
    n = D.n
    l0 = seg_len(D.nodes()).mean()
    jit = (torch.rand(n, 2, device=D.dev) * 2 - 1) * jitter * 0.02
    D.vmax.zero_()
    g, _ = D.pick(14, 0.40, jit)
    D.hold(1.0)
    nodes = D.nodes()
    links = D.links()
    tip = D.pos[:, 2] - D.origins[:, 2] - phases.FINGER_TIP_OFFSET_Z
    # 掴んだ点 (仮想リンク 13/14 の間) の高さ。指の範囲 (指先から 6 cm) にあれば保持
    gz = 0.5 * (links[:, 13, 2] + links[:, 14, 2])
    held = (gz > tip - 0.02) & (gz < tip + 0.08)
    # 吊った両半分の開き: 掴んだ点から 10 cm / 20 cm 下での、左右の半分の水平距離
    spread = {}
    for depth in (0.1, 0.2):
        zq = gz - depth
        sep = []
        for half in (links[:, :14], links[:, 14:].flip(1)):
            dz = (half[..., 2] - zq.unsqueeze(1)).abs()
            k = dz.argmin(dim=1)
            sep.append(half.gather(1, k.view(-1, 1, 1).expand(-1, 1, 3)).squeeze(1)[:, :2])
        spread[f"spread_{int(depth * 100)}cm"] = q((sep[0] - sep[1]).norm(dim=-1)[held])
    v_hold = D.speed()
    res["hang"] = {
        "held": float(held.float().mean()),
        "grasp_z_minus_tip": q((gz - tip)[held]),
        **spread,
        "strain": q((seg_len(nodes) / l0 - 1).abs().amax(dim=1)[held]),
        "penetration": q(penetration(nodes)[held]),
        "residual_speed_after_1s": q(v_hold[held]),
        "max_speed_during": q(D.vmax),
        "diverged": float((D.vmax > 20).float().mean()),
    }
    shapes["hang"] = links.cpu()
    D.segment(D.world(g, phases.GRASP_PALM_Z + 0.05), duration=0.4)
    D.segment(D.pos, r1=phases.OPEN_RATIO, duration=0.25)


def t_drag(D: Driver, res, shapes, jitter):
    D.reset()
    D.hold(0.5)
    n = D.n
    links0 = D.links().clone()
    dist = 0.4 * (1 + (torch.rand(n, device=D.dev) * 2 - 1) * jitter)
    D.vmax.zero_()
    g, yaw = D.pick(3, phases.CARRY_PALM_Z)
    # 紐に垂直な向きへ運ぶ
    perp = torch.stack([-torch.sin(yaw), torch.cos(yaw)], -1)
    dst = g + perp * dist.unsqueeze(-1)
    D.segment(D.world(dst, phases.CARRY_PALM_Z), duration=0.6)
    D.release_and_leave(dst, phases.GRASP_PALM_Z + 0.05)
    links = D.links()
    nodes = D.nodes()
    disp = (links[..., :2] - links0[..., :2]).norm(dim=-1)  # (N, 28)
    moved = disp > 0.02
    res["drag"] = {
        "grasp_link_disp": q(disp[:, 3]),
        "far_end_disp": q(disp[:, -1]),
        "mean_disp": q(disp.mean(dim=1)),
        "frac_links_moved_2cm": q(moved.float().mean(dim=1)),
        "penetration": q(penetration(nodes)),
        "max_speed_during": q(D.vmax),
        "diverged": float((D.vmax > 20).float().mean()),
    }
    shapes["drag"] = links.cpu()


def t_knot(D: Driver, res, shapes, archive_path):
    """`overhand_script.py --dump_knots` で保存した固結び (段 3) から始め、端を持ち上げて降ろす。"""
    if not os.path.isfile(archive_path):
        res["knot"] = {"skipped": f"no archive {archive_path}"}
        return
    u = D.u
    rope = D.rope
    D.reset()
    arch = sa.StateArchive(classes=(3,), capacity=4096, num_joints=rope.num_joints, device=D.dev)
    arch.load(archive_path)
    drawn = arch.sample(D.n, [1.0])
    if drawn is None:
        res["knot"] = {"skipped": "empty archive"}
        return
    _, rel, joint = drawn
    pose = sa.place_at_origin(rel, D.origins, torch.zeros(D.n, device=D.dev))
    rope.write_root_pose_to_sim(pose)
    rope.write_root_velocity_to_sim(torch.zeros(D.n, 6, device=D.dev))
    rope.write_joint_state_to_sim(joint, torch.zeros_like(joint))
    D.hold(0.5)
    s0 = stage_of(u)
    D.vmax.zero_()
    # 段 3 の紐は輪の側がどちらの端か決まっていないので、両端のうち「結び目から遠い側」ではなく、
    # 単純に先端 (仮想リンク 25 = 端から 3 本目) を持ち上げる。台本の尻尾側 (tuck で通した側) にあたる。
    g, _ = D.pick(25, 0.30)
    D.hold(0.5)
    lifted_stage = stage_of(u)
    D.release_and_leave(g, phases.GRASP_PALM_Z + 0.05)
    s1 = stage_of(u)
    nodes = D.nodes()
    valid = s0 == 3
    res["knot"] = {
        "episodes": int(valid.sum()),
        "start_stage_hist": torch.bincount(s0, minlength=4).tolist(),
        "lifted_stage_hist": torch.bincount(lifted_stage, minlength=4).tolist(),
        "end_stage_hist": torch.bincount(s1, minlength=4).tolist(),
        "kept_after_release": float(((s1 == 3) & valid).float().sum() / max(int(valid.sum()), 1)),
        "penetration": q(penetration(nodes)[valid]),
        "max_speed_during": q(D.vmax),
        "diverged": float((D.vmax > 20).float().mean()),
    }
    shapes["knot"] = D.links().cpu()


def t_random(D: Driver, res, moves, seed):
    env, u = D.env, D.u
    g = torch.Generator(device=u.device).manual_seed(seed)
    env.reset()
    tm = u.termination_manager
    div, pen, held, t_move = [], [], [], []
    for _ in range(moves):
        a = (0.5 * torch.randn(u.num_envs, 6, device=u.device, generator=g)).clamp(-1, 1)
        torch.cuda.synchronize()
        t0 = time.time()
        env.step(a)
        torch.cuda.synchronize()
        t_move.append(time.time() - t0)
        d = tm.get_term("diverged").clone()
        div.append(d.float())
        nodes = D.nodes()
        pen.append(torch.where(d, torch.zeros_like(d, dtype=torch.float), penetration(nodes)))
        diag = u.action_manager.get_term("pick_place").diagnostics
        if diag:
            held.append(diag["held"].float())
    div = torch.stack(div)
    pen = torch.stack(pen)
    res["random"] = {
        "moves": int(div.numel()),
        "diverged_per_move": float(div.mean()),
        "penetration_after_move": q(pen),
        "penetration_gt_5mm": float((pen > 0.005).float().mean()),
        "grasp_held": float(torch.stack(held).mean()) if held else None,
        "sec_per_move": sum(t_move[1:]) / max(len(t_move) - 1, 1),
        "moves_per_sec": u.num_envs / (sum(t_move[1:]) / max(len(t_move) - 1, 1)),
    }


def main():
    task = f"Isaac-Twisted-RL-RS-{args_cli.model}-v4"
    cfg = parse_env_cfg(task, device=args_cli.device, num_envs=args_cli.num_envs)
    cfg.seed = args_cli.seed
    # 台本の試験はまっすぐな紐から始める。段のアーカイブ・ログ保存は使わない
    cfg.events.reset_to_stage.params["straight_prob"] = 1.0
    cfg.events.reset_to_stage.params["stage_weights"] = (1.0, 0.0, 0.0)
    cfg.rewards.record_stage_archive.params["save_every"] = 0
    cfg.actions.pick_place.diag = True
    if args_cli.cfg:
        _set_cfg(cfg, args_cli.cfg)
    env = gym.make(task, cfg=cfg)
    D = Driver(env)
    label = args_cli.label or args_cli.model
    out = args_cli.out or os.path.join(_REPO_ROOT, "outputs", "rope_study", "physics", label)
    os.makedirs(out, exist_ok=True)
    res = {"model": args_cli.model, "num_envs": args_cli.num_envs, "jitter": args_cli.jitter, "cfg": args_cli.cfg}
    shapes = {}
    tests = args_cli.tests.split(",")
    torch.manual_seed(args_cli.seed)
    with torch.inference_mode():
        for t in tests:
            t0 = time.time()
            if t == "settle":
                t_settle(D, res, shapes)
            elif t == "hang":
                t_hang(D, res, shapes, args_cli.jitter)
            elif t == "drag":
                t_drag(D, res, shapes, args_cli.jitter)
            elif t == "knot":
                arch = cfg.events.reset_to_stage.params["load_paths"][0].replace(".pt", "_knots.pt")
                t_knot(D, res, shapes, arch)
            elif t == "random":
                t_random(D, res, args_cli.random_moves, args_cli.seed)
            res.setdefault("wall_sec", {})[t] = time.time() - t0
            print(
                f"[bench] {t} done in {time.time() - t0:.0f}s: {json.dumps(res.get(t), ensure_ascii=False)[:600]}",
                flush=True,
            )
            with open(os.path.join(out, "summary.json"), "w") as f:
                json.dump(res, f, indent=2, ensure_ascii=False)
    torch.save(shapes, os.path.join(out, "shapes.pt"))
    print(f"[bench] wrote {out}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
