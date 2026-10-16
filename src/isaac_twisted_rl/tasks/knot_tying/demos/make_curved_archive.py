"""chain 評価用の「曲がった・ぐにゃぐにゃだが交差のない」初期形状のアーカイブ (段 0 だけ) を作る。

    uv run python -u src/isaac_twisted_rl/tasks/knot_tying/demos/make_curved_archive.py \
        --task Isaac-Twisted-RL-Knot-Play-v14 --headless --device cuda:0 \
        --count 1024 --seed 0 --out outputs/archive_eval/curved_{num_links}links.pt

評価は `eval_chain.py --start curved --curved_archive <このファイル>` で読む。手順:

1. ロープの関節ごとの自由度が「平面の中の曲がり」かを、1 自由度ずつ動かして実測する (`probe_yaw_dofs`)。
   関節の並び順を仮定しない (USD の関節の並びと articulation の自由度の並びが同じとは限らないため)。
2. `common/curved_shapes.py` で曲がり角の列を引き、床の少し上に書き込む。
3. 物理で落ち着かせ (ハンドは遠くの待機位置のまま)、落ち着いた後の形が次を満たすものだけを残す:
   有限・床に落ちている・交差 0 (= 段 0)・隣でないリンクが近すぎない・意図した形から大きく崩れていない。
   落ち着かせると、関節のばねで少し戻る形がある。保存するのは落ち着いた後の形なので、評価の開始時の形に
   物理の緩和は残らない。
4. `ResetToStage` が読む形式 (`StateArchive.save`) で保存する。`meta` に生成の条件を持たせる。

意図した形との比較で符号 (曲がる向き) の取り違えを検出する。比較は回転と平行移動を許し、鏡映は許さない。
"""

from __future__ import annotations

import argparse
import json
import os

from isaaclab.app import AppLauncher

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), *[".."] * 5))

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", type=str, default="Isaac-Twisted-RL-Knot-Play-v14", help="評価に使う Play タスク。")
parser.add_argument("--num_envs", type=int, default=128, help="並列に落ち着かせる数 (1 回に試す形の数)。")
parser.add_argument("--count", type=int, default=1024, help="保存する形の数の目標。")
parser.add_argument("--max_rounds", type=int, default=40, help="棄却が多くても、これ以上は試さない。")
parser.add_argument("--seed", type=int, default=0, help="形の乱数のシード (評価のシードとは別)。")
parser.add_argument("--min_total_turn", type=float, default=0.6, help="全曲がり量 [rad] の下限 (ほぼまっすぐを除く)。")
parser.add_argument("--min_clearance_diameters", type=float, default=1.5, help="隣でないリンク中心の最小距離 [直径]。")
parser.add_argument("--max_shape_rms_diameters", type=float, default=1.0, help="意図した形との食い違い [直径] の上限。")
parser.add_argument("--settle_time", type=float, default=1.5, help="落ち着かせる時間 [s]。")
parser.add_argument("--max_speed", type=float, default=0.05, help="落ち着いたとみなす最大のリンク速さ [m/s]。")
parser.add_argument(
    "--out",
    type=str,
    default=os.path.join(_REPO_ROOT, "outputs", "archive_eval", "curved_{num_links}links.pt"),
    help="保存先。{num_links} はリンク数に置き換える。既存のファイルは上書きしない。",
)
parser.add_argument("--plot", type=str, default="", help="確認用の図 (PNG) の保存先。")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaac_twisted_rl.tasks  # noqa: E402, F401
from isaac_twisted_rl.tasks.knot_tying.common import curved_shapes as cs  # noqa: E402
from isaac_twisted_rl.tasks.knot_tying.common import state_archive as sa  # noqa: E402
from isaac_twisted_rl.tasks.knot_tying.common import topology as tp  # noqa: E402
from isaac_twisted_rl.tasks.knot_tying.common.rope_state import RopeTopologyExtractor  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

_PROBE_ANGLE = 0.2
"""1 自由度ずつ動かして向きの変化を測るときの角度 [rad]。"""

_FLOOR_Z_DIAMETERS = 0.55
"""書き込むときのリンク中心の高さ (直径の倍数)。半径 = 0.5 直径 より少し上から落とす。"""


class Sim:
    """シーンを借りて、形を書き込み、物理を進める。"""

    def __init__(self, env):
        self.uenv = env.unwrapped
        self.scene = self.uenv.scene
        self.sim = self.uenv.sim
        self.rope = self.scene["rope"]
        self.device = self.uenv.device
        self.n = self.uenv.num_envs
        self.dt = self.sim.get_physics_dt()
        self.origins = self.scene.env_origins
        pp = self.uenv.cfg.actions.pick_place
        self.diameter = float(pp.rope_diameter)
        self.extractor = RopeTopologyExtractor(self.rope, env_origins=self.origins)
        self.body_index = self.extractor.body_index
        self.num_links = self.extractor.num_links

    def links(self) -> torch.Tensor:
        return self.rope.data.body_pos_w[:, self.body_index] - self.origins.unsqueeze(1)

    def step(self, seconds: float) -> None:
        for _ in range(max(int(round(seconds / self.dt)), 1)):
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(self.dt)

    def write(self, root_xyz: torch.Tensor, heading0: torch.Tensor, joint_pos: torch.Tensor) -> None:
        """全環境に、リンク 0 の位置 (env ローカル)・向き・関節角を書き込む。"""
        pose = torch.zeros(self.n, 7, device=self.device)
        pose[:, 0:3] = root_xyz + self.origins
        pose[:, 3] = torch.cos(0.5 * heading0)
        pose[:, 6] = torch.sin(0.5 * heading0)
        self.rope.write_root_pose_to_sim(pose)
        self.rope.write_root_velocity_to_sim(torch.zeros(self.n, 6, device=self.device))
        self.rope.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos))
        self.scene.write_data_to_sim()


def heading_of(links: torch.Tensor) -> torch.Tensor:
    """リンク中心の折れ線の線分ごとの向き `(N, L-1)` [rad] (xy 平面)。"""
    seg = links[:, 1:, :2] - links[:, :-1, :2]
    return torch.atan2(seg[..., 1], seg[..., 0])


def probe_yaw_dofs(s: Sim) -> torch.Tensor:
    """関節 k (リンク k と k+1 の間) の「平面の曲がり」の自由度の番号 `(L-1,)` を返す。

    自由度の名前 (`Joint{k:02d}:{0,1,2}`。0 = ねじれ、1 = 縦曲げ、2 = 平面の曲がり) で対応づけ、名前の
    取り決めが合っているかを実測で確かめる: 自由度 d だけを `_PROBE_ANGLE` にして 1 ステップ進め、リンクの向きが
    反時計回りに変わり (符号)、高さの差がほぼ出ない (平面の中) ことを見る。ばねで戻るので角度の大きさは見ない。
    """
    names = list(s.rope.joint_names)
    num_joints = s.num_links - 1
    dofs = torch.tensor([names.index(f"Joint{k:02d}:2") for k in range(num_joints)])
    ids_all = dofs.tolist()
    for start in range(0, num_joints, s.n):
        ids = ids_all[start : start + s.n]
        joint_pos = torch.zeros(s.n, s.rope.num_joints, device=s.device)
        joint_pos[torch.arange(len(ids)), torch.tensor(ids)] = _PROBE_ANGLE
        root = torch.zeros(s.n, 3, device=s.device)
        root[:, 2] = _FLOOR_Z_DIAMETERS * s.diameter + 0.05  # 床に触れない高さ (測定だけ)
        s.write(root, torch.zeros(s.n, device=s.device), joint_pos)
        s.step(s.dt)
        links = s.links()
        h = heading_of(links)
        turn = h[:, -1] - h[:, 0]  # 最初と最後の線分の向きの差 (どの関節の曲がりも、この間に入る)
        tilt = (links[:, 1:, 2] - links[:, :-1, 2]).abs().amax(dim=1)
        for e in range(len(ids)):
            if not (float(turn[e]) > 0.02 and float(tilt[e]) < 1.0e-3):
                raise RuntimeError(
                    f"関節 {start + e} の自由度 {ids[e]} ({names[ids[e]]}) は、平面の中の反時計回りの曲がりではない "
                    f"(向きの差 {float(turn[e]):.3f} rad、高さの差 {float(tilt[e]):.4f} m)。自由度の名前の取り決めを確かめる"
                )
    return dofs


def rigid_rms(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """xy 2 点列 `(N, L, 2)` どうしの、回転と平行移動を許した後の二乗平均の平方根 (鏡映は許さない)。"""
    return sa.rigid_aligned_rms(a, b)


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.episode_length_s = 1.0e6  # 物理を直接進めるので、時間切れの reset は起こさない
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()
    s = Sim(env)
    num_links, num_joints = s.num_links, s.num_links - 1
    num_dofs = s.rope.num_joints

    out_path = args_cli.out.format(num_links=num_links)
    if os.path.exists(out_path):
        raise FileExistsError(f"{out_path} が既にある。上書きしない (別のパスを --out で指定する)")

    # まっすぐに reset された直後のリンクの間隔
    straight = s.links()
    spacing = float((straight[0, 1] - straight[0, 0]).norm())
    print(
        f"[INFO] links={num_links} dofs={num_dofs} spacing={spacing * 1000:.2f} mm diameter={s.diameter * 1000:.2f} mm"
    )

    yaw_dof = probe_yaw_dofs(s).to(s.device)
    print(f"[INFO] 平面の曲がりの自由度 (関節 0.. の並び): {yaw_dof.tolist()}")

    gen = torch.Generator().manual_seed(args_cli.seed)
    kept = sa.StateArchive(classes=(0,), capacity=args_cli.count, num_joints=num_dofs, device=s.device)
    stats = {"tried": 0, "settled_ok": 0, "crossing": 0, "clearance": 0, "turn_small": 0, "deviated": 0, "moving": 0}
    turn_hist: list[float] = []
    keep_xy: list[torch.Tensor] = []
    bfs = tp.get_bfs_potential(s.device)

    for rnd in range(args_cli.max_rounds):
        if kept.sizes()[0] >= args_cli.count:
            break
        turns = cs.sample_turning_angles(s.n, num_joints, gen)
        xy = cs.chain_nodes(turns, spacing)  # (n, L, 2)。リンク 0 が原点
        xy_c = xy - xy.mean(dim=1, keepdim=True)
        heading0 = torch.rand(s.n, generator=gen) * 2.0 * torch.pi
        root_xyz = torch.zeros(s.n, 3)
        root_xyz[:, 2] = _FLOOR_Z_DIAMETERS * s.diameter
        # 重心を原点に置き、リンク 0 の位置を書く。向きは heading0 だけ回す (回転後の形が意図した形)。
        c, sn = torch.cos(heading0), torch.sin(heading0)
        rot = torch.stack([torch.stack([c, -sn], -1), torch.stack([sn, c], -1)], -2)  # (n, 2, 2)
        xy_w = xy_c @ rot.transpose(1, 2)
        root_xyz[:, :2] = xy_w[:, 0]
        joint_pos = torch.zeros(s.n, num_dofs)
        joint_pos[:, yaw_dof.cpu()] = turns
        s.write(root_xyz.to(s.device), heading0.to(s.device), joint_pos.to(s.device))
        s.step(args_cli.settle_time)

        links = s.links()
        speed = s.rope.data.body_lin_vel_w.norm(dim=-1).amax(dim=1)
        s.extractor.update(s.rope)
        batch, red = s.extractor.batch, s.extractor.reduced_batch
        stage = bfs.stage(batch, red)
        settled_xy = links[..., :2]
        ok = torch.isfinite(links).all(dim=-1).all(dim=-1)
        ok &= torch.nan_to_num(links[..., 2], nan=1e9).amax(dim=1) <= 3.0 * s.diameter
        moving = ok & (torch.nan_to_num(speed, nan=1e9) > args_cli.max_speed)
        ok &= ~moving
        crossing = ok & ((stage != 0) | (batch.count != 0))
        ok &= ~crossing
        clearance = cs.min_clearance(torch.nan_to_num(settled_xy, nan=0.0).cpu()).to(s.device)
        too_close = ok & (clearance < args_cli.min_clearance_diameters * s.diameter)
        ok &= ~too_close
        deviation = rigid_rms(xy_w.to(s.device), torch.nan_to_num(settled_xy, nan=0.0))
        deviated = ok & (deviation > args_cli.max_shape_rms_diameters * s.diameter)
        ok &= ~deviated
        total = cs.total_turning(torch.nan_to_num(settled_xy, nan=0.0).cpu()).to(s.device)
        small = ok & (total < args_cli.min_total_turn)
        ok &= ~small

        stats["tried"] += s.n
        stats["moving"] += int(moving.sum())
        stats["crossing"] += int(crossing.sum())
        stats["clearance"] += int(too_close.sum())
        stats["deviated"] += int(deviated.sum())
        stats["turn_small"] += int(small.sum())
        stats["settled_ok"] += int(ok.sum())
        rows = ok.nonzero(as_tuple=False).squeeze(-1)
        room = args_cli.count - kept.sizes()[0]
        rows = rows[:room]
        if rows.numel() > 0:
            rel = sa.to_centroid_frame(
                s.rope.data.root_link_pose_w[rows], s.rope.data.body_pos_w[rows][:, s.body_index], s.origins[rows]
            )
            kept.add(torch.zeros(rows.numel(), dtype=torch.long, device=s.device), rel, s.rope.data.joint_pos[rows])
            turn_hist += total[rows].tolist()
            keep_xy.append(settled_xy[rows].cpu())
        print(
            f"[round {rnd}] kept={kept.sizes()[0]}/{args_cli.count} {json.dumps(stats)} max_dev={float(deviation.nan_to_num(1e9).max()):.4f}",
            flush=True,
        )

    n_kept = kept.sizes()[0]
    if n_kept == 0:
        raise RuntimeError("1 つも残らなかった。棄却の内訳 (上の統計) を見て条件を確かめる")
    meta = {
        "task": args_cli.task,
        "seed": args_cli.seed,
        "count": n_kept,
        "args": vars(args_cli),
        "stats": stats,
        "spacing": spacing,
        "diameter": s.diameter,
        "yaw_dof": yaw_dof.tolist(),
        "total_turn_quantiles": torch.tensor(turn_hist).quantile(torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])).tolist(),
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    state = kept.state_dict()
    state["meta"] = meta
    tmp = f"{out_path}.tmp"
    torch.save(state, tmp)
    os.replace(tmp, out_path)
    print(
        f"[dump] {out_path}: 段 0 に {n_kept} 件。全曲がり量 [rad] の分位 (0/25/50/75/100%): {meta['total_turn_quantiles']}"
    )

    if args_cli.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        xy = torch.cat(keep_xy)
        pick = torch.linspace(0, xy.shape[0] - 1, min(32, xy.shape[0])).long()
        fig, axes = plt.subplots(4, 8, figsize=(20, 10))
        for ax, i in zip(axes.flat, pick):
            p = xy[i] - xy[i].mean(dim=0)
            ax.plot(p[:, 0], p[:, 1], "-o", ms=2)
            ax.set_aspect("equal")
            ax.set_xlim(-0.6, 0.6)
            ax.set_ylim(-0.6, 0.6)
            ax.set_title(f"#{int(i)}", fontsize=8)
        fig.savefig(args_cli.plot, dpi=80)
        print(f"[plot] {args_cli.plot}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
