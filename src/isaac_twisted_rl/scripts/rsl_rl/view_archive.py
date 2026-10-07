"""段ごとのアーカイブ (`archive.pt`) の形を GUI で並べて見る。方策も物理の時間発展も使わない。

    uv run python src/isaac_twisted_rl/scripts/rsl_rl/view_archive.py \
        --archive logs/rsl_rl/knot_tying_v7/<run>/archive.pt --per_stage 8 --device cuda:0

## なぜこの形か

* **形の段は床板の色で見せる** (段 0 / 1 / 2 = 青 / 黄 / 赤。`play.py` のグリッパの色と同じ並び)。
  床 (`/World/defaultGroundPlane`) は全環境で 1 枚なので、各環境の原点に見た目だけの薄い板
  (当たり判定なし) を置いて塗る。色はアーカイブの段 (どの段のバッファに入っていたか)。
* **物理を進めない。** 書き込んだ形を表示し続けるだけにする。進めるとハンドとの接触や沈み込みで形が
  変わり、アーカイブに入っていた形ではなくなる。`sim.render()` は物理を止めたまま関節 -> リンクの位置と
  描画を更新する (`SimulationContext.forward`)。
* **書き方は `ResetToStage` と同じ** (重心を env 原点へ、速度 0)。yaw は 0 にそろえる。z 軸まわりの回転は
  位相を変えないので、見比べやすさを優先した。
* **位相も表にして出す。** 見た目だけでは、交差の数や「たどるべき辺」(観測 `next_edge`) の有無は
  読めない。書き込んだ後に位相を計算し直し、環境ごとに アーカイブの段・計算し直した段・交差の数・
  辺の有無・p-data を出す (段が食い違う形は `restore_mismatch` と同じもの)。
* Play 環境 (`-Play-` のタスク) を使う。アーカイブを書き出さない (`save_every=0`) ので、読んだ
  `archive.pt` を上書きしない。

`--headless` では表を出して終わる (形の位相だけを数えるとき)。

## 段の定義を変えた版の初期アーカイブを作る (`--write_filtered`)

段は `--task` の版の段の定義 (`stage_definition`。v10 は `path`) で計算し直す。`--write_filtered <新しいパス>` は、
指定した段の **全部の形** を書き込み、計算し直した段がアーカイブの段と一致した形だけを新しいファイルに書く
(`--per_stage` は無視する)。v10 は台本の初期アーカイブのうち、段 0〜2 の正規の経路の上の形だけで始める
(`docs/decisions/0023-v10-path-stages.md`)。物理を進めないので、残った形は元のファイルの形とまったく同じ:

    uv run python src/isaac_twisted_rl/scripts/rsl_rl/view_archive.py --headless \
        --archive outputs/archive_seed/overhand_28links.pt --task Isaac-Twisted-RL-Knot-Play-v10 \
        --write_filtered outputs/archive_seed/overhand_28links_v10.pt

既にあるファイルには書かない (`outputs/archive_seed/` の初期アーカイブを上書きしないため)。

## 形の p-data を書き出す (`--dump_pdata`)

`--dump_pdata <json>` は、見せた形ごとに `{stage, index, raw, reduced}` (アーカイブの段・番号・R2 除去の前と後の
short p-data) を JSON の配列に書く。表に無い形の p-data も文字列で出るので、段の定義の案を Isaac Sim なしで
数え直せる (`--per_stage` を大きくして、段ごとの形の分布を見る)。
"""

import argparse
import json
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="View the shapes in a stage archive (archive.pt) in the GUI.")
parser.add_argument("--archive", type=str, required=True, help="Archive file (archive.pt of a run, or a seed archive).")
parser.add_argument("--task", type=str, default="Isaac-Twisted-RL-Knot-Play-v7", help="Play task (rope preset).")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--stages", type=str, default="0,1,2", help="Archive stages to show (comma separated).")
parser.add_argument("--per_stage", type=int, default=8, help="Shapes per stage.")
parser.add_argument("--seed", type=int, default=0, help="Seed for picking the shapes.")
parser.add_argument(
    "--dump_pdata", type=str, default=None, help="Write per-shape {stage, index, raw, reduced} p-data as JSON here."
)
parser.add_argument(
    "--write_filtered",
    type=str,
    default=None,
    help="Write the shapes whose recomputed stage matches the archive stage to this new file (uses all shapes).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnvCfg

import isaac_twisted_rl.tasks  # noqa: F401
from isaac_twisted_rl.scripts.rsl_rl.archive_view import filtered_state_dict, pick_shapes
from isaac_twisted_rl.scripts.rsl_rl.stage_colors import STAGE_COLOR_NAMES, STAGE_COLORS
from isaac_twisted_rl.tasks.knot_tying.common import state_archive as sa
from isaac_twisted_rl.tasks.knot_tying.common import topology as tp
from isaac_twisted_rl.tasks.knot_tying.common.mdp.observations import (
    get_extractor,
    get_next_edge_goal,
    invalidate_extractor,
    stage_function,
)
from isaaclab_tasks.utils.hydra import hydra_task_config

FLOOR_FILL = 0.9
"""床板の一辺 / 環境の間隔。隣の環境の板と隙間を空け、境目を見えるようにする。"""

FLOOR_THICKNESS = 0.002
"""床板の厚さ [m]。上面を床 (z = 0) より 1 mm 上に出し、床と重なってちらつかないようにする。"""


def spawn_floor_tiles(unwrapped, stage_of_env: list[int]) -> None:
    """各環境の原点に、アーカイブの段の色の床板を置く (見た目だけ。当たり判定・剛体は付けない)。"""
    side = FLOOR_FILL * float(unwrapped.scene.cfg.env_spacing)
    cfgs = [
        sim_utils.CuboidCfg(
            size=(side, side, FLOOR_THICKNESS),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c, roughness=0.8),
        )
        for c in STAGE_COLORS
    ]
    # 板は env の prim (`/World/envs/env_N`) の下に置かない。env_0 の下に足した子 prim は他の env にも
    # 現れる (env_0 に 1 枚足しただけで env_1, env_2 にも `ArchiveStageFloor` ができる) ので、2 枚目以降が
    # 「prim が既にある」で落ち、環境ごとに色も変えられない。env の外に、env の原点の位置で置く。
    root = "/World/ArchiveStageFloors"
    unwrapped.scene.stage.DefinePrim(root, "Xform")
    origins = unwrapped.scene.env_origins.cpu().tolist()
    for e, s in enumerate(stage_of_env):
        cfg = cfgs[min(s, len(cfgs) - 1)]
        cfg.func(f"{root}/tile_{e}", cfg, translation=tuple(origins[e]))


def look_at_all(unwrapped) -> None:
    """全環境が入るように斜め上からカメラを向ける。"""
    origins = unwrapped.scene.env_origins
    center = origins.mean(dim=0)
    extent = float((origins[:, :2] - center[:2]).abs().max()) + float(unwrapped.scene.cfg.env_spacing)
    target = center.tolist()
    eye = (target[0], target[1] - 1.1 * extent, target[2] + 1.3 * extent)
    unwrapped.sim.set_camera_view(eye=eye, target=target)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg):
    data = torch.load(args_cli.archive, map_location="cpu")
    entries = {int(k): v for k, v in data["entries"].items()}
    sizes = {k: int(v["root_pose"].shape[0]) for k, v in entries.items()}
    stages = [int(s) for s in args_cli.stages.split(",") if s.strip()]
    if args_cli.write_filtered and os.path.exists(args_cli.write_filtered):
        raise FileExistsError(f"{args_cli.write_filtered} は既にある (上書きしない)")
    per_stage = max(sizes.values(), default=0) if args_cli.write_filtered else args_cli.per_stage
    picks = pick_shapes(sizes, stages, per_stage, args_cli.seed)
    if not picks:
        raise ValueError(f"{args_cli.archive} に段 {stages} の形が無い (件数: {sizes})")
    print(f"[INFO] {args_cli.archive}: 段 -> 件数 {sizes}。見せる形 {len(picks)} 個")

    env_cfg.scene.num_envs = len(picks)
    env_cfg.seed = args_cli.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    env = gym.make(args_cli.task, cfg=env_cfg)
    unwrapped = env.unwrapped
    dev = unwrapped.device
    unwrapped.reset()

    # ResetToStage と同じ書き方 (重心を env 原点へ、速度 0)。yaw は 0。
    rel = torch.stack([entries[s]["root_pose"][i] for s, i in picks]).to(dev)
    joints = torch.stack([entries[s]["joint_pos"][i] for s, i in picks]).to(dev)
    ids = torch.arange(len(picks), device=dev)
    rope = unwrapped.scene["rope"]
    pose = sa.place_at_origin(rel, unwrapped.scene.env_origins[ids], torch.zeros(len(picks), device=dev))
    rope.write_root_pose_to_sim(pose, env_ids=ids)
    rope.write_root_velocity_to_sim(torch.zeros(len(picks), 6, device=dev), env_ids=ids)
    rope.write_joint_state_to_sim(joints, torch.zeros_like(joints), env_ids=ids)
    # GPU の物理では、関節を書いただけではリンクの位置が更新されない。位相を計算する前に運動学を回す。
    unwrapped.sim.forward()
    invalidate_extractor(unwrapped)
    unwrapped.observation_manager.compute()

    extractor = get_extractor(unwrapped)
    stage_now = stage_function(unwrapped).stage(extractor.batch, extractor.reduced_batch).tolist()
    crossings = extractor.batch.count.tolist()
    reduced = extractor.reduced_batch.count.tolist()
    goal = get_next_edge_goal(unwrapped)
    has_edge = (goal.cand >= 0).tolist() if goal is not None else [None] * len(picks)
    names = {tp.p_data_key(pd): pd for pd in tp.potential_table()}
    names.update({tp.p_data_key(pd): pd for pds in tp.PATH_STAGE_P_DATA.values() for pd in pds})
    keys = tp.batch_keys(extractor.batch).tolist()

    legend = " / ".join(f"段 {k} = {n}" for k, n in enumerate(STAGE_COLOR_NAMES))
    print(f"[INFO] 床板の色はアーカイブの段: {legend}")
    print("env | アーカイブの段 | 番号 | 計算し直した段 | 交差 (R2 除去後) | たどる辺 | p-data")
    for e, ((s, i), sn, c, cr, he, k) in enumerate(zip(picks, stage_now, crossings, reduced, has_edge, keys)):
        flag = "" if sn == s else "  <- 段が食い違う"
        edge = "-" if he is None else ("あり" if he else "なし")
        print(f"{e:3d} | {s} | {i:5d} | {sn} | {c} ({cr}) | {edge} | {names.get(k, '(表に無い)')}{flag}")
    for s in stages:
        rows = [e for e, (ps, _) in enumerate(picks) if ps == s]
        if rows and goal is not None:
            no_edge = sum(1 for e in rows if not has_edge[e])
            mean_c = sum(crossings[e] for e in rows) / len(rows)
            print(f"[INFO] 段 {s}: {len(rows)} 形、交差の平均 {mean_c:.1f}、たどる辺なし {no_edge} 形")

    if args_cli.dump_pdata:
        raw = tp.p_data_from_batch(extractor.batch, short=True)
        red = tp.p_data_from_batch(extractor.reduced_batch, short=True)
        rows = [{"stage": s, "index": i, "raw": r, "reduced": d} for (s, i), r, d in zip(picks, raw, red)]
        os.makedirs(os.path.dirname(os.path.abspath(args_cli.dump_pdata)), exist_ok=True)
        with open(args_cli.dump_pdata, "w") as f:
            json.dump({"archive": args_cli.archive, "task": args_cli.task, "shapes": rows}, f)
        print(f"[INFO] 形の p-data を書いた: {args_cli.dump_pdata} ({len(rows)} 形)")

    if args_cli.write_filtered:
        keep = [sn == s for (s, _), sn in zip(picks, stage_now)]
        out = filtered_state_dict(entries, int(data["num_joints"]), picks, keep)
        kept = {s: int(v["root_pose"].shape[0]) for s, v in out["entries"].items()}
        os.makedirs(os.path.dirname(os.path.abspath(args_cli.write_filtered)), exist_ok=True)
        torch.save(out, args_cli.write_filtered)
        print(f"[INFO] 段が一致した形を書いた: {args_cli.write_filtered} (段 -> 件数 {kept}、元 {sizes})")

    if not unwrapped.sim.has_gui():
        env.close()
        return

    spawn_floor_tiles(unwrapped, [s for s, _ in picks])
    look_at_all(unwrapped)
    print("[INFO] 物理は止めたまま表示している。ウィンドウを閉じると終わる。")
    while simulation_app.is_running():
        unwrapped.sim.render()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
