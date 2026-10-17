"""寸法の DR (v16) のロープを `MultiAssetSpawnerCfg` で spawn できるか、起動と 1 手の時間がどれだけ変わるかを確かめる。

    uv run python -u src/isaac_twisted_rl/tasks/knot_tying/demos/rope_size_spawn_check.py --headless --device cuda:0 \
        --task Isaac-Twisted-RL-Knot-v15 --num_envs 4096 --sizes grid --moves 3 --label grid_n4096

`docs/decisions/0029-v16-rope-size-dr.md` の「実装の順序」3。確かめること:

1. 28 リンクの Articulation を、env ごとに違う USD (寸法の 9 組) で spawn して `gym.make` が通るか。
2. spawn された寸法が env の番号 % 組の数の割り当てと一致するか (`mdp.check_rope_sizes` で USD の半径を読む)。
3. `scene.replicate_physics=False` での起動の時間 (`gym.make`) と、1 手 (`env.step`) の時間。`--sizes nominal` は
   公称の 1 組だけを同じ経路 (`RopeSizeSpawnerCfg`、replicate なし) で spawn し、`--sizes none` は継承元の版の
   設定のまま (replicate あり)。3 つを比べると、replicate を切った分と、細い・長い組の物理の分を分けられる。
4. 組ごとの 1 手あたりの発散率 (終了項 `diverged` を項の返り値で捕まえる。`bench_step_time.py` と同じ方法)。

継承元の版 (v15) の設定に spawn だけを差し込むので、行動の範囲・床の判定は公称の寸法のまま (寸法を env ごとに
読む実装は v16 の設定で使う)。発散率は寸法ごとの物理の目安で、ゲート (skill `physics-check`) の代わりにはならない。
行動はランダム (`N(0, 0.5)` を [−1, 1] に切る。`bench_step_time.py` と同じ)。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time

from isaaclab.app import AppLauncher

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), *[".."] * 5))

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", type=str, default="Isaac-Twisted-RL-Knot-v15")
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--sizes", choices=("grid", "nominal", "none"), default="grid")
parser.add_argument("--moves", type=int, default=3, help="計時する手の数 (最初の 1 手は計時しない)。")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--label", type=str, default="")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
t_launch = time.perf_counter()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import isaac_twisted_rl.tasks  # noqa: E402, F401
from isaac_twisted_rl.tasks.knot_tying.common import mdp  # noqa: E402
from isaac_twisted_rl.tasks.knot_tying.common.rope_model.rope_specs import (  # noqa: E402
    SIZE_DIAMETER_SCALES,
    SIZE_LENGTH_SCALES,
)
from isaac_twisted_rl.tasks.knot_tying.common.rope_task_cfg import apply_rope_sizes  # noqa: E402


def _gpu_mem() -> str:
    try:
        return subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except Exception as exc:  # noqa: BLE001
        return f"unavailable: {exc}"


def main():
    cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    cfg.seed = args_cli.seed
    cfg.rewards.record_stage_archive.params["save_every"] = 0
    if "history_every" in cfg.rewards.record_stage_archive.params:
        cfg.rewards.record_stage_archive.params["history_every"] = 0
    if args_cli.sizes == "grid":
        apply_rope_sizes(cfg, "knot", [(ls, ds) for ls in SIZE_LENGTH_SCALES for ds in SIZE_DIAMETER_SCALES])
    elif args_cli.sizes == "nominal":
        apply_rope_sizes(cfg, "knot", [(1.0, 1.0)])

    t_make0 = time.perf_counter()
    env = gym.make(args_cli.task, cfg=cfg)
    t_make = time.perf_counter() - t_make0
    u = env.unwrapped
    n, dev = u.num_envs, u.device
    print(
        f"[size_check] gym.make {t_make:.1f} s (num_envs {n}, replicate_physics {cfg.scene.replicate_physics})",
        flush=True,
    )

    sizes = mdp.get_rope_sizes(u)
    if sizes is not None:
        mdp.check_rope_sizes(u, None)
        group = sizes.index
        names = list(sizes.names)
    else:
        group = torch.zeros(n, dtype=torch.long, device=dev)
        names = ["knot (UsdFileCfg)"]

    tm = u.termination_manager
    div_cfg = tm._term_cfgs[tm._term_names.index("diverged")]
    raw_div = div_cfg.func
    div_now = {}

    def div_capture(*a, **k):
        out = raw_div(*a, **k)
        div_now["v"] = out.clone()
        return out

    div_cfg.func = div_capture
    g = torch.Generator(device=dev).manual_seed(args_cli.seed)
    adim = u.action_manager.total_action_dim
    div_count = torch.zeros(len(names), device=dev)
    move_count = torch.zeros(len(names), device=dev)

    def move():
        a = (0.5 * torch.randn(n, adim, device=dev, generator=g)).clamp(-1, 1)
        env.step(a)
        d = div_now.pop("v").float()
        div_count.index_add_(0, group, d)
        move_count.index_add_(0, group, torch.ones_like(d))

    with torch.inference_mode():
        t0 = time.perf_counter()
        env.reset()
        t_reset = time.perf_counter() - t0
        t0 = time.perf_counter()
        move()
        torch.cuda.synchronize()
        t_first = time.perf_counter() - t0
        t0 = time.perf_counter()
        for _ in range(args_cli.moves):
            move()
        torch.cuda.synchronize()
        sec = (time.perf_counter() - t0) / max(args_cli.moves, 1)
    print(
        f"[size_check] reset {t_reset:.1f} s, first move {t_first:.1f} s, {sec:.2f} s/move ({n / sec:.1f} moves/s)",
        flush=True,
    )
    per_size = {
        name: {"moves": int(move_count[i]), "diverged_per_move": float(div_count[i] / move_count[i].clamp_min(1))}
        for i, name in enumerate(names)
    }
    for name, v in per_size.items():
        print(f"[size_check] {name}: diverged/move {v['diverged_per_move']:.4f} ({v['moves']} moves)", flush=True)
    res = {
        "task": args_cli.task,
        "sizes": args_cli.sizes,
        "num_envs": n,
        "replicate_physics": bool(cfg.scene.replicate_physics),
        "app_launch_sec": t_make0 - t_launch,
        "gym_make_sec": t_make,
        "reset_sec": t_reset,
        "first_move_sec": t_first,
        "sec_per_move": sec,
        "moves_per_sec": n / sec,
        "moves_timed": args_cli.moves,
        "per_size": per_size,
        "gpu": _gpu_mem(),
    }
    out = os.path.join(_REPO_ROOT, "outputs", "rope_size_spawn_check", args_cli.label or f"{args_cli.sizes}_n{n}")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(f"[size_check] wrote {out}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
