"""紐モデルの比較のスモークテスト: 環境を立ち上げ、乱数の手を数手打って観測の形・終了の内訳・発散を表示する。

    .venv/bin/python -u src/isaac_twisted_rl/tasks/knot_tying/rope_study/smoke_test.py \
        --task Isaac-Twisted-RL-RS-XpbdC28-v4 --num_envs 16 --steps 3 --headless --device cuda:0
"""

import argparse
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=3)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaac_twisted_rl.tasks  # noqa: E402, F401
from isaac_twisted_rl.tasks.knot_tying.common.mdp.observations import get_extractor  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


def main():
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    env = gym.make(args.task, cfg=cfg)
    u = env.unwrapped
    obs, _ = env.reset()
    rope = u.scene["rope"]
    print("obs", {k: v.shape for k, v in obs.items()}, "decimation", u.cfg.decimation, "rope", type(rope).__name__)
    print("body_pos_w", rope.data.body_pos_w.shape, "num_joints", rope.num_joints)
    for k in range(args.steps):
        a = (0.5 * torch.randn(u.num_envs, 6, device=u.device)).clamp(-1, 1)
        t0 = time.time()
        obs, rew, term, trunc, info = env.step(a)
        torch.cuda.synchronize()
        ex = get_extractor(u)
        z = rope.data.body_pos_w[..., 2] - u.scene.env_origins[:, 2:3]
        print(
            f"step {k}: {time.time() - t0:.1f}s rew {rew.mean().item():+.3f} term {int(term.sum())} trunc {int(trunc.sum())} "
            f"cross {ex.count.float().mean().item():.2f} zmax {z.max(dim=1).values.mean().item():.3f} "
            f"finite {torch.isfinite(rope.data.body_pos_w).all().item()}"
        )
        for key, v in info.get("log", {}).items():
            if "Termination" in key:
                print("   ", key, float(v))
    env.close()


if __name__ == "__main__":
    main()
    app.close()
