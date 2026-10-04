"""紐モデルの比較の評価: chain 評価 (まっすぐな紐から 18 手で固結び) を、学習とは別のモデル・摂動の下で回す。

    # 学習したモデルで評価 (= scripts/rsl_rl/eval_chain.py と同じ数え方)
    uv run python -u src/isaac_twisted_rl/tasks/knot_tying/rope_study/eval_robust.py \
        --task Isaac-Twisted-RL-RS-XpbdC28-Play-v4 --checkpoint logs/rsl_rl/knot_tying_rs_xpbdc28/<run>/model_100.pt \
        --num_envs 512 --headless --seed 7 --out result.json

    # 別のモデルで評価 (モデル間の転移)。観測・行動は仮想リンク 28 本で揃えてあるので、どの方策も
    # どのモデルの Play 環境でも動く
    ... --task Isaac-Twisted-RL-RS-Art28-Play-v4 --checkpoint <XpbdC28 の方策>

    # 摂動: 観測ノイズ (紐の位置、標準偏差 [m])、実行誤差 (掴む点・置く点の xy [m])、物理パラメータ (Hydra)
    ... --obs_noise 0.005 env.actions.pick_place.exec_noise_xy=0.005

    # XPBD の物理パラメータ (`xpbd_rod.XpbdRodParams` の項) を直接上書き (例: 曲げ剛性 x0.5)。
    # `xpbd_params` は dict なので、モデルの設定に無い項を Hydra で足すと `+` が要る。こちらは常に効く
    ... --xpbd bend_stiffness=3.927e-4 --xpbd twist_stiffness=1.9635e-4

数え方は `scripts/rsl_rl/eval_chain.py` と同じ: **各環境の最初の 1 エピソードだけ** を数える
(まっすぐな紐から、固結び・発散・18 手のどれかで終わる)。決定論的な方策 (平均の行動)。

観測ノイズは方策に渡す観測の **紐の位置 (先頭 84 次元 = 仮想リンク 28 x 3)** だけに足す。位相の特徴
(交差の表・段) は真値のまま (位相は画像から別に推定できると仮定)。報酬・終了の判定は真値で行う。
"""

import argparse
import sys

from isaaclab.app import AppLauncher

from isaac_twisted_rl.scripts.rsl_rl import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Chain evaluation under model mismatch and perturbations.")
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--steps", type=int, default=19)
parser.add_argument("--stochastic", action="store_true", default=False)
parser.add_argument("--obs_noise", type=float, default=0.0, help="紐の位置の観測ノイズの標準偏差 [m]。")
parser.add_argument(
    "--xpbd", action="append", default=[], help="XPBD の物理パラメータの上書き key=value (`XpbdRodParams` の項)。"
)
parser.add_argument("--out", type=str, default=None)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import isaaclab_tasks  # noqa: E402, F401
import isaac_twisted_rl.tasks  # noqa: E402, F401
from isaaclab.envs import ManagerBasedRLEnvCfg  # noqa: E402
from isaaclab.utils.assets import retrieve_file_path  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

_STAGE_DIMS = 4
_POS_DIMS = 84  # 仮想リンク 28 x 3 (観測の先頭。v2 の PolicyCfg の並び)


def _divergence_breakdown(div, counted, end_maxd, length) -> dict:
    """発散したエピソードの内訳: 直前の手までに紐が env 原点から 2 m 以上ずれていた割合 (= 徐々にずれて 3 m の
    判定に掛かった「漂流」) と、発散までの手数の平均。"""
    if div is None:
        return {}
    d = div & counted
    m = int(d.sum())
    if m == 0:
        return {"diverged_drift_frac": None, "diverged_mean_actions": None}
    return {
        "diverged_drift_frac": float((end_maxd[d] > 2.0).float().mean()),
        "diverged_mean_actions": float(length[d].float().mean()),
    }


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    resume_path = retrieve_file_path(args_cli.checkpoint)
    env_cfg.log_dir = os.path.dirname(resume_path)
    xpbd_override = {}
    for item in args_cli.xpbd:
        key, val = item.split("=", 1)
        xpbd_override[key] = float(val)
    if xpbd_override:
        if env_cfg.rope_backend.kind != "xpbd":
            raise ValueError(f"--xpbd は XPBD のモデルだけ ({args_cli.task} は {env_cfg.rope_backend.kind})")
        env_cfg.rope_backend.xpbd_params = {**dict(env_cfg.rope_backend.xpbd_params), **xpbd_override}

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    if args_cli.stochastic:
        pol = runner.alg.policy
        pol.eval()
        policy = lambda o: pol.act(o)  # noqa: E731
    else:
        policy = runner.get_inference_policy(device=env.unwrapped.device)

    u = env.unwrapped
    tm = u.termination_manager
    n, dev = u.num_envs, u.device
    names = list(tm.active_terms)
    gen = torch.Generator(device=dev).manual_seed(args_cli.seed + 1000)
    counted = torch.zeros(n, dtype=torch.bool, device=dev)
    outcome = {k: torch.zeros(n, dtype=torch.bool, device=dev) for k in names}
    max_stage = torch.zeros(n, dtype=torch.long, device=dev)
    ep_max_stage = torch.zeros(n, dtype=torch.long, device=dev)
    length = torch.zeros(n, dtype=torch.long, device=dev)
    ep_len = torch.zeros(n, dtype=torch.long, device=dev)
    # 発散の内訳を見るため、エピソード中の「紐のリンクの env 原点からの水平距離の最大」を手ごとに記録する
    # (発散した手の後は reset 済みなので、その直前の手までの最大。徐々にずれて 3 m を超えたのか、急に飛んだのか)
    rope = u.scene["rope"]
    origins_xy = u.scene.env_origins[:, :2].unsqueeze(1)
    ep_maxd = torch.zeros(n, device=dev)
    end_maxd = torch.zeros(n, device=dev)
    t0 = time.time()
    obs = env.get_observations()
    with torch.inference_mode():
        for _ in range(args_cli.steps):
            pobs = obs["policy"] if not torch.is_tensor(obs) else obs
            stage = pobs[:, -_STAGE_DIMS:].argmax(dim=-1)
            ep_max_stage = torch.maximum(ep_max_stage, stage)
            if args_cli.obs_noise > 0.0:
                noise = torch.randn(n, _POS_DIMS, device=dev, generator=gen) * args_cli.obs_noise
                if torch.is_tensor(obs):
                    obs = obs.clone()
                    obs[:, :_POS_DIMS] += noise
                else:
                    obs = obs.clone() if hasattr(obs, "clone") else obs
                    p = obs["policy"].clone()
                    p[:, :_POS_DIMS] += noise
                    obs["policy"] = p
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            ep_len += 1
            done = dones.bool() & ~counted
            end_maxd = torch.where(done, ep_maxd, end_maxd)
            dist = torch.nan_to_num((rope.data.body_pos_w[..., :2] - origins_xy).norm(dim=-1).amax(dim=-1), nan=1e9)
            ep_maxd = torch.where(dones.bool(), torch.zeros_like(ep_maxd), torch.maximum(ep_maxd, dist))
            for k in names:
                outcome[k] |= done & tm.get_term(k)
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
    p = float(knot.sum()) / max(m, 1)
    summary = {
        "task": args_cli.task,
        "checkpoint": resume_path,
        "obs_noise": args_cli.obs_noise,
        "xpbd_override": xpbd_override,
        "hydra": hydra_args,
        "stochastic": args_cli.stochastic,
        "seed": args_cli.seed,
        "episodes": m,
        "uncounted": n - m,
        "knot": p,
        "knot_se": (p * (1 - p) / max(m, 1)) ** 0.5,
        **{k: float((outcome[k] & counted).sum()) / max(m, 1) for k in names if k != "stage_up"},
        **{f"max_stage_ge{s}": float(((max_stage >= s) & counted).sum()) / max(m, 1) for s in (1, 2, 3)},
        "mean_actions_to_knot": float(length[knot].float().mean()) if bool(knot.any()) else None,
        **_divergence_breakdown(outcome.get("diverged"), counted, end_maxd, length),
        "wall_sec": time.time() - t0,
    }
    print("\n=== robust chain evaluation ===")
    for k, v in summary.items():
        print(f"{k}: {v}")
    if args_cli.out:
        os.makedirs(os.path.dirname(os.path.abspath(args_cli.out)), exist_ok=True)
        with open(args_cli.out, "w") as f:
            json.dump(summary, f, indent=2)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
