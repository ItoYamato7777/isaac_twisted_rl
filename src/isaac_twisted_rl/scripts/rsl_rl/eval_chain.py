# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""v4 以降の chain モード評価: チェックポイントを Play 環境で走らせ、エピソード単位で集計する。

以前の比較 (`eval_compare.py`。このファイルはリポジトリに残っていない) は step ごとの `extras["log"]` を平均するので、早く終わるエピソードほど
重く数えられる。ここでは **各環境の最初の 1 エピソードだけ** を数える (まっすぐなロープから、
固結び・発散・18 手のどれかで終わる)。`--steps` は Play の手数の上限 (18) 以上にすること。

    uv run python -u src/isaac_twisted_rl/scripts/rsl_rl/eval_chain.py --task Isaac-Twisted-RL-Knot-Play-v4 \
        --num_envs 1024 --headless --checkpoint logs/rsl_rl/knot_tying_v4/<run>/model_300.pt --out result.json

出力 (JSON と標準出力):
* `knot`: 固結びに達した割合 (`stage_up`。Play では固結びでだけ立つ)
* `knot_strict`: 固結びに達し、その形が正規の経路の終点 (`topology.STRICT_GOAL_P_DATA`、厳密な段 3) だった割合。
  段の定義によらず同じ 2 つの p-data で数える (v10 の段 3 は v7 と同じ判定で、こちらは別の列。
  `docs/decisions/0023-v10-path-stages.md`)。手の後の p-data は `stages.last_stage_transition` (reset の前の値) から読む
* `diverged` / `time_out`: 発散 / 18 手で終わった割合
* `max_stage_ge{k}`: エピソード中に段 k 以上に達した割合 (観測の段の one-hot から)
* `mean_actions_to_knot`: 固結びに達したエピソードの手数の平均

上のキーと既定の挙動は変えない (`summarize_search.py` と過去の `eval_chain_it*.json` が前提にしている)。
足したキー (集計は `common/chain_stats.py`・`common/cfg_diff.py`。単体テストあり):

* 段ごとの内訳: `knot_ci95` (Wilson 95%)、`p_ge2_given_ge1`・`p_knot_given_ge2`、`outcome_by_last_stage`
  (結末 x 最後の手を打ったときの段。和が 1)、`stage_reach` (段 k 以上に着いた割合と手数)、`moves_at_stage`
  (段ごとに使った手数)。崩れ方 (段 0 の時間切れなど) を `diag_stage_moves.py` を別に回さずに見るため
* `action_saturation`: 段 x チャネルごとに、クリップ前の行動が [−1, 1] の外に出た手の割合と BoundedPPO の罰則。
  評価の rollout そのものの手を数える (`action_saturation.py` は v12 の rollout の観測を使い回す)
* `meta`: seed・Hydra の上書き・`num_envs`・`steps`・commit・未コミットの変更・日時。評価のコマンドを
  JSON だけから組み直せるようにするため
* `param_check`: チェックポイントの run の `params/env.yaml` と評価の環境の設定の違い (Play が意図して変える
  項を除く)。違いがあれば評価の前に止める (`--allow_param_diff` で続ける)。学習の Hydra の上書きを評価に
  付け忘れると、違う物理のまま、エラーも出ずに測ってしまうため
"""

import argparse
import sys

from isaaclab.app import AppLauncher

from isaac_twisted_rl.scripts.rsl_rl import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Evaluate a checkpoint per episode in chain mode.")
parser.add_argument("--num_envs", type=int, default=1024, help="Number of environments (= episodes counted).")
parser.add_argument("--task", type=str, default=None, help="Name of the task (use the -Play- variant).")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--steps", type=int, default=19, help="env.step() calls (>= the Play episode length).")
parser.add_argument("--stochastic", action="store_true", default=False, help="Sample actions with the policy std.")
parser.add_argument("--out", type=str, default=None, help="Write the summary as JSON here.")
parser.add_argument(
    "--allow_param_diff",
    action="store_true",
    default=False,
    help="Continue even if the env cfg differs from the run's params/env.yaml beyond the Play changes.",
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
_ARGV = list(sys.argv)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import datetime
import json
import os
import subprocess
import tempfile

import gymnasium as gym
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
import isaac_twisted_rl.tasks  # noqa: F401
from isaac_twisted_rl.tasks.knot_tying.common import topology as tp
from isaac_twisted_rl.tasks.knot_tying.common.cfg_diff import load_cfg_yaml, unexpected_diff
from isaac_twisted_rl.tasks.knot_tying.common.chain_stats import action_saturation, chain_breakdown
from isaac_twisted_rl.tasks.knot_tying.common.mdp.stages import last_stage_transition
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

_STAGE_DIMS = 4
"""観測の末尾にある段の one-hot の次元 (段 0〜3)。"""


def _now() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def _git_info() -> dict:
    """このスクリプトのある checkout の commit と、未コミットの変更 (未追跡を含む)。"""
    cwd = os.path.dirname(os.path.abspath(__file__))

    def _run(*cmd: str) -> str | None:
        try:
            return subprocess.run(["git", *cmd], cwd=cwd, capture_output=True, text=True, check=True).stdout
        except (OSError, subprocess.CalledProcessError):
            return None

    head, status = _run("rev-parse", "HEAD"), _run("status", "--porcelain")
    lines = status.splitlines() if status is not None else None
    return {
        "commit": head.strip() if head else None,
        "git_dirty": bool(lines) if lines is not None else None,
        "git_status": lines,
    }


def _cfg_as_yaml(cfg) -> dict:
    """設定を学習の `params/env.yaml` と同じ `dump_yaml` で書き、読み直す (表し方を揃えて比べるため)。"""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "env.yaml")
        dump_yaml(path, cfg)
        with open(path) as f:
            return load_cfg_yaml(f.read())


def _check_params(env_cfg: ManagerBasedRLEnvCfg, resume_path: str) -> dict:
    """run の `params/env.yaml` と評価の環境の設定の、Play の意図した変更以外の違い (`cfg_diff.unexpected_diff`)。

    Play の意図した変更は、同じ版の学習用のタスク (ID から `-Play` を除いたもの) と Play 用のタスクの設定を
    既定のまま作った差で決める。どちらかが無ければ照合を飛ばし、理由を残す。
    """
    train_yaml = os.path.join(os.path.dirname(resume_path), "params", "env.yaml")
    out: dict = {"train_env_yaml": train_yaml}
    train_task = args_cli.task.replace("-Play", "")
    if not os.path.isfile(train_yaml):
        return out | {"skipped": "params/env.yaml が無い"}
    if train_task == args_cli.task or train_task not in gym.registry:
        return out | {"skipped": f"学習用のタスク {train_task} が登録されていない"}
    with open(train_yaml) as f:
        train = load_cfg_yaml(f.read())
    train_default = _cfg_as_yaml(load_cfg_from_registry(train_task, "env_cfg_entry_point"))
    play_default = _cfg_as_yaml(load_cfg_from_registry(args_cli.task, "env_cfg_entry_point"))
    diff = unexpected_diff(train, _cfg_as_yaml(env_cfg), train_default, play_default)
    return out | {"train_task": train_task, **diff}


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    resume_path = retrieve_file_path(args_cli.checkpoint)
    env_cfg.log_dir = os.path.dirname(resume_path)
    started = _now()

    param_check = _check_params(env_cfg, resume_path)
    found = {k: param_check[k] for k in ("changed", "only_in_train", "only_in_eval") if param_check.get(k)}
    param_check["allowed"] = bool(found) and args_cli.allow_param_diff
    if "skipped" in param_check:
        print(f"[WARN]: 学習の設定との照合を飛ばした: {param_check['skipped']}")
    elif found:
        print("[WARN]: 評価の環境の設定が、学習の params/env.yaml と Play の意図した変更以外で違う:")
        print(json.dumps(found, indent=2, ensure_ascii=False, default=str))
        if not args_cli.allow_param_diff:
            raise RuntimeError(
                "学習の Hydra の上書きを評価に付け忘れていないかを確かめる。意図した違いなら --allow_param_diff を付ける"
            )

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
    tm = unwrapped.termination_manager
    n = unwrapped.num_envs
    dev = unwrapped.device
    names = list(tm.active_terms)

    counted = torch.zeros(n, dtype=torch.bool, device=dev)
    outcome = {k: torch.zeros(n, dtype=torch.bool, device=dev) for k in names}
    strict = torch.zeros(n, dtype=torch.bool, device=dev)
    max_stage = torch.zeros(n, dtype=torch.long, device=dev)
    ep_max_stage = torch.zeros(n, dtype=torch.long, device=dev)
    length = torch.zeros(n, dtype=torch.long, device=dev)
    ep_len = torch.zeros(n, dtype=torch.long, device=dev)
    # 段ごとの内訳と張り付き (`chain_stats`) のための毎手の記録。既存のキーの計算には使わない
    rec_stage: list[torch.Tensor] = []
    rec_active: list[torch.Tensor] = []
    rec_actions: list[torch.Tensor] = []

    obs = env.get_observations()
    with torch.inference_mode():
        for _ in range(args_cli.steps):
            pobs = obs["policy"] if not torch.is_tensor(obs) else obs
            stage = pobs[:, -_STAGE_DIMS:].argmax(dim=-1)
            ep_max_stage = torch.maximum(ep_max_stage, stage)
            actions = policy(obs)
            rec_stage.append(stage.clone())
            rec_active.append(~counted)
            rec_actions.append(actions.clone())
            obs, _, dones, _ = env.step(actions)
            ep_len += 1
            done = dones.bool() & ~counted
            for k in names:
                outcome[k] |= done & tm.get_term(k)
            transition = last_stage_transition(unwrapped)
            if transition is None:
                raise RuntimeError("stage_transition のキャッシュが無い (段の終了項 stage_up が配線されていない?)")
            strict |= done & tm.get_term("stage_up") & tp.strict_goal_mask(transition["key_after"])
            # 固結び (stage_up) で終わった手は観測が次のエピソードに替わっているので 3 を入れる。
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
    summary = {
        "task": args_cli.task,
        "checkpoint": resume_path,
        "stochastic": args_cli.stochastic,
        "episodes": m,
        "uncounted": n - m,
        "knot": float(knot.sum()) / max(m, 1),
        "knot_strict": float((strict & counted).sum()) / max(m, 1),
        **{k: float((outcome[k] & counted).sum()) / max(m, 1) for k in names if k != "stage_up"},
        **{f"max_stage_ge{s}": float(((max_stage >= s) & counted).sum()) / max(m, 1) for s in (1, 2, 3)},
        "mean_actions_to_knot": float(length[knot].float().mean()) if bool(knot.any()) else None,
    }
    stage_t, active_t, actions_t = (torch.stack(r).cpu() for r in (rec_stage, rec_active, rec_actions))
    summary |= chain_breakdown(
        stage_t,
        active_t,
        counted.cpu(),
        {k: (outcome[k] & counted).cpu() for k in names},
        max_stage.cpu(),
        length.cpu(),
        num_stages=_STAGE_DIMS,
    )
    summary["action_saturation"] = action_saturation(stage_t, active_t, actions_t, num_stages=_STAGE_DIMS)
    summary["meta"] = {
        "argv": _ARGV,
        "seed": args_cli.seed,
        "env_seed": env_cfg.seed,
        "hydra_overrides": hydra_args,
        "num_envs": n,
        "steps": args_cli.steps,
        "device": str(dev),
        **_git_info(),
        "started": started,
        "finished": _now(),
    }
    summary["param_check"] = param_check
    print("\n=== chain evaluation ===")
    for k, v in summary.items():
        print(f"{k}: {json.dumps(v, ensure_ascii=False, default=str) if isinstance(v, (dict, list)) else v}")
    if args_cli.out:
        with open(args_cli.out, "w") as f:
            json.dump(summary, f, indent=2)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
