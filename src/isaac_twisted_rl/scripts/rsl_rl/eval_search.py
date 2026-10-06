# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""推論時探索の評価: 学習済みチェックポイントに「探索ありの成功率」を付ける (v4 以降の段ごとの方策)。

chain 評価 (`eval_chain.py`) は方策を 1 回展開しただけの値で、論文 (シミュレータで候補を試す探索、最大
30 分) と土俵が違う (`paper/Report_Paper.md` §5.2)。同じチェックポイントについて、同じ手数の予算で
次の 3 つを並べるためのスクリプト。起動方法と引数は `eval_chain.py` と同じ (Play 環境、Hydra の上書きは
そのまま後ろに付ける)。

| `--mode` | 何を測るか |
|---|---|
| `clone_test` | 状態の複製の検証。複製直後の観測・段が複製元と一致する割合と、同じ手を打ったときの結果の一致 |
| `rollouts` | (a) 確率的な方策の chain 評価 (各環境の最初の 1 エピソード) と、(b) best-of-N (同じ手数の予算で確率的な展開を独立に繰り返す) の成功率の曲線 |
| `search` | (c) ビーム探索 + 到達木からの再開 (`common/beam_search.py`) の成功率の曲線・解の手数・捨てた候補の割合 |
| `shapes` | 段のアーカイブの形ごとに、同じ初期状態・同じ方策で R 回試した固結びの割合の分布 (段 2 の失敗が形で決まるか、試行ごとのばらつきか) |

    uv run python -u src/isaac_twisted_rl/scripts/rsl_rl/eval_search.py --task Isaac-Twisted-RL-Knot-Play-v6 \
        --mode search --num_envs 512 --headless --device cuda:0 --seed 7 \
        --checkpoint logs/rsl_rl/knot_tying_v6/<run>/model_100.pt agent.device=cuda:0

結果は既定でチェックポイントのフォルダの `eval_search_{mode}_it{N}[_{tag}].json`。アルゴリズムと結果の読み方は
`scripts/rsl_rl/README.md`。

## Play 環境の自動 reset との付き合い方

Play 環境は固結び・発散・18 手で自動 reset される。`clone_test` / `search` では:

* **18 手の時間切れは切る** (`episode_length_s` を十分大きくする)。探索のノードは何手目でも続けられる。
* **固結び・発散の reset はそのまま使う。** どちらも探索ではその候補の終わり (成功 / 捨てる) なので、
  reset された後の状態は使わない。結果は reset の前に立った終了項 (`reset_buf` と `get_term`) から読む。
* 毎ラウンド、使う環境すべてにノードの状態を書き込み直す (`mdp/state_clone.py`)。前のラウンドの
  状態や reset の後の状態は残らない。

`rollouts` と `shapes` は Play 環境のまま (18 手で時間切れ) 回す。
"""

import argparse
import sys

from isaaclab.app import AppLauncher

from isaac_twisted_rl.scripts.rsl_rl import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Evaluate a checkpoint with inference-time search.")
parser.add_argument("--num_envs", type=int, default=512, help="Number of environments.")
parser.add_argument("--task", type=str, default=None, help="Name of the task (use the -Play- variant).")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment and the search.")
parser.add_argument("--mode", choices=("clone_test", "rollouts", "search", "shapes"), default="search")
parser.add_argument("--out", type=str, default=None, help="Output JSON (default: next to the checkpoint).")
parser.add_argument("--tag", type=str, default=None, help="Suffix of the default output name.")
parser.add_argument("--budgets", type=str, default="32,64,128,256,512,1024,2048", help="Move budgets for curves.")
# rollouts
parser.add_argument("--episodes_per_env", type=int, default=2, help="rollouts: episodes counted per env.")
parser.add_argument(
    "--deterministic", action="store_true", default=False, help="rollouts / shapes: use the mean action."
)
# search
parser.add_argument("--problems", type=int, default=64, help="search: number of problems (search seeds).")
parser.add_argument("--beam", type=int, default=4, help="search: frontier size M.")
parser.add_argument("--slots", type=int, default=32, help="search: envs per problem per round S (= M x K).")
parser.add_argument("--max_moves", type=int, default=2048, help="search: simulated-move budget per problem.")
parser.add_argument("--max_minutes", type=float, default=0.0, help="search: wall-clock cut-off for the run (0 = none).")
parser.add_argument("--max_same", type=int, default=6, help="search: same-stage moves allowed along a line.")
parser.add_argument("--restart_p", type=float, default=0.05, help="search: uniform restart probability p.")
parser.add_argument("--no_mean_first", action="store_true", default=False, help="search: sample every candidate.")
parser.add_argument(
    "--rank",
    choices=("value", "random"),
    default="value",
    help="search: order within a stage by the critic V(s) or uniformly at random (to check the critic helps).",
)
parser.add_argument("--std_scale", type=float, default=1.0, help="search/rollouts: scale of the policy std.")
# shapes
parser.add_argument(
    "--shape_archive", type=str, default=None, help="shapes: archive file (default: archive.pt next to the checkpoint)."
)
parser.add_argument("--shape_stage", type=int, default=2, help="shapes: archive stage the shapes are taken from.")
parser.add_argument("--shapes", type=int, default=128, help="shapes: number of shapes (0 = all in the stage).")
parser.add_argument("--shape_reps", type=int, default=64, help="shapes: tries per shape (envs per shape).")
parser.add_argument("--shape_moves", type=int, default=6, help="shapes: moves per try (6 = one training transition).")
# clone_test
parser.add_argument("--replicas", type=int, default=8, help="clone_test: envs per source (source included).")
parser.add_argument("--warmup", type=int, default=3, help="clone_test: stochastic moves before each check.")
parser.add_argument("--checks", type=int, default=3, help="clone_test: number of clone-and-step checks.")
parser.add_argument("--clone_skip", type=str, default="", help="clone_test: parts not copied (state_clone.PARTS).")
parser.add_argument("--obs_tol", type=float, default=1e-4, help="Max abs obs difference counted as identical.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import json
import math
import os
import re
import statistics
import time

import gymnasium as gym
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
import isaac_twisted_rl.tasks  # noqa: F401
from isaac_twisted_rl.tasks.knot_tying.common import beam_search as bs
from isaac_twisted_rl.tasks.knot_tying.common import state_archive as sa
from isaac_twisted_rl.tasks.knot_tying.common import topology as tp
from isaac_twisted_rl.tasks.knot_tying.common.mdp import rope_zmax
from isaac_twisted_rl.tasks.knot_tying.common.mdp.observations import get_extractor, invalidate_extractor
from isaac_twisted_rl.tasks.knot_tying.common.mdp.state_clone import PARTS, EnvStateCloner
from isaaclab_tasks.utils.hydra import hydra_task_config

_STAGE_DIMS = 4
"""観測の末尾にある段の one-hot の次元 (段 0〜3)。"""

_NO_TIME_OUT_S = 1.0e7
"""`clone_test` / `search` の `episode_length_s`。18 手の時間切れが立たない長さ (約 3 x 10^6 手)。"""


class Policy:
    """`StageGatedActorCritic` から、平均・std 付きのサンプル・V(s) を取り出す。

    V(s) は観測の段の critic の値 (`GatedMLP` が段の one-hot で振り分ける)。段が違う値どうしは比べないこと
    (`common/beam_search.select_frontier`)。
    """

    def __init__(self, runner: OnPolicyRunner, std_scale: float):
        self.nn = runner.alg.policy
        self.nn.eval()
        self.std_scale = std_scale

    @torch.inference_mode()
    def act(self, obs, mean_mask: torch.Tensor | None = None) -> torch.Tensor:
        self.nn.update_distribution(self.nn.get_actor_obs(obs))
        dist = self.nn.distribution
        sample = dist.mean + self.std_scale * dist.stddev * torch.randn_like(dist.mean)
        if mean_mask is None:
            return sample
        return torch.where(mean_mask.unsqueeze(-1), dist.mean, sample)

    @torch.inference_mode()
    def mean(self, obs) -> torch.Tensor:
        self.nn.update_distribution(self.nn.get_actor_obs(obs))
        return self.nn.distribution.mean

    @torch.inference_mode()
    def value(self, obs) -> torch.Tensor:
        return self.nn.evaluate(obs).squeeze(-1)


def obs_stage(obs) -> torch.Tensor:
    return obs["policy"][:, -_STAGE_DIMS:].argmax(dim=-1)


def obs_term_slices(unwrapped) -> list[tuple[str, int, int]]:
    """観測グループ `policy` の項名と列の範囲。複製の検証で、どの項がずれたかを出すため。"""
    om = unwrapped.observation_manager
    out, start = [], 0
    for name, dims in zip(om.active_terms["policy"], om.group_obs_term_dim["policy"]):
        width = int(torch.tensor(dims).prod())
        out.append((name, start, start + width))
        start += width
    return out


class CloneCheck:
    """複製直後の観測を複製元 (ノードに保存した観測) と比べて数える。"""

    def __init__(self, slices: list[tuple[str, int, int]], tol: float):
        self.slices = slices
        self.tol = tol
        self.n = 0
        self.obs_mismatch = 0
        self.stage_mismatch = 0
        self.max_diff = 0.0
        self.term_max = {name: 0.0 for name, _, _ in slices}
        self.term_mismatch = {name: 0 for name, _, _ in slices}

    def update(self, restored: torch.Tensor, reference: torch.Tensor) -> tuple[int, int]:
        if restored.numel() == 0:
            return 0, 0
        diff = (restored - reference).abs()
        row = diff.amax(dim=-1)
        bad = int((row > self.tol).sum())
        st = int((restored[:, -_STAGE_DIMS:].argmax(-1) != reference[:, -_STAGE_DIMS:].argmax(-1)).sum())
        self.n += int(restored.shape[0])
        self.obs_mismatch += bad
        self.stage_mismatch += st
        self.max_diff = max(self.max_diff, float(row.max()))
        for name, a, b in self.slices:
            d = diff[:, a:b].amax(dim=-1)
            self.term_max[name] = max(self.term_max[name], float(d.max()))
            self.term_mismatch[name] += int((d > self.tol).sum())
        return bad, st

    def summary(self) -> dict:
        n = max(self.n, 1)
        return {
            "restored": self.n,
            "obs_mismatch_frac": self.obs_mismatch / n,
            "stage_mismatch_frac": self.stage_mismatch / n,
            "max_abs_diff": self.max_diff,
            "tol": self.tol,
            "term_max_abs_diff": self.term_max,
            "term_mismatch_frac": {k: v / n for k, v in self.term_mismatch.items()},
        }


def step_outcome(env, unwrapped, obs_after, settled_h: float, diameter: float):
    """`env.step()` の直後に、各環境の (固結び, 発散, 床に落ちている, 手の後の段) を読む。

    固結び・発散の環境はこの手の中で reset されているので、立った終了項から読む (`reset_buf` と
    `get_term`。`get_term` は前のエピソードの値も残すので `reset_buf` で絞る)。床に落ちているかは
    reset されなかった環境についてだけ意味がある。
    """
    tm = unwrapped.termination_manager
    reset = unwrapped.reset_buf.bool()
    knot = reset & tm.get_term("stage_up")
    diverged = reset & tm.get_term("diverged")
    settled = rope_zmax(unwrapped) <= settled_h * diameter
    return knot, diverged, settled, obs_stage(obs_after)


def default_out(resume_path: str, mode: str, tag: str | None) -> str:
    m = re.search(r"model_(\d+)\.pt$", resume_path)
    it = f"_it{m.group(1)}" if m else ""
    suffix = f"_{tag}" if tag else ""
    return os.path.join(os.path.dirname(resume_path), f"eval_search_{mode}{it}{suffix}.json")


# ================================================================ clone_test


def _quantiles(x: list[float]) -> dict:
    """中央値・90 パーセンタイル・最大 (空なら None)。"""
    if not x:
        return {"median": None, "p90": None, "max": None}
    xs = sorted(x)
    return {"median": statistics.median(xs), "p90": xs[int(0.9 * (len(xs) - 1))], "max": xs[-1]}


def run_clone_test(env, unwrapped, policy: Policy, settle: tuple[float, float]) -> dict:
    """状態を R 個の環境に複製し、(1) 複製直後の観測・段の一致、(2) 同じ手を打った結果の一致を測る。

    環境を R 個ずつの区画に分け、区画の先頭 (複製元) の状態を残りの R−1 個へ写す。複製元は書き換えない
    (PhysX の内部状態を持ったまま)。その後、区画の全員に複製元の観測での平均の行動を打たせる。
    """
    n = unwrapped.num_envs
    r = args_cli.replicas
    skip = tuple(s for s in args_cli.clone_skip.split(",") if s)
    for s in skip:
        if s not in PARTS:
            raise ValueError(f"--clone_skip の {s!r} は {PARTS} のどれでもない")
    cloner = EnvStateCloner(unwrapped)
    check = CloneCheck(obs_term_slices(unwrapped), args_cli.obs_tol)
    blocks = n // r
    ids = torch.arange(blocks * r, device=unwrapped.device)
    src = (ids // r) * r
    dst_mask = ids != src
    dst, dst_src = ids[dst_mask], src[dst_mask]
    agree = {k: 0 for k in ("compared", "outcome_same", "stage_same", "both_alive", "knot_src", "knot_rep")}
    agree.update(div_src=0, div_rep=0, rep_compared=0, rep_outcome_same=0, rep_both_alive=0, rep_stage_same=0)
    rep_dev: list[float] = []
    node_dev: list[float] = []
    stage_hist = [0] * _STAGE_DIMS
    obs = env.get_observations()
    for c in range(args_cli.checks):
        for _ in range(args_cli.warmup):
            obs, _, _, _ = env.step(policy.act(obs))
        st = obs_stage(obs)
        for k in range(_STAGE_DIMS):
            stage_hist[k] += int((st[ids[::r]] == k).sum())
        ref_obs = obs["policy"].clone()
        state = cloner.capture(dst_src)
        cloner.restore(dst, state, skip=skip)
        obs = env.get_observations()
        bad, stm = check.update(obs["policy"][dst], ref_obs[dst_src])
        # 複製元の観測は計算し直しても変わらないはず
        src_drift = float((obs["policy"][ids[::r]] - ref_obs[ids[::r]]).abs().max())
        print(
            f"[clone_test] check {c}: obs mismatch {bad}/{dst.numel()}, stage mismatch {stm}, src drift {src_drift:.2e}"
        )

        # 区画の全員に複製元の観測での平均の行動を打たせる
        mean = policy.mean(obs)
        actions = mean.clone()
        actions[ids] = mean[src]
        obs, _, _, _ = env.step(actions)
        knot, div, _, stage_after = step_outcome(env, unwrapped, obs, *settle)
        # 結果の種類: 0 = 固結び、1 = 発散、2 = 続いている (このときだけ段も比べる)
        kind = torch.where(knot, 0, torch.where(div, 1, 2))
        k_src, k_dst = kind[dst_src], kind[dst]
        s_src, s_dst = stage_after[dst_src], stage_after[dst]
        alive_pair = (k_src == 2) & (k_dst == 2)
        agree["compared"] += int(dst.numel())
        agree["outcome_same"] += int(((k_src == k_dst) & ((k_src < 2) | (s_src == s_dst))).sum())
        agree["stage_same"] += int((alive_pair & (s_src == s_dst)).sum())
        agree["both_alive"] += int(alive_pair.sum())
        # 複製どうし (区画の 2 番目の環境 = 最初の複製と、残りの複製) も比べる。複製元だけが PhysX の内部状態
        # (接触のキャッシュ) を持っているので、複製どうしの一致が複製元との一致より高ければ、写せない状態が効いている。
        first = src + 1
        rr = ids[(ids != src) & (ids != first)]
        rf = ((rr // r) * r) + 1
        k_a, k_b = kind[rf], kind[rr]
        rep_alive = (k_a == 2) & (k_b == 2)
        agree["rep_compared"] += int(rr.numel())
        agree["rep_outcome_same"] += int(((k_a == k_b) & ((k_a < 2) | (stage_after[rf] == stage_after[rr]))).sum())
        agree["rep_both_alive"] += int(rep_alive.sum())
        agree["rep_stage_same"] += int((rep_alive & (stage_after[rf] == stage_after[rr])).sum())
        agree["knot_src"] += int(knot[dst_src].sum())
        agree["knot_rep"] += int(knot[dst].sum())
        agree["div_src"] += int(div[dst_src].sum())
        agree["div_rep"] += int(div[dst].sum())
        # reset されなかった組だけ、ロープの節点のずれ (複製元との最大距離) を測る
        alive = ~(unwrapped.reset_buf.bool()[dst] | unwrapped.reset_buf.bool()[dst_src])
        if bool(alive.any()):
            rope = unwrapped.scene["rope"]
            pos = rope.data.body_pos_w - unwrapped.scene.env_origins.unsqueeze(1)
            d = (pos[dst[alive]] - pos[dst_src[alive]]).norm(dim=-1).amax(dim=-1)
            node_dev.extend(d.tolist())
        alive = ~(unwrapped.reset_buf.bool()[rr] | unwrapped.reset_buf.bool()[rf])
        if bool(alive.any()):
            pos = unwrapped.scene["rope"].data.body_pos_w - unwrapped.scene.env_origins.unsqueeze(1)
            d = (pos[rr[alive]] - pos[rf[alive]]).norm(dim=-1).amax(dim=-1)
            rep_dev.extend(d.tolist())
    return {
        "clone_skip": list(skip),
        "replicas": r,
        "source_stage_hist": stage_hist,
        "clone_check": check.summary(),
        "same_action": {
            **agree,
            "outcome_agree_frac": agree["outcome_same"] / max(agree["compared"], 1),
            "stage_agree_frac_alive": agree["stage_same"] / max(agree["both_alive"], 1),
            "rope_node_dev_m": _quantiles(node_dev),
            "replica_outcome_agree_frac": agree["rep_outcome_same"] / max(agree["rep_compared"], 1),
            "replica_stage_agree_frac_alive": agree["rep_stage_same"] / max(agree["rep_both_alive"], 1),
            "replica_rope_node_dev_m": _quantiles(rep_dev),
        },
    }


# ================================================================ rollouts (a) + (b)


def run_rollouts(env, unwrapped, policy: Policy, budgets: list[int]) -> dict:
    """Play 環境のまま確率的な方策を回し、各環境の最初の E エピソードを集める。

    各エピソードは 18 手以内に終わるので、18 E + 1 手回せば全環境が E 本を終える。E 本目より後の
    エピソードは数えない (打ち切りで短いエピソードに偏らないように)。
    """
    n = unwrapped.num_envs
    dev = unwrapped.device
    tm = unwrapped.termination_manager
    names = list(tm.active_terms)
    max_len = int(unwrapped.max_episode_length)
    e_per = args_cli.episodes_per_env
    episodes: list[dict] = []
    count = torch.zeros(n, dtype=torch.long, device=dev)
    ep_len = torch.zeros(n, dtype=torch.long, device=dev)
    ep_max = torch.zeros(n, dtype=torch.long, device=dev)
    obs = env.get_observations()
    mean_mask = torch.ones(n, dtype=torch.bool, device=dev) if args_cli.deterministic else None
    for _ in range(max_len * e_per + 1):
        ep_max = torch.maximum(ep_max, obs_stage(obs))
        obs, _, dones, _ = env.step(policy.act(obs, mean_mask))
        ep_len += 1
        done = dones.bool()
        if bool(done.any()):
            rows = done.nonzero(as_tuple=False).squeeze(-1)
            which = {k: tm.get_term(k)[rows] for k in names}
            for j, env_id in enumerate(rows.tolist()):
                if int(count[env_id]) >= e_per:
                    continue
                outcome = next((k for k in ("stage_up", "diverged", "time_out") if bool(which[k][j])), "other")
                episodes.append(
                    {
                        "env": env_id,
                        "index": int(count[env_id]),
                        "length": int(ep_len[env_id]),
                        "outcome": "knot" if outcome == "stage_up" else outcome,
                        "max_stage": 3 if outcome == "stage_up" else int(ep_max[env_id]),
                    }
                )
            count[rows] += 1
            ep_len[rows] = 0
            ep_max[rows] = 0
        if bool((count >= e_per).all()):
            break

    first = [e for e in episodes if e["index"] == 0]
    m = max(len(first), 1)
    knot_first = [e for e in first if e["outcome"] == "knot"]
    lengths = [e["length"] for e in episodes]
    success = [e["outcome"] == "knot" for e in episodes]
    pool_n = max(len(episodes), 1)
    return {
        "chain": {
            "episodes": len(first),
            "knot": len(knot_first) / m,
            "diverged": sum(e["outcome"] == "diverged" for e in first) / m,
            "time_out": sum(e["outcome"] == "time_out" for e in first) / m,
            **{f"max_stage_ge{s}": sum(e["max_stage"] >= s for e in first) / m for s in (1, 2, 3)},
            "mean_actions_to_knot": (sum(e["length"] for e in knot_first) / len(knot_first)) if knot_first else None,
        },
        "pool": {
            "episodes": len(episodes),
            "knot": sum(success) / pool_n,
            "diverged": sum(e["outcome"] == "diverged" for e in episodes) / pool_n,
            "time_out": sum(e["outcome"] == "time_out" for e in episodes) / pool_n,
            "mean_length": sum(lengths) / pool_n,
            "moves_lost_to_divergence_frac": sum(e["length"] for e in episodes if e["outcome"] == "diverged")
            / max(sum(lengths), 1),
        },
        "best_of_n": {
            "budgets": budgets,
            "success": bs.best_of_n_curve(lengths, success, budgets),
            "expected_moves_to_knot": bs.expected_moves_best_of_n(lengths, success),
        },
        "episodes": episodes,
    }


# ================================================================ search (c)


def run_search(env, unwrapped, policy: Policy, settle: tuple[float, float], budgets: list[int]) -> dict:
    """ビーム探索 (`common/beam_search.BeamSearch`) を回す。1 ラウンド = 複製 -> 1 手 -> 分類 -> 記帳。"""
    n = unwrapped.num_envs
    dev = unwrapped.device
    s = args_cli.slots
    groups = n // s
    if groups < 1:
        raise ValueError(f"--num_envs ({n}) は --slots ({s}) 以上にする")
    cfg = bs.SearchConfig(
        num_groups=groups,
        slots_per_group=s,
        beam_width=args_cli.beam,
        max_moves=args_cli.max_moves,
        num_problems=args_cli.problems,
        max_same=args_cli.max_same,
        restart_random_prob=args_cli.restart_p,
        mean_first=not args_cli.no_mean_first,
    )
    cloner = EnvStateCloner(unwrapped)
    check = CloneCheck(obs_term_slices(unwrapped), args_cli.obs_tol)
    obs = env.get_observations()
    root = cloner.capture(torch.tensor([0], device=dev))
    root["obs"] = obs["policy"][:1].clone()
    gen = torch.Generator(device=dev)
    gen.manual_seed(int(args_cli.seed or 0))
    search = bs.BeamSearch(cfg, root, dev, generator=gen)
    # `--rank random` の順位付けは探索とは別の乱数列にする (他の乱数の引き方を V(s) の場合と揃えるため)。
    rank_gen = torch.Generator(device=dev)
    rank_gen.manual_seed(int(args_cli.seed or 0) + 1)
    print(
        f"[search] G={groups} problems in parallel x S={s} envs (M={cfg.beam_width}), {cfg.num_problems} problems, "
        f"budget {cfg.max_moves} moves/problem"
    )

    t0 = time.time()
    round_times: list[float] = []
    rnd = 0
    with torch.inference_mode():
        while not search.done:
            elapsed = time.time() - t0
            if args_cli.max_minutes > 0 and elapsed > 60.0 * args_cli.max_minutes:
                print(f"[search] wall-clock cut-off after {elapsed / 60:.1f} min")
                search.now = elapsed
                search.stop_all("wall_clock")
                break
            tr = time.time()
            plan = search.plan_round()
            slot_node = plan.slot_node.to(dev)
            ids = (slot_node >= 0).nonzero(as_tuple=False).squeeze(-1)
            state = search.store.gather(slot_node[ids])
            ref_obs = state.pop("obs")
            cloner.restore(ids, state)
            obs = env.get_observations()
            bad, stm = check.update(obs["policy"][ids], ref_obs)

            actions = policy.act(obs, plan.slot_mean.to(dev))
            obs, _, _, _ = env.step(actions)
            knot, div, settled, child_stage = step_outcome(env, unwrapped, obs, *settle)
            parent_stage = search.store.stage[slot_node.clamp_min(0)]
            outcome = bs.classify(parent_stage, child_stage, knot, div, settled)
            if args_cli.rank == "value":
                value = policy.value(obs)
            else:
                value = torch.rand(n, device=dev, generator=rank_gen)
            child_state = cloner.capture()
            child_state["obs"] = obs["policy"].clone()
            finished = search.commit_round(plan, outcome, child_stage, value, child_state, now=time.time() - t0)
            round_times.append(time.time() - tr)
            rnd += 1

            used = outcome[ids]
            counts = {name: int((used == k).sum()) for k, name in enumerate(bs.OUTCOME_NAMES)}
            solved = sum(p.solved for p in search.finished)
            print(
                f"[search] round {rnd} ({round_times[-1]:.1f} s): envs {ids.numel()}, {counts}, "
                f"clone mismatch obs {bad} / stage {stm}, finished {len(search.finished)} (solved {solved}), "
                f"active {len(search.active_problems())}"
            )
            for p in finished:
                tag = f"solved in {p.moves_at_solve} moves (depth {p.depth_at_solve})" if p.solved else p.stop_reason
                print(f"[search]   problem {p.problem_id}: {tag}, rounds {p.rounds}, restarts {p.restarts}")

    probs = sorted(search.finished, key=lambda p: p.problem_id)
    total_cands = sum(sum(p.outcomes.values()) for p in probs)
    sum_out = {k: sum(p.outcomes[k] for p in probs) for k in bs.OUTCOME_NAMES}
    at_solve = [float(p.moves_at_solve) if p.solved else math.inf for p in probs]
    solved = [p for p in probs if p.solved]
    sec_per_round = statistics.median(round_times) if round_times else None
    per_problem_rate = (s / sec_per_round) if sec_per_round else None  # 手/s (1 問題あたり)
    return {
        "config": {
            "num_groups": groups,
            "slots_per_group": s,
            "beam_width": cfg.beam_width,
            "max_moves": cfg.max_moves,
            "problems": cfg.num_problems,
            "max_same": cfg.max_same,
            "restart_random_prob": cfg.restart_random_prob,
            "mean_first": cfg.mean_first,
            "rank": args_cli.rank,
            "std_scale": args_cli.std_scale,
            "max_minutes": args_cli.max_minutes,
        },
        "summary": {
            "problems": len(probs),
            "solved": len(solved) / max(len(probs), 1),
            "budgets": budgets,
            "success": bs.success_curve(at_solve, budgets),
            "moves_at_solve_median": statistics.median([p.moves_at_solve for p in solved]) if solved else None,
            "moves_at_solve_mean": statistics.mean([p.moves_at_solve for p in solved]) if solved else None,
            "depth_at_solve_median": statistics.median([p.depth_at_solve for p in solved]) if solved else None,
            "depth_at_solve_mean": statistics.mean([p.depth_at_solve for p in solved]) if solved else None,
            "candidates": total_cands,
            "candidate_frac": {k: v / max(total_cands, 1) for k, v in sum_out.items()},
            "restarts_mean": statistics.mean([p.restarts for p in probs]) if probs else None,
            "stop_reasons": {r: sum(p.stop_reason == r for p in probs) for r in ("solved", "budget", "wall_clock")},
        },
        "throughput": {
            "rounds": rnd,
            "wall_seconds": time.time() - t0,
            "sec_per_round_median": sec_per_round,
            "moves_per_sec_total": (n / sec_per_round) if sec_per_round else None,
            "moves_per_sec_per_problem": per_problem_rate,
            # この run で 1 問題が実際に受けた処理速度 (G 問題が GPU を分け合う) での、分 -> 手数の換算
            "moves_per_minute_per_problem": (60.0 * per_problem_rate) if per_problem_rate else None,
        },
        "clone_check": check.summary(),
        "problems_detail": [p.to_dict() for p in probs],
    }


# ================================================================ shapes (段 2 の切り分け)


def _key_names() -> dict[int, str]:
    """BFS の表の p-data のキー -> 文字列 (表に無いキーは呼び出し側で数字のまま出す)。"""
    return {tp.p_data_key(pd): pd for pd in tp.potential_table()}


@torch.inference_mode()
def run_shapes(env, unwrapped, policy: Policy, settle: tuple[float, float]) -> dict:
    """段のアーカイブの形ごとに、同じ初期状態から同じ方策で R 回試し、固結びの割合を形ごとに出す。

    推論時探索の記録 (`docs/experiments/2026-10-03_inference_search.md` §3.1) では、同じ状態・同じ手でも手の後の
    段が約 3 割変わった。段 2 -> 固結びの失敗が「形 (初期状態) で決まる」のか「試行ごとのばらつき」なのかを
    分けるため、形ごとの成功率の分布を見る (0 か 1 に分かれれば形、平均の付近に集まればばらつき)。

    * 初期状態はアーカイブのファイル (学習の `archive.pt` か、台本の初期アーカイブ) の `--shape_stage` の段の
      形。`ResetToStage` と同じ書き方 (重心を env 原点へ、速度 0) で、**yaw は 0** に揃える (同じ形の R 個の
      試行を同じ初期状態にするため)。
    * 1 バッチ = `num_envs // R` 個の形 x R 環境。`env.reset()` (グリッパー・手・辺の選択を初期化) の後に
      ロープの状態を書き込み、観測を計算し直す。書き込んだ後の段が `--shape_stage` と違う形は集計から外す
      (アーカイブの復元の不一致。`ResetToStage` の `restore_mismatch` と同じもの)。
    * 関数全体を `inference_mode` で回す。`env.step()` の後の環境のテンソルは inference tensor になり、
      その外 (2 バッチ目の `env.reset()` の書き込み) で更新すると例外になるため (`search` と同じ)。
    * Play 環境 (chain モード) のまま `--shape_moves` 手打つ。段が下がっても続く (chain 評価と同じ)。
      各環境の最初のエピソードだけを数え、固結び・発散で reset された後は数えない。
      `knot` は M 手以内の固結び、`knot_no_drop` はその中で一度も段が下がらなかったもの
      (学習の 1 遷移の成功 `success_s2` に近い定義。学習では段が下がると終わる)。
    """
    n = unwrapped.num_envs
    dev = unwrapped.device
    r = args_cli.shape_reps
    moves = args_cli.shape_moves
    per_batch = n // r
    if per_batch < 1:
        raise ValueError(f"--num_envs ({n}) は --shape_reps ({r}) 以上にする")
    if moves > int(unwrapped.max_episode_length):
        raise ValueError(
            f"--shape_moves ({moves}) は Play のエピソード長 ({int(unwrapped.max_episode_length)} 手) 以下にする"
        )
    path = args_cli.shape_archive
    data = torch.load(path, map_location="cpu")
    entries = data["entries"].get(args_cli.shape_stage)
    if entries is None:
        raise ValueError(f"{path} に段 {args_cli.shape_stage} の形が無い (ある段: {sorted(data['entries'])})")
    total = int(entries["root_pose"].shape[0])
    k = total if args_cli.shapes <= 0 else min(args_cli.shapes, total)
    gen = torch.Generator().manual_seed(int(args_cli.seed or 0))
    picked = torch.arange(total) if k == total else torch.randperm(total, generator=gen)[:k].sort().values
    print(f"[shapes] {path}: stage {args_cli.shape_stage} has {total} shapes, using {k} x {r} tries, {moves} moves")

    rope = unwrapped.scene["rope"]
    names = _key_names()
    mean_mask = torch.ones(n, dtype=torch.bool, device=dev) if args_cli.deterministic else None
    shapes: list[dict] = []
    t0 = time.time()
    for b0 in range(0, k, per_batch):
        bidx = picked[b0 : b0 + per_batch]
        m = int(bidx.numel())
        ids = torch.arange(m * r, device=dev)
        shape_of = ids // r
        env.reset()
        rel = entries["root_pose"][bidx].to(dev)[shape_of]
        joints = entries["joint_pos"][bidx].to(dev)[shape_of]
        pose = sa.place_at_origin(rel, unwrapped.scene.env_origins[ids], torch.zeros(ids.numel(), device=dev))
        rope.write_root_pose_to_sim(pose, env_ids=ids)
        rope.write_root_velocity_to_sim(torch.zeros(ids.numel(), 6, device=dev), env_ids=ids)
        rope.write_joint_state_to_sim(joints, torch.zeros_like(joints), env_ids=ids)
        invalidate_extractor(unwrapped)
        obs = env.get_observations()
        start = obs_stage(obs)
        keys = tp.batch_keys(get_extractor(unwrapped).batch)

        alive = torch.zeros(n, dtype=torch.bool, device=dev)
        alive[ids] = True
        knot_at = torch.full((n,), -1, dtype=torch.long, device=dev)
        div_at = torch.full((n,), -1, dtype=torch.long, device=dev)
        drop_at = torch.full((n,), -1, dtype=torch.long, device=dev)
        for t in range(1, moves + 1):
            obs, _, _, _ = env.step(policy.act(obs, mean_mask))
            knot, div, _, stage_after = step_outcome(env, unwrapped, obs, *settle)
            knot_at[alive & knot] = t
            div_at[alive & div & ~knot] = t
            drop_at[alive & ~knot & ~div & (stage_after < start) & (drop_at < 0)] = t
            alive &= ~(knot | div)

        for j in range(m):
            sl = slice(j * r, (j + 1) * r)
            st = start[sl]
            ka, da, dr = knot_at[sl], div_at[sl], drop_at[sl]
            key = int(keys[j * r])
            shapes.append(
                {
                    "index": int(bidx[j]),
                    "start_stage": int(st[0]),
                    "start_stage_agree": float((st == st[0]).float().mean()),
                    "key": key,
                    "p_data": names.get(key, str(key)),
                    "tries": r,
                    "knot": int((ka > 0).sum()),
                    "knot_no_drop": int(((ka > 0) & (dr < 0)).sum()),
                    "diverged": int((da > 0).sum()),
                    "dropped": int((dr > 0).sum()),
                    "knot_by_move": [int(((ka > 0) & (ka <= t)).sum()) for t in range(1, moves + 1)],
                }
            )
        print(
            f"[shapes] batch {b0 // per_batch + 1}/{math.ceil(k / per_batch)} ({time.time() - t0:.0f} s): "
            f"knot {[s['knot'] for s in shapes[-m:]]} / {r}"
        )

    valid = [s for s in shapes if s["start_stage"] == args_cli.shape_stage]
    by_key: dict[str, list[dict]] = {}
    for s in valid:
        by_key.setdefault(s["p_data"], []).append(s)
    tries = [s["tries"] for s in valid]
    return {
        "config": {
            "archive": path,
            "stage": args_cli.shape_stage,
            "archive_size": total,
            "shapes": k,
            "reps": r,
            "moves": moves,
            "deterministic": args_cli.deterministic,
            "yaw": 0.0,
        },
        "summary": {
            "shapes_excluded_stage_mismatch": len(shapes) - len(valid),
            "knot": bs.shape_rate_summary([s["knot"] for s in valid], tries),
            "knot_no_drop": bs.shape_rate_summary([s["knot_no_drop"] for s in valid], tries),
            "diverged_frac": sum(s["diverged"] for s in valid) / max(sum(tries), 1),
            "dropped_frac": sum(s["dropped"] for s in valid) / max(sum(tries), 1),
            "knot_by_move": [sum(s["knot_by_move"][t] for s in valid) / max(sum(tries), 1) for t in range(moves)],
            "by_p_data": {
                pd: {
                    "shapes": len(g),
                    "knot": sum(s["knot"] for s in g) / sum(s["tries"] for s in g),
                    "shape_rates": sorted(s["knot"] / s["tries"] for s in g),
                }
                for pd, g in sorted(by_key.items(), key=lambda kv: -len(kv[1]))
            },
        },
        "shapes_detail": shapes,
    }


# ================================================================ main


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if args_cli.seed is not None:
        torch.manual_seed(args_cli.seed)

    resume_path = retrieve_file_path(args_cli.checkpoint)
    env_cfg.log_dir = os.path.dirname(resume_path)
    # Play の p-data 表示と初期アーカイブの読み込みは評価に要らない (表示は毎手出て遅い)。
    env_cfg.rewards.topology_stats.params["print_envs"] = 0
    env_cfg.events.reset_to_stage.params["load_paths"] = ()
    if args_cli.mode == "shapes" and args_cli.shape_archive is None:
        args_cli.shape_archive = os.path.join(os.path.dirname(resume_path), "archive.pt")
    if args_cli.mode in ("clone_test", "search"):
        env_cfg.episode_length_s = _NO_TIME_OUT_S
    settle_params = env_cfg.terminations.stage_up.params
    settle = (float(settle_params["settled_height_diameters"]), float(settle_params["rope_diameter"]))

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    runner.load(resume_path)
    policy = Policy(runner, args_cli.std_scale)
    unwrapped = env.unwrapped
    budgets = [int(b) for b in args_cli.budgets.split(",") if b]

    t0 = time.time()
    if args_cli.mode == "clone_test":
        result = run_clone_test(env, unwrapped, policy, settle)
    elif args_cli.mode == "rollouts":
        result = run_rollouts(env, unwrapped, policy, budgets)
    elif args_cli.mode == "shapes":
        result = run_shapes(env, unwrapped, policy, settle)
    else:
        result = run_search(env, unwrapped, policy, settle, budgets)
    summary = {
        "task": args_cli.task,
        "checkpoint": resume_path,
        "mode": args_cli.mode,
        "num_envs": unwrapped.num_envs,
        "seed": args_cli.seed,
        "hydra_overrides": hydra_args,
        "std_scale": args_cli.std_scale,
        "deterministic": args_cli.deterministic if args_cli.mode in ("rollouts", "shapes") else None,
        "wall_seconds": time.time() - t0,
        **result,
    }
    print(f"\n=== eval_search ({args_cli.mode}) ===")
    for k, v in summary.items():
        if k in ("episodes", "problems_detail", "shapes_detail"):
            continue
        print(f"{k}: {v}")
    out = args_cli.out or default_out(resume_path, args_cli.mode, args_cli.tag)
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[INFO]: wrote {out}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
