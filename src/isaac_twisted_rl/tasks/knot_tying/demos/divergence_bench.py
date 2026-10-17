"""1 手あたりの発散を、フェーズ・段・手の幾何ごとに数える計測スクリプト。

v3 の学習では 1 手あたり 5〜6% が発散し、段 2 からの手は約 13% 発散した
(`versions/v3/Result_report_v3.md` §3.3)。どのフェーズで・どんな手で発散しているかを突き止め、
発散対策 (v5) の前後を同じ条件で比べるために作った。

    # ランダムな手 (v3 の smoke test と同じ条件)
    uv run python -u src/isaac_twisted_rl/tasks/knot_tying/demos/divergence_bench.py \
        --task Isaac-Twisted-RL-Knot-v3 --num_envs 1024 --steps 24 --headless --device cuda:1

    # 学習済みの方策 (学習中と同じく std でサンプルする)。v3 の学習と同じくアーカイブ (prob 0.5) から始める
    ... --policy ckpt --checkpoint logs/rsl_rl/knot_tying_v3/2026-09-28_17-28-57/model_150.pt

    # ゲート 1 の台本の途中状態 (段 1 / 2) から始める。結び目の近くの手を集中して測る
    ... --seed_archive outputs/archive_seed/overhand_28links.pt --seed_frac 1.0

    # 設定の上書き (overhand_script.py と同じ書式)。発散対策の切り分けに使う
    ... --cfg 'grasp_limits.release_tip_z=(0.05, 0.10)'

出力 (`--out`、既定は `outputs/divergence_bench/<label>/`):
    summary.log  発散率 (全体 / 段ごと / フェーズごと / 手の幾何ごと) と品質の指標
    moves.pt     手ごとの記録 (後から別の切り口で集計するため)

## 数え方

* **発散**は学習と同じ終了項 `diverged` (NaN・水平 3 m・高さ 5 m・リンク速度 20 m/s) が、その手の
  終わりに立ったもの。
* **発散したフェーズ**は、手の途中でロープのリンク速度が初めて 20 m/s を超えた (または NaN に
  なった) フェーズ (`PickPlaceAction` の診断 `cfg.diag`)。手の終わりには落ち着いていて発散と
  数えられない「途中の暴れ」も、`spike` として別に数える。
* **段**は v4 の段 (`BfsPotential.stage`。0: 自明、1: 輪など、2: C2・cover 後、3: 固結び)。
  手を打つ前の段で層別する。
* **品質**: 把持 (carry の終わりに掴んだリンクが 5 cm 以上上がっている)、手の後にロープが床に
  落ちている (最高点が直径の 3 倍以下)、段の遷移 (前 -> 後)。発散対策で物理を変えても
  結ぶ能力が落ちていないことを確かめるため。
"""

from __future__ import annotations

import argparse
import ast
import os
import time

from isaaclab.app import AppLauncher

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), *[".."] * 5))

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", type=str, default="Isaac-Twisted-RL-Knot-v3")
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--steps", type=int, default=24, help="env.step() の回数 (= 1 環境あたりの手数)。")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--policy", choices=("random", "ckpt"), default="random")
parser.add_argument("--checkpoint", type=str, default="", help="`--policy ckpt` の rsl_rl チェックポイント。")
parser.add_argument(
    "--deterministic", action="store_true", help="ckpt の平均の行動を使う (既定は学習と同じくサンプル)。"
)
parser.add_argument(
    "--archive_prob",
    type=float,
    default=None,
    help="`events.reset_from_archive.params.prob` の上書き (v3 系のみ)。既定はタスクの値のまま。",
)
parser.add_argument("--seed_archive", type=str, default="", help="reset のたびに、このアーカイブの形から始める。")
parser.add_argument("--seed_frac", type=float, default=1.0, help="`--seed_archive` から始める reset の割合。")
parser.add_argument("--seed_stage_weights", type=str, default="0,1,1", help="`--seed_archive` の段 0,1,2 を選ぶ重み。")
parser.add_argument("--cfg", type=str, default="", help="環境設定の上書き。'a.b.c=value;a.b.d=value'。")
parser.add_argument("--label", type=str, default="", help="出力フォルダ名 (既定はタスクと方策から作る)。")
parser.add_argument("--out", type=str, default="")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------- 本体
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaac_twisted_rl.tasks  # noqa: E402, F401
from isaac_twisted_rl.tasks.knot_tying.common import state_archive as sa  # noqa: E402
from isaac_twisted_rl.tasks.knot_tying.common import topology as tp  # noqa: E402
from isaac_twisted_rl.tasks.knot_tying.common.mdp import phases  # noqa: E402
from isaac_twisted_rl.tasks.knot_tying.common.mdp.rope_size import get_rope_sizes, per_env_diameter  # noqa: E402
from isaac_twisted_rl.tasks.knot_tying.common.mdp.observations import (  # noqa: E402
    get_extractor,
    invalidate_extractor,
)
from isaaclab.managers import RewardTermCfg  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

_POST_ATTR = "_divergence_bench_post"


def _segment_distances(nodes: torch.Tensor, min_gap: int = 2) -> torch.Tensor:
    """ロープ芯線の区間どうしの最短距離の最小値 `(N,)` [m]。区間番号が `min_gap` 以上離れた組だけを見る。

    芯線の節点 (`polyline_from_link_centers`) の内部節点は関節の位置 = カプセルの軸の端点なので、
    区間 k はリンク k のカプセルの軸そのもの。`2r − 距離` がめり込みの深さになる。
    区間どうしの最短距離は Ericson "Real-Time Collision Detection" 5.1.9 の式 (端点でクランプ)。
    """
    p0, p1 = nodes[:, :-1], nodes[:, 1:]
    n = p0.shape[1]
    i, j = torch.triu_indices(n, n, offset=min_gap, device=nodes.device)
    a0, a1, b0, b1 = p0[:, i], p1[:, i], p0[:, j], p1[:, j]
    d1, d2, r = a1 - a0, b1 - b0, a0 - b0
    aa, ee, f = (d1 * d1).sum(-1), (d2 * d2).sum(-1), (d2 * r).sum(-1)
    c, bb = (d1 * r).sum(-1), (d1 * d2).sum(-1)
    denom = (aa * ee - bb * bb).clamp_min(1e-12)
    s = ((bb * f - c * ee) / denom).clamp(0.0, 1.0)
    t = (bb * s + f) / ee.clamp_min(1e-12)
    s = torch.where(t < 0.0, (-c / aa.clamp_min(1e-12)).clamp(0.0, 1.0), s)
    s = torch.where(t > 1.0, ((bb - c) / aa.clamp_min(1e-12)).clamp(0.0, 1.0), s)
    t = t.clamp(0.0, 1.0)
    dist = ((a0 + d1 * s.unsqueeze(-1)) - (b0 + d2 * t.unsqueeze(-1))).norm(dim=-1)
    return torch.nan_to_num(dist, nan=1.0e9).amin(dim=-1)


def _capture_post_move(env) -> torch.Tensor:
    """報酬項として手の直後 (reset の前) の段・交差数・最高点を控える。常に 0 を返す。

    `env.step()` が返った時点では、終了した環境はもう reset されていて手の結果が読めないため。
    """
    ex = get_extractor(env)
    stage = tp.get_bfs_potential(env.device).stage(ex.batch, ex.reduced_batch)
    z = ex.nodes[:, :, 2] - env.scene.env_origins[:, 2].unsqueeze(-1)
    zmax = torch.nan_to_num(z, nan=1.0e9).amax(dim=-1)
    setattr(
        env,
        _POST_ATTR,
        {"stage": stage.clone(), "count": ex.count.clone(), "zmax": zmax, "min_seg_dist": _segment_distances(ex.nodes)},
    )
    return torch.zeros(env.num_envs, device=env.device)


def _apply_overrides(env_cfg, text: str) -> None:
    """`overhand_script.py` の `--cfg` と同じ書式の上書き。`durations.*` なら decimation も作り直す。"""
    for item in filter(None, text.split(";")):
        path, value = item.split("=", 1)
        *parents, leaf = path.strip().split(".")
        obj = env_cfg
        for name in parents:
            obj = obj[name] if isinstance(obj, dict) else getattr(obj, name)
        val = ast.literal_eval(value.strip())
        if isinstance(obj, dict):
            obj[leaf] = val
        else:
            setattr(obj, leaf, val)
    # `durations` / `grasp_limits` はプリセット適用のときに行動の設定へ渡されているので渡し直す。
    if "grasp_limits." in text:
        env_cfg.actions.pick_place.limits = env_cfg.grasp_limits
    if "durations." in text:
        env_cfg.actions.pick_place.durations = env_cfg.durations
        env_cfg.decimation = env_cfg.durations.decimation(env_cfg.sim.dt)
        # 積のままだと浮動小数の誤差で ceil(episode_length_s / step_dt) が 1 手ぶん増えることがある
        # (descend 0.5 s = 807 ステップで 12.000000000000002 -> 13 手)。半手ぶん引いて 12 手ちょうどにする。
        env_cfg.episode_length_s = (env_cfg.actions_per_episode - 0.5) * env_cfg.decimation * env_cfg.sim.dt
        env_cfg.sim.render_interval = env_cfg.decimation


class CheckpointPolicy:
    """rsl_rl の `ActorCritic` (MLP, ELU) のチェックポイントから actor だけを組み立てる。"""

    def __init__(self, path: str, device: str, deterministic: bool):
        state = torch.load(path, map_location=device, weights_only=False)["model_state_dict"]
        layers = []
        k = 0
        while f"actor.{k}.weight" in state:
            w, b = state[f"actor.{k}.weight"], state[f"actor.{k}.bias"]
            lin = torch.nn.Linear(w.shape[1], w.shape[0]).to(device)
            lin.weight.data.copy_(w)
            lin.bias.data.copy_(b)
            layers.append(lin)
            if f"actor.{k + 2}.weight" in state:
                layers.append(torch.nn.ELU())
            k += 2
        self.net = torch.nn.Sequential(*layers).eval()
        self.std = state["std"].to(device)
        self.deterministic = deterministic

    @torch.no_grad()
    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        mean = self.net(obs)
        if self.deterministic:
            return mean
        return mean + self.std * torch.randn_like(mean)


class SeedStarts:
    """reset された環境の一部を、アーカイブの形 (段ごと) から始め直す。"""

    def __init__(self, uenv, path: str, frac: float, stage_weights: tuple[float, ...]):
        self.uenv = uenv
        self.rope = uenv.scene["rope"]
        self.archive = sa.StateArchive(
            classes=(0, 1, 2), capacity=100000, num_joints=self.rope.num_joints, device=uenv.device
        )
        loaded = self.archive.load(path)
        print(f"[INFO] seed archive {path}: {loaded}")
        self.frac = frac
        self.weights = stage_weights

    def apply(self, env_ids: torch.Tensor) -> torch.Tensor:
        """`env_ids` の一部に形を書き込み、書き込んだ env を返す。"""
        if env_ids.numel() == 0:
            return env_ids
        pick = env_ids[torch.rand(env_ids.numel(), device=env_ids.device) < self.frac]
        drawn = self.archive.sample(int(pick.numel()), self.weights)
        if drawn is None or pick.numel() == 0:
            return pick[:0]
        _, rel, joint = drawn
        sizes = get_rope_sizes(self.uenv)
        if (
            sizes is not None
        ):  # v16: アーカイブは公称の寸法なので、書き込み先の env の寸法に伸ばす (`ResetToStage` と同じ)
            rel = sa.rescale_rel_pose(rel, sizes.length_scale[pick], sizes.diameter_scale[pick])
        pose = sa.place_at_origin(rel, self.uenv.scene.env_origins[pick], sa.random_yaw(pick.numel(), self.uenv.device))
        self.rope.write_root_pose_to_sim(pose, env_ids=pick)
        self.rope.write_root_velocity_to_sim(torch.zeros(pick.numel(), 6, device=self.uenv.device), env_ids=pick)
        self.rope.write_joint_state_to_sim(joint, torch.zeros_like(joint), env_ids=pick)
        # write_*_to_sim は body 姿勢のバッファを無効化するので、次に読むと書いた形が返る
        # (`ResetToStage` の検算と同じ)。抽出器のキャッシュだけ捨てる。
        invalidate_extractor(self.uenv)
        return pick


class Log:
    def __init__(self, path: str):
        self.path = path
        open(path, "w").close()

    def __call__(self, text: str = ""):
        print(text, flush=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(text + "\n")


def _pct(num, den) -> str:
    num, den = float(num), float(den)
    return f"{100.0 * num / den:5.2f}% ({int(num)}/{int(den)})" if den > 0 else "   -   (0/0)"


def summarize(rec: dict[str, torch.Tensor], log: Log) -> dict[str, float]:
    """手ごとの記録から表を出す。主要な数値を dict で返す。"""
    n = rec["div"].numel()
    div = rec["div"]
    valid = ~rec["invalid"]
    out: dict[str, float] = {}
    log(f"moves={n} (invalid={int((~valid).sum())})")
    out["div_rate"] = float(div[valid].float().mean())
    log(f"[発散] 1 手あたり: {_pct(div[valid].sum(), valid.sum())}")

    log("[発散] 手を打つ前の段ごと:")
    for s in range(3):
        m = valid & (rec["stage0"] == s)
        out[f"div_s{s}"] = float(div[m].float().mean()) if m.any() else float("nan")
        log(f"    段 {s}: {_pct(div[m].sum(), m.sum())}")

    log("[発散] 最初に速度が 20 m/s を超えたフェーズ (発散した手):")
    fb = rec["first_bad"]
    for p, name in enumerate(phases.PHASE_NAMES):
        k = int((div & (fb == p)).sum())
        if k:
            log(f"    {name:10s} {_pct(k, div.sum())}")
    log(f"    (途中は 20 m/s 以下) {_pct((div & (fb < 0)).sum(), div.sum())}")
    spike = valid & ~div & (fb >= 0)
    log(
        f"[途中の暴れ] 手の途中で 20 m/s を超えたが、手の終わりには発散と判定されなかった: {_pct(spike.sum(), valid.sum())}"
    )

    ok = valid & ~div
    vmax = rec["vmax"][ok]
    if vmax.numel() > 0:
        log("[速度] 発散しなかった手の、フェーズごとのリンク最大速度 [m/s] (中央値 / 99% / 99.9%):")
        for p, name in enumerate(phases.PHASE_NAMES):
            v = vmax[:, p]
            q = torch.quantile(v, torch.tensor([0.5, 0.99, 0.999]))
            log(f"    {name:10s} {q[0]:6.2f} / {q[1]:6.2f} / {q[2]:6.2f}")

    def by_bins(key: str, edges: list[float], title: str, unit: str = "m"):
        log(f"[発散] {title}:")
        x = rec[key]
        lo = -float("inf")
        for e in edges + [float("inf")]:
            m = valid & (x > lo) & (x <= e)
            log(f"    ({lo:.3f}, {e:.3f}] {unit}: {_pct(div[m].sum(), m.sum())}")
            lo = e

    by_bins("grasp_top", [0.0, 0.021, 0.035, 0.05], "指の真下の紐の最高点 (0 = 指の下に紐なし)")
    by_bins("grasp_top_stacked", [0.0, 0.035, 0.05], "指の真下の、別の紐に乗った紐の最高点 (0 = なし)")
    by_bins("grasp_link_z", [0.012, 0.025, 0.04], "掴むリンクの高さ")
    by_bins("place_under", [0.0, 0.021, 0.035], "置き先の真下の紐の最高点 (0 = 紐なし)")
    by_bins("release_tip_z", [0.012, 0.022, 0.032, 0.042, 0.052, 0.07, 0.09], "離すときの指先の高さ")
    by_bins("carry_dist", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6], "運搬の水平距離")
    by_bins("yaw_turn", [0.5, 1.0, 1.5, 2.0, 2.5], "運搬中に回す手首の角度", unit="rad")
    log("[発散] フェーズ別 x 離す高さ (< 22 mm / 以上):")
    low = rec["release_tip_z"] < 0.022
    for p, name in enumerate(phases.PHASE_NAMES):
        m = div & (fb == p)
        if m.any():
            log(f"    {name:10s} {int(m.sum()):5d}  低 {int((m & low).sum()):5d}  高 {int((m & ~low).sum()):5d}")

    log("[品質]")
    held = rec["held"][ok]
    out["held"] = float(held.float().mean())
    log(f"    把持 (carry の終わりに掴んだリンクが 5 cm 以上上がっている): {_pct(held.sum(), held.numel())}")
    for name, m in (
        ("指の下に紐なし", rec["grasp_top"] <= 0.0),
        ("床の上の紐だけ", (rec["grasp_top"] > 0.0) & (rec["grasp_top_stacked"] <= 0.0)),
        ("浮いた紐あり", rec["grasp_top_stacked"] > 0.0),
    ):
        h = rec["held"][ok & m]
        log(f"      {name}: {_pct(h.sum(), h.numel())}")
    pen = rec["penetration"][ok]
    q = torch.quantile(pen, torch.tensor([0.5, 0.99, 0.999]))
    log(
        f"    めり込み (手の後、2 つ以上離れたリンクどうし) [mm]: 中央値 {1000 * q[0]:.2f} / 99% {1000 * q[1]:.2f} / "
        f"99.9% {1000 * q[2]:.2f} / 最大 {1000 * float(pen.max()):.2f}。半径 (10 mm) 超: {_pct((pen > 0.01).sum(), pen.numel())}"
    )
    settled = rec["settled"][ok]
    out["settled"] = float(settled.float().mean())
    log(f"    手の後にロープが床に落ちている: {_pct(settled.sum(), settled.numel())}")
    s0, s1 = rec["stage0"][ok], rec["stage1"][ok]
    log("    段の遷移 (発散しなかった手。行 = 前、列 = 後):")
    for a in range(3):
        m = s0 == a
        row = [int(((s1 == b) & m).sum()) for b in range(4)]
        tot = max(int(m.sum()), 1)
        log(
            f"      段 {a} (n={int(m.sum()):6d}): " + "  ".join(f"->{b} {100.0 * row[b] / tot:5.1f}%" for b in range(4))
        )
        for b in range(4):
            out[f"t{a}{b}"] = row[b] / tot
    up = (s1 > s0).float().mean()
    log(
        f"    段が上がった手: {100.0 * float(up):.2f}%  固結びに達した手: {100.0 * float((s1 == 3).float().mean()):.2f}%"
    )
    c0, c1 = rec["count0"][ok], rec["count1"][ok]
    log(f"    交差が増えた手: {100.0 * float((c1 > c0).float().mean()):.2f}%")
    out["up"] = float(up)
    return out


def main():
    task_short = args_cli.task.replace("Isaac-Twisted-RL-", "")
    label = args_cli.label or f"{task_short}_{args_cli.policy}"
    out_dir = args_cli.out or os.path.join(_REPO_ROOT, "outputs", "divergence_bench", label)
    os.makedirs(out_dir, exist_ok=True)
    log = Log(os.path.join(out_dir, "summary.log"))

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed
    _apply_overrides(env_cfg, args_cli.cfg)
    env_cfg.actions.pick_place.diag = True
    if args_cli.archive_prob is not None:
        env_cfg.events.reset_from_archive.params["prob"] = args_cli.archive_prob
    if args_cli.seed_archive and hasattr(env_cfg.events, "reset_from_archive"):
        # アーカイブの形から始めるときは、タスク自身のアーカイブからの reset を止める (重ならないように)。
        env_cfg.events.reset_from_archive.params["prob"] = 0.0
    env_cfg.rewards.bench_post = RewardTermCfg(func=_capture_post_move, weight=1.0)
    torch.manual_seed(args_cli.seed)

    env = gym.make(args_cli.task, cfg=env_cfg)
    # reset 以降はすべて inference_mode で回す。途中の書き込み (`SeedStarts`) が inference_mode の外だと、
    # env.step() の中で作られたテンソルへのインプレース更新で例外になるため。
    with torch.inference_mode():
        _run(env, log, out_dir)
    env.close()


def _run(env, log: Log, out_dir: str) -> None:
    uenv = env.unwrapped
    term = uenv.action_manager.get_term("pick_place")
    dev = uenv.device
    obs, _ = env.reset()
    log(
        f"=== divergence_bench task={args_cli.task} num_envs={uenv.num_envs} steps={args_cli.steps} "
        f"policy={args_cli.policy}{' (mean)' if args_cli.deterministic else ''} seed={args_cli.seed}"
    )
    log(
        f"    cfg='{args_cli.cfg}' archive_prob={args_cli.archive_prob} seed_archive='{args_cli.seed_archive}' frac={args_cli.seed_frac}"
    )
    log(f"    durations={uenv.cfg.durations} ({uenv.cfg.durations.total:.3f} s/手)")
    log(f"    limits={term.cfg.limits}")
    rope_spawn = uenv.cfg.scene.rope.spawn
    if hasattr(
        rope_spawn, "scales"
    ):  # v16 の寸法の DR (`RopeSizeSpawnerCfg`)。接触の設定は組ごとの USD の spawn 設定が持つ
        log(
            f"    rope: sizes={list(rope_spawn.scales)} max_depenetration_velocity={rope_spawn.max_depenetration_velocity} "
            f"solver_position_iteration_count={rope_spawn.solver_position_iteration_count} contact_offset=組ごと (0.4 x 半径)"
        )
    else:
        log(
            f"    rope: max_depenetration_velocity={rope_spawn.rigid_props.max_depenetration_velocity} "
            f"solver_position_iteration_count={rope_spawn.articulation_props.solver_position_iteration_count} "
            f"contact_offset={rope_spawn.collision_props.contact_offset}"
        )

    policy = None
    if args_cli.policy == "ckpt":
        policy = CheckpointPolicy(args_cli.checkpoint, dev, args_cli.deterministic)
    seeds = None
    if args_cli.seed_archive:
        w = tuple(float(v) for v in args_cli.seed_stage_weights.split(","))
        seeds = SeedStarts(uenv, args_cli.seed_archive, args_cli.seed_frac, w)
        seeds.apply(torch.arange(uenv.num_envs, device=dev))
        obs = uenv.observation_manager.compute()

    diverged_term = uenv.termination_manager.get_term  # 立った行だけ書き換わる (terminated と組み合わせて読む)
    names = uenv.termination_manager.active_terms
    cols: dict[str, list[torch.Tensor]] = {}
    ep_end = {k: 0 for k in names}
    t0 = time.time()
    dr_term = None
    # 寸法の DR (v16) では直径が env ごと (寸法を振らない設定では公称の float のまま)
    diameter = per_env_diameter(uenv, term.cfg.rope_diameter)
    sizes = get_rope_sizes(uenv)
    if "rope_dr" in uenv.event_manager.active_terms.get("reset", []):
        dr_term = uenv.event_manager.get_term_cfg("rope_dr").func  # v14: 手ごとの物理の倍率も残す
    for step in range(args_cli.steps):
        # この手の物理。env.step() の中の reset で、終わった env の倍率は引き直される。
        dr_scales = dr_term.scales.clone() if dr_term is not None else None
        if policy is None:
            act = 2.0 * torch.rand(uenv.num_envs, 6, device=dev) - 1.0
        else:
            act = policy(obs["policy"])
        obs, _, terminated, truncated, _ = env.step(act)
        d = term.diagnostics
        post = getattr(uenv, _POST_ATTR)
        ended = terminated | truncated
        div = terminated & diverged_term("diverged")
        for k in names:
            ep_end[k] += int((ended & diverged_term(k)).sum()) if k != "time_out" else int(truncated.sum())
        batch0 = term.batch_at_action_start
        stage0 = tp.get_bfs_potential(dev).stage(batch0, term.reduced_batch_at_action_start)
        a4 = act.clamp(-1.0, 1.0)[:, 4]
        row = {
            "div": div,
            "invalid": term.invalid_action.clone(),
            "stage0": stage0,
            "stage1": post["stage"],
            # crossings_at_action_start も reset で 0 に戻るが、交差数は発散しなかった手でしか使わない
            "count0": term.crossings_at_action_start.clone(),
            "count1": post["count"],
            "settled": post["zmax"] <= 3.0 * diameter,
            # めり込みの深さ [m] (負なら離れている)。リンク番号が 2 以上離れたカプセルどうし
            "penetration": diameter - post["min_seg_dist"],
            "first_bad": d["first_bad_phase"].clone(),
            "vmax": d["phase_vmax"].clone(),
            "held": d["held"].clone(),
            "grasp_link_z": d["grasp_link_z"].clone(),
            "grasp_top": d["grasp_top"].clone(),
            "grasp_top_stacked": d["grasp_top_stacked"].clone(),
            "place_under": d["place_under"].clone(),
            # processed_actions は reset で 0 に戻るので使わない (発散した手の値が消える)
            "release_tip_z": d["release_tip_z"].clone(),
            "carry_dist": d["carry_dist"].clone(),
            "yaw_turn": d["yaw_turn"].clone(),
            "a4": a4,
            "act": act.clamp(-1.0, 1.0).clone(),
            # processed_actions は reset で 0 に戻るので、reset が触らない内部の値を読む
            "grasp_id": term._grasp_id.clone(),
            "target_id": term._target_id.clone(),
            "step": torch.full_like(stage0, step),
        }
        if dr_scales is not None:
            row["dr_scales"] = dr_scales
        if sizes is not None:
            row["size_index"] = sizes.index
        for k, v in row.items():
            cols.setdefault(k, []).append(v.detach().cpu())
        if seeds is not None and bool(ended.any()):
            seeds.apply(ended.nonzero(as_tuple=False).squeeze(-1))
            obs = uenv.observation_manager.compute()
        if (step + 1) % 4 == 0:
            rate = float(torch.cat(cols["div"]).float().mean())
            log(f"  step {step + 1}/{args_cli.steps}: div so far {100 * rate:.2f}%  ({time.time() - t0:.0f} s)")
    elapsed = time.time() - t0
    rec = {k: torch.cat(v) for k, v in cols.items()}
    torch.save({"args": vars(args_cli), "rec": rec}, os.path.join(out_dir, "moves.pt"))
    log(
        f"elapsed {elapsed:.0f} s ({elapsed / args_cli.steps:.1f} s/step, {args_cli.steps * uenv.num_envs / elapsed:.1f} 手/s)"
    )
    total_end = sum(ep_end.values())
    log("[エピソードの終わり方] " + "  ".join(f"{k} {_pct(v, total_end)}" for k, v in ep_end.items()))
    summarize(rec, log)
    if "size_index" in rec:
        sizes = get_rope_sizes(uenv)
        log("[寸法の DR] 組ごとの 1 手あたりの発散率 (組 = env の番号 % 組の数):")
        for k, name in enumerate(sizes.names):
            m = rec["size_index"] == k
            log(f"  [size] {name:16s} div={100 * float(rec['div'][m].float().mean()):5.2f}% ({int(m.sum())} 手)")
    if "dr_scales" in rec:
        from isaac_twisted_rl.tasks.knot_tying.common import domain_rand as dr

        log("[物理の DR] 軸別の 1 手あたりの発散率 (倍率の小 / 中 / 大 の 3 分位、手ごとの倍率で分ける):")
        for i, name in enumerate(dr.AXES):
            v = rec["dr_scales"][:, i]
            thirds = v.argsort().chunk(3)
            log(
                f"  {name:16s} "
                + "  ".join(
                    f"x{float(v[t].mean()):.2f}:{100 * float(rec['div'][t].float().mean()):5.2f}%" for t in thirds
                )
            )


if __name__ == "__main__":
    main()
    simulation_app.close()
