"""1 手 (`env.step` 1 回) の時間の内訳と 1 手あたりの発散率を、学習と同じ環境で測るベンチマーク。

    uv run python -u src/isaac_twisted_rl/scripts/bench_step_time.py --task Isaac-Twisted-RL-Knot-v14 \
        --num_envs 1024 --moves 8 --headless --device cuda:0 \
        --cfg "scene.rope.spawn.articulation_props.solver_position_iteration_count=4"

## 何を測るか

1. **素の速さ** (`clean`): 計時の同期を入れずに `env.step` を回し、1 手の時間とスループット (手/s) を出す。
   学習の収集と同じ条件 (CPU と GPU が重なって動く) の数字はこちら。
2. **内訳** (`breakdown`): Isaac Lab の `ManagerBasedRLEnv.step` が呼ぶ処理 (行動の処理・ハンドの制御・
   書き込み・PhysX・`scene.update`・終了・報酬・reset・観測) と、p-data の抽出 (`RopeTopologyExtractor.update`)、
   報酬・観測・終了の項ごとの時間を、前後で `torch.cuda.synchronize()` して測る。同期の分だけ素の速さより遅い。
   PhysX の時間は 1 手の中の物理ステップの番号ごとにも残し、フェーズ (`PhaseDurations`) ごとに足す。
   PhysX の 1 ステップの重さが、ロープを掴んで動かしている区間と待ちの区間で違うかを見るため。
3. **1 手あたりの発散率**: 終了の項 `diverged` が立った割合 (両方の計測の手を合わせる)。項の返り値をその場で
   捕まえて数える。`TerminationManager.get_term()` は立った環境の値しか書き換えず reset でも消えないので、
   手の後に読むと前の手の発散を数え直してしまう (`.claude/rules/mdp.md`)。
4. **ロープの速さの推移** (`speed_profile`): 物理ステップごとに、環境ごとのリンクの最大速度を取り、1 手の中の
   ステップの番号ごとに環境の中央値・90 パーセンタイル・0.05 m/s を超える割合を残す (内訳の手で集める)。
   待ちのフェーズ (relax・settle_out など) の終わりにロープが止まっているか、どこまで詰められるかを見るため。
   発散した環境は値が大きく外れるので、分位点で読む。

行動はランダム (`N(0, 0.5)` を [−1, 1] に切る。`rope_study/bench_physics.py` の `random` と同じ)。学習済みの
方策の手とは接触の数や reset の頻度が違うので、発散率の絶対値は学習のログと比べない。条件どうしの比で読む。

PPO の更新の時間はここでは測らない (学習ログの `Perf/learning_time` にある)。

## 設定の上書き

`--sim_hz` と `--durations "relax=0.25,settle_out=0.2"` は、物理の刻みとフェーズの尺を変え、それに従う値
(`decimation`・`episode_length_s`・`render_interval`・行動の項のフェーズの尺・報酬の重みの `step_dt` のスケール) を
`KnotTyingEnvCfg.__post_init__` と同じ式で計算し直す。めり込み解消速度は再スケールしないので、変えるなら `--cfg` で
`scene.rope.spawn.rigid_props.max_depenetration_velocity` を書く (`--cfg` はこの 2 つの後に効く)。

`--cfg "a.b=値;c.d=値"` は環境設定の `__post_init__` の **後** に書き込む (値は Python リテラル)。学習の Hydra の
上書き (`env.a.b=値`) と同じ時点なので、同じ上書きを学習に付ければ同じ物理になる。ただし `durations`・
`solver_position_iterations` など `__post_init__` やプリセットの適用で他の値に写されるものは、ここでは効かない
(写された先を直接書く。ソルバの反復はロープとグリッパーの `articulation_props` の両方)。

GPU は他のジョブと共有していることが多いので、計測の前後の `nvidia-smi` の使用率を JSON に残す。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time

from isaaclab.app import AppLauncher

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), *[".."] * 3))

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", type=str, default="Isaac-Twisted-RL-Knot-v14")
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--warmup", type=int, default=2, help="計時しない最初の手の数。")
parser.add_argument("--moves", type=int, default=6, help="素の速さ・内訳のそれぞれで打つ手の数。")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--sim_hz", type=float, default=0.0, help="物理の刻み [Hz]。0 なら設定のまま。")
parser.add_argument("--durations", type=str, default="", help="フェーズの尺の上書き 'relax=0.25,settle_out=0.2' [s]。")
parser.add_argument("--cfg", type=str, default="", help="環境設定の上書き 'a.b=値;...' (値は Python リテラル)。")
parser.add_argument("--no_breakdown", action="store_true", help="内訳を測らない (素の速さと発散率だけ)。")
parser.add_argument("--label", type=str, default="")
parser.add_argument("--out", type=str, default="")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import ast  # noqa: E402
from collections import defaultdict  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import isaac_twisted_rl.tasks  # noqa: E402, F401
from isaac_twisted_rl.tasks.knot_tying.common import rope_state  # noqa: E402
from isaac_twisted_rl.tasks.knot_tying.common.mdp import phases  # noqa: E402


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


def _retime(cfg, sim_hz: float, durations: str):
    """刻みと尺を変え、それに従う値を `KnotTyingEnvCfg.__post_init__` と同じ式で計算し直す。"""
    if not sim_hz and not durations:
        return
    if sim_hz:
        cfg.sim.dt = 1.0 / sim_hz
    for item in filter(None, (s.strip() for s in durations.split(","))):
        k, v = item.split("=")
        setattr(cfg.durations, k.strip(), float(v))
    cfg.actions.pick_place.durations = cfg.durations
    cfg.decimation = cfg.durations.decimation(cfg.sim.dt)
    # 積のままだと浮動小数の誤差で手数が 1 つ増えることがある (`demos/divergence_bench.py` と同じ対処)
    cfg.episode_length_s = (cfg.actions_per_episode - 0.5) * cfg.decimation * cfg.sim.dt
    cfg.sim.render_interval = cfg.decimation
    cfg._rescale_reward_weights()


def _gpu_snapshot() -> list[str]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=20,
        )
        return out.stdout.strip().splitlines()
    except Exception as e:  # noqa: BLE001
        return [f"nvidia-smi failed: {e}"]


class Timer:
    """名前ごとに、前後で CUDA を同期した時間を足す。`enabled` が偽なら素通し。"""

    def __init__(self):
        self.total = defaultdict(float)
        self.calls = defaultdict(int)
        self.enabled = False

    def wrap(self, name, fn):
        if hasattr(fn, "reset") and not hasattr(fn, "__code__"):
            # クラスで書かれた項 (`ManagerTermBase`)。ほかの項が `isinstance` で引くので、包まずにインスタンスの
            # クラスを `__call__` だけ差し替えた派生クラスに付け替える (特殊メソッドはクラスから引かれるため)。
            cls = type(fn)
            fn.__class__ = type(cls.__name__, (cls,), {"__call__": self.wrap(name, cls.__call__)})
            return fn

        def wrapped(*a, **k):
            if not self.enabled:
                return fn(*a, **k)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = fn(*a, **k)
            torch.cuda.synchronize()
            self.total[name] += time.perf_counter() - t0
            self.calls[name] += 1
            return out

        return wrapped


def instrument(u, T: Timer, physx_per_step: list, speeds: list):
    """env の各処理を計時の包みに差し替える (インスタンスの属性を上書きするだけで、クラスには触れない)。"""
    am, sc, sim = u.action_manager, u.scene, u.sim
    am.process_action = T.wrap("process_action", am.process_action)
    am.apply_action = T.wrap("apply_action", am.apply_action)
    sc.write_data_to_sim = T.wrap("write_data_to_sim", sc.write_data_to_sim)
    timed_update = T.wrap("scene_update", sc.update)
    rope = sc["rope"]

    def update(*a, **k):
        out = timed_update(*a, **k)
        if T.enabled:
            v = torch.nan_to_num(rope.data.body_lin_vel_w.norm(dim=-1), nan=1e3).amax(dim=1)
            speeds.append(torch.stack([v.median(), v.quantile(0.9), (v > 0.05).float().mean()]))
        return out

    sc.update = update

    raw_step = sim.step

    def sim_step(*a, **k):
        if not T.enabled:
            return raw_step(*a, **k)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = raw_step(*a, **k)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        T.total["physx"] += dt
        T.calls["physx"] += 1
        physx_per_step.append(dt)
        return out

    sim.step = sim_step

    tm, rm, om = u.termination_manager, u.reward_manager, u.observation_manager
    tm.compute = T.wrap("terminations", tm.compute)
    rm.compute = T.wrap("rewards", rm.compute)
    om.compute = T.wrap("observations", om.compute)
    u._reset_idx = T.wrap("reset", u._reset_idx)

    # 項ごと。p-data の抽出は最初に `get_extractor` を呼んだ項の中で走るので、別に包んで差し引けるようにする。
    for name, tc in zip(rm._term_names, rm._term_cfgs):
        tc.func = T.wrap(f"rew/{name}", tc.func)
    for name, tc in zip(tm._term_names, tm._term_cfgs):
        tc.func = T.wrap(f"term/{name}", tc.func)
    for group, names in om._group_obs_term_names.items():
        for name, tc in zip(names, om._group_obs_term_cfgs[group]):
            tc.func = T.wrap(f"obs/{group}/{name}", tc.func)
    raw_update = rope_state.RopeTopologyExtractor.update

    def ex_update(self, *a, **k):
        return T.wrap("pdata_extract", lambda: raw_update(self, *a, **k))()

    rope_state.RopeTopologyExtractor.update = ex_update


def main():
    cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    cfg.seed = args_cli.seed
    # アーカイブをファイルへ書かない (計測の run にはログのフォルダが無い)
    cfg.rewards.record_stage_archive.params["save_every"] = 0
    if "history_every" in cfg.rewards.record_stage_archive.params:
        cfg.rewards.record_stage_archive.params["history_every"] = 0
    _retime(cfg, args_cli.sim_hz, args_cli.durations)
    if args_cli.cfg:
        _set_cfg(cfg, args_cli.cfg)
    env = gym.make(args_cli.task, cfg=cfg)
    u = env.unwrapped
    n = u.num_envs
    dev = u.device
    rope_ap = cfg.scene.rope.spawn.articulation_props
    meta = {
        "task": args_cli.task,
        "num_envs": n,
        "cfg_override": args_cli.cfg,
        "sim_hz_override": args_cli.sim_hz,
        "durations_override": args_cli.durations,
        "sim_dt": u.physics_dt,
        "decimation": u.cfg.decimation,
        "step_dt": u.step_dt,
        "actions_per_episode": u.cfg.actions_per_episode,
        "max_episode_length": int(u.max_episode_length),
        "phase_steps": dict(zip(phases.PHASE_NAMES, cfg.durations.step_counts(u.physics_dt))),
        "rope_solver_pos_iters": rope_ap.solver_position_iteration_count,
        "rope_solver_vel_iters": rope_ap.solver_velocity_iteration_count,
        "gripper_solver_pos_iters": cfg.scene.gripper.spawn.articulation_props.solver_position_iteration_count,
        "rope_max_depenetration_velocity": cfg.scene.rope.spawn.rigid_props.max_depenetration_velocity,
        "rope_contact_offset": cfg.scene.rope.spawn.collision_props.contact_offset,
        "rope_self_collisions": rope_ap.enabled_self_collisions,
        "physx_solver_type": cfg.sim.physx.solver_type,
        "bounce_threshold_velocity": cfg.sim.physx.bounce_threshold_velocity,
        "gravity": list(cfg.sim.gravity),
        "gpu_before": _gpu_snapshot(),
    }
    print(f"[bench] meta: {json.dumps(meta, ensure_ascii=False)}", flush=True)

    T = Timer()
    physx_per_step: list[float] = []
    speeds: list[torch.Tensor] = []
    instrument(u, T, physx_per_step, speeds)
    g = torch.Generator(device=dev).manual_seed(args_cli.seed)
    adim = u.action_manager.total_action_dim
    div_flags = []
    # 発散の項の返り値をその場で捕まえる (docstring の 3)
    tm = u.termination_manager
    div_cfg = tm._term_cfgs[tm._term_names.index("diverged")]
    div_now = {}
    raw_div = div_cfg.func

    def div_capture(*a, **k):
        out = raw_div(*a, **k)
        div_now["v"] = out.clone()
        return out

    div_cfg.func = div_capture

    def move():
        a = (0.5 * torch.randn(n, adim, device=dev, generator=g)).clamp(-1, 1)
        env.step(a)
        div_flags.append(div_now.pop("v").float().mean().item())
        return a

    with torch.inference_mode():
        env.reset()
        for _ in range(args_cli.warmup):
            move()
        div_flags.clear()
        # 1. 素の速さ
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args_cli.moves):
            move()
        torch.cuda.synchronize()
        clean_sec = (time.perf_counter() - t0) / args_cli.moves
        print(f"[bench] clean: {clean_sec:.2f} s/move, {n / clean_sec:.1f} moves/s", flush=True)
        # 2. 内訳
        bd = None
        if not args_cli.no_breakdown:
            T.enabled = True
            t0 = time.perf_counter()
            for _ in range(args_cli.moves):
                move()
            torch.cuda.synchronize()
            T.enabled = False
            bd_sec = (time.perf_counter() - t0) / args_cli.moves
            per_move = {k: v / args_cli.moves for k, v in sorted(T.total.items(), key=lambda kv: -kv[1])}
            # フェーズごとの PhysX の時間 (1 手の中の物理ステップの番号で集める)
            dec = u.cfg.decimation
            steps = torch.tensor(physx_per_step).view(args_cli.moves, dec).mean(0)
            by_phase, i = {}, 0
            for name, k in zip(phases.PHASE_NAMES, cfg.durations.step_counts(u.physics_dt)):
                seg = steps[i : i + k]
                by_phase[name] = {"steps": k, "sec": float(seg.sum()), "ms_per_step": float(seg.mean() * 1e3)}
                i += k
            sp = torch.stack(speeds).view(args_cli.moves, dec, 3).mean(0).cpu()
            speed_profile, i = {}, 0
            for name, k in zip(phases.PHASE_NAMES, cfg.durations.step_counts(u.physics_dt)):
                seg = sp[i : i + k]
                # フェーズの中の 4 点 (始め・1/3・2/3・終わり)
                idx = sorted({0, k // 3, (2 * k) // 3, k - 1})
                speed_profile[name] = {
                    "at_step": idx,
                    "median": [round(float(seg[j, 0]), 4) for j in idx],
                    "p90": [round(float(seg[j, 1]), 4) for j in idx],
                    "frac_gt_5cm_s": [round(float(seg[j, 2]), 4) for j in idx],
                }
                i += k
            bd = {
                "speed_profile": speed_profile,
                "sec_per_move_synced": bd_sec,
                "sec_per_move_by_part": per_move,
                "physx_ms_per_step": float(steps.mean() * 1e3),
                "physx_by_phase": by_phase,
            }
            print(f"[bench] breakdown: {json.dumps(bd, ensure_ascii=False)}", flush=True)
    res = {
        "meta": meta,
        "clean": {"sec_per_move": clean_sec, "moves_per_sec": n / clean_sec},
        "breakdown": bd,
        "diverged_per_move": sum(div_flags) / max(len(div_flags), 1),
        "moves_counted_for_divergence": len(div_flags) * n,
        "gpu_after": _gpu_snapshot(),
    }
    label = args_cli.label or f"{args_cli.task}_n{n}"
    out = args_cli.out or os.path.join(_REPO_ROOT, "outputs", "bench_step_time", label)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(f"[bench] diverged/move {res['diverged_per_move']:.4f}; wrote {out}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
