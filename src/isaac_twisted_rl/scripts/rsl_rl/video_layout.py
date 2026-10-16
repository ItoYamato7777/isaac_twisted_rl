"""chain 評価の動画 (`eval_chain.py --video`) の置き場所・名前・コマの間隔・manifest・撮るかの判断。

撮影そのもの (カメラ・描画・書き出し) は `video_recorder.py` (Kit が要る)。ここは stdlib だけで書き、
Isaac Sim なしで試験する (`tests/test_video_layout.py`)。`stage_colors.py` と同じ分け方。

## なぜこの形か

* **置き場所は `video/sim/<experiment_name>/<run>/<評価の JSON の名前>/`。** run のフォルダ
  (`logs/rsl_rl/<experiment_name>/<run>/`) と 1 対 1 にし、同じ run の評価 (it ごと・DR 付き) を JSON の名前で分ける。
  動画は git で追跡せず (`logs/` と同じくスクリプトが書く)、`manifest.json` だけを追跡する
  (`docs/decisions/0031-eval-chain-video.md`)。
* **clip の名前に結末と手数を入れる** (`env003_knot_m09.mp4`)。フォルダを `ls` するだけで、見たいエピソード
  (発散・時間切れ) を選べるようにするため。結末は評価の表と同じ 1 つに決めた名前 (`chain_stats.exclusive_outcomes`)。
  撮った環境が評価の終わりまでに終わらなかったら `unfinished`。
* **コマは物理ステップの数で間隔を決める** (既定 8 = 240 Hz で 30 fps、実時間)。1 手 = 1 env.step = 数百物理ステップ
  なので、env.step ごとに撮る Isaac Lab の雛形 (`gym.wrappers.RecordVideo`) では 1 手に 1 コマにしかならない。
* **撮るかは GPU の空きで決める。** 描画は VRAM を足し、GPU は他の学習と共有している。足りなければ評価だけを回し、
  撮らなかった理由を JSON と manifest に残す (評価を遅らせてまで撮らない)。
"""

from __future__ import annotations

import os
import re

VIDEO_ROOT = "video"
SIM_DIR = "sim"
MANIFEST = "manifest.json"
UNFINISHED = "unfinished"
"""撮った環境が評価の終わりまでに 1 エピソードを終えなかったときの結末の名前。"""


def run_of_checkpoint(checkpoint: str) -> tuple[str, str]:
    """チェックポイントのパス (`.../<experiment_name>/<run>/model_N.pt`) から `(experiment_name, run)`。"""
    run_dir = os.path.dirname(os.path.abspath(checkpoint))
    return os.path.basename(os.path.dirname(run_dir)), os.path.basename(run_dir)


def iteration_of(checkpoint: str) -> int | None:
    """`model_N.pt` の N。名前が違えば None。"""
    m = re.fullmatch(r"model_(\d+)\.pt", os.path.basename(checkpoint))
    return int(m.group(1)) if m else None


def eval_name(out: str | None, checkpoint: str) -> str:
    """動画のフォルダの名前 = 評価の JSON の名前から拡張子を除いたもの。`--out` が無ければ `eval_chain_it<N>`。"""
    if out:
        return os.path.splitext(os.path.basename(out))[0]
    it = iteration_of(checkpoint)
    return f"eval_chain_it{it}" if it is not None else "eval_chain"


def clip_dir(repo_root: str, checkpoint: str, out: str | None) -> str:
    """動画と manifest を置くフォルダ。"""
    exp, run = run_of_checkpoint(checkpoint)
    return os.path.join(repo_root, VIDEO_ROOT, SIM_DIR, exp, run, eval_name(out, checkpoint))


def clip_name(env_id: int, outcome: str, moves: int) -> str:
    """`env<番号>_<結末>_m<手数>.mp4`。番号と手数は 0 埋めして `ls` の順を揃える。"""
    return f"env{env_id:03d}_{outcome}_m{moves:02d}.mp4"


def outcome_name(labels: dict[str, bool]) -> str:
    """`chain_stats.exclusive_outcomes` で 1 つに決めた結末のうち、立っているものの名前。無ければ `unfinished`。"""
    for name, fired in labels.items():
        if fired:
            return name
    return UNFINISHED


def frame_plan(physics_dt: float, decimation: int, every: int) -> dict:
    """`every` 物理ステップごとに 1 コマ撮るときの fps と 1 手あたりのコマ数。"""
    if every < 1:
        raise ValueError(f"--video_every は 1 以上: {every}")
    return {
        "every": every,
        "fps": 1.0 / (physics_dt * every),
        "frames_per_move": decimation / every,
    }


def parse_env_ids(spec: str, num_envs: int) -> list[int]:
    """`--video_envs`: 数 (`8` = 先頭の 8 環境) か、番号の並び (`0,5,9`)。評価の環境の数を超えた分は落とす。"""
    spec = spec.strip()
    if "," not in spec:
        return list(range(min(int(spec), num_envs)))
    ids = sorted({int(s) for s in spec.split(",") if s.strip()})
    bad = [i for i in ids if not 0 <= i < num_envs]
    if bad:
        raise ValueError(f"--video_envs の番号が環境の数 {num_envs} の外: {bad}")
    return ids


def parse_resolution(spec: str) -> tuple[int, int]:
    """`960x540` -> `(960, 540)`。yuv420p は幅・高さが偶数でないと書けない。"""
    m = re.fullmatch(r"(\d+)x(\d+)", spec.strip())
    if not m:
        raise ValueError(f"--video_res は <幅>x<高さ>: {spec}")
    w, h = int(m.group(1)), int(m.group(2))
    if w % 2 or h % 2:
        raise ValueError(f"--video_res の幅・高さは偶数にする (yuv420p): {spec}")
    return w, h


def gpu_index(device: str | None, cuda_visible_devices: str | None = None) -> int | None:
    """`cuda:N` を nvidia-smi の GPU の番号にする。`CUDA_VISIBLE_DEVICES` があれば並びで読み替える。"""
    if not device or not device.startswith("cuda"):
        return None
    n = int(device.split(":", 1)[1]) if ":" in device else 0
    if cuda_visible_devices:
        visible = [s.strip() for s in cuda_visible_devices.split(",") if s.strip()]
        if n >= len(visible) or not visible[n].isdigit():
            return None
        return int(visible[n])
    return n


def parse_free_mib(csv_text: str, index: int) -> int | None:
    """`nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits` から GPU `index` の空き [MiB]。"""
    for line in csv_text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2 and parts[0].isdigit() and int(parts[0]) == index:
            return int(float(parts[1]))
    return None


def parse_process_mib(csv_text: str, pid: int) -> int:
    """`nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits` から、`pid` が全 GPU で使う量 [MiB]。"""
    total = 0
    for line in csv_text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2 and parts[0].isdigit() and int(parts[0]) == pid and parts[1].isdigit():
            total += int(parts[1])
    return total


def video_decision(free_mib: int | None, min_free_mib: int) -> str | None:
    """撮らない理由。撮るなら None。空きが読めなければ撮らない (評価を優先する)。"""
    if free_mib is None:
        return "GPU の空きを nvidia-smi から読めなかった"
    if free_mib < min_free_mib:
        return f"GPU の空き {free_mib} MiB < --video_min_free_mib {min_free_mib} MiB"
    return None


def overlay_lines(
    run: str, iteration: int | None, env_id: int, move: int, max_moves: int, phase: str, stage: int
) -> list[str]:
    """コマの上に重ねる文字 (ASCII。cv2 の Hershey の字体は日本語を描けない)。"""
    it = f"it{iteration}" if iteration is not None else "it?"
    return [f"{run}  {it}  env{env_id:03d}", f"move {move}/{max_moves}  {phase}  stage {stage}"]


def ending_line(outcome: str, moves: int) -> str:
    """終わりの静止したコマに出す結末。"""
    return f"{outcome.upper()}  after {moves} moves"


def build_manifest(
    *,
    eval_json: str | None,
    argv: list[str],
    hydra_overrides: list[str],
    checkpoint: str,
    checkpoint_sha256: str | None,
    git: dict,
    plan: dict | None,
    resolution: tuple[int, int] | None,
    clips: list[dict],
    stats: dict | None = None,
    free_mib_before: int | None = None,
    skipped: str | None = None,
) -> dict:
    """`manifest.json` の中身。撮らなかったときも、理由と評価の情報を残すために書く。

    `clips` の各要素は `{"env_id", "outcome", "moves", "file", "frames"}`。`moves` は評価の JSON の手数と同じ数え方
    (その環境の最初のエピソードで打った手)。`stats` は撮影の計測 (1 コマの描画時間・ピークの VRAM と RAM など)。
    """
    return {
        "eval_json": eval_json,
        "argv": list(argv),
        "hydra_overrides": list(hydra_overrides),
        "checkpoint": checkpoint,
        "iteration": iteration_of(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "commit": git.get("commit"),
        "git_dirty": git.get("git_dirty"),
        "recorded": skipped is None,
        "skipped": skipped,
        "fps": plan["fps"] if plan else None,
        "every_physics_steps": plan["every"] if plan else None,
        "resolution": list(resolution) if resolution else None,
        "free_mib_before": free_mib_before,
        "stats": dict(stats or {}),
        "clips": clips,
    }
