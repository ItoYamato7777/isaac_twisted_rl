"""chain 評価 (`eval_chain.py --video`) の途中で、選んだ環境の 1 手の中を mp4 に撮る。

置き場所・名前・manifest は `video_layout.py` (Kit なし、単体テストあり)。仕様は `docs/system/video.md`。

## なぜこの形か

* **評価と同じ実行で撮る。** 動画のエピソードが JSON に数えたエピソードそのもの (環境の番号が対応する) になる。
  別のプロセスで撮ると、環境の数・乱数・GPU の束ね方が違う別の標本になる。その代わり、撮っても評価の値を
  変えてはいけない (`docs/decisions/0031-eval-chain-video.md` のゲート: 既存の JSON と全キーが一致)。
* **カメラは Isaac Lab の `Camera` ではなく、USD のカメラ + replicator の render product。** `Camera` は
  `/isaaclab/render/rtx_sensors` を立て、env.step が物理ステップの間と reset で勝手に描画するようになる。
  scene の設定にも入るので、評価の設定と学習の設定の照合 (`param_check`) に違いとして出る。ここではカメラを
  `/World/VideoCams` に置くだけで、env の設定にも描画の頻度にも触れない (`manager_based_rl_env.py` の
  `render()` と同じ作り方)。
* **コマは `sim.step` をインスタンスで包んで撮る** (`bench_step_time.py` と同じ包み方。クラスには触れない)。
  1 手 = 1 env.step = 数百物理ステップなので、env.step の外からは 1 手に 1 コマしか撮れない。`sim.render()` は
  物理を進めない (`playSimulations` を切って `app.update`)。
* **グリッパの色は `stage_colors.GripperStageColorizer` をそのまま使う** (Play と同じ見え方)。USD の
  `displayColor` だけを書き、物理には触れない。左上の文字の横にも同じ色の四角を描く (色が描画に乗らない
  紐モデル・設定でも段が読めるように)。
* **1 本ずつ ffmpeg に流す** (imageio-ffmpeg、libx264・yuv420p)。18 手 x 約 95 コマを RAM に溜めると 1 本で 1 GB を
  超えるため。結末と手数はエピソードが終わるまで分からないので、仮の名前で書いて最後に名前を付け替える。
* 撮れない設定 (描画が無い・XPBD の紐) では例外にする。XPBD の紐は USD に存在せず、GUI があるときだけ
  マーカーで描く (`common/rope_backends/adapters.py` の `XpbdRope._draw`) ので、headless の動画には映らない。
"""

from __future__ import annotations

import glob
import os
import resource
import subprocess
import time

import numpy as np
import torch

from isaac_twisted_rl.scripts.rsl_rl import video_layout as vl
from isaac_twisted_rl.scripts.rsl_rl.stage_colors import STAGE_COLORS, GripperStageColorizer

_CAM_ROOT = "/World/VideoCams"
_FOCAL_LENGTH = 18.15
"""Kit の既定の viewer (`/OmniverseKit_Persp`) と同じ画角 (横 60° 弱)。"""
_EYE_SCALE = 1.25
"""注視点からカメラまでを `viewer.eye` の何倍にするか。等倍だと手で運んだロープが画面の端に出る (スモークテストで見た)。"""
_WARMUP_RENDERS = 4
"""render product を作った直後は空のコマが返る (`manager_based_rl_env.py` の `render()` の注)。"""
_ENDING_S = 1.0
"""エピソードが終わった後、最後のコマを止めて結末を出す長さ [s]。"""


def _process_vram_mib() -> int | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return vl.parse_process_mib(out, os.getpid())


class EvalVideoRecorder:
    """選んだ環境の最初のエピソードを撮る。

    `before_step(obs, stage)` を `env.step` の直前に、`after_step(done, terms, length)` を直後に呼ぶ
    (`done` はこの手で終わった、まだ数えていない環境。`eval_chain.py` の数え方と同じ)。最後に `close()`。
    """

    def __init__(
        self,
        env,
        policy_nn,
        env_ids: list[int],
        out_dir: str,
        resolution: tuple[int, int],
        every: int,
        run: str,
        iteration: int | None,
    ):
        """`env` は `ManagerBasedRLEnv` (ラッパーを外したもの)、`policy_nn` は `runner.alg.policy`。"""
        import imageio.v2 as imageio
        import omni.replicator.core as rep

        import isaaclab.sim as sim_utils
        from isaaclab.utils.math import create_rotation_matrix_from_view, quat_from_matrix

        from isaac_twisted_rl.tasks.knot_tying.common.mdp.phases import PHASE_NAMES
        from isaac_twisted_rl.tasks.knot_tying.common.rope_backends.adapters import XpbdRope

        sim = env.sim
        if sim.render_mode.value < sim.RenderMode.PARTIAL_RENDERING.value:
            raise RuntimeError(f"描画のモードが {sim.render_mode.name} で撮れない (--enable_cameras が要る)")
        if isinstance(env.scene["rope"], XpbdRope):
            raise RuntimeError("XPBD の紐は headless では描画されない (GUI のマーカーだけ) ので撮れない")

        self._env = env
        self._sim = sim
        self.env_ids = list(env_ids)
        self.out_dir = out_dir
        self.resolution = resolution
        self.plan = vl.frame_plan(sim.get_physics_dt(), env.cfg.decimation, every)
        self._run = run
        self._iteration = iteration
        self._phase_names = PHASE_NAMES
        self._term = env.action_manager.get_term("pick_place")
        self._ids_t = torch.tensor(self.env_ids, dtype=torch.long, device=env.device)
        self._max_moves = int(env.max_episode_length)

        os.makedirs(out_dir, exist_ok=True)
        old = sorted(glob.glob(os.path.join(out_dir, "env*.mp4")) + glob.glob(os.path.join(out_dir, ".env*.mp4")))
        if old:
            print(f"[INFO] 同じ評価の名前の古い動画 {len(old)} 本を消す: {out_dir}")
            for p in old:
                os.remove(p)

        # -- カメラ: 撮り始めのロープのリンクの重心を見る。viewer の lookat (環境の原点) はロープの根元で、
        #    ロープは原点から片側へ伸びる (`rope_task_cfg.py` の init_state) ので、原点を見ると半分が画面の外に出る。
        #    カメラはその後動かさない (ロープが流れたらそのまま映す)。向きは Camera.set_world_poses_from_view と同じ
        origins = env.scene.env_origins[self._ids_t].detach().cpu().float()
        centroid = env.scene["rope"].data.body_pos_w[self._ids_t].detach().cpu().float().mean(dim=1)
        targets = centroid.clone()
        targets[:, 2] = origins[:, 2] + env.cfg.viewer.lookat[2]
        eyes = targets + _EYE_SCALE * torch.tensor(env.cfg.viewer.eye, dtype=torch.float32)
        quats = quat_from_matrix(create_rotation_matrix_from_view(eyes, targets, up_axis="Z"))
        cam_cfg = sim_utils.PinholeCameraCfg(focal_length=_FOCAL_LENGTH, clipping_range=(0.01, 100.0))
        self._render_products = []
        self._annotators = []
        for j, i in enumerate(self.env_ids):
            path = f"{_CAM_ROOT}/cam_{i:03d}"
            cam_cfg.func(path, cam_cfg, translation=tuple(eyes[j].tolist()), orientation=tuple(quats[j].tolist()))
            rp = rep.create.render_product(path, resolution)
            ann = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
            ann.attach([rp])
            self._render_products.append(rp)
            self._annotators.append(ann)

        self._colorizer = GripperStageColorizer(env, policy_nn)

        self._writers = []
        self._tmp_paths = []
        for i in self.env_ids:
            tmp = os.path.join(out_dir, f".env{i:03d}.part.mp4")
            self._tmp_paths.append(tmp)
            self._writers.append(
                imageio.get_writer(
                    tmp,
                    format="FFMPEG",
                    mode="I",
                    fps=self.plan["fps"],
                    codec="libx264",
                    pixelformat="yuv420p",
                    macro_block_size=2,
                    quality=None,
                    ffmpeg_params=["-crf", "23", "-preset", "veryfast"],
                    ffmpeg_log_level="error",
                )
            )
        k = len(self.env_ids)
        self._live = [True] * k
        self._frames = [0] * k
        self._last = [None] * k
        self._stage = [0] * k
        self.clips: list[dict] = []
        self._render_s = 0.0
        self._renders = 0
        self._black_frames = 0
        self.peak_vram_mib = _process_vram_mib()

        for _ in range(_WARMUP_RENDERS):
            sim.render()

        # -- 物理ステップを包む (インスタンスの属性。close() で消してクラスのメソッドに戻す)
        self._sim_steps = 0
        raw_step = sim.step

        def step(*a, **kw):
            out = raw_step(*a, **kw)
            self._sim_steps += 1
            if self._sim_steps % self.plan["every"] == 0 and any(self._live):
                self._capture()
            return out

        sim.step = step

    # ------------------------------------------------------------------ 評価のループから呼ぶ
    def before_step(self, obs, stage: torch.Tensor) -> None:
        """この手を打つ方策の段 (観測の段) を控え、グリッパを塗る。最初の手の前は初期の形を 1 コマ撮る。"""
        self._stage = stage[self._ids_t].tolist()
        self._colorizer.before_step(obs)
        if self._sim_steps == 0 and not any(self._frames):
            self._capture()

    def after_step(self, done: torch.Tensor, terms: dict[str, torch.Tensor], length: torch.Tensor) -> None:
        """この手で最初のエピソードを終えた環境の clip を閉じる。`terms` は終了項ごとの `(n,)`。"""
        from isaac_twisted_rl.tasks.knot_tying.common.chain_stats import exclusive_outcomes

        self._colorizer.after_step(done)
        d = done[self._ids_t].cpu()
        if bool(d.any()):
            excl = exclusive_outcomes({k: v[self._ids_t].cpu() & d for k, v in terms.items()}, d)
            moves = length[self._ids_t].tolist()
            for j in d.nonzero(as_tuple=False).flatten().tolist():
                if self._live[j]:
                    outcome = vl.outcome_name({k: bool(v[j]) for k, v in excl.items()})
                    self._finish(j, outcome, int(moves[j]))
        vram = _process_vram_mib()
        if vram is not None:
            self.peak_vram_mib = max(self.peak_vram_mib or 0, vram)

    def close(self) -> dict:
        """包みを戻し、終わっていない clip を `unfinished` で閉じ、render product を壊す。manifest の値を返す。"""
        self._sim.__dict__.pop("step", None)
        moves = self._env.episode_length_buf[self._ids_t].tolist()
        for j, live in enumerate(self._live):
            if live:
                self._finish(j, vl.UNFINISHED, int(moves[j]))
        for ann, rp in zip(self._annotators, self._render_products):
            try:
                ann.detach([rp.path])
                rp.destroy()
            except Exception as e:  # noqa: BLE001  後片付けの失敗で評価の結果を捨てない
                print(f"[WARN] render product の後片付けに失敗: {e}")
        return {
            "clips": sorted(self.clips, key=lambda c: c["env_id"]),
            "render_ms_per_frame": 1e3 * self._render_s / self._renders if self._renders else None,
            "renders": self._renders,
            "black_frames": self._black_frames,
            "peak_vram_mib": self.peak_vram_mib,
            "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024,
        }

    # ------------------------------------------------------------------ 内部
    def _capture(self) -> None:
        t0 = time.perf_counter()
        self._sim.render()
        data = [ann.get_data() for ann in self._annotators]
        self._render_s += time.perf_counter() - t0
        self._renders += 1
        phase = self._term._phase[self._ids_t].tolist()
        move = (self._env.episode_length_buf[self._ids_t] + 1).tolist()
        w, h = self.resolution
        for j, i in enumerate(self.env_ids):
            if not self._live[j]:
                continue
            img = np.asarray(data[j])
            if img.size == 0:
                self._black_frames += 1
                img = np.zeros((h, w, 3), dtype=np.uint8)
            rgb = np.ascontiguousarray(img[..., :3])
            lines = vl.overlay_lines(
                self._run, self._iteration, i, move[j], self._max_moves, self._phase_names[phase[j]], self._stage[j]
            )
            self._draw_text(rgb, lines, self._stage_rgb(self._stage[j]))
            self._writers[j].append_data(rgb)
            self._frames[j] += 1
            self._last[j] = rgb

    def _finish(self, j: int, outcome: str, moves: int) -> None:
        """最後のコマを `_ENDING_S` 秒止めて結末を出し、名前を付け替える。"""
        last = self._last[j]
        if last is not None:
            end = last.copy()
            self._draw_text(end, [vl.ending_line(outcome, moves)], None, bottom=True)
            for _ in range(round(self.plan["fps"] * _ENDING_S)):
                self._writers[j].append_data(end)
        self._writers[j].close()
        name = vl.clip_name(self.env_ids[j], outcome, moves)
        os.replace(self._tmp_paths[j], os.path.join(self.out_dir, name))
        self._live[j] = False
        self.clips.append(
            {"env_id": self.env_ids[j], "outcome": outcome, "moves": moves, "file": name, "frames": self._frames[j]}
        )

    @staticmethod
    def _stage_rgb(stage: int) -> tuple[int, int, int]:
        c = STAGE_COLORS[min(max(stage, 0), len(STAGE_COLORS) - 1)]
        return tuple(int(round(255 * v)) for v in c)

    def _draw_text(self, rgb: np.ndarray, lines: list[str], swatch, bottom: bool = False) -> None:
        """暗い帯の上に白い文字。`swatch` (RGB) があれば左端に段の色の四角を描く。"""
        import cv2

        h = rgb.shape[0]
        scale = max(0.4, h / 720.0 * 0.8)
        thick = 1 if h < 600 else 2
        line_h = int(28 * scale / 0.8)
        pad = int(8 * scale / 0.8)
        box_w = line_h if swatch is not None else 0
        top = h - pad - line_h * len(lines) - pad if bottom else 0
        band = rgb[top : top + pad * 2 + line_h * len(lines)]
        band[:] = (band.astype(np.uint16) * 2 // 5).astype(np.uint8)
        if swatch is not None:
            cv2.rectangle(rgb, (pad, top + pad), (pad + box_w - 4, top + pad + box_w - 4), swatch, thickness=-1)
        x = pad + (box_w + pad if swatch is not None else 0)
        for k, line in enumerate(lines):
            y = top + pad + line_h * (k + 1) - int(line_h * 0.25)
            cv2.putText(rgb, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thick, cv2.LINE_AA)
