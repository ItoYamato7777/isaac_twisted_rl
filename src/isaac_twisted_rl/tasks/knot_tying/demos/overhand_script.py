"""台本で固結び (overhand knot) を結ぶ検証 (ゲート 1)。

`next_action.md` §4.8.9 / `paper/Report.md` §7 P0-2 の「台本で固結びが結べるか」を
確かめるスクリプト。**これが通らなければ、報酬をどう設計しても PPO は学習できない。**
結果と結論は `next_action.md` §4.8.10 にまとめてある。

    # 台本 (ハンドを直接動かす)。env 0 は既定値、他は ±25% 揺らす
    uv run python -u src/isaac_twisted_rl/tasks/knot_tying/demos/overhand_script.py \
        --task Isaac-Twisted-RL-Knot-v2-R2 --num_envs 128 --jitter 0.25 --headless

    # 同じ台本を学習と同じ 6 次元の行動に写して env.step() で再生する
    ... --replay_actions [--cfg "actions.pick_place.limits.drop_xy_rel_frac=0.30"]

    # 各手の後の形状を段ごとに保存する (v4 の初期アーカイブ。既定のパスは v4 が読む場所)
    ... --dump_archive

出力 (既定は `outputs/overhand_script/<task>/`):
    summary.log         手ごとの位相クラス (R2 除去後) の分布・発散・最終成功率
    envs.log            環境ごと・手ごとの p-data (R2 除去前後)
    move<k>_<名前>.png  xy 投影図 (交点に O/U と符号)。既定では 4 手目で C2 に達した環境を優先

## 何を確かめるのか

学習環境 (`gym.make`) をそのまま立ち上げ、**シーン・物理・ロープ・グリッパー・
摩擦材は学習と完全に同じもの**を使う。違うのはハンドの動かし方だけ:

* 既定 (台本モード): `PickPlaceAction` を通さず、このスクリプトがハンドを直接動かす。
  1 手の粒度は `PickPlaceAction` と同じ (掴む -> 持ち上げる -> 運ぶ -> 降ろす -> 離す
  -> 退避) で、運搬高さを手ごとに変えられる点だけが違う (既定値は学習と同じ 0.20 m)。
  フェーズの尺はデモ相当のゆっくりした値 (7.8 s/手)。`--fast` で学習環境の尺にする。
* `--replay_actions`: 同じ手を `plan_to_action` で 6 次元の行動へ逆変換し、
  `env.step()` で実行する。置き先が行動の範囲 (target リンク ±8 本 + ±0.15 L) に
  収まらなければクリップし、その数をログに出す。

`--cfg 'a.b=値;...'` で環境設定を上書きできる (フェーズの尺・行動の範囲・物理の
切り分け用)。`durations.*` を変えたときは decimation も作り直す。

## 並列の使い方

全環境が **同じフェーズ時刻** で動き、環境ごとに違うのは台本のパラメータ
(`ScriptParams`) だけ。env 0 は既定値、それ以外は `JITTER x --jitter` の幅で
一様に揺らす。1 回の起動で「既定の台本の成功率」と「どのパラメータが効くか」
(終了時に 3 分位ごとの成功率を出す) が同時に分かる。Isaac Sim の起動に
1 分以上かかるので、1 環境ずつ試すのは遅い。

## 台本 (固結びの組み立て方)

pick&place では **掴んで運んだ部分が必ず上に置かれる** (Report §5.4)。
固結びの p-data は紐に沿って上下が交互 (`1U4 2O5 3U6 4O1 5U2 6O3`) なので、
先端 (リンク番号が大きい側) から見ると「輪の根元で上 -> 輪の片側の下 -> 輪の
反対側の上」を通る必要がある。「下」だけは先端を動かしても作れないので、
**輪の側を掴んで先端の上に被せる** 手が必ず 1 つ要る。台本は次の 5 手:

1. slack   先端から 1/4 のリンクを横へ引き出してロープを曲げる (たるみを作る)
2. loop    同じリンクを本体の上へ渡して輪を作る          -> 交差 X (先端側が上)
3. tail    先端を輪の外側 (edge A の脇) へ回しておく     -> 交差は増えない
4. cover   edge A を掴んで先端の上に被せる               -> 尻尾が edge A の下
5. tuck    先端を輪の中を横切って edge B の上へ渡す      -> 固結び

edge A は輪の根元 X から紐に沿って最初に来る側、edge B は最後に来る側。
3〜5 手目の置き先は **輪の座標系** (原点 X、u = X から輪の中心へ、v = edge A の
ある側。単位はロープ全長 L) で指定し、手を打つたびに今のロープの形から作り直す。
輪の大きさ・向き・鏡像は物理次第で毎回違うので、world 座標や「リンク番号の点を
越える」指定より頑健だった (探索の経緯は `next_action.md` §4.8.10)。

4 手目の後は多くの場合 edge A のたるみが尻尾を **2 回** またいだ状態
(`1U4 2O5 3O6 4O1 5U2 6U3`) になる。R2 除去すると C1 と同じに見えるが、
5 手目で尻尾が片方の下から抜けて固結びになる。本物の C2 (`1U3 2O4 3O1 4U2`) を
経由するのは成功例の一部だけ。
"""

from __future__ import annotations

import argparse
import ast
import math
import os
from dataclasses import dataclass, fields

from isaaclab.app import AppLauncher

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), *[".."] * 5))

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", type=str, default="Isaac-Twisted-RL-Knot-v2-R2", help="学習と同じタスク ID。")
parser.add_argument("--num_envs", type=int, default=32, help="並列に試す台本の数。env 0 は既定パラメータ。")
parser.add_argument("--jitter", type=float, default=1.0, help="パラメータの揺らし幅の倍率 (0 なら全環境同じ)。")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--moves", type=int, default=5, help="実行する手の数 (途中で止めて形を見るとき用)。")
parser.add_argument("--plot_envs", type=int, default=8, help="図に描く環境数。")
parser.add_argument(
    "--plot_ids", type=str, default="", help="図に描く環境をカンマ区切りで指定 (--plot_envs より優先)。"
)
parser.add_argument("--out", type=str, default="", help="出力先 (既定: outputs/overhand_script/<task>)。")
parser.add_argument(
    "--replay_actions",
    action="store_true",
    help="台本の各手を学習と同じ 6 次元の行動に写し、env.step() で実行する (行動空間で表現できるかの検証)。",
)
parser.add_argument(
    "--no_diverged_reset", action="store_true", help="再生時に発散判定による reset を切る (原因の切り分け用)。"
)
parser.add_argument(
    "--fast", action="store_true", help="台本モードでもフェーズの尺を学習環境 (env_cfg.durations) と同じにする。"
)
parser.add_argument(
    "--cfg",
    type=str,
    default="",
    help="環境設定の上書き (切り分け用)。'a.b.c=value;a.b.d=value' (value は Python リテラル)。",
)
parser.add_argument("--override", type=str, default="", help="既定パラメータの上書き。'name=value,name=value'。")
parser.add_argument(
    "--dump_archive",
    type=str,
    nargs="?",
    const=os.path.join(_REPO_ROOT, "outputs", "archive_seed", "overhand_{num_links}links.pt"),
    default="",
    help=(
        "各手の後の形状を段 (v4) ごとに保存する。v4 の初期アーカイブになる。パスを省くと v4 が既定で読む "
        "outputs/archive_seed/overhand_{num_links}links.pt ({num_links} はリンク数に置き換える)。"
    ),
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------- 本体
import gymnasium as gym  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

import isaac_twisted_rl.tasks  # noqa: E402, F401
from isaac_twisted_rl.tasks.knot_tying.common import topology as tp  # noqa: E402
from isaac_twisted_rl.tasks.knot_tying.common import state_archive as sa  # noqa: E402
from isaac_twisted_rl.tasks.knot_tying.common.mdp import phases  # noqa: E402
from isaac_twisted_rl.tasks.knot_tying.common.rope_state import RopeTopologyExtractor  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

_CJK_FONTS = [
    f
    for f in ("Noto Sans CJK JP", "IPAexGothic", "IPAGothic", "Droid Sans Fallback")
    if f in {font.name for font in matplotlib.font_manager.fontManager.ttflist}
]
plt.rcParams["font.family"] = [*_CJK_FONTS, "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# ---------------------------------------------------------------- フェーズの尺 [s]
# デモ (`topology_debug.py`) と同じ余裕のある値。ゲート 1 は「物理的に結べるか」を
# 見るので、学習用に詰めた尺 (phases.SHORT_DURATIONS) は使わない。詰めた尺で
# 通るかは `--replay_actions` 側で確かめる。
SETTLE_TIME = 1.5
APPROACH_TIME = 0.8
DESCEND_TIME = 0.6
CLOSE_TIME = 0.5
HOLD_TIME = 0.3
LIFT_TIME = 0.6
CARRY_TIME = 1.6
LOWER_TIME = 0.6
RELEASE_TIME = 0.4
SETTLE_OUT_TIME = 0.5
CLEAR_TIME = 0.4
RETREAT_TIME = 0.5
RELAX_TIME = 1.0
FINAL_SETTLE_TIME = 1.5


@dataclass
class ScriptParams:
    """台本のパラメータ。長さはロープ全長 L に対する比、高さは直径に対する比。

    各フィールドの既定値が env 0 の値。`JITTER` の幅で他の環境を揺らす。
    """

    # 1. slack: 先端側リンクを (dx, dy) だけ動かして曲げる
    slack_link_from_tail: float = 0.253
    slack_dx: float = -0.266
    slack_dy: float = 0.41
    # 2. loop: 同じリンクを、本体のリンク (over_frac) の向こうへ渡す
    loop_over_frac: float = 0.299
    loop_extra: float = 0.065
    loop_drop: float = 1.0
    # 3. tail: 先端を輪の座標 (tail_u, tail_v) へ。v > 0 が edge A の外側
    tail_link_from_tail: float = 0.07
    tail_u: float = 0.075
    tail_v: float = 0.12
    tail_carry: float = 6.25
    # 4. cover: edge A (輪の lo 側から cover_edge_frac) を掴んで (cover_u, cover_v) へ被せる
    cover_edge_frac: float = 0.315
    cover_u: float = 0.105
    cover_v: float = 0.215
    cover_carry: float = 6.25
    cover_drop: float = 1.0
    # 5. tuck: 先端を (tuck_u, tuck_v) へ。v < 0 が edge B の外側
    tuck_link_from_tail: float = 0.07
    tuck_u: float = 0.075
    tuck_v: float = -0.117
    tuck_carry: float = 6.25
    tuck_drop: float = 1.0
    # 共通: 運搬高さ (直径比。指先の高さ)。0.20 m の手のひら = 指先 0.125 m に相当する値が既定
    carry: float = 6.25


JITTER = {
    "slack_link_from_tail": 0.08,
    "slack_dx": 0.08,
    "slack_dy": 0.08,
    "loop_over_frac": 0.08,
    "loop_extra": 0.04,
    "tail_u": 0.08,
    "tail_v": 0.06,
    "cover_edge_frac": 0.12,
    "cover_u": 0.08,
    "cover_v": 0.07,
    "tuck_u": 0.08,
    "tuck_v": 0.07,
}
"""揺らし幅 (± 一様)。ここに無いパラメータは全環境で既定値のまま。"""


def make_params(num_envs: int, device, jitter: float, seed: int, override: str) -> dict[str, torch.Tensor]:
    """環境ごとのパラメータ `{name: (num_envs,)}`。env 0 は既定値 (+ override)。"""
    base = ScriptParams()
    for item in filter(None, override.split(",")):
        k, v = item.split("=")
        setattr(base, k.strip(), float(v))
    gen = torch.Generator(device="cpu").manual_seed(seed)
    out = {}
    for f in fields(ScriptParams):
        nominal = float(getattr(base, f.name))
        val = torch.full((num_envs,), nominal)
        width = JITTER.get(f.name, 0.0) * jitter
        if width > 0 and num_envs > 1:
            noise = (torch.rand(num_envs - 1, generator=gen) * 2.0 - 1.0) * width
            val[1:] += noise
        out[f.name] = val.to(device)
    return out


# ---------------------------------------------------------------- 1 手の指示
@dataclass
class MovePlan:
    """全環境ぶんの 1 手 (env ごとのテンソル)。座標は env ローカル。"""

    label: str
    grasp_link: torch.Tensor  # (N,) long
    way: list[torch.Tensor]  # 運搬の経由点 [(N, 2), ...]。最後が置き場所
    place_z: torch.Tensor  # (N,) 置くときの手のひら高さ [m]
    carry_z: torch.Tensor  # (N,) 運搬時の手のひら高さ [m]
    valid: torch.Tensor  # (N,) bool。False の環境はハンドを動かさない


class ScriptRunner:
    """学習環境のシーンを借りて、台本どおりにハンドを動かす。"""

    def __init__(self, env, out_dir: str):
        self.env = env
        self.uenv = env.unwrapped
        self.scene = self.uenv.scene
        self.sim = self.uenv.sim
        self.rope = self.scene["rope"]
        self.gripper = self.scene["gripper"]
        self.device = self.uenv.device
        self.num_envs = self.uenv.num_envs
        self.dt = self.sim.get_physics_dt()
        self.origins = self.scene.env_origins
        self.out_dir = out_dir

        pp = self.uenv.cfg.actions.pick_place
        self.rope_length = float(pp.rope_length)
        self.diameter = float(pp.rope_diameter)

        self.extractor = RopeTopologyExtractor(self.rope, env_origins=self.origins)
        self.body_index = self.extractor.body_index
        self.num_links = self.extractor.num_links

        self.finger_ids, _ = self.gripper.find_joints(["left_finger_joint", "right_finger_joint"], preserve_order=True)
        limits = self.gripper.data.joint_pos_limits[:, self.finger_ids, :]
        lower, upper = limits[..., 0], limits[..., 1]
        self.open_value = torch.where(lower.abs() > upper.abs(), lower, upper)

        n = self.num_envs
        self.hand_pos = torch.zeros(n, 3, device=self.device)
        self.hand_pos[:, 2] = phases.HOME_PALM_Z
        self.hand_yaw = torch.zeros(n, device=self.device)
        self.diverged = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.div_phase: dict[str, int] = {}
        """初めて発散を検出したフェーズごとの環境数 (原因の切り分け用)。"""
        self._pose = torch.zeros(n, 7, device=self.device)
        self._vel = torch.zeros(n, 6, device=self.device)

    # ------------------------------------------------------------ 物理
    def links(self) -> torch.Tensor:
        """全リンク中心 `(N, num_links, 3)` (env ローカル)。"""
        pos = self.rope.data.body_pos_w[:, self.body_index]
        return pos - self.origins.unsqueeze(1)

    def _write_hand(self, pos, yaw, vel, yaw_rate):
        self._pose[:, 0:3] = pos + self.origins
        self._pose[:, 3] = torch.cos(0.5 * yaw)
        self._pose[:, 4:6] = 0.0
        self._pose[:, 6] = torch.sin(0.5 * yaw)
        self.gripper.write_root_pose_to_sim(self._pose)
        self._vel[:, 0:3] = vel
        self._vel[:, 3:5] = 0.0
        self._vel[:, 5] = yaw_rate
        self.gripper.write_root_velocity_to_sim(self._vel)

    def segment(self, p1, yaw1, r0, r1, duration: float, tag: str = ""):
        """全環境のハンドを現在位置 -> p1 へ smoothstep で動かす (`topology_debug.run_segment` と同じ)。"""
        p0 = self.hand_pos.clone()
        yaw0 = self.hand_yaw.clone()
        steps = max(int(round(duration / self.dt)), 1)
        d_yaw = torch.remainder(yaw1 - yaw0 + math.pi, 2.0 * math.pi) - math.pi
        for k in range(steps):
            u = (k + 1) / steps
            s = u * u * (3.0 - 2.0 * u)
            ds = 6.0 * u * (1.0 - u) / duration
            pos = p0 + (p1 - p0) * s
            vel = (p1 - p0) * ds
            self._write_hand(pos, yaw0 + d_yaw * s, vel, d_yaw * ds)
            ratio = r0 + (r1 - r0) * s
            self.gripper.set_joint_position_target(ratio * self.open_value, joint_ids=self.finger_ids)
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(self.dt)
            if self.sim.has_gui() and k % 8 == 0:
                self.sim.render()
        self.hand_pos = p1.clone()
        self.hand_yaw = yaw0 + d_yaw
        links = self.links()
        speed = self.rope.data.body_lin_vel_w.norm(dim=-1).amax(dim=1)
        # 学習環境の rope_diverged (v2-R2) と同じ基準: NaN / 水平 3 m / リンク速度 20 m/s
        bad = ~torch.isfinite(links).all(dim=-1).all(dim=-1)
        bad |= torch.nan_to_num(links[..., :2], nan=0.0).norm(dim=-1).amax(dim=1) > 3.0
        bad |= torch.nan_to_num(speed, nan=0.0) > 20.0
        new = bad & ~self.diverged
        if tag and bool(new.any()):
            self.div_phase[tag] = self.div_phase.get(tag, 0) + int(new.sum())
        self.diverged |= bad

    def hold(self, duration: float, ratio: float = phases.OPEN_RATIO, tag: str = ""):
        self.segment(self.hand_pos.clone(), self.hand_yaw.clone(), ratio, ratio, duration, tag)

    def tangent_yaw(self, links: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        lo = (ids - 1).clamp(0, self.num_links - 1)
        hi = (ids + 1).clamp(0, self.num_links - 1)
        d = gather(links, hi) - gather(links, lo)
        return torch.atan2(d[:, 1], d[:, 0])

    def execute(self, plan: MovePlan):
        """1 手を実行する。`plan.valid` が False の環境はハンドをその場に留める。"""
        links = self.links()
        v = plan.valid & ~self.diverged
        vv = v.unsqueeze(-1)
        gpos = gather(links, plan.grasp_link)
        g_yaw = torch.where(v, self.tangent_yaw(links, plan.grasp_link), self.hand_yaw)
        OPEN, CLOSED = phases.OPEN_RATIO, phases.CLOSE_RATIO

        def at(xy: torch.Tensor, z) -> torch.Tensor:
            z = z if torch.is_tensor(z) else torch.full_like(xy[:, 0], z)
            p = torch.cat([xy, z.unsqueeze(-1)], dim=-1)
            return torch.where(vv, p, self.hand_pos)

        above = at(gpos[:, :2], phases.HOME_PALM_Z)
        grasp = at(gpos[:, :2], phases.GRASP_PALM_Z)
        lift = at(gpos[:, :2], plan.carry_z)
        self.segment(above, g_yaw, OPEN, OPEN, APPROACH_TIME, "approach")
        self.segment(grasp, g_yaw, OPEN, OPEN, DESCEND_TIME, "descend")
        self.segment(grasp, g_yaw, OPEN, CLOSED, CLOSE_TIME, "close")
        self.hold(HOLD_TIME, CLOSED, "hold")
        self.segment(lift, g_yaw, CLOSED, CLOSED, LIFT_TIME, "lift")
        prev = gpos[:, :2]
        for w in plan.way:
            self.segment(at(w, plan.carry_z), g_yaw, CLOSED, CLOSED, CARRY_TIME / len(plan.way), "carry")
            prev_last = prev
            prev = w
        place_xy = plan.way[-1]
        self.segment(at(place_xy, plan.place_z), g_yaw, CLOSED, CLOSED, LOWER_TIME, "lower")
        self.segment(self.hand_pos.clone(), g_yaw, CLOSED, OPEN, RELEASE_TIME, "release")
        self.hold(SETTLE_OUT_TIME, tag="settle_out")
        # 運んできた向きへ横へ抜けてから上げる (topology_debug の clear と同じ理由)
        d = place_xy - prev_last
        d = d / d.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        clear_xy = place_xy + d * 4.0 * self.diameter
        self.segment(at(clear_xy, plan.place_z), g_yaw, OPEN, OPEN, CLEAR_TIME, "clear")
        self.segment(at(clear_xy, phases.HOME_PALM_Z), g_yaw, OPEN, OPEN, RETREAT_TIME, "retreat")
        self.hold(RELAX_TIME, tag="relax")

    def sync_hand(self):
        """`env.step()` でハンドを動かした後、台本側の現在位置をシミュレータに合わせる。"""
        root = self.gripper.data.root_state_w
        self.hand_pos = root[:, 0:3] - self.origins
        self.hand_yaw = 2.0 * torch.atan2(root[:, 6], root[:, 3])

    def park(self):
        """ハンドをロープから遠ざけて落ち着かせる (最終判定の前)。"""
        park = torch.zeros_like(self.hand_pos)
        park[:, 1] = -0.9 * self.rope_length
        park[:, 2] = phases.HOME_PALM_Z
        self.segment(park, self.hand_yaw.clone(), phases.OPEN_RATIO, phases.OPEN_RATIO, 1.5)
        self.hold(FINAL_SETTLE_TIME)

    # ------------------------------------------------------------ 位相
    def topology(self):
        """現在の (batch, reduced, class, class_r2, p_data, p_data_r2, zmax)。"""
        self.extractor.update(self.rope)
        batch = self.extractor.batch
        red = self.extractor.reduced_batch
        cls = tp.topology_class(batch)
        cls_r2 = tp.topology_class(red)
        return batch, red, cls, cls_r2, self.extractor.p_data(), self.extractor.p_data_r2()

    def dump(self, archive: sa.StateArchive, batch, red) -> torch.Tensor:
        """いまの形状のうち、床に落ちていて発散していないものを段ごとに `archive` へ入れる (`--dump_archive`)。

        判定は v4 の `ResetToStage.record` と同じ (段 = `BfsPotential.stage`、床 = 最高点が直径の 3 倍以下)。
        Returns: 段 `(N,)` (入れなかった env は −1)。
        """
        stage = tp.get_bfs_potential(self.device).stage(batch, red)
        links_w = self.rope.data.body_pos_w[:, self.body_index]
        ok = torch.isfinite(links_w).all(dim=-1).all(dim=-1) & ~self.diverged
        ok &= torch.nan_to_num(self.links()[..., 2], nan=1e9).amax(dim=1) <= 3.0 * self.diameter
        rows = ok.nonzero(as_tuple=False).squeeze(-1)
        if rows.numel() > 0:
            rel = sa.to_centroid_frame(self.rope.data.root_link_pose_w[rows], links_w[rows], self.origins[rows])
            archive.add(stage[rows], rel, self.rope.data.joint_pos[rows])
        return torch.where(ok, stage, torch.full_like(stage, -1))


def gather(links: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """`(N, L, 3)` から env ごとに 1 リンク `(N, 3)`。"""
    return links.gather(1, ids.view(-1, 1, 1).expand(-1, 1, links.shape[-1])).squeeze(1)


def unit(v: torch.Tensor) -> torch.Tensor:
    return v / v.norm(dim=-1, keepdim=True).clamp_min(1e-6)


# ---------------------------------------------------------------- 台本
def loop_crossing(batch: tp.IntersectionBatch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """輪の根元の交差 X を探す。`(valid, seg_lo, seg_hi)`。

    交差が 1 つ以上あれば、紐に沿って最も外側 (seg_hi - seg_lo が最大) の交差を
    輪の根元とみなす。輪の上に余計な交差が乗っていても根元を取り違えないため。
    """
    data = batch.data
    valid = batch.valid_mask
    lo = torch.minimum(data[..., 0], data[..., 1])
    hi = torch.maximum(data[..., 0], data[..., 1])
    span = torch.where(valid, hi - lo, torch.full_like(hi, -1))
    best = span.argmax(dim=1)
    rows = torch.arange(data.shape[0], device=data.device)
    ok = valid.any(dim=1)
    return ok, lo[rows, best].clamp_min(0), hi[rows, best].clamp_min(0)


def build_move(k: int, runner: ScriptRunner, P: dict[str, torch.Tensor], state: dict) -> MovePlan:
    """k 手目 (0 始まり) の指示を今のロープ姿勢から組み立てる。"""
    links = runner.links()
    xy = links[..., :2]
    n = runner.num_links
    L = runner.rope_length
    dia = runner.diameter
    N = runner.num_envs
    dev = runner.device
    ones = torch.ones(N, dtype=torch.bool, device=dev)
    margin = 2

    def link_from_tail(frac: torch.Tensor) -> torch.Tensor:
        # 先端からの距離をリンク数ではなく全長比で持つ (プリセット間で台本を共有するため)
        return (n - 1 - (frac * (n - 1)).round().long()).clamp(margin, n - 1 - margin)

    def palm(diameters: torch.Tensor) -> torch.Tensor:
        # 直径比の「指先の高さ」-> 手のひら高さ
        return phases.GRASP_PALM_Z + diameters * dia

    carry_default = palm(P["carry"])

    if k == 0:
        g = link_from_tail(P["slack_link_from_tail"])
        target = gather(xy, g) + torch.stack([P["slack_dx"], P["slack_dy"]], dim=-1) * L
        state["loop_link"] = g
        return MovePlan("slack", g, [target], palm(torch.zeros(N, device=dev)), carry_default, ones)

    if k == 1:
        g = state["loop_link"]
        over = (P["loop_over_frac"] * (n - 1)).round().long().clamp(0, n - 1)
        o = gather(xy, over)
        target = o + unit(o - gather(xy, g)) * P["loop_extra"].unsqueeze(-1) * L
        return MovePlan("loop", g, [target], palm(P["loop_drop"]), carry_default, ones)

    if "x" not in state:
        # 輪の根元 X は 2 手目の直後に 1 回だけ決める。リンク番号はロープが動いても
        # 変わらないので、以降の手はその番号を使い回す。被せた後は交差が増えて
        # 「最も外側の交差」が X でなくなる (edge A と先端の交差のほうが広い) ため。
        # セグメント s はリンク s そのもの (節点 s, s+1 がリンク s の両端の関節)。
        ok, seg_lo, seg_hi = loop_crossing(runner.extractor.batch)
        state["x"] = (ok, seg_lo.clamp(0, n - 1), seg_hi.clamp(0, n - 1))
    ok, x_lo, x_hi = state["x"]
    loop_len = (x_hi - x_lo).clamp_min(1)
    center = xy.gather(1, torch.stack([x_lo, x_hi], 1).unsqueeze(-1).expand(-1, -1, 2)).mean(1)
    # 輪の中心 = X から輪の上の点の平均
    idx = torch.arange(n, device=dev).unsqueeze(0)
    in_loop = (idx > x_lo.unsqueeze(1)) & (idx < x_hi.unsqueeze(1))
    loop_c = (xy * in_loop.unsqueeze(-1)).sum(1) / in_loop.sum(1, keepdim=True).clamp_min(1)

    def loop_link(frac: torch.Tensor) -> torch.Tensor:
        return (x_lo + (frac * loop_len).round().long()).clamp(0, n - 1)

    # -- 輪の座標系。原点 = X、u = X から輪の中心へ、v = edge A のある側 (単位はロープ全長 L)。
    #    3〜5 手目の置き先はこの座標で指定する。輪の大きさや向き (鏡像を含む) は物理次第で
    #    毎回違うので、world 座標で決め打ちするより頑健。手ごとに今の姿勢から作り直す。
    batch = runner.extractor.batch
    seg = batch.data[..., :2]
    lo_s = torch.minimum(seg[..., 0], seg[..., 1])
    hi_s = torch.maximum(seg[..., 0], seg[..., 1])
    is_x = batch.valid_mask & (lo_s == x_lo.unsqueeze(1)) & (hi_s == x_hi.unsqueeze(1))
    cxy = runner.extractor.crossing_positions()
    rows = torch.arange(N, device=dev)
    xp = cxy[rows, is_x.float().argmax(dim=1)]
    xp = torch.where(is_x.any(dim=1, keepdim=True) & torch.isfinite(xp), xp, center)
    u_ax = unit(loop_c - xp)
    perp = torch.stack([-u_ax[:, 1], u_ax[:, 0]], dim=-1)
    half = x_lo + loop_len // 2
    in_a = (idx > x_lo.unsqueeze(1)) & (idx <= half.unsqueeze(1))
    a_c = (xy * in_a.unsqueeze(-1)).sum(1) / in_a.sum(1, keepdim=True).clamp_min(1)
    side = torch.sign(((a_c - xp) * perp).sum(-1, keepdim=True))
    v_ax = perp * torch.where(side == 0, torch.ones_like(side), side)

    def frame(uu: torch.Tensor, vv: torch.Tensor) -> torch.Tensor:
        return xp + (uu.unsqueeze(-1) * u_ax + vv.unsqueeze(-1) * v_ax) * L

    zero = torch.zeros(N, device=dev)
    if k == 2:
        # tail: 先端を edge A の外側 (v > 0) へ回しておく。交差は増やさない。
        g = link_from_tail(P["tail_link_from_tail"])
        return MovePlan("tail", g, [frame(P["tail_u"], P["tail_v"])], palm(zero), palm(P["tail_carry"]), ok)

    if k == 3:
        # cover: edge A を掴んで、先端より外側 (v がさらに大きい所) へ被せる。
        g = loop_link(P["cover_edge_frac"])
        target = frame(P["cover_u"], P["cover_v"])
        return MovePlan("cover", g, [target], palm(P["cover_drop"]), palm(P["cover_carry"]), ok)

    if k == 4:
        # tuck: 先端を edge B の外側 (v < 0) へ渡す。輪の中を横切って edge B の上を越える。
        g = link_from_tail(P["tuck_link_from_tail"])
        target = frame(P["tuck_u"], P["tuck_v"])
        return MovePlan("tuck", g, [target], palm(P["tuck_drop"]), palm(P["tuck_carry"]), ok)

    raise IndexError(k)


# ---------------------------------------------------------------- 6 次元の行動への写像
def plan_to_action(plan: MovePlan, runner: ScriptRunner) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """1 手を `PickPlaceAction` の 6 次元行動 `[-1, 1]^6` に逆変換する。

    `PickPlaceAction.process_actions` の変換をそのまま逆にたどる。表現できない分は
    クリップし、どのチャネルで何環境がはみ出したかを返す:

    * 置き先は「target リンク (grasp ± 8) の現在位置 + ±0.15 L」でしか書けない。
      17 本の候補から必要な変位が最も小さいリンクを選び、それでも届かなければクリップ。
    * 経由点と運搬高さは行動に無い (運搬は常に手のひら 0.20 m)。ここでは表現できない
      ものとして捨て、その影響は再生の結果に現れる。
    * end_yaw は掴んだ姿勢のまま (台本は運搬中に手首を回さない)。
    """
    term = runner.uenv.action_manager.get_term("pick_place")
    links = runner.links()
    n = runner.num_links
    g = plan.grasp_link.clamp(term._grasp_lo, term._grasp_hi)
    span = term._grasp_hi - term._grasp_lo
    a0 = 2.0 * (g - term._grasp_lo).float() / span - 1.0

    place = plan.way[-1]  # (N, 2)
    offsets = torch.arange(-term._max_delta, term._max_delta + 1, device=runner.device)
    cand = (g.unsqueeze(1) + offsets.unsqueeze(0)).clamp(0, n - 1)  # (N, 17)
    cand_xy = links[..., :2].gather(1, cand.unsqueeze(-1).expand(-1, -1, 2))
    need = (place.unsqueeze(1) - cand_xy) / term._drop_xy_max  # (N, 17, 2)
    best = need.abs().amax(dim=-1).argmin(dim=1)
    rows = torch.arange(runner.num_envs, device=runner.device)
    t = cand[rows, best]
    a1 = (t - g).float() / term._max_delta
    a23 = need[rows, best]
    xy_clip = a23.abs().amax(dim=-1) > 1.0

    a4 = 2.0 * (plan.place_z - phases.GRASP_PALM_Z) / term._drop_z_max - 1.0
    a5 = runner.tangent_yaw(links, g) / math.pi
    act = torch.stack([a0, a1, a23[:, 0], a23[:, 1], a4, a5], dim=-1)
    info = {
        "xy_clip": xy_clip,
        "z_clip": a4.abs() > 1.0,
        "carry_mismatch": (plan.carry_z - phases.CARRY_PALM_Z).abs() > 1e-3,
        "via_points": torch.full_like(xy_clip, len(plan.way) > 1),
    }
    return act.clamp(-1.0, 1.0), info


# ---------------------------------------------------------------- 出力
def crossing_label(over: int, sign: int, pi: int, pj: int) -> str:
    o, u = (pi, pj) if over == 1 else (pj, pi)
    return f"{o}O{u}{'+' if sign > 0 else '-'}"


def snapshot(runner: ScriptRunner, cls_r2, p_data) -> dict:
    """図を後から描くための CPU コピー。"""
    return {
        "nodes": runner.extractor.nodes.cpu().numpy(),
        "batch": runner.extractor.batch.to("cpu"),
        "cxy": runner.extractor.crossing_positions().cpu().numpy(),
        "cls_r2": cls_r2.cpu().tolist(),
        "p_data": list(p_data),
    }


def plot_envs(path: str, snap: dict, title: str, ids: list[int]):
    nodes, batch, cxy = snap["nodes"], snap["batch"], snap["cxy"]
    cls_r2, p_data = snap["cls_r2"], snap["p_data"]
    count = len(ids)
    cols = min(4, count)
    rows = (count + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.2 * rows), squeeze=False)
    for i, e in enumerate(ids):
        ax = axes[i // cols][i % cols]
        xy, z = nodes[e, :, :2], nodes[e, :, 2]
        ax.plot(xy[:, 0], xy[:, 1], "-", color="0.6", lw=1.0)
        ax.scatter(xy[:, 0], xy[:, 1], c=z, cmap="viridis", s=10, vmin=0.0, vmax=max(float(z.max()), 0.08))
        ax.plot(xy[0, 0], xy[0, 1], "s", color="tab:blue", ms=7)
        ax.plot(xy[-1, 0], xy[-1, 1], "^", color="tab:red", ms=7)
        for m in range(int(batch.count[e])):
            _, _, over, sign = (int(v) for v in batch.data[e, m].tolist())
            pi, pj = (int(v) for v in batch.order[e, m].tolist())
            ax.plot(cxy[e, m, 0], cxy[e, m, 1], "x", color="crimson", ms=9, mew=2)
            ax.annotate(
                crossing_label(over, sign, pi, pj),
                cxy[e, m],
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                color="crimson",
            )
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title(f"env {e}  class_r2={int(cls_r2[e])}\n{p_data[e][:40]}", fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


class Log:
    def __init__(self, path: str):
        self.path = path
        open(path, "w").close()

    def __call__(self, text: str = ""):
        print(text, flush=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(text + "\n")


def main():
    task_short = args_cli.task.replace("Isaac-Twisted-RL-", "")
    out_dir = args_cli.out or os.path.join(_REPO_ROOT, "outputs", "overhand_script", task_short)
    os.makedirs(out_dir, exist_ok=True)
    log = Log(os.path.join(out_dir, "summary.log"))
    elog = Log(os.path.join(out_dir, "envs.log"))

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    # 台本はハンドを直接動かすので env.step() を呼ばない。時間切れの reset も起きないが念のため延ばす。
    env_cfg.episode_length_s = 1.0e6
    for item in filter(None, args_cli.cfg.split(";")):
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
    if "durations." in args_cli.cfg:
        # decimation (= 1 手の物理ステップ数) は __post_init__ で尺から計算済みなので作り直す。
        env_cfg.actions.pick_place.durations = env_cfg.durations
        env_cfg.decimation = env_cfg.durations.decimation(env_cfg.sim.dt)
        env_cfg.sim.render_interval = env_cfg.decimation
    if args_cli.replay_actions:
        # 発散した環境だけが reset される (その環境は失敗として数える)。
        mode = "replay (env.step, 6D action)"
        if args_cli.no_diverged_reset:
            # 項そのものは報酬が参照するので消さず、NaN 以外の閾値だけ外す。
            env_cfg.terminations.diverged.params = {"max_height": 1.0e9}
            mode += " no_diverged_reset"
    else:
        mode = "script (direct hand control)"
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()

    runner = ScriptRunner(env, out_dir)
    if args_cli.fast:
        # 学習環境の尺に揃える (尺 0 のフェーズは 1 物理ステップになる。PickPlaceAction と同じ)。
        d = env.unwrapped.cfg.durations
        g = globals()
        for name, val in (
            ("APPROACH_TIME", d.approach),
            ("DESCEND_TIME", d.descend),
            ("CLOSE_TIME", d.close),
            ("HOLD_TIME", d.hold),
            ("LIFT_TIME", d.lift),
            ("CARRY_TIME", d.carry),
            ("LOWER_TIME", d.lower),
            ("RELEASE_TIME", d.release),
            ("SETTLE_OUT_TIME", d.settle_out),
            ("CLEAR_TIME", d.clear),
            ("RETREAT_TIME", d.retreat),
            ("RELAX_TIME", d.relax),
        ):
            g[name] = max(val, runner.dt)
    P = make_params(runner.num_envs, runner.device, args_cli.jitter, args_cli.seed, args_cli.override)
    log(
        f"=== overhand_script task={args_cli.task} num_envs={runner.num_envs} links={runner.num_links} "
        f"L={runner.rope_length:.3f} dia={runner.diameter * 1000:.1f}mm jitter={args_cli.jitter} mode={mode}"
    )

    runner.hold(SETTLE_TIME)
    dump = None
    if args_cli.dump_archive:
        dump = sa.StateArchive(
            classes=(0, 1, 2),
            capacity=runner.num_envs * (args_cli.moves + 1),
            num_joints=runner.rope.num_joints,
            device=runner.device,
        )
    state: dict = {}
    snaps: list[tuple[str, dict]] = []
    if args_cli.plot_ids:
        plot_n = [int(v) for v in args_cli.plot_ids.split(",")]
    else:
        plot_n = list(range(min(args_cli.plot_envs, runner.num_envs)))
    for k in range(args_cli.moves):
        plan = build_move(k, runner, P, state)
        if args_cli.replay_actions:
            act, info = plan_to_action(plan, runner)
            # 無効な環境 (輪ができなかった等) も何かしら打たねばならないので、そのまま打つ。
            _, _, terminated, truncated, _ = env.step(act)
            runner.diverged |= terminated | truncated  # reset されたロープは初期姿勢に戻っている
            runner.sync_hand()
            lv = runner.rope.data.body_lin_vel_w.norm(dim=-1).amax(dim=1)
            rxy = runner.links()[..., :2].norm(dim=-1).amax(dim=1)
            log(
                f"    rope: max link speed median={float(lv.nanmedian()):.2f} max={float(lv.nan_to_num(1e9).max()):.2f} "
                f">20m/s={int((lv > 20).sum())} | max xy median={float(rxy.nanmedian()):.2f} >3m={int((rxy > 3).sum())} "
                f"nonfinite={int((~torch.isfinite(runner.links()).all(-1).all(-1)).sum())}"
            )
            log(
                f"    action: xy clip={int(info['xy_clip'].sum())} z clip={int(info['z_clip'].sum())} "
                f"carry mismatch={int(info['carry_mismatch'].sum())} via={int(info['via_points'].sum())} "
                f"reset: terminated={int(terminated.sum())} truncated={int(truncated.sum())}"
            )
        else:
            runner.execute(plan)
        batch, red, cls, cls_r2, pd, pd_r2 = runner.topology()
        if dump is not None:
            st = runner.dump(dump, batch, red)
            log(f"    dump: stage hist={torch.bincount(st[st >= 0], minlength=4).tolist()} (段 3 は保存しない)")
        hist = torch.bincount(cls_r2[~runner.diverged], minlength=4).tolist()
        log(
            f"[move {k + 1} {plan.label:6s}] class_r2 hist={hist} diverged={int(runner.diverged.sum())} "
            f"invalid={int((~plan.valid).sum())} diverged_at={runner.div_phase}"
        )
        for e in range(runner.num_envs):
            elog(
                f"move{k + 1} env{e:3d} cls={int(cls[e])} cls_r2={int(cls_r2[e])} n={int(batch.count[e])} "
                f"div={bool(runner.diverged[e])} p='{pd[e]}' r2='{pd_r2[e]}'"
            )
        snaps.append((f"move{k + 1}_{plan.label}", snapshot(runner, cls_r2, pd)))

    runner.park()
    batch, red, cls, cls_r2, pd, pd_r2 = runner.topology()
    zmax = runner.links()[..., 2].amax(dim=1)
    ok = (cls_r2 == tp.CLASS_OVERHAND) & ~runner.diverged
    snaps.append(("final", snapshot(runner, cls_r2, pd)))
    if dump is not None:
        runner.dump(dump, batch, red)
        path = args_cli.dump_archive.format(num_links=runner.num_links)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        dump.save(path)
        log(f"[dump] {path}: 段 -> 件数 {dump.sizes()}")
    # 既定では「4 手目で C2 に達した環境」を優先して描く (5 手目の成否を見比べるため)。
    if not args_cli.plot_ids and len(snaps) >= 5:
        c2 = [e for e, c in enumerate(snaps[3][1]["cls_r2"]) if c == tp.CLASS_INTERLEAVED2]
        plot_n = (c2 + [e for e in range(runner.num_envs) if e not in c2])[: len(plot_n)]
    for name, snap in snaps:
        plot_envs(os.path.join(out_dir, f"{name}.png"), snap, f"{args_cli.task} {name}", plot_n)
    log(
        f"[final] class_r2 hist={torch.bincount(cls_r2[~runner.diverged], minlength=4).tolist()} "
        f"overhand={int(ok.sum())}/{runner.num_envs} diverged={int(runner.diverged.sum())} "
        f"zmax_max={float(torch.nan_to_num(zmax, nan=-1).max()):.3f}"
    )
    for e in range(runner.num_envs):
        elog(f"final env{e:3d} cls_r2={int(cls_r2[e])} p='{pd[e]}' r2='{pd_r2[e]}'")
    # パラメータの値を 3 分位に分けたときの成功率 (探索の手がかり)。
    if args_cli.jitter > 0 and runner.num_envs >= 30:
        log("パラメータ別の成功率 (値の小 / 中 / 大 の 3 分位):")
        for name in JITTER:
            vals = P[name]
            order = vals.argsort()
            thirds = order.chunk(3)
            rates = [float(ok[t].float().mean()) for t in thirds]
            edges = [float(vals[t].mean()) for t in thirds]
            log(f"  {name:22s} " + "  ".join(f"{e:+.3f}:{r * 100:4.0f}%" for e, r in zip(edges, rates)))
    if ok.any():
        log("成功した環境のパラメータ:")
        for e in ok.nonzero().flatten().tolist()[:16]:
            log(f"  env{e}: " + ", ".join(f"{name}={float(v[e]):.3f}" for name, v in P.items() if name in JITTER))
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
