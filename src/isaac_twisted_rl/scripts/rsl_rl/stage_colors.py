"""Play で、各環境がいまどの段の方策で手を打っているかをグリッパの色で見せる (`play.py` が使う)。

## なぜこの形か

* **段は方策が実際に使う値から取る。** `StageGatedActorCritic` の actor (`GatedMLP`) は観測の末尾の段の
  one-hot の argmax でヘッドを選ぶ (`common/policies/stage_gated.py`)。その `stage_index` を同じ観測に
  掛けるので、色と実際の振り分けはずれない。段 3 (固結び) は最後のヘッドに丸められるのも同じ。
* **環境の設定ではなく play.py の側で塗る。** 版 (`versions/vN/`) の挙動を変えずに済み、新しい版が要らない。
  学習・`eval_chain.py` (headless) には入らない。
* **ロープではなくグリッパを塗る。** ロープの色はプリセットごとに USD に焼き込まれ (`rope_specs.py`)、
  XPBD の紐はマーカーで描くなど紐モデルごとに描き方が違う。グリッパは 3 つの直方体で、物理の材質は
  `materialPurpose="physics"` で bind されているだけなので (`generate_gripper_usd.py`)、見た目は
  `displayColor` だけで決まり、書き換えても物理に影響しない。
* **固結びに達した環境は数手のあいだ白にする。** chain モードの Play は固結びで終わってすぐ reset し、
  段 0 の色に戻るので、そのままでは到達が見えない。到達は reset の前に立った終了項 `stage_up`
  (chain モードでは固結びのときだけ立つ) を `dones` で絞って読む。`get_term` の値は reset でも
  消えないので、絞らないと前のエピソードの到達を数える (`.claude/rules/mdp.md`)。

色の順番を決める関数 (`color_indices` / `update_hold`) は torch だけで書き、Isaac Sim なしで試験する
(`tests/test_stage_colors.py`)。pxr はクラスの中で import する。
"""

from __future__ import annotations

import torch

STAGE_COLORS: tuple[tuple[float, float, float], ...] = (
    (0.15, 0.45, 0.95),  # 段 0: 青
    (0.95, 0.80, 0.10),  # 段 1: 黄
    (0.90, 0.15, 0.15),  # 段 2: 赤
)
"""段 k の方策 (ヘッド k) を使っている間のグリッパの色。ロープ (Knot は紫) と床の灰色に紛れない色にした。"""

STAGE_COLOR_NAMES: tuple[str, ...] = ("青", "黄", "赤")

KNOT_COLOR: tuple[float, float, float] = (0.95, 0.95, 0.95)
"""固結びに達した環境の色 (白)。"""

KNOT_INDEX = len(STAGE_COLORS)
"""色の番号のうち固結びの到達を表すもの (`STAGE_COLORS` の次)。"""

KNOT_HOLD_STEPS = 3
"""固結びに達した後、到達の色を残す手数。"""

KNOT_TERM = "stage_up"
"""chain モードの Play で、固結びに達したときだけ立つ終了項 (`common/mdp/stages.py` の `stage_changed`)。"""


def color_indices(policy_idx: torch.Tensor, hold: torch.Tensor) -> torch.Tensor:
    """この手で見せる色の番号 `(num_envs,)`。到達の色を残している環境は `KNOT_INDEX`、それ以外は方策の番号。"""
    return torch.where(hold > 0, torch.full_like(policy_idx, KNOT_INDEX), policy_idx)


def update_hold(hold: torch.Tensor, knot: torch.Tensor) -> torch.Tensor:
    """1 手を打った後の、到達の色を残す残りの手数。この手で固結びに達した環境は `KNOT_HOLD_STEPS` に戻す。"""
    return torch.where(knot, torch.full_like(hold, KNOT_HOLD_STEPS), (hold - 1).clamp(min=0))


def legend() -> str:
    """起動時に出す凡例。"""
    stages = " / ".join(f"段 {k} の方策 = {name}" for k, name in enumerate(STAGE_COLOR_NAMES))
    return f"グリッパの色: {stages} / 固結びに到達 (以後 {KNOT_HOLD_STEPS} 手) = 白"


class GripperStageColorizer:
    """各環境のグリッパの `displayColor` を、いま使っている方策の段の色に塗り替える。

    `before_step(obs)` を `env.step` の直前に、`after_step(dones)` を直後に呼ぶ。`obs` は reset の後の
    観測なので、色は「この手を打つ方策」を表す。色の番号が前の手から変わった環境だけ書く。

    使えないとき (段ごとの方策でない・色が足りない・グリッパが見つからない・instance proxy で書けない) は
    `enabled` を False にして、呼ばれても何もしない。
    """

    def __init__(self, env, policy_nn, gripper_name: str = "gripper"):
        """`env` は `ManagerBasedRLEnv` (ラッパーを外したもの)、`policy_nn` は `runner.alg.policy`。"""
        self.enabled = False
        actor = getattr(policy_nn, "actor", None)
        if not hasattr(actor, "stage_index"):
            print("[INFO] 段ごとの方策ではないので、グリッパの色分けはしない。")
            return
        if actor.num_heads > len(STAGE_COLORS):
            print(
                f"[WARN] 方策のヘッドが {actor.num_heads} 個で色 ({len(STAGE_COLORS)} 色) が足りない。色分けはしない。"
            )
            return

        from pxr import Usd, UsdGeom

        scene = env.scene
        name = scene[gripper_name].cfg.prim_path.rsplit("/", 1)[-1]
        self._attrs: list[list] = []
        for env_path in scene.env_prim_paths:
            root = scene.stage.GetPrimAtPath(f"{env_path}/{name}")
            gprims = [p for p in Usd.PrimRange(root, Usd.TraverseInstanceProxies()) if p.IsA(UsdGeom.Gprim)]
            if not gprims:
                print(f"[WARN] {env_path}/{name} の下に形状が無い。グリッパの色分けはしない。")
                return
            if any(p.IsInstanceProxy() for p in gprims):
                print(f"[WARN] {env_path}/{name} は instance proxy で色を書けない。グリッパの色分けはしない。")
                return
            self._attrs.append([UsdGeom.Gprim(p).GetDisplayColorAttr() for p in gprims])

        self._policy = policy_nn
        self._env = env
        self._knot_term = KNOT_TERM if KNOT_TERM in env.termination_manager.active_terms else None
        self._colors = [*STAGE_COLORS, KNOT_COLOR]
        self._hold = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self._shown = [-1] * env.num_envs
        self.enabled = True
        print(f"[INFO] {legend()}")
        if self._knot_term is None:
            print(f"[INFO] 終了項 {KNOT_TERM} が無いので、固結びの到達は色で残さない。")

    def before_step(self, obs) -> None:
        """この手を打つ方策の段の色に塗る。"""
        if not self.enabled:
            return
        from pxr import Gf, Vt

        policy_idx = self._policy.actor.stage_index(self._policy.get_actor_obs(obs))
        idx = color_indices(policy_idx, self._hold).tolist()
        for i, (attrs, c) in enumerate(zip(self._attrs, idx)):
            if c == self._shown[i]:
                continue
            value = Vt.Vec3fArray([Gf.Vec3f(*self._colors[c])])
            for attr in attrs:
                attr.Set(value)
            self._shown[i] = c

    def after_step(self, dones: torch.Tensor) -> None:
        """この手で固結びに達した環境に、到達の色を残す手数を入れる。"""
        if not self.enabled:
            return
        knot = torch.zeros_like(self._hold, dtype=torch.bool)
        if self._knot_term is not None:
            knot = dones.bool() & self._env.termination_manager.get_term(self._knot_term)
        self._hold = update_hold(self._hold, knot)
