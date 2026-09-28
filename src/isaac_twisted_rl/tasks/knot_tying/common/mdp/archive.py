"""到達状態アーカイブからのリセット (Go-Explore 型)。v3 で追加。

`paper/Report.md` §7 P1-1。アーカイブ本体 (`../state_archive.py`) を Isaac Lab に
繋ぐ 3 つの項:

| 項 | 種類 | 役割 |
|---|---|---|
| `ResetFromArchive` | event (`mode="reset"`) | アーカイブ本体を持ち、reset 時に確率 `prob` で保存済みの形状を書き込む |
| `record_archive` | 報酬項 (重み 1.0、常に 0) | 毎手の終わりに、条件を満たす形状をアーカイブへ入れる |
| `log_archive_stats` | カリキュラム項 | アーカイブの大きさと、開始の種別ごとの成功率を TensorBoard へ出す |

保存を報酬項でやるのは、`ManagerBasedRLEnv.step()` の中で「手が終わった後・reset の
前」に毎手呼ばれるのが報酬項だからで、`topology_log.TopologyStats` と同じ流儀。
**重みを 0 にしないこと** (0 だと `RewardManager` が呼ばない)。

## reset の順序

`_reset_idx` は `event(reset) -> observation -> action.reset -> ...` の順に回る。
event の `reset_all` (基底 `EventCfg`) がロープとハンドを初期姿勢へ戻し、その後に
この `reset_from_archive` が選んだ env のロープだけを上書きする (**cfg の並び順 =
適用順** なので、`reset_all` より後ろに置くこと)。`PickPlaceAction.reset()` が
`invalidate_extractor` を呼ぶので、直後の観測は書き込んだ形状で計算される。
関節を書き込んだ後に `body_pos_w` を読むと Isaac Lab が FK を回し直す
(`articulation_data.body_link_pose_w`) ので、復元した位相をその場で検算できる
(`restore_mismatch`)。

## 評価の指標

アーカイブから始めたエピソードは途中の段から始まるので、成功率が水増しされる。
**方策の実力は `scratch_success` (まっすぐなロープから始めたエピソードが
固結びで終わった割合) で見ること。** Play は `prob = 0` にしてある。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg

from .. import topology as tp
from .. import state_archive as sa
from .observations import get_extractor, rope_zmax
from .terminations import diverged_now

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class ResetFromArchive(ManagerTermBase):
    """reset 時に確率 `prob` でアーカイブの形状から始める event 項。アーカイブ本体も持つ。

    params (すべて `__call__` の引数。Isaac Lab は params と `__call__` の引数が一致することを要求する):
        prob: アーカイブから始める確率。0 なら常にまっすぐなロープ (Play / 切り分け用)。
        capacity: 1 クラスあたりの保存件数。
        archive_classes: 保存するクラス。class 3 (固結び) は入れない (そこから始めても学ぶことが無い)。
        class_weights: `archive_classes` と同じ並びの抽出の重み。TWISTED に倣い、交差の多い段を優先する。
        random_yaw: 復元時に重心まわりに一様乱数で回すか (投影を回すだけなので位相は変わらない)。
        success_term: 成功で終わったかを読む終了項の名前 (成功率のログ用)。無ければ成功 0 として数える。
        rope_cfg: ロープ。
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        params = cfg.params
        rope: Articulation = env.scene[params.get("rope_cfg", SceneEntityCfg("rope")).name]
        self.archive = sa.StateArchive(
            classes=params.get("archive_classes", (1, 2)),
            capacity=params.get("capacity", 4096),
            num_joints=rope.num_joints,
            device=env.device,
        )
        # 今のエピソードがアーカイブから始まったか / エピソードが始まっているか
        # (最初の env.reset() はエピソードの終わりではないので成功率に数えない)。
        self._from_archive = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._active = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._sums = {k: 0.0 for k in ("resets", "from_archive", "mismatch", "sc_n", "sc_ok", "ar_n", "ar_ok")}

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor,
        prob: float = 0.5,
        capacity: int = 4096,
        archive_classes: tuple[int, ...] = (1, 2),
        class_weights: tuple[float, ...] = (1.0, 2.0),
        random_yaw: bool = True,
        success_term: str = "knot_tied",
        rope_cfg: SceneEntityCfg = SceneEntityCfg("rope"),
    ) -> None:
        if env_ids is None:
            env_ids = torch.arange(env.num_envs, device=env.device)
        env_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
        self._count_finished_episodes(env, env_ids, success_term)

        # -- どの env をアーカイブから始めるか
        use = torch.rand(env_ids.numel(), device=env.device) < prob
        self._from_archive[env_ids] = False
        self._active[env_ids] = True
        self._sums["resets"] += float(env_ids.numel())
        sel = env_ids[use]
        if sel.numel() == 0:
            return
        drawn = self.archive.sample(int(sel.numel()), class_weights)
        if drawn is None:  # アーカイブがまだ空 (学習の最初)
            return
        cls, rel_pose, joint_pos = drawn

        # -- 書き込む (速度は 0。保存したのは relax 後のほぼ静止した形状)
        rope: Articulation = env.scene[rope_cfg.name]
        origins = env.scene.env_origins[sel]
        pose = sa.place_at_origin(rel_pose, origins, sa.random_yaw(sel.numel(), env.device, random_yaw))
        rope.write_root_pose_to_sim(pose, env_ids=sel)
        rope.write_root_velocity_to_sim(torch.zeros(sel.numel(), 6, device=env.device), env_ids=sel)
        rope.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos), env_ids=sel)
        self._from_archive[sel] = True

        # -- 検算: 書き込んだ形状の位相が保存時のクラスと一致するか (FK は Isaac Lab が回す)
        body_index = get_extractor(env, rope_cfg.name).body_index
        nodes = tp.polyline_from_link_centers(rope.data.body_pos_w[sel][:, body_index])
        restored_cls = tp.topology_class(tp.reduce_r2(tp.segment_intersections(nodes)))
        self._sums["from_archive"] += float(sel.numel())
        self._sums["mismatch"] += float((restored_cls != cls).sum())

    def record(
        self,
        env: ManagerBasedRLEnv,
        settled_height_diameters: float,
        rope_diameter: float,
        diverged_term: str | None,
        rope_cfg: SceneEntityCfg,
    ) -> None:
        """手の終わりの形状のうち、床に落ちていて発散していないものを保存する (`record_archive` から呼ぶ)。"""
        extractor = get_extractor(env, rope_cfg.name)
        cls = tp.topology_class(extractor.reduced_batch)
        ok = rope_zmax(env, rope_cfg) <= settled_height_diameters * rope_diameter
        ok &= torch.isfinite(extractor.nodes).all(dim=-1).all(dim=-1)
        if diverged_term is not None:
            ok &= ~diverged_now(env, diverged_term)
        cls = torch.where(ok, cls, torch.full_like(cls, -1))
        rows = (cls >= 0).nonzero(as_tuple=False).squeeze(-1)
        if rows.numel() == 0:
            return
        rope: Articulation = env.scene[rope_cfg.name]
        link_pos = rope.data.body_pos_w[rows][:, extractor.body_index]
        rel = sa.to_centroid_frame(rope.data.root_link_pose_w[rows], link_pos, env.scene.env_origins[rows])
        self.archive.add(cls[rows], rel, rope.data.joint_pos[rows])

    def pop_stats(self) -> dict[str, float]:
        """前回からの集計を返してリセットする。アーカイブの大きさは現在値。"""
        s = self._sums
        out = {
            "archive_frac": s["from_archive"] / max(s["resets"], 1.0),
            "restore_mismatch": s["mismatch"] / max(s["from_archive"], 1.0),
            "scratch_success": s["sc_ok"] / max(s["sc_n"], 1.0),
            "archive_success": s["ar_ok"] / max(s["ar_n"], 1.0),
            "scratch_episodes": s["sc_n"],
        }
        for c, n in self.archive.sizes().items():
            out[f"arc_size_c{c}"] = float(n)
        self._sums = {k: 0.0 for k in s}
        return out

    def _count_finished_episodes(self, env: ManagerBasedRLEnv, env_ids: torch.Tensor, success_term: str) -> None:
        """reset される env のうち、エピソードを終えたものを開始の種別ごとに数える。

        reset の直前に回った終了判定の値を読む。いま reset される env については
        「このエピソードを終わらせた判定」そのものになる。
        """
        active = self._active[env_ids]
        if not bool(active.any()):
            return
        ids = env_ids[active]
        if success_term in env.termination_manager.active_terms:
            success = env.termination_manager.get_term(success_term)[ids]
        else:
            success = torch.zeros(ids.numel(), dtype=torch.bool, device=env.device)
        from_archive = self._from_archive[ids]
        self._sums["sc_n"] += float((~from_archive).sum())
        self._sums["sc_ok"] += float((~from_archive & success).sum())
        self._sums["ar_n"] += float(from_archive.sum())
        self._sums["ar_ok"] += float((from_archive & success).sum())


def _get_archive_term(env: ManagerBasedRLEnv, event_term: str) -> ResetFromArchive:
    term = env.event_manager.get_term_cfg(event_term).func
    if not isinstance(term, ResetFromArchive):
        raise TypeError(f"event term '{event_term}' is not ResetFromArchive (got {type(term).__name__})")
    return term


def record_archive(
    env: ManagerBasedRLEnv,
    event_term: str = "reset_from_archive",
    settled_height_diameters: float = 3.0,
    rope_diameter: float = 0.02,
    diverged_term: str | None = None,
    rope_cfg: SceneEntityCfg = SceneEntityCfg("rope"),
) -> torch.Tensor:
    """報酬項: 手の終わりの形状をアーカイブへ入れる。**常に 0 を返す** (学習に寄与しない)。

    重みは 0 にしないこと (0 だと `RewardManager` が呼ばない)。1.0 を推奨。
    `rope_diameter` は `rope_task_cfg._apply_rope` がプリセットの値で埋める。
    """
    _get_archive_term(env, event_term).record(env, settled_height_diameters, rope_diameter, diverged_term, rope_cfg)
    return torch.zeros(env.num_envs, device=env.device)


def log_archive_stats(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    event_term: str = "reset_from_archive",
) -> dict[str, float]:
    """カリキュラム項: アーカイブの統計を `Curriculum/<項名>/<key>` として TensorBoard へ流す。

    何も変更しない (カリキュラムとしては no-op)。key は `ResetFromArchive.pop_stats` を参照。
    """
    return _get_archive_term(env, event_term).pop_stats()
