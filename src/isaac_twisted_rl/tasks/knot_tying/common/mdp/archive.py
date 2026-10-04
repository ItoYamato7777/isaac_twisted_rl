"""到達状態アーカイブからのリセット (Go-Explore 型)。v3 で追加。

`paper/Report_Paper.md` §7 P1-1。アーカイブ本体 (`../state_archive.py`) を Isaac Lab に
繋ぐ 3 つの項:

| 項 | 種類 | 役割 |
|---|---|---|
| `ResetFromArchive` | event (`mode="reset"`) | アーカイブ本体を持ち、reset 時に確率 `prob` で保存済みの形状を書き込む |
| `record_archive` | 報酬項 (重み 1.0、常に 0) | 毎手の終わりに、条件を満たす形状をアーカイブへ入れる |
| `log_archive_stats` | カリキュラム項 | アーカイブの大きさと、開始の種別ごとの成功率を TensorBoard へ出す |

v4 は同じ仕組みを **段** (`topology.BfsPotential.stage`) をキーにして持ち直す
(`ResetToStage` / `record_stage_archive` / `log_stage_stats`。ファイル末尾)。

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

import os
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg

from .. import state_archive as sa
from .. import topology as tp
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


# ================================================================ v4: 段ごとのアーカイブ


class ResetToStage(ManagerTermBase):
    """reset 時に「どの段から始めるか」を選び、その段のアーカイブの形状を書き込む event 項。v4 で追加。

    TWISTED-RL §IV-D (explainer `twisted-rl.html` §07) の目標と初期状態の選び方を、段
    (`topology.BfsPotential.stage`) を単位にして行う。v3 の `ResetFromArchive` との違い:

    * **キーが段** (v3 は R2 除去後の class)。v3 では cover 後の状態 (段 2) が class 1 に
      混ざっていた。段 0 (自明な形状) も保存し、初期形状のランダム化を兼ねる。
    * **段を先に選ぶ**。空でない段の中から `stage_weights` で選び、その段の中で一様に選ぶ。
      段 0 は確率 `straight_prob` でまっすぐなロープ (`reset_all` のまま) にする。
    * **ファイルから読む・ファイルへ書く**。`load_paths` を起動時に読み (ゲート 1 の台本の
      途中状態や、前の run の `archive.pt`)、無いファイルは警告して飛ばす (台本の出力をまだ
      作っていない場合)。書き出しは保存と同じく `record_stage_archive` の params
      (`save_every` 手ごとに `{log_dir}/{save_name}`)。
    * 段ごとの結果 (上がった / 下がった / 時間切れ / 発散) を開始した段別に数える。

    保存は `record_stage_archive` (報酬項)、ログは `log_stage_stats` (カリキュラム項) が行う。
    保存するのは段 0〜2 の形状で、段を上がった直後の形状はそのまま次の段の初期状態になる
    (論文の「l -> k の遷移で到達した形を k の初期状態に使う」)。

    params (すべて `__call__` の引数):
        stages: 始める段 (= 保存する段)。
        stage_weights: `stages` と同じ並びの選ぶ重み。空の段は選ばない (段 0 は空でも選ぶ)。
        straight_prob: 段 0 を選んだとき、まっすぐなロープから始める確率。Play は 1.0。
        capacity: 1 段あたりの保存件数。
        random_yaw: 復元時に重心まわりに一様乱数で回すか。
        load_paths: 起動時に読むアーカイブのファイル。`{num_links}` はロープのリンク数に置き換える
            (プリセットごとに別のファイルになる。関節数が違うファイルは `StateArchive.load` が例外にする)。
        outcome_terms: 結果のログに使う終了項の名前 `{"up": ..., "down": ..., "diverged": ..., "time_out": ...}`。
        rope_cfg: ロープ。
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        params = cfg.params
        rope: Articulation = env.scene[params.get("rope_cfg", SceneEntityCfg("rope")).name]
        self.stages = tuple(int(s) for s in params.get("stages", (0, 1, 2)))
        self.archive = sa.StateArchive(
            classes=self.stages,
            capacity=params.get("capacity", 4096),
            num_joints=rope.num_joints,
            device=env.device,
        )
        for path in params.get("load_paths", ()):
            path = path.format(num_links=rope.num_bodies)
            if not os.path.isfile(path):
                print(f"[WARN] ResetToStage: アーカイブのファイルが無いので飛ばす: {path}")
                continue
            loaded = self.archive.load(path)
            print(f"[INFO] ResetToStage: {path} から読み込んだ件数 (段 -> 件数): {loaded}")
        n = env.num_envs
        self._start = torch.zeros(n, dtype=torch.long, device=env.device)
        self._straight = torch.zeros(n, dtype=torch.bool, device=env.device)
        self._active = torch.zeros(n, dtype=torch.bool, device=env.device)
        self._outcomes = ("up", "down", "diverged", "time_out")
        self._sums = self._empty_sums()
        self._last_save_step = -1

    def _empty_sums(self) -> dict[str, float]:
        keys = ["resets", "restored", "mismatch", "straight_n", "straight_up"]
        for s in self.stages:
            keys += [f"start_s{s}", f"episodes_s{s}"] + [f"{o}_s{s}" for o in self._outcomes]
        return {k: 0.0 for k in keys}

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor,
        stages: tuple[int, ...] = (0, 1, 2),
        stage_weights: tuple[float, ...] = (1.0, 1.0, 1.0),
        straight_prob: float = 0.5,
        capacity: int = 4096,
        random_yaw: bool = True,
        load_paths: tuple[str, ...] = (),
        outcome_terms: dict[str, str] | None = None,
        rope_cfg: SceneEntityCfg = SceneEntityCfg("rope"),
    ) -> None:
        if env_ids is None:
            env_ids = torch.arange(env.num_envs, device=env.device)
        env_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
        self._count_finished_episodes(env, env_ids, outcome_terms or {})
        self._active[env_ids] = True
        self._sums["resets"] += float(env_ids.numel())

        # -- 段を選ぶ。段 0 はまっすぐなロープがあるので空でも選べる。
        sizes = self.archive.sizes()
        weights = torch.tensor(
            [w if (s == 0 or sizes[s] > 0) else 0.0 for s, w in zip(self.stages, stage_weights)],
            dtype=torch.float32,
            device=env.device,
        )
        pick = torch.multinomial(weights, env_ids.numel(), replacement=True)
        stage = torch.tensor(self.stages, dtype=torch.long, device=env.device)[pick]
        straight = stage == 0
        if 0 in sizes and sizes[0] > 0:
            straight &= torch.rand(env_ids.numel(), device=env.device) < straight_prob
        self._start[env_ids] = torch.where(straight, torch.zeros_like(stage), stage)
        self._straight[env_ids] = straight
        for s in self.stages:
            self._sums[f"start_s{s}"] += float((self._start[env_ids] == s).sum())

        # -- アーカイブから引いて書き込む (まっすぐな env は reset_all のまま)
        ids_list, cls_list, pose_list, joint_list = [], [], [], []
        for k, s in enumerate(self.stages):
            sel = env_ids[(stage == s) & ~straight]
            if sel.numel() == 0:
                continue
            one_hot = [1.0 if j == k else 0.0 for j in range(len(self.stages))]
            drawn = self.archive.sample(int(sel.numel()), one_hot)
            if drawn is None:  # 段 0 のアーカイブが空 (まっすぐにしておく)
                self._straight[sel] = True
                continue
            ids_list.append(sel)
            cls_list.append(drawn[0])
            pose_list.append(drawn[1])
            joint_list.append(drawn[2])
        if not ids_list:
            return
        sel = torch.cat(ids_list)
        cls = torch.cat(cls_list)
        rope: Articulation = env.scene[rope_cfg.name]
        pose = sa.place_at_origin(
            torch.cat(pose_list), env.scene.env_origins[sel], sa.random_yaw(sel.numel(), env.device, random_yaw)
        )
        joint_pos = torch.cat(joint_list)
        rope.write_root_pose_to_sim(pose, env_ids=sel)
        rope.write_root_velocity_to_sim(torch.zeros(sel.numel(), 6, device=env.device), env_ids=sel)
        rope.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos), env_ids=sel)

        # -- 検算: 書き込んだ形状の段が保存時と一致するか
        body_index = get_extractor(env, rope_cfg.name).body_index
        nodes = tp.polyline_from_link_centers(rope.data.body_pos_w[sel][:, body_index])
        restored = tp.get_bfs_potential(env.device).stage(tp.segment_intersections(nodes))
        self._sums["restored"] += float(sel.numel())
        self._sums["mismatch"] += float((restored != cls).sum())

    def record(
        self,
        env: ManagerBasedRLEnv,
        settled_height_diameters: float,
        rope_diameter: float,
        diverged_term: str | None,
        save_every: int,
        save_name: str,
        rope_cfg: SceneEntityCfg,
    ) -> None:
        """手の終わりの形状のうち、床に落ちていて発散していないものを段ごとに保存する。"""
        extractor = get_extractor(env, rope_cfg.name)
        stage = tp.get_bfs_potential(env.device).stage(extractor.batch, extractor.reduced_batch)
        ok = rope_zmax(env, rope_cfg) <= settled_height_diameters * rope_diameter
        ok &= torch.isfinite(extractor.nodes).all(dim=-1).all(dim=-1)
        if diverged_term is not None:
            ok &= ~diverged_now(env, diverged_term)
        rows = ok.nonzero(as_tuple=False).squeeze(-1)
        if rows.numel() > 0:
            rope: Articulation = env.scene[rope_cfg.name]
            link_pos = rope.data.body_pos_w[rows][:, extractor.body_index]
            rel = sa.to_centroid_frame(rope.data.root_link_pose_w[rows], link_pos, env.scene.env_origins[rows])
            self.archive.add(stage[rows], rel, rope.data.joint_pos[rows])

        log_dir = getattr(env.cfg, "log_dir", None)
        step = env.common_step_counter
        if save_every > 0 and log_dir and step % save_every == 0 and step != self._last_save_step:
            os.makedirs(log_dir, exist_ok=True)
            self.archive.save(os.path.join(log_dir, save_name))
            self._last_save_step = step

    def pop_stats(self) -> dict[str, float]:
        """前回からの集計を返してリセットする。アーカイブの大きさは現在値。

        `success_s{k}` は段 k から始めたエピソードのうち段が上がって終わった割合。
        `chain_est` はその積で、まっすぐなロープから固結びまでつないだときの成功率の粗い見積り
        (段の間の形状の相性を無視している。正確な値は Play の chain モードで測る)。
        """
        s = self._sums
        out = {
            "restore_mismatch": s["mismatch"] / max(s["restored"], 1.0),
            "straight_success": s["straight_up"] / max(s["straight_n"], 1.0),
        }
        chain = 1.0
        for k in self.stages:
            n = s[f"episodes_s{k}"]
            out[f"start_frac_s{k}"] = s[f"start_s{k}"] / max(s["resets"], 1.0)
            out[f"episodes_s{k}"] = n
            out[f"success_s{k}"] = s[f"up_s{k}"] / max(n, 1.0)
            out[f"fail_s{k}"] = s[f"down_s{k}"] / max(n, 1.0)
            out[f"diverged_s{k}"] = s[f"diverged_s{k}"] / max(n, 1.0)
            out[f"timeout_s{k}"] = s[f"time_out_s{k}"] / max(n, 1.0)
            chain *= out[f"success_s{k}"]
        out["chain_est"] = chain
        for k, n in self.archive.sizes().items():
            out[f"arc_size_s{k}"] = float(n)
        self._sums = self._empty_sums()
        return out

    def _count_finished_episodes(self, env: ManagerBasedRLEnv, env_ids: torch.Tensor, terms: dict[str, str]) -> None:
        """reset される env のうち、エピソードを終えたものを開始した段 x 結果で数える。

        reset される env について `termination_manager.get_term` は「このエピソードを終わらせた項」
        を返す (同じ手で複数立ったら cfg の後ろの項。v3 の `ResetFromArchive` と同じ読み方)。
        """
        active = self._active[env_ids]
        if not bool(active.any()):
            return
        ids = env_ids[active]
        start = self._start[ids]
        fired = {}
        for o in self._outcomes:
            name = terms.get(o)
            if name is not None and name in env.termination_manager.active_terms:
                fired[o] = env.termination_manager.get_term(name)[ids]
            else:
                fired[o] = torch.zeros(ids.numel(), dtype=torch.bool, device=env.device)
        for s in self.stages:
            m = start == s
            self._sums[f"episodes_s{s}"] += float(m.sum())
            for o in self._outcomes:
                self._sums[f"{o}_s{s}"] += float((m & fired[o]).sum())
        straight = self._straight[ids]
        self._sums["straight_n"] += float(straight.sum())
        self._sums["straight_up"] += float((straight & fired["up"]).sum())


def _get_stage_term(env: ManagerBasedRLEnv, event_term: str) -> ResetToStage:
    term = env.event_manager.get_term_cfg(event_term).func
    if not isinstance(term, ResetToStage):
        raise TypeError(f"event term '{event_term}' is not ResetToStage (got {type(term).__name__})")
    return term


def record_stage_archive(
    env: ManagerBasedRLEnv,
    event_term: str = "reset_to_stage",
    settled_height_diameters: float = 3.0,
    rope_diameter: float = 0.02,
    diverged_term: str | None = None,
    save_every: int = 0,
    save_name: str = "archive.pt",
    rope_cfg: SceneEntityCfg = SceneEntityCfg("rope"),
) -> torch.Tensor:
    """報酬項: 手の終わりの形状を段ごとのアーカイブへ入れ、`save_every` 手ごとにファイルへ書く。**常に 0**。

    ファイルは `{env.cfg.log_dir}/{save_name}` (train.py が log_dir を設定する)。`save_every=0` なら書かない。

    重みは 0 にしないこと (0 だと `RewardManager` が呼ばない)。`record_archive` (v3) と同じ流儀。
    """
    _get_stage_term(env, event_term).record(
        env, settled_height_diameters, rope_diameter, diverged_term, save_every, save_name, rope_cfg
    )
    return torch.zeros(env.num_envs, device=env.device)


def log_stage_stats(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    event_term: str = "reset_to_stage",
) -> dict[str, float]:
    """カリキュラム項: 段ごとの成功率とアーカイブの大きさを `Curriculum/<項名>/<key>` へ流す (no-op)。"""
    return _get_stage_term(env, event_term).pop_stats()
