"""pick&place 1 手を 1 アクションとして実行する `ActionTerm` (Step 6 の山場)。

## この設計の核心: なぜ `decimation` が巨大なのか

`next_action.md` 4.0 で「1 アクション = 掴む→運ぶ→離す 全部」(Options
Framework 的な粒度) と決めた。これを manager-based Isaac Lab に落とすとき、
`ManagerBasedRLEnv.step()` の構造がそのまま答えになる::

    self.action_manager.process_action(action)   # 1 env ステップに 1 回
    for _ in range(self.cfg.decimation):
        self.action_manager.apply_action()       # <- 毎物理ステップ
        self.scene.write_data_to_sim()
        self.sim.step(render=False)
        self.scene.update(dt=self.physics_dt)    # <- rope.data がここで更新
    # 以降 termination / reward / observation を 1 回だけ計算

つまり **`decimation` = 1 手ぶんの物理ステップ数** にすれば:

* `process_actions()` が「1 手の指示 (6 次元)」を受け取り、掴む点と置く先を確定する
* `apply_actions()` が **1 物理ステップぶんだけ** ステートマシンを進める
* 報酬・観測・終了判定は **手が終わった時点で 1 回だけ** 評価される

これが `next_action.md` の想定 (1 エピソード 3〜5 アクション、報酬は
「アクションごと」) とそのまま一致する。しかも `apply_actions()` の各呼び出しの
直前に `scene.update()` が回っているので、**ロープの最新姿勢を読みながら
ハンドを動かせる**。デモの `over_link_frac` (渡し先をリンクの現在位置から
実行時に決める) に相当する仕掛けが、この構造で自然に作れる。

## デモとの違い: 逐次ループ -> テンソルのステートマシン

`topology_debug_demo.py` の `grasp_move()` は 13 個のフェーズを Python の
for ループで順に実行していた。RL では全環境が同時に別々のフェーズにいるので、
**フェーズ ID と経過ステップ数を `(num_envs,)` のテンソルで持つ**必要がある
(`next_action.md` 4.3)。補間そのもの (smoothstep) や掴む点の決め方といった
ロジックはデモからそのまま持ってきている。

## 離散アクションを連続チャネルで表す理由

`next_action.md` 4.1 は `grasp_link_id` と `delta_link` を離散と決めているが、
このリポジトリの学習器 **RSL-RL の PPO は連続ガウス方策しか持たない**
(離散ヘッドの実装が無い)。そこで 6 チャネルすべてを連続 `[-1, 1]` で受け取り、
離散 2 つは **この ActionTerm 側で量子化する**。方策から見れば連続、
環境に入る値は離散、という形にすれば 4.1 の設計意図
(探索空間を N x N から N x 17 に減らす / `Δlink` に状況非依存の意味を持たせる)
はそのまま保たれる。

## 把持は摩擦のみ (固定ジョイントを使わない)

`rope_catch_demo.py` の方式を踏襲する。指を常に「全閉」で指令し続けると
ロープの太さぶん閉じ切れず、PD の定常偏差がそのまま把持力になる。
固定ジョイントで拘束しないので **どのリンクでも掴める**。
この性質が `grasp_link_id` を自由に選ばせる前提になっている。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.managers.manager_term_cfg import ActionTermCfg
from isaaclab.utils import configclass

from ..rope_state import resolve_rope_bodies
from . import phases

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class PickPlaceAction(ActionTerm):
    """1 手 (掴む→運ぶ→置く→離す) をバッチ実行するアクション項。

    アクションは 6 次元、すべて `[-1, 1]`:

    ===  ================  ===============================================
    ch   意味              変換後
    ===  ================  ===============================================
    0    `grasp_link`      `{margin, ..., N-1-margin}` の離散リンク ID
    1    `delta_link`      `{-8, ..., +8}` の離散。target = clip(grasp+Δ)
    2    `drop_x_rel`      target リンクの現在位置からの x 変位 (±0.15 L)
    3    `drop_y_rel`      同 y
    4    `drop_z_rel`      置く高さ (0 〜 3 ロープ直径ぶん上乗せ)
    5    `end_yaw`         置き終わりの手首 yaw (±π)
    ===  ================  ===============================================

    `next_action.md` 4.1 の 7 次元との差: あちらは `grasp_link_id` と
    `delta_link` を別々に数えて「離散 2 + 連続 4 = 7」としているが、
    列挙されているチャネル自体は 6 つ (grasp / delta / x / y / z / yaw)。
    ここでは実際のチャネル数どおり 6 次元とした。
    """

    cfg: PickPlaceActionCfg
    _asset: Articulation
    """`cfg.asset_name` で指定されたグリッパー。ロープは別途 scene から取る。"""

    def __init__(self, cfg: PickPlaceActionCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self._rope: Articulation = env.scene[cfg.rope_asset_name]
        # ロープの body 順序は **名前で** 解決する。生の index 順と一致する
        # 保証が無く、狂うと p-data が静かに壊れるため (rope_state.py 参照)。
        body_ids = resolve_rope_bodies(self._rope)
        self._rope_body_index = torch.tensor(body_ids, dtype=torch.long, device=self.device)
        self._num_links = len(body_ids)

        # -- 指の関節。開度 1.0 のときの目標値を求めておく。左右で可動域の符号が
        #    逆 (left は [0, +STROKE] / right は [-STROKE, 0]) なので、
        #    絶対値の大きい側 = 「全開」の値を選ぶ。デモと同じ計算。
        self._finger_ids, _ = self._asset.find_joints(cfg.finger_joint_names, preserve_order=True)
        limits = self._asset.data.joint_pos_limits[:, self._finger_ids, :]
        lower, upper = limits[..., 0], limits[..., 1]
        self._open_value = torch.where(lower.abs() > upper.abs(), lower, upper)  # (num_envs, 2)

        # -- フェーズの尺 [物理ステップ]。テンソルで索引するので順序は phases の ID に一致。
        sim_dt = env.sim.get_physics_dt()
        self._phase_steps = torch.tensor(
            cfg.durations.step_counts(sim_dt), dtype=torch.long, device=self.device
        )  # (NUM_PHASES,)

        # -- 寸法由来の定数
        self._rope_length = cfg.rope_length
        self._diameter = cfg.rope_diameter
        lim = cfg.limits
        self._drop_xy_max = lim.drop_xy_rel_frac * self._rope_length
        self._drop_z_max = lim.drop_z_diameters * self._diameter
        self._clear_dist = lim.clear_diameters * self._diameter
        self._max_delta = lim.max_delta_link
        # 端ちょうどは滑るので掴める範囲を内側に寄せる (デモと同じ理由)。
        self._grasp_lo = lim.edge_margin
        self._grasp_hi = self._num_links - 1 - lim.edge_margin

        num_envs = self.num_envs
        dev = self.device
        self._raw_actions = torch.zeros(num_envs, self.action_dim, device=dev)
        self._processed_actions = torch.zeros(num_envs, self.action_dim, device=dev)

        # -- ステートマシンの状態
        self._phase = torch.zeros(num_envs, dtype=torch.long, device=dev)
        self._phase_step = torch.zeros(num_envs, dtype=torch.long, device=dev)
        # 現フェーズの補間区間 (始点 -> 終点)
        self._seg_start = torch.zeros(num_envs, 3, device=dev)
        self._seg_end = torch.zeros(num_envs, 3, device=dev)
        self._seg_yaw0 = torch.zeros(num_envs, device=dev)
        self._seg_yaw1 = torch.zeros(num_envs, device=dev)
        self._seg_r0 = torch.full((num_envs,), phases.OPEN_RATIO, device=dev)
        self._seg_r1 = torch.full((num_envs,), phases.OPEN_RATIO, device=dev)

        # -- 1 手ごとに process_actions で確定する量 (world 座標)
        self._grasp_id = torch.zeros(num_envs, dtype=torch.long, device=dev)
        self._target_id = torch.zeros(num_envs, dtype=torch.long, device=dev)
        self._grasp_xy = torch.zeros(num_envs, 2, device=dev)
        self._place_pos = torch.zeros(num_envs, 3, device=dev)
        self._clear_pos = torch.zeros(num_envs, 3, device=dev)
        self._grasp_yaw = torch.zeros(num_envs, device=dev)
        self._end_yaw = torch.zeros(num_envs, device=dev)
        # ハンドの現在指令位置。次の手の approach の始点になる。
        self._hand_pos = torch.zeros(num_envs, 3, device=dev)
        self._hand_yaw = torch.zeros(num_envs, device=dev)

        # -- 報酬へ持ち越す量 (rewards.py が action_manager 経由で読む)
        self._count_at_start = torch.zeros(num_envs, dtype=torch.long, device=dev)
        self._class_at_start = torch.zeros(num_envs, dtype=torch.long, device=dev)
        # R2 除去版のクラス (v2-R2 以降)。手を始める前の batch を参照で控えて
        # おき、読まれたときだけ計算する。抽出器は update() ごとに新しい
        # テンソルを作るので、参照を持つだけで「手を打つ前」が保存される。
        self._batch_at_start = None
        self._reduced_batch_at_start = None
        self._class_r2_at_start: torch.Tensor | None = None
        self._grasp_dist = torch.zeros(num_envs, device=dev)
        self._invalid = torch.zeros(num_envs, dtype=torch.bool, device=dev)

        # 書き込み用バッファ。毎ステップ確保し直さないよう使い回す。
        self._root_pose = torch.zeros(num_envs, 7, device=dev)
        self._root_vel = torch.zeros(num_envs, 6, device=dev)

        self._initialised = False

    # ------------------------------------------------------------------ 形状
    @property
    def action_dim(self) -> int:
        return 6

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    # -------------------------------------------------------- 報酬向けの公開量
    @property
    def crossings_at_action_start(self) -> torch.Tensor:
        """手を始めた時点の交差数 `(num_envs,)`。progress 報酬が使う。"""
        return self._count_at_start

    @property
    def class_at_action_start(self) -> torch.Tensor:
        """手を始めた時点の位相クラス `(num_envs,)`。ラダー整形報酬が使う。

        ポテンシャル整形 `F = γΦ(s') − Φ(s)` の `Φ(s)` 側。手を打つ前に
        `process_actions` で確定させておかないと、報酬計算の時点では
        既にロープが動いてしまっていて「手を打つ前」が取れない。
        """
        return self._class_at_start

    @property
    def class_r2_at_action_start(self) -> torch.Tensor:
        """`class_at_action_start` の R2 除去版 `(num_envs,)`。v2-R2 の整形報酬が使う。

        `Φ(s)` と `Φ(s')` は **同じ判定器** で測らないとポテンシャル整形の
        前提が崩れるので、`reduce_r2=True` の項はこちらを読むこと。
        まだ 1 手も打っていない (reset 直後) なら 0 (= trivial) を返す。
        """
        if self._class_r2_at_start is None:
            from .. import topology as tp

            reduced = self.reduced_batch_at_action_start
            if reduced is None:
                self._class_r2_at_start = torch.zeros_like(self._class_at_start)
            else:
                self._class_r2_at_start = tp.topology_class(reduced)
        return self._class_r2_at_start

    @property
    def batch_at_action_start(self):
        """手を始めた時点の交差 (`IntersectionBatch`、R2 除去前)。まだ 1 手も打っていなければ None。"""
        return self._batch_at_start

    @property
    def reduced_batch_at_action_start(self):
        """手を始めた時点の交差を R2 除去したもの (`IntersectionBatch`)。まだ 1 手も打っていなければ None。

        v3 の BFS ポテンシャル (`knot_class_shaping(potential="bfs")`) が `Φ(s)` を
        引くのに使う。R2 除去は重いので読まれたときだけ計算し、同じ手の中では使い回す。
        """
        if self._reduced_batch_at_start is None and self._batch_at_start is not None:
            from .. import topology as tp

            self._reduced_batch_at_start = tp.reduce_r2(self._batch_at_start)
        return self._reduced_batch_at_start

    @property
    def grasp_distance(self) -> torch.Tensor:
        """`close` 終了時の 指先とロープの距離 `(num_envs,)` [m]。整形報酬が使う。"""
        return self._grasp_dist

    @property
    def invalid_action(self) -> torch.Tensor:
        """ロープ座標が NaN で手を実行できなかった環境 `(num_envs,)`。"""
        return self._invalid

    # ------------------------------------------------------------------ 本体
    def process_actions(self, actions: torch.Tensor):
        """1 手ぶんの指示を解釈し、フェーズ列の最初 (`approach`) を組み立てる。

        ここで掴む点と置く先を **今のロープ姿勢から** 確定させる。
        `next_action.md` 4.1 が「置く先を絶対座標にせずリンク相対にする」と
        決めた理由がこれで、pick&place 粒度では離した瞬間にロープが物理で
        動くため、絶対座標で持つ意味がほとんど無い。
        """
        self._raw_actions[:] = actions
        act = actions.clamp(-1.0, 1.0)

        link_pos = self._rope_link_positions()  # (num_envs, num_links, 3) world

        # -- ch0: 掴むリンク。[-1,1] を [grasp_lo, grasp_hi] へ写して丸める。
        span = self._grasp_hi - self._grasp_lo
        grasp_f = (act[:, 0] + 1.0) * 0.5 * span + self._grasp_lo
        self._grasp_id = grasp_f.round().long().clamp(self._grasp_lo, self._grasp_hi)

        # -- ch1: Δlink。target はロープ全体を使えるので端まで許す。
        delta = (act[:, 1] * self._max_delta).round().long()
        self._target_id = (self._grasp_id + delta).clamp(0, self._num_links - 1)

        # -- 掴む点と、そこでのロープの接線方向 (指が軸を真横から挟む姿勢)
        grasp_pos = self._gather_links(link_pos, self._grasp_id)  # (num_envs, 3)
        self._grasp_xy = grasp_pos[:, :2]
        self._grasp_yaw = self._tangent_yaw(link_pos, self._grasp_id)

        # -- 置く先 = target リンクの現在位置 + 相対変位
        target_pos = self._gather_links(link_pos, self._target_id)
        place_xy = target_pos[:, :2] + act[:, 2:4] * self._drop_xy_max
        # ch4: [-1,1] -> [0, 3 直径]。置く高さは把持高さからの上乗せ。
        place_z = phases.GRASP_PALM_Z + (act[:, 4] + 1.0) * 0.5 * self._drop_z_max
        self._place_pos = torch.cat([place_xy, place_z.unsqueeze(-1)], dim=-1)

        # -- ch5: 置き終わりの手首 yaw
        self._end_yaw = act[:, 5] * math.pi

        # -- 離した後に横へ抜ける先。運搬してきた向き (掴む点 -> 置く先) へ
        #    数直径ぶん進む。真上へ上げるだけだとロープが指に残って一緒に
        #    持ち上がる (デモの clear フェーズと同じ理由)。
        carry_dir = place_xy - self._grasp_xy
        norm = carry_dir.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        clear_xy = place_xy + carry_dir / norm * self._clear_dist
        self._clear_pos = torch.cat([clear_xy, place_z.unsqueeze(-1)], dim=-1)

        # -- ロープが発散している環境は手を実行しない (NaN が全体へ伝播するのを防ぐ)
        finite = torch.isfinite(grasp_pos).all(dim=-1) & torch.isfinite(target_pos).all(dim=-1)
        self._invalid = ~finite
        if self._invalid.any():
            # 無効な環境は掴む点/置く先を現在のハンド位置に潰しておく。
            # そのまま NaN を書き込むと PhysX 側が壊れる。
            safe = self._hand_pos
            self._grasp_xy = torch.where(self._invalid.unsqueeze(-1), safe[:, :2], self._grasp_xy)
            self._place_pos = torch.where(self._invalid.unsqueeze(-1), safe, self._place_pos)
            self._clear_pos = torch.where(self._invalid.unsqueeze(-1), safe, self._clear_pos)
            self._grasp_yaw = torch.where(self._invalid, self._hand_yaw, self._grasp_yaw)
            self._end_yaw = torch.where(self._invalid, self._hand_yaw, self._end_yaw)

        self._processed_actions[:, 0] = self._grasp_id.float()
        self._processed_actions[:, 1] = self._target_id.float()
        self._processed_actions[:, 2:5] = self._place_pos
        self._processed_actions[:, 5] = self._end_yaw

        # -- 報酬用に「手を始める前の交差数」を控える
        self._count_at_start = self._current_crossing_count()
        self._class_at_start = self._current_topology_class()
        self._batch_at_start = self._current_batch()
        self._reduced_batch_at_start = None
        self._class_r2_at_start = None
        self._grasp_dist = torch.zeros_like(self._grasp_dist)

        # -- フェーズ列を先頭から始める
        self._phase[:] = phases.APPROACH
        self._phase_step[:] = 0
        self._begin_phase(torch.arange(self.num_envs, device=self.device))

    def apply_actions(self):
        """1 物理ステップぶんステートマシンを進め、ハンドの姿勢を書き込む。"""
        if not self._initialised:
            # 最初の呼び出し時点で root の現在位置を拾っておく。reset 直後の
            # approach 始点がゼロベクトルだとハンドが原点へ飛ぶため。
            self._sync_hand_from_sim()
            self._initialised = True

        dur = self._phase_steps[self._phase].clamp_min(1)  # (num_envs,)
        u = ((self._phase_step + 1).float() / dur.float()).clamp(0.0, 1.0)
        # smoothstep とその時間微分。等速補間だと始点・終点で速度が不連続になり、
        # その慣性でロープが指から滑る (デモの run_segment と同じ理由)。
        s = u * u * (3.0 - 2.0 * u)
        seg_time = dur.float() * self._env.sim.get_physics_dt()
        ds = 6.0 * u * (1.0 - u) / seg_time.clamp_min(1e-6)

        pos = self._seg_start + (self._seg_end - self._seg_start) * s.unsqueeze(-1)
        vel = (self._seg_end - self._seg_start) * ds.unsqueeze(-1)
        # yaw は最短回り。差を (-π, π] に畳んでから補間する。
        d_yaw = torch.remainder(self._seg_yaw1 - self._seg_yaw0 + math.pi, 2.0 * math.pi) - math.pi
        yaw = self._seg_yaw0 + d_yaw * s
        yaw_rate = d_yaw * ds
        ratio = self._seg_r0 + (self._seg_r1 - self._seg_r0) * s

        self._write_hand(pos, yaw, vel, yaw_rate)
        self._set_fingers(ratio)

        self._hand_pos = pos
        self._hand_yaw = yaw

        # -- フェーズを進める
        self._phase_step += 1
        done = self._phase_step >= dur
        if done.any():
            # close 終了時に把持距離を記録する (整形報酬用)。この時点なら
            # 「指を閉じ切ったとき、狙ったリンクが実際に指の間にいたか」を測れる。
            closing = done & (self._phase == phases.CLOSE)
            if closing.any():
                self._record_grasp_distance(closing)

            # 「最終フェーズを終えた」判定は **フェーズを進める前** に取る。
            # 進めた後の `_phase >= RELAX` で判定すると、retreat から relax へ
            # 入ったばかりの環境まで「終了済み」に含まれてしまい、
            # relax の待ち時間 (ロープが落ち着くのを待つ) が丸ごと飛ぶ。
            finished = done & (self._phase >= phases.RELAX)
            advancing = done & (self._phase < phases.RELAX)
            self._phase = torch.where(advancing, self._phase + 1, self._phase)
            self._phase_step = torch.where(done, torch.zeros_like(self._phase_step), self._phase_step)
            if advancing.any():
                self._begin_phase(advancing.nonzero(as_tuple=False).squeeze(-1))
            # 最終フェーズ (relax) を終えた環境は、その姿勢のまま静止させる
            # (decimation の端数を吸収する。ここで動くと報酬の測定を汚す)。
            if finished.any():
                idx = finished.nonzero(as_tuple=False).squeeze(-1)
                self._seg_start[idx] = self._seg_end[idx]
                self._seg_yaw0[idx] = self._seg_yaw1[idx]
                self._phase_step[idx] = self._phase_steps[phases.RELAX] - 1

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """エピソード開始時にステートマシンを初期状態へ戻す。"""
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0
        self._phase[env_ids] = phases.APPROACH
        self._phase_step[env_ids] = 0
        self._count_at_start[env_ids] = 0
        self._class_at_start[env_ids] = 0
        self._grasp_dist[env_ids] = 0.0
        self._invalid[env_ids] = False
        # ハンドは待機高さで静止させる。reset イベントが root を初期姿勢へ
        # 戻すので、それに合わせておかないと最初の approach が飛ぶ。
        #
        # `self._initialised` では **ガードしない**。env.reset() は
        # `_reset_idx()` 経由でこの reset() を呼ぶが、その時点では
        # apply_actions() がまだ一度も呼ばれておらず _initialised は False
        # のまま。以前はここを `if self._initialised:` で囲っていたため、
        # 最初のエピソードの 1 手目だけ `_hand_pos` が __init__ 時の
        # ゼロ初期化 (= 原点) のまま approach の始点に使われ、掴む点までの
        # 距離が数 m 〜 10 m 超になって指令速度が非物理的な値 (実測で
        # 平均 78 m/s、最大 131 m/s) になっていた (2026-09-12 発覚)。
        # root_state_w はこの時点で既に (init_state.pos からの) 有効な値を
        # 持っているので、常に同期してよい。
        self._sync_hand_from_sim()
        self._seg_start[env_ids] = self._hand_pos[env_ids]
        self._seg_end[env_ids] = self._hand_pos[env_ids]
        self._seg_yaw0[env_ids] = self._hand_yaw[env_ids]
        self._seg_yaw1[env_ids] = self._hand_yaw[env_ids]
        self._seg_r0[env_ids] = phases.OPEN_RATIO
        self._seg_r1[env_ids] = phases.OPEN_RATIO

        # reset でロープ姿勢が初期状態へ戻ったので、抽出器のキャッシュを捨てる。
        # これをしないと直後の観測が reset 前の形状のまま計算される
        # (observations.invalidate_extractor の docstring に順序の詳細)。
        from .observations import invalidate_extractor

        invalidate_extractor(self._env)

    # ------------------------------------------------------------ 内部ヘルパー
    def _begin_phase(self, idx: torch.Tensor):
        """`idx` の環境について、現在のフェーズの補間区間を組み立てる。

        デモの `grasp_move()` が `plan` として一度に組み立てていたものを、
        「フェーズに入った瞬間に、そのフェーズぶんだけ作る」形に変えた。
        こうしないと環境ごとに違うフェーズにいる状況を表現できない。
        """
        if idx.numel() == 0:
            return
        ph = self._phase[idx]
        origins = self._env.scene.env_origins[idx]

        gx, gy = self._grasp_xy[idx, 0], self._grasp_xy[idx, 1]
        base_z = origins[:, 2]
        above = torch.stack([gx, gy, base_z + phases.HOME_PALM_Z], dim=-1)
        grasp = torch.stack([gx, gy, base_z + phases.GRASP_PALM_Z], dim=-1)
        lift = torch.stack([gx, gy, base_z + phases.CARRY_PALM_Z], dim=-1)
        place = self._place_pos[idx].clone()
        place[:, 2] += base_z
        carry = torch.stack([place[:, 0], place[:, 1], base_z + phases.CARRY_PALM_Z], dim=-1)
        clear = self._clear_pos[idx].clone()
        clear[:, 2] += base_z
        retreat = torch.stack([clear[:, 0], clear[:, 1], base_z + phases.HOME_PALM_Z], dim=-1)

        cur = self._hand_pos[idx]
        cur_yaw = self._hand_yaw[idx]
        g_yaw = self._grasp_yaw[idx]
        e_yaw = self._end_yaw[idx]
        OPEN, CLOSED = phases.OPEN_RATIO, phases.CLOSE_RATIO

        # (始点, 終点, 始 yaw, 終 yaw, 始開度, 終開度) をフェーズごとに定義する。
        # デモの plan と 1 対 1 に対応する。
        table = {
            phases.APPROACH: (cur, above, cur_yaw, g_yaw, OPEN, OPEN),
            phases.DESCEND: (above, grasp, g_yaw, g_yaw, OPEN, OPEN),
            phases.CLOSE: (grasp, grasp, g_yaw, g_yaw, OPEN, CLOSED),
            phases.HOLD: (grasp, grasp, g_yaw, g_yaw, CLOSED, CLOSED),
            phases.LIFT: (grasp, lift, g_yaw, g_yaw, CLOSED, CLOSED),
            # 運搬中に yaw を掴み姿勢から置き姿勢へ回す。デモは yaw 一定だったが、
            # end_yaw をアクションに入れた以上どこかで回す必要があり、
            # 一番長い carry に持たせるのが最も緩やか (急に回すとロープが滑る)。
            phases.CARRY: (lift, carry, g_yaw, e_yaw, CLOSED, CLOSED),
            phases.LOWER: (carry, place, e_yaw, e_yaw, CLOSED, CLOSED),
            phases.RELEASE: (place, place, e_yaw, e_yaw, CLOSED, OPEN),
            phases.SETTLE_OUT: (place, place, e_yaw, e_yaw, OPEN, OPEN),
            phases.CLEAR: (place, clear, e_yaw, e_yaw, OPEN, OPEN),
            phases.RETREAT: (clear, retreat, e_yaw, e_yaw, OPEN, OPEN),
            phases.RELAX: (retreat, retreat, e_yaw, e_yaw, OPEN, OPEN),
        }

        for phase_id, (p0, p1, y0, y1, r0, r1) in table.items():
            mask = ph == phase_id
            if not bool(mask.any()):
                continue
            sel = idx[mask]
            self._seg_start[sel] = p0[mask] if p0.dim() == 2 else p0
            self._seg_end[sel] = p1[mask] if p1.dim() == 2 else p1
            self._seg_yaw0[sel] = y0[mask] if torch.is_tensor(y0) else y0
            self._seg_yaw1[sel] = y1[mask] if torch.is_tensor(y1) else y1
            self._seg_r0[sel] = r0
            self._seg_r1[sel] = r1

    def _write_hand(
        self,
        pos: torch.Tensor,
        yaw: torch.Tensor,
        vel: torch.Tensor,
        yaw_rate: torch.Tensor,
    ):
        """ハンドの root 姿勢と速度を書き込む (キネマティック駆動)。

        **速度も書くのが必須**。接触ソルバは指とロープの相対速度から摩擦を
        計算するので、0 を書くとテレポート扱いになり掴んだロープが滑り落ちる
        (デモの `write_gripper_state` の docstring に実測の経緯がある)。
        """
        self._root_pose[:, 0:3] = pos
        self._root_pose[:, 3] = torch.cos(0.5 * yaw)
        self._root_pose[:, 4] = 0.0
        self._root_pose[:, 5] = 0.0
        self._root_pose[:, 6] = torch.sin(0.5 * yaw)
        self._asset.write_root_pose_to_sim(self._root_pose)
        self._root_vel[:, 0:3] = vel
        self._root_vel[:, 3:5] = 0.0
        self._root_vel[:, 5] = yaw_rate
        self._asset.write_root_velocity_to_sim(self._root_vel)

    def _set_fingers(self, ratio: torch.Tensor):
        """指の開度 (0=全閉, 1=全開) を左右の関節目標へ変換して指令する。"""
        target = ratio.unsqueeze(-1) * self._open_value
        self._asset.set_joint_position_target(target, joint_ids=self._finger_ids)

    def _sync_hand_from_sim(self):
        """シミュレータ上の root 姿勢から `_hand_pos` / `_hand_yaw` を作り直す。"""
        root = self._asset.data.root_state_w
        self._hand_pos = root[:, 0:3].clone()
        qw, qz = root[:, 3], root[:, 6]
        self._hand_yaw = 2.0 * torch.atan2(qz, qw)

    def _rope_link_positions(self) -> torch.Tensor:
        """ロープの全リンク中心 `(num_envs, num_links, 3)` (world 座標)。"""
        return self._rope.data.body_pos_w[:, self._rope_body_index]

    @staticmethod
    def _gather_links(link_pos: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        """`(num_envs, num_links, 3)` から env ごとに 1 リンクを取り出す。"""
        return link_pos.gather(1, ids.view(-1, 1, 1).expand(-1, 1, 3)).squeeze(1)

    def _tangent_yaw(self, link_pos: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        """把持点でのロープ接線方向の yaw `(num_envs,)` [rad]。

        指は base_link のローカル Y 方向に閉じるので、この角度へ向けると
        ロープの軸を真横から挟む姿勢になる (デモの `rope_tangent_yaw` と同じ)。
        """
        lo = (ids - 1).clamp(0, self._num_links - 1)
        hi = (ids + 1).clamp(0, self._num_links - 1)
        delta = self._gather_links(link_pos, hi) - self._gather_links(link_pos, lo)
        return torch.atan2(delta[:, 1], delta[:, 0])

    def _record_grasp_distance(self, mask: torch.Tensor):
        """指先と狙ったリンクの距離を記録する (整形報酬用)。"""
        link_pos = self._rope_link_positions()
        grasp_pos = self._gather_links(link_pos, self._grasp_id)
        tip = self._hand_pos.clone()
        tip[:, 2] -= phases.FINGER_TIP_OFFSET_Z
        dist = (grasp_pos - tip).norm(dim=-1)
        dist = torch.nan_to_num(dist, nan=0.0, posinf=0.0, neginf=0.0)
        self._grasp_dist = torch.where(mask, dist, self._grasp_dist)

    def _current_crossing_count(self) -> torch.Tensor:
        """現在の交差数 `(num_envs,)`。観測側の抽出器を共有する。"""
        from .observations import get_extractor

        extractor = get_extractor(self._env)
        return extractor.count.clone()

    def _current_topology_class(self) -> torch.Tensor:
        """現在の位相クラス `(num_envs,)`。抽出器は観測側と共有する。"""
        from .. import topology as tp
        from .observations import get_extractor

        return tp.topology_class(get_extractor(self._env).batch).clone()

    def _current_batch(self):
        """現在の `IntersectionBatch` (参照)。`class_r2_at_action_start` の遅延計算用。"""
        from .observations import get_extractor

        return get_extractor(self._env).batch


@configclass
class PickPlaceActionCfg(ActionTermCfg):
    """`PickPlaceAction` の設定。"""

    class_type: type[ActionTerm] = PickPlaceAction

    asset_name: str = "gripper"
    """アクションが駆動する資産 (グリッパー)。"""

    rope_asset_name: str = "rope"
    """掴む対象のロープ。`ActionTerm._asset` とは別に scene から引く。"""

    finger_joint_names: list[str] = MISSING
    """指の関節名 (左右の順)。"""

    rope_length: float = MISSING
    """ロープ全長 [m]。`drop_xy_rel` のスケールに使う。"""

    rope_diameter: float = MISSING
    """ロープ直径 [m]。`drop_z_rel` と `clear` 距離のスケールに使う。"""

    durations: phases.PhaseDurations = phases.DEFAULT_DURATIONS
    """各フェーズの尺。合計が 1 手 = 1 RL ステップの長さになる。"""

    limits: phases.GraspLimits = phases.GraspLimits()
    """アクションのチャネルを物理量へ変換するときの範囲。"""
