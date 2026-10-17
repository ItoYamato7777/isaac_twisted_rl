"""環境の状態を **そのまま** 別の環境へ写す (推論時探索の複製)。`scripts/rsl_rl/eval_search.py` が使う。

アーカイブからの復元 (`archive.ResetToStage`) とは目的が違う。あちらは学習の初期状態を作るので、ロープの
重心を env 原点へ移し、ランダムな yaw をかけ、速度を 0 にする。探索の複製は「同じ状態から別の手を試す」
ためのもので、**複製元と同じ姿勢・同じ速度** で、方策が見る観測まで一致しなければならない。
env 原点からの相対位置は保ち、原点の差だけを足し直す (環境はグリッド状に並んでいて原点が違う)。

## 写すもの (手の終わり = `env.step()` の後の状態)

| 持ち主 | 中身 | 写さないと |
|---|---|---|
| ロープ (`Articulation`) | root link の姿勢・root の重心速度・関節角・関節速度 | 形・発散判定 (速度) |
| グリッパー (`Articulation`) | root の姿勢・速度、指の関節角・速度・目標値 | 観測 `hand_pos`、指の開き |
| `PickPlaceAction` | `_hand_pos` / `_hand_yaw` (次の手の approach の始点)、補間区間、フェーズ、`_raw_actions` | 次の手の始点が複製先の古い位置になる |
| `ActionManager` | `_action` / `_prev_action` | 観測 `actions` (`last_action`) |
| `NextEdgeGoal` (v6〜) | 選んだ辺 `key` / `row` / `cand` | 位相が同じ間は辺を選び直さないので、`long_end` の選択が複製先の形で変わりうる |
| env | `episode_length_buf` | 時間切れの判定 |
| 位相抽出器 | 写さずに捨てる (`invalidate_extractor`) | 観測が複製前の形状のまま計算される |

`PickPlaceAction` のその他の量 (掴む点・置く先・`_batch_at_start` など) は次の `process_actions` で
今のロープから作り直されるので写さない。報酬・ログ用の項 (`TopologyStats` など) の内部状態も写さない
(評価の結果に効かない)。

## 写せないもの

寸法の DR (v16) の寸法の違う env の間では写せない (`__init__` が例外にする)。


PhysX の内部状態 (接触のキャッシュ・ソルバのウォームスタート) は API から読み書きできない。同じ状態・
同じ手でも、複製先と複製元の結果は完全には一致しない (`eval_search.py --mode clone_test` で測る)。

`state` (辞書) は純粋なテンソルだけで、`common/beam_search.NodeStore` にそのまま保存できる。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation

from .observations import get_next_edge_goal, invalidate_extractor
from .rope_size import get_rope_sizes

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

PARTS = ("rope", "gripper", "hand", "action_manager", "next_edge", "episode")
"""`EnvStateCloner.restore(skip=...)` で写さずにおける部分。複製の検証 (写さないと観測がずれるか) 用。"""


class EnvStateCloner:
    """`capture(env_ids)` で状態を辞書に取り出し、`restore(env_ids, state)` で別の環境へ書き込む。

    Args:
        env: `ManagerBasedRLEnv` (ラッパーを外したもの)。
        rope_name / gripper_name: シーンの資産名。
        action_name: `PickPlaceAction` の項名。
    """

    def __init__(
        self,
        env: ManagerBasedRLEnv,
        rope_name: str = "rope",
        gripper_name: str = "gripper",
        action_name: str = "pick_place",
    ):
        self.env = env
        self.rope: Articulation = env.scene[rope_name]
        self.gripper: Articulation = env.scene[gripper_name]
        self.term = env.action_manager.get_term(action_name)
        self.next_edge = get_next_edge_goal(env)
        # 寸法の DR (v16) で寸法の違う env の間で写すと、形も行動の範囲も合わない。探索は寸法を 1 組にして回す。
        sizes = get_rope_sizes(env, rope_name)
        if sizes is not None and len(set(sizes.names)) > 1:
            raise ValueError(
                f"寸法の組が {len(sizes.names)} 個ある env では状態を写せない。"
                "env.scene.rope.spawn.scales で 1 組にすること (例: [[1.0,1.0]])"
            )

    def _ids(self, env_ids: torch.Tensor | None) -> torch.Tensor:
        if env_ids is None:
            return torch.arange(self.env.num_envs, device=self.env.device)
        return torch.as_tensor(env_ids, dtype=torch.long, device=self.env.device)

    def capture(self, env_ids: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """`env_ids` の状態 (位置は env 原点相対)。値はすべて複製 (以後の物理で変わらない)。"""
        ids = self._ids(env_ids)
        env = self.env
        origin = env.scene.env_origins[ids]
        rope, grip, term = self.rope.data, self.gripper.data, self.term
        out: dict[str, torch.Tensor] = {}

        pose = rope.root_link_pose_w[ids].clone()
        pose[:, :3] -= origin
        out["rope_root_pose"] = pose
        out["rope_root_vel"] = rope.root_com_vel_w[ids].clone()
        out["rope_joint_pos"] = rope.joint_pos[ids].clone()
        out["rope_joint_vel"] = rope.joint_vel[ids].clone()

        pose = grip.root_link_pose_w[ids].clone()
        pose[:, :3] -= origin
        out["grip_root_pose"] = pose
        out["grip_root_vel"] = grip.root_com_vel_w[ids].clone()
        out["grip_joint_pos"] = grip.joint_pos[ids].clone()
        out["grip_joint_vel"] = grip.joint_vel[ids].clone()
        out["grip_joint_pos_target"] = grip.joint_pos_target[ids].clone()
        out["grip_joint_vel_target"] = grip.joint_vel_target[ids].clone()
        out["grip_joint_effort_target"] = grip.joint_effort_target[ids].clone()

        out["hand_pos"] = term._hand_pos[ids] - origin
        out["hand_yaw"] = term._hand_yaw[ids].clone()
        out["seg_start"] = term._seg_start[ids] - origin
        out["seg_end"] = term._seg_end[ids] - origin
        out["seg_yaw"] = torch.stack([term._seg_yaw0[ids], term._seg_yaw1[ids]], dim=-1)
        out["seg_ratio"] = torch.stack([term._seg_r0[ids], term._seg_r1[ids]], dim=-1)
        out["phase"] = torch.stack([term._phase[ids], term._phase_step[ids]], dim=-1)
        out["raw_actions"] = term._raw_actions[ids].clone()

        am = env.action_manager
        out["am_action"] = am._action[ids].clone()
        out["am_prev_action"] = am._prev_action[ids].clone()

        if self.next_edge is not None:
            ne = self.next_edge
            out["next_edge"] = torch.stack([ne.key[ids], ne.row[ids], ne.cand[ids]], dim=-1)

        out["episode_length"] = env.episode_length_buf[ids].clone()
        return out

    def restore(self, env_ids: torch.Tensor, state: dict[str, torch.Tensor], skip: tuple[str, ...] = ()) -> None:
        """`state` (先頭の次元 = `len(env_ids)`) を `env_ids` へ書き込み、位相抽出器のキャッシュを捨てる。

        `skip` に `PARTS` の名前を入れると、その部分を写さない (検証用。通常は空)。書き込んだ後に観測を
        計算し直すのは呼び出し側 (`observation_manager.compute()`)。
        """
        ids = self._ids(env_ids)
        if ids.numel() == 0:
            return
        env = self.env
        origin = env.scene.env_origins[ids]

        if "rope" not in skip:
            pose = state["rope_root_pose"].clone()
            pose[:, :3] += origin
            self.rope.write_root_pose_to_sim(pose, env_ids=ids)
            self.rope.write_root_velocity_to_sim(state["rope_root_vel"], env_ids=ids)
            self.rope.write_joint_state_to_sim(state["rope_joint_pos"], state["rope_joint_vel"], env_ids=ids)

        if "gripper" not in skip:
            pose = state["grip_root_pose"].clone()
            pose[:, :3] += origin
            self.gripper.write_root_pose_to_sim(pose, env_ids=ids)
            self.gripper.write_root_velocity_to_sim(state["grip_root_vel"], env_ids=ids)
            self.gripper.write_joint_state_to_sim(state["grip_joint_pos"], state["grip_joint_vel"], env_ids=ids)
            self.gripper.set_joint_position_target(state["grip_joint_pos_target"], env_ids=ids)
            self.gripper.set_joint_velocity_target(state["grip_joint_vel_target"], env_ids=ids)
            self.gripper.set_joint_effort_target(state["grip_joint_effort_target"], env_ids=ids)

        if "hand" not in skip:
            term = self.term
            term._hand_pos[ids] = state["hand_pos"] + origin
            term._hand_yaw[ids] = state["hand_yaw"]
            term._seg_start[ids] = state["seg_start"] + origin
            term._seg_end[ids] = state["seg_end"] + origin
            term._seg_yaw0[ids] = state["seg_yaw"][:, 0]
            term._seg_yaw1[ids] = state["seg_yaw"][:, 1]
            term._seg_r0[ids] = state["seg_ratio"][:, 0]
            term._seg_r1[ids] = state["seg_ratio"][:, 1]
            term._phase[ids] = state["phase"][:, 0]
            term._phase_step[ids] = state["phase"][:, 1]
            term._raw_actions[ids] = state["raw_actions"]

        if "action_manager" not in skip:
            am = env.action_manager
            am._action[ids] = state["am_action"]
            am._prev_action[ids] = state["am_prev_action"]

        if "next_edge" not in skip and self.next_edge is not None and "next_edge" in state:
            ne = self.next_edge
            ne.key[ids] = state["next_edge"][:, 0]
            ne.row[ids] = state["next_edge"][:, 1]
            ne.cand[ids] = state["next_edge"][:, 2]

        if "episode" not in skip:
            env.episode_length_buf[ids] = state["episode_length"]

        invalidate_extractor(env)
