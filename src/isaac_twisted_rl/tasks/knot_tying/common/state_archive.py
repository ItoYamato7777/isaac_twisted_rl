"""到達したロープ状態のアーカイブ (Go-Explore 型リセット)。v3 で追加。

`paper/Report.md` §7 P1-1。TWISTED-RL §IV-D は、交差数 k の方策の初期状態を
「それまでの学習で到達したロープ形状」から一様に選ぶ。TWISTED も到達した
状態の木から開始状態を選び、ランダムな手だけでは 3 交差に 1 時間で 27 回しか
届かなかったのが、この方式で 537 回に増えている。v2-R2 は毎回まっすぐな
ロープから始めるので、C2 を練習できるのは「偶然 C1 に着いたエピソードの
残りの手」だけだった (§5.3)。

このモジュールは **純 torch** (isaaclab を import しない) で、
`tests/test_state_archive.py` から Isaac Sim 無しで検証できる。
Isaac Lab との接続 (保存と復元) は `mdp/archive.py`。

## 何を保存するか

ロープは articulation なので、root link の姿勢 (7) と関節角 (J) があれば
形状を完全に復元できる。速度は保存しない (手の終わりの `relax` 後なので
ほぼ静止しており、復元時は 0 を書く)。

root の xy は **ロープの重心からの相対** で持つ。復元時に重心を env 原点へ
置き直すため。アーカイブから始めた env の状態がまたアーカイブへ入る連鎖で、
ロープが原点から離れていって発散判定 (水平 3 m) に近づくのを防ぐ。
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch


class StateArchive:
    """位相クラスごとのリングバッファ。すべてデバイス上のテンソルで持つ。

    Args:
        classes: 保存する位相クラス (例: `(1, 2)`)。それ以外の class は `add` で無視する。
        capacity: 1 クラスあたりの件数。満杯になったら古いものから上書きする。
        num_joints: ロープの関節数。
        device: テンソルを置くデバイス。
    """

    def __init__(self, classes: Sequence[int], capacity: int, num_joints: int, device: torch.device | str):
        self.classes = tuple(int(c) for c in classes)
        self.capacity = int(capacity)
        self.device = torch.device(device)
        num = len(self.classes)
        self.root_pose = torch.zeros(num, self.capacity, 7, device=self.device)
        self.joint_pos = torch.zeros(num, self.capacity, num_joints, device=self.device)
        self._size = [0] * num
        self._ptr = [0] * num
        self._classes_t = torch.tensor(self.classes, dtype=torch.long, device=self.device)
        # class -> スロット番号 (保存しない class は -1)。
        self._slot_of = torch.full((max(self.classes) + 1,), -1, dtype=torch.long, device=self.device)
        self._slot_of[self._classes_t] = torch.arange(num, device=self.device)

    def sizes(self) -> dict[int, int]:
        """クラス -> 保存件数。"""
        return dict(zip(self.classes, self._size))

    def add(self, cls: torch.Tensor, root_pose: torch.Tensor, joint_pos: torch.Tensor) -> None:
        """`(N,)` の class と、対応する `(N, 7)` / `(N, J)` の状態を保存する。

        対象外の class (保存しないクラス) の行は無視する。1 回に容量を超える件数が
        来たら、後ろの `capacity` 件だけを残す。
        """
        if cls.numel() == 0:
            return
        in_range = (cls >= 0) & (cls < self._slot_of.numel())
        slot = torch.where(in_range, self._slot_of[cls.clamp(0, self._slot_of.numel() - 1)], -1)
        for k in range(len(self.classes)):
            rows = (slot == k).nonzero(as_tuple=False).squeeze(-1)
            n = int(rows.numel())
            if n == 0:
                continue
            if n > self.capacity:
                rows = rows[-self.capacity :]
                n = self.capacity
            idx = (self._ptr[k] + torch.arange(n, device=self.device)) % self.capacity
            self.root_pose[k, idx] = root_pose[rows]
            self.joint_pos[k, idx] = joint_pos[rows]
            self._ptr[k] = (self._ptr[k] + n) % self.capacity
            self._size[k] = min(self._size[k] + n, self.capacity)

    def sample(
        self, n: int, class_weights: Sequence[float], generator: torch.Generator | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """`n` 件を取り出す。空のクラスは選ばない。アーカイブが空なら None。

        Args:
            n: 取り出す件数。
            class_weights: `classes` と同じ並びの重み。空でないクラスの中で正規化し、
                クラスを選んでから、そのクラスの中で一様に選ぶ。

        Returns:
            `(cls (n,), root_pose (n, 7), joint_pos (n, J))`。
        """
        sizes = torch.tensor(self._size, dtype=torch.float32, device=self.device)
        weights = torch.tensor(list(class_weights), dtype=torch.float32, device=self.device)
        weights = torch.where(sizes > 0, weights, torch.zeros_like(weights))
        if n == 0 or float(weights.sum()) <= 0.0:
            return None
        k = torch.multinomial(weights, n, replacement=True, generator=generator)
        u = torch.rand(n, device=self.device, generator=generator)
        i = (u * sizes[k]).long().clamp(max=self.capacity - 1)
        return self._classes_t[k], self.root_pose[k, i], self.joint_pos[k, i]


# ---------------------------------------------------------------- 姿勢の変換 (quaternion は w, x, y, z)


def _quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """`q1 ⊗ q2` `(N, 4)`。isaaclab.utils.math.quat_mul と同じ (純 torch にするため自前で書く)。"""
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dim=-1,
    )


def to_centroid_frame(root_pose_w: torch.Tensor, link_pos_w: torch.Tensor, env_origins: torch.Tensor) -> torch.Tensor:
    """world の root 姿勢 `(N, 7)` を「xy はロープ重心から、z は env 原点から」の相対にする。

    Args:
        root_pose_w: root link の姿勢 (world)。
        link_pos_w: `(N, L, 3)` のリンク中心 (world)。重心はこの xy の平均。
        env_origins: `(N, 3)`。
    """
    out = root_pose_w.clone()
    out[:, 0:2] -= link_pos_w[:, :, 0:2].mean(dim=1)
    out[:, 2] -= env_origins[:, 2]
    return out


def place_at_origin(rel_pose: torch.Tensor, env_origins: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """`to_centroid_frame` の逆。重心を env 原点に置き、重心まわりに `yaw` [rad] 回した world 姿勢。

    z 軸まわりの回転は投影図を回すだけなので、p-data (交点の順・上下・符号) は変わらない。
    """
    c, s = torch.cos(yaw), torch.sin(yaw)
    x, y = rel_pose[:, 0], rel_pose[:, 1]
    out = rel_pose.clone()
    out[:, 0] = c * x - s * y + env_origins[:, 0]
    out[:, 1] = s * x + c * y + env_origins[:, 1]
    out[:, 2] = rel_pose[:, 2] + env_origins[:, 2]
    half = 0.5 * yaw
    q_yaw = torch.stack([torch.cos(half), torch.zeros_like(half), torch.zeros_like(half), torch.sin(half)], dim=-1)
    out[:, 3:7] = _quat_mul(q_yaw, rel_pose[:, 3:7])
    return out


def random_yaw(n: int, device: torch.device | str, enabled: bool = True) -> torch.Tensor:
    """`(n,)` の一様乱数の yaw [rad] (`enabled=False` なら 0)。"""
    if not enabled:
        return torch.zeros(n, device=device)
    return (torch.rand(n, device=device) * 2.0 - 1.0) * math.pi
