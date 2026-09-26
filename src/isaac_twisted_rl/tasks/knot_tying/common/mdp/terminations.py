"""終了判定。

`next_action.md` 4.5 の NaN 対策がここの主目的。**エピソードが NaN で落ちるのは
RL では致命的**なので、発散した環境はその場で終了させて reset に回す
(報酬側では `rewards.action_failed` が罰を与える)。

4.4 の「退化 (交差の順序が一意に決まらない) 時の扱い」については、
観測・報酬・終了判定をすべてテンソル層だけで組んだ結果、
`intersections_to_topology` を呼ぶ経路自体が存在しない。したがって
あの `ValueError` は学習中に発生しえない。p-data 文字列はログ用の
任意呼び出しに留めてある。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

from .observations import get_extractor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def rope_diverged(
    env: ManagerBasedRLEnv,
    max_height: float = 5.0,
    rope_cfg: SceneEntityCfg = SceneEntityCfg("rope"),
) -> torch.Tensor:
    """ロープが NaN になった / 明らかに吹き飛んだ `(num_envs,)`。

    NaN そのものだけでなく高さの上限も見るのは、発散は多くの場合
    「NaN になる前に非現実的な速度で飛ぶ」段階を経るため。先に捕まえた方が
    次のエピソードへの汚染 (PhysX 内部状態) が小さい。
    """
    nodes = get_extractor(env, rope_cfg.name).nodes
    not_finite = ~torch.isfinite(nodes).all(dim=-1).all(dim=-1)
    origin_z = env.scene.env_origins[:, 2].unsqueeze(-1)
    height = torch.nan_to_num(nodes[:, :, 2] - origin_z, nan=0.0).max(dim=-1).values
    return not_finite | (height > max_height)


def topology_reached(
    env: ManagerBasedRLEnv,
    min_crossings: int = 1,
    rope_cfg: SceneEntityCfg = SceneEntityCfg("rope"),
) -> torch.Tensor:
    """目標の交差数に達した `(num_envs,)`。

    既定の環境設定では **使っていない**。交差を 1 つ作った時点で打ち切ると
    「作った交差を壊さずに保つ」ことを学習できないため、既定は時間切れまで
    続ける方針にしてある。カリキュラムを組むときのために用意しておく。
    """
    return get_extractor(env, rope_cfg.name).count >= min_crossings
