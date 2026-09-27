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
    max_xy_distance: float | None = None,
    max_link_speed: float | None = None,
    rope_cfg: SceneEntityCfg = SceneEntityCfg("rope"),
) -> torch.Tensor:
    """ロープが NaN になった / 明らかに吹き飛んだ `(num_envs,)`。

    NaN そのものだけでなく高さの上限も見るのは、発散は多くの場合
    「NaN になる前に非現実的な速度で飛ぶ」段階を経るため。先に捕まえた方が
    次のエピソードへの汚染 (PhysX 内部状態) が小さい。

    `max_xy_distance` / `max_link_speed` は v2-R2 で追加した (既定の None は
    判定に加えない = v0〜v2 と同じ挙動)。高さだけだと「床の上を水平に
    吹き飛んだ有限の座標」を見逃すため。ランダム方策の実測 (512 環境 x 24 手,
    2026-09-27) では、正常な手の終わりは xy が env 原点から最大 1.52 m、
    リンク速度が最大 3.3 m/s で、発散時は数十〜数億 m / 60〜175 m/s になる。
    """
    extractor = get_extractor(env, rope_cfg.name)
    nodes = extractor.nodes
    not_finite = ~torch.isfinite(nodes).all(dim=-1).all(dim=-1)
    origin = env.scene.env_origins.unsqueeze(1)
    rel = torch.nan_to_num(nodes - origin, nan=0.0)
    height = rel[:, :, 2].max(dim=-1).values
    out = not_finite | (height > max_height)
    if max_xy_distance is not None:
        out |= rel[:, :, :2].norm(dim=-1).max(dim=-1).values > max_xy_distance
    if max_link_speed is not None:
        rope = env.scene[rope_cfg.name]
        vel = rope.data.body_lin_vel_w[:, extractor.body_index]
        speed = torch.nan_to_num(vel, nan=0.0).norm(dim=-1).max(dim=-1).values
        out |= speed > max_link_speed
    return out


_DIVERGED_CACHE_ATTR = "_knot_tying_diverged_now"


def diverged_now(env: ManagerBasedRLEnv, term_name: str = "diverged") -> torch.Tensor:
    """**この env ステップで** 終了項 `term_name` が立っているか `(num_envs,)` bool。

    報酬項が「発散した手」を見分けるために使う (v2-R2)。判定の閾値は
    終了項の params をそのまま使うので、報酬と終了判定で基準がずれない。

    ## `termination_manager.get_term()` を使ってはいけない理由

    `TerminationManager` の `_term_dones` は「直近のエピソードを **終わらせた**
    項」を保持する作りで、別の項が立つまで True が残り続ける
    (`compute()` は立った行だけを書き換える)。つまり `get_term("diverged")`
    は「前のエピソードが発散で終わった」環境でも True を返す。実測では
    発散が 1 手あたり 3.4% しか起きていないのに、`get_term` は 17.5% を
    True にしていた。そこで終了項の関数を直接呼び直す。

    同じ env ステップ内の 2 回目以降はキャッシュを返す (観測の抽出器と同じく
    `common_step_counter` を見張り番にする)。`ManagerBasedRLEnv.step()` は
    終了判定 -> 報酬 -> reset の順なので、報酬から呼べば reset 前の状態を見る。
    """
    stamp = env.common_step_counter
    cached = getattr(env, _DIVERGED_CACHE_ATTR, None)
    if cached is not None and cached[0] == stamp and cached[1] == term_name:
        return cached[2]
    cfg = env.termination_manager.get_term_cfg(term_name)
    value = cfg.func(env, **cfg.params)
    setattr(env, _DIVERGED_CACHE_ATTR, (stamp, term_name, value))
    return value


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
