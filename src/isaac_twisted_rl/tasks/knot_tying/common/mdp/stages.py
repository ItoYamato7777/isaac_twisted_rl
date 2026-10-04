"""段 (`BfsPotential.stage` = round(3Φ)。0 = 自明・3 = 固結び) を単位にした「1 遷移 = 1 エピソード」の項。v4 で追加。

TWISTED-RL の多段エピソード (explainer `twisted-rl.html` §06) を移植する:

* エピソードは段 k から始まり、**段が変わった時点で終わる** (最大 M 手)。
* 報酬は段が上がれば +1、変わらなければ 0、下がれば −1 の疎報酬だけ。
* 段が変わらない手 (報酬 0) で形を整えてから遷移を起こしてよい。

段は `topology.BfsPotential.stage` (`round(3 Φ)`、0 = 自明 〜 3 = 固結び)。論文の
「交差数」の代わりで、cover 後 (生の交差 3・R2 除去後 1) を段 2 と数えられる。

| 項 | 種類 | 役割 |
|---|---|---|
| `knot_stage_one_hot` | 観測 | 段の one-hot (4)。**観測グループの最後に置く** (`StageGatedActorCritic` がここで振り分ける) |
| `stage_transition_reward` | 報酬 | +1 / 0 / −1 (発散は `diverged` の値) |
| `stage_changed` | 終了 | `direction="up"` / `"down"`。`chain=True` (Play) は固結びでだけ終わる |

## 段が「変わった」の判定

段が上がった・下がったは、手の終わりにロープが床に落ちていて (`settled`) 発散していない
ときだけ数える (`ladder_class_reached` / `LadderProgressBonus` と同じ条件)。吊ったまま
では位相が確定しないので、その手は「変わらない」扱いで続ける。

段が同じなら、p-data が変わっても続ける (論文は「別の位相状態へ移ったら −1 で終了」で、
これより厳しい)。`strict_topology=True` で論文どおりにできる: 段が同じでも R2 除去前の
p-data (BFS 表のキー) が変われば失敗として終える。キーは交差 4 個までしか作れないので、
前後とも 5 個以上のときは「変わらない」と見なす。

## 報酬と終了の判定を揃える

報酬と終了判定は同じ env ステップで別々に呼ばれる (終了 -> 報酬の順)。両者が同じ値を
見るよう、手の前後の段を `stage_transition` で 1 回だけ計算してキャッシュする。
キャッシュの見張り番は `common_step_counter` と、抽出器の batch の同一性 (reset 後に
抽出し直すと別オブジェクトになる)。観測は reset 後の形状を見る必要があるので、
キャッシュを使わず `current_stage` で毎回引く。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

from .. import topology as tp
from .observations import get_extractor, rope_zmax
from .terminations import diverged_now

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

_CACHE_ATTR = "_knot_tying_stage_transition"


def current_stage(env: ManagerBasedRLEnv, rope_cfg: SceneEntityCfg = SceneEntityCfg("rope")) -> torch.Tensor:
    """いまのロープの段 `(num_envs,)` int64 (0〜`STAGE_GOAL`)。"""
    extractor = get_extractor(env, rope_cfg.name)
    return tp.get_bfs_potential(env.device).stage(extractor.batch, extractor.reduced_batch)


def stage_transition(
    env: ManagerBasedRLEnv,
    action_name: str = "pick_place",
    rope_cfg: SceneEntityCfg = SceneEntityCfg("rope"),
) -> dict[str, torch.Tensor]:
    """この手の前後の段と BFS 表のキー。`before` / `after` / `key_before` / `key_after` (各 `(num_envs,)`)。

    手の前は `PickPlaceAction` が控えた batch から引く (まだ 1 手も打っていなければ段 0)。
    同じ env ステップ・同じ抽出結果なら 2 回目以降はキャッシュを返す。
    """
    extractor = get_extractor(env, rope_cfg.name)
    batch = extractor.batch
    cached = getattr(env, _CACHE_ATTR, None)
    if cached is not None and cached[0] == env.common_step_counter and cached[1] is batch:
        return cached[2]
    pot = tp.get_bfs_potential(env.device)
    term = env.action_manager.get_term(action_name)
    raw_before = term.batch_at_action_start
    after = pot.stage(batch, extractor.reduced_batch)
    key_after = tp.batch_keys(batch)
    if raw_before is None:
        before, key_before = torch.zeros_like(after), key_after.clone()
    else:
        before = pot.stage(raw_before, term.reduced_batch_at_action_start)
        key_before = tp.batch_keys(raw_before)
    out = {"before": before, "after": after, "key_before": key_before, "key_after": key_after}
    setattr(env, _CACHE_ATTR, (env.common_step_counter, batch, out))
    return out


def _outcome(
    env: ManagerBasedRLEnv,
    strict_topology: bool,
    settled_height_diameters: float,
    rope_diameter: float,
    diverged_term: str | None,
    action_name: str,
    rope_cfg: SceneEntityCfg,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """`(up, down, diverged, after)`。up / down は床に落ちていて発散していない手だけ。"""
    tr = stage_transition(env, action_name, rope_cfg)
    before, after = tr["before"], tr["after"]
    diverged = (
        diverged_now(env, diverged_term) if diverged_term is not None else torch.zeros_like(after, dtype=torch.bool)
    )
    valid = (rope_zmax(env, rope_cfg) <= settled_height_diameters * rope_diameter) & ~diverged
    up = (after > before) & valid
    down = (after < before) & valid
    if strict_topology:
        down |= (after == before) & (tr["key_after"] != tr["key_before"]) & valid
    return up, down, diverged, after


# ---------------------------------------------------------------- 観測


def knot_stage_one_hot(env: ManagerBasedRLEnv, rope_cfg: SceneEntityCfg = SceneEntityCfg("rope")) -> torch.Tensor:
    """段の one-hot `(num_envs, STAGE_GOAL + 1)`。

    `StageGatedActorCritic` はこの列の argmax でエージェントを選ぶので、**観測グループの
    最後の項** にすること。
    """
    stage = current_stage(env, rope_cfg)
    return torch.nn.functional.one_hot(stage, num_classes=tp.STAGE_GOAL + 1).float()


# ---------------------------------------------------------------- 報酬


def stage_transition_reward(
    env: ManagerBasedRLEnv,
    success: float = 1.0,
    failure: float = -1.0,
    diverged: float = -1.0,
    strict_topology: bool = False,
    settled_height_diameters: float = 3.0,
    rope_diameter: float = 0.02,
    diverged_term: str | None = "diverged",
    action_name: str = "pick_place",
    rope_cfg: SceneEntityCfg = SceneEntityCfg("rope"),
) -> torch.Tensor:
    """段が上がれば `success`、下がれば `failure`、発散は `diverged`、それ以外 0 `(num_envs,)`。

    TWISTED-RL の +1 / 0 / −1 (explainer `twisted-rl.html` §02)。重み 1.0 で使う。
    `rope_diameter` は `rope_task_cfg._apply_rope` がプリセットの値で埋める。
    """
    up, down, div, _ = _outcome(
        env, strict_topology, settled_height_diameters, rope_diameter, diverged_term, action_name, rope_cfg
    )
    out = torch.zeros(env.num_envs, device=env.device)
    out = torch.where(up, torch.full_like(out, success), out)
    out = torch.where(down, torch.full_like(out, failure), out)
    return torch.where(div, torch.full_like(out, diverged), out)


# ---------------------------------------------------------------- 終了


def stage_changed(
    env: ManagerBasedRLEnv,
    direction: str = "up",
    chain: bool = False,
    strict_topology: bool = False,
    settled_height_diameters: float = 3.0,
    rope_diameter: float = 0.02,
    diverged_term: str | None = "diverged",
    action_name: str = "pick_place",
    rope_cfg: SceneEntityCfg = SceneEntityCfg("rope"),
) -> torch.Tensor:
    """段が上がった (`direction="up"`) / 下がった (`"down"`) `(num_envs,)`。`time_out=False` で使う。

    `chain=True` (Play の評価) では段をつないで固結びまで続けるので、`"up"` は固結びに
    達したときだけ、`"down"` は常に False (下がってもやり直す) になる。観測の段が変われば
    `StageGatedActorCritic` が次の段のエージェントに切り替える。
    """
    if direction not in ("up", "down"):
        raise ValueError(f"direction は 'up' か 'down' (got {direction!r})")
    up, down, _, after = _outcome(
        env, strict_topology, settled_height_diameters, rope_diameter, diverged_term, action_name, rope_cfg
    )
    if direction == "up":
        return up & (after >= tp.STAGE_GOAL) if chain else up
    return torch.zeros_like(down) if chain else down
