"""観測項。**テンソル層だけ**を使い、p-data 文字列は一切作らない。

`next_action.md` 4.3 の指示どおり、観測には `IntersectionBatch` の数値を
そのまま流す。理由は `topology/state_2_topology.py` の docstring にある実測値で、
**記号層 (p-data 文字列) は 4096 環境で 94 ms、テンソル層は 1.6 ms** と 60 倍
違う。毎ステップ全環境で文字列を作ると、それだけで学習が実時間で回らなくなる。

p-data 文字列が要るのはログと人間のデバッグだけで、そちらは
`RopeTopologyExtractor.p_data()` を任意のタイミングで呼べばよい。

## 抽出器の共有

`RopeTopologyExtractor` は env につき 1 つで足りる (body 順序の解決結果を
持つだけ)。観測・報酬・終了判定がそれぞれ別に持って `update()` を
重複実行すると交差検出が 3 回走るので、`get_extractor(env)` で 1 つを共有し、
`update()` は **その env ステップで最初に呼ばれたときだけ** 回す。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

from .. import topology as tp
from ..rope_state import RopeTopologyExtractor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

_EXTRACTOR_ATTR = "_knot_tying_extractor"
_STAMP_ATTR = "_knot_tying_extractor_stamp"


def get_extractor(env: ManagerBasedRLEnv, rope_name: str = "rope") -> RopeTopologyExtractor:
    """env に紐づく `RopeTopologyExtractor` を返す (無ければ作る)。

    同じ env ステップ内で何度呼ばれても交差検出は 1 回しか走らない。
    `env.common_step_counter` を見張り番に使う。
    """
    extractor: RopeTopologyExtractor | None = getattr(env, _EXTRACTOR_ATTR, None)
    rope: Articulation = env.scene[rope_name]
    if extractor is None:
        extractor = RopeTopologyExtractor(rope)
        setattr(env, _EXTRACTOR_ATTR, extractor)
        setattr(env, _STAMP_ATTR, -1)
    stamp = getattr(env, _STAMP_ATTR)
    if stamp != env.common_step_counter:
        extractor.update(rope)
        setattr(env, _STAMP_ATTR, env.common_step_counter)
    return extractor


def invalidate_extractor(env: ManagerBasedRLEnv) -> None:
    """次の `get_extractor()` で必ず `update()` を回させる。

    `ManagerBasedRLEnv.step()` は
    `terminations -> rewards -> _reset_idx -> observations` の順に処理する。
    reset された環境はロープの姿勢が初期状態へ戻っているのに、見張り番
    (`common_step_counter`) は同じ env ステップのままなので、**観測が
    reset 前のロープ形状のまま計算されてしまう**。

    `PickPlaceAction.reset()` (= `_reset_idx` の中で呼ばれる) からこれを
    呼ぶことで、直後の観測計算で必ず抽出し直させる。
    """
    if hasattr(env, _STAMP_ATTR):
        setattr(env, _STAMP_ATTR, -1)


def rope_link_positions(env: ManagerBasedRLEnv, rope_cfg: SceneEntityCfg = SceneEntityCfg("rope")) -> torch.Tensor:
    """全リンク中心の env 原点相対座標 `(num_envs, 3 * num_links)`。

    ロープの形そのもの。位相 (交差数・writhe) だけでは「どこを掴めば交差を
    作れるか」が分からないので、幾何も一緒に渡す必要がある。
    env 原点を引くのは、複数環境で同じ方策が使えるようにするため。
    """
    rope: Articulation = env.scene[rope_cfg.name]
    extractor = get_extractor(env, rope_cfg.name)
    pos = rope.data.body_pos_w[:, extractor.body_index] - env.scene.env_origins.unsqueeze(1)
    return torch.nan_to_num(pos, nan=0.0, posinf=0.0, neginf=0.0).reshape(env.num_envs, -1)


def crossing_count(env: ManagerBasedRLEnv, rope_cfg: SceneEntityCfg = SceneEntityCfg("rope")) -> torch.Tensor:
    """交差数 `(num_envs, 1)`。最初の課題 (交差 1 つを作る) の主指標。"""
    return get_extractor(env, rope_cfg.name).count.float().unsqueeze(-1)


def writhe(env: ManagerBasedRLEnv, rope_cfg: SceneEntityCfg = SceneEntityCfg("rope")) -> torch.Tensor:
    """符号の総和 `(num_envs, 1)`。位相の粗い指標。"""
    return get_extractor(env, rope_cfg.name).writhe.float().unsqueeze(-1)


def first_crossing_feature(env: ManagerBasedRLEnv, rope_cfg: SceneEntityCfg = SceneEntityCfg("rope")) -> torch.Tensor:
    """先頭 1 交差の特徴 `(num_envs, 4)` = `(over, sign, point_i/N, point_j/N)`。

    .. deprecated::
        `crossing_features` に置き換えた (交差 1 つしか見えず、正規化の
        分母がバッチごとに変わるため)。ablation 比較用に残してあるだけで
        `ObservationsCfg` からは外れている。新規の用途では使わないこと。

    p-data 文字列 `'1O2+_2U1+'` が持っている情報を、文字列を作らずに数値へ
    落としたもの。最初の課題は「交差 1 つ」なので 1 つぶんで足りる。
    交差が無い環境は 0 埋め。`point_*` は 1 始まりの出現順なので節点数で
    割って正規化する (リンク数の違うプリセット間で意味が揃う)。
    """
    batch = get_extractor(env, rope_cfg.name).batch
    out = torch.zeros(env.num_envs, 4, device=batch.data.device)
    if batch.max_crossings == 0:
        return out
    valid = batch.valid_mask[:, 0]
    data = batch.data[:, 0, :].float()
    order = batch.order[:, 0, :].float()
    denom = float(max(batch.data.shape[1], 1))
    feat = torch.stack([data[:, 2], data[:, 3], order[:, 0] / denom, order[:, 1] / denom], dim=-1)
    return torch.where(valid.unsqueeze(-1), feat, out)


def crossing_features(
    env: ManagerBasedRLEnv,
    max_crossings: int = 5,
    rope_cfg: SceneEntityCfg = SceneEntityCfg("rope"),
) -> torch.Tensor:
    """交差 K 個ぶんの特徴 `(num_envs, max_crossings * 6)`。

    1 交差あたり `(valid, over, sign, point_i/2N, point_j/2N, gap/2N)` の 6 個。
    `first_crossing_feature` を置き換える。変更点は 3 つ:

    **(a) 先頭 1 交差 -> K 個。** 固結びは交差 3 つなので 1 つでは足りない。
    K=5 にしてあるのは、実機では目標の 3 つに加えてノイズの交差が 1〜2 つ
    乗るため (`knot_detect.py` の docstring 参照)。

    **(b) 正規化の分母を定数にした。** 旧実装は `batch.data.shape[1]`
    (= `max_crossings`、**バッチごとに変動する**) で割っていて、docstring の
    「節点数で割る」とも食い違っていた。値が非定常かつ非有界になり、
    `empirical_normalization=False` のまま critic に分布シフトを注入する。
    点番号は `1..2*count` かつ `count <= num_nodes` なので、`2 * num_nodes`
    で割れば全成分が `(0, 1]` に収まり、バッチにもプリセットにも依らない。

    **(c) 走査順に並べ替えた。** `batch.data` の行順は `(seg_i, seg_j)` の
    辞書順であって、ロープを端からたどった順ではない。並べ替えずに渡すと、
    同じ結び目でもセグメントがずれただけで方策には別の並びに見えてしまう。

    `gap` を明示的に入れているのは、これが固結び判定そのもの
    (`partner_gaps`) だから。方策に `|a - b|` を学ばせるより直接渡す方が良い。
    """
    extractor = get_extractor(env, rope_cfg.name)
    batch = extractor.batch
    device = batch.data.device
    out = torch.zeros(env.num_envs, max_crossings, 6, device=device)
    num_slots = min(batch.max_crossings, max_crossings)
    if num_slots == 0:
        return out.reshape(env.num_envs, -1)

    denom = float(max(2 * extractor.num_nodes, 1))
    valid = batch.valid_mask[:, :num_slots]
    order = batch.order[:, :num_slots].float()
    data = batch.data[:, :num_slots].float()
    gaps = tp.partner_gaps(batch)[:, :num_slots].float()

    feat = torch.stack(
        [
            valid.float(),
            data[..., 2],
            data[..., 3],
            order[..., 0] / denom,
            order[..., 1] / denom,
            gaps / denom,
        ],
        dim=-1,
    )
    feat = torch.where(valid.unsqueeze(-1), feat, torch.zeros_like(feat))

    # 走査順 (point_i 昇順) に並べ替える。無効行は末尾へ送る。
    key = torch.where(
        valid,
        batch.order[:, :num_slots, 0],
        torch.full_like(batch.order[:, :num_slots, 0], 1 << 30),
    )
    sort_idx = key.argsort(dim=-1)
    feat = feat.gather(1, sort_idx.unsqueeze(-1).expand_as(feat))

    out[:, :num_slots] = feat
    return out.reshape(env.num_envs, -1)


def knot_class_one_hot(env: ManagerBasedRLEnv, rope_cfg: SceneEntityCfg = SceneEntityCfg("rope")) -> torch.Tensor:
    """位相クラスの one-hot `(num_envs, 4)`。報酬ラダーの段をそのまま観測に渡す。

    段は `{交差なし/行き止まり, 交差 1, 絡み 2 交差, 固結び}`
    (`topology/knot_detect.py`)。報酬が同じ量で計算されるので、方策から
    「今どの段にいるか」が直接見えるようにしておく。
    """
    return tp.class_one_hot(get_extractor(env, rope_cfg.name).batch)


def hand_position(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("gripper")) -> torch.Tensor:
    """ハンド root の env 原点相対位置 `(num_envs, 3)`。"""
    hand: Articulation = env.scene[asset_cfg.name]
    pos = hand.data.root_pos_w - env.scene.env_origins
    return torch.nan_to_num(pos, nan=0.0, posinf=0.0, neginf=0.0)


def rope_zmax(env: ManagerBasedRLEnv, rope_cfg: SceneEntityCfg = SceneEntityCfg("rope")) -> torch.Tensor:
    """ロープの最高点 `(num_envs,)` (env 原点相対)。

    観測ではなく報酬・ログ用。「ハンドが吊っているだけ」を「交差が残った」と
    取り違えないための判定材料 (`next_action.md` 4.2 が必ず出せと指示している)。
    """
    extractor = get_extractor(env, rope_cfg.name)
    z = extractor.nodes[:, :, 2] - env.scene.env_origins[:, 2].unsqueeze(-1)
    return torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0).max(dim=-1).values


def finger_gap(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("gripper")) -> torch.Tensor:
    """指の隙間 `(num_envs,)` [m]。`rope_zmax` と同じくログ用。"""
    hand: Articulation = env.scene[asset_cfg.name]
    finger_ids, _ = hand.find_joints(["left_finger_joint", "right_finger_joint"], preserve_order=True)
    q = hand.data.joint_pos[:, finger_ids]
    return 0.5 * q.abs().sum(dim=-1)
