"""ロープの対称変換 (鏡像・逆順) を、観測と行動のテンソルにかける純 torch の関数。v8 で追加。

v8 は PPO の更新で、集めた遷移を対称変換して学習データを増やす (rsl_rl の
`symmetry_cfg.use_data_augmentation`、`paper/Report_Paper.md` §7 P2-1。Yan2020 / TWISTED の
Mirror / Reverse)。rsl_rl に渡す関数は `common/mdp/symmetry.py` で、ここはその中身。

## 2 つの変換

* **Mirror** (env 原点を通る xz 面での鏡像、y -> −y): 物理的に別の状態。交差の上下は
  そのままで符号が反転する。固結びの判定 (`knot_detect`) は符号の揃い方しか見ず、左右の
  三葉結びをどちらも固結びとするので、段・class は変わらない。ロープは env 原点を通る x 軸
  上にまっすぐ置かれる (`rope_task_cfg.py`) ので、まっすぐなロープは自分自身の鏡像になる。
* **Reverse** (リンクの番号を逆にたどる): 物理的には **同じ状態** の付け替え。点の番号が
  逆順になり、上下と符号はそのまま (両方の紐の向きを同時に変えても交差の符号は変わらない)。

どちらも p-data の上の全単射で、R1 / R2 / cross の辺を辺へ写す。上の p-data の変換は、
ランダムなロープの p-data を直接計算したものと一致することを確かめた (`tests/test_symmetry.py`)。

## 行動 (6 ch、`mdp/actions.py` の `PickPlaceAction`)

| ch | 意味 | Mirror | Reverse |
|---|---|---|---|
| 0 | 掴むリンク | そのまま | 符号を反転 (`grasp_lo + grasp_hi = N−1` なので N−1−l) |
| 1 | Δlink | そのまま | 符号を反転 (`round` は 0 について対称) |
| 2 / 3 | 置き先 xy (target リンクからの変位) | y (ch3) を反転 | そのまま (target は同じ物理のリンク) |
| 4 | 離す高さ | そのまま | そのまま |
| 5 | 置き終わりの手首 yaw (`a·π`) | 符号を反転 | **+π** (`a` を ±1 ずらして [−1, 1) に畳む) |

Reverse の yaw: 掴むときの手首の向きはリンク i−1 -> i+1 の接線で決まる (`_tangent_yaw`) ので、
番号を逆にすると掴む向きが +π になる。運搬中の回転は最短回り (`apply_actions`) なので、置き
終わりの yaw も +π にすると回転量が元と同じになり、ロープは同じ向きで置かれる (平行グリッパーの
指が入れ替わるだけ)。+π にしないと、掴んだリンクを逆向きに置く別の手になる。
ch5 は clip ([−1, 1]) してからずらす。環境も clip してから yaw にするので、clip の外の値も
同じ物理の手へ写る。

## 観測

`policy` グループの項を、項の名前で 1 つずつ変換する (`augment_policy_obs`)。

* `rope_link_pos` / `hand_pos`: 座標を変換する。
* `crossings` / `writhe` / `crossing_feats` / `knot_class`: **変換したリンク位置から計算し直す**。
  環境の観測と同じく、リンク中心のポリライン (`polyline_from_link_centers`) から交差を取る
  ので、環境がその形状に対して出す値と一致する (手で変換式を書くより間違えにくい)。
* `next_edge`: 環境は位相が変わったときだけ辺を選び直す (状態を持つ) ので計算し直せない。
  選んだ辺の特徴を解析的に変換する (`transform_next_edge`)。対応は表の全 552 本の辺で
  全単射になることを確かめた (`tests/test_symmetry.py`)。
* `actions` (直前の行動): 行動と同じ変換。
* `knot_stage`: 変えない (段は不変。`StageGatedActorCritic` が末尾の one-hot で振り分ける)。
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from .intersections import IntersectionBatch, segment_intersections
from .knot_detect import class_one_hot, partner_gaps, reduce_r2
from .next_edge import NUM_MOVE_KINDS
from .polyline import polyline_from_link_centers

__all__ = [
    "TRANSFORMS",
    "ObsTermLayout",
    "augment_actions",
    "augment_policy_obs",
    "crossing_features_from_batch",
    "transform_actions",
    "transform_hand_pos",
    "transform_link_pos",
    "transform_next_edge",
]

TRANSFORMS = {
    "mirror_reverse": ((False, False), (True, False), (False, True), (True, True)),
    "mirror": ((False, False), (True, False)),
    "reverse": ((False, False), (False, True)),
}
"""拡張の組。各要素は `(mirror, reverse)`。先頭は必ず恒等 (rsl_rl は先頭を元のデータとして扱う)。"""

ObsTermLayout = tuple[str, int, dict]
"""観測の 1 項 `(名前, 次元, params)`。`policy` グループの並び順に並べる。"""

_RECOMPUTED_TERMS = ("crossings", "writhe", "crossing_feats", "knot_class")
"""変換したリンク位置から計算し直す項。"""

_UNCHANGED_TERMS = ("knot_stage",)
"""変換しない項。"""


# ---------------------------------------------------------------- 座標と行動


def transform_link_pos(link_pos: torch.Tensor, mirror: bool, reverse: bool) -> torch.Tensor:
    """リンク位置 `(B, N, 3)` (env 原点相対) を変換する。"""
    out = link_pos
    if mirror:
        out = out * out.new_tensor([1.0, -1.0, 1.0])
    if reverse:
        out = out.flip(1)
    return out


def transform_hand_pos(hand_pos: torch.Tensor, mirror: bool, reverse: bool) -> torch.Tensor:
    """手の位置 `(B, 3)`。Reverse は同じ物理の状態なので変えない。"""
    del reverse
    return hand_pos * hand_pos.new_tensor([1.0, -1.0, 1.0]) if mirror else hand_pos


def _wrap_unit(x: torch.Tensor) -> torch.Tensor:
    """`[−1, 1)` に畳む (yaw / π の周期 2)。"""
    return torch.remainder(x + 1.0, 2.0) - 1.0


def transform_actions(actions: torch.Tensor, mirror: bool, reverse: bool) -> torch.Tensor:
    """行動 `(B, 6)` を変換する (表はモジュールの docstring)。Mirror を先にかける。"""
    out = actions.clone()
    if mirror:
        out[:, 3] = -out[:, 3]
        out[:, 5] = -out[:, 5]
    if reverse:
        out[:, 0] = -out[:, 0]
        out[:, 1] = -out[:, 1]
        out[:, 5] = _wrap_unit(out[:, 5].clamp(-1.0, 1.0) + 1.0)
    return out


def transform_next_edge(feat: torch.Tensor, num_links: int, mirror: bool, reverse: bool) -> torch.Tensor:
    """`NextEdgeGoal` の特徴 `(B, 2N + 8)` を変換する。

    並び `[over マスク N, under マスク N, 種類 4, sign, left, obu, 有効]` (`next_edge.NextEdgeTable.features`)。
    表の全ての辺で確かめた対応:

    | | sign | left | obu | マスク |
    |---|---|---|---|---|
    | Mirror | 反転 | 反転 | そのまま | そのまま |
    | Reverse | そのまま | 反転 | 反転 | リンクの並びを逆に |

    該当しない値は 0 なので、符号の反転で 0 のまま残る。
    """
    n = num_links
    out = feat.clone()
    i_sign = 2 * n + NUM_MOVE_KINDS
    i_left, i_obu = i_sign + 1, i_sign + 2
    if mirror:
        out[:, i_sign] = -out[:, i_sign]
        out[:, i_left] = -out[:, i_left]
    if reverse:
        out[:, :n] = out[:, :n].flip(1)
        out[:, n : 2 * n] = out[:, n : 2 * n].flip(1)
        out[:, i_left] = -out[:, i_left]
        out[:, i_obu] = -out[:, i_obu]
    return out


# ---------------------------------------------------------------- 位相の項


def crossing_features_from_batch(batch: IntersectionBatch, num_nodes: int, max_crossings: int) -> torch.Tensor:
    """`mdp.crossing_features` と同じ値 `(B, max_crossings * 6)` を `IntersectionBatch` から作る。

    環境の項は抽出器 (isaaclab に依存) を経由するので、同じ計算をここに持つ。一致は
    Isaac Sim の確認 (`scripts/rsl_rl/symmetry_check.py` の観測の照合) で確かめる。
    """
    device = batch.data.device
    num_envs = batch.num_envs
    out = torch.zeros(num_envs, max_crossings, 6, device=device)
    num_slots = min(batch.max_crossings, max_crossings)
    if num_slots == 0:
        return out.reshape(num_envs, -1)

    denom = float(max(2 * num_nodes, 1))
    valid = batch.valid_mask[:, :num_slots]
    order = batch.order[:, :num_slots].float()
    data = batch.data[:, :num_slots].float()
    gaps = partner_gaps(batch)[:, :num_slots].float()
    feat = torch.stack(
        [valid.float(), data[..., 2], data[..., 3], order[..., 0] / denom, order[..., 1] / denom, gaps / denom],
        dim=-1,
    )
    feat = torch.where(valid.unsqueeze(-1), feat, torch.zeros_like(feat))
    key = torch.where(valid, batch.order[:, :num_slots, 0], torch.full_like(batch.order[:, :num_slots, 0], 1 << 30))
    sort_idx = key.argsort(dim=-1)
    out[:, :num_slots] = feat.gather(1, sort_idx.unsqueeze(-1).expand_as(feat))
    return out.reshape(num_envs, -1)


def _recompute_terms(link_pos: torch.Tensor, layout: Sequence[ObsTermLayout]) -> dict[str, torch.Tensor]:
    """リンク位置 `(B, N, 3)` から、計算し直す項の値を作る。"""
    nodes = polyline_from_link_centers(link_pos)
    batch = segment_intersections(nodes)
    out: dict[str, torch.Tensor] = {}
    for name, _, params in layout:
        if name == "crossings":
            out[name] = batch.count.float().unsqueeze(-1)
        elif name == "writhe":
            out[name] = batch.writhe.float().unsqueeze(-1)
        elif name == "crossing_feats":
            out[name] = crossing_features_from_batch(batch, nodes.shape[1], int(params.get("max_crossings", 5)))
        elif name == "knot_class":
            out[name] = class_one_hot(reduce_r2(batch) if params.get("reduce_r2", False) else batch)
    return out


# ---------------------------------------------------------------- 拡張


def _transform_policy_obs(
    obs: torch.Tensor, layout: Sequence[ObsTermLayout], mirror: bool, reverse: bool
) -> torch.Tensor:
    dims = [d for _, d, _ in layout]
    if sum(dims) != obs.shape[-1]:
        raise ValueError(f"観測の次元 {obs.shape[-1]} が項の次元の和 {sum(dims)} と合わない")
    parts = dict(zip([name for name, _, _ in layout], obs.split(dims, dim=-1)))
    if "rope_link_pos" not in parts:
        raise ValueError("観測に rope_link_pos が無いと位相の項を計算し直せない")
    link_pos = transform_link_pos(parts["rope_link_pos"].reshape(obs.shape[0], -1, 3), mirror, reverse)
    num_links = link_pos.shape[1]
    recomputed = _recompute_terms(link_pos, layout)
    out = []
    for name, _, _ in layout:
        x = parts[name]
        if name == "rope_link_pos":
            x = link_pos.reshape(obs.shape[0], -1)
        elif name in _RECOMPUTED_TERMS:
            x = recomputed[name].to(x.dtype)
        elif name == "hand_pos":
            x = transform_hand_pos(x, mirror, reverse)
        elif name == "actions":
            x = transform_actions(x, mirror, reverse)
        elif name == "next_edge":
            x = transform_next_edge(x, num_links, mirror, reverse)
        elif name not in _UNCHANGED_TERMS:
            raise ValueError(f"対称変換を決めていない観測の項: {name!r}")
        out.append(x)
    return torch.cat(out, dim=-1)


def augment_policy_obs(obs: torch.Tensor, layout: Sequence[ObsTermLayout], transforms: str) -> torch.Tensor:
    """観測 `(B, D)` を `TRANSFORMS[transforms]` の順に変換して縦に並べる `(K·B, D)`。先頭 B 行は元のまま。"""
    blocks = [obs]
    for mirror, reverse in TRANSFORMS[transforms][1:]:
        blocks.append(_transform_policy_obs(obs, layout, mirror, reverse))
    return torch.cat(blocks, dim=0)


def augment_actions(actions: torch.Tensor, transforms: str) -> torch.Tensor:
    """行動 `(B, 6)` を `TRANSFORMS[transforms]` の順に変換して縦に並べる `(K·B, 6)`。"""
    return torch.cat([transform_actions(actions, m, r) for m, r in TRANSFORMS[transforms]], dim=0)
