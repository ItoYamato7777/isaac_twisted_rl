"""固結び (overhand knot / 三葉結び) をテンソル層だけで判定する。

`intersections.py` が返す `IntersectionBatch` から、**記号層 (p-data 文字列) を
一切作らずに** 結び目の種類を判定する。`state_2_topology.py` の実測値どおり
記号層は 4096 環境で 94 ms かかるので、毎ステップの RL パスには載せられない。
テンソル層なら 1.6 ms で、この判定を足しても 1 ms 程度しか増えない。

## なぜテンソル層だけで判定できるのか

移植元 (`itoyama_twisted_rl`) の BFS で **到達可能な 3 交差状態 528 個を
全列挙**して調べた結果、固結びは以下の 3 条件で厳密に特徴づけられる:

1. 交差数が 3
2. **partner gap が全て 3** — 交差点 *i* の相手が *i+3*
   (`|order[..., 0] - order[..., 1]| == 3`)
3. **符号が全て同じ** (`|writhe| == 3`)

528 個のうちこの 3 条件を満たすのはちょうど 4 個で、それは固結びの
p-data 4 通り (キラリティ 2 × 走査方向 2) と完全に一致する::

    1U4+_2O5+_3U6+_4O1+_5U2+_6O3+      1O4+_2U5+_3O6+_4U1+_5O2+_6U3+
    1U4-_2O5-_3U6-_4O1-_5U2-_6O3-      1O4-_2U5-_3O6-_4U1-_5O2-_6U3-

**条件 2 だけでは足りない** (16 個残り、符号条件で 4 個に絞られる)。
**条件 3 だけでも足りない** — 独立した R1 ループを 3 つ作ると
`1U2+_2O1+_3U4+_4O3+_5U6+_6O5+` となり、同符号だが gap は `{1,1,1}`。
**両方必要**。

## なぜ厳密な `count == 3` を使わないのか (重要)

実機のロープでは使えないため。三葉結びの尻尾に小さなループが 1 つ
付いただけで交差数は 7 に増え、gap は `[3,3,3]` から
`[5,5,9,9,3,5,1]` へ **全ての点番号が振り直される**。結び目は物理的に
存在するのに、厳密判定は何も見つけられない。20 リンクのたるんだロープでは
これは例外ではなく常態。

そこで **部分列 (subsequence) 判定**にする。「交差の k 個組のうち、
**その k 個だけで番号を振り直したとき** に gap が全て k かつ同符号に
なる組が存在するか」を見る。実測:

    入力                      交差数   判定
    三葉結び (きれい)            3     True
    三葉結び + 尻尾ループ        7     True   <- 厳密判定では False になる
    結ばれていないコイル         3     False
    ランダムロープ 600 本      0〜5    誤検出 0 %

## 計算量

`MAX_CROSSINGS_CONSIDERED = 8` で打ち切ると 3 個組は C(8,3) = 56 通り。
添字はモジュール読み込み時に 1 回だけ作る定数テンソルで、env ごとの
Python ループは持たない。512 環境で CPU float64 5.8 ms / call
(GPU float32 なら 1 ms を大きく下回る)。1 手が 686 物理ステップなので
無視できる。
"""

from __future__ import annotations

import itertools

import torch

from .intersections import IntersectionBatch

# ---------------------------------------------------------------- 位相クラス
# 報酬ラダーの段。0 は「ラダーに乗っていない」も兼ねる。
CLASS_TRIVIAL = 0
"""交差なし、またはラダーから外れた位相 (独立ツイストなど行き止まり)。"""

CLASS_SINGLE = 1
"""交差 1 つ。これまでのタスクの目標がそのまま 1 段目になる。"""

CLASS_INTERLEAVED2 = 2
"""2 交差・gap 2・同符号。ループに端をくぐらせかけている途中。"""

CLASS_OVERHAND = 3
"""固結び。"""

NUM_CLASSES = 4

MAX_CROSSINGS_CONSIDERED = 8
"""部分列探索で見る交差数の上限。

ロープがぐしゃぐしゃになった環境で C(count, 3) が爆発するのを防ぐ。
8 なら 56 通り。実機で結び目が成立しているときの交差数は 3〜7 程度なので
実用上の取りこぼしは無い (`README.md` の計測項目 `count` ヒストグラムで
監視すること)。
"""


def _subset_indices(num_slots: int, k: int) -> torch.Tensor:
    """`(T, k)` の組み合わせ添字。モジュール内のキャッシュ経由でのみ使う。"""
    return torch.tensor(list(itertools.combinations(range(num_slots), k)), dtype=torch.long)


# (num_slots, k) -> (T, k) の添字テンソル。デバイスはコピー時に合わせる。
_SUBSET_CACHE: dict[tuple[int, int], torch.Tensor] = {}


def _subsets(num_slots: int, k: int, device: torch.device) -> torch.Tensor:
    """`num_slots` 個から `k` 個選ぶ添字 `(T, k)` を `device` 上で返す。

    組み合わせの列挙は Python なので、**呼ばれるたびに作り直さない**よう
    (num_slots, k) をキーにキャッシュする。値は形状固定の定数テンソル。
    """
    key = (num_slots, k)
    cached = _SUBSET_CACHE.get(key)
    if cached is None or cached.device != device:
        cached = _subset_indices(num_slots, k).to(device)
        _SUBSET_CACHE[key] = cached
    return cached


def partner_gaps(batch: IntersectionBatch) -> torch.Tensor:
    """各交差の相手までの距離 `(num_envs, max_crossings)` int64。

    `|order[..., 0] - order[..., 1]|`。無効な行 (パディング) は 0。
    固結びの判定そのものに使うほか、観測にも渡す (方策に `|a - b|` を
    学ばせるより直接与えるほうが良い)。
    """
    order = batch.order
    gaps = (order[..., 0] - order[..., 1]).abs()
    return torch.where(batch.valid_mask, gaps, torch.zeros_like(gaps))


def sign_consistent(batch: IntersectionBatch) -> torch.Tensor:
    """全交差の符号が揃っているか `(num_envs,)` bool。

    `|writhe| == count` と同値 (符号が ±1 のみなので、総和の絶対値が
    個数に一致するのは全て同符号のときだけ)。交差 0 個は False。
    """
    count = batch.count
    return (batch.writhe.abs() == count) & (count > 0)


def has_knot_subsequence(batch: IntersectionBatch, k: int = 3) -> torch.Tensor:
    """k 個組で「交互・同符号」を成す部分列が在るか `(num_envs,)` bool。

    モジュール docstring の「部分列判定」の本体。選んだ k 個の交差だけを
    取り出して点番号を 1..2k に振り直し、gap が全て k かつ符号が全て
    同じであれば、その k 個は交互結び目 (k=3 なら三葉結び) を成す。

    余分な交差があっても中の結び目を見つけられるのがこの形の狙い
    (厳密な `count == k` だと尻尾のループ 1 つで判定が壊れる)。

    env ごとの Python ループは持たない。全て `(B, T, ...)` の一括演算。
    """
    order = batch.order
    device = order.device
    num_envs = order.shape[0]
    num_slots = min(order.shape[1], MAX_CROSSINGS_CONSIDERED)
    if num_slots < k:
        return torch.zeros(num_envs, dtype=torch.bool, device=device)

    sub = _subsets(num_slots, k, device)  # (T, k)

    valid = batch.valid_mask[:, :num_slots][:, sub]  # (B, T, k)
    all_valid = valid.all(dim=-1)

    signs = batch.data[:, :num_slots, 3][:, sub]  # (B, T, k)
    same_sign = (signs == signs[..., :1]).all(dim=-1)

    # 選んだ k 交差の点番号 2k 個。部分列の中だけで順位を取り直す。
    picked = order[:, :num_slots][:, sub].reshape(num_envs, -1, 2 * k)  # (B, T, 2k)
    ordered, _ = picked.sort(dim=-1)
    # 点番号が重複していたら (別の交差が同じ点を共有) その組は無効。
    distinct = (ordered[..., 1:] > ordered[..., :-1]).all(dim=-1)

    # rank[b, t, i] = picked[b, t, i] が部分列内で何番目に小さいか (0 始まり)。
    rank = (picked.unsqueeze(-1) > ordered.unsqueeze(-2)).sum(dim=-1)
    rank = rank.reshape(num_envs, -1, k, 2)
    interleaved = ((rank[..., 0] - rank[..., 1]).abs() == k).all(dim=-1)

    return (all_valid & same_sign & distinct & interleaved).any(dim=-1)


def is_overhand(batch: IntersectionBatch) -> torch.Tensor:
    """固結びか `(num_envs,)` bool。"""
    return has_knot_subsequence(batch, k=3)


def is_interleaved_two(batch: IntersectionBatch) -> torch.Tensor:
    """絡み合った 2 交差 (ラダー 2 段目) か `(num_envs,)` bool。

    ループに端をくぐらせかけた状態。gap 2 の 2 交差が同符号で存在する。
    「独立したツイスト 2 つ」(gap が `{1,1}`) とはここで区別される —
    後者は移植元の BFS グラフでも行き止まりで、固結びへは繋がらない。
    """
    return has_knot_subsequence(batch, k=2)


def topology_class(batch: IntersectionBatch) -> torch.Tensor:
    """報酬ラダーの段 `(num_envs,)` int64。

    上の段から順に判定する (固結びは絡み 2 交差の条件も満たしうるため、
    必ず高い段を優先すること)。どの段にも乗らない位相は 0 に落とす。
    """
    count = batch.count
    overhand = is_overhand(batch)
    inter2 = is_interleaved_two(batch)

    cls = torch.zeros_like(count)
    cls = torch.where(count == 1, torch.full_like(count, CLASS_SINGLE), cls)
    cls = torch.where(inter2, torch.full_like(count, CLASS_INTERLEAVED2), cls)
    cls = torch.where(overhand, torch.full_like(count, CLASS_OVERHAND), cls)
    return cls


def class_one_hot(batch: IntersectionBatch) -> torch.Tensor:
    """位相クラスの one-hot `(num_envs, NUM_CLASSES)` float32。観測用。"""
    cls = topology_class(batch)
    out = torch.zeros(cls.shape[0], NUM_CLASSES, device=cls.device, dtype=torch.float32)
    return out.scatter_(1, cls.unsqueeze(-1), 1.0)
