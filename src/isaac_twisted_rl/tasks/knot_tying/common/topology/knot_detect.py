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

from .intersections import PAD, IntersectionBatch

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
8 なら 56 通り。

## 打ち切る前に gap 1 の交差を後ろへ回す (2026-09-28 修正、全バージョンに効く)

以前は先頭 8 スロットをそのまま使っていた。スロットは `(seg_i, seg_j)` の
辞書順 = **頭側のセグメントから順** なので、交差が 9 個以上あると尻尾側の
交差が探索から漏れ、結び目が紐のどちら側にあるかで判定が変わっていた
(`paper/Report_Paper.md` §6.3。三葉結び + 頭側の R1 ループ 6 個で `is_overhand` が
False、尻尾側なら True)。

gap 1 の交差 (R1 のねじれ) は、どの部分列を取っても 2 点が隣り合ったまま
なので gap k ≥ 2 の条件を **決して満たさない**。これを後ろへ回してから
打ち切れば、結果は変えずに見落としだけが減る。それでも gap ≥ 2 の交差が
8 個を超える環境は `subsequence_truncated` で数えられる (v3 のログ `r2_trunc`)。
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


def _useless_for_subsequence(batch: IntersectionBatch) -> torch.Tensor:
    """部分列判定で決して使われない交差 `(num_envs, M)` bool (無効な行と gap 1 の交差)。"""
    return ~batch.valid_mask | (partner_gaps(batch) == 1)


def _candidate_slots(batch: IntersectionBatch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """部分列探索に使う先頭 `MAX_CROSSINGS_CONSIDERED` スロットの `(order, data, valid)`。

    交差がそれより多いときは、gap 1 の交差を後ろへ回してから打ち切る
    (`MAX_CROSSINGS_CONSIDERED` の docstring)。安定ソートなので、残る交差の
    相対順は変わらない。点番号は振り直さない (部分列の中で順位を取り直すため不要)。
    """
    num_slots = batch.order.shape[1]
    limit = MAX_CROSSINGS_CONSIDERED
    if num_slots <= limit:
        return batch.order, batch.data, batch.valid_mask
    sel = torch.argsort(_useless_for_subsequence(batch).to(torch.int8), dim=1, stable=True)[:, :limit]
    order = batch.order.gather(1, sel.unsqueeze(-1).expand(-1, -1, 2))
    data = batch.data.gather(1, sel.unsqueeze(-1).expand(-1, -1, 4))
    valid = batch.valid_mask.gather(1, sel) & (partner_gaps(batch).gather(1, sel) != 1)
    return order, data, valid


def subsequence_truncated(batch: IntersectionBatch) -> torch.Tensor:
    """gap ≥ 2 の交差が `MAX_CROSSINGS_CONSIDERED` を超え、部分列探索で漏れが出うる env `(num_envs,)`。

    ログ用 (v3 の `r2_trunc`)。これが増えるなら上限を上げることを検討する。
    """
    return (~_useless_for_subsequence(batch)).sum(dim=1) > MAX_CROSSINGS_CONSIDERED


def has_knot_subsequence(batch: IntersectionBatch, k: int = 3) -> torch.Tensor:
    """k 個組で「交互・同符号」を成す部分列が在るか `(num_envs,)` bool。

    モジュール docstring の「部分列判定」の本体。選んだ k 個の交差だけを
    取り出して点番号を 1..2k に振り直し、gap が全て k かつ符号が全て
    同じであれば、その k 個は交互結び目 (k=3 なら三葉結び) を成す。

    余分な交差があっても中の結び目を見つけられるのがこの形の狙い
    (厳密な `count == k` だと尻尾のループ 1 つで判定が壊れる)。

    env ごとの Python ループは持たない。全て `(B, T, ...)` の一括演算。
    """
    order, data, valid_mask = _candidate_slots(batch)
    device = order.device
    num_envs = order.shape[0]
    num_slots = order.shape[1]
    if num_slots < k:
        return torch.zeros(num_envs, dtype=torch.bool, device=device)

    sub = _subsets(num_slots, k, device)  # (T, k)

    valid = valid_mask[:, sub]  # (B, T, k)
    all_valid = valid.all(dim=-1)

    signs = data[:, :, 3][:, sub]  # (B, T, k)
    same_sign = (signs == signs[..., :1]).all(dim=-1)

    # 選んだ k 交差の点番号 2k 個。部分列の中だけで順位を取り直す。
    picked = order[:, sub].reshape(num_envs, -1, 2 * k)  # (B, T, 2k)
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


# ------------------------------------------------------- R2 (ライデマイスター II) 除去
#
# v2-R2 で追加した。v2 以前はここを通らない (既存の関数は一切変えていない)。
#
# ## なぜ必要か (v2 の実測)
#
# `is_interleaved_two` は部分列判定なので、交差が 3 つ以上あると
# **その中の 2 つ** が交互・同符号なら成立する。先端で輪を上から跨ぐだけ
# (先端側が 3 交差とも上) の `1U4+_2U5-_3U6+_4O1+_5O2-_6O3+` では、
# 交差 A (1,4) と C (3,6) が交互・同符号の組を成して class 2 と判定される。
# 実際には B と C (または A と B) が R2 で消せるので、中身は輪 1 つ (class 1)。
# 上から掴んで上へ置く pick&place ではこの「掛け渡し」が最も作りやすく、
# v2 の方策はこれで `success_c2` を稼いでいた疑いが強い。
#
# 逆に、先端を輪の中へ **上から** 置いた状態 (`1U3+_2U4-_3O1+_4O2-`) は
# v2 でも class 0 だが、交差数 2 のまま `offladder` の罰を受けていた。
# R2 除去後は交差 0 (= 何も無い) と正しく扱われる。
#
# ## 判定 (Gauss 符号上の R2)
#
# 交差 X = (p, q) と Y = (r, s) について
#
# * 片方の紐上で X と Y が隣り合う (|p - r| == 1) **かつ**
#   もう片方の紐上でも隣り合う (|q - s| == 1)。(p と s、q と r の対応も見る)
# * その隣り合った 2 点で **同じ紐が両方とも上** (もう片方は両方とも下)。
#
# を満たせば、2 本の弧は他の交差を挟まずに 2 点で交わる二角形を成し、
# 上の弧をそのまま持ち上げて外せる (R2)。上下が交互になっている組は
# 決して消さないので、**交代結び目である固結び (三葉) は絶対に削られない**。
#
# 厳密な R2 は「二角形の内側に何も無い」ことも要求するが、Gauss 符号だけでは
# 見えない。内側に入り得るのは交差を持たずに弧へ入り込めない以上
# 「ロープの自由端の切れ端」か「X に付いた R1 の輪」だけで、どちらも
# 上下が単調なので外しても位相は変わらない (開いた紐なので端は抜ける)。
#
# ## 除去の順序
#
# 1 ラウンドで env ごとに 1 組 (添字が最小の組) だけ消し、番号を振り直して
# 繰り返す。消せる組が無くなるか、交差が 1 つ以下になるまで回る。


def _r2_pair_mask(batch: IntersectionBatch) -> torch.Tensor:
    """R2 で消せる交差の組 `(num_envs, M, M)` bool (対角は False)。"""
    order = batch.order
    valid = batch.valid_mask
    p = order[..., 0]
    q = order[..., 1]
    # order[..., 0] は seg_i 側の点。data[..., 2] は seg_i が上なら +1。
    p_over = batch.data[..., 2] == 1
    q_over = ~p_over

    px, qx, pox = p.unsqueeze(2), q.unsqueeze(2), p_over.unsqueeze(2)
    py, qy, poy, qoy = p.unsqueeze(1), q.unsqueeze(1), p_over.unsqueeze(1), q_over.unsqueeze(1)

    def adj(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return (a - b).abs() == 1

    # 対応 1: p<->r, q<->s。p 側の紐が X, Y の両方で同じ上下。
    same = adj(px, py) & adj(qx, qy) & (pox == poy)
    # 対応 2: p<->s, q<->r。
    cross = adj(px, qy) & adj(qx, py) & (pox == qoy)

    num_slots = order.shape[1]
    eye = torch.eye(num_slots, dtype=torch.bool, device=order.device).unsqueeze(0)
    both_valid = valid.unsqueeze(2) & valid.unsqueeze(1)
    return (same | cross) & both_valid & ~eye


def _drop_crossings(batch: IntersectionBatch, drop: torch.Tensor) -> IntersectionBatch:
    """`drop` `(num_envs, M)` が True の交差を除き、詰め直して点番号を振り直す。"""
    keep = batch.valid_mask & ~drop
    num_envs, num_slots = keep.shape
    device = keep.device

    # 点番号の振り直し: 残る点のうち自分より小さいものの個数 + 1。
    pts = batch.order.reshape(num_envs, 2 * num_slots)  # (B, 2M)
    keep2 = keep.repeat_interleave(2, dim=1)  # order の [m, 0], [m, 1] と並びを揃える
    big = torch.full_like(pts, 1 << 30)
    live = torch.where(keep2, pts, big)
    new_pts = (live.unsqueeze(-1) > live.unsqueeze(-2)).sum(dim=-1) + 1
    new_order = new_pts.reshape(num_envs, num_slots, 2)

    # 残る交差を前へ寄せる (安定ソートなので相対順は保たれる)。
    sel = torch.argsort((~keep).to(torch.int8), dim=1, stable=True)
    count = keep.sum(dim=1).to(torch.int64)
    data = batch.data.gather(1, sel.unsqueeze(-1).expand(-1, -1, 4))
    order = new_order.gather(1, sel.unsqueeze(-1).expand(-1, -1, 2))

    valid = torch.arange(num_slots, device=device).unsqueeze(0) < count.unsqueeze(1)
    data = torch.where(valid.unsqueeze(-1), data, torch.full_like(data, PAD))
    order = torch.where(valid.unsqueeze(-1), order, torch.full_like(order, PAD))
    return IntersectionBatch(data, order, count)


def reduce_r2(batch: IntersectionBatch) -> IntersectionBatch:
    """R2 で消せる交差の組を消せなくなるまで取り除いた `IntersectionBatch`。

    入力は変更しない (新しいテンソルを返す)。env ごとの Python ループは無く、
    ラウンド数 (高々 `M // 2`) だけ回る。交差数の上限までの 1 ラウンドごとに
    `.any()` で打ち切り判定をするので、GPU 同期はラウンド数ぶん発生する。
    消せる組が無ければ 1 回の判定だけで入力そのものを返す。

    `data` の `seg_i` / `seg_j` は元のポリラインのセグメント番号のまま残るので、
    `crossing_positions` などの幾何的な復元にもそのまま使える。
    """
    num_slots = batch.max_crossings
    for _ in range(num_slots // 2):
        mask = _r2_pair_mask(batch)
        has = mask.flatten(1).any(dim=1)  # (B,)
        if not bool(has.any()):
            break
        # 添字が最小の組 (X < Y の行優先) を 1 つ選ぶ。
        flat = mask.flatten(1).to(torch.int8)
        first = flat.argmax(dim=1)
        x = first // num_slots
        y = first % num_slots
        drop = torch.zeros_like(batch.valid_mask)
        rows = torch.arange(batch.num_envs, device=drop.device)
        drop[rows, x] = has
        drop[rows, y] = has
        batch = _drop_crossings(batch, drop)
    return batch


def r2_removed_count(batch: IntersectionBatch, reduced: IntersectionBatch) -> torch.Tensor:
    """R2 で消えた交差数 `(num_envs,)`。ログ用。"""
    return batch.count - reduced.count


def class_one_hot(batch: IntersectionBatch) -> torch.Tensor:
    """位相クラスの one-hot `(num_envs, NUM_CLASSES)` float32。観測用。"""
    cls = topology_class(batch)
    out = torch.zeros(cls.shape[0], NUM_CLASSES, device=cls.device, dtype=torch.float32)
    return out.scatter_(1, cls.unsqueeze(-1), 1.0)
