"""推論時探索 (ビーム探索 + 到達木からの再開) の純 torch 部分。`scripts/rsl_rl/eval_search.py` が使う。

`paper/Report_Paper.md` §5.2 / §7 P2-4。論文 (TWISTED / TWISTED-RL / Yan2020) の成功率は、シミュレータを
世界モデルとして使い、候補の手を実際に試して探索した結果 (最大 30 分) で、chain 評価 (方策を 1 回展開
しただけ) とは土俵が違う。ここでは学習済みの方策を候補生成器、段ごとの critic を同じ段の中の順位付けに
使って、まっすぐなロープから固結びまでを探索する。

このモジュールは **isaaclab を import しない** (`tests/test_beam_search.py` から Isaac Sim 無しで検証する)。
環境の状態の複製は `mdp/state_clone.py`、シミュレータを回すループは `eval_search.py`。

## 1 ラウンド

P 個の問題を G 個の「枠」で並列に解く (枠 1 つ = 環境 S 個)。1 ラウンド = `env.step()` 1 回 (1 手)。

1. 各枠のフロンティア (最大 M ノード) に S 個の環境を割り振り、ノードの状態を複製する
   (`assign_slots`。M が S を割り切らなければ先頭のノードに 1 つずつ多く)。
2. 方策を std 付きでサンプルして 1 手打つ (各ノードの最初の 1 環境は平均の行動にもできる)。
3. 結果を分類する (`classify`): 固結び / 段が上がった / 同じ / 下がった / 発散 / 床に落ちていない。
4. 固結びが 1 つでもあれば解けた。無ければ「上がった」と「同じ」から次のフロンティアを M 個選ぶ
   (`select_frontier`)。**段の高いものを先に取り、同じ段の中だけを V(s) で比べる。**
   段が違う critic は別のネットワークで尺度が揃っていないので、値どうしを比べない。

## 線の打ち切りと再開

同じ段のまま `max_same` 手 (既定 6 = 学習の 1 遷移の上限) 続いた線は捨てる。学習中の方策はこの手数を
超えた状況を経験していない。フロンティアが空になったら (全部が発散・段が下がった・打ち切り)、到達木から
M 個を選んで再開する (`sample_restart`)。TWISTED の「交差数に比例した優先度 + 確率 p でランダム」に倣い、
重みは段 / (1 + そのノードから再開した回数)。段を交差数の代わりに使うのは、段が固結びへの正規の経路上の
交差数に当たるため (cover 後や偽の C2 の生の交差数は近さと合わない。`versions/v4/README.md`)。
再開の回数で割るのは、同じノードからの再開ばかりにならないようにするため (TWISTED には無い)。

## 予算

予算は問題ごとの「シミュレートした手の総数」(枠の環境数 x ラウンド数)。固結びが出たラウンドの手も
全部数える (並列に試した候補は全部シミュレートしているため)。実時間の打ち切りは `eval_search.py` が見る。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

OUT_KNOT = 0
OUT_UP = 1
OUT_SAME = 2
OUT_DOWN = 3
OUT_DIVERGED = 4
OUT_UNSETTLED = 5
OUTCOME_NAMES = ("knot", "up", "same", "down", "diverged", "unsettled")
"""`classify` の結果の名前 (添字 = 値)。"""


def classify(
    parent_stage: torch.Tensor,
    child_stage: torch.Tensor,
    knot: torch.Tensor,
    diverged: torch.Tensor,
    settled: torch.Tensor,
) -> torch.Tensor:
    """1 手の結果 `(n,)` int64 (`OUT_*`)。

    判定の順は 発散 > 固結び > 床に落ちていない > 段の比較。chain 評価の終了判定 (`mdp.stage_changed`、
    固結びは床に落ちていて発散していないときだけ) と揃えてある。`knot` は環境の終了項 (固結びで reset
    された) を渡す。固結びの環境は同じ手の中で reset されて観測が段 0 になるので、段からは判定できない。

    床に落ちていない (吊ったまま) 手は段が確定しないので、観測の段が親と同じなら「同じ」として残し、
    違えば捨てる (`OUT_UNSETTLED`)。同じ段の critic で比べられないため。
    """
    out = torch.full_like(child_stage, OUT_SAME)
    out = torch.where(child_stage > parent_stage, torch.full_like(out, OUT_UP), out)
    out = torch.where(child_stage < parent_stage, torch.full_like(out, OUT_DOWN), out)
    out = torch.where(~settled & (child_stage != parent_stage), torch.full_like(out, OUT_UNSETTLED), out)
    out = torch.where(knot, torch.full_like(out, OUT_KNOT), out)
    return torch.where(diverged, torch.full_like(out, OUT_DIVERGED), out)


def select_frontier(stage: torch.Tensor, value: torch.Tensor, keep: int) -> torch.Tensor:
    """候補 `(n,)` から次のフロンティアの添字を最大 `keep` 個返す。段の降順、同じ段の中は値の降順。

    **段が違う候補の値は比べない。** 値で並べてから段で安定ソートするので、値が効くのは同じ段の中だけ。
    NaN の値は最下位に回す。
    """
    if stage.numel() == 0 or keep <= 0:
        return torch.zeros(0, dtype=torch.long, device=stage.device)
    v = torch.nan_to_num(value.float(), nan=-math.inf)
    by_value = torch.argsort(v, descending=True, stable=True)
    by_stage = torch.argsort(stage[by_value], descending=True, stable=True)
    return by_value[by_stage][:keep]


def assign_slots(num_nodes: int, num_slots: int) -> torch.Tensor:
    """`num_slots` 個の環境を `num_nodes` 個のノードに振り分ける `(num_slots,)` (ノードの添字、昇順に連続)。

    各ノードに `num_slots // num_nodes` 個、余りは先頭から 1 つずつ。フロンティアは良い順に並んでいるので、
    余りは良いノードに回る。
    """
    if num_nodes <= 0 or num_slots <= 0:
        return torch.zeros(0, dtype=torch.long)
    counts = torch.full((num_nodes,), num_slots // num_nodes, dtype=torch.long)
    counts[: num_slots % num_nodes] += 1
    return torch.repeat_interleave(torch.arange(num_nodes), counts)


def restart_weights(stage: torch.Tensor, restarts: torch.Tensor, random_prob: float) -> torch.Tensor:
    """到達木のノードから再開する確率 `(n,)` (和 1)。`(1-p)` x 段 / (1 + 再開回数) の正規化 + `p` x 一様。

    全ノードが段 0 (優先度の和が 0) なら一様。
    """
    n = stage.numel()
    uniform = torch.full((n,), 1.0 / max(n, 1), device=stage.device)
    prio = stage.float().clamp_min(0.0) / (1.0 + restarts.float())
    total = prio.sum()
    if n == 0 or float(total) <= 0.0:
        return uniform
    return (1.0 - random_prob) * prio / total + random_prob * uniform


def sample_restart(
    stage: torch.Tensor,
    restarts: torch.Tensor,
    k: int,
    random_prob: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """`restart_weights` で `k` 個 (ノード数が少なければ全部) を重複なしに選ぶ `(k',)`。"""
    n = stage.numel()
    if n == 0 or k <= 0:
        return torch.zeros(0, dtype=torch.long, device=stage.device)
    w = restart_weights(stage, restarts, random_prob)
    # 確率 0 のノードは multinomial の重複なし抽出で選べないので、選べる数に切り詰める。
    k = min(k, int((w > 0).sum()))
    return torch.multinomial(w, k, replacement=False, generator=generator)


def success_curve(moves_at_solve: list[float], budgets: list[int]) -> list[float]:
    """問題ごとの「解けた時点の手数」(解けなければ inf) から、予算ごとの成功率を返す。"""
    n = max(len(moves_at_solve), 1)
    return [sum(1 for m in moves_at_solve if m <= b) / n for b in budgets]


def best_of_n_curve(lengths: list[int], success: list[bool], budgets: list[int]) -> list[float]:
    """独立な確率的展開を手数の予算まで繰り返したときの成功率 (best-of-N、探索なしの基準)。

    すべての問題が同じまっすぐなロープから始まるので、エピソードは独立同分布とみなせる。そこで集めた
    エピソードの (手数, 成否) の経験分布から、再生方程式で厳密に計算する:

        f(b) = P(成功かつ L <= b) + sum_l P(失敗かつ L = l) f(b - l)

    エピソードは途中で打ち切れない (予算が尽きたら、そのエピソードの成功は数えない) とし、成功した
    エピソードの手数は固結びに着いた手まで。標本を並べ直すモンテカルロと同じ値を、乱数の揺れなしに出す。
    """
    if not lengths:
        return [0.0 for _ in budgets]
    n = len(lengths)
    max_b = max(budgets) if budgets else 0
    max_len = max(lengths)
    p_succ = [0.0] * (max_len + 1)
    p_fail = [0.0] * (max_len + 1)
    for length, ok in zip(lengths, success):
        if ok:
            p_succ[length] += 1.0 / n
        else:
            p_fail[length] += 1.0 / n
    cum_succ = [0.0] * (max_len + 1)
    for length in range(1, max_len + 1):
        cum_succ[length] = cum_succ[length - 1] + p_succ[length]
    f = [0.0] * (max_b + 1)
    for b in range(1, max_b + 1):
        val = cum_succ[min(b, max_len)]
        for length in range(1, min(b, max_len) + 1):
            if p_fail[length] > 0.0:
                val += p_fail[length] * f[b - length]
        f[b] = val
    return [f[b] if b > 0 else 0.0 for b in budgets]


def expected_moves_best_of_n(lengths: list[int], success: list[bool]) -> float | None:
    """best-of-N で最初の成功までにかかる手数の期待値 `E[L] / p` (成功が 1 つも無ければ None)。

    1 本のエピソードの手数 L の平均を成功確率 p で割る (再生報酬定理)。成功したエピソードの L は固結びまで。
    """
    k = sum(1 for s in success if s)
    if k == 0:
        return None
    return (sum(lengths) / len(lengths)) / (k / len(lengths))


# ================================================================ ノードの保存


class NodeStore:
    """到達木のノード。状態 (任意のテンソルの辞書) とメタ情報を GPU 上の配列で持つ。

    状態の中身は `eval_search.py` が `mdp.state_clone.EnvStateCloner.capture` の結果 (+ 観測) を渡す。
    ここは中身を解釈しない。解けた / 打ち切った問題のノードは `release` で空きに戻す。足りなくなったら
    容量を 2 倍にする。
    """

    def __init__(self, capacity: int, device: torch.device | str):
        self.device = torch.device(device)
        self.capacity = int(capacity)
        self.fields: dict[str, torch.Tensor] = {}
        self.parent = torch.full((self.capacity,), -1, dtype=torch.long, device=self.device)
        self.depth = torch.zeros(self.capacity, dtype=torch.long, device=self.device)
        self.stage = torch.zeros(self.capacity, dtype=torch.long, device=self.device)
        self.value = torch.zeros(self.capacity, device=self.device)
        self.same_run = torch.zeros(self.capacity, dtype=torch.long, device=self.device)
        self.restarts = torch.zeros(self.capacity, dtype=torch.long, device=self.device)
        self._free = list(range(self.capacity - 1, -1, -1))

    def __len__(self) -> int:
        return self.capacity - len(self._free)

    def _grow(self, need: int) -> None:
        new_cap = self.capacity
        while new_cap - len(self) < need:
            new_cap *= 2
        if new_cap == self.capacity:
            return

        def pad(t: torch.Tensor, fill: float) -> torch.Tensor:
            extra = torch.full((new_cap - self.capacity, *t.shape[1:]), fill, dtype=t.dtype, device=t.device)
            return torch.cat([t, extra])

        self.fields = {k: pad(v, 0) for k, v in self.fields.items()}
        self.parent = pad(self.parent, -1)
        self.depth = pad(self.depth, 0)
        self.stage = pad(self.stage, 0)
        self.value = pad(self.value, 0.0)
        self.same_run = pad(self.same_run, 0)
        self.restarts = pad(self.restarts, 0)
        self._free = list(range(new_cap - 1, self.capacity - 1, -1)) + self._free
        self.capacity = new_cap

    def add(
        self,
        state: dict[str, torch.Tensor],
        parent: torch.Tensor,
        depth: torch.Tensor,
        stage: torch.Tensor,
        value: torch.Tensor,
        same_run: torch.Tensor,
    ) -> torch.Tensor:
        """`n` 個のノードを足して添字 `(n,)` を返す。`state` の各値は先頭の次元が `n`。"""
        n = int(stage.numel())
        if n == 0:
            return torch.zeros(0, dtype=torch.long, device=self.device)
        self._grow(n)
        ids = torch.tensor([self._free.pop() for _ in range(n)], dtype=torch.long, device=self.device)
        for k, v in state.items():
            if k not in self.fields:
                self.fields[k] = torch.zeros((self.capacity, *v.shape[1:]), dtype=v.dtype, device=self.device)
            self.fields[k][ids] = v.to(self.device)
        self.parent[ids] = parent.to(self.device)
        self.depth[ids] = depth.to(self.device)
        self.stage[ids] = stage.to(self.device)
        self.value[ids] = value.to(self.device).float()
        self.same_run[ids] = same_run.to(self.device)
        self.restarts[ids] = 0
        return ids

    def gather(self, ids: torch.Tensor) -> dict[str, torch.Tensor]:
        """ノード `ids` の状態の辞書 (先頭の次元が `len(ids)`)。"""
        return {k: v[ids] for k, v in self.fields.items()}

    def release(self, ids: list[int]) -> None:
        self._free.extend(int(i) for i in ids)


# ================================================================ 探索の本体


@dataclass
class SearchConfig:
    """ビーム探索の設定。"""

    num_groups: int
    """同時に解く問題の数 G (枠の数)。"""
    slots_per_group: int
    """枠 1 つの環境数 S (1 ラウンドに 1 問題でシミュレートする手の数)。"""
    beam_width: int
    """フロンティアの大きさ M。"""
    max_moves: int
    """問題ごとの手数の予算。"""
    num_problems: int
    """解く問題の総数 (G 個ずつ並べて流す)。"""
    max_same: int = 6
    """同じ段のまま続けてよい手数。これに達した線は捨てる (学習の 1 遷移の上限 6 手に合わせる)。"""
    restart_random_prob: float = 0.05
    """再開のノードを一様に選ぶ確率 p (TWISTED は 0.05)。"""
    mean_first: bool = True
    """各ノードの最初の 1 環境は平均の行動 (決定論的方策) にするか。"""


@dataclass
class ProblemRecord:
    """1 問題の結果。"""

    problem_id: int
    solved: bool = False
    moves: int = 0
    """シミュレートした手の総数。"""
    moves_at_solve: int | None = None
    depth_at_solve: int | None = None
    """解の手数 (まっすぐなロープから固結びまでに実際に打った手の数 = 到達木の深さ)。"""
    rounds: int = 0
    restarts: int = 0
    max_stage: int = 0
    outcomes: dict[str, int] = field(default_factory=lambda: {k: 0 for k in OUTCOME_NAMES})
    start_time: float = 0.0
    end_time: float = 0.0
    stop_reason: str = ""
    """`solved` / `budget` / `wall_clock`。"""

    def to_dict(self) -> dict:
        return {
            "problem_id": self.problem_id,
            "solved": self.solved,
            "moves": self.moves,
            "moves_at_solve": self.moves_at_solve,
            "depth_at_solve": self.depth_at_solve,
            "rounds": self.rounds,
            "restarts": self.restarts,
            "max_stage": self.max_stage,
            "outcomes": dict(self.outcomes),
            "wall_seconds": self.end_time - self.start_time,
            "stop_reason": self.stop_reason,
        }


@dataclass
class _Group:
    problem: ProblemRecord | None = None
    frontier: list[int] = field(default_factory=list)
    tree: list[int] = field(default_factory=list)


@dataclass
class RoundPlan:
    """1 ラウンドの割り振り。添字はすべて環境 (枠 g の環境は `g*S .. g*S+S-1`)。"""

    slot_node: torch.Tensor
    """`(G*S,)` 複製するノード。使わない環境は −1。"""
    slot_mean: torch.Tensor
    """`(G*S,)` 平均の行動を打つ環境。"""


class BeamSearch:
    """P 個の問題を G 個の枠で流すビーム探索の記帳係。シミュレータは触らない。

    `plan_round()` -> (呼び出し側が状態を複製し、1 手打ち、結果を分類する) -> `commit_round()` を繰り返す。
    `done` が True になったら全問題が終わっている。

    Args:
        cfg: 設定。
        root_state: まっすぐなロープの状態 (先頭の次元 1)。新しい問題はここから始まる。
        device: ノードを置くデバイス。
        generator: 再開の抽出に使う乱数。
    """

    def __init__(
        self,
        cfg: SearchConfig,
        root_state: dict[str, torch.Tensor],
        device: torch.device | str,
        generator: torch.Generator | None = None,
    ):
        if cfg.beam_width > cfg.slots_per_group:
            raise ValueError(f"beam_width ({cfg.beam_width}) は slots_per_group ({cfg.slots_per_group}) 以下にする")
        self.cfg = cfg
        self.device = torch.device(device)
        self.root_state = {k: v[:1].to(self.device) for k, v in root_state.items()}
        self.generator = generator
        per_group = cfg.beam_width * (cfg.max_moves // max(cfg.slots_per_group, 1) + 2)
        self.store = NodeStore(max(cfg.num_groups * per_group, 64), self.device)
        self.groups = [_Group() for _ in range(cfg.num_groups)]
        self.finished: list[ProblemRecord] = []
        self._next_problem = 0
        self.now = 0.0
        for g in self.groups:
            self._start_problem(g)

    @property
    def num_slots(self) -> int:
        return self.cfg.num_groups * self.cfg.slots_per_group

    @property
    def done(self) -> bool:
        return all(g.problem is None for g in self.groups)

    def active_problems(self) -> list[ProblemRecord]:
        return [g.problem for g in self.groups if g.problem is not None]

    # ------------------------------------------------------------ 問題の出し入れ
    def _start_problem(self, g: _Group) -> None:
        if self._next_problem >= self.cfg.num_problems:
            g.problem = None
            g.frontier, g.tree = [], []
            return
        g.problem = ProblemRecord(problem_id=self._next_problem, start_time=self.now)
        self._next_problem += 1
        zero = torch.zeros(1, dtype=torch.long, device=self.device)
        root = self.store.add(self.root_state, zero - 1, zero, zero, torch.zeros(1, device=self.device), zero)
        g.frontier = [int(root[0])]
        g.tree = [int(root[0])]

    def _finish(self, g: _Group, reason: str) -> None:
        assert g.problem is not None
        g.problem.stop_reason = reason
        g.problem.end_time = self.now
        self.finished.append(g.problem)
        self.store.release(g.tree)
        self._start_problem(g)

    def stop_all(self, reason: str = "wall_clock") -> None:
        """実時間の打ち切り: 走っている問題をすべて未解決で終え、まだ始めていない問題は始めない。"""
        self.cfg.num_problems = self._next_problem
        for g in self.groups:
            if g.problem is not None:
                self._finish(g, reason)

    # ------------------------------------------------------------ 1 ラウンド
    def plan_round(self) -> RoundPlan:
        cfg = self.cfg
        s = cfg.slots_per_group
        slot_node = torch.full((self.num_slots,), -1, dtype=torch.long)
        slot_mean = torch.zeros(self.num_slots, dtype=torch.bool)
        for gi, g in enumerate(self.groups):
            p = g.problem
            if p is None:
                continue
            if not g.frontier:
                self._restart(g)
            n = min(s, cfg.max_moves - p.moves)
            if n <= 0 or not g.frontier:
                continue
            which = assign_slots(len(g.frontier), n)
            nodes = torch.tensor(g.frontier, dtype=torch.long)[which]
            slot_node[gi * s : gi * s + n] = nodes
            if cfg.mean_first:
                first = torch.ones(n, dtype=torch.bool)
                first[1:] = which[1:] != which[:-1]
                slot_mean[gi * s : gi * s + n] = first
        return RoundPlan(slot_node=slot_node, slot_mean=slot_mean)

    def _restart(self, g: _Group) -> None:
        assert g.problem is not None
        tree = torch.tensor(g.tree, dtype=torch.long, device=self.device)
        pick = sample_restart(
            self.store.stage[tree],
            self.store.restarts[tree],
            self.cfg.beam_width,
            self.cfg.restart_random_prob,
            self.generator,
        )
        ids = tree[pick]
        self.store.restarts[ids] += 1
        self.store.same_run[ids] = 0  # 再開した線は新しい線として数え直す
        g.frontier = [int(i) for i in ids]
        g.problem.restarts += 1

    def commit_round(
        self,
        plan: RoundPlan,
        outcome: torch.Tensor,
        child_stage: torch.Tensor,
        child_value: torch.Tensor,
        child_state: dict[str, torch.Tensor],
        now: float = 0.0,
    ) -> list[ProblemRecord]:
        """1 ラウンドの結果を書き込み、この呼び出しで終わった問題を返す。

        Args:
            plan: `plan_round` の戻り値。
            outcome / child_stage / child_value: `(G*S,)` 各環境の結果 (`classify`)・手の後の段・手の後の V(s)。
                V(s) は手の後の段の critic の値 (段が同じ候補どうしでだけ比べる)。
            child_state: 手の後の状態 (先頭の次元 G*S)。フロンティアに入った行だけ保存する。
            now: 経過秒 (記録用)。
        """
        self.now = now
        cfg = self.cfg
        s = cfg.slots_per_group
        outcome = outcome.to(self.device)
        child_stage = child_stage.to(self.device)
        child_value = child_value.to(self.device)
        slot_node = plan.slot_node.to(self.device)
        done_now: list[ProblemRecord] = []
        for gi, g in enumerate(self.groups):
            p = g.problem
            if p is None:
                continue
            rows = torch.arange(gi * s, (gi + 1) * s, device=self.device)
            rows = rows[slot_node[rows] >= 0]
            if rows.numel() == 0:
                if p.moves >= cfg.max_moves:
                    done_now.append(p)
                    self._finish(g, "budget")
                continue
            p.moves += int(rows.numel())
            p.rounds += 1
            out = outcome[rows]
            for k, name in enumerate(OUTCOME_NAMES):
                p.outcomes[name] += int((out == k).sum())
            parents = slot_node[rows]
            ok_stage = torch.where(out == OUT_KNOT, torch.full_like(child_stage[rows], 3), child_stage[rows])
            ok_stage = torch.where((out == OUT_UP) | (out == OUT_SAME) | (out == OUT_KNOT), ok_stage, 0)
            p.max_stage = max(p.max_stage, int(ok_stage.max()))

            knot = out == OUT_KNOT
            if bool(knot.any()):
                p.solved = True
                p.moves_at_solve = p.moves
                p.depth_at_solve = int(self.store.depth[parents[knot]].min()) + 1
                done_now.append(p)
                self._finish(g, "solved")
                continue

            keep = (out == OUT_UP) | (out == OUT_SAME)
            same_run = torch.where(out == OUT_UP, 0, self.store.same_run[parents] + 1)
            keep &= same_run < cfg.max_same
            cand = keep.nonzero(as_tuple=False).squeeze(-1)
            sel = cand[select_frontier(child_stage[rows][cand], child_value[rows][cand], cfg.beam_width)]
            r = rows[sel]
            ids = self.store.add(
                {k: v[r] for k, v in child_state.items()},
                parents[sel],
                self.store.depth[parents[sel]] + 1,
                child_stage[r],
                child_value[r],
                same_run[sel],
            )
            g.frontier = [int(i) for i in ids]
            g.tree.extend(g.frontier)
            if p.moves >= cfg.max_moves:
                done_now.append(p)
                self._finish(g, "budget")
        return done_now
