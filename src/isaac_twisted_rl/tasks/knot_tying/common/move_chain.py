"""chain モードの手の列の復元と、段ごとの 1 手の結果からの記憶の無いマルコフ連鎖 (Isaac Sim 不要)。

## 手の列の復元 (`split_episodes`)

`diag_stage_moves.py --dump_obs` は、数えた手を「手の番号 (env ステップ) ごとのブロック × そのブロックで
まだ最初のエピソードが終わっていない環境の番号順」に並べて保存し、環境の番号を持たない。chain モードの
エピソードは固結び (`up` で手の後の段 3)・発散・`max_moves` 手でだけ終わるので、各ブロックで終わった環境を
除けば、次のブロックの行と環境が順に対応する。これで環境ごとの手の列が戻る。復元が正しいことは、
数えた行がちょうど使い切られることと、固結び・発散・時間切れの割合が同じチェックポイントの
`eval_chain_it*.json` と一致することで確かめる (`scripts/rsl_rl/chain_moves.py sequences`)。

## マルコフ連鎖 (`markov_absorb`)

段 0 / 1 / 2 の 1 手の結果の割合 (上がる・下がる・留まる・発散) だけで chain 評価を真似る。上がる・下がるは
1 段ずつ、段 2 から上がれば固結び。手の前の段以外の履歴を見ないので、実測とのずれは「同じ段でも、
どの手順で着いたかで結果が違う」ことの大きさの目安になる。
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

OUTCOMES = ("up", "down", "same_shape", "same_stage", "not_settled", "diverged")
"""`diag_stage_moves.py` の `OUTCOMES` と同じ順 (保存された `outcome` はこの添字)。"""

STAY = ("same_shape", "same_stage", "not_settled")
"""マルコフ連鎖で「同じ段に留まる」に数える結果。"""

Move = tuple[int, int, str]
"""1 手 = (手の前の段, 手の後の段, 結果の名前)。"""


def ends_episode(after: int, outcome: str) -> bool:
    """chain モードでエピソードを終わらせる手か (固結びか発散。時間切れは呼び出し側が手数で決める)。"""
    return outcome == "diverged" or (outcome == "up" and after == 3)


def split_episodes(
    before: Sequence[int], after: Sequence[int], outcome: Sequence[int], num_envs: int, max_moves: int
) -> list[list[Move]]:
    """保存された行 (手の番号ごとのブロック) を、環境ごとの手の列に戻す。

    行が余るか足りなければ、並びの前提が崩れているので止める。
    """
    alive = list(range(num_envs))
    seqs: list[list[Move]] = [[] for _ in range(num_envs)]
    i = 0
    for _ in range(max_moves):
        if not alive:
            break
        if i + len(alive) > len(before):
            raise ValueError(f"行が足りない: {i} 行目からの {len(alive)} 行が無い (全 {len(before)} 行)")
        nxt = []
        for env in alive:
            move = (int(before[i]), int(after[i]), OUTCOMES[int(outcome[i])])
            seqs[env].append(move)
            i += 1
            if not ends_episode(move[1], move[2]):
                nxt.append(env)
        alive = nxt
    if i != len(before):
        raise ValueError(f"行が余った: {len(before) - i} 行 (並びの前提か num_envs / max_moves が違う)")
    return seqs


def episode_result(seq: Sequence[Move]) -> str:
    """`knot` / `diverged` / `time_out` (最後の手で終わらなかった)。"""
    _, after, outcome = seq[-1]
    if outcome == "diverged":
        return "diverged"
    if outcome == "up" and after == 3:
        return "knot"
    return "time_out"


def first_up_move(seq: Sequence[Move]) -> int | None:
    """最初の手から続く段 0 の区間で、初めて段が上がった手の番号 (1 から)。上がらずに段 0 を出たか終われば None。"""
    for j, (before, _, outcome) in enumerate(seq):
        if before != 0:
            return None
        if outcome == "up":
            return j + 1
    return None


def visits(seq: Sequence[Move], stage: int) -> list[tuple[str, list[Move]]]:
    """段 `stage` の訪問 (手の前の段が `stage` のまま続く区間) を、入り方の種類と手の列で返す。

    種類: `first` (その段に初めて入った)、`from_above` (上の段から落ちて入った)、`from_below` (下の段から入り直した)。
    """
    out: list[tuple[str, list[Move]]] = []
    for j, move in enumerate(seq):
        if move[0] != stage:
            continue
        if j == 0 or seq[j - 1][0] != stage:
            if not out:
                kind = "first"
            else:
                kind = "from_above" if seq[j - 1][0] > stage else "from_below"
            out.append((kind, []))
        out[-1][1].append(move)
    return out


def visit_table(seqs: Sequence[Sequence[Move]], stage: int) -> dict[str, dict]:
    """入り方の種類ごとに、訪問の数・手の数・1 手の結果の割合・訪問あたりの上昇の割合。"""
    acc: dict[str, dict] = {}
    for seq in seqs:
        for kind, moves in visits(seq, stage):
            a = acc.setdefault(kind, {"visits": 0, "moves": 0, "outcomes": Counter(), "visits_up": 0})
            a["visits"] += 1
            a["moves"] += len(moves)
            a["outcomes"].update(m[2] for m in moves)
            a["visits_up"] += int(any(m[2] == "up" for m in moves))
    out = {}
    for kind, a in acc.items():
        m = max(a["moves"], 1)
        out[kind] = {
            "visits": a["visits"],
            "moves": a["moves"],
            "up": a["outcomes"]["up"] / m,
            "down": a["outcomes"]["down"] / m,
            "stay": sum(a["outcomes"][o] for o in STAY) / m,
            "diverged": a["outcomes"]["diverged"] / m,
            "up_per_visit": a["visits_up"] / max(a["visits"], 1),
        }
    return out


def position_histogram(seqs: Sequence[Sequence[Move]], stage: int, outcome: str) -> dict[int, int]:
    """段 `stage` の訪問の中で、結果 `outcome` の手が何手目 (1 から) に出たかの度数。"""
    hist: Counter = Counter()
    for seq in seqs:
        for _, moves in visits(seq, stage):
            for pos, m in enumerate(moves, start=1):
                if m[2] == outcome:
                    hist[pos] += 1
    return dict(sorted(hist.items()))


def transition_probs(outcome_frac_by_stage: dict[int, dict[str, float]]) -> dict[int, dict[str, float]]:
    """段ごとの 1 手の結果の割合 (`OUTCOMES` の名前) を、上がる・下がる・留まる・発散の 4 つにまとめる。"""
    return {
        s: {
            "up": o.get("up", 0.0),
            "down": o.get("down", 0.0),
            "stay": sum(o.get(k, 0.0) for k in STAY),
            "div": o.get("diverged", 0.0),
        }
        for s, o in outcome_frac_by_stage.items()
    }


def markov_absorb(probs: dict[int, dict[str, float]], budget: int) -> dict[str, float]:
    """段 0 から始めて `budget` 手打ったときの、固結び・発散・まだ終わっていない割合。"""
    x = [1.0, 0.0, 0.0]
    knot = div = 0.0
    for _ in range(budget):
        y = [0.0, 0.0, 0.0]
        for s in range(3):
            p = probs[s]
            if s == 2:
                knot += x[s] * p["up"]
            else:
                y[s + 1] += x[s] * p["up"]
            y[max(s - 1, 0)] += x[s] * p["down"]
            y[s] += x[s] * p["stay"]
            div += x[s] * p["div"]
        x = y
    return {"knot": knot, "diverged": div, "unfinished": sum(x)}
