"""`common/move_chain.py` (chain モードの手の列の復元とマルコフ連鎖) の試験。Isaac Sim は要らない。"""

import pytest
from move_chain import (
    OUTCOMES,
    episode_result,
    first_up_move,
    markov_absorb,
    position_histogram,
    split_episodes,
    transition_probs,
    visit_table,
)

U, D, S, DIV = (OUTCOMES.index(o) for o in ("up", "down", "same_shape", "diverged"))


def _rows():
    """3 環境・最大 3 手。env0: 0→1→2→固結び、env1: 1 手目で発散、env2: 段 0 で 2 手留まって 3 手目に上がる。

    行は手の番号ごとのブロックで、ブロックの中は生き残った環境の番号順。
    """
    rows = [
        # 1 手目: env0, env1, env2
        (0, 1, U), (0, 0, DIV), (0, 0, S),
        # 2 手目: env0, env2
        (1, 2, U), (0, 0, S),
        # 3 手目: env0, env2
        (2, 3, U), (0, 1, U),
    ]  # fmt: skip
    before, after, outcome = zip(*rows, strict=True)
    return before, after, outcome


def test_split_episodes_follows_surviving_envs():
    seqs = split_episodes(*_rows(), num_envs=3, max_moves=3)
    assert [len(s) for s in seqs] == [3, 1, 3]
    assert [episode_result(s) for s in seqs] == ["knot", "diverged", "time_out"]
    assert [first_up_move(s) for s in seqs] == [1, None, 3]


def test_split_episodes_rejects_leftover_or_missing_rows():
    before, after, outcome = _rows()
    with pytest.raises(ValueError, match="余った"):
        split_episodes(before, after, outcome, num_envs=3, max_moves=2)
    with pytest.raises(ValueError, match="足りない"):
        split_episodes(before[:-1], after[:-1], outcome[:-1], num_envs=3, max_moves=3)


def test_visits_split_by_entry_kind():
    seq = [(0, 1, "up"), (1, 2, "up"), (2, 1, "down"), (1, 1, "same_shape"), (1, 2, "up"), (2, 2, "same_stage")]
    t1 = visit_table([seq], stage=1)
    assert t1["first"]["visits"] == 1 and t1["first"]["up_per_visit"] == 1.0
    assert t1["from_above"]["moves"] == 2 and t1["from_above"]["up"] == 0.5
    t2 = visit_table([seq], stage=2)
    assert t2["first"]["down"] == 1.0 and t2["from_below"]["stay"] == 1.0
    assert position_histogram([seq], stage=1, outcome="up") == {1: 1, 2: 1}


def test_markov_absorb_counts_moves_to_knot():
    sure = transition_probs({s: {"up": 1.0} for s in range(3)})
    assert markov_absorb(sure, budget=2)["knot"] == 0.0
    assert markov_absorb(sure, budget=3)["knot"] == 1.0
    p = transition_probs({0: {"up": 1.0}, 1: {"up": 1.0}, 2: {"up": 0.5, "diverged": 0.5}})
    out = markov_absorb(p, budget=3)
    assert out["knot"] == pytest.approx(0.5) and out["diverged"] == pytest.approx(0.5)
    assert out["unfinished"] == pytest.approx(0.0)
