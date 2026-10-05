"""推論時探索の純 torch 部分 `common/beam_search.py` の単体テスト。

Isaac Sim を起動せずに走る::

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/test_beam_search.py
"""

from __future__ import annotations

import math
import random

import beam_search as bs
import pytest
import torch

# ---------------------------------------------------------------- classify


def test_classify_order_of_precedence():
    parent = torch.tensor([1, 1, 1, 2, 2, 2, 2, 1])
    child = torch.tensor([2, 1, 0, 0, 2, 1, 2, 2])
    knot = torch.tensor([False, False, False, True, False, False, False, True])
    div = torch.tensor([False, False, False, False, False, False, True, True])
    settled = torch.tensor([True, True, True, True, False, False, True, True])
    out = bs.classify(parent, child, knot, div, settled)
    assert out.tolist() == [
        bs.OUT_UP,
        bs.OUT_SAME,
        bs.OUT_DOWN,
        bs.OUT_KNOT,  # reset 後の観測は段 0 でも、終了項で固結び
        bs.OUT_SAME,  # 吊ったままでも段が同じなら残す
        bs.OUT_UNSETTLED,  # 吊ったままで段が違えば捨てる
        bs.OUT_DIVERGED,
        bs.OUT_DIVERGED,  # 発散が最優先
    ]


# ---------------------------------------------------------------- select_frontier


def test_select_frontier_never_compares_values_across_stages():
    # 段 1 の候補の値がいくら大きくても、段 2 の候補が先に入る。
    stage = torch.tensor([1, 2, 1, 2, 1])
    value = torch.tensor([100.0, -5.0, 50.0, -1.0, 0.0])
    idx = bs.select_frontier(stage, value, keep=3)
    assert idx.tolist() == [3, 1, 0]


def test_select_frontier_nan_last_and_keep_bounds():
    stage = torch.tensor([0, 0, 0])
    value = torch.tensor([float("nan"), 1.0, 2.0])
    assert bs.select_frontier(stage, value, keep=10).tolist() == [2, 1, 0]
    assert bs.select_frontier(stage[:0], value[:0], keep=3).numel() == 0
    assert bs.select_frontier(stage, value, keep=0).numel() == 0


# ---------------------------------------------------------------- assign_slots


@pytest.mark.parametrize("nodes,slots", [(1, 8), (3, 8), (4, 32), (5, 3), (8, 8)])
def test_assign_slots_balanced_and_contiguous(nodes, slots):
    a = bs.assign_slots(nodes, slots)
    assert a.numel() == slots
    counts = torch.bincount(a, minlength=nodes)
    used = counts[: min(nodes, slots)]
    assert int(used.max() - used.min()) <= 1
    assert bool((counts[:-1] >= counts[1:]).all())  # 余りは先頭 (良いノード) へ
    assert bool((a[1:] >= a[:-1]).all())


# ---------------------------------------------------------------- restart


def test_restart_weights_prefer_higher_stage_and_decay_with_restarts():
    stage = torch.tensor([0, 1, 2, 2])
    restarts = torch.tensor([0, 0, 0, 3])
    w = bs.restart_weights(stage, restarts, random_prob=0.0)
    assert math.isclose(float(w.sum()), 1.0, rel_tol=1e-6)
    assert float(w[0]) == 0.0
    assert float(w[2]) > float(w[1]) > float(w[3])
    w_rand = bs.restart_weights(stage, restarts, random_prob=0.05)
    assert float(w_rand[0]) == pytest.approx(0.05 / 4)


def test_restart_all_stage_zero_is_uniform_and_sampling_without_replacement():
    stage = torch.zeros(5, dtype=torch.long)
    w = bs.restart_weights(stage, torch.zeros(5, dtype=torch.long), random_prob=0.0)
    assert torch.allclose(w, torch.full((5,), 0.2))
    g = torch.Generator().manual_seed(0)
    pick = bs.sample_restart(stage, torch.zeros(5, dtype=torch.long), k=4, random_prob=0.0, generator=g)
    assert len(set(pick.tolist())) == 4
    # 確率が正のノードが 1 つだけなら 1 つしか返らない
    pick = bs.sample_restart(torch.tensor([0, 2, 0]), torch.zeros(3, dtype=torch.long), 3, 0.0, g)
    assert pick.tolist() == [1]


# ---------------------------------------------------------------- 曲線


def test_success_curve():
    assert bs.success_curve([10, 40, math.inf, 5], [5, 10, 100]) == [0.25, 0.5, 0.75]


def test_best_of_n_curve_matches_monte_carlo():
    rng = random.Random(0)
    lengths, success = [], []
    for _ in range(400):
        r = rng.random()
        if r < 0.3:
            lengths.append(rng.randint(4, 12))
            success.append(True)
        elif r < 0.45:
            lengths.append(rng.randint(1, 8))  # 発散で早く終わる
            success.append(False)
        else:
            lengths.append(18)
            success.append(False)
    budgets = [1, 10, 18, 36, 72, 144]
    exact = bs.best_of_n_curve(lengths, success, budgets)
    trials = 20000
    hits = [0] * len(budgets)
    for _ in range(trials):
        used, solved_at = 0, math.inf
        while used < budgets[-1]:
            k = rng.randrange(len(lengths))
            if success[k]:
                solved_at = used + lengths[k]
                break
            used += lengths[k]
        for i, b in enumerate(budgets):
            hits[i] += solved_at <= b
    mc = [h / trials for h in hits]
    for e, m in zip(exact, mc):
        assert abs(e - m) < 0.015
    assert exact[0] == 0.0
    assert all(a <= b for a, b in zip(exact, exact[1:]))


def test_best_of_n_single_episode_budget_and_expected_moves():
    lengths = [5, 18, 18, 10]
    success = [True, False, False, True]
    # 予算 18: 1 本目で成功する確率 0.5 + 1 本目が 18 手の失敗なら 2 本目は打てない
    assert bs.best_of_n_curve(lengths, success, [18]) == [0.5]
    assert bs.expected_moves_best_of_n(lengths, success) == pytest.approx(12.75 / 0.5)
    assert bs.expected_moves_best_of_n([18], [False]) is None


# ---------------------------------------------------------------- NodeStore


def test_node_store_grows_and_releases():
    store = bs.NodeStore(capacity=2, device="cpu")
    z = torch.zeros(3, dtype=torch.long)
    ids = store.add({"x": torch.arange(3.0).unsqueeze(-1)}, z - 1, z, z, torch.zeros(3), z)
    assert store.capacity >= 3 and len(store) == 3
    assert store.gather(ids)["x"].squeeze(-1).tolist() == [0.0, 1.0, 2.0]
    store.release([int(ids[0])])
    assert len(store) == 2
    again = store.add({"x": torch.tensor([[9.0]])}, z[:1] - 1, z[:1], z[:1], torch.zeros(1), z[:1])
    assert int(again[0]) == int(ids[0])
    assert store.gather(again)["x"].item() == 9.0


# ---------------------------------------------------------------- BeamSearch


def _root():
    return {"x": torch.zeros(1, 2)}


def _cfg(**kw):
    base = dict(num_groups=2, slots_per_group=4, beam_width=2, max_moves=40, num_problems=3, max_same=6)
    base.update(kw)
    return bs.SearchConfig(**base)


def _results(search, plan, fn):
    """`fn(slot, parent_stage) -> (outcome, child_stage, value)` で全環境の結果を作る。"""
    n = search.num_slots
    out = torch.full((n,), bs.OUT_SAME)
    stage = torch.zeros(n, dtype=torch.long)
    value = torch.zeros(n)
    for i in range(n):
        node = int(plan.slot_node[i])
        if node < 0:
            continue
        o, st, v = fn(i, int(search.store.stage[node]))
        out[i], stage[i], value[i] = o, st, v
    state = {"x": torch.arange(n, dtype=torch.float32).unsqueeze(-1).expand(n, 2).clone()}
    return out, stage, value, state


def test_plan_round_assigns_root_to_all_slots_and_mean_first():
    search = bs.BeamSearch(_cfg(), _root(), "cpu")
    plan = search.plan_round()
    assert (plan.slot_node >= 0).all()
    # 各枠はフロンティア (根 1 つ) だけなので、平均の行動は枠の先頭の 1 環境
    assert plan.slot_mean.tolist() == [True, False, False, False] * 2


def test_up_candidates_take_priority_and_tree_depth():
    search = bs.BeamSearch(_cfg(num_problems=2), _root(), "cpu")
    plan = search.plan_round()

    def fn(i, ps):
        if i in (1, 5):
            return bs.OUT_UP, ps + 1, -10.0  # 値は低いが段が上がった
        return bs.OUT_SAME, ps, float(i)

    search.commit_round(plan, *_results(search, plan, fn))
    for g in search.groups:
        stages = search.store.stage[torch.tensor(g.frontier)].tolist()
        assert stages[0] == 1 and len(stages) == 2
        assert search.store.depth[torch.tensor(g.frontier)].tolist() == [1, 1]


def test_knot_solves_and_next_problem_starts():
    search = bs.BeamSearch(_cfg(num_problems=3), _root(), "cpu")
    plan = search.plan_round()

    def fn(i, ps):
        if i == 2:
            return bs.OUT_KNOT, 0, 0.0
        return bs.OUT_SAME, ps, 0.0

    done = search.commit_round(plan, *_results(search, plan, fn), now=1.0)
    assert len(done) == 1 and done[0].solved and done[0].moves_at_solve == 4 and done[0].depth_at_solve == 1
    assert search.groups[0].problem.problem_id == 2  # 3 問目が枠 0 に入る
    assert len(search.groups[0].frontier) == 1  # 根から


def test_dead_frontier_restarts_from_tree_and_budget_stops():
    search = bs.BeamSearch(_cfg(num_groups=1, num_problems=1, max_moves=12), _root(), "cpu")
    # ラウンド 1: 段 1 へ上がる
    plan = search.plan_round()
    search.commit_round(plan, *_results(search, plan, lambda i, ps: (bs.OUT_UP, 1, float(i))))
    # ラウンド 2: 全部発散 -> フロンティアが空
    plan = search.plan_round()
    search.commit_round(plan, *_results(search, plan, lambda i, ps: (bs.OUT_DIVERGED, 0, 0.0)))
    assert search.groups[0].frontier == []
    # ラウンド 3: 到達木から再開 (段 1 のノードが選ばれる。根は段 0 で p=0.05 でしか選ばれない)
    gen = torch.Generator().manual_seed(1)
    search.generator = gen
    plan = search.plan_round()
    assert search.groups[0].problem.restarts == 1
    done = search.commit_round(plan, *_results(search, plan, lambda i, ps: (bs.OUT_SAME, ps, 0.0)))
    assert len(done) == 1 and not done[0].solved and done[0].stop_reason == "budget"
    assert done[0].moves == 12
    assert search.done


def test_max_same_drops_stagnant_lines():
    search = bs.BeamSearch(_cfg(num_groups=1, num_problems=1, max_moves=100, max_same=2), _root(), "cpu")
    for _ in range(2):
        plan = search.plan_round()
        search.commit_round(plan, *_results(search, plan, lambda i, ps: (bs.OUT_SAME, ps, 0.0)))
    # 2 手続けて同じ段 -> 線を捨てたのでフロンティアは空、次のラウンドで再開する
    assert search.groups[0].frontier == []
    search.plan_round()
    assert search.groups[0].problem.restarts == 1


def test_partial_budget_uses_only_remaining_slots():
    search = bs.BeamSearch(_cfg(num_groups=1, num_problems=1, max_moves=6), _root(), "cpu")
    plan = search.plan_round()
    search.commit_round(plan, *_results(search, plan, lambda i, ps: (bs.OUT_SAME, ps, 0.0)))
    plan = search.plan_round()
    assert int((plan.slot_node >= 0).sum()) == 2
    done = search.commit_round(plan, *_results(search, plan, lambda i, ps: (bs.OUT_SAME, ps, 0.0)))
    assert done[0].moves == 6


def test_stop_all_marks_wall_clock():
    search = bs.BeamSearch(_cfg(num_problems=5), _root(), "cpu")
    search.stop_all()
    assert search.done and len(search.finished) == 2
    assert all(p.stop_reason == "wall_clock" and not p.solved for p in search.finished)


# ---------------------------------------------------------------- shape_rate_summary


def test_shape_rate_summary_deterministic_shapes():
    # 形で決まる: 半分の形は必ず成功、半分は必ず失敗 -> 成功の分散はすべて形の間
    out = bs.shape_rate_summary([16] * 10 + [0] * 10, [16] * 20)
    assert out["mean_rate"] == pytest.approx(0.5)
    assert out["frac_zero"] == pytest.approx(0.5)
    assert out["frac_one"] == pytest.approx(0.5)
    assert out["frac_mid"] == 0.0
    assert out["hist"] == [0] * 10
    assert out["shape_share"] == pytest.approx(1.0)
    assert out["homogeneous_frac_zero"] == pytest.approx(0.5**16)


def test_shape_rate_summary_homogeneous_rate_has_small_share():
    # どの形も同じ成功率 0.3 (試行ごとのばらつきだけ) -> 形の間の分散は 0 に近い
    gen = torch.Generator().manual_seed(0)
    n, m = 64, 400
    succ = (torch.rand(m, n, generator=gen) < 0.3).sum(dim=1).tolist()
    out = bs.shape_rate_summary(succ, [n] * m)
    assert out["mean_rate"] == pytest.approx(0.3, abs=0.02)
    assert out["shape_share"] < 0.02
    assert out["frac_mid"] > 0.95
    assert out["var_observed"] == pytest.approx(0.3 * 0.7 / n, rel=0.2)
    assert sum(out["hist"]) + out["hist_exact_zero"] + out["hist_exact_one"] == m


def test_shape_rate_summary_bins_and_errors():
    out = bs.shape_rate_summary([0, 1, 5, 9, 10], [10] * 5)
    assert out["hist_exact_zero"] == 1
    assert out["hist_exact_one"] == 1
    assert out["hist"] == [0, 1, 0, 0, 0, 1, 0, 0, 0, 1]
    with pytest.raises(ValueError):
        bs.shape_rate_summary([1], [1])
    with pytest.raises(ValueError):
        bs.shape_rate_summary([1, 2], [4])
    assert bs.shape_rate_summary([], []) == {"shapes": 0}
