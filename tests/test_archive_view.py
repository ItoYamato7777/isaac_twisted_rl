"""アーカイブの形の表示 `scripts/rsl_rl/view_archive.py` の、形を選ぶ部分 (`archive_view.py`) の単体テスト。

Isaac Sim を起動せずに走る (torch だけ)。GUI の描画 (床板の色) は GUI で確かめる。
"""

from __future__ import annotations

from isaac_twisted_rl.scripts.rsl_rl.archive_view import pick_shapes


def test_grouped_by_stage_in_given_order():
    """段ごとに per_stage 件ずつ、指定した段の順に並ぶ (同じ段の形が近い環境に集まる)。"""
    picks = pick_shapes({0: 100, 1: 100, 2: 100}, [2, 0], per_stage=3, seed=0)
    assert [s for s, _ in picks] == [2, 2, 2, 0, 0, 0]
    for s in (0, 2):
        idx = [i for ps, i in picks if ps == s]
        assert len(set(idx)) == 3 and all(0 <= i < 100 for i in idx)


def test_small_and_missing_stages():
    """件数が足りない段は全部、無い・空の段は飛ばす。"""
    picks = pick_shapes({0: 2, 1: 0}, [0, 1, 2], per_stage=5, seed=0)
    assert picks == [(0, 0), (0, 1)]


def test_seed_reproducible():
    """同じシードなら同じ形、違うシードなら (ほぼ確実に) 違う形。"""
    a = pick_shapes({0: 16384}, [0], per_stage=8, seed=1)
    assert a == pick_shapes({0: 16384}, [0], per_stage=8, seed=1)
    assert a != pick_shapes({0: 16384}, [0], per_stage=8, seed=2)


def test_filtered_state_dict_keeps_matching_shapes():
    """keep が True の形だけを、元の番号の順に、`StateArchive.state_dict` と同じ形で残す。"""
    import torch

    from isaac_twisted_rl.scripts.rsl_rl.archive_view import filtered_state_dict

    entries = {
        s: {
            "root_pose": torch.arange(n * 7, dtype=torch.float32).reshape(n, 7) + 100 * s,
            "joint_pos": torch.rand(n, 3),
        }
        for s, n in ((0, 3), (2, 4))
    }
    picks = pick_shapes({0: 3, 2: 4}, [0, 2], per_stage=4, seed=0)
    keep = [True, False, True, False, False, False, False]
    out = filtered_state_dict(entries, 3, picks, keep)
    assert out["num_joints"] == 3
    assert set(out["entries"]) == {0}  # 段 2 は全部外れたのでキーごと無い
    assert torch.equal(out["entries"][0]["root_pose"], entries[0]["root_pose"][[0, 2]])
    assert torch.equal(out["entries"][0]["joint_pos"], entries[0]["joint_pos"][[0, 2]])
