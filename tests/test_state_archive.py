"""到達状態アーカイブ `common/state_archive.py` の単体テスト。v3 で追加。

Isaac Sim を起動せずに走る (純 torch)::

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/test_state_archive.py
"""

from __future__ import annotations

import math

import pytest
import state_archive as sa
import torch

J = 5


def _states(values: list[float]) -> tuple[torch.Tensor, torch.Tensor]:
    v = torch.tensor(values, dtype=torch.float32)
    root = torch.zeros(len(values), 7)
    root[:, 0] = v
    root[:, 3] = 1.0
    return root, v.unsqueeze(-1).expand(-1, J).clone()


def test_add_ignores_other_classes():
    arc = sa.StateArchive(classes=(1, 2), capacity=8, num_joints=J, device="cpu")
    root, jp = _states([10, 11, 12, 13])
    arc.add(torch.tensor([0, 1, 3, 2]), root, jp)
    assert arc.sizes() == {1: 1, 2: 1}


def test_ring_buffer_overwrites_oldest():
    arc = sa.StateArchive(classes=(1,), capacity=3, num_joints=J, device="cpu")
    for v in range(5):
        root, jp = _states([float(v)])
        arc.add(torch.tensor([1]), root, jp)
    assert arc.sizes() == {1: 3}
    assert sorted(arc.root_pose[0, :, 0].tolist()) == [2.0, 3.0, 4.0]


def test_add_more_than_capacity_keeps_last():
    arc = sa.StateArchive(classes=(1,), capacity=3, num_joints=J, device="cpu")
    root, jp = _states([0, 1, 2, 3, 4, 5])
    arc.add(torch.ones(6, dtype=torch.long), root, jp)
    assert sorted(arc.joint_pos[0, :, 0].tolist()) == [3.0, 4.0, 5.0]


def test_sample_empty_returns_none():
    arc = sa.StateArchive(classes=(1, 2), capacity=4, num_joints=J, device="cpu")
    assert arc.sample(5, (1.0, 2.0)) is None


def test_sample_skips_empty_class():
    arc = sa.StateArchive(classes=(1, 2), capacity=4, num_joints=J, device="cpu")
    root, jp = _states([7])
    arc.add(torch.tensor([1]), root, jp)
    cls, root_s, jp_s = arc.sample(50, (1.0, 100.0))
    assert (cls == 1).all()
    assert (root_s[:, 0] == 7).all() and (jp_s == 7).all()


def test_sample_follows_class_weights():
    arc = sa.StateArchive(classes=(1, 2), capacity=4, num_joints=J, device="cpu")
    root, jp = _states([1, 1, 2, 2])
    arc.add(torch.tensor([1, 1, 2, 2]), root, jp)
    gen = torch.Generator().manual_seed(0)
    cls, root_s, _ = arc.sample(20000, (1.0, 3.0), generator=gen)
    frac2 = (cls == 2).float().mean().item()
    assert frac2 == pytest.approx(0.75, abs=0.02)
    # 取り出した状態が class と対応している。
    assert torch.equal(root_s[:, 0], cls.float())


def test_sample_only_from_filled_slots():
    arc = sa.StateArchive(classes=(1,), capacity=100, num_joints=J, device="cpu")
    root, jp = _states([5, 6])
    arc.add(torch.tensor([1, 1]), root, jp)
    _, root_s, _ = arc.sample(1000, (1.0,))
    assert set(root_s[:, 0].tolist()) == {5.0, 6.0}


# ---------------------------------------------------------------- 姿勢の変換


def _yaw_quat(yaw: float) -> torch.Tensor:
    return torch.tensor([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


def test_centroid_roundtrip_without_rotation():
    links = torch.tensor([[[1.0, 2.0, 0.01], [3.0, 2.0, 0.01]]])  # 重心 (2, 2)
    origins = torch.tensor([[1.0, 1.0, 0.0]])
    root = torch.tensor([[1.0, 2.0, 0.01, *_yaw_quat(0.3).tolist()]])
    rel = sa.to_centroid_frame(root, links, origins)
    assert rel[0, :3].tolist() == pytest.approx([-1.0, 0.0, 0.01])
    back = sa.place_at_origin(rel, origins, torch.zeros(1))
    # 重心が env 原点 (1, 1) に来る。
    assert back[0, :3].tolist() == pytest.approx([0.0, 1.0, 0.01])
    assert back[0, 3:].tolist() == pytest.approx(root[0, 3:].tolist())


def test_place_rotates_about_centroid():
    rel = torch.tensor([[1.0, 0.0, 0.02, *_yaw_quat(0.0).tolist()]])
    out = sa.place_at_origin(rel, torch.zeros(1, 3), torch.tensor([math.pi / 2]))
    assert out[0, :3].tolist() == pytest.approx([0.0, 1.0, 0.02], abs=1e-6)
    assert out[0, 3:].tolist() == pytest.approx(_yaw_quat(math.pi / 2).tolist(), abs=1e-6)


def test_random_yaw_range_and_disable():
    y = sa.random_yaw(1000, "cpu")
    assert (y.abs() <= math.pi).all() and y.std() > 1.0
    assert (sa.random_yaw(10, "cpu", enabled=False) == 0).all()


# ---------------------------------------------------------------- ファイル (v4)


def test_save_load_roundtrip(tmp_path):
    arc = sa.StateArchive(classes=(0, 1, 2), capacity=8, num_joints=J, device="cpu")
    root, jp = _states([10, 11, 12, 13, 14])
    arc.add(torch.tensor([0, 1, 1, 2, 3]), root, jp)  # 段 3 は保存しない
    path = str(tmp_path / "arc.pt")
    arc.save(path)

    other = sa.StateArchive(classes=(1, 2), capacity=8, num_joints=J, device="cpu")
    other.add(torch.tensor([2]), *_states([99]))
    loaded = other.load(path)  # 追加で読む (今ある分は消さない)。段 0 は対象外なので無視。
    assert loaded == {1: 2, 2: 1}
    assert other.sizes() == {1: 2, 2: 2}
    # root_pose の 1 つ目の添字はスロット (classes の並び) で、class 1 -> 0、class 2 -> 1。
    assert sorted(other.root_pose[0, :2, 0].tolist()) == [11.0, 12.0]
    assert sorted(other.root_pose[1, :2, 0].tolist()) == [13.0, 99.0]
    assert torch.equal(other.joint_pos[0, :2], other.root_pose[0, :2, 0:1].expand(-1, J))


def test_save_writes_only_filled_entries(tmp_path):
    arc = sa.StateArchive(classes=(1, 2), capacity=8, num_joints=J, device="cpu")
    arc.add(torch.tensor([1]), *_states([5]))
    data = arc.state_dict()
    assert data["num_joints"] == J
    assert set(data["entries"]) == {1}
    assert data["entries"][1]["root_pose"].shape == (1, 7)


def test_load_rejects_other_rope(tmp_path):
    arc = sa.StateArchive(classes=(1,), capacity=4, num_joints=J, device="cpu")
    arc.add(torch.tensor([1]), *_states([1]))
    path = str(tmp_path / "arc.pt")
    arc.save(path)
    other = sa.StateArchive(classes=(1,), capacity=4, num_joints=J + 1, device="cpu")
    with pytest.raises(ValueError, match="関節数"):
        other.load(path)


# ---------------------------------------------------------------- 保存する手の選び方 (v12)


def _rows(before: list[int], after: list[int], rule: str, ok: list[bool] | None = None) -> list[bool]:
    b = torch.tensor(before)
    a = torch.tensor(after)
    m = torch.ones_like(b, dtype=torch.bool) if ok is None else torch.tensor(ok)
    return sa.rows_to_save(m, b, a, rule).tolist()


def test_stage_up_saves_only_moves_up_to_stage_1_or_2():
    #         0->1  1->2  0->2  0->0  2->2  2->1  1->0  2->3  1->3  0->-1 2->-1
    before = [0, 1, 0, 0, 2, 2, 1, 2, 1, 0, 2]
    after = [1, 2, 2, 0, 2, 1, 0, 3, 3, -1, -1]
    want = [True, True, True, False, False, False, False, False, False, False, False]
    assert _rows(before, after, "stage_up") == want


def test_stage_up_requires_ok():
    assert _rows([0, 1, 0], [1, 2, 2], "stage_up", ok=[False, False, True]) == [False, False, True]


def test_every_move_is_ok_only():
    # v4〜v11 の挙動: 段に関係なく ok の手をすべて保存する (段 3 と −1 は StateArchive.add が捨てる)。
    before = [0, 1, 2, 2, 0]
    after = [0, 0, 2, 3, -1]
    ok = [True, False, True, True, False]
    assert _rows(before, after, "every_move", ok=ok) == ok


def test_every_move_does_not_alias_ok():
    ok = torch.tensor([True, False])
    out = sa.rows_to_save(ok, torch.zeros(2, dtype=torch.long), torch.zeros(2, dtype=torch.long), "every_move")
    out[1] = True
    assert ok.tolist() == [True, False]


def test_unknown_rule_raises():
    with pytest.raises(ValueError, match="save rule"):
        _rows([0], [1], "stage_change")
