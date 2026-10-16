"""chain 評価の動画の置き場所・名前・判断 `scripts/rsl_rl/video_layout.py` の単体テスト。

Isaac Sim を起動せずに走る (stdlib だけ)。撮影そのものはスモークテストで目で確かめる
(`docs/system/video.md`)。

## 検証の骨子

* 置き場所が run のフォルダと 1 対 1 で、評価の JSON の名前で分かれる。
* clip の名前に結末と手数が入り、`ls` の順が環境の番号の順になる。
* 結末は `chain_stats.exclusive_outcomes` で 1 つに決めたものを使い、何も立たなければ `unfinished`。
* 240 Hz で 8 物理ステップごとなら 30 fps (実時間)。
* GPU の空きが読めない・足りないときは撮らない。
"""

from __future__ import annotations

import os

import pytest
import torch
from chain_stats import exclusive_outcomes

from isaac_twisted_rl.scripts.rsl_rl import video_layout as vl

CKPT = "/repo/logs/rsl_rl/knot_tying_v14/2026-10-15_15-07-10_v14_s43/model_300.pt"


def test_clip_dir_mirrors_run_and_eval_json():
    """`video/sim/<exp>/<run>/<JSON の名前>/`。`--out` が無ければ it から名前を作る。"""
    out = "/repo/logs/rsl_rl/knot_tying_v14/2026-10-15_15-07-10_v14_s43/eval_chain_dr_it300.json"
    assert vl.clip_dir("/repo", CKPT, out) == os.path.join(
        "/repo", "video", "sim", "knot_tying_v14", "2026-10-15_15-07-10_v14_s43", "eval_chain_dr_it300"
    )
    assert vl.eval_name(None, CKPT) == "eval_chain_it300"
    assert vl.eval_name(None, "/x/run/policy.pt") == "eval_chain"
    assert vl.iteration_of(CKPT) == 300


def test_clip_name_sorts_by_env():
    assert vl.clip_name(3, "knot", 9) == "env003_knot_m09.mp4"
    names = [vl.clip_name(i, "time_out", 18) for i in (10, 2, 100)]
    assert sorted(names) == [vl.clip_name(i, "time_out", 18) for i in (2, 10, 100)]


def test_outcome_follows_exclusive_outcomes():
    """同じ手に固結びと時間切れが立てば固結び (評価の表と同じ決め方)。何も立たなければ unfinished。"""
    t = torch.tensor
    outcomes = {
        "time_out": t([True, True, False, False]),
        "stage_up": t([True, False, False, False]),
        "diverged": t([False, True, False, False]),
    }
    counted = t([True, True, False, True])
    excl = exclusive_outcomes(outcomes, counted)
    names = [vl.outcome_name({k: bool(v[i]) for k, v in excl.items()}) for i in range(4)]
    assert names[:3] == ["knot", "diverged", vl.UNFINISHED]
    assert names[3] == "other"  # 数えたのに項が立っていない (起きない想定) は表と同じ other


def test_frame_plan_real_time():
    plan = vl.frame_plan(1.0 / 240.0, 759, 8)
    assert plan["fps"] == pytest.approx(30.0)
    assert plan["frames_per_move"] == pytest.approx(759 / 8)
    with pytest.raises(ValueError):
        vl.frame_plan(1.0 / 240.0, 759, 0)


def test_parse_env_ids():
    assert vl.parse_env_ids("8", 512) == list(range(8))
    assert vl.parse_env_ids("8", 2) == [0, 1]
    assert vl.parse_env_ids("9,0,5", 512) == [0, 5, 9]
    with pytest.raises(ValueError):
        vl.parse_env_ids("0,512", 512)


def test_parse_resolution():
    assert vl.parse_resolution("960x540") == (960, 540)
    for bad in ("960x541", "960", "x540"):
        with pytest.raises(ValueError):
            vl.parse_resolution(bad)


def test_gpu_index_follows_cuda_visible_devices():
    assert vl.gpu_index("cuda:1") == 1
    assert vl.gpu_index("cuda") == 0
    assert vl.gpu_index("cpu") is None
    assert vl.gpu_index("cuda:0", "1,0") == 1
    assert vl.gpu_index("cuda:2", "1,0") is None


def test_nvidia_smi_parsing():
    gpus = "0, 4243\n1, 12393\n"
    assert vl.parse_free_mib(gpus, 1) == 12393
    assert vl.parse_free_mib(gpus, 2) is None
    apps = "2491044, 7860\n2491146, 314\n2491044, 282\n"
    assert vl.parse_process_mib(apps, 2491044) == 8142
    assert vl.parse_process_mib(apps, 1) == 0


def test_video_decision():
    assert vl.video_decision(12000, 6000) is None
    assert "4000" in vl.video_decision(4000, 6000)
    assert vl.video_decision(None, 6000) is not None


def test_manifest_records_skip_reason():
    m = vl.build_manifest(
        eval_json="e.json",
        argv=["eval_chain.py", "--video"],
        hydra_overrides=["agent.device=cuda:1"],
        checkpoint=CKPT,
        checkpoint_sha256=None,
        git={"commit": "abc", "git_dirty": False},
        plan=None,
        resolution=None,
        clips=[],
        skipped="GPU の空き 1 MiB < --video_min_free_mib 6000 MiB",
    )
    assert m["recorded"] is False and m["skipped"].startswith("GPU")
    assert m["iteration"] == 300 and m["fps"] is None
