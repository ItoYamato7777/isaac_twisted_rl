"""v12: v7 + 到達状態アーカイブに「段が上がった手の後の形」だけを保存する版。

    uv run train --task Isaac-Twisted-RL-Knot-v12 --headless
    uv run play  --task Isaac-Twisted-RL-Knot-Play-v12 --num_envs 16

v7 から何を・なぜ変えたかは `README.md`、仕様は `docs/decisions/0025-archive-save-on-stage-up.md`。
ここは「何を配線したか」だけを書く。

v7 の設定クラスを **継承して、保存の項 `record_stage_archive` の params を 2 つ** 書く (1 版 1 変数):

1. `save_on = "stage_up"`: 段が上がった手 (手の前の段 < 手の後の段、手の後の段が 1 か 2) の後の形だけを保存する
   (`state_archive.rows_to_save`)。
2. `history_every = 200`: 記録のために、200 env ステップ (= 50 iteration) ごとにアーカイブの写しを
   `{log_dir}/archive_history/` に残す。学習の挙動は変えない。

段の定義 (`bfs`)・初期アーカイブ (`overhand_{num_links}links.pt`)・`stage_weights` (1:1:1)・容量・引き方・観測・報酬・
終了・行動・物理・PPO は v7 のまま。v8〜v11 は継承しない。v7 との差がこれだけであることは
`tests/test_v12_cfg.py` が静的に検査する。

Play (chain モード) は v7 と同じ設定に、`history_every = 0` だけを足す。Play は `log_dir` を評価する
チェックポイントのフォルダ (= 学習の run) にするので、0 にしないと学習が残した写しを評価が上書きする。
`save_on` は Play でもそのままにする (Play はアーカイブから引かず、ファイルにも書かないので結果に効かない)。
"""

from __future__ import annotations

from isaaclab.utils import configclass

from ...common.rope_task_cfg import make_preset_cfgs
from ..v5.env_cfg import play_post_init as _v7_play_post_init
from ..v7.env_cfg import KnotTyingV7EnvCfg

__all__ = ["HISTORY_EVERY", "SAVE_ON", "KnotTyingV12EnvCfg", "play_post_init"]

SAVE_ON = "stage_up"
"""どの手の後の形を保存するか (`state_archive.SAVE_RULES`)。v4〜v11 は `every_move`。"""

HISTORY_EVERY = 200
"""アーカイブの写しを残す間隔 [env ステップ]。1 iteration = 4 env ステップなので 50 iteration ごと。

1 つ約 17 MB で、300 iteration で 6 つ。0025 の予想 (段 2 のアーカイブで 1 つの p-data が 20% を超えない) を
`view_archive.py --dump_pdata` で確かめるのに使う。
"""


@configclass
class KnotTyingV12EnvCfg(KnotTyingV7EnvCfg):
    """v12 の環境設定。v7 との差は `rewards.record_stage_archive.params` の `save_on` と `history_every` だけ。"""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.record_stage_archive.params["save_on"] = SAVE_ON
        self.rewards.record_stage_archive.params["history_every"] = HISTORY_EVERY


def play_post_init(cfg: KnotTyingV12EnvCfg) -> None:
    """Play (chain モード): v7 と同じ設定に、アーカイブの写しを書かない設定だけを足す。"""
    _v7_play_post_init(cfg)
    cfg.rewards.record_stage_archive.params["history_every"] = 0


globals().update(make_preset_cfgs(KnotTyingV12EnvCfg, __name__, play_post_init=play_post_init))
