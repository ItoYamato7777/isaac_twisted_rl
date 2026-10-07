"""v11: v7 の段のまま、表に無く R2 除去後のラダーでも 0 の「行き止まりの形」だけを −1 にした版。

    uv run train --task Isaac-Twisted-RL-Knot-v11 --headless
    uv run play  --task Isaac-Twisted-RL-Knot-Play-v11 --num_envs 16

v7 から何を・なぜ変えたかは `README.md`、仕様は `docs/decisions/0024-v11-dead-end-stages.md`。
ここは「何を配線したか」だけを書く。

v7 の設定クラスを **継承して、段の定義だけ** を変える (1 版 1 変数): `stage_definition = "bfs_dead_end"`。
段を引く項 (`common/mdp/` の `stage_function` を通るもの: 段の one-hot・報酬・終了・アーカイブの保存と
復元の検算・`next_edge`・ログ) が `topology.DeadEndStage` を使う。

初期アーカイブは v7 と同じ `overhand_{num_links}links.pt` (台本の形は全部表にあり、v11 でも段が変わらない。
0024 の「データ」)。`next_edge` は v7 と同じ絞らない表 (`get_next_edge_table(device, "bfs_dead_end")`)。
観測の形・報酬の値・PPO・行動・物理・`stage_weights` (1:1:1)・Play (chain モード) は v7 のまま。
v8 (対称性)・v9 (`stage_weights`)・v10 (経路の段と初期アーカイブ) は継承しない。v7 との差がこれだけであることは
`tests/test_v11_cfg.py` が静的に検査する。
"""

from __future__ import annotations

from isaaclab.utils import configclass

from ...common.rope_task_cfg import make_preset_cfgs
from ..v5.env_cfg import play_post_init as _play_post_init
from ..v7.env_cfg import KnotTyingV7EnvCfg

__all__ = ["KnotTyingV11EnvCfg"]


@configclass
class KnotTyingV11EnvCfg(KnotTyingV7EnvCfg):
    """v11 の環境設定。v7 との差は `stage_definition` だけ。"""

    stage_definition: str = "bfs_dead_end"
    """段の定義 (`topology.STAGE_DEFINITIONS`)。v4〜v9 はこの属性を持たず、`bfs` (`BfsPotential.stage`) になる。"""


globals().update(make_preset_cfgs(KnotTyingV11EnvCfg, __name__, play_post_init=_play_post_init))
