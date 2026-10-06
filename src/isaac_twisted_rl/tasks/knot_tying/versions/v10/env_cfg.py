"""v10: v7 の段の定義を、固結びへの正規の経路の上の p-data だけに絞った版。

    uv run train --task Isaac-Twisted-RL-Knot-v10 --headless
    uv run play  --task Isaac-Twisted-RL-Knot-Play-v10 --num_envs 16

v7 から何を・なぜ変えたかは `README.md`、仕様は `docs/decisions/0023-v10-path-stages.md`。
ここは「何を配線したか」だけを書く。

v7 の設定クラスを **継承して、段の定義だけ** を変える (1 版 1 変数):

1. `stage_definition = "path"`。段を引く項 (`common/mdp/` の `stage_function` を通るもの: 段の one-hot・
   報酬・終了・アーカイブの保存と復元の検算・`next_edge`・ログ) が `topology.PathStage` を使う。
2. 初期アーカイブを、台本の初期アーカイブから新しい段の定義に入る形だけを残したファイルにする
   (`SEED_ARCHIVE`)。段の定義を変えた結果で、別の変数ではない (v7 の初期アーカイブの部分集合)。

観測の形・報酬の値・PPO・行動・物理・`stage_weights` (1:1:1)・Play (chain モード) は v7 のまま。
v8 (対称性のデータ拡張)・v9 (`stage_weights`) は継承しない。v7 との差がこの 2 つだけであることは
`tests/test_v10_cfg.py` が静的に検査する。
"""

from __future__ import annotations

import pathlib

from isaaclab.utils import configclass

from ...common.rope_task_cfg import make_preset_cfgs
from ..v5.env_cfg import play_post_init as _play_post_init
from ..v7.env_cfg import KnotTyingV7EnvCfg

__all__ = ["SEED_ARCHIVE", "KnotTyingV10EnvCfg"]

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[6]

SEED_ARCHIVE = str(_REPO_ROOT / "outputs" / "archive_seed" / "overhand_{num_links}links_v10.pt")
"""v4〜v9 の初期アーカイブ (`overhand_{num_links}links.pt`) のうち、v10 の段の定義で段が変わらない形だけ。

`scripts/rsl_rl/view_archive.py --write_filtered` で作る (形は元のファイルとまったく同じ)。`Knot` (28 リンク) では
段 0 の 128 形・段 1 の 264 形・段 2 の 18 形が残り、段 2 の 112 形が外れる見込み (0023 の「実装の見通し」)。
無ければ `ResetToStage` が警告して飛ばし、まっすぐなロープだけから始まる (`Knot` 以外のプリセット)。
外れた形を復元の後に捨てる方法は採らない。段の選び方 (`stage_weights`) の割合が、捨てた分だけ v7 とずれるため。
"""


@configclass
class KnotTyingV10EnvCfg(KnotTyingV7EnvCfg):
    """v10 の環境設定。v7 との差は `stage_definition` と、初期アーカイブのファイルだけ。"""

    stage_definition: str = "path"
    """段の定義 (`topology.STAGE_DEFINITIONS`)。v4〜v9 はこの属性を持たず、`bfs` (`BfsPotential.stage`) になる。"""

    def __post_init__(self):
        super().__post_init__()
        self.events.reset_to_stage.params["load_paths"] = (SEED_ARCHIVE,)


globals().update(make_preset_cfgs(KnotTyingV10EnvCfg, __name__, play_post_init=_play_post_init))
