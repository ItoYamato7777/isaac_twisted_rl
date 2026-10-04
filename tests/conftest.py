"""試験が共通で使う import の経路。

`isaac_twisted_rl.tasks` の `__init__.py` は配下を全部 import する (Isaac Sim が要る) ので、Isaac Sim なしで回す
純粋なモジュール (`topology`・`state_archive`・`beam_search`・`policies`・`mdp/finger_geometry` など) は、
置き場所のフォルダを `sys.path` に足してトップレベルの名前で読む。`common/` と `common/mdp/` の
モジュール名は重ならない。`tests/` は pytest が足すので、`from test_knot_detect import ...` もそのまま動く。
"""

import pathlib
import sys

_COMMON = pathlib.Path(__file__).resolve().parents[1] / "src" / "isaac_twisted_rl" / "tasks" / "knot_tying" / "common"

for _p in (_COMMON / "mdp", _COMMON):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
