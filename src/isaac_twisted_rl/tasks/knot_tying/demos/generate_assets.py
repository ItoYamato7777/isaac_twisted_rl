"""ロープとグリッパーの USD をまとめて生成するブートストラップ。

## なぜこのファイルが要るのか (移植で新たに生じた事情)

このリポジトリの環境では **`pxr` が Kit ランタイム起動後にしか import できない**。
素の `python -c "from pxr import Usd"` は `ModuleNotFoundError` になる
(`isaaclab` は wheel の中に Kit 拡張として同梱されており、`AppLauncher` が
Kit を起動して初めて `pxr` が sys.path に載るため)。

移植元の IsaacLab リポジトリはシステムに pxr があったので
`generate_rope_usd.py` を直接叩けたが、ここではそれができない。そこで
headless の `AppLauncher` を先に起動し、その **後** で各ジェネレータを
import して呼ぶ。ジェネレータ本体 (`generate_rope_usd.py` /
`generate_gripper_usd.py`) は移植元から無改造で、単体でも
`if __name__ == "__main__"` から動く形を保ってある。

## なぜ USD をコミットしないのか

`.gitignore` の `**/*.usd` はこのリポジトリの明示的な方針
(「No USD files allowed in the repo」)。USD はバイナリの crate 形式で
差分が取れないため、生成物として扱いこのスクリプトで再現する。
`data/` ディレクトリは `.gitkeep` だけ追跡している。

## 使い方

デモを動かす前に **一度だけ** 実行すること:

    uv run python -u src/isaac_twisted_rl/tasks/knot_tying/demos/generate_assets.py

    # 一部だけ作り直す
    uv run python -u src/isaac_twisted_rl/tasks/knot_tying/demos/generate_assets.py --rope fine
    uv run python -u src/isaac_twisted_rl/tasks/knot_tying/demos/generate_assets.py --only rope --rope knot_l080_d040 knot_l080_d063
    uv run python -u src/isaac_twisted_rl/tasks/knot_tying/demos/generate_assets.py --only gripper

生成物 (いずれも .gitignore 対象):

    common/rope_model/data/rope_<preset>.usd   (`rope_specs.ROPE_SPECS` の全プリセット)
    common/parallel_gripper/data/gripper_simple.usd

`python -u` で回すのは、出力をリダイレクトしたときにブロックバッファリングで
「数分なにも出ずハングしたように見える」のを避けるため。
"""

from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

# `rope_specs` は pxr も isaaclab も import しない軽量モジュールなので、
# アプリ起動前 (argparse の時点) に読み込んでよい。プリセット名を
# --rope の choices に使いたいのでここで要る。
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "common", "rope_model"))
from rope_specs import ROPE_SPECS  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument(
    "--rope",
    type=str,
    nargs="+",
    default=["all"],
    choices=["all", *ROPE_SPECS],
    help="生成するロープのプリセット (複数可。寸法の DR の 8 組を 1 回の Kit の起動で作るため)。既定の 'all' は全プリセットを生成する。",
)
parser.add_argument(
    "--only",
    type=str,
    default="all",
    choices=["all", "rope", "gripper"],
    help="生成対象を絞る。既定はロープとグリッパーの両方。",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 資産生成に GUI は要らないので headless を既定にする。
# `--headless` は AppLauncher.add_app_launcher_args が既に定義しているので、
# ここで自前に add_argument すると argparse.ArgumentError になる。
# そのため parse 後に代入する形をとる (明示指定は尊重される)。
if not args_cli.headless:
    args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------- 本体
# ここから下は Kit 起動後なので `pxr` を import できる。
from isaac_twisted_rl.tasks.knot_tying.common.parallel_gripper import generate_gripper_usd  # noqa: E402
from isaac_twisted_rl.tasks.knot_tying.common.rope_model import generate_rope_usd  # noqa: E402
from isaac_twisted_rl.tasks.knot_tying.common.rope_model.rope_specs import get_spec  # noqa: E402


def main() -> None:
    made: list[str] = []

    if args_cli.only in ("all", "rope"):
        names = list(ROPE_SPECS) if "all" in args_cli.rope else list(args_cli.rope)
        for name in names:
            # generate() は保存後に bodies 数 / joints 数 / ArticulationRootAPI を
            # assert で検算してくれる。壊れた USD を黙って作らせないために
            # ここでは例外を握りつぶさない。
            made.append(generate_rope_usd.generate(get_spec(name)))

    if args_cli.only in ("all", "gripper"):
        generate_gripper_usd.main()
        made.append(generate_gripper_usd.OUT_PATH)

    print(f"\n=== 生成完了: {len(made)} 個 ===")
    for path in made:
        size = os.path.getsize(path) if os.path.exists(path) else -1
        print(f"  {path} ({size} bytes)")


if __name__ == "__main__":
    main()
    simulation_app.close()
