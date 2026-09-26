"""v1: 固結び (overhand knot) を作る。

    uv run train --task Isaac-Twisted-RL-Twist-v1 --headless

v0 から何を・なぜ変えたかは `README.md` を参照。gym への登録は
`knot_tying/__init__.py` がまとめて行うので、ここには何も書かない。

注意: この __init__ は Kit 起動前に実行される。`isaaclab` を import
すると `list_envs` / `train` が落ちるので docstring だけにしておくこと
(`env_cfg.py` は文字列 entry point 経由で遅延 import される)。
"""
