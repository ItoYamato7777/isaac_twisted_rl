"""v0: 交差を 1 つ作って残す (基準バージョン)。

    uv run train --task Isaac-Twisted-RL-Twist-v0 --headless

役割と設計は `README.md` を参照。gym への登録は
`knot_tying/__init__.py` がまとめて行うので、ここには何も書かない。

注意: この __init__ は Kit 起動前に実行される。`isaaclab` を import
すると `list_envs` / `train` が落ちるので docstring だけにしておくこと
(`env_cfg.py` は文字列 entry point 経由で遅延 import される)。
"""
