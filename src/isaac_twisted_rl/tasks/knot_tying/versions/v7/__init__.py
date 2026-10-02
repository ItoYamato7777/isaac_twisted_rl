"""v7: v6 のロープのめり込み解消速度を v4 の既定に戻した版 (段 2 -> 固結びを学べるようにする)。

    uv run train --task Isaac-Twisted-RL-Knot-v7 --headless
    uv run play  --task Isaac-Twisted-RL-Knot-Play-v7 --num_envs 16

v6 から何を・なぜ変えたかは `README.md` を参照。gym への登録は
`knot_tying/__init__.py` がまとめて行う。

注意: この __init__ は Kit 起動前に実行される。docstring だけにしておくこと。
"""
