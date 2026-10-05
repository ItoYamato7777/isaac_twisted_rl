"""v9: v7 の学習の初期状態で、段 2 を選ぶ重みを 2 倍にした版 (段 2 -> 固結びの学習量を増やす)。

    uv run train --task Isaac-Twisted-RL-Knot-v9 --headless
    uv run play  --task Isaac-Twisted-RL-Knot-Play-v9 --num_envs 16

v7 から何を・なぜ変えたかは `README.md` を参照。gym への登録は
`knot_tying/__init__.py` がまとめて行う。

注意: この __init__ は Kit 起動前に実行される。docstring だけにしておくこと。
"""
