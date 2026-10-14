"""v13: v12 + PPO の損失に「方策の平均が行動の範囲 [−1, 1] の外へ出た分」の罰則を足す版。

    uv run train --task Isaac-Twisted-RL-Knot-v13 --headless
    uv run play  --task Isaac-Twisted-RL-Knot-Play-v13 --num_envs 16

v12 から何を・なぜ変えたかは `README.md` を参照。gym への登録は
`knot_tying/__init__.py` がまとめて行う。

注意: この __init__ は Kit 起動前に実行される。docstring だけにしておくこと。
"""
