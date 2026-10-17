"""v16: v15 + ロープの直径・長さを env ごとに振る DR (寸法の DR)。

    uv run train --task Isaac-Twisted-RL-Knot-v16 --headless
    uv run play  --task Isaac-Twisted-RL-Knot-Play-v16 --num_envs 16

v15 から何を・なぜ変えたかは `README.md` を参照。gym への登録は
`knot_tying/__init__.py` がまとめて行う。

注意: この __init__ は Kit 起動前に実行される。docstring だけにしておくこと。
"""
