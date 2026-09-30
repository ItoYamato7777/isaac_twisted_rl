"""v4: 段ごとのエージェント (TWISTED-RL-C) と 1 遷移 = 1 エピソード。

    uv run train --task Isaac-Twisted-RL-Knot-v4 --headless          # 学習 (段 0 / 1 / 2 の 3 エージェント)
    uv run play  --task Isaac-Twisted-RL-Knot-Play-v4 --num_envs 16  # まっすぐなロープから段をつないで評価

v3 から何を・なぜ変えたかは `README.md` を参照。gym への登録は
`knot_tying/__init__.py` がまとめて行う。

注意: この __init__ は Kit 起動前に実行される。docstring だけにしておくこと。
"""
