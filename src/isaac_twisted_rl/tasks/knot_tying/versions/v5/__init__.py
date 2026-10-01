"""v5: v4 (段ごとのエージェント・1 遷移 = 1 エピソード) に pick&place の発散対策を足した版。

    uv run train --task Isaac-Twisted-RL-Knot-v5 --headless           # 段ごとの MLP (学習はこれ)
    uv run play  --task Isaac-Twisted-RL-Knot-Play-v5 --num_envs 16   # まっすぐなロープから段をつないで評価

v4 / v3 から何を・なぜ変えたかは `README.md` を参照。gym への登録は
`knot_tying/__init__.py` がまとめて行う。

注意: この __init__ は Kit 起動前に実行される。docstring だけにしておくこと。
"""
