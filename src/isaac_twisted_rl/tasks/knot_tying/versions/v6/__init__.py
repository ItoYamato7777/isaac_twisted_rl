"""v6: v5 に「次にたどる位相の辺」のリンクマスクを観測として足した版。

    uv run train --task Isaac-Twisted-RL-Knot-v6 --headless           # 段ごとの MLP (学習はこれ)
    uv run play  --task Isaac-Twisted-RL-Knot-Play-v6 --num_envs 16   # まっすぐなロープから段をつないで評価

v5 から何を・なぜ変えたかは `README.md` を参照。gym への登録は
`knot_tying/__init__.py` がまとめて行う。

注意: この __init__ は Kit 起動前に実行される。docstring だけにしておくこと。
"""
