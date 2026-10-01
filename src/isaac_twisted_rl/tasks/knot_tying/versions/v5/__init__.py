"""v5: v3 に pick&place の発散対策だけを足した版。

    uv run train --task Isaac-Twisted-RL-Knot-v5 --headless          # MLP (学習はこれ)
    uv run play  --task Isaac-Twisted-RL-Knot-Play-v5 --num_envs 1   # まっすぐなロープから評価

v3 から何を・なぜ変えたかは `README.md` を参照。gym への登録は
`knot_tying/__init__.py` がまとめて行う。

注意: この __init__ は Kit 起動前に実行される。docstring だけにしておくこと。
"""
