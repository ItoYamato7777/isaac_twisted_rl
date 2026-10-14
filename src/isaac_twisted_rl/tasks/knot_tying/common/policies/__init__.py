"""独自の方策ネットワーク (rsl_rl の `ActorCritic` の派生)。v4 で追加。

    stage_gated.py   段ごとに独立した actor / critic / std を持つ方策 (TWISTED-RL-C 相当)
    bounded_ppo.py   PPO の損失に方策の平均の範囲外の罰則を足す algorithm (v13 で追加)

注意: この __init__ は Kit 起動前に実行される (`tasks/__init__.py` の `import_packages`)。
docstring だけにしておき、本体は使う側 (`versions/v4/agents/`) が import すること。
"""
