"""v10: v7 の段の定義を、固結びへの正規の経路の上の p-data だけに絞った版。

    uv run train --task Isaac-Twisted-RL-Knot-v10 --headless
    uv run play  --task Isaac-Twisted-RL-Knot-Play-v10 --num_envs 16

v7 から何を・なぜ変えたかは `README.md` を参照。gym への登録は
`knot_tying/__init__.py` がまとめて行う。

注意: この __init__ は Kit 起動前に実行される。docstring だけにしておくこと。
"""
