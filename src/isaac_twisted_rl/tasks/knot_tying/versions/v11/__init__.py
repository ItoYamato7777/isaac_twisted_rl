"""v11: v7 の段のまま、表に無く R2 除去後のラダーでも 0 の「行き止まりの形」だけを −1 にした版。

    uv run train --task Isaac-Twisted-RL-Knot-v11 --headless
    uv run play  --task Isaac-Twisted-RL-Knot-Play-v11 --num_envs 16

v7 から何を・なぜ変えたかは `README.md` を参照。gym への登録は
`knot_tying/__init__.py` がまとめて行う。

注意: この __init__ は Kit 起動前に実行される。docstring だけにしておくこと。
"""
