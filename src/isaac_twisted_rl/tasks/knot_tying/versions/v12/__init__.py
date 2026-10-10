"""v12: v7 + 到達状態アーカイブに「段が上がった手の後の形」だけを保存する版。

    uv run train --task Isaac-Twisted-RL-Knot-v12 --headless
    uv run play  --task Isaac-Twisted-RL-Knot-Play-v12 --num_envs 16

v7 から何を・なぜ変えたかは `README.md` を参照。gym への登録は
`knot_tying/__init__.py` がまとめて行う。

注意: この __init__ は Kit 起動前に実行される。docstring だけにしておくこと。
"""
