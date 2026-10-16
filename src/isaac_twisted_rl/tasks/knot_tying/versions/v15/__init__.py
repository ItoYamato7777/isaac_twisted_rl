"""v15: v14 + 段の変化・発散と同じ手では時間切れを立てない (時間切れの bootstrap の不具合の修正)。

    uv run train --task Isaac-Twisted-RL-Knot-v15 --headless
    uv run play  --task Isaac-Twisted-RL-Knot-Play-v15 --num_envs 16

v14 から何を・なぜ変えたかは `README.md` を参照。gym への登録は
`knot_tying/__init__.py` がまとめて行う。

注意: この __init__ は Kit 起動前に実行される。docstring だけにしておくこと。
"""
