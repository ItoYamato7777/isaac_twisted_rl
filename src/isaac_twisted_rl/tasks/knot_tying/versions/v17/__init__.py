"""v17: v15 + 待ちのフェーズ (close・hold・release・settle_out・relax) の尺を詰める (学習を速くする。物理は変えない)。

    uv run train --task Isaac-Twisted-RL-Knot-v17 --headless
    uv run play  --task Isaac-Twisted-RL-Knot-Play-v17 --num_envs 16

v15 から何を・なぜ変えたかは `README.md` を参照。gym への登録は
`knot_tying/__init__.py` がまとめて行う。

注意: この __init__ は Kit 起動前に実行される。docstring だけにしておくこと。
"""
