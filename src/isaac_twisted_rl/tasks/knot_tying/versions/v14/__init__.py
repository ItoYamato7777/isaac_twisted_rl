"""v14: v13 + 段 0 の保存の規則 (交差なしのまま形を変えた手) + ロープの物理の DR + critic の特権観測。

    uv run train --task Isaac-Twisted-RL-Knot-v14 --headless
    uv run play  --task Isaac-Twisted-RL-Knot-Play-v14 --num_envs 16

v13 から何を・なぜ変えたかは `README.md` を参照。gym への登録は
`knot_tying/__init__.py` がまとめて行う。

注意: この __init__ は Kit 起動前に実行される。docstring だけにしておくこと。
"""
