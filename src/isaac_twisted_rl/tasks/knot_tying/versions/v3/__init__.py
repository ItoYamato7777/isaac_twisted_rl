"""v3: 論文 (TWISTED / TWISTED-RL) の中核を取り込んだ版 (`paper/Report.md` §7 P1)。

    uv run train --task Isaac-Twisted-RL-Knot-v3 --headless          # MLP (学習はこれ)
    uv run train --task Isaac-Twisted-RL-Knot-v3-RNN --headless      # GRU
    uv run train --task Isaac-Twisted-RL-Knot-v3-LSTM --headless     # LSTM
    uv run play  --task Isaac-Twisted-RL-Knot-Play-v3 --num_envs 1   # まっすぐなロープから評価

v2-R2 から何を・なぜ変えたかは `README.md` を参照。gym への登録は
`knot_tying/__init__.py` がまとめて行う。

注意: この __init__ は Kit 起動前に実行される。docstring だけにしておくこと。
"""
