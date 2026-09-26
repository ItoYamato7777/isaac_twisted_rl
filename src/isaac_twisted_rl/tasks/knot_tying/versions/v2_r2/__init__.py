"""v2-R2: v2 の報酬ラダーを R2 (ライデマイスター II) 除去後の位相で判定する版。

    uv run train --task Isaac-Twisted-RL-Twist-v2-R2 --headless        # MLP
    uv run train --task Isaac-Twisted-RL-Twist-v2-R2-RNN --headless    # GRU
    uv run train --task Isaac-Twisted-RL-Twist-v2-R2-LSTM --headless   # LSTM
    uv run play  --task Isaac-Twisted-RL-Twist-Play-v2-R2 --num_envs 1  # p-data を毎手表示

v2 から何を・なぜ変えたかは `README.md` を参照。gym への登録は
`knot_tying/__init__.py` がまとめて行う (タスク ID の `v2-R2` はパッケージ名
`v2_r2` に対応する)。

注意: この __init__ は Kit 起動前に実行される。docstring だけにしておくこと。
"""
