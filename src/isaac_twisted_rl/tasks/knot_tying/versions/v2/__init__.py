"""v2: 固結び (overhand knot) を作る — v1 の学習失敗を受けた作り直し。

    uv run train --task Isaac-Twisted-RL-Twist-v2 --headless        # MLP
    uv run train --task Isaac-Twisted-RL-Twist-v2-RNN --headless    # GRU
    uv run train --task Isaac-Twisted-RL-Twist-v2-LSTM --headless   # LSTM

v1 から何を・なぜ変えたかは `README.md` を参照。gym への登録は
`knot_tying/__init__.py` がまとめて行うので、ここには何も書かない。
v2 は **1 つの環境設定に対して 3 つのエージェント設定**を登録する
(ネットワーク構造の比較)。対応は `knot_tying/__init__.py` の
`_AGENT_VARIANTS`。

注意: この __init__ は Kit 起動前に実行される。`isaaclab` を import
すると `list_envs` / `train` が落ちるので docstring だけにしておくこと
(`env_cfg.py` は文字列 entry point 経由で遅延 import される)。
"""
