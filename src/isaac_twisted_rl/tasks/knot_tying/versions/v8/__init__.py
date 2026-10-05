"""v8: v7 に対称性のデータ拡張 (鏡像・逆順) を足した版。環境は v7 と同じで、PPO の更新だけが変わる。

    uv run train --task Isaac-Twisted-RL-Knot-v8 --headless
    uv run play  --task Isaac-Twisted-RL-Knot-Play-v8 --num_envs 16

v7 から何を・なぜ変えたかは `README.md` を参照。gym への登録は
`knot_tying/__init__.py` がまとめて行う。

注意: この __init__ は Kit 起動前に実行される。docstring だけにしておくこと。
"""
