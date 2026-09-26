"""v2-R2 の PPO ハイパーパラメータ。**v2 と完全に同じで、ログの置き場所だけ変える。**

v2-R2 で変えたのは報酬ラダーの判定器だけ (`../env_cfg.py`)。ハイパーパラメータも
同時に変えると、効果を v2 と比べられなくなるので据え置く。

`experiment_name` を分けるのは v2 と同じ理由 (`../../v2/agents/rsl_rl_ppo_cfg.py`
の docstring)。`--resume` が `logs/rsl_rl/{experiment_name}/` から最新 run を
拾うので、共有すると v2 のチェックポイントを黙って読んでしまう。

## v2 のチェックポイントから暖機開始する場合

観測の次元・並び・ネットワーク形状は v2 と同一 (`knot_class` の中身が R2 除去後の
段に変わるだけ) なので、`--checkpoint` で v2 の `model_*.pt` を直接指定すれば
`strict=True` のまま読める。ただし v2 の方策は偽の C2 (掛け渡し) を覚えている
可能性が高いので、比較の基準としては **ゼロからの学習** を先に回すこと。
"""

from isaaclab.utils import configclass

from ...v2.agents.rsl_rl_ppo_cfg import KnotTyingLSTMPPORunnerCfg as _V2LSTM
from ...v2.agents.rsl_rl_ppo_cfg import KnotTyingPPORunnerCfg as _V2MLP
from ...v2.agents.rsl_rl_ppo_cfg import KnotTyingRNNPPORunnerCfg as _V2RNN


@configclass
class KnotTyingPPORunnerCfg(_V2MLP):
    """v2-R2 既定: MLP 方策 (`Isaac-Twisted-RL-Twist-v2-R2`)。"""

    experiment_name = "knot_tying_v2_r2"


@configclass
class KnotTyingRNNPPORunnerCfg(_V2RNN):
    """v2-R2 再帰版 (GRU): `Isaac-Twisted-RL-Twist-v2-R2-RNN`。"""

    experiment_name = "knot_tying_v2_r2_rnn"


@configclass
class KnotTyingLSTMPPORunnerCfg(_V2LSTM):
    """v2-R2 再帰版 (LSTM): `Isaac-Twisted-RL-Twist-v2-R2-LSTM`。"""

    experiment_name = "knot_tying_v2_r2_lstm"
