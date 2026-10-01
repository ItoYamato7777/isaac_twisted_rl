"""v5 の PPO ハイパーパラメータ。**v3 と同じで、ログの置き場所だけ変える。**

v5 で変えたのは発散対策 (手の動かし方) だけ。v3 の学習ログと並べて発散対策の効果だけを
比べたいので、PPO 側は触らない。観測の次元も v3 と同じ (129) なので、v3 のチェックポイント
から `--resume` で暖機開始することもできる (ただし `drop_z` チャネルの意味は変わっている)。
"""

from isaaclab.utils import configclass

from ...v3.agents.rsl_rl_ppo_cfg import KnotTyingLSTMPPORunnerCfg as _V3LSTM
from ...v3.agents.rsl_rl_ppo_cfg import KnotTyingPPORunnerCfg as _V3MLP
from ...v3.agents.rsl_rl_ppo_cfg import KnotTyingRNNPPORunnerCfg as _V3RNN


@configclass
class KnotTyingPPORunnerCfg(_V3MLP):
    """v5 既定: MLP 方策 (`Isaac-Twisted-RL-Knot-v5`)。"""

    experiment_name = "knot_tying_v5"


@configclass
class KnotTyingRNNPPORunnerCfg(_V3RNN):
    """v5 再帰版 (GRU): `Isaac-Twisted-RL-Knot-v5-RNN`。"""

    experiment_name = "knot_tying_v5_rnn"


@configclass
class KnotTyingLSTMPPORunnerCfg(_V3LSTM):
    """v5 再帰版 (LSTM): `Isaac-Twisted-RL-Knot-v5-LSTM`。"""

    experiment_name = "knot_tying_v5_lstm"
