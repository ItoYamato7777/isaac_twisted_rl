"""v6 の PPO ハイパーパラメータ。**v5 と同じで、ログの置き場所だけを変える。**

観測の次元が v5 (`Knot` で 129) から 64 増えて 193 になるので、v5 のチェックポイントからは
`--resume` で暖機開始できない (`load_state_dict` は strict で、入力層の形が合わない)。
"""

from isaaclab.utils import configclass

from ...v5.agents.rsl_rl_ppo_cfg import KnotTyingLSTMPPORunnerCfg as _V5LSTM
from ...v5.agents.rsl_rl_ppo_cfg import KnotTyingPPORunnerCfg as _V5MLP
from ...v5.agents.rsl_rl_ppo_cfg import KnotTyingRNNPPORunnerCfg as _V5RNN


@configclass
class KnotTyingPPORunnerCfg(_V5MLP):
    """v6 既定: MLP 方策 (`Isaac-Twisted-RL-Knot-v6`)。"""

    experiment_name = "knot_tying_v6"


@configclass
class KnotTyingRNNPPORunnerCfg(_V5RNN):
    """v6 再帰版 (GRU): `Isaac-Twisted-RL-Knot-v6-RNN`。"""

    experiment_name = "knot_tying_v6_rnn"


@configclass
class KnotTyingLSTMPPORunnerCfg(_V5LSTM):
    """v6 再帰版 (LSTM): `Isaac-Twisted-RL-Knot-v6-LSTM`。"""

    experiment_name = "knot_tying_v6_lstm"
