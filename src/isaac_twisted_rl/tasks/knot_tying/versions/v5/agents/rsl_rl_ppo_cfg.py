"""v5 の PPO ハイパーパラメータ。**v3 と同じで、ログの置き場所と 1 iteration の手数を変える。**

v5 で変えたのは発散対策 (手の動かし方) と、v4 (4115db4) と同じ学習速度の改善だけ。観測の
次元も v3 と同じ (129) なので、v3 のチェックポイントから `--resume` で暖機開始することも
できる (ただし `drop_z` チャネルの意味は変わっている)。

* `num_steps_per_env` 12 -> 4、`max_iterations` を総手数が v3 の既定と同じになるよう
  2000 * 12 // 4 に調整。理由と実測は `../env_cfg.py` の `NUM_STEPS_PER_ENV` と
  `../v4/README.md` の「学習速度」。
"""

from isaaclab.utils import configclass

from ...v3.agents.rsl_rl_ppo_cfg import KnotTyingLSTMPPORunnerCfg as _V3LSTM
from ...v3.agents.rsl_rl_ppo_cfg import KnotTyingPPORunnerCfg as _V3MLP
from ...v3.agents.rsl_rl_ppo_cfg import KnotTyingRNNPPORunnerCfg as _V3RNN
from ..env_cfg import NUM_STEPS_PER_ENV, SAVE_INTERVAL


@configclass
class KnotTyingPPORunnerCfg(_V3MLP):
    """v5 既定: MLP 方策 (`Isaac-Twisted-RL-Knot-v5`)。"""

    experiment_name = "knot_tying_v5"
    num_steps_per_env = NUM_STEPS_PER_ENV
    save_interval = SAVE_INTERVAL
    max_iterations = 2000 * 12 // NUM_STEPS_PER_ENV


@configclass
class KnotTyingRNNPPORunnerCfg(_V3RNN):
    """v5 再帰版 (GRU): `Isaac-Twisted-RL-Knot-v5-RNN`。"""

    experiment_name = "knot_tying_v5_rnn"
    num_steps_per_env = NUM_STEPS_PER_ENV
    save_interval = SAVE_INTERVAL
    max_iterations = 2000 * 12 // NUM_STEPS_PER_ENV


@configclass
class KnotTyingLSTMPPORunnerCfg(_V3LSTM):
    """v5 再帰版 (LSTM): `Isaac-Twisted-RL-Knot-v5-LSTM`。"""

    experiment_name = "knot_tying_v5_lstm"
    num_steps_per_env = NUM_STEPS_PER_ENV
    save_interval = SAVE_INTERVAL
    max_iterations = 2000 * 12 // NUM_STEPS_PER_ENV
