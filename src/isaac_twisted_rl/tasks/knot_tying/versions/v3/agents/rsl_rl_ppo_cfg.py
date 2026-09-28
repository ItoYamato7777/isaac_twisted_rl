"""v3 の PPO ハイパーパラメータ。**v2-R2 (= v2) と同じで、ログの置き場所だけ変える。**

v3 で変えたのは環境側 (リセット・報酬・終了・ロープ・行動の範囲・尺) で、
ここを同時に変えると効果を切り分けられなくなるので据え置く。

* `num_steps_per_env` = 12 は `actions_per_episode` (12) と同じ。成功で終わる
  エピソードは短くなるので、整数倍であることにはもう意味は無い。
* `gamma` は v2 の `env_cfg.GAMMA` (0.95) を共有したまま (整形報酬と必ず一致させる)。

`experiment_name` を分けるのは v2 と同じ理由 (`--resume` が
`logs/rsl_rl/{experiment_name}/` から最新 run を拾うため)。v3 は既定のロープが
`Knot` (28 リンク) で観測の次元が v2-R2 (`Twist`, 20 リンク) と違うので、
v2-R2 のチェックポイントからは暖機開始できない。
"""

from isaaclab.utils import configclass

from ...v2_r2.agents.rsl_rl_ppo_cfg import KnotTyingLSTMPPORunnerCfg as _V2R2LSTM
from ...v2_r2.agents.rsl_rl_ppo_cfg import KnotTyingPPORunnerCfg as _V2R2MLP
from ...v2_r2.agents.rsl_rl_ppo_cfg import KnotTyingRNNPPORunnerCfg as _V2R2RNN


@configclass
class KnotTyingPPORunnerCfg(_V2R2MLP):
    """v3 既定: MLP 方策 (`Isaac-Twisted-RL-Knot-v3`)。"""

    experiment_name = "knot_tying_v3"


@configclass
class KnotTyingRNNPPORunnerCfg(_V2R2RNN):
    """v3 再帰版 (GRU): `Isaac-Twisted-RL-Knot-v3-RNN`。"""

    experiment_name = "knot_tying_v3_rnn"


@configclass
class KnotTyingLSTMPPORunnerCfg(_V2R2LSTM):
    """v3 再帰版 (LSTM): `Isaac-Twisted-RL-Knot-v3-LSTM`。"""

    experiment_name = "knot_tying_v3_lstm"
