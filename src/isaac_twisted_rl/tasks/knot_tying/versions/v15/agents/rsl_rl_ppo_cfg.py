"""v15 の PPO ハイパーパラメータ。**v14 と同じで、ログの置き場所だけを変える。**

critic の特権観測 (`obs_groups`)・`BoundedPPO` (`bounds_loss_coef` 0.01)・段ごとの MLP 方策は v14 のまま。
`BoundedPPO` は段ごとの V の平均と範囲外の割合 (`Loss/value_mean_s*`・`Loss/value_over_s*`) を学習ログに出すので、
0028 の予想 1 (段 0 の V が 1/γ を超えない) を学習ログで見られる。
"""

from isaaclab.utils import configclass

from ...v14.agents.rsl_rl_ppo_cfg import KnotTyingPPORunnerCfg as _V14MLP


@configclass
class KnotTyingPPORunnerCfg(_V14MLP):
    """v15 既定: v14 と同じ (`Isaac-Twisted-RL-Knot-v15`)。"""

    experiment_name = "knot_tying_v15"
