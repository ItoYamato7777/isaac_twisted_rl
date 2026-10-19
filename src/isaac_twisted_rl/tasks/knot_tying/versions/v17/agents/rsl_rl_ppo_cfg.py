"""v17 の PPO ハイパーパラメータ。**v15 と同じで、ログの置き場所だけを変える。**

v17 は 1 手の尺だけを変える版なので、PPO・critic の特権観測 (`obs_groups`)・`BoundedPPO`・段ごとの MLP 方策は
v15 のまま。観測の幅も v15 と同じだが、比べる相手は v15 の同じシードを 0 から学習したもの
(`docs/decisions/0030-v17-shorter-wait-phases.md`) なので、v15 のチェックポイントからは再開しない。
"""

from isaaclab.utils import configclass

from ...v15.agents.rsl_rl_ppo_cfg import KnotTyingPPORunnerCfg as _V15MLP


@configclass
class KnotTyingPPORunnerCfg(_V15MLP):
    """v17 既定: v15 と同じ (`Isaac-Twisted-RL-Knot-v17`)。"""

    experiment_name = "knot_tying_v17"
