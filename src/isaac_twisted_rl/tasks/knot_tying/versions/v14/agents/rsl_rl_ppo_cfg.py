"""v14 の PPO ハイパーパラメータ。**v13 と同じで、ログの置き場所と critic の観測だけを変える。**

critic の観測を `["rope_params", "policy"]` にする: 引いたロープの物理の倍率 (正規化した 6 次元、`mdp.rope_physics_params`)
を先に、actor と同じ観測 (197 次元) を後ろに並べる (203 次元)。`StageGatedActorCritic` は critic の観測でも
**最後の 4 列** (段の one-hot) で頭を選ぶので、特権観測は前に置く (`common/policies/stage_gated.py`)。actor の観測は
v13 と同じ。理由は `docs/decisions/0027-v14-s0-reshape-dr-priv-critic.md`。

critic の入力の幅が変わるので、v12・v13 のチェックポイントからは再開できない。特権観測を切るときは
`agent.obs_groups.critic=[policy]`。

PPO (`BoundedPPO`、`bounds_loss_coef` 0.01) と方策 (段ごとの MLP、`init_noise_std` 0.5、`num_steps_per_env` 4) は v13 のまま。
"""

from isaaclab.utils import configclass

from ...v13.agents.rsl_rl_ppo_cfg import KnotTyingPPORunnerCfg as _V13MLP


@configclass
class KnotTyingPPORunnerCfg(_V13MLP):
    """v14 既定: 段ごとの MLP 方策 + `BoundedPPO` + critic の特権観測 (`Isaac-Twisted-RL-Knot-v14`)。"""

    experiment_name = "knot_tying_v14"
    obs_groups = {"policy": ["policy"], "critic": ["rope_params", "policy"]}
