"""v7 の PPO ハイパーパラメータ。**v6 と同じで、ログの置き場所だけを変える。**

v7 で変えたのはロープのめり込み解消速度 (物性) だけなので、方策・PPO は v6 (= v4) のまま:
段ごとの MLP 方策 `StageGatedActorCritic` (段 0 / 1 / 2)、`init_noise_std` 0.5、
`num_steps_per_env` 4。観測も v6 と同じ (`Knot` で 197 次元) なので、v6 のチェックポイントから
`--resume` で暖機開始できる (ただし物性が変わるので、方策が見てきた物理とは違う)。
"""

from isaaclab.utils import configclass

from ...v6.agents.rsl_rl_ppo_cfg import KnotTyingPPORunnerCfg as _V6MLP


@configclass
class KnotTyingPPORunnerCfg(_V6MLP):
    """v7 既定: 段ごとの MLP 方策 (`Isaac-Twisted-RL-Knot-v7`)。"""

    experiment_name = "knot_tying_v7"
