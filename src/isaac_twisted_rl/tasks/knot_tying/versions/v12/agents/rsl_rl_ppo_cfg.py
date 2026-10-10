"""v12 の PPO ハイパーパラメータ。**v7 と同じで、ログの置き場所だけを変える。**

v12 で変えたのはアーカイブの保存の規則 (環境の設定) だけなので、方策・PPO は v7 (= v4) のまま:
段ごとの MLP 方策 `StageGatedActorCritic` (段 0 / 1 / 2)、`init_noise_std` 0.5、
`num_steps_per_env` 4。v8 の対称性のデータ拡張 (`algorithm.symmetry_cfg`) は入れない。
観測の形も v7 と同じ (`Knot` で 197 次元)。
"""

from isaaclab.utils import configclass

from ...v7.agents.rsl_rl_ppo_cfg import KnotTyingPPORunnerCfg as _V7MLP


@configclass
class KnotTyingPPORunnerCfg(_V7MLP):
    """v12 既定: 段ごとの MLP 方策 (`Isaac-Twisted-RL-Knot-v12`)。"""

    experiment_name = "knot_tying_v12"
