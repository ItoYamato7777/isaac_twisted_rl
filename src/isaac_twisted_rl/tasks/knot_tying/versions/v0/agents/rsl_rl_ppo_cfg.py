# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""v0 (交差 1 つ) の PPO ハイパーパラメータ。

## lift / reach と大きく違う点: エピソードが極端に短い

この環境は **1 env ステップ = pick&place 1 手** なので、1 エピソードが
たった 5 ステップしかない (`KnotTyingV0EnvCfg.actions_per_episode`)。
一方 1 ステップの実時間コストは数百物理ステップぶんと非常に重い。

そのため:

* `num_steps_per_env` は 10 (= 2 エピソードぶん)。lift の 24 のままだと
  1 イテレーションが長すぎ、更新が遅すぎる。
* `gamma` は 0.95 と低め (`env_cfg.GAMMA`)。5 ステップしかない地平では、
  lift の 0.98 (実質 50 ステップ相当を見る) は長すぎる。
* `entropy_coef` はやや高め (0.01)。行動が「どのリンクを掴むか」という
  離散選択を連続チャネルの量子化で表しているので、探索が潰れると
  同じリンクばかり掴む方策に嵌まりやすい。

これらは **v0 を基準バージョンとして凍結するための値**で、`main` ブランチ
(コミット 1f5693f 時点) と同一。改良は v1 以降で行う。
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)

from ..env_cfg import GAMMA


@configclass
class KnotTyingPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 10
    max_iterations = 2000
    save_interval = 50
    experiment_name = "knot_tying_v0"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=GAMMA,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
