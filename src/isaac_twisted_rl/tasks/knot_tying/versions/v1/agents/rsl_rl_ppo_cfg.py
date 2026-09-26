"""v1 (固結び) の PPO ハイパーパラメータ。

## lift / reach と大きく違う点: エピソードが極端に短い

この環境は **1 env ステップ = pick&place 1 手** なので、1 エピソードが
たった 8 ステップしかない (`KnotTyingV1EnvCfg.actions_per_episode`)。
一方 1 ステップの実時間コストは 686 物理ステップぶんと非常に重い。

そのため:

* `num_steps_per_env` は 16 (= 2 エピソードぶん)。**エピソード長の整数倍に
  すること**。8 手に対して 10 のような半端な値だと、ロールアウトが毎回
  違うフェーズでエピソード境界を跨いで切れ、GAE の bootstrap が不安定になる。
* `entropy_coef` はやや高め (0.02)。行動が「どのリンクを掴むか」という
  離散選択を連続チャネルの量子化で表しているので、探索が潰れると
  同じリンクばかり掴む方策に嵌まりやすい。地平が伸び目標が疎になった
  ぶん、v0 の 0.01 から上げてある。

## v0 から変えた点

* `gamma` 0.95 -> **0.98** (`env_cfg.GAMMA`)。地平が 5 -> 8 手に伸び、
  報酬が 3 遷移先になった。0.95 だと初手から見た終端報酬が
  0.95^7 ≈ 0.70 まで減衰するが、0.98 なら 0.87 残る。
  **`RewardsCfg.shaping.params["gamma"]` と必ず一致させること。**
  ポテンシャル整形 `F = γΦ(s') − Φ(s)` の最適方策不変性 (Ng et al. 1999)
  は、この γ が方策の割引率と一致していて初めて成り立つ。
  そのため数値はここに書かず `env_cfg.GAMMA` を import している。
* `empirical_normalization` False -> **True**。観測が 75 -> 105 次元に
  増えたが、うち 60 次元はメートル単位のリンク位置で、新しく足した位相
  特徴 30 次元は [0, 1] に収まる。正規化しないと位相側が数値的に埋もれる。
* ネットワークを [256,128,64] -> **[512,256,128]**。観測が増え、価値関数が
  4 段ラダー x 8 手を表現する必要がある。1 手が 686 物理ステップなので、
  ネットワークの計算コストは誤差。
* `num_steps_per_env` 10 -> 16、`max_iterations` 2000 -> 4000。
* `experiment_name` を `knot_tying_v1` にして、ログを v0 と分ける。
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
    num_steps_per_env = 16
    max_iterations = 4000
    save_interval = 50
    experiment_name = "knot_tying_v1"
    empirical_normalization = True
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.02,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=GAMMA,  # env_cfg.GAMMA を共有 (報酬整形と必ず一致させる)
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
