"""v4 の PPO ハイパーパラメータ。**方策を段ごとのエージェント (TWISTED-RL-C) にし、行動の初期 std を下げる。**

PPO 本体 (γ・学習率・エポック数など) は v3 (= v2) のまま。変えたのは方策と、1 iteration の手数:

* `num_steps_per_env` 12 -> **4** (2026-09-30)。1 iteration が約 20 分 -> 約 3 分 (ソルバ反復の
  削減と合わせて) になり、PPO の更新が同じ時間で約 6 倍回る。`max_iterations` は総手数が
  v3 の既定と同じになるよう 6000。理由と実測は `../env_cfg.py` の `NUM_STEPS_PER_ENV` と README。

* `policy.class_name = "StageGatedActorCritic"` (`common/policies/stage_gated.py`)。
  actor・critic・std を段 0 / 1 / 2 ごとに持ち、観測の末尾の段の one-hot で振り分ける。
  各エージェントの MLP は v3 と同じ [512, 256, 128]。`num_stages=1` で論文の G 版になる。
* `init_noise_std` 1.0 -> **0.5**。環境は行動を [−1, 1] でクリップするので、std 1.0 の
  ガウス方策では 1 チャネルあたり約 32% が端に張り付く (`paper/Report_Paper.md` §6.5)。v3 の run
  でも `Policy/mean_noise_std` は 26 iteration で 0.98 のままだった。

`RslRlPpoActorCriticCfg` に `num_stages` / `stage_obs_dims` を足した cfg クラスを使う
(configclass は宣言したフィールドしか `to_dict` に出さないため)。

MLP 版だけを登録する (GRU / LSTM は v2 の比較で差が出ない見込みとした。段ごとに振り分ける
方策は再帰版を用意していない)。
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg

from ....common.policies.stage_gated import register_with_rsl_rl
from ...v3.agents.rsl_rl_ppo_cfg import KnotTyingPPORunnerCfg as _V3MLP
from ..env_cfg import NUM_STEPS_PER_ENV, SAVE_INTERVAL

# OnPolicyRunner の eval(class_name) から引けるようにする (stage_gated.py の docstring)。
# この設定モジュールは gym.make / Hydra の時点で読まれ、runner を作るより必ず前に import される。
register_with_rsl_rl()


@configclass
class StageGatedActorCriticCfg(RslRlPpoActorCriticCfg):
    """`StageGatedActorCritic` の設定。"""

    class_name: str = "StageGatedActorCritic"

    num_stages: int = 3
    """エージェントの数 (段 0 / 1 / 2)。1 にすると 1 つの方策を共有する G 版になる。"""

    stage_obs_dims: int = 4
    """観測の末尾にある段の one-hot の次元 (`mdp.knot_stage_one_hot`、段 0〜3)。"""


@configclass
class KnotTyingPPORunnerCfg(_V3MLP):
    """v4 既定: 段ごとの MLP 方策 (`Isaac-Twisted-RL-Knot-v4`)。"""

    experiment_name = "knot_tying_v4"

    # 1 iteration = 4096 環境 x 4 手 = 16384 手 (v3 は x 12 手 = 49152 手、約 20 分)。
    # 環境設定の `NUM_STEPS_PER_ENV` の docstring と README の「学習速度」。
    num_steps_per_env = NUM_STEPS_PER_ENV
    save_interval = SAVE_INTERVAL
    # 総手数を v3 の既定 (2000 iteration x 12 手) と揃える。
    max_iterations = 2000 * 12 // NUM_STEPS_PER_ENV

    policy = StageGatedActorCriticCfg(
        init_noise_std=0.5,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
