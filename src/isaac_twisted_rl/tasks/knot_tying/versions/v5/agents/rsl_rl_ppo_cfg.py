"""v5 の PPO ハイパーパラメータ。**v4 と同じで、ログの置き場所だけを変える。**

v5 で変えたのは発散対策 (手の動かし方とロープの物性) だけなので、方策・PPO は v4 のまま:
段ごとの MLP 方策 `StageGatedActorCritic` (段 0 / 1 / 2)、`init_noise_std` 0.5、
`num_steps_per_env` 4、`max_iterations` 6000 (= 2000 * 12 // 4)。理由は `../v4/agents/rsl_rl_ppo_cfg.py`
と `../v4/README.md`。

観測の次元も v4 と同じ (`Knot` で 133) なので、v4 のチェックポイントから `--resume` で暖機開始する
こともできる (ただし `drop_z` チャネルの意味は変わっている)。

MLP 版だけを登録する (v4 と同じ。段ごとに振り分ける方策は再帰版を用意していない)。
`StageGatedActorCritic` の rsl_rl への登録 (`register_with_rsl_rl()`) は、下の v4 の設定モジュールを
import した時点で済む。
"""

from isaaclab.utils import configclass

from ...v4.agents.rsl_rl_ppo_cfg import KnotTyingPPORunnerCfg as _V4MLP


@configclass
class KnotTyingPPORunnerCfg(_V4MLP):
    """v5 既定: 段ごとの MLP 方策 (`Isaac-Twisted-RL-Knot-v5`)。"""

    experiment_name = "knot_tying_v5"
