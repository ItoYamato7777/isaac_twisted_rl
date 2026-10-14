"""v13 の PPO ハイパーパラメータ。**v12 の PPO に、平均の範囲外の罰則 (`bounds_loss_coef`) だけを足す。**

algorithm を `BoundedPPO` (`common/policies/bounded_ppo.py`) にし、損失に
`bounds_loss_coef · mean_batch(Σ_チャネル max(|μ| − 1, 0)²)` を足す。仕様と係数の理由は
`docs/decisions/0026-v13-action-bounds-loss.md`。

PPO の他の値 (γ・学習率・エポック数など、v12 = v7 = v4 = v2 の `_algorithm`) と方策 (段ごとの MLP、
`init_noise_std` 0.5、`num_steps_per_env` 4) は v12 のまま。値を書き写さず、`__post_init__` で v12 の
algorithm の値を全部引き継いでクラスだけを替えるので、v12 と食い違わない (`tests/test_v13_cfg.py`)。
"""

import dataclasses

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoAlgorithmCfg

from ....common.policies.bounded_ppo import register_with_rsl_rl
from ...v12.agents.rsl_rl_ppo_cfg import KnotTyingPPORunnerCfg as _V12MLP

# OnPolicyRunner の eval(class_name) から引けるようにする (bounded_ppo.py の docstring)。
# この設定モジュールは gym.make / Hydra の時点で読まれ、runner を作るより必ず前に import される。
register_with_rsl_rl()

BOUNDS_LOSS_COEF = 0.01
"""罰則の係数。v12 s44 の it70 の範囲外の出方 (観測の平均 0.39) で罰則の項が約 0.004 となり、
同じ時期の `Loss/surrogate` (−0.003〜−0.012) と同じ桁になる (0026 の「係数」)。"""


@configclass
class BoundedPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """`BoundedPPO` の設定 (configclass は宣言したフィールドしか `to_dict` に出さないため)。"""

    class_name: str = "BoundedPPO"

    bounds_loss_coef: float = 0.0
    """平均の範囲外の罰則の係数。0 なら更新は PPO と同じ。"""


@configclass
class KnotTyingPPORunnerCfg(_V12MLP):
    """v13 既定: 段ごとの MLP 方策 + `BoundedPPO` (`Isaac-Twisted-RL-Knot-v13`)。"""

    experiment_name = "knot_tying_v13"

    def __post_init__(self):
        super().__post_init__()
        values = {f.name: getattr(self.algorithm, f.name) for f in dataclasses.fields(self.algorithm)}
        del values["class_name"]
        self.algorithm = BoundedPpoAlgorithmCfg(**values, bounds_loss_coef=BOUNDS_LOSS_COEF)
