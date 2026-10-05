"""v8 の PPO ハイパーパラメータ。**v7 に対称性のデータ拡張を足す。それ以外は v7 (= v4) のまま。**

* `algorithm.symmetry_cfg`: PPO の更新のミニバッチを、恒等・鏡像・逆順・鏡像+逆順の 4 倍にして
  学習する (`use_data_augmentation=True`。rsl_rl 3.0.1 の `PPO.update`)。advantage・return・
  旧方策の対数尤度は元のものを 4 回繰り返して使う。変換は `common/topology/symmetry.py`、
  rsl_rl に渡す関数は `common/mdp/symmetry.py`。
* `use_mirror_loss=False`: 鏡像の損失は足さない (データ拡張だけ)。rsl_rl は損失を足さなくても
  `Loss/symmetry` (変換した観測での方策の平均と、変換した平均の行動の差) を記録するので、方策が
  どれだけ対称になったかはそれで見る。
* 方策 (段ごとの MLP `StageGatedActorCritic`)・`init_noise_std` 0.5・`num_steps_per_env` 4・
  学習率などは v7 のまま。観測の次元も v7 と同じ (`Knot` で 197) なので、v7 のチェックポイントから
  `--resume` で暖機開始できる。

変換の組は Hydra で比べられる (既定は 4 倍):
`agent.algorithm.symmetry_cfg.data_augmentation_func=isaac_twisted_rl.tasks.knot_tying.common.mdp.symmetry:augment_mirror`
(`augment_reverse` も同様)。
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlSymmetryCfg

from ....common.mdp.symmetry import augment_mirror_reverse
from ...v7.agents.rsl_rl_ppo_cfg import KnotTyingPPORunnerCfg as _V7MLP

_ALGORITHM = _V7MLP().algorithm.replace(
    symmetry_cfg=RslRlSymmetryCfg(
        use_data_augmentation=True,
        use_mirror_loss=False,
        data_augmentation_func=augment_mirror_reverse,
    )
)
"""v7 の PPO 本体の設定に `symmetry_cfg` だけを足したもの。"""


@configclass
class KnotTyingPPORunnerCfg(_V7MLP):
    """v8 既定: 段ごとの MLP 方策 + 対称性のデータ拡張 (`Isaac-Twisted-RL-Knot-v8`)。"""

    experiment_name = "knot_tying_v8"
    algorithm = _ALGORITHM
