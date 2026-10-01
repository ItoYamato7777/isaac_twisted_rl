"""v6 の PPO ハイパーパラメータ。**v5 (= v4) と同じで、ログの置き場所だけを変える。**

方策は v5 と同じ段ごとの MLP (`StageGatedActorCritic`)。段の one-hot は観測の末尾 4 列のまま
(`../env_cfg.py` の `ObservationsCfg`)。

観測の次元が v5 (`Knot` で 133) から 64 増えて 197 になるので、v5 のチェックポイントからは
`--resume` で暖機開始できない (`load_state_dict` は strict で、入力層の形が合わない)。

MLP 版だけを登録する (v4 / v5 と同じ。段ごとに振り分ける方策は再帰版を用意していない)。
"""

from isaaclab.utils import configclass

from ...v5.agents.rsl_rl_ppo_cfg import KnotTyingPPORunnerCfg as _V5MLP


@configclass
class KnotTyingPPORunnerCfg(_V5MLP):
    """v6 既定: 段ごとの MLP 方策 (`Isaac-Twisted-RL-Knot-v6`)。"""

    experiment_name = "knot_tying_v6"
