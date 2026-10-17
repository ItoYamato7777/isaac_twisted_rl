"""v16 の PPO ハイパーパラメータ。**v15 と同じで、ログの置き場所だけを変える。**

critic の観測は v14 と同じ並び `["rope_params", "policy"]` のまま。v16 は観測のグループ `rope_params` に寸法の 2 次元を
足すので (`env_cfg.py`)、critic の入力は 6 + 2 + 197 = 205 次元になる。`StageGatedActorCritic` は最後の 4 列 (段の
one-hot) で頭を選ぶので、特権観測が前に並ぶ形は変わらない。critic の入力の幅が変わるので、v14・v15 のチェックポイント
からは再開できない (0 から学習する。`docs/decisions/0029-v16-rope-size-dr.md` の判断 3)。
"""

from isaaclab.utils import configclass

from ...v15.agents.rsl_rl_ppo_cfg import KnotTyingPPORunnerCfg as _V15MLP


@configclass
class KnotTyingPPORunnerCfg(_V15MLP):
    """v16 既定: v15 と同じ (`Isaac-Twisted-RL-Knot-v16`)。"""

    experiment_name = "knot_tying_v16"
