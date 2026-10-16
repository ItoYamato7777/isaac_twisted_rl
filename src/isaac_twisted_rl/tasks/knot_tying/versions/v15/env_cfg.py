"""v15: v14 + 段の変化・発散と同じ手では時間切れを立てない (時間切れの bootstrap の不具合の修正)。

    uv run train --task Isaac-Twisted-RL-Knot-v15 --headless
    uv run play  --task Isaac-Twisted-RL-Knot-Play-v15 --num_envs 16

仕様と理由は `docs/decisions/0028-v15-timeout-bootstrap-fix.md`、要約は `README.md`。ここは配線だけを書く。

v14 の設定クラスを継承して、終了項 `time_out` を `mdp.time_out_unless_terminated` に差し替える (1 版 1 変数)。
既存のフィールドへの代入なので、終了項の並び (`time_out`・`diverged`・`stage_up`・`stage_down`) は変わらない。
名前を `time_out` のまま残すのは、`ResetToStage` の `outcome_terms` と `Episode_Termination/time_out` のログが
名前で引くため。

Play (chain 評価) は v14 と同じ設定に戻す (`mdp.time_out`)。chain 評価の定義 (18 手目に固結びになった手は
`knot` と `time_out` の両方に数える) を v14 と揃え、評価の値を版の間で比べられるようにするため。
"""

from __future__ import annotations

from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from ...common import mdp
from ...common.rope_task_cfg import make_preset_cfgs
from ..v14.env_cfg import KnotTyingV14EnvCfg
from ..v14.env_cfg import play_post_init as _v14_play_post_init

__all__ = ["EXCLUDE_TERMS", "KnotTyingV15EnvCfg", "play_post_init"]

EXCLUDE_TERMS = ("diverged", "stage_up", "stage_down")
"""同じ手で立てば時間切れを立てない終了項 (v4 の `TerminationsCfg` の `time_out` 以外の全項)。"""


@configclass
class KnotTyingV15EnvCfg(KnotTyingV14EnvCfg):
    """v15 の環境設定。v14 との差は終了項 `time_out` の関数だけ。"""

    def __post_init__(self):
        super().__post_init__()
        self.terminations.time_out = DoneTerm(
            func=mdp.time_out_unless_terminated, time_out=True, params={"exclude_terms": EXCLUDE_TERMS}
        )


def play_post_init(cfg: KnotTyingV15EnvCfg) -> None:
    """Play (chain モード): v14 と同じ設定にし、時間切れを v14 の `mdp.time_out` に戻す。"""
    _v14_play_post_init(cfg)
    cfg.terminations.time_out = DoneTerm(func=mdp.time_out, time_out=True)


globals().update(make_preset_cfgs(KnotTyingV15EnvCfg, __name__, play_post_init=play_post_init))
