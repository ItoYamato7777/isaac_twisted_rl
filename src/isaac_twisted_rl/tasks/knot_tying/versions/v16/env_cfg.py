"""v16: v15 + ロープの直径・長さを env ごとに振る DR (寸法の DR)。

    uv run train --task Isaac-Twisted-RL-Knot-v16 --headless
    uv run play  --task Isaac-Twisted-RL-Knot-Play-v16 --num_envs 16

仕様と理由は `docs/decisions/0029-v16-rope-size-dr.md`、要約は `README.md`。ここは配線だけを書く。

v15 の設定クラスを継承して、次の 3 つを足す (1 版 1 変数: 寸法の DR とそれを critic に知らせる観測だけ):

1. `rope_size_scales = ROPE_SIZE_SCALES`: 寸法の組を env に順に割り当てる (`rope_task_cfg.apply_rope_sizes`。ロープの
   spawn が `RopeSizeSpawnerCfg` になり、`scene.replicate_physics` が偽になる)。寸法は run の間変わらない。
2. critic の特権観測のグループ `rope_params` に `rope_size` (`mdp.rope_size_params`、2 次元) を足す (6 -> 8 次元)。
3. 起動時の検算 `events.rope_sizes` (`mdp.check_rope_sizes`): spawn された寸法が割り当てと一致するか。

行動の範囲・床に落ちた判定・アーカイブの保存と復元は、ロープの spawn 設定から env ごとの寸法を読む (`mdp.get_rope_sizes`)
ので、ここでは何も書かない。物理の DR (6 軸)・段 0 の保存の規則・時間切れの項・PPO は v15 のまま。

Play (chain 評価) は v15 と同じ設定に、寸法を公称の 1 組にする設定を足す (v15 と同じロープで比べるため)。寸法を振った
評価は Hydra の `env.scene.rope.spawn.scales=[[0.8,0.4],...]` で行う。
"""

from __future__ import annotations

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass

from ...common import mdp
from ...common.rope_model.rope_specs import SIZE_DIAMETER_SCALES, SIZE_LENGTH_SCALES
from ...common.rope_task_cfg import make_preset_cfgs
from ..v14.env_cfg import EventCfg as V14EventCfg
from ..v14.env_cfg import ObservationsCfg as V14ObservationsCfg
from ..v15.env_cfg import KnotTyingV15EnvCfg
from ..v15.env_cfg import play_post_init as _v15_play_post_init

__all__ = ["NOMINAL_SIZE", "ROPE_SIZE_SCALES", "KnotTyingV16EnvCfg", "play_post_init"]

ROPE_SIZE_SCALES = tuple((ls, ds) for ls in SIZE_LENGTH_SCALES for ds in SIZE_DIAMETER_SCALES)
"""学習で env に割り当てる (長さの倍率, 直径の倍率) の組。長さ ×0.8/1.0/1.25 × 直径 ×0.4/0.63/1.0 の 9 組 (0029 の判断 1)。

物理のゲート (skill `physics-check`) で通らない組は、ここから外す。4096 env なら 1 組あたり 455〜456 env。
"""

NOMINAL_SIZE = ((1.0, 1.0),)
"""Play の寸法 (公称の 1 組。v15 と同じ `knot` の USD)。"""


@configclass
class EventCfg(V14EventCfg):
    """イベント。v14 (= v15) の並びの後ろに、寸法の割り当ての検算 (startup) を足す。"""

    rope_sizes = EventTerm(func=mdp.check_rope_sizes, mode="startup")


@configclass
class ObservationsCfg(V14ObservationsCfg):
    """観測。v14 (= v15) の critic の特権観測のグループ `rope_params` の後ろに、寸法の 2 次元を足す。"""

    @configclass
    class RopeParamsCfg(V14ObservationsCfg.RopeParamsCfg):
        """物理の倍率 6 次元 (`domain_rand.AXES` の順) + 寸法の倍率 2 次元 (`domain_rand.SIZE_AXES` の順)。どれも [−1, 1]。"""

        rope_size = ObsTerm(func=mdp.rope_size_params)

    rope_params: RopeParamsCfg = RopeParamsCfg()


@configclass
class KnotTyingV16EnvCfg(KnotTyingV15EnvCfg):
    """v16 の環境設定。v15 との差は寸法の割り当て・critic の特権観測の 2 次元・起動時の検算だけ。"""

    observations: ObservationsCfg = ObservationsCfg()
    events: EventCfg = EventCfg()
    rope_size_scales: tuple[tuple[float, float], ...] | None = ROPE_SIZE_SCALES


def play_post_init(cfg: KnotTyingV16EnvCfg) -> None:
    """Play (chain モード): v15 と同じ設定に、寸法を公称の 1 組にする設定だけを足す (v15 と同じロープで評価する)。"""
    _v15_play_post_init(cfg)
    cfg.scene.rope.spawn.scales = NOMINAL_SIZE


globals().update(make_preset_cfgs(KnotTyingV16EnvCfg, __name__, play_post_init=play_post_init))
