"""v17: v15 + 待ちのフェーズ (close・hold・release・settle_out・relax) の尺を詰める (学習を速くする。物理は変えない)。

    uv run train --task Isaac-Twisted-RL-Knot-v17 --headless
    uv run play  --task Isaac-Twisted-RL-Knot-Play-v17 --num_envs 16

仕様と理由は `docs/decisions/0030-v17-shorter-wait-phases.md`、要約は `README.md`。ここは配線だけを書く。

v15 の設定クラスを継承して、`durations` のフィールドだけを差し替える (1 版 1 変数)。`__post_init__` は書かない。
基底の `__post_init__` が `durations` から `decimation` (591)・`episode_length_s`・`render_interval`・報酬の重みの
`step_dt` のスケールを計算し、`make_preset_cfgs` が `actions.pick_place.durations` に渡す。Hydra の上書きで尺を
変えると、これらが追随しない (0030 の「検討して採らなかった案」)。
"""

from __future__ import annotations

from isaaclab.utils import configclass

from ...common.mdp import phases
from ...common.rope_task_cfg import make_preset_cfgs
from ..v15.env_cfg import KnotTyingV15EnvCfg
from ..v15.env_cfg import play_post_init as _v15_play_post_init

__all__ = ["V17_DURATIONS", "KnotTyingV17EnvCfg", "play_post_init"]

V17_DURATIONS = phases.PhaseDurations(
    approach=0.0,
    descend=0.3,
    close=0.2,
    hold=0.1,
    lift=0.3,
    carry=0.6,
    lower=0.3,
    release=0.15,
    settle_out=0.15,
    clear=0.0,
    retreat=0.105,
    relax=0.25,
)
"""v17 の 1 手の尺 (合計 2.4625 s = 591 物理ステップ @240 Hz。v3〜v16 の `KNOT_DURATIONS` は 3.1625 s = 759)。

待ちのフェーズだけを v3 の値 (close 0.3・hold 0.2・release 0.25・settle_out 0.3・relax 0.5) から詰めた。移動系
(descend・lift・carry・lower・retreat) は v3 のまま: 運ぶ速さが発散の原因になる (v3 の `KNOT_DURATIONS` の docstring)。
v14 の方策をこの尺で chain 評価すると、固結びは 2 シード平均で −0.3 ポイント、発散は増えなかった
(`docs/experiments/2026-10-16_physics-speed.md` の D4)。全項目を書くのは、`PhaseDurations` の既定値
(v3 より移動系が短い) に依らず、表と照らせるようにするため。
"""


@configclass
class KnotTyingV17EnvCfg(KnotTyingV15EnvCfg):
    """v17 の環境設定。v15 との差は 1 手の尺 (`durations`) だけ。"""

    durations: phases.PhaseDurations = V17_DURATIONS


def play_post_init(cfg: KnotTyingV17EnvCfg) -> None:
    """Play (chain モード): v15 と同じ設定にし、18 手の時間切れが 19 手目にずれないよう尺を半手ぶん引く。

    Isaac Lab の手数の上限は `ceil(episode_length_s / (sim.dt * decimation))` で、v4 の Play は
    `episode_length_s = 18 * decimation * sim.dt` にする。decimation 591 ではこの商が浮動小数の誤差で 18 をわずかに
    超え、上限が 19 手になる (759 では 18 ちょうど)。chain 評価の定義 (18 手) を版の間で揃えるため、`demos/divergence_bench.py`・
    `bench_step_time.py` と同じく 17.5 手ぶんにする。学習 (6 手) は 591 でも 6 ちょうどになるので基底の計算のまま
    (`tests/test_v17_cfg.py` が両方を検査する)。
    """
    _v15_play_post_init(cfg)
    cfg.episode_length_s = (cfg.actions_per_episode - 0.5) * cfg.decimation * cfg.sim.dt


globals().update(make_preset_cfgs(KnotTyingV17EnvCfg, __name__, play_post_init=play_post_init))
