"""位相ラダーの集計ログ (TensorBoard) と、Play 時の p-data 表示。v2-R2 で追加。

## なぜ「重み 0 の報酬項」を使わないのか

Isaac Lab の `RewardManager.compute()` は **重み 0 の項を呼びすらしない**
(`isaaclab/managers/reward_manager.py` の "skip if weight is zero")。
v2 の `diverged_class0..3` は重み 0 で登録してあるので、実際には一度も
計算されておらず、TensorBoard にも `Episode_Reward/diverged_class*` は
出ていない (2026-09-24 の v2 run で確認)。

そこで 2 つの項に分ける:

* `TopologyStats` (**報酬項・重み 1.0・常に 0 を返す**): 毎 env ステップ
  (= 1 手ごと) に呼ばれることだけを利用して、統計を内部に積む。
  返り値が 0 なので学習には一切寄与しない。`Episode_Reward/topology_stats`
  は常に 0 と表示されるが、それで正しい。
* `log_topology_stats` (**カリキュラム項**): `_reset_idx` の中で呼ばれ、
  戻り値の dict が `CurriculumManager.reset()` 経由で
  `extras["log"]["Curriculum/<項名>/<key>"]` になる。RSL-RL はこれを
  そのまま TensorBoard に書く。呼ばれるたびに積算をリセットするので、
  値は「前回のリセットから今回までの 1 手あたりの率」。

## 記録する量 (すべて 1 手あたり・全環境平均)

| key | 意味 |
|---|---|
| `raw_c2` / `r2_c2` | R2 除去前 / 後で class 2 (絡み 2 交差) |
| `fake_c2` | 除去前は class 2 だが除去後は class 1 以下 (掛け渡しなどの偽の C2) |
| `raw_c3` / `r2_c3` / `fake_c3` | 同じく class 3 (固結び) |
| `r2_removed` | R2 で消えた交差数の平均 |
| `r2_cls0..3` | R2 除去後のクラス分布 |
| `raw_count` / `r2_count` | 除去前 / 後の交差数の平均 |
| `div_cls0..3` | **手を始めた時点の** R2 除去後クラス別の発散率 |
| `r2_trunc` | R2 除去後に gap ≥ 2 の交差が 8 個を超え、部分列探索で漏れが出うる率 (2026-09-28 追加) |

`fake_c2` が大きく `r2_c2` が小さいなら、v2 と同じく「輪に通さず上から
掛け渡す」方策に寄っているということ。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg

from .. import topology as tp
from .observations import get_extractor
from .terminations import diverged_now

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


_STAT_KEYS = (
    "raw_c2",
    "r2_c2",
    "fake_c2",
    "raw_c3",
    "r2_c3",
    "fake_c3",
    "r2_removed",
    "raw_count",
    "r2_count",
    *(f"r2_cls{k}" for k in range(tp.NUM_CLASSES)),
    *(f"div_cls{k}" for k in range(tp.NUM_CLASSES)),
    "r2_trunc",
)


class TopologyStats(ManagerTermBase):
    """位相ラダーの統計を積む報酬項。**常に 0 を返す** (学習に寄与しない)。

    重みは 0 にしないこと (0 だと `RewardManager` が呼ばない)。1.0 を推奨。

    params:
        action_name: `PickPlaceAction` の項名。
        rope_cfg: ロープ。
        print_envs: 0 より大きければ、先頭からこの数の環境について
            毎手 p-data (R2 除去前 / 後) とクラスを標準出力へ出す。
            **Play 専用**。記号層を回すので学習では 0 のままにすること。
        diverged_term: `div_cls*` の発散の判定に使う終了項の名前。None なら
            NaN だけを発散と数える。指定すると終了判定と同じ基準
            (有限のまま吹き飛んだロープも含む) になる (`terminations.diverged_now`)。
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._sums = {k: 0.0 for k in _STAT_KEYS}
        self._samples = 0

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        # 積算は `log_topology_stats` が読んだときに捨てる。エピソード単位では消さない。
        pass

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        action_name: str = "pick_place",
        rope_cfg: SceneEntityCfg = SceneEntityCfg("rope"),
        print_envs: int = 0,
        diverged_term: str | None = None,
    ) -> torch.Tensor:
        extractor = get_extractor(env, rope_cfg.name)
        raw = extractor.batch
        red = extractor.reduced_batch
        raw_cls = tp.topology_class(raw)
        r2_cls = tp.topology_class(red)

        term = env.action_manager.get_term(action_name)
        cls_start = term.class_r2_at_action_start
        if diverged_term is None:
            diverged = ~torch.isfinite(extractor.nodes).all(dim=-1).all(dim=-1)
        else:
            diverged = diverged_now(env, diverged_term)

        c2, c3 = tp.CLASS_INTERLEAVED2, tp.CLASS_OVERHAND
        values = {
            "raw_c2": raw_cls == c2,
            "r2_c2": r2_cls == c2,
            "fake_c2": (raw_cls == c2) & (r2_cls < c2),
            "raw_c3": raw_cls == c3,
            "r2_c3": r2_cls == c3,
            "fake_c3": (raw_cls == c3) & (r2_cls < c3),
            "r2_removed": tp.r2_removed_count(raw, red),
            "raw_count": raw.count,
            "r2_count": red.count,
            "r2_trunc": tp.subsequence_truncated(red),
        }
        for k in range(tp.NUM_CLASSES):
            values[f"r2_cls{k}"] = r2_cls == k
            values[f"div_cls{k}"] = (cls_start == k) & diverged
        # 1 回の転送でまとめて CPU へ下ろす (項目ごとに .item() すると同期が 17 回)。
        stacked = torch.stack([values[k].float().mean() for k in _STAT_KEYS]).tolist()
        for k, v in zip(_STAT_KEYS, stacked):
            self._sums[k] += v
        self._samples += 1

        if print_envs > 0:
            self._print(env, extractor, raw_cls, r2_cls, print_envs)

        return torch.zeros(env.num_envs, device=env.device)

    def _print(self, env, extractor, raw_cls, r2_cls, print_envs: int) -> None:
        """Play 用: 先頭 `print_envs` 環境の p-data を 1 手ごとに出す。"""
        n = min(print_envs, env.num_envs)
        raw_pd = extractor.p_data(short=True)
        r2_pd = extractor.p_data_r2(short=True)
        steps = env.episode_length_buf[:n].tolist()
        raw_c = raw_cls[:n].tolist()
        r2_c = r2_cls[:n].tolist()
        for i in range(n):
            mark = "  <- R2で降格" if r2_c[i] < raw_c[i] else ""
            print(
                f"[topology] env={i} step={steps[i]} "
                f"raw(class={raw_c[i]}) {raw_pd[i] or '-'} | "
                f"R2後(class={r2_c[i]}) {r2_pd[i] or '-'}{mark}",
                flush=True,
            )

    def pop_means(self) -> dict[str, float] | None:
        """積算の平均を返してリセットする。まだ 1 手も積んでいなければ None。"""
        if self._samples == 0:
            return None
        out = {k: v / self._samples for k, v in self._sums.items()}
        self._sums = {k: 0.0 for k in _STAT_KEYS}
        self._samples = 0
        return out


def log_topology_stats(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term: str = "topology_stats",
) -> dict[str, float] | None:
    """カリキュラム項: `TopologyStats` の積算を TensorBoard へ流す。

    何も変更しない (カリキュラムとしては no-op)。戻り値の dict が
    `Curriculum/<この項の名前>/<key>` としてログされる。
    """
    stats = env.reward_manager.get_term_cfg(reward_term).func
    if not isinstance(stats, TopologyStats):
        raise TypeError(f"reward term '{reward_term}' is not TopologyStats (got {type(stats).__name__})")
    return stats.pop_means()
