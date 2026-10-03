"""紐のバックエンドの設定 (`RopeStudyEnv` が読む)。紐モデルの比較 (`rope_study/`) で追加。"""

from __future__ import annotations

from isaaclab.utils import configclass


@configclass
class RopeBackendCfg:
    """`env.scene["rope"]` に何を置くか。

    | kind | 紐の実体 | `scene["rope"]` |
    |---|---|---|
    | `articulation` | `scene.rope` (PhysX Articulation) | そのもの (アダプタなし。v0〜v7 と同じ) |
    | `articulation_resampled` | `scene.rope_native` (リンク数が仮想リンクと違う Articulation) | `ResampledArticulationRope` |
    | `xpbd` | Warp の `XpbdRod` (USD に紐は無い) | `XpbdRope` |
    | `fem` | `scene.rope_fem` (PhysX の FEM 変形体) | `FemRope` |
    """

    kind: str = "articulation"
    num_virtual_links: int = 28
    """観測・行動・位相に使う仮想リンクの数。`Knot` のリンク数と同じ 28。"""

    native_name: str = "rope_native"
    """`articulation_resampled` / `fem` で、本物の紐の資産の名前。"""

    radius: float = 0.01
    """紐の半径 [m] (めり込みの計測と XPBD の描画に使う)。"""

    init_height: float = 0.06
    """XPBD の初期姿勢の高さ [m] (`rope_task_cfg._ROPE_INIT_Z` と同じ。落として落ち着かせる)。"""

    xpbd_params: dict = {}
    """`xpbd_rod.XpbdRodParams` の引数 (kind=`xpbd` のとき)。"""
