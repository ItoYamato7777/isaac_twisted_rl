"""ロープの形状 (3D 点列) から位相状態 p-data を作る純粋ロジック層。

twisted_rl の `mujoco_infra/mujoco_utils/topology/` に対応する移植。
**このパッケージは `isaaclab` / `omni` / `pxr` を一切 import しない**ので、
Isaac Sim を起動せずに `pytest` 単体で検証できる。Isaac Lab の
`Articulation` との接続は上位の `rope_state.py` が担当する。

## パイプライン

    リンク中心 (num_envs, num_links, 3)          Articulation.data.body_pos_w
      |  polyline.polyline_from_link_centers
      v
    ポリライン (num_envs, num_links+1, 3)
      |  intersections.segment_intersections     ... テンソル層 (バッチ / GPU)
      v
    IntersectionBatch (交差 + 交点の出現順)
      |  state_2_topology.p_data_from_batch      ... 記号層 (env ごとの Python)
      v
    p-data 文字列 "1: U 2 +\\n2: O 1 +"

観測に入れるだけなら `IntersectionBatch` で止めてよい (文字列は不要)。

## 使い方

    >>> import torch
    >>> from ...topology import polyline_to_p_data
    >>> pts = torch.tensor([[0., 0., 1.], [2., 0., 1.], [2., 2., 0.], [1., -1., 0.]])
    >>> polyline_to_p_data(pts, short=True)
    '1O2-_2U1-'

## モジュール構成

    representation.py   Point / Edge / Face / AbstractState (半辺構造と p-data)
    intersections.py    バッチ化 2D 自己交差検出 (テンソル層)
    state_2_topology.py 交差リスト -> AbstractState -> p-data (記号層)
    polyline.py         ポリラインの前処理 (節点生成・中心寄せ・細分・再標本化)
    knot_detect.py      固結びの判定と報酬ラダーの段・R2 除去 (テンソル層のみ)
    knot_graph.py       R1/R2/cross の位相グラフと固結びまでの BFS 距離 (v3、交差 4 個まで) と段 (v4)
"""

from .intersections import (
    DEFAULT_EPS,
    PAD,
    IntersectionBatch,
    crossing_positions,
    has_duplicate_segments,
    segment_intersections,
)
from .knot_detect import (
    CLASS_INTERLEAVED2,
    CLASS_OVERHAND,
    CLASS_SINGLE,
    CLASS_TRIVIAL,
    MAX_CROSSINGS_CONSIDERED,
    NUM_CLASSES,
    class_one_hot,
    has_knot_subsequence,
    is_interleaved_two,
    is_overhand,
    partner_gaps,
    r2_removed_count,
    reduce_r2,
    sign_consistent,
    subsequence_truncated,
    topology_class,
)
from .knot_graph import (
    OVERHAND_P_DATA,
    STAGE_GOAL,
    BfsPotential,
    batch_keys,
    distance_to_overhand,
    enumerate_graph,
    get_bfs_potential,
    p_data_key,
    p_data_to_lists,
    potential_table,
)
from .polyline import (
    center_xy,
    polyline_from_link_centers,
    resample_polyline,
    subdivide_segments,
)
from .representation import AbstractState, Edge, Face, Point
from .state_2_topology import (
    batch_p_data,
    batch_polyline_to_topology,
    intersections_to_topology,
    p_data_from_batch,
    polyline_to_p_data,
    polyline_to_topology,
    topology_from_batch,
)

__all__ = [
    "AbstractState",
    "BfsPotential",
    "OVERHAND_P_DATA",
    "STAGE_GOAL",
    "CLASS_INTERLEAVED2",
    "CLASS_OVERHAND",
    "CLASS_SINGLE",
    "CLASS_TRIVIAL",
    "DEFAULT_EPS",
    "Edge",
    "Face",
    "IntersectionBatch",
    "MAX_CROSSINGS_CONSIDERED",
    "NUM_CLASSES",
    "PAD",
    "Point",
    "batch_keys",
    "batch_p_data",
    "batch_polyline_to_topology",
    "center_xy",
    "class_one_hot",
    "crossing_positions",
    "distance_to_overhand",
    "enumerate_graph",
    "get_bfs_potential",
    "has_duplicate_segments",
    "has_knot_subsequence",
    "intersections_to_topology",
    "is_interleaved_two",
    "is_overhand",
    "p_data_from_batch",
    "p_data_key",
    "p_data_to_lists",
    "partner_gaps",
    "polyline_from_link_centers",
    "polyline_to_p_data",
    "polyline_to_topology",
    "potential_table",
    "r2_removed_count",
    "reduce_r2",
    "resample_polyline",
    "segment_intersections",
    "sign_consistent",
    "subdivide_segments",
    "subsequence_truncated",
    "topology_class",
    "topology_from_batch",
]
