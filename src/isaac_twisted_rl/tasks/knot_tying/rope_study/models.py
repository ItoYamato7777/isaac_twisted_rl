"""比較する紐モデルの一覧。isaaclab を import しない (タスク登録の `__init__.py` から Kit 起動前に読まれる)。

どのモデルも **同じ紐** (直径 20 mm、全長 1.14 m、線密度 0.25 kg/m、曲げ剛性 EI = 7.85e-4 N m^2、
ねじれ剛性 GJ = EI / 2) を表すように材料定数を揃え、表現 (離散化・ソルバ・接触の解き方) と摩擦だけを
変える。基準は v3〜v7 の学習に使ってきた `Knot` プリセット (= `Art28`)。

| 名前 | 表現 | 何を確かめるか |
|---|---|---|
| `Art28` | PhysX Articulation、D6 関節のカプセル 28 本 (4 cm) | 基準。v4 と同じ |
| `Art28LowMu` | 同上、摩擦を実物の範囲 (紐-紐・紐-床 0.5、指 1.0) へ | 摩擦 2.0 という非現実的な値への依存 |
| `Art56` | PhysX Articulation、カプセル 56 本 (2 cm)、1 本おきの接触を除外 | 剛体鎖の離散化の粗さ |
| `XpbdC28` | Warp の XPBD Cosserat 棒、28 区間 | 連続体の棒モデル (伸び・せん断・曲げ・ねじれ)、compliance 付きの指 |
| `XpbdC56` | 同上、56 区間 | 棒モデルの離散化の粗さ |
| `XpbdC28LowMu` | XpbdC28、摩擦を実物の範囲へ | 同上 |
| `XpbdMS28` | Warp の XPBD 質点-ばね (SoftGym / FleX 型)、28 区間 | ねじれと正しい曲げ剛性の無い単純な粒子モデル |
| `Fem` | PhysX の FEM 変形体 (四面体) | Isaac Lab 標準の連続体の変形体 |

摩擦の既定値 (紐-床・紐-紐 2.0、指 4.0) は `rope_model/rope_specs.py` の `ROPE_FRICTION` /
`GROUND_FRICTION` と、`mdp/events.py` の指の摩擦材 (max 合成) に合わせてある。
"""

from __future__ import annotations

from dataclasses import dataclass, field

EI = 7.854e-4
"""曲げ剛性 [N m^2] (`Knot`: E = 1e5 Pa, I = pi r^4 / 4, r = 0.01 m)。"""

GJ = 0.5 * EI
"""ねじれ剛性 [N m^2] (`rope_specs.TORSION_TO_BENDING_RATIO` = 0.5)。"""

LINEAR_DENSITY = 0.2513
"""線密度 [kg/m] (`Knot` のリンク 0.01005 kg / 間隔 0.04 m)。"""

CENTERLINE_LENGTH = 1.12
"""中心線の長さ [m] (`Knot` の節点列 28 x 0.04 m)。"""

LOW_MU = {"rope": 0.5, "ground": 0.5, "finger": 1.0}
"""実物の範囲の摩擦。綿・ポリエステルの紐どうしで 0.3〜0.8、木やラミネートの机の上で 0.3〜0.6、
ゴムの指先と紐で 0.8〜1.2 程度 (報告書の文献表)。"""


@dataclass(frozen=True)
class RopeModel:
    """1 つの紐モデル。`env_cfg._apply_model` がこれを読んで環境を組み立てる。"""

    name: str
    description: str
    kind: str
    """`common/rope_backends/backend_cfg.RopeBackendCfg.kind`。"""

    preset: str | None = None
    """剛体鎖の `rope_specs` のプリセット名。"""

    rope_friction: float | None = None
    """紐の摩擦の上書き (剛体鎖: 起動時に PhysX の材料を書き換える)。None なら USD の値 (2.0)。"""

    ground_friction: float = 2.0
    finger_friction: float = 4.0
    xpbd: dict = field(default_factory=dict)
    fem: dict = field(default_factory=dict)
    archive_capacity: int = 16384


XPBD_FINGER_V2 = {
    "finger_rigid_outside": True,
    "finger_x_faces": True,
    "finger_x_face_margin": 0.001,
    "hand_teleport_distance": 0.02,
}
"""XPBD の指の接触 v2 (2026-10-02 22 時、`xpbd_rod.XpbdRodParams` の各項の docstring)。

v1 (それまでの既定、全部の面が compliance で、挟む面と底面だけ) では、床の紐の上に降ろした指の中に紐が深く入った
まま残り、μ = 4 の摩擦で手に付いてきた (台本の遅い尺で固結び 3/128)。v2 は「挟む力」の分だけ compliance にして
残りを剛に (押し戻しの速さは 1 m/s まで)、指の前後の面にも接触を持たせ、1 物理ステップで 2 cm より大きく動いた
手は瞬間移動として扱う (PhysX のキネマティックな手と同じ)。2026-10-02 の物理試験 (`run_physics_suite.sh` の
最初の回) は v1 で回したもの。v1 に戻すには、これらを False / 0 に上書きする
(`env.rope_backend.xpbd_params.finger_rigid_outside=False` など)。
"""


def _xpbd(model: str, num_segments: int, **kw) -> dict:
    params = {
        "model": model,
        "num_segments": num_segments,
        "length": CENTERLINE_LENGTH,
        "radius": 0.01,
        "linear_density": LINEAR_DENSITY,
        "bend_stiffness": EI,
        "twist_stiffness": GJ,
        **XPBD_FINGER_V2,
    }
    params.update(kw)
    return params


MODELS: dict[str, RopeModel] = {
    m.name: m
    for m in (
        RopeModel("Art28", "PhysX Articulation, カプセル 28 本 (基準 = Knot)", "articulation", preset="knot"),
        RopeModel(
            "Art28LowMu",
            "Art28 で摩擦を実物の範囲に (紐 0.5, 床 0.5, 指 1.0)",
            "articulation",
            preset="knot",
            rope_friction=LOW_MU["rope"],
            ground_friction=LOW_MU["ground"],
            finger_friction=LOW_MU["finger"],
        ),
        RopeModel("Art56", "PhysX Articulation, カプセル 56 本 (2 cm)", "articulation_resampled", preset="knot56"),
        RopeModel("XpbdC28", "Warp XPBD Cosserat 棒, 28 区間", "xpbd", xpbd=_xpbd("cosserat", 28)),
        RopeModel("XpbdC56", "Warp XPBD Cosserat 棒, 56 区間", "xpbd", xpbd=_xpbd("cosserat", 56)),
        RopeModel(
            "XpbdC28LowMu",
            "XpbdC28 で摩擦を実物の範囲に",
            "xpbd",
            ground_friction=LOW_MU["ground"],
            finger_friction=LOW_MU["finger"],
            xpbd=_xpbd(
                "cosserat",
                28,
                friction_ground=LOW_MU["ground"],
                friction_self=LOW_MU["rope"],
                friction_finger=LOW_MU["finger"],
            ),
        ),
        RopeModel("XpbdMS28", "Warp XPBD 質点-ばね (SoftGym/FleX 型), 28 区間", "xpbd", xpbd=_xpbd("mass_spring", 28)),
        RopeModel("Fem", "PhysX FEM 変形体 (四面体)", "fem", fem={}, archive_capacity=4096),
    )
}

MODEL_NAMES = tuple(MODELS)
