"""ロープの「作り方」(構成レシピ) の定義。

同じ D6 ジョイント連鎖でも、曲げ剛性を入れるか / 捩りを解放するか /
どこまで細かく離散化するか で挙動は大きく変わる。ここではそれらを
`RopeSpec` という 1 つのデータ構造にまとめ、名前付きプリセットとして
`ROPE_SPECS` に登録する。

    generate_rope_usd.py --rope <name>   ... USD を生成
    demos/rope_catch.py   --rope <name>   ... そのロープを使ってデモ

このモジュールは `pxr` も `isaaclab` も import しない (純粋な dataclass と
math のみ)。パッケージ自動インポート経路に乗っても軽いままにするため。

## プリセット一覧

| name     | 色   | 狙い                          | 曲げ剛性 | 曲げ角度制限 | 捩り (rotX) | リンク数 | 直径   |
|----------|------|-------------------------------|----------|--------------|-------------|----------|--------|
| `simple` | 青   | 現行モデル (比較用ベースライン) | なし     | なし         | ロック      | 20       | 20 mm  |
| `stiff`  | 緑   | 曲げ剛性を入れた「腰のある」縄 | あり     | ±60°         | ロック      | 20       | 20 mm  |
| `twist`  | 橙   | `stiff` より柔らかい曲げ剛性 (1/10) + 捩り自由度を解放 | あり(弱) | ±60°         | 自由(低剛性) | 20       | 20 mm  |
| `fine`   | 赤   | 結び目が結べる細径・高分解能   | あり     | ±60°         | 自由(低剛性) | 48       | 8 mm   |
| `knot`   | 紫   | `twist` を 28 リンク (1.14 m) に延ばした固結び用 | あり(弱) | ±60° | 自由(低剛性) | 28 | 20 mm |

色は `demos/multi_rope_catch.py` で 4 本を横並びにしたとき、どれがどの
プリセットかを一目で見分けるためのもの。

`simple` → `stiff` → `twist` → `fine` は 1 段ずつ単一の軸だけを変えてあるので、
挙動の差をその変更に帰属させられる (`fine` のみ寸法と分解能を同時に変える)。

## 単位に関する重要な注意

USD の **角度ドライブの stiffness / damping は「度」あたり** で解釈される
(`UsdPhysics.DriveAPI` の仕様。Isaac Lab も `N*m/rad -> N*m/deg` に
`pi/180` を掛けて書き込んでいる: `isaaclab/sim/schemas/schemas.py`)。
本モジュールは一貫して **rad 基準の物理量** で値を保持し、USD へ書き出す
直前に `generate_rope_usd.py` 側で `deg` へ変換する。

`simple` プリセットのみ、既存の `data/rope_simple.usd` とバイト単位で
等価な値を再現するため、この換算を含んだ実効値 (0.01 * 180/pi ≒ 0.573
N*m*s/rad) をそのまま保持している。当時のコード上のコメントは
`0.01 [N*m*s/rad]` だったが、実際に PhysX へ渡っていたのは 57.3 倍の値
だった、という経緯。
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, replace

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# ---------------------------------------------------------------- 共通の材料定数
ROPE_DENSITY = 600.0
"""編組ロープのかさ密度 [kg/m^3]。

素材 (ナイロン ~1140, ポリエステル ~1380) より低いのは、撚り/組みの
隙間に空気を含むため。直径 20 mm で約 0.24 kg/m、直径 8 mm で約 0.043 kg/m
となり、実物のカタログ値と整合する。
"""

ROPE_FRICTION = 2.0
"""ロープ表面の摩擦係数として USD に書く値 (static / dynamic とも同値)。全プリセット共通。

.. warning::
    **この値は PhysX に届いていない。実際のロープの摩擦は 0.5 (PhysX の既定の材質)。**
    `generate_rope_usd.py` は材質を `Scope` の prim に付けて bind するが、physics の bind は
    `Material` の prim でないと解決されない (`ComputeBoundMaterial` が何も返さない)。軸方向の滑りの試験で、
    何も書かないロープと tensor API で 0.5 を書いたロープの止まる距離が一致した
    (`docs/experiments/2026-10-14_friction-binding.md`、`demos/friction_check.py`)。実効摩擦はロープ <-> 床 (0.5 + 2.0) / 2 = 1.25、
    ロープ <-> ロープ 0.5。v0 以降のすべての学習はこの物理で行った。既存の版の挙動を変えないため、USD は直さない。

以下は意図した設計 (上の理由で、実際には成り立っていない):

PhysX は接触する 2 つのマテリアルの摩擦を **合成則** で 1 つにまとめる。
Isaac Lab の既定は ``"average"`` なので、実効摩擦は次のようになる:

    ロープ <-> 床   : (ROPE_FRICTION + GROUND_FRICTION) / 2 = 2.0
    ロープ <-> ロープ: (ROPE_FRICTION + ROPE_FRICTION) / 2   = 2.0

ロープ同士の自己摩擦が上がるのは結び目にとって好都合で、結び目が
締まったまま保持される物理 (キャプスタン効果) が出やすくなる。

指との接触だけは例外で、グリッパー側が ``friction_combine_mode="max"``
の専用マテリアル (係数 4.0) を持つ。PhysX は合成則の優先度が高い方を
採用し ``max`` が最優先なので、指 <-> ロープ は ``max(2.0, 4.0) = 4.0``
のまま変わらない。つまりこの値を上げても **把持力の設計には影響しない**。
"""

GROUND_FRICTION = 2.0
"""ロープを置く床 (ground plane) の摩擦係数。各デモの地面生成で使う。床の材質は PhysX に届いている
(Isaac Lab の `GroundPlaneCfg` が `Material` の prim を作る)。ロープの摩擦が実際は 0.5 なので、ロープ <-> 床 の実効摩擦は
1.25 (`ROPE_FRICTION` の warning)。

以下は意図した設計: `ROPE_FRICTION` と合わせて、ロープ <-> 床 の実効摩擦を 2.0 にする
(従来は両方 1.0 で実効 1.0 だったので 2 倍)。値を大きくすると床の上で
ロープが滑りにくくなり、掴んで動かしたときに引きずられて全体が付いて
くる度合いが強くなる。逆に持ち上げる際は床に残った部分の抵抗が増える。
"""

ROPE_YOUNGS_MODULUS = 1.0e6
"""曲げに関する **実効** ヤング率 [Pa]。

繊維同士がずれるため、引張方向の見かけヤング率 (GPa オーダー) より
3 桁以上小さい。0.1〜10 MPa が実測のおおよその範囲で、その中央付近を採用。
この値だけで「ロープの腰の強さ」が決まるので、実物と合わせる際は
まずここを調整する (後述の片持ち垂れ試験で同定する)。
"""

TWIST_YOUNGS_MODULUS = 1.0e5
"""`twist` プリセット専用の、より柔らかい実効ヤング率 [Pa]。

`stiff`/`fine` は `ROPE_YOUNGS_MODULUS` (1.0e6 Pa) を使うが、`twist` は
グリッパーで挟んで運ぶ用途で使われるため、指を開いたときにロープの腰で
引っかからないよう 1 桁下げて柔らかくしてある。曲げ剛性 EI はヤング率に
比例するので、`ROPE_YOUNGS_MODULUS` の 1/10 で曲げ関節のばね定数も
1/10 になる (角度制限や質量など幾何・慣性由来の量は変わらない)。
"""

TORSION_TO_BENDING_RATIO = 0.5
"""捩り剛性 GJ と曲げ剛性 EI の比。

中実の円形断面なら GJ/EI = 1/(1+nu) ≒ 0.67 だが、編組ロープは捩ると
組みがほどける方向に逃げるため実測はこれより低い。0.3〜0.7 の中央を採用。
"""

BEND_DAMPING_RATIO = 1.0
"""曲げ関節の減衰比 zeta (1.0 = 臨界減衰)。

実ロープはほとんど跳ね返らないので、関節ごとに臨界減衰を与えるのが
見た目にも数値的にも素直。跳ねさせたい場合は 0.3 程度まで下げる。
"""

MIN_BEND_RADIUS_IN_DIAMETERS = 2.0
"""最小曲げ半径を「直径の何倍か」で表したもの。

実ロープはこれ以上きつく曲がらない (曲げると芯が潰れる)。関節の
曲げ角度制限はこの値から導出する。2 倍はきつめの結び目に相当する。
"""


HALF_LENGTH_TO_RADIUS = 2.0
"""カプセル円柱部の半長 / 半径 の比。全プリセットで共通に保つ幾何拘束。

この比を小さくすると、**隣接していないリンク同士が常時接触してしまう**。

リンク間隔は `2 * half_length`、カプセル全長は `2 * half_length + 2 * radius`
なので、1 つ飛ばしのリンク `k` と `k+2` の軸方向の隙間は

    gap = 2 * spacing - capsule_total_length = 2 * (half_length - radius)

一方、接触が生成され始める距離は `2 * contact_offset = 0.8 * radius`
(`RopeSpec.contact_offset` 参照)。比 2.0 のとき

    gap / 接触範囲 = 2 * radius / (0.8 * radius) = 2.5

と余裕があるが、比 1.5 まで落とすと 1.25 になり、ロープがほぼ真っ直ぐな
状態でも k と k+2 が接触判定に入り続ける。すると関節拘束と接触拘束が
競合し、慣性の小さい細いロープでは持ち上げた瞬間に発散する
(実際 `fine` を比 1.5 で作ったときは把持後の lift で必ず NaN になった)。

`k` と `k+1` は関節で直結されているため物理エンジンが自動的に接触除外
するので問題にならない。効いてくるのは常に `k` と `k+2` の組。
"""

_GRASP_SPECIFIC_FORCE = 1500.0
"""摩擦把持時の「リンク質量あたりの押し付け力」[m/s^2]。

既存デモで調整済みの `simple` (指ゲイン 1500 N/m, 半径 0.01 m,
リンク質量 0.01 kg -> 15 N / 0.01 kg) を基準値として採用している。
詳細は `RopeSpec.grasp_finger_stiffness` の docstring 参照。
"""


def _capsule_volume(radius: float, cylinder_height: float) -> float:
    """カプセル (円柱 + 両端の半球) の体積 [m^3]。"""
    return math.pi * radius**2 * cylinder_height + (4.0 / 3.0) * math.pi * radius**3


@dataclass(frozen=True)
class RopeSpec:
    """ロープ 1 本ぶんの構成レシピ。

    ジョイントのゲインは **すべて rad 基準の物理単位** で保持する
    (USD への deg 変換は `generate_rope_usd.py` が行う)。
    """

    name: str
    """プリセット名。USD ファイル名 `rope_<name>.usd` にもなる。"""

    description: str
    """`--rope` のヘルプに出る 1 行説明。"""

    color: tuple[float, float, float]
    """カプセルの表示色 (linear RGB, 0-1)。プリセットごとに変えてある。

    複数のロープを横並びにしたとき (`demos/multi_rope_catch.py`) に、
    どれがどのプリセットかを目視で判別するためのもの。物理には影響しない。
    """

    color_name: str
    """色の日本語名 (ログ表示用)。"""

    num_links: int
    """カプセル剛体リンクの本数。"""

    capsule_radius: float
    """カプセル半径 [m] (= ロープ半径)。"""

    capsule_half_length: float
    """カプセル円柱部の半長 [m]。

    USD の `UsdGeom.Capsule.height` は **円柱部だけ** の長さなので、
    カプセル全長は `2 * capsule_half_length + 2 * capsule_radius` になる。
    リンクの配置間隔は `2 * capsule_half_length` なので、隣接リンクは
    常に直径ぶん (`2 * capsule_radius`) だけ重なる。この重なりが
    曲げたときにリンク間へ隙間が開くのを防ぐ。
    """

    link_mass: float
    """リンク 1 個あたりの質量 [kg]。慣性テンソルは形状から自動導出される。"""

    friction: float
    """ロープ表面の摩擦係数 (static / dynamic とも同値)。"""

    bend_stiffness: float
    """曲げ (rotY/rotZ) ドライブのばね定数 [N*m/rad]。0 なら腰のない紐。"""

    bend_damping: float
    """曲げ (rotY/rotZ) ドライブの粘性減衰 [N*m*s/rad]。"""

    bend_limit_deg: float | None
    """曲げの片側角度制限 [deg]。`None` なら無制限。"""

    twist_stiffness: float | None
    """捩り (rotX) ドライブのばね定数 [N*m/rad]。

    `None` のとき rotX は完全ロックされ、ロープは捩りに対して剛体の棒に
    なる (数値的には最も安定だが、結び目のように捩れが必然的に発生する
    操作は再現できない)。
    """

    twist_damping: float
    """捩り (rotX) ドライブの粘性減衰 [N*m*s/rad]。`twist_stiffness` が
    `None` のときは無視される。"""

    twist_limit_deg: float | None
    """捩りの片側角度制限 [deg]。`None` なら無制限。"""

    filter_neighbor_pairs: int = 1
    """番号の差がこの値以下のリンクどうしの接触を切る (`UsdPhysics.FilteredPairsAPI`)。紐モデルの比較で追加。

    既定の 1 は「関節で直結した隣だけ」(PhysX が自動で除外する) で、従来の 5 プリセットの USD は変わらない。
    リンクを直径より短くした鎖 (`knot56`) では、1 本おきのリンクがまっすぐな状態でも接していて
    (`HALF_LENGTH_TO_RADIUS` の docstring の問題)、関節と接触が競合するので 2 にする。
    """

    # ------------------------------------------------------------ 派生量
    @property
    def link_spacing(self) -> float:
        """隣接リンクの中心間距離 [m]。ジョイントの支点オフセットも同じ値の半分。"""
        return 2.0 * self.capsule_half_length

    @property
    def capsule_total_length(self) -> float:
        """カプセルの全長 (半球のキャップを含む) [m]。"""
        return 2.0 * self.capsule_half_length + 2.0 * self.capsule_radius

    @property
    def rope_length(self) -> float:
        """ロープの全長 (両端のキャップを含む) [m]。"""
        return (self.num_links - 1) * self.link_spacing + self.capsule_total_length

    @property
    def total_mass(self) -> float:
        """ロープ全体の質量 [kg]。"""
        return self.num_links * self.link_mass

    @property
    def linear_density(self) -> float:
        """線密度 [kg/m]。実物のカタログ値と突き合わせるための量。"""
        return self.total_mass / self.rope_length

    @property
    def diameter(self) -> float:
        """ロープ直径 [m]。"""
        return 2.0 * self.capsule_radius

    @property
    def contact_offset(self) -> float:
        """PhysX の contact offset [m]。

        半径に比例させる。PhysX のデフォルト (0.02 m) は半径 4〜10 mm の
        ロープには大きすぎ、逆に小さすぎると高速時に接触を取りこぼす。
        現行 `simple` の 0.004 m / 半径 0.01 m = 0.4 倍を全プリセット共通の
        比率として採用する。
        """
        return 0.4 * self.capsule_radius

    @property
    def grasp_finger_stiffness(self) -> float:
        """このロープを摩擦把持するのに適した指 PD ゲイン [N/m]。

        平行グリッパーのデモは常に全閉を指令するので、ロープを挟むと
        **ロープ半径ぶんの追従誤差**が残り、``押し付け力 = k * 半径`` が
        定常的に出る。ここで k を全プリセット共通の固定値にすると、
        細くて軽いロープほど「質量あたりの押し付け力」が大きくなりすぎる:

            simple: 15 N / 0.0101 kg = 1500 m/s^2
            fine:    6 N / 0.00052 kg = 11500 m/s^2   <- 8 倍。潰れて発散する

        実際 `fine` は固定 k=1500 だと指がロープ半径より深く閉じ込み、
        貫通 -> depenetration による吹き飛びで NaN に至る。そこで
        **質量あたりの押し付け力 (比加速度) を一定** に保つよう k を導出する:

            k * radius / link_mass = _GRASP_SPECIFIC_FORCE   (一定)

        基準値は既存デモで調整済みの `simple` (k=1500, 半径 0.01 m,
        質量 0.01 kg) から取ってあるので、`simple` の挙動は変わらない。
        """
        return _GRASP_SPECIFIC_FORCE * self.link_mass / self.capsule_radius

    @property
    def usd_path(self) -> str:
        """生成される USD ファイルの絶対パス。"""
        return os.path.join(_DATA_DIR, f"rope_{self.name}.usd")

    def summary(self) -> str:
        """人間が読める 1 行サマリ (デモ起動時のログ用)。"""
        twist = "locked" if self.twist_stiffness is None else f"k={self.twist_stiffness:.4g}"
        bend_limit = "none" if self.bend_limit_deg is None else f"+-{self.bend_limit_deg:.0f}deg"
        return (
            f"rope '{self.name}' ({self.color_name}): "
            f"links={self.num_links} dia={self.diameter * 1000:.1f}mm "
            f"len={self.rope_length:.3f}m mass={self.total_mass * 1000:.1f}g "
            f"({self.linear_density:.3f}kg/m) bend(k={self.bend_stiffness:.4g},"
            f"c={self.bend_damping:.4g},{bend_limit}) twist({twist})"
        )


def _make_physical_spec(
    name: str,
    description: str,
    num_links: int,
    capsule_radius: float,
    lock_twist: bool,
    color: tuple[float, float, float],
    color_name: str,
    youngs_modulus: float = ROPE_YOUNGS_MODULUS,
    density: float = ROPE_DENSITY,
    damping_ratio: float = BEND_DAMPING_RATIO,
    friction: float = ROPE_FRICTION,
    twist_limit_deg: float | None = 90.0,
    capsule_half_length: float | None = None,
) -> RopeSpec:
    """材料定数と寸法から関節ゲインを導出して `RopeSpec` を組み立てる。

    「関節のばね定数をいくつにするか」を直接決め打ちするのではなく、
    ロープの実効ヤング率・密度・最小曲げ半径という **実物で測れる量** から
    毎回導出する。こうしておくと寸法や分解能を変えたときにゲインが自動で
    追従し、プリセット間の比較が「同じ材料の別の作り方」として意味を持つ。

    導出:
        断面二次モーメント  I = pi * r^4 / 4
        曲げ剛性            EI = E * I                      [N*m^2]
        関節のばね定数      k_bend = EI / L_seg             [N*m/rad]
        捩り剛性            k_twist = TORSION_RATIO * k_bend
        減衰                c = 2 * zeta * sqrt(k * I_link) [N*m*s/rad]
        曲げ角度制限        theta_max = 2 * asin(L_seg / (2 * R_min))

    `capsule_half_length` を省くと半径から導出する (`HALF_LENGTH_TO_RADIUS`)。寸法の DR (`make_size_variant`) だけが
    長さと直径を別々に振るために与える。
    """
    # 円柱部の半長は半径から導出する。この比は 1 つ飛ばしのリンクが
    # 常時接触してしまうのを避けるための幾何拘束 (HALF_LENGTH_TO_RADIUS 参照)。
    if capsule_half_length is None:
        capsule_half_length = HALF_LENGTH_TO_RADIUS * capsule_radius
    segment_length = 2.0 * capsule_half_length
    total_length = segment_length + 2.0 * capsule_radius

    # -- 質量: 形状の体積 x 密度。リンク数や太さを変えても線密度が保たれる。
    link_mass = density * _capsule_volume(capsule_radius, segment_length)

    # -- 曲げ剛性: 円形断面の EI をセグメント長で割って関節のばね定数にする。
    area_moment = math.pi * capsule_radius**4 / 4.0
    bend_stiffness = youngs_modulus * area_moment / segment_length

    # -- 減衰: リンクの慣性に対する臨界減衰を基準にする。
    #    横曲げ用は長手方向に細長い剛体としての慣性、捩り用は長軸まわりの慣性。
    inertia_transverse = link_mass * (3.0 * capsule_radius**2 + total_length**2) / 12.0
    inertia_axial = 0.5 * link_mass * capsule_radius**2
    bend_damping = 2.0 * damping_ratio * math.sqrt(bend_stiffness * inertia_transverse)

    # -- 曲げ角度制限: 最小曲げ半径 R_min を離散化した 1 関節あたりの角度に直す。
    min_bend_radius = MIN_BEND_RADIUS_IN_DIAMETERS * (2.0 * capsule_radius)
    sin_half = min(segment_length / (2.0 * min_bend_radius), 1.0)
    bend_limit_deg = math.degrees(2.0 * math.asin(sin_half))

    if lock_twist:
        twist_stiffness = None
        twist_damping = 0.0
        twist_limit = None
    else:
        twist_stiffness = TORSION_TO_BENDING_RATIO * bend_stiffness
        twist_damping = 2.0 * damping_ratio * math.sqrt(twist_stiffness * inertia_axial)
        twist_limit = twist_limit_deg

    return RopeSpec(
        name=name,
        description=description,
        color=color,
        color_name=color_name,
        num_links=num_links,
        capsule_radius=capsule_radius,
        capsule_half_length=capsule_half_length,
        link_mass=link_mass,
        friction=friction,
        bend_stiffness=bend_stiffness,
        bend_damping=bend_damping,
        bend_limit_deg=bend_limit_deg,
        twist_stiffness=twist_stiffness,
        twist_damping=twist_damping,
        twist_limit_deg=twist_limit,
    )


# ---------------------------------------------------------------- プリセット定義

_SIMPLE = RopeSpec(
    name="simple",
    description="現行モデル。曲げ剛性なし・角度制限なし・捩りロック (比較用ベースライン)",
    color=(0.10, 0.35, 0.85),
    color_name="青",
    num_links=20,
    capsule_radius=0.01,
    capsule_half_length=0.02,
    link_mass=0.01,
    friction=ROPE_FRICTION,
    bend_stiffness=0.0,
    # 既存 USD は damping 属性へ 0.01 を直接書いていた。USD の角度ドライブは
    # deg 基準なので、rad 基準では 0.01 * 180/pi ≒ 0.5730 が実効値になる。
    # ベースラインを変えないためこの実効値をそのまま保持する。
    bend_damping=0.01 * 180.0 / math.pi,
    bend_limit_deg=None,
    twist_stiffness=None,
    twist_damping=0.0,
    twist_limit_deg=None,
)

_STIFF = _make_physical_spec(
    name="stiff",
    description="曲げ剛性 EI と最小曲げ半径を実物基準で入れた「腰のある」ロープ",
    color=(0.15, 0.65, 0.20),
    color_name="緑",
    num_links=20,
    capsule_radius=0.01,
    lock_twist=True,
)

_TWIST = _make_physical_spec(
    name="twist",
    description="stiff より柔らかい曲げ剛性 + 捩り (rotX) を解放。グリッパー把持・運搬向け",
    color=(0.95, 0.50, 0.05),
    color_name="橙",
    num_links=20,
    capsule_radius=0.01,
    lock_twist=False,
    youngs_modulus=TWIST_YOUNGS_MODULUS,
)

_FINE = _make_physical_spec(
    name="fine",
    description="直径 8 mm・48 リンク。結び目が結べる分解能 (計算コストは高い)",
    color=(0.85, 0.12, 0.15),
    color_name="赤",
    num_links=48,
    capsule_radius=0.004,
    lock_twist=False,
)

KNOT_NUM_LINKS = 28
"""`knot` プリセットのリンク数 (全長 1.14 m)。

## なぜ `twist` を長くしただけなのか (ゲート 1, 2026-09-27)

`demos/overhand_script.py` で台本の固結びを試したところ、`twist` (20 リンク,
0.82 m) で足りなかったのは **長さ** だった。pick&place の固結びは
「本体 -> 輪 -> 輪の片側の下をくぐる尻尾 -> 反対側の上を越える尻尾」の順に
ロープを使い、グリッパーが輪の片側だけを掴めるよう輪の径は 15 cm 前後要る
(開いた指は掴む軸から ±4 cm を占める)。本体 ~8 / 輪 ~12 / 尻尾 ~8 リンクで
20 リンクでは尻尾が 2〜3 リンクしか輪の中に残らず、最後の手で持ち上げると
輪の下から抜けてしまう。

**材料・太さ・リンク長は `twist` と同じ** (直径 20 mm, ヤング率 1e5 Pa)。
1 リンクあたりの質量・慣性・関節ゲイン・把持力が `twist` と一致するので、
`twist` で調整済みの「離すと付いてくる」対策 (docs/decisions/0004-release-follow-physics.md) と発散しにくさを
そのまま引き継げる。細く・細かくした `fine` は結び目の分解能では有利だが、
慣性が小さく発散しやすい (`rope_task_cfg.py` の `fine` の docstring)。

台本 (`overhand_script.py`, ±25% 揺らし, 128 環境) の固結び成功率:

| ロープ | 台本 (ゆっくり, 7.8 s/手) | 台本 (3.2 s/手) |
|---|---|---|
| `twist` | 最高 23/128 (18%)。`knot` 用の既定値では 0/128 | — |
| `knot`  | **122/128 (95%)**, 発散 0 | **116/128 (91%)**, 発散 3 |

詳細と 6 次元の行動空間での再生結果は `docs/experiments/2026-09-27_gate1.md`。

### 注意: 速く運ぶと発散する

学習環境の既定の尺 (v2 の `SHORT_DURATIONS`: carry 0.35 s) で 0.5 m 以上運ぶと、
引きずる本体が長く重いぶん `twist` より発散しやすい (1 手目で 13/32 が発散)。
carry を 0.6 s にすると 0/32 になった。ソルバ反復 32 や指ゲイン 2 倍では
改善しなかったので、原因は運搬速度そのもの。`knot` で学習するなら
carry (と descend/lift/lower) を延ばした尺を使うこと。
"""

_KNOT = _make_physical_spec(
    name="knot",
    description="twist と同じ材料・太さで 28 リンク (全長 1.14 m) に延ばした固結び用ロープ",
    color=(0.55, 0.20, 0.75),
    color_name="紫",
    num_links=KNOT_NUM_LINKS,
    capsule_radius=0.01,
    lock_twist=False,
    youngs_modulus=TWIST_YOUNGS_MODULUS,
)


def _refine_chain(base: RopeSpec, factor: int, name: str, description: str, color, color_name: str) -> RopeSpec:
    """`base` と **同じ材料・同じ太さ・同じ全長** のまま、リンクを `factor` 分の 1 の長さに刻んだ鎖。

    紐モデルの比較 (`rope_study/`) で、剛体鎖の離散化の粗さ (リンク 4 cm = 直径の 2 倍) の影響を
    切り分けるために追加した。`_make_physical_spec` は「1 本おきのリンクが接しない」幾何拘束
    (`HALF_LENGTH_TO_RADIUS` = 2) からリンク長を決めるので、それより細かい鎖は作れない。ここでは
    リンク長を自由に縮め、代わりに近いリンクどうしの接触を `filter_neighbor_pairs` で切る。

    揃える量 (連続体として同じ紐になるように):
        線密度          リンク質量 = base の線密度 x 新しいリンク間隔 (カプセルの重なりで質量が増えないよう体積からは出さない)
        曲げ剛性 EI     関節のばね定数 = EI / 新しいリンク間隔 (base の値 x factor)
        捩り剛性 GJ     同上
        最小曲げ半径    関節の角度制限 = 2 asin(リンク間隔 / (2 R_min))
        減衰            各関節を臨界減衰 (新しいリンクの慣性で計算し直す)
    """
    spacing = base.link_spacing / factor
    half = 0.5 * spacing
    r = base.capsule_radius
    lin_density = base.link_mass / base.link_spacing
    link_mass = lin_density * spacing
    ei = base.bend_stiffness * base.link_spacing
    bend_k = ei / spacing
    total_len = spacing + 2.0 * r
    inertia_t = link_mass * (3.0 * r**2 + total_len**2) / 12.0
    inertia_a = 0.5 * link_mass * r**2
    bend_c = 2.0 * BEND_DAMPING_RATIO * math.sqrt(bend_k * inertia_t)
    r_min = MIN_BEND_RADIUS_IN_DIAMETERS * 2.0 * r
    bend_limit = math.degrees(2.0 * math.asin(min(spacing / (2.0 * r_min), 1.0)))
    twist_k = None if base.twist_stiffness is None else base.twist_stiffness * base.link_spacing / spacing
    twist_c = 0.0 if twist_k is None else 2.0 * BEND_DAMPING_RATIO * math.sqrt(twist_k * inertia_a)
    gap = int(math.floor(1.25 * 2.0 * r / spacing)) + 1
    return RopeSpec(
        name=name,
        description=description,
        color=color,
        color_name=color_name,
        num_links=base.num_links * factor,
        capsule_radius=r,
        capsule_half_length=half,
        link_mass=link_mass,
        friction=base.friction,
        bend_stiffness=bend_k,
        bend_damping=bend_c,
        bend_limit_deg=bend_limit,
        twist_stiffness=twist_k,
        twist_damping=twist_c,
        twist_limit_deg=base.twist_limit_deg,
        filter_neighbor_pairs=max(1, gap),
    )


_KNOT56 = _refine_chain(
    _KNOT,
    2,
    name="knot56",
    description="knot と同じ材料・太さ・全長で、リンクを 2 cm (直径と同じ) に刻んだ 56 リンクの鎖 (紐モデルの比較用)",
    color=(0.75, 0.35, 0.95),
    color_name="薄紫",
)
"""`knot` を 2 倍細かく刻んだ鎖 (紐モデルの比較 `rope_study/` の `Art56`)。

リンク数は 28 x 2 = 56 で、端のキャップを含む全長 (55 x 0.02 + 0.04 = 1.14 m) も、節点列
(`polyline_from_link_centers`) の長さ (56 x 0.02 = 1.12 m) も `knot` と同じ。1 本おきのリンク
(中心の間隔 4 cm、カプセルの全長 4 cm) は
まっすぐな状態で接するので、`filter_neighbor_pairs=2` で接触を切る。
"""

# ---------------------------------------------------------------- 寸法の DR (v16)

SIZE_LENGTH_SCALES = (0.8, 1.0, 1.25)
"""寸法の DR の長さの 3 水準 (公称のリンク間隔 = `capsule_half_length` に掛ける倍率)。

実機の紐 (全長 1.02 m) は ×0.895 で範囲の中。`docs/decisions/0029-v16-rope-size-dr.md` の「寸法の範囲」(ユーザー、2026-10-15)。
"""

SIZE_DIAMETER_SCALES = (0.4, 0.63, 1.0)
"""寸法の DR の直径の 3 水準 (公称の `capsule_radius` に掛ける倍率)。8 mm / 12.6 mm / 20 mm。

実機の紐の直径 8 mm (×0.4) を含むように、公称 (20 mm) を最大の水準にした (0029)。
"""


def size_variant_name(base: str, length_scale: float, diameter_scale: float, keep_joint_limits: bool = False) -> str:
    """寸法の組のプリセット名。公称 (×1.0, ×1.0) は `base` そのもの。

    `keep_joint_limits=False` は角度制限を寸法から導き直した組 (例: `knot_l080_d040`)、`True` は角度制限を `base` の
    まま揃えた長さだけの組 (例: `knot_l080`。直径は ×1.0 だけ。`make_size_variant` の docstring)。
    公称の組を別の USD にしないのは、寸法を振らない env を継承元の版とまったく同じロープにするため。
    """
    if length_scale == 1.0 and diameter_scale == 1.0:
        return base
    if keep_joint_limits:
        if diameter_scale != 1.0:
            raise ValueError(f"角度制限を揃えた組は長さだけ (直径 ×1.0) を想定している (got 直径 ×{diameter_scale:g})")
        return f"{base}_l{round(length_scale * 100):03d}"
    return f"{base}_l{round(length_scale * 100):03d}_d{round(diameter_scale * 100):03d}"


def make_size_variant(
    base: RopeSpec, length_scale: float, diameter_scale: float, keep_joint_limits: bool = False
) -> RopeSpec:
    """`base` と同じ材料・同じリンク数で、リンク間隔を `length_scale` 倍・半径を `diameter_scale` 倍にしたロープ。

    `base` は `_make_physical_spec` で作ったプリセット (`knot`) を想定する。材料定数 (ヤング率 `TWIST_YOUNGS_MODULUS`・
    密度・減衰比・最小曲げ半径) は同じにして、質量・慣性・関節のばね・減衰・曲げの角度制限を寸法から導き直す
    (同じ材料の太さ・長さ違いの紐)。リンク数を変えないのは、DOF の数・観測 (197 次元)・`grasp_link_id` の範囲を
    版の間で揃えるため (0029 の「採らなかった案」)。

    最小曲げ半径 (直径の 2 倍) がリンク間隔の半分より小さい組 (直径 ×0.4 の 3 組) は、導出した角度制限が 180° に
    張り付く。PhysX の曲げの制限 (swing) は 180° 未満でないと扱えないので、その組は角度制限を付けない
    (`bend_limit_deg=None`。`simple` と同じ「制限なし」の経路)。1 本おきのリンクどうしの接触が折れ返りを止める。

    `keep_joint_limits=True` (v16 で使う長さだけの組) は、曲げの角度制限を `base` の値 (`knot` の ±60°) のままにする。
    捩りの制限 (±90°) はどの組も同じ。v16 のアーカイブは形を関節角で持ち、寸法の違う env の間で書き戻すので、
    角度制限が組ごとに違うと、ある組で保存した関節角が書き戻し先の制限を超え、リセットの直後に飛ぶ
    (`docs/experiments/2026-10-17_v16_size-gate.md`。v15 の学習のアーカイブの形は ±60° に張り付き、長さ ×0.8 の導出した
    制限 ±47.2° を超える形が段 2 で 74%)。揃えると、どの組の関節角も全組で有効で、書き戻した形は保存した形と相似になる。
    代わりに最小曲げ半径がリンク間隔に比例する (×0.8 で 32 mm = 直径の 1.6 倍、×1.25 で 50 mm = 2.5 倍。公称は 40 mm)。
    リンク k と k+2 の最も近い点は角度に依らずリンク間隔ぶん (32 mm 以上) 離れていて、接触の届く範囲 (28 mm) より遠いので、
    制限を広げても 1 本おきの自己接触は増えない。ばね・減衰・質量は `keep_joint_limits=False` と同じく寸法から導く。
    """
    if base.name != "knot":
        raise ValueError(f"寸法の DR は knot だけを想定している (got {base.name})")
    limits = "角度制限は公称のまま" if keep_joint_limits else "角度制限は寸法から導出"
    spec = _make_physical_spec(
        name=size_variant_name(base.name, length_scale, diameter_scale, keep_joint_limits),
        description=f"{base.name} の長さ ×{length_scale:g}・直径 ×{diameter_scale:g}、{limits} (寸法の DR、v16)",
        color=base.color,
        color_name=base.color_name,
        num_links=base.num_links,
        capsule_radius=base.capsule_radius * diameter_scale,
        capsule_half_length=base.capsule_half_length * length_scale,
        lock_twist=base.twist_stiffness is None,
        youngs_modulus=TWIST_YOUNGS_MODULUS,
        twist_limit_deg=base.twist_limit_deg,
    )
    if keep_joint_limits:
        spec = replace(spec, bend_limit_deg=base.bend_limit_deg)
    elif spec.bend_limit_deg is not None and spec.bend_limit_deg >= 180.0:
        spec = replace(spec, bend_limit_deg=None)
    return spec


_SIZE_VARIANTS = tuple(
    make_size_variant(_KNOT, ls, ds)
    for ls in SIZE_LENGTH_SCALES
    for ds in SIZE_DIAMETER_SCALES
    if not (ls == 1.0 and ds == 1.0)
)
"""`knot` の寸法の格子 (3 × 3) のうち公称を除く 8 組 (角度制限を寸法から導出)。USD は `generate_assets.py --rope <名前>` で作る。

v16 の物理のゲートで測っただけで、学習には使わない (直径 ×0.4・×0.63 は台本が結べず、長さの組は角度制限が組ごとに違う。
`docs/experiments/2026-10-17_v16_size-gate.md`)。
"""

_LENGTH_VARIANTS = tuple(
    make_size_variant(_KNOT, ls, 1.0, keep_joint_limits=True) for ls in SIZE_LENGTH_SCALES if ls != 1.0
)
"""`knot` の長さだけの組 (`knot_l080`・`knot_l125`。角度制限は公称の ±60°)。v16 の学習に使う (ユーザー、2026-10-17)。"""

LENGTH_VARIANT_SCALES: dict[str, tuple[float, float]] = {
    size_variant_name("knot", ls, 1.0, keep_joint_limits=True): (ls, 1.0) for ls in SIZE_LENGTH_SCALES
}
"""角度制限を揃えた長さだけの組のプリセット名 -> (長さの倍率, 直径の倍率)。公称 (`knot`) を含む 3 組。"""

SIZE_SCALES: dict[str, tuple[float, float]] = {
    size_variant_name("knot", ls, ds): (ls, ds) for ls in SIZE_LENGTH_SCALES for ds in SIZE_DIAMETER_SCALES
}
"""寸法の組のプリセット名 -> (長さの倍率, 直径の倍率)。公称 (`knot`) を含む 9 組。"""

ROPE_SPECS: dict[str, RopeSpec] = {
    spec.name: spec for spec in (_SIMPLE, _STIFF, _TWIST, _FINE, _KNOT, _KNOT56, *_SIZE_VARIANTS, *_LENGTH_VARIANTS)
}
"""名前 -> `RopeSpec` のレジストリ。`--rope` の選択肢はここから引く。"""

DEFAULT_ROPE = "simple"
"""既定のプリセット。既存デモの調整値を壊さないよう現行モデルのままにしてある。"""


def get_spec(name: str) -> RopeSpec:
    """名前からプリセットを引く。未知の名前なら候補を添えて例外を投げる。"""
    try:
        return ROPE_SPECS[name]
    except KeyError:
        raise KeyError(f"unknown rope preset '{name}'. available: {', '.join(ROPE_SPECS)}") from None


def add_rope_arg(parser, default: str = DEFAULT_ROPE) -> None:
    """`--rope` 引数を argparse のパーサに追加する (各デモから呼ぶ共通処理)。"""
    help_lines = "; ".join(f"{n}={s.description}" for n, s in ROPE_SPECS.items())
    parser.add_argument(
        "--rope",
        type=str,
        default=default,
        choices=list(ROPE_SPECS),
        help=f"使用するロープの作り方。{help_lines}",
    )


if __name__ == "__main__":
    for _spec in ROPE_SPECS.values():
        print(_spec.summary())
