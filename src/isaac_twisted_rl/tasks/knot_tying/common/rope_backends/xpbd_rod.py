"""Warp で書いた XPBD の紐ソルバ。紐モデルの比較 (`rope_study/`) で追加。isaaclab を import しない。

PhysX の Articulation (D6 関節のカプセル鎖) とは別の表現の紐を、同じ GPU 上で全環境まとめて解く。
2 つのモデルを同じ接触処理の上に載せてある:

* ``model="cosserat"``: 位置と向きを持つ Cosserat 棒 (Kugelstadt & Schömer, SCA 2016 の PBD 版を
  XPBD (Macklin et al., MIG 2016) の compliance で書いたもの)。節点 (N+1 個) と区間の向き (N 個の
  quaternion) を持ち、伸び・せん断 (ほぼ剛) と、Darboux ベクトルによる曲げ (EI)・ねじれ (GJ) を
  材料定数そのままの compliance で解く。
* ``model="mass_spring"``: 質点とばねの鎖 (SoftGym / NVIDIA FleX の紐と同じ形)。隣どうしの距離
  (ほぼ剛) と、1 つ飛ばしの距離ばね (曲げの代わり) だけ。ねじれは無い。

接触はどちらも同じ:

* 床 (z = 0 の平面) と節点。Coulomb 摩擦 (静止 / 動摩擦。Macklin et al. 2014 "Unified particle
  physics" の位置ベースの摩擦)。
* 指 (グリッパーの 2 本の直方体) と、区間上の標本点 (半径 r の球)。**指は compliance を持つ**:
  PhysX のグリッパーは「全閉を指令し、紐の太さぶん閉じ切れない PD の定常偏差が把持力になる」形
  (`mdp/actions.py`) なので、それと同じ「押し込み量 x ばね定数」の力になるよう compliance を
  1 / k_finger にしてある。指の位置は PhysX のグリッパー (紐とは接触しない) から毎ステップ読む。
* 紐どうし (区間 = 半径 r のカプセル)。弧長で近い組は除外する。

時間積分は small steps (Macklin et al., SCA 2019): 1 物理ステップ (1/240 s) を `substeps` 個に割り、
各小ステップで拘束を 1 回ずつ解く。拘束どうしの競合は、紐の内部 (伸び・曲げ) は偶奇の 2 色に塗った
Gauss-Seidel、接触は Jacobi (補正を足し合わせてから平均) で解く。

## 配列の約束

* 節点 `x` `(E * P, 3)`、P = N + 1。環境 e の節点 k は `e * P + k`。
* 向き `q` `(E * N, 4)` は **Warp の並び (x, y, z, w)**。torch 側へ出すときは (w, x, y, z) に直す
  (`XpbdRod.quat_wxyz`)。局所 z 軸 (d3) が区間の接線。
* 状態は torch のテンソルで持ち、`wp.from_torch` で同じメモリを Warp から触る (コピーしない)。
* **座標は原点の近く (環境ごとの局所座標) で渡す。** float32 なので、座標が大きいと小ステップ 1 回の小さい
  補正 (床の動摩擦は μ g h^2 = 3.3e-7 m (μ 0.5, 16 小ステップ)) が丸めで消える。μ 0.5 では原点から 10 m、
  μ 2.0 では 40 m で床の動摩擦が効かなくなり、紐が一定の速さで滑り続ける。環境につなぐアダプタ
  (`adapters.XpbdRope`) は環境の原点を引いた座標で解く。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import warp as wp

wp.config.quiet = True


# ---------------------------------------------------------------- 設定


@dataclass
class XpbdRodParams:
    """XPBD の紐の物理パラメータ。既定値は `Knot` プリセット (28 リンク, 直径 20 mm) の材料に合わせてある。"""

    model: str = "cosserat"
    """"cosserat" か "mass_spring"。"""

    num_segments: int = 28
    length: float = 1.12
    """中心線の長さ [m]。`Knot` の節点列 (`polyline_from_link_centers`) の長さ 28 x 0.04 m と同じ。"""

    radius: float = 0.01
    linear_density: float = 0.2513
    """線密度 [kg/m]。`Knot` のリンク質量 0.01005 kg / 0.04 m。"""

    bend_stiffness: float = 7.854e-4
    """曲げ剛性 EI [N m^2]。`Knot` と同じ E = 1e5 Pa, I = pi r^4 / 4。"""

    twist_stiffness: float = 3.927e-4
    """ねじれ剛性 GJ [N m^2]。`rope_specs.TORSION_TO_BENDING_RATIO` = 0.5 に合わせた。"""

    stretch_compliance: float = 0.0
    """伸びの compliance [m/N] (0 なら伸びない)。"""

    shear_compliance: float = 0.0

    spring_bend_stiffness: float = 785.0
    """`mass_spring` の 1 つ飛ばしのばねの剛性 [N/m]。曲げ角 0.5 rad で Cosserat の曲げエネルギーと
    一致するように決めた (k = 16 EI / (l^3 theta^2))。ばねの曲げは角度の 4 乗のエネルギーになるので、
    小さい曲げには柔らかすぎ、大きい曲げには硬すぎる。"""

    friction_ground: float = 2.0
    """床との摩擦係数 (静止・動摩擦とも)。`Knot` (PhysX の average 合成で 2.0) に合わせた既定値。"""

    friction_self: float = 2.0
    friction_finger: float = 4.0
    """指との摩擦係数。PhysX 版の指の摩擦材 (mu 4.0, max 合成) と同じ。"""

    dynamic_friction_ratio: float = 1.0
    """動摩擦 / 静止摩擦。PhysX 版は同値。"""

    finger_stiffness: float = 1507.0
    """指 1 本の押し付けのばね定数 [N/m]。`RopeSpec.grasp_finger_stiffness` (1500 x 質量 / 半径)。"""

    finger_samples_per_segment: int = 0
    """区間あたりの指との接触の標本点。0 なら区間長 / (半径 / 2) から決める。"""

    self_contact_exclude_arc: float = -1.0
    """自己接触を見ない弧長 [m] (区間の中心どうし)。負なら pi x 半径。"""

    gravity: float = -9.81
    substeps: int = 16
    iterations: int = 1
    """小ステップあたりの、紐の内部の拘束 (伸び・せん断・曲げ・ねじれ) の反復回数。"""

    inertia_scale: float = 1.0
    """区間の回転慣性の倍率 (cosserat)。"""
    linear_damping: float = 0.2
    """全節点の速度の減衰率 [1/s] (空気抵抗相当)。"""

    bend_damping: float = 60.0
    """曲げの減衰率 [1/s]。節点の速度の 2 階差分 (曲げる速さ) と、隣どうしの区間の相対角速度
    (cosserat のねじれ) を、この割合で減らす (`_k_damp_bend_vel`)。実際の紐はほとんど跳ねないので、
    片持ち梁の試験 (`rope_study/bench_cantilever_xpbd.py`) で 1 s 以内に落ち着く値にした。"""

    angular_damping: float = 200.0
    """区間の角速度そのものの減衰率 [1/s] (cosserat)。

    区間の向きは自分の回転慣性を持ち、せん断の拘束で節点に引きずられる。この回転の自由度に運動
    エネルギーが溜まり、片持ち梁が理論の釣り合いのまわりで 3 s 以上ほとんど減衰せずに振れ続けた
    (曲げの減衰 `bend_damping` を 5 倍にしても変わらない)。角速度そのものを減衰させると 1 s 以内に
    落ち着く (`rope_study/bench_cantilever_xpbd.py`)。剛体としての回転も遅らせる数値的な減衰だが、
    この課題は準静的な pick&place なので影響は小さいと判断した。PhysX の剛体鎖も各関節を臨界減衰に
    しており (`rope_specs.BEND_DAMPING_RATIO`)、同程度に「跳ねない紐」になっている。"""

    max_speed: float = 50.0
    """節点の速度の上限 [m/s]。正常な動作 (1 m/s 程度) には触れない保険。"""

    finger_box_half: tuple[float, float, float] = (0.01, 0.006, 0.03)
    """指の直方体の半幅 (手の x, y, z) [m]。`parallel_gripper/generate_gripper_usd.py` の FINGER_SIZE / 2。"""

    finger_center_y0: float = 0.006
    """全閉のときの指の中心の |y| [m] (FINGER_Y_OFFSET)。"""

    finger_center_z: float = -0.045
    """指の中心の手の root からの z [m] (-(BASE_SIZE[2] / 2 + FINGER_SIZE[2] / 2))。"""

    finger_x_faces: bool = False
    """指の前後の面 (手の x) にも接触を持たせるか。

    False (2026-10-02 の物理試験まで) は挟む面 (y) と底面 (z) だけ。そのため手が x 方向へ動くと紐が前後の面から
    指の直方体の中へ入り、y 方向の深いめり込み (compliance の力) と摩擦で捕まったまま運ばれる、という仮説がある
    (台本の遅い尺で XPBD だけ固結びが 2%、`rope_study/Report_rope_models.md` §7.1.2)。True にすると、中心が指の
    y の範囲 (`finger_x_face_margin` だけ内側へ狭めた範囲) にある標本点を前後の面から押し返す。"""

    finger_rigid_outside: bool = False
    """指の接触のうち、PD の指で挟む力を表す部分だけを compliance にし、残りを剛に解くか。

    指は y (開閉の向き) の PD (1507 N/m) でしか動かないので、compliance が意味を持つのは「相手の指の側の面で
    紐を挟む」部分だけ。False (2026-10-02 の物理試験まで) は全部の面が compliance で、床に置いた紐の上へ指を
    降ろすと紐が指の中に深く (最大 16 mm) 入ったまま残り、その押し込みの力と μ = 4 の摩擦で、手を動かすと
    紐を連れていく (`rope_study/Report_rope_models.md` §7.1.2)。True では:

    * 挟む面と底面: 紐の中心が指の中へ入った分は剛に押し戻し、残りの半径ぶんを compliance (挟む力) にする。
    * 外側の面・前後の面: 全部剛 (位置の補正を標本点で平均)。
    """

    finger_max_depenetration_velocity: float = 1.0
    """`finger_rigid_outside` の剛の押し戻しの速さの上限 [m/s]。PhysX の紐 (`rope_cfg._max_depenetration_velocity`、
    0.96 m/s) と同じ程度。速く動く指 (瞬間移動) は紐を弾かずに通り抜ける。"""

    hand_teleport_distance: float = 0.0
    """1 物理ステップでこれより大きく [m] 動いた手を瞬間移動として扱い、小ステップに補間しない (0 なら常に補間)。

    PhysX はキネマティックな手の姿勢を書き込むと、途中を掃かずに新しい位置で接触を解く。学習の尺の clear (0 s)
    は手を 1 物理ステップで 8 cm 横へ動かすので、補間すると指が 19 m/s で紐を掃く。剛の指 (`finger_rigid_outside`)
    では紐を弾き飛ばし、摩擦で引きずる。台本と行動の通常の動き (1 ステップ 1.3 cm 以下) は補間する。"""

    finger_x_face_margin: float = 0.003
    """前後の面の接触を見る y の範囲を、指の厚みの両側からこれだけ狭める [m]。挟んだ紐の中心は挟む面の上
    (指の y の範囲の端) にあるので、狭めないと挟んだ紐のうち指の前後にはみ出た部分を軸方向へ押してしまう。"""

    @property
    def segment_length(self) -> float:
        return self.length / self.num_segments

    @property
    def num_nodes(self) -> int:
        return self.num_segments + 1

    @property
    def segment_mass(self) -> float:
        return self.linear_density * self.segment_length


# ---------------------------------------------------------------- Warp カーネル


@wp.func
def _rotz(yaw: float, v: wp.vec3):
    c = wp.cos(yaw)
    s = wp.sin(yaw)
    return wp.vec3(c * v[0] - s * v[1], s * v[0] + c * v[1], v[2])


@wp.func
def _quat_small_rot(q: wp.quat, dtheta: wp.vec3):
    """world の小回転 dtheta を q に掛けて正規化する (q + 0.5 [dtheta, 0] q)。"""
    dq = wp.quat(dtheta[0], dtheta[1], dtheta[2], 0.0) * q
    return wp.normalize(wp.quat(q[0] + 0.5 * dq[0], q[1] + 0.5 * dq[1], q[2] + 0.5 * dq[2], q[3] + 0.5 * dq[3]))


@wp.kernel
def _k_predict(
    x: wp.array(dtype=wp.vec3),
    x_prev: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    inv_m: wp.array(dtype=float),
    gravity: float,
    h: float,
    damp: float,
    max_speed: float,
):
    i = wp.tid()
    if inv_m[i] <= 0.0:  # 固定した節点 (片持ち梁の試験など) は動かさない
        v[i] = wp.vec3(0.0, 0.0, 0.0)
        x_prev[i] = x[i]
        return
    vi = v[i] * damp + wp.vec3(0.0, 0.0, gravity * h)
    sp = wp.length(vi)
    if sp > max_speed:
        vi = vi * (max_speed / sp)
    v[i] = vi
    x_prev[i] = x[i]
    x[i] = x[i] + vi * h


@wp.kernel
def _k_predict_rot(
    q: wp.array(dtype=wp.quat), q_prev: wp.array(dtype=wp.quat), omega: wp.array(dtype=wp.vec3), h: float, damp: float
):
    i = wp.tid()
    q_prev[i] = q[i]
    w = omega[i] * damp
    omega[i] = w
    q[i] = _quat_small_rot(q[i], w * h)


@wp.kernel
def _k_stretch_shear(
    x: wp.array(dtype=wp.vec3),
    q: wp.array(dtype=wp.quat),
    inv_m: wp.array(dtype=float),
    inv_i: float,
    l0: float,
    a_stretch: float,
    a_shear: float,
    parity: int,
    n_seg: int,
    n_half: int,
):
    tid = wp.tid()
    e = tid // n_half
    j = 2 * (tid % n_half) + parity
    if j >= n_seg:
        return
    i0 = e * (n_seg + 1) + j
    i1 = i0 + 1
    qi = e * n_seg + j
    w0 = inv_m[i0]
    w1 = inv_m[i1]
    d3 = wp.quat_rotate(q[qi], wp.vec3(0.0, 0.0, 1.0))
    gam = (x[i1] - x[i0]) / l0 - d3
    g_par = wp.dot(gam, d3) * d3
    g_perp = gam - g_par
    wl = (w0 + w1) / (l0 * l0)
    lam = -g_perp / (wl + inv_i + a_shear)
    if wl + a_stretch > 0.0:  # 両端とも固定 (逆質量 0) の区間は伸びを直せない
        lam = lam - g_par / (wl + a_stretch)
    x[i0] = x[i0] - (w0 / l0) * lam
    x[i1] = x[i1] + (w1 / l0) * lam
    q[qi] = _quat_small_rot(q[qi], inv_i * wp.cross(lam, d3))


@wp.kernel
def _k_bend_twist(
    q: wp.array(dtype=wp.quat),
    lam_acc: wp.array(dtype=wp.vec3),
    inv_i: float,
    l0: float,
    a_bend: float,
    a_twist: float,
    parity: int,
    n_seg: int,
    n_half: int,
):
    """曲げ・ねじれ (Darboux ベクトル) の XPBD 拘束。乗数 `lam_acc` は小ステップの中の反復で積み上げる
    (積み上げないと PBD と同じになり、剛性が反復回数と小ステップ数で変わる)。"""
    tid = wp.tid()
    e = tid // n_half
    j = 2 * (tid % n_half) + parity
    if j >= n_seg - 1:
        return
    qa_i = e * n_seg + j
    qb_i = qa_i + 1
    li = e * (n_seg - 1) + j
    qa = q[qa_i]
    qb = q[qb_i]
    r = wp.quat_inverse(qa) * qb
    im = wp.vec3(r[0], r[1], r[2])
    if r[3] < 0.0:
        im = -im
    omega = (2.0 / l0) * im  # Darboux ベクトル (区間 j の物質枠)
    wb = 2.0 * inv_i / (l0 * l0)
    la = lam_acc[li]
    lam = wp.vec3(
        (-omega[0] - a_bend * la[0]) / (wb + a_bend),
        (-omega[1] - a_bend * la[1]) / (wb + a_bend),
        (-omega[2] - a_twist * la[2]) / (wb + a_twist),
    )
    lam_acc[li] = la + lam
    dth = wp.quat_rotate(qa, lam) * (inv_i / l0)
    q[qb_i] = _quat_small_rot(qb, dth)
    q[qa_i] = _quat_small_rot(qa, -dth)


@wp.kernel
def _k_distance(
    x: wp.array(dtype=wp.vec3),
    inv_m: wp.array(dtype=float),
    lam_acc: wp.array(dtype=float),
    rest: float,
    alpha: float,
    stride: int,
    color: int,
    n_nodes: int,
    n_cons: int,
):
    """節点 k と k + stride の距離拘束 (stride 1: 伸び、2: 1 つ飛ばしのばね)。

    同じ色の拘束どうしは節点を共有しないように塗る: stride 1 は k の偶奇、stride 2 は (k // 2) の偶奇。
    """
    tid = wp.tid()
    e = tid // n_cons
    k = tid % n_cons
    if k + stride >= n_nodes:
        return
    c = k % 2
    if stride == 2:
        c = (k // 2) % 2
    if c != color:
        return
    ia = e * n_nodes + k
    ib = ia + stride
    wa = inv_m[ia]
    wb = inv_m[ib]
    d = x[ib] - x[ia]
    dist = wp.length(d)
    if dist < 1.0e-9:
        return
    n = d / dist
    if wa + wb + alpha <= 0.0:
        return
    li = e * n_cons + k
    lam = (-(dist - rest) - alpha * lam_acc[li]) / (wa + wb + alpha)
    lam_acc[li] = lam_acc[li] + lam
    x[ia] = x[ia] - wa * lam * n
    x[ib] = x[ib] + wb * lam * n


@wp.kernel
def _k_ground(
    x: wp.array(dtype=wp.vec3),
    x_prev: wp.array(dtype=wp.vec3),
    radius: float,
    mu_s: float,
    mu_k: float,
):
    i = wp.tid()
    p = x[i]
    pen = radius - p[2]
    if pen <= 0.0:
        return
    p = wp.vec3(p[0], p[1], radius)
    dx = p - x_prev[i]
    dt = wp.vec3(dx[0], dx[1], 0.0)
    ln = wp.length(dt)
    if ln < mu_s * pen:
        p = p - dt
    elif ln > 0.0:
        p = p - dt * wp.min(mu_k * pen / ln, 1.0)
    x[i] = p


@wp.func
def _box_contact(pl: wp.vec3, c: wp.vec3, he: wp.vec3, r: float, x_faces: int, x_margin: float):
    """手の座標系の点 pl (半径 r の球) と、中心 c・半幅 he の指の直方体。(めり込み, 法線 (手の座標系), 最近点)。

    **指の面のうち、挟む面 (y) と底面 (z) だけを見る。** 角や x 側の面とは接触させない。指は compliance を
    持つので、閉じた指は紐に半径の 8 割ほどめり込む (力 = めり込み x ばね定数)。そのとき指の外へ垂れた
    紐の標本点に指の下の角が深く食い込み、紐を下へ押し出して指先から抜け落とした (bench_physics の
    hang 試験、2026-10-02)。PhysX の剛体の指は紐の表面で止まるので角は食い込まない。
    * 挟む面: 球の中心が指の x の範囲と z の範囲 (下は半径の半分まで) にあるとき、y 方向に押す。
    * 底面: 球の中心が指の x・y の範囲にあり、底面より下で半径以内のとき、下へ押す (指を紐の上に
      降ろしたとき。床と挟まれても指は柔らかいので発散しない)。
    """
    d = pl - c
    pen = float(0.0)
    n = wp.vec3(0.0, 0.0, 1.0)
    cp = c
    inside_x = wp.abs(d[0]) <= he[0]
    if inside_x and d[2] >= -he[2] - 0.5 * r and d[2] <= he[2]:
        # 挟む面 (中心が箱の外側の y にある、または箱の中)
        if wp.abs(d[1]) >= he[1]:
            pen = r - (wp.abs(d[1]) - he[1])
        else:
            pen = r + (he[1] - wp.abs(d[1]))
        n = wp.vec3(0.0, wp.sign(d[1]), 0.0)
        cp = c + wp.vec3(d[0], wp.sign(d[1]) * he[1], wp.clamp(d[2], -he[2], he[2]))
    if inside_x and wp.abs(d[1]) <= he[1] and d[2] < -he[2]:
        pb = r - (-he[2] - d[2])
        if pb > pen:
            pen = pb
            n = wp.vec3(0.0, 0.0, -1.0)
            cp = c + wp.vec3(d[0], d[1], -he[2])
    # 前後の面 (`XpbdRodParams.finger_x_faces`): 中心が指の y の範囲の内側 (margin だけ狭めた) と z の範囲にある
    # 標本点を x 方向へ押し返す。箱の中にいるなら、y と x のうち浅いほうへ出す
    if x_faces != 0 and wp.abs(d[1]) <= he[1] - x_margin and d[2] >= -he[2] and d[2] <= he[2]:
        px = r - (wp.abs(d[0]) - he[0])
        if px > 0.0 and (pen <= 0.0 or px < pen):
            sx = 1.0
            if d[0] < 0.0:
                sx = -1.0
            pen = px
            n = wp.vec3(sx, 0.0, 0.0)
            cp = c + wp.vec3(sx * he[0], d[1], d[2])
    return pen, n, cp


@wp.kernel
def _k_finger(
    x: wp.array(dtype=wp.vec3),
    x_prev: wp.array(dtype=wp.vec3),
    inv_m: wp.array(dtype=float),
    hand0: wp.array(dtype=wp.vec4),  # (x, y, z, yaw) 小ステップの始め
    hand1: wp.array(dtype=wp.vec4),  # 小ステップの終わり
    fing0: wp.array(dtype=wp.vec2),  # 左右の指の関節位置 (始め)
    fing1: wp.array(dtype=wp.vec2),
    he: wp.vec3,
    cy0: float,
    cz: float,
    radius: float,
    alpha: float,
    mu_s: float,
    mu_k: float,
    n_seg: int,
    n_samp: int,
    x_faces: int,
    x_margin: float,
    rigid_outside: int,
    max_hard: float,
    dx_n: wp.array(dtype=wp.vec3),
    dx_t: wp.array(dtype=wp.vec3),
    cnt_t: wp.array(dtype=float),
):
    tid = wp.tid()
    per_env = n_seg * n_samp
    e = tid // per_env
    rem = tid % per_env
    j = rem // n_samp
    k = rem % n_samp
    s = (float(k) + 0.5) / float(n_samp)
    ia = e * (n_seg + 1) + j
    ib = ia + 1
    wa = inv_m[ia] * (1.0 - s)
    wb = inv_m[ib] * s
    p = x[ia] * (1.0 - s) + x[ib] * s
    pp = x_prev[ia] * (1.0 - s) + x_prev[ib] * s
    h0 = hand0[e]
    h1 = hand1[e]
    hp1 = wp.vec3(h1[0], h1[1], h1[2])
    hp0 = wp.vec3(h0[0], h0[1], h0[2])
    pl = _rotz(-h1[3], p - hp1)
    weff = (1.0 - s) * wa + s * wb
    if weff <= 0.0:
        return
    for f in range(2):
        sgn = 1.0
        q1 = fing1[e][0]
        q0 = fing0[e][0]
        if f == 1:
            sgn = -1.0
            q1 = fing1[e][1]
            q0 = fing0[e][1]
        c1 = wp.vec3(0.0, sgn * cy0 + q1, cz)
        pen, nl, cp = _box_contact(pl, c1, he, radius, x_faces, x_margin)
        if pen > 0.0:
            n = _rotz(h1[3], nl)
            # 指の表面の点の、この小ステップの移動 (手の動き + 指の開閉) と、それに対する標本点の相対移動
            c0 = wp.vec3(0.0, sgn * cy0 + q0, cz)
            cp_now = hp1 + _rotz(h1[3], cp)
            cp_before = hp0 + _rotz(h0[3], cp - c1 + c0)
            rel = (p - pp) - (cp_now - cp_before)
            # 剛に解く分 (`XpbdRodParams.finger_rigid_outside`): 挟む面 (相手の指の側) と底面は、中心が指の中へ
            # 入った分だけ剛 (残りの半径ぶんが compliance = PD の指で挟む力)。外側の面と前後の面は全部剛。
            # 剛の押し戻しは、指に対して法線の向きへ離れる速さが `max_hard` / 小ステップ (= めり込み解消速度の上限)
            # を超えないところまで (PhysX の max_depenetration_velocity と同じ考え。小ステップごとに押し戻しを
            # 足すと速度が積み上がって紐を弾き飛ばす)。押し切れなかった残りは compliance にもしない
            # (残りを力にすると、それが摩擦の糊になる)
            soft = pen
            hard = float(0.0)
            if rigid_outside != 0:
                soft_face = False
                if nl[1] * sgn < 0.0:
                    soft_face = True
                if nl[2] < -0.5:
                    soft_face = True
                if soft_face:
                    soft = wp.min(pen, radius)
                else:
                    soft = 0.0
                allowed = wp.max(max_hard - wp.dot(rel, n), 0.0)
                hard = wp.min(pen - soft, allowed)
            dlam_soft = soft / (weff + alpha)
            if rigid_outside != 0:
                # compliance の分も同じ速さの上限で切る (指が紐に急に深く重なったとき、例えば瞬間移動の直後に、
                # PD の力 = ばね定数 x 重なりで軽い紐を弾き飛ばさないように)。静かに挟んでいるときの補正は
                # 小ステップあたり数十 um で、上限 (1 m/s で 260 um) には掛からない
                dlam_soft = wp.min(dlam_soft, wp.max(allowed - hard, 0.0) / weff)
            dlam = dlam_soft + hard / weff
            corr = n * dlam_soft
            # 法線 (compliance 付き) は力として足し合わせる。剛の法線と摩擦は節点ごとに平均する
            # (同じ節点に掛かる標本点の補正をすべて足すと、相対変位を何倍にも打ち消して振動し、滑る)。
            wp.atomic_add(dx_n, ia, wa * corr)
            wp.atomic_add(dx_n, ib, wb * corr)
            rel_t = rel - wp.dot(rel, n) * n
            lt = wp.length(rel_t)
            dn = weff * dlam
            ft = n * (hard / weff)
            if lt < mu_s * dn:
                ft = ft - rel_t / weff
            elif lt > 0.0:
                ft = ft - rel_t * (wp.min(mu_k * dn / lt, 1.0) / weff)
            if lt > 0.0 or hard > 0.0:
                wp.atomic_add(dx_t, ia, wa * ft)
                wp.atomic_add(dx_t, ib, wb * ft)
                wp.atomic_add(cnt_t, ia, 1.0)
                wp.atomic_add(cnt_t, ib, 1.0)


@wp.kernel
def _k_self(
    x: wp.array(dtype=wp.vec3),
    x_prev: wp.array(dtype=wp.vec3),
    inv_m: wp.array(dtype=float),
    pa: wp.array(dtype=int),
    pb: wp.array(dtype=int),
    n_pairs: int,
    n_seg: int,
    radius: float,
    mu_s: float,
    mu_k: float,
    dx_acc: wp.array(dtype=wp.vec3),
    cnt: wp.array(dtype=float),
):
    tid = wp.tid()
    e = tid // n_pairs
    k = tid % n_pairs
    base = e * (n_seg + 1)
    a0i = base + pa[k]
    a1i = a0i + 1
    b0i = base + pb[k]
    b1i = b0i + 1
    a0 = x[a0i]
    a1 = x[a1i]
    b0 = x[b0i]
    b1 = x[b1i]
    # 早期棄却: 区間の中点どうしが十分遠い
    da = a1 - a0
    db = b1 - b0
    if wp.length((a0 + a1) * 0.5 - (b0 + b1) * 0.5) > 0.5 * (wp.length(da) + wp.length(db)) + 2.0 * radius:
        return
    r = a0 - b0
    aa = wp.dot(da, da)
    ee = wp.dot(db, db)
    f = wp.dot(db, r)
    c = wp.dot(da, r)
    bb = wp.dot(da, db)
    denom = wp.max(aa * ee - bb * bb, 1.0e-12)
    s = wp.clamp((bb * f - c * ee) / denom, 0.0, 1.0)
    t = (bb * s + f) / wp.max(ee, 1.0e-12)
    if t < 0.0:
        t = 0.0
        s = wp.clamp(-c / wp.max(aa, 1.0e-12), 0.0, 1.0)
    elif t > 1.0:
        t = 1.0
        s = wp.clamp((bb - c) / wp.max(aa, 1.0e-12), 0.0, 1.0)
    ca = a0 + da * s
    cb = b0 + db * t
    dvec = ca - cb
    dist = wp.length(dvec)
    pen = 2.0 * radius - dist
    if pen <= 0.0:
        return
    n = wp.vec3(0.0, 0.0, 1.0)
    if dist > 1.0e-9:
        n = dvec / dist
    wa0 = inv_m[a0i] * (1.0 - s)
    wa1 = inv_m[a1i] * s
    wb0 = inv_m[b0i] * (1.0 - t)
    wb1 = inv_m[b1i] * t
    wsum = (1.0 - s) * wa0 + s * wa1 + (1.0 - t) * wb0 + t * wb1
    if wsum <= 0.0:
        return
    dlam = pen / wsum
    corr = n * dlam
    # 摩擦: 接触点どうしの相対的な接線方向の移動を打ち消す
    cap = x_prev[a0i] * (1.0 - s) + x_prev[a1i] * s
    cbp = x_prev[b0i] * (1.0 - t) + x_prev[b1i] * t
    rel = (ca - cap) - (cb - cbp)
    rel_t = rel - wp.dot(rel, n) * n
    lt = wp.length(rel_t)
    if lt < mu_s * pen:
        corr = corr - rel_t / wsum
    elif lt > 0.0:
        corr = corr - rel_t * (wp.min(mu_k * pen / lt, 1.0) / wsum)
    wp.atomic_add(dx_acc, a0i, wa0 * corr)
    wp.atomic_add(dx_acc, a1i, wa1 * corr)
    wp.atomic_add(dx_acc, b0i, -wb0 * corr)
    wp.atomic_add(dx_acc, b1i, -wb1 * corr)
    wp.atomic_add(cnt, a0i, 1.0)
    wp.atomic_add(cnt, a1i, 1.0)
    wp.atomic_add(cnt, b0i, 1.0)
    wp.atomic_add(cnt, b1i, 1.0)


@wp.kernel
def _k_apply(x: wp.array(dtype=wp.vec3), dx_acc: wp.array(dtype=wp.vec3), cnt: wp.array(dtype=float)):
    i = wp.tid()
    c = cnt[i]
    d = dx_acc[i]
    if c > 1.0:
        d = d / c
    x[i] = x[i] + d
    dx_acc[i] = wp.vec3(0.0, 0.0, 0.0)
    cnt[i] = 0.0


@wp.kernel
def _k_zero_vec3(a: wp.array(dtype=wp.vec3)):
    a[wp.tid()] = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def _k_zero_float(a: wp.array(dtype=float)):
    a[wp.tid()] = 0.0


@wp.kernel
def _k_apply_finger(
    x: wp.array(dtype=wp.vec3),
    dx_n: wp.array(dtype=wp.vec3),
    dx_t: wp.array(dtype=wp.vec3),
    cnt_t: wp.array(dtype=float),
):
    i = wp.tid()
    c = cnt_t[i]
    d = dx_n[i]
    if c > 0.0:
        d = d + dx_t[i] / c
    x[i] = x[i] + d
    dx_n[i] = wp.vec3(0.0, 0.0, 0.0)
    dx_t[i] = wp.vec3(0.0, 0.0, 0.0)
    cnt_t[i] = 0.0


@wp.kernel
def _k_interp_hand(
    hand_prev: wp.array(dtype=wp.vec4),
    hand_now: wp.array(dtype=wp.vec4),
    fing_prev: wp.array(dtype=wp.vec2),
    fing_now: wp.array(dtype=wp.vec2),
    a0: float,
    a1: float,
    teleport: float,
    hand0: wp.array(dtype=wp.vec4),
    hand1: wp.array(dtype=wp.vec4),
    fing0: wp.array(dtype=wp.vec2),
    fing1: wp.array(dtype=wp.vec2),
):
    e = wp.tid()
    hp = hand_prev[e]
    hn = hand_now[e]
    two_pi = 6.283185307179586
    dyaw = hn[3] - hp[3] + 3.141592653589793
    dyaw = dyaw - two_pi * wp.floor(dyaw / two_pi) - 3.141592653589793
    # 1 物理ステップで `teleport` [m] より大きく動いた手は瞬間移動として扱い、小ステップに補間しない
    # (`XpbdRodParams.hand_teleport_distance`)
    b0 = a0
    b1 = a1
    if teleport > 0.0:
        if wp.length(wp.vec3(hn[0] - hp[0], hn[1] - hp[1], hn[2] - hp[2])) > teleport:
            b0 = 1.0
            b1 = 1.0
    hand0[e] = wp.vec4(
        hp[0] + b0 * (hn[0] - hp[0]), hp[1] + b0 * (hn[1] - hp[1]), hp[2] + b0 * (hn[2] - hp[2]), hp[3] + b0 * dyaw
    )
    hand1[e] = wp.vec4(
        hp[0] + b1 * (hn[0] - hp[0]), hp[1] + b1 * (hn[1] - hp[1]), hp[2] + b1 * (hn[2] - hp[2]), hp[3] + b1 * dyaw
    )
    fing0[e] = fing_prev[e] + a0 * (fing_now[e] - fing_prev[e])
    fing1[e] = fing_prev[e] + a1 * (fing_now[e] - fing_prev[e])


@wp.kernel
def _k_copy_hand(
    hand_prev: wp.array(dtype=wp.vec4),
    hand_now: wp.array(dtype=wp.vec4),
    fing_prev: wp.array(dtype=wp.vec2),
    fing_now: wp.array(dtype=wp.vec2),
):
    e = wp.tid()
    hand_prev[e] = hand_now[e]
    fing_prev[e] = fing_now[e]


@wp.kernel
def _k_update_vel(
    x: wp.array(dtype=wp.vec3), x_prev: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3), inv_h: float
):
    i = wp.tid()
    v[i] = (x[i] - x_prev[i]) * inv_h


@wp.kernel
def _k_update_omega(
    q: wp.array(dtype=wp.quat), q_prev: wp.array(dtype=wp.quat), omega: wp.array(dtype=wp.vec3), inv_h: float
):
    i = wp.tid()
    dq = q[i] * wp.quat_inverse(q_prev[i])
    w = wp.vec3(dq[0], dq[1], dq[2])
    if dq[3] < 0.0:
        w = -w
    omega[i] = 2.0 * inv_h * w


@wp.kernel
def _k_damp_pairs_omega(omega: wp.array(dtype=wp.vec3), c: float, parity: int, n_seg: int, n_half: int):
    tid = wp.tid()
    e = tid // n_half
    j = 2 * (tid % n_half) + parity
    if j >= n_seg - 1:
        return
    ia = e * n_seg + j
    ib = ia + 1
    rel = omega[ib] - omega[ia]
    omega[ia] = omega[ia] + 0.5 * c * rel
    omega[ib] = omega[ib] - 0.5 * c * rel


@wp.kernel
def _k_damp_pairs_vel(v: wp.array(dtype=wp.vec3), c: float, color: int, n_nodes: int, n_cons: int):
    """mass_spring の曲げの減衰: 1 つ飛ばしの節点どうしの相対速度を減らす。"""
    tid = wp.tid()
    e = tid // n_cons
    k = tid % n_cons
    if k + 2 >= n_nodes or (k // 2) % 2 != color:
        return
    ia = e * n_nodes + k
    ib = ia + 2
    rel = v[ib] - v[ia]
    v[ia] = v[ia] + 0.5 * c * rel
    v[ib] = v[ib] - 0.5 * c * rel


@wp.kernel
def _k_damp_bend_vel(v: wp.array(dtype=wp.vec3), inv_m: wp.array(dtype=float), c: float, color: int, n_nodes: int):
    """曲げの減衰: 節点 i の速度の 2 階差分 b = v[i-1] - 2 v[i] + v[i+1] を割合 c だけ減らす。

    b は剛体の並進・回転では 0 で、紐が曲がる速さだけを表す。補正 (1, -2, 1) の和は 0 なので運動量を
    変えない。i を 3 色に塗り (i mod 3)、同じ色の補正が節点を共有しないようにする。
    PhysX 版の「関節ごとの臨界減衰」(`rope_specs.BEND_DAMPING_RATIO`) に当たる。
    """
    tid = wp.tid()
    e = tid // n_nodes
    i = tid % n_nodes
    if i < 1 or i >= n_nodes - 1 or i % 3 != color:
        return
    a = e * n_nodes + i - 1
    m = a + 1
    b_ = a + 2
    wa = inv_m[a]
    wm = inv_m[m]
    wb = inv_m[b_]
    wsum = wa + 4.0 * wm + wb
    if wsum <= 0.0:
        return
    bend = v[a] - 2.0 * v[m] + v[b_]
    # 逆質量で重み付けする (固定した節点は動かさない。一様な質量なら補正は (1, -2, 1) x c b / 6)
    lam = bend * (c / wsum)
    v[a] = v[a] - wa * lam
    v[m] = v[m] + 2.0 * wm * lam
    v[b_] = v[b_] - wb * lam


# ---------------------------------------------------------------- ソルバ本体


class XpbdRod:
    """全環境の紐をまとめて解く XPBD ソルバ。

    使い方 (1 物理ステップ = `step()` 1 回):

        rod = XpbdRod(num_envs, XpbdRodParams(), device="cuda:0")
        rod.reset_straight(env_ids, origin_xyz, height)
        rod.set_hand(hand_xyz_yaw, finger_q)   # 物理ステップの終わりの手と指 (始めは前回の値)
        rod.step(1 / 240)
        rod.x.view(E, P, 3)                     # 節点 (world)
    """

    def __init__(self, num_envs: int, params: XpbdRodParams, device: str = "cuda:0"):
        if params.model not in ("cosserat", "mass_spring"):
            raise ValueError(f"unknown model {params.model!r}")
        self.p = params
        self.E = num_envs
        self.N = params.num_segments
        self.P = params.num_nodes
        self.device = device
        self.torch_device = torch.device(device)
        dev = self.torch_device
        f32 = torch.float32
        E, N, P = self.E, self.N, self.P
        # -- 状態 (torch が持ち、Warp は同じメモリを見る)
        self.x = torch.zeros(E * P, 3, device=dev, dtype=f32)
        self.x_prev = torch.zeros_like(self.x)
        self.v = torch.zeros_like(self.x)
        self.q = torch.zeros(E * N, 4, device=dev, dtype=f32)  # (x, y, z, w)
        self.q[:, 3] = 1.0
        self.q_prev = self.q.clone()
        self.omega = torch.zeros(E * N, 3, device=dev, dtype=f32)
        m_seg = params.segment_mass
        m_node = torch.full((P,), m_seg, device=dev, dtype=f32)
        m_node[0] = m_node[-1] = 0.5 * m_seg
        self.inv_m = (1.0 / m_node).repeat(E)
        inertia = params.inertia_scale * m_seg * (params.radius**2 / 4.0 + params.segment_length**2 / 12.0)
        self.inv_i = 1.0 / inertia
        self.dx_acc = torch.zeros_like(self.x)
        self.cnt = torch.zeros(E * P, device=dev, dtype=f32)
        # XPBD の乗数 (小ステップごとに 0 に戻し、反復で積み上げる)
        self.lam_bend = torch.zeros(E * max(N - 1, 1), 3, device=dev, dtype=f32)
        self.lam_stretch = torch.zeros(E * P, device=dev, dtype=f32)
        self.lam_spring = torch.zeros(E * P, device=dev, dtype=f32)
        self.dx_fn = torch.zeros_like(self.x)
        self.dx_ft = torch.zeros_like(self.x)
        self.cnt_ft = torch.zeros(E * P, device=dev, dtype=f32)
        # -- 手 (x, y, z, yaw) と指の関節位置 (左, 右): 物理ステップの始めと終わり
        self.hand_prev = torch.zeros(E, 4, device=dev, dtype=f32)
        self.hand_now = torch.zeros(E, 4, device=dev, dtype=f32)
        self.hand_prev[:, 2] = self.hand_now[:, 2] = 10.0  # 何も指示が無いときは紐から遠く
        self.fing_prev = torch.zeros(E, 2, device=dev, dtype=f32)
        self.fing_now = torch.zeros(E, 2, device=dev, dtype=f32)
        self.hand_sub0 = torch.zeros_like(self.hand_now)
        self.hand_sub1 = torch.zeros_like(self.hand_now)
        self.fing_sub0 = torch.zeros_like(self.fing_now)
        self.fing_sub1 = torch.zeros_like(self.fing_now)
        # -- 自己接触を見る区間の組
        l0 = params.segment_length
        excl = params.self_contact_exclude_arc if params.self_contact_exclude_arc >= 0.0 else math.pi * params.radius
        # 区間 i と i+g は、まっすぐなときに (g-1) 区間ぶん離れている。それが直径以下だと、まっすぐな
        # 紐でも接触の判定に入り続けて伸びの拘束と競合する (`rope_specs.HALF_LENGTH_TO_RADIUS` と同じ問題)。
        # 直径の 1.25 倍を超えるまで除外する。
        geom_gap = int(math.floor(1.25 * 2.0 * params.radius / l0)) + 2
        min_gap = max(2, int(math.ceil(excl / l0 - 1e-9)), geom_gap)
        ii, jj = torch.triu_indices(N, N, offset=min_gap)
        self.pair_a = ii.to(dev, torch.int32).contiguous()
        self.pair_b = jj.to(dev, torch.int32).contiguous()
        self.n_pairs = int(ii.numel())
        self.min_gap = min_gap
        n_samp = params.finger_samples_per_segment
        if n_samp <= 0:
            n_samp = max(2, int(math.ceil(l0 / (0.5 * params.radius))))
        self.n_samp = n_samp
        # 指のばね定数は「標本点 1 つあたり」に配る。指の幅 (x) に入る標本点の数で割り、指 1 本の力が
        # PhysX 版と同じ「押し込み x k_finger」になるようにする。
        samples_under_finger = max(1.0, 2.0 * params.finger_box_half[0] / (l0 / n_samp))
        self.finger_k_per_sample = params.finger_stiffness / samples_under_finger
        # -- Warp のビュー
        self._wp = {}
        self._graph = None
        self._graph_h = None
        self._stream = None

    # ------------------------------------------------------------------ Warp
    def _views(self):
        if not self._wp:
            v3 = lambda t: wp.from_torch(t, dtype=wp.vec3)  # noqa: E731
            self._wp = {
                "x": v3(self.x),
                "x_prev": v3(self.x_prev),
                "v": v3(self.v),
                "q": wp.from_torch(self.q, dtype=wp.quat),
                "q_prev": wp.from_torch(self.q_prev, dtype=wp.quat),
                "omega": v3(self.omega),
                "inv_m": wp.from_torch(self.inv_m),
                "dx_acc": v3(self.dx_acc),
                "lam_bend": v3(self.lam_bend),
                "lam_stretch": wp.from_torch(self.lam_stretch),
                "lam_spring": wp.from_torch(self.lam_spring),
                "cnt": wp.from_torch(self.cnt),
                "dx_fn": v3(self.dx_fn),
                "dx_ft": v3(self.dx_ft),
                "cnt_ft": wp.from_torch(self.cnt_ft),
                "hand_prev": wp.from_torch(self.hand_prev, dtype=wp.vec4),
                "hand_now": wp.from_torch(self.hand_now, dtype=wp.vec4),
                "fing_prev": wp.from_torch(self.fing_prev, dtype=wp.vec2),
                "fing_now": wp.from_torch(self.fing_now, dtype=wp.vec2),
                "pa": wp.from_torch(self.pair_a, dtype=wp.int32),
                "pb": wp.from_torch(self.pair_b, dtype=wp.int32),
                "hand0": wp.from_torch(self.hand_sub0, dtype=wp.vec4),
                "hand1": wp.from_torch(self.hand_sub1, dtype=wp.vec4),
                "fing0": wp.from_torch(self.fing_sub0, dtype=wp.vec2),
                "fing1": wp.from_torch(self.fing_sub1, dtype=wp.vec2),
            }
        return self._wp

    def _launch_substep(self, h: float, sub: int, n_sub: int):
        """小ステップ 1 回ぶんのカーネルを並べる (1 物理ステップぶんまとめて CUDA graph に取り込まれる)。"""
        p, W = self.p, self._views()
        E, N, P = self.E, self.N, self.P
        dev = self.device
        wp.launch(
            _k_interp_hand,
            E,
            [
                W["hand_prev"],
                W["hand_now"],
                W["fing_prev"],
                W["fing_now"],
                sub / n_sub,
                (sub + 1) / n_sub,
                p.hand_teleport_distance,
                W["hand0"],
                W["hand1"],
                W["fing0"],
                W["fing1"],
            ],
            device=dev,
        )
        l0 = p.segment_length
        damp = math.exp(-p.linear_damping * h)
        mu_k = lambda mu: mu * p.dynamic_friction_ratio  # noqa: E731
        wp.launch(
            _k_predict, E * P, [W["x"], W["x_prev"], W["v"], W["inv_m"], p.gravity, h, damp, p.max_speed], device=dev
        )
        if p.model == "cosserat":
            wp.launch(
                _k_predict_rot,
                E * N,
                [W["q"], W["q_prev"], W["omega"], h, math.exp(-p.angular_damping * h)],
                device=dev,
            )
            wp.launch(_k_zero_vec3, self.lam_bend.shape[0], [W["lam_bend"]], device=dev)
            n_half = (N + 1) // 2
            a_bend = 1.0 / (p.bend_stiffness * l0) / h**2
            a_twist = 1.0 / (max(p.twist_stiffness, 1e-12) * l0) / h**2
            for _ in range(p.iterations):
                for par in (0, 1):
                    wp.launch(
                        _k_stretch_shear,
                        E * n_half,
                        [
                            W["x"],
                            W["q"],
                            W["inv_m"],
                            self.inv_i,
                            l0,
                            p.stretch_compliance / h**2,
                            p.shear_compliance / h**2,
                            par,
                            N,
                            n_half,
                        ],
                        device=dev,
                    )
                for par in (0, 1):
                    wp.launch(
                        _k_bend_twist,
                        E * n_half,
                        [W["q"], W["lam_bend"], self.inv_i, l0, a_bend, a_twist, par, N, n_half],
                        device=dev,
                    )
        else:
            a_spring = 1.0 / p.spring_bend_stiffness / h**2
            wp.launch(_k_zero_float, E * P, [W["lam_stretch"]], device=dev)
            wp.launch(_k_zero_float, E * P, [W["lam_spring"]], device=dev)
            for _ in range(p.iterations):
                for color in (0, 1):
                    wp.launch(
                        _k_distance,
                        E * P,
                        [W["x"], W["inv_m"], W["lam_stretch"], l0, p.stretch_compliance / h**2, 1, color, P, P],
                        device=dev,
                    )
                for color in (0, 1):
                    wp.launch(
                        _k_distance,
                        E * P,
                        [W["x"], W["inv_m"], W["lam_spring"], 2.0 * l0, a_spring, 2, color, P, P],
                        device=dev,
                    )
        # 自己接触 (剛、平均)
        if self.n_pairs > 0:
            wp.launch(
                _k_self,
                E * self.n_pairs,
                [
                    W["x"],
                    W["x_prev"],
                    W["inv_m"],
                    W["pa"],
                    W["pb"],
                    self.n_pairs,
                    N,
                    p.radius,
                    p.friction_self,
                    mu_k(p.friction_self),
                    W["dx_acc"],
                    W["cnt"],
                ],
                device=dev,
            )
            wp.launch(_k_apply, E * P, [W["x"], W["dx_acc"], W["cnt"]], device=dev)
        # 指 (法線は compliance 付きで足し合わせ、摩擦は節点ごとに平均)。自己接触の後に解く:
        # 先に解くと、掴んだ点の下で接する紐どうしの補正が摩擦の後に入り、毎小ステップ少しずつ下へずれて
        # (クリープ)、最後は指先の角から抜け落ちた (bench_physics の hang 試験、2026-10-02)
        a_f = 1.0 / self.finger_k_per_sample / h**2
        he = wp.vec3(*p.finger_box_half)
        wp.launch(
            _k_finger,
            E * N * self.n_samp,
            [
                W["x"],
                W["x_prev"],
                W["inv_m"],
                W["hand0"],
                W["hand1"],
                W["fing0"],
                W["fing1"],
                he,
                p.finger_center_y0,
                p.finger_center_z,
                p.radius,
                a_f,
                p.friction_finger,
                mu_k(p.friction_finger),
                N,
                self.n_samp,
                int(p.finger_x_faces),
                p.finger_x_face_margin,
                int(p.finger_rigid_outside),
                p.finger_max_depenetration_velocity * h,
                W["dx_fn"],
                W["dx_ft"],
                W["cnt_ft"],
            ],
            device=dev,
        )
        wp.launch(_k_apply_finger, E * P, [W["x"], W["dx_fn"], W["dx_ft"], W["cnt_ft"]], device=dev)
        # 床 (最後に解いて床下へ抜けないようにする)
        wp.launch(
            _k_ground, E * P, [W["x"], W["x_prev"], p.radius, p.friction_ground, mu_k(p.friction_ground)], device=dev
        )
        wp.launch(_k_update_vel, E * P, [W["x"], W["x_prev"], W["v"], 1.0 / h], device=dev)
        c = 1.0 - math.exp(-p.bend_damping * h)
        for color in (0, 1, 2):
            wp.launch(_k_damp_bend_vel, E * P, [W["v"], W["inv_m"], c, color, P], device=dev)
        if p.model == "cosserat":
            wp.launch(_k_update_omega, E * N, [W["q"], W["q_prev"], W["omega"], 1.0 / h], device=dev)
            n_half = (N + 1) // 2
            for par in (0, 1):
                wp.launch(_k_damp_pairs_omega, E * n_half, [W["omega"], c, par, N, n_half], device=dev)

    def _launch_step(self, h: float, n_sub: int):
        for sub in range(n_sub):
            self._launch_substep(h, sub, n_sub)
        W = self._views()
        wp.launch(
            _k_copy_hand, self.E, [W["hand_prev"], W["hand_now"], W["fing_prev"], W["fing_now"]], device=self.device
        )

    def step(self, dt: float):
        """1 物理ステップ進める。手と指は `set_hand` で渡した始め -> 終わりを小ステップごとに線形補間する。

        CUDA では 1 物理ステップぶん (小ステップ x n_sub) を 1 つの CUDA graph に取り込み、Warp 専用の
        stream で流す (CUDA graph はレガシーの既定 stream では取り込めない)。torch (Isaac Lab は既定 stream)
        との順序は、stream どうしのイベント待ちで保証する (ホストは止めない)。
        """
        n_sub = self.p.substeps
        h = dt / n_sub
        if self.torch_device.type != "cuda":
            self._launch_step(h, n_sub)
            return
        if self._stream is None:
            self._stream = wp.Stream(self.device)
            self._torch_stream = wp.stream_from_torch(torch.cuda.current_stream(self.torch_device))
        self._stream.wait_stream(self._torch_stream)
        if self._graph is None or self._graph_h != h:
            with wp.ScopedStream(self._stream):
                wp.load_module(device=self.device)
                wp.capture_begin(device=self.device, stream=self._stream)
                try:
                    self._launch_step(h, n_sub)
                finally:
                    self._graph = wp.capture_end(device=self.device, stream=self._stream)
                self._graph_h = h
        wp.capture_launch(self._graph, stream=self._stream)
        self._torch_stream.wait_stream(self._stream)

    # ------------------------------------------------------------------ 入出力
    def set_hand(self, hand_xyz_yaw: torch.Tensor, finger_q: torch.Tensor):
        """この物理ステップの終わりの手 `(E, 4)` (x, y, z, yaw) と指の関節位置 `(E, 2)` (左, 右)。"""
        self.hand_now.copy_(hand_xyz_yaw)
        self.fing_now.copy_(finger_q)

    def sync_hand(self, env_ids: torch.Tensor | None = None):
        """手の「始め」を「終わり」に揃える (reset 直後に手が瞬間移動したとき、補間で紐を掃かないように)。"""
        if env_ids is None:
            self.hand_prev.copy_(self.hand_now)
            self.fing_prev.copy_(self.fing_now)
        else:
            self.hand_prev[env_ids] = self.hand_now[env_ids]
            self.fing_prev[env_ids] = self.fing_now[env_ids]

    def nodes(self) -> torch.Tensor:
        """節点 `(E, P, 3)` (world)。"""
        return self.x.view(self.E, self.P, 3)

    def node_vel(self) -> torch.Tensor:
        return self.v.view(self.E, self.P, 3)

    def quat_wxyz(self) -> torch.Tensor:
        """区間の向き `(E, N, 4)` を (w, x, y, z) で。"""
        q = self.q.view(self.E, self.N, 4)
        return torch.cat([q[..., 3:4], q[..., 0:3]], dim=-1)

    def set_state(
        self,
        env_ids: torch.Tensor,
        nodes: torch.Tensor,
        quat_wxyz: torch.Tensor | None = None,
        node_vel: torch.Tensor | None = None,
    ):
        """環境 `env_ids` の節点 `(n, P, 3)` と向き `(n, N, 4)` (w, x, y, z) を書き込む。速度は 0 (指定が無ければ)。

        向きを省くと中心線から捩れ 0 の枠を作る (`centerline.parallel_transport_frames`)。
        """
        from .centerline import parallel_transport_frames

        E, N, P = self.E, self.N, self.P
        env_ids = env_ids.to(self.torch_device, torch.long)
        xs = self.x.view(E, P, 3)
        xs[env_ids] = nodes.to(xs.dtype)
        self.x_prev.view(E, P, 3)[env_ids] = nodes.to(xs.dtype)
        vs = self.v.view(E, P, 3)
        vs[env_ids] = 0.0 if node_vel is None else node_vel.to(vs.dtype)
        if quat_wxyz is None:
            quat_wxyz = parallel_transport_frames(nodes.to(torch.float32))
        q_xyzw = torch.cat([quat_wxyz[..., 1:4], quat_wxyz[..., 0:1]], dim=-1)
        self.q.view(E, N, 4)[env_ids] = q_xyzw.to(self.q.dtype)
        self.q_prev.view(E, N, 4)[env_ids] = q_xyzw.to(self.q.dtype)
        self.omega.view(E, N, 3)[env_ids] = 0.0

    def straight_nodes(self, origins: torch.Tensor, height: float) -> torch.Tensor:
        """x 軸に沿ったまっすぐな紐の節点 `(n, P, 3)`。中心を `origins` の真上 `height` に置く。"""
        l0 = self.p.segment_length
        s = (torch.arange(self.P, device=self.torch_device, dtype=torch.float32) - 0.5 * self.N) * l0
        pts = torch.zeros(origins.shape[0], self.P, 3, device=self.torch_device)
        pts[..., 0] = s
        pts[..., 2] = height
        return pts + torch.stack([origins[:, 0], origins[:, 1], torch.zeros_like(origins[:, 2])], -1).unsqueeze(1)

    @property
    def num_state(self) -> int:
        """アーカイブに保存する状態の次元 (節点 3P + 向き 4N)。"""
        return 3 * self.P + 4 * self.N
