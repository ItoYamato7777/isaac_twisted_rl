"""pick&place 1 手を構成するフェーズ表と、その尺 (時間) の定義。

`demos/topology_debug.py` の `grasp_move()` が組み立てる `plan` (13 個の
フェーズを逐次実行するリスト) を、**RL で使えるよう定数側へ切り出した**もの。
デモ側は 1 環境を Python ループで回せばよかったが、RL では全環境が同時に
別々のフェーズにいるため、フェーズ列そのものは「テンソルで索引できる表」に
なっていなければならない。

## なぜ尺をデモより短くしているか

デモの尺 (合計 11.4 s) は **目視デバッグ用に余裕を持たせた値**で、
初版の `next_action.md` §4.3 に「フェーズの尺は詰められるだけ詰めること」と指示がある
(commit `5fd7e3c`。`git show 5fd7e3c:src/isaac_twisted_rl/tasks/knot_tying/next_action.md`。
今の `docs/history/next_action.md` からは整理で消えた)。1 手 = 11.4 s のままだと 240 Hz で 2736 物理ステップ、
1 エピソード 5 手で 13680 ステップになり、学習の実時間がここで決まってしまう。

そこで各フェーズをおおむね半分以下に詰めて合計 5.0 s とした (その後さらに
2.855 s まで詰めた。`PhaseDurations` の docstring)。詰めるときの
判断基準は「その尺が物理の落ち着きを待つためのものか、単に見やすさのためか」:

* `settle_out` (離した後ロープが落ち切るのを待つ) は **物理の待ち**なので
  比率を保って残す。ここを削ると `docs/decisions/0004-release-follow-physics.md` の
  「離したのに付いてくる」が再発する。
* `approach` / `retreat` (何も掴んでいない移動) は見やすさのためなので大きく削る。

## `RELEASE_RATIO` について

デモの `RELEASE_RATIO` の docstring は「接近時の `OPEN_RATIO` より広く開ける」と
書いてあるが、`docs/decisions/0004-release-follow-physics.md` の対策で `OPEN_RATIO` が 1.0 に上がった結果
両者は同値になっている (初版の `next_action.md` (`5fd7e3c`) §3(c) にも記述と実態のずれとして指摘がある)。
ここでは定数を 1 つ (`OPEN_RATIO`) に統合し、その経緯をこの docstring に残す。
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------- 高さ・開度
# いずれも demos/topology_debug.py と同じ値。ハンドの形状 (指先が base_link
# 中心から 0.075 m 下) に紐づく量なので、勝手に変えると把持できなくなる。
GRASP_PALM_Z = 0.077
"""把持時の手のひら (root) 高さ [m]。指先が床上 0.002 m に来る。"""

HOME_PALM_Z = 0.30
"""待機・退避時の手のひら高さ [m]。"""

CARRY_PALM_Z = 0.20
"""運搬時の手のひら高さ [m]。"""

OPEN_RATIO = 1.0
"""指の開度: 全開 (0=全閉, 1=全開)。

デモでは接近時 `OPEN_RATIO` と解放時 `RELEASE_RATIO` を分けていたが、
`docs/decisions/0004-release-follow-physics.md` の対策で両方 1.0 になったので統合した。
全開にするのは、離した後ハンドを引き上げるときにロープが指に残って
一緒に持ち上がるのを防ぐため (実測でこれが交差を壊していた)。
"""

CLOSE_RATIO = 0.0
"""指の開度: 全閉。

「ロープの太さぶん閉じ切れない」状態を作ることで、PD 制御の定常偏差が
そのまま把持力になる (力制御を書かずに摩擦把持を得る仕掛け)。
"""

FINGER_TIP_OFFSET_Z = 0.075
"""root (base_link 中心) から指先までの距離 [m]。

`generate_gripper_usd.py` の `BASE_SIZE[2]/2 + FINGER_SIZE[2] = 0.015 + 0.06`。
把持整形報酬で「指先とロープの距離」を測るのに使う。
"""

FINGER_HALF_X = 0.01
"""指の直方体の半幅 (手のローカル x = ロープの接線方向) [m]。`FINGER_SIZE[0] / 2`。"""

FINGER_OPEN_Y = (0.030, 0.042)
"""全開のときに指が占めるローカル y の範囲 (片側, 絶対値) [m]。

`FINGER_Y_OFFSET + STROKE ± FINGER_SIZE[1] / 2 = 0.036 ± 0.006`。指を降ろしたときに
ぶつかる紐を探す範囲 (`finger_footprint_top`) に使う。
"""

# ---------------------------------------------------------------- フェーズ ID
# テンソルで索引するので、値は 0 から連番でなければならない。
APPROACH = 0
DESCEND = 1
CLOSE = 2
HOLD = 3
LIFT = 4
CARRY = 5
LOWER = 6
RELEASE = 7
SETTLE_OUT = 8
CLEAR = 9
RETREAT = 10
RELAX = 11
NUM_PHASES = 12

PHASE_NAMES = (
    "approach",
    "descend",
    "close",
    "hold",
    "lift",
    "carry",
    "lower",
    "release",
    "settle_out",
    "clear",
    "retreat",
    "relax",
)


@dataclass
class PhaseDurations:
    """各フェーズの尺 [s]。合計が「1 手 = 1 RL ステップ」の長さになる。

    デモ (`demos/topology_debug.py`) の値を括弧内に併記する。`carry` は
    デモでは経由点で等分される合計時間だったが、本実装は経由点を持たない
    (`docs/decisions/0002-action-space-rope-relative.md` で `via_*` を入れないと決めた) ので 1 区間ぶん。

    ## 移動系フェーズをさらに詰めた (2026-09-12 実測)

    1 イテレーションの実時間が極端に長い問題を調べたところ、1 手 = 5.0 s
    (240 Hz で 1200 物理ステップ) のうち **移動系フェーズ
    (approach/descend/lift/carry/lower/clear/retreat) だけで 3.3 s** を
    占めていた。これらはロープと接触しない純粋な移動区間なので、
    速度を上げても把持そのものには影響しないはずと考え、64 環境 x 20 手
    (=1280 回の手) で把持成功率・発散率・交差生成率を計測した
    (`scratchpad/duration_bench.py`、移動系フェーズのみ係数でスケール、
    物理待ち系は固定):

    | scale | 1 手合計 | 把持成功率 | 発散率 | 交差生成率 |
    |---|---|---|---|---|
    | 1.0 (旧デフォルト) | 5.00 s | 98.8% | 2.8% | 22.2% |
    | 0.7                | 4.01 s | 98.5% | 2.6% | 21.0% |
    | 0.5                | 3.35 s | 99.2% | 2.9% | 20.2% |
    | **0.35 (採用)**    | **2.86 s** | **99.0%** | 3.4% | 20.2% |

    scale=0.35 まで把持成功率・交差生成率はほぼ変化せず、発散率も
    誤差範囲内の微増 (2.8% -> 3.4%) に留まった。移動フェーズは
    「ロープと相互作用しない区間」なので、速度を上げても破綻しにくいのは
    自然な結果。これで **1 手が 5.0 s -> 2.86 s (約 43% 短縮)** になった。

    物理待ち系 (close/hold/release/settle_out/relax) は今回検証していない
    ため据え置き。ここは `docs/decisions/0004-release-follow-physics.md` の「離すと付いてくる」問題
    (`OPEN_RATIO` / `RELEASE_RATIO` の docstring 参照) に直結する箇所なので、
    削るなら同様の実測をしてから行うこと。
    """

    approach: float = 0.175  # 0.5 x 0.35 (2026-09-12 実測、上の docstring 参照)
    descend: float = 0.14  # 0.4 x 0.35
    close: float = 0.3  # (0.6) 握り込む。ロープに触れるフェーズなので未検証・据え置き
    hold: float = 0.2  # (0.4) 接触の安定待ち。同上
    lift: float = 0.14  # 0.4 x 0.35
    carry: float = 0.35  # 1.0 x 0.35。運搬 (最長だった移動区間)
    lower: float = 0.14  # 0.4 x 0.35
    release: float = 0.25  # (0.5) 指を開く。ロープに触れるフェーズなので未検証・据え置き
    settle_out: float = 0.3  # (0.6) 離した後ロープが落ち切るのを待つ (物理の待ち)。同上
    clear: float = 0.105  # 0.3 x 0.35
    retreat: float = 0.105  # 0.3 x 0.35
    relax: float = 0.65  # (1.5) ロープが落ち着くのを待つ (報酬を測る直前)。同上

    def as_tuple(self) -> tuple[float, ...]:
        """フェーズ ID の順に並べた尺のタプル。"""
        return (
            self.approach,
            self.descend,
            self.close,
            self.hold,
            self.lift,
            self.carry,
            self.lower,
            self.release,
            self.settle_out,
            self.clear,
            self.retreat,
            self.relax,
        )

    @property
    def total(self) -> float:
        """1 手にかかる時間 [s]。"""
        return sum(self.as_tuple())

    def step_counts(self, sim_dt: float) -> tuple[int, ...]:
        """各フェーズの物理ステップ数。各フェーズ最低 1 ステップは確保する。"""
        return tuple(max(int(round(d / sim_dt)), 1) for d in self.as_tuple())

    def decimation(self, sim_dt: float) -> int:
        """1 手を実行し切るのに必要な物理ステップ数 (= `env.cfg.decimation`)。

        `total / sim_dt` を丸めるのではなく **各フェーズの丸め後の合計**を返す。
        丸め誤差でフェーズ列が終わり切らない (最後の `relax` が途中で切れて
        ハンドがロープの真上に居るまま報酬を測る) 事故を防ぐため。
        """
        return sum(self.step_counts(sim_dt))


DEFAULT_DURATIONS = PhaseDurations()
"""既定の尺。`KnotTyingEnvCfg.__post_init__` が decimation の計算に使う。"""


@dataclass
class GraspLimits:
    """アクションの各チャネルを物理量へ変換するときの範囲。

    値の根拠は `docs/decisions/0002-action-space-rope-relative.md`。ロープ全長 `L` と直径に対する比で
    持つのは、プリセットによって寸法が違う (simple/stiff/twist は
    L=0.82 m / dia=20 mm、fine は L=0.78 m / dia=8 mm、knot は L=1.14 m / dia=20 mm) ため。
    """

    max_delta_link: int = 8
    """`delta_link` の絶対値の上限。`{-8, ..., +8}` の 17 通り。"""

    drop_xy_rel_frac: float = 0.15
    """`drop_x_rel` / `drop_y_rel` の上限 (ロープ全長 L に対する比)。"""

    drop_z_diameters: float = 3.0
    """`drop_z_rel` の上限 (ロープ直径に対する比)。"""

    clear_diameters: float = 4.0
    """`clear` フェーズで横へ抜ける距離 (ロープ直径に対する比)。

    デモの `clear_dist = 4.0 * rope_spec.diameter` と同じ。
    """

    edge_margin: int = 2
    """掴めるリンクを両端から何個ぶん内側に制限するか。

    端ちょうどを掴むと接触面積が足りず滑る (デモの
    `max(2, min(link_id, num_bodies - 3))` と同じ理由)。
    """

    # ---- v5 の発散対策 (既定値は従来の挙動。`versions/v5/README.md`)
    release_tip_z: tuple[float, float] | None = None
    """離すときの指先の高さの範囲 `(下限, 上限)` [m, 床から]。None なら従来どおり。

    従来は `drop_z` チャネルを「把持と同じ高さ (指先が床上 2 mm) から直径の 3 倍上まで」に
    写していた。指先を床すれすれまで降ろす手は、掴んだ紐を床や下の紐へ押し込む。ハンドは
    位置を直接書き込んでいて力が効かないので、めり込みの解消がすべて紐の側にかかり、
    ここで発散していた (v5 README の計測)。範囲を指定すると、`drop_z` はこの範囲に写り、
    紐は床から浮いた高さで離されて落ちる。
    """

    grasp_clearance: float | None = None
    """指の真下に **浮いた紐** があるとき、その上端からどれだけ離して指先を止めるか [m]。None なら見ない。

    指先を「開いた指 2 本の真下にある紐の上端 (`finger_footprint_top`) + `grasp_clearance`」より
    下へは降ろさない。見る紐は、中心が `grasp_obstacle_min_z` より高いもの (= 床ではなく別の紐の
    上に乗っている紐) だけ。床の上の紐は、指で押されても横へ逃げられるので発散しにくい
    (v5 README の計測: 床の上の紐が指の下にあっても発散率は 4.1% -> 6.5% にしか上がらないが、
    浮いた紐だと 10〜13%)。床の上の紐まで避けると、掴めない手が大きく増える。

    **v5 では使っていない。** 結び目の近くでは掴みたいリンクの横を浮いた紐が跨いでいるのが普通で、
    学習済み方策の手の半分以上の把持が失敗した (v5 README の「掴む高さの対策 (C) を入れなかった理由」)。
    比較の計測を再現できるように項目は残してある。
    """

    grasp_obstacle_min_z: float = 0.0
    """`grasp_clearance` で避ける紐の、中心の高さの下限 [m]。0 なら床の上の紐も避ける。"""


SHORT_DURATIONS = PhaseDurations(
    approach=0.0,
    clear=0.0,
    relax=0.5,
)
"""v2 以降で使う短縮版の尺 (合計 2.4375 s = 585 物理ステップ @240 Hz)。

`DEFAULT_DURATIONS` (2.855 s / 686 ステップ) から **14.7% 短縮**した。

尺 0 のフェーズは `step_counts()` の `max(..., 1)` により **1 物理ステップ**に
なる (0 ステップにはならない)。フェーズ列そのものは 12 個のまま保たれ、
`_begin_phase` の補間区間も従来どおり組まれるので、ステートマシンの構造を
変えずに「そのフェーズを瞬間的に通過する」挙動が得られる。
1 手が短くなるぶん、同じ実時間でより多くのサンプルが取れる。
v1 の学習が 1 イテレーション 185〜200 秒 x 4000 = 9 日超という
実時間の壁に当たっていたので、ここを削るのが最も効く
(`docs/history/next_action.md` §4.8.9 の「他のどの最適化よりも効く」)。

## 何を 0 にしたか、なぜ安全か

* **`approach` = 0.0** (0.175 s 削減)。
  掴む点の真上へ横移動するフェーズ。**ロープに触れない純粋な空中移動**で、
  しかも手は前の手の `retreat` 後 (= `HOME_PALM_Z` = 0.30 m) にいる。
  0 にすると `PickPlaceAction._begin_phase` が尺 0 のフェーズを
  1 ステップで通過し、実質「次の掴む点の真上へ瞬間移動する」ことになる。
  ロープの遥か上 (0.30 m、ロープ直径の 15 倍) での瞬間移動なので、
  ロープとの接触は発生しない。

  **これが v1 の発散対策の本命**でもある。`approach` は前の手の終了位置から
  次の掴む点まで最も長い距離を動くフェーズで、0.175 s に短縮された結果
  指令速度が最大になっていた区間 (`actions.py` の 2026-09-12 のコメントに
  ある「平均 78 m/s」事故と同じ経路)。ここを移動ではなく
  位置の書き換えにすれば、その速度指令自体が消える。

* **`clear` = 0.0** (0.105 s 削減)。
  離した後に運搬方向へ横へ抜けるフェーズ。「指に残って一緒に持ち上がる」
  対策として入れたものだが、その後 `OPEN_RATIO` が 1.0 (全開) に
  上がっており (この module 冒頭の docstring 参照)、全開なら横抜けなしでも
  指から外れる。`settle_out` (0.3 s、全開で静止) がそのまま残るので、
  ロープが落ち着く時間も確保されている。
  なお `clear` を 0 にしても `_clear_pos` の計算自体は残り、
  `retreat` の終点 (= 次の手の始点) として使われ続ける。

* **`relax` = 0.65 -> 0.5** (0.15 s 削減)。
  報酬を測る直前の静定待ち。`settle_out` (0.3 s) と合わせて
  **離してから合計 0.8 s** 待つことになり、物理の落ち着きは確保される。
  ここは「報酬の測定を汚さない」ための待ちなので 0 にはしない。

## 触っていないもの

`close` / `hold` / `release` / `settle_out` は **接触・物理待ち系**で、
`PhaseDurations` の docstring にある 2026-09-12 のベンチマークでも
「未検証・据え置き」とされている区間。把持成功率と
「離すと付いてくる」問題に直結するのでそのまま残す。
`descend` / `lift` / `carry` / `lower` / `retreat` は移動系だが、
既に scale=0.35 まで詰めてあり、これ以上削ると把持中 (= ロープを
掴んだまま) の指令速度が上がって発散を招く。
"""
