"""v2: 固結び (overhand knot) を作る。v1 の学習失敗を受けた作り直し。

    uv run train --task Isaac-Twisted-RL-Twist-v2 --headless

v1 (同じ固結び) からの変更点と、その理由は `README.md` を参照。
ここは「何を配線したか」だけを書く。

シーン・物理・行動・終了判定は `common/env_cfg_base.py` と共通。
このファイルが定義するのは **観測・報酬・エピソード長・フェーズの尺**だけ。
"""

from __future__ import annotations

from dataclasses import MISSING

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from ...common import mdp
from ...common.env_cfg_base import KnotTyingEnvCfg
from ...common.mdp import phases
from ...common.rope_task_cfg import make_preset_cfgs

GAMMA = 0.95
"""割引率。**報酬整形と PPO で同じ値を使う**ための唯一の定義。

ポテンシャル整形 `F = γΦ(s') − Φ(s)` が最適方策を変えないという保証
(Ng et al. 1999) は、この γ が方策の割引率と一致して初めて成り立つ。
`agents/rsl_rl_ppo_cfg.py` はここを import して使う。片方だけ書き換えて
ずれる事故を防ぐため、数値をそこに直接書かないこと。

## v1 の 0.98 から戻した理由

v1 は「地平が 5 -> 8 手に伸びたので終端報酬の減衰を抑える」ため 0.98 に
上げていた。v2 では 2 つの理由で 0.95 (v0 の値) に戻す:

1. **地平が 8 -> 6 手に縮んだ**ので、0.95 でも初手から見た終端報酬は
   0.95^5 ≈ 0.77 残る。減衰を心配する状況ではない。
2. **v1 の critic はターゲットに追いつけていなかった。** TensorBoard 実測で
   `Loss/value_function` が ~1099 までスパイクしている (v0 は 0.17〜2.6 の
   範囲に収まっていた)。しかも適応学習率が 5.1e-4 -> 1.32e-3 と
   **上げられ続けていた** (v0 は下限 1e-5 に張り付いていた) ので、
   critic が発散方向にあったことは間違いない。終端報酬が一度も発火しない
   状態で γ を上げると bootstrap がほぼ学習不能になる。
"""


@configclass
class ObservationsCfg:
    """観測。テンソル層のみ (p-data 文字列は使わない)。

    **v1 から一切変えていない (105 次元)。** 観測を変えなかったのは 2 つの理由:

    * v1 の失敗原因は実測上「疎報酬が発火しない」ことであって観測ではない
      (`README.md` の診断表)。同時に複数変えると原因を切り分けられない。
    * v3 で段別カリキュラム (`-v3-Stage1/2/3`) を組むとき、前段のモデルを
      `--resume` で暖機開始に使う。`OnPolicyRunner.load` は
      `load_state_dict` を **strict のまま** 呼ぶので
      (`rsl_rl/runners/on_policy_runner.py:310`)、観測次元とネットワーク形状が
      段の間で揃っていないと暖機開始そのものができない。
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """方策への観測 (v1 と同一)。

        `rope_link_pos` がロープの形、`crossings` / `writhe` /
        `crossing_feats` / `knot_class` が位相。幾何と位相の両方を渡すのは、
        位相だけでは「次にどこを掴めば結べるか」が決まらないため。
        """

        rope_link_pos = ObsTerm(func=mdp.rope_link_positions)
        crossings = ObsTerm(func=mdp.crossing_count)
        writhe = ObsTerm(func=mdp.writhe)
        crossing_feats = ObsTerm(func=mdp.crossing_features, params={"max_crossings": 5})
        knot_class = ObsTerm(func=mdp.knot_class_one_hot)
        hand_pos = ObsTerm(func=mdp.hand_position)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            # ノイズは入れない。位相量 (交差数・符号) は離散で、ノイズを
            # 乗せると意味が壊れる (1.2 個の交差、という状態は無い)。
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """報酬 (固結び / 3 段の疎報酬ラダー版)。

    **重みは `__post_init__` で `step_dt` で割られる**。Isaac Lab の
    `RewardManager.compute()` は各項に `dt` (= 1 env ステップの秒数) を
    掛ける仕様で、報酬を「毎秒の割合」として扱う設計になっている。
    ところがこの環境は 1 env ステップが 1 手 = 約 2.4 秒と極端に長いため、
    「1 手あたり +1.0」をそのまま書くと +2.4 になってしまう。

    そこで ここには **「1 手あたりの値」をそのまま書き**、
    `KnotTyingEnvCfg.__post_init__` が `step_dt` で割って辻褄を合わせる。

    ## v1 の失敗: 疎報酬が 1 つしか無く、それが一度も発火しなかった

    v1 は疎報酬を最上段 (`overhand_achieved`) だけにした。TensorBoard の
    実測 (157 iteration で打ち切り):

    * `Episode_Reward/success` は 0.0001 -> **厳密に 0.0**
      (1 手あたり 0.00005 回しか発火していない)
    * 終端の報酬構成は `failure` **−0.0924** が最大絶対値の項で、
      `shaping` (+0.0496) を **1.86 倍上回る**純負
    * その結果 `Train/mean_reward` は −0.009 -> −0.058 と**下降を続けた**
    * 報酬がマイナスしか払わないので、唯一プラスを払うエントロピー
      ボーナスの最大化が最適方策になり、`Policy/mean_noise_std` が
      1.003 -> **1.628**、`Loss/entropy` が 8.52 -> **11.39** と
      揃って上昇した (v0 は 0.998 -> 0.610 / 8.51 -> 2.08 と正しく収束)

    つまり「勾配を出せる到達可能な目標が一つも無かった」のが失敗の本質で、
    報酬ラダー (`knot_class_shaping`) や観測拡張が悪かったわけではない。

    ## v2: 到達可能性が実測されている 3 段を疎報酬として置く

    密報酬のラダー (`knot_class_shaping`) はそのまま残し、**その各段に
    対応する疎報酬を重ねる**。3 段それぞれ「実際に発火する」根拠がある:

    | 段 | 項 | 重み | 発火の実測根拠 |
    |---|---|---|---|
    | 交差 1 つ | `success_c1` | +0.5 | **v0 が 1 手あたり 98.5% 発火で学習成功** |
    | 2 交差の絡み | `success_c2` | +1.5 | ランダム方策で 1.4% 到達 (v1 README §12) |
    | 固結び | `success` | +3.0 | ランダム方策で 512 回に 1 回 (疎すぎるので単独では足場にならない) |

    最下段を 1.0 ではなく 0.5 にしたのは、v0 でこの項が 1 手あたり 98.5% =
    ほぼ飽和していたため。飽和した項を強くすると「交差 1 つを作って維持する
    だけ」が局所最適になり、上の段へ登る動機が消える。
    """

    # 密報酬: 位相ラダーをポテンシャル整形で登らせる (v1 から据え置き)。
    # `gamma` はモジュール定数 `GAMMA` から取る (PPO 側と共有)。
    shaping = RewTerm(
        func=mdp.knot_class_shaping,
        weight=1.0,
        params={"gamma": GAMMA, "potentials": (0.0, 0.35, 0.70, 1.0)},
    )

    # 疎報酬 (最下段): 交差が 1 つでき、床に落ちているか。
    # **v0 が実証済みの目標**なので、v2 には必ず発火する学習信号が 1 つある。
    # 新規コードは不要で、v0 が使っていた既存項をそのまま再利用している
    # (`common/mdp/` は追加のみで削除・改名しない方針の恩恵)。
    success_c1 = RewTerm(
        func=mdp.crossing_achieved,
        weight=0.5,
        params={"min_crossings": 1, "settled_height_diameters": 3.0, "rope_diameter": MISSING},
    )

    # 疎報酬 (中段): 2 交差が絡み、床に落ちているか。
    # 「離しても輪が残るか」を問う最初の段 (最下段の交差 1 つは飽和する)。
    success_c2 = RewTerm(
        func=mdp.interleaved2_achieved,
        weight=1.5,
        params={"settled_height_diameters": 3.0, "rope_diameter": MISSING},
    )

    # 疎報酬 (最上段): 固結びができ、かつ床に落ちているか。
    # **ハンド退避後に測る**のが最重要 (rewards.py の docstring 参照)。
    # 属性名を `success` にしておくのは v1 との比較をログ上で揃えるため。
    success = RewTerm(
        func=mdp.overhand_achieved,
        weight=3.0,
        params={"settled_height_diameters": 3.0, "rope_diameter": MISSING},
    )

    # 罰則: 交差はあるがラダーに乗っていない (独立ツイストなど行き止まり)。
    offladder = RewTerm(func=mdp.offladder_penalty, weight=-0.05)

    # 整形: 狙ったリンクを実際に掴めたか。
    grasp_shaping = RewTerm(func=mdp.grasp_distance_penalty, weight=-0.05)

    # 罰則: 把持失敗 / NaN。**v1 の −0.2 から半分に再校正した。**
    #
    # v1 ではこの項が終端で −0.0924 に達し、整形報酬 (+0.0496) を 1.86 倍
    # 上回って報酬全体の符号を決めてしまっていた。しかも `action_failed` は
    # `invalid_action` と「ロープが NaN」の OR (`rewards.py`) なので、
    # 終了判定 `terminations.rope_diverged` と信号が二重に効いている。
    #
    # 0 にはしない。NaN 回避の信号自体は必要で、v0 は −0.2 のままでも
    # 発散を 13.9% -> 0.06% まで学習で潰せている。v0 と v1 の違いは
    # 「プラスを払う項があったか」なので、3 段の疎報酬を入れたうえで
    # 半分に弱めるのが妥当な再校正。
    failure = RewTerm(func=mdp.action_failed, weight=-0.1)

    # 罰則: 手の終わりにロープが浮いたまま (「吊っているだけ」の検出も兼ねる)。
    held_up = RewTerm(
        func=mdp.held_up_penalty,
        weight=-0.1,
        params={"height_diameters": 5.0, "rope_diameter": MISSING},
    )

    # ログ専用 (weight=0.0): class 別に層別した rope_diverged。
    #
    # `docs/history/next_action.md` §4.8.9 / `README.md` §6-2 のゲート。自己接触は交差数に
    # 対して超線形に増えるので、`Episode_Termination/diverged` (全 env 平均
    # 1 本の集計値) だけでは class 3 (固結び) でだけ発散が増えていても埋もれる。
    # `RewardManager` は項ごとに `Episode_Reward/<name>` を自動でログするので、
    # 重み 0 の項として登録するだけで class 別の内訳が TensorBoard に出る
    # (`rewards.diverged_by_class` の docstring 参照)。学習には寄与しない。
    diverged_class0 = RewTerm(func=mdp.diverged_by_class, weight=0.0, params={"knot_class": 0})
    diverged_class1 = RewTerm(func=mdp.diverged_by_class, weight=0.0, params={"knot_class": 1})
    diverged_class2 = RewTerm(func=mdp.diverged_by_class, weight=0.0, params={"knot_class": 2})
    diverged_class3 = RewTerm(func=mdp.diverged_by_class, weight=0.0, params={"knot_class": 3})


@configclass
class KnotTyingV2EnvCfg(KnotTyingEnvCfg):
    """v2 の環境設定。ロープのプリセット別の派生は下の `make_preset_cfgs`。"""

    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()

    # configclass は mutable default を deepcopy するので
    # (`isaaclab/utils/configclass.py` の `_process_mutable_types` ->
    # `_return_f` が `deepcopy(f)` を返す)、モジュール定数の singleton を
    # そのまま渡してもインスタンス間で共有されることはない。
    durations: phases.PhaseDurations = phases.SHORT_DURATIONS
    """1 手を構成するフェーズの尺。**v2 で短縮版に切り替えた** (686 -> 585 step)。

    `common/mdp/phases.py` の `SHORT_DURATIONS` (approach=0.0, clear=0.0,
    relax=0.5、合計 2.4375 s)。同 docstring に「v2 以降で使う」と書いてある
    ものをここで初めて配線した。

    ## `approach = 0.0` が「位置座標の変更による移動」になる仕組み

    `PhaseDurations.step_counts` は `max(int(round(d / sim_dt)), 1)` なので
    尺 0 のフェーズは **1 物理ステップ**になる。`PickPlaceAction.apply_actions`
    を 1 ステップだけ辿ると `dur = 1` -> `u = 1.0` -> smoothstep
    `s = 1·1·(3−2·1) = 1.0` なので、初回ステップで
    `pos = _seg_start + (_seg_end − _seg_start)·1.0 = _seg_end` となり、
    掴む点の真上 (`HOME_PALM_Z` = 0.30 m) へ **補間せず直接書き込まれる**。

    さらに速度指令は `ds = 6·u·(1−u)/seg_time` で `u = 1.0` なので
    **厳密に 0**。つまり「移動させる」のではなく「位置を書き換える」
    挙動そのものになる。

    ## なぜこれが v1 の発散対策になるか

    v1 は `Episode_Termination/diverged` が 0.250 -> **0.480** と増加し、
    終盤は **エピソードの約半分がロープの発散で終了**していた
    (ランダム方策の実測値 1.2% に対して 40 倍)。`approach` は前の手の
    終了位置から次の掴む点まで最も長い距離を動くフェーズで、
    2026-09-12 に「平均 78 m/s、最大 131 m/s」の非物理的な指令速度が
    実測されたのと同じ経路にある (`common/mdp/actions.py` の `reset()` の
    コメント)。ここを位置の書き換えにすれば、その速度指令自体が消える。

    ロープ直径の 15 倍の高さ (0.30 m) でのテレポートなので、
    瞬間移動してもロープとの接触は発生しない。**この高度が安全性の根拠**
    であり、同じことを把持高さでやってはいけない。

    実時間も 1 手あたり 14.7% 短くなる。
    """

    actions_per_episode: int = 6
    """1 エピソードで打てる手の数 (v0 は 5、v1 は 8)。

    固結びの位相的な下限は 3 遷移 (移植元のレシピ
    `Reide1 -> cross -> cross`)。物理的には 1 遷移あたり約 2 手かかる —
    Step 5 の台本も **1 交差を作るのに 2 手** 使っている (1 手目でたるみを
    作り、2 手目で渡す)。3 x 2 = 6 手がこの下限そのもの。

    ## v1 の 8 手から減らした理由

    v1 は「把持失敗や発散からの復帰ぶん」として 6 手に 2 手足していたが、
    実測ではその余裕が裏目に出ている。1 エピソードが長いほど 1 手あたりの
    信用割り当てが薄まるうえ、実時間も比例して伸びる。

    `SHORT_DURATIONS` (585 step) と併せると 1 エピソード
    6 x 585 = 3510 物理ステップで、v1 の 8 x 686 = 5488 ステップから
    **36% の短縮**になる。v1 は 1 iteration 285〜330 秒 x 4000 = 約 14 日
    かかる見積りだったので、ここは実時間の観点でも効く。

    5 手には減らさない。3 遷移 x 2 手の下限を割ると、把持失敗
    (実測で約 3% ) が 1 回起きただけで固結びが原理的に不可能になる。
    """


# ロープのプリセット 4 種 x (通常 / Play) の設定クラスを生成する。
# 生成されるのは TwistEnvCfg / SimpleEnvCfg / StiffEnvCfg / FineEnvCfg と
# それぞれの *PlayEnvCfg。`__init__.py` が文字列 entry point で引く。
globals().update(make_preset_cfgs(KnotTyingV2EnvCfg, __name__))
