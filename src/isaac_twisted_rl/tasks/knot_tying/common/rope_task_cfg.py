"""ロープのプリセットごとの環境設定 (`simple` / `stiff` / `twist` / `fine`)。

`rope_specs.py` の 4 プリセットそれぞれに 1 タスクを割り当てる。基底
(`KnotTyingEnvCfg`) との差は **ロープの USD とそれに紐づく寸法・ゲインだけ**
なので、共通の `_apply_rope` に押し込んである。

## 学習に使うのは `twist`

`next_action.md` 3(c) の決定。`twist` だけヤング率を 1/10 (1.0e5 Pa) に
下げてあり、これが「指を開いてもロープが腰で引っかかって付いてくる」
問題の解決策になっている。他のプリセットは比較用に元の剛性のまま残して
あるので、学習に使うならこの現象が再発しないか必ず確認すること。

## グリッパーの設定がデモと同じでなければならない理由

`fix_root_link=False` (ルートを直接書き込んで動かすため) と
`disable_gravity=True` (キネマティックハンドなので落下させない)、
そして **指のゲインをロープごとにスケールする**
(`RopeSpec.grasp_finger_stiffness = 1500 * link_mass / capsule_radius`) の
3 点が揃って初めて摩擦把持が成立する。1 つでも欠けるとロープを掴めない。
"""

from __future__ import annotations

from collections.abc import Callable

from isaaclab.utils import configclass

from .env_cfg_base import KnotTyingEnvCfg
from .mdp import PickPlaceActionCfg
from .parallel_gripper.gripper_cfg import PARALLEL_GRIPPER_CFG
from .rope_model.rope_cfg import make_rope_cfg
from .rope_model.rope_specs import RopeSpec, get_spec

# 把持まわりの定数。すべて topology_debug_demo.py と同じ値。
_FINGER_DAMPING_RATIO = 0.04
_FINGER_EFFORT_LIMIT = 100.0
_ROPE_INIT_Z = 0.06
"""ロープの初期高さ [m]。落として落ち着かせる前提の値 (デモと同じ)。"""

_HOME_PALM_Z = 0.30
"""ハンドの初期高さ [m]。`phases.HOME_PALM_Z` と揃える。"""


def _apply_rope(cfg: KnotTyingEnvCfg, rope_name: str) -> RopeSpec:
    """プリセット名からロープ・グリッパー・アクションの設定を埋める。"""
    spec = get_spec(rope_name)

    # -- ロープ
    rope_cfg = make_rope_cfg(rope_name)
    rope_cfg.prim_path = "{ENV_REGEX_NS}/Rope"
    rope_cfg.init_state.pos = (0.0, 0.0, _ROPE_INIT_Z)
    cfg.scene.rope = rope_cfg

    # -- グリッパー (キネマティックハンド)
    gripper_cfg = PARALLEL_GRIPPER_CFG.copy()
    gripper_cfg.prim_path = "{ENV_REGEX_NS}/Gripper"
    gripper_cfg.init_state.pos = (0.0, 0.0, _HOME_PALM_Z)
    # ルートを毎ステップ書き込んで動かすので固定しない。
    gripper_cfg.spawn.articulation_props.fix_root_link = False
    # 重力で落ちないようにする (アームが無いので支えるものが無い)。
    gripper_cfg.spawn.rigid_props.disable_gravity = True
    gripper_cfg.spawn.articulation_props.solver_position_iteration_count = 16
    gripper_cfg.spawn.articulation_props.solver_velocity_iteration_count = 1
    # 指のゲインはロープの質量と半径からスケールする。把持力は
    # 「閉じ切れない量 (= ロープ半径) x ばね定数」で決まるので、
    # ロープが変わればゲインも変えないと保持できない。
    stiffness = spec.grasp_finger_stiffness
    finger_actuator = gripper_cfg.actuators["finger_actuator"]
    finger_actuator.stiffness = stiffness
    finger_actuator.damping = _FINGER_DAMPING_RATIO * stiffness
    finger_actuator.effort_limit_sim = _FINGER_EFFORT_LIMIT
    cfg.scene.gripper = gripper_cfg

    # -- アクション
    cfg.actions.pick_place = PickPlaceActionCfg(
        asset_name="gripper",
        rope_asset_name="rope",
        finger_joint_names=["left_finger_joint", "right_finger_joint"],
        rope_length=spec.rope_length,
        rope_diameter=spec.diameter,
        durations=cfg.durations,
    )

    # -- 報酬のうち、ロープ寸法に依存する閾値
    #
    # 項の名前を決め打ちせず、`rope_diameter` を params に宣言している項すべてを
    # 埋める。以前は `success` と `held_up` の 2 つをハードコードしていたが、
    # バージョンが報酬項を足すたびにここも直さないと **例外なく既定値 0.02 が
    # 残る** (= `fine` プリセットでは直径 8 mm なのに 20 mm 前提の閾値で
    # 判定してしまう) という地雷になっていた。v2 で疎報酬を 3 段に増やした
    # ときに踏んだので宣言ベースに変えた。
    #
    # なぜバージョン側の `__post_init__` で埋められないか: `make_preset_cfgs`
    # が組む順序が `_base.__post_init__(self)` -> `_apply_rope(self, preset)`
    # なので、バージョン側の `__post_init__` が走る時点ではまだ
    # プリセットが決まっていない。
    for _term in cfg.rewards.__dict__.values():
        if hasattr(_term, "params") and "rope_diameter" in _term.params:
            _term.params["rope_diameter"] = spec.diameter

    return spec


##
# プリセット別の設定クラスを生成するファクトリ
##

_PRESET_DOCS = {
    "twist": "`twist` (橙, 20 リンク, 直径 20 mm)。**学習に使うのはこれ** (4.0 の決定)。",
    "simple": "`simple` (青, 20 リンク)。曲げ剛性なしの比較用ベースライン。",
    "stiff": "`stiff` (緑, 20 リンク)。曲げ剛性ありの比較用。",
    "fine": (
        "`fine` (赤, 48 リンク, 直径 8 mm)。結び目が結べる高分解能版。\n\n"
        "    ## 4 プリセットの中で最も発散しやすい\n\n"
        "    リンクが細く軽いほど発散しやすい。慣性が `simple` の 1/170 しか無く、\n"
        "    わずかな接触トルクで角加速度が跳ね上がるため\n"
        "    (`rope_cfg.py` の `MAX_ANGULAR_VELOCITY_DEG` の docstring)。\n\n"
        "    基底の 240 Hz でも、ランダム方策で 20 手あたり 2 回ほど\n"
        "    `rope_diverged` が立つ (`twist` は同条件で 1.2%)。学習に使うのは\n"
        "    `twist` なので当面はこのままにしてあるが、`fine` で本格的に\n"
        "    学習を回すならさらに `sim.dt` を詰める必要がある。"
    ),
    "knot": "`knot` (紫, 28 リンク, 直径 20 mm)。`twist` と同じ材料で長くした固結び用 (ゲート 1)。",
}

_PLAY_RENDER_STEPS = 8
"""Play 環境の描画間隔 [物理ステップ]。240 Hz で 30 Hz 相当 (約 0.033 s ごと)。

学習時 (`KnotTyingEnvCfg.__post_init__`) は `render_interval = decimation`
(= 1 手ぶん) にしてある。これは「1 手ごとに 1 回描画できれば十分」という
学習中の GUI 観察向けの値で、`uv run play` で推論の様子を見る用途には
不向き: 1 手の間ずっと画面が固まったまま最後に一瞬でワープしたように見え、
ロボットが「止まっている」ように見える原因になる (`apply_actions()` 自体は
毎物理ステップ動いているが、それが画面に反映されない)。

Play 系のクラスだけ `render_interval` を短くして、フェーズ内の smoothstep
補間による滑らかな動きが見えるようにする。学習の速度には影響しない
(`render_interval` は描画タイミングのみを制御し、物理・報酬・観測の
頻度である `decimation` とは独立)。
"""

_PLAY_NUM_ENVS = 9
"""Play 環境の数。GUI で見るだけなので絞る。"""


def make_preset_cfgs(
    base_cls: type,
    module_name: str,
    play_post_init: Callable[[KnotTyingEnvCfg], None] | None = None,
) -> dict[str, type]:
    """バージョンの基底クラスから、プリセット別の設定クラス 8 個を作る。

    返すのは ``{"TwistEnvCfg": <class>, "TwistPlayEnvCfg": <class>, ...}``。
    呼び出し側は ``globals().update(make_preset_cfgs(...))`` で自分の
    モジュールに生やす。`__init__.py` はこれを文字列 entry point で引く。

    **なぜファクトリにするか。** プリセット 4 種 x (通常/Play) の 8 クラスは
    「基底が違うだけで中身は同じ」。バージョンを増やすたびに 8 個の空クラスを
    書き写すのは間違いの元なので、ここで 1 回だけ定義する。バージョン側が
    書くのは課題の中身 (観測・報酬・エピソード長) だけで済む。

    Args:
        base_cls: そのバージョンの環境設定クラス (`KnotTyingV0EnvCfg` など)。
        module_name: 生成したクラスに付ける ``__module__``。呼び出し側の
            ``__name__`` を渡す。これが合っていないと Hydra や
            `dump_yaml` がクラスを解決できない。
        play_post_init: Play 用クラスだけに追加で掛ける設定変更 (省略可)。
            Play の `__post_init__` の **最後** (プリセット適用と環境数の
            絞り込みの後) に呼ばれる。v2-R2 が「Play のときだけ p-data を
            表示する」ために追加した。バージョン側の `__post_init__` は
            プリセット適用より前に走るので、そこでは Play か否かを区別できない。
    """
    out: dict[str, type] = {}
    for preset, doc in _PRESET_DOCS.items():
        name = preset.capitalize()

        def _make_post_init(_preset: str, _base: type):
            def __post_init__(self):
                # super() は使わない: type(self) を基準にすると、Play 側から
                # 呼ばれたときに同じ関数へ戻って無限再帰する。基底を閉包で
                # 固定して直接呼ぶ。
                _base.__post_init__(self)
                _apply_rope(self, _preset)
                if _preset == "fine":
                    # リンク数が 48 と多く自己接触も多いので接触ペアの上限を上げる。
                    self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 8

            return __post_init__

        env_cls = configclass(
            type(
                f"{name}EnvCfg",
                (base_cls,),
                {
                    "__doc__": doc,
                    "__module__": module_name,
                    "__post_init__": _make_post_init(preset, base_cls),
                },
            )
        )

        def _make_play_post_init(_base: type):
            def __post_init__(self):
                _base.__post_init__(self)
                self.scene.num_envs = _PLAY_NUM_ENVS
                self.sim.render_interval = _PLAY_RENDER_STEPS
                if play_post_init is not None:
                    play_post_init(self)

            return __post_init__

        play_cls = configclass(
            type(
                f"{name}PlayEnvCfg",
                (env_cls,),
                {
                    "__doc__": f"`{preset}` の再生用 (環境数を絞り、描画を増やす)。",
                    "__module__": module_name,
                    "__post_init__": _make_play_post_init(env_cls),
                },
            )
        )

        out[f"{name}EnvCfg"] = env_cls
        out[f"{name}PlayEnvCfg"] = play_cls
    return out
