"""報酬項 (`next_action.md` 4.2 の設計をそのまま実装)。

## 報酬関数が「1 手につき 1 回」呼ばれること

`decimation` を 1 手ぶんの物理ステップ数にしてあるので、
`ManagerBasedRLEnv.step()` の中で報酬計算は **フェーズ列を全部終えた後に
1 回だけ** 走る。つまり `next_action.md` 4.2 の「アクションごと」の項
(`r_progress`) が、特別な仕掛け無しにそのまま書ける。

## 成功判定を「ハンドが退避した後」に測ること

4.2 が **最重要** と書いている点。ハンドで吊って交差を見せかける方策を
学習させないため、成功報酬は必ずハンドがロープから離れた状態で測る。

本実装ではフェーズ列の最後が `relax` (`clear` で横へ抜けてから `retreat` で
待機高さ 0.30 m まで上がった後、指を全開で静止) なので、報酬計算の時点で
ハンドは既にロープから離れている。

それに加えて **`rope_zmax` による保険** を入れる。`next_action.md` 3(c) の
物理パラメータ調整で「離すと付いてくる」現象は解決済みだが、4.2 は
「報酬の測定タイミングはそれとは別に明示的に押さえる必要がある」と
釘を刺している。ロープが床付近に落ちていることを条件に加えることで、
仮に再発しても「吊っているだけ」が成功に化けない。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

from .. import topology as tp
from . import phases
from .observations import get_extractor, rope_zmax

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _pick_place_term(env: ManagerBasedRLEnv, action_name: str):
    """`PickPlaceAction` 本体を取り出す (報酬へ持ち越した量を読むため)。"""
    return env.action_manager.get_term(action_name)


def crossing_progress(
    env: ManagerBasedRLEnv,
    action_name: str = "pick_place",
    rope_cfg: SceneEntityCfg = SceneEntityCfg("rope"),
) -> torch.Tensor:
    """交差数の増分 `(num_envs,)`。密報酬 (`r_progress`)。

    p-data は離散量なので一致 / 不一致だけでは勾配が出ない (4.2)。
    「手を打つ前より交差が増えた」を毎手評価することで、成功に至る前から
    方策に方向を与える。減った場合は負になる (作った交差を壊す手を罰する)。
    """
    extractor = get_extractor(env, rope_cfg.name)
    before = _pick_place_term(env, action_name).crossings_at_action_start
    return (extractor.count - before).float()


def crossing_achieved(
    env: ManagerBasedRLEnv,
    min_crossings: int = 1,
    settled_height_diameters: float = 3.0,
    rope_diameter: float = 0.02,
    rope_cfg: SceneEntityCfg = SceneEntityCfg("rope"),
) -> torch.Tensor:
    """目標の交差数に達し、かつロープが床に落ちているか `(num_envs,)`。

    `settled_height_diameters` は「ロープの最高点がこの高さ以下なら、
    ハンドに吊られてはいない」と判断する閾値 (ロープ直径に対する比)。
    把持高さ `GRASP_PALM_Z` は指先が床上 2 mm に来る高さなので、
    交差した状態で床に落ちていればロープの最高点はロープ直径の数倍に収まる。
    ハンドに吊られていれば 0.2 m 以上になり、この条件で弾かれる。
    """
    extractor = get_extractor(env, rope_cfg.name)
    reached = extractor.count >= min_crossings
    settled = rope_zmax(env, rope_cfg) <= settled_height_diameters * rope_diameter
    return (reached & settled).float()


def knot_class_shaping(
    env: ManagerBasedRLEnv,
    action_name: str = "pick_place",
    gamma: float = 0.98,
    potentials: tuple[float, ...] = (0.0, 0.35, 0.70, 1.0),
    rope_cfg: SceneEntityCfg = SceneEntityCfg("rope"),
) -> torch.Tensor:
    """位相ラダーのポテンシャル整形 `(num_envs,)`。密報酬。

    `F(s, s') = γ·Φ(s') − Φ(s)`。`Φ` は位相クラス
    (`topology/knot_detect.py`) ごとの値で、既定は
    `trivial=0.0 / 交差1つ=0.35 / 絡み2交差=0.70 / 固結び=1.0`。

    ## なぜ素朴なボーナスではなくポテンシャル形式なのか

    `if class == 2: +0.70` のように段そのものに報酬を付けると、
    **2 段目に居座って毎手ボーナスを稼ぐ**のが最適方策になってしまう。
    このタスクで最も起こりやすい報酬ハッキングがこれ。

    ポテンシャル形式なら、段に留まるだけでは `γ·0.70 − 0.70 ≈ −0.014` に
    しかならず、登らないと純益が出ない。しかも Ng et al. (1999) により
    **最適方策を変えないことが保証されている**ので、整形を足したせいで
    解きたい問題がずれる心配が無い。

    ## γ を合わせること

    1 env ステップ = 1 手なので、ここの `gamma` は PPO 側の
    `KnotTyingPPORunnerCfg.algorithm.gamma` と**必ず一致させる**。
    ずれると上記の最適方策不変性が崩れる。既定値をどちらも 0.98 にし、
    cfg 側にも注意書きを置いてある。

    `Φ(s)` は `PickPlaceAction.class_at_action_start` から読む
    (手を打つ前に確定させてある値)。`Φ(s')` は手を打ち終えた現在の状態。
    """
    extractor = get_extractor(env, rope_cfg.name)
    table = torch.tensor(potentials, dtype=torch.float32, device=extractor.count.device)
    before = _pick_place_term(env, action_name).class_at_action_start
    after = tp.topology_class(extractor.batch)
    return gamma * table[after] - table[before]


def overhand_achieved(
    env: ManagerBasedRLEnv,
    settled_height_diameters: float = 3.0,
    rope_diameter: float = 0.02,
    rope_cfg: SceneEntityCfg = SceneEntityCfg("rope"),
) -> torch.Tensor:
    """固結びができ、かつロープが床に落ちているか `(num_envs,)`。疎報酬。

    判定は `topology/knot_detect.py` の部分列検出器。**厳密な
    `交差数 == 3` を使っていない**のが要点で、実機では結び目の他に
    尻尾のループなど余分な交差が乗るため、厳密判定だと本物の固結びを
    取りこぼす (三葉結び + 尻尾ループ 1 つで交差数 7 になる実測あり)。

    `settled_height_diameters` の条件は `crossing_achieved` と同じ意図で、
    **ハンドで吊って結び目を見せかける方策を学習させない**ための保険
    (`next_action.md` 4.2 が最重要と指定している点)。フェーズ列の最後が
    `relax` なので報酬計算時点でハンドは既に離れているが、二重にかける。
    """
    extractor = get_extractor(env, rope_cfg.name)
    reached = tp.is_overhand(extractor.batch)
    settled = rope_zmax(env, rope_cfg) <= settled_height_diameters * rope_diameter
    return (reached & settled).float()


def interleaved2_achieved(
    env: ManagerBasedRLEnv,
    settled_height_diameters: float = 3.0,
    rope_diameter: float = 0.02,
    rope_cfg: SceneEntityCfg = SceneEntityCfg("rope"),
) -> torch.Tensor:
    """2 交差が絡んだ状態ができ、かつロープが床に落ちているか `(num_envs,)`。疎報酬。

    `overhand_achieved` と同じ構造で、判定を `is_interleaved_two`
    (= `has_knot_subsequence(k=2)`) に替えただけ。**ラダーの中段に
    「離しても残ったか」を測る疎報酬を置くため**に v2 で追加した。

    ## なぜ中段の疎報酬が必要か (v1 の実測)

    v1 は疎報酬を `overhand_achieved` (ラダー最上段) だけにしたが、
    TensorBoard の実測では **1 手あたり 0.00005 回しか発火せず、終端では
    厳密に 0.0** だった。その結果 PPO が受け取る勾配が `action_failed`
    (罰則) だけになり、終端の報酬構成は `failure` −0.0924 が
    `shaping` +0.0496 を 1.86 倍上回る純負の状態になっていた。
    報酬がマイナスしか払わないと、唯一プラスを払うエントロピーボーナスの
    最大化が最適方策になり、実際に `noise_std` が 1.003 -> 1.628、
    `entropy` が 8.52 -> 11.39 と**発散方向へ**動いていた。

    中段の段としてこれを選んだ理由は 2 つ:

    * **発火し得ることが実測済み**。ランダム方策 (16 手 x 32 環境) で
      クラス 2 に 1.4% 到達している (`versions/v1/README.md` §12)。
      最上段の固結びは同条件で 512 回中 1 回で、疎すぎて足場にならない。
    * **「離しても輪が残るか」を測る唯一の段**。最下段
      (`crossing_achieved`) は交差 1 つで、v0 では 1 手あたり 98.5% 発火
      = ほぼ飽和していた。固結びに必要な「作った絡みが解けない」技能を
      問うのはこの段が最初になる。

    `settled_height_diameters` の意図は `overhand_achieved` と同じで、
    ハンドで吊って絡みを見せかける方策を学習させないための保険。
    """
    extractor = get_extractor(env, rope_cfg.name)
    reached = tp.is_interleaved_two(extractor.batch)
    settled = rope_zmax(env, rope_cfg) <= settled_height_diameters * rope_diameter
    return (reached & settled).float()


def offladder_penalty(
    env: ManagerBasedRLEnv,
    rope_cfg: SceneEntityCfg = SceneEntityCfg("rope"),
) -> torch.Tensor:
    """交差はあるがラダーに乗っていない `(num_envs,)`。軽い罰則。

    独立した R1 ツイストを複数作った状態など。交差数だけ見ると進んで
    いるように見えるが、移植元の BFS グラフでも固結びへ繋がらない
    行き止まりなので、そこへ行く癖を軽く抑える。

    旧 `crossing_progress` (Δ交差数) が持っていた抜け道
    (R1 を 3 回ひねれば交差数 3) を塞ぐ役割でもある。
    """
    extractor = get_extractor(env, rope_cfg.name)
    cls = tp.topology_class(extractor.batch)
    return ((extractor.count >= 2) & (cls == tp.CLASS_TRIVIAL)).float()


def grasp_distance_penalty(
    env: ManagerBasedRLEnv,
    action_name: str = "pick_place",
) -> torch.Tensor:
    """指を閉じた時点での 指先とロープの距離 `(num_envs,)` [m]。整形報酬。

    「狙ったリンクを実際に掴めたか」を連続量で表す。負の重みを掛けて使う。
    掴めていれば 0 に近く、空振りしていればロープまでの距離ぶん罰が入る。
    """
    return _pick_place_term(env, action_name).grasp_distance


def action_failed(
    env: ManagerBasedRLEnv,
    action_name: str = "pick_place",
    rope_cfg: SceneEntityCfg = SceneEntityCfg("rope"),
) -> torch.Tensor:
    """手を実行できなかった / ロープが発散した `(num_envs,)`。罰則。

    `next_action.md` 4.5 の NaN 対策の報酬側。終了判定 (`terminations.rope_diverged`)
    と対になっていて、こちらは「その手は失敗だった」という信号を与える。
    """
    invalid = _pick_place_term(env, action_name).invalid_action
    extractor = get_extractor(env, rope_cfg.name)
    diverged = ~torch.isfinite(extractor.nodes).all(dim=-1).all(dim=-1)
    return (invalid | diverged).float()


def held_up_penalty(
    env: ManagerBasedRLEnv,
    height_diameters: float = 5.0,
    rope_diameter: float = 0.02,
    rope_cfg: SceneEntityCfg = SceneEntityCfg("rope"),
) -> torch.Tensor:
    """手の終わりにロープが浮いたままか `(num_envs,)`。罰則。

    `crossing_achieved` の条件を裏返しただけに見えるが、こちらは交差の有無に
    依らず効く。`next_action.md` 3(c) の「離したのに付いてくる」が再発した
    ことを **学習曲線ではなくログで** 気付けるようにする狙いもある
    (この項だけが立ち上がったら物理パラメータを疑う)。
    """
    return (rope_zmax(env, rope_cfg) > height_diameters * rope_diameter).float()


def diverged_by_class(
    env: ManagerBasedRLEnv,
    knot_class: int,
    action_name: str = "pick_place",
    rope_cfg: SceneEntityCfg = SceneEntityCfg("rope"),
) -> torch.Tensor:
    """手を始めた時点の位相クラスが `knot_class` で、その手の間に発散したか。

    `next_action.md` 4.8.9 の「class 別に層別した rope_diverged を見ること」
    への対応。自己接触は交差数に対して超線形に増えるので、
    `Episode_Termination/diverged` の集計値 (全 env 平均 1 本) だけでは
    class 3 (固結び) でだけ発散が増えていても埋もれる。

    重み 0.0 の報酬項として登録し、`Episode_Reward/diverged_class{0..3}` に
    class 別の発散率を出すためだけに存在する (学習には寄与しない)。
    `RewardManager` は項ごとに `Episode_Reward/<name>` を自動でログするので、
    新しい集計インフラを作らずに済む。

    判定基準は `action_failed` と同じ `~isfinite` (NaN/発散)。どの class から
    発散したかを見たいので、発散が起きた**後**の class ではなく
    `class_at_action_start` (手を打つ前の class) で層別する。
    """
    term = _pick_place_term(env, action_name)
    extractor = get_extractor(env, rope_cfg.name)
    diverged = ~torch.isfinite(extractor.nodes).all(dim=-1).all(dim=-1)
    return ((term.class_at_action_start == knot_class) & diverged).float()


def phase_incomplete(env: ManagerBasedRLEnv, action_name: str = "pick_place") -> torch.Tensor:
    """手が最終フェーズまで到達しなかった `(num_envs,)`。

    `decimation` はフェーズの尺の合計から自動計算しているので通常は 0。
    尺の設定を変えたときに気付けるようにしてある健全性チェック。
    """
    term = _pick_place_term(env, action_name)
    return (term._phase < phases.RELAX).float()
