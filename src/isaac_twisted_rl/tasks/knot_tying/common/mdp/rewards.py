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
from .terminations import diverged_now

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _pick_place_term(env: ManagerBasedRLEnv, action_name: str):
    """`PickPlaceAction` 本体を取り出す (報酬へ持ち越した量を読むため)。"""
    return env.action_manager.get_term(action_name)


def _zero_if_diverged(env: ManagerBasedRLEnv, value: torch.Tensor, diverged_term: str | None) -> torch.Tensor:
    """`diverged_term` が指定されていれば、この手で発散した環境の値を 0 にする。

    発散した手のロープ座標は NaN か巨大な有限値で、そこから測った位相・高さ・
    把持距離は意味を持たない (v2-R2 の修正。`README.md` の「発散した手の報酬」)。
    None なら何もしない (v0〜v2 の挙動)。
    """
    if diverged_term is None:
        return value
    return torch.where(diverged_now(env, diverged_term), torch.zeros_like(value), value)


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
    reduce_r2: bool = False,
    diverged_term: str | None = None,
) -> torch.Tensor:
    """目標の交差数に達し、かつロープが床に落ちているか `(num_envs,)`。

    `settled_height_diameters` は「ロープの最高点がこの高さ以下なら、
    ハンドに吊られてはいない」と判断する閾値 (ロープ直径に対する比)。
    把持高さ `GRASP_PALM_Z` は指先が床上 2 mm に来る高さなので、
    交差した状態で床に落ちていればロープの最高点はロープ直径の数倍に収まる。
    ハンドに吊られていれば 0.2 m 以上になり、この条件で弾かれる。
    """
    extractor = get_extractor(env, rope_cfg.name)
    reached = extractor.ladder_batch(reduce_r2).count >= min_crossings
    settled = rope_zmax(env, rope_cfg) <= settled_height_diameters * rope_diameter
    return _zero_if_diverged(env, (reached & settled).float(), diverged_term)


def knot_class_shaping(
    env: ManagerBasedRLEnv,
    action_name: str = "pick_place",
    gamma: float = 0.98,
    potentials: tuple[float, ...] = (0.0, 0.35, 0.70, 1.0),
    rope_cfg: SceneEntityCfg = SceneEntityCfg("rope"),
    reduce_r2: bool = False,
    diverged_term: str | None = None,
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

    `reduce_r2=True` (v2-R2) では前後とも R2 除去後の位相で段を決める
    (`topology/knot_detect.py` の `reduce_r2`)。前後で判定器を揃えないと
    `F` がポテンシャル差にならないので、`before` も R2 版を読む。

    ## `diverged_term` (v2-R2)

    指定すると、その手で発散した環境では `Φ(s') = 0` とする (`F = −Φ(s)`)。
    発散はエピソードの終端 (bootstrap しない) なので、終端状態のポテンシャルを
    0 と置くのがポテンシャル整形の正しい扱い。**0 にはしない** — そうすると
    「登った段から発散で降りる」ことの損失 `−Φ(s)` が消えてしまう。
    指定しない (v0〜v2) と、発散したロープ座標から測った出鱈目な段で
    `Φ(s')` を払ってしまう。
    """
    extractor = get_extractor(env, rope_cfg.name)
    table = torch.tensor(potentials, dtype=torch.float32, device=extractor.count.device)
    term = _pick_place_term(env, action_name)
    before = term.class_r2_at_action_start if reduce_r2 else term.class_at_action_start
    after = tp.topology_class(extractor.ladder_batch(reduce_r2))
    phi_after = table[after]
    if diverged_term is not None:
        phi_after = torch.where(diverged_now(env, diverged_term), torch.zeros_like(phi_after), phi_after)
    return gamma * phi_after - table[before]


def overhand_achieved(
    env: ManagerBasedRLEnv,
    settled_height_diameters: float = 3.0,
    rope_diameter: float = 0.02,
    rope_cfg: SceneEntityCfg = SceneEntityCfg("rope"),
    reduce_r2: bool = False,
    diverged_term: str | None = None,
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
    reached = tp.is_overhand(extractor.ladder_batch(reduce_r2))
    settled = rope_zmax(env, rope_cfg) <= settled_height_diameters * rope_diameter
    return _zero_if_diverged(env, (reached & settled).float(), diverged_term)


def interleaved2_achieved(
    env: ManagerBasedRLEnv,
    settled_height_diameters: float = 3.0,
    rope_diameter: float = 0.02,
    rope_cfg: SceneEntityCfg = SceneEntityCfg("rope"),
    reduce_r2: bool = False,
    diverged_term: str | None = None,
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

    ## `reduce_r2` (v2-R2)

    部分列判定のため、既定 (False) では交差 3 つ以上の中の 2 つが交互・同符号
    なら成立してしまう。先端で輪を上から跨いだだけの状態 (R2 で輪 1 つに
    戻る) もここで class 2 と判定され、v2 の方策はこれで報酬を得ていた疑いが
    強い。True なら R2 で消せる組を除いてから判定する。
    """
    extractor = get_extractor(env, rope_cfg.name)
    reached = tp.is_interleaved_two(extractor.ladder_batch(reduce_r2))
    settled = rope_zmax(env, rope_cfg) <= settled_height_diameters * rope_diameter
    return _zero_if_diverged(env, (reached & settled).float(), diverged_term)


def offladder_penalty(
    env: ManagerBasedRLEnv,
    rope_cfg: SceneEntityCfg = SceneEntityCfg("rope"),
    reduce_r2: bool = False,
    diverged_term: str | None = None,
) -> torch.Tensor:
    """交差はあるがラダーに乗っていない `(num_envs,)`。軽い罰則。

    独立した R1 ツイストを複数作った状態など。交差数だけ見ると進んで
    いるように見えるが、移植元の BFS グラフでも固結びへ繋がらない
    行き止まりなので、そこへ行く癖を軽く抑える。

    旧 `crossing_progress` (Δ交差数) が持っていた抜け道
    (R1 を 3 回ひねれば交差数 3) を塞ぐ役割でもある。

    `reduce_r2=True` (v2-R2) では R2 除去後の交差数・クラスで判定する。
    先端を輪の上へ置いただけ (R2 で交差 0 に戻る) の状態は「行き止まり」
    ではなく「何もしていない」なので罰しない。
    """
    extractor = get_extractor(env, rope_cfg.name)
    batch = extractor.ladder_batch(reduce_r2)
    cls = tp.topology_class(batch)
    return _zero_if_diverged(env, ((batch.count >= 2) & (cls == tp.CLASS_TRIVIAL)).float(), diverged_term)


def grasp_distance_penalty(
    env: ManagerBasedRLEnv,
    action_name: str = "pick_place",
    max_distance_diameters: float | None = None,
    rope_diameter: float = 0.02,
    diverged_term: str | None = None,
) -> torch.Tensor:
    """指を閉じた時点での 指先とロープの距離 `(num_envs,)` [m]。整形報酬。

    「狙ったリンクを実際に掴めたか」を連続量で表す。負の重みを掛けて使う。
    掴めていれば 0 に近く、空振りしていればロープまでの距離ぶん罰が入る。

    ## 上限と発散のマスク (v2-R2 で追加。既定の None は v0〜v2 と同じ無制限)

    距離は `close` 終了時に測るので、**その瞬間にロープが吹き飛んでいると
    巨大な有限値** が入る (`nan_to_num` は NaN / Inf しか消さない)。
    v2-R2 の最初の run はこれで `Episode_Reward/grasp_shaping` が −107.8、
    `Loss/value_function` が 7.2e9 になり、適応学習率が下限まで落ちた。
    ランダム方策の実測では正常な手の距離は最大 0.25 m、発散した手では
    97 m が記録された (2026-09-27, 512 環境 x 24 手)。

    * `max_distance_diameters`: ロープ直径の何倍で頭打ちにするか。
    * `diverged_term`: この手で発散した環境は 0 にする (罰は `failure` が持つ)。

    上限だけでなくマスクも要るのは、`close` で吹き飛んだロープが手の終わりまでに
    有限・範囲内へ戻る場合と、戻らず発散と判定される場合の両方があるため。
    """
    dist = _pick_place_term(env, action_name).grasp_distance
    if max_distance_diameters is not None:
        dist = dist.clamp(max=max_distance_diameters * rope_diameter)
    return _zero_if_diverged(env, dist, diverged_term)


def action_failed(
    env: ManagerBasedRLEnv,
    action_name: str = "pick_place",
    rope_cfg: SceneEntityCfg = SceneEntityCfg("rope"),
    diverged_term: str | None = None,
) -> torch.Tensor:
    """手を実行できなかった / ロープが発散した `(num_envs,)`。罰則。

    `next_action.md` 4.5 の NaN 対策の報酬側。終了判定 (`terminations.rope_diverged`)
    と対になっていて、こちらは「その手は失敗だった」という信号を与える。

    `diverged_term` (v2-R2) を指定すると、NaN だけでなくその終了項の判定
    (有限のまま吹き飛んだロープを含む) も失敗に数える。報酬側で発散した手の
    他の項を 0 にするので、発散の罰はこの項に一本化される。
    """
    invalid = _pick_place_term(env, action_name).invalid_action
    extractor = get_extractor(env, rope_cfg.name)
    diverged = ~torch.isfinite(extractor.nodes).all(dim=-1).all(dim=-1)
    if diverged_term is not None:
        diverged = diverged | diverged_now(env, diverged_term)
    return (invalid | diverged).float()


def held_up_penalty(
    env: ManagerBasedRLEnv,
    height_diameters: float = 5.0,
    rope_diameter: float = 0.02,
    rope_cfg: SceneEntityCfg = SceneEntityCfg("rope"),
    diverged_term: str | None = None,
) -> torch.Tensor:
    """手の終わりにロープが浮いたままか `(num_envs,)`。罰則。

    `crossing_achieved` の条件を裏返しただけに見えるが、こちらは交差の有無に
    依らず効く。`next_action.md` 3(c) の「離したのに付いてくる」が再発した
    ことを **学習曲線ではなくログで** 気付けるようにする狙いもある
    (この項だけが立ち上がったら物理パラメータを疑う)。
    """
    return _zero_if_diverged(env, (rope_zmax(env, rope_cfg) > height_diameters * rope_diameter).float(), diverged_term)


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

    .. warning::
        **重み 0.0 では実際には呼ばれない。** Isaac Lab の
        `RewardManager.compute()` は重み 0 の項をスキップするため、
        v2 の `Episode_Reward/diverged_class*` は一度も記録されていない
        (2026-09-24 の v2 run で確認)。v2-R2 以降は
        `topology_log.TopologyStats` の `div_cls*` を使うこと。
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
