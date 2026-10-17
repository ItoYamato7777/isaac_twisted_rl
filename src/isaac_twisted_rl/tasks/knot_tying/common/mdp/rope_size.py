"""ロープの寸法 (長さ・直径) の DR の、env ごとの寸法を読む項。v16 で追加。

| 項 | 種類 | 役割 |
|---|---|---|
| `get_rope_sizes` | (関数) | env ごとの寸法 `RopeSizes` を返す。寸法を振らない設定 (v15 まで) では None |
| `per_env_diameter` | (関数) | 直径を 1 つの数で持つ項 (床に落ちた判定など) が、env ごとの直径に置き換えるために呼ぶ |
| `rope_size_params` | 観測 | 寸法の倍率を [−1, 1] に正規化した `(N, 2)`。critic の特権観測に使う (actor には入れない) |
| `check_rope_sizes` | event (`mode="startup"`) | spawn された USD の半径・リンク間隔が、割り当てと一致するかを確かめる |

## 寸法の出どころ

ロープの spawn 設定が `RopeSizeSpawnerCfg` (`rope_model/rope_cfg.py`) のときだけ、寸法を振る。env `i` の寸法は
`scales[i % len(scales)]` (`domain_rand.size_assignment`。spawn の規則と同じ)。spawn 設定を唯一の出どころにするのは、
Hydra の上書き (`env.scene.rope.spawn.scales=...`) で寸法を変えたとき、spawn と寸法を読む側が食い違わないため。
spawn 設定が普通の `UsdFileCfg` (v15 まで) なら `get_rope_sizes` は None で、寸法を読む項は 1 つの数のまま動く
(既存の版の挙動を変えない)。

## 長さの倍率が 2 種類ある理由

* `length_scale`: リンク間隔 (`capsule_half_length`) の倍率。関節角が同じなら形全体がこの倍率で伸び縮みするので、
  アーカイブの復元 (`ResetToStage`) はこれで root の位置を伸ばす。critic の特権観測もこれ (格子の水準そのもの)。
* `rope_length`: 端のキャップを含む全長 [m] (`RopeSpec.rope_length` = 28 × 間隔 + 直径)。行動の置き先の範囲
  (`drop_xy_rel_frac × L`) はこちら。公称の組では継承元の版と同じ値になる。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from .. import domain_rand as dr
from ..rope_model.rope_specs import SIZE_DIAMETER_SCALES, SIZE_LENGTH_SCALES, get_spec, size_variant_name

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

_SIZES_ATTR = "_knot_tying_rope_sizes"


@dataclass(frozen=True)
class RopeSizes:
    """env ごとの寸法。テンソルはすべて `(num_envs,)` で env のデバイスにある。"""

    names: tuple[str, ...]
    """組のプリセット名 (`scales` の順)。"""

    index: torch.Tensor
    """env ごとの組の番号 (long)。"""

    length_scale: torch.Tensor
    """リンク間隔の倍率 (格子の水準)。"""

    diameter_scale: torch.Tensor
    """直径の倍率 (格子の水準)。"""

    rope_length: torch.Tensor
    """端のキャップを含む全長 [m]。"""

    diameter: torch.Tensor
    """直径 [m]。"""

    capsule_radius: torch.Tensor
    """カプセルの半径 [m] (`check_rope_sizes` の検算用)。"""

    capsule_half_length: torch.Tensor
    """カプセルの円柱部の半長 [m] (`check_rope_sizes` の検算用)。"""


def get_rope_sizes(env: ManagerBasedRLEnv, rope_name: str = "rope") -> RopeSizes | None:
    """env ごとの寸法 (初回に作って env に持たせる)。ロープの spawn 設定が `RopeSizeSpawnerCfg` でなければ None。"""
    if hasattr(env, _SIZES_ATTR):
        return getattr(env, _SIZES_ATTR)
    sizes = _build(env, rope_name)
    setattr(env, _SIZES_ATTR, sizes)
    return sizes


def _build(env: ManagerBasedRLEnv, rope_name: str) -> RopeSizes | None:
    spawn = getattr(getattr(env.cfg.scene, rope_name), "spawn", None)
    # 型ではなく属性で見分ける (isaaclab.sim をここで import しない。`RopeSizeSpawnerCfg` だけが `scales` と `base` を持つ)。
    if spawn is None or not (hasattr(spawn, "scales") and hasattr(spawn, "base")):
        return None
    pairs = [(float(ls), float(ds)) for ls, ds in spawn.scales]
    keep = bool(getattr(spawn, "keep_joint_limits", False))
    names = tuple(size_variant_name(spawn.base, ls, ds, keep) for ls, ds in pairs)
    specs = [get_spec(n) for n in names]
    dev = env.device
    index = dr.size_assignment(env.num_envs, len(pairs), device=dev)

    def per_env(values) -> torch.Tensor:
        return torch.tensor(values, dtype=torch.float32, device=dev)[index]

    return RopeSizes(
        names=names,
        index=index,
        length_scale=per_env([p[0] for p in pairs]),
        diameter_scale=per_env([p[1] for p in pairs]),
        rope_length=per_env([s.rope_length for s in specs]),
        diameter=per_env([s.diameter for s in specs]),
        capsule_radius=per_env([s.capsule_radius for s in specs]),
        capsule_half_length=per_env([s.capsule_half_length for s in specs]),
    )


def per_env_diameter(env: ManagerBasedRLEnv, rope_diameter: float) -> float | torch.Tensor:
    """直径 `rope_diameter` (公称の値) を env ごとの直径 `(num_envs,)` にする。寸法を振らない設定ではそのまま返す。

    `rope_diameter` は `rope_task_cfg._apply_rope` が公称のプリセットの直径で埋めた値なので、直径の倍率を掛けると
    その env の直径になる。寸法を振らない設定で float のまま返すのは、継承元の版の計算を 1 ビットも変えないため。
    """
    sizes = get_rope_sizes(env)
    if sizes is None:
        return rope_diameter
    return rope_diameter * sizes.diameter_scale


def size_scales(env: ManagerBasedRLEnv) -> torch.Tensor:
    """env ごとの寸法の倍率 `(num_envs, 2)` (`domain_rand.SIZE_AXES` の順)。寸法を振らない設定では 1。"""
    sizes = get_rope_sizes(env)
    if sizes is None:
        return torch.ones(env.num_envs, len(dr.SIZE_AXES), device=env.device)
    return torch.stack([sizes.length_scale, sizes.diameter_scale], dim=-1)


def rope_size_params(env: ManagerBasedRLEnv) -> torch.Tensor:
    """観測: 寸法の倍率の log2 を、格子の範囲で [−1, 1] に写した `(N, 2)` (長さ、直径)。

    範囲は格子の最小と最大 (長さ ×0.8〜×1.25、直径 ×0.4〜×1.0)。公称の寸法は長さ 0 (0.8 × 1.25 = 1)・直径 +1 になる
    (直径の範囲は公称を上端にしたため)。使う組を Hydra で絞っても同じ写し方なので、公称だけの評価でも
    学習のときと同じ値が critic に入る。
    """
    dev = env.device
    lo = torch.tensor([min(SIZE_LENGTH_SCALES), min(SIZE_DIAMETER_SCALES)], device=dev)
    hi = torch.tensor([max(SIZE_LENGTH_SCALES), max(SIZE_DIAMETER_SCALES)], device=dev)
    return dr.normalize_size(size_scales(env), lo, hi)


def check_rope_sizes(env: ManagerBasedRLEnv, env_ids, rope_asset_name: str = "rope", atol: float = 1.0e-6):
    """spawn された USD の寸法が、env ごとの割り当て (`get_rope_sizes`) と一致するかを確かめる (`mode="startup"`)。

    割り当ての規則 (env の番号 % 組の数) は Isaac Lab の `spawn_multi_asset` の実装に依る。食い違うと、行動の範囲・
    床の判定・アーカイブの復元・critic の観測が別の寸法の値で黙って動くので、起動時に全 env の先頭のリンクの
    カプセルの半径と円柱の長さを USD から読んで比べる。寸法を振らない設定では何もしない。
    """
    sizes = get_rope_sizes(env, rope_asset_name)
    if sizes is None:
        return
    from isaacsim.core.utils.stage import get_current_stage
    from pxr import UsdGeom

    stage = get_current_stage()
    root = env.scene[rope_asset_name].cfg.prim_path
    radius = torch.empty(env.num_envs)
    half = torch.empty(env.num_envs)
    for i in range(env.num_envs):
        path = root.replace(env.scene.env_regex_ns, env.scene.env_prim_paths[i])
        geom = UsdGeom.Capsule(stage.GetPrimAtPath(f"{path}/Link00/geom"))
        if not geom:
            raise RuntimeError(f"ロープのカプセルが無い: {path}/Link00/geom")
        radius[i] = float(geom.GetRadiusAttr().Get())
        half[i] = 0.5 * float(geom.GetHeightAttr().Get())
    bad_r = (radius - sizes.capsule_radius.cpu()).abs() > atol
    bad_h = (half - sizes.capsule_half_length.cpu()).abs() > atol
    bad = bad_r | bad_h
    if bool(bad.any()):
        first = int(bad.nonzero()[0])
        raise RuntimeError(
            f"ロープの寸法が割り当てと違う env が {int(bad.sum())} 個 (最初は env {first}: USD の半径 {float(radius[first]):.5f}・"
            f"半長 {float(half[first]):.5f}、割り当て {sizes.names[int(sizes.index[first])]})"
        )
    _check_joint_limits(env, sizes, rope_asset_name)
    counts = torch.bincount(sizes.index.cpu(), minlength=len(sizes.names)).tolist()
    print(
        f"[knot_tying] check_rope_sizes: {env.num_envs} env の寸法・関節の角度制限が割り当てと一致 "
        f"({dict(zip(sizes.names, counts))})"
    )


def _check_joint_limits(env: ManagerBasedRLEnv, sizes: RopeSizes, rope_asset_name: str, atol_deg: float = 0.01) -> None:
    """PhysX から読んだ関節の角度制限 (`joint_pos_limits`) が、env の組のプリセットと一致するかを確かめる。

    v16 は長さの組の角度制限を公称に揃える (`rope_specs.make_size_variant(..., keep_joint_limits=True)`)。揃っていないと、
    寸法の間で書き戻した関節角が制限を超えて飛ぶ (`docs/experiments/2026-10-17_v16_size-gate.md`)。DOF の名前の規則
    (`<関節>:0` = 捩り、`:1`・`:2` = 曲げ) は `RandomizeRopePhysics` と同じ。制限の無い組 (`bend_limit_deg=None`) は見ない。
    """
    import math

    rope = env.scene[rope_asset_name]
    suffix = [name.rsplit(":", 1)[-1] for name in rope.joint_names]
    twist = torch.tensor([sfx == "0" for sfx in suffix])
    upper = torch.rad2deg(rope.data.joint_pos_limits[..., 1]).cpu()  # (N, J)
    for k, name in enumerate(sizes.names):
        spec = get_spec(name)
        rows = (sizes.index == k).cpu()
        for mask, limit, label in ((~twist, spec.bend_limit_deg, "曲げ"), (twist, spec.twist_limit_deg, "捩り")):
            if limit is None:
                continue
            err = (upper[rows][:, mask] - limit).abs().max()
            if not math.isfinite(float(err)) or float(err) > atol_deg:
                raise RuntimeError(
                    f"{name} の{label}の角度制限が PhysX と違う (プリセット ±{limit:.2f}°、最大のずれ {float(err):.3f}°)"
                )
