"""rsl_rl の PPO に渡す対称性のデータ拡張の関数 (`RslRlSymmetryCfg.data_augmentation_func`)。v8 で追加。

変換の中身 (鏡像・逆順、観測の項ごとの扱い) は `topology/symmetry.py`。ここは環境から観測の
項の並びを読み、`TensorDict` の各グループに当てはめるだけ。

rsl_rl (3.0.1) の PPO は、更新のミニバッチごとに `data_augmentation_func(obs=..., actions=..., env=...)`
を呼び、返ってきた `K` 倍のバッチで方策と価値を学習する (先頭の 1 倍が元のデータ。
`rsl_rl/algorithms/ppo.py` の `update`)。advantage・return・旧方策の対数尤度は元のものを
`K` 回繰り返して使う。

変換の組は関数で選ぶ (Hydra の上書き
`agent.algorithm.symmetry_cfg.data_augmentation_func=<module>:augment_mirror` で比べられるように):

* `augment_mirror_reverse` (v8 の既定): 恒等・鏡像・逆順・両方の 4 倍
* `augment_mirror` / `augment_reverse`: 2 倍
"""

from __future__ import annotations

import math

import torch

from ..topology import symmetry as sym

__all__ = ["augment_mirror", "augment_mirror_reverse", "augment_reverse", "policy_obs_layout"]


def policy_obs_layout(env, group: str) -> list[sym.ObsTermLayout]:
    """観測グループ `group` の項の並び `[(名前, 次元, params), ...]` を環境から読む。

    `env` はラッパー (`RslRlVecEnvWrapper`) でも `ManagerBasedRLEnv` でもよい。添字を固定で
    書かないのは、版で観測の項を足したときに黙ってずれないようにするため。
    """
    manager = env.unwrapped.observation_manager
    names = manager.active_terms[group]
    dims = [math.prod(d) for d in manager.group_obs_term_dim[group]]
    cfgs = manager._group_obs_term_cfgs[group]
    return [(name, dim, dict(cfg.params or {})) for name, dim, cfg in zip(names, dims, cfgs)]


def _augment(env, obs, actions, transforms: str):
    obs_out = None
    if obs is not None:
        from tensordict import TensorDict

        k = len(sym.TRANSFORMS[transforms])
        groups = {key: sym.augment_policy_obs(obs[key], policy_obs_layout(env, key), transforms) for key in obs.keys()}
        obs_out = TensorDict(groups, batch_size=[obs.batch_size[0] * k], device=obs.device)
    act_out = sym.augment_actions(actions, transforms) if actions is not None else None
    return obs_out, act_out


def augment_mirror_reverse(env, obs=None, actions=None) -> tuple:
    """恒等・鏡像・逆順・鏡像+逆順の 4 倍 (v8 の既定)。"""
    with torch.no_grad():
        return _augment(env, obs, actions, "mirror_reverse")


def augment_mirror(env, obs=None, actions=None) -> tuple:
    """恒等・鏡像の 2 倍。"""
    with torch.no_grad():
        return _augment(env, obs, actions, "mirror")


def augment_reverse(env, obs=None, actions=None) -> tuple:
    """恒等・逆順の 2 倍。"""
    with torch.no_grad():
        return _augment(env, obs, actions, "reverse")
