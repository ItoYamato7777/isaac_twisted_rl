"""学習の run の `params/env.yaml` と、評価の環境の設定の照合。isaaclab を import しない (yaml だけ)。

評価に学習の Hydra の上書き (物理など) を付け忘れると、違う物理のまま、エラーも出ずに測ってしまう。
chain 評価 (`scripts/rsl_rl/eval_chain.py`) は評価の環境の設定を同じ `dump_yaml` で書き出し、ここで学習の
env.yaml と比べる。違いのうち、Play が意図して変える項は許す。どの項を Play が変えるかは手で並べず、同じ版の
学習用と Play 用の設定を既定のまま作って比べた差で決める (版ごとの `play_post_init` の中身を写さずに済む)。

Play も変える項を学習でも上書きしていた場合は、付け忘れても見つけられない (その項は Play の差として許す)。
"""

from __future__ import annotations

from typing import Any

import yaml

DEFAULT_IGNORE = ("seed", "log_dir", "io_descriptors_output_dir", "scene.num_envs", "sim.device")
"""学習と評価で必ず違い、物理を変えない項 (`eval_chain.py` と `train.py` がコマンドの引数と run のフォルダから書く)。"""

_PATH_TOKENS = {"{ENV_REGEX_NS}": "/World/envs/env_.*"}
"""学習の env.yaml は環境を作った後に書かれ、scene の prim_path の名前空間の記号が展開されている
(isaaclab の `InteractiveScene` が設定を書き換える)。評価の設定は環境を作る前に比べるので、記号を展開して揃える。"""


class _CfgLoader(yaml.SafeLoader):
    """isaaclab の `dump_yaml` が書く `!!python/tuple` を list として読む (それ以外の python タグは書かれない)。"""


_CfgLoader.add_constructor("tag:yaml.org,2002:python/tuple", lambda loader, node: loader.construct_sequence(node))


def load_cfg_yaml(text: str) -> dict:
    """`dump_yaml` が書いた設定の yaml を読む。"""
    return yaml.load(text, Loader=_CfgLoader)  # noqa: S506 (SafeLoader に tuple だけを足したもの)


def _normalize(v: Any) -> Any:
    if isinstance(v, str):
        for token, resolved in _PATH_TOKENS.items():
            v = v.replace(token, resolved)
    return v


def flatten(d: Any, prefix: str = "") -> dict[str, Any]:
    """入れ子の辞書を `a.b.c` のキーの平らな辞書にする。list と辞書でない値は葉。文字列の名前空間の記号は展開する。"""
    if not isinstance(d, dict):
        return {prefix: _normalize(d)}
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict) and v:
            out.update(flatten(v, key))
        else:
            out[key] = _normalize(v)
    return out


def diff_flat(a: dict[str, Any], b: dict[str, Any]) -> tuple[dict[str, tuple[Any, Any]], list[str], list[str]]:
    """`(値の違うキー -> (a, b), a だけのキー, b だけのキー)`。キーは並べて返す。"""
    changed = {k: (a[k], b[k]) for k in sorted(a.keys() & b.keys()) if a[k] != b[k]}
    return changed, sorted(a.keys() - b.keys()), sorted(b.keys() - a.keys())


def _ignored(key: str, prefixes: tuple[str, ...]) -> bool:
    return any(key == p or key.startswith(p + ".") for p in prefixes)


def unexpected_diff(
    train: dict,
    evaluation: dict,
    train_default: dict,
    play_default: dict,
    ignore: tuple[str, ...] = DEFAULT_IGNORE,
) -> dict[str, Any]:
    """学習の設定 `train` と評価の設定 `evaluation` の違いから、Play の意図した変更と `ignore` を除いたもの。

    Play の意図した変更は、`train_default` (学習用の設定の既定) と `play_default` (Play 用の設定の既定) で
    値が違う・片方にしか無いキー。返り値は `{"changed": {キー: (学習, 評価)}, "only_in_train": [...],
    "only_in_eval": [...]}`。片方にしか無いキーは、学習の後にコードが項や引数を足した・消したことを表す。
    """
    changed, only_t, only_e = diff_flat(flatten(train), flatten(evaluation))
    p_changed, p_only_t, p_only_e = diff_flat(flatten(train_default), flatten(play_default))
    intended = set(p_changed) | set(p_only_t) | set(p_only_e)

    def _keep(key: str) -> bool:
        return key not in intended and not _ignored(key, ignore)

    return {
        "changed": {k: v for k, v in changed.items() if _keep(k)},
        "only_in_train": [k for k in only_t if _keep(k)],
        "only_in_eval": [k for k in only_e if _keep(k)],
    }
