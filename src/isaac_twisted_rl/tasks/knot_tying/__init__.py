"""紐結び (knot tying) タスクパッケージ。

TWISTED-RL (itoyama_twisted_rl リポジトリ) の MuJoCo 実装を Isaac Lab へ
移植したもの。**課題はバージョン制**で、改良は既存バージョンを書き換えず
新しいバージョンを足していく:

    uv run train --task Isaac-Twisted-RL-Twist-v0 --headless   # 交差を 1 つ作る
    uv run train --task Isaac-Twisted-RL-Twist-v1 --headless   # 固結び (学習失敗)
    uv run train --task Isaac-Twisted-RL-Twist-v2 --headless   # 固結び (作り直し)
    uv run train --task Isaac-Twisted-RL-Twist-v2-R2 --headless   # v2 + R2 除去で段を判定
    uv run train --task Isaac-Twisted-RL-Knot-v3 --headless    # 論文の中核を取り込んだ版 (Knot で学習)
    uv run train --task Isaac-Twisted-RL-Knot-v4 --headless    # 段ごとのエージェント + 1 遷移 = 1 エピソード
    uv run train --task Isaac-Twisted-RL-Knot-v5 --headless    # v3 + pick&place の発散対策
    uv run train --task Isaac-Twisted-RL-Knot-v6 --headless    # v5 + 次にたどる位相の辺のリンクマスクを観測に

構成と設計方針、バージョンの増やし方は `README.md` を参照。

    common/     バージョン非依存の共有層 (物理・MDP 項・位相判定・資産)
    versions/   バージョンごとの配線 (観測・報酬・PPO・README)
    demos/      RL 環境からは import されない独立スクリプト

USD 資産 (ロープ 5 種とグリッパー) はリポジトリにコミットしていない
(`.gitignore` の `**/*.usd`)。学習やデモの前に一度だけ生成すること:

    uv run python -u src/isaac_twisted_rl/tasks/knot_tying/demos/generate_assets.py

注意: `isaac_twisted_rl.tasks/__init__.py` の `import_packages` による自動
インポートで、この __init__ と配下のパッケージの __init__ が読み込まれる。
**これらに重い import を置かないこと** (`isaaclab.sim` などは Kit 起動前に
import できず、`list_envs` / `train` が丸ごと落ちる)。デモ本体は plain
module なので自動 import されない (`import_packages` はパッケージの
__init__ しか実行しない)。
"""

import gymnasium as gym

# 登録するバージョン。**新しいバージョンを足すときはここに 1 行足すだけ**
# (`versions/vN/` に env_cfg.py と agents/ を置いたうえで)。
_VERSIONS = ("v0", "v1", "v2", "v2-R2", "v3", "v4", "v5", "v6")


def _version_package(ver: str) -> str:
    """タスク ID のバージョン表記 -> `versions/` 配下のパッケージ名。

    `v2-R2` のようにハイフンを含む表記はそのままでは Python の
    パッケージ名にできないので `v2_r2` に写す。`v0`〜`v2` は恒等写像。
    """
    return ver.replace("-", "_").lower()


# ロープのプリセット。学習に使うのは `Twist` (v0〜v2-R2) と `Knot` (v3) で、
# `Simple` / `Stiff` / `Fine` は比較用ベースライン。
_ROPE_PRESETS = ("Twist", "Simple", "Stiff", "Fine", "Knot")

# バージョンごとの **エージェント変種**。キーがタスク ID の接尾辞、値が
# `versions/vN/agents/rsl_rl_ppo_cfg.py` 内の Runner クラス名。
#
# v2 で入れた仕組み。**環境設定 (観測・報酬・物理) は 1 つで、方策の
# ネットワーク構造だけを変えたい**という要求のため (MLP / GRU / LSTM の比較)。
# `OnPolicyRunner` は `policy_cfg["class_name"]` から方策を 1 つだけ作る
# (`rsl_rl/runners/on_policy_runner.py:418`) ので、1 タスクの中で複数の
# ネットワークを切り替えることはできない。**同じ env cfg に対して別の
# agent cfg を登録する**のが、ライブラリを改造せずにできる唯一の形。
#
# 接尾辞はバージョンの **後ろ** に付ける (`...-Twist-v2-RNN`)。ID の末尾が
# バージョンという既存の規則を崩さないため。
#
# ここに載っていないバージョンは `_DEFAULT_VARIANTS` が使われ、
# 発行される ID は従来と完全に同一になる (v0 / v1 は影響を受けない)。
_AGENT_VARIANTS: dict[str, dict[str, str]] = {
    "v2": {
        "": "KnotTyingPPORunnerCfg",
        "-RNN": "KnotTyingRNNPPORunnerCfg",
        "-LSTM": "KnotTyingLSTMPPORunnerCfg",
    },
    "v2-R2": {
        "": "KnotTyingPPORunnerCfg",
        "-RNN": "KnotTyingRNNPPORunnerCfg",
        "-LSTM": "KnotTyingLSTMPPORunnerCfg",
    },
    "v3": {
        "": "KnotTyingPPORunnerCfg",
        "-RNN": "KnotTyingRNNPPORunnerCfg",
        "-LSTM": "KnotTyingLSTMPPORunnerCfg",
    },
    "v5": {
        "": "KnotTyingPPORunnerCfg",
        "-RNN": "KnotTyingRNNPPORunnerCfg",
        "-LSTM": "KnotTyingLSTMPPORunnerCfg",
    },
    "v6": {
        "": "KnotTyingPPORunnerCfg",
        "-RNN": "KnotTyingRNNPPORunnerCfg",
        "-LSTM": "KnotTyingLSTMPPORunnerCfg",
    },
}

_DEFAULT_VARIANTS = {"": "KnotTyingPPORunnerCfg"}
"""変種を定義していないバージョンの既定 (1 タスク = 1 エージェント)。"""

# 環境設定は **文字列 entry point** で参照する。この __init__ は
# `tasks/__init__.py` の `import_packages` によって **Kit 起動前** に実行され、
# `isaaclab.sim` を import すると `list_envs` / `train` が丸ごと落ちるため
# (上の docstring の注意を参照)。文字列にしておけば、実際の import は
# `gym.make` の時点まで遅延される。
for _ver in _VERSIONS:
    _pkg = f"{__name__}.versions.{_version_package(_ver)}"
    _variants = _AGENT_VARIANTS.get(_ver, _DEFAULT_VARIANTS)
    for _preset in _ROPE_PRESETS:
        for _suffix, _runner in _variants.items():
            _agent_ep = f"{_pkg}.agents.rsl_rl_ppo_cfg:{_runner}"
            gym.register(
                id=f"Isaac-Twisted-RL-{_preset}-{_ver}{_suffix}",
                entry_point="isaaclab.envs:ManagerBasedRLEnv",
                kwargs={
                    "env_cfg_entry_point": f"{_pkg}.env_cfg:{_preset}EnvCfg",
                    "rsl_rl_cfg_entry_point": _agent_ep,
                },
                disable_env_checker=True,
            )
            gym.register(
                id=f"Isaac-Twisted-RL-{_preset}-Play-{_ver}{_suffix}",
                entry_point="isaaclab.envs:ManagerBasedRLEnv",
                kwargs={
                    "env_cfg_entry_point": f"{_pkg}.env_cfg:{_preset}PlayEnvCfg",
                    "rsl_rl_cfg_entry_point": _agent_ep,
                },
                disable_env_checker=True,
            )
