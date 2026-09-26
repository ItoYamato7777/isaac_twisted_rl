"""バージョン非依存の共有層。

ここに入るのは「どのバージョンの課題でも変わらないもの」:

    env_cfg_base.py     シーン・物理・decimation の基底設定 (KnotTyingEnvCfg)
    rope_task_cfg.py    ロープのプリセットを設定へ流し込むヘルパ (_apply_rope)
    rope_state.py       Articulation <-> topology/ の接続層
    mdp/                MDP 項のライブラリ。**全バージョンの項をここに集める**
    topology/           3D 点列 -> 位相 (交差・p-data・結び目判定) の純ロジック層
    rope_model/         ロープの USD 生成と設定
    parallel_gripper/   グリッパーの USD 生成と設定

課題そのもの (どの観測を渡し、どう報酬を付けるか) は `../versions/vN/` 側。
この分離の意図は `../README.md` を参照。

## 共有層を触るときの注意

**ここを変えると全バージョンの挙動が変わる。** バグ修正ならそれが狙いだが、
課題の設計を変えたいときは `mdp/` に**新しい項を足して**新バージョンだけが
使うようにすること。既存の項の削除・改名は過去バージョンを壊す
(`../README.md` の「バージョンを追加するときの規約」)。

注意: この __init__ は `tasks/__init__.py` の `import_packages` によって
**Kit 起動前**に実行される。`isaaclab.sim` などを import すると
`list_envs` / `train` が丸ごと落ちるので、ここは docstring だけにしておく。
"""
