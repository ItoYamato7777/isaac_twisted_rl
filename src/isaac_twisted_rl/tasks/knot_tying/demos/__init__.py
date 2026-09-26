"""RL 環境からは import されない、独立したデモ・生成スクリプト。

    generate_assets.py    ロープ 4 種とグリッパーの USD をまとめて生成する
    rope_catch.py         1 本のロープを平行グリッパーで掴むデモ
    multi_rope_catch.py   全プリセットを横並びにして同時に掴み比べるデモ
    topology_debug.py     台本どおりに操作しながら p-data の変化を記録し、
                          xy 投影図に交差点を描くデモ (位相抽出の目視検証用)
    spawn_rope.py         ロープを落として落ち着かせるだけのデモ
    parallel_gripper.py   グリッパーの開閉デモ

どれも `if __name__ == "__main__"` を持つ実行スクリプトで、
`uv run python -u <path>` で直接起動する。使い方は `../README.md`。

注意: この __init__ は `import_packages` によって Kit 起動前に実行される
(docstring だけなので安全)。デモ本体はプレーンモジュールなので
**自動 import されない** — `import_packages` はパッケージの __init__ しか
実行しないため、`isaaclab.app` を import していても `list_envs` は壊れない。
"""
