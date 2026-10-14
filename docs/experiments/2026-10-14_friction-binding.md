# ロープと指の実際の摩擦は文書の値 (2.0 / 4.0) でなく、PhysX の既定の 0.5 (2026-10-14)

- 状態: 凍結
- 読むべき状況: ロープ・指・床の摩擦の値を使う・変えるとき。物理の DR (v14 以降) の公称値を知りたいとき。過去の学習の物理の条件を書くとき

## 目的と仮説

v14 の摩擦の DR を実装するときに、PhysX の tensor API (`get_material_properties`) がロープの 28 形状と指の形状の摩擦を
どれも 0.5 と読み戻した。文書 (`rope_specs.py` の `ROPE_FRICTION` = 2.0、`mdp/events.py` の `bind_grasp_material` = 4.0 の `max` 合成)
と違う。読み戻しが間違っているのか、材質が PhysX に届いていないのかを確かめる。

## 条件

| 項目 | 値 |
|---|---|
| commit | `2bcbfc8` (未コミットの差分: あり。v14 の実装の途中で、計測のスクリプト `demos/friction_check.py` を含む。job の `git.diff`) |
| タスク ID とプリセット | `Isaac-Twisted-RL-Knot-Play-v13` (`Knot`)。Play なのでまっすぐなロープから始まる。物理は v4〜v13 で同じ |
| 計測のコマンド | `src/isaac_twisted_rl/scripts/run_job.sh --name friction_check --timeout 30m --detach -- uv run python src/isaac_twisted_rl/tasks/knot_tying/demos/friction_check.py --headless --device cuda:0` (既定: 滑り 4 env × 5 条件、指 24 env × 6 条件、seed 3) |
| job | `outputs/jobs/2026-10-14_23-12-52_friction_check` (`signal:` 無し、Traceback 無し)。出力 `outputs/friction_check/summary.log` |
| 試験の中身 | `demos/friction_check.py` の docstring |

## 結果

### 1. USD の bind

| 形状 | bind の先 | 先の prim の型 | 先の μ | `ComputeBoundMaterial("physics")` |
|---|---|---|---|---|
| ロープ `Link00/geom` | `/World/envs/env_0/Rope/RopeMaterial` | **`Scope`** | 2.0 | **なし** |
| 指 `left_finger/geom` | `/World/PhysicsMaterials/GraspMaterial` | `Material` | 4.0 | あり |

ロープの USD (`generate_rope_usd.py`) は材質の API を `Scope` の prim に付けて bind している。physics の bind は `Material` の prim でないと解決されない。

### 2. 軸方向の滑り (ロープ↔床、決定論的)

まっすぐなロープを床に 1 s 落ち着かせ、軸の方向に 1 m/s を与えて 2 s 後の重心の移動距離 d を測った。μ_eff = 1 / (2 g d)。

| ロープの材質 | d [m] (4 env の中央値。最小〜最大の差は 1e-5 以下) | μ_eff | (μ_rope + 2.0) / 2 |
|---|---|---|---|
| 何も書かない (USD のまま) | **0.03869** | 1.317 | — |
| tensor API で μ = 0.1 | 0.04648 | 1.097 | 1.05 |
| tensor API で μ = 0.5 | **0.03868** | 1.318 | 1.25 |
| tensor API で μ = 1.0 | 0.03181 | 1.602 | 1.5 |
| tensor API で μ = 2.0 | 0.02331 | 2.186 | 2.0 |

- 何も書かないロープは μ = 0.5 を書いたロープと 4 桁まで同じに止まる。μ = 2.0 なら 0.0233 m のはずで、そうはならない。
- μ_eff は average 合成の値より 5〜10% 大きい (止まる直前の静止摩擦などで等減速から外れるためと見るが、未確認)。順序と間隔は average 合成 (床 2.0) と合う。

### 3. 指の摩擦 (ロープ↔指、混沌的なので組の平均で比べる)

同じ 1 手 (固定の乱数の行動) を、指の μ だけを tensor API で変えた 6 組 (各 24 env) で打ち、1 手目の後のロープの形の組の平均が、組 0 (何も書かない) からどれだけ離れるか。

| 組 | 指の材質 | 組 0 からの距離 [m] | 組の中のばらつき [m] |
|---|---|---|---|
| 0 | 何も書かない | 0 | 0.062 |
| 1 | 何も書かない | **0.0155** (雑音の目安) | 0.054 |
| 2 | μ = 0.05 | 0.104 | 0.062 |
| 3 | μ = 0.5 | **0.0165** | 0.071 |
| 4 | μ = 2.0 | 0.063 | 0.049 |
| 5 | μ = 4.0 | 0.097 | 0.070 |

- 何も書かない指は、μ = 0.5 を書いた指と雑音の範囲で同じに動き、μ = 4.0 を書いた指とは違う動きになった。tensor API の読み戻しも 0.5。

## 結論

- 分かった:
  - **ロープの摩擦は 0.5 (PhysX の既定の材質) で、USD に書いた 2.0 は効いていない。** bind の先が `Scope` で解決されないため。ロープ↔床は average 合成で約 1.25、ロープ↔ロープは 0.5。
  - tensor API (`set_material_properties`) で書いた摩擦は効き、読み戻しは実際の値を返す (滑りの試験で、書いた値の順に止まる距離が変わり、何も書かない値は読み戻しの 0.5 と一致した)。
  - v0 以降のすべての学習・評価・ゲート 1 は、この摩擦で行われた。文書・docstring の「2.0」「4.0 (`max`)」は意図で、実際の物理ではない。
- 推測:
  - **指の摩擦も 0.5 (average 合成) で、`bind_grasp_material` の 4.0 (`max`) は効いていない。** USD の上の bind は解決するが、startup のイベントは PhysX が scene を読み込んだ後に回るので、後から付けた bind が PhysX に届かないと見る。読み戻しと試験 3 が合うが、試験 3 は混沌的で雑音が大きい。
  - rope study の剛体鎖の `LowMu` (ロープどうし・床 0.5、指 1.0。tensor API の `randomize_rigid_body_material` で書く) は、ロープの摩擦では既定 (実際は 0.5) と違わなかった。違ったのは床と指だけ。
- 未確認:
  - 摩擦を意図どおり (ロープ 2.0、指 4.0 の `max`) にしたときに、ゲート 1 (台本の固結び) と学習がどう変わるか。
  - tensor API で書いた指の材質の合成のモード (v14 の DR の範囲では、指もロープも average の材質なので結果に関わらない)。

## 次の作業

- v14 の DR は、PhysX から読んだ実際の公称値 (0.5) に倍率 ×0.5〜×2 を掛ける ([0027](../decisions/0027-v14-s0-reshape-dr-priv-critic.md))。
- 既存の版の挙動は変えない。`rope_specs.py`・`mdp/events.py` の docstring に、実際の値を warning として書いた。
- 摩擦を意図どおりにする (USD の材質を `Material` の prim にする・指の bind を scene の読み込みの前にする) のは、後の版の候補。物理を変えるので skill `physics-check` から。
