# v16 の寸法の DR: MultiAssetSpawnerCfg でロープを spawn できるか、起動・速さ・PhysX のバッファ (2026-10-17)

- 状態: 凍結
- 読むべき状況: v16 (寸法の DR) の spawn の仕組みを確かめるとき。replicate_physics を切ったときの起動の時間・学習の速さ・PhysX のバッファのあふれを知りたいとき

## 要約

- 結論: 28 リンクのロープを寸法の 9 組の USD で `MultiAssetSpawnerCfg` (env の番号 % 9) に spawn でき、4096 env の全部で USD の寸法が割り当てと一致した。replicate_physics を切ると起動が +19 s (48 -> 67 s)。PhysX の aggregate pairs のバッファが基底の上限からあふれる (「interactions を取りこぼす」) ので、寸法の DR のときだけ上限を上げた (total 2**20・found/lost 2**27)。上げた後はエラー 0、GPU のメモリの最大 12.1 GB、ランダムな手で 69.5 手/s (v15 のまま 90.4 手/s の 77%)。
- 有効か: ランダムな手 (`N(0, 0.5)`) で 4 手 (計時は 3 手) だけの計測。行動の範囲・床の判定は公称の寸法のまま (v15 の設定に spawn だけを差し込んだ)。学習中の速さ・バッファの必要量は方策の発散の多さで変わる (未確認)。組ごとの発散率は物理の DR (6 軸) を入れたままの値で、ゲート (skill `physics-check`) の代わりにはならない。

## 目的と仮説

[0029](../decisions/0029-v16-rope-size-dr.md) の「実装の順序」3 (最大の未確認): Isaac Lab 2.3.0 の `MultiAssetSpawnerCfg` が Articulation (28 リンクのロープ) で動くか、`scene.replicate_physics=False` での 4096 env の起動の時間と 1 手の時間。

## 条件

| 項目 | 値 |
|---|---|
| commit | 1 回目 `a7c81fd` + 未コミットの差分 (`outputs/jobs/2026-10-17_19-49-32_size_spawn_check/git.diff`。**未追跡だった `common/mdp/rope_size.py` と `demos/rope_size_spawn_check.py` は diff に無い**。中身は `df4d205` と同じ)、2 回目 `df4d205`、3 回目 `b6a21ff` (未コミットの差分: 未追跡の `video/sim/` だけ) |
| タスク ID とプリセット | `Isaac-Twisted-RL-Knot-v15` (`Knot`、学習の設定。物理の DR あり) に、ロープの spawn だけを `rope_task_cfg.apply_rope_sizes` で差し込んだ |
| 条件 | `grid`: 寸法の 9 組 (長さ ×0.8/1.0/1.25 × 直径 ×0.4/0.63/1.0)、replicate なし。`nominal`: 公称の 1 組を同じ経路 (`RopeSizeSpawnerCfg`)、replicate なし。`none`: v15 のまま (`UsdFileCfg`、replicate あり) |
| num_envs | 4096 (grid は 1 組 455〜456 env) |
| コマンド | `run_job.sh --name size_spawn_check{,2,3} ... -- uv run python -u src/isaac_twisted_rl/tasks/knot_tying/demos/rope_size_spawn_check.py --headless --device cuda:0 --num_envs 4096 --sizes <条件> --moves 3 [--found_lost_pairs 134217728] --label <名前>` |
| job | `outputs/jobs/2026-10-17_19-49-32_size_spawn_check` (3 条件)、`..._20-03-41_size_spawn_check2` (total の上限を上げた grid・nominal)、`..._20-13-46_size_spawn_check3` (found/lost も上げた grid) |
| 出力 | `outputs/rope_size_spawn_check/<label>/summary.json` |
| 行動 | ランダム `N(0, 0.5)` を [−1, 1] に切る。最初の 1 手は計時しない |
| 発散の判定 | 終了項 `diverged` の返り値を、項の呼び出しで捕まえる (1 手あたり) |
| GPU | RTX 3090 の GPU 0。ほかのジョブなし (GPU 1 は他のユーザーの 2.7 GB) |

## 結果

### spawn と割り当て

- 3 回とも `gym.make` が通った。`mdp.check_rope_sizes` (全 env の `Link00/geom` のカプセルの半径と円柱の長さを USD から読む) が全 4096 env で割り当て (env の番号 % 組の数) と一致した。

### 起動と速さ

| 条件 | replicate | aggregate pairs の上限 (total / found-lost) | `gym.make` [s] | 手/s | PhysX のエラー |
|---|---|---|---|---|---|
| none (v15 のまま) | あり | 16384 / 2**22 | 48.4 | 90.4 | 0 |
| nominal | なし | 16384 / 2**22 | 66.0 | 89.4 | total のあふれ 32 回 (必要 16652) |
| grid | なし | 16384 / 2**22 | 66.8 | 105.9 | total のあふれ 720 回 (必要 18379) |
| nominal | なし | 2**20 / 2**22 | 64.9 | 89.6 | found/lost のあふれ 2 回 (必要 665〜909 万) |
| grid | なし | 2**20 / 2**22 | 67.6 | 79.1 | found/lost のあふれ 3 回 (必要 8792〜9775 万) |
| grid | なし | 2**20 / **2**27** | 67.6 | **69.5** | **0** |

- found/lost 2**27 の grid の run の GPU 0 のメモリの最大は 12.1 GB (5 s ごとに `nvidia-smi` で読んだ)。

### 組ごとの 1 手あたりの発散 (物理の DR あり、ランダムな手、1 組あたり約 1820 手)

| 組 (長さ・直径) | 上限 16384 / 2**22 (あふれあり) | 2**20 / 2**27 (あふれなし) |
|---|---|---|
| ×0.8・×0.4 | 20.9% | 8.8% |
| ×0.8・×0.63 | 32.1% | 14.6% |
| ×0.8・×1.0 | 16.5% | 8.6% |
| ×1.0・×0.4 | 26.3% | 14.1% |
| ×1.0・×0.63 | 24.9% | 14.7% |
| ×1.0・×1.0 (`knot`) | 9.8% | 4.4% |
| ×1.25・×0.4 | 33.4% | 19.1% |
| ×1.25・×0.63 | 27.1% | 18.2% |
| ×1.25・×1.0 | 15.8% | 8.0% |

公称の 1 組だけの run (16384 手): none 2.89%、nominal (あふれあり) 3.22%、nominal (total 2**20、found/lost は 2 回あふれ) 3.20%。

## 結論

- 分かった:
  - `MultiAssetSpawnerCfg` (`random_choice=False`) で 28 リンクの Articulation を env ごとに別の USD で spawn でき、割り当ては env の番号 % 組の数 (Isaac Lab の `spawn_multi_asset` の実装どおり)。
  - replicate_physics を切ると、起動が 4096 env で約 18 s 長くなる。公称の 1 組だけなら 1 手の時間は変わらない (89.6 対 90.4 手/s)。
  - replicate なしでは、基底の `gpu_total_aggregate_pairs_capacity` (16384) と `gpu_found_lost_aggregate_pairs_capacity` (2**22) があふれ、PhysX が「interactions を取りこぼす」とエラーを出す。total のあふれは寸法の 9 組の run の発散を大きく増やした (公称の組で 9.8% 対 あふれなしで 4.4%)。found/lost を 2**27 にするとエラーは 0。
  - Isaac Lab 2.3.0 は replicate なしのとき env の ID を PhysX に渡さず、env どうしの接触を collision group で切る (`InteractiveScene.__init__` のコメント "env_ids is only applicable when replicating physics")。
  - 細い組・長い組の発散は公称より多い (ランダムな手・物理の DR ありで 8〜19% 対 4.4%)。
- 推測:
  - found/lost の必要量 (約 1 億) は、発散して飛んだ (または NaN の) ロープの AABB が他の env の aggregate と重なって、broadphase が env をまたぐペアを作るため。必要量が発散の多い grid で 10 倍大きいことと合う。collision group の判定は broadphase の後なので、バッファには入る。
  - grid の 69.5 手/s (v15 の 77%) の遅さの多くは、この env をまたぐペアの処理。学習の方策は発散がランダムな手より少ないので、学習ではここまで遅くならない見込み。
  - 公称の組の 4.4% (grid の中) と 3.2% (公称だけ) の差は、1820 手の揺れ (標準誤差 約 0.5 ポイント) の範囲に近い。
- 未確認:
  - 学習中の速さと、found/lost の必要量 (2**27 で足りるか)。学習のログの `PhysX error` で確かめる。
  - 物理の DR を切った、寸法だけの組ごとの発散とゲート 1 (`physics-check` のゲートで測る)。

## 次の作業

- 寸法の組ごとのゲート 1 と発散の計測 (物理の DR なし / あり)。通らない組を `versions/v16/env_cfg.py` の `ROPE_SIZE_SCALES` から外すかを決める。
- 学習のログで `PhysX error` (aggregate pairs のあふれ) が出ていないかを確かめる手順を、v16 の学習の確認に入れる。
