# v4: 段ごとのエージェント (TWISTED-RL-C) と 1 遷移 = 1 エピソード

```bash
# 最初に一度: ゲート 1 の台本の途中状態を初期アーカイブとして書き出す
uv run python -u src/isaac_twisted_rl/tasks/knot_tying/demos/overhand_script.py \
    --task Isaac-Twisted-RL-Knot-v2-R2 --num_envs 128 --jitter 0.25 --headless --dump_archive

uv run train --task Isaac-Twisted-RL-Knot-v4 --headless           # 学習 (段 0 / 1 / 2 の 3 エージェント)
uv run play  --task Isaac-Twisted-RL-Knot-Play-v4 --num_envs 16   # まっすぐなロープから段をつないで評価
```

ログ: `logs/rsl_rl/knot_tying_v4/`。**学習には `Knot` プリセットを使う** (v3 と同じ理由)。

## なぜ作り直したか

v3 の最初の run (`logs/rsl_rl/knot_tying_v3/2026-09-28_13-25-00`、1024 環境。既定の環境数を
4096 にしたコミット `a1c75cc` の際に run ごと置き換えられた) は、26 iteration (約 3.3 h) の時点で
次の状態だった:

| 指標 | 値 | 読み |
|---|---|---|
| `scratch_success` | 0〜0.8% で横ばい | まっすぐなロープからはほぼ結べていない |
| `archive_success` | 5% 前後で横ばい | C1 / C2 から始めても伸びていない |
| `Policy/mean_noise_std` / `Loss/entropy` | 0.98 / 8.4 (初期値のまま) | PPO はまだほぼランダム方策 |
| `Episode_Termination/diverged` | 33% | エピソードの 1/3 が物理の発散で終わる |
| `r2_c2` / `r2_c3` (1 手あたり) | 1.2% / 0.3% | ランダムな手と同じ水準 |
| `arc_size_c1` / `c2` | 4096 (1 iter で満杯) / 3403 | cover 後の状態は R2 除去後 class 1 なので **C1 側に入る** |
| 速度 | 455 s/iter (26 手/s) | 2000 iter で約 10 日 |

v3 は「まっすぐなロープから固結びまでを 1 本の方策・1 エピソード (12 手) で解く」定式化の
ままだった。TWISTED の SAC+HER ベースラインが「交差 1 は解けるが交差 2 でほぼ 0」だったのと
同じ形で、論文 (TWISTED-RL) はこれを次の 2 つで越えている
(`paper/explainer/twisted-rl.html` §04, §06, §09):

* **エージェントを交差数ごとに分ける (C 版)**。G (1 つ) / A (移動の種類別) / C / AC のうち、
  C が全テストセットで最良 (3-Medium 44%、3-Hard 46%。G は 28% / 38%)。
* **1 回の位相遷移 = 1 エピソード (多段エピソード)**。最大 M 手、位相が変わったら終了。
  報酬は目標に達すれば +1、変わらなければ 0、別の状態へ移れば −1。

## v3 から何を・なぜ変えたか

| 項目 | v3 | v4 | 理由 |
|---|---|---|---|
| エージェント | 1 つ | **段 0 / 1 / 2 ごとに独立した actor・critic・std** (`StageGatedActorCritic`) | TWISTED-RL-C |
| エピソード | まっすぐ or アーカイブから固結びまで (12 手) | **段 k から始め、段が変わったら終了** (最大 6 手) | 多段エピソード (§06) |
| 報酬 | BFS ポテンシャル整形 + 段を初めて上がった手のボーナス | **段が上がる +1 / 下がる −1 / 発散 −1 / それ以外 0** | 論文の +1 / 0 / −1 |
| 終了 | 時間切れ・発散・固結び | 時間切れ・発散・**段が上がる・段が下がる** | 同上 |
| 初期状態 | 確率 0.5 でアーカイブ (class 1 : 2 = 1 : 2) | **段を等確率で選び、その段のアーカイブから**。段 0 は半分がまっすぐなロープ | 論文 §IV-D (遷移を一様に選び、出発点の形を一様に選ぶ) |
| アーカイブのキー | R2 除去後の class (1, 2) | **段 (0, 1, 2)** | cover 後の状態が C1 に混ざっていた |
| アーカイブの中身 | 学習中に到達した形状 (起動時は空) | + **ゲート 1 の台本の途中状態をファイルから読む**。学習中も `{log_dir}/archive.pt` へ書く | 段 2 のエージェントが最初から練習できる。`--resume` で空に戻らない |
| 観測 | 129 次元 (`Knot`) | + **段の one-hot (4) を末尾に** (133 次元) | 方策がこの列でエージェントを選ぶ |
| `init_noise_std` | 1.0 | **0.5** | [−1, 1] のクリップで 1 チャネルあたり約 32% が端に張り付いていた (Report §6.5) |
| 1 iteration の手数 (`num_steps_per_env`) | 12 (4096 環境で 49152 手、約 20 分) | **4** (16384 手) | 下の「学習速度」。PPO の更新回数が律速だった |
| PhysX のソルバの位置反復 | 16 | **8** | 下の「学習速度」。1 物理ステップの時間のほぼ全部が反復の固定費だった |
| ロープ・尺・置き先の範囲・発散の扱い・PPO 本体・γ | | v3 のまま | |

共有層の変更はすべて **追加** (既存の項の挙動は変えていない):
`topology.BfsPotential.stage` / `STAGE_GOAL`、`StateArchive.save` / `load` / `state_dict`、
`mdp/stages.py` (新規)、`mdp/archive.py` の `ResetToStage` / `record_stage_archive` /
`log_stage_stats`、`common/policies/` (新規)、`demos/overhand_script.py` の `--dump_archive`。

## 段 (論文の「交差数」の代わり)

`stage = round(3 Φ)`。`Φ` は v3 の BFS ポテンシャル (`max(1 − d/3, R2 除去後のラダー/3)`)。
固結びへの正規の経路 (交差 0 → 1 → 2 → 3) に対応する。

| 段 | 含む状態 | エージェント |
|---|---|---|
| 0 | 自明 (まっすぐ・曲がっただけ) | π0 |
| 1 | 輪 1 つ・独立したねじれ 2 つ・偽の C2 | π1 |
| 2 | 本物の C2・**cover 後** (`1U4+_2O5+_3O6-_4O1+_5U2+_6U3-` など) | π2 |
| 3 | 固結び (ゴール) | — |

生の交差数を使わないのは、cover 後 (生 3・R2 除去後 1) や偽の C2 (生 3) が固結びまでの近さと
合わないため。値は v3 の BFS 表そのもので、新しく数え直した状態は無い (`tests/test_knot_graph.py`)。

## 1 遷移 = 1 エピソード

* 段が上がった・下がったは、手の終わりにロープが床に落ちていて (最高点が直径の 3 倍以下)、
  発散していないときだけ数える。吊ったままの手は「変わらない」扱いで続ける。
* 段が同じなら p-data が変わっても続ける (論文より寛容)。`strict_topology=true` で論文どおり
  「段が同じでも p-data が変わったら −1 で終了」にできる (下の切り分け)。
* 時間切れ (6 手) は `time_out=True` なので PPO は bootstrap する。段の変化による終了は bootstrap しない。
* `grasp_shaping` (−0.05 × 把持距離) / `failure` (−0.1) / `held_up` (−0.1) は v3 のまま残した。

## 段ごとのエージェント (`common/policies/stage_gated.py`)

* actor・critic・std を段ごとに持ち、観測の末尾 4 列 (段の one-hot) の argmax で振り分ける。
  各サンプルは自分の段のネットワークしか通らないので、勾配は他の段へ流れない
  (`tests/test_stage_gated_policy.py`)。共有しているのは適応学習率と、ミニバッチ内の
  advantage の正規化だけ。
* RSL-RL は `eval(class_name)` で方策クラスを作るので、`agents/rsl_rl_ppo_cfg.py` が import 時に
  `register_with_rsl_rl()` で登録する (train.py / play.py は変えていない)。
* `self.actor` が振り分け込みのモジュールなので、`play.py` の JIT / ONNX 出力はそのまま動く。
* 段 3 (固結び) の観測は段 2 のエージェントに回す (学習中は固結びで終わるので使われない)。

## 段ごとのアーカイブ (`ResetToStage`)

* 毎手の終わりに、床に落ちていて発散していない形状を段 0 / 1 / 2 のリングバッファ (各 16384。
  既定の 4096 環境では段 1 に 1 手で数千件入るので、v3 の 4096 件の 4 倍にした)
  へ入れる。段を上がった直後の形状は次の段の初期状態になる (論文の「l → k の遷移で到達した
  形を k の初期状態に使う」)。
* reset では、空でない段から `stage_weights = (1, 1, 1)` で段を選び、その段の中で一様に選ぶ。
  段 0 は確率 `straight_prob = 0.5` でまっすぐなロープにする。
* 起動時に `load_paths` を読む。既定は `outputs/archive_seed/overhand_{num_links}links.pt`
  (`overhand_script.py --dump_archive` の出力。`{num_links}` はリンク数)。無ければ警告して空から始める。
* 200 手 (= 50 iteration x 4 手、チェックポイントの間隔) ごとに `{log_dir}/archive.pt` へ書く。
  `--resume` するときは `env.events.reset_to_stage.params.load_paths` にそれを足す。

## Play (chain モード)

`Isaac-Twisted-RL-Knot-Play-v4` は評価用。まっすぐなロープから始め、**固結び・発散・18 手でだけ
終わる** (段が上がっても下がっても続ける)。観測の段が変わると次の段のエージェントに切り替わる。
`Episode_Termination/stage_up` (Play では固結びに達した割合) が v3 の `scratch_success` と比べる主指標。

## ログ (TensorBoard)

| key | 意味 |
|---|---|
| `Curriculum/stage_log/success_s{k}` | 段 k から始めたエピソードのうち、段が上がって終わった割合 (エージェント k の成功率) |
| `Curriculum/stage_log/fail_s{k}` / `diverged_s{k}` / `timeout_s{k}` | 同じく、段が下がった / 発散した / 6 手で時間切れの割合 |
| `Curriculum/stage_log/episodes_s{k}` | 前回のログから終わった、段 k 始まりのエピソード数 |
| `Curriculum/stage_log/chain_est` | `success_s0 × success_s1 × success_s2`。まっすぐなロープから固結びまでの粗い見積り (正確な値は Play で測る) |
| `Curriculum/stage_log/straight_success` | まっすぐなロープから始めた段 0 のエピソードの成功率 |
| `Curriculum/stage_log/start_frac_s{k}` | reset のうち段 k から始めた割合 |
| `Curriculum/stage_log/arc_size_s{k}` | 段 k のアーカイブの件数 (上限 16384) |
| `Curriculum/stage_log/restore_mismatch` | 復元した形状の段が保存時と違った割合。**0 に近くないなら復元が壊れている** |
| `Episode_Termination/stage_up` / `stage_down` | reset のうち段が上がって / 下がって終わった割合 (段は混ざる) |
| `Episode_Reward/transition` | +1 / 0 / −1 の報酬 |
| `Curriculum/topology_log/*` | v2-R2 / v3 と同じ |

`Episode_Reward/record_stage_archive` と `topology_stats` は常に 0 と表示されるが正しい。

## 切り分け (Hydra の CLI 上書き)

```bash
# G 版 (1 つの方策を全段で共有する。論文の G と C の比較)
uv run train --task Isaac-Twisted-RL-Knot-v4 --headless agent.policy.num_stages=1
# 初期アーカイブを読まない (学習中に到達した形状だけ)
uv run train --task Isaac-Twisted-RL-Knot-v4 --headless 'env.events.reset_to_stage.params.load_paths=[]'
# 論文どおりの厳密な終了 (段が同じでも p-data が変われば −1 で終了)
uv run train --task Isaac-Twisted-RL-Knot-v4 --headless env.rewards.transition.params.strict_topology=true \
    env.terminations.stage_down.params.strict_topology=true
# 行動の初期 std を v3 に戻す
uv run train --task Isaac-Twisted-RL-Knot-v4 --headless agent.policy.init_noise_std=1.0
```

## 学習速度 (2026-09-30)

v4 を v3 のままの設定 (4096 環境 x 12 手、ソルバ反復 16) で回すと **1 iteration が約 20 分**、
既定の 2000 iteration で約 27 日かかる。原因を測り、2 か所を変えた。**1 iteration は約 3 分になる。**

計測 (1 iteration の時間の内訳、ソルバの反復と台本の再生・発散率、A / B / C の 4 時間ずつの学習の比較) は実験記録
[`docs/experiments/2026-09-30_v4_speed.md`](../../../../../../docs/experiments/2026-09-30_v4_speed.md)、判断の要約は
[`docs/decisions/0020-v4-solver-iterations-and-steps-per-iteration.md`](../../../../../../docs/decisions/0020-v4-solver-iterations-and-steps-per-iteration.md)。
要点: 時間のほぼ全部が PhysX の物理ステップで、その固定費の正体はソルバの位置反復だった。序盤の PPO は手数より
更新回数で進んでいた。**反復と成否の関係は単調ではない** (12 が最悪) ので、8 以外の値に変えるときは必ず測り直すこと。

### 変えたこと

| 項目 | v3 から | v4 | 効果 |
|---|---|---|---|
| `num_steps_per_env` | 12 | **4** | 1 iteration の手数が 1/3。スループットは同じまま、更新が 3 倍の頻度で回る |
| ソルバの位置反復 | 16 | **8** (`KnotTyingV4EnvCfg.solver_position_iterations`) | スループット 47 -> 87 手/s (4096 環境) |
| `max_iterations` | 2000 | 6000 | 総手数は v3 の既定と同じ |
| アーカイブの保存間隔 | 600 手 | 200 手 | チェックポイント (50 iteration) と揃える |

合わせて **1 iteration 約 20 分 -> 約 3 分**、同じ時間で回る更新は約 6 倍、集まる手は約 1.8 倍。

`num_steps_per_env` を 4 にしても良い理由: v4 のエピソードは段が変われば終わり (最大 6 手)、報酬
(+1 / −1) は段を変えた手そのものに付く。4 手で打ち切られたエピソードは、GAE が次の状態の価値で
bootstrap する。環境数ではなく手数を減らしたのは、上の固定費のため (1024 環境 x 6 手でも 1 iteration は
約 4 分になるが、スループットが半分になる)。

### 変更前の設定に戻す (切り分け)

```bash
# 1 iteration 12 手に戻す (保存間隔も合わせる)
uv run train --task Isaac-Twisted-RL-Knot-v4 --headless agent.num_steps_per_env=12 \
    env.rewards.record_stage_archive.params.save_every=600
# ソルバの反復を 16 に戻す (solver_position_iterations は Hydra からは効かないので、資産の値を直接上書きする)
uv run train --task Isaac-Twisted-RL-Knot-v4 --headless \
    env.scene.rope.spawn.articulation_props.solver_position_iteration_count=16 \
    env.scene.gripper.spawn.articulation_props.solver_position_iteration_count=16
```

### 注意

* `--device cuda:1` は環境だけを移し、PPO は `agent.device` (既定 `cuda:0`) に残る。別の GPU で回すときは
  `agent.device=cuda:1` も付けること (`train.py` の既存の挙動)。
* 1 手 = 759 物理ステップ (3.16 s) の尺はまだ詰めていない。発散との関係 (`lower` / `release` / `carry` で
  多い) を調べてから。

## 学習の結果 (2026-10-01〜02、v5 / v6 との比較)

v5 / v6 と同じ条件で学習した overnight の run と chain 評価の結果は
[`docs/experiments/2026-10-01_v4-v6_overnight.md`](../../../../../../docs/experiments/2026-10-01_v4-v6_overnight.md) (数値の正本)。
run のフォルダ名・最後のチェックポイント・報告書に無い値 (iteration 70 の平均手数など) は
[`docs/experiments/2026-10-01_v4-v7_readme-notes.md`](../../../../../../docs/experiments/2026-10-01_v4-v7_readme-notes.md)。

## 初期アーカイブと smoke test (2026-09-28)

初期アーカイブの段ごとの件数と、smoke test (64 環境 x 3 iteration) の結果は
[`docs/experiments/2026-09-28_v4_smoke-test.md`](../../../../../../docs/experiments/2026-09-28_v4_smoke-test.md)。実装の注意だけをここに残す:

* ONNX の exporter は `policy.actor[0].in_features` を読むので、
  `GatedMLP` に先頭のヘッドの層を返す `__getitem__` を持たせている。
* 終了時の `TypeError: post_quit()` は、学習を終えた後の Kit の終了処理 (`kit_exit` に None が渡る) で出るもので、v4 の項とは関係ない。

## この版で解決しないこと (今後の候補)

* **Mirror / Reverse 対称性の拡張** (Yan2020 / TWISTED)。RSL-RL の `symmetry_cfg` で入れられる。
  交差特徴 (点番号・上下・符号) の付け替えが要る。
* **推論時の探索** (TWISTED の K 候補、Yan2020 の木探索)。同じ形を N 環境に複製し、方策の
  手を N 本試して段が上がったものを残す。論文と同じ「探索ありの成功率」で比べられる。
  (済み: 評価として実装し、評価は途中で打ち切り中。[`docs/experiments/2026-10-03_inference_search.md`](../../../../../../docs/experiments/2026-10-03_inference_search.md))
* **次の段の V(s′) を報酬にする多段の微調整** (Yan2020)。v4 は段を上がった時点で終わるので、
  エージェント k は「次の段が打ちやすい形」を学ばない。chain モードの env を学習に一部混ぜると、
  GAE が次の段の critic の値で bootstrap するので、この形になる。
* **p-data ごとに一様に選ぶ 2 段の抽出** (論文 §IV-D の厳密な形)。今は段の中で形状を一様に
  選ぶので、多く現れる p-data (輪など) に偏る。
* **発散 (v3 で 1 手あたり約 3%) の原因調査**。どのフェーズで発散するか。ゲート 1 の
  「`env.step` で再生すると台本より発散が多い」と同じ問題の可能性がある。
  (済み: v5 でフェーズ別に計測した。[`docs/experiments/2026-10-01_v5_divergence.md`](../../../../../../docs/experiments/2026-10-01_v5_divergence.md))
* アーカイブはファイルに書くが、`--resume` で自動では読まない (`load_paths` に指定する)。
