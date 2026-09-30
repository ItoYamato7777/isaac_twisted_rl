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
* 600 手 (= 50 iteration、チェックポイントの間隔) ごとに `{log_dir}/archive.pt` へ書く。
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

## 初期アーカイブと smoke test (2026-09-28)

**初期アーカイブ** (`overhand_script.py --dump_archive`、`Knot`、128 環境、±25%):
段 0 / 1 / 2 = 128 / 264 / 130 件。台本の固結びは 123/128 (発散 0)。cover の手の後は
**126/128 が段 2** だったが、R2 除去後の class で数えると C2 は 18 だけで、110 は C1 だった
(v3 のアーカイブではこれらが C1 側に入っていた)。

**smoke test** (`Knot`、64 環境 x 3 iteration、V3 の学習と GPU を共有):

| iter | `success_s0` / `s1` / `s2` | `fail_s1` / `s2` | `diverged_s0` / `s1` / `s2` | `arc_size_s0` / `s1` / `s2` | `restore_mismatch` | `value_function` |
|---|---|---|---|---|---|---|
| 0 | 0.69 / 0.21 / 0.08 | 0.52 / 0.77 | 0.07 / 0.08 / 0.07 | 304 / 437 / 169 | 0.000 | 0.52 |
| 1 | 0.74 / 0.10 / 0.07 | 0.64 / 0.86 | | 610 / 794 / 220 | 0.000 | 0.41 |
| 2 | 0.80 / 0.11 / 0.09 | 0.69 / 0.86 | | 926 / 1136 / 277 | 0.000 | 0.35 |

* 初期アーカイブのおかげで、3 段とも iteration 0 からエピソードがある。ほぼランダムな初期方策では、
  段 1 / 2 から始めると 5〜8 割が段を下げて (輪を崩して) 終わる。
* 1 iteration 約 780 s。V3 (4096 環境) と GPU を共有していて物理ステップが遅く、環境数を減らしても
  短くならない。
* 切り分けの上書き 4 種を同時に指定した 1 iteration も走り、`params/*.yaml` に反映され、
  `archive.pt` が書き出されて読み戻せることを確かめた (`save_every=6` に上書き)。
* Play (chain モード) は 18 手まで続き (段が変わっても終わらない)、`exported/policy.pt` と
  `policy.onnx` が出力された。ONNX の exporter は `policy.actor[0].in_features` を読むので、
  `GatedMLP` に先頭のヘッドの層を返す `__getitem__` を持たせている。
* 終了時の `TypeError: post_quit()` は、学習を終えた後の Kit の終了処理 (`kit_exit` に None が渡る) で出るもので、v4 の項とは関係ない。

## この版で解決しないこと (今後の候補)

* **Mirror / Reverse 対称性の拡張** (Yan2020 / TWISTED)。RSL-RL の `symmetry_cfg` で入れられる。
  交差特徴 (点番号・上下・符号) の付け替えが要る。
* **推論時の探索** (TWISTED の K 候補、Yan2020 の木探索)。同じ形を N 環境に複製し、方策の
  手を N 本試して段が上がったものを残す。論文と同じ「探索ありの成功率」で比べられる。
* **次の段の V(s′) を報酬にする多段の微調整** (Yan2020)。v4 は段を上がった時点で終わるので、
  エージェント k は「次の段が打ちやすい形」を学ばない。chain モードの env を学習に一部混ぜると、
  GAE が次の段の critic の値で bootstrap するので、この形になる。
* **p-data ごとに一様に選ぶ 2 段の抽出** (論文 §IV-D の厳密な形)。今は段の中で形状を一様に
  選ぶので、多く現れる p-data (輪など) に偏る。
* **発散 (v3 で 1 手あたり約 3%) の原因調査**。どのフェーズで発散するか。ゲート 1 の
  「`env.step` で再生すると台本より発散が多い」と同じ問題の可能性がある。
* アーカイブはファイルに書くが、`--resume` で自動では読まない (`load_paths` に指定する)。
