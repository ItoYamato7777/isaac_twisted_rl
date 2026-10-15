# 評価: chain 評価の定義と、版どうしの比べ方

- 状態: 現行 (v4 以降の評価の正本)
- 読むべき状況: 版や設定の良し悪しを比べるとき、評価のコマンドを組むとき、評価の数字を読むとき
- 最終更新: 2026-10-15

手順 (学習から評価・集計まで) は skill `train-eval` にある。この文書は「何を測るか」と「どう比べるか」だけを書く。

## 1. chain 評価

`src/isaac_twisted_rl/scripts/rsl_rl/eval_chain.py` で測る。v4 以降の版の比較は、この評価で行う。

| 項目 | 内容 | 根拠 |
|---|---|---|
| 環境 | Play タスク (`Isaac-Twisted-RL-Knot-Play-v*`)。v4 以降の Play は chain モード (§3) | `versions/v4/env_cfg.py:217-224` |
| 始まり | まっすぐなロープ | 同上 (`straight_prob = 1.0`) |
| 終わり | 固結び・発散・18 手 (3 遷移 x 6 手) のどれか | `versions/v4/env_cfg.py:213-214` |
| 数えるもの | **各環境の最初の 1 エピソードだけ** (早く終わるエピソードを重く数えないため) | `eval_chain.py:7-11` |
| 方策 | 決定論的 (平均の行動)。`--stochastic` を付けたときだけ確率的 | `eval_chain.py:54` |
| 学習の設定との照合 | 評価の前に、チェックポイントの run の `params/env.yaml` と評価の環境の設定を比べ、Play が意図して変える項以外に違いがあれば止める (`--allow_param_diff` で続ける)。Play が変える項は、同じ版の学習用と Play 用のタスクの設定を既定のまま作った差で決める。`seed`・`log_dir`・`io_descriptors_output_dir`・`scene.num_envs`・`sim.device` は比べない | `common/cfg_diff.py`、`eval_chain.py` の `_check_params` |

出力 (JSON と標準出力。`eval_chain.py:13-36`)。上の 6 行は 2026-10-15 より前からあるキーで、意味を変えない (`summarize_search.py` と過去の `eval_chain_it*.json` が前提にしている)。割合の分母は数えたエピソードの数 (`episodes`):

| キー | 意味 |
|---|---|
| `knot` | 固結びに達した割合 (`stage_up`。Play では固結びでだけ立つ) |
| `knot_strict` | 固結びに達し、その形が正規の経路の終点 (厳密な段 3) だった割合 |
| `diverged` / `time_out` / `stage_down` | 発散 / 18 手 / 段の下降で終わった割合。終了項ごとに独立に数える (同じ手に 2 つ立てば両方に入る) |
| `max_stage_ge{k}` | エピソードの中で段 k 以上に達した割合 |
| `mean_actions_to_knot` | 固結びに達したエピソードの手数の平均 |
| `episodes` / `uncounted` | 数えたエピソード / `--steps` の中で最初のエピソードが終わらなかった環境 |
| `knot_ci95` | `knot` の Wilson の 95% 区間。chain 評価は決定論的 (§2) なので、測り直しの揺れではなく、初期状態の揺らぎの組を引き直したときの揺れの二項の見積り |
| `p_ge2_given_ge1` / `p_knot_given_ge2` | 段 1 以上に着いたエピソードのうち段 2 以上に着いた割合 / 段 2 以上に着いたうち固結びの割合 (分母が 0 なら null) |
| `outcome_by_last_stage` | 結末 (`knot`・`diverged`・`stage_down`・`time_out`) x 最後の手を打ったときの段 (0〜3。段 3 は下の注) の割合。和が 1 になるよう、同じ手に 2 つ立てば 固結び > 発散 > 段の下降 > 時間切れ の 1 つに数える。段 0 の時間切れ (v13 s44 型の崩れ) はここで見る |
| `stage_reach.{k}` | 段 k 以上に着いた割合 (`frac`、`max_stage_ge{k}` と同じ) と、着いたエピソードで初めて着くまでの手数の平均 (`mean_moves`)。段 3 には、固結びで終わったエピソード (固結びの手) と、観測で段 3 が見えたエピソード (その前の手) が入るので、`mean_actions_to_knot` とは一致しない |
| `moves_at_stage.{s}` | 段 s で打った手の数の平均 (`mean`、全エピソード)、段 s で 1 手以上打った割合 (`visited_frac`)、そのエピソードでの平均 (`mean_if_visited`) |
| `action_saturation` | 段 x チャネル (`channels`: grasp / delta_link / drop_x / drop_y / drop_z / yaw) ごとに、クリップ前の行動が [−1, 1] の外に出た手の割合 (`sat_frac`)、どれかのチャネルが出た手の割合 (`sat_any`)、BoundedPPO の罰則 Σ max(\|a\| − 1, 0)² の手の平均 (`penalty`)、手の数 (`moves`)。`by_stage` と全段の `all`。各環境の最初のエピソードの手を数える。`--stochastic` ではサンプルした行動の値 |
| `meta` | `argv`・`seed`・`env_seed`・`hydra_overrides`・`num_envs`・`steps`・`device`・`commit`・`git_dirty` / `git_status` (未追跡を含む未コミットの変更)・`started` / `finished` |
| `param_check` | 上の「学習の設定との照合」の結果。`changed` (キー -> [学習, 評価])・`only_in_train`・`only_in_eval`・`allow_param_diff`。照合を飛ばしたときは `skipped` に理由 |

段は手を打つ前の観測の段 (方策が使うヘッドと同じ)。手の後の段は、エピソードが続くときだけ次の手の前の観測として見えるので、固結び以外で最後の手の後に段が上がっても数えない (`max_stage_ge{k}` も同じ)。観測の段は床に落ちているか (`settled`) を見ないが、終了項 `stage_up` は床に落ちて発散していない手でだけ立つ (`common/mdp/stages.py` の `_outcome`)。そのため、固結びの形のまま吊っている手の後は観測の段が 3 になり、エピソードが続く (段 3 で打つ手。`max_stage_ge3` が `knot` より大きいのもこのため)。集計は `common/chain_stats.py` (単体テスト `tests/test_chain_stats.py`)。

### 1.1 標準の条件

過去の比較 (`docs/experiments/2026-10-01_v4-v6_overnight.md` §2) と同じ条件で測る。スクリプトの既定値はこれと違う (`--num_envs` 1024、`--seed` なし: `eval_chain.py:47`、`:52`) ので、必ず指定する。

| 条件 | 値 |
|---|---|
| 環境の数 (= エピソード数) | 512 |
| シード | 7 (値には効かない。§2 の決定論。過去の記録と揃えるために付ける) |
| 方策 | 決定論的 (`--stochastic` を付けない) |
| Hydra の上書き | **学習と同じもの**。学習で物理を上書きした run は、評価にも同じ上書きを付ける。付け忘れると評価の前に止まる (§1 の「学習の設定との照合」。Play も変える項の上書きは見つけられない) |
| デバイス | `--device` と `agent.device` を同じにする |
| チェックポイント | 最後のもの。番号は run のフォルダを `ls` して確かめる (`docs/system/logs.md` §3)。過去の記録の値を再現するときは、最後のものではなく記録の iteration (`eval_chain_it{N}.json` の N) のもの |

```bash
uv run python -u src/isaac_twisted_rl/scripts/rsl_rl/eval_chain.py --task Isaac-Twisted-RL-Knot-Play-v6 \
    --num_envs 512 --headless --device cuda:0 --seed 7 \
    --checkpoint logs/rsl_rl/knot_tying_v6/<run>/model_100.pt agent.device=cuda:0 --out <run のフォルダ>/eval_chain_it100.json
```

- 結果は run のフォルダの `eval_chain_it{N}.json` に置く (過去の結果もこの名前)。
- 2026-10-15 (commit `9aa89f1`) からの `eval_chain_*.json` は、`meta` にコマンド・シード・Hydra の上書き・commit を持つ。それより前の JSON には入っていない (キーは `task`・`checkpoint`・`stochastic` と結果だけ) ので、過去の評価のコマンドは実験記録の側を見る。新しい評価でも、実験記録にコマンドを書く。

## 2. 比べ方

| 規則 | 理由 | 根拠 |
|---|---|---|
| 版どうしは chain 評価の `knot` で比べる | 学習ログの段ごとの成功率と `Curriculum/stage_log/chain_est` は、本番の成功率を表さない。学習は段ごとに最大 6 手・アーカイブから始まるのに対して、本番はまっすぐなロープから 18 手で段をつなぐ | `docs/experiments/2026-10-01_v4-v6_overnight.md` §4.5 |
| シード 2 本以上で比べる (学習のシード) | 512 エピソードの標準誤差は 2 ポイント前後 (二項の見積り。次の行)。シード差のほうがずっと大きい: v6 の seed 42 と 43 は iteration 70 で 19 ポイント、iteration 100 で 15 ポイント離れた。**この推定は v6 の 1 組だけが出典** | `docs/experiments/2026-10-01_v4-v6_overnight.md:283-284` (v6 s43 の値は表 `:264`) |
| chain 評価は決定論的。測り直しや `--seed` の違いを、別の標本として数えない | 同じチェックポイントは、測り直しても、GPU・混み方・`--seed` を変えても全キーがビット単位で同じ (RTX 3090、v13 s44 it70・v12 s43 it300)。512 の結末は、同じ初期状態の丸めの差が物理で増幅されたもので、環境の番号で決まると読める (推測)。標準誤差の 2 ポイントは二項の仮定の見積りで、測り直しでは確かめられない (未実測) | `docs/experiments/2026-10-15_eval-chain-repro.md` |
| `eval_chain.py` を変えたときは、変更の前と同じ条件の JSON と既存のキーが完全に一致することを確かめる | 上の決定論から、一致しなければ変更が値を変えている | 同上 |
| 1 シードどうしの 10〜15 ポイントの差は結論に使わない | 上のシード差に埋もれる | 同 `:283-284` |
| 学習ログや、学習済みの方策での計測 (代理の指標) だけで出した結論は「推測」と書く | 学習ログで出した結論を事実として書き、本番の指標で確かめられず撤回したことがある | `docs/experiments/2026-10-01_v4-v6_overnight.md:275-276` 付近、refactor の調査 D-7 |

## 3. Play タスクの意味 (版で違う)

| 版 | Play タスクの中身 | 根拠 |
|---|---|---|
| v0〜v2-R2 | 推論の様子を GUI で見る環境。環境は 9 個、描画の間隔を短くする | `common/rope_task_cfg.py:145-161` |
| v3 | 上に加えて、常にまっすぐなロープから始め、毎手 p-data を表示する | `versions/v3/env_cfg.py:205-208` |
| v4 以降・rope study | **chain 評価の環境** (§1)。段の重み (1, 0, 0)、まっすぐなロープ、段の上下の終了を chain モードに、18 手 | `versions/v4/env_cfg.py:213-224` |

タスク ID の接尾辞 `-Play` は同じでも、v3 までと v4 以降で評価の定義が違う。版をまたいで Play の数字を比べない。

## 4. 探索ありの成功率

シミュレータで候補の手を試すビーム探索と、探索なしの best-of-N は `scripts/rsl_rl/eval_search.py` で測る (論文と同じ土俵の数字)。
使い方とアルゴリズムは [`scripts/rsl_rl/README.md`](../../src/isaac_twisted_rl/scripts/rsl_rl/README.md)、結果は [`docs/experiments/2026-10-03_inference_search.md`](../experiments/2026-10-03_inference_search.md) (打ち切り中)。
結果は run のフォルダの `eval_search_*_it{N}.json` に置く。探索ありの成功率は chain 評価の `knot` と別に報告する。

## 5. 用語

- 「chain 評価」は `eval_chain.py` の評価だけを指す。`chain_est` は学習ログの見積りで、chain 評価ではない。
- 「発散」の判定は 3 通りある (終了項 `diverged`、NaN、`divergence_bench` の spike)。chain 評価の `diverged` は終了項 `diverged`。定義は [`docs/glossary.md`](../glossary.md)。
