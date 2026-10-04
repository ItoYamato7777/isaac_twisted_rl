# 評価: chain 評価の定義と、版どうしの比べ方

- 状態: 現行 (v4 以降の評価の正本)
- 読むべき状況: 版や設定の良し悪しを比べるとき、評価のコマンドを組むとき、評価の数字を読むとき
- 最終更新: 2026-10-03

手順 (学習から評価・集計まで) は skill `train-eval` にある。この文書は「何を測るか」と「どう比べるか」だけを書く。

## 1. chain 評価

`src/isaac_twisted_rl/scripts/rsl_rl/eval_chain.py` で測る。v4 以降の版の比較は、この評価で行う。

| 項目 | 内容 | 根拠 |
|---|---|---|
| 環境 | Play タスク (`Isaac-Twisted-RL-Knot-Play-v*`)。v4 以降の Play は chain モード (§3) | `versions/v4/env_cfg.py:217-224` |
| 始まり | まっすぐなロープ | 同上 (`straight_prob = 1.0`) |
| 終わり | 固結び・発散・18 手 (3 遷移 x 6 手) のどれか | `versions/v4/env_cfg.py:213-214` |
| 数えるもの | **各環境の最初の 1 エピソードだけ** (早く終わるエピソードを重く数えないため) | `eval_chain.py:6-10` |
| 方策 | 決定論的 (平均の行動)。`--stochastic` を付けたときだけ確率的 | `eval_chain.py:36` |

出力 (JSON と標準出力。`eval_chain.py:13-17`):

| キー | 意味 |
|---|---|
| `knot` | 固結びに達した割合 (`stage_up`。Play では固結びでだけ立つ) |
| `diverged` / `time_out` | 発散 / 18 手で終わった割合 |
| `max_stage_ge{k}` | エピソードの中で段 k 以上に達した割合 |
| `mean_actions_to_knot` | 固結びに達したエピソードの手数の平均 |

### 1.1 標準の条件

過去の比較 (`docs/experiments/2026-10-01_v4-v6_overnight.md` §2) と同じ条件で測る。スクリプトの既定値はこれと違う (`--num_envs` 1024、`--seed` なし: `eval_chain.py:30`、`:35`) ので、必ず指定する。

| 条件 | 値 |
|---|---|
| 環境の数 (= エピソード数) | 512 |
| シード | 7 |
| 方策 | 決定論的 (`--stochastic` を付けない) |
| Hydra の上書き | **学習と同じもの**。学習で物理を上書きした run は、評価にも同じ上書きを付ける |
| デバイス | `--device` と `agent.device` を同じにする |
| チェックポイント | 最後のもの。番号は run のフォルダを `ls` して確かめる (`docs/system/logs.md` §3)。過去の記録の値を再現するときは、最後のものではなく記録の iteration (`eval_chain_it{N}.json` の N) のもの |

```bash
uv run python -u src/isaac_twisted_rl/scripts/rsl_rl/eval_chain.py --task Isaac-Twisted-RL-Knot-Play-v6 \
    --num_envs 512 --headless --device cuda:0 --seed 7 \
    --checkpoint logs/rsl_rl/knot_tying_v6/<run>/model_100.pt agent.device=cuda:0 --out <run のフォルダ>/eval_chain_it100.json
```

- 結果は run のフォルダの `eval_chain_it{N}.json` に置く (過去の結果もこの名前)。
- `eval_chain_*.json` には、シードと Hydra の上書きが入らない (キーは `task`・`checkpoint`・`stochastic` と結果だけ)。そのため、評価のコマンドは実験記録の側に必ず書く。

## 2. 比べ方

| 規則 | 理由 | 根拠 |
|---|---|---|
| 版どうしは chain 評価の `knot` で比べる | 学習ログの段ごとの成功率と `Curriculum/stage_log/chain_est` は、本番の成功率を表さない。学習は段ごとに最大 6 手・アーカイブから始まるのに対して、本番はまっすぐなロープから 18 手で段をつなぐ | `docs/experiments/2026-10-01_v4-v6_overnight.md` §4.5 |
| シード 2 本以上で比べる | 512 エピソードの標準誤差は 2 ポイント前後。シード差のほうがずっと大きい: v6 の seed 42 と 43 は iteration 70 で 19 ポイント、iteration 100 で 15 ポイント離れた。**この推定は v6 の 1 組だけが出典** | `docs/experiments/2026-10-01_v4-v6_overnight.md:283-284` (v6 s43 の値は表 `:264`) |
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
