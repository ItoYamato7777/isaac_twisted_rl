# v15 のゲート: 時間切れの項の差し替えで terminated と truncated の重なりが消える (2026-10-16)

- 状態: 凍結
- 読むべき状況: v15 の学習・評価の結果を読むとき。`time_out_unless_terminated` が意図どおり動いたかを知りたいとき

## 要約

- 結論: v12 s44 の `model_150.pt` の方策を学習用のタスクで 256 環境 × 24 手走らせると、v14 では同じ手に `terminated` と `truncated` が両方立つ手が 124 回あり (うち `stage_up` 119・`stage_down` 3・`diverged` 2)、v15 では 0 回だった。純粋な時間切れ (`truncated` だけ) は 396 回で変わらず、`terminated` も 1808 回で一致した。
- 有効か: 方策 1 つ・シード 1 つ・256 環境 × 24 手の計数。不具合の消滅の確認であり、学習の効果は何も示さない。

## 目的と仮説

[0028](../decisions/0028-v15-timeout-bootstrap-fix.md) の実装の見通し 5。仮説: 継承元 (v14) では 6 手目に段が変わる・発散する手で両方が立ち、v15 では立たない。

## 条件

| 項目 | 値 |
|---|---|
| commit | `b92f936` (検査のスクリプトを足した commit。`62331ac` まで差分は同じ)。未コミットの差分: なし |
| タスク ID とプリセット | `Isaac-Twisted-RL-Knot-v14` / `Isaac-Twisted-RL-Knot-v15` (学習用、`Knot`) |
| 方策 | `logs/rsl_rl/knot_tying_v12/2026-10-13_15-17-31_v12_s44/model_150.pt` (6 手目に段 0 を上げる方策。critic は 197 次元なので `agent.obs_groups.critic=[policy]`)。行動は方策の std で引く (`--deterministic` なし) |
| シード・num_envs・手数 | seed 7・256 環境・24 手 (6 手のエピソード 4 回ぶん) |
| コマンド | `uv run python -u src/isaac_twisted_rl/scripts/rsl_rl/check_timeout_overlap.py --task Isaac-Twisted-RL-Knot-v{14,15} --num_envs 256 --headless --device cuda:0 --seed 7 --checkpoint <上の model_150.pt> agent.device=cuda:0 'agent.obs_groups.critic=[policy]' --out outputs/v15_gate/overlap_v{14,15}.json` |
| job | `outputs/jobs/2026-10-16_16-35-27_v15_gate_overlap` (exit 0、`signal:` の行なし) |
| 出力 | `outputs/v15_gate/overlap_v14.json`・`overlap_v15.json` |

## ゲート

数えた値 (`env.step` の後の `termination_manager` の `terminated` / `time_outs`。手の数 = 環境 × 手):

| | `time_out` の関数 | terminated | truncated | 両方 | truncated だけ | 両方の内訳 (diverged / stage_up / stage_down) |
|---|---|---|---|---|---|---|
| v14 | `time_out` | 1808 | 520 | **124** | 396 | 2 / 119 / 3 |
| v15 | `time_out_unless_terminated` | 1808 | 396 | **0** | 396 | 0 / 0 / 0 |

- v15 の `truncated` は v14 の `truncated だけ` と一致し、`terminated` は両版で 1808。差し替えは「両方立った 124 手から `truncated` を外す」だけに効いている。
- 同じ seed・同じ方策だが乱数の流れが同じとは限らない (計数が一致したのは観測事実)。

## 位相の抽出の順序 (実装の前に読んで確かめた)

`time_out` は終了項の並びの先頭で `stage_up` より先に呼ばれるが、`ManagerBasedRLEnv.step()` は `common_step_counter` を進めてから終了判定に入り、`get_extractor` はその見張り番で抽出し直す。`time_out_unless_terminated` が `stage_changed` を呼んで見る形は今の手の後の形で、並べ替えは要らなかった。上の計数で v15 の `both_by_term` が 0 なのは、この読みと整合する (別の順序なら `stage_up` が立つ手で除外が漏れる)。

## 結果

学習の前に確かめられたのは上の重なりの計数と、学習の run の `params/env.yaml` の `terminations.time_out.func` が `time_out_unless_terminated` になっていること (s43・s44 とも)、s43 の TensorBoard に `Loss/value_mean_s0` が出ること。

**2 iteration のスモークテストは別には回していない。** 学習の run を本番として起動し、最初の iteration で `params/env.yaml` とログを確かめた (問題があれば止めてやり直す前提)。Play の `time_out` が `mdp.time_out` に戻ることは `tests/test_v15_cfg.py` の静的な検査だけで、組み立てた Play の設定は最初の chain 評価の `params` で確かめる。

## v14 の段 0 の V (0028 の予想 1 の比べる相手)

固定した観測 (`outputs/diag_stage_moves/obs/` の 4 ファイル、v12 の方策の観測 25040 行) を v14 の critic に入れて数えた。v14 の critic は特権観測 6 次元を前に取るので、公称の物理の値 `0,0,-1,0,0,0` を全行に足した (`critic_stage_values.py --critic_prefix`)。出力 `outputs/critic_stage_values/v14.json`、閾値 1/γ = 1.053。

| run | 段 0 の V の平均の最大 (it10〜300) | 1/γ を超えた割合の最大 | it300 の平均 |
|---|---|---|---|
| v14 s43 | 1.077 (it100) | 0.431 (it100) | 0.972 |
| v14 s44 | 0.991 | 0.143 (it300) | 0.984 |
| (比較) v13 s44 (崩れた run) | 9.423 (it300) | 0.983 | 9.423 |

- 分かった: v14 でも V が 1/γ をやや超える点がある (s43 の it30〜300 で 5〜43%) が、崩れた run のような膨らみ (v13 s44 で 3〜9) は無い。
- 注意: 観測は v12 の方策が訪れた状態で、v14 の方策が訪れた状態ではない。物理は公称の値に固定した (DR で引いた物理での V は数えていない)。よって「v14 の学習中の V」ではなく「固定した観測での V」。v15 の学習ログの `Loss/value_mean_s0` (自分の方策の状態での V) とは直接比べられず、予想 1 を区別する材料は弱い。v15 の学習の後に、v15 のチェックポイントも同じ観測・同じ数え方で数えて比べる。

## 結論

- 分かった: 重なりは v14 に実在し (124 / 6144 手 ≈ 2%)、v15 では 0 になった。純粋な時間切れの bootstrap は残っている。
- 推測: なし。
- 未確認: Play の組み立てた設定 (`time_out` が `mdp.time_out`)。学習が V の膨らみ・段 0 の崩れに効くか。

## 次の作業

seed 43 / 44 を it300 まで学習 (job `outputs/jobs/2026-10-16_17-07-47_v15_s43`・`..._s44`、commit `62331ac`) -> chain 評価 -> v14 と並べる。
