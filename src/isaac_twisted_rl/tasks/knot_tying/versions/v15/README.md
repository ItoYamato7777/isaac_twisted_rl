# v15: v14 + 段の変化・発散と同じ手では時間切れを立てない

```bash
uv run train --task Isaac-Twisted-RL-Knot-v15 --headless           # v14 + 時間切れの bootstrap の不具合の修正
uv run play  --task Isaac-Twisted-RL-Knot-Play-v15 --num_envs 16   # v14 と同じ chain 評価 (時間切れは mdp.time_out)
```

ログ: `logs/rsl_rl/knot_tying_v15/`。**学習には `Knot` プリセットを使う** (v3〜v14 と同じ)。継承元は **v14**。
仕様と理由の正本は [`docs/decisions/0028-v15-timeout-bootstrap-fix.md`](../../../../../../docs/decisions/0028-v15-timeout-bootstrap-fix.md)。ここには配線と、見るべき指標を書く。
初期アーカイブは v14 と同じ。

## なぜ作ったか

Isaac Lab の `TerminationManager` は `time_out=True` の項を `truncated`、それ以外を `terminated` に別々に立てるので、6 手目に段が上がる・下がる・発散すると両方が立つ。rsl-rl 3.0.1 の PPO は `truncated` の手の報酬に `terminated` を見ずに γ·V(s) を足すので、6 手目に段を上げた手の目標は 1 + γ·V(s) になる。6 手目にだけ訪れる状態の V は 1 を超えて膨らみうる (不動点の上限 1/(1 − γ) = 20) し、6 手目まで待って上げる得が生まれる。崩れた run (v12 s44 it150) の方策は、まっすぐなロープから 512 環境中 490 で、ちょうど 6 手目に段 0 を上げていた。

rsl-rl に渡るのは `dones` と `time_outs` だけで、両方が立った手の `terminated` を PPO の側から取り戻せないので、終了項の側で直す。

## v14 から何を・なぜ変えたか

| 項目 | v14 | v15 |
|---|---|---|
| 終了項 `time_out` の関数 | `mdp.time_out` (6 手目で常に立つ) | **`mdp.time_out_unless_terminated`**: 6 手目でも、同じ手で `diverged`・`stage_up`・`stage_down` のどれかが立てば立たない |
| Play | v14 | v14 と同じ (`time_out` を `mdp.time_out` に戻す。chain 評価の定義を v14 と揃える) |
| それ以外 (観測・報酬・行動・アーカイブの保存の規則・物理の DR・critic の特権観測・PPO・方策の構造) | | v14 のまま |

- 6 手目に段が変わる・発散する手は `terminated` だけになり、PPO はその手を bootstrap しない。目標は 1〜5 手目に上げた手と同じ +1 (下がれば −1) になる。
- 段も変わらず発散もしない 6 手目は、今までどおり `truncated` で γ·V(s) を bootstrap する (観測に手の番号が無いので、時間切れは打ち切りとして扱う)。
- 項の名前は `time_out` のまま (`ResetToStage` の `outcome_terms`・`Episode_Termination/time_out` が名前で引く)。終了項の並びも v14 と同じ。6 手目に段が上がった手は、v14 でも `stage_up` が後ろにあるのでログでは成功に数えていた。ログの数え方は変わらない。
- 終了項の関数は、`exclude_terms` の各項の関数を params ごと呼び直す (`get_term()` は前のエピソードの値が残るので使わない)。`time_out` は並びの先頭だが、`common_step_counter` が終了判定の前に進むので、位相の抽出は今の手の後の形で行われる。

## 見るべき指標

- `Loss/value_mean_s0` (段 0 の V の平均。`BoundedPPO` が出す): 1/γ = 1.053 を超えないはず (0028 の予想 1)。
- `Curriculum/stage_log/timeout_s0`: 段 0 の崩れの印。
- 学習中の `terminated & truncated` は 0 になる (ゲートで確かめる)。
