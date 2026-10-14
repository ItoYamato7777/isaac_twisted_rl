# 0026 v13 は PPO の損失に「方策の平均が行動の範囲の外へ出た分」の罰則を足す

- 状態: 有効 (v13 以降。v13 は未実装)
- 読むべき状況: v13 を実装するとき。方策の平均が行動の範囲の外へ出る問題・学習の途中で chain 評価が下がる問題を扱うとき。PPO の損失や行動の分布を変えたくなったとき
- 最終更新: 2026-10-14

## 決定

v13 = v12 + **PPO の損失に、段ごとの方策の平均が行動の範囲 [−1, 1] の外へ出た分の罰則を足す** (1 版 1 変数。ユーザーの指示、2026-10-14)。

```
loss = surrogate + value_loss_coef · value − entropy_coef · entropy + bounds_loss_coef · mean_batch( Σ_チャネル max(|μ| − 1, 0)² )
```

- μ は actor の出力 (クリップ前の平均)。範囲の中 (|μ| ≤ 1) では罰則は 0 で、学習を変えない。
- 境目は 1.0 (環境が `PickPlaceAction.process_actions` でクリップする値と同じ)。
- `bounds_loss_coef` は **0.01 を推奨** (下の「係数」)。学習の前にユーザーに確かめる。
- 環境の設定 (観測・報酬・終了・アーカイブの保存の規則 `stage_up`・物理) と方策の構造は v12 のまま。変わるのは PPO の損失だけ。
- 学習: seed 43 / 44 を 0 から iteration 300 まで (v12 と同じ長さ)。比べる相手は v12 (同じ seed)。

## 理由

- **v7・v11・v12 のどれも、chain 評価の山が it70〜100 にあり、その後に下がる。** v12 でアーカイブのループを断っても下がった (v12 s44: it70 の 58.8% -> it150 の 2.0%)。版の間に共通の原因がある ([v12 の記録](../experiments/2026-10-13_v12_archive-stage-up.md))。
- **全 run で、方策の平均が範囲の外へ出る割合が学習とともに増え続けた** ([調査の記録](../experiments/2026-10-14_action-saturation.md))。段 0 で it300 に 37〜73%、平均は最大で |μ| 14。chain 評価が大きく崩れた 3 つ (v12 s44・v11 s44・v11 s43 の it1000) では、段 0 の置き方のチャネル (dx・release_z・yaw) まで 50〜97% が範囲の外だった。崩れない v12 s43 の it300 では dx 12%・release_z 2%。
- **範囲の外では、その次元の学習の信号が無くなる。** 環境はクリップし、PPO の log prob はクリップ前の行動で計算する。平均が範囲の外にあると標本の多くが同じ端の行動になり、その次元の違いが報酬に現れないので、平均を範囲の中へ戻す勾配が無い。平均は流れていき、段の間の分布が変わったとき (v12 s44 では段 0 の打ち方が変わり、段 1 に入る形が変わった) に方策が付いていけなくなる (推測)。
- **罰則なら、端が本当に良い手のチャネルを壊さない。** grasp と Δlink は崩れない run でも 76〜92% が範囲の外で、「ロープの端を掴み、いちばん遠くへ置く」が良い手になっていると読める。罰則は平均を 1 のあたりに留めるだけで、クリップ後の行動 (決定論的な評価の行動) は変わらない。標本の半分ほどは範囲の中に残るので、学習の信号も残る。
- 罰則は rl_games の `bounds_loss` (IsaacGymEnvs の連続行動のタスクで使う) と同じ考え方 (向こうの境目と係数の既定値は未確認)。

### 係数

- 範囲の外への出方 Σ max(|μ| − 1, 0)² の観測の平均は、v12 s44 で it70 に 0.39、it300 に 5.3 (段 0 は 14.6)。同じ時期の `Loss/surrogate` は −0.003〜−0.012。
- 0.01 なら、罰則の項は it70 の出方で約 0.004 となり、surrogate と同じ桁になる。範囲の外では surrogate の勾配がほぼ 0 なので、同じ桁の罰則で平均は範囲の近くへ戻る見込み (推測)。範囲の中では 0 なので、大きすぎて学習を壊す心配は小さい。
- 値を振る比較は、1 版 1 変数の原則で v13 では行わない。効き方が足りなければ (下の予想 1 が外れたら)、係数を変えた次の版を作る。

### 見込みの弱点

- 張り付きは崩れの**相関**で、因果は確かめていない。時期もきれいに重ならない (v12 s44 の it70 は dx が 43% 範囲の外で、chain 評価は最高の 58.8%)。v13 で張り付きを抑えても崩れたら、この仮説は誤り。
- std が小さくなりすぎること (段 1 の最小の std がどの run も 0.03〜0.06) には効かない。崩れた run と崩れない run で差が無いので、v13 には入れない。
- s44 の価値関数の損失の増加 (0.35 -> 1.73) が原因側なら、v13 では直らない。

## 外れたら仮説が誤りと言える予想

1. 罰則が効く: v13 の段 0 の置き方のチャネル (dx・release_z・yaw) で |μ| > 1 の割合が、iteration 300 まで 20% を超えない。Σ max(|μ| − 1, 0)² の観測の平均が 0.05 を超えない (`action_saturation.py`。観測は v12 の 4 ファイルをそのまま使う)。
2. chain 評価が it70〜100 の後に下がらない: 2 シードとも it300 が、it70・it100 の高い方から 5 ポイント以上は下がらない。
3. s44 が段 0 で崩れない: 学習ログの `timeout_s0` が 0.1 を超えない。

1 が当たって 2・3 が外れたら、張り付きは崩れの原因ではない。1 が外れたら、罰則が足りないので、係数を見直す (この版の結論は保留)。

## 検討して採らなかった案

| 案 | 採らなかった理由 |
|---|---|
| 平均を tanh で [−1, 1] に押し込む | 端が良い手のチャネル (grasp) では平均が tanh の飽和域に入り、勾配が消えるので、同じ型の詰まりが残りうる |
| std の下限 | 範囲の外への張り付きを止めない。最小の std は崩れた run と崩れない run で差が無い |
| Beta 分布の方策 | 方策の構造と出力の意味が変わり、既存の版のチェックポイントと比べにくい。rsl-rl の PPO に無く、実装が大きい |
| クリップ後の行動で log prob を計算する | rsl-rl の rollout の保存と PPO の更新の両方を変える必要がある。端の確率質量の扱いが要る |
| 早めに止める (it70〜100 のチェックポイントを使う) | 原因を直さない。山の位置はシードで違う (v11 s44 は it150 が最高) |

## 実装の見通し (推奨。2026-10-14 の rsl-rl 3.0.1 のコードで確かめた)

- **rsl-rl の PPO の損失は `PPO.update` の中で 1 行で組む** (`.venv/lib/python3.11/site-packages/rsl_rl/algorithms/ppo.py:315`)。足す口が無い。`update` を写して 1 行足すと約 250 行の写しになるので、次の形を推奨する:
  - `kt/common/policies/bounded_ppo.py` (isaaclab を import しない。`policies/stage_gated.py` と同じ置き場所で、`tests/` から単体テストできる) に `BoundedPPO(PPO)` を置く。引数 `bounds_loss_coef: float = 0.0`。
  - `BoundedPPO.update()` は、`self.policy.update_distribution` を一時的に包んでから `super().update()` を呼ぶ。包んだ関数は、勾配を取る呼び出し (`torch.is_grad_enabled()`。rollout の `act` は inference mode なので掛からない) のとき、分布の平均を「順方向は恒等で、逆方向で `bounds_loss_coef · ∂penalty/∂μ` を足す」autograd の関数に通して `Normal` を作り直す。`update` の中の `self.policy.act(obs_batch)` は mini-batch ごとに 1 回なので、これは損失に罰則の項を足したのと同じ勾配になる。罰則の値は貯めておき、`update` の返す `loss_dict` に `"bounds"` を足す (`Loss/bounds` として TensorBoard に出る。`on_policy_runner.py:212`)。
  - mini-batch の平均の取り方 (batch の平均、チャネルの和) を surrogate・entropy と揃える。
  - `on_policy_runner` は `eval(alg_cfg["class_name"])` で PPO を作る (`on_policy_runner.py:424`)。`register_with_rsl_rl` (`policies/stage_gated.py`) と同じく、`on_policy_runner.BoundedPPO = BoundedPPO` で登録する。
- **単体テスト** (`tests/test_bounded_ppo.py`): (1) 包んだ平均の勾配が、`loss + coef · penalty` を直接書いたときの勾配と一致する。(2) |μ| ≤ 1 なら勾配も値も変わらない。(3) `bounds_loss_coef = 0` なら PPO と同じ。(4) inference mode では何もしない。
- **版**: `versions/v13/env_cfg.py` は v12 のクラスを継承して本体なし (Play も v12 の `play_post_init`)。`versions/v13/agents/rsl_rl_ppo_cfg.py` は v12 の Runner を継承し、`algorithm` を `RslRlPpoAlgorithmCfg` の派生 (`class_name = "BoundedPPO"`、フィールド `bounds_loss_coef = 0.01`。configclass は宣言したフィールドしか `to_dict` に出さない) にする。それ以外の PPO の値は v12 (= v4) と同じであることを `tests/test_v13_cfg.py` で検査する (`tests/test_v12_cfg.py` が手本)。
- **スモークテスト**: `params/agent.yaml` に `class_name: BoundedPPO`・`bounds_loss_coef: 0.01`、TensorBoard に `Loss/bounds` が出る、学習ログの他の値が v12 の最初の iteration と同じ桁。
- **評価**: chain 評価 (seed 7、512 環境、Hydra の上書きなし) を it70 / 100 / 150 / 200 / 250 / 300 で、v12 と並べる。`action_saturation.py --run v13_sS=<run>` で予想 1 を数える。

## 出典

- 調査の記録: [2026-10-14_action-saturation](../experiments/2026-10-14_action-saturation.md)、[2026-10-13_v12_archive-stage-up](../experiments/2026-10-13_v12_archive-stage-up.md)。
- 方針はユーザーとの検討 (2026-10-14) で決めた (案 A、v12 を継承)。
