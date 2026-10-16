# 0028 v15 は、段の変化・発散と同じ手では時間切れを立てない (時間切れの bootstrap の不具合を直す)

- 状態: 有効 (v15 以降。2026-10-15 にユーザーが v16 で直すと決め、2026-10-16 に **v15 へ繰り上げた** (寸法 DR より先に直す)。**継承元は v14** (ユーザー、2026-10-16)。未実装)
- 読むべき状況: v15 を実装・学習・評価するとき。時間切れ (`time_out`) の終了項や PPO の bootstrap を扱うとき。段 0 の崩れ・critic の V の膨らみを扱うとき
- 最終更新: 2026-10-16

## 決定

v15 = 継承元 + **時間切れの終了項を「同じ手で段の上昇・段の下降・発散が立たないときだけ立つ」項に差し替える** (1 版 1 変数)。

- **継承元は v14** (ユーザー、2026-10-16。v14 の chain 評価が 2 シードとも崩れず、公称の物理で v13 を大きく上回ったため。[v14 の記録](../experiments/2026-10-15_v14_s0-reshape-dr-priv-critic.md))。v14 の切り分けの run (3 つの変更のどれが効いたか) は回さない (ユーザー、同日)。
  - 比べる相手は v14 の同じシード (43 / 44)。この修正だけの差で比べるため、v14 には他の変更を足さない (段 0 の保存の規則・ロープの物理の DR・critic の特権観測は v14 のまま)。
  - 2026-10-16 に v14 のコードで確かめた: 終了項は v4 の `TerminationsCfg` のまま (`time_out`・`diverged`・`stage_up`・`stage_down`)。PPO は v13 の `BoundedPPO` (係数 0.01) のままで、`Loss/value_mean_s*`・`Loss/value_over_s*` は commit `5484ad9` (v14 の学習の後) から出る。v15 の学習ログには出るが、**v14 の run には無い**。段は actor の観測の最後の 4 列から読むので、v14 の critic の観測 (203 次元) でも数えられる (`bounded_ppo.py` の `_stage_value_stats`)。
- 終了項の名前は `time_out` のまま (`ResetToStage` の `outcome_terms`・`Episode_Termination/time_out` のログが名前で引く)。`time_out=True` のまま。
- 6 手目の手で段の上昇・段の下降・発散のどれかが立てば `terminated` だけが立ち、PPO はその手を bootstrap しない。どれも立たない 6 手目は、今と同じく `truncated` で、γ·V(s) を bootstrap する。
- **Play (chain 評価) は継承元と同じ時間切れに戻す**。chain 評価の定義 (18 手目に固結びになった手は `knot` と `time_out` の両方に数える) を継承元と揃え、評価の値を版の間で比べられるようにする。
- PPO・観測・報酬・アーカイブの保存の規則・物理は継承元のまま。
- 学習: seed 43 / 44 を 0 から iteration 300 まで (継承元と同じ長さ・同じコマンドの形)。

## 理由

詳細と数値は [2026-10-15_timeout-bootstrap-and-stage2-bottleneck](../experiments/2026-10-15_timeout-bootstrap-and-stage2-bottleneck.md)。

- **不具合**: Isaac Lab の `TerminationManager.compute` は `time_out=True` の項を `truncated`、それ以外を `terminated` に別々に OR する。6 手目に段が上がると両方が立つ。rsl-rl 3.0.1 の `PPO.process_env_step` は、`extras["time_outs"]` (= `truncated`) の手の報酬に、`terminated` を見ずに γ·V(s) を足す (`.venv/lib/python3.11/site-packages/rsl_rl/algorithms/ppo.py:161-164`)。そのため 6 手目に段を上げた手の目標は 1 + γ·V(s) になる。6 手目にだけ訪れる状態では V の不動点が p/(1 − γ) (上限 20) になり、段 0 の V が 1 を超えて膨らみうる。v4 の `TerminationsCfg` の docstring はこの重なりを知っていて、ログの数え方だけを手当てしている。
- **崩れた方策はこの不具合を突いていた**: v12 s44 it150 の方策は、まっすぐなロープから 512 環境中 490 で、ちょうど 6 手目に段 0 を上げた (chain モードに 6 手の時間切れは無い)。報酬と終了の中で 6 手目に上げる手を得にするのは、この不具合だけ。崩れた 4 本の run の段 0 の V (2.3〜15.3) は上限 20 の範囲に入る。
- **環境の側で直す**: rsl-rl に渡るのは `dones = terminated | truncated` と `time_outs = truncated` だけで、両方が立った手から `terminated` を取り戻せない。PPO の側では直せない。終了項の側で「同じ手に段の変化・発散が立つなら時間切れを立てない」とすれば、PPO・ラッパーを変えずに済み、版の設定だけで閉じる (`docs/decisions/0012-version-is-config.md`)。
- **6 手目まで待つ得が消える**: 修正の後、6 手目に段を上げた手の目標は 1 (終端)。1〜5 手目に上げるのと同じなので、待つ得は無い。純粋な時間切れ (段も変わらず発散もしない 6 手目) の bootstrap は残すので、時間切れの意味 (観測に手の番号が無い打ち切り) は今と同じ。

## 外れたら仮説が誤りと言える予想

v15 の 2 シードについて:

1. **V が膨らまない**: 学習ログの `Loss/value_mean_s0` (`BoundedPPO` が出す段 0 の V の平均。10 iteration の平均) が it300 まで 1/γ = 1.053 を超えない。固定した観測での数え方 (`critic_stage_values.py`、[critic-inflation の記録](../experiments/2026-10-15_critic-inflation.md)) でも、段 0 の V の平均が全チェックポイントで 1.053 を超えない。
2. **6 手目に段 0 を上げる山が出ない**: s44 の it150・it300 で `diag_stage_moves.py --dump_obs` を回し、`chain_moves.py sequences` で、段 0 を初めて上げた手が 6 手目の環境が 5% を超えない (v12 s44 it150 は 490 / 512)。
3. **段 0 が崩れない**: 学習ログの `timeout_s0` が 2 シードとも it300 まで 0.1 を超えない。
4. **chain 評価が段 0 から崩れない**: 2 シードとも it150〜300 の `knot` が 20% を下回らない (v12 s44・v13 s44 は it150 以降 2〜22%)。頭打ち (30〜50%) が上がることは予想しない (主因は段 2 の詰まりと見ている)。

継承元を v14 にしたので (v14 は 2 シードとも崩れなかった)、3・4 は当たっても仮説を支持しない (外れれば修正が悪さをした印)。v14 の V が膨らんでいたかは学習ログに無いので、v14 のチェックポイントを固定した観測で数えて (`critic_stage_values.py`) 1 の比べる相手にする。v14 でも膨らんでいなければ、1 も仮説を区別しない。そのときは、v15 の chain 評価が 2 シードの平均で v14 から 5 ポイント以上下がらないこと (修正が害にならない) を主に見る。v14 が崩れなかったのは、段 0 の保存の規則が不具合の「突き所」(6 手目にしか訪れない状態) を潰したためという推測がある ([v14 の記録の考察](../experiments/2026-10-15_v14_s0-reshape-dr-priv-critic.md))。そうなら v15 は v14 からあまり伸びない。v15 がさらに伸びれば、v14 でも不具合が性能を削っていた。

読み方: 1 が当たって 3 が外れたら、V の膨らみ (この不具合) は崩れの原因ではない。1 が外れたら、V を膨らませる別の経路がある (関数近似の外挿など)。1〜3 が当たれば、崩れの原因はこの不具合だった (介入での確認)。

## 検討して採らなかった案

| 案 | 採らなかった理由 |
|---|---|
| 時間切れの bootstrap の V を 1 で頭打ち (critic-inflation の記録の案 B) | 6 手目に上げた手の目標が 1 + 0.95·1 = 1.95 のままで、待つ得が残る |
| critic の出力を [−1, 1] に制限 (同じく案 C) | 同上。V の上限を付けても、+1 の手に γ·V が足される不具合は残る |
| 時間切れを終端にする (同じく案 A。`time_out=False`) | 待つ得は消えるが、純粋な時間切れの bootstrap もやめる。観測に手の番号が無いので、6 手目の時間切れの 0 が 1〜5 手目の状態の V に混ざり、時間切れの意味が変わる。不具合を直すより大きな変更 |
| PPO の `process_env_step` で直す (`BoundedPPO` を上書き) | rsl-rl に渡るのは `dones` と `time_outs` だけで、両方が立った手の `terminated` を取り戻せない。ラッパーも変える必要がある |
| `RslRlVecEnvWrapper` を派生して `time_outs = truncated & ~terminated` を渡す | 動くが、`train.py`・`play.py`・評価のスクリプトのラッパーの作り方を版ごとに変える仕組みが要る。終了項の差し替えなら版の設定だけで済む |

## 実装の見通し (推奨。2026-10-15 のコードで確かめた)

(`kt/` = `src/isaac_twisted_rl/tasks/knot_tying/`。手順は skill `add-version`)

継承元は v14。下の手順は v14 のコード (2026-10-16) に合わせてある。

1. **終了項を `common/mdp/terminations.py` に追加する** (既存の項は変えない。`common/mdp/` は追加だけ)。

   ```python
   def time_out_unless_terminated(
       env: ManagerBasedRLEnv, exclude_terms: tuple[str, ...] = ("diverged", "stage_up", "stage_down")
   ) -> torch.Tensor:
       """時間切れ `(num_envs,)`。ただし同じ手で `exclude_terms` のどれかが立つなら False。v15 で追加。
       (docstring に「なぜ」を書く: 不具合の仕組み・PPO の側で直せない理由・0028 へのリンク)"""
       out = isaaclab.envs.mdp.time_out(env)          # episode_length_buf >= max_episode_length
       for name in exclude_terms:
           cfg = env.termination_manager.get_term_cfg(name)
           out = out & ~cfg.func(env, **cfg.params)
       return out
   ```

   - `get_term()` は使わない (`_term_dones` は前のエピソードの値が残る。`diverged_now` の docstring と同じ理由)。終了項の関数を params ごと呼び直す (`diverged_now` と同じ方法。発散は `diverged_now(env, "diverged")` でキャッシュを使ってもよい)。
   - `time_out` は終了項の並びの先頭なので、`stage_up` などより先に呼ばれる。`stage_changed` は `stage_transition` を env ステップ (`common_step_counter`) と抽出結果 (`extractor.batch`) でキャッシュするので、先に呼んでも同じ値になるはず。**実装のときに確かめる**: 抽出結果がこの時点で今の手の後の形に更新されているか (`get_extractor` の更新の仕組みを読む)。更新が遅れるなら、`time_out` の項を `stage_down` の後ろへ並べ替える (v15 の `TerminationsCfg` で項を定義し直す) か、抽出を先に起こす。
   - `exclude_terms` に無い名前があれば (`get_term_cfg` が `ValueError`)、そのまま落とす (配線の誤りを黙って通さない)。
2. **`versions/v15/`** を作る (`__init__.py`・`agents/__init__.py` は docstring だけ)。
   - `env_cfg.py`: `KnotTyingV15EnvCfg(KnotTyingV14EnvCfg)` の `__post_init__` で、`super().__post_init__()` の後に `self.terminations.time_out = DoneTerm(func=mdp.time_out_unless_terminated, time_out=True, params={"exclude_terms": ("diverged", "stage_up", "stage_down")})` だけを書く。既存の属性への代入なので、終了項の並びは変わらない (`params/env.yaml` で確かめる)。
   - `play_post_init(cfg)`: v14 の `play_post_init(cfg)` (v13 の Play の後に DR を切る) を呼んだ後に、`cfg.terminations.time_out = DoneTerm(func=mdp.time_out, time_out=True)` に戻す (Play の chain 評価を v14 と同じ定義にする)。`make_preset_cfgs(KnotTyingV15EnvCfg, __name__, play_post_init=play_post_init)`。
   - `agents/rsl_rl_ppo_cfg.py`: v14 の Runner を継承し、`experiment_name = "knot_tying_v15"` だけを書く (critic の特権観測 `obs_groups`・`BoundedPPO` と係数 0.01 は v14 のまま。`Loss/value_mean_s*`・`Loss/value_over_s*` が学習ログに出る)。
   - `README.md`: この文書の「決定」と「理由」の要約 (結果は書かない)。
3. **登録と一覧**: `kt/__init__.py` の `_VERSIONS` に `"v15"`、`kt/versions/__init__.py` の docstring の一覧と `kt/README.md` の版の系譜に 1 行ずつ。番号は v15 (v14 の次。寸法 DR は v16 に回した)。版の一覧の検査 (`tests/test_invariants.py` など) が番号の連続を前提にしていないかを確かめる。
4. **テスト**: `tests/test_v15_cfg.py` (手本は `tests/test_v14_cfg.py`・`tests/test_v13_cfg.py`。`ast` で静的に): env の設定は v14 のクラスだけを継承し、`__post_init__` は `terminations.time_out` の差し替えだけ。Play は v14 の `play_post_init` を呼んで `time_out` を `mdp.time_out` に戻すだけ。PPO の設定は v14 を継承してログ名だけ。`tests/test_v14_cfg.py` の「古い版が v14 の項を使わない」検査が v15 で落ちないかも見る。終了項の関数そのものは Isaac Sim が要るので、下のゲートで確かめる。
5. **ゲート (Isaac Sim。学習の前)**:
   - 重なりの検査: v14 と v15 の学習用のタスクで、v12 s44 の `model_150.pt` (6 手目に段 0 を上げる方策。critic は 197 次元なので、読み込むなら `agent.obs_groups.critic=[policy]` を付ける) を 64 環境 × 13 手打ち、`env.step` が返す `terminated & truncated` の数を数える。v14 では 0 より大きく、v15 では 0 になること。小さな確認のスクリプトを `scripts/rsl_rl/` に足すか、スモークテストの run で `Episode_Termination/time_out` と `stage_up` を v14 と比べる (どちらで確かめたかを実験記録に書く)。
   - スモークテスト (skill `train-eval`、2 iteration): `params/env.yaml` の `terminations.time_out.func` が `time_out_unless_terminated`、Play の `params` では `time_out` に戻っていること、`Loss/value_mean_s0` が TensorBoard に出ること。
6. commit してから学習する。学習のコマンドは v14 と同じ形: `src/isaac_twisted_rl/scripts/run_job.sh --name v15_sS --timeout 30h --detach -- uv run train --task Isaac-Twisted-RL-Knot-v15 --headless --device cuda:N --seed S --max_iterations 301 --run_name v15_sS agent.device=cuda:N agent.save_interval=10` (S = 43 / 44。起動の前に `nvidia-smi` を見る)。
7. 評価: chain 評価を it70 / 100 / 150 / 200 / 250 / 300 で (`--task Isaac-Twisted-RL-Knot-Play-v15`、標準の条件、Hydra の上書きなし)、v14 の同じシード (`logs/rsl_rl/knot_tying_v14/2026-10-15_15-07-10_v14_s43`・`2026-10-15_16-07-22_v14_s44` の `eval_chain_it*.json`) と並べる。予想 2 のための `diag_stage_moves.py --dump_obs` (s44 の it150・it300 と s43 の it300)。結果は skill `experiment-record`。

## 出典

- 調査: [2026-10-15_timeout-bootstrap-and-stage2-bottleneck](../experiments/2026-10-15_timeout-bootstrap-and-stage2-bottleneck.md)、[2026-10-15_critic-inflation](../experiments/2026-10-15_critic-inflation.md) (冒頭の注記)。
- コード: `.venv/lib/python3.11/site-packages/isaaclab/source/isaaclab/isaaclab/managers/termination_manager.py:164-170`、`.venv/lib/python3.11/site-packages/isaaclab/source/isaaclab_rl/isaaclab_rl/rsl_rl/vecenv_wrapper.py:163`、`.venv/lib/python3.11/site-packages/rsl_rl/algorithms/ppo.py:161-164`、`kt/versions/v4/env_cfg.py:125-130`。
- 継承元を結果の後に決めることはユーザーが決めた (2026-10-15)。bootstrap の修正を v15・寸法 DR を v16 に入れ替えることはユーザーが決めた (2026-10-16)。継承元を v14 にし、v14 の切り分けの run を回さないこともユーザーが決めた (2026-10-16)。
