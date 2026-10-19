# isaac_twisted_rl

Isaac Lab 上で、ロボットハンドがロープを固結び (overhand knot) にする強化学習の研究コード。
現在地は `docs/STATUS.md`、文書の地図は `docs/README.md`、用語は `docs/glossary.md`。

## 最終的な目標 (`docs/decisions/0032-sim-only-sim2real-goal.md`)

- 実機 (別リポジトリ `hiro_knot`) で固結びにすること。このリポジトリは Sim の学習にとどめる。実機の達成条件は「Sim でできることを実機でもできる」で、Sim2Real のギャップに強い方策を作ることを判断の軸にする。
- 評価の初期状態は今はまっすぐなロープ。交差なしの様々な形へ広げるのは、ユーザーが明示的に指示したとき (まっすぐな状態の成功率が高くなるか、直す内容が尽きたとき)。Claude が先に始めない。

## セッションの始めと終わり

- 始め: git status・`docs/STATUS.md` の要点・走っているジョブ (`nvidia-smi`) を確かめる (SessionStart hook が冒頭に出す)。次の作業は STATUS の「次の作業」の 1 番目から。
- 終わり・区切り: skill `handoff` で `docs/STATUS.md` を上書きする。追記で伸ばさない。結果は skill `experiment-record` で `docs/experiments/` へ。

## 検証のコマンド

- テスト (約 50 s、Isaac Sim 不要): `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run python -m pytest tests/ -q`
  - 環境変数は、`/opt/ros/jazzy` の `launch_testing` が pytest のプラグインの自動読み込みに引っかかり `ModuleNotFoundError: lark` で落ちるのを避けるため。
  - `uv run pytest` は使わない (`.venv/bin/pytest` の shebang が別のリポジトリの python を指す)。
- lint: 設定は `ruff.toml` (版を固定)。`uvx ruff@0.16.10 check <ファイル>`・`uvx ruff@0.16.10 format <ファイル>` (`uv run ruff` は無い)。`--select` で規則を足して `--fix` をかけない (起動スクリプトの import の順序や、Warp のカーネルの中の `float(0.0)` を壊すものがある)。

## 壊してはいけない制約 (`tests/test_invariants.py` が検査する)

- `tasks/` の下の `__init__.py` で isaaclab.sim などを import しない。`import_packages` が Kit の起動前に全部実行する。
- フォルダ名 `mdp` を変えない。タスクの登録から外す除外の一覧が、この名前の部分一致で決まる。
- `common/topology/` と `common/state_archive.py` は isaaclab・omni・pxr を import しない。Isaac Sim なしで単体テストを回すため。
- `common/mdp/` の項は追加だけ。削除・改名しない。過去の版の設定が名前で参照している。
- 観測・行動・フェーズ・段の定義・`common/topology/` の関数を変える版は、実機側 `hiro_knot` の再実装とずれる (エラーは出ない)。変えるときは `docs/decisions/0032-sim-only-sim2real-goal.md` の「hiro_knot との境界」を読み、版の README に hiro_knot 側の更新の要否を書く。
- 既存の版 (`versions/vN/`) の挙動を変えない。変えたいときは新しい版を作る (skill `add-version`)。

## 実験の約束

- 版どうしの比較は chain 評価 (定義: `docs/system/evaluation.md`) で、シード 2 本以上。学習ログの段ごとの成功率や `chain_est` で比べない。
- 学習・評価の前にコードを commit する。未追跡のファイルは run のフォルダの `git/*.diff` に残らず、run を再現できなくなる。
- chain 評価には `--video` を付けて動画を `video/sim/` に残すが、**動画の中身は見ない・判断しない。動画の判断はユーザーがする** (ユーザーが「動画も踏まえて評価して」と明示したときだけ見る)。評価・考察は JSON の数値で書き、実験記録の動画の欄には場所と `manifest.json` の事実 (撮れたか・クリップ数・結末・手数) だけを書く。`docs/decisions/0031-eval-chain-video.md` の決定 3。
- 学習ログや代理の指標だけで出した結論は「推測」と書く。事実として commit メッセージや README に書かない。
- 評価には学習と同じ Hydra の上書きを付ける (付けないと違う物理で評価され、エラーは出ない)。GPU を指定するときは `--device cuda:N` と `agent.device=cuda:N` を揃える (`--device` だけでは PPO が `cuda:0` に残る)。
- 最後のチェックポイントの番号は `ls` で確かめる (rope study の run は `model_99.pt` で、`model_100.pt` は無い)。
- 手順は skill `train-eval` (学習と chain 評価)、`physics-check` (物理を変えたときのゲート 1 と発散の計測)。

## 長時間のジョブと GPU

- GPU は他のジョブと共有している。起動の前に `nvidia-smi` を見る。
- Isaac Sim を使うジョブは `src/isaac_twisted_rl/scripts/run_job.sh --name <名前> --timeout <上限> --detach -- <コマンド>` で起動する (バッファ無効・`timeout -k`・ログと記録 `outputs/jobs/<時刻>_<名前>/` をまとめて行う。Kit は固まると SIGTERM で止まらないことがある。`--detach` が無いと session の終わりに止まる)。
- exit 0 は完了を意味しない (Isaac Lab は SIGTERM でも 0 で終わる)。job の `exit.txt` に `signal:` の行が無いことと、最後のチェックポイントで確かめる。
- 待つときは前景の `sleep` を使わず、`run_in_background` か Monitor を使う。
- 自分が起動していないプロセスを kill しない。PID をユーザーに伝える。`pkill -f` は使わない (パターンを含む自分のシェルも殺す)。
- 残すスクリプトとログは、追跡する場所か `outputs/` に置く。scratchpad (`/tmp/claude-*`) を文書やスクリプトから参照しない。

## 触らないもの

- `/home/itoyama/work/IsaacLab/` の下の旧実装の knot_tying (rsl-rl 5.x 前提の移植前のコード) は参照しない。このリポジトリは rsl-rl 3.0.1・isaaclab 2.3.0 (`.venv` の中)。IsaacLab の checkout の例をそのまま写さない。
- `logs/`・`outputs/` は学習・評価のスクリプトが書く。手で編集しない (git の管理外で戻せない)。
- ロープの USD (`common/rope_model/data/`) と `outputs/archive_seed/` を上書きしない。USD は `generate_assets.py --rope <名前>`、初期アーカイブは `overhand_script.py --dump_archive <新しいパス>` で作る。

## 書き方

- 文書と docstring は日本語。ルートの `README.md` だけ英語。
- docstring には「何をするか」より「なぜそうしたか (今も有効な理由)」を書く。移植元と挙動を変えるときは、変えた理由と検証の結果を残す。経緯は書かない。
- 現在の仕様は現在形で書く。結果には条件・シード・iteration・評価の方法を付け、「分かった / 推測 / 未確認」を分ける。
- 1 つの情報の正本は 1 か所。現在地は `docs/STATUS.md`、結果は `docs/experiments/`、横断的な判断は `docs/decisions/`、経緯は `docs/history/` (通常は読まない)。
