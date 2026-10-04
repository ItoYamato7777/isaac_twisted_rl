# 作業の型: 研究の 1 サイクル・セッション・長時間ジョブ

- 状態: 現行
- 読むべき状況: 研究の次の 1 サイクルを始めるとき、どの段階を飛ばしてよいかを決めるとき、作業の流れの全体を見渡したいとき
- 最終更新: 2026-10-04

この文書は段階・終了の条件・残す記録だけを書く。コマンドと確認の項目は skill にある (`.claude/skills/<名前>/SKILL.md`)。
規則のうち毎セッション要るものは `CLAUDE.md` にあり、ここでは繰り返さない。

## 1. 研究の 1 サイクル

| 段階 | やること | 終了の条件 | 残す記録 | skill |
|---|---|---|---|---|
| 1. 仮説 | 何を変えると何がどれだけ良くなるかを書く。比べる相手 (基準の版) と指標 (chain 評価の何 %) とシード数を決める | 仮説・基準・指標・シード数が STATUS の「次の作業」にある | [`STATUS.md`](STATUS.md) | `handoff` |
| 2. 実装 | 新しい版か設定の変更。既存の版の挙動は変えない (変えたいときは新しい版) | テストと lint が通る。**commit がある** (未追跡のファイルも含めて) | commit | `add-version` |
| 3. ゲート (学習の前) | スモークテスト (数 iteration)。物理・ロープのプリセット・紐モデルを変えたときだけ、ゲート 1 と 1 手あたりの発散の計測も | スモークテストが落ちない。物理を変えたなら、ゲート 1 の結果の表と発散率が出ている | 実験記録の「ゲート」欄 | `train-eval` §1、`physics-check` |
| 4. 学習 | シード 2 本以上。GPU の空きを確かめてから起動 | 最後のチェックポイントがある (番号は `ls` で確かめる)。run のフォルダ名が分かっている | STATUS の「走っているジョブ」(PID・GPU・run のフォルダ) | `train-eval` §2 |
| 5. 評価 | chain 評価。条件は [`system/evaluation.md`](system/evaluation.md) §1 の標準の条件、Hydra の上書きは学習と同じ | シードごとの評価 JSON がある | 評価 JSON のパス | `train-eval` §3 |
| 6. 実験記録 | テンプレートで書く。否定的な結果も書く | 再現のメタデータに空欄が無い (分からないものは「記録なし」) | [`experiments/`](experiments/README.md) の 1 ファイルと一覧の 1 行 | `experiment-record` |
| 7. 判断 | 標準誤差とシード差を並べて、仮説を採るか捨てるかを決める。「分かった / 推測 / 未確認」を分ける | 結論が実験記録の「結論」と STATUS にある | STATUS。設計を変えたなら [`decisions/`](decisions/README.md) | `train-eval` §4、`handoff` |

### 飛ばしてよい段階

- 3 のゲート 1 と発散の計測は、物理・プリセット・紐モデルを変えたときだけ要る。報酬・観測・カリキュラムだけを変えたときは、スモークテストだけでよい。
- 1 シードの学習は、設定が動くかを見る切り分けにだけ使う。版どうしの結論には使わない (4 を 2 シード以上にする)。

### この型の根拠になった知見

- シード差が大きい: v6 の seed 42 と 43 は chain 評価で iteration 70 に 19 ポイント、iteration 100 に 15 ポイント離れた。出典は [overnight の記録](experiments/2026-10-01_v4-v6_overnight.md) の 1 組だけで、一般則ではない。それでも 1 シードどうしの 10〜15 ポイントの差は判別できないので、2 シード以上で比べる。
- 学習ログの段ごとの成功率と `chain_est` は、本番の成功率を表さない (同じ記録の §4.5。v6 では `chain_est` 0.03 に対して chain 評価 23〜38%)。比べるのは chain 評価だけにする。
- 物理の計測で効いた変更が、学習では逆効果になったことがある ([0021](decisions/0021-measure-physics-changes-by-training.md))。計測だけで採らず、学習と chain 評価で確かめる。
- 整形報酬は、測り方を誤ると効いて見える ([0015](decisions/0015-shaping-verification-pitfall.md))。

## 2. セッションの規約

| 時 | やること | 仕組み |
|---|---|---|
| 始め | git status・STATUS の要点・走っているジョブを確かめる。次の作業は STATUS の「次の作業」の 1 番目から | SessionStart hook が冒頭に出す。`CLAUDE.md` |
| 途中 | 長い待ちは前景の `sleep` を使わない (バックグラウンドの実行か監視を使う) | `CLAUDE.md` |
| 終わり・区切り | STATUS を上書きする (走っているジョブと PID を含む)。結果は実験記録へ。テストと lint が通り、コミットできる状態にする | skill `handoff` |

引き継ぎメモを追記で伸ばす形 (過去の [`history/rope_study_HANDOFF.md`](history/rope_study_HANDOFF.md)) は使わない。
情報を「上書きで最新を保つ STATUS」と「一度書いたら変えない実験記録」に分ける。

## 3. 長時間ジョブの規約

| 項目 | 規約 | 仕組み |
|---|---|---|
| 起動 | Isaac Sim を使うジョブは `src/isaac_twisted_rl/scripts/run_job.sh` で起動する。session をまたぐジョブは `--detach` を付ける (付けないと session の終わりに止まる) | `run_job.sh`、`CLAUDE.md` |
| 記録 | `run_job.sh` が `outputs/jobs/<時刻>_<名前>/` に PID・GPU・commit・未コミットの差分・コマンド・ログ・終了の状態を書く。STATUS の「走っているジョブ」に PID と run のフォルダを書く | `run_job.sh`、skill `handoff` |
| 完了の確かめ | exit 0 は完了を意味しない (Isaac Lab は SIGTERM でも 0 で終わる)。`exit.txt` に `signal:` の行が無いことと、最後のチェックポイントで確かめる | `CLAUDE.md` |
| 止め方 | 自分が起動したジョブ (自分の記録にある PID) だけを止める。それ以外は PID を人に伝える。`pkill -f` は使わない | permissions (kill は確認、`pkill -f` は拒否) |
| GPU の共有 | GPU は他のジョブと共有している。起動の前に `nvidia-smi` を見る。時間を測るときは、そのときの GPU の負荷を記録に書く | `CLAUDE.md` |
