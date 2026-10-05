# scripts/rsl_rl: 学習・評価のスクリプト

| スクリプト | 用途 |
|---|---|
| `train.py` / `play.py` | 学習と再生 (`uv run train` / `uv run play`) |
| `eval_chain.py` | v4 以降の **chain 評価**: Play 環境でまっすぐなロープから 18 手以内に固結びを結べるか。各環境の最初の 1 エピソードを数える |
| `eval_search.py` | **推論時探索の評価**: 同じチェックポイントに「探索ありの成功率」を付ける (この README の本題) |
| `summarize_search.py` | `eval_chain.py` / `eval_search.py` の JSON から版ごとの比較表 (Markdown) を出す |
| `stage_colors.py` | `play.py` が使う: 段ごとの方策 (v4 以降) で、各環境がいま使っている段の方策をグリッパの色で見せる (下の「Play のグリッパの色」) |
| `symmetry_check.py` | v8 の対称性のデータ拡張の前提の確認: 計算し直した位相の観測が環境と一致するか、鏡像・逆順の手が物理で同じ結果になるか (同じ手どうしのずれと比べる) |

## Play のグリッパの色

`uv run play` は、段ごとの方策 (`StageGatedActorCritic`、v4 以降) のとき、各環境のグリッパを **この手を打つ方策の段** の色に塗る
(描画のあるときだけ。`--no_stage_colors` で切る)。段は方策が振り分けに使う `GatedMLP.stage_index` で取るので、色と
実際に使われたヘッドは一致する。

| 色 | 意味 |
|---|---|
| 青 / 黄 / 赤 | 段 0 / 1 / 2 の方策 |
| 白 | 固結びに達した (reset の後も 3 手のあいだ残す) |

色は USD の `displayColor` だけを書き換え、物理・観測・環境の設定には触れない。ロープは紐モデルごとに描き方が違うので塗らない。

結果のまとめは [`docs/experiments/2026-10-03_inference_search.md`](../../../../docs/experiments/2026-10-03_inference_search.md) (打ち切り中)。

## なぜ推論時探索を測るか

論文 (TWISTED / TWISTED-RL / Yan2020) の成功率は、シミュレータを世界モデルとして使い、候補の手を実際に
試して探索した結果 (最大 30 分) で、chain 評価 (方策を 1 回展開しただけ) とは土俵が違う
([`paper/Report_Paper.md`](../../../../paper/Report_Paper.md) §5.2、§7 P2-4)。探索を入れると、発散した候補は捨てるだけで済み、段 2 で手を
使い切る方策も別の候補でやり直せる。版の順位が変わるかを見るため、同じチェックポイント・同じ手数の予算で
3 つを並べる:

| | 方式 | `eval_search.py` のモード |
|---|---|---|
| (a) | chain 評価 (決定論的・確率的) | 決定論的は `eval_chain.py`、確率的は `rollouts` の各環境の最初の 1 エピソード |
| (b) | best-of-N: 同じ手数の予算で確率的な展開を独立に繰り返す (探索なしの基準) | `rollouts` |
| (c) | ビーム探索 + 到達木からの再開 | `search` |

**環境の新しいバージョンではない** (`versions/` には足していない)。共有層への追加は
`common/beam_search.py` (純 torch) と `common/mdp/state_clone.py` (状態の複製) の 2 ファイルだけで、
既存の項・既存のバージョンの挙動は変わらない。

## 使い方

起動方法と引数は `eval_chain.py` と同じ (Play 環境、Hydra の上書きは後ろに付ける。**学習で物理を上書きした
run は評価にも同じ上書きを付ける**)。結果は既定でチェックポイントのフォルダの `eval_search_{mode}_it{N}.json`。

```bash
C=logs/rsl_rl/knot_tying_v6/2026-10-01_19-23-03_overnight_s42/model_100.pt
# 複製の検証 (下の「状態の複製」)
uv run python -u src/isaac_twisted_rl/scripts/rsl_rl/eval_search.py --task Isaac-Twisted-RL-Knot-Play-v6 \
    --mode clone_test --num_envs 128 --headless --device cuda:0 --seed 7 --checkpoint $C agent.device=cuda:0
# (a) 確率的な chain 評価 + (b) best-of-N (512 環境 x 2 エピソード)
uv run python -u src/isaac_twisted_rl/scripts/rsl_rl/eval_search.py --task Isaac-Twisted-RL-Knot-Play-v6 \
    --mode rollouts --episodes_per_env 2 --num_envs 512 --budgets 16,32,64,128,256,512,1024 \
    --headless --device cuda:0 --seed 7 --checkpoint $C agent.device=cuda:0
# (c) ビーム探索 (32 問題を並列 x 16 環境、計 128 問題、予算 1024 手 / 問題)
uv run python -u src/isaac_twisted_rl/scripts/rsl_rl/eval_search.py --task Isaac-Twisted-RL-Knot-Play-v6 \
    --mode search --slots 16 --beam 4 --problems 128 --max_moves 1024 --num_envs 512 \
    --budgets 16,32,64,128,256,512,1024 --headless --device cuda:0 --seed 7 --checkpoint $C agent.device=cuda:0
# 段 2 の形ごとの固結びの割合 (下の「形ごとの切り分け」。128 形 x 64 回、6 手、決定論的な方策)
uv run python -u src/isaac_twisted_rl/scripts/rsl_rl/eval_search.py --task Isaac-Twisted-RL-Knot-Play-v6 \
    --mode shapes --shape_stage 2 --shapes 128 --shape_reps 64 --shape_moves 6 --deterministic --num_envs 512 \
    --headless --device cuda:0 --seed 7 --checkpoint $C agent.device=cuda:0
# 比較表
uv run python src/isaac_twisted_rl/scripts/rsl_rl/summarize_search.py v6s42=$C ...
```

## (c) ビーム探索のアルゴリズム (`common/beam_search.py`)

### 問題と枠

問題は「まっすぐなロープから固結びまで」。**すべての問題が同じまっすぐなロープから始まる** ので、問題の違いは
探索の乱数と物理の非決定性だけ (論文の「同じ問題を 5 seed」と同じ形)。成功率は「予算内に固結びを見つけた
問題の割合」。

環境を S 個ずつの **枠** に分け、G = 環境数 / S 個の問題を並列に解く。問題が終わると (解けた / 予算切れ)、
その枠で次の問題を始める。1 ラウンド = `env.step()` 1 回 (全環境が 1 手)。

### 1 ラウンド

1. 各枠のフロンティア (最大 M ノード) に S 個の環境を割り振り、ノードの状態を複製する。M が S を割り切らない
   ときは、良い順で先頭のノードに 1 つずつ多く。
2. 方策を std 付きでサンプルして 1 手打つ。**各ノードの最初の 1 環境だけは平均の行動** (決定論的方策) にする
   (`--no_mean_first` で全部サンプル)。
3. 結果を分類する:

   | 結果 | 条件 | 扱い |
   |---|---|---|
   | 固結び | Play の終了項 `stage_up` (chain: 固結び・床に落ちている・発散していない) | **解けた** |
   | 発散 | 終了項 `diverged` | 捨てる |
   | 段が上がった | 床に落ちていて、観測の段が親より上 | 次のフロンティアへ (優先) |
   | 同じ | 観測の段が親と同じ (吊ったままでも段が同じなら含める) | V(s) で順位付け |
   | 段が下がった | 床に落ちていて、観測の段が親より下 | 捨てる |
   | 吊ったまま | 床に落ちておらず、観測の段が親と違う | 捨てる (同じ段の critic で比べられない) |

4. 「上がった」と「同じ」から次のフロンティアを M 個選ぶ: **段の高い候補を先に取り、同じ段の中だけを
   その段の critic の V(s) で比べる** (値で並べてから段で安定ソート)。`StageGatedActorCritic` の critic は
   段ごとに別のネットワークで、尺度が揃っていないので、段が違う値どうしは比べない。
5. フロンティアに入ったノードを到達木に足す (親・深さ・段・値・同じ段が続いた手数を持つ)。

### 線の打ち切りと再開

* 同じ段のまま **6 手** (`--max_same`) 続いた線は捨てる。学習の 1 遷移の上限 (6 手) に合わせた。方策はこれより
  長く同じ段に居る状況を経験していない。
* フロンティアが空になったら (全部が発散・段が下がった・打ち切り)、到達木から M 個を重複なしに選んで再開する。
  TWISTED の「交差数に比例した優先度 + 確率 p でランダム」に倣い、重みは
  `(1 − p) x 段 / (1 + そのノードから再開した回数)` を正規化したもの + `p x 一様` (p = 0.05)。
  * 交差数の代わりに段 (round(3Φ)、0 = 自明・3 = 固結び) を使うのは、段は固結びへの正規の経路の上では交差数と一致し、それ以外の状態でも固結びへの近さを表すため (cover 後や偽の C2 の
    生の交差数は近さと合わない。`versions/v4/README.md` の「段」)。
  * 再開の回数で割るのは、同じノードからの再開ばかりにならないようにするため (TWISTED には無い)。
  * 再開したノードの「同じ段が続いた手数」は 0 に戻す (新しい線として数える)。
  * まっすぐなロープ (根) は段 0 なので、確率 p の一様な抽出でだけ選ばれる。

### 予算

* **問題ごとの「シミュレートした手の総数」**。1 ラウンドで枠の S 環境ぶん増える。固結びが出たラウンドの手も
  全部数える (並列に試した候補は全部シミュレートしているため)。予算の残りが S より少なければ、残りの数の
  環境だけを使う。
* 実時間の打ち切り `--max_minutes`: 超えたら走っている問題を未解決で終え、残りの問題は始めない。
* 1 回の run で予算の最大値まで回し、「解けた時点の手数」から小さい予算の成功率も読む (`summary.success`)。
  ラウンド単位 (S 手) で数えるので、S より細かい予算では少し悲観的に出る。

### 実時間との換算

JSON の `throughput.moves_per_minute_per_problem` は、**この run で 1 問題が実際に受けた処理速度**
(G 問題が GPU を分け合う) での 1 分あたりの手数。「x 分相当の予算」はこれを掛けた手数になる。GPU を他の
ジョブと共有しているときは小さく出るので、比べるときは手数の予算で比べること。

## (b) best-of-N の計算 (`rollouts`)

Play 環境のまま (18 手で時間切れ) 確率的な方策を回し、各環境の最初の E エピソード (`--episodes_per_env`、
既定 2) を集める。18 E + 1 手回せば全環境が E 本を終えるので、打ち切りで短いエピソードに偏らない。

すべての問題が同じまっすぐなロープから始まるので、エピソードは独立同分布とみなせる。集めたエピソードの
(手数, 成否) の経験分布から、予算 b 手での成功率 f(b) を再生方程式で厳密に計算する
(`beam_search.best_of_n_curve`):

    f(b) = P(成功かつ L <= b) + Σ_l P(失敗かつ L = l) f(b − l)

エピソードは途中で打ち切れない (予算内に終わらなかったエピソードの成功は数えない)。成功したエピソードの手数は
固結びに着いた手まで、発散したエピソードは発散した手まで。逐次に 1 本ずつ打つ勘定なので、成功した時点で
止められる分だけ (c) より有利な数え方になっている。最初の固結びまでの手数の期待値は `E[L] / p`。

## 状態の複製 (`common/mdp/state_clone.py`)

アーカイブからの復元 (`ResetToStage`) は重心を原点へ移し、ランダムな yaw をかけ、速度を 0 にする。探索の複製は
**複製元と同じ姿勢・同じ速度** で、方策が見る観測まで一致しなければならない。env 原点からの相対位置を保ち、
原点の差だけを足し直す。

| 持ち主 | 写すもの | 写さないと |
|---|---|---|
| ロープ | root link の姿勢・root の重心速度・関節角・関節速度 | 形・発散判定 (速度) |
| グリッパー | root の姿勢・速度、指の関節角・速度・目標値 | 観測 `hand_pos`、指の開き |
| `PickPlaceAction` | `_hand_pos` / `_hand_yaw` (次の手の approach の始点)、補間区間、フェーズ、`_raw_actions` | 次の手の始点が複製先の古い位置になる |
| `ActionManager` | `_action` / `_prev_action` | 観測 `actions` (`last_action`) |
| `NextEdgeGoal` (v6〜) | 選んだ辺 `key` / `row` / `cand` | 位相が同じ間は選び直さないので、複製先の形で `long_end` の選択が変わりうる |
| env | `episode_length_buf` | 時間切れ |
| 位相抽出器 | 写さずに捨てる (`invalidate_extractor`) | 観測が複製前の形状のまま計算される |

PhysX の内部状態 (接触のキャッシュ・ソルバのウォームスタート) は API から読み書きできないので写せない。

### Play 環境の自動 reset との付き合い方

* `clone_test` / `search` では **18 手の時間切れを切る** (`episode_length_s` を 10^7 s に)。ノードは何手目でも続けられる。
* **固結び・発散の reset はそのまま使う。** どちらも探索ではその候補の終わり (成功 / 捨てる) なので、reset の後の
  状態は使わない。結果は reset の前に立った終了項 (`reset_buf` と `termination_manager.get_term`) から読む。
* 毎ラウンド、使う環境すべてにノードの状態を書き込み直すので、前のラウンドや reset の後の状態は残らない。
* Play の p-data の表示 (`print_envs`) と初期アーカイブの読み込みは切る (評価に要らない。表示は毎手出て遅い)。

### 検証 (`--mode clone_test`、v6 seed 42 の `model_100`、2026-10-03)

環境を 8 個ずつの区画に分け、確率的な方策で 2 手進めた後、区画の先頭 (複製元) の状態を残り 7 個へ写す。
(1) 複製直後の観測と段を複製元と比べ、(2) 区画の全員に複製元の観測での平均の行動を打たせて結果を比べる。
`search` でも毎ラウンド (1) を数えて JSON の `clone_check` に出す。

| 確認 | 結果 |
|---|---|
| 複製直後の観測の不一致 (最大の差が 1e-4 を超えた行) | **0 / 448** (64 環境 x 2 回 + 128 環境 x 3 回)。最大の差 7e-6 (リンク位置の FK の丸め) |
| 複製直後の段の不一致 | 0 / 448 |
| 検証の感度: 直前の行動と選んだ辺を写さない (`--clone_skip next_edge,action_manager`) | 観測の不一致 100% (`actions` 100%、`next_edge` 6.2%) |
| 同じ手の後の段の一致: 複製元と複製 | 73% (330 組) |
| 同じ手の後の段の一致: 複製どうし | 71% (282 組) |
| 同じ手の後のロープの節点のずれ (中央値 / 90%) | 9 cm / 33 cm (複製元と複製)、10 cm / 38 cm (複製どうし) |

* 複製は、方策が見る観測の上では完全に一致する。選んだ辺を写さないと 6% の環境で `next_edge` が変わる
  (`long_end` の選び直しが複製先の形で変わる) ので、写す必要がある。
* **同じ状態・同じ手でも、手の後の段は 3 割ほど変わる。** 複製元 (PhysX の内部状態を持っている唯一の環境) との
  一致率が複製どうしの一致率と変わらないので、写せない状態のせいではなく、物理そのものの非決定性 (接触の多い
  1 手 = 759 物理ステップの間に小さな差が育つ) と読める (解釈)。探索の側から見ると、同じノード・同じ手でも
  候補が分かれる。

## 形ごとの切り分け (`--mode shapes`)

段 2 -> 固結びの失敗が「初期の形で決まる」のか「試行ごとのばらつき (物理の非決定性・方策のサンプル)」なのかを分ける。
段のアーカイブ (`--shape_archive`。既定はチェックポイントの隣の `archive.pt` = 学習中に到達した形。台本の初期アーカイブは
`outputs/archive_seed/overhand_28links.pt`) から `--shape_stage` の段の形を `--shapes` 個選び、それぞれを R = `--shape_reps`
個の環境に**同じ初期状態** (重心を原点へ、速度 0、yaw 0) で書き込んで、同じ方策で `--shape_moves` 手打つ。

* 環境は Play (chain モード) のまま。段が下がっても続ける。各環境の最初のエピソードだけを数える。
* `--deterministic` (平均の行動) なら、試行の違いは物理の非決定性だけになる。付けなければ方策のサンプルも入る。
* 書き込んだ後の段が `--shape_stage` と違う形 (アーカイブの復元の不一致) は集計から外し、件数を出す。
* 形ごとの成功率が 0 と 1 に分かれれば形で決まり、全体の平均の付近に集まればばらつき。`shape_share` (級内相関) は、
  成功の分散のうち形の間の成功率の違いで説明できる割合 (0 = どの形も同じ、1 = 形ごとに必ず成功か必ず失敗)。
  集計は `common/beam_search.shape_rate_summary` (単体テストあり)。

## 出力 (JSON の主な key)

| モード | key | 意味 |
|---|---|---|
| `rollouts` | `chain.knot` / `diverged` / `time_out` | 各環境の最初の 1 エピソード (= 確率的な方策の chain 評価) |
| | `pool.*` | 集めた全エピソード。`moves_lost_to_divergence_frac` は全手数のうち発散したエピソードに使った割合 |
| | `best_of_n.success` | `best_of_n.budgets` ごとの best-of-N の成功率 |
| | `best_of_n.expected_moves_to_knot` | 最初の固結びまでの手数の期待値 |
| `search` | `summary.success` | `summary.budgets` ごとのビーム探索の成功率 |
| | `summary.moves_at_solve_*` / `depth_at_solve_*` | 解けた時点のシミュレートした手数 / 解の手数 (まっすぐなロープから固結びまでに打った手) |
| | `summary.candidate_frac` | 全候補のうち固結び・上がった・同じ・下がった・発散・吊ったままの割合 |
| | `throughput.*` | 1 ラウンドの秒数、1 問題あたりの手/分 |
| | `clone_check` | 複製直後の観測・段の不一致 (項ごとの最大の差も) |
| | `problems_detail` | 問題ごとの記録 |
| `clone_test` | `clone_check` / `same_action` | 上の「検証」 |
| `shapes` | `summary.knot` / `summary.knot_no_drop` | `shape_rate_summary` の結果。`knot` は M 手以内の固結び、`knot_no_drop` はその中で一度も段が下がらなかったもの (学習の 1 遷移の成功に近い) |
| | `summary.knot_by_move` | 手数ごとの固結びの割合 (全試行) |
| | `summary.by_p_data` | 初期の形の p-data (R2 除去前) ごとの形の数・成功率 |
| | `shapes_detail` | 形ごとの記録 (アーカイブの添字・初期の段・p-data・成功数・発散・段が下がった数) |
