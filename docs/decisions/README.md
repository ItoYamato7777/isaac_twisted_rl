# decisions: 版をまたぐ設計の判断と理由

- 状態: 現行 (判断を足したら一覧に足す。判断が変わったら、その文書の「状態」を書き換え、新しい判断を別の番号で足す)
- 読むべき状況: 「なぜこうなっているか」を知りたいとき。行動空間・物理・報酬・終了・初期状態・ロープ・版の制度を変えたくなったとき
- 最終更新: 2026-10-20

1 判断 1 ファイル (`NNNN-<題>.md`)。各ファイルは「決定」(現在形)・「理由」・「検討して採らなかった案」・「今の実装」(file:line と、今のコードと合っているか)・「出典」の順に書く。1 つのモジュールで閉じる理由はそのモジュールの docstring に置き、ここには書かない。経緯の記録は [`docs/history/`](../history/) にある (読まなくてよい)。

文書の中の `kt/` は `src/isaac_twisted_rl/tasks/knot_tying/` の略。
`docs/history/next_action.md:NNN` の行番号は、その文書に頭書き (33 行) を足す前のもの。

## 一覧

| 番号 | 題 | 要旨 | 旧節番号 (`docs/history/next_action.md` ほか) |
|---|---|---|---|
| [0001](0001-kinematic-hand-and-one-move.md) | キネマティックハンドと「1 手 = 掴む→運ぶ→離す 全部」 | アームを付けず、1 RL ステップ = pick&place 1 回。最初の課題は交差 1 つ (v0) | §4.0 |
| [0002](0002-action-space-rope-relative.md) | 行動空間: ロープ相対 + `Δlink` | 6 次元。置き先は target リンクの現在位置からの相対。`Δlink` で N×N を N×17 に。端リンクは掴まない | §4.1、§3(b)、§4.6(a)(b) |
| [0003](0003-physics-240hz.md) | 物理は 240 Hz、残った発散はその場で reset | 120 Hz は発散 11.2%、240 Hz は 1.2%。めり込み解消速度が 240 Hz 前提 | §4.5 |
| [0004](0004-release-follow-physics.md) | 「離したのに付いてくる」は物理パラメータで直す | `twist` のヤング率 1.0e5 Pa、`OPEN_RATIO` 1.0。`settle_out` を削らない | §3(c) |
| [0005](0005-measure-after-retreat.md) | 成功の判定はハンド退避後、床に落ちているときだけ | 吊って見せかける方策を防ぐ。最高点が直径の 3 倍以下。`rope_zmax`・`finger_gap` をログに | §4.2 |
| [0006](0006-no-symbolic-planner-her.md) | 移植元の構成 (記号プランナ + SAC/HER) を使わない | HER と replay buffer のカリキュラムは off-policy 前提で PPO に載らない | §4.8.1、§4.7 |
| [0007](0007-topology-class-ladder-shaping.md) | 位相クラスのラダーをポテンシャル整形で登らせる (v1〜v3) | 交差数の抜け道を避け、居座りを防ぐ。γ を PPO と一致させる | §4.8.4、v1 README §1 |
| [0008](0008-transition-reward-and-termination.md) | 状態ボーナスを毎手払わない: 遷移で 1 回だけ払い、成功・段の変化で終了 | v2 の居座りの原因。v3 は初めて上がった手だけ、v4 は +1 / −1 で終了 | `paper/Report_Paper.md` §5.5、v3 README P1-2、v4 README |
| [0009](0009-reached-state-archive.md) | 到達した形状からエピソードを始める | 「干し草の中の針」を初期状態で解く。v4 は段ごと・初期アーカイブを読む | `paper/Report_Paper.md` §5.3、v3 README P1-1、v4 README、§4.8.10 の 6 |
| [0010](0010-stage-granularity-bfs.md) | 段の粒度: p-data グラフの BFS 距離で数える | cover 後を段 2 に数える。生の交差数も 4 クラスのラダーも粗い | `paper/Report_Paper.md` §5.4、§4.8.10 の 5、v3 README P1-3、v4 README |
| [0011](0011-success-rate-without-search.md) | 成功率は探索なしで測り、探索込みの数字と分ける | 論文の成功率は 30〜60 分の探索込み | `paper/Report_Paper.md` §5.2 |
| [0012](0012-version-is-config.md) | 「版 = 設定」であって「版 = コードのコピー」ではない | 版の差は配線と重みだけ。共有層の項は追加のみ。バグ修正は全版に効かせる | §6、`kt/README.md:154-173` (commit 62b251b) |
| [0013](0013-rope-knot.md) | 学習のロープは `Knot` (剛体鎖 28 リンク・1.14 m) | 長さが決め手。rope study でも剛体鎖を推す (摩擦は未決)。仕様の表 | §4.8.10 の 1、`docs/experiments/2026-10-02_rope_models_detail.md` §2.1・§10 |
| [0014](0014-gate1-wiring-for-knot.md) | `Knot` で学習するための 3 点 | 置き先 ±0.30 L、1 手 3.16 s (carry 0.6 s)、運搬高さ 0.20 m 固定 | §4.8.9 の 1 つ目、§4.8.10 の 2・3・4・7 |
| [0015](0015-shaping-verification-pitfall.md) | 整形報酬は計算の瞬間に捕まえて検証する | `env.step()` の後に読むと reset 後の値を見る | §4.8.8 |
| [0020](0020-v4-solver-iterations-and-steps-per-iteration.md) | v4 以降はソルバの位置反復を 8、1 iteration を 4 手にする | (別の文書。番号 0020〜0029 は v4 の速度と v5 の教訓) | — |
| [0021](0021-measure-physics-changes-by-training.md) | 物性を変えるときは、計測だけでなく学習と chain 評価で確かめる | (同上) | — |
| [0022](0022-p2-one-version-each.md) | 探索と汎化の改善 (P2) は 1 版に 1 つずつ。v8 は対称性のデータ拡張 | RND は rsl_rl の `rnd_cfg` のままでは使わない (終了した環境では reset 後の観測で新しさを数える) | `paper/Report_Paper.md` §7 P2 |
| [0023](0023-v10-path-stages.md) | v10 は段を固結びへの正規の経路の上の p-data だけに絞る (段 0〜3 に 1 / 2 / 4 / 2 個) | 段 2 には Fig.1 の C2 と台本の C2 の両方を入れる (Fig.1 の C2 は台本で 1 度も通っていない)。道から外れたら −1 で終了 | 論文 Fig.1、`overhand_script.py` |
| [0024](0024-v11-dead-end-stages.md) | v11 は v7 の段のまま、表に無くラダーでも 0 の「行き止まりの形」だけを −1 にする (偽の交差は R2 除去後の形で救う) | R2 除去は生の p-data が表に無いときだけ使う。段 2 の表に無い形 (ラダー 2) は外さない。iteration 300 まで学習 | v7 のアーカイブの p-data (`outputs/archive_pdata/`) |
| [0025](0025-archive-save-on-stage-up.md) | v12 は到達状態アーカイブに「段が上がった手の後の形」だけを保存する | 毎手を保存すると、段 k の方策が抜け出せない形が段 k のアーカイブに溜まるループになる (v11 s43 の段 2)。論文の l→k の規則に戻す。v7 を継承 | v11 s43 の段 2 のアーカイブの切り分け (`docs/experiments/2026-10-09_v11_stage2-archive-loop.md`) |
| [0026](0026-v13-action-bounds-loss.md) | v13 は PPO の損失に、方策の平均が行動の範囲 [−1, 1] の外へ出た分の罰則を足す | 範囲の外では行動がクリップされて学習の信号が無くなり、平均が流れる。tanh・std の下限は採らない。係数 0.01 を推奨。v12 を継承 | `docs/experiments/2026-10-14_action-saturation.md` |
| [0027](0027-v14-s0-reshape-dr-priv-critic.md) | v14 は段 0 で交差なしのまま形を変えた手の保存 (1 エピソード 1 回)・ロープの物理の DR (6 軸 ×0.5〜×2)・critic の特権観測を入れる | 段 0 のアーカイブがまっすぐに近い形だけ。物理の違いへの頑健性。3 つとも Hydra で切れる。摩擦の公称値は実測の 0.5。v13 を継承 | `docs/experiments/2026-10-14_friction-binding.md` |
| [0028](0028-v15-timeout-bootstrap-fix.md) | v15 は段の変化・発散と同じ手では時間切れを立てない (時間切れの bootstrap の不具合を直す) | 6 手目に段が上がった手に γ·V が足され、段 0 の V が膨らみうる。PPO の側では直せない。継承元は v14 (2026-10-16) | `docs/experiments/2026-10-15_timeout-bootstrap-and-stage2-bottleneck.md` |
| [0029](0029-v16-rope-size-dr.md) | v16 はロープの直径・長さを env ごとに振る (3 × 3 の格子、ゲートで外す) | 長さ ×0.8/1.0/1.25・直径 ×0.4/0.63/1.0 (実機は直径 8 mm・長さ ×0.895)。観測は正規化しない・critic に寸法 2 次元・指のゲインは固定。実装は v15 の結果の後。継承元はそのとき決める | — |
| [0030](0030-v17-shorter-wait-phases.md) | **不採用** (2026-10-20、ユーザー)。v17 は待ちのフェーズの尺を詰める (close 0.2・hold 0.1・release 0.15・settle_out 0.15・relax 0.25 s。1 手 759 → 591 ステップ) | 1 手の時間は v15 の 0.766 倍になったが、0 から学ぶと 2 シードとも固結びが v15 を下回り (it150〜300 で −31.9・−7.1 ポイント)、chain 評価の発散が増えた。v14 の方策の評価 (−0.3 ポイント) は学習の結果を予測しなかった (0021 と同じ型) | `docs/experiments/2026-10-16_physics-speed.md`・`2026-10-20_v17_shorter-wait-phases.md` |
| [0031](0031-eval-chain-video.md) | chain 評価を回すときは、評価したエピソードの動画も `video/` に残す (学習の途中は撮らない) | 目的は報酬のハックと物理的におかしい挙動を人が見ること。動画は結論の根拠にしない。`--video` で数値が変わらないことを実装のゲートにする。`video/highlights/` だけ動画ごと commit し、`sim/`・`real/` の動画は追跡しない。**未実装** (次のセッション) | — |
| [0032](0032-sim-only-sim2real-goal.md) | このリポジトリは Sim の学習にとどめ、Sim でできることを実機でもできる方策を作る。評価の初期状態はまっすぐなロープから始め、広げるのはユーザーの指示のとき | 最終目標は実機 (`hiro_knot`) で固結び。実機の達成条件は「Sim でできることは実機でもできる」(数値の目標は未決)。Sim2Real のギャップに強いことを判断の軸にする。初期状態を交差なしの様々な形へ広げるのは、まっすぐな状態の成功率が高くなるか直す内容が尽きたときに、ユーザーが指示する | — |

## decisions を作らなかった理由の候補

理由がすでにコードの docstring に全部あるので、二重にしない。

| 話題 | 旧節 | 理由の置き場所 |
|---|---|---|
| 固結びの判定 (テンソル層だけで判定できる・厳密な `交差数 == 3` を使わない) | §4.8.2、§4.8.3 | `kt/common/topology/knot_detect.py` の docstring (3 条件・528 個の列挙・部分列判定・gap 1 を後ろへ回す修正) |
| 観測・報酬・終了をテンソル層だけで組む (p-data 文字列を使わない、退化時の `ValueError` が起きない) | §4.3、§4.4 | `kt/common/mdp/observations.py`・`kt/common/mdp/terminations.py` の docstring |
| body を名前で解決する (順序の実測) | §2 | `kt/common/rope_state.py` の docstring |
| 移植元から意図的に変えた点 (shapely 不使用・細分の廃止・camelCase) | §1 | `kt/common/topology/intersections.py:1-60`・`representation.py:38` の docstring。面の走査の上限 (定数 10 → 辺の総数) と「4096 環境で 2735 ms → 1.6 ms」は docstring に見当たらず、旧 §1 (`docs/history/next_action.md:125-134`) にだけある (局所の理由なので、docstring へ移すのが本来の行き先) |
