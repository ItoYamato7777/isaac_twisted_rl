> **状態: 経緯の記録 (凍結)。2026-10-02〜03 の rope study の 4 つのセッションの引き継ぎメモ。通常は読まない**
> - 旧パス `src/isaac_twisted_rl/tasks/knot_tying/rope_study/HANDOFF.md`。中の「最新」「ここから読む」の印は当時のもの。
> - 結果と有効な run の一覧は `docs/experiments/2026-10-02_rope_models.md`、保留の作業は `docs/STATUS.md`、踏んだ注意の主なもの (kill の拒否・`pkill -f`・Kit の固まりと `timeout -k`・前景の `sleep`) は `CLAUDE.md` に移した。
> - 中の相対パス `logs/…` の多くは `outputs/rope_study/logs/…` の意味 (リポジトリ直下の `logs/` ではない)。

# 引き継ぎメモ: 紐モデルの調査・実装・比較 (rope study)

- 1 回目のセッション: 〜2026-10-02 17:57 (コンテキストの都合で中断)。4 回目 (〜2026-10-03 13:10) で最終報告まで完了
- 2 回目のセッション: 2026-10-02 17:58〜18:50 頃。**途中から Bash が一切使えなくなった** (下の §0)
- 最終成果物: `Report_rope_models.md` (この フォルダ)。**中間版** として全節を記入済み (§7.3 強化学習・§7.4 転移は未実施のまま)
- 2026-10-03 にユーザーの指示でコミットした (5 コミット: rope_model の knot56 → overhand_script → rope_backends → rope_study → 報告書)。出力 (`outputs/`・`logs/`) と USD はコミットしていない (.gitignore)

## 00000. 最終状態 (2026-10-03 13:10、4 回目のセッションで終了) ← 最初にここを読む

* **最終報告書 `Report_rope_models.md` は完成** (§0 要旨、§10 結論)。結論: 実機の検証へ持っていく最有力は PhysX の剛体鎖
  (28 リンク、`Art28` 系)。摩擦 (2.0 / 0.5) は決めきれない。XPBD Cosserat は物理では最良だが、学習の途中で段 0 の方策が崩れ、
  転移の最悪値が低いので今回は推さない。
* ユーザーの指示で、このセッションの評価と実装はここで打ち切った。**走らせているプロセスは無い**。2026-10-03 にコミット済み (6 行目)。
* テスト 247 件・ruff 通過。表: `outputs/rope_study/eval/summary.md`、`outputs/rope_study/train_summary.md`。

### 次にやること (報告書 §12 と同じ。優先順)

1. **XPBD の段 0 の方策の崩れの切り分け** (全 4 モデル・2 seed で再現、it 20〜80 で s0 の成功 98% → 30〜65%。結論を変えうる最大の未解決)。
   崩れる前後のチェックポイント (it 40 / 60 / 80) の段 0 の行動と 1 手の結果を比べる → PPO 側の対策 (学習率・エントロピー・
   行動のノイズの範囲) → 直ったら XPBD を学習し直して転移・摂動を回し直す (XPBD は 1 本 30 分)。
2. **seed 43 の評価**: 学習は済み (`logs/rsl_rl/knot_tying_rs_xpbdc{28,56}/*_s43`)。一覧 `outputs/rope_study/eval/jobs_s43.txt` (32 本) を
   `run_eval_queue.sh` で回す (止めた途中の出力は `outputs/rope_study/eval_s43_stopped/`)。`summarize_eval.py` は `*_s43` を扱える。
3. `Art28LowMu` を 200 iteration に延長、`Art28` / `Art28LowMu` に seed 43、摩擦のランダム化の版 (`Art28MuDR`、要実装)。
4. `XpbdC28LowMu` の漂流 (評価の「発散」はすべて紐が原点から 3 m ずれたもの) への対策。台本では 79% 結べる。
5. 曲げ剛性のランダム化、共通の初期アーカイブ、学習曲線の chain 評価、GPU 占有時の速さ、FEM の較正、`Art56` の切り分け (報告書 §12 の 5〜12)。

### 注意 (このセッションで踏んだもの)

* 最後のチェックポイントは `model_99.pt` (100 iteration = 0〜99)。`make_eval_jobs.py --iters 50,99`。
* 前のセッションが起動したプロセスの kill は権限判定で拒否される。ユーザーに PID を伝えて止めてもらう。
* FEM は GPU が混雑していると CUDA illegal memory access で固まり、`timeout` の SIGTERM が効かない (`timeout -k` か `kill -9`)。
* `cd X && nohup ... &` は cd ごとバックグラウンドに行くので、続くコマンドの cwd は変わらない。

## 0000. 4 回目のセッション (2026-10-02 23:40〜) の状況

### 0000.1 分かったこと・したこと

1. **低摩擦の XPBD が床を滑り続けた原因 = XPBD を world 座標で解いていたこと (float32 の丸め)**。小ステップの床の動摩擦の
   補正 μ g h^2 (3.3e-7 m @ μ 0.5、1.3e-6 m @ μ 2.0) が座標の ulp より小さいと消える。4096 環境 (原点 ±94.5 m) では
   μ 2.0 でも約 88% の環境で床・紐どうしの動摩擦が無かった。**修正: `adapters.XpbdRope` を環境の局所座標に** (23:50)。
   単体テスト 2 件追加 (247 件通過、ruff 通過)。環境の中の再現も直った (`logs/diag_lowmu_localfix.log`)。報告書 §7.1.6。
2. 修正前の XPBD の結果 (ゲート 1・物理ベンチ・アーカイブ・v4 方策の XPBD 上の評価) は `outputs/rope_study/xpbd_world_coords/`
   (README あり) と `outputs/archive_seed/rope_study/xpbd_world_coords/` に退避。
3. **修正前の物理で走っている学習 2 本 (`XpbdC28` PID 1586101、`XpbdC56` PID 1655235) は止められなかった** (kill が権限判定で
   拒否された。ユーザーに依頼する)。run のフォルダに `INVALID.txt` を置き、`make_eval_jobs.py` は飛ばす。
   ログは `xpbd_world_coords/logs/train_Xpbd{C28,C56}.log` に移した。
4. FEM: `scene.replicate_physics = False` で起動するようになった (16 環境で 3 手、1 手 47 s、発散なし。`logs/smoke_Fem_v2.log`)。
   前のセッションの固まった FEM (PID 1635316、GPU1 3.5 GB) は残っている (止められない)。
5. 道具: `eval_robust.py --xpbd key=value`、`make_eval_jobs.py` に物理の摂動先 (`XpbdC28EIx0.5` / `EIx2`)・`--perturb_iters`・
   `--noise_mm` (既定 10)・`INVALID.txt` の飛ばし、`summarize_eval.py` に選定の表、`run_eval_loop.sh`、`run_train_chain.sh`。
6. 剛体鎖の学習の尺のゲート 1: `Art28LowMu` 40/128 (発散 4)、`Art56` 3/128 (発散 103)。`Art56` は候補から外した。

### 0000.1b 2026-10-03 10:40 の状況 (セッションが一度切れて再開)

* ユーザーが 1586101 (無効な XpbdC28) と 1635316 (固まった FEM) と古い評価キューを止めた。
* 学習は 6 本とも 100 iteration 完了 (最後のチェックポイントは **`model_99.pt`**。評価は `--iters 50,99`)。
  run: `logs/rsl_rl/knot_tying_rs_<m>/` の最新 (`Art28`/`Art28LowMu` 10-02_21-39、`XpbdC28` 10-03_00-08、`XpbdC56` 00-05、
  `XpbdMS28` 03-44、`XpbdC28LowMu` 05-38)。
* 物理試験 v3 (修正後の XPBD) 完了。報告書 §7.1.3 / §7.1.7 に記入済み。`XpbdC28LowMu` は台本で最も結べる (遅い尺 96%、学習の尺 79%)。
* `XpbdC28LowMu` の方策は chain 評価で固結び 0%、「発散」40〜45% はすべて紐の漂流 (原点から 3 m。`eval_robust.py` に
  `diverged_drift_frac` を追加して確認。`outputs/rope_study/diag/lowmu_eval_breakdown.json`)。
* FEM: 32 環境の gate1 は GPU 混雑時に CUDA illegal memory access で固まった (`fem_crash_32env/`)。空いた GPU では 32 環境で
  3 手動く (1 手 33 s、`logs/smoke_Fem_32.log`)。物理試験を 32 環境で回し直し中 (`logs_suite_fem_gpu0.txt`)。
* 評価ループ 4 本 (`eval/loop_cuda{0,1}_{a,b}.txt`、`--policies <6 モデル> --iters 50,99`)。66 本中 6 本済み (10:23 開始)。
* seed 43 の学習: `XpbdC28` (cuda:0)、`XpbdC56` (cuda:1)、ログ `logs/train_<m>_s43.log`。評価は
  `make_eval_jobs.py --policies "" --extra_policy XpbdC28_s43=<run> --extra_home XpbdC28 --iters 50,99` のように別に作る。
* 報告書: §5・§6.3・§6.4・§7.1.3・§7.1.5〜§7.1.7・§7.2 は更新済み。§7.1.8 (FEM)、§7.3、§7.4、§0、§8〜§13 が残り。

### 0000.1c 2026-10-03 11:20

* **XPBD の 4 モデルはすべて学習の途中 (it 20〜60) で段 0 の方策が崩れる** (まっすぐな紐からの成功 99% → 20〜35%、時間切れ 65%、
  行動のノイズが 0.45 → 0.49 に戻る)。剛体鎖は崩れない。アーカイブの不一致は 0。`XpbdC28` は chain 評価 it50 23.8% → it99 2.1%。
* そこで各方策を「自分のモデルで良い方の iteration」(it50 / it99) で比べる。XPBD の it50 の転移・摂動を
  `eval/jobs_it50.txt` (36 本) に作り、`run_eval_queue.sh` を各 GPU 1 本ずつ (`eval/queue_it50_cuda{0,1}.txt`)。
  `summarize_eval.py` は「最後の iteration」と「良い方の iteration」の 2 通りの表を出す。
* FEM は §7.1.8 に記入済み (保持 0%、把持 55%、台本 3/32)。

### 0000.2 走っているもの (23:58 時点)

| 何 | ログ |
|---|---|
| 物理試験 v3 (修正後の XPBD): GPU0 `XpbdC28 XpbdC28LowMu`、GPU1 `XpbdC56 XpbdMS28` | `outputs/rope_study/logs_suite_v3_gpu{0,1}.txt` |
| 学習のチェーン: GPU0 は PID 1586101 の後に `XpbdC28` → `XpbdMS28`、GPU1 は PID 1655235 の後に `XpbdC56`、PID 1567147 (`Art28LowMu`) の後に `XpbdC28LowMu` | `outputs/rope_study/logs_train_chain_gpu{0,1a,1b}.txt` |
| 学習 `Art28` (GPU0)、`Art28LowMu` (GPU1) | `logs/train_Art28*.log` |
| v4 の方策の評価キュー (GPU0、`jobs_v4.txt` の残り: 観測ノイズ・実行誤差) | `eval/queue_v4_gpu0.txt` |

## 000. 4 回目のセッションへ (2026-10-02 23:05 に 3 回目を終了)

### 000.1 走っているもの (nohup。確認: `nvidia-smi`、`pgrep -af "train --task Isaac-Twisted-RL-RS|eval_robust|overhand_script|bench_"`)

| 何 | GPU | 23:05 の状態 | ログ / 出力 |
|---|---|---|---|
| 学習 `Art28` (seed 42, 100 it) | cuda:0 | it 9、約 390 s/it (残り約 10 h) | `outputs/rope_study/logs/train_Art28.log`、`logs/rsl_rl/knot_tying_rs_art28/*_rs_Art28_s42/` |
| 学習 `Art28LowMu` | cuda:1 | it 9、約 240 s/it | `.../train_Art28LowMu.log`、`logs/rsl_rl/knot_tying_rs_art28lowmu/` |
| 学習 `XpbdC28` (指 v2) | cuda:0 | it 15、約 140 s/it (残り約 3.5 h) | `.../train_XpbdC28.log`、`logs/rsl_rl/knot_tying_rs_xpbdc28/` |
| v4 の `Knot` の方策 (it100) の評価キュー (転移 5 → 自分 → 摂動 4) | cuda:0 | 1/10 済 | `outputs/rope_study/eval/queue_v4_gpu0.txt`、`eval/v4Knot/*.json` |
| ゲート 1 のやり直し (まっすぐな紐から) | cuda:0 | Art28LowMu 学習の尺 → Art56 学習の尺 | `outputs/rope_study/logs_gate1_redo.txt` (スクリプトは `outputs/rope_study/diag_scripts/gate1_redo.sh`) |
| 物理試験 v2 (`run_physics_suite.sh`) | cuda:1 | XpbdMS28 の物理ベンチ → 学習の尺 | `outputs/rope_study/logs_suite_v2_gpu1.txt` |
| 小さいキュー | cuda:1 | 剛体鎖の片持ち梁は書き出し済み (Kit の終了で固まっているかも。timeout 1800 で切れる) → FEM のスモークテスト | `outputs/rope_study/logs_small_queue.txt`、`logs/smoke_Fem.log` |

別セッション (`2f001223-...`) は `eval_queue2.sh` で chain 評価を 2 本/GPU 回している。**止めない**。1 GPU に学習は 2 本まで。

### 000.2 3 回目で分かったこと・したこと (報告書 §7.1.1〜§7.1.5 に記入済み)

1. **学習の尺のゲート 1 は全モデルで無効だった** → 直してやり直し中。`overhand_script.py` は env.reset() を学習の設定のまま使い、
   初期アーカイブのファイルがあると段 0〜2 の途中の形から始まっていた (suite は遅い尺がアーカイブを書いた後に学習の尺を回す)。
   スクリプトを Play と同じ「まっすぐな紐」に固定した。無効な結果は `outputs/rope_study/contaminated_start/` (README あり)。
2. **XPBD の指の接触 v2** (`xpbd_rod.XpbdRodParams`: `finger_rigid_outside`, `finger_x_faces`, `finger_x_face_margin`,
   `finger_max_depenetration_velocity`, `hand_teleport_distance`。`rope_study/models.py` の `XPBD_FINGER_V2` で全 XPBD モデルの既定)。
   原因は compliance の指の中に床の紐が深く残り、押し込みの力 x μ 4 が糊になること。剛の押し戻しは「離れる速さ」1 m/s で頭打ち
   (小ステップごとに足すと速度が積み上がって弾く)、1 ステップ 2 cm 超の手は瞬間移動 (学習の尺の clear = 0 s で 19 m/s で掃いていた)。
   v1 の結果は `outputs/rope_study/finger_v1/` と `outputs/archive_seed/rope_study/finger_v1/`。
3. まっすぐな紐からのゲート 1 (128 環境): 遅い尺 / 学習の尺 = `Art28` 126 / **68** (発散 14)、`XpbdC28` v2 49 / **78** (発散 0)、
   `XpbdC56` v2 32 / **73** (発散 0)、`XpbdC28LowMu` v2 29 (発散 61) / 21 (発散 33)、`XpbdMS28` v2 12 / (実行中)、
   `Art28LowMu` 113 (発散 4) / (実行中)、`Art56` 0 (発散 45) / (実行待ち)。**学習の尺では XPBD v2 が剛体鎖より結べて発散しない。**
4. 物理ベンチ (random = 学習と同じ行動の発散率/手): Art28 10.2%、Art28LowMu 3.0%、Art56 58.7% (反復 16: 33%、32: 36% → 反復では直らない)、
   XPBD 0% (LowMu 1.8%)。drag で同じ μ 2.0 でも反対の端の移動が剛体鎖 0.32 m / XPBD 0.13 m (摩擦の実装で挙動が違う)。
5. 片持ち梁: 剛体鎖も XPBD も同じ区間長なら同じ誤差 (4 cm で実効 EI 0.43 倍、2 cm で 0.62 倍)。`elastica` の射撃法のバグを直した。
6. `eval_robust.py` は動く (v4 の方策 it100 を `Art28` で 64 環境: 42.2% ± 6.2、発散 10.9%)。v4 の方策を `Art28LowMu` で: 24.6%。
7. 道具を足した: `rope_study/make_eval_jobs.py` (評価ジョブ一覧)、`run_eval_queue.sh` (1 本ずつ、出力済みは飛ばす。失敗したジョブは
   `.log` を消すと再実行)、`summarize_eval.py` (自分のモデル・転移行列・摂動の表)。テスト 4 件追加 (計 245 件通過)。

### 000.3 未解決 (優先順)

1. **低摩擦の XPBD が環境の中で床を滑り続ける** (`XpbdC28LowMu` の「発散」は全部「3 m 以上離れた」判定)。
   再現: `ROPE_DEBUG_ENV=31 .venv/bin/python -u .../demos/overhand_script.py --task Isaac-Twisted-RL-RS-XpbdC28LowMu-v4 --num_envs 64 --jitter 0.25 --moves 2 --headless --device cuda:0 --out outputs/rope_study/diag/x`
   (決定的。env 31 が 1 手目の clear / retreat の後から約 0.26 m/s の一定の速さで +y へ滑り続け、全節点 z = 0.010、手は遠い。
   ログ: `outputs/rope_study/logs/diag_lowmu_div4.log`)。XPBD 単体 (`outputs/rope_study/diag_scripts/diag_drift.py`、`diag_loop.py`。scratchpad から写した。`diag_finger.py` を import するので同じフォルダで実行) では起きない
   (初速 1 m/s は理論どおり 0.10 m で止まる)。疑い: 床の摩擦の位置補正が効かない状態 (節点が床からわずかに浮いて pen ≤ 0?
   `x_prev` の扱い?)、環境の中だけの違い (PhysX から読む手・指、アダプタ)。次の手: 環境の中で env 31 の `rod.x / x_prev / v`
   を 1 手目の後に保存し、XPBD 単体で再生して切り分ける。μ 2.0 のモデルでも小さく起きているかを drag / ゲート 1 の紐の重心の動きで確認。
   直したら XPBD 4 モデルの物理試験をやり直す (ゲート 1 と物理ベンチ。出力は退避してから)。
2. `XpbdC28` の学習の初期アーカイブは無効な (途中の形から始まった) 台本の結果 (`archive_seed/rope_study/XpbdC28_trainseed.pt` と同一)。
   学習中に段のアーカイブは更新されるので影響は小さいと判断して続けている (報告書に記載)。まっすぐからの版は `XpbdC28_straight.pt`。
3. FEM は未起動 (小さいキューの最後)。Art56 は反復を上げても発散が多い (学習の候補から外す理由として報告書に書く)。
4. 報告書 §0・§8・§9・§10・§12 は 18:50 時点の記述が残っている (§0 冒頭に注意書きあり)。結果に合わせて書き直す。

### 000.4 次にやること

1. 上の未解決 1 (低摩擦の XPBD の滑り) を切り分けて直す。
2. 学習: `XpbdC56` (アーカイブはまっすぐからの v2: `archive_seed/rope_study/XpbdC56.pt`) を GPU が空き次第。次に `XpbdMS28`、
   `XpbdC28LowMu` (1 を直してから)。コマンドは §5 の 4。
3. 学習が it50 / it100 に達したら評価: `make_eval_jobs.py --policies Art28,Art28LowMu,XpbdC28,... --iters 50,100` →
   `run_eval_queue.sh <jobs> cuda:N` (2 GPU で同じ一覧を同時に回してよい) → `summarize_eval.py --md outputs/rope_study/eval/summary.md`。
   1 本 15〜25 分 (GPU 共有時)。
4. 報告書を仕上げる (§0・§7.3・§7.4・§8・§9・§10・§12・§13 に make_eval_jobs / run_eval_queue / summarize_eval を追記)。

### 000.5 注意 (3 回目で踏んだもの)

* `pkill -f <パターン>` はパターンを含む自分のシェルまで殺す (コマンドが途中で止まった)。PID を確かめてから `kill <PID>` を使う。
  別のセッション由来のプロセスを kill しようとすると権限判定で拒否され、その後 Bash 全体が拒否されたことがある。
* Kit (Isaac Sim) は終了時に固まることがある (generate_assets、bench_cantilever_art)。出力は書けているので、`timeout` で切れるのを待つ。
* `overhand_script.py` の切り分け用の出力: `ROPE_DEBUG_DIV=1` (発散の初回の様子、3 m/s を超えた瞬間 `[kick]`)、`ROPE_DEBUG_ENV=<env>`
  (フェーズごとの紐の重心・手 `[trace]`)。既定では何もしない。発散の判定の内訳 (`reason={nan, far3m, speed20}`) は常に出る。
* 長い待ちは `sleep` ではなく Bash の `run_in_background` + `until` ループか Monitor。
* GPU を共有しているので時間 (s/it、評価の所要) は参考値。

## 00. 3 回目のセッション (2026-10-02 21:37〜、Bash が戻った後) の状況 ← 最新

### 00.1 実行中のもの (nohup。セッションが終わっても走る)

| 何 | GPU | ログ | 出力 |
|---|---|---|---|
| 学習 `Art28` (21:40〜、100 it、約 480 s/it) | cuda:0 | `outputs/rope_study/logs/train_Art28.log` | `logs/rsl_rl/knot_tying_rs_art28/*_rs_Art28_s42/` |
| 学習 `Art28LowMu` (21:40〜、約 530 s/it) | cuda:1 | `.../train_Art28LowMu.log` | `logs/rsl_rl/knot_tying_rs_art28lowmu/` |
| 学習 `XpbdC28` (22:05〜、指 v2) | cuda:0 | `.../train_XpbdC28.log` | `logs/rsl_rl/knot_tying_rs_xpbdc28/` |
| 小さい試験のキュー (Art28 の物理ベンチ再実行 → Art56 反復 16/32 → 剛体鎖の片持ち梁 → FEM のスモークテスト) | cuda:1 | `outputs/rope_study/logs_small_queue.txt` | `physics/Art28`, `physics/Art56_it{16,32}`, `cantilever/art.json`, `logs/smoke_Fem.log` |
| 物理試験 v2 (XPBD 4 モデル、指 v2) | cuda:1 | `outputs/rope_study/logs_suite_v2_gpu1.txt` | `gate1/`, `gate1_fast/`, `physics/`, `archive_seed/rope_study/<M>.pt` |

キューのスクリプト: `/tmp/claude-1002/-home-itoyama-work-isaac-twisted-rl/7e308401-.../scratchpad/small_queue.sh`。
別セッションは v7 の学習を終え、`eval_queue2.sh` (chain 評価 2 本/GPU) を回している。止めない。

### 00.2 このセッションでしたこと

* pytest 241 件・ruff は通過 (Bash が戻った直後)。その後テストを 4 件足した (指 v2 の 3 件 + elastica)。
* 物理試験のキュー (17:51〜19:43) は完走していた。全 7 モデルの結果を報告書 §7.1.2 / §7.1.3 に入れた。
  `Art28` の物理ベンチは self_penetration の修正後に再実行して完走 (random で 1 手あたり 10.2% 発散)。
* **XPBD の指の接触を直した (v2)**。原因は「compliance の指の中に紐が深く残り、押し込みの力 x μ 4 が糊になる」。
  `xpbd_rod.XpbdRodParams` に `finger_rigid_outside` / `finger_x_faces` / `finger_x_face_margin` /
  `finger_max_depenetration_velocity` / `hand_teleport_distance` を追加 (既定は v1 のまま)。`rope_study/models.py` の
  `XPBD_FINGER_V2` で全 XPBD モデルの既定を v2 にした。ゲート 1 (遅い尺) 3 → 16/128、2 手目の輪 24 → 89/128。報告書 §7.1.4。
  v1 の結果は `outputs/rope_study/finger_v1/` と `outputs/archive_seed/rope_study/finger_v1/` に退避。
  切り分けの再現スクリプトは scratchpad の `diag_finger.py` (Warp だけ。台本の 1 手目と吊り下げ)。
* `elastica.heavy_cantilever` の射撃法を直した (大たわみで θ が水平より上や真下を越えて振り子のように戻り、偽の根に
  落ちていた)。`cantilever/xpbd.json` を再計算中 (修正前は `xpbd_before_elastica_fix.json`)。

### 00.3 次にやること

1. 小さいキューと v2 の物理試験の結果を読み、報告書に入れる (Art56 の反復、剛体鎖の片持ち梁、FEM)。
2. v2 の物理試験が終わったモデル (XpbdC28LowMu → XpbdC56 → XpbdMS28) の学習を、GPU が空き次第 (1 GPU に学習 2 本まで)。
3. 学習が 50 / 100 iteration に達したら `eval_robust.py` で自分のモデル・転移・摂動 (§5 の 5)。
4. 報告書 §0・§10 の書き直し。

## 0. 2 回目のセッションの状況 (次のセッションはここから読む)

### 0.1 Bash が使えなくなった経緯

* 前のセッションが 16:49 に起動した `generate_assets.py --rope knot56 --only rope --headless` (PID 1396214、親の
  `timeout 600` は PID 1396210) が、`rope_knot56.usd` を書いた後に Kit の終了で固まり、`timeout` の SIGTERM でも
  終わらずに残っている (GPU0 で約 340 MB、GPU1 で約 280 MB)。`rope_knot56.usd` は 16:49 に書かれた正常な USD crate。
* これを `kill -9` しようとしたところ自動の権限判定に「他の作業への干渉」として拒否され、**その後は `date` を含む
  すべての Bash が拒否された**。ジョブの起動・テスト・`ls` ができないので、このセッションでは Read / Edit / Write
  だけで、実行中の物理試験の出力の読み取りと文書化をした。
* **更新 (18:50 過ぎ)**: 1 回目のセッション (まだ生きていた) から連絡があり、固まったプロセス (PID 1396214 / 1396210) は
  向こうで止めた。2 つとも向こうが 16:49 に起動したもので、`rope_knot56.usd` (16:49:55, 7839 B) は無傷とのこと
  (こちらは Bash が使えないので `ps` で確かめていない)。
  向こうが仕掛けていた「GPU1 のキューが終わったら `Art28` の物理ベンチを自動で再実行する」ジョブは二重実行を避けるため
  取り消された → **`Art28` の物理ベンチの再実行はこちら (次のセッション) の担当**。
  `centerline.self_penetration` の NaN 対策はこちらの修正だけが入っている。向こうは今後 `rope_study/` と
  `common/rope_backends/` を編集しない。
* **ユーザーに頼むこと**: Bash の権限を戻してもらう (他のセッションからは戻せない)。

### 0.2 このセッションで分かったこと (報告書に反映済み)

* 物理試験のキュー (§4) は動いている。18:50 時点で: GPU0 は `XpbdC28` のゲート 1 (学習の尺) を実行中、
  GPU1 は `Art28` のゲート 1 (学習の尺) を実行中。
* **ゲート 1 (遅い尺)**: `Art28` 126/128 (98%)、発散 0、アーカイブ 段 0/1/2 = 128/258/130。
  `XpbdC28` **3/128 (2%)**、発散 0、最後の最高点 25.4 cm、アーカイブ 545/55/26。
  `XpbdC28` は 2 手目 (輪) で 24/128 しか輪ができない。1 手目の後に床に落ちていたのは 21/128 だけ。
  `move1_slack.png` を見ると、`Art28` は根元側が元の位置に残るのに、XPBD は紐のほぼ全体が付いてくる。
* **ゲート 1 (学習の尺 3.16 s/手)**: `XpbdC28` **36/128 (28%)**、発散 0 (2 手目で 84/128 が輪)。
  `Art28` は 5 手目の後で 49/128、**発散 13/128** (最終の行は 18:50 時点で未出力)。
  → 学習に使う尺では差が小さい。XPBD が悪いのは遅い尺 (指が紐の近くにいる時間が長い) だけ。
  2026-09-27 の `knot` (反復 16) の学習の尺 91% と比べ `Art28` (v4、反復 8) は大きく悪い。
  XPBD の初期アーカイブは遅い尺の台本から作ったので段 1・2 が少ない → 学習の前に `--fast --dump_archive` で作り直す案。
* **XPBD の指の仮説 (未検証)**: `xpbd_rod._box_contact` は挟む面 (y) と底面 (z) しか接触しない。指の前後 (x) の面と
  上面には接触が無いので、手が x 方向に動くと紐が指の直方体の中へ入れる。中に入った標本点は y に最大 16 mm の
  めり込みとして押され、μ = 4.0 の位置ベースの摩擦で x・z に固定されて、手と一緒に持ち上がる。
  最後の最高点 25.4 cm = 手のひら 0.30 m − 指の中心 0.045 m と一致する。
  確かめ方: `XpbdC28LowMu` のゲート 1 が大きく良くなるか、指の全ての面と接触させて (角を丸めた SDF) ゲート 1 を
  やり直す (ただし底の角が垂れた紐を押し出す問題 (§3.4 の 6) が再発しないかを hang で確かめる)。
  **キューに XPBD のモデルが残っている間は `xpbd_rod.py` を変えない** (未テストのカーネルが壊れると残りが全部止まる)。
* **物理ベンチ**: `XpbdC28` は完走 (発散 0、めり込み ≈ 0、hang 保持 100%、drag で全リンクが動く、knot で結び目が
  残るのは 30%、random の把持 92%)。**`Art28` は hang で落ちた**: 少なくとも 1 環境が NaN (発散) になり、
  `centerline.self_penetration` の区間長の平均が NaN → `ValueError`。
  → `self_penetration` を NaN に強く直した (有限の区間だけで平均。18:45)。有限の入力では結果は同じ。
  `bench_physics.py` の hang に `diverged` を足した。**`Art28` の物理ベンチは再実行が要る** (`run_physics_suite.sh` を
  もう一度回せば、`shapes.pt` が無い `Art28` だけやり直す)。`Art28LowMu` / `Art56` の物理ベンチは修正後のコードで回るはず。
* **片持ち梁の注意**: `xpbd.json` の自由長 16 cm の `effective_EI_ratio` が 4 行とも 0.03688767 で同じ
  → `elastica.effective_ei` (か `heavy_cantilever`) が大たわみで解けていない。報告書では 8 cm だけを使った。
  区間 0.5 cm の Cosserat は離散鎖の解析解より 25% 柔らかい (小ステップ 16 x 反復 1 の収束不足と解釈)。
  質点-ばねのばね定数 785 N/m は区間 4 cm 用の固定値 (`spring_bend_stiffness`) なので、4 cm 以外の行は意味が無い。

### 0.3 このセッションで書いた・変えたファイル

* `rope_study/bench_cantilever_art.py` (新規、**未実行・未 lint**): 剛体鎖の片持ち梁。`knot` 4 cm x 自由リンク 2/4、
  `knot56` 2 cm x 4/8 を、位置反復 8/16/32 で 1 つのステージに並べ、根元を `fix_root_link=True` で固定、接触なし、
  3 s 後の先端の下がりを elastica と比べる。USD は `outputs/rope_study/cantilever/usd/` に作る。
  初回の実行で落ちたら、`fix_root_link` と USD の root (Link00 に ArticulationRootAPI) の組み合わせと、
  `find_bodies` の名前を疑う。
* `rope_study/README.md` (新規): 使い方。
* `common/rope_backends/centerline.py`: `self_penetration` を NaN に強く (上記)。
* `rope_study/bench_physics.py`: hang に `diverged`。
* `rope_study/bench_cantilever_xpbd.py`, `rope_study/elastica.py`: docstring の参照先を `bench_cantilever_art.py` に。
* `Report_rope_models.md`: §0 (中間の要旨)、§5〜§13、参考文献を記入。§2.3 の 1 文を正確に。
* テスト (`pytest`) と lint (`ruff`) は **このセッションでは回せていない**。最初に回すこと。

### 0.4 次のセッションの最初にやること

1. `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/` と
   `uvx ruff@0.14.10 check --line-length 120 src/isaac_twisted_rl/tasks/knot_tying/rope_study src/isaac_twisted_rl/tasks/knot_tying/common/rope_backends`。
2. `ps` / `nvidia-smi` で、固まっていた generate_assets.py が消えていることと、キュー 2 本の状態を確かめる。
   キューの残りの結果を読む (`outputs/rope_study/logs_suite_gpu{0,1}.txt`)。GPU1 のキューが終わっていたら、
   `run_physics_suite.sh cuda:1 "Art28" 128` で `Art28` の物理ベンチだけやり直す (自動の再実行は仕掛けられていない)。
3. 報告書 §7.1.2 / §7.1.3 の「実行待ち」を埋める。
4. あとは下の §5 の順 (ただし XPBD の指の修正を、XPBD の学習より前に入れる)。

## 1. 依頼の要約 (元の依頼の要件。すべて満たすこと)

紐結び強化学習の紐のシミュレーションモデルを体系的に調査し、複数の紐モデルを実装して、それぞれで強化学習し、
**実機に適応したときに最も良い結果が期待できるモデル** を実験的に選ぶ。

* Isaac Lab / Isaac Sim を第一候補にする。現行の紐モデルが不十分なら、Isaac Lab 内の別の表現や外部手法との連携も検討する。
* 文献 (DER・mass-spring・Cosserat・FEM・PBD・粒子・剛体リンク近似など) を、曲げ・伸び・ねじれ・摩擦・自己接触・他物体との接触・
  把持・数値安定性・並列化・RL との相性・実機移行性で比較する。
* 候補を複数実装し、公平に比べられるように設定・観測・報酬・行動・接触判定・リセット・評価の変更を整理する。
  seed・評価条件・学習時間・計算資源・失敗条件を記録する。
* RL は報酬や sim の成功率だけで判断しない。成功率と再現性、物理の妥当性、接触・摩擦・自己接触の安定性、発散・めり込み・
  不自然な伸び・過剰な振動、学習の安定性と必要な学習量、方策の頑健性、パラメータ変動・初期状態変動への耐性、実機との挙動差、
  必要なランダム化・キャリブレーション、Isaac Lab 上での保守性・拡張性・計算コスト を比較する。
* 実機評価ができない場合は、実機適応の成否を予測する sim 上の検証を提案する。sim で最良のモデルをそのまま結論にしない。
  実機で有利/不利になる可能性を明確に説明する。
* 報告書に入れるもの: 現行モデルの評価 / 文献の整理 / 候補 / 実装した変更 / 実験条件と比較方法 / RL 結果の比較 /
  物理的妥当性の比較 / 実機適応性の比較 / 未解決の問題と限界 / 最有力モデルと選定理由 / 今後の実機検証計画 /
  追加で必要な実験 / 再現可能な実行手順。**最終結果はマークダウンファイルで、適切なディレクトリに適切な名前で** (→ `Report_rope_models.md`)。
* 候補を早く絞りすぎない。結果が不十分なら断定しない。既存コードを壊さない (変更を分離)。成果物・設定・ログ・チェックポイント・評価結果を整理する。
* 質問があれば最初に列挙し、答えが無くても合理的な仮定で進める (前のセッションで列挙済み。報告書 §1.1)。

## 2. 環境と制約 (必ず守る)

* **GPU 2 枚 (RTX 3090) は別セッションと共有。** 別セッション (scratchpad `.../2f001223-.../`) が v6/v7 の切り分け学習 4 本
  (`logs/rsl_rl/knot_tying_v6/*_abl_{relA,depB,sw2}_s42`, `knot_tying_v7/*_abl_toP_s42`, 101 iteration, 12:33 開始, 約 460 s/iter)
  を回しており、終わると `pipeline2.sh` が chain 評価を 2 本/GPU で流す (2026-10-03 01:30〜03:00 頃に終わる見込み)。
  **それらのプロセスを止めない。** 自分の重い学習は GPU が空いてから。起動前に `nvidia-smi` で空きメモリを見る
  (4096 環境の学習 1 本 ≈ 7.8 GB、128 環境の試験 ≈ 2.5 GB)。
* 別セッションの未コミットの作業 (`versions/v7/`, `versions/v6/Result_Report_v6.md`, `scripts/rsl_rl/eval_chain.py`,
  `knot_tying/{README.md,__init__.py}`, `common/mdp/{__init__,stages}.py`) には触らない。
* `common/rope_model/data/rope_knot.usd` と `outputs/archive_seed/overhand_28links.pt` (v4〜v7 が使う) を上書きしない。
  紐の USD を作り直すときは `--rope knot56` のように対象を限定する (`generate_assets.py --rope <名前> --only rope`)。
* rsl-rl 3.0.1 / Isaac Lab 2.3.0 / Isaac Sim 5.1 / Warp 1.10.1 (`.venv`)。`/home/itoyama/work/IsaacLab` は別物 (使わない)。
* Isaac Sim を起動するスクリプトは `python -u` で。`pxr` は Kit 起動後にしか import できない。
* テスト: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/` (中断時点で 241 件すべて通過)。
  lint: `uvx ruff@0.14.10 check --line-length 120 <path>` (新規ファイルは通過済み)。
* GPU を別の学習と共有していると、小さい環境数でも 1 物理ステップが 50 ms 前後になり、時間の計測は当てにならない。
  速度の比較は GPU が空いてから測り直す。

## 3. ここまでにしたこと

### 3.1 調査 (報告書 §2・§3 に下書き済み)

* 現行の紐 (`Knot` = 本研究の `Art28`) の構造・パラメータ・既知の失敗を整理し、「物理的に不十分 + 学習・実機適応で問題あり」と判断。
  主な問題: 摩擦 2.0 / 指 4.0 が非現実的、4 cm リンクが自重曲げの特性長 6.8 cm を分解できない、発散とその数値的な手当てへの依存、
  ソルバ設定への非単調な依存、遅い (87 手/s)。
* 文献の表 (§3.1)・シミュレータ (§3.2)・紐の RL と sim-to-real (§3.3)・物性値 (§3.4)・較正 (§3.5)。
  **参考文献の URL の節がまだ無い** → 報告書の最後に下の §7 のリストを入れること。

### 3.2 実装 (すべて追加。既定値のままなら v0〜v7 の挙動は変わらない)

| ファイル | 内容 |
|---|---|
| `common/rope_backends/__init__.py` | パッケージの説明 (docstring だけ。Kit 起動前に import されるため) |
| `common/rope_backends/centerline.py` | 純 torch: 弧長での再標本化、仮想リンク (中心線を弧長で 28 等分した区間の中点)、捩れ 0 の枠 (平行移動)、めり込み・伸びの計測 |
| `common/rope_backends/xpbd_rod.py` | Warp の XPBD ソルバ (isaaclab 非依存)。`model="cosserat"` (Kugelstadt & Schömer 型、伸び・せん断・曲げ・ねじれ) と `"mass_spring"` (FleX 型)。床・指 (compliance 付き)・自己接触、位置ベースの Coulomb 摩擦、small steps (既定 16 小ステップ/物理ステップ)、1 物理ステップ全体を CUDA graph にして Warp 専用 stream で流す |
| `common/rope_backends/adapters.py` | `scene["rope"]` を Articulation と同じ読み書き口に見せるアダプタ。`ResampledArticulationRope` (56 リンク鎖を仮想 28 リンクで見せる。関節位置はリンクの向きから正確に出す)、`XpbdRope` (PhysX のグリッパーの root 姿勢と指の関節位置を毎ステップ読み、XPBD を 1 ステップ進める。GUI ではマーカーで描画)、`FemRope` (FEM の頂点を軸方向の区画で平均した中心線。**未テスト**) |
| `common/rope_backends/backend_cfg.py` | `RopeBackendCfg` (kind: articulation / articulation_resampled / xpbd / fem) |
| `common/rope_backends/env.py` | `RopeStudyEnv`: シーン生成の直後に (`InteractiveScene` を `__init__` の間だけ包んで) アダプタを `scene._extras["rope"]` に入れ、`scene.update` / `scene.reset` にアダプタの処理を挟む。`load_managers` の直前に `adapter.bind(env)` |
| `common/rope_backends/noisy_action.py` | `NoisyPickPlaceAction(Cfg)`: 掴む点・置く点の xy と手首角の実行誤差 (既定 0)。頑健性評価用 |
| `common/rope_model/rope_specs.py` | `RopeSpec.filter_neighbor_pairs` (既定 1) と `knot56` プリセット (`_refine_chain`: 同じ材料・太さ・全長でリンク 2 cm、56 本、1 本おきの接触を除外) を追加 |
| `common/rope_model/generate_rope_usd.py` | `filter_neighbor_pairs > 1` のときだけ `UsdPhysics.FilteredPairsAPI` を付ける。`rope_knot56.usd` は生成済み (`rope_knot.usd` は 9/27 のまま) |
| `demos/overhand_script.py` | `--dump_knots PATH` を追加 (最後に固結びになった形を段 3 だけのアーカイブとして保存) |
| `rope_study/__init__.py` | タスク登録 `Isaac-Twisted-RL-RS-<モデル>[-Play]-v4` (entry point は `RopeStudyEnv`) |
| `rope_study/models.py` | モデルの一覧 (純 dataclass): `Art28`, `Art28LowMu`, `Art56`, `XpbdC28`, `XpbdC56`, `XpbdC28LowMu`, `XpbdMS28`, `Fem` |
| `rope_study/env_cfg.py` | v4 の設定を継承し、モデルごとに紐・摩擦・グリッパー (指のゲインは全モデル Knot と同じ 1507 N/m)・行動 (`NoisyPickPlaceActionCfg`)・初期アーカイブ (`outputs/archive_seed/rope_study/<モデル>.pt`) を埋める |
| `rope_study/agents/rsl_rl_ppo_cfg.py` | v4 と同一の PPO。ログは `logs/rsl_rl/knot_tying_rs_<モデル小文字>/` |
| `rope_study/bench_physics.py` | 全モデル共通の物理ベンチマーク (settle / hang / drag / knot / random)。出力 `outputs/rope_study/physics/<モデル>/{summary.json,shapes.pt}` |
| `rope_study/run_physics_suite.sh` | モデルごとに ゲート 1 の台本 (遅い尺、`--dump_archive`, `--dump_knots`) → bench_physics → ゲート 1 (学習の尺 `--fast`) を順に回す |
| `rope_study/elastica.py` | heavy elastica (重い片持ち梁) の数値解と、離散鎖の小たわみの解析解 |
| `rope_study/bench_cantilever_xpbd.py` | XPBD の片持ち梁の試験 (Isaac Sim 不要)。出力 `outputs/rope_study/cantilever/xpbd.json` |
| `rope_study/eval_robust.py` | chain 評価 (各環境の最初の 1 エピソード) を別モデル・観測ノイズ (`--obs_noise`、紐の位置 84 次元だけ)・Hydra の上書き (実行誤差 `env.actions.pick_place.exec_noise_xy=...`、摩擦など) の下で回す |
| `tests/test_rope_backends.py` | centerline と XPBD (Warp CPU) の単体テスト 10 件 |

### 3.3 確認済みのこと

* スモークテスト (16 環境): `XpbdC28` / `Art56` / `Art28LowMu` が学習環境の中で動く (観測 133 次元、仮想 28 リンク、段の遷移・終了が働く)。
* `bench_physics.py` は `XpbdC28` (32 環境) で最後まで動く (settle: 残留速度 0・伸び 1e-5、drag: 紐全体が付いてくる、random: 発散 0 / 把持 98%)。
* **`Art56` は v4 と同じソルバ反復 8 だと 2 手目で 16 中 7 環境が発散した** → 反復を振る必要あり (下の §5)。
* XPBD の単体 (スタンドアロン) の速度: 4096 環境で Cosserat 28 区間 953 手/s、質点-ばね 1274 手/s、Cosserat 56 区間 464 手/s
  (GPU を別の学習 2 本と共有した状態。PhysX の剛体鎖は単独 GPU で 87 手/s)。環境に入れると PhysX のグリッパーと MDP の処理が足される (未計測)。
* 片持ち梁 (XPBD、自由長 8 cm): 区間 4 / 2 / 1 cm で先端の下がり 32.1 / 23.7 / 20.5 mm、連続体 15.6 mm、同じ離散鎖の解析解 36.2 / 25.1 / 20.3 mm。
  **XPBD は離散鎖の解析解と一致 (実装は正しい)。4 cm の離散化は連続体の約 2 倍たわむ (実効 EI ≈ 0.43 倍)。**

### 3.4 XPBD のデバッグで分かったこと (報告書 §5・§9 に書く)

1. 固定節点 (逆質量 0) の区間で割り算が 0 になって NaN → ガード済み。
2. 角速度の減衰だけでは振動が止まらない (振動は節点の速度が持つ) → 節点速度の 2 階差分を減らす運動量保存の曲げ減衰 (`_k_damp_bend_vel`、逆質量で重み付け) を追加。
3. それでも片持ち梁が 3 s 以上振れ続けた → 区間の回転の自由度に溜まる振動。`angular_damping=200 /s` (角速度そのものの減衰) で 1 s 以内に収まる。準静的な課題なので許容と判断 (数値的な減衰であることを報告書に明記する)。
4. compliance 付きの拘束で λ を反復をまたいで積み上げていなかった (PBD になり、剛性が反復回数・小ステップ数で変わった) → λ を積み上げる XPBD に修正。修正後は反復・小ステップによらず同じ値に収束。
5. 指の摩擦の補正を標本点ぶん足し合わせると過補正で振動して滑った → 法線は足し合わせ (力)、摩擦は節点ごとに平均。
6. compliance の指は紐に約 8 mm めり込むので、指の下の角が垂れた紐を下へ押し出し、吊った紐が抜け落ちた → 指の接触を「挟む面 (y) と底面 (z)」だけに限定。指の接触は自己接触の後に解く。保持中のクリープは 1.2 s で約 1 mm まで減った (ゼロではない)。

## 4. 実行中のジョブ (セッションが終わっても nohup で走り続ける)

* GPU0: `run_physics_suite.sh cuda:0 "XpbdC28 XpbdC28LowMu XpbdMS28 XpbdC56" 128` (17:51 開始)
* GPU1: `run_physics_suite.sh cuda:1 "Art28 Art28LowMu Art56" 128` (17:51 開始)
* 進み具合: `outputs/rope_study/logs_suite_gpu{0,1}.txt`。各ステップのログ: `outputs/rope_study/logs/{gate1,physics,gate1_fast}_<モデル>.log`
* 出力: `outputs/rope_study/gate1/<モデル>/summary.log` (`[final] ... overhand=N/128`)、`outputs/rope_study/physics/<モデル>/summary.json`、
  `outputs/rope_study/gate1_fast/<モデル>/summary.log`、初期アーカイブ `outputs/archive_seed/rope_study/<モデル>.pt` と `<モデル>_knots.pt`
* GPU を共有しているので遅い (1 モデルあたり 30〜60 分の見込み)。スクリプトは出力済みのステップを飛ばすので、止めても再実行で続きから回る。
* 注意: 実行中のキューは各ステップで新しい Python を起動するので、**`xpbd_rod.py` などを書き換えると、まだ回っていないモデルにはその変更が効く**。
  ソルバを変えたら、どの結果が変更前/後かを記録すること。

## 5. 次にやること (優先順)

1. **物理試験の結果を読む** (上の出力)。ゲート 1 (台本) の固結びの成功率・発散、bench の各試験、学習の尺での台本。
   `Art56` が発散するなら、ソルバ反復を振って再試験:
   `bench_physics.py --model Art56 --cfg "scene.rope_native.spawn.articulation_props.solver_position_iteration_count=16;scene.gripper.spawn.articulation_props.solver_position_iteration_count=16" --label Art56_it16`
   (反復 16 / 32。学習でも同じ値を `env.scene.rope_native.spawn.articulation_props.solver_position_iteration_count=...` で渡す)。
   bench の hang 試験の「保持」の判定 (`held`) は、指先から −2〜+8 cm に掴んだ点があるか。XPBD で 0% だったのは指の接触の修正前。修正後の値を確認すること。
1.5 **XPBD の指の接触を直す** (§0.2 の仮説。キューの XPBD のモデルが終わってから): `_box_contact` に指の前後 (x) の面と
   上面を足す (角を丸めた箱の SDF で法線を連続に)。hang (垂れた紐が抜け落ちないか) とゲート 1 で確かめ、XPBD の
   ゲート 1 と物理ベンチをやり直す (出力のフォルダを分けるか、修正前の結果を `*_v1` などに退避して記録を残す)。
   指の摩擦 4.0 は PhysX の値に合わせたものだが、位置ベースの摩擦では捕まえすぎる可能性もある (`XpbdC28LowMu` と比べる)。
2. **FEM の実行可能性を確かめる**: ``rope_study/smoke_test.py` で `Isaac-Twisted-RL-RS-Fem-v4` を 16 環境で起動 (未テスト。`FemRope` と `_fem_cfg` は書いただけ)。
   起動・把持・持ち上げ・交差・発散・速度を見る。曲げはロックするので、片持ち梁で `youngs_modulus` を較正する (`models.py` の `Fem` の `fem` dict)。
   動かなければ理由を報告書に書いて候補から外す。
3. **剛体鎖の片持ち梁** (Kit が要る): 短い鎖 (4 cm リンク x 3 / 5、2 cm リンク x 5 / 9) の USD を `generate_rope_usd.build_stage(spec)` で
   scratchpad に作り、`fix_root_link=True` で吊って先端の下がりを `elastica.py` と比べる (XPBD の表 §3.3 と並べる)。
4. **強化学習** (GPU が空いてから。各 4096 環境、seed 42、100 iteration、`agent.save_interval=10`):
   ```bash
   uv run train --task Isaac-Twisted-RL-RS-<モデル>-v4 --headless --device cuda:0 --seed 42 --max_iterations 100 \
       --run_name rs_<モデル>_s42 agent.device=cuda:0 agent.save_interval=10
   ```
   * 物理試験で台本の固結びが極端に低い・発散が多いモデルは、理由を書いたうえで後回し。
   * `Art28` は v4 の `Knot` と同じ環境なので、既存の v4 の run (`logs/rsl_rl/knot_tying_v4/2026-10-01_19-22-*_overnight_s42`、
     chain 評価 it50 39.3% / it100 41.6%) を参照値に使える。ただし同じ条件 (同じコード・初期アーカイブの作り方) で 1 本回すのが望ましい。
   * 見積り: 剛体鎖 ≈ 200 s/iter (単独 GPU)、56 リンクは 2 倍以上、XPBD は数倍速いはず (未計測)。1 GPU に 2 本まで (相乗りで +12%)。
   * 余裕があれば上位モデルに seed 43、摩擦などのドメインランダム化版 (例: XPBD の摩擦を 0.3〜2.0 で一様に) も。
5. **評価** (`eval_robust.py`、512 環境、seed 7、決定論的方策、iteration 50 と 100):
   * 自分のモデルでの chain 評価 (固結び率・発散・時間切れ・段 2 以上に届いた割合・手数)。
   * **モデル間の転移行列**: 各方策を全モデルの `-Play-v4` で評価 (観測・行動は仮想 28 リンクで共通なのでそのまま動く)。
   * 摂動: 摩擦 (XPBD は `env.rope_backend.xpbd_params.friction_ground=...` など、剛体鎖は `LowMu` モデル)、
     観測ノイズ `--obs_noise 0.005 / 0.01`、実行誤差 `env.actions.pick_place.exec_noise_xy=0.005 / 0.01`、XPBD の曲げ剛性 x0.5 / x2。
   * 学習曲線: 10 iteration ごとのチェックポイントを chain 評価 (学習ログの `chain_est` は本番の成功率を表さない。v6 レポート §4.5)。
6. **報告書を仕上げる** (`Report_rope_models.md`): 2 回目のセッションで全節を中間版として記入済み。結果が出たら
   §7.1.1 (剛体鎖の片持ち梁)、§7.1.2 / §7.1.3 の「実行待ち」、§7.2 (速さを GPU が空いてから)、§7.3、§7.4 を埋め、
   **§0 と §10 を書き直す** (選定は「他のモデル・摂動の下での成功率の平均と最悪値」で。差が誤差より小さければ断定しない)。
   `rope_study/README.md` (使い方) は作成済み。

## 6. 既知の注意点

* `Art28` はアダプタを通さない (v4 と完全に同じ)。他のモデルの観測は仮想リンク。位相は仮想リンクの 29 節点の折れ線で判定する
  (表現本来の細かい中心線ではない)。実機でも同じ本数に再標本化するという前提。差 (位相の判定の食い違い) は未計測。
* XPBD の紐は USD に存在しない (GUI では球のマーカー)。PhysX のグリッパーは紐と接触しないので指は全閉まで閉じる。XPBD は
  PhysX の指の関節位置を「指令」として読み、指を compliance (1507 N/m を標本点に配分) で紐に押し付ける。
* `FemRope.state_dim` は `native.max_sim_vertices_per_body` に依存 (アーカイブの容量は FEM だけ 4096)。
* `bench_physics.py` の knot 試験は `<モデル>_knots.pt` (ゲート 1 の台本の最後の固結び) が無いと飛ばす。台本の先端側 (仮想リンク 25) を持ち上げる。
* `overhand_script.py --replay_actions` は v4 系のタスクでは段の変化で reset されるので、使うなら
  `--cfg "terminations.stage_up=None;terminations.stage_down=None"` を付ける (未検証)。

## 7. 参考文献 (報告書に URL 付きで入れる)

* DeformX (2026): https://arxiv.org/abs/2606.22116 / https://deformx.github.io/
* DLO-Lab (ICML 2026): https://arxiv.org/abs/2606.04206
* DEFORM (CoRL 2024): https://arxiv.org/abs/2406.05931
* DER in generalized coordinates: https://arxiv.org/abs/2310.00911
* MuJoCo elasticity / cable plugin: https://github.com/google-deepmind/mujoco/blob/main/plugin/elasticity/README.md
* Kugelstadt & Schömer, Position and Orientation Based Cosserat Rods (SCA 2016): https://animation.rwth-aachen.de/media/papers/2016-SCA-Cosserat-Rods.pdf
* Deul et al., Direct Position-Based Solver for Stiff Rods (CGF 2018): https://www.animation.rwth-aachen.de/media/papers/2018-CGF-Rods.pdf
* Real2Sim2Real (Lim et al.): https://arxiv.org/abs/2111.04814
* TWISTED-RL: https://arxiv.org/abs/2602.14526
* SoftGym: https://arxiv.org/abs/2011.07215
* Wiggle and Go! (CoRL 2026): https://arxiv.org/abs/2604.22102
* Effective Parameters, Real Behavior: Renormalization for Robotics (2026): https://arxiv.org/abs/2607.24079
* Caporali et al., DLO 知覚と操作のサーベイ (IJRR 2026): https://journals.sagepub.com/doi/10.1177/02783649261432253
* Newton の cable (VBD) の議論: https://github.com/newton-physics/newton/discussions/1271
* TieBot: https://arxiv.org/abs/2407.03245 / KnotDLO: https://arxiv.org/abs/2506.22176
* 紐の摩擦: Samson の技術資料 https://www.samsonrope.com/docs/default-source/technical-bulletins/tb_coeffecient_of_friction.pdf 、
  結び目の摩擦 https://arxiv.org/html/2604.06962v1
* Isaac Lab の変形体: https://isaac-sim.github.io/IsaacLab/main/source/tutorials/01_assets/run_deformable_object.html
