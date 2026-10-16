# 0031 chain 評価を回すときは、評価したエピソードの動画も残す (`video/`)

- 状態: 有効 (2026-10-16 にユーザーが方針として決めた。**未実装**。実装は次のセッション)
- 読むべき状況: chain 評価を回すとき。動画を撮る・置く・見返すとき。実機の撮影を `video/` に置くとき
- 最終更新: 2026-10-16

## 決定

1. **Claude が chain 評価 (`eval_chain.py`) を回すときは、同じ評価で数環境の動画も撮る。** 目的は、数値 (`knot` など) では見えない「学習した方策が実際にどう動くか」を人が確かめること。特に **報酬のハック** (例: 吊ったまま交差を見せる、判定の穴に居座る) と **物理的におかしい挙動** (めり込み・跳ね・発散の直前の振動) を見つける。
2. **学習の途中の様子の動画は撮らない。** 学習のプロセスでの録画 (`train --video`) も、学習と並行した録画もしない。チェックポイントの評価のときだけ撮る。
3. **動画は結論の根拠にしない。** 数は chain 評価の JSON が正本。動画で見つけたことは「観察」として実験記録に書き、数で確かめるまでは推測と書く。
4. **数値を変えない。** `--video` を付けても、付けないときと JSON の全キー (`meta` の argv・日時などを除く) がビット単位で同じであること ([evaluation.md](../system/evaluation.md) §2 の決定論)。これを実装のゲートにする。一致しなければ評価とは別のプロセスで撮る (下の「代わりの案」)。
5. **余裕が無ければ撮らない。** 撮影 (描画) は VRAM を数 GB 足す (見積り。実測で決める)。評価の前に GPU の空きが足りなければ、動画なしで評価だけを回し、撮らなかったことを記録に残す。評価を遅らせてまで撮らない。
6. **置き場所はリポジトリのルートの `video/`。`video/highlights/` は動画ごと git で追跡する。`video/sim/`・`video/real/` の動画は追跡しない** (`manifest.json` と `video/README.md` だけを追跡する)。

### 置き場所

```
video/
  README.md                         追跡。正本 (実装後に作る docs/system/video.md) への案内だけ
  sim/<experiment_name>/<run>/      動画は追跡しない。logs/rsl_rl/<experiment_name>/<run>/ と 1 対 1
      <評価の JSON の名前>/          例 eval_chain_it300/ (--out の拡張子を除いた名前。DR 付きなら eval_chain_it300_dr/)
          env003_knot_m09.mp4       env<番号>_<結末>_m<手数>.mp4。結末は knot / diverged / time_out / unfinished
          manifest.json             追跡する
  real/<YYYYMMDD_HHMMSS>[_<題>]/    動画は追跡しない。hiro_knot の real_logs/<run> と同じ名前。実機の動画 + 手で書く manifest.json
  highlights/<YYYY-MM-DD>_<題>/     動画ごと追跡する (commit に含める)。sim・real から選んで編集したもの。manifest.json に元の clip を書く
```

- `video/sim/` はスクリプトが書く。手で編集しない (`logs/`・`outputs/` と同じ扱い)。
- `sim/`・`real/` の動画は手元にしか無い。残したいもの (README・発表・論文に使うもの) は `highlights/` に選んで commit する。
- `highlights/` の動画は git の履歴から消せない (消すには履歴の書き換えが要る)。GitHub は 1 ファイル 100 MB で拒み、50 MB を超えると警告する。入れる前に短く切り、再エンコードして小さくする (目安は 1 本 20 MB 以下。上限は実装のときに決めて `docs/system/video.md` に書く)。
- 実機: フォルダ名を hiro_knot の `real_logs/<YYYYMMDD_HHMMSS>` に揃える。そこの `header.json` が使った方策の run 名を持つので、`video/sim/<exp>/<run>/` と辿れる。`manifest.json` には real_logs の run・方策の run とチェックポイント・撮影の機材と視点・結果 (固結びできたか・何手目まで)・メモを書く。

### 動画の中身 (人が確かめやすくするため)

- **1 手の中を撮る。** 既定は 8 物理ステップごと (240 Hz で 30 fps = 実時間)。1 手 (759 物理ステップ) が約 95 フレーム、18 手で約 1 分。
- **撮る環境ごとにカメラを置く** (環境の原点 + viewer の eye / lookat)。今の viewer は world の原点を見るので、環境の数によっては環境の間を映す。
- 画面の上に文字を重ねる (ASCII): run 名・iteration・環境の番号・手 k/18・フェーズ名・段。グリッパは段の色 (`scripts/rsl_rl/stage_colors.py` をそのまま使う)。終わりの 1 s は止めて結末を出す。
- `manifest.json`: 評価の JSON のパス・コマンド (argv)・Hydra の上書き・チェックポイント (iteration・sha256)・commit と未コミットの有無・撮った環境ごとの結末と手数・fps・解像度・1 フレームの描画時間・ピークの VRAM・撮らなかった理由 (撮らなかったとき)。
- 撮る環境の数の既定は 8 程度 (固結び・発散・時間切れが混ざる数。v14 は固結び約 57%・発散約 14%)。解像度・数は実測で決める。

## 理由

- 今は動画が 1 本も無く、方策の動きは数値と学習ログからしか見ていない。報酬のハック (v2 の偽の C2 への居座り: [記録](../experiments/2026-09-24_v2_fake-c2.md)) や、吊って交差を見せかける方策 ([0005](0005-measure-after-retreat.md)) は、過去にも数値だけでは気づくのが遅れた。
- 評価と同じ実行で撮れば、動画のエピソードは JSON に数えたエピソードそのものになる (環境の番号が対応する)。別に撮ると、環境の数・乱数・GPU の束ね方が違う別の標本になる。
- 学習の途中の動画は、学習を遅くし (描画)、GPU を共有している他のジョブの VRAM を圧迫する。確かめたいのは学習した結果なので、チェックポイントの評価で足りる。

## 検討して採らなかった案

| 案 | 採らなかった理由 |
|---|---|
| `train --video` / `play --video` (Isaac Lab の雛形の `gym.wrappers.RecordVideo`) | このタスクは 1 env.step = 1 手 (約 759 物理ステップ) なので 1 手に 1 フレーム (約 0.3 fps) しか撮れない。`render_fps = 1/step_dt` (`isaaclab/envs/manager_based_rl_env.py:88`) |
| 学習と並行して 100 iteration ごとに評価と動画を回す見張り | 学習の途中の動画は要らない (ユーザー、2026-10-16) |
| Isaac Lab の `Camera` / `TiledCamera` を scene の設定に足す | 全環境にカメラが付く。`rtx_sensors` が立って env.step が物理ステップの間と reset で勝手に描画する。評価の設定と学習の設定の照合 (`param_check`) にも違いとして出る |
| RecorderManager の項で撮る | 項があると step で観測を余計に 1 回計算する (`manager_based_rl_env.py:211-214`)。env_cfg も変わる |
| `sim/`・`real/` の動画も追跡する / Git LFS で追跡する | 評価のたびに数十本・数百 MB 増え、git の履歴から消せない。LFS は GitHub の無料枠 (約 1 GB) をすぐ超え、git-lfs も入っていない。追跡するのは選んだ `highlights/` だけにする |

### 代わりの案 (ゲート 4 が通らなかったとき)

評価とは別のプロセスで、同じチェックポイント・同じ Hydra の上書きで少数の環境を撮る。動画は評価の標本ではなく「同じ分布からの別の標本」と manifest に書く。

## 実装の見通し (2026-10-16 のコードと isaaclab 2.3.0 の source で確かめた。実行はしていない)

(`rl/` = `src/isaac_twisted_rl/scripts/rsl_rl/`)

1. **`rl/video_layout.py`** (Kit なし。stdlib と torch だけ。`stage_colors.py` と同じ分け方。テスト `tests/test_video_layout.py`): 出力先 (チェックポイントのパスから `<exp>/<run>`、`--out` から JSON の名前)、clip の名前、`frame_plan(physics_dt, decimation, every)` (fps・1 手のフレーム数)、結末の名前 (`stage_up` → knot、`diverged`、`time_out`、何も立たなければ unfinished。同じ手に 2 つ立てば `chain_stats` の優先順位と同じ)、manifest の組み立て、`nvidia-smi` の出力から空きを読んで撮るかを決める関数。
2. **`rl/video_recorder.py`** (Kit。import はクラスの中):
   - **カメラ**: 撮る環境ごとに `/World/VideoCams/cam_i` に `sim_utils.PinholeCameraCfg` で置く。位置は `scene.env_origins[i] + viewer.eye`、向きは `isaaclab.utils.math.create_rotation_matrix_from_view` (`Camera.set_world_poses_from_view` と同じ OpenGL の向き)。`rep.create.render_product` + `rgb` annotator (`manager_based_rl_env.py:285-291` と同じ作り方)。作った後に数回 `sim.render()` で暖める (最初は空のフレームが返る)。8 台を超えるなら `render_product_tiled` で 1 回の描画にまとめる案もある。
   - **フック**: `env.unwrapped.sim.step` と `sim.render` をインスタンスで包む (前例 `scripts/bench_step_time.py:178-193`)。k 物理ステップごとに `sim.render()` を呼んでフレームを取る。headless では `has_rtx_sensors()` が False のままなので、env.step は自分では描画しない (`manager_based_rl_env.py:180,195`)。GUI があるときは env 自身の描画に乗る (二重に描画しない)。
   - **重ね書き**: フェーズは `action_manager.get_term("pick_place")._phase[i]` と `common/mdp/phases.py` の `PHASE_NAMES`、手の番号は `episode_length_buf[i] + 1`、段は手の前の観測の段 (`eval_chain.py` と同じ読み方)。結末は env.step の後に `dones` で絞って終了項を読む (`.claude/rules/mdp.md`)。
   - 書き出しは imageio-ffmpeg (libx264・yuv420p・`macro_block_size=2`。540 は 16 の倍数でない)。`close()` で包みを戻し、annotator・render product を壊す (`finally`)。
   - 止める条件: 描画モードが PARTIAL_RENDERING 未満 (`--enable_cameras` が無い)、XPBD の紐 (GUI が無いと描かれない: `common/rope_backends/adapters.py:344-346`)。
3. **`rl/eval_chain.py` に `--video` を足す** (既定は off。既定の挙動・キーは変えない約束): `--video_envs` (既定 8 個)・`--video_every` (既定 8)・`--video_res`・`--video_min_free_mib`。`--video` のときは AppLauncher の前に `enable_cameras` を立てる。起動の前に GPU の空きを読み、足りなければ動画を切って評価だけを回し、JSON の新しいキー `video` (撮ったか・出力先・撮らなかった理由) に残す。評価のループ (撮った環境が全部終わるまで) は今のまま。
4. **ゲート (実装の後、commit してから)**:
   - 単体テスト・`tests/test_invariants.py`・`tests/test_docs.py`。
   - スモークテスト: 2 環境・2 手・640x360 で、`ffprobe` のフレーム数と fps、0.5 s と 4 s のフレームを PNG に出して目で見る (ロープとハンドが画面の中・グリッパが段の色・文字が読める・最初が黒くない)。
   - **数値が変わらないこと**: v14 s43 の `model_300.pt` を標準の条件 (512 環境・seed 7・決定論的) で `--video` 付きで評価し、既存の `eval_chain_it300.json` と全キー (`meta`・`video` を除く) が一致すること。`--enable_cameras` で Kit の設定 (rendering の kit) が変わるので、付けない評価とも比べる。
   - 所要時間・VRAM・RAM を測り、`--video_min_free_mib` の既定を決める。
5. **文書とルール** (ゲートが通った後): 正本 `docs/system/video.md` (この文書の「置き場所」「動画の中身」とコマンド・実測・しきい値)、`docs/system/evaluation.md` §1 に `--video` が値を変えないことと JSON の `video` キー、`.claude/rules/video.md` (`paths` に `video/**` と `rl/video_*.py`。守る約束だけ)、`CLAUDE.md` の「実験の約束」に 1 行 (chain 評価には `--video` を付ける。動画は結論の根拠にしない)、skill `train-eval` §3 のコマンドに `--video`、skill `experiment-record` に「動画の場所と、動画で見た観察 (推測として)」の 1 行、`docs/README.md`・`rl/README.md` の表、`video/README.md`、`.gitignore` (`/video/sim/**`・`/video/real/**` を外し、`manifest.json` は例外で追跡。`video/highlights/` と `video/README.md` は追跡。既存の `**/videos/*` は `video/` に当たらない。`git check-ignore -v` で確かめる)、`.claude/settings.json` の deny に `Edit(/video/sim/**)`。
6. hiro_knot (`~/work/hiro_knot`) の側に `video/real/` の約束を書くかは、実装の後にユーザーに確かめる (このリポジトリの外)。

## 出典

- `isaaclab/envs/manager_based_rl_env.py` (`.venv` の中): step の物理ループと描画 (L176-241)、`render()` の render product (L243-300)、`render_fps` (L88)。
- `isaaclab/sim/simulation_context.py`: 描画モード (headless + `--enable_cameras` で PARTIAL_RENDERING)、`has_rtx_sensors()` は `Camera` だけが立てる。
- [evaluation.md](../system/evaluation.md) §2: chain 評価の決定論 (`eval_chain.py` を変えたら既存の JSON と一致を確かめる規則)。
