# v4 の初期アーカイブと smoke test (2026-09-28)

- 状態: 凍結
- 読むべき状況: v4 以降が読む初期アーカイブ (`outputs/archive_seed/overhand_28links.pt`) の中身の件数を知りたいとき。v4 の配線 (段ごとのアーカイブ・切り分けの上書き・Play の chain モード・ONNX の出力) が動いたことの記録を見たいとき

## 要約

- 結論: 台本の途中の形を、段 0 / 1 / 2 の初期アーカイブ (128 / 264 / 130 形) にした。アーカイブの保存と読み戻し・切り分けの上書き・Play・ONNX の配線が動いた。
- 有効か: 有効 (配線の確認)。シード・コマンド・run のフォルダは記録なし。

> 転記元: `src/isaac_twisted_rl/tasks/knot_tying/versions/v4/README.md` の「初期アーカイブと smoke test (2026-09-28)」(`:312-336`、commit 62b251b の時点)。
> 実装の注意 (ONNX の `__getitem__`・終了時の `TypeError`) は [v4 README](../../src/isaac_twisted_rl/tasks/knot_tying/versions/v4/README.md) にも残っている。

## 目的と仮説

ゲート 1 の台本の途中状態を段ごとの初期アーカイブとして書き出し、v4 の学習が 3 段とも iteration 0 から回ること、切り分けの上書き・アーカイブの保存と読み戻し・Play (chain モード) と方策の書き出しが動くことを確かめる。

## 条件

| 項目 | 値 |
|---|---|
| commit | 記録なし (v4 の実装のコミットは `5115f9f`、2026-09-30。smoke test がどのコミットの上かは記録なし) |
| タスク ID とプリセット | 初期アーカイブ: `overhand_script.py --dump_archive` (`Knot`。今のコマンドは `--task Isaac-Twisted-RL-Knot-v2-R2 --num_envs 128 --jitter 0.25 --headless --dump_archive`: v4 README の冒頭)。smoke test: `Isaac-Twisted-RL-Knot-v4` (`Knot`) |
| 変えた変数 | — |
| シード | 記録なし |
| iteration (最後のチェックポイント) | 3 iteration (切り分けの上書きの確認は 1 iteration) |
| num_envs | 初期アーカイブ 128 (±25%)。smoke test 64 |
| 学習のコマンド | 記録なし |
| 評価のコマンド | 記録なし |
| run のフォルダ | 記録なし (`logs/rsl_rl/knot_tying_v4/` に 2026-09-28 の run は無い) |
| 出力 | 初期アーカイブ: `outputs/archive_seed/overhand_{num_links}links.pt` (`Knot` は 28: ls で `outputs/archive_seed/overhand_28links.pt` を確認)。Play: `exported/policy.pt`・`policy.onnx` |
| GPU | V3 の学習 (4096 環境) と共有 |
| 無効にした run | なし |

## ゲート (学習の前の確認)

初期アーカイブを作る台本の固結びは 123/128 (発散 0)。ゲート 1 の本体は [`2026-09-27_gate1.md`](2026-09-27_gate1.md)。

## 結果

**初期アーカイブ** (`overhand_script.py --dump_archive`、`Knot`、128 環境、±25%):
段 0 / 1 / 2 = 128 / 264 / 130 件。台本の固結びは 123/128 (発散 0)。cover の手の後は
**126/128 が段 2** だったが、R2 除去後の class で数えると C2 は 18 だけで、110 は C1 だった
(v3 のアーカイブではこれらが C1 側に入っていた)。

**smoke test** (`Knot`、64 環境 x 3 iteration、V3 の学習と GPU を共有):

| iter | `success_s0` / `s1` / `s2` | `fail_s1` / `s2` | `diverged_s0` / `s1` / `s2` | `arc_size_s0` / `s1` / `s2` | `restore_mismatch` | `value_function` |
|---|---|---|---|---|---|---|
| 0 | 0.69 / 0.21 / 0.08 | 0.52 / 0.77 | 0.07 / 0.08 / 0.07 | 304 / 437 / 169 | 0.000 | 0.52 |
| 1 | 0.74 / 0.10 / 0.07 | 0.64 / 0.86 | | 610 / 794 / 220 | 0.000 | 0.41 |
| 2 | 0.80 / 0.11 / 0.09 | 0.69 / 0.86 | | 926 / 1136 / 277 | 0.000 | 0.35 |

* 初期アーカイブのおかげで、3 段とも iteration 0 からエピソードがある。ほぼランダムな初期方策では、
  段 1 / 2 から始めると 5〜8 割が段を下げて (輪を崩して) 終わる。
* 1 iteration 約 780 s。V3 (4096 環境) と GPU を共有していて物理ステップが遅く、環境数を減らしても
  短くならない。
* 切り分けの上書き 4 種を同時に指定した 1 iteration も走り、`params/*.yaml` に反映され、
  `archive.pt` が書き出されて読み戻せることを確かめた (`save_every=6` に上書き)。
* Play (chain モード) は 18 手まで続き (段が変わっても終わらない)、`exported/policy.pt` と
  `policy.onnx` が出力された。ONNX の exporter は `policy.actor[0].in_features` を読むので、
  `GatedMLP` に先頭のヘッドの層を返す `__getitem__` を持たせている。
* 終了時の `TypeError: post_quit()` は、学習を終えた後の Kit の終了処理 (`kit_exit` に None が渡る) で出るもので、v4 の項とは関係ない。

## 結論

- 分かった: 初期アーカイブは段 0 / 1 / 2 = 128 / 264 / 130 件。cover の後の形は段 2 に入る (R2 除去後の class では大半が C1)。`restore_mismatch` は 3 iteration とも 0.000。切り分けの上書き・アーカイブの保存と読み戻し・Play と ONNX の出力が動いた。
- 推測: 記録なし。
- 未確認: smoke test のシード・コマンド・run のフォルダは記録なし。速度 (約 780 s/iter) は GPU を共有した値で、単独の GPU の速度は [`2026-09-30_v4_speed.md`](2026-09-30_v4_speed.md)。

## 次の作業

本番の学習の速度の調査 ([`2026-09-30_v4_speed.md`](2026-09-30_v4_speed.md)) と overnight の比較 ([`2026-10-01_v4-v6_overnight.md`](2026-10-01_v4-v6_overnight.md))。
