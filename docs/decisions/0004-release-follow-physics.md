# 0004 「離したのに付いてくる」は物理パラメータで直す (柔らかいロープ・指は常に全開)

- 状態: 有効 (`twist` と、同じ材料の `knot` に適用。他のプリセットは未確認)
- 読むべき状況: ロープのプリセット・ヤング率・指の開度・離した後の待ち時間 (`settle_out`) を変えるとき。学習ログで `held_up` だけが立ち上がったとき
- 最終更新: 2026-10-04

## 決定

| 対策 | 場所 | 内容 |
|---|---|---|
| ロープを柔らかく | `rope_specs.py` の `TWIST_YOUNGS_MODULUS` | `twist` のヤング率だけ 1.0e6 → **1.0e5 Pa**。曲げ剛性 EI はヤング率に比例するので、曲げ関節のばね定数も 1/10 になる。指を開いたときに、ロープが腰で引っかからなくなる |
| 指を常に全開に | `OPEN_RATIO` | 0.8 → **1.0**。接近時も離すときも全開。`common/mdp/phases.py` では、解放時の `RELEASE_RATIO` を `OPEN_RATIO` に統合した |

- `twist` だけヤング率を下げたのは、**グリッパーで挟んで運ぶ用途に特化したプリセット**にしたため。`stiff` / `fine` は比較用に元の剛性のまま残してある。後から足した `knot` は `twist` と同じ材料。
- **他のプリセットで学習するときは、この現象が再発しないか必ず確認すること。**
- 離した後にロープが落ち切るのを待つ `settle_out` は物理の待ちなので、尺を詰めるときも比率を保って残す。削ると再発する。物理待ち系のフェーズ (close / hold / release / settle_out / relax) を削るなら、移動系と同様の実測をしてから行う。

## 理由

当初、指を開いて真上へ退避すると、ロープが指に残ったまま持ち上がり、作った交差が壊れていた (台本デモ `demos/topology_debug.py`、Step 5)。上の 2 つの対策で解決し、どちらもコードに反映済み。

報酬の測定タイミングは、この対策とは別に押さえる ([0005](0005-measure-after-retreat.md))。

## 今の実装

(`kt/` = `src/isaac_twisted_rl/tasks/knot_tying/`)

| 何 | 場所 | 確認 |
|---|---|---|
| `TWIST_YOUNGS_MODULUS = 1.0e5` (`ROPE_YOUNGS_MODULUS = 1.0e6`) | `kt/common/rope_model/rope_specs.py:98`、`:89` | 一致 |
| `twist` と `knot` が `TWIST_YOUNGS_MODULUS` を使う | `kt/common/rope_model/rope_specs.py:451`、`:508` | 一致 |
| `OPEN_RATIO = 1.0`、`RELEASE_RATIO` は統合して無い | `kt/common/mdp/phases.py:49-56`、`:25-31` | 一致 |
| `settle_out` を残す理由 | `kt/common/mdp/phases.py:19-22`、`PhaseDurations` の docstring (`:145-148`) | 一致 |
| 再発をログで気付くための `held_up_penalty` | `kt/common/mdp/rewards.py:410-424` | 一致 |
| 学習に使うプリセットと再確認の注意 | `kt/common/rope_task_cfg.py:7-13` | 見出しは「`knot` (v3)」のまま。v4〜v7 も `knot` で学習している (`refactor/audit/C.md` §2.4) |

## 出典

`docs/history/next_action.md` の行番号は、頭書き (33 行) を足す前のもの (今の行番号は 33 を足す)。

- `docs/history/next_action.md:204-216` (旧 §3(c))
- `kt/common/mdp/phases.py:9-31` (docstring)
