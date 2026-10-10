# v12: v7 + 到達状態アーカイブに「段が上がった手の後の形」だけを保存する

```bash
uv run train --task Isaac-Twisted-RL-Knot-v12 --headless           # 段ごとの MLP
uv run play  --task Isaac-Twisted-RL-Knot-Play-v12 --num_envs 16   # まっすぐなロープから段をつないで評価 (chain モード)
```

ログ: `logs/rsl_rl/knot_tying_v12/`。**学習には `Knot` プリセットを使う** (v3〜v11 と同じ)。継承元は **v7** (v8〜v11 ではない)。
仕様と理由の正本は [`docs/decisions/0025-archive-save-on-stage-up.md`](../../../../../../docs/decisions/0025-archive-save-on-stage-up.md)。ここには配線と、見るべき指標を書く。
初期アーカイブは v7 と同じファイルなので、作り直しは要らない。

## なぜ作ったか

v4〜v11 はアーカイブに毎手の後の形を保存する。段 k の方策が抜け出せない形は、時間切れのエピソードの毎手 (最大 6 回) 段 k に
書き足され、解ける形はほとんど書き足されないので、古い順に捨てるアーカイブは方策の一番苦手な形に占められる。v11 s43 の延長では
段 2 のアーカイブの 4 割が 1 つの形になり、段 2 の成功が落ちた (方策は悪くなっていなかった)。段が上がった手だけを保存すると、
段 k のアーカイブに入るのは段 k より下の方策が作った形だけになり、このループの道が無くなる。段 k の訓練の始まりの分布も、
chain 評価で段 k に着く形の分布と同じになる (TWISTED-RL §IV-D の「l -> k の遷移で到達した形を k の初期状態に使う」)。

## v7 から何を・なぜ変えたか

| 項目 | v7 | v12 | 理由 |
|---|---|---|---|
| 保存する手 (`record_stage_archive` の `save_on`) | `every_move` (床に落ちていて発散していない手の後の形をすべて) | **`stage_up`** (そのうち段が上がり、手の後の段が 1 か 2 の手だけ) | 上 |
| アーカイブの写し (`history_every`) | 0 (書かない) | **200 env ステップ (50 iteration) ごと**に `{log_dir}/archive_history/archive_step{N}.pt` | 記録のため。学習の挙動は変えない (`archive.pt` は上書きなので中身の推移を追えない) |
| Play (chain) | v4 の `play_post_init` | 同じ + `history_every = 0` | Play の `log_dir` は学習の run のフォルダなので、写しを上書きしないため |
| それ以外 (段の定義 `bfs`・初期アーカイブ・`stage_weights` 1:1:1・`straight_prob`・容量 16384・古い順に捨てる・観測・報酬・終了・行動・物理・PPO) | | v7 のまま | 1 版 1 変数 |

v7 との差がこれだけであることは `tests/test_v12_cfg.py` が静的に検査する。保存する行の選び方は `tests/test_state_archive.py`。

v12 での段ごとのアーカイブ:

| 段 | 学習中に入る形 |
|---|---|
| 0 | 何も入らない (台本の初期アーカイブの 128 形のまま) |
| 1 | 段 0 から上がった手の後の形 |
| 2 | 段 0・1 から上がった手の後の形 |
| 3 (固結び) | 保存しない (v7 と同じ) |

手の前後の段は `common/mdp/stages.py` の `stage_transition` から読む (報酬・終了と同じ値)。保存する段は v7 と同じく
保存の項が引いた手の後の段で、`stage_transition` の手の後の段と食い違えば `save_stage_mismatch` に数える (0 のはず)。

## 見るべき指標

版の比較は chain 評価で行う ([`docs/system/evaluation.md`](../../../../../../docs/system/evaluation.md))。比べる相手は v7 の seed 43 / 44 を
iteration 300 まで延ばした run と v11。学習ログは代理の指標で、版の良し悪しの結論には使わない。

* chain 評価の `knot` を iteration 100 / 200 / 300 で並べる。v7 s43 の延長は it100 53.7% -> it300 39.3%、v11 s43 は 47.1% -> 40.0% に下がった。
  0025 の予想では v12 は下がらない。
* 段 2 のアーカイブで 1 つの p-data が占める割合 (`archive_history/` の写しを `view_archive.py --dump_pdata` で数える): 0025 の予想では iteration 300 まで 20% を超えない。
* `Curriculum/stage_log/success_s2`: 0025 の予想では v11 s43 のように途中で半分以下に落ちない。`timeout_s0` は v7 の延長のように増えない。
* `Curriculum/stage_log/saved_s0` は 0、`arc_size_s0` は 128 のまま。`saved_s1`・`saved_s2` で段 1・2 の入れ替わりの速さが見える
  (段 2 は 1 iteration に約 1750 件の見込み。容量 16384 は約 9 iteration で入れ替わる。概算)。
* `Curriculum/stage_log/save_stage_mismatch`・`restore_mismatch`: 0 のはず。
