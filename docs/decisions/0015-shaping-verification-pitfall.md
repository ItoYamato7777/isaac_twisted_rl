# 0015 整形報酬は、報酬が計算されるその瞬間に捕まえて検証する (`env.step()` の後に読まない)

- 状態: 有効 (Isaac Lab の `ManagerBasedRLEnv` を使う限り)
- 読むべき状況: 整形報酬 (`knot_class_shaping` など) や、状態に依存する報酬・終了項が正しいかを確かめるとき。`env.step()` の後に位相や報酬の積算値を読んで比べたくなったとき
- 最終更新: 2026-10-04

## 決定

整形報酬が正しいかを確かめるときは、**`env.step()` から返ってきた後に位相クラスを読まない**。報酬関数自体を差し替えて、**報酬が計算されるその瞬間に** `Φ(s)` と `Φ(s')` を捕まえる。実際にそうすると、誤差は厳密に 0 になる。

## 理由

`env.step()` の後に読むと、次の 2 つの理由で「報酬が使った値」とは違うものを見てしまう。

1. **reset のタイミング。** `ManagerBasedRLEnv.step()` は `termination -> reward -> _reset_idx -> observation` の順に処理する。終了した env は、報酬計算の後にロープが初期姿勢へ戻る。そのため step 後に読むと、`Φ(s')` が reset 後の値になる。
2. **`_episode_sums` は reset でゼロクリアされる** (`reward_manager.py` の `reset()`)。エピソードをまたいで積算値を比較すると、必ずずれる。

素朴に step 後に読んだ版では、最大 0.455 のずれが出た。しかも、終了した env が無いステップでもずれた (抽出器が `common_step_counter` でスタンプされているので、どのタイミングで呼ぶかで返る batch が変わる)。**報酬のバグではなく検証方法のバグ**だったが、気付きにくいので残す (旧 §4.8 = v1 への拡張、2026-09-15 の節に書かれていた)。

同じ `env.step()` の順序から来る罠は、抽出器のキャッシュ (reset された環境の観測が reset 前の形状のままになる。`PickPlaceAction.reset()` から `invalidate_extractor()` を呼ぶ: 旧 §4.6(d)) でも踏んでいる。

## 今の実装

(`kt/` = `src/isaac_twisted_rl/tasks/knot_tying/`)

| 何 | 場所 | 確認 |
|---|---|---|
| 抽出器の `common_step_counter` によるスタンプ | `kt/common/mdp/observations.py:39-71` (`get_extractor`・`invalidate_extractor`) | 一致 |
| 規則としての要約 | `.claude/rules/mdp.md` | **元の記述と食い違う**: 規則は「状態アーカイブへの退避の後に測り、割引率 γ を学習と揃える」と書くが、旧 §4.8.8 にあるのは上の「報酬関数を差し替えて計算の瞬間に捕まえる」だけ (アーカイブの話は無い)。γ の一致は [0007](0007-topology-class-ladder-shaping.md) の話 |
| この検証を自動で行うテスト | 未確認 (`tests/` に、報酬関数を差し替えて Φ を捕まえるテストがあるかは確かめていない) | |

## 出典

`docs/history/next_action.md` の行番号は、頭書き (33 行) を足す前のもの (今の行番号は 33 を足す)。

- `docs/history/next_action.md:447-463` (旧 §4.8.8)、`:351-353` (旧 §4.6(d))
