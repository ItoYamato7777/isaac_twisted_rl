# 0009 到達した形状からエピソードを始める (到達状態アーカイブ)

- 状態: 有効 (v3 は位相クラスごと、v4〜v11 は段ごとのアーカイブ。v12 以降の保存の規則は [0025](0025-archive-save-on-stage-up.md))
- 読むべき状況: 初期状態の分布・アーカイブの中身・初期アーカイブのファイルを変えるとき。「毎回まっすぐなロープから始める」に戻したくなったとき
- 最終更新: 2026-10-09

## 決定

reset の一部を、学習中に到達した形状から始める。

| 版 | キー | 選び方 | 容量 | 初期の中身 |
|---|---|---|---|---|
| v3 | R2 除去後の位相クラス (1, 2) | 確率 0.5 でアーカイブ、C1 : C2 = 1 : 2 | 各 4096 件 | 空 (ランダムな手が C1 / C2 に着いた時点から埋まる) |
| v4〜v7 | 段 (0, 1, 2) | 空でない段から `stage_weights = (1, 1, 1)` で段を選び、その段の中で一様に選ぶ。段 0 は確率 `straight_prob = 0.5` でまっすぐなロープ | 各 16384 件 | ゲート 1 の台本の途中状態をファイルから読む。学習中も `{log_dir}/archive.pt` へ書く |

- 保存するのは、毎手の終わりに床に落ちていて発散していない形状 (v4〜v11。段が変わらない手の後の形も保存するので、方策が抜け出せない形が溜まるループになる。v12 は段が上がった手の後の形だけ: [0025](0025-archive-save-on-stage-up.md))。v3 は class 1 と 2 だけ (固結びから始めても学ぶことが無い)。root の姿勢 (xy はロープ重心からの相対) と関節角を持つ。
- 復元は、重心を env 原点に置いて z 軸まわりに一様乱数で回して書き込む (回しても p-data は変わらない)。直後に FK で位相を計算し直し、保存時の class と一致したかを `restore_mismatch` に数える (v3)。
- v4: 段を上がった直後の形状は次の段の初期状態になる (論文の「l → k の遷移で到達した形を k の初期状態に使う」)。起動時に `load_paths` を読む。既定は `outputs/archive_seed/overhand_{num_links}links.pt` (`overhand_script.py --dump_archive` の出力)。無ければ警告して空から始める。600 手 (= 50 iteration、チェックポイントの間隔) ごとに `{log_dir}/archive.pt` へ書く。`--resume` するときは `env.events.reset_to_stage.params.load_paths` にそれを足す。

## 理由

### 「干し草の中の針」を、論文は初期状態の選び方で解いている

TWISTED の記述では、ランダムな手では 3 交差の状態に**1 コア 1 時間で 27 個しか**到達できず、到達済みの状態から再開する方式にして初めて 537 個に増えた。TWISTED-RL も交差数 k の方策の初期状態を「前の段で到達した形状」から取る (TWISTED-RL §IV-D、Go-Explore)。

v2-R2 は毎回まっすぐなロープから 6 手で始めるので、C2 の練習ができるのは「偶然 C1 に着いたエピソードの残りの手」だけになる。`paper/Report_Paper.md` はこれを「論文との差で最も効いているのはここ」と考えた (同 §5.3。推測として書かれている)。

移植元では replay buffer を次段の初期状態プールにしていた (off-policy 前提)。初期状態プールの考え方自体は PPO でも使えることが後で分かり、v3 で取り込んだ ([0006](0006-no-symbolic-planner-her.md))。

### v4 で段ごとにし、初期アーカイブを読むようにした理由

| 変更 | 理由 |
|---|---|
| キーを R2 除去後の class (1, 2) から段 (0, 1, 2) へ | cover 後の状態 (R2 除去後は class 1) が C1 に混ざっていた。v3 の最初の run では `arc_size_c1` が 1 iteration で 4096 (満杯)、`c2` が 3403 |
| ゲート 1 の台本の途中状態をファイルから読む | 段 2 のエージェントが最初から練習できる。`--resume` で空に戻らない。台本の途中状態 (輪 / cover 後 / C2) がそのまま初期データに使えることは、ゲート 1 の時点で分かっていた (旧 §4.8.10 の 6) |
| 段を等確率で選び、その段のアーカイブから | 論文 §IV-D (遷移を一様に選び、出発点の形を一様に選ぶ) |
| 容量を 4096 から 16384 へ | 既定の 4096 環境では段 1 に 1 手で数千件入る |

## 今の実装

(`kt/` = `src/isaac_twisted_rl/tasks/knot_tying/`)

| 何 | 場所 | 確認 |
|---|---|---|
| リングバッファ (純 torch) | `kt/common/state_archive.py:34` (`StateArchive`) | 一致 |
| v3 の保存・復元 | `kt/common/mdp/archive.py:57` (`ResetFromArchive`)、`:199` (`record_archive`)、配線 `kt/versions/v3/env_cfg.py:155-163` (`prob = 0.5`、`class_weights = (1.0, 2.0)`) | 一致 |
| v4 の段ごとのアーカイブ | `kt/common/mdp/archive.py:231` (`ResetToStage`)、`:461` (`record_stage_archive`)、配線 `kt/versions/v4/env_cfg.py:150-160` (`stage_weights = (1.0, 1.0, 1.0)`、`straight_prob = 0.5`、`capacity = 16384`、`load_paths = (SEED_ARCHIVE,)`) | 一致 |
| 初期アーカイブのパス | `kt/versions/v4/env_cfg.py:53` (`outputs/archive_seed/overhand_{num_links}links.pt`) | 一致。git の管理外のファイル (作り直しは skill `physics-check`) |
| Play (chain 評価) は段 0・まっすぐなロープから | `kt/versions/v4/env_cfg.py:219-220` | 一致 |

## 出典

`docs/history/next_action.md` の行番号は、頭書き (33 行) を足す前のもの (今の行番号は 33 を足す)。

- `paper/Report_Paper.md:169-173` (§5.3)
- `docs/history/next_action.md:395-396` (旧 §4.8.1 の末尾)、`:532` (旧 §4.8.10 の 6)
- `kt/versions/v3/README.md:56-69` (P1-1)
- `kt/versions/v4/README.md:14-38`、`:40-60` (変更表)、`:97-108` (段ごとのアーカイブ)
