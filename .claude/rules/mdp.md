---
paths:
  - "src/isaac_twisted_rl/tasks/knot_tying/common/mdp/**"
---

# `common/mdp/` を変えるときの規則

- **項は追加だけ。削除・改名しない。** 過去の版の設定が名前で参照している。役目を終えた項は `.. deprecated::` で残す。`tests/test_invariants.py` の I4 が検査する。挙動を変えたい項は、新しい名前の項を足し、新しい版で使う。
- **未コミットの項で学習しない。** 学習した後で項を消すと、その run はコードごと再現できなくなる (旧 v7 の `stage_time_out` がそうなった)。
- **`env.step()` の順序** は「終了判定 → 報酬 → reset → 観測」。そこから来る罠を 3 回踏んでいる:
  - 報酬と終了の項が見る状態は reset の前のもの。観測は reset の後のもの。
  - 抽出器などのキャッシュは、どの時点で更新されるかを確かめてから使う。
  - `TerminationManager.get_term()` の値は、立った環境だけが書き換わり、reset でも消えない。reset した環境で絞ってから数える。
- **成功の判定はハンドを退避させた後に測る** (`docs/decisions/0005-measure-after-retreat.md`)。手の途中で測ると、ハンドで吊って交差を見せかける方策を学ぶ。
- **整形報酬の γ は PPO の `gamma` と一致させる** (一致しないと整形が最適方策を変える)。
- **整形報酬を検証するときは、`env.step()` から返った後に位相クラスを読まない。** 報酬関数を差し替えて、報酬を計算するその瞬間の値を捕まえる (`docs/decisions/0015-shaping-verification-pitfall.md`)。
