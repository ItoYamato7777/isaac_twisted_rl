---
paths:
  - "src/isaac_twisted_rl/tasks/knot_tying/versions/**"
---

# `versions/` を変えるときの規則

- **既存の版の挙動を変えない。** 版は「設定」で、過去の run の再現は版のコードで行う。挙動を変えたいときは新しい版を作る (skill `add-version`)。既存の版のバグを直すときは、その版の README と実験記録に、直した内容と影響する run を書く。
- **継承の連鎖に注意する。** v5・v6・v7 と rope study は v4 の `env_cfg.py` を継承しているので、v4 を変えると全部に波及する。private な名前 (`_play_post_init` など) も版をまたいで import されている。
- 版の README には、その版の仕様と理由だけを書く。結果は `docs/experiments/`、現在地は `docs/STATUS.md`。
