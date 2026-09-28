# v0: 交差を 1 つ作る (基準バージョン)

```bash
uv run train --task Isaac-Twisted-RL-Twist-v0 --headless
uv run play  --task Isaac-Twisted-RL-Twist-Play-v0 --num_envs 1
```

ログ: `logs/rsl_rl/knot_tying_v0/`

## 位置づけ

**紐結びの最小版で、以降のバージョンの基準点。**
ロープの一部を持ち上げてロープ本体の上をまたがせて置くと、「交差」ができる。
これを位相不変量として検出し、報酬に変える。**学習は成功した** (結果は [v2 README](../v2/README.md) §1 の比較表)。

このバージョンは、`main` ブランチのコミット `1f5693f` 時点の実装と同一
(`experiment_name` を `knot_tying_v0` に分けた点だけが違う)。
改良はここを書き換えずに、新しいバージョンとして足すこと
(規約は [`../../README.md`](../../README.md))。

シーン・物理・行動・終了判定・イベントの仕組みは [`common/README.md`](../../common/README.md) にまとめてある。
ここに書くのは v0 が決めている部分 (観測・報酬・エピソード長・PPO) だけ。

## タスク

**ロープを掴んで交差を 1 つ作り、手を離しても残す。**

- ロープ: `twist` プリセット (D6 ジョイント連鎖の 20 リンク、直径 20 mm、全長 0.82 m)。
  柔らかい曲げ剛性と捩り自由度を持つ ([`rope_specs.py`](../../common/rope_model/rope_specs.py))。
- ロボット: **アームの無いキネマティックハンド** ([`parallel_gripper/`](../../common/parallel_gripper/))。
  IK の失敗・可達領域外・自己衝突・特異点が存在しないので、学習は「紐を操作する」ことだけに集中できる。
- 1 エピソード: **5 手** (`actions_per_episode`)。フェーズの尺は `DEFAULT_DURATIONS`
  (1 手 2.855 s = 686 物理ステップ)。

## 観測 (75 次元)

| 項 | 内容 |
|---|---|
| `rope_link_pos` | 全リンク中心の、env 原点からの相対座標 (ロープの「形」) |
| `crossings` | 現在の交差数 |
| `writhe` | 交差符号の総和 (位相の粗い指標) |
| `first_crossing` | 先頭の交差の特徴量 `(over, sign, point_i/N, point_j/N)` |
| `hand_pos` | ハンド root の、env 原点からの相対位置 |
| `actions` | 直前に取った行動 |

`first_crossing` には正規化の分母がバッチごとに変わるバグがある。v0 の挙動を保つために残してあり、
v1 以降は修正版の `crossing_features` を使う ([v1 README](../v1/README.md) §1)。

## 報酬 ([`mdp/rewards.py`](../../common/mdp/rewards.py))

| 項 | 関数 | 重み (1 手あたり) | 内容 |
|---|---|---|---|
| `progress` | `crossing_progress` | +0.3 | 交差数の増分 (密報酬)。交差が減れば負になる |
| `success` | `crossing_achieved` | +1.0 | 交差数 ≥ 1 **かつ**ロープが床付近にある (吊られていない) |
| `grasp_shaping` | `grasp_distance_penalty` | −0.05 | `close` 終了時の指先とロープの距離 |
| `failure` | `action_failed` | −0.2 | ロープが NaN で手を実行できなかった / 発散した |
| `held_up` | `held_up_penalty` | −0.1 | 手の終わりにロープが浮いたまま (吊っている兆候) |

p-data は離散量で、一致 / 不一致だけでは勾配が出ない。そこで密報酬 (`progress`) と疎報酬 (`success`) を併用している。

既定では、交差ができても終了しない (`terminations.py` の `topology_reached` は用意してあるが使っていない)。
交差を作るだけでなく、**壊さずに保つ**ことも学習させるため。

## PPO ([`agents/rsl_rl_ppo_cfg.py`](agents/rsl_rl_ppo_cfg.py))

```python
num_steps_per_env = 10      # 2 エピソード分 (1 エピソード 5 ステップ)
max_iterations = 2000
gamma = 0.95                # 地平が 5 ステップしかないので低め (env_cfg.GAMMA)
lam = 0.95
entropy_coef = 0.01         # 離散選択を連続値の量子化で表すので、探索の崩壊を防ぐ
learning_rate = 1.0e-4      # schedule="adaptive"
clip_param = 0.2
actor_hidden_dims = critic_hidden_dims = [256, 128, 64]
empirical_normalization = False
```

`lift` / `reach` 系のタスクとの最大の違いは、**エピソードは極端に短い (5 ステップ) のに、1 ステップの
実時間コストは非常に重い (686 物理ステップ)** こと。`num_steps_per_env` を 10 と小さくして
1 イテレーションの物理ステップ数を抑え、`gamma` も短い地平に合わせて低めにしている。
