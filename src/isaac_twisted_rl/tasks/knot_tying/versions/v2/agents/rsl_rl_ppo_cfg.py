"""v2 の PPO ハイパーパラメータ。MLP / GRU / LSTM の 3 変種。

## なぜ 1 つの環境に 3 つの Runner cfg を置くのか

要求は「Actor, Critic を時系列モデルにした変種も作る」。`OnPolicyRunner` は
`policy_cfg["class_name"]` から方策を **1 つだけ** `eval()` で作る
(`rsl_rl/runners/on_policy_runner.py:418`) ので、1 つのタスクの中で
複数のネットワークを切り替えることはできない。そこで
**環境設定 (観測・報酬・物理) を固定し、エージェント設定だけを変えた
3 タスクを登録して別々に学習させ、比べる**形にした。
タスク ID との対応は `knot_tying/__init__.py` の `_AGENT_VARIANTS`。

なお移植元 TWISTED-RL の「交差数ごとに SAC を切り替える」構造
(`../v1/diff_to_TwistedRL.md` §2) はここでは実現していない。あれは
`eval()` の名前空間の制約 (解決できるのは import 済みの `ActorCritic` /
`ActorCriticRecurrent` のみ) に加え、HER と共有リプレイバッファという
off-policy 固有の仕組みが前提なので PPO には移植できない。
段別の学習は v3 で「段ごとに別タスクを登録し `--resume` で暖機開始する」
カリキュラムとして実装する。

## v1 から変えた点と理由 (すべて TensorBoard の実測にもとづく)

v1 は 157/4000 iteration で打ち切られ、`Train/mean_reward` が
−0.009 -> −0.058 と**下降を続けていた**。

* `entropy_coef` 0.02 -> **0.01** (v0 の値に戻す)。
  v1 は `Policy/mean_noise_std` 1.003 -> **1.628**、`Loss/entropy`
  8.52 -> **11.39** と**どちらも上昇**していた。タスク報酬が純負
  (`failure` −0.0924 が `shaping` +0.0496 を 1.86 倍上回る) だったため、
  **唯一プラスを払う項がエントロピーボーナスになり、最大限ランダムに
  振る舞うことが最適方策になっていた**。v0 (0.01) では
  0.998 -> 0.610 / 8.51 -> 2.08 と正しく収束している。
  v1 が 0.02 に上げた理由は「目標が疎だから探索を保つ」だったが、
  正しい対処は目標を発火させること (`../env_cfg.py` の 3 段ラダー)。
* `gamma` 0.98 -> **0.95** (`env_cfg.GAMMA`)。理由は `env_cfg.GAMMA` の
  docstring 参照 (地平が 6 手に縮んだ + critic が追いつけていなかった)。
  **`RewardsCfg.shaping.params["gamma"]` と必ず一致させること** —
  ポテンシャル整形の最適方策不変性 (Ng et al. 1999) の前提。
  そのため数値はここに書かず `env_cfg.GAMMA` を import している。
* `empirical_normalization` を**廃止し、per-policy の
  `actor_obs_normalization` / `critic_obs_normalization` を False で明示**。
  v1 でこれを True にしたのと `Loss/value_function` の ~1099 への
  スパイク (v0 は 0.17〜2.6) が同時に起きている。正規化が原因と断定は
  できないが「v1 で変えた変数」であり、v2 は保守的に v0 側へ戻す。
  単変数のアブレーションは v3 で行う。
  なお `RslRlPpoActorCriticCfg` ではこの 2 つが `MISSING` なので
  **明示的に渡さないと configclass の検証で落ちる**
  (`isaaclab_rl/rsl_rl/rl_cfg.py`)。`empirical_normalization` 自体は
  deprecated で、設定すると `OnPolicyRunner` が警告を出す
  (`on_policy_runner.py:406-415`)。
* `num_steps_per_env` 16 -> **12**。**`actions_per_episode` (6) の整数倍で
  なければならない** (2 エピソードぶん)。半端な値だとロールアウトが毎回
  違うフェーズでエピソード境界を跨いで切れ、GAE の bootstrap が不安定に
  なる。16 は 6 の倍数ではないので、これは必須の変更。
* `max_iterations` 4000 -> **2000**。v1 は 285〜330 s/iteration で
  4000 iteration = 約 14 日という見積りだった。`SHORT_DURATIONS` (−14.7%)
  と 8 -> 6 手 (−25%) で 1 iteration あたり約 36% 減るが、それでも
  2000 iteration で約 4.5 日。実時間を本当に縮めるのは `num_envs` の
  スループット測定 (`../../next_action.md` 4.8.9 の 2 番目)。
* `obs_groups` を **明示**。`RslRlBaseRunnerCfg.obs_groups` は `MISSING` で、
  v0/v1 は未設定のまま `resolve_obs_groups` の deprecated な
  フォールバックに頼っていた。
* `experiment_name` を変種ごとに分ける。`train.py` が
  `logs/rsl_rl/{experiment_name}/` をログ根にし、`--resume` がそこから
  最新 run を正規表現で拾うため、共有すると **GRU のチェックポイントを
  MLP に読ませる**事故が起きる。`ActorCritic.load_state_dict` は
  `strict=True` が既定なので (`rsl_rl/modules/actor_critic.py:154-167`)
  その場合は `RuntimeError` になる。
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoActorCriticRecurrentCfg,
    RslRlPpoAlgorithmCfg,
)

from ..env_cfg import GAMMA

_OBS_GROUPS = {"policy": ["policy"], "critic": ["policy"]}
"""観測グループの対応。critic も policy と同じ観測を見る (特権観測は無い)。

`RslRlBaseRunnerCfg.obs_groups` は `MISSING` で、未設定だと
`resolve_obs_groups` が deprecated なフォールバック (policy -> ["policy"]、
critic は policy と同じ) を適用して警告を出す。挙動は同じだが、
将来 critic に特権観測を足すときの変更点をここに明示しておく。
"""

_HIDDEN = [512, 256, 128]
"""MLP 部の隠れ層。**v1 から据え置き。**

変えないのは、v3 で段別カリキュラム (`-v3-Stage1/2/3`) を組むときに
`OnPolicyRunner.load` が `load_state_dict` を **strict のまま** 呼ぶ
(`rsl_rl/runners/on_policy_runner.py:310`) ため。観測次元とネットワーク
形状が段の間で揃っていないと `--resume` による暖機開始ができない。
再帰版では RNN の出力 (`rnn_hidden_dim`) を受ける MLP の形になる。
"""


def _algorithm(num_mini_batches: int) -> RslRlPpoAlgorithmCfg:
    """PPO 本体の設定。変種で変えるのは `num_mini_batches` だけ。

    再帰方策では `RolloutStorage.recurrent_mini_batch_generator` が
    `mini_batch_size = num_envs // num_mini_batches`
    (`rsl_rl/storage/rollout_storage.py:211`) とするので、**ミニバッチが
    時間ではなく環境で切られる**。MLP 側の
    `num_envs * num_steps_per_env` とは意味が違い、同じ分割数だと
    1 バッチの実効サンプル数が桁で減る。そのため再帰側は分割数を
    減らして 1 バッチの環境数を確保する。
    """
    return RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=num_mini_batches,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=GAMMA,  # env_cfg.GAMMA を共有 (報酬整形と必ず一致させる)
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class KnotTyingPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """v2 既定: MLP 方策 (`Isaac-Twisted-RL-Twist-v2`)。

    3 変種のうち**これがベースライン**。再帰版より先に回して、
    3 段の疎報酬ラダーが実際に発火するか (`Episode_Reward/success_c1` が
    立ち上がるか) を確認する。
    """

    num_steps_per_env = 12
    max_iterations = 2000
    save_interval = 50
    experiment_name = "knot_tying_v2"
    obs_groups = _OBS_GROUPS
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=_HIDDEN,
        critic_hidden_dims=_HIDDEN,
        activation="elu",
    )
    algorithm = _algorithm(num_mini_batches=4)


@configclass
class KnotTyingRNNPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """v2 再帰版 (GRU): `Isaac-Twisted-RL-Twist-v2-RNN`。

    Actor と Critic がそれぞれ独立した GRU を持つ
    (`ActorCriticRecurrent` は `memory_a` / `memory_c` を別々に作る)。

    ## `rnn_type` の指定は文字列一致で、間違えても例外が出ない

    `rsl_rl.networks.Memory` は
    `rnn_cls = nn.GRU if type.lower() == "gru" else nn.LSTM`
    なので、**"gru" 以外のあらゆる文字列が黙って LSTM になる**。
    `"GRU"` は `.lower()` が効くので通るが、typo (`"grus"` など) は
    気付けない。GRU のつもりで LSTM を学習してしまう事故を避けるため
    ここは小文字の `"gru"` に固定し、学習開始時のログ
    (`ActorCriticRecurrent.__init__` が Actor/Critic の RNN を print する)
    で `GRU(...)` と出ていることを必ず目視確認すること。

    ## `rnn_hidden_dim` であって `rnn_hidden_size` ではない

    `ActorCriticRecurrent` は `rnn_hidden_size` を kwargs で受けても
    **`rnn_hidden_dim` が既定の 256 のときだけ**上書きする
    (`rsl_rl/modules/actor_critic_recurrent.py` の DeprecationWarning 分岐)。
    `RslRlPpoActorCriticRecurrentCfg` に `rnn_hidden_size` フィールドは
    無いので configclass 経由では渡せないが、古い例を写すときは注意。

    ## この環境で再帰が有効とは考えにくい (それでも比べる価値はある)

    観測はリンク 3D 位置 20 個 + 交差表 + クラス one-hot で **ほぼ完全状態**
    であり、報酬は `settle_out` (0.3 s) + `relax` (0.5 s) の静定後に測るので
    **覚えておくべき速度も残っていない** (ほぼマルコフ)。さらに
    1 エピソードが 6 手しかなく、`recurrent_mini_batch_generator` は
    エピソード境界で軌跡を切ってパディングするため、隠れ状態が育つ窓が
    そもそも 6 ステップしかない。

    一応の理屈はある: `crossing_feats` は交差 5 個ぶんの固定長で、
    実際の交差数が変わるとパディング部分の意味が曖昧になるので、
    再帰的な critic がそれを平滑化する可能性はある。ただし薄い。

    **MLP と同等以下になる想定**で回し、結果は `../README.md` に
    負の知見として残す。
    """

    num_steps_per_env = 12
    max_iterations = 2000
    save_interval = 50
    experiment_name = "knot_tying_v2_rnn"
    obs_groups = _OBS_GROUPS
    policy = RslRlPpoActorCriticRecurrentCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=_HIDDEN,
        critic_hidden_dims=_HIDDEN,
        activation="elu",
        rnn_type="gru",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
    )
    algorithm = _algorithm(num_mini_batches=2)


@configclass
class KnotTyingLSTMPPORunnerCfg(KnotTyingRNNPPORunnerCfg):
    """v2 再帰版 (LSTM): `Isaac-Twisted-RL-Twist-v2-LSTM`。

    GRU 版との差は `rnn_type` と `experiment_name` だけ。LSTM はゲートが
    3 つ (GRU は 2 つ) でパラメータが約 4/3 になるが、この環境では
    `KnotTyingRNNPPORunnerCfg` の docstring にある理由で差は出ない見込み。

    `policy` を丸ごと書き直しているのは、configclass の継承で
    ネストした設定の 1 フィールドだけを差し替えるより、
    **何を渡しているかがその場で読める**ほうが事故が少ないため
    (特に `rnn_type` は間違えても例外が出ない)。
    """

    experiment_name = "knot_tying_v2_lstm"
    policy = RslRlPpoActorCriticRecurrentCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=_HIDDEN,
        critic_hidden_dims=_HIDDEN,
        activation="elu",
        rnn_type="lstm",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
    )
