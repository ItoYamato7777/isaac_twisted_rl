"""段ごとに独立した actor / critic / std を持つ方策 (TWISTED-RL-C 相当)。v4 で追加。

TWISTED-RL は、高レベルの遷移を「元の状態の交差数」でグループに分け、グループごとに
SAC エージェントを 1 つ学習する。分け方 G (1 つ) / A (移動の種類) / C (交差数) / AC の
うち、**C が全テストセットで最良** だった (explainer `twisted-rl.html` §04, §09)。
v4 はこれを「段」(`topology.BfsPotential.stage`、固結びへの経路上の交差数に当たる) で行う。

## 1 つの PPO の中に K 個のエージェントを置く

`OnPolicyRunner` は方策を 1 つしか作れない。そこで actor・critic・行動の std を段ごとに
K 組持ち、**観測の末尾にある段の one-hot で振り分ける** (hard gating)。各サンプルは自分の
段のネットワークしか通らないので、勾配は他の段のパラメータへ流れない。パラメータの上では
K 個の独立したエージェントと同じで、共有しているのは次の 2 つだけ:

* optimizer の学習率 (適応スケジュールは全サンプルの KL 平均で決まる)
* ミニバッチ内の advantage の正規化

`num_stages=1` にすると振り分けが無くなり、`ActorCritic` と同じ形 (論文の G 版) になる。

## 観測の約束

* 段の one-hot (`stage_obs_dims` 次元) は actor・critic どちらの観測でも **最後の列**
  に置くこと (`mdp.knot_stage_one_hot` を観測グループの最後の項にする)。
* 段が `num_stages` 以上 (固結び) なら最後のエージェントを使う (学習中は固結びで
  エピソードが終わるので、その観測で行動を選ぶことは無い)。
* 観測の正規化は使えない (one-hot の列が平行移動・拡大され、argmax が変わりうる)。

## export

`self.actor` を振り分け込みの `GatedMLP` にしてあるので、Isaac Lab の
`export_policy_as_jit` / `export_policy_as_onnx` (`policy.actor` をコピーして
`torch.jit.script` する) がそのまま動く。

## rsl_rl への登録

`OnPolicyRunner` は `eval(policy_cfg["class_name"])` で方策クラスを作る
(`rsl_rl/runners/on_policy_runner.py:418`)。解決できるのは同モジュールが import 済みの
名前だけなので、`register_with_rsl_rl()` でそのモジュールの名前空間へ登録する。
v4 の agents 設定モジュールが import 時に呼ぶので、train.py / play.py は変えなくてよい。

このモジュールは isaaclab を import しない (`tests/test_stage_gated_policy.py` から
Isaac Sim 無しで検証できる)。
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.modules import ActorCritic
from rsl_rl.networks import MLP


class GatedMLP(nn.Module):
    """K 個の MLP を持ち、入力の末尾 `stage_dims` 列の argmax で 1 つを選んで出力する。

    全ヘッドを計算してから選ぶ (段ごとに行を分けるより単純で、K = 3 なら十分速い)。
    選ばれなかったヘッドの出力は捨てるので、そのパラメータには勾配が流れない。
    """

    def __init__(self, heads: list[nn.Module], stage_dims: int):
        super().__init__()
        self.heads = nn.ModuleList(heads)
        self.num_heads = len(heads)
        self.stage_dims = stage_dims

    def __getitem__(self, idx: int) -> nn.Module:
        """先頭のヘッドの層を返す (全ヘッドで形は同じ)。

        Isaac Lab の `export_policy_as_onnx` が入力の次元を `policy.actor[0].in_features`
        で読むため (`isaaclab_rl/rsl_rl/exporter.py`)。
        """
        return self.heads[0][idx]

    def stage_index(self, x: torch.Tensor) -> torch.Tensor:
        """`(..., obs)` -> 使うヘッドの番号 `(...)`。範囲外 (固結び) は最後のヘッド。"""
        return x[..., -self.stage_dims :].argmax(dim=-1).clamp(max=self.num_heads - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_heads == 1:
            return self.heads[0](x)
        outs: list[torch.Tensor] = []
        for head in self.heads:
            outs.append(head(x))
        stacked = torch.stack(outs, dim=-2)  # (..., K, D)
        idx = self.stage_index(x).unsqueeze(-1).unsqueeze(-1)
        idx = idx.expand(idx.shape[:-1] + (stacked.shape[-1],))  # (..., 1, D)
        return stacked.gather(-2, idx).squeeze(-2)


class StageGatedActorCritic(ActorCritic):
    """段ごとに独立した actor / critic / std を持つ `ActorCritic`。

    Args (ActorCritic に加えて):
        num_stages: エージェントの数 K。v4 は 3 (段 0 / 1 / 2)。1 なら G 版。
        stage_obs_dims: 観測の末尾にある段の one-hot の次元 (v4 は段 0〜3 の 4)。
    """

    is_recurrent = False

    def __init__(
        self,
        obs,
        obs_groups,
        num_actions,
        num_stages: int = 3,
        stage_obs_dims: int = 4,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        **kwargs,
    ):
        if actor_obs_normalization or critic_obs_normalization:
            raise ValueError("StageGatedActorCritic は観測の正規化を使えない (段の one-hot が崩れる)")
        if num_stages < 1 or stage_obs_dims < num_stages:
            raise ValueError(f"num_stages={num_stages}, stage_obs_dims={stage_obs_dims} が不正")
        super().__init__(
            obs,
            obs_groups,
            num_actions,
            actor_obs_normalization=False,
            critic_obs_normalization=False,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
            noise_std_type=noise_std_type,
            **kwargs,
        )
        self.num_stages = num_stages
        self.stage_obs_dims = stage_obs_dims

        num_actor_obs = sum(obs[g].shape[-1] for g in obs_groups["policy"])
        num_critic_obs = sum(obs[g].shape[-1] for g in obs_groups["critic"])
        # 基底が作った 1 組は捨てて、段ごとの K 組に置き換える。
        self.actor = GatedMLP(
            [MLP(num_actor_obs, num_actions, actor_hidden_dims, activation) for _ in range(num_stages)],
            stage_obs_dims,
        )
        self.critic = GatedMLP(
            [MLP(num_critic_obs, 1, critic_hidden_dims, activation) for _ in range(num_stages)],
            stage_obs_dims,
        )
        if noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_stages, num_actions))
        else:  # "log" (それ以外は基底が例外にしている)
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_stages, num_actions)))
        print(f"StageGatedActorCritic: {num_stages} 段 x (actor {actor_hidden_dims}, critic {critic_hidden_dims})")

    def update_distribution(self, obs):
        mean = self.actor(obs)
        stage = self.actor.stage_index(obs)
        if self.noise_std_type == "scalar":
            std = self.std[stage]
        else:
            std = torch.exp(self.log_std[stage])
        self.distribution = Normal(mean, std)


def register_with_rsl_rl() -> None:
    """`OnPolicyRunner` の `eval(class_name)` から `"StageGatedActorCritic"` を引けるようにする。"""
    import rsl_rl.runners.on_policy_runner as on_policy_runner

    on_policy_runner.StageGatedActorCritic = StageGatedActorCritic
