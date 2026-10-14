"""チェックポイントの方策を TorchScript / ONNX に書き出す (Isaac Sim なし、CPU)。

`play.py` も起動時に同じ書き出しをするが、Kit を起動して Play 環境を作り、その後は描画のループに入るので、
GPU が学習で埋まっているときや書き出しだけしたいときに使えない。ここでは run の `params/agent.yaml` から
方策 (`StageGatedActorCritic`) を組み立ててチェックポイントを読み、`play.py` と同じ Isaac Lab の exporter
(`isaaclab_rl/rsl_rl/exporter.py`) で書き出す。出力の場所と名前も `play.py` と同じ
(`<チェックポイントのフォルダ>/exported/policy.pt`・`policy.onnx`)::

    uv run python src/isaac_twisted_rl/scripts/rsl_rl/export_policy.py \
        --checkpoint logs/rsl_rl/knot_tying_v11/<run>/model_1000.pt

* exporter は `isaaclab_rl` のパッケージとしてではなくファイルとして読む。パッケージの `__init__` が
  `isaaclab.envs` を import し、Kit が要るため。
* 書き出した後、段ごとの one-hot を付けた乱数の観測で、TorchScript と ONNX の出力が `act_inference` (決定論の
  平均の行動) と一致するかを確かめる (ONNX は onnxruntime があるときだけ)。
* `exported/export_info.json` に、元のチェックポイント・iteration・観測と行動の次元・確かめた誤差を書く。
  同じフォルダに別のチェックポイントを書き出すと上書きされるので、どれを書き出したかをここで分かるようにする。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import sys

import torch
import yaml

_KT_COMMON = pathlib.Path(__file__).resolve().parents[2] / "tasks" / "knot_tying" / "common"


def _load_by_path(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _isaaclab_exporter():
    import isaaclab

    root = pathlib.Path(isaaclab.__file__).resolve().parent
    candidates = sorted(root.rglob("isaaclab_rl/rsl_rl/exporter.py")) + sorted(
        root.parent.rglob("isaaclab_rl/rsl_rl/exporter.py")
    )
    if not candidates:
        raise FileNotFoundError("isaaclab_rl/rsl_rl/exporter.py が見つからない")
    return _load_by_path("_isaaclab_rl_exporter", candidates[0])


def _stage_obs(n: int, obs_dim: int, stage_dims: int, seed: int = 0) -> torch.Tensor:
    """段 0〜3 を順に回した one-hot を末尾に付けた乱数の観測 `(n, obs_dim)`。"""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, obs_dim, generator=g)
    x[:, -stage_dims:] = 0.0
    x[torch.arange(n), obs_dim - stage_dims + torch.arange(n) % stage_dims] = 1.0
    return x


def _obs_shapes(state: dict) -> tuple[dict[str, torch.Tensor], dict[str, list[str]]]:
    """方策を組み立てるための観測の形とグループ。critic の幅はチェックポイントから読む。

    v14 から critic は actor と違う幅 (特権観測を前に足す) になる。critic を actor の幅で作ると `load_state_dict`
    (strict) が落ちるので、critic の入力の幅を `critic.heads.0.0.weight` から読み、別のグループにする。
    書き出すのは actor だけなので、critic の観測の中身は使わない。v4〜v13 は critic と actor が同じ幅で、今までと同じ形になる。
    """
    obs_dim = state["actor.heads.0.0.weight"].shape[1]
    critic_dim = state["critic.heads.0.0.weight"].shape[1]
    if critic_dim == obs_dim:
        return {"policy": torch.zeros(1, obs_dim)}, {"policy": ["policy"], "critic": ["policy"]}
    obs = {"policy": torch.zeros(1, obs_dim), "critic": torch.zeros(1, critic_dim)}
    return obs, {"policy": ["policy"], "critic": ["critic"]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a stage-gated policy checkpoint to TorchScript / ONNX.")
    parser.add_argument("--checkpoint", required=True, help="model_<it>.pt")
    parser.add_argument("--out_dir", default=None, help="Output folder (default: <checkpoint folder>/exported).")
    args = parser.parse_args()

    ckpt = pathlib.Path(args.checkpoint).resolve()
    run_dir = ckpt.parent
    out_dir = pathlib.Path(args.out_dir).resolve() if args.out_dir else run_dir / "exported"
    agent_cfg = yaml.unsafe_load((run_dir / "params" / "agent.yaml").read_text())
    pcfg = dict(agent_cfg["policy"])
    if pcfg.pop("class_name") != "StageGatedActorCritic":
        raise ValueError("StageGatedActorCritic の run だけを扱う")

    sys.path.insert(0, str(_KT_COMMON))
    from policies.stage_gated import StageGatedActorCritic

    loaded = torch.load(ckpt, map_location="cpu", weights_only=False)
    state = loaded["model_state_dict"]
    obs_dim = state["actor.heads.0.0.weight"].shape[1]
    num_actions = state["std"].shape[-1]
    obs, groups = _obs_shapes(state)
    policy = StageGatedActorCritic(obs, groups, num_actions, **pcfg)
    policy.load_state_dict(state)
    policy.eval()

    exporter = _isaaclab_exporter()
    out_dir.mkdir(parents=True, exist_ok=True)
    exporter.export_policy_as_jit(policy, normalizer=None, path=str(out_dir), filename="policy.pt")
    exporter.export_policy_as_onnx(policy, normalizer=None, path=str(out_dir), filename="policy.onnx")

    x = _stage_obs(64, obs_dim, pcfg.get("stage_obs_dims", 4))
    with torch.inference_mode():
        ref = policy.act_inference({"policy": x})
        jit_err = float((torch.jit.load(str(out_dir / "policy.pt"))(x) - ref).abs().max())
    onnx_err = None
    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(str(out_dir / "policy.onnx"))
        name = sess.get_inputs()[0].name
        got = torch.cat([torch.from_numpy(sess.run(None, {name: x[i : i + 1].numpy()})[0]) for i in range(x.shape[0])])
        onnx_err = float((got - ref).abs().max())
    except ImportError:
        pass

    info = {
        "checkpoint": str(ckpt),
        "iteration": loaded.get("iter"),
        "obs_dim": obs_dim,
        "num_actions": num_actions,
        "num_stages": pcfg.get("num_stages"),
        "stage_obs_dims": pcfg.get("stage_obs_dims"),
        "max_abs_err_jit": jit_err,
        "max_abs_err_onnx": onnx_err,
    }
    (out_dir / "export_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False))
    print(json.dumps(info, indent=2, ensure_ascii=False))
    print(f"[export] {out_dir / 'policy.pt'}, {out_dir / 'policy.onnx'}")


if __name__ == "__main__":
    main()
