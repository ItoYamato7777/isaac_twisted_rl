"""`scripts/rsl_rl/cli_args.py` の試験。Isaac Sim なしで回る (cli_args は isaaclab を型注釈でしか使わない)。

ログの根は `logs/rsl_rl/<agent_cfg.experiment_name>` (`train.py`・`play.py`)。引数を渡さないときにログ名が
変わると、既存の run を `--resume`・`play` で探せなくなるので、それを固定する。
"""

import argparse
from types import SimpleNamespace

from isaac_twisted_rl.scripts.rsl_rl import cli_args


def _agent_cfg():
    return SimpleNamespace(
        seed=42,
        resume=False,
        load_run=".*",
        load_checkpoint="model_.*.pt",
        run_name="",
        logger="tensorboard",
        experiment_name="knot_tying_v6",
    )


def _parse(argv):
    parser = argparse.ArgumentParser()
    cli_args.add_rsl_rl_args(parser)
    args, _ = parser.parse_known_args(argv)
    return args


def test_experiment_name_unchanged_without_flag():
    cfg = cli_args.update_rsl_rl_cfg(_agent_cfg(), _parse(["--run_name", "s42"]))
    assert cfg.experiment_name == "knot_tying_v6"
    assert cfg.run_name == "s42"


def test_experiment_name_flag_overrides_cfg():
    cfg = cli_args.update_rsl_rl_cfg(_agent_cfg(), _parse(["--experiment_name", "knot_tying_v6_ablation"]))
    assert cfg.experiment_name == "knot_tying_v6_ablation"


def test_hydra_style_args_are_left_for_hydra():
    # train.py は parse_known_args の残りを Hydra に渡す。agent.experiment_name=... はここで消費しない
    parser = argparse.ArgumentParser()
    cli_args.add_rsl_rl_args(parser)
    _, rest = parser.parse_known_args(["agent.experiment_name=x"])
    assert rest == ["agent.experiment_name=x"]
