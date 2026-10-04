"""登録された全タスクの環境設定と PPO 設定を、parse_env_cfg / load_cfg_from_registry で作って JSON に出す。

P4-9 (版をまたぐ import を公開名に) の前後で設定の値が同じことを確かめるため。
前のコードは git worktree に置き、PYTHONPATH=<worktree>/src で読ませる (出力の isaac_twisted_rl_file で確かめる)。
リポジトリの場所に依存する文字列 (rope study の SEED_DIR など) は <REPO> に、repr のアドレスは 0x? に置き換えてから比べる。

    run_job.sh --name p49_cfg_after --timeout 30m --detach -- \
        uv run python -u docs/history/refactor/phase4/dump_env_cfgs.py --headless --out docs/history/refactor/phase4/env_cfgs_after.json
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--out", required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import json  # noqa: E402
import pathlib  # noqa: E402
import re  # noqa: E402
import traceback  # noqa: E402

import gymnasium as gym  # noqa: E402
import isaac_twisted_rl  # noqa: E402
import isaac_twisted_rl.tasks  # noqa: E402, F401
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry, parse_env_cfg  # noqa: E402

root = str(pathlib.Path(isaac_twisted_rl.__file__).resolve().parents[2])
# どちらのコードを読んだかの印 (<REPO> に置き換える前の生のパスと、P4-9 で足した名前の有無) をログに出す
import importlib  # noqa: E402

v5 = importlib.import_module("isaac_twisted_rl.tasks.knot_tying.versions.v5.env_cfg")
print("isaac_twisted_rl:", isaac_twisted_rl.__file__, "v5.play_post_init:", hasattr(v5, "play_post_init"), flush=True)
out = {"isaac_twisted_rl_file": isaac_twisted_rl.__file__, "tasks": {}}
for tid in sorted(t for t in gym.registry if t.startswith("Isaac-Twisted-RL")):
    rec = {}
    try:
        rec["env"] = parse_env_cfg(tid, device="cuda:0").to_dict()
    except Exception:
        rec["env_error"] = traceback.format_exc(limit=3)
    try:
        rec["agent"] = load_cfg_from_registry(tid, "rsl_rl_cfg_entry_point").to_dict()
    except Exception:
        rec["agent_error"] = traceback.format_exc(limit=3)
    out["tasks"][tid] = rec
    print(tid, "ok" if "env" in rec and "agent" in rec else "ERROR", flush=True)

text = json.dumps(out, indent=1, sort_keys=True, ensure_ascii=False, default=repr).replace(root, "<REPO>")
text = re.sub(r"0x[0-9a-f]+", "0x?", text)  # repr のアドレスは実行ごとに変わる
pathlib.Path(args.out).write_text(text)
print("wrote", args.out, len(out["tasks"]), "tasks", flush=True)
app.close()
