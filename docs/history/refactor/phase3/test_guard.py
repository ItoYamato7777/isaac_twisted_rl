"""guard_bash.py に想定の PreToolUse の JSON を渡し、判定を表にする。

cwd は一時的な git リポジトリ 2 つ (未追跡の .py がある dirty と、空の clean) にする。
本体の作業ツリーの状態で「ask」の期待が変わらないようにするため。
"""
import json, os, subprocess, tempfile

HOOK = "/home/itoyama/work/isaac_twisted_rl/.claude/hooks/guard_bash.py"
tmp = tempfile.mkdtemp()
REPO, clean = os.path.join(tmp, "dirty"), os.path.join(tmp, "clean")
for d in (REPO, clean):
    os.makedirs(d)
    subprocess.run(["git", "init", "-q", d], check=True)
open(os.path.join(REPO, "a.py"), "w").close()

CASES = [  # (コマンド, cwd, 期待)
    ("uv run train --task Isaac-Twisted-RL-Knot-v6 --headless --seed 42", REPO, "ask"),
    ("PYTHONUNBUFFERED=1 timeout -k 30 7200 uv run train --task X > log 2>&1", REPO, "ask"),
    ("nohup .venv/bin/train --task X &", REPO, "ask"),
    ("uv run python -u src/isaac_twisted_rl/scripts/rsl_rl/eval_chain.py --task X", REPO, "ask"),
    ("cd /x && uv run python -u src/isaac_twisted_rl/scripts/rsl_rl/train.py --task X", REPO, "ask"),
    ("src/isaac_twisted_rl/scripts/run_job.sh --name v6_s42 --timeout 10h -- uv run train --task X --seed 42", REPO, "ask"),
    ("bash src/isaac_twisted_rl/scripts/run_job.sh --name g --timeout 1h --detach -- uv run python -u demos/generate_assets.py --headless", REPO, "deny"),
    ("src/isaac_twisted_rl/scripts/run_job.sh --name g --timeout 1h -- uv run python -u demos/overhand_script.py --task X --dump_archive", REPO, "deny"),
    ("src/isaac_twisted_rl/scripts/run_job.sh --name g --timeout 1h -- uv run python -u demos/generate_assets.py --rope fine", REPO, "pass"),
    ("python src/isaac_twisted_rl/tasks/knot_tying/demos/generate_assets.py --headless", REPO, "deny"),
    ("uv run python -u demos/generate_assets.py --rope fine", REPO, "pass"),
    ("uv run python -u demos/generate_assets.py --rope=fine --only rope", REPO, "pass"),
    ("uv run python -u src/isaac_twisted_rl/tasks/knot_tying/demos/overhand_script.py --task X --headless --dump_archive", REPO, "deny"),
    ("uv run python -u demos/overhand_script.py --task X --dump_archive --headless", REPO, "deny"),
    ("uv run python -u demos/overhand_script.py --task X --dump_archive outputs/archive_seed/new.pt --headless", REPO, "pass"),
    ("uv run python -u demos/overhand_script.py --task X --headless", REPO, "pass"),
    ("grep -n train src/isaac_twisted_rl/scripts/rsl_rl/train.py", REPO, "pass"),
    ("cat src/isaac_twisted_rl/scripts/rsl_rl/eval_chain.py", REPO, "pass"),
    ("git log --oneline -- src/isaac_twisted_rl/scripts/rsl_rl/train.py", REPO, "pass"),
    ("PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run python -m pytest tests/ -q", REPO, "pass"),
    ("ls logs/rsl_rl/knot_tying_v6", REPO, "pass"),
]
if clean:
    CASES += [
        ("uv run train --task Isaac-Twisted-RL-Knot-v6 --headless --seed 42", clean, "pass"),
        ("uv run python -u src/isaac_twisted_rl/scripts/rsl_rl/eval_chain.py --task X", clean, "pass"),
        ("src/isaac_twisted_rl/scripts/run_job.sh --name v6_s42 --timeout 10h -- uv run train --task X", clean, "pass"),
    ]

bad = 0
for cmd, cwd, want in CASES:
    inp = json.dumps({"hook_event_name": "PreToolUse", "tool_name": "Bash", "tool_input": {"command": cmd}, "cwd": cwd})
    r = subprocess.run(["python3", HOOK], input=inp, capture_output=True, text=True)
    got = json.loads(r.stdout)["hookSpecificOutput"]["permissionDecision"] if r.stdout.strip() else "pass"
    ok = got == want and r.returncode == 0
    bad += not ok
    print(f"{'OK ' if ok else 'NG '} {got:5} (want {want:5}) {'clean' if cwd != REPO else 'dirty'}  {cmd}")
print(f"{len(CASES) - bad}/{len(CASES)} as expected")
