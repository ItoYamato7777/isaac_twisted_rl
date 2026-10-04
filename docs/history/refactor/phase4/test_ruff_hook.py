"""ruff_after_edit.py (H3) に想定の PostToolUse の JSON を渡して試す。

一時的なプロジェクト (本物の ruff.toml の写し) の中にファイルを作り、CLAUDE_PROJECT_DIR をそこに向ける。
"""
import json, os, shutil, subprocess, tempfile

REPO = "/home/itoyama/work/isaac_twisted_rl"
HOOK = f"{REPO}/.claude/hooks/ruff_after_edit.py"
root = tempfile.mkdtemp()
shutil.copy(f"{REPO}/ruff.toml", root)
outside = tempfile.mkdtemp()

LAUNCH = '''import argparse
from isaaclab.app import AppLauncher

app = AppLauncher(argparse.Namespace()).app
import torch
import gymnasium as gym
import isaaclab_tasks

print(torch, gym, isaaclab_tasks)
'''
CASES = [  # (相対パス or 絶対パス, 中身, 期待: 整形されるか, 指摘に含まれる文字列 or None, 中身が変わってはいけない行)
    ("src/pkg/lib.py", "x = {'a':1}\n", True, None, None),
    ("src/pkg/unused.py", "import os\nimport sys\n\nprint(sys.argv)\n", False, "F401", "import os"),
    ("src/isaac_twisted_rl/scripts/rsl_rl/train.py", LAUNCH, False, None, "import torch\nimport gymnasium as gym\nimport isaaclab_tasks"),
    ("src/pkg/launch_like.py", LAUNCH, False, "E402", "import torch\nimport gymnasium as gym\nimport isaaclab_tasks"),  # 対照: 除外の外なら指摘される
    ("src/pkg/notes.md", "x = {'a':1}\n", False, None, "x = {'a':1}"),
    (os.path.join(outside, "far.py"), "x = {'a':1}\n", False, None, "x = {'a':1}"),
]
bad = 0
for rel, body, want_fmt, want_find, keep in CASES:
    p = rel if os.path.isabs(rel) else os.path.join(root, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "w").write(body)
    inp = json.dumps({"hook_event_name": "PostToolUse", "tool_name": "Write", "tool_input": {"file_path": p}, "cwd": root})
    r = subprocess.run(["python3", HOOK], input=inp, capture_output=True, text=True, env={**os.environ, "CLAUDE_PROJECT_DIR": root})
    ctx = json.loads(r.stdout)["hookSpecificOutput"]["additionalContext"] if r.stdout.strip() else ""
    after = open(p).read()
    ok = r.returncode == 0
    ok &= (after != body) == want_fmt and (("整形した" in ctx) == want_fmt)
    ok &= (want_find in ctx) if want_find else ("ruff check" not in ctx)
    ok &= want_find == "E402" or ("I001" not in ctx and "E402" not in ctx)
    ok &= (keep in after) if keep else True
    bad += not ok
    print(f"{'OK' if ok else 'NG'}  fmt={after != body!s:5} ctx={ctx[:90]!r}  {rel}")
print(f"{len(CASES) - bad}/{len(CASES)} as expected")
