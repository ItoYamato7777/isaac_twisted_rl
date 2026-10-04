#!/usr/bin/env python3
"""PostToolUse (Edit|Write): 編集した .py を ruff で整形し、残った lint の指摘を Claude に返す (H3)。

整形 (`ruff format`) は AST を変えないので挙動は変わらない。lint は指摘を返すだけで `--fix` はしない
(起動スクリプトの import の順序や Warp のカーネルの中の書き方を、自動の修正が壊すことがあるため)。
規則と対象外のファイルは ruff.toml に従う (`--force-exclude` で、明示したファイルにも除外を効かせる)。
PostToolUse は操作を止められないので、どの失敗でも exit 0。
"""

import json
import os
import subprocess
import sys

RUFF = ["uvx", "ruff@0.16.10"]


def run(args, cwd):
    return subprocess.run(RUFF + args, cwd=cwd, capture_output=True, text=True, timeout=25)


def main():
    try:
        data = json.load(sys.stdin)
    except Exception:
        return
    path = data.get("tool_input", {}).get("file_path", "")
    root = os.environ.get("CLAUDE_PROJECT_DIR") or data.get("cwd") or ""
    if not path.endswith(".py") or not root or not os.path.isfile(path):
        return
    path, root = os.path.realpath(path), os.path.realpath(root)
    if os.path.commonpath([path, root]) != root:
        return
    notes = []
    try:
        fmt = run(["format", "--force-exclude", path], root)
        if "1 file reformatted" in fmt.stdout + fmt.stderr:
            notes.append(
                f"ruff format が {os.path.relpath(path, root)} を整形した (AST は同じ)。続けて編集する前に読み直す。"
            )
        chk = run(["check", "--force-exclude", "--output-format", "concise", path], root)
        found = [
            line
            for line in chk.stdout.splitlines()
            if line.startswith(path) or line.startswith(os.path.relpath(path, root))
        ]
        if found:
            notes.append("ruff check の指摘 (自動では直していない):\n" + "\n".join(found[:20]))
    except Exception:
        return
    if notes:
        print(
            json.dumps(
                {"hookSpecificOutput": {"hookEventName": "PostToolUse", "additionalContext": "\n".join(notes)}},
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
