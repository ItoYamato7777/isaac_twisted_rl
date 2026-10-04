#!/usr/bin/env python3
"""PreToolUse (Bash): 2 つの操作を照合する。それ以外のコマンドは何もしない (exit 0、出力なし)。

H2 学習・評価の起動 (train / train.py / eval_chain.py / eval_search.py / eval_robust.py) で、
   作業ツリーに未コミットの .py / .sh / .yaml / .toml があれば確認 (ask) を求める。
   未追跡のファイルは run の git/*.diff に残らず、その run を再現できなくなるため。
H4 共有の資産 (git の管理外で戻せない) を上書きするコマンドを拒否 (deny) する。
   - generate_assets.py を --rope なしで呼ぶ (全プリセットのロープの USD を作り直す)。
   - overhand_script.py の --dump_archive をパスなしで使う (v4〜v7 が読む
     outputs/archive_seed/overhand_{num_links}links.pt に書く)。

照合は「コマンドの先頭の語」で行う。grep・cat などの引数としてファイル名が出るだけのものは対象にしない。
"""
import json
import os
import re
import shlex
import subprocess
import sys

WRAPPERS = {"timeout", "nohup", "nice", "stdbuf", "env", "time", "command", "exec"}
LAUNCH_SCRIPTS = re.compile(r"(^|/)(train|eval_chain|eval_search|eval_robust)\.py$")


def segments(cmd: str):
    """複合コマンドを、単純なコマンドの語の列に分ける。"""
    try:
        lex = shlex.shlex(cmd, posix=True, punctuation_chars=";&|()")
        lex.whitespace_split = True
        tokens = list(lex)
    except ValueError:
        tokens = cmd.split()
    seg = []
    for tok in tokens:
        if tok and set(tok) <= set(";&|()"):
            if seg:
                yield seg
            seg = []
        else:
            seg.append(tok)
    if seg:
        yield seg


def strip_prefix(words):
    """先頭の環境変数の代入と、timeout などの包みを外す。"""
    i = 0
    while i < len(words):
        w = words[i]
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", w):
            i += 1
        elif w in WRAPPERS:
            i += 1
            while i < len(words) and (words[i].startswith("-") or re.match(r"^\d+(\.\d+)?[smhd]?$", words[i])):
                i += 1
        else:
            break
    return words[i:]


def program(words):
    """(実行するもの, 引数) を返す。uv run と python -u を外す。"""
    w = strip_prefix(words)
    if w[:2] == ["uv", "run"]:
        w = w[2:]
        while w and w[0].startswith("-"):
            w = w[1:]
    if w and re.match(r"^(.*/)?python3?(\.\d+)?$", w[0]):
        w = w[1:]
        while w and w[0].startswith("-") and w[0] != "-m":
            w = w[1:]
    return (w[0], w[1:]) if w else ("", [])


def decide(kind: str, reason: str):
    print(json.dumps({"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": kind,
                                             "permissionDecisionReason": reason}}, ensure_ascii=False))
    sys.exit(0)


def main():
    try:
        data = json.load(sys.stdin)
    except Exception:
        return
    cmd = data.get("tool_input", {}).get("command", "")
    if not cmd:
        return
    launch = False
    for words in segments(cmd):
        prog, args = program(words)
        name = os.path.basename(prog)
        if name == "generate_assets.py" and not any(a == "--rope" or a.startswith("--rope=") for a in args):
            decide("deny", "generate_assets.py を --rope なしで呼ぶと、全プリセットのロープの USD (git の管理外の共有資産) を"
                           "作り直す。--rope <名前> で対象を絞る。")
        if name == "overhand_script.py":
            for i, a in enumerate(args):
                bare = a == "--dump_archive" and (i + 1 == len(args) or args[i + 1].startswith("-"))
                if bare or a in ("--dump_archive=",):
                    decide("deny", "overhand_script.py の --dump_archive をパスなしで使うと、v4〜v7 が読む共有の初期アーカイブ "
                                   "outputs/archive_seed/overhand_{num_links}links.pt を上書きする。新しいパスを明示する。")
        if name == "train" or LAUNCH_SCRIPTS.search(prog):
            launch = True
    if not launch:
        return
    cwd = data.get("cwd") or os.environ.get("CLAUDE_PROJECT_DIR") or "."
    try:
        out = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"], cwd=cwd,
                             capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return
    dirty = [l[3:] for l in out.splitlines() if re.search(r"\.(py|sh|ya?ml|toml)$", l)]
    if dirty:
        decide("ask", f"未コミットのコード・設定が {len(dirty)} 件ある ({', '.join(dirty[:5])}{' …' if len(dirty) > 5 else ''})。"
                      "未追跡のファイルは run の git/*.diff に残らず、この run を再現できなくなる。"
                      "先に commit するか、このまま起動するかを確かめる。")


if __name__ == "__main__":
    main()
