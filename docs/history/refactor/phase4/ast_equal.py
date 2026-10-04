"""整形の前後で、変更された .py の AST が一致するかを確かめる (P4-3・H3 の検証)。

使い方: python3 ast_equal.py [基準の commit (既定 HEAD)]
判定: exact = ast.dump (位置の情報を除く) が完全に一致。
      docws = 文字列の定数の行ごとの前後の空白だけが違う (ruff format は docstring の字下げと行末の空白を直す)。
      DIFF  = それ以外。1 件でもあれば exit 1。
"""
import ast
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
base = sys.argv[1] if len(sys.argv) > 1 else "HEAD"


def dump(src, norm):
    tree = ast.parse(src)
    if norm:
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                node.value = "\n".join(line.strip() for line in node.value.strip().splitlines())
    return ast.dump(tree, include_attributes=False)


files = subprocess.run(["git", "diff", "--name-only", base, "--", "*.py"], cwd=ROOT,
                       capture_output=True, text=True, check=True).stdout.split()
counts = {"exact": 0, "docws": 0, "DIFF": 0}
for f in files:
    old = subprocess.run(["git", "show", f"{base}:{f}"], cwd=ROOT, capture_output=True, text=True).stdout
    new = (ROOT / f).read_text()
    kind = "exact" if dump(old, False) == dump(new, False) else "docws" if dump(old, True) == dump(new, True) else "DIFF"
    counts[kind] += 1
    if kind != "exact":
        print(f"{kind:5} {f}")
print(f"changed .py: {len(files)}  {counts}")
sys.exit(1 if counts["DIFF"] else 0)
