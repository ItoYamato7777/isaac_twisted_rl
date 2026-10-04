"""変更された .py が、docstring とコメントの変更だけかを確かめる。

使い方: python3 ast_same.py <基準の commit>   (既定 HEAD)
基準の commit と作業ツリーで、docstring を除いた AST (ast.dump) を比べる。コメントは AST に入らない。
"""
import ast, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
base = sys.argv[1] if len(sys.argv) > 1 else "HEAD"


def strip(tree):
    """docstring と、属性の docstring (代入の後の文字列) を含む、裸の文字列の文を全部 pass にする。"""
    for node in ast.walk(tree):
        for field in ("body", "orelse", "finalbody"):
            body = getattr(node, field, None)
            if isinstance(body, list):
                for i, st in enumerate(body):
                    if isinstance(st, ast.Expr) and isinstance(st.value, ast.Constant) and isinstance(st.value.value, str):
                        body[i] = ast.Pass()
    return ast.dump(tree, include_attributes=False)


files = subprocess.run(["git", "diff", "--name-only", base, "--", "*.py"], cwd=ROOT,
                       capture_output=True, text=True, check=True).stdout.split()
bad = []
for f in files:
    p = ROOT / f
    if not p.exists():
        bad.append(f"{f}: 削除された")
        continue
    old = subprocess.run(["git", "show", f"{base}:{f}"], cwd=ROOT, capture_output=True, text=True)
    if old.returncode:
        bad.append(f"{f}: 新しいファイル")
        continue
    if strip(ast.parse(old.stdout)) != strip(ast.parse(p.read_text())):
        bad.append(f"{f}: docstring 以外の AST が変わった")
print(f"changed .py: {len(files)}, non-docstring changes: {len(bad)}")
for b in bad:
    print("  -", b)
sys.exit(1 if bad else 0)
