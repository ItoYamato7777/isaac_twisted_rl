"""P4-4 (安全な自動修正) の前後で、変更された .py が「同じ意味」かを確かめる。

ast_equal.py (完全一致) では通らない、次の書き換えだけを同じとみなして比べる:
- 連続した import の文の塊の中の、並べ替え・まとめ・分割 (束縛の多重集合で比べる)
- `import a.b as b` と `from a import b` (Python 3.7 以降、どちらも a.b を import して getattr で束縛する)
- `__all__` の要素の順序
- `functools.lru_cache(maxsize=None)` と `functools.cache` (cache は lru_cache(maxsize=None) の別名)
それ以外の違いがあれば DIFF。使い方: python3 ast_equiv.py [基準の commit (既定 HEAD)]
"""
import ast
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
base = sys.argv[1] if len(sys.argv) > 1 else "HEAD"


def bindings(node):
    out = []
    if isinstance(node, ast.Import):
        for a in node.names:
            if a.asname and "." in a.name:
                mod, _, attr = a.name.rpartition(".")
                out.append((a.asname, 0, mod, attr))
            elif a.asname:
                out.append((a.asname, 0, a.name, None))
            else:
                out.append((a.name.split(".")[0], 0, a.name, None))
    else:
        for a in node.names:
            out.append((a.asname or a.name, node.level, node.module or "", a.name))
    return out


class Norm(ast.NodeTransformer):
    def generic_visit(self, node):
        super().generic_visit(node)
        for field in ("body", "orelse", "finalbody"):
            body = getattr(node, field, None)
            if not isinstance(body, list):
                continue
            new, block = [], None
            for st in body:
                if isinstance(st, (ast.Import, ast.ImportFrom)):
                    if block is None:
                        block = []
                        new.append(block)
                    block.extend(bindings(st))
                else:
                    block = None
                    new.append(st)
            setattr(node, field, [ast.Constant(repr(sorted(x))) if isinstance(x, list) else x for x in new])
        return node

    def visit_Assign(self, node):
        self.generic_visit(node)
        if any(isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets) and isinstance(node.value, (ast.List, ast.Tuple)):
            node.value.elts.sort(key=ast.dump)
        return node

    def visit_Call(self, node):
        self.generic_visit(node)
        f = node.func
        if (isinstance(f, ast.Attribute) and f.attr == "lru_cache" and not node.args and len(node.keywords) == 1
                and node.keywords[0].arg == "maxsize" and isinstance(node.keywords[0].value, ast.Constant)
                and node.keywords[0].value.value is None):
            return ast.Attribute(value=f.value, attr="cache", ctx=ast.Load())
        return node


def norm(src):
    return ast.dump(Norm().visit(ast.parse(src)), include_attributes=False)


files = subprocess.run(["git", "diff", "--name-only", base, "--", "*.py"], cwd=ROOT,
                       capture_output=True, text=True, check=True).stdout.split()
bad = 0
for f in files:
    old = subprocess.run(["git", "show", f"{base}:{f}"], cwd=ROOT, capture_output=True, text=True).stdout
    same = norm(old) == norm((ROOT / f).read_text())
    bad += not same
    print(f"{'same' if same else 'DIFF'}  {f}")
print(f"changed .py: {len(files)}, DIFF: {bad}")
sys.exit(1 if bad else 0)
