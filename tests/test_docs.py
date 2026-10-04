"""文書と構成の食い違いを、静的に検査する。

どれもファイルを読むだけで、Isaac Sim は要らない:

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run python -m pytest tests/test_docs.py

- 版の一覧: `versions/` のフォルダ、`knot_tying/__init__.py` の `_VERSIONS`、`versions/__init__.py` の docstring の一覧が
  一致する。版を足すたびに一覧の更新が漏れてきたため (6 回の版の追加で 3 回)。
- Markdown の相対リンクが、実在するファイルを指す。文書を改名・移動したときに参照が切れるのを捕まえる
  (`paper/Report.md` の改名で 11 本が切れたことがある)。
- 追跡しているファイル (経緯の `docs/history/` を除く) が、Claude Code の一時的な scratchpad (`/tmp/claude-*`) を
  参照しない (`/tmp/claude-<数字>/...` の形のパス)。session が終わると消えるので、参照は必ず切れる。
"""

from __future__ import annotations

import ast
import re
import subprocess
from pathlib import Path
from urllib.parse import unquote

REPO = Path(__file__).resolve().parents[1]
KT = REPO / "src" / "isaac_twisted_rl" / "tasks" / "knot_tying"


def _tracked(*patterns: str) -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "--", *patterns],
        cwd=REPO, capture_output=True, text=True, check=True,
    ).stdout
    return [REPO / p for p in out.splitlines() if (REPO / p).exists()]


def test_version_lists_agree():
    tree = ast.parse((KT / "__init__.py").read_text())
    versions = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "_VERSIONS" for t in node.targets):
            versions = ast.literal_eval(node.value)
    assert versions, "_VERSIONS が見つからない"
    registered = {v.replace("-", "_").lower() for v in versions}

    folders = {p.name for p in (KT / "versions").iterdir() if p.is_dir() and (p / "__init__.py").exists()}

    doc = ast.get_docstring(ast.parse((KT / "versions" / "__init__.py").read_text())) or ""
    listed = set(re.findall(r"^\s+(v\w+)/", doc, flags=re.M))

    assert registered == folders, f"_VERSIONS {sorted(registered)} と versions/ のフォルダ {sorted(folders)} が違う"
    assert listed == folders, f"versions/__init__.py の一覧 {sorted(listed)} と versions/ のフォルダ {sorted(folders)} が違う"


_LINK = re.compile(r"(?<!!)\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")


def _strip_code(text: str) -> str:
    text = re.sub(r"^(```|~~~).*?^\1", "", text, flags=re.M | re.S)
    return re.sub(r"`[^`\n]*`", "", text)


def test_markdown_relative_links_resolve():
    broken = []
    for md in _tracked("*.md"):
        for target in _LINK.findall(_strip_code(md.read_text(errors="replace"))):
            if re.match(r"^[a-z][a-z0-9+.-]*:", target) or target.startswith("#"):
                continue  # URL・メール・同じ文書の中の見出し
            path = unquote(target.split("#", 1)[0])
            if not path:
                continue
            resolved = (REPO / path.lstrip("/")) if path.startswith("/") else (md.parent / path)
            if not resolved.exists():
                broken.append(f"{md.relative_to(REPO)}: {target}")
    assert not broken, "存在しないファイルを指すリンク:\n" + "\n".join(broken)


def test_no_scratchpad_references():
    hits = []
    for f in _tracked():
        rel = f.relative_to(REPO).as_posix()
        if rel.startswith(("docs/history/", "refactor/")) or rel == "tests/test_docs.py" or f.suffix in {".usd", ".png", ".pdf", ".pt"}:
            continue
        try:
            text = f.read_text()
        except (UnicodeDecodeError, IsADirectoryError):
            continue
        for i, line in enumerate(text.splitlines(), 1):
            if re.search(r"/tmp/claude-\d", line):
                hits.append(f"{rel}:{i}")
    assert not hits, "一時的な scratchpad (/tmp/claude-*) への参照:\n" + "\n".join(hits)
