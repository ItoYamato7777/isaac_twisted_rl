"""移動 (commit 62b251b) と改名 (dbf9a97) で切れた Markdown・HTML の相対リンクを、新しい場所へ書き換える。

使い方: python3 fix_links.py <file> [<file> ...]   (dry run は --dry)
- 移動したファイルの中のリンクは、移動前の場所からの相対として解いてから、今の場所からの相対に直す。
- 解いた先が移動・改名されたファイルなら、新しいパスに置き換える。
- 書き換えるのはリンクの括弧の中と HTML の href だけ。本文は変えない。
"""
import os, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
KT = "src/isaac_twisted_rl/tasks/knot_tying"
MOVES = {
    f"{KT}/next_action.md": "docs/history/next_action.md",
    f"{KT}/rope_study/HANDOFF.md": "docs/history/rope_study_HANDOFF.md",
    f"{KT}/versions/v1/diff_to_TwistedRL.md": "docs/history/v1_diff_to_TwistedRL.md",
    f"{KT}/versions/v3/HTML/v3-implementation.html": "docs/history/v3_html/v3-implementation.html",
    f"{KT}/versions/v3/HTML/v3-vs-papers.html": "docs/history/v3_html/v3-vs-papers.html",
    f"{KT}/versions/v3/Result_report_v3.md": "docs/experiments/2026-09-28_v3_main.md",
    f"{KT}/versions/v6/Result_Report_v6.md": "docs/experiments/2026-10-01_v4-v6_overnight.md",
    f"{KT}/rope_study/Report_rope_models.md": "docs/experiments/2026-10-02_rope_models.md",
    "paper/Report_Search.md": "docs/experiments/2026-10-03_inference_search.md",
    "paper/Report.md": "paper/Report_Paper.md",
}
NEW2OLD = {v: k for k, v in MOVES.items() if not k.endswith("Report.md")}
LINK = re.compile(r"(\]\()([^)\s#]+)((?:#[^)\s]*)?\))|(href=\")([^\"#:]+)((?:#[^\"]*)?\")")


def fix(path: Path, dry: bool):
    rel = path.relative_to(ROOT).as_posix()
    old_dir = Path(NEW2OLD.get(rel, rel)).parent
    new_dir = Path(rel).parent
    text = path.read_text()
    changes = []

    def repl(m):
        pre, target, post = (m.group(1), m.group(2), m.group(3)) if m.group(1) else (m.group(4), m.group(5), m.group(6))
        if re.match(r"^[a-z][a-z0-9+.-]*:", target) or target.startswith("/"):
            return m.group(0)
        if (ROOT / new_dir / target).exists():
            return m.group(0)
        resolved = os.path.normpath((old_dir / target).as_posix())
        resolved = MOVES.get(resolved, resolved)
        if not (ROOT / resolved).exists():
            return m.group(0)
        new = os.path.relpath(ROOT / resolved, ROOT / new_dir)
        changes.append(f"{target} -> {new}")
        return pre + new + post

    out = LINK.sub(repl, text)
    print(f"{rel}: {len(changes)} links")
    for c in changes:
        print("   ", c)
    if not dry and out != text:
        path.write_text(out)


if __name__ == "__main__":
    dry = "--dry" in sys.argv
    for f in [a for a in sys.argv[1:] if a != "--dry"]:
        fix(ROOT / f, dry)
