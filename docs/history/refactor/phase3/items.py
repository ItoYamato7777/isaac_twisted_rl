"""Phase 3 の情報の保存の照合 (BRIEF §4 Phase 3【必須】)。

baseline: 追跡している *.md / *.html から再現の項目を抜き出して baseline_items.json に保存する。
check:    今の作業ツリー (追跡 + 未追跡で無視されていない *.md / *.html / *.py) のどこかに、
          各項目が残っているかを確かめ、消えた項目を元の文書ごとに出す。
"""
import json, re, subprocess, sys, collections
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).with_name("baseline_items.json")

PATTERNS = {
    "commit": r"(?<![0-9A-Za-z_/.-])(?=[0-9a-f]*[a-f])(?=[0-9a-f]*[0-9])[0-9a-f]{7,40}(?![0-9A-Za-z_])",
    "commit_bt": r"(?<=`)[0-9a-f]{7,12}(?=`)",
    "seed": r"(?:seed|シード)\s*[=:]?\s*\d+|_s\d{2,3}\b|\bs4[0-9]\b",
    "iteration": r"(?:iteration|iter|it)\s?\d+|\d+\s?iteration|--max_iterations[ =]\d+|save_interval=\d+",
    "checkpoint": r"model_\d+\.pt",
    "path": r"(?:logs|outputs)/[^\s`'\"()|,<>\]]+",
    "run_dir": r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}[A-Za-z0-9_]*",
    "number": r"\d+(?:\.\d+)?\s?(?:%|ポイント|ms|秒|分|時間|GB|GiB|MiB|MB|m/s|mm|cm|Hz|環境|手/s|手|件|本|回|倍)"
              r"|\d+(?:\.\d+)?\s?(?:s|h|m|kg|g)\b",
}
RX = {k: re.compile(v) for k, v in PATTERNS.items()}


def files(cmd):
    out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=True).stdout.split("\n")
    return [f for f in out if f]


def norm(s):
    return re.sub(r"\s+", "", s).rstrip(".:;。、")


def baseline():
    items = collections.defaultdict(lambda: collections.defaultdict(set))
    for f in files(["git", "ls-files", "*.md", "*.html"]):
        text = (ROOT / f).read_text(errors="replace")
        for kind, rx in RX.items():
            for m in rx.finditer(text):
                items[f][kind].add(norm(m.group(0)))
    data = {f: {k: sorted(v) for k, v in d.items()} for f, d in items.items()}
    OUT.write_text(json.dumps(data, ensure_ascii=False, indent=1))
    n = sum(len(v) for d in data.values() for v in d.values())
    print(f"baseline: {len(data)} files, {n} (file, item) pairs -> {OUT}")


def check():
    data = json.loads(OUT.read_text())
    corpus = []
    for f in files(["git", "ls-files", "-co", "--exclude-standard", "*.md", "*.html", "*.py"]):
        p = ROOT / f
        # この道具の置き場所 (リファクタリングの記録) は数値を写しているので、照合に入れると欠落を隠す
        if p.exists() and not f.startswith("docs/history/refactor/"):
            corpus.append(re.sub(r"\s+", "", p.read_text(errors="replace")))
    big = "\n".join(corpus)
    missing = collections.defaultdict(list)
    total = 0
    for f, d in data.items():
        for kind, vals in d.items():
            for v in vals:
                total += 1
                if v not in big:
                    missing[f].append(f"{kind}: {v}")
    n = sum(len(v) for v in missing.values())
    print(f"check: {total} pairs, missing {n}")
    for f, v in sorted(missing.items()):
        print(f"## {f} ({len(v)})")
        for x in v:
            print("  -", x)


if __name__ == "__main__":
    {"baseline": baseline, "check": check}[sys.argv[1]]()
