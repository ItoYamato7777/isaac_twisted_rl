"""壊してはいけない構造上の約束 (不変条件) を、静的に検査する。

どれも `ast` でソースを読むだけで、リポジトリのコードを import しない。
そのため Isaac Sim なしで、ほかの単体テストと一緒に回る:

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run python -m pytest tests/test_invariants.py

検査する約束と、その理由:

- I1 `tasks/` の下のパッケージの `__init__.py` は、最上位で isaaclab・omni・pxr などを
  (推移的にも) import しない。`tasks/__init__.py` の `import_packages` が配下の全パッケージを
  import するので、ここで Kit の起動後でないと読めないものを import すると、
  `list_envs` や `train` がタスクの登録の段階で落ちる (またはタスクが黙って消える)。
- I2 `common/topology/` と `common/state_archive.py` は isaaclab・omni・pxr などを import しない
  (関数の中も含む)。単体テストを Isaac Sim なしで回すため。
- I3 `mdp` という名前のディレクトリは `common/mdp` だけで、`import_packages` の除外の一覧
  (`_BLACKLIST_PKGS`) がそれを外す。除外は部分一致なので、名前に `utils` や `.mdp` を含む
  パッケージを新しく作ると、黙って登録から外れる。
- I4 `common/mdp/` の項 (最上位の def / class) は追加だけ。削除・改名しない。
  過去の版の設定が名前で参照しているため。役目を終えた項は `.. deprecated::` で残す。
- I5 追跡しているシェルスクリプトの `timeout` は `-k` (`--kill-after`) を付ける。Kit・FEM は固まると
  SIGTERM で止まらないことがあり、`-k` が無いと timeout の後もプロセスが GPU を握ったまま残る。
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
TASKS = SRC / "isaac_twisted_rl" / "tasks"
KT = TASKS / "knot_tying"

# Kit (Isaac Sim) の起動後でないと import できない、またはそれに依存するパッケージ。
FORBIDDEN = {"isaaclab", "omni", "pxr", "isaacsim", "carb"}

# `import_packages(__name__, _BLACKLIST_PKGS)` の除外の一覧 (`tasks/__init__.py`)。
BLACKLIST = ("utils", ".mdp")

# I4 の基準: commit 6f41ef1 の時点で `common/mdp/*.py` の最上位にあった def / class の名前 (58 個)。
# 名前を足すのは自由 (この一覧は更新しなくてよい)。消えたらテストが落ちる。
MDP_NAMES_6F41EF1 = (
    "EnvStateCloner",
    "GraspLimits",
    "LadderProgressBonus",
    "NextEdgeGoal",
    "NextEdgeStats",
    "PhaseDurations",
    "PickPlaceAction",
    "PickPlaceActionCfg",
    "ResetFromArchive",
    "ResetToStage",
    "TopologyStats",
    "_get_archive_term",
    "_get_stage_term",
    "_outcome",
    "_pick_place_term",
    "_zero_if_diverged",
    "action_failed",
    "bind_grasp_material",
    "check_rope_link_order",
    "crossing_achieved",
    "crossing_count",
    "crossing_features",
    "crossing_progress",
    "current_stage",
    "diverged_by_class",
    "diverged_now",
    "finger_footprint_top",
    "finger_gap",
    "first_crossing_feature",
    "get_extractor",
    "get_next_edge_goal",
    "grasp_distance_penalty",
    "hand_position",
    "held_up_penalty",
    "interleaved2_achieved",
    "invalidate_extractor",
    "knot_class_one_hot",
    "knot_class_shaping",
    "knot_stage_one_hot",
    "ladder_class_reached",
    "log_archive_stats",
    "log_next_edge_stats",
    "log_stage_stats",
    "log_topology_stats",
    "offladder_penalty",
    "overhand_achieved",
    "phase_incomplete",
    "record_archive",
    "record_stage_archive",
    "rope_diverged",
    "rope_link_positions",
    "rope_sample_points",
    "rope_zmax",
    "stage_changed",
    "stage_transition",
    "stage_transition_reward",
    "topology_reached",
    "writhe",
)


def _module_name(path: Path) -> str:
    rel = path.relative_to(SRC).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _module_path(name: str) -> Path | None:
    base = SRC.joinpath(*name.split("."))
    if (base / "__init__.py").exists():
        return base / "__init__.py"
    if base.with_suffix(".py").exists():
        return base.with_suffix(".py")
    return None


def _top_level_nodes(tree: ast.Module):
    """import 時に実行される文 (関数・クラスの本体の中を除く)。"""
    stack = list(tree.body)
    while stack:
        node = stack.pop()
        yield node
        if isinstance(node, (ast.If, ast.Try, ast.With)):
            for field in ("body", "orelse", "finalbody"):
                stack.extend(getattr(node, field, []))
            for handler in getattr(node, "handlers", []):
                stack.extend(handler.body)


def _imports(path: Path, top_level_only: bool) -> list[str]:
    """`path` が import するモジュールの絶対名の一覧。"""
    tree = ast.parse(path.read_text(), filename=str(path))
    nodes = _top_level_nodes(tree) if top_level_only else ast.walk(tree)
    pkg = _module_name(path)
    if path.name != "__init__.py":
        pkg = pkg.rpartition(".")[0]
    names = []
    for node in nodes:
        if isinstance(node, ast.Import):
            names += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = pkg.split(".")
                base = base[: len(base) - (node.level - 1)]
                mod = ".".join(base + ([node.module] if node.module else []))
            else:
                mod = node.module or ""
            names.append(mod)
            # `from pkg import sub` の sub がサブモジュールなら、それも辿る
            names += [f"{mod}.{a.name}" for a in node.names if _module_path(f"{mod}.{a.name}")]
    return names


def _forbidden_reach(start: Path) -> list[str]:
    """`start` から最上位の import を推移的にたどり、禁止のパッケージに届く経路を返す。"""
    seen: set[str] = set()
    found = []
    stack = [(_module_name(start), [_module_name(start)])]
    while stack:
        name, trail = stack.pop()
        if name in seen:
            continue
        seen.add(name)
        path = _module_path(name)
        if path is None:
            continue
        for imp in _imports(path, top_level_only=True):
            root = imp.split(".")[0]
            if root in FORBIDDEN:
                found.append(" -> ".join(trail + [imp]))
            elif root == "isaac_twisted_rl":
                stack.append((imp, trail + [imp]))
    return found


def _auto_imported_packages() -> list[Path]:
    """`import_packages` が import するパッケージの `__init__.py` (除外を適用したもの)。"""
    out = []
    for init in sorted(TASKS.rglob("__init__.py")):
        if "__pycache__" in init.parts or init == TASKS / "__init__.py":
            continue
        name = _module_name(init)
        if any(b in name for b in BLACKLIST):
            continue
        out.append(init)
    return out


def test_i1_auto_imported_inits_do_not_reach_kit():
    assert _auto_imported_packages(), "パッケージが 1 つも見つからない (パスの計算の誤り)"
    violations = [v for init in _auto_imported_packages() for v in _forbidden_reach(init)]
    assert not violations, "Kit の起動前に実行される __init__ から禁止の import に届く:\n" + "\n".join(violations)


def test_i2_pure_layers_do_not_import_kit():
    targets = sorted((KT / "common" / "topology").rglob("*.py")) + [KT / "common" / "state_archive.py"]
    violations = []
    for path in targets:
        if "__pycache__" in path.parts:
            continue
        for imp in _imports(path, top_level_only=False):
            if imp.split(".")[0] in FORBIDDEN:
                violations.append(f"{path.relative_to(REPO)}: {imp}")
    assert not violations, "Isaac Sim なしで動くべき層が禁止のパッケージを import している:\n" + "\n".join(violations)


def test_i3_mdp_directory_and_blacklist():
    tree = ast.parse((TASKS / "__init__.py").read_text())
    blacklist = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "_BLACKLIST_PKGS" for t in node.targets):
            blacklist = ast.literal_eval(node.value)
    assert blacklist is not None and set(BLACKLIST) <= set(blacklist), f"_BLACKLIST_PKGS = {blacklist}"

    mdp_dirs = sorted(
        p.relative_to(SRC).as_posix()
        for p in SRC.rglob("*")
        if p.is_dir() and "mdp" in p.name and "__pycache__" not in p.parts
    )
    assert mdp_dirs == ["isaac_twisted_rl/tasks/knot_tying/common/mdp"], mdp_dirs

    # 除外の部分一致に、意図しないパッケージが引っかかっていないこと
    caught = sorted(
        _module_name(init)
        for init in TASKS.rglob("__init__.py")
        if "__pycache__" not in init.parts
        and any(b in _module_name(init) for b in BLACKLIST)
        and ".common.mdp" not in _module_name(init)
    )
    assert not caught, f"除外の一覧 {BLACKLIST} に部分一致して、黙って登録から外れるパッケージ: {caught}"


def test_i4_mdp_terms_are_additive():
    names = set()
    for path in (KT / "common" / "mdp").glob("*.py"):
        tree = ast.parse(path.read_text())
        names |= {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
    missing = sorted(set(MDP_NAMES_6F41EF1) - names)
    assert not missing, (
        "common/mdp/ の項が消えた (削除・改名は禁止。役目を終えた項は .. deprecated:: で残す): " + ", ".join(missing)
    )


def _timeouts_without_kill(text: str) -> list[str]:
    """`timeout` の呼び出しのうち、時間の引数より前に `-k` / `--kill-after` が無い行を返す。"""
    bad = []
    for line in text.splitlines():
        if line.lstrip().startswith("#"):
            continue
        code = re.sub(r"\s#.*$", "", line)
        for m in re.finditer(r"(?:^|[\s;&|(`$])timeout\s+(.*)", code):
            for tok in m.group(1).split():
                if tok in ("-k", "--kill-after") or tok.startswith(("--kill-after=", "-k")):
                    break
                if tok.startswith("-"):
                    continue
                # 時間の引数 (数字か変数) が先に来たら -k が無い。それ以外 (文中の「timeout の」など) は呼び出しではない
                if re.match(r"""^["']?(\d|\$)""", tok):
                    bad.append(line.strip())
                break
    return bad


def test_i5_shell_timeouts_have_kill_after():
    out = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "--", "*.sh"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    bad = {
        p: b for p in out.splitlines() if (REPO / p).exists() and (b := _timeouts_without_kill((REPO / p).read_text()))
    }
    assert not bad, f"timeout に -k が無い (固まった Kit が残る): {bad}"


def test_i5_detector():
    assert _timeouts_without_kill("timeout 7200 $PY x.py") == ["timeout 7200 $PY x.py"]
    assert _timeouts_without_kill("  out=$(timeout 10 nvidia-smi)") == ["out=$(timeout 10 nvidia-smi)"]
    assert not _timeouts_without_kill("timeout -k 60 7200 $PY x.py")
    assert not _timeouts_without_kill('timeout -k "$k" "$d" "$@"')
    assert not _timeouts_without_kill("timeout --kill-after=5 10 cmd")
    assert not _timeouts_without_kill("run_job.sh --timeout 10h -- cmd   # timeout 1")
    assert not _timeouts_without_kill('echo "timeout: ${d} で止めた"')
    assert not _timeouts_without_kill('echo "SIGKILL で止まった (timeout の kill-after か)"')


if __name__ == "__main__":
    sys.exit(__import__("pytest").main([__file__, "-q"]))
