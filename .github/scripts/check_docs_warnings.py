"""Fail a PR's docs build on warnings that come from files the PR changed.

Usage:
    python .github/scripts/check_docs_warnings.py <sphinx-warnings-file> --base <sha> [--head <sha>]

main still has older docs warnings, so treating every warning as an error would fail every
PR. This keeps new and edited pages, and the docstrings of edited modules, warning-free in
the meantime. Once main builds cleanly, switch the workflow to ``sphinx-build -W`` and drop
this script.
"""

import argparse
import os
import re
import sys
from pathlib import Path

from dataset_files import ROOT, changed_files


# "<path>:<line>: WARNING: msg" or "<path>:docstring of <obj>:<line>: ERROR: msg"; the line may be empty.
WARNING_RE = re.compile(
    r"^(?P<path>[^:\s][^:]*?)(?::docstring of (?P<obj>[^:]+))?(?::(?P<line>\d*))?: "
    r"(?P<level>WARNING|ERROR|SEVERE|CRITICAL): (?P<msg>.*)$"
)
# autodoc reports modules it can't import without a file path:
# "WARNING: Failed to import stable_datasets.images.x." or
# "WARNING: autodoc: failed to import 'x' from module 'stable_datasets.images'; ..."
IMPORT_FAILURE_RES = (
    re.compile(r"^WARNING: Failed to import (?P<module>stable_datasets[\w.]*?)\.?$"),
    re.compile(
        r"^WARNING: autodoc: failed to import '(?P<name>\w+)' from module '(?P<package>stable_datasets[\w.]*)'"
    ),
)


def module_path(dotted: str) -> str | None:
    """Repo-relative file for a dotted module name, if it exists."""
    base = ROOT / dotted.replace(".", "/")
    for candidate in (base.with_suffix(".py"), base / "__init__.py"):
        if candidate.is_file():
            return candidate.relative_to(ROOT).as_posix()
    return None


def parse_import_failure(line: str) -> dict | None:
    for regex in IMPORT_FAILURE_RES:
        m = regex.match(line)
        if m:
            dotted = m.groupdict().get("module") or f"{m['package']}.{m['name']}"
            path = module_path(dotted)
            if path:
                return {
                    "path": path,
                    "obj": None,
                    "line": None,
                    "level": "WARNING",
                    "msg": f"`{dotted}` can't be imported",
                }
    return None


def parse(warnings_file: Path) -> list[dict]:
    """Warnings that point at a file, with the path made relative to the repo root."""
    found = []
    for line in warnings_file.read_text(encoding="utf-8", errors="replace").splitlines():
        if failure := parse_import_failure(line):
            found.append(failure)
            continue
        m = WARNING_RE.match(line)
        if not m:
            continue
        path = Path(m["path"])
        if path.is_absolute():
            try:
                path = path.resolve().relative_to(ROOT)
            except ValueError:
                continue
        found.append({**m.groupdict(), "path": path.as_posix()})
    # Sphinx can report the same problem more than once (e.g. an import failure in two forms).
    unique = {(w["path"], w["obj"], w["line"], w["msg"]): w for w in found}
    return list(unique.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("warnings_file", type=Path)
    parser.add_argument("--base", required=True, help="commit to diff against")
    parser.add_argument("--head", default="HEAD")
    args = parser.parse_args()

    changed = {c.path for c in changed_files(args.base, args.head) if c.status != "D"}
    warnings = parse(args.warnings_file)
    mine = [w for w in warnings if w["path"] in changed]
    others = len(warnings) - len(mine)

    lines = [f"**Docs warnings in files this PR changed: {len(mine)}**", ""]
    for w in mine:
        where = f"{w['path']}" + (f" (docstring of `{w['obj']}`)" if w["obj"] else f":{w['line'] or '?'}")
        lines.append(f"- `{where}`: {w['level']}: {w['msg']}")
    lines += ["", f"{others} older warnings in other files aren't checked here."]
    report = "\n".join(lines)
    print(report)
    for w in mine:
        # Docstring line numbers count from the docstring, not the file, so only page warnings get a line.
        line = f",line={w['line']}" if w["line"] and not w["obj"] else ""
        print(f"::error file={w['path']}{line}::{w['msg']}")
    if summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(summary, "a") as f:
            f.write(report + "\n")
    return 1 if mine else 0


if __name__ == "__main__":
    sys.exit(main())
