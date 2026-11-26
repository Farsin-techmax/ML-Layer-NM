"""
tidy.py -- move generated clutter out of the repo root, reversibly.

WHY THIS IS CAREFUL
-------------------
.gitignore in this repo blanket-ignores *.csv, *.log and *.xlsx. That means git holds NO
history for any of those files, so anything deleted here is gone permanently -- there is no
`git checkout` to undo it. data/ alone is 926 MB of irreplaceable source data, and
refactored_test_dir/ holds the training matrices whose SHA256s are quoted in all nine
production_models/*/metadata.json files.

So this script never works out what to delete by asking "what is ignored?" or "what is
untracked?" -- both answers would include the entire project. It only ever touches files that
match an explicit, hand-written list of patterns (SECTION 1), in the repo root only, and it
MOVES them to a quarantine folder outside the repo rather than deleting them.

Usage:
    python tidy.py                          # dry run: show what would move, touch nothing
    python tidy.py --apply                  # move the default categories to quarantine
    python tidy.py --apply --category gates # also sweep the leak-screen / rescore files
    python tidy.py --apply --older-than 7   # only things untouched for a week
    python tidy.py --apply --delete         # hard delete instead of quarantine (asks first)
"""
from __future__ import annotations

import argparse
import datetime
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
QUARANTINE_ROOT = Path(r"c:\Techmax\cwf\backups")


# =====================================================================================
# SECTION 1 -- What counts as clutter.
#
# This is the only place that decides what may be swept. Everything else in the file is
# machinery. If you ever wonder "why did tidy remove that?", the answer is here.
# =====================================================================================
CATEGORIES = {
    "logs": {
        "on_by_default": True,
        "globs": ["run_*.log", "pms_processing.log", "debug.txt"],
        "why": "stdout captures and logging.FileHandler output; recreated by the next run",
    },
    "pycache": {
        "on_by_default": True,
        "globs": ["__pycache__"],          # a directory, handled separately below
        "why": "compiled bytecode; Python rebuilds it automatically",
    },
    "gates": {
        "on_by_default": False,
        "globs": ["leak_screen_*.csv", "rescore_*.csv"],
        "why": "G6/G8 raw evidence -- regenerable, but it is the measurement behind a "
               "promotion decision, so it is opt-in rather than automatic",
    },
    "scratch": {
        "on_by_default": False,
        "globs": ["*.bak", "early_slice_metrics.csv", "actual_*.csv",
                  "retrained_test_metrics_final.csv"],
        "why": "one-off analysis outputs and backup copies",
    },
}

# __pycache__ is searched recursively, so these subtrees are skipped: venv and legacy hold
# tens of thousands of files between them, and .git / feature must never be walked at all.
PYCACHE_SKIP = {"venv", "legacy", ".git", "feature", ".agents"}


# =====================================================================================
# SECTION 2 -- Finding candidates.
#
# Note both functions below are deliberately narrow. File patterns are matched in the repo
# ROOT ONLY (REPO.glob, never REPO.rglob), because every valuable artifact in this project
# lives in a subdirectory -- data/, models/, test_sets/, salman/ and so on. Refusing to
# descend is what keeps the blast radius small, and it costs nothing.
# =====================================================================================
def find_matching_files(globs):
    """Files in the repo root matching any of these patterns. Not recursive, by design."""
    found = []
    for pattern in globs:
        for path in REPO.glob(pattern):
            if path.is_file():
                found.append(path)
    return found


def find_pycache_dirs():
    """__pycache__ folders anywhere in the project, minus the subtrees in PYCACHE_SKIP."""
    found = []
    for path in REPO.rglob("__pycache__"):
        if not path.is_dir():
            continue
        parts_above = path.relative_to(REPO).parts[:-1]      # everything except __pycache__
        if any(part in PYCACHE_SKIP for part in parts_above):
            continue
        found.append(path)
    return found


# =====================================================================================
# SECTION 3 -- The three refusals.
#
# Every candidate must survive all three checks before it is allowed to move. Each one
# exists because of a specific way this could go wrong.
# =====================================================================================
def git_tracked_paths():
    """Absolute paths of every file git tracks, or None if git could not be asked.

    Returning None matters: the caller ABORTS rather than continuing. If we cannot tell
    what is tracked, we must not guess -- failing closed is the whole point.
    """
    try:
        result = subprocess.run(["git", "ls-files"], cwd=REPO,
                                capture_output=True, text=True, check=True)
    except Exception as exc:                                  # noqa: BLE001
        print(f"  could not run 'git ls-files': {exc}")
        return None

    tracked = set()
    for line in result.stdout.splitlines():
        line = line.strip()
        if line:
            tracked.add((REPO / line).resolve())
    return tracked


def refusal_reason(path, tracked_paths):
    """Why this path must NOT be swept, or None if it is safe to move."""
    resolved = path.resolve()

    # 1. Path escape. A pattern or symlink must never reach outside the project.
    if REPO not in resolved.parents:
        return "outside the repo root"

    # 2. Tracked by git. Source and docs are tracked; generated clutter never is.
    if resolved in tracked_paths:
        return "tracked by git"

    # 3. Declared provenance. write_manifest.py writes <file>.manifest.json next to any
    #    artifact whose origin has been recorded. That is this project's existing signal for
    #    "this matters", so honouring it means anything you manifest in future is protected
    #    automatically, with no change to this script.
    if Path(str(resolved) + ".manifest.json").exists():
        return "has a .manifest.json sibling"

    return None


# =====================================================================================
# SECTION 4 -- Small helpers.
# =====================================================================================
def size_of(path):
    """Bytes used by a file, or by everything inside a directory."""
    if path.is_file():
        return path.stat().st_size
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            total += item.stat().st_size
    return total


def age_in_days(path):
    now = datetime.datetime.now().timestamp()
    return (now - path.stat().st_mtime) / 86400.0


def human(num_bytes):
    for unit in ["B", "KB", "MB", "GB"]:
        if num_bytes < 1024 or unit == "GB":
            return f"{num_bytes:,.1f} {unit}"
        num_bytes = num_bytes / 1024.0


# =====================================================================================
# SECTION 5 -- Main flow: collect, report, then (only with --apply) move.
# =====================================================================================
def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--apply", action="store_true",
                        help="actually move files (without this it is a dry run)")
    parser.add_argument("--category", action="append", default=None, metavar="NAME",
                        help=f"categories to sweep; repeatable. "
                             f"choices: {', '.join(CATEGORIES)}")
    parser.add_argument("--older-than", type=float, default=0.0, metavar="DAYS",
                        help="only sweep files untouched for this many days (default 0)")
    parser.add_argument("--delete", action="store_true",
                        help="hard delete instead of quarantine -- asks for confirmation")
    args = parser.parse_args()

    # Which categories are in play: those named with --category, else the defaults.
    if args.category:
        for name in args.category:
            if name not in CATEGORIES:
                sys.exit(f"unknown category {name!r}; choose from {', '.join(CATEGORIES)}")
        active = list(args.category)
    else:
        active = [name for name, cfg in CATEGORIES.items() if cfg["on_by_default"]]

    print(f"repo       : {REPO}")
    print(f"mode       : {'APPLY' if args.apply else 'DRY RUN (nothing will move)'}")
    print(f"categories : {', '.join(active)}")
    if args.older_than:
        print(f"age filter : older than {args.older_than} day(s)")

    tracked_paths = git_tracked_paths()
    if tracked_paths is None:
        sys.exit("ABORT: cannot determine which files git tracks, so nothing is safe to move.")
    print(f"git tracks : {len(tracked_paths):,} file(s) -- all of them are off limits\n")

    # ---- collect -------------------------------------------------------------------
    selected = []          # (category, path, size)
    refused = []           # (path, reason)
    too_new = 0
    for name in active:
        if name == "pycache":
            candidates = find_pycache_dirs()
        else:
            candidates = find_matching_files(CATEGORIES[name]["globs"])

        for path in candidates:
            reason = refusal_reason(path, tracked_paths)
            if reason is not None:
                refused.append((path, reason))
                continue
            if args.older_than and age_in_days(path) < args.older_than:
                too_new += 1
                continue
            selected.append((name, path, size_of(path)))

    # ---- report --------------------------------------------------------------------
    if refused:
        print("REFUSED (protected):")
        for path, reason in refused:
            print(f"  {path.name:<44} {reason}")
        print()

    if not selected:
        print("Nothing to sweep -- the root is already clean.")
        return

    total = 0
    for name in active:
        rows = [row for row in selected if row[0] == name]
        if not rows:
            continue
        subtotal = sum(row[2] for row in rows)
        total += subtotal
        print(f"{name}  ({len(rows)} item(s), {human(subtotal)})")
        print(f"  why: {CATEGORIES[name]['why']}")
        for _, path, size in sorted(rows, key=lambda row: -row[2])[:8]:
            print(f"    {path.relative_to(REPO).as_posix():<46} {human(size):>10}")
        if len(rows) > 8:
            print(f"    ... and {len(rows) - 8} more")
        print()

    print(f"TOTAL: {len(selected)} item(s), {human(total)}")
    if too_new:
        print(f"({too_new} item(s) skipped as newer than {args.older_than} day(s))")

    if not args.apply:
        print("\nDry run -- nothing moved. Re-run with --apply to act on this.")
        return

    # ---- act -----------------------------------------------------------------------
    if args.delete:
        answer = input(f"\nPermanently delete {len(selected)} item(s)? "
                       f"This cannot be undone. Type 'delete' to confirm: ")
        if answer.strip() != "delete":
            sys.exit("aborted -- nothing was touched")
        for _, path, _ in selected:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
        print(f"deleted {len(selected)} item(s)")
        return

    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    quarantine = QUARANTINE_ROOT / f"tidy_{stamp}"
    quarantine.mkdir(parents=True, exist_ok=True)

    moved = []
    for category, path, size in selected:
        relative = path.relative_to(REPO)
        destination = quarantine / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(destination))
        moved.append({"category": category, "original_path": relative.as_posix(),
                      "bytes": size})

    manifest = {
        "moved_at": stamp,
        "repo": str(REPO),
        "categories": active,
        "item_count": len(moved),
        "total_bytes": total,
        "items": moved,
        "restore_hint": "copy the contents of this folder back over the repo root",
    }
    (quarantine / "tidy_manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"\nmoved {len(moved)} item(s), {human(total)}")
    print(f"quarantine -> {quarantine}")
    print(f"restore    -> Copy-Item '{quarantine}\\*' '{REPO}' -Recurse -Force"
          f"   (then delete tidy_manifest.json)")


if __name__ == "__main__":
    main()
