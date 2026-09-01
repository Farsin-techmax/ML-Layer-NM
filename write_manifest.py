"""
Record provenance for a produced artifact (matrix, test set, model dir, metrics file).

The 2026-08-19 label regression was caused by a silently swapped data file, not a commit -- git
history did not catch it. Every metric therefore needs its input files recorded at production time.

Writes <artifact>.manifest.json (or <dir>/manifest.json for a directory) containing: SHA256, size
and mtime of the artifact and every --input, git HEAD, the producing command line, all PMS_* env
vars, and a timestamp.

Usage:
    python write_manifest.py refactored_test_dir/final_processed_20k_candidates.csv ^
        --inputs data/EDA_Q2-2026.csv data/Service_History_Q2-2026.csv ^
        --cmd "python pmstrainfeatureeng_refactored.py 2026 Q1 20 --label-mode candidates --grace-days 15"
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime


def file_record(path):
    if not os.path.exists(path):
        return {"error": "MISSING"}
    if os.path.isdir(path):
        entries = {}
        for name in sorted(os.listdir(path)):
            full = os.path.join(path, name)
            if os.path.isfile(full):
                entries[name] = file_record(full)
        return {"dir": True, "files": entries}
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    st = os.stat(path)
    return {
        "sha256": h.hexdigest(),
        "size": st.st_size,
        "mtime": datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
    }


def main():
    p = argparse.ArgumentParser(description="Write <artifact>.manifest.json provenance record")
    p.add_argument("artifact", help="File or directory the manifest describes")
    p.add_argument("--inputs", nargs="*", default=[], help="Input files that produced it")
    p.add_argument("--cmd", default="", help="The command line that produced the artifact")
    args = p.parse_args()

    if not os.path.exists(args.artifact):
        sys.exit(f"Artifact does not exist: {args.artifact}")

    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        head = "unknown"

    manifest = {
        "artifact": args.artifact.replace("\\", "/"),
        "artifact_record": file_record(args.artifact),
        "inputs": {f.replace("\\", "/"): file_record(f) for f in args.inputs},
        "command": args.cmd,
        "git_head": head,
        "env_pms": {k: v for k, v in sorted(os.environ.items()) if k.startswith("PMS_")},
        "written_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    if os.path.isdir(args.artifact):
        out = os.path.join(args.artifact, "manifest.json")
    else:
        out = args.artifact + ".manifest.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
