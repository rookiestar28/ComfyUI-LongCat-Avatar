from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from LongCat_Video.mps_smoke_matrix import run_mps_smoke_matrix  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run LongCat Avatar MPS smoke matrix diagnostics.")
    parser.add_argument("--branch", default=_git_read("branch"), help="Branch name to record in JSON evidence.")
    parser.add_argument("--commit", default=_git_read("commit"), help="Commit SHA to record in JSON evidence.")
    args = parser.parse_args()

    result = run_mps_smoke_matrix(branch=args.branch, commit=args.commit)
    print(json.dumps(result.to_public_dict(), indent=2, sort_keys=True))
    return 0 if result.status == "pass" else 2


def _git_read(kind: str) -> str:
    command = ["git", "branch", "--show-current"] if kind == "branch" else ["git", "rev-parse", "--short", "HEAD"]
    try:
        completed = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
    except Exception:
        return "unknown"
    return completed.stdout.strip() or "unknown"


if __name__ == "__main__":
    raise SystemExit(main())
