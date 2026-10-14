#!/usr/bin/env python3
"""Reproducibility check for the FreightTiger pipeline.

Runs the complete pipeline (``main.py``) three times against the
unchanged supplied inputs, writes each run to a separate temporary
output file, and compares the resulting CSV bytes.

Result is saved to ``reports/reproducibility_check.txt``.
Exit code 0 when all three runs are byte-identical, 1 otherwise.

Usage (from the repository root):
    python3 reports/run_reproducibility_check.py
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "reports" / "reproducibility_check.txt"

RUNS = 3


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    hashes: list[str] = []
    returncodes: list[int] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for run in range(1, RUNS + 1):
            out_path = Path(tmpdir) / f"output.run{run}.csv"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "main.py"),
                    "--output",
                    str(out_path),
                ],
                capture_output=True,
                text=True,
                cwd=ROOT,
            )
            returncodes.append(proc.returncode)
            if proc.returncode != 0 or not out_path.is_file():
                hashes.append("<run failed>")
            else:
                hashes.append(sha256(out_path))

    identical = len(set(hashes)) == 1 and "<run failed>" not in hashes
    lines = [
        "FreightTiger pipeline reproducibility check",
        f"Runs compared          : {RUNS}",
        "Inputs                 : data/raw/shipment_records.csv, "
        "data/raw/context_notes.csv (unchanged)",
        "Procedure              : python3 main.py --output <tmp/runN.csv>, "
        "three times; compare SHA-256 of generated CSV bytes",
        "",
    ]
    for run, (code, digest) in enumerate(zip(returncodes, hashes), start=1):
        lines.append(f"Run {run}: exit={code} sha256={digest}")
    lines.append("")
    lines.append(f"Byte-identical across {RUNS} runs: {'PASS' if identical else 'FAIL'}")
    lines.append("")
    result = "\n".join(lines) + "\n"
    REPORT_PATH.write_text(result, encoding="utf-8")
    print(result, end="")
    print(f"Result saved to {REPORT_PATH}")
    return 0 if identical else 1


if __name__ == "__main__":
    sys.exit(main())
