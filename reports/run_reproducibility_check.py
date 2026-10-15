#!/usr/bin/env python3
"""Reproducibility check for the FreightTiger pipeline.

Runs the complete pipeline (``main.py``) three times against the
unchanged supplied inputs, writes each run to a separate temporary
output file, and compares the resulting CSV.

* Default (offline) mode compares the full CSV **bytes** and expects
  every run to be byte-identical.
* ``--llm`` mode runs with ``--llm`` enabled. If no API key is present
  the runs stay offline and are still byte-compared. When an API key is
  present, every column **except** ``reason`` must be identical across
  runs (LLM wording may vary between runs, but flags, numbers, and
  matched note ids must not); the ``reason`` column is compared for
  informational purposes only.

Result is saved to ``reports/reproducibility_check.txt``.
Exit code 0 when the relevant comparison passes, 1 otherwise.

Usage (from the repository root):
    python3 reports/run_reproducibility_check.py [--llm]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "reports" / "reproducibility_check.txt"

RUNS = 3
REASON_COLUMN = "reason"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def non_reason_view(path: Path) -> list[dict[str, str]]:
    return [
        {k: v for k, v in row.items() if k != REASON_COLUMN}
        for row in read_rows(path)
    ]


def llm_key_present() -> bool:
    return bool(os.environ.get("FREIGHTTIGER_LLM_API_KEY"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Run main.py with --llm and compare non-reason columns",
    )
    args = parser.parse_args()
    online = args.llm and llm_key_present()

    hashes: list[str] = []
    returncodes: list[int] = []
    run_files: list[Path] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for run in range(1, RUNS + 1):
            out_path = Path(tmpdir) / f"output.run{run}.csv"
            cmd = [sys.executable, str(ROOT / "main.py"), "--output", str(out_path)]
            if args.llm:
                cmd.append("--llm")
            proc = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
            returncodes.append(proc.returncode)
            run_files.append(out_path)
            if proc.returncode != 0 or not out_path.is_file():
                hashes.append("<run failed>")
            else:
                hashes.append(sha256(out_path))

        failed_run = "<run failed>" in hashes
        if not failed_run and online:
            views = [non_reason_view(p) for p in run_files]
            non_reason_identical = all(v == views[0] for v in views[1:])
            byte_identical = len(set(hashes)) == 1
            passed = non_reason_identical
        else:
            # Offline runs must be byte-identical (including reason text).
            passed = not failed_run and len(set(hashes)) == 1
            non_reason_identical = passed
            byte_identical = passed

        mode = (
            "llm (online, API key present)"
            if online
            else "llm requested, no API key (offline fallback)"
            if args.llm
            else "offline (default)"
        )
        lines = [
            "FreightTiger pipeline reproducibility check",
            f"Mode                   : {mode}",
            f"Runs compared          : {RUNS}",
            "Inputs                 : data/raw/shipment_records.csv, "
            "data/raw/context_notes.csv (unchanged)",
            "Procedure              : "
            + "python3 main.py"
            + (" --llm" if args.llm else "")
            + " --output <tmp/runN.csv>, three times; "
            + (
                "compare all CSV columns except 'reason' "
                "(flags/numbers/note ids must not vary; "
                "reason wording may)"
                if online
                else "compare SHA-256 of generated CSV bytes"
            ),
            "",
        ]
        for run, (code, digest) in enumerate(zip(returncodes, hashes), start=1):
            lines.append(f"Run {run}: exit={code} sha256={digest}")
        lines.append("")
        if online:
            lines.append(
                f"Byte-identical across {RUNS} runs: "
                f"{'yes' if byte_identical else 'no'} "
                "(expected when LLM wording varies)"
            )
            lines.append(
                f"Non-'reason' columns identical across {RUNS} runs: "
                f"{'PASS' if non_reason_identical else 'FAIL'}"
            )
        else:
            lines.append(
                f"Byte-identical across {RUNS} runs: "
                f"{'PASS' if passed else 'FAIL'}"
            )
        lines.append("")
        result = "\n".join(lines) + "\n"
        REPORT_PATH.write_text(result, encoding="utf-8")
        print(result, end="")
        print(f"Result saved to {REPORT_PATH}")
        return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
