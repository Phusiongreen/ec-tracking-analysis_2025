#!/usr/bin/env python
"""Execute all pipeline scripts (00–08) consecutively.

Usage:
    python run_all.py            # run all scripts 00–08
    python run_all.py --start 3  # start from script 03 onward
"""

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPTS = [
    "00_correct_time_points_from_trackmate.py",
    "01_Preprocess_TackMate_Data.py",
    "02_Quality_Control.py",
    "03_A_Plot_Trajectories.py",
    "04_Plot_Velocity_Field.py",
    "05_Graph_Analysis.py",
    "06_Time_Graph_Analysis.py",
    "07_Plot_Migration_Speeds.py",
    "08_Compare_Average_Velocities.py",
]


def main():
    parser = argparse.ArgumentParser(description="Run pipeline scripts 00–08.")
    parser.add_argument(
        "--start", type=int, default=0, metavar="N",
        help="Script number to start from (0–8, default: 0)",
    )
    args = parser.parse_args()

    if not 0 <= args.start <= 8:
        parser.error(f"--start must be between 0 and 8, got {args.start}")

    scripts = [s for s in SCRIPTS if int(s[:2]) >= args.start]

    files_dir = Path(__file__).resolve().parent
    failed = []

    for script in scripts:
        script_path = files_dir / script
        print(f"\n{'=' * 60}")
        print(f"Running {script} ...")
        print(f"{'=' * 60}\n")

        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(files_dir),
        )

        if result.returncode != 0:
            print(f"\n✗ {script} failed (exit code {result.returncode})")
            failed.append(script)
            break  # stop on first failure

    if failed:
        print(f"\nPipeline stopped — failed script: {failed[0]}")
        sys.exit(1)
    else:
        print(f"\n{'=' * 60}")
        print("All scripts completed successfully.")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

