#!/usr/bin/env python
"""Combine rank-level simulation outputs into a single parquet file."""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cp_claims.analysis import combine_result_files


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine split simulation outputs into a single parquet file")
    parser.add_argument("inputs", nargs="+", help="Parquet files or directories containing parquet files")
    parser.add_argument("--output", type=Path, default=Path("results") / "results_combined.parquet")
    args = parser.parse_args()

    combined = combine_result_files(args.inputs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(args.output, index=False)
    print(f"Saved {len(combined):,} rows to {args.output}")

    if "method" in combined.columns:
        print(f"\nMethods: {combined['method'].nunique()}")
        print(f"DGPs: {combined['dgp_type'].nunique()}")
        print(f"Learners: {combined['learner'].nunique()}")
        print(f"Repetitions: {combined['repetition'].nunique()}")

    if "coverage" in combined.columns:
        cov_summary = combined.groupby("method")["coverage"].agg(["mean", "std"]).round(4)
        print(f"\nCoverage by method:")
        print(cov_summary)


if __name__ == "__main__":
    main()
