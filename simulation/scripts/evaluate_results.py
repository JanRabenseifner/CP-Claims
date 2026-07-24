#!/usr/bin/env python
"""Evaluate combined simulation outputs and save summary tables."""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cp_claims.analysis import (
    DEFAULT_METHODS,
    apply_baseline_filter,
    compute_method_rankings,
    compute_summary_stats,
    filter_methods,
    generate_latex_table,
    infer_baseline,
    load_results,
    summarize_results,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate combined simulation outputs")
    parser.add_argument("input", nargs="+", help="Combined result file or directories")
    parser.add_argument("--output-dir", type=Path, default=Path("analysis_output"))
    parser.add_argument("--keep-all-methods", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_results(args.input)
    if not args.keep_all_methods:
        df = filter_methods(df, DEFAULT_METHODS)

    baseline = infer_baseline(df)
    df_baseline = apply_baseline_filter(df, baseline)

    compute_summary_stats(df_baseline).to_csv(output_dir / "summary_statistics.csv", index=False)
    compute_method_rankings(df_baseline).to_csv(output_dir / "method_rankings.csv", index=False)
    df_baseline.groupby("method").agg({"coverage": "mean", "mean_width": "mean"}).round(4).to_csv(output_dir / "method_summary.csv")

    (output_dir / "baseline.json").write_text(json.dumps(baseline, indent=2), encoding="utf-8")
    (output_dir / "dataset_summary.json").write_text(json.dumps(summarize_results(df), indent=2), encoding="utf-8")

    generate_latex_table(df_baseline, output_dir)

    print(f"Baseline rows: {len(df_baseline):,}")
    print(f"Saved outputs to {output_dir}")


if __name__ == "__main__":
    main()
