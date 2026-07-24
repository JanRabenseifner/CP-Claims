import json
from pathlib import Path

from .analysis import (
    DEFAULT_METHODS,
    apply_baseline_filter,
    compute_method_rankings,
    compute_summary_stats,
    filter_methods,
    infer_baseline,
    load_results,
    summarize_results,
)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate combined simulation outputs")
    parser.add_argument("input", nargs="+", help="Result file or directories")
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
    (output_dir / "baseline.json").write_text(json.dumps(baseline, indent=2), encoding="utf-8")
    (output_dir / "dataset_summary.json").write_text(json.dumps(summarize_results(df), indent=2), encoding="utf-8")

    print(f"Baseline rows: {len(df_baseline):,}")
    print(f"Saved outputs to {output_dir}")


if __name__ == "__main__":
    main()
