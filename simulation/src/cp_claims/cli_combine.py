from pathlib import Path

from .analysis import combine_result_files


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Combine split simulation outputs into a single parquet file")
    parser.add_argument("inputs", nargs="+", help="Parquet files or directories")
    parser.add_argument("--output", type=Path, default=Path("results") / "results_combined.parquet")
    args = parser.parse_args()

    combined = combine_result_files(args.inputs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(args.output, index=False)
    print(f"Saved {len(combined):,} rows to {args.output}")


if __name__ == "__main__":
    main()
