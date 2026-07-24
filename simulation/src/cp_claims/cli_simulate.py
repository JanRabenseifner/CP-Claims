from pathlib import Path

from .simulation import generate_configs, run_single_simulation


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Run the count-data CP simulation locally (without MPI)")
    parser.add_argument("--reps", type=int, default=2)
    parser.add_argument("--output", type=Path, default=Path("results") / "results.parquet")
    args = parser.parse_args()

    configs = generate_configs(n_reps=args.reps)
    results = []
    for config in configs:
        results.extend(run_single_simulation(config))

    import pandas as pd
    df = pd.DataFrame(results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)
    print(f"Saved {len(df):,} rows to {args.output}")


if __name__ == "__main__":
    main()
