#!/usr/bin/env python
"""
MPI driver for the count-data conformal prediction simulation study.

Usage on HPC cluster:
    mpirun python scripts/run_simulation.py [options]

Usage locally (without MPI):
    python scripts/run_simulation.py --repetitions 2 --output-dir results/local_test
"""

import argparse
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cp_claims.config import DGCP_METHOD_TYPES, LEARNERS, MIN_GROUP_SIZES
from cp_claims.simulation import (
    SimulationConfig,
    generate_configs,
    run_single_simulation,
)

try:
    from mpi4py import MPI
    COMM = MPI.COMM_WORLD
    RANK = COMM.Get_rank()
    SIZE = COMM.Get_size()
except ImportError:
    COMM = None
    RANK = 0
    SIZE = 1


def parse_int_list(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item]


def parse_float_list(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item]


def parse_str_list(value: str) -> list[str]:
    return [item for item in value.split(",") if item]


def parse_bool_list(value: str) -> list[bool]:
    mapping = {"true": True, "false": False}
    return [mapping[item.lower()] for item in value.split(",") if item]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MPI simulation driver for count-data conformal prediction")
    parser.add_argument("--output-dir", default="results/run")
    parser.add_argument("--repetitions", type=int, default=96)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--n-obs", default="1000,2000,4000,6000,10000")
    parser.add_argument("--dim-x", default="10,20,50")
    parser.add_argument("--dgp-types", default="poisson,zip,negbin,hurdle")
    parser.add_argument("--correlation-types", default="toeplitz")
    parser.add_argument("--correlation-strength", default="0.0,0.3,0.5")
    parser.add_argument("--categorical-ratio", default="0.1,0.3,0.5")
    parser.add_argument("--interaction-effects", default="false,true")
    parser.add_argument("--spatial-effects", default="false,true")
    parser.add_argument("--exposure-types", default="realistic")
    parser.add_argument("--learners", default="Poisson_GLM,LightGBM,RandomForest")
    parser.add_argument("--dgcp-methods", default="hybrid,full,exact")
    parser.add_argument("--min-group-sizes", default="10,20,50,100,200")
    parser.add_argument("--include-two-stage", action="store_true")
    parser.add_argument("--include-binary-mondrian", action="store_true", default=True)
    parser.add_argument("--no-binary-mondrian", action="store_true")
    return parser


def build_param_grid(args: argparse.Namespace) -> dict:
    return {
        "dgp_type": parse_str_list(args.dgp_types),
        "n_obs": parse_int_list(args.n_obs),
        "dim_x": parse_int_list(args.dim_x),
        "correlation_type": parse_str_list(args.correlation_types),
        "correlation_strength": parse_float_list(args.correlation_strength),
        "categorical_ratio": parse_float_list(args.categorical_ratio),
        "exposure_type": parse_str_list(args.exposure_types),
        "interaction_effects": parse_bool_list(args.interaction_effects),
        "spatial_effects": parse_bool_list(args.spatial_effects),
    }


def main() -> None:
    args = build_parser().parse_args()
    learners = parse_str_list(args.learners)
    dgcp_methods = parse_str_list(args.dgcp_methods)
    min_group_sizes = parse_int_list(args.min_group_sizes)
    param_grid = build_param_grid(args)
    include_binary_mondrian = not args.no_binary_mondrian

    reps_per_process = args.repetitions // SIZE
    extra_reps = args.repetitions % SIZE
    if RANK < extra_reps:
        reps_per_process += 1
        start_rep = RANK * reps_per_process
    else:
        start_rep = RANK * reps_per_process + extra_reps
    end_rep = start_rep + reps_per_process

    if RANK == 0:
        print("=" * 80)
        print("COUNT DATA CONFORMAL PREDICTION SIMULATION")
        print("=" * 80)
        print(f"MPI Processes: {SIZE}")
        print(f"Total Repetitions: {args.repetitions}")
        print(f"DGCP methods: {dgcp_methods}")
        print(f"Min group sizes: {min_group_sizes}")
        print(f"Learners: {learners}")
        print(f"Alpha: {args.alpha}")
        print("=" * 80)

    # Generate configs for this process's repetitions
    all_configs = generate_configs(
        param_grid=param_grid,
        n_reps=args.repetitions,
        base_seed=args.seed,
        exposure_type=parse_str_list(args.exposure_types)[0],
        correlation_type=parse_str_list(args.correlation_types)[0],
    )

    # Filter to only this rank's repetitions
    my_configs = [c for c in all_configs if start_rep <= c.repetition < end_rep]

    start_time = time.time()
    results: list[dict] = []

    for config in my_configs:
        try:
            sim_results = run_single_simulation(
                config=config,
                alpha=args.alpha,
                learners=learners,
                dgcp_methods=dgcp_methods,
                min_group_sizes=min_group_sizes,
                include_two_stage=args.include_two_stage,
                include_binary_mondrian=include_binary_mondrian,
            )
            results.extend(sim_results)
        except Exception as exc:
            results.append({"error": str(exc), **asdict(config)})

    df_results = pd.DataFrame(results)
    output_dir = Path(args.output_dir) / f"rank_{RANK:03d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"results_{RANK:03d}.parquet"
    df_results.to_parquet(output_file, index=False)

    print(f"Rank {RANK}: Saved {len(df_results)} results to {output_file}")

    if COMM is not None:
        COMM.Barrier()

    if RANK == 0:
        elapsed = time.time() - start_time
        print(f"\nSimulation completed in {elapsed:.1f} seconds")

    print(f"Rank {RANK}: Completed in {time.time() - start_time:.1f} seconds")


if __name__ == "__main__":
    main()
