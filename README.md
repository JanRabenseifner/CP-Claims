# CP-Claims

Simulation study for count-data conformal prediction methods in insurance claim frequency modeling.

## Overview

This repository implements a Monte Carlo simulation comparing prediction interval methods for insurance claim counts, including:

- **DGCP** (Discrete Group-Conditional Prediction): Exact, Full, and Hybrid variants with configurable minimum group sizes (m = 10, 20, 50, 100, 200)
- **Standard ICP**: Inductive conformal prediction with Poisson deviance and zero-adjusted relative nonconformity scores
- **Cluster-Conditional ICP** (Mondrian): Conformal prediction conditioned on data-driven clusters
- **Binary CP**: Marginal and Mondrian binary classification conformal prediction
- **Parametric benchmarks**: Poisson Normal approximation, Poisson Bootstrap, NB Plugin, NB Chebyshev, NB Bootstrap
- **Two-Stage methods**: Weighted, Calibrated, Balanced, Direct, Empirical, and Exact variants (optional)

Four data generating processes (DGPs) are used: Poisson, Zero-Inflated Poisson (ZIP), Negative Binomial, and Hurdle models, with configurable covariate structure, correlation, zero inflation, and overdispersion.

## Installation

```bash
pip install -e .
```

Optional dependencies:

```bash
pip install -e ".[lightgbm]"    # LightGBM learner
pip install -e ".[hpc]"         # MPI parallel execution
pip install -e ".[dev]"         # Jupyter notebook support
```

## Running the simulation

### Local execution

```bash
python scripts/run_simulation.py --output-dir results/local_test --repetitions 2
```

### HPC cluster execution

```bash
sbatch scripts/run_simulation_hpc.sh
```

The shell script uses SLURM and MPI (`mpirun`) to distribute repetitions across 96 processes. Adjust `--ntasks`, `--time`, and `--partition` in the SBATCH directives to match your cluster configuration. Set `PROJECT_DIR` and `RESULTS_DIR` environment variables if needed.

### Command-line options

The simulation driver supports flexible configuration via CLI arguments:

```bash
python scripts/run_simulation.py \
    --repetitions 96 \
    --seed 42 \
    --alpha 0.1 \
    --n-obs 1000,2000,4000,6000,10000 \
    --dim-x 10,20,50 \
    --dgp-types poisson,zip,negbin,hurdle \
    --correlation-strength 0.0,0.3,0.5 \
    --categorical-ratio 0.1,0.3,0.5 \
    --interaction-effects false,true \
    --spatial-effects false,true \
    --exposure-types realistic \
    --correlation-types toeplitz \
    --learners Poisson_GLM,LightGBM,RandomForest \
    --dgcp-methods hybrid,full,exact \
    --min-group-sizes 10,20,50,100,200
```

## Combining results

After the simulation completes, rank-level parquet files are stored in the output directory. Combine them into a single file:

```bash
python scripts/combine_results.py results/run --output results/results_combined.parquet
```

## Evaluation

Generate summary statistics, method rankings, and publication tables:

```bash
python scripts/evaluate_results.py results/results_combined.parquet --output-dir analysis_output
```

This produces:
- `summary_statistics.csv`
- `method_rankings.csv`
- `method_summary.csv`
- `results_table.tex`
- `baseline.json`
- `dataset_summary.json`

## Notebook

Open `notebooks/evaluate_results.ipynb` for interactive analysis of the combined results, including coverage plots, sensitivity analysis, and outcome-conditional coverage comparisons.

## Project structure

```
CP-Claims/
├── src/cp_claims/          # Reusable package
│   ├── config.py           # Method display names and default grids
│   ├── simulation.py       # Core simulation logic
│   ├── analysis.py         # Result loading, filtering, and summary utilities
│   ├── utils_dgps.py       # Data generating processes
│   ├── utils_dgcp.py       # DGCP method implementations
│   ├── utils_benchmarks.py # Parametric benchmark intervals
│   ├── utils_pinterval.py  # ICP, Mondrian, Binary CP implementations
│   ├── utils_eval.py       # Evaluation metrics
│   ├── utils_two_stage.py  # Two-stage method implementations
│   ├── cli_simulate.py     # Local simulation entrypoint
│   ├── cli_combine.py      # Combine entrypoint
│   ├── cli_evaluate.py     # Evaluate entrypoint
├── scripts/
│   ├── run_simulation.py       # MPI driver (HPC)
│   ├── run_simulation_hpc.sh   # SLURM submission script
│   ├── combine_results.py
│   ├── evaluate_results.py
├── notebooks/
│   └── evaluate_results.ipynb
├── results/                # Simulation outputs
├── analysis_output/        # Evaluation outputs
├── pyproject.toml
└── README.md
```

## License

See the accompanying license file.
