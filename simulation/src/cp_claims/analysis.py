from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    BENCHMARK_METHODS,
    CP_METHODS,
    DEFAULT_BASELINE,
    DEFAULT_MAIN_METHODS,
    METHOD_DISPLAY_NAMES,
)

DEFAULT_METHODS = DEFAULT_MAIN_METHODS
DGCP_METHODS = [m for m in DEFAULT_METHODS if m.startswith("DGCP_")]

NUMERIC_COLUMNS = [
    "coverage",
    "mean_width",
    "median_width",
    "interval_score",
    "coverage_y0",
    "coverage_y1",
    "coverage_y2plus",
    "coverage_y1_5",
    "coverage_y6_10",
    "coverage_y11_20",
    "coverage_y21_50",
    "coverage_y51plus",
    "coverage_gap",
    "risk_differentiation",
    "pct_singleton",
    "pct_informative",
    "tail_coverage",
    "var_95_coverage",
    "var_99_coverage",
    "calibration_range",
    "n_obs",
    "dim_x",
    "correlation_strength",
    "categorical_ratio",
    "zero_inflation",
    "overdispersion",
    "min_group_size",
    "repetition",
]


def get_display_name(method: str) -> str:
    return METHOD_DISPLAY_NAMES.get(method, method.replace("_", " "))


def _as_paths(paths: str | Path | Sequence[str | Path]) -> list[Path]:
    if isinstance(paths, (str, Path)):
        return [Path(paths)]
    return [Path(p) for p in paths]


def combine_result_files(paths: str | Path | Sequence[str | Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in _as_paths(paths):
        candidates = sorted(path.rglob("*.parquet")) if path.is_dir() else [path]
        for candidate in candidates:
            suffix = candidate.suffix.lower()
            if suffix == ".parquet":
                frame = pd.read_parquet(candidate)
            elif suffix == ".csv":
                frame = pd.read_csv(candidate)
            elif suffix == ".pkl":
                frame = pd.read_pickle(candidate)
            else:
                continue
            frame["source_file"] = candidate.name
            frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No result files found for {paths!r}")
    combined = pd.concat(frames, ignore_index=True)
    if "error" in combined.columns:
        combined = combined[combined["error"].isna()].copy()
    return combined


def preprocess_results(df: pd.DataFrame) -> pd.DataFrame:
    clean = df.copy()
    if "method" in clean.columns:
        clean["method"] = clean["method"].replace({"NB_PlugIn": "NB_Plugin"})
    for col in NUMERIC_COLUMNS:
        if col in clean.columns:
            clean[col] = pd.to_numeric(clean[col], errors="coerce")
    if "coverage" in clean.columns:
        clean["is_valid"] = clean["coverage"] >= 0.88
    return clean


def load_results(paths: str | Path | Sequence[str | Path]) -> pd.DataFrame:
    return preprocess_results(combine_result_files(paths))


def filter_methods(df: pd.DataFrame, methods: Iterable[str] | None = None) -> pd.DataFrame:
    selected = list(methods or DEFAULT_METHODS)
    return df[df["method"].isin(selected)].copy()


def infer_baseline(df: pd.DataFrame, desired: dict | None = None) -> dict:
    target = desired or DEFAULT_BASELINE
    baseline: dict = {}
    for col, expected in target.items():
        if col not in df.columns:
            continue
        available = pd.Series(df[col].dropna().unique())
        if available.empty:
            continue
        if (available == expected).any():
            baseline[col] = expected
        elif pd.api.types.is_numeric_dtype(available):
            baseline[col] = min(available.tolist(), key=lambda v: abs(v - expected))
        else:
            mode = df[col].mode(dropna=True)
            baseline[col] = mode.iloc[0] if not mode.empty else available.iloc[0]
    return baseline


def apply_baseline_filter(
    df: pd.DataFrame,
    baseline: dict | None = None,
    exclude: str | None = None,
) -> pd.DataFrame:
    resolved = baseline or DEFAULT_BASELINE
    mask = pd.Series(True, index=df.index)
    for col, value in resolved.items():
        if col == exclude or col not in df.columns:
            continue
        mask &= df[col].eq(value)
    return df.loc[mask].copy()


def compute_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    agg_dict = {
        "coverage": ["mean", "std", "min", "max"],
        "mean_width": ["mean", "std"],
        "coverage_y0": ["mean", "std"],
        "coverage_y1": ["mean", "std"],
        "coverage_gap": ["mean", "std"],
        "interval_score": ["mean", "std"],
        "is_valid": ["mean"],
    }
    available = {k: v for k, v in agg_dict.items() if k in df.columns}
    summary = df.groupby(["dgp_type", "learner", "method"]).agg(available).round(4)
    summary.columns = ["_".join(c).strip() for c in summary.columns.values]
    return summary.reset_index()


def compute_method_rankings(df: pd.DataFrame) -> pd.DataFrame:
    rankings: list[pd.DataFrame] = []
    for (dgp_type, learner), group in df.groupby(["dgp_type", "learner"]):
        method_stats = group.groupby("method").agg({
            "coverage": "mean",
            "mean_width": "mean",
            "interval_score": "mean",
            "coverage_gap": "mean",
        }).reset_index()
        method_stats["rank_coverage"] = method_stats["coverage"].rank(ascending=False)
        method_stats["rank_width"] = method_stats["mean_width"].rank(ascending=True)
        method_stats["rank_interval_score"] = method_stats["interval_score"].rank(ascending=True)
        method_stats["rank_gap"] = method_stats["coverage_gap"].rank(ascending=True)
        method_stats["rank_combined"] = (
            method_stats["rank_coverage"]
            + method_stats["rank_width"]
            + method_stats["rank_interval_score"]
            + method_stats["rank_gap"]
        ) / 4
        method_stats["dgp_type"] = dgp_type
        method_stats["learner"] = learner
        rankings.append(method_stats)
    return pd.concat(rankings, ignore_index=True)


def summarize_results(df: pd.DataFrame) -> dict:
    summary: dict = {"rows": len(df)}
    for col in [
        "dgp_type",
        "learner",
        "method",
        "n_obs",
        "dim_x",
        "correlation_strength",
        "categorical_ratio",
        "zero_inflation",
    ]:
        if col in df.columns:
            summary[col] = sorted(df[col].dropna().unique().tolist())
    return summary


def generate_latex_table(df: pd.DataFrame, output_dir: str | Path) -> str:
    """Generate a LaTeX-formatted results table for the paper."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_methods = df.groupby("method").agg({
        "coverage": "mean",
        "mean_width": "mean",
        "coverage_y0": "mean",
        "coverage_y1": "mean",
        "coverage_gap": "mean",
        "interval_score": "mean",
    }).round(4)

    rows = []
    for method in baseline_methods.index:
        display = get_display_name(method)
        row = baseline_methods.loc[method]
        rows.append(
            f"{display} & {row['coverage']:.4f} & {row['mean_width']:.2f} "
            f"& {row['coverage_y0']:.4f} & {row['coverage_y1']:.4f} "
            f"& {row['coverage_gap']:.4f} & {row['interval_score']:.4f}"
        )

    latex = (
        "\\begin{tabular}{lcccccc}\n"
        "\\toprule\n"
        "Method & Coverage & Width & Cov($y=0$) & Cov($y=1$) & Gap & Score \\\\\n"
        "\\midrule\n"
        + "\\\\\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}\n"
    )

    (output_dir / "results_table.tex").write_text(latex, encoding="utf-8")
    return latex
