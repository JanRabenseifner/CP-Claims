import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
import warnings
from pathlib import Path
from dataclasses import dataclass

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
warnings.filterwarnings('ignore')


# CELL 11: Evaluation Metrics

def interval_coverage(truth: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Empirical coverage rate."""
    covered = (truth >= lower) & (truth <= upper)
    return np.nanmean(covered)


def interval_width(lower: np.ndarray, upper: np.ndarray) -> float:
    """Mean interval width."""
    return np.nanmean(upper - lower)


def interval_score(truth: np.ndarray, lower: np.ndarray, upper: np.ndarray,
                   alpha: float) -> float:
    """Interval score (proper scoring rule)."""
    truth = np.asarray(truth, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)

    width = upper - lower
    below_penalty = (2 / alpha) * np.abs(lower - truth) * (truth < lower)
    above_penalty = (2 / alpha) * np.abs(truth - upper) * (truth > upper)

    return np.nanmean(width + below_penalty + above_penalty)


def outcome_conditional_coverage(
    truth: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray
) -> Dict[str, float]:
    """Coverage by outcome group - KEY metric for class-conditional validity."""
    truth = np.asarray(truth)
    lower = np.asarray(lower)
    upper = np.asarray(upper)

    covered = (truth >= lower) & (truth <= upper)

    results = {}

    # Claims = 0
    mask_0 = truth == 0
    if mask_0.sum() > 0:
        results["Coverage_Claims_0"] = covered[mask_0].mean()
        results["N_Claims_0"] = mask_0.sum()

    # Claims = 1
    mask_1 = truth == 1
    if mask_1.sum() > 0:
        results["Coverage_Claims_1"] = covered[mask_1].mean()
        results["N_Claims_1"] = mask_1.sum()

    # Claims >= 2
    mask_2plus = truth >= 2
    if mask_2plus.sum() > 0:
        results["Coverage_Claims_2plus"] = covered[mask_2plus].mean()
        results["N_Claims_2plus"] = mask_2plus.sum()

    # Overall
    results["Coverage_Overall"] = covered.mean()
    results["N_Total"] = len(truth)

    return results


def evaluate_method(
    truth: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    alpha: float,
    clusters: np.ndarray = None,
    mu_pred: np.ndarray = None
) -> Dict[str, Any]:
    """Comprehensive evaluation of a prediction interval method."""
    truth = np.asarray(truth)
    lower = np.asarray(lower)
    upper = np.asarray(upper)

    widths = upper - lower

    # Masks for claimants vs non-claimants
    claimant_mask = truth > 0
    non_claimant_mask = truth == 0

    results = {
        "Coverage": interval_coverage(truth, lower, upper),
        "Coverage_Dev": interval_coverage(truth, lower, upper) - (1 - alpha),
        "Mean_Width": interval_width(lower, upper),
        "Median_Width": np.nanmedian(widths),
        "Interval_Score": interval_score(truth, lower, upper, alpha),
        "Pct_Singleton": 100 * np.mean(widths == 0),
        "Pct_Width_1": 100 * np.mean(widths <= 1),
        "Pct_Width_2": 100 * np.mean(widths <= 2),
    }

    # NEW METRICS: Singleton analysis
    # Pct_Singleton_Zero: % of intervals that are exactly {0}
    singleton_zero_mask = (lower == 0) & (upper == 0)
    results["Pct_Singleton_Zero"] = 100 * np.mean(singleton_zero_mask)

    # Pct_Singleton_Positive: % of intervals that are a single positive count
    singleton_positive_mask = (lower == upper) & (lower > 0)
    results["Pct_Singleton_Positive"] = 100 * np.mean(singleton_positive_mask)

    # NEW METRICS: Width by claimant status
    if non_claimant_mask.sum() > 0:
        results["Mean_Width_Non_Claimants"] = np.mean(widths[non_claimant_mask])
    else:
        results["Mean_Width_Non_Claimants"] = np.nan

    if claimant_mask.sum() > 0:
        results["Mean_Width_Claimants"] = np.mean(widths[claimant_mask])
    else:
        results["Mean_Width_Claimants"] = np.nan

    # NEW METRIC: Pct_Informative - % of intervals not {0,1} or [0,1]
    # Informative = not (lower==0 and upper<=1)
    trivial_mask = (lower == 0) & (upper <= 1)
    results["Pct_Informative"] = 100 * (1 - np.mean(trivial_mask))

    # NEW METRIC: Risk_Differentiation - correlation(μ, interval_width)
    if mu_pred is not None:
        mu_pred = np.asarray(mu_pred)
        if len(mu_pred) > 1 and np.std(widths) > 0 and np.std(mu_pred) > 0:
            results["Risk_Differentiation"] = np.corrcoef(mu_pred, widths)[0, 1]
        else:
            results["Risk_Differentiation"] = np.nan
    else:
        results["Risk_Differentiation"] = np.nan

    # Outcome-conditional coverage
    outcome_cov = outcome_conditional_coverage(truth, lower, upper)
    results.update(outcome_cov)

    # Cluster-conditional coverage
    if clusters is not None:
        covered = (truth >= lower) & (truth <= upper)
        cluster_cov = pd.DataFrame({"cluster": clusters, "covered": covered})
        cluster_stats = cluster_cov.groupby("cluster")["covered"].mean()
        results["Min_Cluster_Cov"] = cluster_stats.min()
        results["Max_Cluster_Cov"] = cluster_stats.max()
        results["Cov_Range"] = cluster_stats.max() - cluster_stats.min()

    return results




def actuarial_metrics(truth, lower, upper, mu_pred, clusters=None):
    """
    Metrics focusing on actuarially relevant performance.

    Includes:
    - Pct_Singleton_Zero: % of intervals that are exactly {0}
    - Pct_Singleton_Positive: % of intervals that are a single positive count
    - Mean_Width_Non_Claimants: average width for policies without claims
    - Mean_Width_Claimants: average width for policies with claims > 0
    - Pct_Informative: % of intervals not {0,1} or [0,1]
    - Risk_Differentiation: correlation(μ, interval_width)
    """
    metrics = {}

    truth = np.asarray(truth)
    lower = np.asarray(lower)
    upper = np.asarray(upper)
    mu_pred = np.asarray(mu_pred)
    width = upper - lower

    # 1. Coverage for claimants only
    claimant_mask = truth > 0
    non_claimant_mask = truth == 0

    if claimant_mask.sum() > 0:
        metrics["Coverage_Claimants"] = (
            (truth[claimant_mask] >= lower[claimant_mask]) &
            (truth[claimant_mask] <= upper[claimant_mask])
        ).mean()

    # 2. Singleton metrics
    # Singleton {0}: lower == 0 and upper == 0
    singleton_zero_mask = (lower == 0) & (upper == 0)
    metrics["Pct_Singleton_Zero"] = 100 * singleton_zero_mask.mean()

    # Singleton positive: lower == upper and lower > 0
    singleton_positive_mask = (lower == upper) & (lower > 0)
    metrics["Pct_Singleton_Positive"] = 100 * singleton_positive_mask.mean()

    # 3. Width by claimant status
    if non_claimant_mask.sum() > 0:
        metrics["Mean_Width_Non_Claimants"] = width[non_claimant_mask].mean()
    else:
        metrics["Mean_Width_Non_Claimants"] = np.nan

    if claimant_mask.sum() > 0:
        metrics["Mean_Width_Claimants"] = width[claimant_mask].mean()
    else:
        metrics["Mean_Width_Claimants"] = np.nan

    # 4. Interval width vs. predicted μ (relative efficiency)
    # Efficiency ratio: width / (μ + 1)
    efficiency_ratio = width / (mu_pred + 1)
    metrics["Median_Efficiency_Ratio"] = np.median(efficiency_ratio)
    metrics["Efficiency_Ratio_IQ"] = np.percentile(efficiency_ratio, 75) - np.percentile(efficiency_ratio, 25)

    # 5. Risk Differentiation (Discrimination power)
    # How well does interval width correlate with predicted risk?
    if len(mu_pred) > 1:
        metrics["Risk_Differentiation"] = np.corrcoef(mu_pred, width)[0, 1]
    else:
        metrics["Risk_Differentiation"] = np.nan

    # Also compute for claimants only
    if claimant_mask.sum() > 1:
        metrics["Risk_Width_Correlation"] = np.corrcoef(mu_pred[claimant_mask], width[claimant_mask])[0, 1]
    else:
        metrics["Risk_Width_Correlation"] = np.nan

    # 6. Over-prediction vs under-prediction bias
    lower_error = np.where(truth < lower, truth - lower, 0)
    upper_error = np.where(truth > upper, truth - upper, 0)
    metrics["Mean_Underprediction"] = np.mean(lower_error)
    metrics["Mean_Overprediction"] = np.mean(upper_error)

    # 7. Practical utility: % of intervals that are informative
    # Informative = not {0}, not {0,1}, not [0,1]
    trivial_mask = ((lower == 0) & (upper <= 1))
    metrics["Pct_Informative"] = 100 * (1 - trivial_mask.mean())
    metrics["Pct_Informative_Intervals"] = metrics["Pct_Informative"]  # Alias for compatibility

    return metrics


def cluster_risk_metrics(test_data, intervals, cluster_col="Cluster"):
    """
    Evaluate risk segmentation quality by cluster.
    """
    results = []

    for cluster in sorted(test_data[cluster_col].unique()):
        mask = test_data[cluster_col] == cluster
        cluster_y = test_data.loc[mask, "Claims"].values
        cluster_mu = test_data.loc[mask, "pred_mu"].values

        # Get intervals
        if "lower" in intervals.columns:
            lower = intervals.loc[mask, "lower"].values
            upper = intervals.loc[mask, "upper"].values
        else:
            lower = intervals.loc[mask, "lower_bound"].values
            upper = intervals.loc[mask, "upper_bound"].values

        # Actual vs predicted risk in cluster
        actual_freq = cluster_y.mean()
        predicted_freq = cluster_mu.mean()

        # Risk discrimination
        claimant_mask = cluster_y > 0
        non_claimant_mask = cluster_y == 0

        if claimant_mask.sum() > 0 and non_claimant_mask.sum() > 0:
            # Mean width for claimants vs non-claimants
            width_claimants = (upper[claimant_mask] - lower[claimant_mask]).mean()
            width_non = (upper[non_claimant_mask] - lower[non_claimant_mask]).mean()

            # Risk differentiation metric
            risk_diff = width_claimants - width_non
        else:
            width_claimants = width_non = risk_diff = np.nan

        results.append({
            "Cluster": cluster,
            "N": mask.sum(),
            "Actual_Freq": actual_freq,
            "Predicted_Freq": predicted_freq,
            "Freq_Ratio": actual_freq / (predicted_freq + 1e-6),
            "Mean_Width": (upper - lower).mean(),
            "Width_Claimants": width_claimants,
            "Width_NonClaimants": width_non,
            "Risk_Differentiation": risk_diff,
            "Pct_Claimants": 100 * claimant_mask.mean()
        })

    return pd.DataFrame(results)



def evaluate_group_specific_intervals(
    test_y: np.ndarray,
    intervals_by_group: Dict[str, Tuple[np.ndarray, np.ndarray]]
) -> pd.DataFrame:
    """
    Evaluate coverage using group-specific intervals (without union).

    intervals_by_group: Dictionary with keys 'y0', 'y1', 'y2plus'
                       containing (lower, upper) arrays
    """
    results = []

    for outcome, (lower, upper) in intervals_by_group.items():
        if outcome == "y0":
            mask = test_y == 0
        elif outcome == "y1":
            mask = test_y == 1
        else:  # y2plus
            mask = test_y >= 2

        if mask.sum() > 0:
            coverage = ((test_y[mask] >= lower[mask]) &
                       (test_y[mask] <= upper[mask])).mean()
            width = (upper[mask] - lower[mask]).mean()

            results.append({
                "Outcome_Group": outcome,
                "Coverage": coverage,
                "Mean_Width": width,
                "N": mask.sum()
            })

    return pd.DataFrame(results)



def evaluate_cluster_performance(
    test_data: pd.DataFrame,
    intervals: pd.DataFrame,
    alpha: float = 0.1,
    cluster_col: str = "Cluster"
) -> pd.DataFrame:
    """
    Comprehensive cluster-level evaluation.

    Returns coverage and width metrics for each cluster,
    broken down by outcome group.
    """
    results = []

    for cluster in sorted(test_data[cluster_col].unique()):
        mask = test_data[cluster_col] == cluster
        cluster_y = test_data.loc[mask, "Claims"].values
        cluster_mu = test_data.loc[mask, "pred_mu"].values

        if "lower" in intervals.columns:
            cluster_lower = intervals.loc[mask, "lower"].values
            cluster_upper = intervals.loc[mask, "upper"].values
        else:
            cluster_lower = intervals.loc[mask, "lower_bound"].values
            cluster_upper = intervals.loc[mask, "upper_bound"].values

        n_total = mask.sum()

        if n_total == 0:
            continue

        # Overall cluster metrics
        covered = (cluster_y >= cluster_lower) & (cluster_y <= cluster_upper)
        coverage_overall = covered.mean()
        mean_width = (cluster_upper - cluster_lower).mean()

        # Outcome-specific metrics
        for outcome_label in ["0", "1", "2plus"]:
            if outcome_label == "0":
                outcome_mask = cluster_y == 0
            elif outcome_label == "1":
                outcome_mask = cluster_y == 1
            else:  # 2plus
                outcome_mask = cluster_y >= 2

            n_outcome = outcome_mask.sum()

            if n_outcome > 0:
                coverage_outcome = covered[outcome_mask].mean()
                width_outcome = (cluster_upper[outcome_mask] - cluster_lower[outcome_mask]).mean()

                results.append({
                    "Cluster": cluster,
                    "Outcome_Group": outcome_label,
                    "N_Total": n_total,
                    "N_Outcome": n_outcome,
                    "Coverage": coverage_outcome,
                    "Mean_Width": width_outcome,
                    "Coverage_Dev": coverage_outcome - (1 - alpha),
                    "Pct_Singleton": 100 * (cluster_upper[outcome_mask] - cluster_lower[outcome_mask] == 0).mean()
                })

        # Add overall cluster summary
        results.append({
            "Cluster": cluster,
            "Outcome_Group": "Overall",
            "N_Total": n_total,
            "N_Outcome": n_total,
            "Coverage": coverage_overall,
            "Mean_Width": mean_width,
            "Coverage_Dev": coverage_overall - (1 - alpha),
            "Pct_Singleton": 100 * (cluster_upper - cluster_lower == 0).mean()
        })

    return pd.DataFrame(results)




# Add to Evaluation: Binary vs. Count Comparison

def compare_binary_vs_count_performance():
    """
    Compare how well binary classification identifies claimants
    vs. count models' ability to predict claim counts.
    """
    # Binary metrics from your current count intervals
    def binary_metrics_from_count_intervals(truth, lower, upper):
        """
        Convert count intervals to binary predictions for comparison.
        """
        # If interval includes 0, predict "no claim" possible
        # If interval excludes 0, predict "claim" definite
        # Otherwise ambiguous

        binary_predictions = []
        for t, l, u in zip(truth, lower, upper):
            if l > 0:
                # Interval excludes 0 → definite claim
                binary_predictions.append(1)
            elif u == 0:
                # Interval is exactly [0,0] → definite no claim
                binary_predictions.append(0)
            else:
                # Interval includes both 0 and >0 → ambiguous
                binary_predictions.append(None)  # Can't decide

        return np.array(binary_predictions)

    # Compare methods
    comparison_results = []

    for method_name in ["ICP_poisson_pearson", "Parametric_Poisson",
                        "Outcome_Cond_CP", "Mondrian_Poisson"]:
        if method_name in results:
            result_df = results[method_name]
            lower = result_df["lower"].values if "lower" in result_df else result_df["lower_bound"].values
            upper = result_df["upper"].values if "upper" in result_df else result_df["upper_bound"].values

            # Binary predictions from count intervals
            binary_pred = binary_metrics_from_count_intervals(test_y, lower, upper)
            valid_mask = ~np.isnan(binary_pred)

            if valid_mask.sum() > 0:
                # Binary classification metrics
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

                y_true_binary = (test_y > 0).astype(int)
                y_pred_binary = binary_pred[valid_mask].astype(int)
                y_true_masked = y_true_binary[valid_mask]

                metrics = {
                    "Method": method_name,
                    "Binary_Accuracy": accuracy_score(y_true_masked, y_pred_binary),
                    "Binary_Precision": precision_score(y_true_masked, y_pred_binary, zero_division=0),
                    "Binary_Recall": recall_score(y_true_masked, y_pred_binary, zero_division=0),
                    "Binary_F1": f1_score(y_true_masked, y_pred_binary, zero_division=0),
                    "Pct_Definite_Predictions": 100 * valid_mask.mean(),
                    "Mean_Count_Width": (upper - lower).mean()
                }
                comparison_results.append(metrics)

    return pd.DataFrame(comparison_results)