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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
warnings.filterwarnings('ignore')




# CELL 4: Non-Conformity Score Functions

def poisson_pearson_ncs(pred: np.ndarray, truth: np.ndarray, eps: float = 0.1) -> np.ndarray:
    """
    Poisson Pearson Residual - Variance-Stabilizing NCS.

    For Poisson:  Var(Y) = μ, so (Y - μ)/√μ has approximately unit variance.
    RECOMMENDED for claims frequency.
    """
    pred = np.asarray(pred, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    return (truth - pred) / np.sqrt(pred + eps)


def poisson_deviance_ncs(pred: np.ndarray, truth: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Poisson Deviance Residual NCS.

    Better tail behavior than Pearson, based on likelihood ratio.
    """
    pred = np.asarray(pred, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    dev_contrib = np.where(
        truth == 0,
        2 * pred,
        2 * (truth * np.log(np.maximum(truth, eps) / np.maximum(pred, eps)) - (truth - pred))
    )
    return np.sign(truth - pred) * np.sqrt(np.maximum(dev_contrib, 0))


def za_relative_ncs(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Zero-Adjusted Relative Error:  |Y - μ| / (1 + μ)"""
    pred = np.asarray(pred, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    return np.abs(truth - pred) / (1 + pred)


def absolute_ncs(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Standard Absolute Error NCS."""
    return np.abs(np.asarray(truth) - np.asarray(pred))


# Registry
NCS_FUNCTIONS = {
    "poisson_pearson": poisson_pearson_ncs,
    "poisson_deviance": poisson_deviance_ncs,
    "za_relative": za_relative_ncs,
    "absolute": absolute_ncs,
}

SIGNED_NCS_TYPES = {"poisson_pearson", "poisson_deviance"}

NCS_DESCRIPTIONS = {
    "poisson_pearson": "(Y - μ) / √μ [Variance-stabilizing]",
    "poisson_deviance": "sign(Y-μ)·√(2·dev) [Likelihood-based]",
    "za_relative": "|Y - μ| / (1 + μ) [Zero-adjusted]",
    "absolute": "|Y - μ| [Standard]",
}


def get_ncs_function(ncs_type: str) -> Callable:
    if ncs_type not in NCS_FUNCTIONS:
        raise ValueError(f"Unknown NCS:  {ncs_type}")
    return NCS_FUNCTIONS[ncs_type]


def is_signed_ncs(ncs_type: str) -> bool:
    return ncs_type in SIGNED_NCS_TYPES




# CELL 5: Conformal Prediction Core Functions

def compute_conformal_quantiles(calib_ncs: np.ndarray, alpha: float,
                                 ncs_type: str) -> Dict[str, float]:
    """Compute conformal quantile thresholds from calibration scores."""
    n_calib = len(calib_ncs)
    q_level = np.ceil((n_calib + 1) * (1 - alpha)) / n_calib
    q_level = min(q_level, 1.0)

    if is_signed_ncs(ncs_type):
        q_lower = np.nanquantile(calib_ncs, alpha / 2)
        q_upper = np.nanquantile(calib_ncs, min(1 - alpha / 2, 1.0))
        return {"q_lower": q_lower, "q_upper": q_upper}
    else:
        q_threshold = np.nanquantile(calib_ncs, q_level)
        return {"q_threshold": q_threshold}


def find_prediction_set(mu: float, quantiles: Dict[str, float],
                        ncs_type: str, max_count: int = 50) -> Tuple[int, int]:
    """Find the prediction set for a single observation."""
    if np.isnan(mu) or mu <= 0:
        return (np.nan, np.nan)

    ncs_fun = get_ncs_function(ncs_type)
    candidates = np.arange(0, max_count + 1)
    candidate_ncs = ncs_fun(np.full_like(candidates, mu, dtype=float), candidates)

    if is_signed_ncs(ncs_type):
        in_set = (candidate_ncs >= quantiles["q_lower"]) & (candidate_ncs <= quantiles["q_upper"])
    else:
        in_set = candidate_ncs <= quantiles["q_threshold"]

    valid_counts = candidates[in_set]

    if len(valid_counts) == 0:
        return (max(0, int(np.floor(mu))), int(np.ceil(mu)))
    return (int(np.min(valid_counts)), int(np.max(valid_counts)))


def discrete_conformal_poisson(pred_mu: np.ndarray, calib_mu: np.ndarray,
                                calib_y: np.ndarray, alpha: float = 0.1,
                                ncs_type: str = "poisson_pearson",
                                max_count: int = None) -> pd.DataFrame:
    """Standard Inductive Conformal Prediction for count data."""
    pred_mu = np.asarray(pred_mu, dtype=np.float64)
    calib_mu = np.asarray(calib_mu, dtype=np.float64)
    calib_y = np.asarray(calib_y, dtype=np.float64)

    ncs_fun = get_ncs_function(ncs_type)
    calib_ncs = ncs_fun(calib_mu, calib_y)
    quantiles = compute_conformal_quantiles(calib_ncs, alpha, ncs_type)

    if max_count is None:
        max_count = min(int(max(np.nanmax(calib_y), np.ceil(np.nanmax(pred_mu) * 3))) + 5, 50)

    results = []
    for mu in pred_mu:
        lower, upper = find_prediction_set(mu, quantiles, ncs_type, max_count)
        results.append({"pred": mu, "lower": lower, "upper": upper})

    return pd.DataFrame(results)




# CELL 5B: Binary Classification CP Functions
from typing import Set, List, Dict

def binary_classification_cp(
    pred_probs: np.ndarray,  # Shape [n_samples, 2] - probabilities for classes 0 and 1
    calib_probs: np.ndarray,
    calib_y_binary: np.ndarray,  # 0 or 1
    alpha: float = 0.1,
    mondrian: bool = False
) -> List[Set[int]]:
    """
    Binary Classification Conformal Prediction.

    Based on claims_cp_binary.md approach.

    Parameters:
    -----------
    pred_probs: Probabilities for test set [P(Y=0), P(Y=1)]
    calib_probs: Probabilities for calibration set
    calib_y_binary: Binary labels for calibration set
    alpha: Significance level
    mondrian: If True, use label-conditional (Mondrian) CP

    Returns:
    --------
    List of prediction sets: {0}, {1}, or {0, 1}
    """
    # Non-conformity measure: 1 - P(true class)
    calib_scores = np.array([
        1 - p[y] for p, y in zip(calib_probs, calib_y_binary)
    ])

    if mondrian:
        # Label-Conditional Mondrian CP
        calib_scores_0 = calib_scores[calib_y_binary == 0]
        calib_scores_1 = calib_scores[calib_y_binary == 1]

        n0 = len(calib_scores_0)
        n1 = len(calib_scores_1)

        # Compute class-specific thresholds with proper clamping
        q0 = min(np.ceil((n0 + 1) * (1 - alpha)) / n0, 1.0) if n0 > 0 else 1.0
        q1 = min(np.ceil((n1 + 1) * (1 - alpha)) / n1, 1.0) if n1 > 0 else 1.0

        threshold_0 = np.quantile(calib_scores_0, q0) if n0 > 0 else 1.0
        threshold_1 = np.quantile(calib_scores_1, q1) if n1 > 0 else 1.0

        # Generate prediction sets
        prediction_sets = []
        for p in pred_probs:
            set_classes = set()
            if (1 - p[0]) <= threshold_0:  # Include class 0
                set_classes.add(0)
            if (1 - p[1]) <= threshold_1:  # Include class 1
                set_classes.add(1)
            # Ensure non-empty set (conservative fallback)
            if len(set_classes) == 0:
                set_classes = {0, 1}
            prediction_sets.append(set_classes)
    else:
        # Marginal CP
        n_calib = len(calib_scores)
        q_level = min(np.ceil((n_calib + 1) * (1 - alpha)) / n_calib, 1.0)
        threshold = np.quantile(calib_scores, q_level)

        # Generate prediction sets
        prediction_sets = []
        for p in pred_probs:
            set_classes = set()
            if (1 - p[0]) <= threshold:  # Include class 0
                set_classes.add(0)
            if (1 - p[1]) <= threshold:  # Include class 1
                set_classes.add(1)
            # Ensure non-empty set (conservative fallback)
            if len(set_classes) == 0:
                set_classes = {0, 1}
            prediction_sets.append(set_classes)

    return prediction_sets


def binary_metrics_from_prediction_sets(
    prediction_sets: List[Set[int]],
    true_labels: np.ndarray) -> Dict[str, float]:
    """
    Compute binary classification metrics from prediction sets.

    Handles ambiguous predictions ({0,1}) by excluding or treating as uncertain.
    """
    metrics = {}

    # Convert to definite predictions when possible
    definite_preds = []
    definite_true = []

    for pred_set, true_label in zip(prediction_sets, true_labels):
        if len(pred_set) == 1:
            # Definite prediction
            definite_preds.append(list(pred_set)[0])
            definite_true.append(true_label)

    metrics["Pct_Definite"] = 100 * len(definite_preds) / len(prediction_sets)

    if len(definite_preds) > 0:

        definite_preds = np.array(definite_preds)
        definite_true = np.array(definite_true)

        metrics["Binary_Accuracy"] = accuracy_score(definite_true, definite_preds)
        metrics["Binary_Precision"] = precision_score(definite_true, definite_preds, zero_division=0)
        metrics["Binary_Recall"] = recall_score(definite_true, definite_preds, zero_division=0)
        metrics["Binary_F1"] = f1_score(definite_true, definite_preds, zero_division=0)
        metrics["N_Definite"] = len(definite_preds)

    # Coverage metrics
    covered = [true_label in pred_set for pred_set, true_label in zip(prediction_sets, true_labels)]
    metrics["Coverage_Binary"] = np.mean(covered)

    # Conditional coverage
    mask_0 = true_labels == 0
    mask_1 = true_labels == 1

    if mask_0.sum() > 0:
        metrics["Coverage_Binary_Y0"] = np.mean([true_label in pred_set
                                                 for pred_set, true_label
                                                 in zip(prediction_sets, true_labels)
                                                 if true_label == 0])
    if mask_1.sum() > 0:
        metrics["Coverage_Binary_Y1"] = np.mean([true_label in pred_set
                                                 for pred_set, true_label
                                                 in zip(prediction_sets, true_labels)
                                                 if true_label == 1])

    # Average set size (0.5 for {0} or {1}, 1 for {0,1})
    metrics["Avg_Set_Size"] = np.mean([len(s) for s in prediction_sets])

    return metrics

# CELL 6: Mondrian (Cluster-Conditional) Conformal Prediction

def mondrian_conformal_poisson(
    pred_mu: np.ndarray,
    pred_class: np.ndarray,
    calib_mu: np.ndarray,
    calib_y: np.ndarray,
    calib_class: np.ndarray,
    alpha: float = 0.1,
    ncs_type: str = "poisson_pearson",
    min_calib_size: int = 30,
    max_count: int = 50
) -> pd.DataFrame:
    """
    Mondrian Conformal Prediction - calibrates separately per cluster.

    Provides cluster-conditional coverage guarantees.
    """
    pred_mu = np.asarray(pred_mu, dtype=np.float64)
    pred_class = np.asarray(pred_class)
    calib_mu = np.asarray(calib_mu, dtype=np.float64)
    calib_y = np.asarray(calib_y, dtype=np.float64)
    calib_class = np.asarray(calib_class)

    ncs_fun = get_ncs_function(ncs_type)
    results = []

    for i, (mu_i, class_i) in enumerate(zip(pred_mu, pred_class)):
        if np.isnan(mu_i) or mu_i <= 0:
            results.append({"pred": mu_i, "lower": np.nan, "upper": np.nan, "class": class_i})
            continue

        # Get class-specific calibration data
        mask = calib_class == class_i
        n_class = mask.sum()

        if n_class < min_calib_size:
            # Fallback to global calibration
            calib_ncs = ncs_fun(calib_mu, calib_y)
        else:
            calib_ncs = ncs_fun(calib_mu[mask], calib_y[mask])

        quantiles = compute_conformal_quantiles(calib_ncs, alpha, ncs_type)
        lower, upper = find_prediction_set(mu_i, quantiles, ncs_type, max_count)

        results.append({"pred": mu_i, "lower": lower, "upper": upper, "class": class_i})

    return pd.DataFrame(results)





# CELL 7: Outcome-Conditional CP (DGCP-Inspired)

def get_outcome_group(y: np.ndarray, min_group_size: int = 30) -> np.ndarray:
    """
    Assign observations to outcome groups.
    Groups rare outcomes together (DGCP principle).
    """
    y = np.asarray(y)
    groups = np.where(y == 0, "zero", np.where(y == 1, "one", "two_plus"))

    # Check if we need to merge groups
    group_counts = pd.Series(groups).value_counts()

    # If "two_plus" too small, merge with "one" into "one_plus"
    if group_counts.get("two_plus", 0) < min_group_size:
        groups = np.where(groups == "two_plus", "one_plus", groups)
        groups = np.where(groups == "one", "one_plus", groups)

    return groups


def outcome_conditional_cp(
    pred_mu: np.ndarray,
    calib_mu: np.ndarray,
    calib_y: np.ndarray,
    alpha: float = 0.1,
    ncs_type: str = "poisson_pearson",
    min_samples: int = 30,
    max_count: int = 50
) -> pd.DataFrame:
    """
    Outcome-Conditional Conformal Prediction (DGCP-inspired).

    Calibrates separately per outcome group, then returns union of intervals.
    This ensures valid coverage for each outcome group.

    Key insight: We compute prediction sets that are valid conditional on
    each possible outcome, then take the union (conservative but valid).
    """
    pred_mu = np.asarray(pred_mu, dtype=np.float64)
    calib_mu = np.asarray(calib_mu, dtype=np.float64)
    calib_y = np.asarray(calib_y, dtype=np.float64)

    ncs_fun = get_ncs_function(ncs_type)

    # Get outcome groups (with automatic merging of rare groups)
    calib_groups = get_outcome_group(calib_y, min_samples)
    unique_groups = np.unique(calib_groups)

    # Compute quantiles per outcome group
    group_quantiles = {}
    for group in unique_groups:
        mask = calib_groups == group
        if mask.sum() >= min_samples:
            calib_ncs_g = ncs_fun(calib_mu[mask], calib_y[mask])
            group_quantiles[group] = compute_conformal_quantiles(calib_ncs_g, alpha, ncs_type)
        else:
            # Fallback to global
            calib_ncs = ncs_fun(calib_mu, calib_y)
            group_quantiles[group] = compute_conformal_quantiles(calib_ncs, alpha, ncs_type)

    # For each prediction, compute union of intervals across outcome groups
    results = []
    for mu_i in pred_mu:
        if np.isnan(mu_i) or mu_i <= 0:
            results.append({"pred": mu_i, "lower": np.nan, "upper": np.nan})
            continue

        all_lowers = []
        all_uppers = []

        for group, quantiles in group_quantiles.items():
            lower_g, upper_g = find_prediction_set(mu_i, quantiles, ncs_type, max_count)
            if not np.isnan(lower_g):
                all_lowers.append(lower_g)
                all_uppers.append(upper_g)

        if len(all_lowers) > 0:
            lower = int(min(all_lowers))
            upper = int(max(all_uppers))
        else:
            lower = max(0, int(np.floor(mu_i)))
            upper = int(np.ceil(mu_i))

        results.append({"pred": mu_i, "lower": lower, "upper": upper})

    return pd.DataFrame(results)



# CELL 8: Hybrid (Cluster × Outcome) Conditional CP

def hybrid_conditional_cp(
    pred_mu: np.ndarray,
    pred_cluster: np.ndarray,
    calib_mu: np.ndarray,
    calib_y: np.ndarray,
    calib_cluster: np.ndarray,
    alpha: float = 0.1,
    ncs_type: str = "poisson_pearson",
    min_samples: int = 30,
    max_count: int = 50
) -> pd.DataFrame:
    """
    Hybrid Conditional CP:  Cluster × Outcome stratification.

    Strategy:
    1. Stratify by cluster (actuarial segments)
    2. Within each cluster, apply outcome-conditional calibration
    3. For rare strata, apply DGCP grouping principle
    """
    pred_mu = np.asarray(pred_mu, dtype=np.float64)
    pred_cluster = np.asarray(pred_cluster)
    calib_mu = np.asarray(calib_mu, dtype=np.float64)
    calib_y = np.asarray(calib_y, dtype=np.float64)
    calib_cluster = np.asarray(calib_cluster)

    ncs_fun = get_ncs_function(ncs_type)

    # Get outcome groups
    calib_outcome = np.where(calib_y == 0, "zero", "one_plus")

    # Create combined strata:  cluster × outcome
    calib_strata = np.array([f"{c}_{o}" for c, o in zip(calib_cluster, calib_outcome)])

    # Compute quantiles per stratum (with fallback)
    strata_quantiles = {}
    unique_strata = np.unique(calib_strata)

    for stratum in unique_strata:
        mask = calib_strata == stratum
        if mask.sum() >= min_samples:
            calib_ncs_s = ncs_fun(calib_mu[mask], calib_y[mask])
            strata_quantiles[stratum] = compute_conformal_quantiles(calib_ncs_s, alpha, ncs_type)

    # Global fallback quantiles
    global_ncs = ncs_fun(calib_mu, calib_y)
    global_quantiles = compute_conformal_quantiles(global_ncs, alpha, ncs_type)

    results = []
    for i, (mu_i, cluster_i) in enumerate(zip(pred_mu, pred_cluster)):
        if np.isnan(mu_i) or mu_i <= 0:
            results.append({"pred": mu_i, "lower": np.nan, "upper": np.nan, "cluster": cluster_i})
            continue

        all_lowers = []
        all_uppers = []

        # Check both outcome groups for this cluster
        for outcome in ["zero", "one_plus"]:
            stratum = f"{cluster_i}_{outcome}"

            if stratum in strata_quantiles:
                quantiles = strata_quantiles[stratum]
            else:
                # Fallback:  try cluster-only, then global
                cluster_mask = calib_cluster == cluster_i
                if cluster_mask.sum() >= min_samples:
                    calib_ncs_c = ncs_fun(calib_mu[cluster_mask], calib_y[cluster_mask])
                    quantiles = compute_conformal_quantiles(calib_ncs_c, alpha, ncs_type)
                else:
                    quantiles = global_quantiles

            lower_g, upper_g = find_prediction_set(mu_i, quantiles, ncs_type, max_count)
            if not np.isnan(lower_g):
                all_lowers.append(lower_g)
                all_uppers.append(upper_g)

        if len(all_lowers) > 0:
            lower = int(min(all_lowers))
            upper = int(max(all_uppers))
        else:
            lower = max(0, int(np.floor(mu_i)))
            upper = int(np.ceil(mu_i))

        results.append({"pred": mu_i, "lower": lower, "upper": upper, "cluster": cluster_i})

    return pd.DataFrame(results)





def adaptive_outcome_cp(
    pred_mu: np.ndarray,
    calib_mu: np.ndarray,
    calib_y: np.ndarray,
    alpha: float = 0.1,
    ncs_type: str = "poisson_pearson",
    min_samples: int = 30,
    max_count: int = 50
) -> Dict[str, pd.DataFrame]:
    """
    Adaptive Outcome-Conditional CP (DGCP-inspired).

    Returns:
    - union_df: Union intervals (original conservative approach)
    - group_dfs: Dictionary of group-specific intervals for evaluation
    """
    pred_mu = np.asarray(pred_mu, dtype=np.float64)
    calib_mu = np.asarray(calib_mu, dtype=np.float64)
    calib_y = np.asarray(calib_y, dtype=np.float64)

    ncs_fun = get_ncs_function(ncs_type)

    # Define outcome groups
    outcome_masks = {
        "y0": calib_y == 0,
        "y1": calib_y == 1,
        "y2plus": calib_y >= 2
    }

    # Compute quantiles per group with adaptive strategy
    group_quantiles = {}
    group_sizes = {}

    for group_name, mask in outcome_masks.items():
        n_samples = mask.sum()
        group_sizes[group_name] = n_samples

        if n_samples >= min_samples:
            # Group has sufficient data - use group-specific calibration
            calib_ncs_g = ncs_fun(calib_mu[mask], calib_y[mask])
            group_quantiles[group_name] = compute_conformal_quantiles(calib_ncs_g, alpha, ncs_type)
        else:
            # Insufficient data - fall back to marginal calibration
            calib_ncs_all = ncs_fun(calib_mu, calib_y)
            group_quantiles[group_name] = compute_conformal_quantiles(calib_ncs_all, alpha, ncs_type)

    # Generate intervals
    union_results = []
    group_intervals = {g: {"lower": [], "upper": []} for g in outcome_masks.keys()}

    for mu_i in pred_mu:
        if np.isnan(mu_i) or mu_i <= 0:
            for g in outcome_masks.keys():
                group_intervals[g]["lower"].append(np.nan)
                group_intervals[g]["upper"].append(np.nan)
            union_results.append({"pred": mu_i, "lower": np.nan, "upper": np.nan})
            continue

        all_lowers = []
        all_uppers = []

        for group_name, quantiles in group_quantiles.items():
            lower_g, upper_g = find_prediction_set(mu_i, quantiles, ncs_type, max_count)

            # Store group-specific intervals
            group_intervals[group_name]["lower"].append(lower_g)
            group_intervals[group_name]["upper"].append(upper_g)

            if not np.isnan(lower_g):
                all_lowers.append(lower_g)
                all_uppers.append(upper_g)

        # Union interval (conservative)
        if len(all_lowers) > 0:
            lower_union = int(min(all_lowers))
            upper_union = int(max(all_uppers))
        else:
            lower_union = max(0, int(np.floor(mu_i)))
            upper_union = int(np.ceil(mu_i))

        union_results.append({"pred": mu_i, "lower": lower_union, "upper": upper_union})

    # Create DataFrames
    union_df = pd.DataFrame(union_results)

    group_dfs = {}
    for group_name in outcome_masks.keys():
        group_dfs[group_name] = pd.DataFrame({
            "pred": pred_mu,
            "lower": group_intervals[group_name]["lower"],
            "upper": group_intervals[group_name]["upper"]
        })

    return {
        "union": union_df,
        "groups": group_dfs,
        "group_sizes": group_sizes
    }


# CELL 15B: Two-Stage Insurance CP (optimized for minimal interval widths)

def two_stage_insurance_cp(
    pred_mu: np.ndarray,
    calib_mu: np.ndarray,
    calib_y: np.ndarray,
    alpha: float = 0.1,
    alpha_stage1: float = None,
    ncs_type: str = "poisson_deviance"
) -> pd.DataFrame:
    """
    Two-stage CP optimized for minimal interval widths in insurance.

    Stage 1: Binary CP for claim/no-claim (gets more singletons)
    Stage 2: Count CP for claimants (calibrated on claims > 0 only)

    Parameters:
    -----------
    pred_mu : array
        Predicted mean (μ) for test observations
    calib_mu : array
        Predicted mean for calibration observations
    calib_y : array
        True outcomes for calibration observations
    alpha : float
        Overall miscoverage rate (default 0.1 for 90% coverage)
    alpha_stage1 : float
        Alpha for Stage 1 (default: alpha/2 for more aggressive Stage 1)
    ncs_type : str
        NCS type for Stage 2 count CP (poisson_deviance works better for positive counts)

    Returns:
    --------
    DataFrame with pred, lower, upper columns
    """
    pred_mu = np.asarray(pred_mu, dtype=np.float64)
    calib_mu = np.asarray(calib_mu, dtype=np.float64)
    calib_y = np.asarray(calib_y, dtype=np.float64)

    if alpha_stage1 is None:
        alpha_stage1 = alpha  # More aggressive for Stage 1

    # -------------------------------------------------------------------------
    # Stage 1: Binary CP for claim/no-claim
    # -------------------------------------------------------------------------
    # Use efficient binary CP with Mondrian adjustment
    calib_probs_binary = np.column_stack([
        np.exp(-calib_mu),           # P(Y=0)
        1 - np.exp(-calib_mu)        # P(Y>0)
    ])

    test_probs_binary = np.column_stack([
        np.exp(-pred_mu),            # P(Y=0)
        1 - np.exp(-pred_mu)         # P(Y>0)
    ])

    calib_y_binary = (calib_y > 0).astype(int)

    # Optimize for definite predictions (singletons)
    # Use smaller α for Stage 1 to get more {0} or {1} predictions
    binary_sets = binary_classification_cp(
        test_probs_binary, calib_probs_binary, calib_y_binary,
        alpha=alpha_stage1, mondrian=True  # Use Mondrian for better class-conditional coverage
    )

    # -------------------------------------------------------------------------
    # Stage 2: Count CP for claimants
    # -------------------------------------------------------------------------
    # Focus calibration on claims > 0 only
    claimant_mask = calib_y > 0
    claimant_calib_mu = calib_mu[claimant_mask]
    claimant_calib_y = calib_y[claimant_mask]

    # Compute conformal quantiles on positive claims
    ncs_fun = get_ncs_function(ncs_type)

    if len(claimant_calib_y) > 0:
        claimant_ncs = ncs_fun(claimant_calib_mu, claimant_calib_y)
        claimant_quantiles = compute_conformal_quantiles(claimant_ncs, alpha, ncs_type)
    else:
        claimant_quantiles = None

    # -------------------------------------------------------------------------
    # Combine stages
    # -------------------------------------------------------------------------
    results = []

    for i, (mu_i, binary_set) in enumerate(zip(pred_mu, binary_sets)):
        if np.isnan(mu_i) or mu_i <= 0:
            results.append({"pred": mu_i, "lower": np.nan, "upper": np.nan})
            continue

        # Get count interval for claimants
        if claimant_quantiles is not None:
            count_lower, count_upper = find_prediction_set(mu_i, claimant_quantiles, ncs_type, max_count=50)
        else:
            # Fallback to parametric if no claimant data
            count_lower, count_upper = 1, max(1, int(np.ceil(mu_i * 2)))

        if binary_set == {0}:
            # Definitely no claim
            lower, upper = 0, 0
        elif binary_set == {1}:
            # Definitely claim - use count interval
            # Shift to exclude zero: [max(1, lower), upper]
            lower = max(1, count_lower)
            upper = max(1, count_upper)
        else:  # {0,1} or empty - ambiguous
            # Union of {0} and count interval
            # Result: [0, max(1, upper)]
            lower = 0
            upper = max(1, count_upper)

        results.append({"pred": mu_i, "lower": int(lower), "upper": int(upper)})

    return pd.DataFrame(results)



def two_stage_exact_cp(
    pred_mu: np.ndarray,
    calib_mu: np.ndarray,
    calib_y: np.ndarray,
    alpha: float = 0.1,
    ncs_type: str = "poisson_deviance"
) -> pd.DataFrame:
    """
    Two-stage CP with exact coverage control.

    Key innovation: Uses conformal p-values from both stages
    to construct exact prediction sets.
    """
    pred_mu = np.asarray(pred_mu, dtype=np.float64)
    calib_mu = np.asarray(calib_mu, dtype=np.float64)
    calib_y = np.asarray(calib_y, dtype=np.float64)

    # -----------------------------------------------------------------
    # Step 1: Compute binary p-values (for Y=0 vs Y>0)
    # -----------------------------------------------------------------
    calib_y_binary = (calib_y > 0).astype(int)
    calib_probs_binary = np.column_stack([
        np.exp(-calib_mu),           # P(Y=0)
        1 - np.exp(-calib_mu)        # P(Y>0)
    ])

    # Non-conformity for binary: 1 - P(true class)
    calib_scores_binary = np.array([
        1 - p[y] for p, y in zip(calib_probs_binary, calib_y_binary)
    ])

    # -----------------------------------------------------------------
    # Step 2: Compute count p-values for claimants (Y>0)
    # -----------------------------------------------------------------
    claimant_mask = calib_y > 0
    claimant_calib_mu = calib_mu[claimant_mask]
    claimant_calib_y = calib_y[claimant_mask]

    if len(claimant_calib_y) > 0:
        ncs_fun = get_ncs_function(ncs_type)
        claimant_ncs = ncs_fun(claimant_calib_mu, claimant_calib_y)
    else:
        claimant_ncs = np.array([])

    # -----------------------------------------------------------------
    # Step 3: For each test point, construct exact prediction set
    # -----------------------------------------------------------------
    results = []
    max_count = min(int(max(np.nanmax(calib_y), np.ceil(np.nanmax(pred_mu) * 3))) + 5, 50)
    n_obs = len(pred_mu)

    for i, mu in enumerate(pred_mu):
        # Print progress every 10%
        if i > 0 and i % (max(1, n_obs // 10)) == 0:
             print(f"Processed {i}/{n_obs} ({100 * i / n_obs:.0f}%) of observations.")

        if np.isnan(mu) or mu <= 0:
            results.append({"pred": mu, "lower": np.nan, "upper": np.nan})
            continue

        # Consider candidate y values
        candidates = np.arange(0, max_count + 1)
        prediction_set = []

        for y_candidate in candidates:
            # Compute binary p-value for this candidate
            if y_candidate == 0:
                # Test if y=0 is plausible
                candidate_prob_0 = np.exp(-mu)
                candidate_score_0 = 1 - candidate_prob_0

                # Mondrian p-value: only compare with calib_y=0
                calib_scores_0 = calib_scores_binary[calib_y_binary == 0]
                if len(calib_scores_0) > 0:
                    pval_binary = (np.sum(calib_scores_0 >= candidate_score_0) + 1) / (len(calib_scores_0) + 1)
                else:
                    pval_binary = 1.0

                # Keep if p-value > α
                if pval_binary > alpha:
                    prediction_set.append(y_candidate)

            else:  # y_candidate > 0
                # Test in two parts:
                # Part 1: Is y>0 plausible? (binary test)
                candidate_prob_1 = 1 - np.exp(-mu)
                candidate_score_1 = 1 - candidate_prob_1  # NCS = 1 - P(true class) for y>0

                # Mondrian p-value for class y>0
                calib_scores_1 = calib_scores_binary[calib_y_binary == 1]
                if len(calib_scores_1) > 0:
                    pval_binary = (np.sum(calib_scores_1 >= candidate_score_1) + 1) / (len(calib_scores_1) + 1)
                else:
                    pval_binary = 1.0

                # Part 2: Is this specific y value plausible? (count test)
                if len(claimant_ncs) > 0:
                    candidate_ncs = ncs_fun(np.array([mu]), np.array([y_candidate]))[0]
                    pval_count = (np.sum(claimant_ncs >= candidate_ncs) + 1) / (len(claimant_ncs) + 1)
                else:
                    pval_count = 1.0

                # Keep if BOTH tests pass (p-value > α for both)
                if pval_binary > alpha and pval_count > alpha:
                    prediction_set.append(y_candidate)

        # Final interval (contiguous if possible)
        if prediction_set:
            lower = int(min(prediction_set))
            upper = int(max(prediction_set))

            # Optional: Make contiguous if small gap
            if upper - lower <= 2 and len(prediction_set) / (upper - lower + 1) > 0.7:
                # Fill small gaps
                pass
            # Keep as set if disjoint
        else:
            # Fallback: point prediction
            lower = upper = int(np.round(mu))

        results.append({"pred": mu, "lower": lower, "upper": upper})

    return pd.DataFrame(results)

def two_stage_bonferroni_cp(
    pred_mu: np.ndarray,
    calib_mu: np.ndarray,
    calib_y: np.ndarray,
    alpha: float = 0.1,
    ncs_type: str = "poisson_deviance") -> pd.DataFrame:
    """
    Two-stage with Bonferroni correction for exact coverage.
    Uses α/2 for each stage to ensure overall α.
    """
    pred_mu = np.asarray(pred_mu, dtype=np.float64)
    calib_mu = np.asarray(calib_mu, dtype=np.float64)
    calib_y = np.asarray(calib_y, dtype=np.float64)

    # -----------------------------------------------------------------
    # Stage 1: Binary CP with α/2
    # -----------------------------------------------------------------
    calib_probs_binary = np.column_stack([
        np.exp(-calib_mu),
        1 - np.exp(-calib_mu)
    ])
    test_probs_binary = np.column_stack([
        np.exp(-pred_mu),
        1 - np.exp(-pred_mu)
    ])
    calib_y_binary = (calib_y > 0).astype(int)

    # Use more conservative: mondrian + α/2
    binary_sets = binary_classification_cp(
        test_probs_binary, calib_probs_binary, calib_y_binary,
        alpha=alpha/2, mondrian=True
    )

    # -----------------------------------------------------------------
    # Stage 2: Count CP for claimants with α/2
    # -----------------------------------------------------------------
    claimant_mask = calib_y > 0
    claimant_calib_mu = calib_mu[claimant_mask]
    claimant_calib_y = calib_y[claimant_mask]

    ncs_fun = get_ncs_function(ncs_type)
    if len(claimant_calib_y) > 0:
        claimant_ncs = ncs_fun(claimant_calib_mu, claimant_calib_y)
        # Compute quantile at 1-α/2 with proper clamping
        n_calib = len(claimant_ncs)
        q_level = min(np.ceil((n_calib + 1) * (1 - alpha/2)) / n_calib, 1.0)
        claimant_threshold = np.nanquantile(claimant_ncs, q_level)
        # Handle edge cases
        if np.isnan(claimant_threshold) or np.isinf(claimant_threshold):
            claimant_threshold = np.nanmax(claimant_ncs) if len(claimant_ncs) > 0 else 10.0
    else:
        claimant_threshold = 10.0  # Reasonable fallback instead of inf

    # -----------------------------------------------------------------
    # Combine with Bonferroni rule
    # -----------------------------------------------------------------
    results = []

    for i, (mu, binary_set) in enumerate(zip(pred_mu, binary_sets)):
        if np.isnan(mu) or mu <= 0:
            results.append({"pred": mu, "lower": np.nan, "upper": np.nan})
            continue

        # Find count interval using threshold
        count_set = []
        for y in range(0, 51):  # 0 to 50
            if y == 0:
                # y=0 included if binary_set contains 0
                if 0 in binary_set:
                    count_set.append(0)
            else:
                # y>0 included if:
                # 1. binary_set contains 1
                # 2. NCS score passes threshold
                if 1 in binary_set:
                    score = ncs_fun(np.array([mu]), np.array([y]))[0]
                    if score <= claimant_threshold:
                        count_set.append(y)

        if count_set:
            lower = min(count_set)
            upper = max(count_set)
        else:
            # Fallback: conservative interval [0, upper_bound]
            lower = 0
            upper = max(1, int(np.ceil(mu * 2)))

        results.append({"pred": mu, "lower": int(lower), "upper": int(upper)})

    return pd.DataFrame(results)

# CELL 9: Parametric and Bootstrap Baselines

def poisson_parametric_interval(mu: np.ndarray, alpha: float = 0.1) -> pd.DataFrame:
    """Exact Poisson prediction intervals."""
    mu = np.asarray(mu, dtype=np.float64)
    lower = stats.poisson.ppf(alpha / 2, mu).astype(int)
    upper = stats.poisson.ppf(1 - alpha / 2, mu).astype(int)
    lower = np.maximum(lower, 0)
    return pd.DataFrame({"pred": mu, "lower": lower, "upper": upper})


def negbinom_parametric_interval(mu: np.ndarray, calib_mu: np.ndarray,
                                  calib_y: np.ndarray, alpha: float = 0.1) -> pd.DataFrame:
    """Negative binomial intervals with estimated dispersion."""
    mu = np.asarray(mu, dtype=np.float64)

    # Estimate dispersion via method of moments
    mean_y = np.mean(calib_y)
    var_y = np.var(calib_y)

    if var_y > mean_y:
        theta = mean_y**2 / (var_y - mean_y)
    else:
        theta = 100  # Low overdispersion

    theta = max(theta, 0.1)  # Ensure positive

    p = theta / (theta + mu)
    lower = stats.nbinom.ppf(alpha / 2, theta, p).astype(int)
    upper = stats.nbinom.ppf(1 - alpha / 2, theta, p).astype(int)
    lower = np.maximum(lower, 0)

    return pd.DataFrame({"pred": mu, "lower": lower, "upper": upper})


def bootstrap_poisson_interval(pred_mu: np.ndarray, calib_mu: np.ndarray,
                                calib_y: np.ndarray, alpha: float = 0.1,
                                n_bootstraps: int = 500) -> pd.DataFrame:
    """Bootstrap intervals using Pearson residuals."""
    pred_mu = np.asarray(pred_mu, dtype=np.float64)
    calib_mu = np.asarray(calib_mu, dtype=np.float64)
    calib_y = np.asarray(calib_y, dtype=np.float64)

    # Compute Pearson residuals
    eps = 0.1
    pearson_resid = (calib_y - calib_mu) / np.sqrt(calib_mu + eps)

    results = []
    for mu in pred_mu:
        boot_resid = np.random.choice(pearson_resid, size=n_bootstraps, replace=True)
        boot_counts = mu + boot_resid * np.sqrt(mu + eps)
        boot_counts = np.maximum(boot_counts, 0)

        lower = int(max(0, np.floor(np.quantile(boot_counts, alpha / 2))))
        upper = int(np.ceil(np.quantile(boot_counts, 1 - alpha / 2)))

        results.append({"pred": mu, "lower": lower, "upper": upper})

    return pd.DataFrame(results)


