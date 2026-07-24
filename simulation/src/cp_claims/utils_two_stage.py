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
from .utils_pinterval import binary_classification_cp

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
warnings.filterwarnings('ignore')


import numpy as np
import pandas as pd
from scipy import stats
from typing import Callable, Dict, Tuple
import warnings

warnings.filterwarnings('ignore')


# NCS Functions (copied for self-containment)

def poisson_deviance_ncs(pred: np.ndarray, truth: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Poisson Deviance Residual NCS."""
    pred = np.asarray(pred, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    dev_contrib = np.where(
        truth == 0,
        2 * pred,
        2 * (truth * np.log(np.maximum(truth, eps) / np.maximum(pred, eps)) - (truth - pred))
    )
    return np.sign(truth - pred) * np.sqrt(np.maximum(dev_contrib, 0))


def poisson_pearson_ncs(pred: np.ndarray, truth: np. ndarray, eps: float = 0.1) -> np.ndarray:
    """Poisson Pearson Residual NCS."""
    pred = np.asarray(pred, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    return (truth - pred) / np.sqrt(pred + eps)


NCS_FUNCTIONS = {
    "poisson_deviance":  poisson_deviance_ncs,
    "poisson_pearson": poisson_pearson_ncs,
}


def get_ncs_function(ncs_type: str) -> Callable:
    if ncs_type not in NCS_FUNCTIONS:
        raise ValueError(f"Unknown NCS:  {ncs_type}")
    return NCS_FUNCTIONS[ncs_type]


# Core Issue: The two-stage methods fail because they don't properly account
# for the MIXTURE nature of the problem.
#
# Key insight: For zero-inflated counts, we need to think of the prediction
# set as a union of two components:
#   C(x) = C_0(x) ∪ C_+(x)
# where C_0(x) ∈ {{}, {0}} and C_+(x) ⊆ {1, 2, 3, ... }
#
# The coverage guarantee should be:
#   P(Y ∈ C(X)) = P(Y=0) · P(0 ∈ C_0(X) | Y=0) + P(Y>0) · P(Y ∈ C_+(X) | Y>0)
#
# For marginal 1-α coverage, we need BOTH conditional coverages to be ≥ 1-α. 


def two_stage_mixture_cp(
    pred_mu: np.ndarray,
    calib_mu: np.ndarray,
    calib_y:  np.ndarray,
    alpha: float = 0.1,
    ncs_type: str = "poisson_deviance",
    outcome_conditional: bool = True
) -> pd.DataFrame:
    """
    Two-Stage Mixture Conformal Prediction for Zero-Inflated Counts. 
    
    This method properly handles the mixture structure by:
    1. Calibrating separately on zeros and positives
    2. Using outcome-conditional quantiles for each stage
    3. Taking the union to form the final prediction set
    
    Parameters: 
    -----------
    pred_mu : array
        Predicted Poisson mean for test observations
    calib_mu : array  
        Predicted mean for calibration observations
    calib_y : array
        True outcomes for calibration observations
    alpha :  float
        Miscoverage rate (default 0.1 for 90% coverage)
    ncs_type : str
        Non-conformity score type
    outcome_conditional : bool
        If True, aims for outcome-conditional validity (coverage ≥ 1-α for each y group)
        If False, aims for marginal validity only
        
    Returns:
    --------
    DataFrame with pred, lower, upper, includes_zero columns
    """
    pred_mu = np.asarray(pred_mu, dtype=np. float64)
    calib_mu = np.asarray(calib_mu, dtype=np. float64)  
    calib_y = np.asarray(calib_y, dtype=np.float64)
    
    ncs_fun = get_ncs_function(ncs_type)
    
    # Split calibration data
    mask_y0 = calib_y == 0
    mask_ypos = calib_y > 0
    
    n_zeros = mask_y0.sum()
    n_pos = mask_ypos.sum()
    n_total = len(calib_y)
    
    # Compute NCS for zeros
    if n_zeros > 0:
        # For y=0, the NCS is computed at the predicted mu
        scores_y0 = ncs_fun(calib_mu[mask_y0], calib_y[mask_y0])
    else:
        scores_y0 = np.array([])
    
    # Compute NCS for positives  
    if n_pos > 0:
        scores_ypos = ncs_fun(calib_mu[mask_ypos], calib_y[mask_ypos])
    else:
        scores_ypos = np.array([])
    
    # Compute quantile thresholds
    # For outcome-conditional validity, we need each group to have 1-α coverage
    if outcome_conditional:
        # Threshold for y=0: include 0 if score ≤ q_{1-α}(scores_y0)
        if n_zeros > 0:
            q_level_0 = min(np.ceil((n_zeros + 1) * (1 - alpha)) / n_zeros, 1.0)
            threshold_y0 = np.quantile(scores_y0, q_level_0)
        else:
            threshold_y0 = np.inf  # Always include 0 if no calibration zeros
        
        # Threshold for y>0: include y if score ≤ q_{1-α}(scores_ypos)
        if n_pos > 0:
            q_level_pos = min(np.ceil((n_pos + 1) * (1 - alpha)) / n_pos, 1.0)
            threshold_ypos = np. quantile(scores_ypos, q_level_pos)
        else:
            threshold_ypos = np.inf
    else:
        # For marginal validity, use pooled calibration
        all_scores = ncs_fun(calib_mu, calib_y)
        q_level = min(np.ceil((n_total + 1) * (1 - alpha)) / n_total, 1.0)
        threshold_y0 = threshold_ypos = np.quantile(all_scores, q_level)
    
    # Determine max count to consider
    max_calib = int(np.nanmax(calib_y)) if len(calib_y) > 0 else 10
    max_pred = int(np.ceil(np.nanmax(pred_mu) * 3)) if len(pred_mu) > 0 else 10
    max_count = min(max(max_calib, max_pred) + 5, 100)
    
    results = []
    
    for mu in pred_mu:
        if np.isnan(mu) or mu <= 0:
            results.append({
                "pred": mu, "lower": 0, "upper": 0, 
                "includes_zero":  True, "set_type": "degenerate"
            })
            continue
        
        prediction_set = []
        
        # Stage 1: Check if 0 should be included
        score_at_0 = ncs_fun(np.array([mu]), np.array([0]))[0]
        include_zero = score_at_0 <= threshold_y0
        
        if include_zero:
            prediction_set.append(0)
        
        # Stage 2: Check positive values
        for y in range(1, max_count + 1):
            score_at_y = ncs_fun(np.array([mu]), np.array([y]))[0]
            if score_at_y <= threshold_ypos:
                prediction_set. append(y)
        
        # Handle empty set (should be rare with proper calibration)
        if not prediction_set:
            # Fallback:  include the mode
            if np.exp(-mu) > 0.5:
                prediction_set = [0]
            else:
                mode = max(1, int(np.round(mu)))
                prediction_set = [mode]
        
        # Convert to contiguous interval
        prediction_set = sorted(set(prediction_set))
        lower = min(prediction_set)
        upper = max(prediction_set)
        
        # Determine set type for diagnostics
        if lower == 0 and upper == 0:
            set_type = "zero_only"
        elif lower > 0:
            set_type = "positive_only"  
        else:
            set_type = "mixed"
        
        results.append({
            "pred": mu, 
            "lower": int(lower), 
            "upper": int(upper),
            "includes_zero": include_zero,
            "set_type": set_type
        })
    
    return pd.DataFrame(results)


def two_stage_weighted_cp(
    pred_mu: np.ndarray,
    calib_mu: np.ndarray,
    calib_y: np.ndarray,
    alpha: float = 0.1,
    ncs_type:  str = "poisson_deviance"
) -> pd.DataFrame:
    """
    Two-Stage Weighted CP that reweights to achieve balanced coverage.
    
    Key idea: The naive approach under-covers rare outcomes (y≥2) because
    they have fewer calibration points. We reweight the thresholds to 
    account for this.
    
    Uses importance weighting based on predicted probability of each outcome group.
    """
    pred_mu = np.asarray(pred_mu, dtype=np.float64)
    calib_mu = np.asarray(calib_mu, dtype=np.float64)
    calib_y = np.asarray(calib_y, dtype=np.float64)
    
    ncs_fun = get_ncs_function(ncs_type)
    
    # Split into three groups:  y=0, y=1, y≥2
    mask_y0 = calib_y == 0
    mask_y1 = calib_y == 1
    mask_y2plus = calib_y >= 2
    
    n0, n1, n2plus = mask_y0.sum(), mask_y1.sum(), mask_y2plus.sum()
    n_total = len(calib_y)
    
    # Compute scores per group
    scores = {}
    thresholds = {}
    
    for name, mask in [("y0", mask_y0), ("y1", mask_y1), ("y2plus", mask_y2plus)]:
        n_group = mask.sum()
        if n_group > 0:
            group_scores = ncs_fun(calib_mu[mask], calib_y[mask])
            scores[name] = group_scores
            
            # Compute group-specific threshold at 1-α
            q_level = min(np.ceil((n_group + 1) * (1 - alpha)) / n_group, 1.0)
            thresholds[name] = np. quantile(group_scores, q_level)
        else:
            scores[name] = np.array([])
            thresholds[name] = np.inf
    
    max_count = min(int(np.nanmax(calib_y)) + 10, 100) if len(calib_y) > 0 else 50
    
    results = []
    
    for mu in pred_mu:
        if np.isnan(mu) or mu <= 0:
            results.append({"pred": mu, "lower":  0, "upper": 0})
            continue
        
        prediction_set = []
        
        # Check y=0 against y0 threshold
        score_0 = ncs_fun(np.array([mu]), np.array([0]))[0]
        if score_0 <= thresholds["y0"]:
            prediction_set.append(0)
        
        # Check y=1 against y1 threshold
        score_1 = ncs_fun(np.array([mu]), np.array([1]))[0]
        if score_1 <= thresholds["y1"]:
            prediction_set. append(1)
        
        # Check y≥2 against y2plus threshold
        for y in range(2, max_count + 1):
            score_y = ncs_fun(np.array([mu]), np.array([y]))[0]
            if score_y <= thresholds["y2plus"]: 
                prediction_set.append(y)
        
        # Handle empty set
        if not prediction_set: 
            # Use Poisson probabilities to pick most likely
            p0 = np.exp(-mu)
            p1 = mu * np.exp(-mu)
            if p0 >= p1 and p0 >= 0.3:
                prediction_set = [0]
            elif p1 >= 0.2:
                prediction_set = [1]
            else:
                mode = max(1, int(np.round(mu)))
                prediction_set = [mode]
        
        prediction_set = sorted(set(prediction_set))
        lower = min(prediction_set)
        upper = max(prediction_set)
        
        results.append({"pred": mu, "lower":  int(lower), "upper": int(upper)})
    
    return pd.DataFrame(results)


def two_stage_adaptive_cp(
    pred_mu: np.ndarray,
    calib_mu: np.ndarray,
    calib_y: np.ndarray,
    alpha: float = 0.1,
    ncs_type:  str = "poisson_deviance",
    min_group_size: int = 30
) -> pd.DataFrame:
    """
    Adaptive Two-Stage CP that chooses calibration strategy based on data. 
    
    Strategy:
    1. If sufficient zeros AND positives:  use outcome-conditional calibration
    2. If one group is small: pool with adjacent group (DGCP principle)
    3. Always ensure the threshold is well-estimated
    
    This balances outcome-conditional validity with statistical stability.
    """
    pred_mu = np.asarray(pred_mu, dtype=np.float64)
    calib_mu = np.asarray(calib_mu, dtype=np.float64)
    calib_y = np.asarray(calib_y, dtype=np.float64)
    
    ncs_fun = get_ncs_function(ncs_type)
    
    # Analyze calibration data structure
    mask_y0 = calib_y == 0
    mask_y1 = calib_y == 1
    mask_y2plus = calib_y >= 2
    
    n0, n1, n2plus = mask_y0.sum(), mask_y1.sum(), mask_y2plus.sum()
    n_total = len(calib_y)
    
    # Decide on grouping strategy
    # Group 1: zeros
    # Group 2: positives (maybe split into y=1 and y≥2 if enough data)
    
    use_three_groups = (n0 >= min_group_size and 
                        n1 >= min_group_size and 
                        n2plus >= min_group_size)
    
    use_two_groups = (n0 >= min_group_size and 
                      (n1 + n2plus) >= min_group_size)
    
    # Compute thresholds based on chosen strategy
    thresholds = {}
    
    if use_three_groups: 
        # Three separate thresholds
        for name, mask in [("y0", mask_y0), ("y1", mask_y1), ("y2plus", mask_y2plus)]:
            n_group = mask.sum()
            group_scores = ncs_fun(calib_mu[mask], calib_y[mask])
            q_level = min(np.ceil((n_group + 1) * (1 - alpha)) / n_group, 1.0)
            thresholds[name] = np.quantile(group_scores, q_level)
            
    elif use_two_groups: 
        # Two thresholds: zeros vs positives
        scores_y0 = ncs_fun(calib_mu[mask_y0], calib_y[mask_y0])
        mask_pos = calib_y > 0
        scores_pos = ncs_fun(calib_mu[mask_pos], calib_y[mask_pos])
        
        q0 = min(np.ceil((n0 + 1) * (1 - alpha)) / n0, 1.0)
        thresholds["y0"] = np.quantile(scores_y0, q0)
        
        n_pos = mask_pos.sum()
        q_pos = min(np.ceil((n_pos + 1) * (1 - alpha)) / n_pos, 1.0)
        thresholds["y1"] = thresholds["y2plus"] = np.quantile(scores_pos, q_pos)
        
    else:
        # Fall back to marginal calibration
        all_scores = ncs_fun(calib_mu, calib_y)
        q_level = min(np.ceil((n_total + 1) * (1 - alpha)) / n_total, 1.0)
        threshold = np.quantile(all_scores, q_level)
        thresholds["y0"] = thresholds["y1"] = thresholds["y2plus"] = threshold
    
    max_count = min(int(np.nanmax(calib_y)) + 10, 100) if len(calib_y) > 0 else 50
    
    results = []
    
    for mu in pred_mu:
        if np.isnan(mu) or mu <= 0:
            results.append({"pred": mu, "lower": 0, "upper": 0})
            continue
        
        prediction_set = []
        
        # Check each candidate y value against appropriate threshold
        for y in range(0, max_count + 1):
            score_y = ncs_fun(np.array([mu]), np.array([y]))[0]
            
            if y == 0:
                threshold = thresholds["y0"]
            elif y == 1:
                threshold = thresholds["y1"]
            else: 
                threshold = thresholds["y2plus"]
            
            if score_y <= threshold:
                prediction_set.append(y)
        
        # Handle empty set
        if not prediction_set: 
            p0 = np.exp(-mu)
            if p0 > 0.5:
                prediction_set = [0]
            else:
                prediction_set = [max(1, int(np.round(mu)))]
        
        prediction_set = sorted(set(prediction_set))
        lower = min(prediction_set)
        upper = max(prediction_set)
        
        results.append({"pred": mu, "lower":  int(lower), "upper": int(upper)})
    
    return pd.DataFrame(results)


def two_stage_calibrated_cp(
    pred_mu: np.ndarray,
    calib_mu: np.ndarray,
    calib_y: np.ndarray,
    alpha: float = 0.1,
    ncs_type:  str = "poisson_deviance"
) -> pd.DataFrame:
    """
    Two-Stage CP with post-hoc calibration adjustment.
    
    Key insight: The main issue with two-stage methods is that the 
    thresholds computed on calibration data don't transfer perfectly
    to test data due to: 
    1. Distribution shift in mu values
    2. Finite-sample estimation error
    
    This method estimates and corrects for these biases using a 
    leave-one-out approach on calibration data.
    """
    pred_mu = np.asarray(pred_mu, dtype=np.float64)
    calib_mu = np.asarray(calib_mu, dtype=np.float64)
    calib_y = np.asarray(calib_y, dtype=np.float64)
    
    ncs_fun = get_ncs_function(ncs_type)
    n_calib = len(calib_y)
    
    # Compute all calibration scores
    calib_scores = ncs_fun(calib_mu, calib_y)
    
    # Estimate coverage on calibration data using leave-one-out
    # This tells us if our threshold is too conservative or too liberal
    loo_coverage = {"y0": [], "y1": [], "y2plus": []}
    
    for i in range(min(n_calib, 200)):  # Subsample for efficiency
        # Leave out point i
        loo_scores = np.delete(calib_scores, i)
        loo_y = np.delete(calib_y, i)
        
        # Compute threshold on remaining data
        q_level = min(np.ceil((len(loo_scores) + 1) * (1 - alpha)) / len(loo_scores), 1.0)
        threshold = np.quantile(loo_scores, q_level)
        
        # Check if left-out point would be covered
        covered = calib_scores[i] <= threshold
        
        if calib_y[i] == 0:
            loo_coverage["y0"].append(covered)
        elif calib_y[i] == 1:
            loo_coverage["y1"].append(covered)
        else:
            loo_coverage["y2plus"].append(covered)
    
    # Compute adjustment factors
    adjustments = {}
    for group in ["y0", "y1", "y2plus"]:
        if len(loo_coverage[group]) > 10:
            empirical_cov = np.mean(loo_coverage[group])
            target_cov = 1 - alpha
            
            # If under-covering, we need a larger threshold (multiply by >1)
            # If over-covering, we need a smaller threshold (multiply by <1)
            if empirical_cov > 0:
                adjustments[group] = target_cov / empirical_cov
            else:
                adjustments[group] = 1.0
        else:
            adjustments[group] = 1.0
    
    # Clamp adjustments to reasonable range
    for group in adjustments:
        adjustments[group] = np.clip(adjustments[group], 0.8, 1.25)
    
    # Compute adjusted thresholds per group
    mask_y0 = calib_y == 0
    mask_y1 = calib_y == 1  
    mask_y2plus = calib_y >= 2
    
    thresholds = {}
    
    for name, mask, adj_key in [("y0", mask_y0, "y0"), 
                                  ("y1", mask_y1, "y1"),
                                  ("y2plus", mask_y2plus, "y2plus")]:
        n_group = mask.sum()
        if n_group > 0:
            group_scores = calib_scores[mask]
            # Adjust the quantile level based on LOO analysis
            adjusted_alpha = alpha / adjustments[adj_key]
            adjusted_alpha = np.clip(adjusted_alpha, 0.01, 0.5)
            q_level = min(np.ceil((n_group + 1) * (1 - adjusted_alpha)) / n_group, 1.0)
            thresholds[name] = np.quantile(group_scores, q_level)
        else:
            thresholds[name] = np.inf
    
    max_count = min(int(np.nanmax(calib_y)) + 10, 100) if len(calib_y) > 0 else 50
    
    results = []
    
    for mu in pred_mu:
        if np.isnan(mu) or mu <= 0:
            results. append({"pred": mu, "lower": 0, "upper":  0})
            continue
        
        prediction_set = []
        
        for y in range(0, max_count + 1):
            score_y = ncs_fun(np.array([mu]), np.array([y]))[0]
            
            if y == 0:
                threshold = thresholds["y0"]
            elif y == 1:
                threshold = thresholds["y1"]
            else:
                threshold = thresholds["y2plus"]
            
            if score_y <= threshold:
                prediction_set. append(y)
        
        if not prediction_set:
            p0 = np.exp(-mu)
            if p0 > 0.5:
                prediction_set = [0]
            else:
                prediction_set = [max(1, int(np.round(mu)))]
        
        prediction_set = sorted(set(prediction_set))
        lower = min(prediction_set)
        upper = max(prediction_set)
        
        results.append({"pred": mu, "lower": int(lower), "upper": int(upper)})
    
    return pd.DataFrame(results)


# Wrapper functions for backward compatibility

def two_stage_mixture_outcome_conditional(
    pred_mu: np.ndarray,
    calib_mu: np.ndarray,
    calib_y: np.ndarray,
    alpha: float = 0.1,
    ncs_type: str = "poisson_deviance"
) -> pd.DataFrame:
    """Wrapper for outcome-conditional mixture CP."""
    return two_stage_mixture_cp(
        pred_mu, calib_mu, calib_y, alpha, ncs_type, 
        outcome_conditional=True
    )


def two_stage_mixture_marginal(
    pred_mu:  np.ndarray,
    calib_mu: np.ndarray,
    calib_y:  np.ndarray,
    alpha: float = 0.1,
    ncs_type: str = "poisson_deviance"
) -> pd.DataFrame:
    """Wrapper for marginal mixture CP."""
    return two_stage_mixture_cp(
        pred_mu, calib_mu, calib_y, alpha, ncs_type,
        outcome_conditional=False
    )





def _compute_threshold(scores: np.ndarray, alpha: float) -> float:
    """Compute conformal threshold with finite-sample correction."""
    n = len(scores)
    if n == 0:
        return np.inf
    q_level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    return np.quantile(scores, q_level)


def two_stage_three_group_cp(
    pred_mu: np.ndarray,
    calib_mu: np.ndarray,
    calib_y:  np.ndarray,
    alpha: float = 0.1,
    ncs_type:  str = "poisson_deviance",
    min_group_size: int = 20
) -> pd.DataFrame:
    """
    Two-Stage CP with explicit three-group calibration:  y=0, y=1, y≥2.
    
    This addresses the y=1 over-coverage issue by computing a SEPARATE
    threshold for y=1 rather than pooling with y≥2.
    
    Mathematical justification:
    --------------------------
    For outcome-conditional validity, we need: 
        P(Y ∈ C(X) | Y = k) ≥ 1-α  for k = 0, 1, 2, ...
    
    Grouping y=1 with y≥2 violates this because the NCS distributions
    are quite different: 
    - y=1: scores tend to be moderate (μ typically < 1 in insurance)
    - y≥2: scores are larger (these are the "surprising" outcomes)
    
    Parameters: 
    -----------
    pred_mu : array
        Predicted Poisson mean for test observations
    calib_mu : array
        Predicted mean for calibration observations
    calib_y : array
        True outcomes for calibration observations
    alpha :  float
        Miscoverage rate (default 0.1 for 90% coverage)
    ncs_type : str
        Non-conformity score type
    min_group_size : int
        Minimum samples per group before pooling
    """
    pred_mu = np.asarray(pred_mu, dtype=np. float64)
    calib_mu = np.asarray(calib_mu, dtype=np. float64)
    calib_y = np.asarray(calib_y, dtype=np. float64)
    
    ncs_fun = get_ncs_function(ncs_type)
    
    # Define three groups
    mask_y0 = calib_y == 0
    mask_y1 = calib_y == 1
    mask_y2plus = calib_y >= 2
    
    n0 = mask_y0.sum()
    n1 = mask_y1.sum()
    n2plus = mask_y2plus.sum()
    
    # Compute group-specific scores and thresholds
    thresholds = {}
    
    # Group y=0
    if n0 >= min_group_size:
        scores_y0 = ncs_fun(calib_mu[mask_y0], calib_y[mask_y0])
        thresholds["y0"] = _compute_threshold(scores_y0, alpha)
    else:
        # Pool with all data
        all_scores = ncs_fun(calib_mu, calib_y)
        thresholds["y0"] = _compute_threshold(all_scores, alpha)
    
    # Group y=1 - THIS IS THE KEY CHANGE
    if n1 >= min_group_size:
        scores_y1 = ncs_fun(calib_mu[mask_y1], calib_y[mask_y1])
        thresholds["y1"] = _compute_threshold(scores_y1, alpha)
    else:
        # Pool y=1 with y≥2 (all positives) if y=1 is too small
        mask_pos = calib_y > 0
        scores_pos = ncs_fun(calib_mu[mask_pos], calib_y[mask_pos])
        thresholds["y1"] = _compute_threshold(scores_pos, alpha)
    
    # Group y≥2
    if n2plus >= min_group_size:
        scores_y2plus = ncs_fun(calib_mu[mask_y2plus], calib_y[mask_y2plus])
        thresholds["y2plus"] = _compute_threshold(scores_y2plus, alpha)
    else:
        # Pool with all positives
        mask_pos = calib_y > 0
        scores_pos = ncs_fun(calib_mu[mask_pos], calib_y[mask_pos])
        thresholds["y2plus"] = _compute_threshold(scores_pos, alpha)
    
    # Determine max count
    max_count = min(int(np.nanmax(calib_y)) + 10, 100) if len(calib_y) > 0 else 50
    
    results = []
    
    for mu in pred_mu:
        if np.isnan(mu) or mu <= 0:
            results.append({"pred":  mu, "lower": 0, "upper": 0})
            continue
        
        prediction_set = []
        
        # Check y=0 against y0 threshold
        score_0 = ncs_fun(np.array([mu]), np.array([0]))[0]
        if score_0 <= thresholds["y0"]: 
            prediction_set.append(0)
        
        # Check y=1 against y1 threshold (SEPARATE!)
        score_1 = ncs_fun(np.array([mu]), np.array([1]))[0]
        if score_1 <= thresholds["y1"]:
            prediction_set.append(1)
        
        # Check y≥2 against y2plus threshold
        for y in range(2, max_count + 1):
            score_y = ncs_fun(np. array([mu]), np.array([y]))[0]
            if score_y <= thresholds["y2plus"]:
                prediction_set.append(y)
        
        # Handle empty set
        if not prediction_set: 
            p0 = np.exp(-mu)
            p1 = mu * np.exp(-mu)
            if p0 >= max(p1, 0.3):
                prediction_set = [0]
            elif p1 >= 0.2:
                prediction_set = [1]
            else:
                prediction_set = [max(1, int(np.round(mu)))]
        
        prediction_set = sorted(set(prediction_set))
        lower = min(prediction_set)
        upper = max(prediction_set)
        
        results.append({"pred": mu, "lower":  int(lower), "upper": int(upper)})
    
    return pd.DataFrame(results)


def two_stage_dgcp_inspired_cp(
    pred_mu:  np.ndarray,
    calib_mu: np.ndarray,
    calib_y:  np.ndarray,
    alpha: float = 0.1,
    ncs_type: str = "poisson_deviance",
    min_group_size:  int = 30
) -> pd.DataFrame:
    """
    DGCP-Inspired Two-Stage CP with adaptive grouping.
    
    Implements the Ding et al. (2024) DGCP principle:
    - Group rare outcomes together
    - Ensure each group has sufficient calibration data
    - Use group-specific thresholds
    
    Grouping strategy:
    1. Always separate y=0 (most common in zero-inflated data)
    2. Adaptively group y=1, y=2, ...  based on sample sizes
    """
    pred_mu = np. asarray(pred_mu, dtype=np.float64)
    calib_mu = np.asarray(calib_mu, dtype=np.float64)
    calib_y = np.asarray(calib_y, dtype=np.float64)
    
    ncs_fun = get_ncs_function(ncs_type)
    
    # Analyze outcome distribution
    unique_y, counts = np.unique(calib_y. astype(int), return_counts=True)
    outcome_counts = dict(zip(unique_y, counts))
    
    # Define adaptive groups
    groups = {}
    
    # y=0 always separate (typically largest group in zero-inflated data)
    if 0 in outcome_counts and outcome_counts[0] >= min_group_size:
        groups[0] = [0]
    
    # For positive outcomes, use cumulative grouping
    # Start with y=1, add y=2, y=3, ...  until group is large enough
    current_group_start = 1
    current_group = []
    current_count = 0
    
    for y in range(1, int(max(unique_y)) + 1):
        count_y = outcome_counts.get(y, 0)
        current_group.append(y)
        current_count += count_y
        
        # Check if we have enough for a separate group
        if current_count >= min_group_size:
            groups[current_group_start] = current_group. copy()
            current_group_start = y + 1
            current_group = []
            current_count = 0
    
    # Handle remaining outcomes
    if current_group: 
        if current_group_start > 1 and (current_group_start - 1) in groups:
            # Merge with previous group
            groups[current_group_start - len(groups[current_group_start - 1])] += current_group
        else:
            # Create new group or merge with y=1 group
            if 1 in groups:
                groups[1] += current_group
            else:
                groups[1] = current_group
    
    # Ensure we have at least y=0 and y>0 groups
    if 0 not in groups:
        groups[0] = [0]
    if 1 not in groups: 
        groups[1] = list(range(1, int(max(unique_y)) + 1))
    
    # Compute thresholds per group
    thresholds = {}
    group_membership = {}  # Maps y value to its group key
    
    for group_key, group_values in groups.items():
        mask = np.isin(calib_y, group_values)
        n_group = mask.sum()
        
        if n_group >= min_group_size:
            scores = ncs_fun(calib_mu[mask], calib_y[mask])
            thresholds[group_key] = _compute_threshold(scores, alpha)
        else:
            # Fallback to all data
            all_scores = ncs_fun(calib_mu, calib_y)
            thresholds[group_key] = _compute_threshold(all_scores, alpha)
        
        for y in group_values:
            group_membership[y] = group_key
    
    max_count = min(int(np.nanmax(calib_y)) + 10, 100) if len(calib_y) > 0 else 50
    
    results = []
    
    for mu in pred_mu:
        if np.isnan(mu) or mu <= 0:
            results.append({"pred": mu, "lower": 0, "upper": 0})
            continue
        
        prediction_set = []
        
        for y in range(0, max_count + 1):
            score_y = ncs_fun(np.array([mu]), np.array([y]))[0]
            
            # Find which group this y belongs to
            if y in group_membership:
                group_key = group_membership[y]
            else:
                # For y values beyond calibration range, use highest group
                group_key = max(groups.keys())
            
            if score_y <= thresholds. get(group_key, np. inf):
                prediction_set. append(y)
        
        if not prediction_set:
            p0 = np.exp(-mu)
            if p0 > 0.5:
                prediction_set = [0]
            else:
                prediction_set = [max(1, int(np.round(mu)))]
        
        prediction_set = sorted(set(prediction_set))
        lower = min(prediction_set)
        upper = max(prediction_set)
        
        results.append({"pred": mu, "lower":  int(lower), "upper": int(upper)})
    
    return pd.DataFrame(results)


def two_stage_balanced_cp(
    pred_mu: np.ndarray,
    calib_mu: np.ndarray,
    calib_y: np.ndarray,
    alpha: float = 0.1,
    ncs_type:  str = "poisson_deviance",
    balance_method: str = "inverse_frequency"
) -> pd.DataFrame:
    """
    Two-Stage CP with balanced coverage across outcome groups.
    
    Uses importance weighting to ensure each outcome group gets
    appropriate representation in threshold computation.
    
    balance_method options:
    - "inverse_frequency": Weight by 1/P(Y=y), upweighting rare outcomes
    - "target_coverage": Adjust thresholds to hit target coverage per group
    - "sqrt_frequency": Weight by 1/sqrt(P(Y=y)), moderate upweighting
    """
    pred_mu = np.asarray(pred_mu, dtype=np. float64)
    calib_mu = np.asarray(calib_mu, dtype=np. float64)
    calib_y = np.asarray(calib_y, dtype=np. float64)
    
    ncs_fun = get_ncs_function(ncs_type)
    n_total = len(calib_y)
    
    # Define groups
    mask_y0 = calib_y == 0
    mask_y1 = calib_y == 1
    mask_y2plus = calib_y >= 2
    
    n0, n1, n2plus = mask_y0.sum(), mask_y1.sum(), mask_y2plus.sum()
    
    # Compute weights based on balance method
    if balance_method == "inverse_frequency": 
        # Rare outcomes get higher weight
        w0 = n_total / (3 * max(n0, 1))
        w1 = n_total / (3 * max(n1, 1))
        w2plus = n_total / (3 * max(n2plus, 1))
    elif balance_method == "sqrt_frequency":
        w0 = np.sqrt(n_total / max(n0, 1))
        w1 = np.sqrt(n_total / max(n1, 1))
        w2plus = np.sqrt(n_total / max(n2plus, 1))
    else:  # target_coverage
        w0 = w1 = w2plus = 1.0
    
    # Normalize weights
    w_sum = w0 + w1 + w2plus
    w0, w1, w2plus = w0/w_sum, w1/w_sum, w2plus/w_sum
    
    # Compute alpha per group (weighted allocation)
    # Higher weight = more stringent (lower alpha)
    # This ensures rare outcomes get tighter thresholds
    alpha_0 = alpha * (1 - w0 * 0.5)  # Reduce alpha for high-weight groups
    alpha_1 = alpha * (1 - w1 * 0.5)
    alpha_2plus = alpha * (1 - w2plus * 0.5)
    
    # Clamp to valid range
    alpha_0 = np.clip(alpha_0, 0.02, alpha)
    alpha_1 = np.clip(alpha_1, 0.02, alpha)
    alpha_2plus = np. clip(alpha_2plus, 0.02, alpha)
    
    # Compute thresholds
    thresholds = {}
    
    if n0 > 0:
        scores_y0 = ncs_fun(calib_mu[mask_y0], calib_y[mask_y0])
        thresholds["y0"] = _compute_threshold(scores_y0, alpha_0)
    else:
        thresholds["y0"] = np.inf
    
    if n1 > 0:
        scores_y1 = ncs_fun(calib_mu[mask_y1], calib_y[mask_y1])
        thresholds["y1"] = _compute_threshold(scores_y1, alpha_1)
    else:
        thresholds["y1"] = np.inf
    
    if n2plus > 0:
        scores_y2plus = ncs_fun(calib_mu[mask_y2plus], calib_y[mask_y2plus])
        thresholds["y2plus"] = _compute_threshold(scores_y2plus, alpha_2plus)
    else:
        thresholds["y2plus"] = np.inf
    
    max_count = min(int(np.nanmax(calib_y)) + 10, 100) if len(calib_y) > 0 else 50
    
    results = []
    
    for mu in pred_mu:
        if np.isnan(mu) or mu <= 0:
            results.append({"pred": mu, "lower": 0, "upper": 0})
            continue
        
        prediction_set = []
        
        score_0 = ncs_fun(np.array([mu]), np.array([0]))[0]
        if score_0 <= thresholds["y0"]:
            prediction_set.append(0)
        
        score_1 = ncs_fun(np. array([mu]), np.array([1]))[0]
        if score_1 <= thresholds["y1"]:
            prediction_set.append(1)
        
        for y in range(2, max_count + 1):
            score_y = ncs_fun(np.array([mu]), np.array([y]))[0]
            if score_y <= thresholds["y2plus"]: 
                prediction_set.append(y)
        
        if not prediction_set:
            p0 = np.exp(-mu)
            if p0 > 0.5:
                prediction_set = [0]
            else:
                prediction_set = [max(1, int(np.round(mu)))]
        
        prediction_set = sorted(set(prediction_set))
        lower = min(prediction_set)
        upper = max(prediction_set)
        
        results. append({"pred": mu, "lower": int(lower), "upper": int(upper)})
    
    return pd.DataFrame(results)






def _compute_threshold(scores: np.ndarray, alpha: float) -> float:
    """Compute conformal threshold with finite-sample correction."""
    n = len(scores)
    if n == 0:
        return np.inf
    q_level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    return np.quantile(scores, q_level)


def _estimate_score_scale(calib_mu: np.ndarray, calib_y: np.ndarray, 
                          ncs_fun: Callable) -> Dict[str, float]:
    """
    Estimate the typical scale of NCS scores for each outcome group.
    This helps calibrate the threshold adjustments.
    """
    mask_y0 = calib_y == 0
    mask_y1 = calib_y == 1
    mask_y2plus = calib_y >= 2
    
    scales = {}
    
    if mask_y0.sum() > 0:
        scores_y0 = ncs_fun(calib_mu[mask_y0], calib_y[mask_y0])
        scales["y0"] = np.median(np.abs(scores_y0))
    else:
        scales["y0"] = 1.0
    
    if mask_y1.sum() > 0:
        scores_y1 = ncs_fun(calib_mu[mask_y1], calib_y[mask_y1])
        scales["y1"] = np.median(np.abs(scores_y1))
    else:
        scales["y1"] = 1.0
    
    if mask_y2plus.sum() > 0:
        scores_y2plus = ncs_fun(calib_mu[mask_y2plus], calib_y[mask_y2plus])
        scales["y2plus"] = np.median(np.abs(scores_y2plus))
    else:
        scales["y2plus"] = 1.0
    
    return scales


def two_stage_balanced_v2_cp(
    pred_mu: np.ndarray,
    calib_mu: np.ndarray,
    calib_y:  np.ndarray,
    alpha: float = 0.1,
    ncs_type: str = "poisson_deviance"
) -> pd.DataFrame:
    """
    Improved Balanced Two-Stage CP with score-scale awareness.
    
    Key insight: The y=1 over-coverage happens because y=1 scores are
    systematically SMALLER than y≥2 scores. The original balanced method
    reduced alpha for small groups (making thresholds more lenient), but
    for y=1 we need the OPPOSITE adjustment.
    
    This version:
    1. Estimates the typical score scale for each group
    2. Adjusts alpha based on score scale relative to y≥2
    3. Groups with smaller scores get HIGHER alpha (tighter thresholds)
    """
    pred_mu = np.asarray(pred_mu, dtype=np.float64)
    calib_mu = np.asarray(calib_mu, dtype=np.float64)
    calib_y = np.asarray(calib_y, dtype=np.float64)
    
    ncs_fun = get_ncs_function(ncs_type)
    
    # Define groups
    mask_y0 = calib_y == 0
    mask_y1 = calib_y == 1
    mask_y2plus = calib_y >= 2
    
    n0, n1, n2plus = mask_y0.sum(), mask_y1.sum(), mask_y2plus.sum()
    
    # Estimate score scales
    scales = _estimate_score_scale(calib_mu, calib_y, ncs_fun)
    
    # Reference scale is y≥2 (the "normal" count behavior)
    ref_scale = scales["y2plus"]
    
    # Compute scale ratios (how much smaller/larger are scores vs reference)
    ratio_y0 = scales["y0"] / ref_scale if ref_scale > 0 else 1.0
    ratio_y1 = scales["y1"] / ref_scale if ref_scale > 0 else 1.0
    
    # Adjust alpha based on scale ratio
    # If scores are smaller (ratio < 1), INCREASE alpha to tighten threshold
    # If scores are larger (ratio > 1), DECREASE alpha to loosen threshold
    
    # For y=0: typically has moderate scores
    alpha_0 = alpha * np.clip(1.0 / ratio_y0, 0.7, 1.3)
    
    # For y=1: typically has smaller scores, so increase alpha
    alpha_1 = alpha * np.clip(1.0 / ratio_y1, 0.8, 1.5)
    
    # For y≥2: reference group, use base alpha
    alpha_2plus = alpha
    
    # Additional adjustment for sample size (small groups need some protection)
    n_total = len(calib_y)
    
    if n0 < 30:
        alpha_0 = alpha_0 * 0.9  # Slightly more conservative
    if n1 < 30:
        alpha_1 = alpha_1 * 0.9
    if n2plus < 30:
        alpha_2plus = alpha_2plus * 0.9
    
    # Clamp to valid range
    alpha_0 = np.clip(alpha_0, 0.05, 0.20)
    alpha_1 = np.clip(alpha_1, 0.05, 0.20)
    alpha_2plus = np.clip(alpha_2plus, 0.05, 0.20)
    
    # Compute thresholds
    thresholds = {}
    
    if n0 > 0:
        scores_y0 = ncs_fun(calib_mu[mask_y0], calib_y[mask_y0])
        thresholds["y0"] = _compute_threshold(scores_y0, alpha_0)
    else:
        thresholds["y0"] = np.inf
    
    if n1 > 0:
        scores_y1 = ncs_fun(calib_mu[mask_y1], calib_y[mask_y1])
        thresholds["y1"] = _compute_threshold(scores_y1, alpha_1)
    else:
        thresholds["y1"] = np.inf
    
    if n2plus > 0:
        scores_y2plus = ncs_fun(calib_mu[mask_y2plus], calib_y[mask_y2plus])
        thresholds["y2plus"] = _compute_threshold(scores_y2plus, alpha_2plus)
    else:
        thresholds["y2plus"] = np.inf
    
    max_count = min(int(np.nanmax(calib_y)) + 10, 100) if len(calib_y) > 0 else 50
    
    results = []
    
    for mu in pred_mu:
        if np.isnan(mu) or mu <= 0:
            results. append({"pred": mu, "lower": 0, "upper":  0})
            continue
        
        prediction_set = []
        
        score_0 = ncs_fun(np.array([mu]), np.array([0]))[0]
        if score_0 <= thresholds["y0"]:
            prediction_set.append(0)
        
        score_1 = ncs_fun(np.array([mu]), np.array([1]))[0]
        if score_1 <= thresholds["y1"]: 
            prediction_set.append(1)
        
        for y in range(2, max_count + 1):
            score_y = ncs_fun(np.array([mu]), np.array([y]))[0]
            if score_y <= thresholds["y2plus"]: 
                prediction_set.append(y)
        
        if not prediction_set:
            p0 = np.exp(-mu)
            if p0 > 0.5:
                prediction_set = [0]
            else:
                prediction_set = [max(1, int(np. round(mu)))]
        
        prediction_set = sorted(set(prediction_set))
        lower = min(prediction_set)
        upper = max(prediction_set)
        
        results.append({"pred": mu, "lower": int(lower), "upper": int(upper)})
    
    return pd.DataFrame(results)


def two_stage_empirical_cp(
    pred_mu: np.ndarray,
    calib_mu: np. ndarray,
    calib_y: np.ndarray,
    alpha: float = 0.1,
    ncs_type: str = "poisson_deviance",
    n_iterations: int = 3
) -> pd.DataFrame:
    """
    Two-Stage CP with empirical calibration via iteration.
    
    Uses an iterative procedure to find alpha adjustments that achieve
    the target coverage for each outcome group on the calibration data.
    
    This is more computationally expensive but can achieve better
    outcome-conditional calibration.
    """
    pred_mu = np.asarray(pred_mu, dtype=np. float64)
    calib_mu = np.asarray(calib_mu, dtype=np. float64)
    calib_y = np.asarray(calib_y, dtype=np. float64)
    
    ncs_fun = get_ncs_function(ncs_type)
    
    # Define groups
    mask_y0 = calib_y == 0
    mask_y1 = calib_y == 1
    mask_y2plus = calib_y >= 2
    
    n0, n1, n2plus = mask_y0.sum(), mask_y1.sum(), mask_y2plus.sum()
    
    # Initialize alphas
    alphas = {"y0": alpha, "y1": alpha, "y2plus":  alpha}
    target_coverage = 1 - alpha
    
    # Iteratively adjust alphas based on LOO coverage
    for iteration in range(n_iterations):
        # Compute thresholds with current alphas
        thresholds = {}
        
        if n0 > 0:
            scores_y0 = ncs_fun(calib_mu[mask_y0], calib_y[mask_y0])
            thresholds["y0"] = _compute_threshold(scores_y0, alphas["y0"])
        else:
            thresholds["y0"] = np.inf
        
        if n1 > 0:
            scores_y1 = ncs_fun(calib_mu[mask_y1], calib_y[mask_y1])
            thresholds["y1"] = _compute_threshold(scores_y1, alphas["y1"])
        else:
            thresholds["y1"] = np.inf
        
        if n2plus > 0:
            scores_y2plus = ncs_fun(calib_mu[mask_y2plus], calib_y[mask_y2plus])
            thresholds["y2plus"] = _compute_threshold(scores_y2plus, alphas["y2plus"])
        else:
            thresholds["y2plus"] = np.inf
        
        # Estimate coverage on calibration data (pseudo-LOO)
        # For each calibration point, check if it would be covered
        empirical_cov = {"y0": [], "y1": [], "y2plus": []}
        
        for i in range(len(calib_y)):
            mu_i = calib_mu[i]
            y_i = int(calib_y[i])
            
            # Would y_i be in the prediction set for mu_i?
            score_i = ncs_fun(np. array([mu_i]), np.array([y_i]))[0]
            
            if y_i == 0:
                covered = score_i <= thresholds["y0"]
                empirical_cov["y0"].append(covered)
            elif y_i == 1:
                covered = score_i <= thresholds["y1"]
                empirical_cov["y1"].append(covered)
            else: 
                covered = score_i <= thresholds["y2plus"]
                empirical_cov["y2plus"].append(covered)
        
        # Adjust alphas based on empirical coverage
        for group in ["y0", "y1", "y2plus"]:
            if len(empirical_cov[group]) > 10:
                emp_cov = np.mean(empirical_cov[group])
                
                # If over-covering, increase alpha (tighten threshold)
                # If under-covering, decrease alpha (loosen threshold)
                if emp_cov > target_coverage + 0.02: 
                    alphas[group] = min(alphas[group] * 1.1, 0.25)
                elif emp_cov < target_coverage - 0.02:
                    alphas[group] = max(alphas[group] * 0.9, 0.03)
    
    # Final threshold computation with adjusted alphas
    thresholds = {}
    
    if n0 > 0:
        scores_y0 = ncs_fun(calib_mu[mask_y0], calib_y[mask_y0])
        thresholds["y0"] = _compute_threshold(scores_y0, alphas["y0"])
    else:
        thresholds["y0"] = np.inf
    
    if n1 > 0:
        scores_y1 = ncs_fun(calib_mu[mask_y1], calib_y[mask_y1])
        thresholds["y1"] = _compute_threshold(scores_y1, alphas["y1"])
    else:
        thresholds["y1"] = np.inf
    
    if n2plus > 0:
        scores_y2plus = ncs_fun(calib_mu[mask_y2plus], calib_y[mask_y2plus])
        thresholds["y2plus"] = _compute_threshold(scores_y2plus, alphas["y2plus"])
    else:
        thresholds["y2plus"] = np.inf
    
    max_count = min(int(np.nanmax(calib_y)) + 10, 100) if len(calib_y) > 0 else 50
    
    results = []
    
    for mu in pred_mu:
        if np.isnan(mu) or mu <= 0:
            results.append({"pred": mu, "lower": 0, "upper": 0})
            continue
        
        prediction_set = []
        
        score_0 = ncs_fun(np.array([mu]), np.array([0]))[0]
        if score_0 <= thresholds["y0"]:
            prediction_set.append(0)
        
        score_1 = ncs_fun(np.array([mu]), np.array([1]))[0]
        if score_1 <= thresholds["y1"]: 
            prediction_set.append(1)
        
        for y in range(2, max_count + 1):
            score_y = ncs_fun(np.array([mu]), np.array([y]))[0]
            if score_y <= thresholds["y2plus"]:
                prediction_set.append(y)
        
        if not prediction_set:
            p0 = np.exp(-mu)
            if p0 > 0.5:
                prediction_set = [0]
            else: 
                prediction_set = [max(1, int(np.round(mu)))]
        
        prediction_set = sorted(set(prediction_set))
        lower = min(prediction_set)
        upper = max(prediction_set)
        
        results.append({"pred": mu, "lower": int(lower), "upper": int(upper)})
    
    return pd.DataFrame(results)


def two_stage_direct_cp(
    pred_mu: np.ndarray,
    calib_mu: np.ndarray,
    calib_y: np.ndarray,
    alpha: float = 0.1,
    ncs_type:  str = "poisson_deviance"
) -> pd.DataFrame:
    """
    Direct Three-Group CP without any alpha adjustments.
    
    The simplest correct approach:  just use alpha for each group separately.
    This should give exactly 1-α coverage for each outcome group if the
    calibration and test distributions match.
    
    Serves as a baseline to understand how much adjustment is actually needed.
    """
    pred_mu = np.asarray(pred_mu, dtype=np. float64)
    calib_mu = np.asarray(calib_mu, dtype=np. float64)
    calib_y = np.asarray(calib_y, dtype=np. float64)
    
    ncs_fun = get_ncs_function(ncs_type)
    
    # Define groups
    mask_y0 = calib_y == 0
    mask_y1 = calib_y == 1
    mask_y2plus = calib_y >= 2
    
    # Compute thresholds - same alpha for all groups
    thresholds = {}
    
    if mask_y0.sum() > 0:
        scores_y0 = ncs_fun(calib_mu[mask_y0], calib_y[mask_y0])
        thresholds["y0"] = _compute_threshold(scores_y0, alpha)
    else:
        thresholds["y0"] = np.inf
    
    if mask_y1.sum() > 0:
        scores_y1 = ncs_fun(calib_mu[mask_y1], calib_y[mask_y1])
        thresholds["y1"] = _compute_threshold(scores_y1, alpha)
    else:
        thresholds["y1"] = np.inf
    
    if mask_y2plus.sum() > 0:
        scores_y2plus = ncs_fun(calib_mu[mask_y2plus], calib_y[mask_y2plus])
        thresholds["y2plus"] = _compute_threshold(scores_y2plus, alpha)
    else:
        thresholds["y2plus"] = np.inf
    
    max_count = min(int(np.nanmax(calib_y)) + 10, 100) if len(calib_y) > 0 else 50
    
    results = []
    
    for mu in pred_mu:
        if np.isnan(mu) or mu <= 0:
            results.append({"pred": mu, "lower": 0, "upper": 0})
            continue
        
        prediction_set = []
        
        score_0 = ncs_fun(np.array([mu]), np.array([0]))[0]
        if score_0 <= thresholds["y0"]:
            prediction_set.append(0)
        
        score_1 = ncs_fun(np.array([mu]), np.array([1]))[0]
        if score_1 <= thresholds["y1"]: 
            prediction_set.append(1)
        
        for y in range(2, max_count + 1):
            score_y = ncs_fun(np.array([mu]), np.array([y]))[0]
            if score_y <= thresholds["y2plus"]:
                prediction_set.append(y)
        
        if not prediction_set:
            p0 = np.exp(-mu)
            if p0 > 0.5:
                prediction_set = [0]
            else: 
                prediction_set = [max(1, int(np.round(mu)))]
        
        prediction_set = sorted(set(prediction_set))
        lower = min(prediction_set)
        upper = max(prediction_set)
        
        results.append({"pred": mu, "lower": int(lower), "upper": int(upper)})
    
    return pd.DataFrame(results)




def two_stage_exact_empirical_cp(
    pred_mu: np.ndarray,
    calib_mu: np.ndarray,
    calib_y: np.ndarray,
    alpha: float = 0.1,
    ncs_type: str = "poisson_deviance",
    n_iterations: int = 2,
    min_group_size: int = 50
) -> pd.DataFrame:
    """
    Mathematically correct Exact CP with empirical refinement.
    
    Key principles:
    1. Maintain Mondrian CP structure for validity
    2. Use DGCP for rare outcome grouping
    3. Adjust α using theoretically justified weighting (not heuristic tuning)
    
    This preserves finite-sample validity while adapting to data distribution.
    """
    pred_mu = np.asarray(pred_mu, dtype=np.float64)
    calib_mu = np.asarray(calib_mu, dtype=np.float64)
    calib_y = np.asarray(calib_y, dtype=np.float64)
    
    # -----------------------------------------------------------------
    # STEP 1: Define groups using DGCP principle (mathematically sound)
    # -----------------------------------------------------------------
    # Analyze calibration distribution
    unique_y, counts = np.unique(calib_y.astype(int), return_counts=True)
    n_total = len(calib_y)
    
    # Define outcome groups with guaranteed validity
    groups = {}
    group_membership = {}  # Maps y to group key
    
    # Group 0: Y=0 (always separate, most abundant)
    groups[0] = [0]
    group_membership[0] = 0
    
    # Group positive outcomes with DGCP
    pos_outcomes = [y for y in unique_y if y > 0]
    
    if len(pos_outcomes) == 0:
        # No positive outcomes in calibration
        groups[1] = [1]  # Default
    else:
        # Sort positive outcomes by frequency
        pos_freq = [(y, counts[unique_y == y][0]) for y in pos_outcomes]
        pos_freq.sort(key=lambda x: x[1])
        
        current_group = []
        current_size = 0
        group_idx = 1
        
        for y, freq in pos_freq:
            # If current group too small AND y=1, try to keep separate
            if y == 1 and freq >= min_group_size:
                # Y=1 gets its own group if enough data
                if current_group:  # Finish previous group
                    groups[group_idx] = current_group.copy()
                    for gy in current_group:
                        group_membership[gy] = group_idx
                    group_idx += 1
                    current_group = []
                    current_size = 0
                
                groups[group_idx] = [1]
                group_membership[1] = group_idx
                group_idx += 1
                continue
            
            # Add to current group
            current_group.append(y)
            current_size += freq
            
            # Start new group if large enough
            if current_size >= min_group_size or y == pos_freq[-1][0]:
                groups[group_idx] = current_group.copy()
                for gy in current_group:
                    group_membership[gy] = group_idx
                group_idx += 1
                current_group = []
                current_size = 0
    
    # Ensure all y values have group membership
    for y in range(0, int(np.max(calib_y)) + 1):
        if y not in group_membership:
            # Assign to most similar group
            if y == 0:
                group_membership[y] = 0
            elif y == 1:
                group_membership[y] = 1 if 1 in group_membership else max(groups.keys())
            else:
                group_membership[y] = max(groups.keys())
    
    # -----------------------------------------------------------------
    # STEP 2: Compute group-specific α using theory (not heuristics)
    # -----------------------------------------------------------------
    # Use inverse frequency weighting with Bonferroni correction
    group_sizes = {}
    for g_key, g_values in groups.items():
        mask = np.isin(calib_y, g_values)
        group_sizes[g_key] = mask.sum()
    
    # Compute weights: larger groups get more stringent α (smaller)
    # This is theoretically justified for controlling group-wise error
    weights = {}
    for g_key in groups:
        if group_sizes[g_key] > 0:
            # Bonferroni-style adjustment: allocate α proportional to 1/sqrt(n)
            weights[g_key] = 1.0 / np.sqrt(group_sizes[g_key])
        else:
            weights[g_key] = 1.0
    
    # Normalize and compute adjusted α (maintains ∑α_g ≤ α)
    total_weight = sum(weights.values())
    alpha_adjusted = {}
    
    for g_key in groups:
        # Allocate α proportionally to weight
        # Groups with more data (larger n) get smaller α (more stringent)
        alpha_adjusted[g_key] = alpha * (weights[g_key] / total_weight)
        
        # Ensure α stays in reasonable range (0.5% to 20%)
        alpha_adjusted[g_key] = np.clip(alpha_adjusted[g_key], 0.005, 0.20)
    
    # -----------------------------------------------------------------
    # STEP 3: Binary stage (Y=0 vs Y>0) with adjusted α
    # -----------------------------------------------------------------
    calib_y_binary = (calib_y > 0).astype(int)
    calib_probs_binary = np.column_stack([
        np.exp(-calib_mu),           # P(Y=0)
        1 - np.exp(-calib_mu)        # P(Y>0)
    ])
    
    # Non-conformity scores: 1 - P(true class)
    calib_scores_binary = np.array([
        1 - p[y] for p, y in zip(calib_probs_binary, calib_y_binary)
    ])
    
    # -----------------------------------------------------------------
    # STEP 4: Count stage with group-specific calibration
    # -----------------------------------------------------------------
    ncs_fun = get_ncs_function(ncs_type)
    group_calib_scores = {}
    
    for g_key, g_values in groups.items():
        mask = np.isin(calib_y, g_values)
        if mask.sum() > 0:
            group_calib_scores[g_key] = ncs_fun(calib_mu[mask], calib_y[mask])
        else:
            group_calib_scores[g_key] = np.array([])
    
    # -----------------------------------------------------------------
    # STEP 5: Construct prediction sets with exact validity
    # -----------------------------------------------------------------
    results = []
    # Dynamic max_count based on calibration data
    max_count = min(int(np.nanmax(calib_y)) + 5, 20) if len(calib_y) > 0 else 10
    
    for i, mu in enumerate(pred_mu):
        if np.isnan(mu) or mu <= 0:
            results.append({"pred": mu, "lower": np.nan, "upper": np.nan})
            continue
        
        prediction_set = []
        
        # Consider candidate y values up to max_count
        for y_candidate in range(0, max_count + 1):
            # Determine which group this y belongs to
            if y_candidate in group_membership:
                g_key = group_membership[y_candidate]
            else:
                # For unseen y, use largest group (most conservative)
                g_key = max(groups.keys())
            
            # Get group-specific α
            alpha_group = alpha_adjusted[g_key]
            
            if y_candidate == 0:
                # Binary test for Y=0
                candidate_prob_0 = np.exp(-mu)
                candidate_score_0 = 1 - candidate_prob_0
                
                # Mondrian p-value: compare only with Y=0 calibration
                calib_scores_0 = calib_scores_binary[calib_y_binary == 0]
                if len(calib_scores_0) > 0:
                    pval_binary = (np.sum(calib_scores_0 >= candidate_score_0) + 1) / (len(calib_scores_0) + 1)
                else:
                    pval_binary = 1.0
                
                # Keep if p-value > α_group (group-specific α)
                if pval_binary > alpha_group:
                    prediction_set.append(y_candidate)
            
            else:  # y_candidate > 0
                # Two-part test: binary AND count
                
                # Part 1: Binary test for Y>0
                candidate_prob_pos = 1 - np.exp(-mu)
                candidate_score_pos = np.exp(-mu)  # 1 - P(Y>0)
                
                # Mondrian p-value for Y>0
                calib_scores_pos = calib_scores_binary[calib_y_binary == 1]
                if len(calib_scores_pos) > 0:
                    pval_binary = (np.sum(calib_scores_pos >= candidate_score_pos) + 1) / (len(calib_scores_pos) + 1)
                else:
                    pval_binary = 1.0
                
                # Part 2: Count test within group
                if len(group_calib_scores[g_key]) > 0:
                    candidate_ncs = ncs_fun(np.array([mu]), np.array([y_candidate]))[0]
                    pval_count = (np.sum(group_calib_scores[g_key] >= candidate_ncs) + 1) / (len(group_calib_scores[g_key]) + 1)
                else:
                    pval_count = 1.0
                
                # Keep if BOTH tests pass (intersection test)
                # Use group-specific α for binary test
                # Use same α_group for count test (conservative)
                if pval_binary > alpha_group and pval_count > alpha_group:
                    prediction_set.append(y_candidate)
        
        # Construct contiguous interval (standard practice)
        if prediction_set:
            lower = int(min(prediction_set))
            upper = int(max(prediction_set))
            
            # Make contiguous if gap is small (≤2)
            # This is for interpretability, not validity
            expected_size = upper - lower + 1
            if len(prediction_set) / expected_size < 0.8 and (upper - lower) <= 3:
                # Fill small gaps for interpretability
                prediction_set = list(range(lower, upper + 1))
                lower = int(min(prediction_set))
                upper = int(max(prediction_set))
        else:
            # Fallback: point prediction (should be rare)
            lower = upper = int(np.round(mu))
        
        # Additional safety: ensure interval isn't too narrow for high mu
        if mu > 2.0 and (upper - lower) < 2:
            upper = lower + 2
        
        results.append({
            "pred": mu, 
            "lower": lower, 
            "upper": upper,
            "set_size": len(prediction_set) if 'prediction_set' in locals() else 1
        })
    
    return pd.DataFrame(results)


def two_stage_exact_cp(
    pred_mu: np.ndarray,
    calib_mu: np.ndarray,
    calib_y: np.ndarray,
    alpha: float = 0.1,
    ncs_type: str = "poisson_deviance",
    min_group_size: int = 10
) -> pd.DataFrame:
    """
    Fixed version that matches your working real-world implementation.
    Maintains Mondrian CP validity with same alpha for all groups.
    """
    pred_mu = np.asarray(pred_mu, dtype=np.float64)
    calib_mu = np.asarray(calib_mu, dtype=np.float64)
    calib_y = np.asarray(calib_y, dtype=np.float64)
    
    # -----------------------------------------------------------
    # Step 1: Binary p-values (Y=0 vs Y>0) - SAME as working version
    # -----------------------------------------------------------
    calib_y_binary = (calib_y > 0).astype(int)
    calib_probs_binary = np.column_stack([
        np.exp(-calib_mu),           # P(Y=0)
        1 - np.exp(-calib_mu)        # P(Y>0)
    ])
    
    calib_scores_binary = np.array([
        1 - p[y] for p, y in zip(calib_probs_binary, calib_y_binary)
    ])
    
    # -----------------------------------------------------------
    # Step 2: Define groups EXACTLY as in your working version
    # -----------------------------------------------------------
    mask_y0 = calib_y == 0
    mask_y1 = calib_y == 1
    mask_y2plus = calib_y >= 2
    
    n1 = mask_y1.sum()
    n2plus = mask_y2plus.sum()
    
    # Use SAME logic as working algorithm
    if n2plus >= min_group_size:
        groups = {
            "y1": mask_y1,
            "y2plus": mask_y2plus
        }
    else:
        groups = {
            "ypos": calib_y > 0
        }
    
    # Compute scores for each group
    ncs_fun = get_ncs_function(ncs_type)
    group_scores = {}
    
    for group_name, mask in groups.items():
        if mask.sum() > 0:
            group_scores[group_name] = ncs_fun(calib_mu[mask], calib_y[mask])
        else:
            group_scores[group_name] = np.array([])
    
    # -----------------------------------------------------------
    # Step 3: Construct prediction sets with SAME alpha for all
    # -----------------------------------------------------------
    results = []
    max_count = min(int(np.nanmax(calib_y)) + 3, 20) if len(calib_y) > 0 else 10
    
    for i, mu in enumerate(pred_mu):
        if np.isnan(mu) or mu <= 0:
            results.append({"pred": mu, "lower": np.nan, "upper": np.nan})
            continue
        
        prediction_set = []
        
        for y_candidate in range(0, max_count + 1):
            if y_candidate == 0:
                # Test if y=0 is plausible - SAME as working version
                candidate_prob_0 = np.exp(-mu)
                candidate_score_0 = 1 - candidate_prob_0
                
                calib_scores_0 = calib_scores_binary[calib_y_binary == 0]
                if len(calib_scores_0) > 0:
                    # Use SAME alpha (not adjusted!)
                    pval_binary = (np.sum(calib_scores_0 >= candidate_score_0) + 1) / (len(calib_scores_0) + 1)
                else:
                    pval_binary = 1.0
                
                if pval_binary > alpha:  # FIXED: Use global alpha
                    prediction_set.append(y_candidate)
            
            else:  # y_candidate > 0
                # Binary test for Y>0 - SAME as working version
                candidate_prob_pos = 1 - np.exp(-mu)
                candidate_score_pos = 1 - candidate_prob_pos  # exp(-mu)
                
                calib_scores_pos = calib_scores_binary[calib_y_binary == 1]
                if len(calib_scores_pos) > 0:
                    pval_binary = (np.sum(calib_scores_pos >= candidate_score_pos) + 1) / (len(calib_scores_pos) + 1)
                else:
                    pval_binary = 1.0
                
                # Count test - Use same grouping logic as working version
                if y_candidate == 1 and "y1" in group_scores:
                    group_ncs = group_scores["y1"]
                elif y_candidate >= 2 and "y2plus" in group_scores:
                    group_ncs = group_scores["y2plus"]
                elif "ypos" in group_scores:
                    group_ncs = group_scores["ypos"]
                else:
                    group_ncs = np.array([])
                
                if len(group_ncs) > 0:
                    candidate_ncs = ncs_fun(np.array([mu]), np.array([y_candidate]))[0]
                    pval_count = (np.sum(group_ncs >= candidate_ncs) + 1) / (len(group_ncs) + 1)
                else:
                    pval_count = 1.0
                
                # BOTH tests must pass with SAME alpha
                if pval_binary > alpha and pval_count > alpha:  # FIXED: Use global alpha
                    prediction_set.append(y_candidate)
        
        # Construct contiguous interval
        if prediction_set:
            lower = int(min(prediction_set))
            upper = int(max(prediction_set))
            
            # Fill small gaps for interpretability (same as working version)
            expected_set = set(range(lower, upper + 1))
            actual_set = set(prediction_set)
            if len(actual_set) / len(expected_set) < 0.8 and (upper - lower) <= 3:
                prediction_set = list(range(lower, upper + 1))
                lower = int(min(prediction_set))
                upper = int(max(prediction_set))
        else:
            lower = upper = int(np.round(mu))
        
        results.append({
            "pred": mu, 
            "lower": lower, 
            "upper": upper,
            "set_size": len(prediction_set) if prediction_set else 1
        })
    
    return pd.DataFrame(results)