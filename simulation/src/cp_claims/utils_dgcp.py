import numpy as np
import pandas as pd
from typing import Callable, Dict, List, Tuple, Set
import warnings

warnings.filterwarnings('ignore')


# Non-Conformity Score Functions

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


def poisson_pearson_ncs(pred: np.ndarray, truth: np.ndarray, eps: float = 0.1) -> np.ndarray:
    """Poisson Pearson Residual NCS."""
    pred = np.asarray(pred, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    return (truth - pred) / np.sqrt(pred + eps)


NCS_FUNCTIONS = {
    "poisson_deviance": poisson_deviance_ncs,
    "poisson_pearson": poisson_pearson_ncs,
}


def get_ncs_function(ncs_type: str) -> Callable:
    if ncs_type not in NCS_FUNCTIONS:
        raise ValueError(f"Unknown NCS: {ncs_type}")
    return NCS_FUNCTIONS[ncs_type]


def _compute_pvalue(test_score: float, calib_scores: np.ndarray) -> float:
    """Compute conformal p-value with finite-sample correction."""
    if len(calib_scores) == 0:
        return 1.0
    return (np.sum(calib_scores >= test_score) + 1) / (len(calib_scores) + 1)


# Helper Functions

def get_contiguous_runs(prediction_set: List[int]) -> List[Tuple[int, int]]:
    """Convert a prediction set to a list of contiguous intervals."""
    if not prediction_set:
        return []
    
    prediction_set = sorted(set(prediction_set))
    
    intervals = []
    start = prediction_set[0]
    end = prediction_set[0]
    
    for i in range(1, len(prediction_set)):
        if prediction_set[i] == end + 1:
            end = prediction_set[i]
        else:
            intervals.append((start, end))
            start = prediction_set[i]
            end = prediction_set[i]
    
    intervals.append((start, end))
    return intervals


def _compute_max_count(calib_y: np.ndarray, pred_mu: np.ndarray = None, 
                       use_observed_only: bool = True) -> int:
    """
    Compute maximum count to test based on calibration data.
    
    Parameters:
    -----------
    calib_y : array
        Calibration outcomes
    pred_mu : array, optional
        Predicted means (used if use_observed_only=False)
    use_observed_only : bool
        If True, only test counts observed in calibration data.
        If False, extend slightly beyond observed max.
        
    Returns:
    --------
    Maximum count value to test
    """
    if len(calib_y) == 0:
        return 10
    
    max_calib_y = int(np.nanmax(calib_y))
    
    if use_observed_only:
        # Only test values actually observed in calibration
        # This prevents 100% coverage for unseen high counts
        return max_calib_y
    else:
        # Small extension for edge cases
        if pred_mu is not None and len(pred_mu) > 0:
            max_pred = int(np.ceil(np.nanmax(pred_mu) * 2))
        else:
            max_pred = max_calib_y
        return max(max_calib_y, max_pred) + 1


def _build_dgcp_count_groups(
    calib_y: np.ndarray,
    min_group_size: int = 10,
    max_count: int = None
) -> Dict[int, List[int]]:
    """
    Build DGCP-style groups for count data.
    
    For counts, "semantically related" = neighboring count values.
    We expand symmetrically (y-1, y+1, y-2, y+2, ...) until we have >= m samples.
    """
    if max_count is None:
        max_count = _compute_max_count(calib_y, use_observed_only=True)
    
    unique_y, counts = np.unique(calib_y[calib_y >= 0].astype(int), return_counts=True)
    count_freq = dict(zip(unique_y, counts))
    
    groups = {}
    
    for y in range(0, max_count + 1):
        n_y = count_freq.get(y, 0)
        
        if n_y >= min_group_size:
            groups[y] = [y]
        else:
            group = [y]
            total_samples = n_y
            radius = 1
            
            while total_samples < min_group_size and radius <= max_count:
                for neighbor in [y - radius, y + radius]:
                    if 0 <= neighbor <= max_count and neighbor not in group:
                        group.append(neighbor)
                        total_samples += count_freq.get(neighbor, 0)
                radius += 1
            
            groups[y] = sorted(group)
    
    return groups


def _build_pos_count_groups(
    calib_y_pos: np.ndarray,
    min_group_size: int = 10,
    max_count: int = None
) -> Dict[int, List[int]]:
    """
    Build DGCP-style groups for positive counts only (Y >= 1).
    Used in the Hybrid method where Y=0 is handled separately.
    """
    if max_count is None:
        max_count = _compute_max_count(calib_y_pos, use_observed_only=True)
    
    calib_y_int = calib_y_pos[calib_y_pos >= 1].astype(int)
    unique_y, counts = np.unique(calib_y_int, return_counts=True)
    count_freq = dict(zip(unique_y, counts))
    
    groups = {}
    
    for y in range(1, max_count + 1):
        n_y = count_freq.get(y, 0)
        
        if n_y >= min_group_size:
            groups[y] = [y]
        else:
            group = [y]
            total_samples = n_y
            radius = 1
            
            while total_samples < min_group_size and radius <= max_count:
                for neighbor in [y - radius, y + radius]:
                    if 1 <= neighbor <= max_count and neighbor not in group:
                        group.append(neighbor)
                        total_samples += count_freq.get(neighbor, 0)
                radius += 1
            
            groups[y] = sorted(group)
    
    return groups


# DGCP FULL - Fixed with Hypothetical Scores and Proper max_count

def two_stage_dgcp_full_cp(
    pred_mu: np.ndarray,
    calib_mu: np.ndarray,
    calib_y: np.ndarray,
    alpha: float = 0.1,
    ncs_type: str = "poisson_deviance",
    min_group_size: int = 10,
    max_count: int = None,
    verbose: bool = False
) -> pd.DataFrame:
    """
    DGCP Full: Tests each candidate y using HYPOTHETICAL scores.
    
    Key Fix: Calibration scores are computed AS IF all calibration points
    in the group had the candidate y value being tested.
    
    Parameters:
    -----------
    pred_mu : array
        Predicted Poisson mean for test observations
    calib_mu : array
        Predicted mean for calibration observations
    calib_y : array
        True outcomes for calibration observations
    alpha : float
        Miscoverage rate (default 0.1 for 90% coverage)
    ncs_type : str
        Non-conformity score type
    min_group_size : int
        Minimum calibration samples per count (m in DGCP)
    max_count : int, optional
        Maximum count to test. If None, uses max observed in calib_y.
    verbose : bool
        Print diagnostic information
        
    Returns:
    --------
    DataFrame with pred, lower, upper columns
    """
    pred_mu = np.asarray(pred_mu, dtype=np.float64)
    calib_mu = np.asarray(calib_mu, dtype=np.float64)
    calib_y = np.asarray(calib_y, dtype=np.float64)
    
    ncs_fun = get_ncs_function(ncs_type)
    
    # FIX: Use observed max count only, no arbitrary extension to 50
    if max_count is None:
        max_count = _compute_max_count(calib_y, pred_mu, use_observed_only=True)
    
    if verbose:
        print(f"DGCP Full - max_count set to {max_count} (max observed in calib)")
    
    # Build DGCP groups
    groups = _build_dgcp_count_groups(calib_y, min_group_size, max_count)
    
    if verbose:
        print(f"DGCP Full - Groups (m={min_group_size}):")
        unique_y, counts = np.unique(calib_y[calib_y >= 0].astype(int), return_counts=True)
        count_freq = dict(zip(unique_y, counts))
        shown = set()
        for y in range(min(15, max_count + 1)):
            g = tuple(groups.get(y, [y]))
            if g not in shown:
                n_samples = sum(count_freq.get(yy, 0) for yy in g)
                print(f"  y={y} -> group {list(g)}: {n_samples} samples")
                shown.add(g)
    
    # Pre-compute group masks and μ values
    group_calib_mu = {}
    for y in range(0, max_count + 1):
        group = tuple(groups.get(y, [y]))
        if group not in group_calib_mu:
            mask = np.isin(calib_y.astype(int), list(group))
            if mask.sum() > 0:
                group_calib_mu[group] = calib_mu[mask]
            else:
                group_calib_mu[group] = np.array([])
    
    # Pre-compute HYPOTHETICAL scores for each candidate y
    hypothetical_scores = {}
    
    for y in range(0, max_count + 1):
        group = tuple(groups.get(y, [y]))
        group_mu = group_calib_mu.get(group, np.array([]))
        
        if len(group_mu) > 0:
            hypothetical_scores[y] = ncs_fun(group_mu, np.full(len(group_mu), y))
        else:
            hypothetical_scores[y] = np.array([])
    
    # Construct prediction sets
    results = []
    
    for mu in pred_mu:
        if np.isnan(mu) or mu <= 0:
            results.append({
                "pred": mu,
                "lower": np.nan,
                "upper": np.nan,
                "set_size": 0
            })
            continue
        
        prediction_set = []
        
        # Test each candidate y - ONLY up to max_count
        for y in range(0, max_count + 1):
            calib_scores = hypothetical_scores.get(y, np.array([]))
            
            if len(calib_scores) == 0:
                # No calibration data for this y - skip, don't include
                # This is the key change: we don't conservatively include
                # values we have no calibration data for
                continue
            
            test_score = ncs_fun(np.array([mu]), np.array([y]))[0]
            pval = _compute_pvalue(test_score, calib_scores)
            
            if pval > alpha:
                prediction_set.append(y)
        
        # Finalize
        if not prediction_set:
            if np.exp(-mu) > 0.5:
                prediction_set = [0]
            else:
                prediction_set = [max(1, min(int(np.round(mu)), max_count))]
        
        prediction_set = sorted(set(prediction_set))
        lower = min(prediction_set)
        upper = max(prediction_set)
        
        results.append({
            "pred": mu,
            "lower": int(lower),
            "upper": int(upper),
            "set_size": upper - lower + 1,
            "includes_zero": 0 in prediction_set
        })
    
    return pd.DataFrame(results)


# DGCP EXACT - Fixed with Proper max_count

def two_stage_dgcp_exact_cp(
    pred_mu: np.ndarray,
    calib_mu: np.ndarray,
    calib_y: np.ndarray,
    alpha: float = 0.1,
    ncs_type: str = "poisson_deviance",
    min_group_size: int = 10,
    max_count: int = None,
    return_sets: bool = False,
    verbose: bool = False
) -> pd.DataFrame:
    """
    DGCP Exact: Tests each candidate y using HYPOTHETICAL scores.
    Allows disjoint prediction sets (doesn't force contiguous intervals).
    
    Parameters:
    -----------
    pred_mu : array
        Predicted Poisson mean for test observations
    calib_mu : array
        Predicted mean for calibration observations
    calib_y : array
        True outcomes for calibration observations
    alpha : float
        Miscoverage rate (default 0.1 for 90% coverage)
    ncs_type : str
        Non-conformity score type
    min_group_size : int
        Minimum calibration samples per count (m in DGCP)
    max_count : int, optional
        Maximum count to test. If None, uses max observed in calib_y.
    return_sets : bool
        If True, include full prediction set in output
    verbose : bool
        Print diagnostic information
        
    Returns:
    --------
    DataFrame with pred, lower, upper, set_size, n_intervals columns
    """
    pred_mu = np.asarray(pred_mu, dtype=np.float64)
    calib_mu = np.asarray(calib_mu, dtype=np.float64)
    calib_y = np.asarray(calib_y, dtype=np.float64)
    
    ncs_fun = get_ncs_function(ncs_type)
    
    # FIX: Use observed max count only
    if max_count is None:
        max_count = _compute_max_count(calib_y, pred_mu, use_observed_only=True)
    
    if verbose:
        print(f"DGCP Exact - max_count set to {max_count} (max observed in calib)")
    
    # Build DGCP groups
    groups = _build_dgcp_count_groups(calib_y, min_group_size, max_count)
    
    if verbose:
        print(f"DGCP Exact - Groups (m={min_group_size}):")
        unique_y, counts = np.unique(calib_y[calib_y >= 0].astype(int), return_counts=True)
        count_freq = dict(zip(unique_y, counts))
        shown = set()
        for y in range(min(15, max_count + 1)):
            g = tuple(groups.get(y, [y]))
            if g not in shown:
                n_samples = sum(count_freq.get(yy, 0) for yy in g)
                print(f"  y={y} -> group {list(g)}: {n_samples} samples")
                shown.add(g)
    
    # Pre-compute group masks and μ values
    group_calib_mu = {}
    for y in range(0, max_count + 1):
        group = tuple(groups.get(y, [y]))
        if group not in group_calib_mu:
            mask = np.isin(calib_y.astype(int), list(group))
            if mask.sum() > 0:
                group_calib_mu[group] = calib_mu[mask]
            else:
                group_calib_mu[group] = np.array([])
    
    # Pre-compute HYPOTHETICAL scores for each candidate y
    hypothetical_scores = {}
    
    for y in range(0, max_count + 1):
        group = tuple(groups.get(y, [y]))
        group_mu = group_calib_mu.get(group, np.array([]))
        
        if len(group_mu) > 0:
            hypothetical_scores[y] = ncs_fun(group_mu, np.full(len(group_mu), y))
        else:
            hypothetical_scores[y] = np.array([])
    
    # Construct prediction sets
    results = []
    
    for mu in pred_mu:
        if np.isnan(mu) or mu <= 0:
            results.append({
                "pred": mu,
                "lower": np.nan,
                "upper": np.nan,
                "set_size": 0,
                "n_intervals": 0,
                "includes_zero": False
            })
            continue
        
        prediction_set = []
        
        # Test each candidate y - ONLY up to max_count
        for y in range(0, max_count + 1):
            calib_scores = hypothetical_scores.get(y, np.array([]))
            
            if len(calib_scores) == 0:
                # No calibration data - skip
                continue
            
            test_score = ncs_fun(np.array([mu]), np.array([y]))[0]
            pval = _compute_pvalue(test_score, calib_scores)
            
            if pval > alpha:
                prediction_set.append(y)
        
        # Finalize
        if not prediction_set:
            if np.exp(-mu) > 0.5:
                prediction_set = [0]
            else:
                prediction_set = [max(1, min(int(np.round(mu)), max_count))]
        
        prediction_set = sorted(set(prediction_set))
        intervals = get_contiguous_runs(prediction_set)
        
        result = {
            "pred": mu,
            "lower": min(prediction_set),
            "upper": max(prediction_set),
            "set_size": len(prediction_set),
            "n_intervals": len(intervals),
            "includes_zero": 0 in prediction_set
        }
        
        if return_sets:
            result["prediction_set"] = prediction_set
            result["intervals"] = intervals
        
        results.append(result)
    
    return pd.DataFrame(results)


# DGCP HYBRID - Fixed with Proper max_count

def two_stage_dgcp_hybrid_cp(
    pred_mu: np.ndarray,
    calib_mu: np.ndarray,
    calib_y: np.ndarray,
    alpha: float = 0.1,
    ncs_type: str = "poisson_deviance",
    min_group_size: int = 10,
    max_count: int = None,
    verbose: bool = False
) -> pd.DataFrame:
    """
    DGCP Hybrid: Binary NCS for Y=0, Hypothetical Poisson NCS for Y>=1.
    
    Stage 1 (Binary): Uses probability-based NCS for Y=0
        - Score = 1 - P(Y=0|μ) = 1 - exp(-μ)
        - Calibrated on Y=0 observations only (Mondrian/LCMICP style)
    
    Stage 2 (Count): Uses HYPOTHETICAL Poisson NCS for Y>=1
        - For candidate y, scores = NCS(μ_calib, y) for all points in group
        - Calibrated on Y>0 observations with DGCP neighbor grouping
    
    Parameters:
    -----------
    pred_mu : array
        Predicted Poisson mean for test observations
    calib_mu : array
        Predicted mean for calibration observations
    calib_y : array
        True outcomes for calibration observations
    alpha : float
        Miscoverage rate (default 0.1 for 90% coverage)
    ncs_type : str
        Non-conformity score type for count stage
    min_group_size : int
        Minimum calibration samples per count (m in DGCP)
    max_count : int, optional
        Maximum count to test. If None, uses max observed in calib_y.
    verbose : bool
        Print diagnostic information
        
    Returns:
    --------
    DataFrame with pred, lower, upper columns
    """
    pred_mu = np.asarray(pred_mu, dtype=np.float64)
    calib_mu = np.asarray(calib_mu, dtype=np.float64)
    calib_y = np.asarray(calib_y, dtype=np.float64)
    
    ncs_fun = get_ncs_function(ncs_type)
    
    # FIX: Use observed max count only
    if max_count is None:
        max_count = _compute_max_count(calib_y, pred_mu, use_observed_only=True)
    
    if verbose:
        print(f"DGCP Hybrid - max_count set to {max_count} (max observed in calib)")
    
    # =========================================================================
    # Stage 1: Binary NCS for Y=0 (LCMICP approach)
    # =========================================================================
    mask_y0 = calib_y == 0
    mask_ypos = calib_y > 0
    
    calib_mu_y0 = calib_mu[mask_y0]
    calib_mu_pos = calib_mu[mask_ypos]
    calib_y_pos = calib_y[mask_ypos].astype(int)
    
    # Binary NCS: score = 1 - P(Y=0|μ) = 1 - exp(-μ) = P(Y>0|μ)
    if len(calib_mu_y0) > 0:
        calib_binary_scores_y0 = 1.0 - np.exp(-calib_mu_y0)
    else:
        calib_binary_scores_y0 = np.array([])
    
    n_zeros = mask_y0.sum()
    n_pos = mask_ypos.sum()
    
    if verbose:
        print(f"DGCP Hybrid - Stage 1 (Binary):")
        print(f"  Calibration: {n_zeros} zeros, {n_pos} positives")
        if len(calib_binary_scores_y0) > 0:
            print(f"  Y=0 binary scores: mean={calib_binary_scores_y0.mean():.3f}, "
                  f"std={calib_binary_scores_y0.std():.3f}, "
                  f"q90={np.percentile(calib_binary_scores_y0, 90):.3f}")
    
    # =========================================================================
    # Stage 2: DGCP with Hypothetical Scores for Y>=1
    # =========================================================================
    # For positive counts, max_count should be based on positive observations
    max_count_pos = int(np.nanmax(calib_y_pos)) if len(calib_y_pos) > 0 else 1
    
    pos_groups = _build_pos_count_groups(calib_y_pos, min_group_size, max_count_pos)
    
    if verbose:
        print(f"DGCP Hybrid - Stage 2 (Count) Groups (m={min_group_size}), max_count_pos={max_count_pos}:")
        unique_y, counts = np.unique(calib_y_pos, return_counts=True)
        count_freq = dict(zip(unique_y, counts))
        shown = set()
        for y in range(1, min(15, max_count_pos + 1)):
            g = tuple(pos_groups.get(y, [y]))
            if g not in shown:
                n_samples = sum(count_freq.get(yy, 0) for yy in g)
                print(f"  y={y} -> group {list(g)}: {n_samples} samples")
                shown.add(g)
    
    # Pre-compute group μ values for positive counts
    pos_group_calib_mu = {}
    for y in range(1, max_count_pos + 1):
        group = tuple(pos_groups.get(y, [y]))
        if group not in pos_group_calib_mu:
            mask = np.isin(calib_y_pos, list(group))
            if mask.sum() > 0:
                pos_group_calib_mu[group] = calib_mu_pos[mask]
            else:
                pos_group_calib_mu[group] = np.array([])
    
    # Pre-compute HYPOTHETICAL scores for each positive candidate y
    pos_hypothetical_scores = {}
    
    for y in range(1, max_count_pos + 1):
        group = tuple(pos_groups.get(y, [y]))
        group_mu = pos_group_calib_mu.get(group, np.array([]))
        
        if len(group_mu) > 0:
            pos_hypothetical_scores[y] = ncs_fun(group_mu, np.full(len(group_mu), y))
        else:
            pos_hypothetical_scores[y] = np.array([])
    
    # =========================================================================
    # Construct prediction sets
    # =========================================================================
    results = []
    
    for mu in pred_mu:
        if np.isnan(mu) or mu <= 0:
            results.append({
                "pred": mu,
                "lower": np.nan,
                "upper": np.nan,
                "set_size": 0
            })
            continue
        
        prediction_set = []
        
        # ----- Stage 1: Test Y=0 using BINARY NCS -----
        if len(calib_binary_scores_y0) > 0:
            test_binary_score_y0 = 1.0 - np.exp(-mu)
            pval_y0 = _compute_pvalue(test_binary_score_y0, calib_binary_scores_y0)
            
            if pval_y0 > alpha:
                prediction_set.append(0)
        else:
            # No zero calibration data - include conservatively
            prediction_set.append(0)
        
        # ----- Stage 2: Test each positive count using HYPOTHETICAL scores -----
        # ONLY test up to max_count_pos (observed positive counts)
        for y in range(1, max_count_pos + 1):
            calib_scores = pos_hypothetical_scores.get(y, np.array([]))
            
            if len(calib_scores) == 0:
                # No calibration data - skip (don't conservatively include)
                continue
            
            test_score = ncs_fun(np.array([mu]), np.array([y]))[0]
            pval = _compute_pvalue(test_score, calib_scores)
            
            if pval > alpha:
                prediction_set.append(y)
        
        # ----- Finalize -----
        if not prediction_set:
            if np.exp(-mu) > 0.5:
                prediction_set = [0]
            else:
                prediction_set = [max(1, min(int(np.round(mu)), max_count_pos))]
        
        prediction_set = sorted(set(prediction_set))
        lower = min(prediction_set)
        upper = max(prediction_set)
        
        results.append({
            "pred": mu,
            "lower": int(lower),
            "upper": int(upper),
            "set_size": upper - lower + 1,
            "includes_zero": 0 in prediction_set
        })
    
    return pd.DataFrame(results)


# Convenience Wrapper

def two_stage_dgcp_cp(
    pred_mu: np.ndarray,
    calib_mu: np.ndarray,
    calib_y: np.ndarray,
    alpha: float = 0.1,
    ncs_type: str = "poisson_deviance",
    min_group_size: int = 10,
    max_count: int = None,
    method: str = "hybrid",
    verbose: bool = False
) -> pd.DataFrame:
    """
    Two-Stage DGCP Conformal Prediction for Count Data.
    
    Wrapper that selects the appropriate DGCP variant.
    All variants use HYPOTHETICAL scores for proper calibration.
    
    Parameters:
    -----------
    pred_mu : array
        Predicted Poisson mean for test observations
    calib_mu : array
        Predicted mean for calibration observations
    calib_y : array
        True outcomes for calibration observations
    alpha : float
        Miscoverage rate (default 0.1 for 90% coverage)
    ncs_type : str
        Non-conformity score type ("poisson_deviance" or "poisson_pearson")
    min_group_size : int
        Minimum calibration samples per group (m in DGCP paper)
    max_count : int, optional
        Maximum count to test. If None, uses max observed in calib_y.
    method : str
        - "hybrid": Binary NCS for Y=0 + Hypothetical Poisson NCS for Y>=1
        - "exact": Hypothetical scores, allows disjoint sets
        - "full": Hypothetical scores, contiguous intervals
    verbose : bool
        Print grouping information
        
    Returns:
    --------
    DataFrame with pred, lower, upper columns
    """
    if method == "hybrid":
        return two_stage_dgcp_hybrid_cp(
            pred_mu, calib_mu, calib_y, alpha, ncs_type, min_group_size, max_count, verbose
        )
    elif method == "exact":
        return two_stage_dgcp_exact_cp(
            pred_mu, calib_mu, calib_y, alpha, ncs_type, min_group_size, max_count,
            return_sets=False, verbose=verbose
        )
    elif method == "full":
        return two_stage_dgcp_full_cp(
            pred_mu, calib_mu, calib_y, alpha, ncs_type, min_group_size, max_count, verbose
        )
    else:
        raise ValueError(f"Unknown method: {method}. Use 'hybrid', 'exact', or 'full'.")


