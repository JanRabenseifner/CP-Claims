import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import gammaln, betaln
from scipy.optimize import minimize_scalar, brentq
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.discrete.discrete_model import NegativeBinomial
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from dataclasses import dataclass, field
import warnings
import time
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# LOG-LIKELIHOOD FUNCTIONS (Merz & Wüthrich aligned)

def poisson_log_likelihood_with_exposure(y_obs, mu_pred, exposure=None):
    """
    Compute Poisson log-likelihood for claim counts with exposure.
    
    Following Merz & Wüthrich (2023), for observations N_i with exposure v_i:
    L = sum(N_i * log(v_i * mu_i) - v_i * mu_i - log(N_i!))
    
    where mu_i is the predicted rate per unit exposure.
    
    Parameters:
    -----------
    y_obs : array-like
        Observed claim counts N_i
    mu_pred : array-like
        Predicted claim rate (per unit exposure if exposure given)
    exposure : array-like, optional
        Exposure v_i for each observation. If None, assumed to be 1.
    
    Returns:
    --------
    float : Total log-likelihood
    """
    y_obs = np.asarray(y_obs, dtype=np.float64)
    mu_pred = np.asarray(mu_pred, dtype=np.float64)
    
    if exposure is not None:
        exposure = np.asarray(exposure, dtype=np.float64)
        # Expected count = exposure * rate
        lambda_pred = exposure * mu_pred
    else:
        lambda_pred = mu_pred
    
    # Poisson log-likelihood: N*log(lambda) - lambda - log(N!)
    # Use gammaln(N+1) = log(N!)
    with np.errstate(divide='ignore', invalid='ignore'):
        ll = y_obs * np.log(lambda_pred + 1e-10) - lambda_pred - gammaln(y_obs + 1)
        ll = np.where(np.isfinite(ll), ll, 0)
    
    return np.sum(ll)


def nb_log_likelihood_with_exposure(y_obs, mu_pred, alpha, exposure=None):
    """
    Compute Negative Binomial (NB2) log-likelihood with exposure.
    
    Following Merz & Wüthrich (2023), Eq. (5.38), for NB2 with:
    - Claim counts N_i = v_i * Y_i
    - Exposure v_i
    - Shape parameter alpha (nuisance parameter)
    
    The NB2 pmf for claim count n with exposure v and rate mu:
    P(N=n) = Gamma(n + v*alpha) / (n! * Gamma(v*alpha)) 
             * (mu/(mu+alpha))^n * (alpha/(mu+alpha))^(v*alpha)
    
    Log-likelihood:
    LL = sum( log(Gamma(n+v*alpha)) - log(n!) - log(Gamma(v*alpha))
              + n*log(mu/(mu+alpha)) + v*alpha*log(alpha/(mu+alpha)) )
    
    Parameters:
    -----------
    y_obs : array-like
        Observed claim counts N_i
    mu_pred : array-like
        Predicted claim rate (per unit exposure)
    alpha : float
        NB2 dispersion parameter (alpha > 0). Var(N) = E[N] * (1 + mu/alpha)
    exposure : array-like, optional
        Exposure v_i for each observation. If None, assumed to be 1.
    
    Returns:
    --------
    float : Total log-likelihood
    """
    y_obs = np.asarray(y_obs, dtype=np.float64)
    mu_pred = np.asarray(mu_pred, dtype=np.float64)
    
    if exposure is not None:
        v = np.asarray(exposure, dtype=np.float64)
    else:
        v = np.ones_like(y_obs)
    
    # Ensure positive values
    mu_pred = np.maximum(mu_pred, 1e-10)
    alpha = max(alpha, 1e-10)
    
    n = y_obs  # Claim counts
    
    # Log probability ratio terms
    log_p = np.log(mu_pred / (mu_pred + alpha))  # log(mu/(mu+alpha))
    log_q = np.log(alpha / (mu_pred + alpha))     # log(alpha/(mu+alpha))
    
    # Log-likelihood components
    # gammaln(n + v*alpha) - gammaln(n+1) - gammaln(v*alpha)
    # + n*log_p + v*alpha*log_q
    
    with np.errstate(divide='ignore', invalid='ignore'):
        ll = (gammaln(n + v * alpha) 
              - gammaln(n + 1) 
              - gammaln(v * alpha)
              + n * log_p 
              + v * alpha * log_q)
        ll = np.where(np.isfinite(ll), ll, 0)
    
    return np.sum(ll)


# DEVIANCE LOSS FUNCTIONS

def poisson_deviance_loss(y_true, y_pred):
    """
    Compute Poisson deviance loss: D = 2 * mean(y * log(y/mu) - (y - mu))
    
    This is the standard metric for comparing Poisson regression models,
    as used in Wüthrich & Merz (2023) and the actuarial literature.
    """
    with np.errstate(divide='ignore'):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            xlogy = np.where(y_true != 0, y_true * np.log(y_true / (y_pred + 1e-10)), 0)
            dev = 2 * (xlogy - y_true + y_pred)
    return dev.mean()


def nb_deviance_loss(y_true, y_pred, alpha):
    """
    Compute Negative Binomial (NB2) deviance loss.
    
    Following Merz & Wüthrich (2023), Remark 5.21, the unit deviance is:
    d(y, mu) = 2 * [y * log(y/mu) - (y + alpha) * log((y + alpha)/(mu + alpha))]
    
    IMPORTANT: Deviance can only be compared across models with the SAME alpha!
    
    Parameters:
    -----------
    y_true : array-like
        Observed values
    y_pred : array-like
        Predicted values (mu)
    alpha : float
        NB2 dispersion parameter
    
    Returns:
    --------
    float : Mean deviance loss
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        # Term 1: y * log(y/mu)
        term1 = np.where(y_true > 0, 
                        y_true * np.log(y_true / (y_pred + 1e-10)), 
                        0)
        
        # Term 2: (y + alpha) * log((y + alpha)/(mu + alpha))
        term2 = (y_true + alpha) * np.log((y_true + alpha) / (y_pred + alpha))
        
        dev = 2 * (term1 - term2)
        dev = np.where(np.isfinite(dev), dev, 0)
    
    return dev.mean()


# AIC/BIC FUNCTIONS (Comparable across Poisson and NB)

def compute_aic(n_params, log_likelihood):
    """Compute AIC = 2k - 2ln(L)"""
    return 2 * n_params - 2 * log_likelihood


def compute_bic(n_params, log_likelihood, n_obs):
    """Compute BIC = k*ln(n) - 2ln(L)"""
    return n_params * np.log(n_obs) - 2 * log_likelihood


def compute_comparable_aic_poisson(y_obs, mu_pred, n_params, exposure=None):
    """
    Compute AIC for Poisson model using consistent log-likelihood.
    
    This ensures comparability with NB AIC as per Merz & Wüthrich recommendation.
    """
    ll = poisson_log_likelihood_with_exposure(y_obs, mu_pred, exposure)
    return compute_aic(n_params, ll)


def compute_comparable_aic_nb(y_obs, mu_pred, alpha, n_params, exposure=None):
    """
    Compute AIC for NB model using consistent log-likelihood.
    
    Note: n_params should include +1 for the alpha parameter.
    This ensures comparability with Poisson AIC as per Merz & Wüthrich recommendation.
    """
    ll = nb_log_likelihood_with_exposure(y_obs, mu_pred, alpha, exposure)
    return compute_aic(n_params, ll)


# ITERATIVE ALPHA ESTIMATION (Merz & Wüthrich, Section 5.3.5)

def estimate_alpha_for_nb_v01(y_obs, mu_pred, exposure=None, alpha_init=1.0, 
                          max_iter=20, tol=1e-4, verbose=False):
    """
    Estimate optimal alpha for NB2 model given fixed beta (mu predictions).
    
    Following Merz & Wüthrich iterative procedure:
    For given beta (and thus mu), maximize log-likelihood in alpha.
    
    Parameters:
    -----------
    y_obs : array-like
        Observed claim counts
    mu_pred : array-like
        Predicted rates from the model
    exposure : array-like, optional
        Exposure values
    alpha_init : float
        Initial alpha value
    max_iter : int
        Maximum iterations
    tol : float
        Convergence tolerance
    verbose : bool
        Print progress
    
    Returns:
    --------
    float : Estimated alpha
    """
    def neg_ll(alpha):
        if alpha <= 0:
            return np.inf
        return -nb_log_likelihood_with_exposure(y_obs, mu_pred, alpha, exposure)
    
    # Use bounded optimization
    result = minimize_scalar(neg_ll, bounds=(0.01, 100), method='bounded')
    
    if verbose:
        print(f"    α estimation: α = {result.x:.4f}, LL = {-result.fun:.2f}")
    
    return result.x


def estimate_alpha_for_nb_v02(y_obs, mu_pred, exposure=None, alpha_init=1.0, 
                          max_iter=20, tol=1e-4, verbose=False):
    """
    Estimate optimal alpha for NB2 model given fixed beta (mu predictions).
    
    Following Merz & Wüthrich iterative procedure:
    For given beta (and thus mu), maximize log-likelihood in alpha.
    
    Parameters:
    -----------
    y_obs : array-like
        Observed claim counts
    mu_pred : array-like
        Predicted rates from the model
    exposure : array-like, optional
        Exposure values
    alpha_init : float
        Initial alpha value
    max_iter : int
        Maximum iterations
    tol : float
        Convergence tolerance
    verbose : bool
        Print progress
    
    Returns:
    --------
    float : Estimated alpha
    """    
    def neg_ll(alpha):
        if alpha <= 0:
            return np.inf
        ll = nb_log_likelihood_with_exposure(y_obs, mu_pred, alpha, exposure)
        return -ll if np.isfinite(ll) else np.inf
    
    # Use wider bounds and try different methods
    try:
        # Method 1: Brent's method (more robust)
        result = minimize_scalar(neg_ll, bounds=(1e-4, 1000), method='bounded')
        alpha_opt = result.x
        
        # If α is very large, check if Poisson is better
        if alpha_opt > 100:
            # Try ML estimation via method of moments as sanity check
            if exposure is not None:
                v = np.asarray(exposure)
            else:
                v = np.ones_like(y_obs)
            
            pearson_resid = (y_obs - mu_pred) / np.sqrt(mu_pred + 1e-10)
            phi = np.mean(pearson_resid**2)  # Dispersion index
            
            if phi > 1:
                # Method of moments estimator
                alpha_mom = np.mean(mu_pred**2) / (np.var(y_obs) - np.mean(mu_pred))
                alpha_mom = max(alpha_mom, 1e-4)
                
                # Choose the one with better likelihood
                ll_opt = -result.fun
                ll_mom = nb_log_likelihood_with_exposure(y_obs, mu_pred, alpha_mom, exposure)
                
                alpha_opt = alpha_opt if ll_opt > ll_mom else alpha_mom
            else:
                # No overdispersion detected
                alpha_opt = 1000  # Effectively Poisson
    except:
        # Fallback to method of moments
        if exposure is not None:
            v = np.asarray(exposure)
            mu_counts = mu_pred * v
        else:
            mu_counts = mu_pred
        
        var_counts = np.var(y_obs)
        mean_counts = np.mean(mu_counts)
        
        if var_counts > mean_counts:
            alpha_opt = mean_counts**2 / (var_counts - mean_counts)
            alpha_opt = max(alpha_opt, 1e-4)
        else:
            alpha_opt = 1000  # No overdispersion
    
    return min(alpha_opt, 1000)  # Cap at 1000


def fit_nb_glm_iterative(X_train, y_train, exposure_train=None, 
                                       alpha_init=1.0, max_iter=10, tol=1e-4, verbose=True):
    """
    Fit NB2 GLM following Merz & Wüthrich iterative procedure.
    
    This implementation correctly handles:
    1. Poisson GLM for β estimation given α (same score equations as NB2 with fixed α)
    2. Separate α optimization using NB log-likelihood
    3. Exposure adjustment
    
    Parameters:
    -----------
    X_train : array-like
        Design matrix with intercept
    y_train : array-like
        Response (claim counts)
    exposure_train : array-like, optional
        Exposure values
    alpha_init : float
        Initial alpha value
    max_iter : int
        Maximum iterations
    tol : float
        Convergence tolerance for alpha
    verbose : bool
        Print progress
    
    Returns:
    --------
    dict with keys:
        'beta': fitted coefficients
        'alpha': estimated dispersion parameter  
        'mu_pred': predicted rates
        'converged': bool
        'n_iter': number of iterations
    """
    # Convert to arrays
    X_arr = np.asarray(X_train, dtype=np.float64)
    y_arr = np.asarray(y_train, dtype=np.float64)
    
    if exposure_train is not None:
        exposure_arr = np.asarray(exposure_train, dtype=np.float64)
        offset = np.log(exposure_arr + 1e-10)
    else:
        exposure_arr = np.ones_like(y_arr)
        offset = None
    
    # Add constant if needed
    has_constant = np.any(np.all(X_arr == 1, axis=0))
    if not has_constant:
        X_arr = sm.add_constant(X_arr)
    
    alpha_current = alpha_init
    beta_current = None
    mu_rate = None
    poisson_model = None
    
    if verbose:
        print(f"  Starting iterative NB fitting with α_init = {alpha_init:.4f}")
    
    for iteration in range(max_iter):
        # Step 1: Fit Poisson GLM to get β estimates
        # For NB2 with fixed α, the MLE for β satisfies the same score equations as Poisson
        try:
            if offset is not None:
                poisson_model = sm.GLM(
                    y_arr, X_arr,
                    family=sm.families.Poisson(),
                    offset=offset
                ).fit(maxiter=100, disp=0)
            else:
                poisson_model = sm.GLM(
                    y_arr, X_arr,
                    family=sm.families.Poisson()
                ).fit(maxiter=100, disp=0)
            
            beta_current = poisson_model.params
            mu_counts = poisson_model.predict()  # λ = v * μ
            mu_rate = mu_counts / exposure_arr
            
        except Exception as e:
            if verbose:
                print(f"    Iteration {iteration+1} Poisson fit failed: {e}")
            # Fallback initialization
            if beta_current is None:
                beta_current = np.zeros(X_arr.shape[1])
                beta_current[0] = np.log(np.mean(y_arr / exposure_arr) + 1e-10)
                mu_rate = np.exp(X_arr @ beta_current)
                mu_counts = mu_rate * exposure_arr
        
        # Step 2: Optimize α given current μ (rates)
        alpha_new = estimate_alpha_for_nb(
            y_arr, mu_rate, exposure_arr, 
            alpha_init=alpha_current, verbose=False
        )
        
        if verbose:
            print(f"    Iteration {iteration+1}: α = {alpha_new:.4f}")
        
        # Check convergence
        if iteration > 0 and abs(alpha_new - alpha_current) / (alpha_current + 1e-10) < tol:
            if verbose:
                print(f"  Converged after {iteration+1} iterations")
            alpha_current = alpha_new
            break
        
        alpha_current = alpha_new
    
    # Ensure we have predictions
    if mu_rate is None and beta_current is not None:
        mu_rate = np.exp(X_arr @ beta_current)
        mu_counts = mu_rate * exposure_arr
    
    return {
        'beta': beta_current,
        'alpha': alpha_current,
        'mu_pred': mu_rate,
        'mu_counts': mu_counts,
        'converged': iteration < max_iter - 1,
        'n_iter': iteration + 1,
        'poisson_model': poisson_model
    }


def estimate_alpha_for_nb(y_obs, mu_pred, exposure=None, alpha_init=1.0, 
                                 max_iter=20, tol=1e-4, verbose=False):
    """
    Robust α estimation for NB2 model.
    
    Parameters:
    -----------
    y_obs : array-like
        Observed claim counts
    mu_pred : array-like
        Predicted rates from the model
    exposure : array-like, optional
        Exposure values
    alpha_init : float
        Initial alpha value
    max_iter : int
        Maximum iterations
    tol : float
        Convergence tolerance
    verbose : bool
        Print progress
    
    Returns:
    --------
    float : Estimated alpha
    """
    def neg_ll(alpha):
        # Ensure alpha is a scalar
        alpha_scalar = float(alpha)
        if alpha_scalar <= 0:
            return np.inf
        ll = nb_log_likelihood_with_exposure(y_obs, mu_pred, alpha_scalar, exposure)
        return -ll if np.isfinite(ll) else np.inf
    
    try:
        # Use bounded optimization with reasonable bounds
        result = minimize_scalar(neg_ll, bounds=(1e-4, 100), method='bounded')
        alpha_opt = float(result.x)
        
        # Check if result is meaningful
        if alpha_opt >= 50:  # Very large α ≈ Poisson
            # Check if data shows overdispersion
            if exposure is not None:
                v = np.asarray(exposure)
                mu_counts = mu_pred * v
            else:
                mu_counts = mu_pred
            
            # Calculate dispersion index
            var_obs = np.var(y_obs)
            mean_obs = np.mean(mu_counts)
            
            if var_obs <= mean_obs * 1.1:  # Less than 10% overdispersion
                alpha_opt = 100.0  # Effectively Poisson
            else:
                # Method of moments estimator as fallback
                if var_obs > mean_obs:
                    alpha_mom = mean_obs**2 / (var_obs - mean_obs)
                    alpha_opt = min(max(alpha_mom, 0.1), 50)
        
        return max(alpha_opt, 1e-4)
        
    except Exception as e:
        if verbose:
            print(f"    α estimation error: {e}")
        
        # Fallback to method of moments
        if exposure is not None:
            v = np.asarray(exposure)
            mu_counts = mu_pred * v
        else:
            mu_counts = mu_pred
        
        var_obs = np.var(y_obs)
        mean_obs = np.mean(mu_counts)
        
        if var_obs > mean_obs:
            alpha_mom = mean_obs**2 / (var_obs - mean_obs)
            return min(max(alpha_mom, 0.1), 50)
        else:
            return 100.0  # No overdispersion detected


def fit_nb_glm_iterative_v02(X_train, y_train, exposure_train=None, 
                                   alpha_init=1.0, max_iter=10, tol=1e-4, verbose=True):
    """
    Fit NB2 GLM with iterative alpha estimation following Merz & Wüthrich.
    
    Algorithm:
    1. Start with alpha^(0)
    2. Fit NB GLM with fixed alpha to get beta^(1)
    3. For fixed beta^(1), optimize alpha to get alpha^(1)
    4. Repeat until convergence
    
    This is more robust than simultaneous MLE and handles exposure properly.
    
    Parameters:
    -----------
    X_train : array-like
        Design matrix with intercept
    y_train : array-like
        Response (claim counts)
    exposure_train : array-like, optional
        Exposure values
    alpha_init : float
        Initial alpha value
    max_iter : int
        Maximum iterations
    tol : float
        Convergence tolerance for alpha
    verbose : bool
        Print progress
    
    Returns:
    --------
    dict with keys:
        'beta': fitted coefficients
        'alpha': estimated dispersion parameter  
        'mu_pred': predicted rates
        'converged': bool
        'n_iter': number of iterations
        'cov_beta': covariance matrix for beta
    """   
    # Convert to arrays
    X_arr = np.asarray(X_train, dtype=np.float64)
    y_arr = np.asarray(y_train, dtype=np.float64)
    
    if exposure_train is not None:
        exposure_arr = np.asarray(exposure_train, dtype=np.float64)
        offset = np.log(exposure_arr + 1e-10)
    else:
        exposure_arr = np.ones_like(y_arr)
        offset = None
    
    # Add constant if needed
    has_constant = np.any(np.all(X_arr == 1, axis=0))
    if not has_constant:
        X_arr = sm.add_constant(X_arr)
    
    alpha_current = alpha_init
    beta_current = None
    mu_rate = None
    mu_counts = None
    
    for iteration in range(max_iter):
        # Step 1: Fit Poisson GLM to get beta estimates
        # (Under NB2, the MLE for beta given alpha uses the same score equations as Poisson)
        try:
            if offset is not None:
                poisson_model = sm.GLM(y_arr, X_arr, family=sm.families.Poisson(),
                                     offset=offset).fit(maxiter=100, disp=0)
            else:
                poisson_model = sm.GLM(y_arr, X_arr, family=sm.families.Poisson()
                                     ).fit(maxiter=100, disp=0)
            beta_current = poisson_model.params
            mu_counts = poisson_model.predict()
            mu_rate = mu_counts / exposure_arr
            
        except Exception as e:
            if verbose:
                print(f"    Iteration {iteration+1} Poisson fit failed: {e}")
            # Initialize with simple mean
            beta_current = np.zeros(X_arr.shape[1])
            beta_current[0] = np.log(np.mean(y_arr / exposure_arr) + 1e-10)
            eta = X_arr @ beta_current
            mu_rate = np.exp(eta)
            mu_counts = mu_rate * exposure_arr
        
        # Step 2: Optimize alpha given current mu_rate
        alpha_new = estimate_alpha_for_nb(y_arr, mu_rate, exposure_arr, 
                                         alpha_init=alpha_current, verbose=False)
        
        if verbose:
            print(f"    Iteration {iteration+1}: α = {alpha_new:.4f}")
        
        # Check convergence
        if abs(alpha_new - alpha_current) / (alpha_current + 1e-10) < tol:
            if verbose:
                print(f"  Converged after {iteration+1} iterations")
            break
        
        alpha_current = alpha_new
    
    return {
        'beta': beta_current,
        'alpha': alpha_current,
        'mu_pred': mu_rate,
        'mu_counts': mu_counts,
        'converged': iteration < max_iter - 1,
        'n_iter': iteration + 1
    }


def fit_nb_glm_iterative_v01(X_train, y_train, exposure_train=None, 
                         alpha_init=1.0, max_iter=10, tol=1e-4, verbose=True):

    # Ensure proper format
    if hasattr(X_train, 'values'):
        X_train = X_train.values
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    
    if exposure_train is not None:
        exposure_train = np.asarray(exposure_train, dtype=np.float64)
        offset = np.log(exposure_train + 1e-10)
    else:
        offset = None
        exposure_train = np.ones_like(y_train)
    
    # Check for constant column
    has_constant = np.any(np.all(X_train == 1, axis=0))
    if not has_constant:
        X_train = sm.add_constant(X_train)
    
    alpha_current = alpha_init
    beta_current = None
    
    if verbose:
        print(f"  Starting iterative NB fitting with α_init = {alpha_init:.4f}")
    
    for iteration in range(max_iter):
        # Step 1: Fit Poisson GLM to get initial beta (faster, more stable)
        if iteration == 0:
            try:
                if offset is not None:
                    poisson_model = sm.GLM(
                        y_train, X_train,
                        family=sm.families.Poisson(),
                        offset=offset
                    ).fit(maxiter=100, disp=0)
                else:
                    poisson_model = sm.GLM(
                        y_train, X_train,
                        family=sm.families.Poisson()
                    ).fit(maxiter=100, disp=0)
                beta_current = poisson_model.params
                mu_pred = poisson_model.predict()  # This gives exposure * rate
                # Convert back to rate
                mu_rate = mu_pred / exposure_train
            except Exception as e:
                if verbose:
                    print(f"    Poisson init failed: {e}")
                # Simple initialization
                beta_current = np.zeros(X_train.shape[1])
                beta_current[0] = np.log(np.mean(y_train / exposure_train) + 1e-10)
                mu_rate = np.exp(X_train @ beta_current)
        else:
            # Use NB GLM with fixed alpha (via family)
            # statsmodels doesn't support fixed alpha easily, so we use Poisson
            # with quasi-likelihood adjustment (the beta estimates are the same)
            try:
                if offset is not None:
                    poisson_model = sm.GLM(
                        y_train, X_train,
                        family=sm.families.Poisson(),
                        offset=offset
                    ).fit(maxiter=100, disp=0)
                else:
                    poisson_model = sm.GLM(
                        y_train, X_train,
                        family=sm.families.Poisson()
                    ).fit(maxiter=100, disp=0)
                beta_current = poisson_model.params
                mu_pred = poisson_model.predict()
                mu_rate = mu_pred / exposure_train
            except Exception as e:
                if verbose:
                    print(f"    Iteration {iteration+1} fit failed: {e}")
                break
        
        # Step 2: Optimize alpha for fixed beta
        alpha_new = estimate_alpha_for_nb(
            y_train, mu_rate, exposure_train, 
            alpha_init=alpha_current, verbose=False
        )
        
        if verbose:
            print(f"    Iteration {iteration+1}: α = {alpha_new:.4f}")
        
        # Check convergence
        if abs(alpha_new - alpha_current) / (alpha_current + 1e-10) < tol:
            if verbose:
                print(f"  Converged after {iteration+1} iterations")
            alpha_current = alpha_new
            break
        
        alpha_current = alpha_new
    
    # Final covariance matrix (from Poisson, adjusted for NB)
    # Under NB2, the asymptotic covariance is scaled by (1 + mu/alpha)
    try:
        cov_beta = poisson_model.cov_params()
    except:
        cov_beta = np.eye(len(beta_current))
    
    return {
        'beta': beta_current,
        'alpha': alpha_current,
        'mu_pred': mu_rate,
        'mu_counts': mu_pred if 'mu_pred' in dir() else mu_rate * exposure_train,
        'converged': iteration < max_iter - 1,
        'n_iter': iteration + 1,
        'cov_beta': cov_beta,
        'poisson_model': poisson_model if 'poisson_model' in dir() else None
    }


# LEGACY FUNCTIONS (for backward compatibility)

def poisson_log_likelihood(y_true, y_pred):
    """
    Compute Poisson log-likelihood: sum(y * log(mu) - mu - log(y!))
    DEPRECATED: Use poisson_log_likelihood_with_exposure for proper exposure handling.
    """
    ll = y_true * np.log(y_pred + 1e-10) - y_pred - gammaln(y_true + 1)
    return np.sum(ll)


@dataclass
class ModelComparisonResults:
    """Data class to store model comparison results."""
    model: str
    n_params: int = field(default=None)
    run_time: float = field(default=None)
    poisson_deviance_loss_train: float = field(default=None)
    poisson_deviance_loss_test: float = field(default=None)
    pred_avg_freq_train: float = field(default=None)
    pred_avg_freq_test: float = field(default=None)
    aic: float = field(default=None)
    bic: float = field(default=None)
    rmse_test: float = field(default=None)


# PREPROCESSING UTILITIES (aligned with glm_preprocess_fit_actuarial_models_with_transformers.ipynb)

def prepare_glm_design_matrices(train_data, test_data, cat_vars, num_vars, 
                                 add_driv_age_polynomials=False,
                                 add_interaction_terms=False,
                                 sf_var='SF_Class'):
    """
    Prepare design matrices for GLM fitting, aligned with the preprocessing
    in glm_preprocess_fit_actuarial_models_with_transformers.ipynb.
    
    Parameters:
    -----------
    train_data, test_data : DataFrames
    cat_vars : list of categorical variable names
    num_vars : list of numerical variable names
    add_driv_age_polynomials : bool
        If True, add DrivAge^1, ..., DrivAge^4, log(DrivAge) features (GLM2 style)
    add_interaction_terms : bool
        If True, add DrivAge × SF_Class interaction terms (GLM3 style)
    sf_var : str
        Name of the SF/BonusMalus-like variable for interactions
    
    Returns:
    --------
    X_train, X_test : Design matrices with constant term
    feature_info : dict with feature counts
    """
    X_train_parts = []
    X_test_parts = []
    
    # Categorical variables with dummy encoding
    for v in cat_vars:
        if v not in train_data.columns:
            continue
        # Skip DrivAge if we're adding polynomials
        if add_driv_age_polynomials and v == 'DrivAge':
            continue
            
        dummies_train = pd.get_dummies(train_data[v].astype(str), prefix=v, drop_first=True)
        dummies_test = pd.get_dummies(test_data[v].astype(str), prefix=v, drop_first=True)
        X_train_parts.append(dummies_train)
        X_test_parts.append(dummies_test)
    
    # Numerical variables
    for v in num_vars:
        if v not in train_data.columns:
            continue
        num_train = pd.to_numeric(train_data[v], errors='coerce').fillna(0).values.reshape(-1, 1)
        num_test = pd.to_numeric(test_data[v], errors='coerce').fillna(0).values.reshape(-1, 1)
        X_train_parts.append(pd.DataFrame(num_train, columns=[v], index=train_data.index))
        X_test_parts.append(pd.DataFrame(num_test, columns=[v], index=test_data.index))
    
    # Polynomial DrivAge features (GLM2 style)
    if add_driv_age_polynomials and 'DrivAge' in train_data.columns:
        driv_train = pd.to_numeric(train_data['DrivAge'].astype(str), errors='coerce').fillna(40)
        driv_test = pd.to_numeric(test_data['DrivAge'].astype(str), errors='coerce').fillna(40)
        
        poly_features_train = pd.DataFrame(index=train_data.index)
        poly_features_test = pd.DataFrame(index=test_data.index)
        
        for i in range(1, 5):
            poly_features_train[f'DrivAge_{i}'] = driv_train ** i
            poly_features_test[f'DrivAge_{i}'] = driv_test ** i
        
        poly_features_train['DrivAge_log'] = np.log(driv_train + 1)
        poly_features_test['DrivAge_log'] = np.log(driv_test + 1)
        
        # Normalize by training mean
        for col in poly_features_train.columns:
            mean_val = poly_features_train[col].mean()
            if mean_val != 0:
                poly_features_train[col] = poly_features_train[col] / mean_val
                poly_features_test[col] = poly_features_test[col] / mean_val
        
        X_train_parts.append(poly_features_train)
        X_test_parts.append(poly_features_test)
    
    # Interaction terms (GLM3 style)
    if add_interaction_terms and 'DrivAge' in train_data.columns and sf_var in train_data.columns:
        driv_train = pd.to_numeric(train_data['DrivAge'].astype(str), errors='coerce').fillna(40)
        driv_test = pd.to_numeric(test_data['DrivAge'].astype(str), errors='coerce').fillna(40)
        sf_train = pd.to_numeric(train_data[sf_var], errors='coerce').fillna(0)
        sf_test = pd.to_numeric(test_data[sf_var], errors='coerce').fillna(0)
        
        interact_train = pd.DataFrame(index=train_data.index)
        interact_test = pd.DataFrame(index=test_data.index)
        
        interact_train['DrivAge_x_SF'] = driv_train * sf_train
        interact_test['DrivAge_x_SF'] = driv_test * sf_test
        interact_train['DrivAge2_x_SF'] = (driv_train ** 2) * sf_train
        interact_test['DrivAge2_x_SF'] = (driv_test ** 2) * sf_test
        
        # Normalize
        for col in interact_train.columns:
            mean_val = interact_train[col].mean()
            if mean_val != 0:
                interact_train[col] = interact_train[col] / mean_val
                interact_test[col] = interact_test[col] / mean_val
        
        X_train_parts.append(interact_train)
        X_test_parts.append(interact_test)
    
    # Combine all parts
    X_train = pd.concat(X_train_parts, axis=1) if X_train_parts else pd.DataFrame(index=train_data.index)
    X_test = pd.concat(X_test_parts, axis=1) if X_test_parts else pd.DataFrame(index=test_data.index)
    
    # Align columns
    for col in X_train.columns:
        if col not in X_test.columns:
            X_test[col] = 0.0
    for col in list(X_test.columns):
        if col not in X_train.columns:
            X_test = X_test.drop(columns=[col])
    X_test = X_test[X_train.columns]
    
    # Add constant and ensure float
    X_train = sm.add_constant(X_train.astype(np.float64))
    X_test = sm.add_constant(X_test.astype(np.float64))
    
    feature_info = {
        'n_features': len(X_train.columns) - 1,  # Exclude constant
        'n_cat_features': len([c for c in X_train.columns if any(v in c for v in cat_vars)]),
        'n_num_features': len(num_vars)
    }
    
    return X_train, X_test, feature_info


# POISSON REGRESSION BENCHMARKS

def poisson_normal_approximation_interval(model_or_result, X_test, exposure_test=None, alpha=0.1,
                                           test_df=None, use_formula_api=False):
    """
    $\tilde{\\Gamma}_1$: Normal approximation prediction interval for Poisson regression.
    
    This is a TRUE prediction interval that accounts for parameter estimation uncertainty.
    Based on Myers & Montgomery (1997) and Kim et al. (2022), Section 4.2.
    
    Parameters:
    -----------
    model_or_result : fitted statsmodels Poisson GLM OR dict with 'mu_pred', 'cov_beta'
    X_test : array-like, shape (n_test, n_features)
        Test covariates (with intercept if needed) - used for variance calculation
    exposure_test : array-like, optional
        Exposure for test points
    alpha : float, default=0.1
    test_df : DataFrame, optional
        Test data for formula-based prediction (required if use_formula_api=True)
    use_formula_api : bool, default=False
        If True, use test_df for prediction (formula API)
    
    Returns:
    --------
    DataFrame with columns: pred, lower, upper
    """
    # Handle both statsmodels model and result dict
    if isinstance(model_or_result, dict):
        # Result dict from our fitting procedure
        mu_pred = np.asarray(model_or_result['mu_pred_test'])
        cov_beta = np.asarray(model_or_result['cov_beta'])
    elif use_formula_api and test_df is not None:
        # Use DataFrame-based prediction for formula API
        if exposure_test is not None:
            offset_test = np.log(exposure_test + 1e-10)
            mu_pred = model_or_result.predict(test_df, offset=offset_test)
        else:
            mu_pred = model_or_result.predict(test_df)
        mu_pred = np.asarray(mu_pred)
        cov_beta = np.asarray(model_or_result.cov_params())
    else:
        # Statsmodels model - use direct matrix prediction
        if exposure_test is not None:
            offset_test = np.log(exposure_test + 1e-10)
            # Manual linear predictor calculation
            X_array = np.asarray(X_test)
            eta = X_array @ model_or_result.params.values
            mu_pred = np.exp(eta + offset_test)
        else:
            X_array = np.asarray(X_test)
            eta = X_array @ model_or_result.params.values
            mu_pred = np.exp(eta)
        cov_beta = np.asarray(model_or_result.cov_params())
    
    # Get linear predictor and its variance
    # η = Xβ, with Var(η) = X * Cov(β) * Xᵀ
    X_array = np.asarray(X_test)
    
    # Compute Var(η) for each test point
    var_eta = np.zeros(len(mu_pred))
    for i in range(len(mu_pred)):
        x_i = X_array[i, :]
        var_eta[i] = x_i @ cov_beta @ x_i.T
    
    # For Poisson with log link: μ = exp(η)
    # By delta method: Var(μ) ≈ (dμ/dη)² * Var(η) = μ² * Var(η)
    var_mu = (mu_pred**2) * var_eta
    
    # Total variance for prediction: V0 = μ + μ² * Var(η)
    V0 = mu_pred + var_mu
    
    # Normal approximation prediction interval
    z = stats.norm.ppf(1 - alpha/2)
    half_width = z * np.sqrt(V0)
    
    lower = np.maximum(0, mu_pred - half_width)
    upper = mu_pred + half_width
    
    return pd.DataFrame({
        "pred": mu_pred,
        "lower": np.floor(lower).astype(int),
        "upper": np.ceil(upper).astype(int)
    })


def poisson_residual_bootstrap_interval(model_or_result, X_calib, y_calib, exposure_calib,
                                         X_test, exposure_test=None, 
                                         alpha=0.1, n_bootstraps=500,
                                         calib_df=None, test_df=None, use_formula_api=False):
    """
    $\tilde{\\Gamma}_5$: Residual bootstrap prediction interval for Poisson regression.
    
    This is a TRUE prediction interval that accounts for parameter estimation uncertainty.
    Based on Davison & Hinkley (1997) and Kim et al. (2022), Section 4.2.
    
    Parameters:
    -----------
    model_or_result : fitted statsmodels Poisson GLM OR dict with predictions
    X_calib, y_calib : calibration data (design matrix)
    exposure_calib : exposure for calibration data
    X_test : test covariates (design matrix)
    exposure_test : exposure for test data
    alpha : float, default=0.1
    n_bootstraps : int, default=500
    calib_df, test_df : DataFrames for formula-based prediction
    use_formula_api : bool, default=False
    
    Returns:
    --------
    DataFrame with columns: pred, lower, upper
    """
    # Handle both statsmodels model and result dict
    if isinstance(model_or_result, dict):
        # Result dict with pre-computed predictions
        mu_calib = np.asarray(model_or_result['mu_pred_calib'])
        mu_test = np.asarray(model_or_result['mu_pred_test'])
    elif use_formula_api and calib_df is not None and test_df is not None:
        # Use DataFrame-based prediction for formula API
        if exposure_calib is not None:
            offset_calib = np.log(exposure_calib + 1e-10)
            mu_calib = np.asarray(model_or_result.predict(calib_df, offset=offset_calib))
        else:
            mu_calib = np.asarray(model_or_result.predict(calib_df))
        
        if exposure_test is not None:
            offset_test = np.log(exposure_test + 1e-10)
            mu_test = np.asarray(model_or_result.predict(test_df, offset=offset_test))
        else:
            mu_test = np.asarray(model_or_result.predict(test_df))
    else:
        # Statsmodels model - use direct matrix prediction
        X_calib_arr = np.asarray(X_calib)
        X_test_arr = np.asarray(X_test)
        
        if exposure_calib is not None:
            offset_calib = np.log(exposure_calib + 1e-10)
            eta = X_calib_arr @ model_or_result.params.values
            mu_calib = np.exp(eta + offset_calib)
        else:
            eta = X_calib_arr @ model_or_result.params.values
            mu_calib = np.exp(eta)
        
        if exposure_test is not None:
            offset_test = np.log(exposure_test + 1e-10)
            eta = X_test_arr @ model_or_result.params.values
            mu_test = np.exp(eta + offset_test)
        else:
            eta = X_test_arr @ model_or_result.params.values
            mu_test = np.exp(eta)
    
    # Compute Pearson residuals
    eps = 1e-6
    pearson_resid = (y_calib - mu_calib) / np.sqrt(mu_calib + eps)
    
    # Standardize residuals
    resid_mean = np.mean(pearson_resid)
    resid_std = np.std(pearson_resid)
    if resid_std > 0:
        pearson_resid = (pearson_resid - resid_mean) / resid_std
    
    results = []
    for i, mu in enumerate(mu_test):
        # Resample residuals
        boot_resid = np.random.choice(pearson_resid, size=n_bootstraps, replace=True)
        
        # Generate bootstrap predictions
        boot_counts = mu + boot_resid * np.sqrt(mu + eps)
        boot_counts = np.maximum(boot_counts, 0)
        
        # Get quantiles
        lower = int(max(0, np.floor(np.quantile(boot_counts, alpha/2))))
        upper = int(np.ceil(np.quantile(boot_counts, 1 - alpha/2)))
        
        results.append({
            "pred": mu,
            "lower": lower,
            "upper": upper
        })
    
    return pd.DataFrame(results)


# NEGATIVE BINOMIAL REGRESSION BENCHMARKS (Kim et al. 2022 aligned)

def negative_binomial_chebyshev_interval(model_or_result, X_test, exposure_test=None, 
                                          X_calib=None, y_calib=None, exposure_calib=None,
                                          alpha=0.1, disp_param=None):
    """
    Chebyshev prediction interval for Negative Binomial regression.
    
    Following Kim et al. (2022), Section 5.4, Table 3 - Gamma mixture (NB):
    
    For NB2 (Poisson-Gamma mixture) with shape r = 1/α:
    - Mean: E[Y] = μ
    - Variance: Var[Y] = μ + μ²/r = μ + α*μ²
    
    The Chebyshev PI with parameter uncertainty (Eq. from Table 3):
    σ₀² = λ₀² * {Var[η̂₀] + (1 + Var[η̂₀]) / r}
    
    Total variance: V₀ = μ + σ₀²
    Upper bound: μ + sqrt((1/α) - 1) * sqrt(V₀)
    
    Parameters:
    -----------
    model_or_result : fitted model OR dict from fit_nb_glm_iterative
    X_test : test covariates
    exposure_test : exposure for test data
    X_calib, y_calib : optional calibration data for dispersion estimation
    exposure_calib : exposure for calibration data
    alpha : float, default=0.1
        Confidence level (PI is 1-alpha)
    disp_param : float, optional
        Override for dispersion parameter (α in NB2)
    
    Returns:
    --------
    DataFrame with columns: pred, lower, upper
    """
    # Handle both statsmodels model and our custom result dict
    if isinstance(model_or_result, dict):
        # Result from fit_nb_glm_iterative
        nb_result = model_or_result
        beta = nb_result['beta']
        disp_alpha = disp_param if disp_param is not None else nb_result['alpha']
        cov_beta = nb_result['cov_beta']
        
        # Compute predictions
        X_array = np.asarray(X_test)
        if X_array.shape[1] != len(beta):
            X_array = sm.add_constant(X_array)
        
        eta = X_array @ beta
        if exposure_test is not None:
            exposure_test = np.asarray(exposure_test)
            mu_pred = exposure_test * np.exp(eta)
        else:
            mu_pred = np.exp(eta)
            exposure_test = np.ones(len(mu_pred))
    else:
        # Standard statsmodels model
        model = model_or_result
        
        if exposure_test is not None:
            offset_test = np.log(np.asarray(exposure_test) + 1e-10)
            mu_pred = model.predict(X_test, offset=offset_test)
        else:
            mu_pred = model.predict(X_test)
            exposure_test = np.ones(len(mu_pred))
        
        # Get dispersion parameter
        if disp_param is not None:
            disp_alpha = disp_param
        elif hasattr(model, 'alpha'):
            disp_alpha = model.alpha
        elif hasattr(model, 'params') and len(model.params) > 0:
            n_betas = X_test.shape[1]
            if len(model.params) > n_betas:
                disp_alpha = np.exp(model.params[-1])
            else:
                disp_alpha = 1.0
        else:
            disp_alpha = 1.0
        
        # Get covariance matrix
        try:
            cov_params_full = model.cov_params()
            X_array = np.asarray(X_test)
            n_betas = X_array.shape[1]
            if cov_params_full.shape[0] > n_betas:
                cov_beta = cov_params_full[:n_betas, :n_betas]
            else:
                cov_beta = cov_params_full
        except:
            cov_beta = None
            X_array = np.asarray(X_test)
    
    # Estimate dispersion from calibration data if needed
    if disp_alpha <= 0 or np.isnan(disp_alpha):
        if X_calib is not None and y_calib is not None:
            if isinstance(model_or_result, dict):
                X_calib_arr = np.asarray(X_calib)
                if X_calib_arr.shape[1] != len(beta):
                    X_calib_arr = sm.add_constant(X_calib_arr)
                eta_calib = X_calib_arr @ beta
                if exposure_calib is not None:
                    mu_calib = np.asarray(exposure_calib) * np.exp(eta_calib)
                else:
                    mu_calib = np.exp(eta_calib)
            else:
                if exposure_calib is not None:
                    offset_calib = np.log(np.asarray(exposure_calib) + 1e-10)
                    mu_calib = model.predict(X_calib, offset=offset_calib)
                else:
                    mu_calib = model.predict(X_calib)
            
            # Estimate alpha from method of moments
            disp_alpha = estimate_alpha_for_nb(y_calib, mu_calib, exposure_calib)
        else:
            disp_alpha = 1.0
    
    # NB shape parameter r = 1/α
    r = 1.0 / (disp_alpha + 1e-10)
    
    # Compute Var[η̂] for each test point: Var[η̂] = x' Cov(β) x
    if cov_beta is not None and cov_beta.shape[0] == X_array.shape[1]:
        var_eta = np.zeros(len(mu_pred))
        for i in range(len(mu_pred)):
            x_i = X_array[i, :]
            var_eta[i] = x_i @ cov_beta @ x_i.T
    else:
        # Fallback: no parameter uncertainty
        var_eta = np.zeros(len(mu_pred))
    
    # Kim et al. (2022) Table 3 formula for Gamma mixture (NB):
    # σ₀² = λ₀² * {Var[η̂₀] + (1 + Var[η̂₀]) / r}
    # where λ₀ = μ (predicted mean)
    sigma_0_sq = (mu_pred**2) * (var_eta + (1 + var_eta) / r)
    
    # Total prediction variance: V₀ = μ + σ₀²
    # (the μ term accounts for Poisson-like variance, σ₀² for overdispersion + parameter uncertainty)
    total_var = mu_pred + sigma_0_sq
    
    # Chebyshev bound: P(|Y - μ| ≥ t*σ) ≤ 1/t²
    # For one-sided: P(Y - μ ≥ t*σ) ≤ 1/(1 + t²)
    # To get coverage 1-α: t = sqrt(1/α - 1)
    t = np.sqrt(1/alpha - 1)
    upper = mu_pred + t * np.sqrt(total_var)
    
    return pd.DataFrame({
        "pred": mu_pred,
        "lower": np.zeros_like(mu_pred).astype(int),  # One-sided (conservative)
        "upper": np.ceil(upper).astype(int)
    })


def negative_binomial_parametric_bootstrap_interval(model_or_result, X_calib, y_calib, exposure_calib,
                                                     X_test, exposure_test=None,
                                                     alpha=0.1, n_bootstraps=500, 
                                                     disp_param=None, include_param_uncertainty=True):
    """
    Parametric bootstrap prediction interval for Negative Binomial regression.
    
    Following Kim et al. (2022), Section 5.2 and Davison & Hinkley (1997):
    This accounts for parameter estimation uncertainty through residual resampling.
    
    For NB2: The bootstrap incorporates both:
    1. Sampling variability (from NB distribution)
    2. Parameter estimation uncertainty (via residual resampling)
    
    Parameters:
    -----------
    model_or_result : fitted model OR dict from fit_nb_glm_iterative
    X_calib, y_calib : calibration data
    exposure_calib : exposure for calibration data
    X_test : test covariates
    exposure_test : exposure for test data
    alpha : float, default=0.1
    n_bootstraps : int, default=500
    disp_param : float, optional
        Override for dispersion parameter
    include_param_uncertainty : bool, default=True
        If True, use residual resampling to account for parameter uncertainty
        If False, just sample from fitted NB distribution
    
    Returns:
    --------
    DataFrame with columns: pred, lower, upper
    """
    # Handle both statsmodels model and our custom result dict
    if isinstance(model_or_result, dict):
        nb_result = model_or_result
        beta = nb_result['beta']
        disp_alpha = disp_param if disp_param is not None else nb_result['alpha']
        
        # Compute test predictions
        X_test_arr = np.asarray(X_test)
        if X_test_arr.shape[1] != len(beta):
            X_test_arr = sm.add_constant(X_test_arr)
        
        eta_test = X_test_arr @ beta
        if exposure_test is not None:
            exposure_test = np.asarray(exposure_test)
            mu_test = exposure_test * np.exp(eta_test)
        else:
            mu_test = np.exp(eta_test)
            exposure_test = np.ones(len(mu_test))
        
        # Compute calibration predictions
        X_calib_arr = np.asarray(X_calib)
        if X_calib_arr.shape[1] != len(beta):
            X_calib_arr = sm.add_constant(X_calib_arr)
        
        eta_calib = X_calib_arr @ beta
        if exposure_calib is not None:
            exposure_calib = np.asarray(exposure_calib)
            mu_calib = exposure_calib * np.exp(eta_calib)
        else:
            mu_calib = np.exp(eta_calib)
            exposure_calib = np.ones(len(mu_calib))
    else:
        model = model_or_result
        
        # Get test predictions
        if exposure_test is not None:
            exposure_test = np.asarray(exposure_test)
            offset_test = np.log(exposure_test + 1e-10)
            mu_test = model.predict(X_test, offset=offset_test)
        else:
            mu_test = model.predict(X_test)
            exposure_test = np.ones(len(mu_test))
        
        # Get calibration predictions
        if exposure_calib is not None:
            exposure_calib = np.asarray(exposure_calib)
            offset_calib = np.log(exposure_calib + 1e-10)
            mu_calib = model.predict(X_calib, offset=offset_calib)
        else:
            mu_calib = model.predict(X_calib)
            exposure_calib = np.ones(len(mu_calib))
        
        # Get dispersion parameter
        if disp_param is not None:
            disp_alpha = disp_param
        elif hasattr(model, 'alpha'):
            disp_alpha = model.alpha
        elif hasattr(model, 'params') and len(model.params) > 0:
            n_betas = X_test.shape[1]
            if len(model.params) > n_betas:
                disp_alpha = np.exp(model.params[-1])
            else:
                disp_alpha = estimate_alpha_for_nb(y_calib, mu_calib, exposure_calib)
        else:
            disp_alpha = estimate_alpha_for_nb(y_calib, mu_calib, exposure_calib)
    
    y_calib = np.asarray(y_calib)
    
    # Compute standardized Pearson residuals from calibration data
    # For NB2: Var(Y) = μ + α*μ², so SD = sqrt(μ + α*μ²)
    var_calib = mu_calib + disp_alpha * (mu_calib**2)
    pearson_resid = (y_calib - mu_calib) / np.sqrt(var_calib + 1e-10)
    
    # Standardize residuals (mean 0, std 1)
    resid_mean = np.mean(pearson_resid)
    resid_std = np.std(pearson_resid)
    if resid_std > 0:
        std_resid = (pearson_resid - resid_mean) / resid_std
    else:
        std_resid = pearson_resid - resid_mean
    
    # NB parameters for scipy: n = 1/α, p = n/(n + μ)
    r_param = 1.0 / (disp_alpha + 1e-10)
    
    results = []
    
    for i, mu in enumerate(mu_test):
        if include_param_uncertainty:
            # Method: Residual-based bootstrap (Davison & Hinkley, 1997)
            # 1. Sample residuals with replacement
            # 2. Apply to get bootstrap Y values
            # 3. Collect quantiles
            
            boot_samples = []
            for _ in range(n_bootstraps):
                # Sample a residual
                eps = np.random.choice(std_resid)
                
                # Generate bootstrap Y: Y* = μ + ε * sqrt(Var)
                # where Var = μ + α*μ²
                var_pred = mu + disp_alpha * (mu**2)
                y_boot = mu + eps * np.sqrt(var_pred)
                
                # Ensure non-negative integer
                y_boot = max(0, int(np.round(y_boot)))
                boot_samples.append(y_boot)
            
            boot_samples = np.array(boot_samples)
        else:
            # Simple approach: just sample from NB distribution
            p_param = r_param / (r_param + mu + 1e-10)
            boot_samples = stats.nbinom.rvs(r_param, p_param, size=n_bootstraps)
        
        # Get quantiles
        lower = int(max(0, np.floor(np.percentile(boot_samples, 100 * alpha/2))))
        upper = int(np.ceil(np.percentile(boot_samples, 100 * (1 - alpha/2))))
        
        results.append({
            "pred": mu,
            "lower": lower,
            "upper": upper
        })
    
    return pd.DataFrame(results)


def negative_binomial_plug_in_interval(model_or_result, X_test, exposure_test=None,
                                        alpha=0.1, disp_param=None):
    """
    Plug-in prediction interval for Negative Binomial regression (Γ̄₁ in Kim et al.).
    
    Following Kim et al. (2022), Eq. (13):
    PI = [μ̂ ± z_{α/2} * sqrt(μ̂(1 + μ̂)/ξ + μ̂ + n⁻¹μ̂²ψ'Ξ₁₁ψ)]
    
    This is a two-sided interval using normal approximation.
    
    Parameters:
    -----------
    model_or_result : fitted model OR dict from fit_nb_glm_iterative
    X_test : test covariates
    exposure_test : exposure for test data
    alpha : float, default=0.1
    disp_param : float, optional
        Override for dispersion parameter
    
    Returns:
    --------
    DataFrame with columns: pred, lower, upper
    """
    # Handle both statsmodels model and our custom result dict
    if isinstance(model_or_result, dict):
        nb_result = model_or_result
        beta = nb_result['beta']
        disp_alpha = disp_param if disp_param is not None else nb_result['alpha']
        cov_beta = nb_result['cov_beta']
        
        X_array = np.asarray(X_test)
        if X_array.shape[1] != len(beta):
            X_array = sm.add_constant(X_array)
        
        eta = X_array @ beta
        if exposure_test is not None:
            mu_pred = np.asarray(exposure_test) * np.exp(eta)
        else:
            mu_pred = np.exp(eta)
    else:
        model = model_or_result
        
        if exposure_test is not None:
            offset_test = np.log(np.asarray(exposure_test) + 1e-10)
            mu_pred = model.predict(X_test, offset=offset_test)
        else:
            mu_pred = model.predict(X_test)
        
        if disp_param is not None:
            disp_alpha = disp_param
        elif hasattr(model, 'alpha'):
            disp_alpha = model.alpha
        else:
            disp_alpha = 1.0
        
        try:
            cov_params_full = model.cov_params()
            X_array = np.asarray(X_test)
            n_betas = X_array.shape[1]
            if cov_params_full.shape[0] > n_betas:
                cov_beta = cov_params_full[:n_betas, :n_betas]
            else:
                cov_beta = cov_params_full
        except:
            cov_beta = None
            X_array = np.asarray(X_test)
    
    # Compute parameter uncertainty contribution
    if cov_beta is not None and cov_beta.shape[0] == X_array.shape[1]:
        var_eta = np.zeros(len(mu_pred))
        for i in range(len(mu_pred)):
            x_i = X_array[i, :]
            var_eta[i] = x_i @ cov_beta @ x_i.T
    else:
        var_eta = np.zeros(len(mu_pred))
    
    # Kim et al. (2022) Eq. (13) variance formula:
    # V = μ(1+μ)/ξ + μ + μ²*Var[η̂]
    # where ξ = 1/α (so μ(1+μ)/ξ = α*μ(1+μ))
    # Simplified: V ≈ μ + α*μ² + α*μ + μ²*Var[η̂]
    
    # Total prediction variance
    total_var = (mu_pred + disp_alpha * mu_pred * (1 + mu_pred) + 
                 (mu_pred**2) * var_eta)
    
    # Normal approximation
    z = stats.norm.ppf(1 - alpha/2)
    half_width = z * np.sqrt(total_var)
    
    lower = np.maximum(0, mu_pred - half_width)
    upper = mu_pred + half_width
    
    return pd.DataFrame({
        "pred": mu_pred,
        "lower": np.floor(lower).astype(int),
        "upper": np.ceil(upper).astype(int)
    })


# MODEL FITTING FUNCTIONS

def fit_poisson_glm_for_benchmarks(X_train, y_train, exposure_train=None):
    """
    Fit Poisson GLM for benchmark methods.
    Returns fitted model and design matrix for predictions.
    
    Parameters:
    -----------
    X_train : numpy array or DataFrame
        Design matrix (should already be numeric, with constant term)
    y_train : array-like
        Response variable
    exposure_train : array-like, optional
        Exposure for offset
    """
    # Ensure numpy array and add constant if needed
    if hasattr(X_train, 'values'):
        X_train = X_train.values
    X_train = np.asarray(X_train, dtype=np.float64)
    
    # Check if constant column already exists (column of all 1s)
    has_constant = np.any(np.all(X_train == 1, axis=0))
    if not has_constant:
        X_train_const = sm.add_constant(X_train)
    else:
        X_train_const = X_train
    
    if exposure_train is not None:
        offset_train = np.log(np.asarray(exposure_train) + 1e-10)
        model = sm.GLM(
            y_train, X_train_const,
            family=sm.families.Poisson(),
            offset=offset_train
        ).fit(maxiter=100, disp=0)
    else:
        model = sm.GLM(
            y_train, X_train_const,
            family=sm.families.Poisson()
        ).fit(maxiter=100, disp=0)
    
    return model


def fit_negative_binomial_for_benchmarks(X_train, y_train, exposure_train=None):
    """
    Fit Negative Binomial (NB2) model for benchmark methods.
    Returns fitted model.
    
    Parameters:
    -----------
    X_train : numpy array or DataFrame
        Design matrix (should already be numeric, with constant term)
    y_train : array-like
        Response variable
    exposure_train : array-like, optional
        Exposure for offset
    """
    # Ensure numpy array and add constant if needed
    if hasattr(X_train, 'values'):
        X_train = X_train.values
    X_train = np.asarray(X_train, dtype=np.float64)
    
    # Check if constant column already exists
    has_constant = np.any(np.all(X_train == 1, axis=0))
    if not has_constant:
        X_train_const = sm.add_constant(X_train)
    else:
        X_train_const = X_train
    
    if exposure_train is not None:
        offset_train = np.log(np.asarray(exposure_train) + 1e-10)
        model = NegativeBinomial(
            y_train, X_train_const,
            loglike_method='nb2',
            offset=offset_train
        ).fit(maxiter=100, disp=0)
    else:
        model = NegativeBinomial(
            y_train, X_train_const,
            loglike_method='nb2'
        ).fit(maxiter=100, disp=0)
    
    return model


# UPDATED BENCHMARK RUNNER

def _build_benchmark_formula(cat_vars, num_vars):
    """
    Build formula string for smf.glm.
    
    Parameters:
    -----------
    cat_vars : list
        Categorical variable names (will be wrapped with C())
    num_vars : list
        Numerical variable names
    
    Returns:
    --------
    str : Formula string for smf.glm (RHS only, without response)
    """
    terms = []
    
    for v in cat_vars:
        terms.append(f"C({v})")
    
    for v in num_vars:
        terms.append(v)
    
    if not terms:
        return "1"  # Intercept only
    
    return " + ".join(terms)


# Global scaler storage for consistent scaling across train/calib/test
_BENCHMARK_SCALERS = {}


def _prepare_benchmark_data(data, cat_vars, num_vars, reference_data=None, fit_scaler=False):
    """
    Prepare DataFrame for formula-based GLM fitting.
    
    Parameters:
    -----------
    data : DataFrame
        Data to prepare
    cat_vars : list
        Categorical variable names
    num_vars : list
        Numerical variable names
    reference_data : DataFrame, optional
        Reference data for aligning category levels (use training data when preparing test/calib)
    fit_scaler : bool
        If True, fit the scaler on this data (use for training data only)
    
    Returns:
    --------
    DataFrame : Prepared data with Claims, Exposure, log_exposure, and all covariates
    """
    global _BENCHMARK_SCALERS
    df = pd.DataFrame({
        'Claims': data['Claims'].values,
        'Exposure': data['Exposure'].values,
        'log_exposure': np.log(data['Exposure'].values + 1e-10)
    }, index=data.index)
    
    # Add categorical variables
    for v in cat_vars:
        if v in data.columns:
            if reference_data is not None and v in reference_data.columns:
                # Align categories with reference data
                ref_categories = reference_data[v].astype(str).unique()
                df[v] = pd.Categorical(
                    data[v].astype(str),
                    categories=ref_categories
                )
            else:
                df[v] = pd.Categorical(data[v].astype(str))
    
    # Add numerical variables with scaling to prevent overflow
    for v in num_vars:
        if v in data.columns:
            values = pd.to_numeric(data[v], errors='coerce').fillna(0).values
            
            if fit_scaler:
                # Fit scaler on training data
                mean_val = np.mean(values)
                std_val = np.std(values)
                if std_val < 1e-10:
                    std_val = 1.0  # Avoid division by zero
                _BENCHMARK_SCALERS[v] = {'mean': mean_val, 'std': std_val}
                df[v] = (values - mean_val) / std_val
            elif v in _BENCHMARK_SCALERS:
                # Use pre-fitted scaler
                mean_val = _BENCHMARK_SCALERS[v]['mean']
                std_val = _BENCHMARK_SCALERS[v]['std']
                df[v] = (values - mean_val) / std_val
            else:
                # No scaler available, use raw values (fallback)
                df[v] = values
    
    return df


def run_parametric_benchmarks(train_data, calib_data, test_data, valid_vars, alpha=0.1):
    """
    Run prediction interval benchmarks from Kim et al. (2022).
    Uses smf.glm() for formula-based GLM fitting.
    
    Returns:
    --------
    dict: Dictionary with benchmark results for both Poisson and NB
    """
    results = {}
    
    # Variables to exclude from modeling (internal/metadata columns)
    exclude_vars = {'pred_mu_true', 'pi_true', 'pred_mu', 'mu_true', 'lambda_true', 
                    'Cluster', 'log_exposure', 'weight', 'index'}
    
    # Separate categorical and numerical variables
    cat_vars = []
    num_vars = []
    for v in valid_vars:
        if v not in train_data.columns:
            continue
        if v in exclude_vars:
            continue
        col = train_data[v]
        if col.dtype.name == 'category' or col.dtype == object or not np.issubdtype(col.dtype, np.number):
            cat_vars.append(v)
        else:
            num_vars.append(v)
    
    print(f"  Categorical variables: {cat_vars}")
    print(f"  Numerical variables: {num_vars}")
    
    # Build formula (RHS only)
    formula_rhs = _build_benchmark_formula(cat_vars, num_vars)
    print(f"  Formula RHS: {formula_rhs}")
    
    # Clear previous scalers and prepare DataFrames for formula API
    global _BENCHMARK_SCALERS
    _BENCHMARK_SCALERS = {}
    
    train_df = _prepare_benchmark_data(train_data, cat_vars, num_vars, fit_scaler=True)
    calib_df = _prepare_benchmark_data(calib_data, cat_vars, num_vars, reference_data=train_data, fit_scaler=False)
    test_df = _prepare_benchmark_data(test_data, cat_vars, num_vars, reference_data=train_data, fit_scaler=False)
    
    # Full formula with response
    formula = f"Claims ~ {formula_rhs}"
    
    y_calib = calib_data['Claims'].values
    exposure_train = train_data['Exposure'].values
    exposure_calib = calib_data['Exposure'].values
    exposure_test = test_data['Exposure'].values
    
    # Import patsy for design matrix extraction
    from patsy import dmatrix
    
    # 1. POISSON BENCHMARKS
    print("Fitting Poisson GLM for benchmarks (smf.glm)...")
    
    try:
        poisson_model = smf.glm(
            formula=formula,
            data=train_df,
            family=sm.families.Poisson(),
            offset=train_df['log_exposure']
        ).fit(maxiter=100, disp=0)
        
        print(f"  Poisson model fitted with {len(poisson_model.params)} parameters")
        
        # Get design matrices from fitted model for interval functions
        X_train_const = poisson_model.model.exog
        design_info = poisson_model.model.data.design_info
        X_calib_const = dmatrix(design_info, calib_df, return_type='dataframe').values
        X_test_const = dmatrix(design_info, test_df, return_type='dataframe').values
        
        print(f"  Design matrix shapes: Train={X_train_const.shape}, Calib={X_calib_const.shape}, Test={X_test_const.shape}")
        
        print("  - Poisson Normal Approximation (Γ̃₁)...")
        results['Poisson_Normal'] = poisson_normal_approximation_interval(
            poisson_model, X_test_const, exposure_test, alpha,
            test_df=test_df, use_formula_api=True
        )
        
        print("  - Poisson Residual Bootstrap (Γ̃₅)...")
        results['Poisson_Bootstrap'] = poisson_residual_bootstrap_interval(
            poisson_model, X_calib_const, y_calib, exposure_calib,
            X_test_const, exposure_test, alpha, n_bootstraps=500,
            calib_df=calib_df, test_df=test_df, use_formula_api=True
        )
        
    except Exception as e:
        print(f"  WARNING: Poisson GLM fitting failed: {e}")
        import traceback
        traceback.print_exc()
    
    # 2. NEGATIVE BINOMIAL BENCHMARKS
    # Use iterative approach for proper α estimation (smf.glm doesn't auto-estimate well)
    print("\nFitting Negative Binomial GLM for benchmarks (iterative α estimation)...")
    try:
        # First fit Poisson to get initial β estimates via formula API
        # Then use iterative approach for NB with proper α estimation
        
        # Use the Poisson design matrices we already have
        if 'X_train_const' in dir() and 'poisson_model' in dir():
            # Use iterative Merz & Wüthrich algorithm for α estimation
            nb_result = fit_nb_glm_iterative_v01(
                X_train_const, train_data['Claims'].values, exposure_train,
                alpha_init=1.0, max_iter=20, tol=1e-6, verbose=True
            )
            
            nb_alpha = nb_result['alpha']
            print(f"  Estimated α = {nb_alpha:.4f} (converged: {nb_result['converged']}, iterations: {nb_result['n_iter']})")
            
            # Compute predictions for calib and test using fitted β
            beta = nb_result['beta']
            eta_calib = X_calib_const @ beta
            eta_test = X_test_const @ beta
            
            mu_pred_calib = np.exp(eta_calib + np.log(exposure_calib))
            mu_pred_test = np.exp(eta_test + np.log(exposure_test))
            
            # Add predictions to result dict
            nb_result['mu_pred_calib'] = mu_pred_calib
            nb_result['mu_pred_test'] = mu_pred_test
            
            print("  - Negative Binomial Chebyshev (Γ̄₄ - Kim et al.)...")
            results['NB_Chebyshev'] = negative_binomial_chebyshev_interval(
                nb_result, X_test_const, exposure_test,
                X_calib_const, y_calib, exposure_calib,
                alpha, disp_param=nb_alpha
            )
            
            print("  - Negative Binomial Parametric Bootstrap (Γ̄₂ - Kim et al.)...")
            results['NB_Bootstrap'] = negative_binomial_parametric_bootstrap_interval(
                nb_result, X_calib_const, y_calib, exposure_calib,
                X_test_const, exposure_test, alpha, n_bootstraps=500,
                include_param_uncertainty=True, disp_param=nb_alpha
            )
            
            print("  - Negative Binomial Plug-in (Γ̄₁ - Kim et al.)...")
            results['NB_Plugin'] = negative_binomial_plug_in_interval(
                nb_result, X_test_const, exposure_test, alpha, disp_param=nb_alpha
            )
            
            # Store fitted NB result for later use
            results['_nb_result'] = nb_result
            results['_nb_alpha'] = nb_alpha
        else:
            print("  WARNING: Poisson model not available for NB initialization")
        
    except Exception as e:
        print(f"  WARNING: Negative Binomial fitting failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Fallback: Use Poisson model with overdispersion adjustment
        if '_poisson_result' in results:
            poisson_pred_test = results['_poisson_result']['mu_pred_test']
            poisson_pred_calib = results['_poisson_result']['mu_pred_calib']
            
            # Estimate dispersion from calibration residuals
            pearson_resid = (y_calib - poisson_pred_calib) / np.sqrt(poisson_pred_calib + 1e-10)
            disp_est = np.var(pearson_resid)
            
            # Simple NB quantile intervals (NOT true prediction intervals)
            n_param = 1.0 / (disp_est + 1e-10)
            p_param = n_param / (n_param + poisson_pred_test + 1e-10)
            
            lower = stats.nbinom.ppf(alpha/2, n_param, p_param)
            upper = stats.nbinom.ppf(1 - alpha/2, n_param, p_param)
            
            results['NB_Fallback'] = pd.DataFrame({
                "pred": poisson_pred_test,
                "lower": np.maximum(lower, 0).astype(int),
                "upper": np.ceil(upper).astype(int)
            })
    
    return results


# COMPREHENSIVE MODEL COMPARISON WITH ALL METRICS

def run_comprehensive_model_comparison(train_data, test_data, cat_vars, num_vars, 
                                        exposure_col='Exposure', target_col='Claims'):
    """
    Run comprehensive model comparison with metrics aligned to 
    glm_preprocess_fit_actuarial_models_with_transformers.ipynb.
    
    Models fitted:
    - Null Model (intercept only)
    - Poisson GLM1 (standard dummy encoding)
    - Poisson GLM2 (+ DrivAge polynomials)
    - Poisson GLM3 (+ interaction terms)
    - Negative Binomial (NB2)
    
    Metrics computed:
    - Poisson Deviance Loss (train & test)
    - Predicted Avg Frequency (train & test)
    - AIC, BIC
    - RMSE
    - Number of parameters
    - Run time
    
    Returns:
    --------
    results_df : DataFrame with all model comparison metrics
    fitted_models : dict with fitted model objects
    predictions : dict with train/test predictions
    """
    print("="*80)
    print("COMPREHENSIVE MODEL COMPARISON")
    print("="*80)
    
    y_train = train_data[target_col].values
    y_test = test_data[target_col].values
    exposure_train = train_data[exposure_col].values
    exposure_test = test_data[exposure_col].values
    exposure_test = exposure_test + 1e-10
    
    n_train = len(train_data)
    n_test = len(test_data)
    
    freq_train = y_train.sum() / exposure_train.sum()
    freq_test = y_test.sum() / exposure_test.sum()
    
    print(f"Train: {n_train:,} obs, Test: {n_test:,} obs")
    print(f"Observed freq - Train: {freq_train:.4%}, Test: {freq_test:.4%}")
    
    results = []
    fitted_models = {}
    predictions = {'train': {}, 'test': {}}
    
    # -------------------------------------------------------------------------
    # 1. NULL MODEL
    # -------------------------------------------------------------------------
    print("\n1. Null Model (Intercept only)...")
    start = time.time()
    null_rate = y_train.sum() / exposure_train.sum()
    run_time = time.time() - start
    
    predictions['train']['Null'] = null_rate * exposure_train
    predictions['test']['Null'] = null_rate * exposure_test
    
    ll_null = poisson_log_likelihood(y_test, predictions['test']['Null'])
    
    results.append({
        'Model': 'Null (Intercept)',
        'N_Params': 1,
        'Run_Time': run_time,
        'Poisson_Dev_Train': poisson_deviance_loss(y_train, predictions['train']['Null']),
        'Poisson_Dev_Test': poisson_deviance_loss(y_test, predictions['test']['Null']),
        'Pred_Freq_Train': null_rate,
        'Pred_Freq_Test': null_rate,
        'AIC': compute_aic(1, ll_null),
        'BIC': compute_bic(1, ll_null, n_test),
        'RMSE': np.sqrt(np.mean((y_test - predictions['test']['Null'])**2))
    })
    
    # -------------------------------------------------------------------------
    # 2. GLM1 - Standard dummy encoding
    # -------------------------------------------------------------------------
    print("\n2. Poisson GLM1 (standard encoding)...")
    try:
        X_train_glm1, X_test_glm1, _ = prepare_glm_design_matrices(
            train_data, test_data, cat_vars, num_vars,
            add_driv_age_polynomials=False, add_interaction_terms=False
        )
        
        start = time.time()
        glm1 = sm.GLM(y_train, X_train_glm1, family=sm.families.Poisson(),
                      offset=np.log(exposure_train + 1e-10)).fit(disp=0)
        run_time = time.time() - start
        
        predictions['train']['GLM1'] = glm1.predict(X_train_glm1) * exposure_train
        predictions['test']['GLM1'] = glm1.predict(X_test_glm1) * exposure_test
        fitted_models['GLM1'] = glm1
        
        results.append({
            'Model': 'Poisson GLM1',
            'N_Params': len(glm1.params),
            'Run_Time': run_time,
            'Poisson_Dev_Train': poisson_deviance_loss(y_train, predictions['train']['GLM1']),
            'Poisson_Dev_Test': poisson_deviance_loss(y_test, predictions['test']['GLM1']),
            'Pred_Freq_Train': predictions['train']['GLM1'].sum() / exposure_train.sum(),
            'Pred_Freq_Test': predictions['test']['GLM1'].sum() / exposure_test.sum(),
            'AIC': glm1.aic,
            'BIC': glm1.bic,
            'RMSE': np.sqrt(np.mean((y_test - predictions['test']['GLM1'])**2))
        })
        print(f"   GLM1: {len(glm1.params)} params, AIC={glm1.aic:,.0f}")
        
    except Exception as e:
        print(f"   WARNING: GLM1 error: {e}")
    
    # -------------------------------------------------------------------------
    # 3. GLM2 - With DrivAge polynomials
    # -------------------------------------------------------------------------
    print("\n3. Poisson GLM2 (+ DrivAge polynomials)...")
    try:
        X_train_glm2, X_test_glm2, _ = prepare_glm_design_matrices(
            train_data, test_data, cat_vars, num_vars,
            add_driv_age_polynomials=True, add_interaction_terms=False
        )
        
        start = time.time()
        glm2 = sm.GLM(y_train, X_train_glm2, family=sm.families.Poisson(),
                      offset=np.log(exposure_train + 1e-10)).fit(disp=0)
        run_time = time.time() - start
        
        predictions['train']['GLM2'] = glm2.predict(X_train_glm2) * exposure_train
        predictions['test']['GLM2'] = glm2.predict(X_test_glm2) * exposure_test
        fitted_models['GLM2'] = glm2
        
        results.append({
            'Model': 'Poisson GLM2',
            'N_Params': len(glm2.params),
            'Run_Time': run_time,
            'Poisson_Dev_Train': poisson_deviance_loss(y_train, predictions['train']['GLM2']),
            'Poisson_Dev_Test': poisson_deviance_loss(y_test, predictions['test']['GLM2']),
            'Pred_Freq_Train': predictions['train']['GLM2'].sum() / exposure_train.sum(),
            'Pred_Freq_Test': predictions['test']['GLM2'].sum() / exposure_test.sum(),
            'AIC': glm2.aic,
            'BIC': glm2.bic,
            'RMSE': np.sqrt(np.mean((y_test - predictions['test']['GLM2'])**2))
        })
        print(f"   GLM2: {len(glm2.params)} params, AIC={glm2.aic:,.0f}")
        
    except Exception as e:
        print(f"   WARNING: GLM2 error: {e}")
    
    # -------------------------------------------------------------------------
    # 4. GLM3 - With interactions
    # -------------------------------------------------------------------------
    print("\n4. Poisson GLM3 (+ interactions)...")
    try:
        X_train_glm3, X_test_glm3, _ = prepare_glm_design_matrices(
            train_data, test_data, cat_vars, num_vars,
            add_driv_age_polynomials=True, add_interaction_terms=True
        )
        
        start = time.time()
        glm3 = sm.GLM(y_train, X_train_glm3, family=sm.families.Poisson(),
                      offset=np.log(exposure_train + 1e-10)).fit(disp=0)
        run_time = time.time() - start
        
        predictions['train']['GLM3'] = glm3.predict(X_train_glm3) * exposure_train
        predictions['test']['GLM3'] = glm3.predict(X_test_glm3) * exposure_test
        fitted_models['GLM3'] = glm3
        
        results.append({
            'Model': 'Poisson GLM3',
            'N_Params': len(glm3.params),
            'Run_Time': run_time,
            'Poisson_Dev_Train': poisson_deviance_loss(y_train, predictions['train']['GLM3']),
            'Poisson_Dev_Test': poisson_deviance_loss(y_test, predictions['test']['GLM3']),
            'Pred_Freq_Train': predictions['train']['GLM3'].sum() / exposure_train.sum(),
            'Pred_Freq_Test': predictions['test']['GLM3'].sum() / exposure_test.sum(),
            'AIC': glm3.aic,
            'BIC': glm3.bic,
            'RMSE': np.sqrt(np.mean((y_test - predictions['test']['GLM3'])**2))
        })
        print(f"   GLM3: {len(glm3.params)} params, AIC={glm3.aic:,.0f}")
        
    except Exception as e:
        print(f"   WARNING: GLM3 error: {e}")
    
    # -------------------------------------------------------------------------
    # 5. NEGATIVE BINOMIAL (NB2) using statsmodels
    # -------------------------------------------------------------------------
    print("\n5. Negative Binomial (NB2) using statsmodels...")
    try:
        # Use GLM1 design matrix
        start = time.time()
        
        # Use statsmodels NegativeBinomial directly
        offset_train = np.log(exposure_train + 1e-10)
        nb_model = NegativeBinomial(
            y_train, X_train_glm1, 
            loglike_method='nb2',
            offset=offset_train
        ).fit(maxiter=200, disp=0, method='nm')
        run_time = time.time() - start
        
        # Extract alpha from statsmodels NB model with robust handling
        nb_alpha = None
        
        if hasattr(nb_model, 'alpha'):
            nb_alpha = nb_model.alpha
        
        # Check if alpha is valid (not inf, nan, or too large)
        if nb_alpha is None or not np.isfinite(nb_alpha) or nb_alpha > 100:
            # Fallback: try to extract from params
            n_betas = X_train_glm1.shape[1]
            if len(nb_model.params) > n_betas:
                log_alpha = nb_model.params[-1]
                if np.isfinite(log_alpha) and log_alpha < 10:
                    nb_alpha = np.exp(log_alpha)
        
        # If still invalid, estimate from Pearson residuals
        if nb_alpha is None or not np.isfinite(nb_alpha) or nb_alpha > 100:
            mu_train = nb_model.predict(X_train_glm1, offset=offset_train)
            pearson_resid = (y_train - mu_train) / np.sqrt(mu_train + 1e-10)
            phi = np.sum(pearson_resid**2) / (len(y_train) - X_train_glm1.shape[1])
            
            if phi > 1.05:
                mu_bar = np.mean(mu_train)
                nb_alpha = max(0.1, min((phi - 1) / (mu_bar + 1e-10), 10.0))
            else:
                nb_alpha = 0.1
                print("  Note: No significant overdispersion detected, using α=0.1")
        
        # Get predictions
        offset_test = np.log(exposure_test + 1e-10)
        predictions['train']['NB'] = nb_model.predict(X_train_glm1, offset=offset_train) * exposure_train
        predictions['test']['NB'] = nb_model.predict(X_test_glm1, offset=offset_test) * exposure_test
        fitted_models['NB'] = nb_model
        
        # Store alpha for later use
        fitted_models['_nb_alpha'] = nb_alpha
        
        # NB-specific deviance
        nb_dev_train = nb_deviance_loss(y_train, predictions['train']['NB'], nb_alpha)
        nb_dev_test = nb_deviance_loss(y_test, predictions['test']['NB'], nb_alpha)
        
        # Number of parameters (betas + alpha)
        n_params_nb = len(nb_model.params)
        
        results.append({
            'Model': f'Negative Binomial',
            'N_Params': n_params_nb,
            'Run_Time': run_time,
            'Poisson_Dev_Train': poisson_deviance_loss(y_train, predictions['train']['NB']),
            'Poisson_Dev_Test': poisson_deviance_loss(y_test, predictions['test']['NB']),
            'NB_Dev_Train': nb_dev_train,
            'NB_Dev_Test': nb_dev_test,
            'Pred_Freq_Train': predictions['train']['NB'].sum() / exposure_train.sum(),
            'Pred_Freq_Test': predictions['test']['NB'].sum() / exposure_test.sum(),
            'AIC': nb_model.aic,
            'BIC': nb_model.bic,
            'RMSE': np.sqrt(np.mean((y_test - predictions['test']['NB'])**2)),
            'Alpha': nb_alpha
        })
        print(f"   NB2: {n_params_nb} params, AIC={nb_model.aic:,.0f}, α={nb_alpha:.4f}")
        
    except Exception as e:
        print(f"   WARNING: Negative Binomial fitting failed: {e}")
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    print("\n" + "="*80)
    print("MODEL COMPARISON SUMMARY")
    print("="*80)
    print(f"\nObserved Frequencies: Train={freq_train:.4%}, Test={freq_test:.4%}")
    print(results_df.to_string(index=False))
    
    return results_df, fitted_models, predictions


def generate_comparison_latex_table(results_df, obs_freq_train, obs_freq_test, output_path=None):
    """
    Generate a LaTeX table from model comparison results.
    
    Parameters:
    -----------
    results_df : DataFrame from run_comprehensive_model_comparison
    obs_freq_train, obs_freq_test : observed frequencies
    output_path : optional path to save the LaTeX file
    
    Returns:
    --------
    latex_str : LaTeX table as string
    """
    latex_table = r"""
\begin{table}[htbp]
\centering
\caption{Model Comparison: Predictive Performance}
\label{tab:model_comparison}
\small
\begin{tabular}{lcccccccc}
\toprule
\textbf{Model} & \textbf{\# Params} & \textbf{Dev Train} & \textbf{Dev Test} & \textbf{Freq Train} & \textbf{Freq Test} & \textbf{AIC} & \textbf{RMSE} \\
\midrule
"""
    
    for _, row in results_df.iterrows():
        latex_table += f"{row['Model']} & {row['N_Params']:,} & {row['Poisson_Dev_Train']:.4f} & {row['Poisson_Dev_Test']:.4f} & {row['Pred_Freq_Train']:.4f} & {row['Pred_Freq_Test']:.4f} & {row['AIC']:,.0f} & {row['RMSE']:.4f} \\\\\n"
    
    latex_table += r"""\midrule
\multicolumn{8}{l}{\textit{Note: Observed frequency: Train=""" + f"{obs_freq_train:.4f}" + r""", Test=""" + f"{obs_freq_test:.4f}" + r"""}} \\
\bottomrule
\end{tabular}
\end{table}
"""
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(latex_table)
    
    return latex_table


def check_overdispersion(y_obs, mu_pred, exposure=None):
    """Check for overdispersion in count data."""
    if exposure is not None:
        y_rate = y_obs / exposure
        mu_rate = mu_pred / exposure
    else:
        y_rate = y_obs
        mu_rate = mu_pred

    # Pearson dispersion statistic
    pearson_resid = (y_obs - mu_pred) / np.sqrt(mu_pred + 1e-10)
    phi_pearson = np.sum(pearson_resid**2) / (len(y_obs) - 1)

    # Deviance dispersion
    with np.errstate(divide='ignore'):
        deviance = 2 * np.sum(y_obs * np.log(y_obs/(mu_pred + 1e-10)) - (y_obs - mu_pred))
    phi_deviance = deviance / (len(y_obs) - 1)

    print(f"  Overdispersion diagnostics:")
    print(f"    Pearson φ = {phi_pearson:.4f} (φ > 1 indicates overdispersion)")
    print(f"    Deviance φ = {phi_deviance:.4f}")
    print(f"    Variance/Mean ratio = {np.var(y_obs)/np.mean(y_obs):.4f}")

    return phi_pearson > 1.5  # Conservative threshold