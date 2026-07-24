import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import toeplitz
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class EnhancedCountDGP:
    """Enhanced DGP for insurance claim count simulation"""
    
    def __init__(self, seed=42):
        self.seed = seed
        np.random.seed(seed)
    
    def generate_covariates(self, n_obs, dim_x, correlation_type='toeplitz', 
                          categorical_ratio=0.3, correlation_strength=0.5):
        """
        Generate realistic insurance covariates:
        - Policyholder characteristics (age, vehicle age, etc.)
        - Geographic regions
        - Policy features (coverage type, deductible)
        """
        n_continuous = int(dim_x * (1 - categorical_ratio))
        n_categorical = dim_x - n_continuous
        
        # Continuous covariates with realistic distributions
        X_cont = np.zeros((n_obs, n_continuous))
        
        # Age (truncated normal)
        X_cont[:, 0] = np.clip(np.random.normal(45, 15, n_obs), 18, 90)
        
        # Vehicle age (exponential decay)
        X_cont[:, 1] = np.random.exponential(8, n_obs)
        
        # Policy duration (uniform)
        X_cont[:, 2] = np.random.uniform(0, 20, n_obs)
        
        # Remaining continuous variables with correlation
        if n_continuous > 3:
            # Create correlation structure
            if correlation_type == 'toeplitz':
                corr = toeplitz([correlation_strength**i for i in range(n_continuous-3)])
            elif correlation_type == 'block':
                corr = np.eye(n_continuous-3) * 0.7 + 0.3
            else:
                corr = np.eye(n_continuous-3)
            
            # Generate correlated covariates
            means = np.random.uniform(-1, 1, n_continuous-3)
            X_cont[:, 3:] = np.random.multivariate_normal(means, corr, n_obs)
        
        # Categorical covariates
        cat_data = {}
        for i in range(n_categorical):
            if i == 0:  # Region
                n_levels = 5
                probs = [0.3, 0.25, 0.2, 0.15, 0.1]
                cat_data[f'Region'] = np.random.choice(n_levels, size=n_obs, p=probs)
            elif i == 1:  # Coverage type
                n_levels = 3
                cat_data[f'Coverage'] = np.random.choice(n_levels, size=n_obs, p=[0.5, 0.3, 0.2])
            else:
                n_levels = np.random.choice([2, 3, 4])
                cat_data[f'Cat_{i}'] = np.random.choice(n_levels, size=n_obs)
        
        # Combine into DataFrame
        df = pd.DataFrame(X_cont, columns=[f'X_cont_{i}' for i in range(n_continuous)])
        for name, values in cat_data.items():
            df[name] = pd.Categorical(values)
        
        return df
    
    def generate_exposure(self, n_obs, exposure_type='realistic'):
        """Generate realistic exposure measures"""
        if exposure_type == 'realistic':
            # Mixture: most policies are annual, some are partial years
            is_annual = np.random.binomial(1, 0.8, n_obs)
            annual_exposure = np.ones(n_obs)
            partial_exposure = np.random.beta(2, 5, n_obs)  # Skewed toward shorter exposures
            
            exposure = np.where(is_annual == 1, annual_exposure, partial_exposure)
            exposure = np.clip(exposure, 0.1, 1.0)
        else:
            exposure = np.ones(n_obs)
        
        return exposure
    
    def generate_claims(self, df, exposure, dgp_type='poisson', 
                       zero_inflation=0.0, overdispersion=1.0,
                       interaction_effects=True, spatial_effects=False):
        """
        Generate claim counts with enhanced realism
        """
        n_obs = len(df)
        
        # Create design matrix (numeric) - ensure float64 numpy array
        X_numeric = pd.get_dummies(df, drop_first=True).values.astype(np.float64)
        
        # Generate coefficients with realistic structure
        n_features = X_numeric.shape[1]
        beta = np.zeros(n_features, dtype=np.float64)
        
        # Strong effects for key variables (age, region, etc.)
        key_vars = min(5, n_features)
        beta[:key_vars] = np.random.normal(0.5, 0.2, key_vars)
        
        # Weak effects for remaining variables
        if n_features > key_vars:
            beta[key_vars:] = np.random.normal(0, 0.1, n_features - key_vars)
        
        # Add interaction effects if requested
        if interaction_effects and n_features >= 3:
            # Age × Vehicle age interaction
            age_idx = 0 if 'X_cont_0' in df.columns else None
            veh_age_idx = 1 if 'X_cont_1' in df.columns else None
            
            if age_idx is not None and veh_age_idx is not None:
                interaction = df.iloc[:, age_idx] * df.iloc[:, veh_age_idx]
                X_numeric = np.column_stack([X_numeric, interaction.values])
                beta = np.append(beta, np.random.normal(0.2, 0.05))
        
        # Compute linear predictor
        eta = np.dot(X_numeric, beta)
        eta = np.asarray(eta, dtype=np.float64)
        
        # CRITICAL: Clip eta to prevent exp() overflow
        eta = np.clip(eta, -10, 3)  # Keeps mu in reasonable range [~0, ~20]
        mu_base = np.exp(eta) * exposure
        mu_base = np.clip(mu_base, 1e-6, 50)  # Ensure valid Poisson lambda
        
        # Apply DGP-specific modifications
        if dgp_type == 'poisson':
            mu = mu_base
            y = np.random.poisson(mu)
        
        elif dgp_type == 'zip':
            # Zero-inflation probability depends on covariates - handle small feature dimensions
            n_zi_features = min(3, n_features)
            zi_beta = np.random.normal(0.5, 0.1, n_zi_features).astype(np.float64)
            pi = 1.0 / (1.0 + np.exp(-np.dot(X_numeric[:, :n_zi_features], zi_beta)))
            pi = np.clip(zero_inflation * pi / (np.mean(pi) + 1e-10), 0, 1)
            
            structural_zero = np.random.binomial(1, pi)
            poisson_counts = np.random.poisson(mu_base)
            y = np.where(structural_zero == 1, 0, poisson_counts)
            mu = mu_base * (1 - pi)  # Adjusted mean
        
        elif dgp_type == 'negbin':
            theta = max(overdispersion, 0.1)  # Ensure positive theta
            p = theta / (theta + mu_base)
            p = np.clip(p, 1e-10, 1 - 1e-10)  # Ensure valid probability
            y = np.random.negative_binomial(theta, p)
            mu = mu_base
        
        elif dgp_type == 'hurdle':
            # Hurdle probability - handle small feature dimensions
            n_hurdle_features = min(2, n_features)
            hurdle_beta = np.random.normal(-0.5, 0.2, n_hurdle_features).astype(np.float64)
            pi = 1.0 / (1.0 + np.exp(-np.dot(X_numeric[:, :n_hurdle_features], hurdle_beta)))
            pi = np.clip(zero_inflation * pi / (np.mean(pi) + 1e-10), 0, 0.99)
            
            cross_hurdle = np.random.binomial(1, 1 - pi)
            y = np.zeros(n_obs, dtype=int)
            
            # Truncated Poisson for positive counts with max iteration protection
            for i in np.where(cross_hurdle == 1)[0]:
                count = 0
                while count < 100:
                    sample = np.random.poisson(mu_base[i])
                    if sample > 0:
                        y[i] = sample
                        break
                    count += 1
                if count >= 100:
                    y[i] = 1  # Fallback
            
            mu = mu_base * (1 - pi) / (1 - np.exp(-mu_base) + 1e-10)
        
        # Add spatial effects if requested
        if spatial_effects:
            spatial_risk = np.zeros(n_obs)
            if 'Region' in df.columns:
                # Higher risk in certain regions
                region_effects = np.random.normal(0, 0.3, df['Region'].nunique())
                for r in range(df['Region'].nunique()):
                    mask = (df['Region'] == r).values
                    spatial_risk[mask] = region_effects[r]
            
            # Fix: Ensure lambda is non-negative by using max(0, exp(spatial_risk) - 1)
            # or use a different formulation: additional_claims ~ Poisson(max(0, exp(spatial_risk) - 1))
            spatial_lambda = np.maximum(0, np.exp(spatial_risk) - 1)
            y = y + np.random.poisson(spatial_lambda)
        
        return y, mu, {
            'beta': beta,
            'zero_inflation': zero_inflation if dgp_type in ['zip', 'hurdle'] else 0.0,
            'overdispersion': overdispersion if dgp_type == 'negbin' else np.nan,
            'true_mean': mu,
            'exposure': exposure
        }


# DATA SPLITTING UTILITIES

def split_data(
    dgp_output: Dict[str, Any],
    train_frac: float = 0.6,
    calib_frac: float = 0.2,
    seed: int = None
) -> Dict[str, pd.DataFrame]:
    """
    Split DGP output into train/calibration/test sets.
    
    Parameters:
    -----------
    dgp_output : dict
        Output from a DGP function
    train_frac : float
        Fraction for training (default 0.6)
    calib_frac : float
        Fraction for calibration (default 0.2)
        Test fraction = 1 - train_frac - calib_frac
    seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    dict with keys: 'train', 'calib', 'test' containing DataFrames
    """
    if seed is not None:
        np.random.seed(seed)
    
    df = dgp_output['df'].copy()
    n = len(df)
    
    # Create random split indices
    indices = np.random.permutation(n)
    
    n_train = int(n * train_frac)
    n_calib = int(n * calib_frac)
    
    train_idx = indices[:n_train]
    calib_idx = indices[n_train:n_train + n_calib]
    test_idx = indices[n_train + n_calib:]
    
    return {
        'train': df.iloc[train_idx].reset_index(drop=True),
        'calib': df.iloc[calib_idx].reset_index(drop=True),
        'test': df.iloc[test_idx].reset_index(drop=True),
        'params': dgp_output['params']
    }

# SUMMARY STATISTICS

def summarize_dgp_output(dgp_output: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute summary statistics for DGP output.
    """
    y = dgp_output['y']
    mu = dgp_output['mu']
    
    summary = {
        'n_obs': len(y),
        'mean_y': np.mean(y),
        'var_y': np.var(y),
        'pct_zeros': 100 * np.mean(y == 0),
        'pct_ones': 100 * np.mean(y == 1),
        'pct_two_plus': 100 * np.mean(y >= 2),
        'max_y': np.max(y),
        'mean_mu': np.mean(mu),
        'var_mu': np.var(mu),
        'dispersion_ratio': np.var(y) / np.mean(y) if np.mean(y) > 0 else np.nan,
    }
    
    if 'pi' in dgp_output:
        summary['mean_zero_inflation'] = np.mean(dgp_output['pi'])
    
    return summary


# TESTING

if __name__ == "__main__":
    print("Testing DGPs for Count Data Simulation Study")
    print("=" * 60)
    
    # Test each DGP
    for dgp_type in ['poisson', 'zip', 'negbin', 'hurdle']:
        print(f"\nDGP: {dgp_type.upper()}")
        print("-" * 40)
        
        dgp_output = dgp_count_wrapper(
            dgp_type=dgp_type,
            n_obs=5000,
            dim_x=10,
            rho=0.5,
            sparsity=0.3,
            effect_size=0.5,
            base_rate=0.1,
            zero_inflation=0.3,
            overdispersion=2.0,
            seed=42
        )
        
        summary = summarize_dgp_output(dgp_output)
        
        for key, value in summary.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
    
    print("\n" + "=" * 60)
    print("DGP module loaded successfully!")
