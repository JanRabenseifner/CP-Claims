import itertools
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OrdinalEncoder

from .config import DGCP_METHOD_TYPES, LEARNERS, MIN_GROUP_SIZES
from .utils_benchmarks import run_parametric_benchmarks
from .utils_dgcp import two_stage_dgcp_cp
from .utils_dgps import EnhancedCountDGP, split_data, summarize_dgp_output
from .utils_eval import interval_score
from .utils_pinterval import (
    binary_classification_cp,
    discrete_conformal_poisson,
    mondrian_conformal_poisson,
)
from .utils_two_stage import (
    two_stage_balanced_v2_cp,
    two_stage_calibrated_cp,
    two_stage_direct_cp,
    two_stage_empirical_cp,
    two_stage_exact_cp,
    two_stage_weighted_cp,
)

try:
    from lightgbm import LGBMRegressor
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

warnings.filterwarnings("ignore")


@dataclass(frozen=True)
class SimulationConfig:
    dgp_type: str
    n_obs: int
    dim_x: int
    correlation_strength: float
    categorical_ratio: float
    interaction_effects: bool
    spatial_effects: bool
    zero_inflation: float
    overdispersion: float
    repetition: int
    seed: int
    exposure_type: str = "realistic"
    correlation_type: str = "toeplitz"


DEFAULT_PARAM_GRID = {
    "dgp_type": ["poisson", "zip", "negbin", "hurdle"],
    "n_obs": [1000, 2000, 4000, 6000, 10000],
    "dim_x": [10, 20, 50],
    "correlation_strength": [0.0, 0.3, 0.5],
    "categorical_ratio": [0.1, 0.3, 0.5],
    "interaction_effects": [False, True],
    "spatial_effects": [False, True],
}


def generate_configs(
    param_grid: dict | None = None,
    n_reps: int = 96,
    base_seed: int = 42,
    exposure_type: str = "realistic",
    correlation_type: str = "toeplitz",
) -> list[SimulationConfig]:
    grid = param_grid or DEFAULT_PARAM_GRID
    configs: list[SimulationConfig] = []
    keys = list(grid.keys())
    for rep in range(n_reps):
        for index, values in enumerate(itertools.product(*(grid[k] for k in keys))):
            config = dict(zip(keys, values))
            zi_values = [0.0, 0.1, 0.3, 0.5]
            od_values = [2.0, 3.0, 4.0, 5.0]
            config["zero_inflation"] = zi_values[index % len(zi_values)] if config["dgp_type"] in {"zip", "hurdle"} else 0.0
            config["overdispersion"] = od_values[index % len(od_values)] if config["dgp_type"] == "negbin" else 1.0
            config["repetition"] = rep
            config["seed"] = base_seed + rep * 1000 + index
            config["exposure_type"] = exposure_type
            config["correlation_type"] = correlation_type
            configs.append(SimulationConfig(**config))
    return configs


def _fit_poisson_glm(train_data: pd.DataFrame, valid_vars: list[str]):
    design = pd.get_dummies(train_data[valid_vars], drop_first=True, dtype=float)
    design = sm.add_constant(design)
    model = sm.GLM(train_data["Claims"], design, family=sm.families.Poisson(), exposure=train_data["Exposure"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            result = model.fit(maxiter=100)
        except Exception:
            result = model.fit_regularized(alpha=0.01)

    def predict(data_subset: pd.DataFrame) -> np.ndarray:
        x = pd.get_dummies(data_subset[valid_vars], drop_first=True, dtype=float)
        x = sm.add_constant(x)
        for col in result.model.exog_names:
            if col not in x.columns:
                x[col] = 0
        x = x[result.model.exog_names]
        return np.clip(result.predict(x) * data_subset["Exposure"].values, 1e-6, 100)

    return predict


def _fit_lgbm(train_data: pd.DataFrame, valid_vars: list[str]):
    if not LGBM_AVAILABLE:
        raise ImportError("lightgbm is not installed")
    num_cols = [c for c in valid_vars if np.issubdtype(train_data[c].dtype, np.number)]
    cat_cols = [c for c in valid_vars if c not in num_cols]
    transformers = [("num", "passthrough", num_cols)]
    if cat_cols:
        transformers.append(("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols))
    preprocessor = ColumnTransformer(transformers=transformers, verbose_feature_names_out=False).set_output(transform="pandas")
    x_train = preprocessor.fit_transform(train_data[valid_vars])
    y_train = train_data["Claims"] / train_data["Exposure"]
    model = LGBMRegressor(
        objective="poisson", learning_rate=0.05, num_leaves=31,
        reg_lambda=2.0, reg_alpha=2.0, n_estimators=200,
        random_state=42, verbosity=-1,
    )
    model.fit(x_train, y_train, sample_weight=train_data["Exposure"])

    def predict(data_subset: pd.DataFrame) -> np.ndarray:
        x = preprocessor.transform(data_subset[valid_vars])
        return np.clip(model.predict(x) * data_subset["Exposure"].values, 1e-6, 100)

    return predict


def _fit_random_forest(train_data: pd.DataFrame, valid_vars: list[str]):
    num_cols = [c for c in valid_vars if np.issubdtype(train_data[c].dtype, np.number)]
    cat_cols = [c for c in valid_vars if c not in num_cols]
    transformers = [("num", "passthrough", num_cols)]
    if cat_cols:
        transformers.append(("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols))
    preprocessor = ColumnTransformer(transformers=transformers, verbose_feature_names_out=False).set_output(transform="pandas")
    x_train = preprocessor.fit_transform(train_data[valid_vars])
    y_train = train_data["Claims"] / train_data["Exposure"]
    model = RandomForestRegressor(
        criterion="poisson", n_estimators=100, max_features="sqrt",
        max_depth=10, min_samples_leaf=20, random_state=42, n_jobs=1,
    )
    model.fit(x_train, y_train, sample_weight=train_data["Exposure"])

    def predict(data_subset: pd.DataFrame) -> np.ndarray:
        x = preprocessor.transform(data_subset[valid_vars])
        return np.clip(model.predict(x) * data_subset["Exposure"].values, 1e-6, 100)

    return predict


LEARNER_FUNCTIONS = {
    "Poisson_GLM": _fit_poisson_glm,
    "LightGBM": _fit_lgbm,
    "RandomForest": _fit_random_forest,
}


def compute_metrics(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray,
                    pred_mu: np.ndarray, alpha: float = 0.1) -> dict[str, float]:
    covered = (y_true >= lower) & (y_true <= upper)
    widths = upper - lower
    metrics: dict[str, float] = {
        "coverage": float(np.mean(covered)),
        "mean_width": float(np.mean(widths)),
        "median_width": float(np.median(widths)),
        "width_std": float(np.std(widths)),
        "interval_score": float(interval_score(y_true, lower, upper, alpha)),
        "pct_singleton": float(100 * np.mean(widths == 0)),
    }
    groups = {
        "coverage_y0": y_true == 0,
        "coverage_y1": y_true == 1,
        "coverage_y2plus": y_true >= 2,
    }
    min_obs = 5
    for label, mask in groups.items():
        metrics[label] = float(np.mean(covered[mask])) if mask.sum() >= min_obs else np.nan
        n_label = f"n_{label.replace('coverage_', 'y')}"
        metrics[n_label] = int(mask.sum()) if mask.sum() >= min_obs else 0
    metrics["coverage_gap"] = abs(metrics["coverage_y0"] - metrics["coverage_y1"]) if not np.isnan(metrics["coverage_y0"]) and not np.isnan(metrics["coverage_y1"]) else np.nan

    bin_edges = [(1, 5), (6, 10), (11, 20), (21, 50), (51, np.inf)]
    max_y = int(np.max(y_true)) if len(y_true) > 0 else 0
    for low, high in bin_edges:
        if max_y < low:
            continue
        if np.isinf(high):
            mask = y_true >= low
            name = f"coverage_y{int(low)}plus"
        else:
            mask = (y_true >= low) & (y_true <= high)
            name = f"coverage_y{int(low)}_{int(high)}"
        if mask.any():
            metrics[name] = float(np.mean(covered[mask]))

    if len(pred_mu) > 1 and np.std(widths) > 0:
        metrics["risk_differentiation"] = float(np.corrcoef(pred_mu, widths)[0, 1])
    else:
        metrics["risk_differentiation"] = np.nan

    for level in [0.95, 0.99]:
        threshold = np.percentile(y_true, level * 100)
        mask = y_true >= threshold
        if mask.any():
            metrics[f"var_{int(level * 100)}_coverage"] = float(np.mean(covered[mask]))

    tail_mask = y_true >= np.percentile(y_true, 90)
    if tail_mask.any():
        metrics["tail_mean_width"] = float(np.mean(widths[tail_mask]))
        metrics["tail_coverage"] = float(np.mean(covered[tail_mask]))

    informative = (lower > 0) | (upper < np.inf)
    metrics["pct_informative"] = float(100 * np.mean(informative))

    if "pred_mu" in metrics or len(pred_mu) > 10:
        risk_deciles = pd.qcut(pred_mu, q=10, duplicates="drop", labels=False)
        calibration_by_risk = []
        for decile in np.unique(risk_deciles):
            decile_mask = risk_deciles == decile
            if decile_mask.any():
                calibration_by_risk.append(np.mean(covered[decile_mask]))
        metrics["calibration_range"] = float(np.max(calibration_by_risk) - np.min(calibration_by_risk)) if calibration_by_risk else np.nan

    return metrics


def run_single_simulation(
    config: SimulationConfig,
    alpha: float = 0.1,
    learners: list[str] | None = None,
    dgcp_methods: list[str] | None = None,
    min_group_sizes: list[int] | None = None,
    include_two_stage: bool = False,
    include_binary_mondrian: bool = True,
) -> list[dict]:
    learner_list = learners or LEARNERS
    dgcp_method_list = dgcp_methods or DGCP_METHOD_TYPES
    min_size_list = min_group_sizes or MIN_GROUP_SIZES

    dgp = EnhancedCountDGP(seed=config.seed)
    df = dgp.generate_covariates(
        n_obs=config.n_obs, dim_x=config.dim_x,
        correlation_type=config.correlation_type,
        categorical_ratio=config.categorical_ratio,
        correlation_strength=config.correlation_strength,
    )
    exposure = dgp.generate_exposure(n_obs=config.n_obs, exposure_type=config.exposure_type)
    y, mu_true, _ = dgp.generate_claims(
        df=df, exposure=exposure,
        dgp_type=config.dgp_type,
        zero_inflation=config.zero_inflation,
        overdispersion=config.overdispersion,
        interaction_effects=config.interaction_effects,
        spatial_effects=config.spatial_effects,
    )
    df["Claims"] = y
    df["Exposure"] = exposure
    df["pred_mu_true"] = mu_true

    splits = split_data({"df": df, "params": asdict(config)}, train_frac=0.6, calib_frac=0.2, seed=config.seed)
    train_data = splits["train"]
    calib_data = splits["calib"]
    test_data = splits["test"]

    from sklearn.cluster import KMeans

    def generate_clusters(data, n_clusters=5):
        cluster_features = []
        for col in data.columns:
            if col.startswith("Region") or col.startswith("Coverage") or "age" in col.lower():
                cluster_features.append(col)
        if not cluster_features:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                cluster_features = [numeric_cols[0]]
            else:
                return np.ones(len(data))
        cluster_data = pd.get_dummies(data[cluster_features], drop_first=True).values
        n_clusters = min(n_clusters, len(cluster_data) // 10, 10)
        if len(cluster_data) >= n_clusters and n_clusters > 0:
            return KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(cluster_data) + 1
        return np.ones(len(data))

    train_data["Cluster"] = generate_clusters(train_data)
    calib_data["Cluster"] = generate_clusters(calib_data)
    test_data["Cluster"] = generate_clusters(test_data)

    valid_vars = [c for c in train_data.columns if c not in {"Claims", "Exposure", "pred_mu_true", "pi_true", "Cluster"}]
    dgp_summary = summarize_dgp_output({"y": y, "mu": mu_true})
    results: list[dict] = []

    for learner_name in learner_list:
        if learner_name == "LightGBM" and not LGBM_AVAILABLE:
            continue
        try:
            predict_func = LEARNER_FUNCTIONS[learner_name](train_data, valid_vars)
        except Exception as exc:
            results.append({"learner": learner_name, "error": str(exc), **asdict(config)})
            continue

        calib_copy = calib_data.copy()
        test_copy = test_data.copy()
        calib_copy["pred_mu"] = predict_func(calib_copy)
        test_copy["pred_mu"] = predict_func(test_copy)
        pred_mu = test_copy["pred_mu"].values
        calib_mu = calib_copy["pred_mu"].values
        calib_y = calib_copy["Claims"].values
        test_y = test_copy["Claims"].values
        calib_cluster = calib_copy["Cluster"].values
        test_cluster = test_copy["Cluster"].values

        method_outputs: dict[str, object] = {}

        # Standard ICP methods
        try:
            method_outputs["ICP_poisson_deviance"] = discrete_conformal_poisson(pred_mu, calib_mu, calib_y, alpha, "poisson_deviance")
        except Exception:
            method_outputs["ICP_poisson_deviance"] = None
        try:
            method_outputs["ICP_za_relative"] = discrete_conformal_poisson(pred_mu, calib_mu, calib_y, alpha, "za_relative")
        except Exception:
            method_outputs["ICP_za_relative"] = None
        try:
            method_outputs["Mondrian_Poisson"] = mondrian_conformal_poisson(
                pred_mu, test_cluster, calib_mu, calib_y, calib_cluster, alpha, "poisson_deviance")
        except Exception:
            method_outputs["Mondrian_Poisson"] = None

        # Binary CP methods
        calib_binary = (calib_y > 0).astype(int)
        calib_probs = np.column_stack([np.exp(-calib_mu), 1 - np.exp(-calib_mu)])
        test_probs = np.column_stack([np.exp(-pred_mu), 1 - np.exp(-pred_mu)])
        try:
            method_outputs["Binary_CP_Marginal"] = binary_classification_cp(test_probs, calib_probs, calib_binary, alpha, mondrian=False)
        except Exception:
            method_outputs["Binary_CP_Marginal"] = None
        if include_binary_mondrian:
            try:
                method_outputs["Binary_Marginal_CP"] = binary_classification_cp(test_probs, calib_probs, calib_binary, alpha, mondrian=True)
            except Exception:
                method_outputs["Binary_Marginal_CP"] = None

        # Parametric benchmarks (only for Poisson_GLM learner)
        if learner_name == "Poisson_GLM":
            try:
                benchmarks = run_parametric_benchmarks(train_data, calib_data, test_data, valid_vars, alpha)
                method_outputs.update({k: v for k, v in benchmarks.items() if not k.startswith("_")})
            except Exception:
                pass

        # DGCP methods
        for dgcp_method in dgcp_method_list:
            for min_group_size in min_size_list:
                method_name = f"DGCP_{dgcp_method.title()}_m{min_group_size}"
                try:
                    intervals = two_stage_dgcp_cp(
                        pred_mu=pred_mu, calib_mu=calib_mu, calib_y=calib_y,
                        alpha=alpha, ncs_type="poisson_deviance",
                        min_group_size=min_group_size, method=dgcp_method, verbose=False,
                    )
                    method_outputs[method_name] = (intervals, min_group_size)
                except Exception:
                    method_outputs[method_name] = (None, min_group_size)

        # Two-Stage methods (optional)
        if include_two_stage:
            for method_name, func in [
                ("Two_Stage_Weighted_CP", two_stage_weighted_cp),
                ("Two_Stage_Calibrated_CP", two_stage_calibrated_cp),
                ("Two_Stage_Balanced_V2_CP", two_stage_balanced_v2_cp),
                ("Two_Stage_Direct_CP", two_stage_direct_cp),
                ("Two_Stage_Empirical_CP", two_stage_empirical_cp),
                ("Two_Stage_Exact_CP", two_stage_exact_cp),
            ]:
                try:
                    method_outputs[method_name] = func(pred_mu, calib_mu, calib_y, alpha)
                except Exception:
                    method_outputs[method_name] = None

        # Evaluate all methods
        for method_name, output in method_outputs.items():
            row: dict[str, object] = {
                "method": method_name,
                "learner": learner_name,
                **asdict(config),
                "pct_zeros_data": dgp_summary["pct_zeros"],
                "mean_y_data": dgp_summary["mean_y"],
                "var_y_data": dgp_summary["var_y"],
                "dispersion_ratio": dgp_summary["dispersion_ratio"],
            }
            min_group_size_value = np.nan
            if method_name.startswith("DGCP_"):
                output, min_group_size_value = output
            row["min_group_size"] = min_group_size_value

            if output is None:
                row["error"] = "Method failed"
                results.append(row)
                continue

            if method_name in {"Binary_CP_Marginal", "Binary_Marginal_CP"}:
                from scipy import stats as scipy_stats
                binary_truth = (test_y > 0).astype(int)
                covered_binary = np.array([t in s for t, s in zip(binary_truth, output)])
                lower = np.zeros(len(test_y))
                upper = np.zeros(len(test_y))
                for idx, (pred_set, mu_val) in enumerate(zip(output, pred_mu)):
                    if 0 in pred_set and 1 in pred_set:
                        lower[idx] = 0
                        upper[idx] = max(scipy_stats.poisson.ppf(1 - alpha / 2, mu_val), 1)
                    elif 0 in pred_set:
                        lower[idx] = 0
                        upper[idx] = 0
                    else:
                        lower[idx] = 1
                        upper[idx] = max(scipy_stats.poisson.ppf(1 - alpha / 2, mu_val), 1)
                m = compute_metrics(test_y, lower, upper, pred_mu, alpha)
                m["coverage"] = float(np.mean(covered_binary))
                row.update(m)
            else:
                lower = output["lower"].values if "lower" in output.columns else output["lower_bound"].values
                upper = output["upper"].values if "upper" in output.columns else output["upper_bound"].values
                row.update(compute_metrics(test_y, lower, upper, pred_mu, alpha))

            results.append(row)

    return results
