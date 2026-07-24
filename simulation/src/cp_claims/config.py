METHOD_DISPLAY_NAMES = {
    "DGCP_Exact_m10": "DGCP Exact m=10",
    "DGCP_Exact_m20": "DGCP Exact m=20",
    "DGCP_Exact_m50": "DGCP Exact m=50",
    "DGCP_Exact_m100": "DGCP Exact m=100",
    "DGCP_Exact_m200": "DGCP Exact m=200",
    "DGCP_Full_m10": "DGCP Full m=10",
    "DGCP_Full_m20": "DGCP Full m=20",
    "DGCP_Full_m50": "DGCP Full m=50",
    "DGCP_Full_m100": "DGCP Full m=100",
    "DGCP_Full_m200": "DGCP Full m=200",
    "DGCP_Hybrid_m10": "DGCP Hybrid m=10",
    "DGCP_Hybrid_m20": "DGCP Hybrid m=20",
    "DGCP_Hybrid_m50": "DGCP Hybrid m=50",
    "DGCP_Hybrid_m100": "DGCP Hybrid m=100",
    "DGCP_Hybrid_m200": "DGCP Hybrid m=200",
    "Poisson_Normal": "Parametric Poisson",
    "Poisson_Bootstrap": "Bootstrap Poisson",
    "NB_Plugin": "NB Plugin",
    "NB_Chebyshev": "NB Chebyshev",
    "NB_Bootstrap": "NB Bootstrap",
    "ICP_poisson_deviance": "Standard ICP",
    "ICP_za_relative": "ZA Relative ICP",
    "Mondrian_Poisson": "Cluster Cond. ICP",
    "Binary_CP_Marginal": "Binary CP Marginal",
    "Binary_Marginal_CP": "Binary Marginal CP",
    "Two_Stage_CP": "Two Stage CP",
    "Two_Stage_Bootstrap": "Two Stage Bootstrap",
    "Two_Stage_Plugin": "Two Stage Plugin",
}

BENCHMARK_METHODS = [
    "Poisson_Normal",
    "Poisson_Bootstrap",
    "NB_Plugin",
    "NB_Chebyshev",
    "NB_Bootstrap",
]

CP_METHODS = [
    "ICP_poisson_deviance",
    "ICP_za_relative",
    "Mondrian_Poisson",
    "Binary_CP_Marginal",
    "Binary_Marginal_CP",
]

DEFAULT_MAIN_METHODS = BENCHMARK_METHODS + CP_METHODS + [
    "DGCP_Hybrid_m200",
    "DGCP_Full_m200",
    "DGCP_Exact_m200",
]

DEFAULT_BASELINE = {
    "interaction_effects": True,
    "spatial_effects": False,
    "n_obs": 10000,
    "dim_x": 10,
    "correlation_strength": 0.3,
    "categorical_ratio": 0.3,
}

DGCP_METHOD_TYPES = ["hybrid", "full", "exact"]
MIN_GROUP_SIZES = [10, 20, 50, 100, 200]

LEARNERS = ["Poisson_GLM", "LightGBM", "RandomForest"]
