"""
Trustworthy experimental band-gap prediction from chemical composition.

Designed as a single Google Colab / Python script.
The default STRICT_EXPERIMENTAL_ONLY=True removes zero-gap metal labels,
because only the non-zero gaps in this source are literature experimental values.
"""

# -----------------------------------------------------------------------------
# 0. Install missing dependency (safe in Colab; skipped if already installed)
# -----------------------------------------------------------------------------
import importlib.util
import subprocess
import sys

if importlib.util.find_spec("matminer") is None:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "matminer>=0.9.3,<0.11"]
    )

# -----------------------------------------------------------------------------
# 1. Imports and configuration
# -----------------------------------------------------------------------------
from collections import Counter
from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import pearsonr, spearmanr
from pymatgen.core import Composition
from matminer.datasets import load_dataset
from matminer.featurizers.base import MultipleFeaturizer
from matminer.featurizers.composition import (
    ElementFraction,
    ElementProperty,
    Stoichiometry,
    ValenceOrbital,
)

from sklearn.cluster import MiniBatchKMeans
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    explained_variance_score,
    f1_score,
    matthews_corrcoef,
    max_error,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, KFold, train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

SEED = 42
STRICT_EXPERIMENTAL_ONLY = True
N_TREES = 300
N_BOOTSTRAP = 1000
N_CLUSTERS = 12
LOEO_MAX_ELEMENTS = 8
SCREEN_LOW_EV = 1.0
SCREEN_HIGH_EV = 2.0
OUTPUT_DIR = Path("bandgap_results")
OUTPUT_DIR.mkdir(exist_ok=True)

np.random.seed(SEED)
pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 160)

try:
    from IPython.display import display
except ImportError:
    display = print


# -----------------------------------------------------------------------------
# 2. Metric functions: each metric checks a different failure mode
# -----------------------------------------------------------------------------
def safe_corr(function, a, b):
    """Return a correlation safely when a fold is constant or too small."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(function(a, b)[0])


def concordance_correlation_coefficient(y_true, y_pred):
    """Agreement metric: correlation plus mean/scale agreement; ideal value = 1."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    vx = np.var(y_true)
    vy = np.var(y_pred)
    covariance = np.mean((y_true - y_true.mean()) * (y_pred - y_pred.mean()))
    denominator = vx + vy + (y_true.mean() - y_pred.mean()) ** 2
    return float(2 * covariance / denominator) if denominator > 0 else np.nan


def regression_metrics(y_true, y_pred):
    """A broad but non-redundant regression evidence card."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residual = y_pred - y_true
    absolute_error = np.abs(residual)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    baseline_abs = np.sum(np.abs(y_true - np.median(y_true)))
    target_sd = float(np.std(y_true, ddof=1)) if len(y_true) > 1 else np.nan

    # Regress observed y on predicted y. Ideal slope=1 and intercept=0.
    if len(y_true) >= 3 and np.std(y_pred) > 0:
        calibration_slope, calibration_intercept = np.polyfit(y_pred, y_true, 1)
    else:
        calibration_slope, calibration_intercept = np.nan, np.nan

    return {
        "n": int(len(y_true)),
        "MAE_eV": float(mean_absolute_error(y_true, y_pred)),
        "RMSE_eV": rmse,
        "MedianAE_eV": float(median_absolute_error(y_true, y_pred)),
        "P90_AE_eV": float(np.quantile(absolute_error, 0.90)),
        "MaxAE_eV": float(max_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
        "ExplainedVariance": float(explained_variance_score(y_true, y_pred)),
        "Pearson_r": safe_corr(pearsonr, y_true, y_pred),
        "Spearman_rho": safe_corr(spearmanr, y_true, y_pred),
        "CCC": concordance_correlation_coefficient(y_true, y_pred),
        "MeanBias_eV": float(np.mean(residual)),
        "ResidualSD_eV": float(np.std(residual, ddof=1)),
        "RelativeAbsoluteError": (
            float(np.sum(absolute_error) / baseline_abs) if baseline_abs > 0 else np.nan
        ),
        "RPD": float(target_sd / rmse) if rmse > 0 else np.inf,
        "CalibrationSlope": float(calibration_slope),
        "CalibrationIntercept_eV": float(calibration_intercept),
    }


def bootstrap_confidence_intervals(y_true, y_pred, n_boot=N_BOOTSTRAP, seed=SEED):
    """Paired bootstrap 95% confidence intervals for headline metrics."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    metric_names = ["MAE_eV", "RMSE_eV", "R2", "Spearman_rho", "CCC"]
    samples = {name: [] for name in metric_names}
    n = len(y_true)

    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        values = regression_metrics(y_true[idx], y_pred[idx])
        for name in metric_names:
            samples[name].append(values[name])

    rows = []
    point = regression_metrics(y_true, y_pred)
    for name in metric_names:
        values = np.asarray(samples[name], dtype=float)
        rows.append(
            {
                "metric": name,
                "estimate": point[name],
                "CI95_low": np.nanquantile(values, 0.025),
                "CI95_high": np.nanquantile(values, 0.975),
            }
        )
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# 3. Models
# -----------------------------------------------------------------------------
def make_extra_trees(n_trees=N_TREES):
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                ExtraTreesRegressor(
                    n_estimators=n_trees,
                    min_samples_leaf=2,
                    max_features=0.80,
                    random_state=SEED,
                    n_jobs=-1,
                ),
            ),
        ]
    )


MODEL_FACTORIES = {
    "Median baseline": lambda: Pipeline(
        [("imputer", SimpleImputer(strategy="median")),
         ("model", DummyRegressor(strategy="median"))]
    ),
    "Ridge": lambda: Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=10.0)),
        ]
    ),
    "HistGradientBoosting": lambda: Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                HistGradientBoostingRegressor(
                    learning_rate=0.05,
                    max_iter=350,
                    max_leaf_nodes=31,
                    l2_regularization=1.0,
                    random_state=SEED,
                ),
            ),
        ]
    ),
    "RandomForest": lambda: Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=N_TREES,
                    min_samples_leaf=2,
                    max_features=0.80,
                    random_state=SEED,
                    n_jobs=-1,
                ),
            ),
        ]
    ),
    "ExtraTrees": make_extra_trees,
}


def cross_validated_predictions(model_factory, splitter, X, y, groups=None):
    """Out-of-fold predictions plus fold-to-fold stability metrics."""
    y = np.asarray(y, dtype=float)
    predictions = np.full(len(y), np.nan)
    fold_id = np.full(len(y), -1, dtype=int)
    fold_rows = []
    split_iterator = splitter.split(X, y, groups) if groups is not None else splitter.split(X, y)

    for fold, (train_idx, test_idx) in enumerate(split_iterator, start=1):
        model = model_factory()
        model.fit(X.iloc[train_idx], y[train_idx])
        fold_pred = model.predict(X.iloc[test_idx])
        predictions[test_idx] = fold_pred
        fold_id[test_idx] = fold
        row = {"fold": fold, "n_train": len(train_idx), "n_test": len(test_idx)}
        row.update(regression_metrics(y[test_idx], fold_pred))
        fold_rows.append(row)

    if np.isnan(predictions).any() or (fold_id < 0).any():
        raise RuntimeError("Some rows did not receive an out-of-fold prediction.")
    return predictions, fold_id, pd.DataFrame(fold_rows)


# -----------------------------------------------------------------------------
# 4. Load the real dataset and audit it
# -----------------------------------------------------------------------------
print("\nLoading matbench_expt_gap ...")
raw = load_dataset("matbench_expt_gap").copy()
FORMULA_COLUMN = "composition"
TARGET_COLUMN = "gap expt"

if FORMULA_COLUMN not in raw.columns or TARGET_COLUMN not in raw.columns:
    raise KeyError(f"Unexpected columns: {raw.columns.tolist()}")

raw["composition_object"] = raw[FORMULA_COLUMN].map(
    lambda x: x if isinstance(x, Composition) else Composition(str(x))
)
raw["formula"] = raw["composition_object"].map(lambda c: c.reduced_formula)
raw["chemical_system"] = raw["composition_object"].map(
    lambda c: "-".join(sorted(element.symbol for element in c.elements))
)
raw[TARGET_COLUMN] = pd.to_numeric(raw[TARGET_COLUMN], errors="coerce")

audit = {
    "source_rows": int(len(raw)),
    "missing_targets": int(raw[TARGET_COLUMN].isna().sum()),
    "duplicate_reduced_formulas": int(raw["formula"].duplicated().sum()),
    "zero_gap_rows": int((raw[TARGET_COLUMN] == 0).sum()),
}

data = raw.dropna(subset=[TARGET_COLUMN]).copy()
if STRICT_EXPERIMENTAL_ONLY:
    data = data[data[TARGET_COLUMN] > 0].copy()

data.reset_index(drop=True, inplace=True)
audit["rows_used"] = int(len(data))
audit["strict_experimental_only"] = STRICT_EXPERIMENTAL_ONLY

print("\nDATA AUDIT")
print(json.dumps(audit, indent=2))
print("\nTarget summary (eV)")
display(data[TARGET_COLUMN].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).to_frame().T)

if data["formula"].duplicated().any():
    raise ValueError("Duplicate formulas remain; stop to prevent formula leakage.")


# -----------------------------------------------------------------------------
# 5. Convert every composition into fixed-size chemistry descriptors
# -----------------------------------------------------------------------------
print("\nGenerating composition descriptors ...")
featurizer = MultipleFeaturizer(
    [
        Stoichiometry(),
        ElementFraction(),
        ElementProperty.from_preset("magpie"),
        ValenceOrbital(props=["avg"]),
    ]
)

feature_values = featurizer.featurize_many(
    data["composition_object"].tolist(),
    ignore_errors=True,
    pbar=True,
)
X = pd.DataFrame(feature_values, columns=featurizer.feature_labels())
X = X.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
X = X.loc[:, ~X.columns.duplicated()].copy()

# Remove columns that carry no usable information. This uses X only, never y.
drop_columns = [
    column
    for column in X.columns
    if X[column].isna().all() or X[column].nunique(dropna=True) <= 1
]
X.drop(columns=drop_columns, inplace=True)
y = data[TARGET_COLUMN].to_numpy(dtype=float)

print(f"Rows used: {len(X):,}")
print(f"Usable descriptors: {X.shape[1]:,}")
print(f"Dropped unusable descriptors: {len(drop_columns):,}")
print(f"Total missing descriptor cells: {int(X.isna().sum().sum()):,}")


# -----------------------------------------------------------------------------
# 6. Standard random 5-fold comparison against simple baselines
# -----------------------------------------------------------------------------
print("\nRANDOM 5-FOLD MODEL COMPARISON")
random_splitter = KFold(n_splits=5, shuffle=True, random_state=SEED)
comparison_rows = []
random_outputs = {}

for model_name, factory in MODEL_FACTORIES.items():
    print(f"  Evaluating {model_name} ...")
    pred, folds, fold_table = cross_validated_predictions(factory, random_splitter, X, y)
    row = {"model": model_name}
    row.update(regression_metrics(y, pred))
    row["Fold_MAE_mean"] = fold_table["MAE_eV"].mean()
    row["Fold_MAE_std"] = fold_table["MAE_eV"].std(ddof=1)
    row["Fold_R2_mean"] = fold_table["R2"].mean()
    row["Fold_R2_std"] = fold_table["R2"].std(ddof=1)
    comparison_rows.append(row)
    random_outputs[model_name] = (pred, folds, fold_table)

model_comparison = pd.DataFrame(comparison_rows).sort_values("MAE_eV")
display(model_comparison)
model_comparison.to_csv(OUTPUT_DIR / "model_comparison_random_cv.csv", index=False)


# -----------------------------------------------------------------------------
# 7. Three validation regimes: interpolation and two distribution shifts
# -----------------------------------------------------------------------------
selected_name = "ExtraTrees"
random_pred, random_fold_id, random_fold_metrics = random_outputs[selected_name]

print("\nCHEMICAL-SYSTEM GROUP 5-FOLD TEST")
chemical_groups = data["chemical_system"].to_numpy()
chemical_splitter = GroupKFold(n_splits=5)
chemical_pred, chemical_fold_id, chemical_fold_metrics = cross_validated_predictions(
    make_extra_trees, chemical_splitter, X, y, groups=chemical_groups
)

print("\nCOMPOSITION-CLUSTER GROUP 5-FOLD TEST")
# Cluster creation uses descriptors only; no target values enter this operation.
cluster_imputer = SimpleImputer(strategy="median")
cluster_scaler = StandardScaler()
X_cluster = cluster_imputer.fit_transform(X)
X_cluster = cluster_scaler.fit_transform(X_cluster)
cluster_pca = PCA(n_components=0.95, svd_solver="full")
X_cluster = cluster_pca.fit_transform(X_cluster)
cluster_model = MiniBatchKMeans(
    n_clusters=N_CLUSTERS,
    random_state=SEED,
    n_init=10,
    batch_size=512,
)
cluster_groups = cluster_model.fit_predict(X_cluster)
cluster_splitter = GroupKFold(n_splits=5)
cluster_pred, cluster_fold_id, cluster_fold_metrics = cross_validated_predictions(
    make_extra_trees, cluster_splitter, X, y, groups=cluster_groups
)

scenario_predictions = {
    "Random 5-fold": random_pred,
    "Chemical-system 5-fold": chemical_pred,
    "Composition-cluster 5-fold": cluster_pred,
}
scenario_rows = []
for scenario, pred in scenario_predictions.items():
    row = {"scenario": scenario}
    row.update(regression_metrics(y, pred))
    scenario_rows.append(row)

scenario_metrics = pd.DataFrame(scenario_rows)
random_mae = float(scenario_metrics.loc[scenario_metrics["scenario"] == "Random 5-fold", "MAE_eV"].iloc[0])
scenario_metrics["MAE_increase_vs_random_eV"] = scenario_metrics["MAE_eV"] - random_mae
scenario_metrics["MAE_ratio_vs_random"] = scenario_metrics["MAE_eV"] / random_mae

print("\nGENERALIZATION EVIDENCE CARD")
display(scenario_metrics)
scenario_metrics.to_csv(OUTPUT_DIR / "generalization_scenarios.csv", index=False)

all_fold_metrics = pd.concat(
    [
        random_fold_metrics.assign(scenario="Random 5-fold"),
        chemical_fold_metrics.assign(scenario="Chemical-system 5-fold"),
        cluster_fold_metrics.assign(scenario="Composition-cluster 5-fold"),
    ],
    ignore_index=True,
)
all_fold_metrics.to_csv(OUTPUT_DIR / "fold_stability_metrics.csv", index=False)


# -----------------------------------------------------------------------------
# 8. Bootstrap confidence intervals: uncertainty of the reported metrics
# -----------------------------------------------------------------------------
print("\nBOOTSTRAP 95% CONFIDENCE INTERVALS")
bootstrap_tables = []
for scenario, pred in scenario_predictions.items():
    table = bootstrap_confidence_intervals(y, pred)
    table.insert(0, "scenario", scenario)
    bootstrap_tables.append(table)
bootstrap_results = pd.concat(bootstrap_tables, ignore_index=True)
display(bootstrap_results)
bootstrap_results.to_csv(OUTPUT_DIR / "bootstrap_confidence_intervals.csv", index=False)


# -----------------------------------------------------------------------------
# 9. Leave-one-element-out: a deliberately extreme unseen-element test
# -----------------------------------------------------------------------------
print("\nLEAVE-ONE-ELEMENT-OUT TESTS")
element_sets = [set(element.symbol for element in c.elements) for c in data["composition_object"]]
element_counts = Counter(element for elements in element_sets for element in elements)

# Avoid tiny test sets and cases where removing one element destroys most training data.
candidate_elements = [
    element
    for element, count in element_counts.most_common()
    if count >= 75 and count <= 0.35 * len(data)
][:LOEO_MAX_ELEMENTS]

loeo_rows = []
for element in candidate_elements:
    test_mask = np.array([element in elements for elements in element_sets])
    train_idx = np.where(~test_mask)[0]
    test_idx = np.where(test_mask)[0]
    print(f"  Holding out {element}: train={len(train_idx)}, test={len(test_idx)}")
    model = make_extra_trees()
    model.fit(X.iloc[train_idx], y[train_idx])
    pred = model.predict(X.iloc[test_idx])
    row = {"held_out_element": element, "n_train": len(train_idx), "n_test": len(test_idx)}
    row.update(regression_metrics(y[test_idx], pred))
    loeo_rows.append(row)

loeo_results = pd.DataFrame(loeo_rows).sort_values("MAE_eV") if loeo_rows else pd.DataFrame()
display(loeo_results)
loeo_results.to_csv(OUTPUT_DIR / "leave_one_element_out.csv", index=False)


# -----------------------------------------------------------------------------
# 10. Independent train/calibration/test split for uncertainty evaluation
# -----------------------------------------------------------------------------
print("\nUNCERTAINTY, OOD, AND SELECTIVE-PREDICTION TEST")
all_indices = np.arange(len(y))
target_bins = pd.qcut(y, q=10, labels=False, duplicates="drop")
train_idx, temp_idx = train_test_split(
    all_indices,
    test_size=0.40,
    random_state=SEED,
    stratify=target_bins,
)
cal_idx, test_idx = train_test_split(
    temp_idx,
    test_size=0.50,
    random_state=SEED,
    stratify=np.asarray(target_bins)[temp_idx],
)

uncertainty_model = make_extra_trees(n_trees=500)
uncertainty_model.fit(X.iloc[train_idx], y[train_idx])
cal_pred = uncertainty_model.predict(X.iloc[cal_idx])
test_pred = uncertainty_model.predict(X.iloc[test_idx])
y_cal = y[cal_idx]
y_test = y[test_idx]

# Tree-to-tree spread is a useful ranking signal, but not calibrated by itself.
imputer = uncertainty_model.named_steps["imputer"]
forest = uncertainty_model.named_steps["model"]
X_cal_imputed = imputer.transform(X.iloc[cal_idx])
X_test_imputed = imputer.transform(X.iloc[test_idx])
cal_tree_predictions = np.vstack([tree.predict(X_cal_imputed) for tree in forest.estimators_])
test_tree_predictions = np.vstack([tree.predict(X_test_imputed) for tree in forest.estimators_])
cal_scale = np.maximum(cal_tree_predictions.std(axis=0, ddof=1), 0.05)
test_scale = np.maximum(test_tree_predictions.std(axis=0, ddof=1), 0.05)


def finite_sample_conformal_quantile(scores, alpha):
    """Higher empirical quantile with the split-conformal finite-sample correction."""
    scores = np.asarray(scores, dtype=float)
    level = min(1.0, np.ceil((len(scores) + 1) * (1 - alpha)) / len(scores))
    return float(np.quantile(scores, level, method="higher"))


normalized_calibration_scores = np.abs(y_cal - cal_pred) / cal_scale
coverage_rows = []
intervals = {}
for nominal_coverage in [0.80, 0.90, 0.95]:
    alpha = 1 - nominal_coverage
    q = finite_sample_conformal_quantile(normalized_calibration_scores, alpha)
    lower = test_pred - q * test_scale
    upper = test_pred + q * test_scale
    covered = (y_test >= lower) & (y_test <= upper)
    intervals[nominal_coverage] = (lower, upper)
    coverage_rows.append(
        {
            "nominal_coverage": nominal_coverage,
            "observed_coverage": covered.mean(),
            "coverage_error": covered.mean() - nominal_coverage,
            "mean_interval_width_eV": np.mean(upper - lower),
            "median_interval_width_eV": np.median(upper - lower),
            "P90_interval_width_eV": np.quantile(upper - lower, 0.90),
        }
    )

conformal_results = pd.DataFrame(coverage_rows)
display(conformal_results)
conformal_results.to_csv(OUTPUT_DIR / "conformal_calibration.csv", index=False)

# Conditional coverage by predicted-gap quartile; this can reveal hidden weak regions.
lower90, upper90 = intervals[0.90]
predicted_quartile = pd.qcut(test_pred, 4, labels=["Q1 low", "Q2", "Q3", "Q4 high"], duplicates="drop")
conditional_rows = []
for group in pd.unique(predicted_quartile):
    mask = np.asarray(predicted_quartile == group)
    conditional_rows.append(
        {
            "predicted_gap_group": str(group),
            "n": int(mask.sum()),
            "observed_90pct_coverage": float(((y_test[mask] >= lower90[mask]) & (y_test[mask] <= upper90[mask])).mean()),
            "mean_interval_width_eV": float(np.mean(upper90[mask] - lower90[mask])),
            "MAE_eV": float(mean_absolute_error(y_test[mask], test_pred[mask])),
        }
    )
conditional_coverage = pd.DataFrame(conditional_rows)
display(conditional_coverage)
conditional_coverage.to_csv(OUTPUT_DIR / "conditional_coverage.csv", index=False)


# -----------------------------------------------------------------------------
# 11. Applicability domain: distance from training chemistry
# -----------------------------------------------------------------------------
ood_imputer = SimpleImputer(strategy="median")
ood_scaler = StandardScaler()
X_train_ood = ood_imputer.fit_transform(X.iloc[train_idx])
X_cal_ood = ood_imputer.transform(X.iloc[cal_idx])
X_test_ood = ood_imputer.transform(X.iloc[test_idx])
X_train_ood = ood_scaler.fit_transform(X_train_ood)
X_cal_ood = ood_scaler.transform(X_cal_ood)
X_test_ood = ood_scaler.transform(X_test_ood)

ood_pca = PCA(n_components=0.95, svd_solver="full")
X_train_ood = ood_pca.fit_transform(X_train_ood)
X_cal_ood = ood_pca.transform(X_cal_ood)
X_test_ood = ood_pca.transform(X_test_ood)

nearest_neighbors = NearestNeighbors(n_neighbors=5)
nearest_neighbors.fit(X_train_ood)
cal_distances = nearest_neighbors.kneighbors(X_cal_ood, return_distance=True)[0].mean(axis=1)
test_distances = nearest_neighbors.kneighbors(X_test_ood, return_distance=True)[0].mean(axis=1)
absolute_test_error = np.abs(y_test - test_pred)

high_error_cutoff = np.quantile(absolute_test_error, 0.75)
high_error_label = (absolute_test_error >= high_error_cutoff).astype(int)
ood_threshold = np.quantile(cal_distances, 0.95)
ood_warning = test_distances > ood_threshold
relative_far = test_distances >= np.quantile(test_distances, 0.75)
relative_near = test_distances <= np.quantile(test_distances, 0.75)

ood_results = pd.DataFrame(
    [
        {
            "Distance_error_Spearman": safe_corr(spearmanr, test_distances, absolute_test_error),
            "Distance_high_error_AUROC": roc_auc_score(high_error_label, test_distances),
            "Uncertainty_error_Spearman": safe_corr(spearmanr, test_scale, absolute_test_error),
            "Uncertainty_high_error_AUROC": roc_auc_score(high_error_label, test_scale),
            "Calibration_95pct_distance_threshold": ood_threshold,
            "OOD_warning_rate": ood_warning.mean(),
            "OOD_warning_MAE_eV": mean_absolute_error(y_test[ood_warning], test_pred[ood_warning]) if ood_warning.any() else np.nan,
            "Non_OOD_MAE_eV": mean_absolute_error(y_test[~ood_warning], test_pred[~ood_warning]) if (~ood_warning).any() else np.nan,
            "Nearest_75pct_MAE_eV": mean_absolute_error(y_test[relative_near], test_pred[relative_near]),
            "Farthest_25pct_MAE_eV": mean_absolute_error(y_test[relative_far], test_pred[relative_far]),
        }
    ]
)
display(ood_results)
ood_results.to_csv(OUTPUT_DIR / "applicability_domain.csv", index=False)


# -----------------------------------------------------------------------------
# 12. Selective prediction: does abstaining reduce risk?
# -----------------------------------------------------------------------------
order = np.argsort(test_scale)  # most confident first
selective_rows = []
for retained_fraction in [1.00, 0.90, 0.80, 0.70, 0.50, 0.30]:
    n_keep = max(1, int(np.ceil(retained_fraction * len(test_idx))))
    keep = order[:n_keep]
    metrics = regression_metrics(y_test[keep], test_pred[keep])
    selective_rows.append(
        {
            "retained_fraction": retained_fraction,
            "n_predictions_issued": n_keep,
            "n_abstained": len(test_idx) - n_keep,
            "MAE_eV": metrics["MAE_eV"],
            "RMSE_eV": metrics["RMSE_eV"],
            "P90_AE_eV": metrics["P90_AE_eV"],
            "R2": metrics["R2"],
        }
    )

selective_results = pd.DataFrame(selective_rows)
display(selective_results)
selective_results.to_csv(OUTPUT_DIR / "selective_prediction_risk_coverage.csv", index=False)


# -----------------------------------------------------------------------------
# 13. Screening test for an explicitly defined application window
# -----------------------------------------------------------------------------
true_in_window = ((y_test >= SCREEN_LOW_EV) & (y_test <= SCREEN_HIGH_EV)).astype(int)
predicted_in_window = ((test_pred >= SCREEN_LOW_EV) & (test_pred <= SCREEN_HIGH_EV)).astype(int)
tn, fp, fn, tp = confusion_matrix(true_in_window, predicted_in_window, labels=[0, 1]).ravel()
precision = precision_score(true_in_window, predicted_in_window, zero_division=0)
prevalence = true_in_window.mean()
center = (SCREEN_LOW_EV + SCREEN_HIGH_EV) / 2
window_ranking_score = -np.abs(test_pred - center)

screening_results = pd.DataFrame(
    [
        {
            "window_low_eV": SCREEN_LOW_EV,
            "window_high_eV": SCREEN_HIGH_EV,
            "prevalence": prevalence,
            "selection_rate": predicted_in_window.mean(),
            "accuracy": accuracy_score(true_in_window, predicted_in_window),
            "balanced_accuracy": balanced_accuracy_score(true_in_window, predicted_in_window),
            "precision": precision,
            "recall": recall_score(true_in_window, predicted_in_window, zero_division=0),
            "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
            "F1": f1_score(true_in_window, predicted_in_window, zero_division=0),
            "MCC": matthews_corrcoef(true_in_window, predicted_in_window),
            "AUROC_window_ranking": roc_auc_score(true_in_window, window_ranking_score),
            "AveragePrecision_window_ranking": average_precision_score(true_in_window, window_ranking_score),
            "enrichment_factor": precision / prevalence if prevalence > 0 else np.nan,
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
        }
    ]
)
display(screening_results)
screening_results.to_csv(OUTPUT_DIR / "screening_metrics.csv", index=False)


# -----------------------------------------------------------------------------
# 14. Error slices: a good average can hide poor target regions
# -----------------------------------------------------------------------------
error_bins = [0, 1, 2, 3, 4, np.inf]
error_labels = ["0-1", "1-2", "2-3", "3-4", ">4"]
test_ranges = pd.cut(y_test, bins=error_bins, labels=error_labels, right=False)
range_rows = []
for label in error_labels:
    mask = np.asarray(test_ranges == label)
    if mask.sum() == 0:
        continue
    values = regression_metrics(y_test[mask], test_pred[mask])
    range_rows.append(
        {
            "true_gap_range_eV": label,
            "n": int(mask.sum()),
            "MAE_eV": values["MAE_eV"],
            "RMSE_eV": values["RMSE_eV"],
            "MedianAE_eV": values["MedianAE_eV"],
            "MeanBias_eV": values["MeanBias_eV"],
        }
    )
error_slices = pd.DataFrame(range_rows)
display(error_slices)
error_slices.to_csv(OUTPUT_DIR / "error_by_true_gap_range.csv", index=False)


# -----------------------------------------------------------------------------
# 15. Learning curve: would more measurements likely help?
# -----------------------------------------------------------------------------
rng = np.random.default_rng(SEED)
learning_rows = []
for fraction in [0.20, 0.40, 0.60, 0.80, 1.00]:
    n_subset = max(100, int(len(train_idx) * fraction))
    subset = rng.choice(train_idx, size=n_subset, replace=False)
    model = make_extra_trees(n_trees=200)
    model.fit(X.iloc[subset], y[subset])
    pred = model.predict(X.iloc[test_idx])
    values = regression_metrics(y_test, pred)
    learning_rows.append(
        {
            "training_fraction": fraction,
            "n_train": n_subset,
            "MAE_eV": values["MAE_eV"],
            "RMSE_eV": values["RMSE_eV"],
            "R2": values["R2"],
        }
    )
learning_curve_results = pd.DataFrame(learning_rows)
display(learning_curve_results)
learning_curve_results.to_csv(OUTPUT_DIR / "learning_curve.csv", index=False)


# -----------------------------------------------------------------------------
# 16. Feature importance for interpretation (association, not causation)
# -----------------------------------------------------------------------------
feature_importance = pd.DataFrame(
    {
        "feature": X.columns,
        "importance": forest.feature_importances_,
    }
).sort_values("importance", ascending=False)
display(feature_importance.head(20))
feature_importance.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)


# -----------------------------------------------------------------------------
# 17. Save row-level predictions for full auditability
# -----------------------------------------------------------------------------
cv_predictions_table = data[["formula", "chemical_system", TARGET_COLUMN]].copy()
cv_predictions_table["random_cv_prediction_eV"] = random_pred
cv_predictions_table["random_cv_fold"] = random_fold_id
cv_predictions_table["chemical_system_cv_prediction_eV"] = chemical_pred
cv_predictions_table["chemical_system_cv_fold"] = chemical_fold_id
cv_predictions_table["composition_cluster_cv_prediction_eV"] = cluster_pred
cv_predictions_table["composition_cluster_cv_fold"] = cluster_fold_id
cv_predictions_table["composition_cluster"] = cluster_groups
cv_predictions_table.to_csv(OUTPUT_DIR / "all_cross_validated_predictions.csv", index=False)

test_predictions_table = data.loc[test_idx, ["formula", "chemical_system", TARGET_COLUMN]].copy()
test_predictions_table["prediction_eV"] = test_pred
test_predictions_table["absolute_error_eV"] = absolute_test_error
test_predictions_table["tree_uncertainty_scale"] = test_scale
test_predictions_table["distance_to_training"] = test_distances
test_predictions_table["OOD_warning"] = ood_warning
test_predictions_table["interval_90_low_eV"] = lower90
test_predictions_table["interval_90_high_eV"] = upper90
test_predictions_table.to_csv(OUTPUT_DIR / "heldout_test_predictions.csv", index=False)

with open(OUTPUT_DIR / "dataset_audit.json", "w", encoding="utf-8") as file:
    json.dump(audit, file, indent=2)


# -----------------------------------------------------------------------------
# 18. One dashboard containing the major independent checks
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(3, 3, figsize=(19, 16))

# A. Predicted versus measured
ax = axes[0, 0]
ax.scatter(y_test, test_pred, s=18, alpha=0.55)
limits = [min(y_test.min(), test_pred.min()), max(y_test.max(), test_pred.max())]
ax.plot(limits, limits, "k--", linewidth=1.5, label="Ideal")
ax.set(xlabel="Measured gap (eV)", ylabel="Predicted gap (eV)", title="Held-out predictions")
ax.legend()

# B. Residual diagnostic
ax = axes[0, 1]
ax.scatter(test_pred, test_pred - y_test, s=18, alpha=0.55)
ax.axhline(0, color="black", linestyle="--")
ax.set(xlabel="Predicted gap (eV)", ylabel="Residual: predicted - measured (eV)", title="Residual pattern")

# C. Generalization shift
ax = axes[0, 2]
ax.bar(scenario_metrics["scenario"], scenario_metrics["MAE_eV"], color=["#4C78A8", "#F58518", "#E45756"])
ax.set(ylabel="MAE (eV)", title="Generalization becomes harder")
ax.tick_params(axis="x", rotation=20)

# D. Conformal calibration
ax = axes[1, 0]
ax.plot(conformal_results["nominal_coverage"], conformal_results["observed_coverage"], "o-", label="Observed")
ax.plot([0.75, 1.0], [0.75, 1.0], "k--", label="Ideal")
ax.set(xlabel="Nominal coverage", ylabel="Observed coverage", title="Interval calibration", xlim=(0.77, 0.98), ylim=(0.77, 0.98))
ax.legend()

# E. Distance versus error
ax = axes[1, 1]
ax.scatter(test_distances, absolute_test_error, s=18, alpha=0.55)
ax.axvline(ood_threshold, color="red", linestyle="--", label="Calibrated OOD threshold")
ax.set(xlabel="Distance from training chemistry", ylabel="Absolute error (eV)", title="Applicability domain")
ax.legend()

# F. Risk-coverage curve
ax = axes[1, 2]
ax.plot(selective_results["retained_fraction"], selective_results["MAE_eV"], "o-")
ax.set(xlabel="Fraction receiving predictions", ylabel="MAE (eV)", title="Selective prediction")
ax.invert_xaxis()

# G. Uncertainty versus error
ax = axes[2, 0]
ax.scatter(test_scale, absolute_test_error, s=18, alpha=0.55)
ax.set(xlabel="Tree uncertainty scale", ylabel="Absolute error (eV)", title="Does uncertainty track failure?")

# H. Screening confusion matrix
ax = axes[2, 1]
cm = np.array([[tn, fp], [fn, tp]])
image = ax.imshow(cm, cmap="Blues")
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=13)
ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Outside", "Inside"], yticklabels=["Outside", "Inside"], xlabel="Predicted", ylabel="Measured", title=f"Screening: {SCREEN_LOW_EV}-{SCREEN_HIGH_EV} eV")
fig.colorbar(image, ax=ax, fraction=0.046)

# I. Top feature importance
ax = axes[2, 2]
top_features = feature_importance.head(15).sort_values("importance")
ax.barh(top_features["feature"], top_features["importance"])
ax.set(xlabel="ExtraTrees importance", title="Top composition descriptors")

fig.suptitle("Trustworthy Experimental Band-Gap Validation Dashboard", fontsize=18, y=1.01)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "validation_dashboard.png", dpi=220, bbox_inches="tight")
plt.show()


# -----------------------------------------------------------------------------
# 19. Final concise summary
# -----------------------------------------------------------------------------
print("\n" + "=" * 90)
print("FINAL EVIDENCE SUMMARY")
print("=" * 90)
display(scenario_metrics[["scenario", "MAE_eV", "RMSE_eV", "R2", "Spearman_rho", "CCC", "MAE_ratio_vs_random"]])
display(conformal_results)
display(ood_results)
display(selective_results)
display(screening_results)
print(f"\nAll tables, row-level predictions, and the dashboard were saved to: {OUTPUT_DIR.resolve()}")
print("\nInterpretation rule:")
print("Do not call the model trustworthy from random-CV R2 alone.")
print("Trust is supported only if group/cluster errors remain acceptable, intervals calibrate,")
print("uncertainty and distance identify failures, and abstention genuinely lowers error.")
