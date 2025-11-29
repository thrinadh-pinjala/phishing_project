# --- Interpretability Framework ---

from sklearn.inspection import permutation_importance

def compute_permutation_importance(model, X, y, n_repeats=10, random_state=42):
    """
    Compute permutation importance for a fitted model.
    Returns a dict {feature: importance}.
    """
    result = permutation_importance(model, X, y, n_repeats=n_repeats, random_state=random_state)
    return dict(zip(X.columns, result.importances_mean))

def compute_shap_values(model, X, nsamples=100):
    """
    Compute SHAP values for a fitted model and a sample of X.
    Returns the SHAP explainer and values.
    Handles both binary and multiclass models.
    """
    import shap
    # Use TreeExplainer for tree models, LinearExplainer for linear, KernelExplainer otherwise
    if hasattr(model, 'feature_importances_'):
        explainer = shap.TreeExplainer(model)
    elif hasattr(model, 'coef_'):
        explainer = shap.LinearExplainer(model, X)
    else:
        explainer = shap.KernelExplainer(model.predict_proba, X)
    sample_X = X.sample(min(nsamples, len(X)), random_state=42)
    try:
        shap_values = explainer.shap_values(sample_X)
    except Exception:
        # fallback for some models
        shap_values = explainer(sample_X)
    return explainer, shap_values

def plot_probability_path(model, X_row, feature_names=None):
    """
    Visualize how each feature contributes to the final probability for a single sample.
    Uses SHAP waterfall plot if available. For linear models, also prints stepwise contributions.
    """
    import shap
    if hasattr(model, 'feature_importances_') or hasattr(model, 'coef_'):
        explainer = shap.Explainer(model, X_row.to_frame().T)
        shap_values = explainer(X_row.to_frame().T)
        try:
            shap.plots.waterfall(shap_values[0])
        except Exception:
            print('SHAP waterfall plot not available for this model.')
        # For linear models, print stepwise contributions
        if hasattr(model, 'coef_'):
            coefs = model.coef_[0] if hasattr(model.coef_, '__len__') else model.coef_
            intercept = model.intercept_[0] if hasattr(model, 'intercept_') else 0
            contribs = X_row.values * coefs
            print('--- Linear Model Probability Path ---')
            print(f'Intercept: {intercept:.4f}')
            for i, f in enumerate(feature_names or X_row.index):
                print(f'{f}: {X_row.values[i]:.4f} * {coefs[i]:.4f} = {contribs[i]:.4f}')
            print(f'Total (logit): {intercept + contribs.sum():.4f}')
    else:
        print('Probability path visualization is only available for supported models.')

def risk_factor_decomposition(model, X, feature_groups, normalize=True):
    """
    Decompose risk by feature group.
    feature_groups: dict {group_name: [feature1, feature2, ...]}
    Returns dict {group_name: group_score}
    If normalize=True, group scores sum to 1.
    """
    importances = track_feature_importance(model, X.columns)
    group_scores = {}
    for group, features in feature_groups.items():
        group_scores[group] = sum(importances.get(f, 0) for f in features)
    if normalize:
        total = sum(group_scores.values())
        if total > 0:
            for group in group_scores:
                group_scores[group] /= total
    return group_scores
def global_feature_importance_summary(model, X, y=None, nsamples=100):
    """
    Return a summary of feature importance using multiple methods: model-based, permutation, SHAP.
    """
    summary = {}
    # Model-based
    summary['model_importance'] = track_feature_importance(model, X.columns)
    # Permutation
    if y is not None:
        try:
            summary['permutation_importance'] = compute_permutation_importance(model, X, y, n_repeats=5)
        except Exception:
            summary['permutation_importance'] = {}
    # SHAP
    try:
        explainer, shap_values = compute_shap_values(model, X, nsamples=nsamples)
        if isinstance(shap_values, list):
            # For multiclass, take mean abs across classes
            shap_imp = dict(zip(X.columns, np.mean(np.abs(shap_values), axis=(0,1))))
        else:
            shap_imp = dict(zip(X.columns, np.mean(np.abs(shap_values.values), axis=0)))
        summary['shap_importance'] = shap_imp
    except Exception:
        summary['shap_importance'] = {}
    return summary
import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import CalibratedClassifierCV

def calculate_kl_divergence(p, q, eps=1e-8):
    """
    Calculate Kullback-Leibler divergence D_KL(P||Q) between two distributions.
    p, q: arrays representing probability distributions (should sum to 1)
    """
    p = np.asarray(p) + eps
    q = np.asarray(q) + eps
    p = p / np.sum(p)
    q = q / np.sum(q)
    return np.sum(p * np.log(p / q))

def calculate_psi(expected, actual, buckets=10):
    """
    Calculate Population Stability Index (PSI) between two samples.
    expected: baseline (reference) feature values
    actual: new (current) feature values
    buckets: number of bins
    """
    expected = np.asarray(expected)
    actual = np.asarray(actual)
    # Remove nan/inf
    expected = expected[np.isfinite(expected)]
    actual = actual[np.isfinite(actual)]
    if expected.size == 0 or actual.size == 0:
        return np.nan
    # Bin edges based on expected
    quantiles = np.linspace(0, 1, buckets + 1)
    try:
        bin_edges = np.unique(np.quantile(expected, quantiles))
        if len(bin_edges) < 2:
            bin_edges = np.array([np.min(expected), np.max(expected)])
    except Exception:
        bin_edges = np.array([np.min(expected), np.max(expected)])
    expected_counts, _ = np.histogram(expected, bins=bin_edges)
    actual_counts, _ = np.histogram(actual, bins=bin_edges)
    expected_percents = expected_counts / (len(expected) + 1e-8)
    actual_percents = actual_counts / (len(actual) + 1e-8)
    # Avoid division by zero
    expected_percents = np.where(expected_percents == 0, 1e-8, expected_percents)
    actual_percents = np.where(actual_percents == 0, 1e-8, actual_percents)
    psi = np.sum((actual_percents - expected_percents) * np.log(actual_percents / expected_percents))
    return psi

def recalibrate_probabilities(model, X_val, y_val, method='isotonic'):
    """
    Recalibrate model probabilities using Platt scaling (sigmoid) or isotonic regression.
    Returns a fitted CalibratedClassifierCV.
    """
    if method == 'isotonic':
        calibrator = CalibratedClassifierCV(model, method='isotonic', cv='prefit')
    else:
        calibrator = CalibratedClassifierCV(model, method='sigmoid', cv='prefit')
    calibrator.fit(X_val, y_val)
    return calibrator

def track_feature_importance(model, feature_names):
    """
    Track feature importance for a model. Returns a dict {feature: importance}.
    Supports tree-based models and linear models.
    """
    if hasattr(model, 'feature_importances_'):
        return dict(zip(feature_names, model.feature_importances_))
    elif hasattr(model, 'coef_'):
        # For multiclass, take mean absolute value
        coefs = model.coef_
        if coefs.ndim == 1:
            importances = np.abs(coefs)
        else:
            importances = np.mean(np.abs(coefs), axis=0)
        return dict(zip(feature_names, importances))
    else:
        return {}

# --- Longevity Analysis Framework ---

class LongevityAnalyzer:
    """
    Framework for temporal feature tracking, drift detection, calibration, and feature relevance monitoring.
    """
    def __init__(self, baseline_df=None, feature_names=None):
        self.baseline_df = baseline_df.copy() if baseline_df is not None else None
        self.feature_names = feature_names
        self.baseline_distributions = {}
        self.drift_history = []
        self.psi_history = []
        self.feature_importance_history = []
        if self.baseline_df is not None and self.feature_names is not None:
            self._compute_baseline_distributions()

    def _compute_baseline_distributions(self):
        for col in self.feature_names:
            values = self.baseline_df[col].dropna()
            hist, _ = np.histogram(values, bins=20, density=True)
            self.baseline_distributions[col] = hist / (np.sum(hist) + 1e-8)

    def track_feature_distributions(self, new_df):
        """
        Track and compare feature distributions in new_df to baseline.
        Returns dict of KL divergences and PSI values for each feature.
        """
        kl_results = {}
        psi_results = {}
        for col in self.feature_names:
            if col not in new_df.columns or col not in self.baseline_distributions:
                continue
            new_values = new_df[col].dropna()
            if new_values.empty:
                continue
            new_hist, _ = np.histogram(new_values, bins=20, density=True)
            new_hist = new_hist / (np.sum(new_hist) + 1e-8)
            kl = calculate_kl_divergence(self.baseline_distributions[col], new_hist)
            psi = calculate_psi(self.baseline_df[col], new_values)
            kl_results[col] = kl
            psi_results[col] = psi
        self.drift_history.append(kl_results)
        self.psi_history.append(psi_results)
        return kl_results, psi_results

    def detect_drift(self, kl_threshold=0.1, psi_threshold=0.1):
        """
        Detect features with drift above thresholds.
        Returns list of features with significant drift.
        """
        if not self.drift_history or not self.psi_history:
            return []
        latest_kl = self.drift_history[-1]
        latest_psi = self.psi_history[-1]
        drifted_features = [
            col for col in self.feature_names
            if (latest_kl.get(col, 0) > kl_threshold) or (latest_psi.get(col, 0) > psi_threshold)
        ]
        return drifted_features

    def update_baseline(self, new_df):
        """
        Update baseline distributions with new data (e.g., after retraining).
        """
        self.baseline_df = new_df.copy()
        self._compute_baseline_distributions()

    def track_feature_relevance(self, model):
        """
        Track and store feature importance for the current model.
        """
        importances = track_feature_importance(model, self.feature_names)
        self.feature_importance_history.append(importances)
        return importances

    def get_feature_importance_trends(self, window=5):
        """
        Get rolling window trends of feature importance.
        Returns dict: {feature: [history]}
        """
        trends = {f: [] for f in self.feature_names}
        for history in self.feature_importance_history[-window:]:
            for f in self.feature_names:
                trends[f].append(history.get(f, 0))
        return trends