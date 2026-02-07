"""
Statistical testing utilities for comparing models across subjects.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy import stats


def wilcoxon_test(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    alternative: str = "two-sided",
) -> Dict[str, float]:
    """
    Wilcoxon signed-rank test for paired subject-level scores.

    Args:
        scores_a: Accuracy array for model A (one per subject)
        scores_b: Accuracy array for model B (one per subject)
        alternative: "two-sided", "greater", or "less"

    Returns:
        Dict with statistic, p_value, effect_size (r = Z / sqrt(N))
    """
    diff = np.array(scores_a) - np.array(scores_b)

    # Remove zeros (ties)
    nonzero = diff[diff != 0]
    n = len(nonzero)

    if n < 5:
        return {
            "statistic": np.nan,
            "p_value": np.nan,
            "effect_size_r": np.nan,
            "n_subjects": n,
            "warning": "Too few non-tied pairs for reliable test",
        }

    result = stats.wilcoxon(scores_a, scores_b, alternative=alternative)

    # Effect size: r = Z / sqrt(N)
    # Approximate Z from p-value
    if result.pvalue > 0:
        z = stats.norm.ppf(1 - result.pvalue / 2)
    else:
        z = np.inf
    effect_size_r = z / np.sqrt(n)

    return {
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "effect_size_r": float(effect_size_r),
        "n_subjects": n,
        "mean_diff": float(np.mean(diff)),
        "std_diff": float(np.std(diff, ddof=1)),
    }


def aggregate_results(
    log_dir: str, experiment_prefix: str = ""
) -> Dict[str, List[float]]:
    """
    Aggregate results from experiment JSON logs.

    Args:
        log_dir: Directory containing .json log files.
        experiment_prefix: Filter logs by prefix (e.g., "hybrid_cnn_se_").

    Returns:
        Dict mapping metric_name → list of per-subject values.
    """
    log_dir = Path(log_dir)
    results = {}

    for fpath in sorted(log_dir.glob("*.json")):
        if experiment_prefix and not fpath.stem.startswith(experiment_prefix):
            continue

        with open(fpath) as f:
            log = json.load(f)

        fm = log.get("final_metrics", {})
        for key in ["accuracy", "balanced_accuracy", "roc_auc"]:
            if key in fm:
                results.setdefault(key, []).append(fm[key])

    return results


def compare_models(
    log_dir: str,
    model_a_prefix: str,
    model_b_prefix: str,
    metric: str = "accuracy",
) -> Dict:
    """
    Compare two models using Wilcoxon signed-rank test.

    Args:
        log_dir: Path to results/logs directory.
        model_a_prefix: Prefix for model A logs (e.g., "hybrid_cnn_se").
        model_b_prefix: Prefix for model B logs (e.g., "eegnet").
        metric: Which metric to compare.

    Returns:
        Dict with test results and descriptive stats.
    """
    results_a = aggregate_results(log_dir, model_a_prefix)
    results_b = aggregate_results(log_dir, model_b_prefix)

    if metric not in results_a or metric not in results_b:
        return {"error": f"Metric '{metric}' not found in logs."}

    scores_a = np.array(results_a[metric])
    scores_b = np.array(results_b[metric])

    n = min(len(scores_a), len(scores_b))
    scores_a = scores_a[:n]
    scores_b = scores_b[:n]

    test_result = wilcoxon_test(scores_a, scores_b)
    test_result["model_a"] = model_a_prefix
    test_result["model_b"] = model_b_prefix
    test_result["metric"] = metric
    test_result["model_a_mean"] = float(scores_a.mean())
    test_result["model_a_std"] = float(scores_a.std(ddof=1))
    test_result["model_b_mean"] = float(scores_b.mean())
    test_result["model_b_std"] = float(scores_b.std(ddof=1))

    return test_result
