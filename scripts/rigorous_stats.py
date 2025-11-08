"""Rigorous Statistical Testing Utilities

Provides gold-standard statistical methods that can't be gamed:
- Permutation tests for p-values
- Bootstrap confidence intervals
- Model selection (AIC/BIC)
- Stability metrics
- Null test generators
"""

from __future__ import annotations

from typing import Callable, Tuple, Optional, List
import numpy as np
from dataclasses import dataclass
from scipy import stats


@dataclass
class PermutationTestResult:
    """Result from permutation test."""
    observed_statistic: float
    p_value: float
    null_distribution: np.ndarray
    n_permutations: int

    def is_significant(self, alpha: float = 0.05) -> bool:
        return self.p_value < alpha


@dataclass
class BootstrapCI:
    """Bootstrap confidence interval."""
    point_estimate: float
    ci_lower: float
    ci_upper: float
    confidence_level: float
    n_bootstrap: int

    def contains_zero(self) -> bool:
        """Check if CI contains zero (null hypothesis)."""
        return self.ci_lower <= 0 <= self.ci_upper


@dataclass
class ModelComparison:
    """Piecewise vs smooth model comparison."""
    break_point: Optional[float]
    smooth_aic: float
    piecewise_aic: float
    delta_aic: float
    delta_bic: float
    prefers_piecewise: bool

    def is_strong_evidence(self) -> bool:
        """Strong evidence requires ΔAIC > 10."""
        return abs(self.delta_aic) > 10


def permutation_test_correlation(
    x: np.ndarray,
    y: np.ndarray,
    n_permutations: int = 10000,
    seed: Optional[int] = None
) -> PermutationTestResult:
    """Permutation test for correlation coefficient.

    Tests H0: correlation = 0 by randomly permuting y labels.

    Args:
        x: First variable
        y: Second variable
        n_permutations: Number of random permutations
        seed: Random seed for reproducibility

    Returns:
        PermutationTestResult with p-value and null distribution
    """
    if seed is not None:
        np.random.seed(seed)

    # Observed statistic
    observed_r = np.corrcoef(x, y)[0, 1]

    # Generate null distribution
    null_dist = np.zeros(n_permutations)

    for i in range(n_permutations):
        # Permute y
        y_perm = np.random.permutation(y)
        # Compute correlation under null
        null_dist[i] = np.corrcoef(x, y_perm)[0, 1]

    # Two-tailed p-value: proportion as extreme as observed
    p_value = np.mean(np.abs(null_dist) >= np.abs(observed_r))

    return PermutationTestResult(
        observed_statistic=float(observed_r),
        p_value=float(p_value),
        null_distribution=null_dist,
        n_permutations=n_permutations,
    )


def bootstrap_ci(
    data: np.ndarray,
    statistic_fn: Callable[[np.ndarray], float],
    n_bootstrap: int = 10000,
    confidence_level: float = 0.95,
    seed: Optional[int] = None
) -> BootstrapCI:
    """Bootstrap confidence interval for arbitrary statistic.

    Args:
        data: Input data
        statistic_fn: Function to compute statistic (e.g., np.mean)
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level (e.g., 0.95 for 95% CI)
        seed: Random seed

    Returns:
        BootstrapCI with point estimate and intervals
    """
    if seed is not None:
        np.random.seed(seed)

    n = len(data)
    bootstrap_stats = np.zeros(n_bootstrap)

    for i in range(n_bootstrap):
        # Resample with replacement
        sample = np.random.choice(data, size=n, replace=True)
        bootstrap_stats[i] = statistic_fn(sample)

    # Percentile method
    alpha = 1 - confidence_level
    ci_lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))

    point_estimate = statistic_fn(data)

    return BootstrapCI(
        point_estimate=float(point_estimate),
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        confidence_level=confidence_level,
        n_bootstrap=n_bootstrap,
    )


def fit_piecewise_linear(
    x: np.ndarray,
    y: np.ndarray,
    break_candidates: Optional[np.ndarray] = None
) -> Tuple[Optional[float], float, np.ndarray]:
    """Fit piecewise linear model with one breakpoint.

    Searches over breakpoint locations and returns best fit.

    Args:
        x: Independent variable
        y: Dependent variable
        break_candidates: Candidate breakpoints (if None, use middle 60%)

    Returns:
        (best_break, best_sse, fitted_values)
    """
    if break_candidates is None:
        # Use middle 60% as candidates
        x_sorted = np.sort(x)
        n = len(x_sorted)
        start_idx = int(0.2 * n)
        end_idx = int(0.8 * n)
        break_candidates = x_sorted[start_idx:end_idx]

    best_break = None
    best_sse = np.inf
    best_fitted = None

    for break_point in break_candidates:
        # Split data
        left_mask = x <= break_point
        right_mask = x > break_point

        if np.sum(left_mask) < 2 or np.sum(right_mask) < 2:
            continue

        # Fit two linear segments
        try:
            # Left segment
            left_fit = np.polyfit(x[left_mask], y[left_mask], deg=1)
            # Right segment
            right_fit = np.polyfit(x[right_mask], y[right_mask], deg=1)

            # Compute fitted values
            fitted = np.zeros_like(y)
            fitted[left_mask] = np.polyval(left_fit, x[left_mask])
            fitted[right_mask] = np.polyval(right_fit, x[right_mask])

            # SSE
            sse = np.sum((y - fitted) ** 2)

            if sse < best_sse:
                best_sse = sse
                best_break = break_point
                best_fitted = fitted

        except np.linalg.LinAlgError:
            continue

    return best_break, best_sse, best_fitted


def compare_piecewise_vs_smooth(
    x: np.ndarray,
    y: np.ndarray
) -> ModelComparison:
    """Compare piecewise vs smooth (single linear) model using AIC/BIC.

    Args:
        x: Independent variable
        y: Dependent variable

    Returns:
        ModelComparison with AIC/BIC deltas
    """
    n = len(x)

    # Fit smooth model (single linear regression)
    smooth_fit = np.polyfit(x, y, deg=1)
    smooth_pred = np.polyval(smooth_fit, x)
    smooth_sse = np.sum((y - smooth_pred) ** 2)
    smooth_k = 2  # slope + intercept

    smooth_aic = n * np.log(smooth_sse / n) + 2 * smooth_k
    smooth_bic = n * np.log(smooth_sse / n) + smooth_k * np.log(n)

    # Fit piecewise model
    break_point, piecewise_sse, piecewise_pred = fit_piecewise_linear(x, y)

    if break_point is None:
        # Piecewise fit failed, smooth wins
        return ModelComparison(
            break_point=None,
            smooth_aic=smooth_aic,
            piecewise_aic=np.inf,
            delta_aic=np.inf,
            delta_bic=np.inf,
            prefers_piecewise=False,
        )

    piecewise_k = 4  # 2 slopes + 2 intercepts

    piecewise_aic = n * np.log(piecewise_sse / n) + 2 * piecewise_k
    piecewise_bic = n * np.log(piecewise_sse / n) + piecewise_k * np.log(n)

    delta_aic = smooth_aic - piecewise_aic  # Positive = piecewise better
    delta_bic = smooth_bic - piecewise_bic

    return ModelComparison(
        break_point=float(break_point),
        smooth_aic=float(smooth_aic),
        piecewise_aic=float(piecewise_aic),
        delta_aic=float(delta_aic),
        delta_bic=float(delta_bic),
        prefers_piecewise=delta_aic > 0,
    )


def stability_rate(
    results: List[float],
    threshold: float,
    above: bool = True
) -> float:
    """Compute stability rate (fraction above/below threshold).

    Args:
        results: List of metric values across trials
        threshold: Success threshold
        above: If True, count values >= threshold. If False, count values <= threshold.

    Returns:
        Fraction of results meeting criterion
    """
    results_arr = np.array(results)

    if above:
        success = np.sum(results_arr >= threshold)
    else:
        success = np.sum(results_arr <= threshold)

    return float(success / len(results_arr))


def shuffle_labels_test(
    x: np.ndarray,
    y: np.ndarray,
    detector_fn: Callable[[np.ndarray, np.ndarray], float],
    n_shuffles: int = 100,
    seed: Optional[int] = None
) -> Tuple[float, float, float]:
    """Null test: shuffle x labels and check if detector goes quiet.

    A good detector should find no signal when labels are shuffled.

    Args:
        x: Independent variable (will be shuffled)
        y: Dependent variable
        detector_fn: Function that returns detection statistic (e.g., ΔAIC)
        n_shuffles: Number of shuffle trials
        seed: Random seed

    Returns:
        (observed_statistic, mean_null_statistic, false_positive_rate)
    """
    if seed is not None:
        np.random.seed(seed)

    # Observed statistic (unshuffled)
    observed = detector_fn(x, y)

    # Null distribution (shuffled labels)
    null_stats = []
    for _ in range(n_shuffles):
        x_shuffled = np.random.permutation(x)
        null_stat = detector_fn(x_shuffled, y)
        null_stats.append(null_stat)

    null_stats = np.array(null_stats)

    # False positive rate: fraction of shuffles that "detect" signal
    # (depends on what detector returns, e.g., ΔAIC > 10)
    # For simplicity, use mean of null distribution
    mean_null = np.mean(null_stats)

    # FPR: fraction as extreme as observed
    fpr = np.mean(np.abs(null_stats) >= np.abs(observed))

    return float(observed), float(mean_null), float(fpr)


def scale_invariance_test(
    x: np.ndarray,
    y: np.ndarray,
    analysis_fn: Callable[[np.ndarray, np.ndarray], dict],
    scales: List[float] = [0.5, 1.0, 2.0, 10.0]
) -> dict:
    """Test if analysis conclusions are invariant to scaling.

    Scales y by different factors and checks if conclusions flip.

    Args:
        x: Independent variable
        y: Dependent variable
        analysis_fn: Function that returns dict with analysis results
        scales: Scale factors to test

    Returns:
        Dict mapping scale -> analysis results
    """
    results = {}

    for scale in scales:
        y_scaled = y * scale
        result = analysis_fn(x, y_scaled)
        results[scale] = result

    return results


def inject_synthetic_step_test(
    x: np.ndarray,
    y: np.ndarray,
    injected_break: float,
    injected_jump: float,
    detector_fn: Callable[[np.ndarray, np.ndarray], Optional[float]]
) -> Tuple[bool, float, Optional[float]]:
    """Adversarial test: inject known step, check if detector finds it.

    Args:
        x: Independent variable
        y: Dependent variable
        injected_break: Location of injected step
        injected_jump: Size of step
        detector_fn: Function that detects breakpoints (returns detected break or None)

    Returns:
        (detected_correctly, injected_break, detected_break)
    """
    # Inject step
    y_injected = y.copy()
    step_mask = x >= injected_break
    y_injected[step_mask] += injected_jump

    # Run detector
    detected_break = detector_fn(x, y_injected)

    # Check if within tolerance (±0.5 * range)
    x_range = np.max(x) - np.min(x)
    tolerance = 0.1 * x_range

    if detected_break is None:
        detected_correctly = False
    else:
        detected_correctly = abs(detected_break - injected_break) <= tolerance

    return detected_correctly, float(injected_break), detected_break


def pre_register_tolerance(
    measurement_noise_std: float,
    n_measurements: int,
    confidence_level: float = 0.95
) -> float:
    """Pre-register conservation tolerance based on measurement uncertainty.

    For conservation claims, the tolerance should be set BEFORE seeing data,
    based on instrument noise and sample size.

    Args:
        measurement_noise_std: Standard deviation of measurement noise
        n_measurements: Number of independent measurements
        confidence_level: Confidence level for tolerance band

    Returns:
        Tolerance threshold (e.g., 0.5%)
    """
    # Standard error of the mean
    sem = measurement_noise_std / np.sqrt(n_measurements)

    # Critical value for confidence level
    z_crit = stats.norm.ppf((1 + confidence_level) / 2)

    # Tolerance band
    tolerance = z_crit * sem

    return float(tolerance)
