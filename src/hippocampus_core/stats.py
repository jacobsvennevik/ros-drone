"""Statistical utilities for aggregating experimental results."""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np


def compute_confidence_interval(
    data: np.ndarray, confidence: float = 0.95
) -> Tuple[float, float]:
    """Compute confidence interval for data.

    Parameters
    ----------
    data:
        Array of values.
    confidence:
        Confidence level (e.g., 0.95 for 95% CI).

    Returns
    -------
    Tuple[float, float]
        (lower_bound, upper_bound)
    """
    if len(data) == 0:
        return (0.0, 0.0)
    if len(data) == 1:
        return (float(data[0]), float(data[0]))

    alpha = 1.0 - confidence
    mean = np.mean(data)
    std = np.std(data, ddof=1)  # Sample standard deviation
    n = len(data)

    # Use t-distribution for small samples, normal for large
    if n < 30:
        try:
            from scipy import stats

            t_critical = stats.t.ppf(1 - alpha / 2, df=n - 1)
            margin = t_critical * std / np.sqrt(n)
        except ImportError:
            # Fallback to normal approximation if scipy not available
            z_critical = 1.96 if confidence == 0.95 else 2.576 if confidence == 0.99 else 1.645
            margin = z_critical * std / np.sqrt(n)
    else:
        # Normal approximation
        z_critical = 1.96 if confidence == 0.95 else 2.576 if confidence == 0.99 else 1.645
        margin = z_critical * std / np.sqrt(n)

    return (float(mean - margin), float(mean + margin))


def bootstrap_statistic(
    data: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
) -> Tuple[float, Tuple[float, float]]:
    """Compute bootstrap confidence interval for a statistic.

    Parameters
    ----------
    data:
        Array of values.
    statistic:
        Function to compute statistic (e.g., np.mean, np.median).
    n_bootstrap:
        Number of bootstrap samples.
    confidence:
        Confidence level.

    Returns
    -------
    Tuple[float, Tuple[float, float]]
        (statistic_value, (lower_ci, upper_ci))
    """
    if len(data) == 0:
        return (0.0, (0.0, 0.0))

    # Compute statistic on original data
    stat_value = statistic(data)

    # Bootstrap sampling
    bootstrap_stats = []
    rng = np.random.default_rng(42)  # Fixed seed for reproducibility

    for _ in range(n_bootstrap):
        sample = rng.choice(data, size=len(data), replace=True)
        bootstrap_stats.append(statistic(sample))

    bootstrap_stats = np.array(bootstrap_stats)

    # Compute confidence interval
    alpha = 1.0 - confidence
    lower_percentile = 100 * (alpha / 2)
    upper_percentile = 100 * (1 - alpha / 2)

    lower_ci = np.percentile(bootstrap_stats, lower_percentile)
    upper_ci = np.percentile(bootstrap_stats, upper_percentile)

    return (float(stat_value), (float(lower_ci), float(upper_ci)))


def aggregate_trials(
    results_list: List[Dict[str, Any]], keys: Optional[List[str]] = None
) -> Dict[str, Dict[str, float]]:
    """Aggregate statistics across multiple trial results.

    Parameters
    ----------
    results_list:
        List of result dictionaries from individual trials.
    keys:
        Optional list of keys to aggregate. If None, aggregates all numeric keys.

    Returns
    -------
    Dict[str, Dict[str, float]]
        Dictionary mapping metric names to statistics:
        {
            "metric_name": {
                "mean": float,
                "std": float,
                "median": float,
                "q25": float,
                "q75": float,
                "min": float,
                "max": float,
                "ci_lower": float,
                "ci_upper": float,
            }
        }
    """
    if not results_list:
        return {}

    # Determine which keys to aggregate
    if keys is None:
        # Find all keys that contain numeric arrays or single values
        all_keys = set()
        for result in results_list:
            for key, value in result.items():
                if isinstance(value, (int, float, np.number)):
                    all_keys.add(key)
                elif isinstance(value, (list, np.ndarray)):
                    if len(value) > 0 and isinstance(value[0], (int, float, np.number)):
                        all_keys.add(key)

        keys = sorted(all_keys)

    aggregated: Dict[str, Dict[str, float]] = {}

    for key in keys:
        # Extract values for this key across all trials
        values = []
        for result in results_list:
            if key in result:
                value = result[key]
                if isinstance(value, (int, float, np.number)):
                    values.append(float(value))
                elif isinstance(value, (list, np.ndarray)):
                    # For time series, take final value
                    if len(value) > 0:
                        values.append(float(value[-1]))

        if not values:
            continue

        values_array = np.array(values)

        # Compute statistics
        mean = float(np.mean(values_array))
        std = float(np.std(values_array, ddof=1))
        median = float(np.median(values_array))
        q25 = float(np.percentile(values_array, 25))
        q75 = float(np.percentile(values_array, 75))
        min_val = float(np.min(values_array))
        max_val = float(np.max(values_array))

        # Confidence interval
        ci_lower, ci_upper = compute_confidence_interval(values_array)

        aggregated[key] = {
            "mean": mean,
            "std": std,
            "median": median,
            "q25": q25,
            "q75": q75,
            "min": min_val,
            "max": max_val,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
        }

    return aggregated


def aggregate_time_series(
    results_list: List[Dict[str, Any]], time_key: str = "times"
) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Aggregate time series data across trials.

    Parameters
    ----------
    results_list:
        List of result dictionaries with time series.
    time_key:
        Key for time array.

    Returns
    -------
    Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]
        Dictionary mapping metric names to (times, means, stds) tuples.
        Times are aligned across trials.
    """
    if not results_list:
        return {}

    # Find common time points (use first trial's times as reference)
    if time_key not in results_list[0]:
        return {}

    reference_times = np.array(results_list[0][time_key])

    aggregated: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    # Get all metric keys (excluding time_key)
    metric_keys = set()
    for result in results_list:
        for key in result.keys():
            if key != time_key and isinstance(result[key], (list, np.ndarray)):
                metric_keys.add(key)

    for metric_key in metric_keys:
        # Interpolate each trial's data to reference times
        interpolated_values = []

        for result in results_list:
            if metric_key not in result:
                continue

            times = np.array(result[time_key])
            values = np.array(result[metric_key])

            if len(times) == 0 or len(values) == 0:
                continue

            # Interpolate to reference times
            if len(times) == 1:
                # Constant value
                interpolated = np.full_like(reference_times, values[0])
            else:
                # Linear interpolation
                interpolated = np.interp(reference_times, times, values)

            interpolated_values.append(interpolated)

        if not interpolated_values:
            continue

        # Stack and compute statistics
        values_matrix = np.array(interpolated_values)
        means = np.mean(values_matrix, axis=0)
        stds = np.std(values_matrix, axis=0, ddof=1)

        aggregated[metric_key] = (reference_times, means, stds)

    return aggregated


def compute_success_rate(
    results_list: List[Dict[str, Any]],
    condition: Callable[[Dict[str, Any]], bool],
) -> Tuple[float, int, int]:
    """Compute success rate based on a condition.

    Parameters
    ----------
    results_list:
        List of result dictionaries.
    condition:
        Function that returns True for successful trials.

    Returns
    -------
    Tuple[float, int, int]
        (success_rate, num_successful, total)
    """
    if not results_list:
        return (0.0, 0, 0)

    successful = sum(1 for result in results_list if condition(result))
    total = len(results_list)
    rate = successful / total if total > 0 else 0.0

    return (rate, successful, total)


def format_statistic(value: float, std: Optional[float] = None, precision: int = 3) -> str:
    """Format a statistic with optional standard deviation.

    Parameters
    ----------
    value:
        Mean or median value.
    std:
        Optional standard deviation.
    precision:
        Number of decimal places.

    Returns
    -------
    str
        Formatted string like "1.234 ± 0.056" or "1.234"
    """
    if std is not None:
        return f"{value:.{precision}f} ± {std:.{precision}f}"
    else:
        return f"{value:.{precision}f}"

