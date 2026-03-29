#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import statistics
import sys
from collections.abc import Callable, Sequence
from typing import Optional

"""Command-line utility to compute bootstrap confidence intervals.

Usage examples:
  python -m src.utils.bootstrap_confidence_intervals --data 10 20 30 40 50 --stat mean --n-bootstrap 1000 --confidence 0.95
  echo "10 20 30 40 50" | python -m src.utils.bootstrap_confidence_intervals --stat median --n-bootstrap 5000
  python -m src.utils.bootstrap_confidence_intervals --data 1.2 3.4 5.6 7.8 --stat stdev --confidence 0.90

Computes bootstrap confidence intervals for a given statistic (mean, median, or stdev)
using the percentile method.
"""


def bootstrap_resample(
    data: list[float], n_bootstrap: int, stat_func: Callable[[list[float]], float]
) -> list[float]:
    """Generate bootstrap resamples and compute statistic for each.

    Args:
        data: Original data sample
        n_bootstrap: Number of bootstrap resamples to generate
        stat_func: Function to compute statistic (e.g., statistics.mean)

    Returns:
        List of computed statistics from bootstrap resamples
    """
    bootstrap_stats = []
    n = len(data)

    for _ in range(n_bootstrap):
        # Resample with replacement
        resample = [data[random.randint(0, n - 1)] for _ in range(n)]
        bootstrap_stats.append(stat_func(resample))

    return bootstrap_stats


def compute_confidence_interval(
    bootstrap_stats: Sequence[float], confidence: float
) -> tuple[float, float]:
    """Compute confidence interval using the percentile method.

    Args:
        bootstrap_stats: List of statistics from bootstrap resamples
        confidence: Confidence level (e.g., 0.95 for 95% CI)

    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    alpha = 1 - confidence
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    sorted_stats = sorted(bootstrap_stats)
    n = len(sorted_stats)

    lower_idx = int(lower_percentile / 100 * n)
    upper_idx = int(upper_percentile / 100 * n)

    # Clamp indices to valid range
    lower_idx = max(0, min(lower_idx, n - 1))
    upper_idx = max(0, min(upper_idx, n - 1))

    return sorted_stats[lower_idx], sorted_stats[upper_idx]


def get_stat_function(stat_name: str) -> Callable[[list[float]], float]:
    """Return the appropriate statistics function.

    Args:
        stat_name: Name of statistic ('mean', 'median', 'stdev')

    Returns:
        Callable function from statistics module

    Raises:
        ValueError: If stat_name is not recognized
    """
    stat_funcs = {
        "mean": statistics.mean,
        "median": statistics.median,
        "stdev": statistics.stdev,
    }

    if stat_name not in stat_funcs:
        raise ValueError(
            f"Unknown statistic: {stat_name}. Must be one of {list(stat_funcs.keys())}"
        )

    return stat_funcs[stat_name]


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute bootstrap confidence intervals for a statistic",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  bootci --data 10 20 30 40 50 --stat mean --n-bootstrap 1000 --confidence 0.95
  echo "10 20 30 40 50" | bootci --stat median
  bootci --data 1.2 3.4 5.6 7.8 --stat stdev --confidence 0.90
        """,
    )

    parser.add_argument(
        "--data",
        type=float,
        nargs="+",
        help="Data points (space-separated). If not provided, reads from stdin.",
    )
    parser.add_argument(
        "--stat",
        type=str,
        required=True,
        choices=["mean", "median", "stdev"],
        help="Statistic to compute (mean, median, or stdev)",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=10000,
        help="Number of bootstrap resamples (default: 10000)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="Confidence level (default: 0.95 for 95%% CI)",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=4,
        help="Decimal places for printed values (default: 4)",
    )

    return parser.parse_args(argv)


def read_stdin_data() -> list[float]:
    """Read numeric data from stdin.

    Returns:
        List of float values
    """
    data = []
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        # Split line and parse each value
        for value in line.split():
            try:
                data.append(float(value))
            except ValueError:
                print(f"Warning: Skipping non-numeric value: {value}", file=sys.stderr)
    return data


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    # Get data from args or stdin
    if args.data:
        data = args.data
    else:
        data = read_stdin_data()

    # Validate data
    if len(data) < 2:
        print("Error: Need at least 2 data points", file=sys.stderr)
        return 2

    # Validate confidence level
    if not (0.0 < args.confidence < 1.0):
        print(
            "Error: --confidence must be between 0 and 1 (exclusive)", file=sys.stderr
        )
        return 2

    # Validate n_bootstrap
    if args.n_bootstrap < 1:
        print("Error: --n-bootstrap must be at least 1", file=sys.stderr)
        return 2

    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)

    try:
        stat_func = get_stat_function(args.stat)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2

    # Compute original statistic
    try:
        original_stat = stat_func(data)
    except statistics.StatisticsError as e:
        print(f"Error computing statistic: {e}", file=sys.stderr)
        return 2

    # Perform bootstrap
    bootstrap_stats = bootstrap_resample(data, args.n_bootstrap, stat_func)

    # Compute confidence interval
    lower, upper = compute_confidence_interval(bootstrap_stats, args.confidence)

    # Format output
    precision = args.precision
    conf_pct = args.confidence * 100

    print(f"Data: n={len(data)}, statistic={args.stat}")
    print(f"Original {args.stat}: {original_stat:.{precision}f}")
    print(f"Bootstrap samples: {args.n_bootstrap}")
    print(
        f"{conf_pct:.1f}% Confidence Interval: [{lower:.{precision}f}, {upper:.{precision}f}]"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
