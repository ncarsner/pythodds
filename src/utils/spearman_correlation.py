#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import sys
from typing import Optional, Sequence

from scipy.stats import norm
from scipy.stats import t as t_dist

"""Command-line utility to compute Spearman rank correlation coefficient.

The Spearman rank correlation coefficient (ρ) measures the monotonic relationship
between two variables by ranking the data first. Values range from -1 to 1:
  - ρ = 1: perfect monotonically increasing relationship
  - ρ = 0: no monotonic relationship
  - ρ = -1: perfect monotonically decreasing relationship

This tool computes:
  - Spearman's ρ (rank correlation coefficient)
  - ρ² (coefficient of determination for ranks)
  - t-statistic and p-value for testing H₀: ρ = 0
  - Confidence interval for ρ (population correlation)

Usage examples:
  # From command-line values
  python spearman_correlation.py --x 1,2,3,4,10 --y 2,3,5,6,20

  # From CSV file
  python spearman_correlation.py --file survey.csv --x-col satisfaction --y-col loyalty

  # With hypothesis test at α=0.01
  python spearman_correlation.py --x 100,150,120,180,200 --y 5,3,4,2,1 --alpha 0.01

  # One-tailed test with rank display
  python spearman_correlation.py --x 1,2,3,4,5 --y 2,4,5,4,5 --alpha 0.05 --sided one --show-ranks
"""


def mean(values: Sequence[float]) -> float:
    """Compute the arithmetic mean of a sequence of values."""
    if not values:
        raise ValueError("Cannot compute mean of empty list")
    return sum(values) / len(values)


def rank_data(values: Sequence[float]) -> list[float]:
    """Assign ranks to values, handling ties by averaging ranks.

    Args:
        values: Sequence of numeric values

    Returns:
        List of ranks (1-based), with tied values receiving average rank
    """
    n = len(values)

    # Create list of (value, original_index) pairs and sort by value
    indexed_values = [(val, idx) for idx, val in enumerate(values)]
    indexed_values.sort(key=lambda x: x[0])

    # Assign ranks, handling ties
    ranks = [0.0] * n
    i = 0

    while i < n:
        # Find all values equal to current value (tie group)
        j = i
        while j < n and indexed_values[j][0] == indexed_values[i][0]:
            j += 1

        # Average rank for this tie group
        # Ranks are 1-based: (i+1) to j inclusive
        avg_rank = sum(range(i + 1, j + 1)) / (j - i)

        # Assign average rank to all tied values
        for k in range(i, j):
            original_idx = indexed_values[k][1]
            ranks[original_idx] = avg_rank

        i = j

    return ranks


def pearson_r_from_ranks(rank_x: Sequence[float], rank_y: Sequence[float]) -> float:
    """Compute Pearson correlation coefficient between ranks.

    This is the core of Spearman's ρ calculation.

    Returns r in [-1, 1]
    """
    if len(rank_x) != len(rank_y):
        raise ValueError("rank_x and rank_y must have the same length")
    if len(rank_x) < 2:
        raise ValueError("Need at least 2 data points")

    mean_x = mean(rank_x)
    mean_y = mean(rank_y)

    # Compute deviations
    dev_x = [xi - mean_x for xi in rank_x]
    dev_y = [yi - mean_y for yi in rank_y]

    # Compute covariance and variances
    cov_xy = sum(dx * dy for dx, dy in zip(dev_x, dev_y))
    var_x = sum(dx * dx for dx in dev_x)
    var_y = sum(dy * dy for dy in dev_y)

    # Handle zero variance (all same rank - shouldn't happen with proper ranking)
    if var_x == 0 or var_y == 0:
        return 0.0

    r = cov_xy / math.sqrt(var_x * var_y)

    # Clamp to [-1, 1] to handle floating-point errors
    return max(-1.0, min(1.0, r))


def spearman_rho(x: Sequence[float], y: Sequence[float]) -> float:
    """Compute Spearman rank correlation coefficient between x and y.

    Args:
        x: First variable values
        y: Second variable values

    Returns:
        Spearman's ρ in [-1, 1]
    """
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    if len(x) < 2:
        raise ValueError("Need at least 2 data points")

    # Rank the data
    rank_x = rank_data(x)
    rank_y = rank_data(y)

    # Compute Pearson correlation of ranks
    return pearson_r_from_ranks(rank_x, rank_y)


def correlation_t_statistic(rho: float, n: int) -> float:
    """Compute t-statistic for testing H₀: ρ = 0.

    t = ρ * sqrt(n - 2) / sqrt(1 - ρ²)

    Follows t-distribution with df = n - 2.
    """
    if n < 3:
        raise ValueError("Need at least 3 data points for t-test")

    # Handle ρ = ±1 case (perfect correlation)
    if abs(rho) >= 1.0:
        return float("inf") if rho > 0 else float("-inf")

    df = n - 2
    t = rho * math.sqrt(df) / math.sqrt(1 - rho * rho)
    return t


def t_cdf(t: float, df: int) -> float:
    """CDF of Student's t-distribution.

    Uses scipy.stats.t for numerically robust implementation.
    """
    if df < 1:
        raise ValueError("Degrees of freedom must be at least 1")
    return float(t_dist.cdf(t, df))


def standard_normal_cdf(z: float) -> float:
    """CDF of standard normal distribution using error function."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def correlation_p_value(rho: float, n: int, sided: str = "two") -> float:
    """Compute p-value for correlation test.

    Args:
        rho: Spearman correlation coefficient
        n: Sample size
        sided: "one" for one-tailed, "two" for two-tailed test

    Returns:
        p-value for testing H₀: ρ = 0
    """
    t = correlation_t_statistic(rho, n)
    df = n - 2

    # Handle infinite t (perfect correlation)
    if math.isinf(t):
        return 0.0

    # Compute two-tailed p-value: P(|T| > |t|)
    p_upper = 1.0 - t_cdf(abs(t), df)

    if sided == "two":
        return 2.0 * p_upper
    else:  # one-tailed
        if t > 0:
            return p_upper
        else:
            return 1.0 - p_upper


def fisher_z_transform(rho: float) -> float:
    """Fisher Z-transformation for correlation coefficient.

    Z = 0.5 * ln((1 + ρ) / (1 - ρ)) = arctanh(ρ)
    """
    if abs(rho) >= 1.0:
        return float("inf") if rho > 0 else float("-inf")
    return 0.5 * math.log((1.0 + rho) / (1.0 - rho))


def inverse_fisher_z(z: float) -> float:
    """Inverse Fisher Z-transformation.

    ρ = (e^(2z) - 1) / (e^(2z) + 1) = tanh(z)
    """
    if math.isinf(z):
        return 1.0 if z > 0 else -1.0
    return math.tanh(z)


def inverse_normal_cdf(p: float) -> float:
    """Inverse CDF of standard normal distribution (quantile function).

    Uses scipy.stats.norm.ppf for numerically robust implementation.
    """
    if p <= 0.0 or p >= 1.0:
        raise ValueError("p must be between 0 and 1")
    return float(norm.ppf(p))


def correlation_confidence_interval(
    rho: float, n: int, alpha: float = 0.05
) -> tuple[float, float]:
    """Compute confidence interval for population correlation ρ.

    Uses Fisher Z-transformation:
    1. Transform ρ to Z
    2. Compute CI for Z using normal approximation
    3. Transform back to ρ scale

    Args:
        rho: Sample correlation coefficient
        n: Sample size
        alpha: Significance level (default 0.05 for 95% CI)

    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    if n < 4:
        # CI not meaningful for very small samples
        return (-1.0, 1.0)

    # Handle perfect correlation specially
    if abs(rho) >= 0.9999:
        # For near-perfect correlation, CI is essentially the value itself
        if rho > 0:
            return (0.9, 1.0)
        else:
            return (-1.0, -0.9)

    # Fisher Z-transformation
    z = fisher_z_transform(rho)

    # Standard error of Z
    se_z = 1.0 / math.sqrt(n - 3)

    # Critical value from standard normal (two-tailed)
    z_crit = inverse_normal_cdf(1.0 - alpha / 2.0)

    # CI for Z
    z_lower = z - z_crit * se_z
    z_upper = z + z_crit * se_z

    # Transform back to ρ scale
    rho_lower = inverse_fisher_z(z_lower)
    rho_upper = inverse_fisher_z(z_upper)

    # Clamp to [-1, 1]
    rho_lower = max(-1.0, min(1.0, rho_lower))
    rho_upper = max(-1.0, min(1.0, rho_upper))

    return (rho_lower, rho_upper)


def parse_csv_file(
    filepath: str, x_col: str, y_col: str
) -> tuple[list[float], list[float]]:
    """Parse x and y values from a CSV file.

    Args:
        filepath: Path to CSV file
        x_col: Column name for x values
        y_col: Column name for y values

    Returns:
        Tuple of (x_values, y_values)
    """
    x_values = []
    y_values = []

    try:
        with open(filepath, "r") as f:
            reader = csv.DictReader(f)

            # Check if columns exist
            if reader.fieldnames is None:
                raise ValueError("CSV file appears to be empty or has no header")

            if x_col not in reader.fieldnames:
                raise ValueError(
                    f"Column '{x_col}' not found in CSV. Available: {reader.fieldnames}"
                )
            if y_col not in reader.fieldnames:
                raise ValueError(
                    f"Column '{y_col}' not found in CSV. Available: {reader.fieldnames}"
                )

            for row in reader:
                try:
                    x_values.append(float(row[x_col]))
                    y_values.append(float(row[y_col]))
                except ValueError as e:
                    raise ValueError(f"Failed to parse numeric value: {e}")

    except FileNotFoundError:
        raise ValueError(f"File not found: {filepath}")
    except Exception as e:
        raise ValueError(f"Error reading CSV file: {e}")

    return x_values, y_values


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compute Spearman rank correlation coefficient and hypothesis test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # From command-line values
  %(prog)s --x 1,2,3,4,10 --y 2,3,5,6,20

  # From CSV file
  %(prog)s --file survey.csv --x-col satisfaction --y-col loyalty

  # With hypothesis test
  %(prog)s --x 100,150,120,180,200 --y 5,3,4,2,1 --alpha 0.01

  # One-tailed test with rank display
  %(prog)s --x 1,2,3,4,5 --y 2,4,5,4,5 --alpha 0.05 --sided one --show-ranks
        """,
    )

    # Input methods (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--x",
        type=str,
        help="comma-separated x values (use with --y)",
    )
    input_group.add_argument(
        "--file",
        type=str,
        help="CSV file path (use with --x-col and --y-col)",
    )

    parser.add_argument(
        "--y",
        type=str,
        help="comma-separated y values (required with --x)",
    )
    parser.add_argument(
        "--x-col",
        type=str,
        help="column name for x values (required with --file)",
    )
    parser.add_argument(
        "--y-col",
        type=str,
        help="column name for y values (required with --file)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="significance level for hypothesis test and CI (e.g., 0.05)",
    )
    parser.add_argument(
        "--sided",
        choices=["one", "two"],
        default="two",
        help="one-tailed or two-tailed test (default: two)",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=6,
        help="decimal places for output (default: 6)",
    )
    parser.add_argument(
        "--show-ranks",
        action="store_true",
        help="display ranked data for inspection",
    )

    args = parser.parse_args(argv)

    # Validate argument combinations
    if args.x is not None and args.y is None:
        parser.error("--y is required when using --x")
    if args.file is not None and (args.x_col is None or args.y_col is None):
        parser.error("--x-col and --y-col are required when using --file")

    return args


def format_number(x: float, precision: int) -> str:
    """Format a number with specified precision."""
    fmt = f"{{:.{precision}f}}"
    return fmt.format(x)


def interpret_correlation(rho: float) -> str:
    """Provide interpretation of correlation strength."""
    abs_rho = abs(rho)
    if abs_rho >= 0.9:
        strength = "very strong"
    elif abs_rho >= 0.7:
        strength = "strong"
    elif abs_rho >= 0.5:
        strength = "moderate"
    elif abs_rho >= 0.3:
        strength = "weak"
    else:
        strength = "very weak"

    direction = "positive" if rho >= 0 else "negative"
    return f"{strength} {direction}"


def main(argv: Optional[list[str]] = None) -> int:
    """Main entry point for CLI."""
    args = parse_args(argv)

    # Parse input data
    try:
        if args.x is not None:
            # Parse from command-line values
            x_values = [float(v.strip()) for v in args.x.split(",")]
            y_values = [float(v.strip()) for v in args.y.split(",")]
        else:
            # Parse from CSV file
            x_values, y_values = parse_csv_file(args.file, args.x_col, args.y_col)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2

    # Validate data
    if len(x_values) != len(y_values):
        print("Error: x and y must have the same number of values", file=sys.stderr)
        return 2

    if len(x_values) < 2:
        print("Error: Need at least 2 data points", file=sys.stderr)
        return 2

    n = len(x_values)
    precision = args.precision

    # Compute ranks if requested for display
    if args.show_ranks:
        rank_x = rank_data(x_values)
        rank_y = rank_data(y_values)

        print("Data and Ranks:")
        print(f"{'Index':>6} {'X':>10} {'Rank(X)':>10} {'Y':>10} {'Rank(Y)':>10}")
        print("-" * 56)
        for i in range(n):
            print(
                f"{i+1:>6} {x_values[i]:>10.3f} {rank_x[i]:>10.1f} {y_values[i]:>10.3f} {rank_y[i]:>10.1f}"
            )
        print()

    # Compute correlation
    try:
        rho = spearman_rho(x_values, y_values)
        rho_squared = rho * rho
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2

    # Print basic results
    print(f"Sample size (n):               {n}")
    print(f"Spearman's ρ:                  {format_number(rho, precision)}")
    print(f"ρ² (coefficient of determination): {format_number(rho_squared, precision)}")
    print(f"Interpretation:                {interpret_correlation(rho)}")

    # Compute hypothesis test if alpha is provided
    if args.alpha is not None:
        alpha = args.alpha

        if not (0.0 < alpha < 1.0):
            print("Error: --alpha must be between 0 and 1", file=sys.stderr)
            return 2

        if n < 3:
            print(
                "Error: Need at least 3 data points for hypothesis test",
                file=sys.stderr,
            )
            return 2

        try:
            t_stat = correlation_t_statistic(rho, n)
            p_value = correlation_p_value(rho, n, args.sided)
            ci_lower, ci_upper = correlation_confidence_interval(rho, n, alpha)

            print(f"\nHypothesis test: H₀: ρ = 0 vs H₁: ρ ≠ 0 ({args.sided}-tailed)")
            print(f"t-statistic:                   {format_number(t_stat, precision)}")
            print(f"Degrees of freedom (df):       {n - 2}")
            print(f"p-value:                       {format_number(p_value, precision)}")
            print(f"Significance level (α):        {format_number(alpha, precision)}")

            significant = p_value < alpha
            print(
                f"Result:                        {'Significant' if significant else 'Not significant'} at α={alpha}"
            )

            if significant:
                print(
                    "Conclusion:                    Reject H₀. Evidence of correlation."
                )
            else:
                print(
                    "Conclusion:                    Fail to reject H₀. No significant correlation."
                )

            ci_pct = int((1.0 - alpha) * 100)
            print(f"\n{ci_pct}% Confidence interval for ρ:")
            print(
                f"Lower bound:                   {format_number(ci_lower, precision)}"
            )
            print(
                f"Upper bound:                   {format_number(ci_upper, precision)}"
            )
            print(
                f"Interval:                      [{format_number(ci_lower, precision)}, {format_number(ci_upper, precision)}]"
            )

        except ValueError as e:
            print(f"Error in hypothesis test: {e}", file=sys.stderr)
            return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
