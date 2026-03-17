#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import sys
from typing import Optional, Sequence

"""Command-line utility to compute Pearson correlation coefficient.

The Pearson correlation coefficient (r) measures the linear relationship
between two continuous variables. Values range from -1 to 1:
  - r = 1: perfect positive linear relationship
  - r = 0: no linear relationship
  - r = -1: perfect negative linear relationship

This tool computes:
  - Pearson's r (correlation coefficient)
  - r² (coefficient of determination)
  - t-statistic and p-value for testing H₀: ρ = 0
  - Confidence interval for ρ (population correlation)

Usage examples:
  # From command-line values
  python pearson_correlation.py --x 1,2,3,4,5 --y 2.1,3.8,6.2,7.9,10.1

  # From CSV file
  python pearson_correlation.py --file data.csv --x-col height --y-col weight

  # With hypothesis test at α=0.05
  python pearson_correlation.py --x 10,20,30,40,50 --y 15,28,41,55,68 --alpha 0.05

  # One-tailed test
  python pearson_correlation.py --x 1,2,3,4,5 --y 2,4,5,4,5 --alpha 0.05 --sided one
"""


def mean(values: Sequence[float]) -> float:
    """Compute the arithmetic mean of a sequence of values."""
    if not values:
        raise ValueError("Cannot compute mean of empty list")
    return sum(values) / len(values)


def pearson_r(x: Sequence[float], y: Sequence[float]) -> float:
    """Compute Pearson correlation coefficient between x and y.

    Returns r in [-1, 1] where:
      r = Σ((xᵢ - x̄)(yᵢ - ȳ)) / sqrt(Σ(xᵢ - x̄)² * Σ(yᵢ - ȳ)²)
    """
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    if len(x) < 2:
        raise ValueError("Need at least 2 data points")

    # n = len(x)
    mean_x = mean(x)
    mean_y = mean(y)

    # Compute deviations
    dev_x = [xi - mean_x for xi in x]
    dev_y = [yi - mean_y for yi in y]

    # Compute covariance and variances
    cov_xy = sum(dx * dy for dx, dy in zip(dev_x, dev_y))
    var_x = sum(dx * dx for dx in dev_x)
    var_y = sum(dy * dy for dy in dev_y)

    # Handle zero variance (constant values)
    if var_x == 0 or var_y == 0:
        return 0.0 if var_x == 0 or var_y == 0 else float("nan")

    r = cov_xy / math.sqrt(var_x * var_y)

    # Clamp to [-1, 1] to handle floating-point errors
    return max(-1.0, min(1.0, r))


def correlation_t_statistic(r: float, n: int) -> float:
    """Compute t-statistic for testing H₀: ρ = 0.

    t = r * sqrt(n - 2) / sqrt(1 - r²)

    Follows t-distribution with df = n - 2.
    """
    if n < 3:
        raise ValueError("Need at least 3 data points for t-test")

    # Handle r = ±1 case (perfect correlation)
    if abs(r) >= 1.0:
        return float("inf") if r > 0 else float("-inf")

    df = n - 2
    t = r * math.sqrt(df) / math.sqrt(1 - r * r)
    return t


def t_cdf(t: float, df: int) -> float:
    """Approximate CDF of Student's t-distribution.

    Uses the relationship between t-distribution and incomplete beta function.
    For two-tailed test, we need P(|T| > |t|) = 2 * P(T > |t|).
    """
    if df < 1:
        raise ValueError("Degrees of freedom must be at least 1")

    # Special cases
    if math.isinf(t):
        return 1.0 if t > 0 else 0.0

    # x = df / (df + t * t)

    # Use regularized incomplete beta function
    # P(T ≤ t) = 1/2 + (sgn(t) * (1 - I_x(df/2, 1/2)) / 2)
    # where I_x is the regularized incomplete beta

    # For simplicity, use a normal approximation for large df
    if df > 30:
        # For large df, t-distribution approaches standard normal
        return standard_normal_cdf(t)

    # For small df, use Series approximation
    # This is a simplified implementation
    p = math.atan2(t, math.sqrt(df)) / math.pi
    if t >= 0:
        return 0.5 + p
    else:
        return 0.5 + p


def standard_normal_cdf(z: float) -> float:
    """CDF of standard normal distribution using error function."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def correlation_p_value(r: float, n: int, sided: str = "two") -> float:
    """Compute p-value for correlation test.

    Args:
        r: Pearson correlation coefficient
        n: Sample size
        sided: "one" for one-tailed, "two" for two-tailed test

    Returns:
        p-value for testing H₀: ρ = 0
    """
    t = correlation_t_statistic(r, n)
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


def fisher_z_transform(r: float) -> float:
    """Fisher Z-transformation for correlation coefficient.

    Z = 0.5 * ln((1 + r) / (1 - r)) = arctanh(r)
    """
    if abs(r) >= 1.0:
        return float("inf") if r > 0 else float("-inf")
    return 0.5 * math.log((1.0 + r) / (1.0 - r))


def inverse_fisher_z(z: float) -> float:
    """Inverse Fisher Z-transformation.

    r = (e^(2z) - 1) / (e^(2z) + 1) = tanh(z)
    """
    if math.isinf(z):
        return 1.0 if z > 0 else -1.0
    return math.tanh(z)


# Coefficients for Beasley-Springer-Moro rational approximation
# Used by inverse_normal_cdf() - defined at module level to avoid recreation
_INVERSE_NORMAL_A = (
    -3.969683028665376e1,
    2.209460984245205e2,
    -2.759285104469687e2,
    1.383577518672690e2,
    -3.066479806614716e1,
    2.506628277459239e0,
)
_INVERSE_NORMAL_B = (
    -5.447609879822406e1,
    1.615858368580409e2,
    -1.556989798598866e2,
    6.680131188771972e1,
    -1.328068155288572e1,
)
_INVERSE_NORMAL_C = (
    -7.784894002430293e-3,
    -3.223964580411365e-1,
    -2.400758277161838e0,
    -2.549732539343734e0,
    4.374664141464968e0,
    2.938163982698783e0,
)
_INVERSE_NORMAL_D = (
    7.784695709041462e-3,
    3.224671290700398e-1,
    2.445134137142996e0,
    3.754408661907416e0,
)

# Break-points for rational approximation regions
_P_LOW = 0.02425
_P_HIGH = 1.0 - _P_LOW


def inverse_normal_cdf(p: float) -> float:
    """Approximate inverse of standard normal CDF using rational approximation.

    This is an implementation of the Beasley-Springer-Moro algorithm.
    Accurate to about 1e-9 for 0 < p < 1.
    """
    if p <= 0.0 or p >= 1.0:
        raise ValueError("p must be between 0 and 1")

    # Rational approximation for lower region
    if p < _P_LOW:
        q = math.sqrt(-2.0 * math.log(p))
        x = (
            (
                (
                    (
                        (_INVERSE_NORMAL_C[0] * q + _INVERSE_NORMAL_C[1]) * q
                        + _INVERSE_NORMAL_C[2]
                    )
                    * q
                    + _INVERSE_NORMAL_C[3]
                )
                * q
                + _INVERSE_NORMAL_C[4]
            )
            * q
            + _INVERSE_NORMAL_C[5]
        ) / (
            (
                (
                    (_INVERSE_NORMAL_D[0] * q + _INVERSE_NORMAL_D[1]) * q
                    + _INVERSE_NORMAL_D[2]
                )
                * q
                + _INVERSE_NORMAL_D[3]
            )
            * q
            + 1.0
        )
        return x

    # Rational approximation for upper region
    if p > _P_HIGH:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        x = -(
            (
                (
                    (
                        (
                            (_INVERSE_NORMAL_C[0] * q + _INVERSE_NORMAL_C[1]) * q
                            + _INVERSE_NORMAL_C[2]
                        )
                        * q
                        + _INVERSE_NORMAL_C[3]
                    )
                    * q
                    + _INVERSE_NORMAL_C[4]
                )
                * q
                + _INVERSE_NORMAL_C[5]
            )
            / (
                (
                    (
                        (_INVERSE_NORMAL_D[0] * q + _INVERSE_NORMAL_D[1]) * q
                        + _INVERSE_NORMAL_D[2]
                    )
                    * q
                    + _INVERSE_NORMAL_D[3]
                )
                * q
                + 1.0
            )
        )
        return x

    # Rational approximation for central region
    q = p - 0.5
    r = q * q
    x = (
        (
            (
                (
                    (
                        (_INVERSE_NORMAL_A[0] * r + _INVERSE_NORMAL_A[1]) * r
                        + _INVERSE_NORMAL_A[2]
                    )
                    * r
                    + _INVERSE_NORMAL_A[3]
                )
                * r
                + _INVERSE_NORMAL_A[4]
            )
            * r
            + _INVERSE_NORMAL_A[5]
        )
        * q
    ) / (
        (
            (
                (
                    (_INVERSE_NORMAL_B[0] * r + _INVERSE_NORMAL_B[1]) * r
                    + _INVERSE_NORMAL_B[2]
                )
                * r
                + _INVERSE_NORMAL_B[3]
            )
            * r
            + _INVERSE_NORMAL_B[4]
        )
        * r
        + 1.0
    )

    return x


def correlation_confidence_interval(
    r: float, n: int, alpha: float = 0.05
) -> tuple[float, float]:
    """Compute confidence interval for population correlation ρ.

    Uses Fisher Z-transformation:
    1. Transform r to Z
    2. Compute CI for Z using normal approximation
    3. Transform back to r scale

    Args:
        r: Sample correlation coefficient
        n: Sample size
        alpha: Significance level (default 0.05 for 95% CI)

    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    if n < 4:
        # CI not meaningful for very small samples
        return (-1.0, 1.0)

    # Handle perfect correlation specially
    if abs(r) >= 0.9999:
        # For near-perfect correlation, CI is essentially the value itself
        # (very small standard error)
        if r > 0:
            return (0.9, 1.0)
        else:
            return (-1.0, -0.9)

    # Fisher Z-transformation
    z = fisher_z_transform(r)

    # Standard error of Z
    se_z = 1.0 / math.sqrt(n - 3)

    # Critical value from standard normal (two-tailed)
    z_crit = inverse_normal_cdf(1.0 - alpha / 2.0)

    # CI for Z
    z_lower = z - z_crit * se_z
    z_upper = z + z_crit * se_z

    # Transform back to r scale
    r_lower = inverse_fisher_z(z_lower)
    r_upper = inverse_fisher_z(z_upper)

    # Clamp to [-1, 1]
    r_lower = max(-1.0, min(1.0, r_lower))
    r_upper = max(-1.0, min(1.0, r_upper))

    return (r_lower, r_upper)


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
        description="Compute Pearson correlation coefficient and hypothesis test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # From command-line values
  %(prog)s --x 1,2,3,4,5 --y 2.1,3.8,6.2,7.9,10.1

  # From CSV file
  %(prog)s --file data.csv --x-col height --y-col weight

  # With hypothesis test
  %(prog)s --x 10,20,30,40,50 --y 15,28,41,55,68 --alpha 0.05

  # One-tailed test (test if correlation is positive)
  %(prog)s --x 1,2,3,4,5 --y 2,4,5,4,5 --alpha 0.05 --sided one
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


def interpret_correlation(r: float) -> str:
    """Provide interpretation of correlation strength."""
    abs_r = abs(r)
    if abs_r >= 0.9:
        strength = "very strong"
    elif abs_r >= 0.7:
        strength = "strong"
    elif abs_r >= 0.5:
        strength = "moderate"
    elif abs_r >= 0.3:
        strength = "weak"
    else:
        strength = "very weak"

    direction = "positive" if r >= 0 else "negative"
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

    # Compute correlation
    try:
        r = pearson_r(x_values, y_values)
        r_squared = r * r
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2

    # Print basic results
    print(f"Sample size (n):               {n}")
    print(f"Pearson's r:                   {format_number(r, precision)}")
    print(f"r² (coefficient of determination): {format_number(r_squared, precision)}")
    print(f"Interpretation:                {interpret_correlation(r)}")

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
            t_stat = correlation_t_statistic(r, n)
            p_value = correlation_p_value(r, n, args.sided)
            ci_lower, ci_upper = correlation_confidence_interval(r, n, alpha)

            print(f"\nHypothesis test: H₀: ρ = 0 vs H₁: ρ ≠ 0 ({args.sided}-tailed)")
            print(f"t-statistic:                   {format_number(t_stat, precision)}")
            print(f"Degrees of freedom (df):       {n - 2}")
            print(f"p-value:                       {format_number(p_value, precision)}")
            print(f"Significance level (α):        {format_number(alpha, precision)}")

            significant = p_value < alpha
            print(
                f"Result:                   {'Significant' if significant else 'Not significant'} at α={alpha}"
            )

            if significant:
                print("Conclusion:            Reject H₀. Evidence of correlation.")
            else:
                print(
                    "Conclusion:            Fail to reject H₀. No significant correlation."
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
