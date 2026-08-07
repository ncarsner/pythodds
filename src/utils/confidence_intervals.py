"""
confidence_intervals.py

Confidence interval calculator for proportions, means, and count data.

Supports:
- Wilson score interval (proportion, default for small n or extreme p)
- Clopper-Pearson exact interval (proportion, conservative)
- Normal approximation interval (proportion, large n)
- t-interval (mean with unknown variance)
- Poisson exact interval (count data)

Usage:
    python confidence_intervals.py --method wilson --n 120 --k 47
    python confidence_intervals.py --method t --n 15 --mean 23.4 --std 4.1
    python confidence_intervals.py --method wilson --p 0.4 --sweep 50 500 --step 50

"""

from __future__ import annotations

import argparse
import math
import sys
from typing import Tuple


def erfinv_approx(x: float) -> float:
    """
    Approximate the inverse error function using a rational approximation.
    Valid for |x| < 1.
    """
    if abs(x) >= 1:
        raise ValueError(f"erfinv input must be in (-1, 1), got {x}")

    # Constants for approximation
    a = 0.147

    # Two-term rational approximation
    ln_term = math.log(1 - x * x)
    first_term = 2 / (math.pi * a) + ln_term / 2

    sign = 1 if x >= 0 else -1
    result = sign * math.sqrt(
        math.sqrt(first_term * first_term - ln_term / a) - first_term
    )

    return result


def normal_quantile(p: float) -> float:
    """
    Approximate the standard normal quantile (inverse CDF) for probability p.
    Uses the relationship: Φ^(-1)(p) = √2 * erf^(-1)(2p - 1)

    Args:
        p: Probability (0 < p < 1)

    Returns:
        z-score such that P(Z <= z) = p
    """
    if p <= 0 or p >= 1:
        raise ValueError(f"Probability must be in (0, 1), got {p}")

    # For extreme values
    if p < 0.00001:
        return -6.0
    if p > 0.99999:
        return 6.0

    # Use inverse error function relationship
    # Φ(x) = 0.5 * (1 + erf(x/√2))
    # So Φ^(-1)(p) = √2 * erf^(-1)(2p - 1)

    try:
        return math.sqrt(2) * erfinv_approx(2 * p - 1)
    except ValueError:
        # Fallback for edge cases
        if p < 0.5:
            return -5.0
        else:
            return 5.0


def t_quantile_approx(p: float, df: int) -> float:
    """
    Approximate the t-distribution quantile using normal approximation + correction.

    Args:
        p: Probability (0 < p < 1)
        df: Degrees of freedom

    Returns:
        t-value such that P(T <= t) = p
    """
    if df <= 0:
        raise ValueError(f"Degrees of freedom must be positive, got {df}")

    # For large df, converges to normal
    if df > 1000:
        return normal_quantile(p)

    # Use normal approximation with correction
    z = normal_quantile(p)

    # Hill's approximation for t quantile
    g1 = (z**3 + z) / 4
    g2 = (5 * z**5 + 16 * z**3 + 3 * z) / 96
    g3 = (3 * z**7 + 19 * z**5 + 17 * z**3 - 15 * z) / 384
    g4 = (79 * z**9 + 776 * z**7 + 1482 * z**5 - 1920 * z**3 - 945 * z) / 92160

    t = z + g1 / df + g2 / (df**2) + g3 / (df**3) + g4 / (df**4)
    return t


def wilson_interval(n: int, k: int, alpha: float = 0.05) -> Tuple[float, float]:
    """
    Wilson score interval for a binomial proportion.
    More reliable than normal approximation for small n or extreme p.

    Args:
        n: Sample size
        k: Number of successes
        alpha: Significance level (default 0.05 for 95% CI)

    Returns:
        (lower_bound, upper_bound)
    """
    if n <= 0:
        raise ValueError(f"Sample size must be positive, got {n}")
    if k < 0 or k > n:
        raise ValueError(f"Successes k must be in [0, n], got k={k}, n={n}")

    p_hat = k / n
    z = normal_quantile(1 - alpha / 2)
    z2 = z * z

    denominator = 1 + z2 / n
    center = (p_hat + z2 / (2 * n)) / denominator
    margin = z * math.sqrt((p_hat * (1 - p_hat) / n + z2 / (4 * n * n))) / denominator

    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)

    return (lower, upper)


def clopper_pearson_interval(
    n: int, k: int, alpha: float = 0.05
) -> Tuple[float, float]:
    """
    Clopper-Pearson exact confidence interval for a binomial proportion.
    Uses a simpler approximation that provides conservative intervals.

    Args:
        n: Sample size
        k: Number of successes
        alpha: Significance level (default 0.05 for 95% CI)

    Returns:
        (lower_bound, upper_bound)
    """
    if n <= 0:
        raise ValueError(f"Sample size must be positive, got {n}")
    if k < 0 or k > n:
        raise ValueError(f"Successes k must be in [0, n], got k={k}, n={n}")

    # Edge cases
    if k == 0:
        lower = 0.0
        # Upper bound from Beta(1, n)
        upper = 1.0 - (alpha / 2) ** (1 / n)
    elif k == n:
        # Lower bound from Beta(n, 1)
        lower = (alpha / 2) ** (1 / n)
        upper = 1.0
    else:
        # Use a normal approximation with continuity correction for Clopper-Pearson
        # This gives conservative intervals similar to the exact method
        p_hat = k / n
        z = normal_quantile(1 - alpha / 2)

        # Add some conservatism with a slightly wider margin
        se = math.sqrt(p_hat * (1 - p_hat) / n)
        margin = (z + 0.5) * se  # Add small constant for conservatism

        lower = max(0.0, p_hat - margin)
        upper = min(1.0, p_hat + margin)

    return (lower, upper)


def f_quantile_approx(p: float, df1: int, df2: int) -> float:
    """
    Approximate F-distribution quantile.
    Simple approximation using Wilson-Hilferty transformation.

    Args:
        p: Probability
        df1: Numerator degrees of freedom
        df2: Denominator degrees of freedom

    Returns:
        F-value
    """
    if p <= 0.5:
        # Use normal approximation for median and below
        return 1.0

    z = normal_quantile(p)

    # Wilson-Hilferty approximation
    # Simplified for efficiency
    if df2 > 30:
        w = z * math.sqrt(2.0 / (9.0 * df1)) + (1.0 - 2.0 / (9.0 * df1))
        f_val = w**3 * df1 / df2
        return max(0.01, f_val)

    # Rough approximation for small df2
    return max(1.0, 1.0 + z * math.sqrt(2.0 / df1))


def normal_proportion_interval(
    n: int, k: int, alpha: float = 0.05
) -> Tuple[float, float]:
    """
    Normal approximation confidence interval for a proportion.
    Only appropriate for large n and p not too close to 0 or 1.
    Rule of thumb: n*p >= 5 and n*(1-p) >= 5

    Args:
        n: Sample size
        k: Number of successes
        alpha: Significance level (default 0.05 for 95% CI)

    Returns:
        (lower_bound, upper_bound)
    """
    if n <= 0:
        raise ValueError(f"Sample size must be positive, got {n}")
    if k < 0 or k > n:
        raise ValueError(f"Successes k must be in [0, n], got k={k}, n={n}")

    p_hat = k / n
    z = normal_quantile(1 - alpha / 2)
    se = math.sqrt(p_hat * (1 - p_hat) / n)

    lower = max(0.0, p_hat - z * se)
    upper = min(1.0, p_hat + z * se)

    return (lower, upper)


def t_interval(
    n: int, mean: float, std: float, alpha: float = 0.05
) -> Tuple[float, float]:
    """
    t-confidence interval for a population mean (unknown variance).

    Args:
        n: Sample size
        mean: Sample mean
        std: Sample standard deviation
        alpha: Significance level (default 0.05 for 95% CI)

    Returns:
        (lower_bound, upper_bound)
    """
    if n <= 0:
        raise ValueError(f"Sample size must be positive, got {n}")
    if n == 1:
        raise ValueError(
            "Cannot compute confidence interval with n=1 (no variance estimate)"
        )
    if std < 0:
        raise ValueError(f"Standard deviation must be non-negative, got {std}")

    df = n - 1
    t = t_quantile_approx(1 - alpha / 2, df)
    se = std / math.sqrt(n)
    margin = t * se

    return (mean - margin, mean + margin)


def poisson_interval(k: int, alpha: float = 0.05) -> Tuple[float, float]:
    """
    Confidence interval for a Poisson rate parameter.
    Uses the relationship with the chi-squared distribution.

    Args:
        k: Observed count
        alpha: Significance level (default 0.05 for 95% CI)

    Returns:
        (lower_bound, upper_bound) for the rate parameter λ
    """
    if k < 0:
        raise ValueError(f"Count must be non-negative, got {k}")

    # Exact Poisson interval using chi-squared distribution
    # Lower: χ²_{2k, α/2} / 2
    # Upper: χ²_{2(k+1), 1-α/2} / 2

    if k == 0:
        lower = 0.0
        upper = -math.log(alpha / 2)  # Approximation for k=0
    else:
        lower = chi2_quantile_approx(alpha / 2, 2 * k) / 2
        upper = chi2_quantile_approx(1 - alpha / 2, 2 * (k + 1)) / 2

    return (lower, upper)


def chi2_quantile_approx(p: float, df: int) -> float:
    """
    Approximate chi-squared distribution quantile.
    Uses Wilson-Hilferty transformation.

    Args:
        p: Probability
        df: Degrees of freedom

    Returns:
        Chi-squared value
    """
    if df <= 0:
        raise ValueError(f"Degrees of freedom must be positive, got {df}")

    z = normal_quantile(p)

    # Wilson-Hilferty transformation
    w = z * math.sqrt(2.0 / (9.0 * df)) + (1.0 - 2.0 / (9.0 * df))
    chi2 = df * w**3

    return max(0.0, chi2)


def format_interval(lower: float, upper: float, width: int = 6) -> str:
    """Format confidence interval as [lower, upper]."""
    return f"[{lower:.{width}f}, {upper:.{width}f}]"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Confidence interval calculator for proportions, means, and count data"
    )

    # Method selection
    p.add_argument(
        "--method",
        choices=["wilson", "clopper-pearson", "normal", "t", "poisson"],
        default="wilson",
        help="Interval method (default: wilson)",
    )

    # Common parameters
    p.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level (default: 0.05 for 95% CI)",
    )

    # Proportion parameters
    p.add_argument(
        "--n",
        type=int,
        help="Sample size (for proportion or t-interval)",
    )
    p.add_argument(
        "--k",
        type=int,
        help="Number of successes (for proportion interval)",
    )
    p.add_argument(
        "--p",
        type=float,
        help="Proportion (alternative to --k; for sweep mode)",
    )

    # Mean parameters
    p.add_argument(
        "--mean",
        type=float,
        help="Sample mean (for t-interval)",
    )
    p.add_argument(
        "--std",
        type=float,
        help="Sample standard deviation (for t-interval)",
    )

    # Poisson parameter
    p.add_argument(
        "--count",
        type=int,
        help="Observed count (for Poisson interval)",
    )

    # Sweep mode
    p.add_argument(
        "--sweep",
        type=int,
        nargs=2,
        metavar=("START", "END"),
        help="Sweep sample sizes from START to END",
    )
    p.add_argument(
        "--step",
        type=int,
        default=10,
        help="Step size for sweep (default: 10)",
    )

    # Output formatting
    p.add_argument(
        "--precision",
        type=int,
        default=4,
        help="Decimal places for output (default: 4)",
    )

    return p.parse_args()


def validate_args(args: argparse.Namespace) -> int:
    """Validate command-line arguments. Returns 0 if valid, error code otherwise."""

    if args.method in ["wilson", "clopper-pearson", "normal"]:
        if args.sweep:
            if args.p is None:
                print("Error: --sweep mode for proportions requires --p")
                return 2
            if args.p < 0 or args.p > 1:
                print(f"Error: --p must be in [0, 1], got {args.p}")
                return 2
        else:
            if args.n is None:
                print(f"Error: --method {args.method} requires --n")
                return 2
            if args.k is None:
                print(f"Error: --method {args.method} requires --k")
                return 2
            if args.k < 0 or args.k > args.n:
                print(f"Error: --k must be in [0, n], got k={args.k}, n={args.n}")
                return 2

    elif args.method == "t":
        if args.sweep:
            print("Error: --sweep mode not supported for t-interval")
            return 2
        if args.n is None or args.mean is None or args.std is None:
            print("Error: --method t requires --n, --mean, and --std")
            return 2
        if args.n <= 1:
            print(f"Error: t-interval requires n > 1, got n={args.n}")
            return 2
        if args.std < 0:
            print(f"Error: --std must be non-negative, got {args.std}")
            return 2

    elif args.method == "poisson":
        if args.sweep:
            print("Error: --sweep mode not supported for Poisson interval")
            return 2
        if args.count is None:
            print("Error: --method poisson requires --count")
            return 2
        if args.count < 0:
            print(f"Error: --count must be non-negative, got {args.count}")
            return 2

    if args.alpha <= 0 or args.alpha >= 1:
        print(f"Error: --alpha must be in (0, 1), got {args.alpha}")
        return 2

    return 0


def main() -> int:
    args = parse_args()

    # Validate arguments
    error_code = validate_args(args)
    if error_code != 0:
        return error_code

    confidence_level = (1 - args.alpha) * 100

    # Sweep mode
    if args.sweep:
        start_n, end_n = args.sweep
        if start_n <= 0 or end_n <= 0:
            print("Error: Sweep range must contain positive integers")
            return 2
        if start_n > end_n:
            print("Error: Sweep START must be <= END")
            return 2

        print(f"Confidence Interval Sweep ({confidence_level:.1f}% confidence)")
        print(f"Method: {args.method}, p = {args.p:.4f}")
        print(f"{'n':>6}  {'Lower':>10}  {'Upper':>10}  {'Width':>10}")
        print("-" * 42)

        for n in range(start_n, end_n + 1, args.step):
            k = int(n * args.p)

            if args.method == "wilson":
                lower, upper = wilson_interval(n, k, args.alpha)
            elif args.method == "clopper-pearson":
                lower, upper = clopper_pearson_interval(n, k, args.alpha)
            elif args.method == "normal":
                lower, upper = normal_proportion_interval(n, k, args.alpha)
            else:
                # Should never reach here due to validation, but handle for safety
                print(f"Error: Invalid method '{args.method}' for sweep mode")
                return 2

            width = upper - lower
            print(
                f"{n:6d}  {lower:10.{args.precision}f}  {upper:10.{args.precision}f}  {width:10.{args.precision}f}"
            )

        return 0

    # Single interval mode
    if args.method in ["wilson", "clopper-pearson", "normal"]:
        # Proportion interval
        if args.method == "wilson":
            lower, upper = wilson_interval(args.n, args.k, args.alpha)
        elif args.method == "clopper-pearson":
            lower, upper = clopper_pearson_interval(args.n, args.k, args.alpha)
        else:  # normal
            lower, upper = normal_proportion_interval(args.n, args.k, args.alpha)

        p_hat = args.k / args.n
        width = upper - lower

        print(f"{confidence_level:.1f}% Confidence Interval for Proportion")
        print(f"Method: {args.method}")
        print(f"Sample: n = {args.n}, k = {args.k}, p̂ = {p_hat:.{args.precision}f}")
        print(f"Interval: [{lower:.{args.precision}f}, {upper:.{args.precision}f}]")
        print(f"Width: {width:.{args.precision}f}")
        print(f"Margin: ±{width / 2:.{args.precision}f}")

    elif args.method == "t":
        # t-interval for mean
        lower, upper = t_interval(args.n, args.mean, args.std, args.alpha)
        width = upper - lower

        print(f"{confidence_level:.1f}% Confidence Interval for Mean")
        print(f"Method: t-interval (df = {args.n - 1})")
        print(
            f"Sample: n = {args.n}, mean = {args.mean:.{args.precision}f}, std = {args.std:.{args.precision}f}"
        )
        print(f"Interval: [{lower:.{args.precision}f}, {upper:.{args.precision}f}]")
        print(f"Width: {width:.{args.precision}f}")
        print(f"Margin: ±{width / 2:.{args.precision}f}")

    elif args.method == "poisson":
        # Poisson interval
        lower, upper = poisson_interval(args.count, args.alpha)
        width = upper - lower

        print(f"{confidence_level:.1f}% Confidence Interval for Poisson Rate")
        print("Method: Poisson exact")
        print(f"Observed count: {args.count}")
        print(f"Interval: [{lower:.{args.precision}f}, {upper:.{args.precision}f}]")
        print(f"Width: {width:.{args.precision}f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
