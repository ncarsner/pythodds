#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from typing import Optional

"""Command-line utility for sample size calculations.

Computes the minimum sample size needed for:
- Proportion estimation within a specified margin of error
- Mean difference detection with specified power
- Two-proportion comparison with specified power

Uses only the Python standard library (math.erf / math.erfinv).

Usage examples:
  # Minimum n to estimate a proportion within ±3% at 95% confidence
  sample --type proportion --prop 0.5 --margin 0.03

  # Minimum n to detect a mean shift of 5 units (std=12) with 80% power
  sample --type mean --delta 5 --std 12 --power 0.80

  # Sweep: show achieved power across n = 50-300 for a two-proportion comparison
  sample --type comparison --p1 0.40 --p2 0.50 --alpha 0.05 --sweep 50 300
"""


# ---------------------------------------------------------------------------
# Z-score and distribution utilities
# ---------------------------------------------------------------------------

_SQRT2 = math.sqrt(2.0)


def _erfinv(y: float) -> float:
    """Inverse of math.erf, implemented via Halley's method with a rational initial guess.

    Converges to full float64 precision in ≤ 3 iterations for |y| < 0.9999.
    """
    if y == 0.0:
        return 0.0
    if abs(y) >= 1.0:
        raise ValueError("erfinv argument must be in (-1, 1)")
    # Rational approximation for initial guess (Winitzki 2008 approximation)
    a = 0.147
    ln1y2 = math.log(1.0 - y * y)
    term = 2.0 / (math.pi * a) + ln1y2 / 2.0
    x = math.copysign(math.sqrt(math.sqrt(term * term - ln1y2 / a) - term), y)
    # Halley refinement — two iterations are sufficient for float64 accuracy
    for _ in range(3):
        fx = math.erf(x) - y
        dfx = 2.0 / math.sqrt(math.pi) * math.exp(-(x * x))
        x -= fx / (dfx - fx * x)  # Halley's formula simplified for erf
    return x


def normal_cdf(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """Cumulative distribution P(X ≤ x) for X ~ N(μ, σ²)."""
    return 0.5 * (1.0 + math.erf((x - mu) / (sigma * _SQRT2)))


def normal_ppf(p: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """Percent-point (quantile / inverse CDF) function for N(μ, σ²).

    Returns x such that P(X ≤ x) = p. Requires 0 < p < 1.
    """
    return mu + sigma * _SQRT2 * _erfinv(2.0 * p - 1.0)


def z_critical(alpha: float, sided: str = "two") -> float:
    """Return the critical z-value for a given significance level.

    Args:
        alpha: Significance level (e.g., 0.05 for 95% confidence)
        sided: "one" or "two" for one-tailed or two-tailed test

    Returns:
        Critical z-value (always positive)
    """
    if sided == "two":
        return abs(normal_ppf(alpha / 2.0))
    else:
        return abs(normal_ppf(alpha))


# ---------------------------------------------------------------------------
# Sample size calculations
# ---------------------------------------------------------------------------


def sample_size_proportion(
    p: float, margin: float, alpha: float = 0.05, sided: str = "two"
) -> int:
    """Calculate sample size for proportion estimation within a margin of error.

    Args:
        p: Expected proportion (0 to 1)
        margin: Desired margin of error (e.g., 0.03 for ±3%)
        alpha: Significance level (default 0.05 for 95% confidence)
        sided: "one" or "two" for one-tailed or two-tailed test

    Returns:
        Minimum required sample size
    """
    z = z_critical(alpha, sided)
    # Formula: n = (z² * p * (1-p)) / margin²
    n = (z * z * p * (1.0 - p)) / (margin * margin)
    return math.ceil(n)


def sample_size_mean(
    sigma: float,
    delta: float,
    alpha: float = 0.05,
    power: float = 0.80,
    sided: str = "two",
) -> int:
    """Calculate sample size for detecting a mean difference with specified power.

    Args:
        sigma: Population standard deviation
        delta: Minimum detectable effect size (difference to detect)
        alpha: Significance level (default 0.05)
        power: Statistical power (1 - β), default 0.80
        sided: "one" or "two" for one-tailed or two-tailed test

    Returns:
        Minimum required sample size
    """
    z_alpha = z_critical(alpha, sided)
    z_beta = abs(normal_ppf(1.0 - power))

    # Formula: n = ((z_α + z_β) * σ / δ)²
    n = ((z_alpha + z_beta) * sigma / delta) ** 2
    return math.ceil(n)


def sample_size_comparison(
    p1: float,
    p2: float,
    alpha: float = 0.05,
    power: float = 0.80,
    sided: str = "two",
    equal_n: bool = True,
) -> tuple[int, int]:
    """Calculate sample size for comparing two proportions.

    Args:
        p1: Proportion in group 1
        p2: Proportion in group 2
        alpha: Significance level (default 0.05)
        power: Statistical power (1 - β), default 0.80
        sided: "one" or "two" for one-tailed or two-tailed test
        equal_n: If True, return equal sample sizes; if False, optimize for unequal

    Returns:
        Tuple of (n1, n2) sample sizes
    """
    z_alpha = z_critical(alpha, sided)
    z_beta = abs(normal_ppf(1.0 - power))

    # Pooled proportion for null hypothesis
    p_bar = (p1 + p2) / 2.0

    if equal_n:
        # Formula for equal sample sizes:
        # n = 2 * (z_α * sqrt(2*p_bar*(1-p_bar)) + z_β * sqrt(p1*(1-p1) + p2*(1-p2)))² / (p1 - p2)²
        numerator = z_alpha * math.sqrt(2 * p_bar * (1 - p_bar)) + z_beta * math.sqrt(
            p1 * (1 - p1) + p2 * (1 - p2)
        )
        n = 2 * (numerator / (p1 - p2)) ** 2
        n_ceil = math.ceil(n)
        return (n_ceil, n_ceil)
    else:
        # For unequal n, use the equal formula as a starting point
        numerator = z_alpha * math.sqrt(2 * p_bar * (1 - p_bar)) + z_beta * math.sqrt(
            p1 * (1 - p1) + p2 * (1 - p2)
        )
        n_total = 2 * (numerator / (p1 - p2)) ** 2
        n1 = math.ceil(n_total / 2)
        n2 = math.ceil(n_total / 2)
        return (n1, n2)


def achieved_power_mean(
    n: int, sigma: float, delta: float, alpha: float, sided: str
) -> float:
    """Calculate the achieved power for a given sample size in a mean test."""
    z_alpha = z_critical(alpha, sided)
    # Power = P(Z > z_α - δ*sqrt(n)/σ)
    z_stat = delta * math.sqrt(n) / sigma - z_alpha
    return normal_cdf(z_stat)


def achieved_power_comparison(
    n: int, p1: float, p2: float, alpha: float, sided: str
) -> float:
    """Calculate the achieved power for a given sample size in a two-proportion test."""
    z_alpha = z_critical(alpha, sided)
    p_bar = (p1 + p2) / 2.0

    # Standard error under null and alternative
    se_null = math.sqrt(2 * p_bar * (1 - p_bar) / n)
    se_alt = math.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / n)

    # Non-centrality parameter
    delta = abs(p1 - p2)
    z_stat = delta / se_alt - z_alpha * se_null / se_alt
    return normal_cdf(z_stat)


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def _fmt(x: float, precision: int) -> str:
    """Format a float value to specified precision."""
    return f"{x:.{precision}f}"


def _fmt_pct(x: float, precision: int) -> str:
    """Format a proportion as percentage."""
    return f"{x * 100:.{precision}f}%"


def format_proportion_result(
    n: int, p: float, margin: float, alpha: float, sided: str, precision: int
) -> str:
    """Format the output for proportion sample size calculation."""
    lines = [
        "Sample Size Calculation: Proportion Estimation",
        "─" * 60,
        f"Expected proportion (p):        {_fmt(p, precision)}",
        f"Margin of error:                ±{_fmt_pct(margin, precision)}",
        f"Confidence level:               {_fmt_pct(1 - alpha, precision)}",
        f"Test type:                      {sided}-sided",
        "",
        f"Required sample size:           n = {n}",
    ]
    return "\n".join(lines)


def format_mean_result(
    n: int,
    sigma: float,
    delta: float,
    alpha: float,
    power: float,
    sided: str,
    precision: int,
) -> str:
    """Format the output for mean difference sample size calculation."""
    lines = [
        "Sample Size Calculation: Mean Difference Detection",
        "─" * 60,
        f"Population std dev (σ):         {_fmt(sigma, precision)}",
        f"Minimum detectable effect (δ):  {_fmt(delta, precision)}",
        f"Significance level (α):         {_fmt(alpha, precision)}",
        f"Statistical power (1-β):        {_fmt_pct(power, precision)}",
        f"Test type:                      {sided}-sided",
        "",
        f"Required sample size:           n = {n}",
    ]
    return "\n".join(lines)


def format_comparison_result(
    n1: int,
    n2: int,
    p1: float,
    p2: float,
    alpha: float,
    power: float,
    sided: str,
    precision: int,
) -> str:
    """Format the output for two-proportion comparison sample size calculation."""
    lines = [
        "Sample Size Calculation: Two-Proportion Comparison",
        "─" * 60,
        f"Proportion in group 1 (p₁):     {_fmt(p1, precision)}",
        f"Proportion in group 2 (p₂):     {_fmt(p2, precision)}",
        f"Difference (p₁ - p₂):           {_fmt(p1 - p2, precision)}",
        f"Significance level (α):         {_fmt(alpha, precision)}",
        f"Statistical power (1-β):        {_fmt_pct(power, precision)}",
        f"Test type:                      {sided}-sided",
        "",
        f"Required sample sizes:          n₁ = {n1}, n₂ = {n2}",
        f"Total sample size:              N = {n1 + n2}",
    ]
    return "\n".join(lines)


def format_sweep_table(
    test_type: str,
    sweep_range: list[int],
    params: dict,
    precision: int,
) -> str:
    """Format a sweep table showing power across different sample sizes."""
    lines = [
        f"Power Analysis Sweep: {test_type.title()}",
        "─" * 60,
    ]

    # Add parameter summary
    if test_type == "mean":
        lines.extend(
            [
                f"Parameters: σ={params['sigma']}, δ={params['delta']}, "
                f"α={params['alpha']}, {params['sided']}-sided",
                "",
                f"{'Sample Size':>12}  {'Power':>10}",
                f"{'─' * 12}  {'─' * 10}",
            ]
        )
        for n in sweep_range:
            power = achieved_power_mean(
                n, params["sigma"], params["delta"], params["alpha"], params["sided"]
            )
            lines.append(f"{n:>12}  {_fmt_pct(power, precision):>10}")
    elif test_type == "comparison":
        lines.extend(
            [
                f"Parameters: p₁={params['p1']}, p₂={params['p2']}, "
                f"α={params['alpha']}, {params['sided']}-sided",
                "",
                f"{'Sample Size':>12}  {'Power':>10}",
                f"{'─' * 12}  {'─' * 10}",
            ]
        )
        for n in sweep_range:
            power = achieved_power_comparison(
                n, params["p1"], params["p2"], params["alpha"], params["sided"]
            )
            lines.append(f"{n:>12}  {_fmt_pct(power, precision):>10}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Argument parsing and main
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample size calculator for statistical studies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Proportion estimation
  sample --type proportion --prop 0.5 --margin 0.03

  # Mean difference detection
  sample --type mean --delta 5 --std 12 --power 0.80

  # Two-proportion comparison
  sample --type comparison --p1 0.40 --p2 0.50 --alpha 0.05 --power 0.80

  # Power analysis sweep
  sample --type mean --delta 5 --std 12 --sweep 50 300 --step 25
""",
    )

    parser.add_argument(
        "--type",
        choices=["proportion", "mean", "comparison"],
        required=True,
        help="Type of sample size calculation",
    )

    # Proportion arguments
    parser.add_argument(
        "--prop", type=float, dest="p", help="Expected proportion (0 to 1)"
    )
    parser.add_argument(
        "--margin",
        type=float,
        help="Desired margin of error for proportion (e.g., 0.03 for ±3%%)",
    )

    # Mean arguments
    parser.add_argument(
        "--std",
        "--sigma",
        type=float,
        dest="sigma",
        help="Population standard deviation",
    )
    parser.add_argument(
        "--delta", type=float, help="Minimum detectable effect size (mean difference)"
    )

    # Comparison arguments
    parser.add_argument("--p1", type=float, help="Proportion in group 1")
    parser.add_argument("--p2", type=float, help="Proportion in group 2")

    # Common arguments
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level (default: 0.05)",
    )
    parser.add_argument(
        "--power",
        type=float,
        default=0.80,
        help="Statistical power for mean/comparison (default: 0.80)",
    )
    parser.add_argument(
        "--sided",
        choices=["one", "two"],
        default="two",
        help="One-sided or two-sided test (default: two)",
    )

    # Sweep arguments
    parser.add_argument(
        "--sweep",
        nargs=2,
        type=int,
        metavar=("MIN", "MAX"),
        help="Show power across range of sample sizes (MIN MAX)",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=10,
        help="Step size for sweep (default: 10)",
    )

    # Output formatting
    parser.add_argument(
        "--precision",
        type=int,
        default=4,
        help="Decimal places for printed values (default: 4)",
    )

    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    # Validate alpha
    if not (0.0 < args.alpha < 1.0):
        print("Error: --alpha must be between 0 and 1", file=sys.stderr)
        return 2

    # Validate power
    if args.power is not None and not (0.0 < args.power < 1.0):
        print("Error: --power must be between 0 and 1", file=sys.stderr)
        return 2

    try:
        if args.type == "proportion":
            # Validate required arguments
            if args.p is None:
                print("Error: --p required for proportion type", file=sys.stderr)
                return 2
            if args.margin is None:
                print("Error: --margin required for proportion type", file=sys.stderr)
                return 2
            if not (0.0 < args.p < 1.0):
                print("Error: --p must be between 0 and 1", file=sys.stderr)
                return 2
            if not (0.0 < args.margin < 1.0):
                print("Error: --margin must be between 0 and 1", file=sys.stderr)
                return 2

            n = sample_size_proportion(args.p, args.margin, args.alpha, args.sided)
            print(
                format_proportion_result(
                    n, args.p, args.margin, args.alpha, args.sided, args.precision
                )
            )

        elif args.type == "mean":
            # Validate required arguments
            if args.sigma is None:
                print("Error: --std required for mean type", file=sys.stderr)
                return 2
            if args.delta is None:
                print("Error: --delta required for mean type", file=sys.stderr)
                return 2
            if args.sigma <= 0:
                print("Error: --std must be positive", file=sys.stderr)
                return 2
            if args.delta <= 0:
                print("Error: --delta must be positive", file=sys.stderr)
                return 2

            if args.sweep:
                # Power analysis sweep
                min_n, max_n = args.sweep
                sweep_range = list(range(min_n, max_n + 1, args.step))
                params = {
                    "sigma": args.sigma,
                    "delta": args.delta,
                    "alpha": args.alpha,
                    "sided": args.sided,
                }
                print(format_sweep_table("mean", sweep_range, params, args.precision))
            else:
                n = sample_size_mean(
                    args.sigma, args.delta, args.alpha, args.power, args.sided
                )
                print(
                    format_mean_result(
                        n,
                        args.sigma,
                        args.delta,
                        args.alpha,
                        args.power,
                        args.sided,
                        args.precision,
                    )
                )

        elif args.type == "comparison":
            # Validate required arguments
            if args.p1 is None:
                print("Error: --p1 required for comparison type", file=sys.stderr)
                return 2
            if args.p2 is None:
                print("Error: --p2 required for comparison type", file=sys.stderr)
                return 2
            if not (0.0 < args.p1 < 1.0):
                print("Error: --p1 must be between 0 and 1", file=sys.stderr)
                return 2
            if not (0.0 < args.p2 < 1.0):
                print("Error: --p2 must be between 0 and 1", file=sys.stderr)
                return 2
            if args.p1 == args.p2:
                print("Error: --p1 and --p2 must be different", file=sys.stderr)
                return 2

            if args.sweep:
                # Power analysis sweep
                min_n, max_n = args.sweep
                sweep_range = list(range(min_n, max_n + 1, args.step))
                params = {
                    "p1": args.p1,
                    "p2": args.p2,
                    "alpha": args.alpha,
                    "sided": args.sided,
                }
                print(
                    format_sweep_table(
                        "comparison", sweep_range, params, args.precision
                    )
                )
            else:
                n1, n2 = sample_size_comparison(
                    args.p1, args.p2, args.alpha, args.power, args.sided
                )
                print(
                    format_comparison_result(
                        n1,
                        n2,
                        args.p1,
                        args.p2,
                        args.alpha,
                        args.power,
                        args.sided,
                        args.precision,
                    )
                )

    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
