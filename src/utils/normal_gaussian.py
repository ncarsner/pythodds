#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from typing import Optional

"""Command-line utility for Gaussian (normal) distribution calculations.

Computes PDF, CDF, and survival probabilities for a N(μ, σ²) distribution,
evaluates the probability of a value falling between two bounds, and inverts
the CDF to find the value at a given quantile (percent-point function).

Uses only the Python standard library (math.erf / math.erfinv).

Usage examples:
  normal -x 1.96 -m 0 -s 1
  normal --between -1.96 1.96 -m 0 -s 1
  normal --quantile 0.975 -m 0 -s 1
  normal -x 75 -m 70 -s 5
"""


# ---------------------------------------------------------------------------
# Core probability functions
# ---------------------------------------------------------------------------

_SQRT2 = math.sqrt(2.0)
_SQRT2PI = math.sqrt(2.0 * math.pi)


def normal_pdf(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """Probability density function for N(μ, σ²) evaluated at x."""
    z = (x - mu) / sigma
    return math.exp(-0.5 * z * z) / (sigma * _SQRT2PI)


def normal_cdf(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """Cumulative distribution P(X ≤ x) for X ~ N(μ, σ²)."""
    return 0.5 * (1.0 + math.erf((x - mu) / (sigma * _SQRT2)))


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


def normal_ppf(p: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """Percent-point (quantile / inverse CDF) function for N(μ, σ²).

    Returns x such that P(X ≤ x) = p. Requires 0 < p < 1.
    """
    return mu + sigma * _SQRT2 * _erfinv(2.0 * p - 1.0)


def normal_prob_between(
    low: float, high: float, mu: float = 0.0, sigma: float = 1.0
) -> float:
    """P(low ≤ X ≤ high) for X ~ N(μ, σ²)."""
    return normal_cdf(high, mu, sigma) - normal_cdf(low, mu, sigma)


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def _fmt(x: float, precision: int) -> str:
    pct = x * 100.0
    fmt = f"{{:.{precision}f}}"
    return f"{fmt.format(x)} ({fmt.format(pct)}%)"


def _fmt_val(x: float, precision: int) -> str:
    return f"{x:.{precision}f}"


def format_single(
    x: float,
    mu: float,
    sigma: float,
    pdf: float,
    cdf_le: float,
    cdf_ge: float,
    precision: int,
) -> str:
    z = (x - mu) / sigma
    lines = [
        f"Mean (μ):          {_fmt_val(mu, precision)}",
        f"Std dev (σ):       {_fmt_val(sigma, precision)}",
        f"Value (x):         {_fmt_val(x, precision)}",
        f"Z-score:           {_fmt_val(z, precision)}",
        f"PDF f(x):          {_fmt_val(pdf, precision)}",
        f"P(X ≤ {_fmt_val(x, precision)}):    {_fmt(cdf_le, precision)}",
        f"P(X ≥ {_fmt_val(x, precision)}):    {_fmt(cdf_ge, precision)}",
    ]
    return "\n".join(lines)


def format_between(
    low: float,
    high: float,
    mu: float,
    sigma: float,
    prob: float,
    precision: int,
) -> str:
    lines = [
        f"Mean (μ):                            {_fmt_val(mu, precision)}",
        f"Std dev (σ):                         {_fmt_val(sigma, precision)}",
        f"Lower bound:                         {_fmt_val(low, precision)}",
        f"Upper bound:                         {_fmt_val(high, precision)}",
        f"P({_fmt_val(low, precision)} ≤ X ≤ {_fmt_val(high, precision)}):  {_fmt(prob, precision)}",
    ]
    return "\n".join(lines)


def format_quantile(
    p: float,
    mu: float,
    sigma: float,
    x: float,
    precision: int,
) -> str:
    z = (x - mu) / sigma
    lines = [
        f"Mean (μ):          {_fmt_val(mu, precision)}",
        f"Std dev (σ):       {_fmt_val(sigma, precision)}",
        f"Quantile (p):      {_fmt(p, precision)}",
        f"x at P(X ≤ x) = p: {_fmt_val(x, precision)}",
        f"Z-score:           {_fmt_val(z, precision)}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gaussian (normal) distribution calculator.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  normal -x 1.96 -m 0 -s 1
  normal -x 75 -m 70 -s 5
  normal --between -1.96 1.96 -m 0 -s 1
  normal --quantile 0.975 -m 0 -s 1
""",
    )

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--value",
        "-x",
        type=float,
        metavar="X",
        help="compute PDF, P(X ≤ x), and P(X ≥ x) for this value",
    )
    mode.add_argument(
        "--between",
        nargs=2,
        type=float,
        metavar=("LOW", "HIGH"),
        help="compute P(LOW ≤ X ≤ HIGH)",
    )
    mode.add_argument(
        "--quantile",
        "-q",
        type=float,
        metavar="P",
        help="find the value x such that P(X ≤ x) = P (inverse CDF)",
    )

    parser.add_argument(
        "--mean",
        "-m",
        type=float,
        default=0.0,
        metavar="μ",
        help="distribution mean μ (default: 0)",
    )
    parser.add_argument(
        "--std",
        "-s",
        type=float,
        default=1.0,
        metavar="σ",
        help="distribution standard deviation σ (default: 1)",
    )
    parser.add_argument(
        "--precision",
        "-P",
        type=int,
        default=6,
        help="decimal places for printed values (default: 6)",
    )

    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate(args: argparse.Namespace) -> Optional[str]:
    if args.std <= 0:
        return "-s/--std must be greater than 0"

    if args.value is None and args.between is None and args.quantile is None:
        return "one of -x/--value, --between, or -q/--quantile is required"

    if args.between is not None:
        low, high = args.between
        if high <= low:
            return "--between HIGH must be strictly greater than LOW"

    if args.quantile is not None and not (0.0 < args.quantile < 1.0):
        return "-q/--quantile must be strictly between 0 and 1"

    return None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    error = validate(args)
    if error:
        print(f"Error: {error}", file=sys.stderr)
        return 2

    mu = args.mean
    sigma = args.std
    precision = args.precision

    # --- single value ---
    if args.value is not None:
        x = args.value
        pdf = normal_pdf(x, mu, sigma)
        cdf_le = normal_cdf(x, mu, sigma)
        cdf_ge = 1.0 - cdf_le
        print(format_single(x, mu, sigma, pdf, cdf_le, cdf_ge, precision))
        return 0

    # --- between two bounds ---
    if args.between is not None:
        low, high = args.between
        prob = normal_prob_between(low, high, mu, sigma)
        print(format_between(low, high, mu, sigma, prob, precision))
        return 0

    # --- quantile / inverse CDF ---
    p = args.quantile
    x = normal_ppf(p, mu, sigma)
    print(format_quantile(p, mu, sigma, x, precision))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
