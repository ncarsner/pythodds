#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from typing import Optional

"""Command-line utility to compute binomial distribution probabilities.

Usage examples:
  python scripts/probability_binom.py -n 10 -k 3 -p 0.4
  python scripts/probability_binom.py -n 100 -k 30 -p 0.35 --target 40 --min-prob 0.05

Prints PMF for exactly `k`, CDF for `<= k`, survival `>= k`, and optionally
the probability of meeting a `target` number of successes and whether that
probability meets a provided minimum threshold (`--min-prob`).
"""


def binomial_pmf(n: int, k: int, p: float) -> float:
    """Probability mass function P(X = k) for Binomial(n, p)."""
    if k < 0 or k > n:
        return 0.0
    return math.comb(n, k) * (p**k) * (1 - p) ** (n - k)


def binomial_cdf_le(n: int, k: int, p: float) -> float:
    """Cumulative distribution P(X <= k) for Binomial(n, p)."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    total = 0.0
    for i in range(0, k + 1):
        total += binomial_pmf(n, i, p)
    return total


def binomial_cdf_ge(n: int, k: int, p: float) -> float:
    """Survival function P(X >= k) for Binomial(n, p)."""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    return 1.0 - binomial_cdf_le(n, k - 1, p)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Binomial distribution helper")
    parser.add_argument(
        "-n", "--trials", type=int, required=True, help="number of trials (n)"
    )
    parser.add_argument(
        "-k",
        "--successes",
        type=int,
        required=True,
        help="number of successes (k) to evaluate",
    )
    parser.add_argument(
        "-p",
        "--p",
        type=float,
        required=True,
        help="historical success probability (0..1)",
    )
    parser.add_argument(
        "--target",
        type=int,
        default=None,
        help="optional target number of successes to evaluate P(X >= target)",
    )
    parser.add_argument(
        "--min-prob",
        type=float,
        default=None,
        help="optional minimum acceptable probability for the target (0..1)",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=6,
        help="decimal places for printed probabilities",
    )
    return parser.parse_args(argv)


def format_prob(x: float, precision: int) -> str:
    pct = x * 100.0
    fmt = f"{{:.{precision}f}}"
    return f"{fmt.format(x)} ({fmt.format(pct)}%)"


def render_distribution_bar(le: float, exact: float, ge: float, width: int) -> str:
    """Return a 100%-stacked ANSI bar representing P(X<k) | P(X=k) | P(X>k)."""
    left = le - exact  # P(X < k)
    mid = exact  # P(X = k)
    right = ge - exact  # P(X > k)

    left_w = round(left * width)
    mid_w = round(mid * width)
    right_w = max(0, width - left_w - mid_w)

    RED = "\033[41m"
    YEL = "\033[43m"
    GRN = "\033[42m"
    RST = "\033[0m"

    bar = RED + " " * left_w + YEL + " " * mid_w + GRN + " " * right_w + RST
    legend = (
        f"{RED} {RST} <k {left * 100:.1f}%  "
        f"{YEL} {RST} =k {mid * 100:.1f}%  "
        f"{GRN} {RST} >k {right * 100:.1f}%"
    )
    return bar + "\n" + legend


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    n = args.trials
    k = args.successes
    p = args.p
    precision = args.precision

    if not (0.0 <= p <= 1.0):
        print("Error: --p must be between 0 and 1", file=sys.stderr)
        return 2
    if n < 0:
        print("Error: --trials must be non-negative", file=sys.stderr)
        return 2

    exact = binomial_pmf(n, k, p)
    le = binomial_cdf_le(n, k, p)
    ge = binomial_cdf_ge(n, k, p)

    lines = [
        f"n={n}, k={k}, p={p}",
        f"P(X = {k}):  {format_prob(exact, precision)}",
        f"P(X <= {k}): {format_prob(le, precision)}",
        f"P(X >= {k}): {format_prob(ge, precision)}",
    ]
    for line in lines:
        print(line)
    bar_width = max(len(line) for line in lines)
    print(render_distribution_bar(le, exact, ge, bar_width))

    if args.target is not None:
        target = args.target
        pt = binomial_cdf_ge(n, target, p)
        print(f"P(X >= {target}): {format_prob(pt, precision)}")
        if args.min_prob is not None:
            minp = args.min_prob
            if not (0.0 <= minp <= 1.0):
                print("Error: --min-prob must be between 0 and 1", file=sys.stderr)
                return 2
            meets = pt >= minp
            print(f"Meets minimum {format_prob(minp, precision)}: {meets}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
