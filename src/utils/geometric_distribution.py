#!/usr/bin/env python3
"""Command-line utility for the geometric distribution.

Models the number of independent Bernoulli trials until the first success
(support k = 1, 2, 3, ...).  Pure Python via ``math`` — no external
dependencies.

Usage examples:
  geometric -k 5 -p 0.3
  geometric -k 10 -p 0.2 --survival
  geometric -p 0.25 --table 1 15
"""

from __future__ import annotations

import argparse
import sys

# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def geo_pmf(k: int, p: float) -> float:
    """Probability mass function P(X = k) for Geometric(p).

    Args:
        k: Trial number of the first success; must be >= 1.
        p: Success probability per trial; must be in (0, 1].

    Returns:
        P(X = k) = (1 - p)^(k - 1) * p.

    Raises:
        ValueError: If ``k`` < 1 or ``p`` is not in (0, 1].
    """
    _validate_k(k)
    _validate_p(p)
    return ((1 - p) ** (k - 1)) * p


def geo_cdf(k: int, p: float) -> float:
    """Cumulative distribution function P(X <= k) for Geometric(p).

    Args:
        k: Trial number; values < 1 yield 0.0.
        p: Success probability per trial; must be in (0, 1].

    Returns:
        P(X <= k) = 1 - (1 - p)^k, clipped to [0, 1].

    Raises:
        ValueError: If ``p`` is not in (0, 1].
    """
    _validate_p(p)
    if k < 1:
        return 0.0
    return max(0.0, min(1.0, 1 - (1 - p) ** k))


def geo_survival(k: int, p: float) -> float:
    """Survival function P(X > k) for Geometric(p).

    Args:
        k: Trial number; values < 0 yield 1.0.
        p: Success probability per trial; must be in (0, 1].

    Returns:
        P(X > k) = (1 - p)^k, clipped to [0, 1].

    Raises:
        ValueError: If ``p`` is not in (0, 1].
    """
    _validate_p(p)
    if k < 0:
        return 1.0
    return max(0.0, min(1.0, (1 - p) ** k))


def geo_mean(p: float) -> float:
    """Expected number of trials until the first success.

    Args:
        p: Success probability per trial; must be in (0, 1].

    Returns:
        Mean = 1 / p.

    Raises:
        ValueError: If ``p`` is not in (0, 1].
    """
    _validate_p(p)
    return 1 / p


def geo_variance(p: float) -> float:
    """Variance of the number of trials until the first success.

    Args:
        p: Success probability per trial; must be in (0, 1].

    Returns:
        Variance = (1 - p) / p^2.

    Raises:
        ValueError: If ``p`` is not in (0, 1].
    """
    _validate_p(p)
    return (1 - p) / (p**2)


def _validate_k(k: int) -> None:
    """Raise if ``k`` is not a valid trial number (>= 1)."""
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")


def _validate_p(p: float) -> None:
    """Raise if ``p`` is not a valid success probability in (0, 1]."""
    if not (0 < p <= 1):
        raise ValueError(f"p must be in (0, 1], got {p}")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Build and return the argument parser namespace.

    Args:
        argv: Argument list (uses ``sys.argv`` when ``None``).

    Returns:
        Parsed :class:`argparse.Namespace`.
    """
    parser = argparse.ArgumentParser(
        description="Geometric distribution calculator (trials until first success).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  geometric -k 5 -p 0.3
  geometric -k 10 -p 0.2 --survival
  geometric -p 0.25 --table 1 15
""",
    )
    parser.add_argument(
        "-k",
        type=int,
        default=None,
        metavar="INT",
        help="trial number of the first success (required unless --table is used)",
    )
    parser.add_argument(
        "-p",
        type=float,
        required=True,
        metavar="F",
        help="success probability per trial, in (0, 1]",
    )
    parser.add_argument(
        "--survival",
        action="store_true",
        help="report P(X > k) instead of the cumulative P(X <= k)",
    )
    parser.add_argument(
        "--table",
        type=int,
        nargs=2,
        default=None,
        metavar=("MIN", "MAX"),
        help="print PMF/CDF/survival for k = MIN..MAX instead of a single report",
    )
    parser.add_argument(
        "--precision",
        "-P",
        type=int,
        default=4,
        metavar="PREC",
        help="decimal places for output (default: 4)",
    )
    return parser.parse_args(argv)


def validate(args: argparse.Namespace) -> str | None:
    """Return an error message string, or ``None`` if arguments are valid.

    Args:
        args: Parsed argument namespace from :func:`parse_args`.

    Returns:
        Error description string, or ``None`` when validation passes.
    """
    if not (0 < args.p <= 1):
        return f"-p must be in (0, 1], got {args.p}"
    if args.precision < 0:
        return "--precision must be non-negative"

    if args.table is not None:
        min_k, max_k = args.table
        if min_k < 1:
            return f"--table MIN must be >= 1, got {min_k}"
        if max_k < min_k:
            return f"--table MAX must be >= MIN, got MIN={min_k}, MAX={max_k}"
        return None

    if args.k is None:
        return "-k is required unless --table is used"
    if args.k < 1:
        return f"-k must be >= 1, got {args.k}"
    return None


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def _fmt(value: float, precision: int) -> str:
    """Format a float to the requested number of decimal places."""
    return f"{value:.{precision}f}"


def format_single(k: int, p: float, survival: bool, precision: int) -> str:
    """Format a single-k geometric distribution report.

    Args:
        k: Trial number of the first success.
        p: Success probability per trial.
        survival: If True, report P(X > k) instead of P(X <= k).
        precision: Decimal places for all floating-point output.

    Returns:
        Multi-line string ready to print.
    """
    f = lambda v: _fmt(v, precision)  # noqa: E731
    lines = [
        f"Geometric distribution  (p={f(p)})",
        "",
        f"  PMF  P(X = {k}):  {f(geo_pmf(k, p))}",
    ]
    if survival:
        lines.append(f"  P(X > {k}):        {f(geo_survival(k, p))}")
    else:
        lines.append(f"  CDF  P(X <= {k}):  {f(geo_cdf(k, p))}")
    lines += [
        "",
        f"  mean:      {f(geo_mean(p))}",
        f"  variance:  {f(geo_variance(p))}",
    ]
    return "\n".join(lines)


def format_table(
    min_k: int, max_k: int, p: float, survival: bool, precision: int
) -> str:
    """Format a PMF/CDF-or-survival table for k = min_k..max_k.

    Args:
        min_k: First trial number in the table (>= 1).
        max_k: Last trial number in the table (>= min_k).
        p: Success probability per trial.
        survival: If True, show P(X > k) instead of P(X <= k).
        precision: Decimal places for all floating-point output.

    Returns:
        Multi-line string ready to print.
    """
    f = lambda v: _fmt(v, precision)  # noqa: E731
    third_header = "P(X > k)" if survival else "P(X <= k)"
    lines = [
        f"Geometric distribution  (p={f(p)})",
        "",
        f"  {'k':>4}  {'PMF':>10}  {third_header:>10}",
    ]
    for k in range(min_k, max_k + 1):
        third = geo_survival(k, p) if survival else geo_cdf(k, p)
        lines.append(f"  {k:>4}  {f(geo_pmf(k, p)):>10}  {f(third):>10}")
    lines += [
        "",
        f"  mean:      {f(geo_mean(p))}",
        f"  variance:  {f(geo_variance(p))}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Run the geometric distribution CLI.

    Args:
        argv: Argument list override for testing (uses ``sys.argv`` when ``None``).

    Returns:
        0 on success, 2 on input or computation error.
    """
    args = parse_args(argv)

    error = validate(args)
    if error:
        print(f"Error: {error}", file=sys.stderr)
        return 2

    try:
        if args.table is not None:
            min_k, max_k = args.table
            print(format_table(min_k, max_k, args.p, args.survival, args.precision))
        else:
            print(format_single(args.k, args.p, args.survival, args.precision))
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
