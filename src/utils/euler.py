#!/usr/bin/env python3
"""Command-line utility for Euler's number and related functions.

Demonstrates the mathematical constant e ≈ 2.71828 via the limit definition,
Taylor series for e^x, natural logarithm series, Euler's identity, and the
Euler-Mascheroni constant.

Usage examples:
  euler --approx 1000000
  euler --exp 2 --order 10 --compare
  euler --identity --precision 15
  euler --ln 10 --compare
  euler --mascheroni
"""

from __future__ import annotations

import argparse
import json
import math
import sys

# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

_EULER_MASCHERONI_KNOWN = 0.5772156649015328606065120900824024310421593359399235988


def e_approx(n: int) -> float:
    """Approximate e using the limit definition (1 + 1/n)^n.

    Args:
        n: Number of terms; larger n gives a closer approximation.

    Returns:
        Approximation of e.

    Raises:
        ValueError: If n < 1.
    """
    if n < 1:
        raise ValueError("n must be at least 1")
    return (1.0 + 1.0 / n) ** n


def exp_series(x: float, n: int) -> float:
    """Compute e^x via the Taylor series sum_{k=0}^{n} x^k / k!.

    Args:
        x: Exponent value.
        n: Number of terms (order); larger n improves accuracy for large |x|.

    Returns:
        Approximation of e^x.

    Raises:
        ValueError: If n < 1.
    """
    if n < 1:
        raise ValueError("n must be at least 1")
    total = 0.0
    term = 1.0
    for k in range(n + 1):
        total += term
        term *= x / (k + 1)
    return total


def natural_log(x: float, n: int) -> float:
    """Approximate ln(x) via the arctanh series 2·sum[(t^(2k+1))/(2k+1)] where t=(x-1)/(x+1).

    This series converges for all x > 0. Accuracy improves with larger n,
    but more terms are needed for x far from 1.

    Args:
        x: Argument; must be positive.
        n: Number of terms; larger n improves accuracy.

    Returns:
        Approximation of ln(x).

    Raises:
        ValueError: If x <= 0 or n < 1.
    """
    if x <= 0:
        raise ValueError("x must be positive for natural log")
    if n < 1:
        raise ValueError("n must be at least 1")
    t = (x - 1.0) / (x + 1.0)
    total = 0.0
    t_power = t
    t_sq = t * t
    for k in range(n + 1):
        total += t_power / (2 * k + 1)
        t_power *= t_sq
    return 2.0 * total


def euler_identity() -> tuple[float, float, float]:
    """Compute e^(iπ) using Euler's formula: e^(iθ) = cos(θ) + i·sin(θ).

    Returns:
        Tuple of (real_part, imaginary_part, sum) where sum = real_part + 1.
        Euler's identity states e^(iπ) + 1 = 0, so sum ≈ 0.
    """
    real = math.cos(math.pi)
    imag = math.sin(math.pi)
    return real, imag, real + 1.0


def euler_mascheroni(n: int = 1_000_000) -> float:
    """Approximate the Euler-Mascheroni constant γ = lim(H_n - ln(n)).

    Args:
        n: Number of harmonic series terms; larger n gives higher accuracy.

    Returns:
        Approximation of γ ≈ 0.5772156649.
    """
    harmonic = sum(1.0 / k for k in range(1, n + 1))
    return harmonic - math.log(n)


# ---------------------------------------------------------------------------
# Convergence table helper
# ---------------------------------------------------------------------------


def _approx_table(n: int) -> list[tuple[int, float, float]]:
    """Build a convergence table for (1+1/k)^k at logarithmic steps up to n.

    Args:
        n: Maximum value; always included as the final row.

    Returns:
        List of (k, approximation, error_vs_e) tuples.
    """
    e = math.e
    rows: list[tuple[int, float, float]] = []
    seen: set[int] = set()
    k = 1
    while k <= n:
        approx = (1.0 + 1.0 / k) ** k
        rows.append((k, approx, abs(e - approx)))
        seen.add(k)
        k *= 10
    if n not in seen:
        approx = (1.0 + 1.0 / n) ** n
        rows.append((n, approx, abs(e - approx)))
    return rows


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the euler utility.

    Args:
        argv: Argument list; uses sys.argv if None.

    Returns:
        Parsed namespace with validated attribute types.
    """
    parser = argparse.ArgumentParser(
        description="Euler's number and related functions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  euler --approx 1000000
  euler --exp 2 --order 10 --compare
  euler --identity --precision 15
  euler --ln 10 --compare
  euler --mascheroni
""",
    )

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--approx",
        type=int,
        metavar="N",
        help="show convergence of (1+1/N)^N to e with a table at logarithmic steps",
    )
    mode.add_argument(
        "--exp",
        type=float,
        metavar="X",
        help="compute e^X via Taylor series",
    )
    mode.add_argument(
        "--identity",
        action="store_true",
        help="display Euler's identity: e^(iπ) + 1 = 0",
    )
    mode.add_argument(
        "--ln",
        type=float,
        metavar="X",
        help="approximate ln(X) via the arctanh series",
    )
    mode.add_argument(
        "--mascheroni",
        action="store_true",
        help="display the Euler-Mascheroni constant γ ≈ 0.5772",
    )

    parser.add_argument(
        "--order",
        type=int,
        default=20,
        metavar="N",
        help="number of Taylor/series terms for --exp and --ln (default: 20)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="compare series result to math.exp / math.log reference value",
    )
    parser.add_argument(
        "--precision",
        "-P",
        type=int,
        default=10,
        help="decimal places for printed values (default: 10)",
    )
    parser.add_argument(
        "--format",
        "-f",
        choices=["table", "json"],
        default="table",
        help="output format: table (default) or json",
    )

    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate(args: argparse.Namespace) -> str | None:
    """Validate parsed arguments.

    Args:
        args: Parsed namespace from parse_args.

    Returns:
        Error message string if invalid, or None if all arguments are valid.
    """
    if args.precision < 0:
        return "--precision must be non-negative"

    if (
        args.approx is None
        and args.exp is None
        and not args.identity
        and args.ln is None
        and not args.mascheroni
    ):
        return "one of --approx, --exp, --identity, --ln, or --mascheroni is required"

    if args.approx is not None and args.approx < 1:
        return "--approx N must be at least 1"

    if args.exp is not None and args.order < 1:
        return "--order must be at least 1"

    if args.ln is not None:
        if args.ln <= 0:
            return "--ln x must be positive"
        if args.order < 1:
            return "--order must be at least 1"

    return None


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def format_approx(table: list[tuple[int, float, float]], precision: int) -> str:
    """Format the convergence table for --approx mode.

    Args:
        table: List of (n, approximation, error) rows from _approx_table.
        precision: Decimal places.

    Returns:
        Formatted table string.
    """
    w = max(precision + 6, 14)
    header = f"{'n':>12}  {'(1+1/n)^n':>{w}}  {'Error vs e':>{w}}"
    sep = "-" * len(header)
    lines = [header, sep]
    for n_val, approx, err in table:
        lines.append(f"{n_val:>12}  {approx:{w}.{precision}f}  {err:{w}.{precision}f}")
    lines.append(f"\nmath.e = {math.e:.{precision}f}")
    return "\n".join(lines)


def format_exp(
    x: float,
    order: int,
    result: float,
    precision: int,
    compare: bool,
) -> str:
    """Format output for --exp mode.

    Args:
        x: Exponent.
        order: Number of Taylor terms used.
        result: Series approximation.
        precision: Decimal places.
        compare: Whether to include comparison to math.exp.

    Returns:
        Formatted string.
    """
    lines = [
        f"Value (x):          {x:.{precision}f}",
        f"Taylor order:       {order}",
        f"e^x (series):       {result:.{precision}f}",
    ]
    if compare:
        true_val = math.exp(x)
        lines.append(f"e^x (math.exp):     {true_val:.{precision}f}")
        lines.append(f"Absolute error:     {abs(true_val - result):.{precision}f}")
    return "\n".join(lines)


def format_identity(real: float, imag: float, total: float, precision: int) -> str:
    """Format output for --identity mode.

    Args:
        real: Real part of e^(iπ) (should be -1).
        imag: Imaginary part of e^(iπ) (should be ≈ 0).
        total: e^(iπ) + 1 (should be ≈ 0).
        precision: Decimal places.

    Returns:
        Formatted string showing Euler's identity.
    """
    p = precision
    lines = [
        "Euler's Identity: e^(iπ) + 1 = 0",
        "",
        "  e^(iπ) = cos(π) + i·sin(π)",
        f"  Real part:      {real:.{p}f}  (exact: -1)",
        f"  Imaginary part: {imag:.{p}f}  (exact: 0)",
        f"  e^(iπ) + 1:     {total:.{p}f}  (exact: 0)",
    ]
    return "\n".join(lines)


def format_ln(
    x: float,
    order: int,
    result: float,
    precision: int,
    compare: bool,
) -> str:
    """Format output for --ln mode.

    Args:
        x: Argument to ln.
        order: Number of series terms used.
        result: Series approximation.
        precision: Decimal places.
        compare: Whether to include comparison to math.log.

    Returns:
        Formatted string.
    """
    lines = [
        f"Value (x):          {x:.{precision}f}",
        f"Series order:       {order}",
        f"ln(x) (series):     {result:.{precision}f}",
    ]
    if compare:
        true_val = math.log(x)
        lines.append(f"ln(x) (math.log):   {true_val:.{precision}f}")
        lines.append(f"Absolute error:     {abs(true_val - result):.{precision}f}")
    return "\n".join(lines)


def format_mascheroni(gamma: float, precision: int) -> str:
    """Format output for --mascheroni mode.

    Args:
        gamma: Approximation of the Euler-Mascheroni constant.
        precision: Decimal places.

    Returns:
        Formatted string showing γ and the series definition.
    """
    p = precision
    lines = [
        "Euler-Mascheroni Constant (γ)",
        "",
        f"  Series approx (n=1,000,000): {gamma:.{p}f}",
        f"  Known value:                 {_EULER_MASCHERONI_KNOWN:.{p}f}",
        "  γ = lim(H_n − ln(n))  as n → ∞",
        "  where H_n = 1 + 1/2 + 1/3 + ··· + 1/n",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Run the euler CLI.

    Args:
        argv: Argument list; uses sys.argv if None.

    Returns:
        Exit code: 0 on success, 2 on input error.
    """
    args = parse_args(argv)

    error = validate(args)
    if error:
        print(f"Error: {error}", file=sys.stderr)
        return 2

    precision = args.precision
    fmt = args.format

    if args.approx is not None:
        table = _approx_table(args.approx)
        if fmt == "json":
            rows = [
                {"n": n_val, "approximation": approx, "error": err}
                for n_val, approx, err in table
            ]
            print(
                json.dumps(
                    {"mode": "approx", "math_e": math.e, "table": rows}, indent=2
                )
            )
        else:
            print(format_approx(table, precision))
        return 0

    if args.exp is not None:
        result = exp_series(args.exp, args.order)
        if fmt == "json":
            data: dict[str, object] = {
                "mode": "exp",
                "x": args.exp,
                "order": args.order,
                "series": result,
            }
            if args.compare:
                true_val = math.exp(args.exp)
                data["math_exp"] = true_val
                data["error"] = abs(true_val - result)
            print(json.dumps(data, indent=2))
        else:
            print(format_exp(args.exp, args.order, result, precision, args.compare))
        return 0

    if args.identity:
        real, imag, total = euler_identity()
        if fmt == "json":
            print(
                json.dumps(
                    {
                        "mode": "identity",
                        "real": real,
                        "imaginary": imag,
                        "sum": total,
                    },
                    indent=2,
                )
            )
        else:
            print(format_identity(real, imag, total, precision))
        return 0

    if args.ln is not None:
        result_ln = natural_log(args.ln, args.order)
        if fmt == "json":
            data_ln: dict[str, object] = {
                "mode": "ln",
                "x": args.ln,
                "order": args.order,
                "series": result_ln,
            }
            if args.compare:
                true_val_ln = math.log(args.ln)
                data_ln["math_log"] = true_val_ln
                data_ln["error"] = abs(true_val_ln - result_ln)
            print(json.dumps(data_ln, indent=2))
        else:
            print(format_ln(args.ln, args.order, result_ln, precision, args.compare))
        return 0

    # mascheroni mode
    gamma = euler_mascheroni()
    if fmt == "json":
        print(
            json.dumps(
                {
                    "mode": "mascheroni",
                    "gamma": gamma,
                    "known_value": _EULER_MASCHERONI_KNOWN,
                },
                indent=2,
            )
        )
    else:
        print(format_mascheroni(gamma, precision))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
