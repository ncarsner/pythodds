#!/usr/bin/env python3
"""Command-line utility for the sigmoid (logistic) function.

Computes σ(x) = 1 / (1 + e^(−x)), its derivative σ'(x) = σ(x)(1 − σ(x)),
and the inverse logit (logit function). Supports single-value evaluation,
range tables, and an optional Unicode sparkline.

Usage examples:
  sigmoid -x 2.0
  sigmoid -x 0 --derivative
  sigmoid --range -5 5 1
  sigmoid --inverse --prob 0.75
  sigmoid --range -4 4 0.5 --plot
"""

from __future__ import annotations

import argparse
import math
import sys
from typing import Sequence

# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def sigmoid(x: float) -> float:
    """Compute the sigmoid function σ(x) = 1 / (1 + e^(−x)).

    Uses a numerically stable formulation to avoid overflow for large |x|.

    Args:
        x: Input value.

    Returns:
        Sigmoid output in the open interval (0, 1).
    """
    if x >= 0:
        exp_neg = math.exp(-x)
        return 1.0 / (1.0 + exp_neg)
    exp_pos = math.exp(x)
    return exp_pos / (1.0 + exp_pos)


def sigmoid_derivative(x: float) -> float:
    """Compute the derivative σ'(x) = σ(x) · (1 − σ(x)).

    Args:
        x: Input value.

    Returns:
        Derivative of the sigmoid at x; maximum value 0.25 at x = 0.
    """
    s = sigmoid(x)
    return s * (1.0 - s)


def inverse_logit(p: float) -> float:
    """Compute the inverse sigmoid (logit function): log(p / (1 − p)).

    Args:
        p: Probability strictly between 0 and 1.

    Returns:
        Log-odds value x such that σ(x) = p.

    Raises:
        ValueError: If p is not strictly between 0 and 1.
    """
    if not (0.0 < p < 1.0):
        raise ValueError("p must be strictly between 0 and 1")
    return math.log(p / (1.0 - p))


# ---------------------------------------------------------------------------
# Range generation
# ---------------------------------------------------------------------------


def _frange(start: float, stop: float, step: float) -> list[float]:
    """Generate evenly-spaced float values from start to stop inclusive.

    Args:
        start: First value in the range.
        stop: Last value (included if reachable within floating-point tolerance).
        step: Step size; must be positive.

    Returns:
        List of values from start to stop in increments of step.
    """
    n = math.floor((stop - start) / step + 1e-9) + 1
    return [start + i * step for i in range(n)]


# ---------------------------------------------------------------------------
# Sparkline
# ---------------------------------------------------------------------------

_SPARK_CHARS = "▁▂▃▄▅▆▇█"


def sparkline(values: Sequence[float]) -> str:
    """Create a Unicode block-character sparkline for values in [0, 1].

    Args:
        values: Sequence of values; each should be in [0, 1].

    Returns:
        A string of Unicode block characters visualising the values.
    """
    if not values:
        return ""
    n = len(_SPARK_CHARS)
    result = []
    for v in values:
        idx = min(n - 1, int(v * n))
        result.append(_SPARK_CHARS[idx])
    return "".join(result)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the sigmoid utility.

    Args:
        argv: Argument list; uses sys.argv if None.

    Returns:
        Parsed namespace with validated attribute types.
    """
    parser = argparse.ArgumentParser(
        description="Sigmoid function calculator.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  sigmoid -x 2.0
  sigmoid -x 0 --derivative
  sigmoid --range -5 5 1
  sigmoid --inverse --prob 0.75
  sigmoid --range -4 4 0.5 --plot
""",
    )

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--value",
        "-x",
        type=float,
        metavar="X",
        help="evaluate sigmoid at a single value",
    )
    mode.add_argument(
        "--range",
        nargs=3,
        type=float,
        metavar=("MIN", "MAX", "STEP"),
        help="print a table of sigmoid values over [MIN, MAX] in steps of STEP",
    )
    mode.add_argument(
        "--inverse",
        action="store_true",
        help="compute the inverse logit: find x such that σ(x) = P",
    )

    parser.add_argument(
        "--prob",
        "-p",
        type=float,
        metavar="P",
        help="probability in (0, 1) for --inverse mode",
    )
    parser.add_argument(
        "--derivative",
        "-d",
        action="store_true",
        help="also show the derivative σ'(x)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="append a Unicode sparkline visualising the sigmoid curve",
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


def validate(args: argparse.Namespace) -> str | None:
    """Validate parsed arguments.

    Args:
        args: Parsed namespace from parse_args.

    Returns:
        Error message string if invalid, or None if all arguments are valid.
    """
    if args.precision < 0:
        return "--precision must be non-negative"

    if args.inverse:
        if args.prob is None:
            return "--prob is required with --inverse"
        if not (0.0 < args.prob < 1.0):
            return "--prob must be strictly between 0 and 1"
        return None

    if args.value is None and args.range is None:
        return "one of --value, --range, or --inverse is required"

    if args.range is not None:
        lo, hi, step = args.range
        if step <= 0:
            return "STEP must be greater than 0"
        if lo >= hi:
            return "MIN must be less than MAX"

    return None


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def _fmt(x: float, precision: int) -> str:
    return f"{x:.{precision}f}"


def format_single(
    x: float,
    sig: float,
    dsig: float,
    precision: int,
    show_deriv: bool,
) -> str:
    """Format output for single-value mode.

    Args:
        x: Input value.
        sig: Sigmoid value σ(x).
        dsig: Derivative value σ'(x).
        precision: Decimal places.
        show_deriv: Whether to include the derivative line.

    Returns:
        Formatted string.
    """
    lines = [
        f"Value (x):         {_fmt(x, precision)}",
        f"Sigmoid σ(x):      {_fmt(sig, precision)}",
    ]
    if show_deriv:
        lines.append(f"Derivative σ'(x):  {_fmt(dsig, precision)}")
    return "\n".join(lines)


def format_table(
    xs: list[float],
    sigs: list[float],
    dsigs: list[float],
    precision: int,
    show_deriv: bool,
    show_plot: bool,
) -> str:
    """Format output for range-table mode.

    Args:
        xs: Input x values.
        sigs: Corresponding σ(x) values.
        dsigs: Corresponding σ'(x) values.
        precision: Decimal places.
        show_deriv: Whether to include the derivative column.
        show_plot: Whether to append a Unicode sparkline.

    Returns:
        Formatted string with header, separator, data rows, and optional sparkline.
    """
    w = max(12, precision + 6)
    p = precision
    deriv_label = "σ'(x)"

    if show_deriv:
        header = f"{'x':>{w}}  {'σ(x)':>{w}}  {deriv_label:>{w}}"
        sep = "-" * len(header)
        rows = [
            f"{x:{w}.{p}f}  {s:{w}.{p}f}  {d:{w}.{p}f}"
            for x, s, d in zip(xs, sigs, dsigs)
        ]
    else:
        header = f"{'x':>{w}}  {'σ(x)':>{w}}"
        sep = "-" * len(header)
        rows = [f"{x:{w}.{p}f}  {s:{w}.{p}f}" for x, s in zip(xs, sigs)]

    lines = [header, sep] + rows

    if show_plot:
        lines.extend(["", sparkline(sigs)])

    return "\n".join(lines)


def format_inverse(p: float, x: float, precision: int) -> str:
    """Format output for inverse-logit mode.

    Args:
        p: Input probability.
        x: Computed logit value.
        precision: Decimal places.

    Returns:
        Formatted string showing p, logit(p), and a round-trip check.
    """
    lines = [
        f"Probability (p):      {_fmt(p, precision)}",
        f"Logit log(p/(1-p)):   {_fmt(x, precision)}",
        f"Check σ(logit):       {_fmt(sigmoid(x), precision)}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Run the sigmoid CLI.

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

    if args.inverse:
        try:
            x = inverse_logit(args.prob)
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 2
        print(format_inverse(args.prob, x, precision))
        return 0

    if args.range is not None:
        lo, hi, step = args.range
        xs = _frange(lo, hi, step)
        sigs = [sigmoid(v) for v in xs]
        dsigs = [sigmoid_derivative(v) for v in xs]
        print(format_table(xs, sigs, dsigs, precision, args.derivative, args.plot))
        return 0

    # Single-value mode
    x = args.value
    sig = sigmoid(x)
    dsig = sigmoid_derivative(x)
    output = format_single(x, sig, dsig, precision, args.derivative)

    if args.plot:
        plot_xs = _frange(-6.0, 6.0, 0.25)
        plot_sigs = [sigmoid(v) for v in plot_xs]
        print(output)
        print("")
        print(sparkline(plot_sigs))
    else:
        print(output)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
