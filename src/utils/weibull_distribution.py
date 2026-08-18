#!/usr/bin/env python3
"""Command-line utility for the Weibull distribution.

Computes PDF, CDF, survival, hazard, quantiles, and moments for the two-
parameter Weibull(k, lambda) distribution.  Pure Python via ``math.lgamma`` —
no external dependencies.

The shape parameter k sets the failure mode: k < 1 is a decreasing hazard
(infant mortality), k = 1 is the constant-hazard exponential case, and k > 1 is
an increasing hazard (wear-out).

Usage examples:
  weibull -x 500 -k 2 --lambda 1000
  weibull -x 500 -k 2 --lambda 1000 --survival
  weibull --quantile 0.05 -k 1.5 --lambda 800
  weibull -k 2.5 --lambda 1200 --table 0 2000 200
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from typing import Any

# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def weibull_pdf(x: float, k: float, lam: float) -> float:
    """Probability density f(x) for Weibull(k, lam).

    Args:
        x: Evaluation point; values < 0 yield 0.0.
        k: Shape parameter; must be > 0.
        lam: Scale parameter; must be > 0.

    Returns:
        f(x) = (k/lam) * (x/lam)^(k-1) * exp(-(x/lam)^k), and 0.0 for x < 0.

    Raises:
        ValueError: If ``k`` or ``lam`` is not > 0.
    """
    _validate_params(k, lam)
    if x < 0:
        return 0.0
    if x == 0:
        # The density at the origin is 0, 1/lam, or unbounded depending on k.
        if k > 1:
            return 0.0
        if k == 1:
            return 1 / lam
        return math.inf
    z = x / lam
    return (k / lam) * z ** (k - 1) * math.exp(-(z**k))


def weibull_cdf(x: float, k: float, lam: float) -> float:
    """Cumulative distribution F(x) = P(X <= x) for Weibull(k, lam).

    Args:
        x: Evaluation point; values <= 0 yield 0.0.
        k: Shape parameter; must be > 0.
        lam: Scale parameter; must be > 0.

    Returns:
        F(x) = 1 - exp(-(x/lam)^k), clipped to [0, 1].

    Raises:
        ValueError: If ``k`` or ``lam`` is not > 0.
    """
    _validate_params(k, lam)
    if x <= 0:
        return 0.0
    return max(0.0, min(1.0, -math.expm1(-((x / lam) ** k))))


def weibull_survival(x: float, k: float, lam: float) -> float:
    """Survival function S(x) = P(X > x) for Weibull(k, lam).

    Args:
        x: Evaluation point; values <= 0 yield 1.0.
        k: Shape parameter; must be > 0.
        lam: Scale parameter; must be > 0.

    Returns:
        S(x) = exp(-(x/lam)^k), clipped to [0, 1].

    Raises:
        ValueError: If ``k`` or ``lam`` is not > 0.
    """
    _validate_params(k, lam)
    if x <= 0:
        return 1.0
    return max(0.0, min(1.0, math.exp(-((x / lam) ** k))))


def weibull_hazard(x: float, k: float, lam: float) -> float:
    """Hazard rate h(x) = f(x) / S(x) for Weibull(k, lam).

    Args:
        x: Evaluation point; values < 0 yield 0.0.
        k: Shape parameter; must be > 0.
        lam: Scale parameter; must be > 0.

    Returns:
        h(x) = (k/lam) * (x/lam)^(k-1) — the instantaneous failure rate of a
        unit that has already survived to x.  Computed in closed form, so it
        stays finite in the far tail where f(x)/S(x) would divide 0 by 0.

    Raises:
        ValueError: If ``k`` or ``lam`` is not > 0.
    """
    _validate_params(k, lam)
    if x < 0:
        return 0.0
    if x == 0:
        if k > 1:
            return 0.0
        if k == 1:
            return 1 / lam
        return math.inf
    return (k / lam) * (x / lam) ** (k - 1)


def weibull_quantile(p: float, k: float, lam: float) -> float:
    """Inverse CDF: the value x with P(X <= x) = p.

    Args:
        p: Target cumulative probability, in [0, 1).
        k: Shape parameter; must be > 0.
        lam: Scale parameter; must be > 0.

    Returns:
        x = lam * (-ln(1 - p))^(1/k).

    Raises:
        ValueError: If ``p`` is not in [0, 1), or ``k`` or ``lam`` is not > 0.
    """
    _validate_params(k, lam)
    if not (0 <= p < 1):
        raise ValueError(f"p must be in [0, 1), got {p}")
    if p == 0:
        return 0.0
    return lam * (-math.log1p(-p)) ** (1 / k)


def weibull_mean(k: float, lam: float) -> float:
    """Expected value E[X] for Weibull(k, lam).

    Args:
        k: Shape parameter; must be > 0.
        lam: Scale parameter; must be > 0.

    Returns:
        lam * Gamma(1 + 1/k), evaluated through ``math.lgamma`` so large
        shape or scale values do not overflow the gamma function.

    Raises:
        ValueError: If ``k`` or ``lam`` is not > 0.
    """
    _validate_params(k, lam)
    return lam * math.exp(math.lgamma(1 + 1 / k))


def weibull_variance(k: float, lam: float) -> float:
    """Variance Var(X) for Weibull(k, lam).

    Args:
        k: Shape parameter; must be > 0.
        lam: Scale parameter; must be > 0.

    Returns:
        lam^2 * [Gamma(1 + 2/k) - Gamma(1 + 1/k)^2].

    Raises:
        ValueError: If ``k`` or ``lam`` is not > 0.
    """
    _validate_params(k, lam)
    g1 = math.exp(math.lgamma(1 + 1 / k))
    g2 = math.exp(math.lgamma(1 + 2 / k))
    return max(0.0, lam**2 * (g2 - g1**2))


def weibull_median(k: float, lam: float) -> float:
    """Median failure time for Weibull(k, lam).

    Args:
        k: Shape parameter; must be > 0.
        lam: Scale parameter; must be > 0.

    Returns:
        lam * (ln 2)^(1/k).

    Raises:
        ValueError: If ``k`` or ``lam`` is not > 0.
    """
    return weibull_quantile(0.5, k, lam)


def failure_mode(k: float) -> str:
    """Describe the hazard behaviour implied by the shape parameter.

    Args:
        k: Shape parameter; must be > 0.

    Returns:
        One of ``"infant mortality (decreasing hazard)"``,
        ``"random failure (constant hazard, exponential)"``, or
        ``"wear-out (increasing hazard)"``.

    Raises:
        ValueError: If ``k`` is not > 0.
    """
    if k <= 0:
        raise ValueError(f"k must be > 0, got {k}")
    if k < 1:
        return "infant mortality (decreasing hazard)"
    if k == 1:
        return "random failure (constant hazard, exponential)"
    return "wear-out (increasing hazard)"


def table_rows(
    min_x: float, max_x: float, step: float, k: float, lam: float
) -> list[tuple[float, float, float, float, float]]:
    """Build a PDF/CDF/survival/hazard table across a range of x.

    Args:
        min_x: First evaluation point; must be >= 0.
        max_x: Last evaluation point; must be >= ``min_x``.
        step: Increment between rows; must be > 0.
        k: Shape parameter; must be > 0.
        lam: Scale parameter; must be > 0.

    Returns:
        List of (x, pdf, cdf, survival, hazard) tuples.

    Raises:
        ValueError: If the range or step is invalid, or a parameter is not > 0.
    """
    if step <= 0:
        raise ValueError(f"step must be > 0, got {step}")
    if min_x < 0:
        raise ValueError(f"min_x must be >= 0, got {min_x}")
    if max_x < min_x:
        raise ValueError(f"max_x must be >= min_x, got min={min_x}, max={max_x}")
    _validate_params(k, lam)

    rows: list[tuple[float, float, float, float, float]] = []
    # Step through with an integer counter so float accumulation cannot drift.
    count = int((max_x - min_x) / step) + 1
    for i in range(count):
        x = min_x + i * step
        rows.append(
            (
                x,
                weibull_pdf(x, k, lam),
                weibull_cdf(x, k, lam),
                weibull_survival(x, k, lam),
                weibull_hazard(x, k, lam),
            )
        )
    return rows


def _validate_params(k: float, lam: float) -> None:
    """Raise if the shape or scale parameter is not strictly positive."""
    if k <= 0:
        raise ValueError(f"k must be > 0, got {k}")
    if lam <= 0:
        raise ValueError(f"lambda must be > 0, got {lam}")


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
        description="Weibull distribution calculator for reliability analysis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  weibull -x 500 -k 2 --lambda 1000
  weibull -x 500 -k 2 --lambda 1000 --survival
  weibull --quantile 0.05 -k 1.5 --lambda 800
  weibull -k 2.5 --lambda 1200 --table 0 2000 200
""",
    )
    parser.add_argument(
        "-x",
        type=float,
        default=None,
        metavar="F",
        help="evaluation point (required unless --quantile or --table is used)",
    )
    parser.add_argument(
        "-k",
        "--shape",
        type=float,
        required=True,
        dest="k",
        metavar="F",
        help="shape parameter k; <1 infant mortality, 1 constant, >1 wear-out",
    )
    parser.add_argument(
        "--lambda",
        "--scale",
        type=float,
        required=True,
        dest="lam",
        metavar="F",
        help="scale parameter lambda (characteristic life)",
    )
    parser.add_argument(
        "--quantile",
        type=float,
        default=None,
        metavar="F",
        help="report the x at this cumulative probability instead of a point report",
    )
    parser.add_argument(
        "--survival",
        action="store_true",
        help="lead the point report with S(x) = 1 - F(x)",
    )
    parser.add_argument(
        "--table",
        type=float,
        nargs=3,
        default=None,
        metavar=("MIN", "MAX", "STEP"),
        help="print a table over x = MIN..MAX instead of a single report",
    )
    parser.add_argument(
        "--format",
        choices=["table", "json"],
        default="table",
        help="output format (default: table)",
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
    if args.k <= 0:
        return f"-k must be > 0, got {args.k}"
    if args.lam <= 0:
        return f"--lambda must be > 0, got {args.lam}"
    if args.precision < 0:
        return "--precision must be non-negative"

    if args.table is not None:
        min_x, max_x, step = args.table
        if min_x < 0:
            return f"--table MIN must be >= 0, got {min_x}"
        if max_x < min_x:
            return f"--table MAX must be >= MIN, got MIN={min_x}, MAX={max_x}"
        if step <= 0:
            return f"--table STEP must be > 0, got {step}"
        return None

    if args.quantile is not None:
        if not (0 <= args.quantile < 1):
            return f"--quantile must be in [0, 1), got {args.quantile}"
        return None

    if args.x is None:
        return "-x is required unless --quantile or --table is used"
    if args.x < 0:
        return f"-x must be >= 0, got {args.x}"
    return None


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def _fmt(value: float, precision: int) -> str:
    """Format a float to the requested number of decimal places."""
    if math.isinf(value):
        return "inf"
    return f"{value:.{precision}f}"


def build_result(args: argparse.Namespace) -> dict[str, Any]:
    """Compute the requested figures into a single result mapping.

    Args:
        args: Validated argument namespace from :func:`parse_args`.

    Returns:
        Mapping of parameters, distribution moments, and whichever of the
        point, quantile, or table sections was requested.

    Raises:
        ValueError: If a core computation rejects its inputs.
    """
    result: dict[str, Any] = {
        "k": args.k,
        "lambda": args.lam,
        "failure_mode": failure_mode(args.k),
        "mean": weibull_mean(args.k, args.lam),
        "median": weibull_median(args.k, args.lam),
        "variance": weibull_variance(args.k, args.lam),
    }

    if args.table is not None:
        min_x, max_x, step = args.table
        result["table"] = table_rows(min_x, max_x, step, args.k, args.lam)
    elif args.quantile is not None:
        result["quantile"] = {
            "p": args.quantile,
            "x": weibull_quantile(args.quantile, args.k, args.lam),
        }
    else:
        result["point"] = {
            "x": args.x,
            "pdf": weibull_pdf(args.x, args.k, args.lam),
            "cdf": weibull_cdf(args.x, args.k, args.lam),
            "survival": weibull_survival(args.x, args.k, args.lam),
            "hazard": weibull_hazard(args.x, args.k, args.lam),
        }
    return result


def format_table(result: dict[str, Any], precision: int, survival: bool) -> str:
    """Format the result mapping as an aligned text report.

    Args:
        result: Mapping from :func:`build_result`.
        precision: Decimal places for all floating-point output.
        survival: If True, lead the point report with S(x) rather than F(x).

    Returns:
        Multi-line string ready to print.
    """
    lines = [
        f"Weibull distribution  (k={_fmt(float(result['k']), precision)}, "
        f"lambda={_fmt(float(result['lambda']), precision)})",
        "",
        f"  Failure mode: {result['failure_mode']}",
        "",
    ]

    point = result.get("point")
    if isinstance(point, dict):
        x = _fmt(point["x"], precision)
        ordered = [
            (f"S(x > {x}):", "survival"),
            (f"F(x <= {x}):", "cdf"),
        ]
        if not survival:
            ordered.reverse()
        # Labels carry the formatted x value, so their width varies with
        # --precision; pad every row to the widest one in this report.
        width = max(len(label) for label, _key in ordered + [("hazard h(x):", "")])
        lines.append(f"  {'f(x):':<{width}} {_fmt(point['pdf'], precision):>14}")
        for label, key in ordered:
            lines.append(f"  {label:<{width}} {_fmt(point[key], precision):>14}")
        lines.append(
            f"  {'hazard h(x):':<{width}} {_fmt(point['hazard'], precision):>14}"
        )
        lines.append("")

    quantile = result.get("quantile")
    if isinstance(quantile, dict):
        lines += [
            f"  {'p:':<16} {_fmt(quantile['p'], precision):>14}",
            f"  {'x at p:':<16} {_fmt(quantile['x'], precision):>14}",
            "",
        ]

    rows = result.get("table")
    if isinstance(rows, list):
        lines.append(
            f"  {'x':>10}  {'f(x)':>12}  {'F(x)':>12}  {'S(x)':>12}  {'h(x)':>12}"
        )
        for x, pdf, cdf, surv, hazard in rows:
            lines.append(
                f"  {_fmt(x, precision):>10}  {_fmt(pdf, precision):>12}  "
                f"{_fmt(cdf, precision):>12}  {_fmt(surv, precision):>12}  "
                f"{_fmt(hazard, precision):>12}"
            )
        lines.append("")

    lines += [
        f"  {'mean:':<16} {_fmt(float(result['mean']), precision):>14}",
        f"  {'median:':<16} {_fmt(float(result['median']), precision):>14}",
        f"  {'variance:':<16} {_fmt(float(result['variance']), precision):>14}",
    ]
    return "\n".join(lines)


def format_json(result: dict[str, Any]) -> str:
    """Format the result mapping as JSON.

    Args:
        result: Mapping from :func:`build_result`.

    Returns:
        JSON string.
    """
    data = dict(result)
    rows = data.get("table")
    if isinstance(rows, list):
        data["table"] = [
            {"x": x, "pdf": pdf, "cdf": cdf, "survival": surv, "hazard": hazard}
            for x, pdf, cdf, surv, hazard in rows
        ]
    return json.dumps(data, indent=2)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Run the Weibull distribution CLI.

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
        result = build_result(args)
        if args.format == "json":
            print(format_json(result))
        else:
            print(format_table(result, args.precision, args.survival))
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
