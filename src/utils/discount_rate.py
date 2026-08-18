#!/usr/bin/env python3
"""Command-line utility for discount rates and present value.

Relates nominal rates, real rates, and inflation through the Fisher equation,
and applies the result to lump sums and cash-flow series.  Pure Python via
``math`` — no external dependencies.

    real_rate = (1 + nominal) / (1 + inflation) - 1

Usage examples:
  discount --nominal 0.08 --inflation 0.03
  discount --nominal 0.08 --fv 10000 --periods 5
  discount --nominal 0.08 --inflation 0.03 --cashflows -1000 300 400 500
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from typing import Any

# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def real_rate(nominal: float, inflation: float) -> float:
    """Real discount rate implied by a nominal rate and inflation.

    Args:
        nominal: Nominal (money) rate per period, e.g. 0.08 for 8%; must be > -1.
        inflation: Inflation rate per period, e.g. 0.03 for 3%; must be > -1.

    Returns:
        (1 + nominal) / (1 + inflation) - 1, the Fisher-equation real rate.

    Raises:
        ValueError: If ``nominal`` or ``inflation`` is <= -1.
    """
    _validate_rate(nominal, "nominal")
    _validate_rate(inflation, "inflation")
    return (1 + nominal) / (1 + inflation) - 1


def nominal_rate(real: float, inflation: float) -> float:
    """Nominal discount rate implied by a real rate and inflation.

    Args:
        real: Real (purchasing-power) rate per period; must be > -1.
        inflation: Inflation rate per period; must be > -1.

    Returns:
        (1 + real) * (1 + inflation) - 1.

    Raises:
        ValueError: If ``real`` or ``inflation`` is <= -1.
    """
    _validate_rate(real, "real")
    _validate_rate(inflation, "inflation")
    return (1 + real) * (1 + inflation) - 1


def discount_factor(rate: float, periods: float) -> float:
    """Present-value factor for one amount ``periods`` periods out.

    Args:
        rate: Discount rate per period; must be > -1.
        periods: Number of periods; must be >= 0.

    Returns:
        1 / (1 + rate)^periods.

    Raises:
        ValueError: If ``rate`` <= -1 or ``periods`` < 0.
    """
    _validate_rate(rate, "rate")
    if periods < 0:
        raise ValueError(f"periods must be >= 0, got {periods}")
    return 1 / (1 + rate) ** periods


def present_value(fv: float, rate: float, periods: float) -> float:
    """Present value of a future lump sum.

    Args:
        fv: Future value of the lump sum.
        rate: Discount rate per period; must be > -1.
        periods: Number of periods until the amount is received; must be >= 0.

    Returns:
        fv / (1 + rate)^periods.

    Raises:
        ValueError: If ``rate`` <= -1 or ``periods`` < 0.
    """
    return fv * discount_factor(rate, periods)


def npv(rate: float, cash_flows: Sequence[float]) -> float:
    """Net present value of a cash-flow series.

    The first cash flow is treated as occurring at period 0 (undiscounted), so
    an up-front investment is entered as a negative first element.

    Args:
        rate: Discount rate per period; must be > -1.
        cash_flows: Cash flows in period order; must be non-empty.

    Returns:
        sum over t of cash_flows[t] / (1 + rate)^t.

    Raises:
        ValueError: If ``rate`` <= -1 or ``cash_flows`` is empty.
    """
    _validate_rate(rate, "rate")
    if not cash_flows:
        raise ValueError("cash_flows must have at least one value")
    return sum(cf / (1 + rate) ** t for t, cf in enumerate(cash_flows))


def real_npv(nominal: float, inflation: float, cash_flows: Sequence[float]) -> float:
    """Inflation-adjusted NPV of cash flows stated in today's purchasing power.

    Constant-dollar (real) cash flows must be discounted at the real rate; the
    real rate is derived from ``nominal`` and ``inflation`` via the Fisher
    equation.

    Args:
        nominal: Nominal discount rate per period; must be > -1.
        inflation: Inflation rate per period; must be > -1.
        cash_flows: Real cash flows in period order; must be non-empty.

    Returns:
        NPV of ``cash_flows`` discounted at ``real_rate(nominal, inflation)``.

    Raises:
        ValueError: If a rate is <= -1 or ``cash_flows`` is empty.
    """
    return npv(real_rate(nominal, inflation), cash_flows)


def payback_period(cash_flows: Sequence[float], rate: float) -> float | None:
    """First period at which the discounted cumulative cash flow turns positive.

    Args:
        cash_flows: Cash flows in period order; must be non-empty.
        rate: Discount rate per period; must be > -1.

    Returns:
        The fractional period at which cumulative discounted cash flow crosses
        zero, interpolating within the crossing period, or ``None`` if it never
        does.

    Raises:
        ValueError: If ``rate`` <= -1 or ``cash_flows`` is empty.
    """
    _validate_rate(rate, "rate")
    if not cash_flows:
        raise ValueError("cash_flows must have at least one value")
    cumulative = 0.0
    for t, cf in enumerate(cash_flows):
        discounted = cf / (1 + rate) ** t
        previous = cumulative
        cumulative += discounted
        if cumulative >= 0 > previous:
            # Interpolate inside the crossing period rather than rounding up.
            return t - 1 + (-previous / discounted)
        if t == 0 and cumulative >= 0:
            return 0.0
    return None


def _validate_rate(rate: float, name: str) -> None:
    """Raise if a per-period rate is at or below -100%."""
    if rate <= -1:
        raise ValueError(f"{name} must be > -1, got {rate}")


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
        description="Discount rate, present value, and inflation-adjusted NPV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  discount --nominal 0.08 --inflation 0.03
  discount --real 0.05 --inflation 0.03
  discount --nominal 0.08 --fv 10000 --periods 5
  discount --nominal 0.08 --inflation 0.03 --cashflows -1000 300 400 500
""",
    )
    parser.add_argument(
        "--nominal",
        type=float,
        default=None,
        metavar="F",
        help="nominal discount rate per period, e.g. 0.08 for 8%%",
    )
    parser.add_argument(
        "--real",
        type=float,
        default=None,
        metavar="F",
        help="real discount rate per period (requires --inflation)",
    )
    parser.add_argument(
        "--inflation",
        type=float,
        default=None,
        metavar="F",
        help="inflation rate per period, e.g. 0.03 for 3%%",
    )
    parser.add_argument(
        "--fv",
        type=float,
        default=None,
        metavar="F",
        help="future value of a lump sum to discount (requires --periods)",
    )
    parser.add_argument(
        "--periods",
        type=float,
        default=1.0,
        metavar="F",
        help="number of periods for the lump sum and discount factor (default: 1)",
    )
    parser.add_argument(
        "--cashflows",
        type=float,
        nargs="+",
        default=None,
        metavar="F",
        help="cash flow series starting at period 0, for NPV",
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
    if args.nominal is None and args.real is None:
        return "one of --nominal or --real is required"
    if args.nominal is None and args.inflation is None:
        return "--real requires --inflation to derive the nominal rate"
    if args.nominal is not None and args.nominal <= -1:
        return f"--nominal must be > -1, got {args.nominal}"
    if args.real is not None and args.real <= -1:
        return f"--real must be > -1, got {args.real}"
    if args.inflation is not None and args.inflation <= -1:
        return f"--inflation must be > -1, got {args.inflation}"
    if args.periods < 0:
        return f"--periods must be >= 0, got {args.periods}"
    if args.precision < 0:
        return "--precision must be non-negative"
    if args.cashflows is not None and not args.cashflows:
        return "--cashflows must list at least one value"
    return None


def resolve_rates(args: argparse.Namespace) -> tuple[float, float | None, float | None]:
    """Reconcile whichever of the nominal, real, and inflation rates were given.

    Args:
        args: Validated argument namespace from :func:`parse_args`.

    Returns:
        Tuple of (nominal, real, inflation).  ``real`` is ``None`` when no
        inflation rate was supplied and none was given directly.

    Raises:
        ValueError: If a supplied rate is <= -1.
    """
    inflation = args.inflation
    if args.nominal is not None:
        nominal = args.nominal
        real = real_rate(nominal, inflation) if inflation is not None else None
        # An explicit --real alongside --nominal is informational only; the
        # Fisher-derived value is what the report uses.
    else:
        # validate() guarantees inflation is present on this branch.
        nominal = nominal_rate(args.real, inflation)
        real = args.real
    return nominal, real, inflation


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def _fmt(value: float, precision: int) -> str:
    """Format a float to the requested number of decimal places."""
    return f"{value:,.{precision}f}"


def _pct(value: float, precision: int) -> str:
    """Format a per-period rate as a percentage string."""
    return f"{value * 100:.{precision}f}%"


def build_result(args: argparse.Namespace) -> dict[str, Any]:
    """Compute every requested figure into a single result mapping.

    Args:
        args: Validated argument namespace from :func:`parse_args`.

    Returns:
        Mapping of the reconciled rates plus any ``lump_sum`` and ``npv``
        sections that the flags asked for.

    Raises:
        ValueError: If a core computation rejects its inputs.
    """
    nominal, real, inflation = resolve_rates(args)
    result: dict[str, Any] = {
        "nominal_rate": nominal,
        "inflation": inflation,
        "real_rate": real,
        "periods": args.periods,
        "discount_factor_nominal": discount_factor(nominal, args.periods),
        "discount_factor_real": (
            discount_factor(real, args.periods) if real is not None else None
        ),
    }

    if args.fv is not None:
        result["lump_sum"] = {
            "future_value": args.fv,
            "present_value_nominal": present_value(args.fv, nominal, args.periods),
            "present_value_real": (
                present_value(args.fv, real, args.periods) if real is not None else None
            ),
        }

    if args.cashflows is not None:
        nominal_npv = npv(nominal, args.cashflows)
        result["npv"] = {
            "cash_flows": list(args.cashflows),
            "nominal": nominal_npv,
            "real": (
                real_npv(nominal, inflation, args.cashflows)
                if inflation is not None
                else None
            ),
            "payback_period": payback_period(args.cashflows, nominal),
        }
    return result


def format_table(result: dict[str, Any], precision: int) -> str:
    """Format the result mapping as an aligned text report.

    Args:
        result: Mapping from :func:`build_result`.
        precision: Decimal places for all floating-point output.

    Returns:
        Multi-line string ready to print.
    """
    periods = float(result["periods"])
    lines = [
        "Discount rate analysis",
        "",
        f"  {'Nominal rate:':<32} {_pct(float(result['nominal_rate']), precision):>14}",
    ]
    if result["inflation"] is not None:
        lines.append(
            f"  {'Inflation:':<32} {_pct(float(result['inflation']), precision):>14}"
        )
        lines.append(
            f"  {'Real rate (Fisher):':<32} "
            f"{_pct(float(result['real_rate']), precision):>14}"
        )
    lines += [
        "",
        f"  {'Periods:':<32} {periods:>14g}",
        f"  {'Discount factor (nominal):':<32} "
        f"{_fmt(float(result['discount_factor_nominal']), precision):>14}",
    ]
    if result["discount_factor_real"] is not None:
        lines.append(
            f"  {'Discount factor (real):':<32} "
            f"{_fmt(float(result['discount_factor_real']), precision):>14}"
        )

    lump = result.get("lump_sum")
    if isinstance(lump, dict):
        lines += [
            "",
            f"  {'Future value:':<32} {_fmt(lump['future_value'], precision):>14}",
            f"  {'Present value (nominal):':<32} "
            f"{_fmt(lump['present_value_nominal'], precision):>14}",
        ]
        if lump["present_value_real"] is not None:
            lines.append(
                f"  {'Present value (real):':<32} "
                f"{_fmt(lump['present_value_real'], precision):>14}"
            )

    series = result.get("npv")
    if isinstance(series, dict):
        lines += [
            "",
            f"  {'Cash flows:':<32} {len(series['cash_flows']):>14}",
            f"  {'NPV (nominal):':<32} {_fmt(series['nominal'], precision):>14}",
        ]
        if series["real"] is not None:
            lines.append(f"  {'NPV (real):':<32} {_fmt(series['real'], precision):>14}")
        payback = series["payback_period"]
        payback_text = "never" if payback is None else _fmt(payback, precision)
        lines.append(f"  {'Discounted payback (periods):':<32} {payback_text:>14}")
        lines += [
            "",
            "  Note: real NPV assumes the cash flows are stated in today's",
            "  purchasing power and discounts them at the real rate.",
        ]
    return "\n".join(lines)


def format_json(result: dict[str, Any]) -> str:
    """Format the result mapping as JSON.

    Args:
        result: Mapping from :func:`build_result`.

    Returns:
        JSON string.
    """
    return json.dumps(result, indent=2)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Run the discount rate CLI.

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
            print(format_table(result, args.precision))
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
