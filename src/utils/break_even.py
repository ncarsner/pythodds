#!/usr/bin/env python3
"""Command-line utility for break-even (cost-volume-profit) analysis.

Answers the core viability question: at what unit volume does revenue cover
total cost?  Pure Python — no external dependencies.  The optional ``--chart``
flag renders a text bar chart rather than pulling in a plotting library.

Usage examples:
  breakeven --fixed 50000 --price 25 --variable 10
  breakeven --fixed 50000 --price 25 --variable 10 --target-profit 20000
  breakeven --fixed 50000 --price 25 --variable 10 --sweep 0 8000 500 --chart
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def contribution_margin(price: float, variable: float) -> float:
    """Contribution margin per unit.

    Args:
        price: Selling price per unit; must be > 0.
        variable: Variable cost per unit; must be >= 0.

    Returns:
        price - variable, the cash each unit contributes toward fixed costs.

    Raises:
        ValueError: If ``price`` <= 0 or ``variable`` < 0.
    """
    _validate_price_variable(price, variable)
    return price - variable


def contribution_margin_ratio(price: float, variable: float) -> float:
    """Contribution margin as a fraction of the selling price.

    Args:
        price: Selling price per unit; must be > 0.
        variable: Variable cost per unit; must be >= 0.

    Returns:
        (price - variable) / price, in (-inf, 1].

    Raises:
        ValueError: If ``price`` <= 0 or ``variable`` < 0.
    """
    _validate_price_variable(price, variable)
    return (price - variable) / price


def breakeven_units(fixed: float, price: float, variable: float) -> float:
    """Unit volume at which total revenue equals total cost.

    Args:
        fixed: Total fixed costs; must be >= 0.
        price: Selling price per unit; must be > 0.
        variable: Variable cost per unit; must be >= 0 and < ``price``.

    Returns:
        fixed / (price - variable).

    Raises:
        ValueError: If an input is out of range, or if ``price`` <= ``variable``
            (no contribution margin means the break-even point never arrives).
    """
    _validate_fixed(fixed)
    margin = contribution_margin(price, variable)
    if margin <= 0:
        raise ValueError(
            f"price must exceed variable cost, got price={price}, variable={variable}"
        )
    return fixed / margin


def breakeven_revenue(fixed: float, margin_ratio: float) -> float:
    """Sales revenue at which total revenue equals total cost.

    Args:
        fixed: Total fixed costs; must be >= 0.
        margin_ratio: Contribution margin ratio, in (0, 1].

    Returns:
        fixed / margin_ratio.

    Raises:
        ValueError: If ``fixed`` < 0 or ``margin_ratio`` is not in (0, 1].
    """
    _validate_fixed(fixed)
    if not (0 < margin_ratio <= 1):
        raise ValueError(f"margin_ratio must be in (0, 1], got {margin_ratio}")
    return fixed / margin_ratio


def margin_of_safety(actual_units: float, be_units: float) -> float:
    """Fraction by which actual volume may fall before hitting break-even.

    Args:
        actual_units: Actual or projected unit volume; must be > 0.
        be_units: Break-even unit volume; must be >= 0.

    Returns:
        (actual_units - be_units) / actual_units.  Negative when actual volume
        is already below break-even.

    Raises:
        ValueError: If ``actual_units`` <= 0 or ``be_units`` < 0.
    """
    if actual_units <= 0:
        raise ValueError(f"actual_units must be > 0, got {actual_units}")
    if be_units < 0:
        raise ValueError(f"be_units must be >= 0, got {be_units}")
    return (actual_units - be_units) / actual_units


def target_profit_units(
    fixed: float, price: float, variable: float, profit: float
) -> float:
    """Unit volume required to reach a target operating profit.

    Args:
        fixed: Total fixed costs; must be >= 0.
        price: Selling price per unit; must be > 0.
        variable: Variable cost per unit; must be >= 0 and < ``price``.
        profit: Target profit; may be negative to model an acceptable loss.

    Returns:
        (fixed + profit) / (price - variable).

    Raises:
        ValueError: If an input is out of range, or if ``price`` <= ``variable``.
    """
    _validate_fixed(fixed)
    margin = contribution_margin(price, variable)
    if margin <= 0:
        raise ValueError(
            f"price must exceed variable cost, got price={price}, variable={variable}"
        )
    return (fixed + profit) / margin


def profit_at(units: float, fixed: float, price: float, variable: float) -> float:
    """Operating profit at a given unit volume.

    Args:
        units: Unit volume; must be >= 0.
        fixed: Total fixed costs; must be >= 0.
        price: Selling price per unit; must be > 0.
        variable: Variable cost per unit; must be >= 0.

    Returns:
        units * (price - variable) - fixed.

    Raises:
        ValueError: If an input is out of range.
    """
    if units < 0:
        raise ValueError(f"units must be >= 0, got {units}")
    _validate_fixed(fixed)
    return units * contribution_margin(price, variable) - fixed


def sweep_rows(
    min_units: float,
    max_units: float,
    step: float,
    fixed: float,
    price: float,
    variable: float,
) -> list[tuple[float, float, float, float]]:
    """Build a profit/loss table across a unit range.

    Args:
        min_units: First unit volume in the table; must be >= 0.
        max_units: Last unit volume in the table; must be >= ``min_units``.
        step: Increment between rows; must be > 0.
        fixed: Total fixed costs; must be >= 0.
        price: Selling price per unit; must be > 0.
        variable: Variable cost per unit; must be >= 0.

    Returns:
        List of (units, revenue, total_cost, profit) tuples.

    Raises:
        ValueError: If the range or step is invalid, or a cost input is invalid.
    """
    if step <= 0:
        raise ValueError(f"step must be > 0, got {step}")
    if min_units < 0:
        raise ValueError(f"min_units must be >= 0, got {min_units}")
    if max_units < min_units:
        raise ValueError(
            f"max_units must be >= min_units, got min={min_units}, max={max_units}"
        )
    _validate_fixed(fixed)
    _validate_price_variable(price, variable)

    rows: list[tuple[float, float, float, float]] = []
    # Step through with an integer counter so float accumulation cannot drift.
    count = int((max_units - min_units) / step) + 1
    for i in range(count):
        units = min_units + i * step
        revenue = units * price
        cost = fixed + units * variable
        rows.append((units, revenue, cost, revenue - cost))
    return rows


def _validate_fixed(fixed: float) -> None:
    """Raise if total fixed costs are negative."""
    if fixed < 0:
        raise ValueError(f"fixed must be >= 0, got {fixed}")


def _validate_price_variable(price: float, variable: float) -> None:
    """Raise if the per-unit price or variable cost is out of range."""
    if price <= 0:
        raise ValueError(f"price must be > 0, got {price}")
    if variable < 0:
        raise ValueError(f"variable must be >= 0, got {variable}")


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
        description="Break-even (cost-volume-profit) analysis calculator.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  breakeven --fixed 50000 --price 25 --variable 10
  breakeven --fixed 50000 --price 25 --variable 10 --target-profit 20000
  breakeven --fixed 50000 --price 25 --variable 10 --actual-units 5000
  breakeven --fixed 50000 --price 25 --variable 10 --sweep 0 8000 500 --chart
""",
    )
    parser.add_argument(
        "--fixed",
        type=float,
        required=True,
        metavar="F",
        help="total fixed costs",
    )
    parser.add_argument(
        "--price",
        type=float,
        required=True,
        metavar="F",
        help="selling price per unit",
    )
    parser.add_argument(
        "--variable",
        type=float,
        required=True,
        metavar="F",
        help="variable cost per unit",
    )
    parser.add_argument(
        "--target-profit",
        type=float,
        default=None,
        metavar="F",
        help="also report the unit volume needed for this profit",
    )
    parser.add_argument(
        "--actual-units",
        type=float,
        default=None,
        metavar="F",
        help="actual or projected volume, for the margin of safety",
    )
    parser.add_argument(
        "--sweep",
        type=float,
        nargs=3,
        default=None,
        metavar=("MIN", "MAX", "STEP"),
        help="append a profit/loss table across a unit range",
    )
    parser.add_argument(
        "--chart",
        action="store_true",
        help="render a text bar chart of the sweep profit column (needs --sweep)",
    )
    parser.add_argument(
        "--format",
        choices=["table", "json", "csv"],
        default="table",
        help="output format (default: table)",
    )
    parser.add_argument(
        "--precision",
        "-P",
        type=int,
        default=2,
        metavar="PREC",
        help="decimal places for output (default: 2)",
    )
    return parser.parse_args(argv)


def validate(args: argparse.Namespace) -> str | None:
    """Return an error message string, or ``None`` if arguments are valid.

    Args:
        args: Parsed argument namespace from :func:`parse_args`.

    Returns:
        Error description string, or ``None`` when validation passes.
    """
    if args.fixed < 0:
        return f"--fixed must be >= 0, got {args.fixed}"
    if args.price <= 0:
        return f"--price must be > 0, got {args.price}"
    if args.variable < 0:
        return f"--variable must be >= 0, got {args.variable}"
    if args.price <= args.variable:
        return (
            "--price must exceed --variable; with no contribution margin there "
            "is no break-even volume"
        )
    if args.precision < 0:
        return "--precision must be non-negative"
    if args.actual_units is not None and args.actual_units <= 0:
        return f"--actual-units must be > 0, got {args.actual_units}"

    if args.sweep is not None:
        min_units, max_units, step = args.sweep
        if min_units < 0:
            return f"--sweep MIN must be >= 0, got {min_units}"
        if max_units < min_units:
            return f"--sweep MAX must be >= MIN, got MIN={min_units}, MAX={max_units}"
        if step <= 0:
            return f"--sweep STEP must be > 0, got {step}"
    elif args.chart:
        return "--chart requires --sweep"
    return None


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def _fmt(value: float, precision: int) -> str:
    """Format a float to the requested number of decimal places."""
    return f"{value:.{precision}f}"


def _pct(value: float, precision: int) -> str:
    """Format a fraction as a percentage string."""
    return f"{value * 100:.{precision}f}%"


def build_result(args: argparse.Namespace) -> dict[str, Any]:
    """Compute every requested figure into a single result mapping.

    Args:
        args: Validated argument namespace from :func:`parse_args`.

    Returns:
        Mapping of input echoes, core results, and any optional sections
        (``target_profit``, ``margin_of_safety``, ``sweep``).

    Raises:
        ValueError: If a core computation rejects its inputs.
    """
    margin = contribution_margin(args.price, args.variable)
    ratio = contribution_margin_ratio(args.price, args.variable)
    be_units = breakeven_units(args.fixed, args.price, args.variable)

    result: dict[str, Any] = {
        "fixed": args.fixed,
        "price": args.price,
        "variable": args.variable,
        "contribution_margin": margin,
        "contribution_margin_ratio": ratio,
        "breakeven_units": be_units,
        "breakeven_revenue": breakeven_revenue(args.fixed, ratio),
    }

    if args.target_profit is not None:
        units = target_profit_units(
            args.fixed, args.price, args.variable, args.target_profit
        )
        result["target_profit"] = {
            "profit": args.target_profit,
            "units": units,
            "revenue": units * args.price,
        }

    if args.actual_units is not None:
        result["margin_of_safety"] = {
            "actual_units": args.actual_units,
            "units": args.actual_units - be_units,
            "ratio": margin_of_safety(args.actual_units, be_units),
            "profit": profit_at(
                args.actual_units, args.fixed, args.price, args.variable
            ),
        }

    if args.sweep is not None:
        min_units, max_units, step = args.sweep
        result["sweep"] = sweep_rows(
            min_units, max_units, step, args.fixed, args.price, args.variable
        )

    return result


def format_chart(rows: list[tuple[float, float, float, float]], width: int = 40) -> str:
    """Render the sweep profit column as a text bar chart.

    Losses extend left of a zero axis and profits extend right, so the
    break-even crossing is visible as the point where the bars flip sides.

    Args:
        rows: Sweep rows from :func:`sweep_rows`.
        width: Maximum bar length in characters for the largest magnitude.

    Returns:
        Multi-line string ready to print; empty string when ``rows`` is empty.
    """
    if not rows:
        return ""
    peak = max(abs(row[3]) for row in rows)
    lines = ["", "  Profit / loss chart  (| = break-even axis)", ""]
    for units, _revenue, _cost, profit in rows:
        # Scale each bar against the largest absolute profit or loss in the set.
        bars = 0 if peak == 0 else round(abs(profit) / peak * width)
        if profit < 0:
            left, right = "#" * bars, ""
        else:
            left, right = "", "#" * bars
        lines.append(f"  {units:>10,.0f}  {left:>{width}}|{right}")
    return "\n".join(lines)


def format_table(result: dict[str, Any], precision: int, chart: bool) -> str:
    """Format the result mapping as an aligned text report.

    Args:
        result: Mapping from :func:`build_result`.
        precision: Decimal places for all floating-point output.
        chart: If True, append a text bar chart of the sweep profit column.

    Returns:
        Multi-line string ready to print.
    """
    lines = [
        "Break-even analysis",
        "",
        f"  Fixed costs:               {_fmt(float(result['fixed']), precision):>14}",
        f"  Price per unit:            {_fmt(float(result['price']), precision):>14}",
        f"  Variable cost per unit:    {_fmt(float(result['variable']), precision):>14}",
        "",
        "  Contribution margin:       "
        f"{_fmt(float(result['contribution_margin']), precision):>14}",
        "  Contribution margin ratio: "
        f"{_pct(float(result['contribution_margin_ratio']), precision):>14}",
        "",
        "  Break-even units:          "
        f"{_fmt(float(result['breakeven_units']), precision):>14}",
        "  Break-even revenue:        "
        f"{_fmt(float(result['breakeven_revenue']), precision):>14}",
    ]

    target = result.get("target_profit")
    if isinstance(target, dict):
        lines += [
            "",
            f"  Target profit:             {_fmt(target['profit'], precision):>14}",
            f"  Units required:            {_fmt(target['units'], precision):>14}",
            f"  Revenue required:          {_fmt(target['revenue'], precision):>14}",
        ]

    mos = result.get("margin_of_safety")
    if isinstance(mos, dict):
        lines += [
            "",
            f"  Actual units:              {_fmt(mos['actual_units'], precision):>14}",
            f"  Profit at actual units:    {_fmt(mos['profit'], precision):>14}",
            f"  Margin of safety (units):  {_fmt(mos['units'], precision):>14}",
            f"  Margin of safety:          {_pct(mos['ratio'], precision):>14}",
        ]

    rows = result.get("sweep")
    if isinstance(rows, list):
        lines += [
            "",
            f"  {'units':>10}  {'revenue':>14}  {'cost':>14}  {'profit':>14}",
        ]
        for units, revenue, cost, profit in rows:
            lines.append(
                f"  {_fmt(units, precision):>10}  {_fmt(revenue, precision):>14}  "
                f"{_fmt(cost, precision):>14}  {_fmt(profit, precision):>14}"
            )
        if chart:
            lines.append(format_chart(rows))

    return "\n".join(lines)


def format_csv(result: dict[str, Any], precision: int) -> str:
    """Format the result mapping as CSV.

    Scalar figures are emitted as ``key,value`` rows; a sweep table, when
    present, follows as its own header plus one row per unit volume.

    Args:
        result: Mapping from :func:`build_result`.
        precision: Decimal places for all floating-point output.

    Returns:
        Multi-line CSV string ready to print.
    """
    lines = ["metric,value"]
    for key in (
        "fixed",
        "price",
        "variable",
        "contribution_margin",
        "contribution_margin_ratio",
        "breakeven_units",
        "breakeven_revenue",
    ):
        lines.append(f"{key},{_fmt(float(result[key]), precision)}")

    target = result.get("target_profit")
    if isinstance(target, dict):
        lines.append(f"target_profit,{_fmt(target['profit'], precision)}")
        lines.append(f"target_profit_units,{_fmt(target['units'], precision)}")
        lines.append(f"target_profit_revenue,{_fmt(target['revenue'], precision)}")

    mos = result.get("margin_of_safety")
    if isinstance(mos, dict):
        lines.append(f"actual_units,{_fmt(mos['actual_units'], precision)}")
        lines.append(f"profit_at_actual_units,{_fmt(mos['profit'], precision)}")
        lines.append(f"margin_of_safety_units,{_fmt(mos['units'], precision)}")
        lines.append(f"margin_of_safety_ratio,{_fmt(mos['ratio'], precision)}")

    rows = result.get("sweep")
    if isinstance(rows, list):
        lines += ["", "units,revenue,cost,profit"]
        for units, revenue, cost, profit in rows:
            lines.append(
                f"{_fmt(units, precision)},{_fmt(revenue, precision)},"
                f"{_fmt(cost, precision)},{_fmt(profit, precision)}"
            )
    return "\n".join(lines)


def format_json(result: dict[str, Any]) -> str:
    """Format the result mapping as JSON.

    Args:
        result: Mapping from :func:`build_result`.

    Returns:
        JSON string.
    """
    data = dict(result)
    rows = data.get("sweep")
    if isinstance(rows, list):
        data["sweep"] = [
            {"units": u, "revenue": r, "cost": c, "profit": p} for u, r, c, p in rows
        ]
    return json.dumps(data, indent=2)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Run the break-even analysis CLI.

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
        elif args.format == "csv":
            print(format_csv(result, args.precision))
        else:
            print(format_table(result, args.precision, args.chart))
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
