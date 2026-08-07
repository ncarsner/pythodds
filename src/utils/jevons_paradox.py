#!/usr/bin/env python3
"""Command-line utility for Jevons' Paradox (rebound effect) analysis.

Jevons' Paradox states that technological efficiency improvements often
stimulate greater resource consumption due to lower effective costs,
potentially offsetting or reversing expected conservation gains. This
tool quantifies the direct rebound effect and determines whether
'backfire' (net consumption increase despite efficiency gains) occurs.

Usage examples:
  jevons --efficiency 0.30 --elasticity 0.5
  jevons -e 0.30 -d 0.5 --baseline 1000 --resource coal
  jevons --sweep-efficiency 0.1 0.9 --elasticity 0.7
  jevons --efficiency 0.30 --sweep-elasticity 0.1 2.0
"""

from __future__ import annotations

import argparse
import json
import math
import sys

# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def expected_savings(baseline: float, eta: float) -> float:
    """Compute expected resource savings without any rebound effect.

    Args:
        baseline: Baseline resource consumption before efficiency gain.
        eta: Efficiency improvement fraction (0 < eta < 1).

    Returns:
        Expected savings assuming demand is perfectly inelastic.

    Raises:
        ValueError: If baseline is non-positive or eta is out of (0, 1).
    """
    if baseline <= 0:
        raise ValueError("baseline must be positive")
    if not (0 < eta < 1):
        raise ValueError("efficiency (eta) must be in (0, 1)")
    return baseline * eta


def rebound_consumption(baseline: float, eta: float, epsilon: float) -> float:
    """Compute additional resource consumption caused by the rebound effect.

    When efficiency lowers the effective cost per unit of energy service,
    demand rises. Rebound = baseline × epsilon × eta × (1 − eta).

    Args:
        baseline: Baseline resource consumption.
        eta: Efficiency improvement fraction (0 < eta < 1).
        epsilon: Absolute price elasticity of demand (epsilon >= 0).

    Returns:
        Additional resource consumption attributable to demand rebound.

    Raises:
        ValueError: If any argument is out of valid range.
    """
    if baseline <= 0:
        raise ValueError("baseline must be positive")
    if not (0 < eta < 1):
        raise ValueError("efficiency (eta) must be in (0, 1)")
    if epsilon < 0:
        raise ValueError("elasticity (epsilon) must be non-negative")
    return baseline * epsilon * eta * (1.0 - eta)


def net_consumption(baseline: float, eta: float, epsilon: float) -> float:
    """Compute net resource consumption after efficiency gain and rebound.

    Net consumption = baseline × (1 + epsilon × eta) × (1 − eta).

    Args:
        baseline: Baseline resource consumption.
        eta: Efficiency improvement fraction (0 < eta < 1).
        epsilon: Absolute price elasticity of demand (epsilon >= 0).

    Returns:
        Net resource consumption after accounting for demand rebound.

    Raises:
        ValueError: If any argument is out of valid range.
    """
    if baseline <= 0:
        raise ValueError("baseline must be positive")
    if not (0 < eta < 1):
        raise ValueError("efficiency (eta) must be in (0, 1)")
    if epsilon < 0:
        raise ValueError("elasticity (epsilon) must be non-negative")
    return baseline * (1.0 + epsilon * eta) * (1.0 - eta)


def rebound_rate(eta: float, epsilon: float) -> float:
    """Compute the rebound rate as a fraction of expected savings.

    Rebound rate = epsilon × (1 − eta). Values: 0 = no rebound (full
    conservation); 1 = 100% rebound (savings fully cancelled); >1 =
    backfire (net consumption exceeds baseline).

    Args:
        eta: Efficiency improvement fraction (0 < eta < 1).
        epsilon: Absolute price elasticity of demand (epsilon >= 0).

    Returns:
        Rebound rate fraction (0.0 to >1.0 for backfire scenarios).

    Raises:
        ValueError: If eta or epsilon is out of valid range.
    """
    if not (0 < eta < 1):
        raise ValueError("efficiency (eta) must be in (0, 1)")
    if epsilon < 0:
        raise ValueError("elasticity (epsilon) must be non-negative")
    return epsilon * (1.0 - eta)


def actual_savings(baseline: float, eta: float, epsilon: float) -> float:
    """Compute actual resource savings after the rebound effect.

    Actual savings = expected_savings − rebound_consumption.
    A negative value indicates backfire (net consumption increase).

    Args:
        baseline: Baseline resource consumption.
        eta: Efficiency improvement fraction (0 < eta < 1).
        epsilon: Absolute price elasticity of demand (epsilon >= 0).

    Returns:
        Actual resource savings (negative if backfire occurs).

    Raises:
        ValueError: If any argument is out of valid range.
    """
    return expected_savings(baseline, eta) - rebound_consumption(baseline, eta, epsilon)


def is_backfire(eta: float, epsilon: float) -> bool:
    """Determine whether the Jevons backfire condition is met.

    Backfire occurs when the rebound rate exceeds 100%, meaning net
    resource consumption rises above the pre-efficiency baseline.

    Args:
        eta: Efficiency improvement fraction (0 < eta < 1).
        epsilon: Absolute price elasticity of demand (epsilon >= 0).

    Returns:
        True if backfire occurs (net consumption > baseline).

    Raises:
        ValueError: If eta or epsilon is out of valid range.
    """
    return rebound_rate(eta, epsilon) > 1.0


def outcome_label(rate: float) -> str:
    """Return a qualitative description of the rebound scenario.

    Args:
        rate: Rebound rate fraction from rebound_rate().

    Returns:
        Human-readable description of the outcome.
    """
    if rate == 0.0:
        return "Full conservation — all efficiency gains become savings"
    if rate < 0.5:
        return "Weak rebound — most efficiency gains translate to savings"
    if rate < 1.0:
        return "Strong rebound — efficiency gains are significantly offset"
    if math.isclose(rate, 1.0, rel_tol=1e-9):
        return "Full rebound — efficiency gains are exactly cancelled"
    return "BACKFIRE — consumption increases despite efficiency improvement"


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def analyze(
    baseline: float,
    eta: float,
    epsilon: float,
) -> dict[str, float | bool]:
    """Run a complete Jevons' Paradox analysis for one scenario.

    Args:
        baseline: Baseline resource consumption.
        eta: Efficiency improvement fraction (0 < eta < 1).
        epsilon: Absolute price elasticity of demand (epsilon >= 0).

    Returns:
        Dict with keys: baseline, eta, epsilon, expected_savings,
        rebound_consumption, actual_savings, net_consumption,
        rebound_rate, rebound_pct, backfire.

    Raises:
        ValueError: If any argument is out of valid range.
    """
    exp_sav = expected_savings(baseline, eta)
    reb_con = rebound_consumption(baseline, eta, epsilon)
    act_sav = actual_savings(baseline, eta, epsilon)
    net_con = net_consumption(baseline, eta, epsilon)
    rate = rebound_rate(eta, epsilon)
    return {
        "baseline": baseline,
        "eta": eta,
        "epsilon": epsilon,
        "expected_savings": exp_sav,
        "rebound_consumption": reb_con,
        "actual_savings": act_sav,
        "net_consumption": net_con,
        "rebound_rate": rate,
        "rebound_pct": rate * 100.0,
        "backfire": rate > 1.0,
    }


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def _fmt(value: float, precision: int) -> str:
    """Format a float to a fixed number of decimal places."""
    return f"{value:.{precision}f}"


def _pct(value: float, precision: int) -> str:
    """Format a fraction as a percentage string."""
    return f"{value * 100:.{precision}f}%"


def format_analysis(
    result: dict[str, float | bool],
    resource: str,
    precision: int,
) -> str:
    """Format a single analysis result as a human-readable table.

    Args:
        result: Analysis result dict from analyze().
        resource: Resource label for display (e.g., 'coal', 'units').
        precision: Decimal places for numeric output.

    Returns:
        Formatted multi-line string.
    """
    baseline = float(result["baseline"])
    eta = float(result["eta"])
    epsilon = float(result["epsilon"])
    exp_sav = float(result["expected_savings"])
    reb_con = float(result["rebound_consumption"])
    act_sav = float(result["actual_savings"])
    net_con = float(result["net_consumption"])
    rate = float(result["rebound_rate"])
    lbl = resource or "units"
    p = precision

    lines = [
        "Jevons' Paradox Analysis",
        "=" * 44,
        f"Efficiency improvement:  {_pct(eta, p)}",
        f"Price elasticity:        {_fmt(epsilon, p)}",
        f"Baseline consumption:    {_fmt(baseline, p)} {lbl}",
        "",
        f"Expected savings (no rebound):  {_fmt(exp_sav, p)} {lbl} ({_pct(eta, p)})",
        f"Rebound consumption:            {_fmt(reb_con, p)} {lbl}",
        f"Actual savings (after rebound): {_fmt(act_sav, p)} {lbl} "
        f"({_pct(act_sav / baseline, p)})",
        f"Net consumption:                {_fmt(net_con, p)} {lbl} "
        f"({_pct(net_con / baseline, p)})",
        f"Rebound rate:                   {_fmt(rate * 100.0, p)}% of expected savings",
        "",
        f"Outcome: {outcome_label(rate)}",
    ]
    return "\n".join(lines)


def format_sweep_table(
    rows: list[dict[str, float | bool]],
    sweep_param: str,
    precision: int,
) -> str:
    """Format sweep results as an aligned table.

    Args:
        rows: List of analysis result dicts from analyze().
        sweep_param: Swept parameter name: 'eta' or 'epsilon'.
        precision: Decimal places for numeric output.

    Returns:
        Formatted table string with one row per sweep value.
    """
    p = precision
    w = precision + 6
    header = (
        f"{'Efficiency':>{w}}  "
        f"{'Elasticity':>{w}}  "
        f"{'Rebound%':>{w}}  "
        f"{'Backfire':>10}  "
        f"{'Net Change%':>{w}}"
    )
    sep = "-" * len(header)
    lines = [header, sep]
    for row in rows:
        eta = float(row["eta"])
        epsilon = float(row["epsilon"])
        rate_pct = float(row["rebound_pct"])
        bf = "YES" if row["backfire"] else "no"
        net_chg_pct = (
            float(row["net_consumption"]) / float(row["baseline"]) - 1.0
        ) * 100.0
        lines.append(
            f"{eta * 100:>{w}.{p}f}%  "
            f"{epsilon:>{w}.{p}f}  "
            f"{rate_pct:>{w}.{p}f}  "
            f"{bf:>10}  "
            f"{net_chg_pct:>{w}.{p}f}%"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Sweep helper
# ---------------------------------------------------------------------------


def _sweep_values(lo: float, hi: float, step: float) -> list[float]:
    """Generate evenly spaced values from lo to hi (inclusive)."""
    n = max(0, round((hi - lo) / step))
    return [lo + i * step for i in range(n + 1)]


# ---------------------------------------------------------------------------
# Argument parsing and validation
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the jevons command.

    Args:
        argv: Argument list; uses sys.argv[1:] when None.

    Returns:
        Parsed namespace.
    """
    parser = argparse.ArgumentParser(
        description="Jevons' Paradox — rebound effect and backfire analysis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  jevons --efficiency 0.30 --elasticity 0.5
  jevons -e 0.30 -d 0.5 --baseline 1000 --resource coal
  jevons --sweep-efficiency 0.1 0.9 --elasticity 0.7
  jevons --efficiency 0.30 --sweep-elasticity 0.1 2.0
""",
    )

    sweep_group = parser.add_mutually_exclusive_group()
    sweep_group.add_argument(
        "--sweep-efficiency",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        help="sweep efficiency from MIN to MAX (requires --elasticity)",
    )
    sweep_group.add_argument(
        "--sweep-elasticity",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        help="sweep elasticity from MIN to MAX (requires --efficiency)",
    )

    parser.add_argument(
        "--efficiency",
        "-e",
        type=float,
        metavar="ETA",
        help="efficiency improvement fraction (0 < eta < 1)",
    )
    parser.add_argument(
        "--elasticity",
        "-d",
        type=float,
        metavar="EPSILON",
        help="absolute price elasticity of demand (>= 0)",
    )
    parser.add_argument(
        "--baseline",
        "-b",
        type=float,
        default=1.0,
        metavar="C0",
        help="baseline resource consumption (default: 1.0)",
    )
    parser.add_argument(
        "--resource",
        "-r",
        type=str,
        default="units",
        metavar="LABEL",
        help="resource label for display (default: units)",
    )
    parser.add_argument(
        "--step",
        type=float,
        default=0.05,
        metavar="STEP",
        help="step size for sweep (default: 0.05)",
    )
    parser.add_argument(
        "--format",
        "-f",
        choices=["table", "json"],
        default="table",
        help="output format: table (default) or json",
    )
    parser.add_argument(
        "--precision",
        "-P",
        type=int,
        default=4,
        metavar="DIGITS",
        help="decimal places for printed values (default: 4)",
    )

    return parser.parse_args(argv)


def validate(args: argparse.Namespace) -> str | None:
    """Validate parsed CLI arguments.

    Args:
        args: Parsed argument namespace from parse_args().

    Returns:
        Error message if invalid; None if all arguments are valid.
    """
    if args.precision < 0:
        return "--precision must be non-negative"
    if args.baseline <= 0:
        return "--baseline must be positive"
    if args.step <= 0:
        return "--step must be positive"

    is_sweep_eff = args.sweep_efficiency is not None
    is_sweep_ela = args.sweep_elasticity is not None

    if is_sweep_eff:
        lo, hi = args.sweep_efficiency
        if not (0 < lo < 1):
            return "--sweep-efficiency MIN must be in (0, 1)"
        if not (0 < hi < 1):
            return "--sweep-efficiency MAX must be in (0, 1)"
        if lo >= hi:
            return "--sweep-efficiency MIN must be less than MAX"
        if args.elasticity is None:
            return "--elasticity is required when using --sweep-efficiency"
        if args.elasticity < 0:
            return "--elasticity must be non-negative"
        return None

    if is_sweep_ela:
        lo, hi = args.sweep_elasticity
        if lo < 0:
            return "--sweep-elasticity MIN must be non-negative"
        if hi < 0:
            return "--sweep-elasticity MAX must be non-negative"
        if lo >= hi:
            return "--sweep-elasticity MIN must be less than MAX"
        if args.efficiency is None:
            return "--efficiency is required when using --sweep-elasticity"
        if not (0 < args.efficiency < 1):
            return "--efficiency must be in (0, 1)"
        return None

    # Single analysis mode
    if args.efficiency is None:
        return "--efficiency is required"
    if args.elasticity is None:
        return "--elasticity is required"
    if not (0 < args.efficiency < 1):
        return "--efficiency must be in (0, 1)"
    if args.elasticity < 0:
        return "--elasticity must be non-negative"
    return None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Entry point for the jevons CLI command.

    Args:
        argv: Argument list; uses sys.argv[1:] when None.

    Returns:
        Exit code: 0 on success, 2 on argument or calculation error.
    """
    args = parse_args(argv)

    error = validate(args)
    if error:
        print(f"Error: {error}", file=sys.stderr)
        return 2

    precision = args.precision
    fmt = args.format
    baseline = args.baseline
    resource = args.resource

    if args.sweep_efficiency is not None or args.sweep_elasticity is not None:
        rows: list[dict[str, float | bool]] = []

        if args.sweep_efficiency is not None:
            lo, hi = args.sweep_efficiency
            for eta in _sweep_values(lo, hi, args.step):
                if 0 < eta < 1:
                    rows.append(analyze(baseline, eta, args.elasticity))
            sweep_param = "eta"
        else:
            lo, hi = args.sweep_elasticity
            for epsilon in _sweep_values(lo, hi, args.step):
                if epsilon >= 0:
                    rows.append(analyze(baseline, args.efficiency, epsilon))
            sweep_param = "epsilon"

        if fmt == "json":
            print(json.dumps(rows, indent=2))
        else:
            print(format_sweep_table(rows, sweep_param, precision))
        return 0

    try:
        result = analyze(baseline, args.efficiency, args.elasticity)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    if fmt == "json":
        print(json.dumps(result, indent=2))
    else:
        print(format_analysis(result, resource, precision))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
