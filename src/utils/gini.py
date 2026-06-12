#!/usr/bin/env python3
"""Command-line utility for the Gini coefficient and Lorenz curve.

Computes inequality metrics from raw observations, weighted samples, or
pre-aggregated population/income share pairs.  Pure Python — no external
dependencies.

Usage examples:
  gini --data 1 2 3 4 5
  gini --data 1 2 3 4 5 --lorenz --precision 4
  gini --data 1 2 3 --weights 1 2 1 --correct
  gini --groups 0.2 0.05 0.3 0.15 0.5 0.80
  gini --data 1 2 3 --data 4 5 6 7 --data 2 2 2
"""

from __future__ import annotations

import argparse
import json
import math
import sys

# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def _gini_trapezoid(values: list[float], weights: list[float]) -> float:
    """Compute Gini via sorted Lorenz-trapezoid formula.

    Args:
        values: Non-negative observation values.
        weights: Positive weights parallel to values; must sum to a positive number.

    Returns:
        Gini coefficient in [0, 1).
    """
    pairs = sorted(zip(values, weights), key=lambda t: t[0])
    W = sum(w for _, w in pairs)
    total = sum(x * w for x, w in pairs)
    if total == 0.0:
        return 0.0
    prev_F = 0.0
    prev_L = 0.0
    area = 0.0
    cum_w = 0.0
    cum_xw = 0.0
    for x, w in pairs:
        cum_w += w
        cum_xw += x * w
        F = cum_w / W
        L = cum_xw / total
        area += (F - prev_F) * (L + prev_L) / 2.0
        prev_F = F
        prev_L = L
    return 1.0 - 2.0 * area


def gini_coefficient(
    data: list[float],
    weights: list[float] | None = None,
    corrected: bool = False,
) -> float:
    """Compute the Gini coefficient for a dataset.

    Args:
        data: Non-negative observations.  Must not be empty; sum must be > 0.
        weights: Optional positive weights parallel to data.  When None, each
            observation receives equal weight.
        corrected: If True, apply the n/(n−1) small-sample bias correction.

    Returns:
        Gini coefficient; 0 = perfect equality, approaching 1 = maximum inequality.

    Raises:
        ValueError: If data is empty, contains negative values, sums to zero,
            or weights are invalid.
    """
    if not data:
        raise ValueError("data must not be empty")
    if any(x < 0 for x in data):
        raise ValueError("data values must be non-negative")
    n = len(data)
    if weights is None:
        w = [1.0] * n
    else:
        if len(weights) != n:
            raise ValueError("weights length must match data length")
        if any(ww <= 0 for ww in weights):
            raise ValueError("weights must be positive")
        w = list(weights)
    if sum(x * ww for x, ww in zip(data, w)) == 0.0:
        raise ValueError("sum of values must be positive")
    g = _gini_trapezoid(data, w)
    if corrected and n > 1:
        g = g * n / (n - 1)
    return g


def lorenz_curve(
    data: list[float],
    weights: list[float] | None = None,
) -> list[tuple[float, float]]:
    """Compute Lorenz curve coordinates.

    Args:
        data: Non-negative observations.
        weights: Optional positive weights parallel to data.

    Returns:
        List of (cumulative_population_share, cumulative_income_share) tuples,
        starting at (0.0, 0.0) and ending at (1.0, 1.0).

    Raises:
        ValueError: If data is invalid or sum is zero.
    """
    if not data:
        raise ValueError("data must not be empty")
    if any(x < 0 for x in data):
        raise ValueError("data values must be non-negative")
    n = len(data)
    if weights is None:
        w = [1.0] * n
    else:
        if len(weights) != n:
            raise ValueError("weights length must match data length")
        if any(ww <= 0 for ww in weights):
            raise ValueError("weights must be positive")
        w = list(weights)
    W = sum(w)
    total = sum(x * ww for x, ww in zip(data, w))
    if total == 0.0:
        raise ValueError("sum of values must be positive for Lorenz curve")
    pairs = sorted(zip(data, w), key=lambda t: t[0])
    points: list[tuple[float, float]] = [(0.0, 0.0)]
    cum_w = 0.0
    cum_xw = 0.0
    for x, ww in pairs:
        cum_w += ww
        cum_xw += x * ww
        points.append((cum_w / W, cum_xw / total))
    return points


def relative_mad(
    data: list[float],
    weights: list[float] | None = None,
) -> float:
    """Compute the relative mean absolute difference (= 2 × Gini coefficient).

    Args:
        data: Non-negative observations.
        weights: Optional positive weights.

    Returns:
        Relative mean absolute difference.

    Raises:
        ValueError: See gini_coefficient.
    """
    return 2.0 * gini_coefficient(data, weights=weights, corrected=False)


def gini_grouped(groups: list[tuple[float, float]]) -> float:
    """Compute the Gini coefficient from pre-aggregated (pop_share, inc_share) pairs.

    Groups are sorted internally by per-capita income (inc_share / pop_share).

    Args:
        groups: List of (population_share, income_share) tuples.  Population
            shares must be positive and sum to 1; income shares must be
            non-negative and sum to 1.

    Returns:
        Gini coefficient computed via Lorenz trapezoid area.

    Raises:
        ValueError: If groups are empty, shares are invalid, or sums deviate from 1.
    """
    if not groups:
        raise ValueError("groups must not be empty")
    if any(p <= 0 for p, _ in groups):
        raise ValueError("population shares must be positive")
    if any(s < 0 for _, s in groups):
        raise ValueError("income shares must be non-negative")
    total_pop = sum(p for p, _ in groups)
    total_inc = sum(s for _, s in groups)
    if not math.isclose(total_pop, 1.0, rel_tol=1e-5, abs_tol=1e-5):
        raise ValueError(f"population shares must sum to 1.0 (got {total_pop:.6f})")
    if not math.isclose(total_inc, 1.0, rel_tol=1e-5, abs_tol=1e-5):
        raise ValueError(f"income shares must sum to 1.0 (got {total_inc:.6f})")
    # Sort groups by ascending per-capita income
    sorted_groups = sorted(groups, key=lambda g: g[1] / g[0])
    area = 0.0
    prev_x = 0.0
    prev_y = 0.0
    cum_pop = 0.0
    cum_inc = 0.0
    for pop, inc in sorted_groups:
        cum_pop += pop
        cum_inc += inc
        area += (cum_pop - prev_x) * (cum_inc + prev_y) / 2.0
        prev_x = cum_pop
        prev_y = cum_inc
    return 1.0 - 2.0 * area


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list; uses sys.argv[1:] when None.

    Returns:
        Parsed namespace.
    """
    parser = argparse.ArgumentParser(
        description="Gini coefficient and Lorenz curve calculator.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  gini --data 1 2 3 4 5
  gini --data 1 2 3 4 5 --lorenz
  gini --data 1 2 3 --weights 2 1 1 --correct
  gini --groups 0.2 0.05 0.3 0.15 0.5 0.80
  gini --data 10 30 50 --data 5 5 90 --data 30 35 35
""",
    )

    parser.add_argument(
        "--data",
        type=float,
        nargs="+",
        action="append",
        metavar="VALUE",
        help="observed values (repeat flag for multiple datasets)",
    )
    parser.add_argument(
        "--groups",
        type=float,
        nargs="+",
        metavar="VALUE",
        help="alternating population and income shares: POP INC POP INC ...",
    )
    parser.add_argument(
        "--weights",
        type=float,
        nargs="+",
        metavar="W",
        help="positive weights parallel to --data (single dataset only)",
    )
    parser.add_argument(
        "--lorenz",
        action="store_true",
        help="display Lorenz curve coordinates (single dataset only)",
    )
    parser.add_argument(
        "--correct",
        action="store_true",
        help="apply n/(n−1) small-sample bias correction",
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
        default=6,
        metavar="DIGITS",
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
        Error message if invalid; None if all arguments are valid.
    """
    if args.precision < 0:
        return "--precision must be non-negative"

    has_data = bool(args.data)
    has_groups = args.groups is not None

    if not has_data and not has_groups:
        return "one of --data or --groups is required"
    if has_data and has_groups:
        return "--data and --groups are mutually exclusive"

    if has_groups:
        if len(args.groups) < 2 or len(args.groups) % 2 != 0:
            return "--groups requires an even number of values (POP INC pairs)"

    if has_data:
        n_datasets = len(args.data)
        if args.weights is not None:
            if n_datasets > 1:
                return "--weights is only supported with a single --data group"
            if len(args.weights) != len(args.data[0]):
                return "--weights length must match --data length"
            if any(w <= 0 for w in args.weights):
                return "--weights values must be positive"
        if n_datasets > 1 and args.lorenz:
            return "--lorenz is only supported with a single dataset"

    return None


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def _fmt(x: float, p: int) -> str:
    return f"{x:.{p}f}"


def format_single(
    n: int,
    mean: float,
    gini: float,
    rmad: float,
    corrected: float | None,
    precision: int,
) -> str:
    """Format point-estimate output for a single dataset.

    Args:
        n: Number of observations.
        mean: Arithmetic mean (or weighted mean).
        gini: Gini coefficient.
        rmad: Relative mean absolute difference.
        corrected: Bias-corrected Gini, or None if not requested.
        precision: Decimal places.

    Returns:
        Formatted string.
    """
    p = precision
    lines = [
        "Gini Coefficient Analysis",
        "=" * 40,
        f"Observations:           {n}",
        f"Mean:                   {_fmt(mean, p)}",
        f"Gini coefficient:       {_fmt(gini, p)}",
    ]
    if corrected is not None:
        lines.append(f"Corrected Gini (n/n-1): {_fmt(corrected, p)}")
    lines.append(f"Rel. mean abs. diff.:   {_fmt(rmad, p)}")
    return "\n".join(lines)


def format_lorenz(
    points: list[tuple[float, float]],
    gini: float,
    precision: int,
) -> str:
    """Format Lorenz curve coordinates as a table.

    Args:
        points: List of (pop_share, inc_share) from lorenz_curve().
        gini: Gini coefficient for the header.
        precision: Decimal places.

    Returns:
        Formatted string with header and coordinate table.
    """
    p = precision
    w = max(14, p + 8)
    header = f"{'Population %':>{w}}  {'Income %':>{w}}"
    sep = "-" * len(header)
    rows = [f"{pop:{w}.{p}f}  {inc:{w}.{p}f}" for pop, inc in points]
    lines = [
        f"Lorenz Curve  (Gini = {_fmt(gini, p)})",
        "=" * 40,
        header,
        sep,
    ] + rows
    return "\n".join(lines)


def format_grouped(gini: float, n_groups: int, precision: int) -> str:
    """Format output for grouped-data mode.

    Args:
        gini: Computed Gini coefficient.
        n_groups: Number of input groups.
        precision: Decimal places.

    Returns:
        Formatted string.
    """
    lines = [
        "Gini Coefficient (Grouped Data)",
        "=" * 40,
        f"Groups:                 {n_groups}",
        f"Gini coefficient:       {_fmt(gini, precision)}",
        f"Rel. mean abs. diff.:   {_fmt(2.0 * gini, precision)}",
    ]
    return "\n".join(lines)


def format_comparison(
    results: list[dict[str, float | int]],
    corrected: bool,
    precision: int,
) -> str:
    """Format a multi-dataset comparison table ranked by Gini.

    Args:
        results: List of dicts with keys 'index', 'n', 'mean', 'gini',
            'corrected', 'rank'.
        corrected: Whether to include the corrected Gini column.
        precision: Decimal places.

    Returns:
        Formatted comparison table string.
    """
    p = precision
    w = max(12, p + 6)
    if corrected:
        header = (
            f"{'Dataset':>10}  {'N':>6}  {'Mean':>{w}}  "
            f"{'Gini':>{w}}  {'Corrected':>{w}}  {'Rank':>6}"
        )
    else:
        header = f"{'Dataset':>10}  {'N':>6}  {'Mean':>{w}}  {'Gini':>{w}}  {'Rank':>6}"
    sep = "-" * len(header)
    rows = []
    for r in results:
        if corrected:
            rows.append(
                f"{r['index']:>10}  {r['n']:>6}  {r['mean']:{w}.{p}f}  "
                f"{r['gini']:{w}.{p}f}  {r['corrected']:{w}.{p}f}  {r['rank']:>6}"
            )
        else:
            rows.append(
                f"{r['index']:>10}  {r['n']:>6}  {r['mean']:{w}.{p}f}  "
                f"{r['gini']:{w}.{p}f}  {r['rank']:>6}"
            )
    lines = ["Dataset Comparison", "=" * 40, header, sep] + rows
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _weighted_mean(data: list[float], weights: list[float]) -> float:
    W = sum(weights)
    return sum(x * w for x, w in zip(data, weights)) / W


def main(argv: list[str] | None = None) -> int:
    """Run the gini CLI.

    Args:
        argv: Argument list; uses sys.argv[1:] when None.

    Returns:
        Exit code: 0 on success, 2 on input error.
    """
    args = parse_args(argv)

    error = validate(args)
    if error:
        print(f"Error: {error}", file=sys.stderr)
        return 2

    p = args.precision
    fmt = args.format

    # ---- grouped-data mode ----
    if args.groups is not None:
        vals = args.groups
        groups = [(vals[i], vals[i + 1]) for i in range(0, len(vals), 2)]
        try:
            g = gini_grouped(groups)
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 2
        if fmt == "json":
            print(
                json.dumps(
                    {
                        "mode": "grouped",
                        "n_groups": len(groups),
                        "gini": round(g, p + 4),
                    },
                    indent=2,
                )
            )
        else:
            print(format_grouped(g, len(groups), p))
        return 0

    # ---- raw data mode ----
    datasets: list[list[float]] = args.data
    weights = args.weights

    # Single dataset
    if len(datasets) == 1:
        data = datasets[0]
        w = weights
        try:
            g = gini_coefficient(data, weights=w, corrected=False)
            rmad = relative_mad(data, weights=w)
            g_corr: float | None = None
            if args.correct:
                g_corr = gini_coefficient(data, weights=w, corrected=True)
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 2

        if args.lorenz:
            try:
                points = lorenz_curve(data, weights=w)
            except ValueError as exc:
                print(f"Error: {exc}", file=sys.stderr)
                return 2
            if fmt == "json":
                payload: dict = {
                    "mode": "lorenz",
                    "n": len(data),
                    "gini": round(g, p + 4),
                    "lorenz_curve": [
                        [round(x, p + 4), round(y, p + 4)] for x, y in points
                    ],
                }
                if g_corr is not None:
                    payload["corrected_gini"] = round(g_corr, p + 4)
                print(json.dumps(payload, indent=2))
            else:
                print(format_lorenz(points, g, p))
            return 0

        ew = w if w is not None else [1.0] * len(data)
        mean = _weighted_mean(data, ew)
        if fmt == "json":
            payload = {
                "mode": "single",
                "n": len(data),
                "mean": round(mean, p + 4),
                "gini": round(g, p + 4),
                "relative_mad": round(rmad, p + 4),
            }
            if g_corr is not None:
                payload["corrected_gini"] = round(g_corr, p + 4)
            print(json.dumps(payload, indent=2))
        else:
            print(format_single(len(data), mean, g, rmad, g_corr, p))
        return 0

    # Multiple datasets — comparison mode
    try:
        results = []
        for i, data in enumerate(datasets, start=1):
            g = gini_coefficient(data, corrected=False)
            g_corr = gini_coefficient(data, corrected=True)
            ew = [1.0] * len(data)
            mean = _weighted_mean(data, ew)
            results.append(
                {
                    "index": i,
                    "n": len(data),
                    "mean": mean,
                    "gini": g,
                    "corrected": g_corr,
                }
            )
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    # Assign ranks (1 = most equal)
    ranked = sorted(results, key=lambda r: r["gini"])
    for rank, r in enumerate(ranked, start=1):
        r["rank"] = rank

    if fmt == "json":
        payload = {
            "mode": "compare",
            "datasets": [
                {
                    "index": r["index"],
                    "n": r["n"],
                    "mean": round(r["mean"], p + 4),
                    "gini": round(r["gini"], p + 4),
                    "corrected_gini": round(r["corrected"], p + 4),
                    "rank": r["rank"],
                }
                for r in results
            ],
        }
        print(json.dumps(payload, indent=2))
    else:
        print(format_comparison(results, args.correct, p))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
