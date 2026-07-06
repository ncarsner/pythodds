#!/usr/bin/env python3
"""Command-line utility for chi-square statistics.

Supports goodness-of-fit tests (do observed categorical frequencies match
expected ones?) and tests of independence (are two categorical variables
associated?) on a contingency table.

The chi-square CDF is computed via the regularised incomplete gamma function
(series expansion / continued fraction, using ``math.lgamma``) — pure Python,
no external dependencies.

Usage examples:
  chisq --test gof --observed 18,22,17,25,19,19 --expected 20,20,20,20,20,20
  chisq --test independence --table "40,30,20" --table "25,45,30"
  chisq --test gof --observed 52,48 --expected 50,50 --alpha 0.10
"""

from __future__ import annotations

import argparse
import math
import sys
from typing import Sequence

# ---------------------------------------------------------------------------
# Regularised incomplete gamma function (Numerical Recipes gser/gcf)
# ---------------------------------------------------------------------------

_ITMAX = 200
_EPS = 1e-14
_FPMIN = 1e-300


def _gamma_series(a: float, x: float) -> float:
    """Regularised lower incomplete gamma P(a, x) via series expansion.

    Valid and rapidly convergent for x < a + 1.

    Args:
        a: Shape parameter; must be > 0.
        x: Upper integration bound; must be >= 0.

    Returns:
        P(a, x) in [0, 1].
    """
    gln = math.lgamma(a)
    ap = a
    delta = total = 1.0 / a
    for _ in range(_ITMAX):
        ap += 1
        delta *= x / ap
        total += delta
        if abs(delta) < abs(total) * _EPS:
            break
    return total * math.exp(-x + a * math.log(x) - gln)


def _gamma_cf(a: float, x: float) -> float:
    """Regularised upper incomplete gamma Q(a, x) via continued fraction.

    Valid and rapidly convergent for x >= a + 1.

    Args:
        a: Shape parameter; must be > 0.
        x: Lower integration bound; must be >= a.

    Returns:
        Q(a, x) in [0, 1].
    """
    gln = math.lgamma(a)
    b = x + 1 - a
    c = 1.0 / _FPMIN
    d = 1.0 / b
    h = d
    for i in range(1, _ITMAX + 1):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = b + an / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _EPS:
            break
    return math.exp(-x + a * math.log(x) - gln) * h


def regularized_gamma_p(a: float, x: float) -> float:
    """Regularised lower incomplete gamma function P(a, x).

    Args:
        a: Shape parameter; must be > 0.
        x: Upper integration bound; must be >= 0.

    Returns:
        P(a, x) in [0, 1].

    Raises:
        ValueError: If ``a`` <= 0 or ``x`` < 0.
    """
    if a <= 0:
        raise ValueError(f"a must be > 0, got {a}")
    if x < 0:
        raise ValueError(f"x must be >= 0, got {x}")
    if x == 0:
        return 0.0
    if x < a + 1:
        return _gamma_series(a, x)
    return 1.0 - _gamma_cf(a, x)


def chi2_cdf(x: float, df: int) -> float:
    """Cumulative distribution function P(X <= x) for chi-square(df).

    Args:
        x: Observed chi-square statistic; must be >= 0.
        df: Degrees of freedom; must be >= 1.

    Returns:
        P(X <= x), the regularised lower incomplete gamma P(df/2, x/2).

    Raises:
        ValueError: If ``df`` < 1 or ``x`` < 0.
    """
    if df < 1:
        raise ValueError(f"df must be >= 1, got {df}")
    return regularized_gamma_p(df / 2, x / 2)


def chi2_sf(x: float, df: int) -> float:
    """Survival function (upper tail p-value) for chi-square(df).

    Args:
        x: Observed chi-square statistic; must be >= 0.
        df: Degrees of freedom; must be >= 1.

    Returns:
        P(X > x) = 1 - chi2_cdf(x, df), clipped to [0, 1].
    """
    return max(0.0, min(1.0, 1.0 - chi2_cdf(x, df)))


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


class GofResult:
    """Result of a chi-square goodness-of-fit test."""

    def __init__(
        self,
        statistic: float,
        df: int,
        p_value: float,
        contributions: list[float],
        observed: list[float],
        expected: list[float],
        alpha: float,
    ) -> None:
        """Store all computed quantities."""
        self.statistic = statistic
        self.df = df
        self.p_value = p_value
        self.contributions = contributions
        self.observed = observed
        self.expected = expected
        self.alpha = alpha


class IndependenceResult:
    """Result of a chi-square test of independence."""

    def __init__(
        self,
        statistic: float,
        df: int,
        p_value: float,
        contributions: list[list[float]],
        expected: list[list[float]],
        table: list[list[float]],
        alpha: float,
    ) -> None:
        """Store all computed quantities."""
        self.statistic = statistic
        self.df = df
        self.p_value = p_value
        self.contributions = contributions
        self.expected = expected
        self.table = table
        self.alpha = alpha


# ---------------------------------------------------------------------------
# Core statistical functions
# ---------------------------------------------------------------------------


def chisq_gof(
    observed: Sequence[float], expected: Sequence[float], alpha: float = 0.05
) -> GofResult:
    """Perform a chi-square goodness-of-fit test.

    Tests H₀: the observed frequencies follow the given expected distribution.

    Args:
        observed: Observed category counts/frequencies (non-negative).
        expected: Expected category counts/frequencies (strictly positive).
        alpha: Significance level (default 0.05).

    Returns:
        :class:`GofResult` with statistic, df, p-value, and per-cell contributions.

    Raises:
        ValueError: If lengths mismatch, fewer than 2 categories, any observed
            value is negative, or any expected value is not strictly positive.
    """
    if len(observed) != len(expected):
        raise ValueError(
            f"observed and expected must have the same length, "
            f"got {len(observed)} and {len(expected)}"
        )
    if len(observed) < 2:
        raise ValueError(f"need at least 2 categories, got {len(observed)}")
    if any(o < 0 for o in observed):
        raise ValueError("observed values must be non-negative")
    if any(e <= 0 for e in expected):
        raise ValueError("expected values must be strictly positive")
    if not (0 < alpha < 1):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    contributions = [(o - e) ** 2 / e for o, e in zip(observed, expected)]
    statistic = sum(contributions)
    df = len(observed) - 1
    p_value = chi2_sf(statistic, df)

    return GofResult(
        statistic=statistic,
        df=df,
        p_value=p_value,
        contributions=contributions,
        observed=list(observed),
        expected=list(expected),
        alpha=alpha,
    )


def chisq_independence(
    table: Sequence[Sequence[float]], alpha: float = 0.05
) -> IndependenceResult:
    """Perform a chi-square test of independence on a contingency table.

    Tests H₀: the row and column categorical variables are independent.

    Args:
        table: Rows of observed cell counts/frequencies (non-negative);
            at least 2 rows and 2 columns, each row the same length.
        alpha: Significance level (default 0.05).

    Returns:
        :class:`IndependenceResult` with statistic, df, p-value, expected
        cell values, and per-cell contributions.

    Raises:
        ValueError: If the table has fewer than 2 rows/columns, ragged rows,
            negative values, or a zero row/column total (expected cell = 0).
    """
    if len(table) < 2:
        raise ValueError(f"need at least 2 rows, got {len(table)}")
    ncols = len(table[0])
    if ncols < 2:
        raise ValueError(f"need at least 2 columns, got {ncols}")
    if any(len(row) != ncols for row in table):
        raise ValueError("all rows must have the same number of columns")
    if any(v < 0 for row in table for v in row):
        raise ValueError("table values must be non-negative")

    row_totals = [sum(row) for row in table]
    col_totals = [sum(row[j] for row in table) for j in range(ncols)]
    grand_total = sum(row_totals)
    if grand_total <= 0:
        raise ValueError("grand total of table must be positive")
    if any(rt <= 0 for rt in row_totals) or any(ct <= 0 for ct in col_totals):
        raise ValueError("every row and column total must be positive")
    if not (0 < alpha < 1):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    expected = [
        [row_totals[i] * col_totals[j] / grand_total for j in range(ncols)]
        for i in range(len(table))
    ]
    contributions = [
        [(table[i][j] - expected[i][j]) ** 2 / expected[i][j] for j in range(ncols)]
        for i in range(len(table))
    ]
    statistic = sum(sum(row) for row in contributions)
    df = (len(table) - 1) * (ncols - 1)
    p_value = chi2_sf(statistic, df)

    return IndependenceResult(
        statistic=statistic,
        df=df,
        p_value=p_value,
        contributions=contributions,
        expected=expected,
        table=[list(row) for row in table],
        alpha=alpha,
    )


def parse_number_list(raw: str) -> list[float]:
    """Parse a comma-separated string of finite numbers.

    Args:
        raw: Comma-separated numeric string (e.g. ``"18,22,17"``).

    Returns:
        List of parsed floats.

    Raises:
        ValueError: If the string is empty, non-numeric, or contains non-finite values.
    """
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        raise ValueError("at least one value is required")
    values = [float(p) for p in parts]
    if any(not math.isfinite(v) for v in values):
        raise ValueError("all values must be finite")
    return values


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
        description="Chi-square test calculator (goodness-of-fit, independence).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  chisq --test gof --observed 18,22,17,25,19,19 --expected 20,20,20,20,20,20
  chisq --test independence --table "40,30,20" --table "25,45,30"
  chisq --test gof --observed 52,48 --expected 50,50 --alpha 0.10
""",
    )
    parser.add_argument(
        "--test",
        choices=["gof", "independence"],
        required=True,
        help="test type: 'gof' (goodness-of-fit) or 'independence' (contingency table)",
    )
    parser.add_argument(
        "--observed",
        type=str,
        metavar="O1,O2,...",
        help="comma-separated observed frequencies (gof mode)",
    )
    parser.add_argument(
        "--expected",
        type=str,
        metavar="E1,E2,...",
        help="comma-separated expected frequencies (gof mode)",
    )
    parser.add_argument(
        "--table",
        action="append",
        type=str,
        metavar="R1,R2,...",
        help="one contingency table row of comma-separated values; "
        "repeat --table once per row (independence mode)",
    )
    parser.add_argument(
        "--alpha",
        "-a",
        type=float,
        default=0.05,
        metavar="ALPHA",
        help="significance level (default: 0.05)",
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
    if not (0 < args.alpha < 1):
        return f"--alpha must be in (0, 1), got {args.alpha}"
    if args.precision < 0:
        return "--precision must be non-negative"

    if args.test == "gof":
        if args.observed is None or args.expected is None:
            return "--test gof requires both --observed and --expected"
        if args.table is not None:
            return "--table is not used with --test gof"
        return None

    # independence
    if args.table is None or len(args.table) < 2:
        return "--test independence requires at least two --table rows"
    if args.observed is not None or args.expected is not None:
        return "--observed/--expected are not used with --test independence"
    return None


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def _fmt(value: float, precision: int) -> str:
    """Format a float to the requested number of decimal places."""
    return f"{value:.{precision}f}"


def _decision(p_value: float, alpha: float) -> str:
    """Return reject / fail-to-reject decision string."""
    if p_value < alpha:
        return f"Reject H₀  (p < α = {alpha})"
    return f"Fail to reject H₀  (p ≥ α = {alpha})"


def format_gof(result: GofResult, precision: int) -> str:
    """Format a goodness-of-fit result for display.

    Args:
        result: Computed :class:`GofResult`.
        precision: Decimal places for all floating-point output.

    Returns:
        Multi-line string ready to print.
    """
    f = lambda v: _fmt(v, precision)  # noqa: E731
    lines = [
        "Chi-square goodness-of-fit test",
        "H₀: observed frequencies match the expected distribution",
        "",
        f"  {'category':>10}  {'observed':>10}  {'expected':>10}  {'χ² contrib':>12}",
    ]
    for i, (o, e, c) in enumerate(
        zip(result.observed, result.expected, result.contributions), start=1
    ):
        lines.append(f"  {i:>10}  {f(o):>10}  {f(e):>10}  {f(c):>12}")
    lines += [
        "",
        f"  χ² statistic:  {f(result.statistic)}",
        f"  df:            {result.df}",
        f"  p-value:       {f(result.p_value)}",
        "",
        f"  {_decision(result.p_value, result.alpha)}",
    ]
    return "\n".join(lines)


def format_independence(result: IndependenceResult, precision: int) -> str:
    """Format an independence test result for display.

    Args:
        result: Computed :class:`IndependenceResult`.
        precision: Decimal places for all floating-point output.

    Returns:
        Multi-line string ready to print.
    """
    f = lambda v: _fmt(v, precision)  # noqa: E731
    lines = [
        "Chi-square test of independence",
        "H₀: the row and column variables are independent",
        "",
        "  Observed:",
    ]
    for row in result.table:
        lines.append("    " + "  ".join(f"{f(v):>10}" for v in row))
    lines.append("")
    lines.append("  Expected:")
    for row in result.expected:
        lines.append("    " + "  ".join(f"{f(v):>10}" for v in row))
    lines += [
        "",
        f"  χ² statistic:  {f(result.statistic)}",
        f"  df:            {result.df}",
        f"  p-value:       {f(result.p_value)}",
        "",
        f"  {_decision(result.p_value, result.alpha)}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Run the chi-square test CLI.

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
        if args.test == "gof":
            observed = parse_number_list(args.observed)
            expected = parse_number_list(args.expected)
            result_gof = chisq_gof(observed, expected, args.alpha)
            print(format_gof(result_gof, args.precision))
        else:
            table = [parse_number_list(row) for row in args.table]
            result_ind = chisq_independence(table, args.alpha)
            print(format_independence(result_ind, args.precision))
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
