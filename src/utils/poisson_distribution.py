#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import sys
from typing import Optional

"""Command-line utility for Poisson distribution probabilities.

Computes PMF, CDF, and survival probabilities for a Poisson(λ) distribution,
finds the minimum event count that reaches a target cumulative probability,
and generates full probability tables over a range of event counts.

The Poisson distribution models the number of rare, independent events
occurring in a fixed interval when the average rate λ is known — server
errors per hour, calls per minute, defects per batch, and so on.

Usage examples:
  poisson -l 3.0 -k 7
  poisson -l 3.0 --target-prob 0.95
  poisson -l 3.0 --range 0 15
  poisson -l 0.5 -k 2 --target 5 --min-prob 0.01
  poisson -l 3.0 --range 0 20 --format json
"""


# ---------------------------------------------------------------------------
# Core probability functions
# ---------------------------------------------------------------------------

def poisson_pmf(k: int, lam: float) -> float:
    """P(X = k) for Poisson(λ).

    Uses log-space arithmetic (math.lgamma) for numerical stability at large k.
    """
    if k < 0:
        return 0.0
    if lam == 0.0:
        return 1.0 if k == 0 else 0.0
    return math.exp(k * math.log(lam) - lam - math.lgamma(k + 1))


def poisson_cdf_le(k: int, lam: float) -> float:
    """P(X ≤ k) for Poisson(λ)."""
    if k < 0:
        return 0.0
    return min(sum(poisson_pmf(i, lam) for i in range(k + 1)), 1.0)


def poisson_cdf_ge(k: int, lam: float) -> float:
    """P(X ≥ k) for Poisson(λ)."""
    if k <= 0:
        return 1.0
    return max(1.0 - poisson_cdf_le(k - 1, lam), 0.0)


def min_k_for_prob(target_prob: float, lam: float, max_k: int = 100_000) -> Optional[int]:
    """Return the smallest k such that P(X ≤ k) >= target_prob (inverse CDF / quantile)."""
    if target_prob <= 0.0:
        return 0
    cumulative = 0.0
    for k in range(max_k + 1):
        cumulative += poisson_pmf(k, lam)
        if cumulative >= target_prob:
            return k
    return None  # not reached within max_k


def prob_table(lam: float, min_k: int, max_k: int) -> list[dict]:
    """Return rows of {k, pmf, cdf_le, cdf_ge} for k in [min_k, max_k]."""
    rows = []
    # Accumulate CDF from 0 up to min_k - 1
    cumulative = sum(poisson_pmf(i, lam) for i in range(min_k))
    for k in range(min_k, max_k + 1):
        pmf = poisson_pmf(k, lam)
        cdf_ge = max(1.0 - cumulative, 0.0)   # P(X >= k) = 1 - P(X <= k-1)
        cumulative = min(cumulative + pmf, 1.0)
        rows.append({
            "k": k,
            "pmf": pmf,
            "cdf_le": cumulative,
            "cdf_ge": cdf_ge,
        })
    return rows


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def _fmt_prob(x: float, precision: int) -> str:
    pct = x * 100.0
    fmt = f"{{:.{precision}f}}"
    return f"{fmt.format(x)} ({fmt.format(pct)}%)"


def format_single(
    k: int,
    lam: float,
    pmf: float,
    cdf_le: float,
    cdf_ge: float,
    target: Optional[int],
    min_prob: Optional[float],
    precision: int,
) -> str:
    lines = [
        f"Rate (λ):                {lam:.{precision}f}",
        f"Events (k):              {k:,}",
        f"E[X] = Var[X]:           {lam:.{precision}f}",
        f"P(X = {k}):              {_fmt_prob(pmf, precision)}",
        f"P(X ≤ {k}):              {_fmt_prob(cdf_le, precision)}",
        f"P(X ≥ {k}):              {_fmt_prob(cdf_ge, precision)}",
    ]
    if target is not None:
        pt = poisson_cdf_ge(target, lam)
        lines.append(f"P(X ≥ {target}):              {_fmt_prob(pt, precision)}")
        if min_prob is not None:
            meets = pt >= min_prob
            lines.append(f"Meets minimum {_fmt_prob(min_prob, precision)}: {meets}")
    return "\n".join(lines)


def format_table_output(rows: list[dict], precision: int) -> str:
    headers = ["k", "P(X=k)", "P(X≤k)", "P(X≥k)"]
    col_data = [
        [
            str(r["k"]),
            _fmt_prob(r["pmf"], precision),
            _fmt_prob(r["cdf_le"], precision),
            _fmt_prob(r["cdf_ge"], precision),
        ]
        for r in rows
    ]
    col_widths = [
        max(len(h), max((len(c[i]) for c in col_data), default=0))
        for i, h in enumerate(headers)
    ]
    sep = "  "
    header_line = sep.join(h.rjust(col_widths[i]) for i, h in enumerate(headers))
    divider = sep.join("-" * w for w in col_widths)
    lines = [header_line, divider]
    for cols in col_data:
        lines.append(sep.join(c.rjust(col_widths[i]) for i, c in enumerate(cols)))
    return "\n".join(lines)


def format_json_output(rows: list[dict]) -> str:
    def _round(d: dict) -> dict:
        return {k: round(v, 8) if isinstance(v, float) else v for k, v in d.items()}
    return json.dumps([_round(r) for r in rows], indent=2)


def format_csv_output(rows: list[dict]) -> str:
    if not rows:
        return ""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Poisson distribution probability calculator.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  poisson -l 3.0 -k 7
  poisson -l 3.0 -t 0.95
  poisson -l 3.0 -r 0 15
  poisson -l 0.5 -k 2 --target 5 --min-prob 0.01
  poisson -l 3.0 -r 0 20 -f json
""",
    )

    parser.add_argument(
        "--rate", "-l", type=float, required=True, metavar="λ",
        help="average event rate (λ, lambda); must be > 0",
    )

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--events", "-k", type=int, metavar="K",
        help="compute PMF and CDF for exactly this number of events",
    )
    mode.add_argument(
        "--target-prob", "-t", type=float, metavar="PROB",
        help="find the minimum k such that P(X ≤ k) >= PROB (inverse CDF)",
    )
    mode.add_argument(
        "--range", "-r", nargs=2, type=int, metavar=("MIN_K", "MAX_K"),
        help="print a probability table for event counts MIN_K through MAX_K",
    )

    parser.add_argument(
        "--target", type=int, default=None, metavar="T",
        help="with -k/--events: also print P(X ≥ T) for this target count",
    )
    parser.add_argument(
        "--min-prob", type=float, default=None, metavar="P",
        help="with --target: report whether P(X ≥ T) meets this threshold",
    )
    parser.add_argument(
        "--format", "-f", choices=["table", "json", "csv"], default="table",
        help="output format (default: table; applies only with -r/--range)",
    )
    parser.add_argument(
        "--precision", "-P", type=int, default=6,
        help="decimal places for printed probabilities (default: 6)",
    )

    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(args: argparse.Namespace) -> Optional[str]:
    if args.rate <= 0:
        return "-l/--rate must be greater than 0"

    if args.events is None and args.target_prob is None and args.range is None:
        return "one of -k/--events, -t/--target-prob, or -r/--range is required"

    if args.events is not None and args.events < 0:
        return "-k/--events must be non-negative"

    if args.target_prob is not None and not (0.0 < args.target_prob < 1.0):
        return "-t/--target-prob must be strictly between 0 and 1"

    if args.range is not None:
        min_k, max_k = args.range
        if min_k < 0:
            return "-r/--range MIN_K must be non-negative"
        if max_k < min_k:
            return "-r/--range MAX_K must be >= MIN_K"
        if max_k - min_k > 100_000:
            return "-r/--range span must not exceed 100,000 rows"

    if args.target is not None and args.events is None:
        return "--target can only be used with -k/--events"

    if args.min_prob is not None and args.target is None:
        return "--min-prob can only be used with --target"

    if args.min_prob is not None and not (0.0 <= args.min_prob <= 1.0):
        return "--min-prob must be between 0 and 1"

    return None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    error = validate(args)
    if error:
        print(f"Error: {error}", file=sys.stderr)
        return 2

    lam = args.rate
    precision = args.precision

    # --- single event count ---
    if args.events is not None:
        k = args.events
        pmf = poisson_pmf(k, lam)
        cdf_le = poisson_cdf_le(k, lam)
        cdf_ge = poisson_cdf_ge(k, lam)
        print(format_single(k, lam, pmf, cdf_le, cdf_ge, args.target, args.min_prob, precision))
        return 0

    # --- find minimum k for target probability ---
    if args.target_prob is not None:
        target = args.target_prob
        k = min_k_for_prob(target, lam)
        if k is None:
            print(
                f"No event count within search range reaches {target * 100:.{precision}f}%",
                file=sys.stderr,
            )
            return 1
        actual = poisson_cdf_le(k, lam)
        print(f"Rate (λ):                {lam:.{precision}f}")
        print(f"Target P(X ≤ k):         {_fmt_prob(target, precision)}")
        print(f"Minimum k:               {k:,}")
        print(f"Actual P(X ≤ {k}):       {_fmt_prob(actual, precision)}")
        return 0

    # --- range table ---
    min_k, max_k = args.range
    rows = prob_table(lam, min_k, max_k)

    if args.format == "json":
        print(format_json_output(rows))
    elif args.format == "csv":
        print(format_csv_output(rows))
    else:
        print(f"Rate (λ): {lam:.{precision}f}  |  Event range: {min_k}–{max_k}\n")
        print(format_table_output(rows, precision))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
