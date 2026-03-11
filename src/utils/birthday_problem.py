#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import sys
from typing import Optional

"""Command-line utility for birthday-problem collision probability.

Computes the probability that at least two items in a group of `n` share the
same value when drawn from a pool of `d` equally-likely possibilities — or
from a non-uniform pool described by relative frequency weights.

The "birthday problem" generalises beyond dates: it applies to any scenario
where you want to know the chance of a duplicate identifier, record, or event
in a finite population (citizen IDs, case numbers, ZIP codes, audit samples…).

Usage examples:
  birthday --pool-size 365 --group-size 23
  birthday --pool-size 365 --target-prob 0.50
  birthday --pool-size 365 --range 1 60 --format table
  birthday --pool-size 1000000 --group-size 1180 --format json
  birthday --group-size 30 --weights 0.10,0.15,0.20,0.30,0.25
"""


# ---------------------------------------------------------------------------
# Core probability functions
# ---------------------------------------------------------------------------


def collision_prob_uniform(n: int, d: float) -> float:
    """P(at least one duplicate) for n items drawn from a uniform pool of size d.

    Uses log-space arithmetic for numerical stability with large d or n.
    Returns 1.0 immediately when n > d (pigeonhole guarantee).
    """
    if n <= 1:
        return 0.0
    if n > d:
        return 1.0
    # log P(no collision) = sum_{i=1}^{n-1} log(1 - i/d)
    log_no_collision = sum(math.log1p(-i / d) for i in range(1, n))
    return -math.expm1(
        log_no_collision
    )  # 1 - exp(log_no_collision), more precise near 0


def collision_prob_nonuniform(n: int, weights: list[float]) -> float:
    """P(at least one duplicate) for n items drawn from a non-uniform pool.

    Uses the Poisson approximation:
        P(at least one collision) ≈ 1 - exp(-C(n,2) * sum(p_i^2))

    where p_i are the normalised category probabilities and sum(p_i^2) is the
    pairwise collision probability (Simpson's diversity complement).

    This is accurate when n is small relative to the effective pool size.
    """
    if n <= 1:
        return 0.0
    total = sum(weights)
    if total <= 0:
        raise ValueError("weights must contain at least one positive value")
    probs = [w / total for w in weights]
    pairwise_collision = sum(p * p for p in probs)
    pairs = n * (n - 1) / 2
    return -math.expm1(-pairs * pairwise_collision)


def effective_pool_size(weights: list[float]) -> float:
    """Effective pool size (inverse Simpson index) for a non-uniform distribution.

    Equals d for a uniform pool of size d; smaller than d whenever the
    distribution is skewed, reflecting reduced collision resistance.
    """
    total = sum(weights)
    if total <= 0:
        raise ValueError("weights must contain at least one positive value")
    probs = [w / total for w in weights]
    return 1.0 / sum(p * p for p in probs)


def min_group_for_prob(
    target_prob: float, d: float, max_n: int = 1_000_000
) -> Optional[int]:
    """Return the smallest n such that collision_prob_uniform(n, d) >= target_prob."""
    if target_prob <= 0.0:
        return 1
    if target_prob >= 1.0:
        return int(d) + 1
    for n in range(2, min(max_n, int(d)) + 2):
        if collision_prob_uniform(n, d) >= target_prob:
            return n
    return None  # not reached within max_n


def expected_duplicate_pairs(n: int, d: float) -> float:
    """Expected number of duplicate pairs among n items in a uniform pool of d.

    E[matching pairs] = C(n, 2) / d
    """
    return n * (n - 1) / 2 / d


def prob_table(d: float, min_n: int, max_n: int) -> list[dict]:
    """Return rows of {n, probability, expected_pairs} for n in [min_n, max_n]."""
    rows = []
    for n in range(min_n, max_n + 1):
        prob = collision_prob_uniform(n, d)
        rows.append(
            {
                "n": n,
                "probability": prob,
                "expected_duplicate_pairs": expected_duplicate_pairs(n, d),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def _fmt_prob(x: float, precision: int) -> str:
    pct = x * 100.0
    fmt = f"{{:.{precision}f}}"
    return f"{fmt.format(x)} ({fmt.format(pct)}%)"


def format_single(
    n: int,
    d: int,
    prob: float,
    exp_pairs: Optional[float],
    weights: Optional[list[float]],
    precision: int,
) -> str:
    lines = []
    if weights:
        eff = effective_pool_size(weights)
        lines.append(f"Pool type:              non-uniform ({len(weights)} categories)")
        lines.append(f"Effective pool size:    {eff:.{precision}f}")
    else:
        lines.append(f"Pool size (d):          {d:,}")
    lines.append(f"Group size (n):         {n:,}")
    lines.append(f"P(at least 1 duplicate):  {_fmt_prob(prob, precision)}")
    if not weights:
        lines.append(f"Expected duplicate pairs: {exp_pairs:.{precision}f}")
    return "\n".join(lines)


def format_table_output(rows: list[dict], precision: int) -> str:
    headers = ["n", "P(duplicate)", "Expected pairs"]
    col_data = [
        [
            str(r["n"]),
            _fmt_prob(r["probability"], precision),
            f"{r['expected_duplicate_pairs']:.{precision}f}",
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


def _parse_weights(value: str) -> list[float]:
    try:
        weights = [float(w.strip()) for w in value.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"--weights must be comma-separated numbers, got: {value!r}"
        )
    if any(w < 0 for w in weights):
        raise argparse.ArgumentTypeError("all weights must be non-negative")
    if not any(w > 0 for w in weights):
        raise argparse.ArgumentTypeError("at least one weight must be positive")
    return weights


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Birthday-problem collision probability calculator.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  birthday --pool-size 365 --group-size 23
  birthday --pool-size 365 --target-prob 0.50
  birthday --pool-size 365 --range 1 60
  birthday --pool-size 1000000 --group-size 1180
  birthday --group-size 30 --weights 0.10,0.15,0.20,0.30,0.25
""",
    )

    pool = parser.add_mutually_exclusive_group()
    pool.add_argument(
        "--pool-size",
        "-p",
        type=float,
        default=365.25,
        metavar="D",
        help="number of equally-likely outcomes in the pool (default: 365.25)",
    )
    pool.add_argument(
        "--weights",
        type=_parse_weights,
        metavar="W1,W2,...",
        help="comma-separated relative frequencies for a non-uniform pool",
    )

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--group-size",
        "-n",
        type=int,
        metavar="N",
        help="compute collision probability for exactly this group size",
    )
    mode.add_argument(
        "--target-prob",
        type=float,
        metavar="PROB",
        help="find the minimum group size that reaches this collision probability",
    )
    mode.add_argument(
        "--range",
        nargs=2,
        type=int,
        metavar=("MIN_N", "MAX_N"),
        help="print a probability table for group sizes MIN_N through MAX_N",
    )

    parser.add_argument(
        "--format",
        choices=["table", "json", "csv"],
        default="table",
        help="output format (default: table; applies only with --range)",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=6,
        help="decimal places for printed probabilities (default: 6)",
    )

    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate(args: argparse.Namespace) -> Optional[str]:
    if args.pool_size is not None and args.weights is None and args.pool_size < 1:
        return "--pool-size must be at least 1"

    if args.weights is not None and args.group_size is None:
        # weights mode only supports --group-size currently
        if args.target_prob is not None:
            return "--target-prob is not supported with --weights (use --pool-size)"
        if args.range is not None:
            return "--range is not supported with --weights (use --pool-size)"

    if args.group_size is not None and args.group_size < 1:
        return "--group-size must be at least 1"

    if args.target_prob is not None and not (0.0 < args.target_prob < 1.0):
        return "--target-prob must be strictly between 0 and 1"

    if args.range is not None:
        min_n, max_n = args.range
        if min_n < 1:
            return "range MIN_N must be at least 1"
        if max_n < min_n:
            return "range MAX_N must be >= MIN_N"
        if max_n - min_n > 100_000:
            return "range span must not exceed 100,000 rows"

    if args.group_size is None and args.target_prob is None and args.range is None:
        return "one of --group-size, --target-prob, or --range is required"

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

    d = args.pool_size
    weights = args.weights
    precision = args.precision

    # --- single group size ---
    if args.group_size is not None:
        n = args.group_size
        if weights:
            prob = collision_prob_nonuniform(n, weights)
            exp_pairs = None
        else:
            prob = collision_prob_uniform(n, d)
            exp_pairs = expected_duplicate_pairs(n, d)
        print(format_single(n, d, prob, exp_pairs, weights, precision))
        return 0

    # --- find minimum group for target probability ---
    if args.target_prob is not None:
        target = args.target_prob
        n = min_group_for_prob(target, d)
        if n is None:
            print(
                f"No group size up to {int(d) + 1:,} reaches {target * 100:.{precision}f}%",
                file=sys.stderr,
            )
            return 1
        prob = collision_prob_uniform(n, d)
        exp_pairs = expected_duplicate_pairs(n, d)
        print(f"Pool size (d):            {d:,}")
        print(f"Target probability:       {_fmt_prob(target, precision)}")
        print(f"Minimum group size (n):   {n:,}")
        print(f"Actual probability:       {_fmt_prob(prob, precision)}")
        print(f"Expected duplicate pairs: {exp_pairs:.{precision}f}")
        return 0

    # --- range table ---
    min_n, max_n = args.range
    rows = prob_table(d, min_n, max_n)

    if args.format == "json":
        print(format_json_output(rows))
    elif args.format == "csv":
        print(format_csv_output(rows))
    else:
        print(f"Pool size (d): {d:,}  |  Group range: {min_n}-{max_n}\n")
        print(format_table_output(rows, precision))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
