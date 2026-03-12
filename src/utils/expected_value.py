#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from typing import Optional

"""Command-line utility for expected value and discrete distribution statistics.

Accepts a discrete probability distribution as paired outcome/probability lists
or as a CSV/JSON file, and computes E[X], Var(X), SD(X), Shannon entropy, and
optionally the moment generating function (MGF) at a given t.

Usage examples:
  expected --outcomes 0,1,5,10 --probs 0.50,0.25,0.15,0.10
  expected --outcomes 1,2,3,4,5,6 --probs 0.1,0.2,0.3,0.2,0.1,0.1
  expected --file payouts.csv
  expected --outcomes 0,1 --probs 0.3,0.7 --mgf 0.5
"""


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def expected_value(outcomes: list[float], probs: list[float]) -> float:
    """E[X] = Σ xᵢ · pᵢ"""
    return sum(x * p for x, p in zip(outcomes, probs))


def variance(outcomes: list[float], probs: list[float]) -> float:
    """Var(X) = E[X²] - (E[X])²"""
    mu = expected_value(outcomes, probs)
    return sum((x - mu) ** 2 * p for x, p in zip(outcomes, probs))


def std_dev(outcomes: list[float], probs: list[float]) -> float:
    """SD(X) = sqrt(Var(X))"""
    return math.sqrt(variance(outcomes, probs))


def entropy(probs: list[float]) -> float:
    """Shannon entropy H(X) = -Σ pᵢ log₂(pᵢ)  (bits; zero terms skipped)."""
    return -sum(p * math.log2(p) for p in probs if p > 0.0)


def mgf(outcomes: list[float], probs: list[float], t: float) -> float:
    """Moment generating function M_X(t) = E[e^{tX}] = Σ e^{t·xᵢ} · pᵢ"""
    return sum(math.exp(t * x) * p for x, p in zip(outcomes, probs))


# ---------------------------------------------------------------------------
# File loading
# ---------------------------------------------------------------------------


def _load_csv(path: str) -> tuple[list[float], list[float]]:
    """Load outcomes and probabilities from a two-column CSV file.

    Expects a header row with columns named 'outcome' and 'prob' (or 'probability'),
    or a headerless file with outcomes in the first column and probs in the second.
    """
    outcomes: list[float] = []
    probs: list[float] = []
    with open(path, newline="") as f:
        sample = f.read(1024)
        f.seek(0)
        has_header = csv.Sniffer().has_header(sample)
        reader = csv.reader(f)
        if has_header:
            headers = [h.strip().lower() for h in next(reader)]
            try:
                oi = next(
                    i
                    for i, h in enumerate(headers)
                    if h in ("outcome", "outcomes", "value", "x")
                )
                pi = next(
                    i
                    for i, h in enumerate(headers)
                    if h in ("prob", "probability", "p", "weight")
                )
            except StopIteration:
                oi, pi = 0, 1
        else:
            oi, pi = 0, 1
        for row in reader:
            if not row:
                continue
            outcomes.append(float(row[oi]))
            probs.append(float(row[pi]))
    return outcomes, probs


def _load_json(path: str) -> tuple[list[float], list[float]]:
    """Load outcomes and probabilities from a JSON file.

    Accepts either:
      - A list of {"outcome": x, "prob": p} objects
      - An object {"outcomes": [...], "probs": [...]}
    """
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        outcomes = [float(item["outcome"]) for item in data]
        probs = [
            float(item.get("prob", item.get("probability", item.get("p"))))
            for item in data
        ]
    else:
        outcomes = [float(v) for v in data["outcomes"]]
        probs = [float(v) for v in data.get("probs", data.get("probabilities", []))]
    return outcomes, probs


def load_file(path: str) -> tuple[list[float], list[float]]:
    """Dispatch to CSV or JSON loader based on file extension."""
    lower = path.lower()
    if lower.endswith(".json"):
        return _load_json(path)
    return _load_csv(path)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_distribution(outcomes: list[float], probs: list[float]) -> Optional[str]:
    if len(outcomes) == 0:
        return "at least one outcome/probability pair is required"
    if len(outcomes) != len(probs):
        return f"number of outcomes ({len(outcomes)}) must match number of probabilities ({len(probs)})"
    if any(p < 0.0 for p in probs):
        return "all probabilities must be non-negative"
    total = sum(probs)
    if abs(total - 1.0) > 1e-6:
        return f"probabilities must sum to 1.0 (got {total:.8f})"
    return None


def validate(args: argparse.Namespace) -> Optional[str]:
    if args.file is None and args.outcomes is None:
        return "one of --outcomes/--file is required"
    if args.file is not None and args.outcomes is not None:
        return "--outcomes and --file are mutually exclusive"
    if args.outcomes is not None and args.probs is None:
        return "--probs is required when --outcomes is provided"
    if args.mgf_t is not None and math.isinf(args.mgf_t):
        return "--mgf value must be finite"
    return None


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Expected value and discrete distribution statistics calculator.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  expected --outcomes 0,1,5,10 --probs 0.50,0.25,0.15,0.10
  expected --outcomes 1,2,3,4,5,6 --probs 0.1,0.2,0.3,0.2,0.1,0.1
  expected --file payouts.csv
  expected --outcomes 0,1 --probs 0.3,0.7 --mgf 0.5
""",
    )

    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--outcomes",
        "-o",
        type=str,
        metavar="X1,X2,...",
        help="comma-separated outcome values",
    )
    source.add_argument(
        "--file",
        "-f",
        type=str,
        metavar="PATH",
        help="CSV or JSON file with outcomes and probabilities",
    )
    parser.add_argument(
        "--probs",
        "-p",
        type=str,
        metavar="P1,P2,...",
        help="comma-separated probabilities (required with --outcomes)",
    )
    parser.add_argument(
        "--mgf",
        dest="mgf_t",
        type=float,
        default=None,
        metavar="T",
        help="also compute the moment generating function M_X(t) at this t",
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
# Output formatting
# ---------------------------------------------------------------------------


def _fmt(x: float, precision: int) -> str:
    return f"{x:.{precision}f}"


def format_output(
    outcomes: list[float],
    probs: list[float],
    ev: float,
    var: float,
    sd: float,
    ent: float,
    mgf_val: Optional[float],
    mgf_t: Optional[float],
    precision: int,
) -> str:
    n = len(outcomes)
    lines = [
        f"Outcomes (n):      {n}",
        f"E[X]:              {_fmt(ev, precision)}",
        f"Var(X):            {_fmt(var, precision)}",
        f"SD(X):             {_fmt(sd, precision)}",
        f"Entropy H(X):      {_fmt(ent, precision)} bits",
    ]
    if mgf_val is not None and mgf_t is not None:
        lines.append(f"MGF M_X({_fmt(mgf_t, precision)}):   {_fmt(mgf_val, precision)}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    error = validate(args)
    if error:
        print(f"Error: {error}", file=sys.stderr)
        return 2

    # Load distribution
    if args.file is not None:
        try:
            outcomes, probs = load_file(args.file)
        except (OSError, KeyError, ValueError, StopIteration) as exc:
            print(f"Error loading file: {exc}", file=sys.stderr)
            return 2
    else:
        try:
            outcomes = [float(v.strip()) for v in args.outcomes.split(",")]
            probs = [float(v.strip()) for v in args.probs.split(",")]
        except ValueError as exc:
            print(f"Error parsing values: {exc}", file=sys.stderr)
            return 2

    dist_error = validate_distribution(outcomes, probs)
    if dist_error:
        print(f"Error: {dist_error}", file=sys.stderr)
        return 2

    precision = args.precision
    ev = expected_value(outcomes, probs)
    var = variance(outcomes, probs)
    sd = std_dev(outcomes, probs)
    ent = entropy(probs)

    mgf_val = None
    if args.mgf_t is not None:
        mgf_val = mgf(outcomes, probs, args.mgf_t)

    print(
        format_output(outcomes, probs, ev, var, sd, ent, mgf_val, args.mgf_t, precision)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
