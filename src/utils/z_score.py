#!/usr/bin/env python3
"""Command-line utility for computing z-scores.

Computes standardized scores using z = (x - mean) / standard deviation.
The tool supports either a single value with an explicit mean and standard
deviation, or a list of values where the mean and standard deviation are
computed from the data.

Usage examples:
  zscore --value 85 --mean 70 --std 10
  zscore --values 2,4,4,4,5,5,7,9
  zscore --values 2,4,4,4,5,5,7,9 --sample
"""

from __future__ import annotations

import argparse
import math
import sys
from typing import Sequence

# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def mean(values: Sequence[float]) -> float:
    """Compute the arithmetic mean."""
    if not values:
        raise ValueError("Cannot compute mean of empty list")
    return sum(values) / len(values)


def variance(values: Sequence[float], sample: bool = False) -> float:
    """Compute population or sample variance."""
    n = len(values)
    if n == 0:
        raise ValueError("Cannot compute variance of empty list")
    if sample and n < 2:
        raise ValueError("Need at least 2 values for sample variance")

    mu = mean(values)
    denominator = n - 1 if sample else n
    return sum((value - mu) ** 2 for value in values) / denominator


def std_dev(values: Sequence[float], sample: bool = False) -> float:
    """Compute population or sample standard deviation."""
    return math.sqrt(variance(values, sample))


def z_score(value: float, mu: float, sigma: float) -> float:
    """Compute the z-score for one value."""
    if sigma <= 0:
        raise ValueError("standard deviation must be greater than 0")
    return (value - mu) / sigma


def z_scores(values: Sequence[float], sample: bool = False) -> list[float]:
    """Compute z-scores for a sequence of values."""
    sigma = std_dev(values, sample)
    if sigma == 0:
        raise ValueError("standard deviation must be greater than 0")
    mu = mean(values)
    return [z_score(value, mu, sigma) for value in values]


# ---------------------------------------------------------------------------
# Parsing and validation
# ---------------------------------------------------------------------------


def parse_number_list(raw: str) -> list[float]:
    """Parse a comma-separated list of finite numbers."""
    values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("at least one value is required")
    if any(not math.isfinite(value) for value in values):
        raise ValueError("all values must be finite")
    return values


def validate(args: argparse.Namespace) -> str | None:
    if args.precision < 0:
        return "--precision must be non-negative"

    if args.value is not None:
        if args.mean is None or args.std is None:
            return "--mean and --std are required when --value is provided"
        if not all(math.isfinite(v) for v in (args.value, args.mean, args.std)):
            return "--value, --mean, and --std must be finite"
        if args.std <= 0:
            return "--std must be greater than 0"
        return None

    if args.values is None:
        return "one of --value or --values is required"

    return None


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Z-score calculator.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  zscore --value 85 --mean 70 --std 10
  zscore --values 2,4,4,4,5,5,7,9
  zscore --values 2,4,4,4,5,5,7,9 --sample
""",
    )

    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--value",
        "-x",
        type=float,
        metavar="X",
        help="single value to standardize",
    )
    source.add_argument(
        "--values",
        "-v",
        type=str,
        metavar="X1,X2,...",
        help="comma-separated values to standardize as a dataset",
    )

    parser.add_argument(
        "--mean",
        "-m",
        type=float,
        metavar="MEAN",
        help="mean to use with --value",
    )
    parser.add_argument(
        "--std",
        "-s",
        type=float,
        metavar="STD",
        help="standard deviation to use with --value",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="use sample standard deviation for --values",
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


def _fmt(value: float, precision: int) -> str:
    return f"{value:.{precision}f}"


def format_single(
    value: float, mu: float, sigma: float, z: float, precision: int
) -> str:
    lines = [
        f"Value (x):         {_fmt(value, precision)}",
        f"Mean:              {_fmt(mu, precision)}",
        f"Std dev:           {_fmt(sigma, precision)}",
        f"Z-score:           {_fmt(z, precision)}",
    ]
    return "\n".join(lines)


def format_dataset(
    values: Sequence[float],
    mu: float,
    sigma: float,
    scores: Sequence[float],
    sample: bool,
    precision: int,
) -> str:
    std_label = "sample" if sample else "population"
    lines = [
        f"Values (n):        {len(values)}",
        f"Mean:              {_fmt(mu, precision)}",
        f"Std dev ({std_label}): {_fmt(sigma, precision)}",
        "Z-scores:",
    ]
    lines.extend(
        f"  {_fmt(value, precision)} -> {_fmt(score, precision)}"
        for value, score in zip(values, scores)
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    error = validate(args)
    if error:
        print(f"Error: {error}", file=sys.stderr)
        return 2

    precision = args.precision

    if args.value is not None:
        try:
            z = z_score(args.value, args.mean, args.std)
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 2
        print(format_single(args.value, args.mean, args.std, z, precision))
        return 0

    try:
        values = parse_number_list(args.values)
        scores = z_scores(values, args.sample)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    mu = mean(values)
    sigma = std_dev(values, args.sample)
    print(format_dataset(values, mu, sigma, scores, args.sample, precision))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
