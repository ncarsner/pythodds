#!/usr/bin/env python3
"""Command-line utility for information-theoretic measures.

Computes Shannon entropy, KL divergence, cross-entropy, mutual information,
and conditional entropy.  Pure Python via ``math.log`` — no external
dependencies.  Results are reported in bits (base 2), nats (base e), or
hartleys (base 10).

Input vectors are normalized to sum to 1, so raw counts are accepted
alongside probabilities.

Usage examples:
  entropy --probs 0.167,0.167,0.167,0.167,0.167,0.167
  entropy --measure kl --probs-p 0.7,0.3 --probs-q 0.5,0.5
  entropy --measure mi --joint 0.25,0.25 --joint 0.25,0.25
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence
from typing import Any

# Display names for the supported logarithm bases.
UNIT_NAMES: dict[str, str] = {"2": "bits", "e": "nats", "10": "hartleys"}

# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def normalize(values: Sequence[float]) -> list[float]:
    """Scale non-negative weights into a probability distribution.

    Args:
        values: Non-negative weights or probabilities; must be non-empty and
            sum to a positive total.

    Returns:
        The values divided by their sum, so the result sums to 1.

    Raises:
        ValueError: If ``values`` is empty, contains a negative entry, or sums
            to zero.
    """
    if not values:
        raise ValueError("distribution must have at least one outcome")
    for value in values:
        if value < 0:
            raise ValueError(f"probabilities must be >= 0, got {value}")
    total = math.fsum(values)
    if total <= 0:
        raise ValueError("probabilities must sum to a positive total")
    return [value / total for value in values]


def shannon_entropy(probs: Sequence[float], base: float = 2.0) -> float:
    """Shannon entropy H(X) of a discrete distribution.

    Args:
        probs: Non-negative weights or probabilities; normalized internally.
        base: Logarithm base; must be > 1.

    Returns:
        H(X) = -sum p_i * log_base(p_i), with the 0 * log 0 = 0 convention.

    Raises:
        ValueError: If ``probs`` is not a valid distribution or ``base`` <= 1.
    """
    _validate_base(base)
    p = normalize(probs)
    # Zero-probability outcomes contribute nothing (0 log 0 -> 0 by convention).
    return -math.fsum(pi * math.log(pi, base) for pi in p if pi > 0)


def max_entropy(outcomes: int, base: float = 2.0) -> float:
    """Entropy of the uniform distribution over ``outcomes`` outcomes.

    Args:
        outcomes: Number of outcomes; must be >= 1.
        base: Logarithm base; must be > 1.

    Returns:
        log_base(outcomes) — the largest entropy any distribution of this size
        can reach.

    Raises:
        ValueError: If ``outcomes`` < 1 or ``base`` <= 1.
    """
    _validate_base(base)
    if outcomes < 1:
        raise ValueError(f"outcomes must be >= 1, got {outcomes}")
    return math.log(outcomes, base)


def kl_divergence(p: Sequence[float], q: Sequence[float], base: float = 2.0) -> float:
    """Kullback-Leibler divergence D(P || Q).

    Args:
        p: Reference distribution, normalized internally.
        q: Comparison distribution, normalized internally; must be the same
            length as ``p``.
        base: Logarithm base; must be > 1.

    Returns:
        D(P||Q) = sum p_i * log_base(p_i / q_i) — the extra information cost of
        coding samples from P using a code built for Q.

    Raises:
        ValueError: If the vectors differ in length, either is not a valid
            distribution, ``base`` <= 1, or some ``q_i`` is 0 where ``p_i`` > 0
            (the divergence is then infinite).
    """
    _validate_base(base)
    _validate_same_length(p, q)
    pn, qn = normalize(p), normalize(q)
    total = 0.0
    for pi, qi in zip(pn, qn):
        if pi == 0:
            continue
        if qi == 0:
            raise ValueError(
                "KL divergence is infinite: q assigns zero probability to an "
                "outcome p gives positive probability"
            )
        total += pi * math.log(pi / qi, base)
    return total


def cross_entropy(p: Sequence[float], q: Sequence[float], base: float = 2.0) -> float:
    """Cross-entropy H(P, Q).

    Args:
        p: True distribution, normalized internally.
        q: Predicted distribution, normalized internally; must be the same
            length as ``p``.
        base: Logarithm base; must be > 1.

    Returns:
        H(P, Q) = -sum p_i * log_base(q_i) = H(P) + D(P||Q).

    Raises:
        ValueError: If the vectors differ in length, either is not a valid
            distribution, ``base`` <= 1, or some ``q_i`` is 0 where ``p_i`` > 0.
    """
    _validate_base(base)
    _validate_same_length(p, q)
    pn, qn = normalize(p), normalize(q)
    total = 0.0
    for pi, qi in zip(pn, qn):
        if pi == 0:
            continue
        if qi == 0:
            raise ValueError(
                "cross-entropy is infinite: q assigns zero probability to an "
                "outcome p gives positive probability"
            )
        total -= pi * math.log(qi, base)
    return total


def normalize_joint(joint: Sequence[Sequence[float]]) -> list[list[float]]:
    """Scale a joint weight table into a joint probability distribution.

    Args:
        joint: Rectangular table of non-negative weights, one row per X outcome
            and one column per Y outcome.

    Returns:
        The table divided by its grand total, so all cells sum to 1.

    Raises:
        ValueError: If the table is empty, ragged, has a negative cell, or sums
            to zero.
    """
    if not joint:
        raise ValueError("joint distribution must have at least one row")
    width = len(joint[0])
    if width == 0:
        raise ValueError("joint distribution rows must have at least one column")
    for row in joint:
        if len(row) != width:
            raise ValueError(
                f"joint rows must all have the same length, got {width} and {len(row)}"
            )
    flat = normalize([cell for row in joint for cell in row])
    return [flat[i * width : (i + 1) * width] for i in range(len(joint))]


def marginals(
    joint: Sequence[Sequence[float]],
) -> tuple[list[float], list[float]]:
    """Row and column marginal distributions of a joint table.

    Args:
        joint: Rectangular table of non-negative weights.

    Returns:
        Tuple of (P(X), P(Y)) — the row sums and the column sums of the
        normalized joint table.

    Raises:
        ValueError: If the joint table is invalid.
    """
    table = normalize_joint(joint)
    px = [math.fsum(row) for row in table]
    py = [math.fsum(row[j] for row in table) for j in range(len(table[0]))]
    return px, py


def joint_entropy(joint: Sequence[Sequence[float]], base: float = 2.0) -> float:
    """Joint entropy H(X, Y) of a joint distribution table.

    Args:
        joint: Rectangular table of non-negative weights.
        base: Logarithm base; must be > 1.

    Returns:
        H(X, Y), the entropy of the flattened joint distribution.

    Raises:
        ValueError: If the joint table is invalid or ``base`` <= 1.
    """
    table = normalize_joint(joint)
    return shannon_entropy([cell for row in table for cell in row], base)


def mutual_information(joint: Sequence[Sequence[float]], base: float = 2.0) -> float:
    """Mutual information I(X; Y) of a joint distribution table.

    Args:
        joint: Rectangular table of non-negative weights.
        base: Logarithm base; must be > 1.

    Returns:
        I(X; Y) = H(X) + H(Y) - H(X, Y) — the information the two variables
        share.  Zero exactly when X and Y are independent.

    Raises:
        ValueError: If the joint table is invalid or ``base`` <= 1.
    """
    px, py = marginals(joint)
    value = (
        shannon_entropy(px, base)
        + shannon_entropy(py, base)
        - joint_entropy(joint, base)
    )
    # Floating-point cancellation can push an independent table slightly below
    # zero; mutual information is non-negative by definition.
    return max(0.0, value)


def conditional_entropy(joint: Sequence[Sequence[float]], base: float = 2.0) -> float:
    """Conditional entropy H(Y | X) of a joint distribution table.

    Args:
        joint: Rectangular table of non-negative weights, rows indexing X.
        base: Logarithm base; must be > 1.

    Returns:
        H(Y | X) = H(X, Y) - H(X) — the uncertainty left in Y once X is known.

    Raises:
        ValueError: If the joint table is invalid or ``base`` <= 1.
    """
    px, _py = marginals(joint)
    return max(0.0, joint_entropy(joint, base) - shannon_entropy(px, base))


def _kl_or_none(p: Sequence[float], q: Sequence[float], base: float) -> float | None:
    """Return D(P||Q), or ``None`` when the divergence is infinite.

    Args:
        p: Reference distribution.
        q: Comparison distribution.
        base: Logarithm base.

    Returns:
        The divergence, or ``None`` if ``q`` puts zero mass where ``p`` does not.
    """
    try:
        return kl_divergence(p, q, base)
    except ValueError as exc:
        if "infinite" in str(exc):
            return None
        raise


def _validate_base(base: float) -> None:
    """Raise if the logarithm base is not usable for an entropy measure."""
    if base <= 1:
        raise ValueError(f"base must be > 1, got {base}")


def _validate_same_length(p: Sequence[float], q: Sequence[float]) -> None:
    """Raise if two distributions do not have matching support sizes."""
    if len(p) != len(q):
        raise ValueError(
            f"p and q must have the same length, got {len(p)} and {len(q)}"
        )


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_values(text: str) -> list[float]:
    """Parse a comma- or whitespace-separated list of numbers.

    Args:
        text: Raw string such as ``"0.7,0.3"`` or ``"3 1 1"``.

    Returns:
        List of parsed floats.

    Raises:
        ValueError: If ``text`` holds no values or an entry is not numeric.
    """
    tokens = [token for token in text.replace(",", " ").split() if token]
    if not tokens:
        raise ValueError("no numeric values found")
    try:
        return [float(token) for token in tokens]
    except ValueError:
        raise ValueError(f"could not parse '{text}' as a list of numbers") from None


def base_value(base: str) -> float:
    """Convert a ``--base`` choice into a numeric logarithm base.

    Args:
        base: One of ``"2"``, ``"e"``, or ``"10"``.

    Returns:
        The numeric base.

    Raises:
        ValueError: If ``base`` is not a supported choice.
    """
    if base == "e":
        return math.e
    if base in ("2", "10"):
        return float(base)
    raise ValueError(f"base must be one of 2, e, 10, got {base}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Build and return the argument parser namespace.

    Args:
        argv: Argument list (uses ``sys.argv`` when ``None``).

    Returns:
        Parsed :class:`argparse.Namespace`.
    """
    parser = argparse.ArgumentParser(
        description="Information entropy and divergence calculator.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  entropy --probs 0.167,0.167,0.167,0.167,0.167,0.167
  entropy --measure kl --probs-p 0.7,0.3 --probs-q 0.5,0.5
  entropy --measure cross --probs-p 1,0 --probs-q 0.9,0.1 --base e
  entropy --measure mi --joint 0.25,0.25 --joint 0.25,0.25
""",
    )
    parser.add_argument(
        "--probs",
        type=str,
        default=None,
        metavar="VALUES",
        help="distribution for the entropy measure (comma or space separated)",
    )
    parser.add_argument(
        "--measure",
        choices=["entropy", "kl", "cross", "mi", "conditional"],
        default="entropy",
        help="quantity to compute (default: entropy)",
    )
    parser.add_argument(
        "--base",
        choices=["2", "e", "10"],
        default="2",
        help="logarithm base: 2=bits, e=nats, 10=hartleys (default: 2)",
    )
    parser.add_argument(
        "--probs-p",
        type=str,
        default=None,
        metavar="VALUES",
        help="distribution P for the kl and cross measures",
    )
    parser.add_argument(
        "--probs-q",
        type=str,
        default=None,
        metavar="VALUES",
        help="distribution Q for the kl and cross measures",
    )
    parser.add_argument(
        "--joint",
        type=str,
        action="append",
        default=None,
        metavar="ROW",
        help="one row of the joint table for mi and conditional; repeat per row",
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
    if args.precision < 0:
        return "--precision must be non-negative"

    if args.measure == "entropy":
        if args.probs is None:
            return "--probs is required for --measure entropy"
    elif args.measure in ("kl", "cross"):
        if args.probs_p is None or args.probs_q is None:
            return f"--probs-p and --probs-q are required for --measure {args.measure}"
    elif args.joint is None:
        return f"--joint is required for --measure {args.measure}"
    return None


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def _fmt(value: float, precision: int) -> str:
    """Format a float to the requested number of decimal places."""
    return f"{value:.{precision}f}"


def build_result(args: argparse.Namespace) -> dict[str, Any]:
    """Compute the requested measure into a result mapping.

    Args:
        args: Validated argument namespace from :func:`parse_args`.

    Returns:
        Mapping with the measure name, unit, and every figure the report shows.

    Raises:
        ValueError: If an input vector or the joint table is invalid.
    """
    base = base_value(args.base)
    unit = UNIT_NAMES[args.base]
    result: dict[str, Any] = {
        "measure": args.measure,
        "base": args.base,
        "unit": unit,
    }

    if args.measure == "entropy":
        probs = normalize(parse_values(args.probs))
        maximum = max_entropy(len(probs), base)
        value = shannon_entropy(probs, base)
        result.update(
            {
                "outcomes": len(probs),
                "entropy": value,
                "max_entropy": maximum,
                # Efficiency is undefined for a single outcome (max entropy 0).
                "efficiency": value / maximum if maximum > 0 else 1.0,
            }
        )
    elif args.measure in ("kl", "cross"):
        p = normalize(parse_values(args.probs_p))
        q = normalize(parse_values(args.probs_q))
        result.update(
            {
                "entropy_p": shannon_entropy(p, base),
                "kl_divergence": kl_divergence(p, q, base),
                "cross_entropy": cross_entropy(p, q, base),
                # Reverse KL is supplementary, so an infinite value is reported
                # rather than failing the run the user actually asked for.
                "kl_reverse": _kl_or_none(q, p, base),
            }
        )
    else:
        joint = [parse_values(row) for row in args.joint]
        px, py = marginals(joint)
        result.update(
            {
                "entropy_x": shannon_entropy(px, base),
                "entropy_y": shannon_entropy(py, base),
                "joint_entropy": joint_entropy(joint, base),
                "mutual_information": mutual_information(joint, base),
                "conditional_entropy_y_given_x": conditional_entropy(joint, base),
                "conditional_entropy_x_given_y": conditional_entropy(
                    [list(col) for col in zip(*normalize_joint(joint))], base
                ),
            }
        )
    return result


def format_table(result: dict[str, Any], precision: int) -> str:
    """Format the result mapping as an aligned text report.

    Args:
        result: Mapping from :func:`build_result`.
        precision: Decimal places for all floating-point output.

    Returns:
        Multi-line string ready to print.
    """
    unit = str(result["unit"])
    measure = str(result["measure"])

    def line(label: str, key: str) -> str:
        """Render one labelled value row in the report's unit."""
        value = result[key]
        if value is None:
            return f"  {label:<26} {'infinite':>12}"
        return f"  {label:<26} {_fmt(float(value), precision):>12} {unit}"

    if measure == "entropy":
        lines = [
            f"Shannon entropy  (base {result['base']}, {unit})",
            "",
            f"  {'Outcomes:':<26} {int(result['outcomes']):>12}",
            line("Entropy H(X):", "entropy"),
            line("Maximum (uniform):", "max_entropy"),
            f"  {'Efficiency:':<26} "
            f"{_fmt(float(result['efficiency']) * 100, precision):>12} %",
        ]
    elif measure in ("kl", "cross"):
        title = "KL divergence" if measure == "kl" else "Cross-entropy"
        lines = [
            f"{title}  (base {result['base']}, {unit})",
            "",
            line("Entropy H(P):", "entropy_p"),
            line("KL divergence D(P||Q):", "kl_divergence"),
            line("Cross-entropy H(P,Q):", "cross_entropy"),
            line("Reverse KL D(Q||P):", "kl_reverse"),
            "",
            "  Note: KL is asymmetric — D(P||Q) is the extra cost of coding",
            "  samples from P with a code built for Q.",
        ]
    else:
        title = "Mutual information" if measure == "mi" else "Conditional entropy"
        lines = [
            f"{title}  (base {result['base']}, {unit})",
            "",
            line("Entropy H(X):", "entropy_x"),
            line("Entropy H(Y):", "entropy_y"),
            line("Joint entropy H(X,Y):", "joint_entropy"),
            line("Mutual information I(X;Y):", "mutual_information"),
            line("Conditional H(Y|X):", "conditional_entropy_y_given_x"),
            line("Conditional H(X|Y):", "conditional_entropy_x_given_y"),
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
    """Run the information entropy CLI.

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
