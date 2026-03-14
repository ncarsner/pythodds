#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from typing import Optional

"""Command-line utility for Bayes' theorem calculations.

Computes posterior probability P(A|B) using:

    P(A|B) = P(B|A) * P(A) / P(B)

You can provide the evidence term P(B) directly, or provide the false-positive
rate P(B|not A) and let the tool derive P(B) via the law of total probability.

Real-world examples:

  Medical test (disease has 1% prevalence, test is 99% accurate, 5% false-positive rate):
    "If I test positive, what's the chance I actually have the disease?"
    bayes -p 0.01 -l 0.99 -f 0.05
    → Answer: ~16.7% (most positive results are false alarms due to low prevalence)

  Email spam filter (20% of emails are spam, filter catches 80% of spam):
    "Given an email triggered the spam filter, what's the chance it's really spam?"
    bayes -p 0.2 -l 0.8 -e 0.5
    → Answer: 32% (if half of all emails trigger the filter)

  Weather forecast (30% chance of rain, forecaster is right 90% of the time):
    "Forecaster says it will rain. What's the actual probability?"
    bayes -p 0.3 -l 0.9 -f 0.1
    → Answer: ~79.4%

Parameters explained:
  -p / --prior:          Base rate probability before seeing evidence
  -l / --likelihood:     How often the evidence appears when A is true
  -f / --false-positive: How often the evidence appears when A is false
  -e / --evidence:       Overall probability of seeing the evidence (alternative to -f)
"""


def bayes_posterior(prior: float, likelihood: float, evidence: float) -> float:
    """Return P(A|B) given P(A), P(B|A), and P(B)."""
    if evidence <= 0.0:
        raise ValueError("evidence must be greater than 0")
    return (likelihood * prior) / evidence


def evidence_from_false_positive(
    prior: float,
    likelihood: float,
    false_positive: float,
) -> float:
    """Return P(B) from P(B|A), P(B|not A), and P(A)."""
    return (likelihood * prior) + (false_positive * (1.0 - prior))


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bayes' theorem posterior probability calculator.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Real-world examples:

  Medical test (1% disease prevalence, 99% accurate, 5% false-positive):
    bayes -p 0.01 -l 0.99 -f 0.05

  Email spam filter (20% spam rate, 80% detection rate, 50% evidence):
    bayes -p 0.2 -l 0.8 -e 0.5

  Weather forecast (30% rain chance, 90% forecaster accuracy, 10% false alarm):
    bayes -p 0.3 -l 0.9 -f 0.1

  Pregnancy test (5% prevalence, 98% accuracy, 2% false-positive rate):
    bayes -p 0.05 -l 0.98 -f 0.02

  Security alarm (0.5% chance of break-in, alarm triggers 95% for break-in, 3% false alarm):
    bayes -p 0.005 -l 0.95 -f 0.03

  Job interview (60% qualified, 80% likeable, 30% chance interviewer likes unqualified):
    bayes -p 0.6 -l 0.8 -f 0.3
""",
    )

    parser.add_argument(
        "--prior",
        "-p",
        type=float,
        required=True,
        metavar="P_A",
        help="prior probability P(A), between 0 and 1",
    )
    parser.add_argument(
        "--likelihood",
        "-l",
        type=float,
        required=True,
        metavar="P_B_GIVEN_A",
        help="likelihood P(B|A), between 0 and 1",
    )

    evidence_group = parser.add_mutually_exclusive_group()
    evidence_group.add_argument(
        "--evidence",
        "-e",
        type=float,
        metavar="P_B",
        help="evidence probability P(B), between 0 and 1",
    )
    evidence_group.add_argument(
        "--false-positive",
        "-f",
        type=float,
        metavar="P_B_GIVEN_NOT_A",
        help="false-positive rate P(B|not A), between 0 and 1",
    )

    parser.add_argument(
        "--precision",
        "-P",
        type=int,
        default=6,
        help="decimal places for output (default: 6)",
    )

    return parser.parse_args(argv)


def validate(args: argparse.Namespace) -> Optional[str]:
    if args.evidence is None and args.false_positive is None:
        return "one of -e/--evidence or -f/--false-positive is required"

    if not (0.0 <= args.prior <= 1.0):
        return "-p/--prior must be between 0 and 1"

    if not (0.0 <= args.likelihood <= 1.0):
        return "-l/--likelihood must be between 0 and 1"

    if args.evidence is not None:
        if not (0.0 < args.evidence <= 1.0):
            return "-e/--evidence must be greater than 0 and at most 1"

    if args.false_positive is not None:
        if not (0.0 <= args.false_positive <= 1.0):
            return "-f/--false-positive must be between 0 and 1"

    if args.precision < 0:
        return "-P/--precision must be non-negative"

    return None


def _fmt_prob(x: float, precision: int) -> str:
    pct = x * 100.0
    fmt = f"{{:.{precision}f}}"
    return f"{fmt.format(x)} ({fmt.format(pct)}%)"


def format_output(
    prior: float,
    likelihood: float,
    evidence: float,
    posterior: float,
    false_positive: Optional[float],
    precision: int,
) -> str:
    lines = [
        f"Prior P(A):                 {_fmt_prob(prior, precision)}",
        f"Likelihood P(B|A):          {_fmt_prob(likelihood, precision)}",
    ]
    if false_positive is not None:
        lines.append(
            f"False-positive P(B|not A):  {_fmt_prob(false_positive, precision)}"
        )
    lines.extend(
        [
            f"Evidence P(B):              {_fmt_prob(evidence, precision)}",
            f"Posterior P(A|B):           {_fmt_prob(posterior, precision)}",
        ]
    )
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    error = validate(args)
    if error:
        print(f"Error: {error}", file=sys.stderr)
        return 2

    if args.evidence is not None:
        evidence = args.evidence
        false_positive = None
    else:
        false_positive = args.false_positive
        evidence = evidence_from_false_positive(
            args.prior, args.likelihood, false_positive
        )

    # Guard against pathological floating-point / invalid inputs that collapse evidence.
    if evidence <= 0.0 or not math.isfinite(evidence):
        print(
            "Error: derived evidence must be finite and greater than 0", file=sys.stderr
        )
        return 2

    posterior = bayes_posterior(args.prior, args.likelihood, evidence)
    print(
        format_output(
            args.prior,
            args.likelihood,
            evidence,
            posterior,
            false_positive,
            args.precision,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
