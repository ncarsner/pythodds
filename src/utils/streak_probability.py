#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from typing import Optional

"""Command-line utility for streak (consecutive run) probability calculations.

Computes the probability of at least one run of k consecutive successes in n
independent Bernoulli trials with success probability p, and the expected
length of the longest run of consecutive successes.

Usage examples:
  streak -n 100 -k 5 -p 0.5
  streak -n 162 -p 0.300 --longest
"""


# ---------------------------------------------------------------------------
# Core probability functions
# ---------------------------------------------------------------------------

def prob_at_least_one_streak(n: int, k: int, p: float) -> float:
    """P(at least one run of k consecutive successes in n Bernoulli(p) trials).

    Uses a DP state vector of length k: dp[j] = probability that the last j
    trials were all successes and no run of length k has occurred yet.
    Time: O(n·k).  Space: O(k).
    """
    if n <= 0 or k <= 0 or not (0.0 <= p <= 1.0):
        raise ValueError("Requires n > 0, k > 0, and 0 <= p <= 1")
    if p == 0.0:
        return 0.0
    if p == 1.0:
        return 1.0 if n >= k else 0.0
    if k > n:
        return 0.0

    q = 1.0 - p
    # dp[j] = P(currently j consecutive successes at end, no streak of k yet)
    dp = [0.0] * k
    dp[0] = 1.0  # initial state: 0 consecutive successes, no streak seen

    for _ in range(n):
        new_dp = [0.0] * k
        for j in range(k):
            if dp[j] == 0.0:
                continue
            # failure: reset current run to 0
            new_dp[0] += dp[j] * q
            # success: extend current run by 1
            if j + 1 < k:
                new_dp[j + 1] += dp[j] * p
            # if j + 1 == k: streak achieved — probability is absorbed
        dp = new_dp

    prob_no_streak = sum(dp)
    return max(0.0, min(1.0, 1.0 - prob_no_streak))


def expected_longest_streak(n: int, p: float) -> float:
    """Expected length of the longest run of consecutive successes in n trials.

    Uses the identity  E[L] = Σ_{k=1}^{n} P(L ≥ k)
                             = Σ_{k=1}^{n} prob_at_least_one_streak(n, k, p).
    Terminates early when the incremental term is negligible (< 1e-12).
    """
    if n <= 0 or not (0.0 <= p <= 1.0):
        raise ValueError("Requires n > 0 and 0 <= p <= 1")
    if p == 0.0:
        return 0.0
    if p == 1.0:
        return float(n)

    total = 0.0
    for k in range(1, n + 1):
        term = prob_at_least_one_streak(n, k, p)
        total += term
        if term < 1e-12:
            break
    return total


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def _fmt_prob(x: float, precision: int) -> str:
    pct = x * 100.0
    fmt = f"{{:.{precision}f}}"
    return f"{fmt.format(x)} ({fmt.format(pct)}%)"


def format_streak_output(n: int, k: int, p: float, at_least_one: float, precision: int) -> str:
    label_at_least = f"P(streak \u2265 {k}):"
    label_none = f"P(no streak \u2265 {k}):"
    width = max(len(label_at_least), len(label_none)) + 2
    lines = [
        f"{'Trials (n):'.ljust(width)}{n:,}",
        f"{'Streak length (k):'.ljust(width)}{k:,}",
        f"{'Success probability (p):'.ljust(width)}{p:.{precision}f}",
        f"{label_at_least.ljust(width)}{_fmt_prob(at_least_one, precision)}",
        f"{label_none.ljust(width)}{_fmt_prob(1.0 - at_least_one, precision)}",
    ]
    return "\n".join(lines)


def format_longest_output(n: int, p: float, expected: float, precision: int) -> str:
    width = len("Success probability (p):") + 2
    lines = [
        f"{'Trials (n):'.ljust(width)}{n:,}",
        f"{'Success probability (p):'.ljust(width)}{p:.{precision}f}",
        f"{'E[longest streak]:'.ljust(width)}{expected:.{precision}f}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Streak (consecutive run) probability calculator.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  streak -n 100 -k 5 -p 0.5
  streak -n 162 -p 0.300 --longest
""",
    )
    parser.add_argument(
        "--trials", "-n", type=int, required=True, metavar="N",
        help="total number of independent trials",
    )
    parser.add_argument(
        "--prob", "-p", type=float, required=True, metavar="P",
        help="probability of success on a single trial (0..1)",
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--streak-length", "-k", type=int, metavar="K",
        help="compute P(at least one run of K consecutive successes)",
    )
    mode.add_argument(
        "--longest", action="store_true",
        help="compute E[length of longest run of consecutive successes]",
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
    if args.trials <= 0:
        return "-n/--trials must be a positive integer"
    if not (0.0 <= args.prob <= 1.0):
        return "-p/--prob must be between 0 and 1"
    if args.streak_length is not None and args.streak_length <= 0:
        return "-k/--streak-length must be a positive integer"
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

    n = args.trials
    p = args.prob
    precision = args.precision

    if args.longest:
        expected = expected_longest_streak(n, p)
        print(format_longest_output(n, p, expected, precision))
        return 0

    # streak-length mode
    k = args.streak_length
    at_least_one = prob_at_least_one_streak(n, k, p)
    print(format_streak_output(n, k, p, at_least_one, precision))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
