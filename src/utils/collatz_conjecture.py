"""
collatz_conjecture.py

Script to evaluate positive integers in the context of the Collatz conjecture.

Behavior:
- For each start value from 1..n, follow the Collatz sequence until reaching 1 or a number already
  known to resolve to 1.
- Maintain a set `resolved` of numbers proven to reach 1; when a sequence reaches a resolved
  number, mark all visited numbers as resolved.
- Maintain `max_valid`: the largest consecutive integer k such that all integers 1..k are in `resolved`.
- Periodically (configurable interval) update/print `max_valid` to avoid scanning the whole set repeatedly.

Usage:
    python collatz_conjecture.py --n 100000 --interval 1000
    python collatz_conjecture.py --trace 27
    python collatz_conjecture.py --trace-range 1 10
    python collatz_conjecture.py --trace-range 1 10 --summary-only

"""

from __future__ import annotations

import argparse
import time
from typing import Dict, List, Set


def collatz_next(x: int) -> int:
    return x // 2 if x % 2 == 0 else 3 * x + 1


def trace_collatz_sequence(
    start: int, show_steps: bool = True
) -> tuple[list[int], int]:
    """
    Trace the Collatz sequence from a starting number to 1.

    Args:
        start: Starting positive integer
        show_steps: If True, prints each step in the sequence

    Returns:
        Tuple of (sequence, step_count) where sequence is the list of values
        and step_count is the number of steps to reach 1
    """
    if start < 1:
        raise ValueError(f"Starting value must be positive, got {start}")

    sequence = [start]
    x = start

    if show_steps:
        print(f"\nTracing Collatz sequence for {start}:")
        print(f"  Step 0: {x}")

    step = 0
    while x != 1:
        x = collatz_next(x)
        step += 1
        sequence.append(x)

        if show_steps:
            operation = "÷ 2" if sequence[-2] % 2 == 0 else "× 3 + 1"
            print(f"  Step {step}: {sequence[-2]} → {x} ({operation})")

    if show_steps:
        print(f"  Reached 1 in {step} steps")
        print(f"  Max value in sequence: {max(sequence)}")
        print(f"  Sequence length: {len(sequence)}")

    return sequence, step


class CollatzChecker:
    def __init__(self) -> None:
        # Numbers known to reach 1
        self.resolved: Set[int] = {1}
        # Largest k such that all numbers 1..k are known to resolve
        self.max_valid: int = 1
        # Steps required for known numbers to reach 1 (1 -> 0)
        self.steps_for: Dict[int, int] = {1: 0}
        # Histogram: steps -> count of starting integers (within checked range) that take that many steps
        self.steps_histogram: Dict[int, int] = {}

    def _update_max_valid(self) -> None:
        # Incrementally raise max_valid while next integer is in resolved
        while (self.max_valid + 1) in self.resolved:
            self.max_valid += 1

    def ensure_up_to(
        self, n: int, check_interval: int = 1000, verbose: bool = False
    ) -> None:
        """
        Ensure that all start values from 1..n have been processed (not necessarily proven),
        populating `self.resolved` with numbers known to reach 1. Periodically updates `max_valid`.

        Args:
            n: upper bound (inclusive) for starting values
            check_interval: how many iterations between max_valid updates/prints
            verbose: if True, prints progress messages
        """
        start_time = time.time()
        last_print = start_time
        for start in range(1, n + 1):
            if start in self.steps_for:
                # already known (either pre-resolved or precomputed as a path member of an
                # earlier traversal); use the cached step count directly
                steps = self.steps_for[start]
                self.steps_histogram.setdefault(steps, 0)
                self.steps_histogram[steps] += 1
                # Mark this integer as a directly-evaluated resolved start
                self.resolved.add(start)
                if start == self.max_valid + 1:
                    self._update_max_valid()
                continue

            x = start
            path: List[int] = []
            # Walk until we hit any number whose step count is already known
            # (either a directly-evaluated start or a precomputed path member)
            while x not in self.steps_for:
                path.append(x)
                x = collatz_next(x)
                # Safety: avoid infinite loops (should not happen with Collatz sequences)
                if x < 1:
                    raise ValueError(f"Unexpected value in Collatz sequence: {x}")

            # Steps = number of hops until the path reaches a number with a known step count.
            # Any number in steps_for (direct start or precomputed path member) terminates the walk.
            total_steps = len(path)

            # Assign steps for each visited node (first-write-wins):
            # - The start itself (i==0) gets its full hop count.
            # - The last path member (i==len-1, i>0) gets 1 if its immediate Collatz
            #   successor is a directly-evaluated start (in self.resolved), else 0.
            # - All other intermediate members get 0: they are "validated" (known to
            #   reach 1 via a previously-traversed path) so no further hops are needed.
            last_idx = len(path) - 1
            for i, val in enumerate(path):
                if val not in self.steps_for:
                    if i == 0:
                        self.steps_for[val] = total_steps
                    elif i == last_idx:
                        self.steps_for[val] = (
                            1 if collatz_next(val) in self.resolved else 0
                        )
                    else:
                        self.steps_for[val] = 0

            # Record histogram entry for this starting number
            self.steps_histogram.setdefault(self.steps_for[start], 0)
            self.steps_histogram[self.steps_for[start]] += 1

            # Mark this integer as a directly-evaluated resolved start
            self.resolved.add(start)
            if start == self.max_valid + 1:
                self._update_max_valid()

            # Periodic progress
            if check_interval > 0 and start % check_interval == 0:
                elapsed = time.time() - last_print
                last_print = time.time()
                if verbose:
                    print(
                        f"Checked up to {start}; max_valid={self.max_valid}; resolved_size={len(self.resolved)}; time={elapsed:.2f}s"
                    )

        # Final update
        self._update_max_valid()
        total_time = time.time() - start_time
        if verbose:
            print(
                f"Finished checking 1..{n}; max_valid={self.max_valid}; resolved_size={len(self.resolved)}; total_time={total_time:.2f}s"
            )
            # Print basic histogram summary
            print("Steps histogram (steps -> count of starts):")
            for steps in sorted(self.steps_histogram.keys()):
                print(f"  {steps}: {self.steps_histogram[steps]}")

    def proven(self, k: int) -> bool:
        """Return True if k is known to resolve to 1."""
        return k in self.resolved


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Collatz checker with caching and incremental max_valid"
    )

    # Mode selection (mutually exclusive)
    mode_group = p.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--n",
        type=int,
        help="Upper bound (inclusive) of starting integers to check",
    )
    mode_group.add_argument(
        "--trace",
        type=int,
        metavar="NUM",
        help="Show step-by-step Collatz sequence for a specific number",
    )
    mode_group.add_argument(
        "--trace-range",
        type=int,
        nargs=2,
        metavar=("START", "END"),
        help="Show step-by-step sequences for a range of numbers (inclusive)",
    )

    # Options
    p.add_argument(
        "--interval",
        type=int,
        default=1000,
        help="How often (in starts) to update/print progress (for --n mode)",
    )
    p.add_argument("--verbose", action="store_true", help="Print progress messages")
    p.add_argument(
        "--summary-only",
        action="store_true",
        help="For trace modes, show only summary statistics instead of step-by-step",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # Validate that at least one mode is selected
    if not any([args.n, args.trace, args.trace_range]):
        print("Error: Must specify one of --n, --trace, or --trace-range")
        return 2

    # Handle trace mode for a single number
    if args.trace:
        try:
            sequence, steps = trace_collatz_sequence(
                args.trace, show_steps=not args.summary_only
            )
            if args.summary_only:
                print(f"Number: {args.trace}")
                print(f"Steps to reach 1: {steps}")
                print(f"Max value in sequence: {max(sequence)}")
                print(f"Sequence length: {len(sequence)}")
        except ValueError as e:
            print(f"Error: {e}")
            return 1
        return 0

    # Handle trace-range mode
    if args.trace_range:
        start, end = args.trace_range
        if start < 1 or end < 1:
            print("Error: Range values must be positive integers")
            return 1
        if start > end:
            print("Error: START must be less than or equal to END")
            return 1

        print(f"Tracing Collatz sequences for range [{start}, {end}]:\n")
        results = []

        for num in range(start, end + 1):
            sequence, steps = trace_collatz_sequence(
                num, show_steps=not args.summary_only
            )
            results.append((num, steps, max(sequence), len(sequence)))

            if args.summary_only:
                print(
                    f"{num}: {steps} steps, max={max(sequence)}, length={len(sequence)}"
                )

        # Print summary statistics
        print(f"\n{'='*60}")
        print(f"Summary for range [{start}, {end}]:")
        print(f"  Numbers tested: {len(results)}")
        print(f"  Average steps: {sum(r[1] for r in results) / len(results):.2f}")
        print(
            f"  Max steps: {max(r[1] for r in results)} (number: {max(results, key=lambda x: x[1])[0]})"
        )
        print(
            f"  Min steps: {min(r[1] for r in results)} (number: {min(results, key=lambda x: x[1])[0]})"
        )
        print(
            f"  Highest peak: {max(r[2] for r in results)} (number: {max(results, key=lambda x: x[2])[0]})"
        )
        return 0

    # Handle standard mode with --n
    if args.n:
        checker = CollatzChecker()
        checker.ensure_up_to(args.n, check_interval=args.interval, verbose=args.verbose)
        print(f"max_valid = {checker.max_valid}")
        return 0


if __name__ == "__main__":
    main()
