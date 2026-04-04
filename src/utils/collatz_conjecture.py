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

"""

from __future__ import annotations

import argparse
import time
from typing import Dict, List, Set


def collatz_next(x: int) -> int:
    return x // 2 if x % 2 == 0 else 3 * x + 1


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
    p.add_argument(
        "--n",
        type=int,
        required=True,
        help="Upper bound (inclusive) of starting integers to check",
    )
    p.add_argument(
        "--interval",
        type=int,
        default=1000,
        help="How often (in starts) to update/print progress",
    )
    p.add_argument("--verbose", action="store_true", help="Print progress messages")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    checker = CollatzChecker()
    checker.ensure_up_to(args.n, check_interval=args.interval, verbose=args.verbose)
    print(f"max_valid = {checker.max_valid}")


if __name__ == "__main__":
    main()
