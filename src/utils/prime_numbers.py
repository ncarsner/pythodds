#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from typing import Optional

"""Command-line utility for prime number operations.

Provides functionality for prime checking, finding nth primes, counting primes,
listing primes in ranges, and computing prime factorizations.

Usage examples:
  prime --check 97
  prime --nth 100
  prime --count 1000
  prime --range 50 100
  prime --factorize 360
  prime --range 1 50 --format json
"""


# ---------------------------------------------------------------------------
# Core prime functions
# ---------------------------------------------------------------------------


def is_prime(n: int) -> bool:
    """Check if n is a prime number.

    Uses trial division with optimizations:
    - Returns False for n < 2
    - Checks 2 and 3 explicitly
    - Tests only 6k±1 candidates up to √n
    """
    if n < 2:
        return False
    if n == 2 or n == 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False

    # Check for divisors of the form 6k±1 up to √n
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6

    return True


def nth_prime(n: int) -> int:
    """Return the nth prime number (1-indexed).

    Uses a sieve-based approach for efficiency with larger n.
    For n=1, returns 2 (the first prime).
    """
    if n < 1:
        raise ValueError("n must be at least 1")

    if n == 1:
        return 2
    if n == 2:
        return 3

    # Use prime counting approximation to estimate upper bound
    # The nth prime is approximately n * ln(n) for large n
    if n < 6:
        estimate = 15
    else:
        estimate = int(n * (math.log(n) + math.log(math.log(n))) * 1.3)

    # Generate primes using sieve until we have at least n primes
    while True:
        primes = list(sieve_of_eratosthenes(estimate))
        if len(primes) >= n:
            return primes[n - 1]
        estimate = int(estimate * 1.5)


def count_primes(limit: int) -> int:
    """Count the number of primes up to and including limit.

    This is the prime counting function π(n).
    """
    if limit < 2:
        return 0
    return len(list(sieve_of_eratosthenes(limit)))


def sieve_of_eratosthenes(limit: int) -> list[int]:
    """Generate all prime numbers up to and including limit using the Sieve of Eratosthenes.

    Returns a list of primes in ascending order.
    """
    if limit < 2:
        return []

    # Initialize sieve
    is_prime_arr = [True] * (limit + 1)
    is_prime_arr[0] = is_prime_arr[1] = False

    # Sieve
    for i in range(2, int(math.sqrt(limit)) + 1):
        if is_prime_arr[i]:
            for j in range(i * i, limit + 1, i):
                is_prime_arr[j] = False

    # Collect primes
    return [i for i in range(limit + 1) if is_prime_arr[i]]


def primes_in_range(start: int, end: int) -> list[int]:
    """Return all prime numbers in the range [start, end] inclusive."""
    if start < 2:
        start = 2
    if end < start:
        return []

    # Generate all primes up to end, then filter
    all_primes = sieve_of_eratosthenes(end)
    return [p for p in all_primes if p >= start]


def prime_factorization(n: int) -> dict[int, int]:
    """Return the prime factorization of n as a dict of {prime: exponent}.

    Example: prime_factorization(360) returns {2: 3, 3: 2, 5: 1}
    representing 2³ × 3² × 5¹ = 360
    """
    if n < 2:
        raise ValueError("n must be at least 2")

    factors = {}

    # Handle factor of 2
    while n % 2 == 0:
        factors[2] = factors.get(2, 0) + 1
        n //= 2

    # Handle odd factors
    i = 3
    while i * i <= n:
        while n % i == 0:
            factors[i] = factors.get(i, 0) + 1
            n //= i
        i += 2

    # If n is still > 1, then it's a prime factor
    if n > 1:
        factors[n] = factors.get(n, 0) + 1

    return factors


def format_factorization(factors: dict[int, int]) -> str:
    """Format prime factorization as a readable string.

    Example: {2: 3, 3: 2, 5: 1} → "2³ × 3² × 5"
    """
    if not factors:
        return "1"

    superscripts = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
    terms = []
    for prime in sorted(factors.keys()):
        exp = factors[prime]
        if exp == 1:
            terms.append(str(prime))
        else:
            terms.append(f"{prime}{str(exp).translate(superscripts)}")
    return " × ".join(terms)


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prime number operations: check, find, count, list, factorize"
    )

    # Mutually exclusive operation modes
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "-is",
        "--check",
        type=int,
        metavar="N",
        help="check if N is prime",
    )
    group.add_argument(
        "-n",
        "--nth",
        type=int,
        metavar="N",
        help="find the Nth prime number (1-indexed)",
    )
    group.add_argument(
        "-C",
        "--count",
        type=int,
        metavar="LIMIT",
        help="count primes up to and including LIMIT (π function)",
    )
    group.add_argument(
        "-R",
        "--range",
        type=int,
        nargs=2,
        metavar=("START", "END"),
        help="list primes in range [START, END] inclusive",
    )
    group.add_argument(
        "-F",
        "--factorize",
        type=int,
        metavar="N",
        help="compute prime factorization of N",
    )

    # Output formatting
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="output format (default: text)",
    )

    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def format_check_result(n: int, result: bool, fmt: str) -> str:
    """Format the result of a primality check."""
    if fmt == "json":
        return json.dumps({"n": n, "is_prime": result}, indent=2)
    else:
        status = "is prime" if result else "is NOT prime"
        return f"{n} {status}"


def format_nth_result(n: int, prime: int, fmt: str) -> str:
    """Format the result of finding the nth prime."""
    if fmt == "json":
        return json.dumps({"n": n, "nth_prime": prime}, indent=2)
    else:
        return f"The {n}th prime number is {prime:,}"


def format_count_result(limit: int, count: int, fmt: str) -> str:
    """Format the result of counting primes."""
    if fmt == "json":
        return json.dumps({"limit": limit, "prime_count": count}, indent=2)
    else:
        return f"π({limit:,}) = {count:,} (there are {count:,} primes ≤ {limit:,})"


def format_range_result(start: int, end: int, primes: list[int], fmt: str) -> str:
    """Format the result of listing primes in a range."""
    if fmt == "json":
        return json.dumps(
            {"start": start, "end": end, "count": len(primes), "primes": primes},
            indent=2,
        )
    else:
        lines = [
            f"Primes in [{start}, {end}]: {len(primes)} found",
            "",
        ]
        # Display primes in rows of 10
        for i in range(0, len(primes), 10):
            row = primes[i : i + 10]
            lines.append("  " + ", ".join(str(p) for p in row))
        return "\n".join(lines)


def format_factorization_result(n: int, factors: dict[int, int], fmt: str) -> str:
    """Format the result of prime factorization."""
    if fmt == "json":
        return json.dumps({"n": n, "factors": factors}, indent=2)
    else:
        factorization = format_factorization(factors)
        lines = [
            f"Prime factorization of {n:,}:",
            f"  {n:,} = {factorization}",
            "",
            "Factor breakdown:",
        ]
        for prime in sorted(factors.keys()):
            exp = factors[prime]
            lines.append(f"  {prime}^{exp} = {prime**exp}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    """Main entry point for the prime CLI tool."""
    try:
        args = parse_args(argv)

        if args.check is not None:
            n = args.check
            if n < 0:
                print("Error: number must be non-negative", file=sys.stderr)
                return 2
            result = is_prime(n)
            print(format_check_result(n, result, args.format))

        elif args.nth is not None:
            n = args.nth
            if n < 1:
                print("Error: n must be at least 1", file=sys.stderr)
                return 2
            prime = nth_prime(n)
            print(format_nth_result(n, prime, args.format))

        elif args.count is not None:
            limit = args.count
            if limit < 0:
                print("Error: limit must be non-negative", file=sys.stderr)
                return 2
            count = count_primes(limit)
            print(format_count_result(limit, count, args.format))

        elif args.range is not None:
            start, end = args.range
            if start < 0 or end < 0:
                print("Error: range values must be non-negative", file=sys.stderr)
                return 2
            if start > end:
                print("Error: start must be <= end", file=sys.stderr)
                return 2
            primes = primes_in_range(start, end)
            print(format_range_result(start, end, primes, args.format))

        elif args.factorize is not None:
            n = args.factorize
            if n < 2:
                print(
                    "Error: number must be at least 2 for factorization",
                    file=sys.stderr,
                )
                return 2
            factors = prime_factorization(n)
            print(format_factorization_result(n, factors, args.format))

        return 0

    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nInterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
