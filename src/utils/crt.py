#!/usr/bin/env python3
"""Command-line utility for Sunzi's Theorem (CRT).

Solves systems of simultaneous congruences:
  x ≡ a₁ (mod n₁)
  x ≡ a₂ (mod n₂)
  ...

Works for both coprime and non-coprime moduli (general CRT).  Remainders
are normalized to [0, n) automatically.  Pure Python — no external
dependencies.

Usage examples:
  crt --solve 2 3 3 5 2 7
  crt --solve 0 4 3 6 --format json
  crt --solve 2 3 3 5 2 7 --all 200
"""

from __future__ import annotations

import argparse
import json
import sys

# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def extended_gcd(a: int, b: int) -> tuple[int, int, int]:
    """Compute the extended greatest common divisor.

    Returns (g, s, t) such that a·s + b·t = g = gcd(a, b).

    Args:
        a: First integer.
        b: Second integer.

    Returns:
        Tuple (g, s, t) with a·s + b·t = g.
    """
    if b == 0:
        return a, 1, 0
    g, x, y = extended_gcd(b, a % b)
    return g, y, x - (a // b) * y


def mod_inverse(a: int, m: int) -> int:
    """Compute the modular inverse of a modulo m.

    Args:
        a: Integer to invert.
        m: Modulus; must be >= 2.

    Returns:
        x in [0, m) such that a·x ≡ 1 (mod m).

    Raises:
        ValueError: If gcd(a, m) ≠ 1 (inverse does not exist) or m < 2.
    """
    if m < 2:
        raise ValueError(f"modulus must be at least 2, got {m}")
    g, s, _ = extended_gcd(a % m, m)
    if g != 1:
        raise ValueError(f"{a} has no inverse modulo {m} (gcd={g})")
    return s % m


def crt(remainders: list[int], moduli: list[int]) -> tuple[int, int]:
    """Solve a system of congruences using Sunzi's Theorem (general CRT).

    Handles both coprime and non-coprime moduli via pairwise merge.
    Remainders are normalized to [0, nᵢ) internally.

    Args:
        remainders: Residues a₁, a₂, ..., aₖ (any integers).
        moduli: Moduli n₁, n₂, ..., nₖ; each must be >= 1.

    Returns:
        Tuple (x, N) where x is the unique solution in [0, N) and
        N = lcm(n₁, ..., nₖ).

    Raises:
        ValueError: If lengths differ, any modulus < 1, or the system is
            inconsistent (no solution).
    """
    if len(remainders) != len(moduli):
        raise ValueError("remainders and moduli must have the same length")
    if not remainders:
        raise ValueError("at least one congruence is required")
    for n in moduli:
        if n < 1:
            raise ValueError(f"moduli must be >= 1, got {n}")

    x = remainders[0] % moduli[0]
    m = moduli[0]
    for a, n in zip(remainders[1:], moduli[1:]):
        a = a % n
        g, p, _ = extended_gcd(m, n)
        if (a - x) % g != 0:
            raise ValueError(
                f"No solution: x≡{x} (mod {m}) and x≡{a} (mod {n}) are inconsistent"
            )
        lcm = m * (n // g)
        # Step k along m until congruent with a (mod n)
        x = (x + m * ((a - x) // g * p % (n // g))) % lcm
        m = lcm
    return x, m


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list; uses sys.argv[1:] when None.

    Returns:
        Parsed namespace.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Sunzi's Theorem (Chinese Remainder Theorem) solver.\n\n"
            "Finds x satisfying a system of simultaneous congruences:\n"
            "  x ≡ A₁ (mod N₁)\n"
            "  x ≡ A₂ (mod N₂)\n"
            "  ...\n\n"
            "Returns the unique solution x in [0, lcm(N₁, N₂, ...)) and the combined\n"
            "modulus. Handles both coprime and non-coprime moduli; reports an error\n"
            "when no solution exists."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  crt --solve 2 3 3 5 2 7
      # remainder=2 mod=3, remainder=3 mod=5, remainder=2 mod=7
      # → x = 23 (mod 105)

  crt --solve 0 4 2 6
      # remainder=0 mod=4, remainder=2 mod=6  (non-coprime moduli)
      # → x = 8 (mod 12)

  crt --solve 2 3 3 5 2 7 --all 300   # list all solutions ≤ 300
  crt --solve 2 3 3 5 2 7 --format json
""",
    )
    parser.add_argument(
        "--solve",
        type=int,
        nargs="+",
        metavar="A N",
        help=(
            "alternating remainder/modulus pairs: A₁ N₁ A₂ N₂ ...\n"
            "Each pair encodes one congruence x ≡ Aᵢ (mod Nᵢ): Aᵢ is the\n"
            "remainder (residue) and Nᵢ is the modulus (divisor).\n"
            "At least one pair is required."
        ),
    )
    parser.add_argument(
        "--all",
        type=int,
        metavar="MAX",
        help="list all solutions in [0, MAX]",
    )
    parser.add_argument(
        "--format",
        "-f",
        choices=["table", "json"],
        default="table",
        help="output format: table (default) or json",
    )

    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate(args: argparse.Namespace) -> str | None:
    """Validate parsed arguments.

    Args:
        args: Parsed namespace from parse_args.

    Returns:
        Error message if invalid; None if all arguments are valid.
    """
    if args.solve is None:
        return "--solve is required"
    if len(args.solve) < 2:
        return "--solve requires at least one remainder/modulus pair (A N)"
    if len(args.solve) % 2 != 0:
        return "--solve requires an even number of values (A N pairs)"
    vals = args.solve
    for i in range(1, len(vals), 2):
        if vals[i] < 1:
            return f"moduli must be >= 1 (got {vals[i]} at position {i + 1})"
    if args.all is not None and args.all < 0:
        return "--all MAX must be non-negative"
    return None


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def format_solution(
    remainders: list[int],
    moduli: list[int],
    x: int,
    N: int,
    all_max: int | None,
) -> str:
    """Format CRT solution as a human-readable table.

    Args:
        remainders: Input residues (raw, before normalization).
        moduli: Input moduli.
        x: Unique solution in [0, N).
        N: Period (lcm of moduli).
        all_max: If not None, list all solutions up to this bound.

    Returns:
        Formatted string.
    """
    lines = [
        "Sunzi's Theorem",
        "=" * 40,
        "Congruences:",
    ]
    for a, n in zip(remainders, moduli):
        lines.append(f"  x ≡ {a} (mod {n})")

    lines += [
        "",
        f"Solution:  x ≡ {x} (mod {N})",
        "",
        "Verification:",
    ]
    for a, n in zip(remainders, moduli):
        check = x % n
        tick = "✓" if check == (a % n) else "✗"
        lines.append(f"  {x} mod {n} = {check} {tick}")

    if all_max is not None:
        solutions = list(range(x, all_max + 1, N))
        lines += [
            "",
            f"All solutions in [0, {all_max}]:",
            "  " + ", ".join(str(s) for s in solutions),
        ]

    return "\n".join(lines)


def format_json(
    remainders: list[int],
    moduli: list[int],
    x: int,
    N: int,
    all_max: int | None,
) -> str:
    """Format CRT solution as JSON.

    Args:
        remainders: Input residues.
        moduli: Input moduli.
        x: Unique solution in [0, N).
        N: Period.
        all_max: If not None, include all solutions up to this bound.

    Returns:
        JSON string.
    """
    payload: dict = {
        "solution": x,
        "modulus": N,
        "congruences": [
            {
                "remainder": a,
                "modulus": n,
                "check": x % n,
                "valid": (x % n) == (a % n),
            }
            for a, n in zip(remainders, moduli)
        ],
    }
    if all_max is not None:
        payload["all_solutions"] = list(range(x, all_max + 1, N))
    return json.dumps(payload, indent=2)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Run the crt CLI.

    Args:
        argv: Argument list; uses sys.argv[1:] when None.

    Returns:
        Exit code: 0 on success, 2 on input error.
    """
    args = parse_args(argv)

    error = validate(args)
    if error:
        print(f"Error: {error}", file=sys.stderr)
        return 2

    vals = args.solve
    remainders = [vals[i] for i in range(0, len(vals), 2)]
    moduli = [vals[i] for i in range(1, len(vals), 2)]

    try:
        x, N = crt(remainders, moduli)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    if args.format == "json":
        print(format_json(remainders, moduli, x, N, args.all))
    else:
        print(format_solution(remainders, moduli, x, N, args.all))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
