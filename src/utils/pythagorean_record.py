#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from typing import Optional

"""Command-line utility for Pythagorean expectation record predictions.

Computes the expected winning percentage for a team based on runs/points scored
and allowed using either the traditional Pythagorean formula or the newer
linear formula from SABR research (Rothman, 2014).

Traditional Pythagorean Formula (Bill James):
    EXP(W%) = RS^exp / (RS^exp + RA^exp)

Linear Formula (Rothman, 2014):
    MLB: EXP(W%) = 0.000683(RS - RA) + 0.5
    NFL: EXP(W%) = 0.001538(PS - PA) + 0.5
    NBA: EXP(W%) = 0.000351(PS - PA) + 0.5

Usage examples:

  MLB team with 800 runs scored, 650 runs allowed (using linear formula):
    pythagorean_record --scored 800 --allowed 650
    → Expected W%: 60.25% (98 wins in 162 games)

  Same team using traditional Pythagorean (exponent 2):
    pythagorean_record --scored 800 --allowed 650 --method pythagorean
    → Expected W%: 60.19%

  NFL team with 420 points scored, 300 points allowed:
    pythagorean_record --scored 420 --allowed 300 --sport nfl
    → Expected W%: 68.46% (~12 wins in 17 games)

  NBA team with custom exponent comparison:
    pythagorean_record --scored 8500 --allowed 8200 --sport nba --method both
    → Shows both linear and Pythagorean predictions

  Custom game count (baseball season in progress):
    pythagorean_record --scored 450 --allowed 380 --games 100
    → Expected W%: 54.78% (~55 wins in 100 games)

Reference:
  Rothman, Stanley. "A New Formula to Predict a Team's Winning Percentage."
  SABR Baseball Research Journal, Fall 2014.
  https://sabr.org/journal/article/a-new-formula-to-predict-a-teams-winning-percentage/
"""


# ---------------------------------------------------------------------------
# Core calculation functions
# ---------------------------------------------------------------------------


def pythagorean_expectation(
    scored: float, allowed: float, exponent: float = 2.0
) -> float:
    """Traditional Pythagorean formula: RS^exp / (RS^exp + RA^exp)

    Args:
        scored: Runs or points scored by the team
        allowed: Runs or points allowed by the team
        exponent: Power to raise both terms (typically 2.0 for baseball)

    Returns:
        Expected winning percentage (0.0 to 1.0)
    """
    if scored < 0 or allowed < 0:
        raise ValueError("scored and allowed must be non-negative")
    if exponent <= 0:
        raise ValueError("exponent must be positive")

    # Handle edge cases
    if scored == 0 and allowed == 0:
        return 0.5
    if allowed == 0:
        return 1.0
    if scored == 0:
        return 0.0

    scored_exp = scored**exponent
    allowed_exp = allowed**exponent
    return scored_exp / (scored_exp + allowed_exp)


def linear_expectation(scored: float, allowed: float, sport: str = "mlb") -> float:
    """Linear formula from Rothman (2014): m*(scored - allowed) + 0.50


    Args:
        scored: Runs or points scored by the team
        allowed: Runs or points allowed by the team
        sport: One of "mlb", "nfl", or "nba"

    Returns:
        Expected winning percentage (0.0 to 1.0)
    """
    coefficients = {
        "mlb": 0.000683,
        "nfl": 0.001538,
        "nba": 0.000351,
    }

    if sport not in coefficients:
        raise ValueError(f"sport must be one of {list(coefficients.keys())}")

    m = coefficients[sport]
    diff = scored - allowed
    win_pct = m * diff + 0.50

    # Clamp to [0, 1] range (formula can theoretically exceed bounds with extreme values)
    return max(0.0, min(1.0, win_pct))


def expected_wins(win_percentage: float, games: int) -> float:
    """Calculate expected wins from winning percentage and games played.

    Args:
        win_percentage: Expected winning percentage (0.0 to 1.0)
        games: Number of games in the season

    Returns:
        Expected number of wins
    """
    return win_percentage * games


# ---------------------------------------------------------------------------
# CLI argument parsing and validation
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pythagorean expectation record calculator.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  Calculate expected record for MLB team using linear formula:
    pythagorean_record --scored 800 --allowed 650

  Use traditional Pythagorean formula with default exponent 2:
    pythagorean_record --scored 800 --allowed 650 --method pythagorean

  Use custom exponent (e.g., 1.83 for more accurate baseball):
    pythagorean_record --scored 800 --allowed 650 --method pythagorean --exponent 1.83

  Compare both methods side by side:
    pythagorean_record --scored 800 --allowed 650 --method both

  NFL team prediction:
    pythagorean_record --scored 420 --allowed 300 --sport nfl

  NBA team prediction:
    pythagorean_record --scored 8500 --allowed 8200 --sport nba

  Mid-season projection (100 games played):
    pythagorean_record --scored 450 --allowed 380 --games 100

  In-progress season projection with current record:
    pythagorean_record --scored 550 --allowed 490 --current-wins 45 --games-played 82

  In-progress season with current record (team is 45-37 with 550 RS, 490 RA):
    pythagorean_record --scored 550 --allowed 490 --current-wins 45 --games-played 82
    → Shows current record, expected record for games played, and projected final record

For in-progress seasons:
  Use --current-wins and --games-played to compare actual performance vs. expected,
  and project the final season record based on current run differential.
""",
    )

    parser.add_argument(
        "--scored",
        "-s",
        type=float,
        required=True,
        metavar="RUNS",
        help="runs/points scored by the team",
    )
    parser.add_argument(
        "--allowed",
        "-a",
        type=float,
        required=True,
        metavar="RUNS",
        help="runs/points allowed by the team",
    )

    parser.add_argument(
        "--sport",
        choices=["mlb", "nfl", "nba"],
        default="mlb",
        help="sport/league (default: mlb)",
    )

    parser.add_argument(
        "--method",
        "-m",
        choices=["linear", "pythagorean", "both"],
        default="linear",
        help="calculation method: linear (SABR 2014), pythagorean (Bill James), or both (default: linear)",
    )

    parser.add_argument(
        "--exponent",
        "-e",
        type=float,
        default=2.0,
        metavar="EXP",
        help="exponent for Pythagorean formula (default: 2.0; baseball optimal ~1.83)",
    )

    parser.add_argument(
        "--games",
        "-g",
        type=int,
        metavar="N",
        help="games in season (default: 162 for mlb, 17 for nfl, 82 for nba)",
    )

    parser.add_argument(
        "--precision",
        "-P",
        type=int,
        default=2,
        help="decimal places for percentages (default: 2)",
    )

    parser.add_argument(
        "--current-wins",
        "-w",
        type=int,
        metavar="W",
        help="current wins (for in-progress season projection)",
    )

    parser.add_argument(
        "--games-played",
        "-p",
        type=int,
        metavar="GP",
        help="games already played (for in-progress season projection)",
    )

    return parser.parse_args(argv)


def validate(args: argparse.Namespace) -> Optional[str]:
    """Validate command-line arguments."""
    if args.scored < 0:
        return "--scored must be non-negative"

    if args.allowed < 0:
        return "--allowed must be non-negative"

    if args.exponent <= 0:
        return "--exponent must be positive"

    if args.games is not None and args.games <= 0:
        return "--games must be positive"

    if args.precision < 0:
        return "--precision must be non-negative"

    # Validate in-progress season parameters
    if args.current_wins is not None and args.games_played is None:
        return "--games-played is required when --current-wins is specified"

    if args.games_played is not None and args.current_wins is None:
        return "--current-wins is required when --games-played is specified"

    if args.current_wins is not None:
        if args.current_wins < 0:
            return "--current-wins must be non-negative"
        if args.games_played <= 0:
            return "--games-played must be positive"
        if args.current_wins > args.games_played:
            return "--current-wins cannot exceed --games-played"

    # Check games-played doesn't exceed total games
    if args.games_played is not None:
        total_games = (
            args.games if args.games is not None else get_default_games(args.sport)
        )
        if args.games_played > total_games:
            return f"--games-played ({args.games_played}) cannot exceed total games ({total_games})"

    return None


def get_default_games(sport: str) -> int:
    """Return the standard number of games for a sport."""
    defaults = {
        "mlb": 162,
        "nfl": 17,
        "nba": 82,
    }
    return defaults[sport]


def format_result(
    win_pct: float,
    wins: float,
    games: int,
    precision: int,
    method_name: str,
    label: str = "Expected",
) -> str:
    """Format a single result line."""
    pct_fmt = f"{{:.{precision}f}}"
    pct_str = pct_fmt.format(win_pct * 100)
    wins_rounded = round(wins)
    losses = games - wins_rounded
    return f"{method_name:12s}: {pct_str}% ({wins_rounded}-{losses} in {games} games) [{label}]"


def format_output(
    scored: float,
    allowed: float,
    sport: str,
    method: str,
    exponent: float,
    games: int,
    precision: int,
    current_wins: Optional[int] = None,
    games_played: Optional[int] = None,
) -> str:
    """Generate formatted output for the calculation."""
    score_label = "Runs" if sport == "mlb" else "Points"
    diff = scored - allowed

    lines = [
        f"{score_label} Scored:  {scored:.0f}",
        f"{score_label} Allowed: {allowed:.0f}",
        f"Differential: {diff:+.0f}",
    ]

    # Show current record if provided
    if current_wins is not None and games_played is not None:
        current_losses = games_played - current_wins
        current_pct = (current_wins / games_played) * 100 if games_played > 0 else 0
        pct_fmt = f"{{:.{precision}f}}"
        lines.extend(
            [
                "",
                f"Current Record: {current_wins}-{current_losses} ({pct_fmt.format(current_pct)}% in {games_played} games)",
            ]
        )

    lines.append("")

    # Calculate and display expected records
    if method == "linear" or method == "both":
        win_pct_linear = linear_expectation(scored, allowed, sport)

        if current_wins is not None and games_played is not None:
            # Show expected for games played so far
            expected_wins_so_far = expected_wins(win_pct_linear, games_played)
            diff_from_expected = current_wins - expected_wins_so_far
            lines.append(
                format_result(
                    win_pct_linear,
                    expected_wins_so_far,
                    games_played,
                    precision,
                    "Linear",
                    "Expected so far",
                )
            )

            # Show over/under performance
            sign = "+" if diff_from_expected >= 0 else ""
            lines.append(
                f"              ({sign}{diff_from_expected:.1f} wins vs expected)"
            )

            # Project final season record
            remaining_games = games - games_played
            projected_remaining_wins = expected_wins(win_pct_linear, remaining_games)
            projected_total_wins = current_wins + projected_remaining_wins
            lines.append(
                format_result(
                    win_pct_linear,
                    projected_total_wins,
                    games,
                    precision,
                    "Linear",
                    "Projected final",
                )
            )
        else:
            # Standard full-season expectation
            wins_linear = expected_wins(win_pct_linear, games)
            lines.append(
                format_result(
                    win_pct_linear,
                    wins_linear,
                    games,
                    precision,
                    "Linear",
                    "Full season",
                )
            )

    if method == "pythagorean" or method == "both":
        win_pct_pyth = pythagorean_expectation(scored, allowed, exponent)
        exp_label = f"Pyth (^{exponent:.2f})" if exponent != 2.0 else "Pythagorean"

        if current_wins is not None and games_played is not None:
            # Show expected for games played so far
            expected_wins_so_far = expected_wins(win_pct_pyth, games_played)
            diff_from_expected = current_wins - expected_wins_so_far
            lines.append(
                format_result(
                    win_pct_pyth,
                    expected_wins_so_far,
                    games_played,
                    precision,
                    exp_label,
                    "Expected so far",
                )
            )

            # Show over/under performance
            sign = "+" if diff_from_expected >= 0 else ""
            lines.append(
                f"              ({sign}{diff_from_expected:.1f} wins vs expected)"
            )

            # Project final season record
            remaining_games = games - games_played
            projected_remaining_wins = expected_wins(win_pct_pyth, remaining_games)
            projected_total_wins = current_wins + projected_remaining_wins
            lines.append(
                format_result(
                    win_pct_pyth,
                    projected_total_wins,
                    games,
                    precision,
                    exp_label,
                    "Projected final",
                )
            )
        else:
            # Standard full-season expectation
            wins_pyth = expected_wins(win_pct_pyth, games)
            lines.append(
                format_result(
                    win_pct_pyth, wins_pyth, games, precision, exp_label, "Full season"
                )
            )

    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    """Main CLI entry point."""
    args = parse_args(argv)

    error = validate(args)
    if error:
        print(f"Error: {error}", file=sys.stderr)
        return 2

    # Determine games count
    games = args.games if args.games is not None else get_default_games(args.sport)

    try:
        output = format_output(
            args.scored,
            args.allowed,
            args.sport,
            args.method,
            args.exponent,
            games,
            args.precision,
            args.current_wins,
            args.games_played,
        )
        print(output)
        return 0
    except (ValueError, ZeroDivisionError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
