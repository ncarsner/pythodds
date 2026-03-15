"""Tests for Pythagorean expectation functions and CLI."""

from unittest.mock import patch

import pytest

from src.utils.pythagorean_record import (
    linear_expectation,
    main,
    pythagorean_expectation,
)

# ---------------------------------------------------------------------------
# Core function tests
# ---------------------------------------------------------------------------


def test_pythagorean_expectation_edge_cases():
    """Test edge cases for Pythagorean formula."""
    # Both zero should return 0.5
    assert pythagorean_expectation(0, 0) == 0.5

    # Zero allowed should return 1.0 (perfect team)
    assert pythagorean_expectation(100, 0) == 1.0

    # Zero scored should return 0.0 (winless team)
    assert pythagorean_expectation(0, 100) == 0.0


def test_pythagorean_expectation_raises_on_negative():
    """Test that negative values raise ValueError."""
    with pytest.raises(ValueError, match="non-negative"):
        pythagorean_expectation(-100, 50)

    with pytest.raises(ValueError, match="non-negative"):
        pythagorean_expectation(100, -50)


def test_pythagorean_expectation_raises_on_invalid_exponent():
    """Test that invalid exponents raise ValueError."""
    with pytest.raises(ValueError, match="positive"):
        pythagorean_expectation(100, 50, exponent=0)

    with pytest.raises(ValueError, match="positive"):
        pythagorean_expectation(100, 50, exponent=-2)


def test_linear_expectation_raises_on_invalid_sport():
    """Test that invalid sport raises ValueError."""
    with pytest.raises(ValueError, match="sport must be"):
        linear_expectation(100, 50, sport="nhl")


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------


def test_main_basic_linear_mlb(capsys):
    """Test basic MLB linear calculation."""
    rc = main(["--scored", "800", "--allowed", "650"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Runs Scored:  800" in out
    assert "Runs Allowed: 650" in out
    assert "Linear" in out
    assert "60." in out


def test_main_custom_exponent(capsys):
    """Test custom Pythagorean exponent."""
    rc = main(
        [
            "--scored",
            "800",
            "--allowed",
            "650",
            "--method",
            "pythagorean",
            "--exponent",
            "1.83",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "Pyth (^1.83)" in out


def test_main_negative_scored_returns_2(capsys):
    """Test that negative scored returns error code 2."""
    rc = main(["--scored", "-100", "--allowed", "50"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "non-negative" in err


def test_main_negative_allowed_returns_2(capsys):
    """Test that negative allowed returns error code 2."""
    rc = main(["--scored", "100", "--allowed", "-50"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "non-negative" in err


def test_main_invalid_exponent_returns_2(capsys):
    """Test that non-positive exponent returns error code 2."""
    rc = main(["--scored", "100", "--allowed", "50", "--exponent", "-1"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "positive" in err


def test_main_invalid_games_returns_2(capsys):
    """Test that non-positive games returns error code 2."""
    rc = main(["--scored", "100", "--allowed", "50", "--games", "0"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "positive" in err


def test_main_negative_precision_returns_2(capsys):
    """Test that negative precision returns error code 2."""
    rc = main(["--scored", "100", "--allowed", "50", "--precision", "-1"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "non-negative" in err


# ---------------------------------------------------------------------------
# In-progress season projection tests
# ---------------------------------------------------------------------------


def test_main_in_progress_season_both_methods(capsys):
    """Test in-progress season showing both methods."""
    rc = main(
        [
            "--scored",
            "550",
            "--allowed",
            "490",
            "--current-wins",
            "45",
            "--games-played",
            "82",
            "--method",
            "both",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "Linear" in out
    assert "Pythagorean" in out
    assert "Expected" in out
    assert "Projected" in out


def test_main_in_progress_season_nfl(capsys):
    """Test in-progress NFL season."""
    rc = main(
        [
            "--scored",
            "250",
            "--allowed",
            "180",
            "--current-wins",
            "6",
            "--games-played",
            "10",
            "--sport",
            "nfl",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "Current Record: 6-4" in out
    assert "in 10 games" in out
    assert "in 17 games" in out


def test_main_in_progress_season_nba_custom_games(capsys):
    """Test in-progress NBA season with custom total games."""
    rc = main(
        [
            "--scored",
            "5500",
            "--allowed",
            "5200",
            "--current-wins",
            "30",
            "--games-played",
            "50",
            "--sport",
            "nba",
            "--games",
            "82",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "Current Record: 30-20" in out
    assert "in 50 games" in out
    assert "in 82 games" in out


def test_main_current_wins_without_games_played_returns_2(capsys):
    """Test that current-wins without games-played returns error."""
    rc = main(["--scored", "550", "--allowed", "490", "--current-wins", "45"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "--games-played is required" in err


def test_main_games_played_without_current_wins_returns_2(capsys):
    """Test that games-played without current-wins returns error."""
    rc = main(["--scored", "550", "--allowed", "490", "--games-played", "82"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "--current-wins is required" in err


def test_main_current_wins_exceeds_games_played_returns_2(capsys):
    """Test that current-wins exceeding games-played returns error."""
    rc = main(
        [
            "--scored",
            "550",
            "--allowed",
            "490",
            "--current-wins",
            "90",
            "--games-played",
            "82",
        ]
    )
    assert rc == 2
    err = capsys.readouterr().err
    assert "cannot exceed --games-played" in err


def test_main_games_played_exceeds_season_returns_2(capsys):
    """Test that games-played exceeding total games returns error."""
    rc = main(
        [
            "--scored",
            "550",
            "--allowed",
            "490",
            "--current-wins",
            "45",
            "--games-played",
            "200",
        ]
    )
    assert rc == 2
    err = capsys.readouterr().err
    assert "cannot exceed total games" in err


def test_main_negative_current_wins_returns_2(capsys):
    """Test that negative current-wins returns error."""
    rc = main(
        [
            "--scored",
            "550",
            "--allowed",
            "490",
            "--current-wins",
            "-5",
            "--games-played",
            "82",
        ]
    )
    assert rc == 2
    err = capsys.readouterr().err
    assert "non-negative" in err


def test_main_zero_games_played_returns_2(capsys):
    """Test that zero games-played returns error."""
    rc = main(
        [
            "--scored",
            "550",
            "--allowed",
            "490",
            "--current-wins",
            "0",
            "--games-played",
            "0",
        ]
    )
    assert rc == 2
    err = capsys.readouterr().err
    assert "positive" in err


# ---------------------------------------------------------------------------
# Exception handling tests
# ---------------------------------------------------------------------------


def test_main_zero_division_error_exception_returns_2(capsys):
    """Test that ZeroDivisionError in format_output is caught and returns error code 2."""
    with patch("src.utils.pythagorean_record.pythagorean_expectation") as mock_pyth:
        mock_pyth.side_effect = ZeroDivisionError("Division by zero in calculation")
        rc = main(["--scored", "800", "--allowed", "650", "--method", "pythagorean"])
        assert rc == 2
        err = capsys.readouterr().err
        assert "Error: Division by zero in calculation" in err
