"""Tests for streak probability functions."""

import pytest

from src.utils.streak_probability import (
    expected_longest_streak,
    main,
    prob_at_least_one_streak,
)

# ---------------------------------------------------------------------------
# prob_at_least_one_streak — edge cases
# ---------------------------------------------------------------------------


def test_streak_p_zero_returns_zero():
    """No successes means no streaks."""
    assert prob_at_least_one_streak(100, 5, 0.0) == 0.0


def test_streak_p_one_k_lte_n_returns_one():
    """Certain success and k <= n guarantees a streak."""
    assert prob_at_least_one_streak(10, 5, 1.0) == 1.0


def test_streak_k_gt_n_returns_zero():
    """k longer than n: streak impossible regardless of p."""
    assert prob_at_least_one_streak(4, 10, 0.5) == 0.0


def test_streak_invalid_n_raises():
    with pytest.raises(ValueError):
        prob_at_least_one_streak(0, 3, 0.5)


# ---------------------------------------------------------------------------
# expected_longest_streak — correctness
# ---------------------------------------------------------------------------


def test_expected_longest_p_zero():
    """No successes means longest streak is 0."""
    assert expected_longest_streak(100, 0.0) == 0.0


def test_expected_longest_p_one():
    """Certain success means longest streak equals n."""
    assert expected_longest_streak(10, 1.0) == 10.0


def test_expected_longest_invalid_n_raises():
    with pytest.raises(ValueError):
        expected_longest_streak(0, 0.5)


def test_expected_longest_increases_with_n():
    """Longer sequences should yield longer expected streaks."""
    e50 = expected_longest_streak(50, 0.3)
    e100 = expected_longest_streak(100, 0.3)
    assert e100 > e50


# ---------------------------------------------------------------------------
# main — streak-length mode
# ---------------------------------------------------------------------------


def test_main_streak_length_output_contains_probability(capsys):
    main(["-n", "3", "-k", "2", "-p", "0.5"])
    out = capsys.readouterr().out
    # P(at least one streak ≥ 2) = 0.375 → 37.5%
    assert "37.5" in out


# ---------------------------------------------------------------------------
# main — longest mode
# ---------------------------------------------------------------------------


def test_main_longest_mode_returns_zero():
    rc = main(["-n", "162", "-p", "0.300", "--longest"])
    assert rc == 0


# ---------------------------------------------------------------------------
# main — validation / error handling
# ---------------------------------------------------------------------------


def test_main_invalid_n_returns_2(capsys):
    rc = main(["-n", "0", "-k", "3", "-p", "0.5"])
    assert rc == 2


def test_main_invalid_k_returns_2(capsys):
    rc = main(["-n", "10", "-k", "0", "-p", "0.5"])
    assert rc == 2


def test_main_invalid_p_error_message(capsys):
    main(["-n", "10", "-k", "3", "-p", "-0.1"])
    err = capsys.readouterr().err
    assert "Error" in err
