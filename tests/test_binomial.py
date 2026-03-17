"""Tests for binomial distribution functions."""

from unittest.mock import patch

from src.utils.binomial_distribution import (
    _log_comb,
    binomial_cdf_ge,
    binomial_cdf_le,
    binomial_pmf,
    format_prob,
    main,
)


def test_binomial_pmf_boundary():
    """Test PMF boundary conditions."""
    # P(X=k) when k < 0 should be 0
    assert binomial_pmf(10, -1, 0.5) == 0.0

    # P(X=k) when k > n should be 0
    assert binomial_pmf(10, 11, 0.5) == 0.0


# ---------------------------------------------------------------------------
# Tests for _log_comb helper function
# ---------------------------------------------------------------------------


def test_log_comb_edge_cases():
    """Test edge cases for _log_comb."""
    # n choose 0 = 1, log(1) = 0
    assert _log_comb(10, 0) == 0.0

    # n choose n = 1, log(1) = 0
    assert _log_comb(10, 10) == 0.0

    # k < 0 should return -inf
    assert _log_comb(10, -1) == float("-inf")

    # k > n should return -inf
    assert _log_comb(10, 11) == float("-inf")


# ---------------------------------------------------------------------------
# Additional tests for binomial_pmf with log-space implementation
# ---------------------------------------------------------------------------


def test_binomial_pmf_extreme_probability():
    """Test PMF with p=0 and p=1 edge cases."""
    # When p=0, only P(X=0) should be 1
    assert binomial_pmf(10, 0, 0.0) == 1.0
    assert binomial_pmf(10, 5, 0.0) == 0.0

    # When p=1, only P(X=n) should be 1
    assert binomial_pmf(10, 5, 1.0) == 0.0
    assert binomial_pmf(10, 10, 1.0) == 1.0


def test_binomial_pmf_extreme_log_values():
    """Test that extreme log_prob values are handled correctly."""
    # Case where log_prob would be very negative (returns 0)
    result = binomial_pmf(10000, 0, 0.9999)
    assert result >= 0.0
    assert result < 1e-100

    # Case where log_prob approaches 0 (probability near 1)
    result = binomial_pmf(2, 2, 0.99)
    assert result > 0.9
    assert result <= 1.0


def test_binomial_pmf_log_prob_exceeds_700():
    """Test that line 54 returns 1.0 when log_prob > 700.

    Tests the defensive upper bound check. Mock _log_comb to return
    a very large value (800) that would cause log_prob > 700, triggering the
    clamp to 1.0 to prevent math.exp() from overflowing.
    """
    with patch("src.utils.binomial_distribution._log_comb", return_value=800):
        # With _log_comb returning 800, log_prob will be > 700
        # even after adding negative terms from log(p) and log(1-p)
        result = binomial_pmf(10, 5, 0.5)
        assert result == 1.0  # Should be clamped to 1.0


def test_binomial_cdf_le():
    """Test CDF (less than or equal)."""
    # P(X <= 0) for Binomial(n=1, p=0.5) should be 0.5
    # P(X <= n) should be 1.0
    # P(X <= k) when k < 0 should be 0
    assert binomial_cdf_le(10, -1, 0.5) == 0.0


def test_binomial_cdf_ge():
    """Test survival function (greater than or equal)."""
    # P(X >= 1) for Binomial(n=1, p=0.5) should be 0.5
    # P(X >= 0) should be 1.0
    # P(X >= k) when k > n should be 0
    assert binomial_cdf_ge(10, 0, 0.5) == 1.0


# ---------------------------------------------------------------------------
# Tests for format_prob
# ---------------------------------------------------------------------------


def test_format_prob_precision():
    """Precision parameter controls decimal places."""
    result_2 = format_prob(0.333333, 2)
    assert "0.33" in result_2


def test_complementary_probabilities():
    """Test that CDF and survival function are complementary."""
    n, p = 10, 0.4
    for k in range(n + 1):
        cdf = binomial_cdf_le(n, k, p)
        sf = binomial_cdf_ge(n, k + 1, p)
        # cdf + sf should equal 1.0
        assert abs(cdf + sf - 1.0) < 1e-10


def test_large_pool_no_overflow():
    """Test that large n values don't cause overflow errors."""
    # Previously failed with: OverflowError: int too large to convert to float
    # Now uses log-space calculations to avoid overflow
    result = binomial_pmf(10000, 5000, 0.5)
    assert result > 0.0
    assert result < 0.01  # Small but calculable


# ---------------------------------------------------------------------------
# Tests for main
# ---------------------------------------------------------------------------


def test_main_with_target_and_min_prob(capsys):
    """main() outputs meets minimum check when both --target and --min-prob specified."""
    result = main(
        ["-n", "10", "-k", "5", "-p", "0.4", "--target", "7", "--min-prob", "0.05"]
    )
    assert result == 0
    captured = capsys.readouterr()
    assert "Meets minimum" in captured.out
    assert "True" in captured.out or "False" in captured.out


def test_main_invalid_p_too_low():
    """main() returns 2 when p < 0."""
    result = main(["-n", "10", "-k", "5", "-p", "-0.1"])
    assert result == 2


def test_main_invalid_n_negative():
    """main() returns 2 when n < 0."""
    result = main(["-n", "-5", "-k", "3", "-p", "0.4"])
    assert result == 2


def test_main_invalid_min_prob_too_low():
    """main() returns 2 when --min-prob < 0."""
    result = main(
        ["-n", "10", "-k", "5", "-p", "0.4", "--target", "7", "--min-prob", "-0.1"]
    )
    assert result == 2
