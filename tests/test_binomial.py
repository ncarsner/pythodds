"""Tests for binomial distribution functions."""

from src.utils.binomial_distribution import (
    binomial_pmf,
    binomial_cdf_le,
    binomial_cdf_ge,
    format_prob,
    main,
    parse_args,
)


def test_binomial_pmf_basic():
    """Test PMF with simple values."""
    # P(X=0) for Binomial(n=1, p=0.5) should be 0.5
    result = binomial_pmf(1, 0, 0.5)
    assert abs(result - 0.5) < 1e-10

    # P(X=1) for Binomial(n=1, p=0.5) should be 0.5
    result = binomial_pmf(1, 1, 0.5)
    assert abs(result - 0.5) < 1e-10


def test_binomial_pmf_boundary():
    """Test PMF boundary conditions."""
    # P(X=k) when k < 0 should be 0
    assert binomial_pmf(10, -1, 0.5) == 0.0

    # P(X=k) when k > n should be 0
    assert binomial_pmf(10, 11, 0.5) == 0.0


def test_binomial_cdf_le():
    """Test CDF (less than or equal)."""
    # P(X <= 0) for Binomial(n=1, p=0.5) should be 0.5
    result = binomial_cdf_le(1, 0, 0.5)
    assert abs(result - 0.5) < 1e-10

    # P(X <= n) should be 1.0
    assert binomial_cdf_le(10, 10, 0.5) == 1.0

    # P(X <= k) when k < 0 should be 0
    assert binomial_cdf_le(10, -1, 0.5) == 0.0


def test_binomial_cdf_ge():
    """Test survival function (greater than or equal)."""
    # P(X >= 1) for Binomial(n=1, p=0.5) should be 0.5
    result = binomial_cdf_ge(1, 1, 0.5)
    assert abs(result - 0.5) < 1e-10

    # P(X >= 0) should be 1.0
    assert binomial_cdf_ge(10, 0, 0.5) == 1.0

    # P(X >= k) when k > n should be 0
    assert binomial_cdf_ge(10, 11, 0.5) == 0.0


def test_parse_args_all_options():
    """Parse all arguments together."""
    args = parse_args(
        [
            "-n",
            "100",
            "-k",
            "30",
            "-p",
            "0.35",
            "--target",
            "40",
            "--min-prob",
            "0.10",
            "--precision",
            "4",
        ]
    )
    assert args.trials == 100
    assert args.successes == 30
    assert args.p == 0.35
    assert args.target == 40
    assert args.min_prob == 0.10
    assert args.precision == 4


# ---------------------------------------------------------------------------
# Tests for format_prob
# ---------------------------------------------------------------------------


def test_format_prob_precision():
    """Precision parameter controls decimal places."""
    result_2 = format_prob(0.333333, 2)
    result_6 = format_prob(0.333333, 6)
    assert "0.33" in result_2
    assert "33.33%" in result_2
    assert "0.333333" in result_6
    assert "33.333300%" in result_6


def test_complementary_probabilities():
    """Test that CDF and survival function are complementary."""
    n, p = 10, 0.4
    for k in range(n + 1):
        cdf = binomial_cdf_le(n, k, p)
        sf = binomial_cdf_ge(n, k + 1, p)
        # cdf + sf should equal 1.0
        assert abs(cdf + sf - 1.0) < 1e-10


# ---------------------------------------------------------------------------
# Tests for main
# ---------------------------------------------------------------------------


def test_main_basic_output(capsys):
    """main() outputs basic probabilities."""
    result = main(["-n", "10", "-k", "5", "-p", "0.4"])
    assert result == 0
    captured = capsys.readouterr()
    assert "n=10, k=5, p=0.4" in captured.out
    assert "P(X = 5):" in captured.out
    assert "P(X <= 5):" in captured.out
    assert "P(X >= 5):" in captured.out


def test_main_with_target(capsys):
    """main() outputs target probability when --target specified."""
    result = main(["-n", "10", "-k", "5", "-p", "0.4", "--target", "7"])
    assert result == 0
    captured = capsys.readouterr()
    assert "P(X >= 7):" in captured.out


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
