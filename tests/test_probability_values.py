"""
Tests for probability_values.py (pvalue CLI tool)

Tests cover:
- Z-test for proportions
- Binomial exact test
- Chi-squared goodness-of-fit test
- t-test for means
- Helper functions (normal_cdf, quantiles, etc.)
- CLI argument validation
- Sweep mode
- main() function
"""

import io
import sys
from unittest.mock import patch

import pytest

from src.utils.probability_values import (
    binomial_cdf_le,
    binomial_exact_test,
    binomial_pmf,
    chi2_cdf_approx,
    chi2_goodness_of_fit,
    erfinv_approx,
    main,
    normal_quantile,
    parse_args,
    t_quantile_approx,
    t_test_mean,
    validate_args,
    z_test_proportion,
)

# --- Test helper functions ---


@pytest.mark.parametrize(
    "x", [1.0, -1.0, 1.5, -1.5], ids=["x=1.0", "x=-1.0", "x=1.5", "x=-1.5"]
)
def test_erfinv_approx_invalid(x):
    """Test erfinv_approx raises error for |x| >= 1."""
    with pytest.raises(ValueError, match="erfinv input must be in"):
        erfinv_approx(x)


@pytest.mark.parametrize(
    "p,expected",
    [
        (0.000001, -6.0),
        (0.999999, 6.0),
    ],
    ids=["very_low", "very_high"],
)
def test_normal_quantile_extreme(p, expected):
    """Test normal quantile with extreme values."""
    assert normal_quantile(p) == expected


@pytest.mark.parametrize("p", [0.0, 1.0, -0.5, 1.5], ids=["p=0", "p=1", "p<0", "p>1"])
def test_normal_quantile_invalid(p):
    """Test normal quantile raises error for invalid probabilities."""
    with pytest.raises(ValueError, match="Probability must be in"):
        normal_quantile(p)


def test_normal_quantile_fallback():
    """Test normal quantile fallback when erfinv raises error."""
    with patch(
        "src.utils.probability_values.erfinv_approx", side_effect=ValueError("Mock")
    ):
        assert normal_quantile(0.3) == -5.0
        assert normal_quantile(0.7) == 5.0


def test_t_quantile_large_df():
    """Test t-quantile converges to normal for large df."""
    z = normal_quantile(0.975)
    t = t_quantile_approx(0.975, 2000)
    assert abs(t - z) < 0.05


@pytest.mark.parametrize(
    "df,p",
    [
        (10, 0.975),
        (30, 0.95),
        (100, 0.99),
        (5, 0.75),
    ],
    ids=["df=10", "df=30", "df=100", "df=5"],
)
def test_t_quantile_small_df(df, p):
    """Test t-quantile using Hill's approximation for small/medium df."""
    # Test that Hill's approximation produces reasonable values
    t = t_quantile_approx(p, df)
    # t-quantile should be larger than z-quantile for same p (heavier tails)
    z = normal_quantile(p)
    if p > 0.5:
        assert t >= z  # For upper tail
    else:
        assert t <= z  # For lower tail
    # Result should be finite and reasonable
    assert abs(t) < 100


@pytest.mark.parametrize("df", [0, -1], ids=["df=0", "df=-1"])
def test_t_quantile_invalid_df(df):
    """Test t-quantile raises error for invalid df."""
    with pytest.raises(ValueError, match="Degrees of freedom must be positive"):
        t_quantile_approx(0.5, df)


def test_chi2_cdf_invalid_df():
    """Test chi-squared CDF raises error for invalid df."""
    with pytest.raises(ValueError, match="Degrees of freedom must be positive"):
        chi2_cdf_approx(1.0, 0)


# --- Test binomial functions ---


@pytest.mark.parametrize(
    "n,k,p,expected",
    [
        (10, -1, 0.5, 0.0),
        (10, 11, 0.5, 0.0),
        (10, 0, 0.0, 1.0),
        (10, 5, 0.0, 0.0),
        (10, 10, 1.0, 1.0),
        (10, 5, 1.0, 0.0),
    ],
    ids=["k<0", "k>n", "p=0_k=0", "p=0_k>0", "p=1_k=n", "p=1_k<n"],
)
def test_binomial_pmf_edge_cases(n, k, p, expected):
    """Test binomial PMF edge cases."""
    assert binomial_pmf(n, k, p) == expected


def test_binomial_pmf_log_prob_bounds():
    """Test binomial PMF handles edge cases in log probability calculation."""
    # Test that extremely small probabilities clamp to 0.0
    result = binomial_pmf(5000, 4999, 0.0001)
    assert result == 0.0

    # Test reasonable probability remains in valid range
    result = binomial_pmf(100, 50, 0.5)
    assert 0 < result < 1


def test_binomial_pmf_extreme_overflow():
    """Test binomial PMF handles the defensive log_prob > 700 case."""
    # Use mocking to simulate a numerical error that produces log_prob > 700

    # Patch lgamma to return values that would make log_prob > 700
    with patch("src.utils.probability_values.math.lgamma") as mock_lgamma:
        # Set up lgamma to return values that create log_prob > 700
        # lgamma(n+1) - lgamma(k+1) - lgamma(n-k+1) could theoretically be large
        mock_lgamma.side_effect = lambda x: 1000 if x == 11 else 0

        # This should trigger the log_prob > 700 branch and return 1.0
        result = binomial_pmf(10, 5, 0.5)
        assert result == 1.0


def test_binomial_cdf_le():
    """Test binomial CDF."""
    # P(X <= 0) when p=0 should be 1
    assert binomial_cdf_le(10, 0, 0.0) == 1.0

    # P(X <= k) where k >= n should be 1
    assert binomial_cdf_le(10, 10, 0.5) == 1.0

    # P(X <= k) where k < 0 should be 0
    assert binomial_cdf_le(10, -1, 0.5) == 0.0


# --- Test z-test ---


def test_z_test_proportion_one_sided():
    """Test one-sided z-tests."""
    # Test "less"
    z_stat_less, p_value_less = z_test_proportion(1000, 480, 0.5, "less")
    assert p_value_less < 0.5

    # Test "greater"
    z_stat_greater, p_value_greater = z_test_proportion(1000, 480, 0.5, "greater")
    assert p_value_greater > 0.5


@pytest.mark.parametrize(
    "n,k,p0,match",
    [
        (0, 5, 0.5, "Sample size must be positive"),
        (-10, 5, 0.5, "Sample size must be positive"),
        (10, -1, 0.5, "Successes k must be in"),
        (10, 15, 0.5, "Successes k must be in"),
        (10, 5, 0.0, "Null proportion must be in"),
        (10, 5, 1.0, "Null proportion must be in"),
        (10, 5, 1.5, "Null proportion must be in"),
    ],
    ids=["n=0", "n<0", "k<0", "k>n", "p0=0", "p0=1", "p0>1"],
)
def test_z_test_invalid_inputs(n, k, p0, match):
    """Test z-test raises errors for invalid inputs."""
    with pytest.raises(ValueError, match=match):
        z_test_proportion(n, k, p0)


def test_z_test_invalid_sided():
    """Test z-test raises error for invalid sided argument."""
    with pytest.raises(ValueError, match="sided must be"):
        z_test_proportion(10, 5, 0.5, "invalid")


def test_z_test_zero_se():
    """Test z-test handles zero standard error (defensive coding for se == 0)."""
    # Use a targeted mock that only affects the SE calculation
    original_sqrt = __import__("math").sqrt

    def selective_mock_sqrt(x):
        # If x is the SE calculation result (p0 * (1-p0) / n)
        # For our test cases, this will be 0.25/100 = 0.0025 or similar small values
        if 0.0001 < x < 0.01:  # SE calculation range
            return 0.0  # Force se = 0
        else:
            return original_sqrt(x)  # Normal sqrt for other calls

    with patch(
        "src.utils.probability_values.math.sqrt", side_effect=selective_mock_sqrt
    ):
        # Test case 1: p_hat != p0, should return inf
        z_stat, p_value = z_test_proportion(100, 60, 0.5, "two")
        # p_hat = 0.6, p0 = 0.5, they're different, so z_stat should be inf
        assert z_stat == float("inf")

    with patch(
        "src.utils.probability_values.math.sqrt", side_effect=selective_mock_sqrt
    ):
        # Test case 2: p_hat == p0 with se == 0, should return 0.0
        # k/n = 50/100 = 0.5 = p0, so p_hat == p0
        z_stat, p_value = z_test_proportion(100, 50, 0.5, "two")
        # When se == 0 and p_hat == p0, z_stat should be 0.0
        assert z_stat == 0.0


# --- Test binomial exact test ---


def test_binomial_exact_test_one_sided():
    """Test one-sided binomial exact tests."""
    # Test "less"
    stat_less, p_value_less = binomial_exact_test(30, 15, 0.6, "less")
    assert p_value_less < 0.5

    # Test "greater"
    stat_greater, p_value_greater = binomial_exact_test(30, 25, 0.6, "greater")
    assert p_value_greater < 0.5


@pytest.mark.parametrize(
    "n,k,p0,match",
    [
        (0, 5, 0.5, "Sample size must be positive"),
        (10, -1, 0.5, "Successes k must be in"),
        (10, 15, 0.5, "Successes k must be in"),
        (10, 5, 0.0, "Null proportion must be in"),
    ],
    ids=["n=0", "k<0", "k>n", "p0=0"],
)
def test_binomial_exact_invalid_inputs(n, k, p0, match):
    """Test binomial exact test raises errors for invalid inputs."""
    with pytest.raises(ValueError, match=match):
        binomial_exact_test(n, k, p0)


def test_binomial_exact_invalid_sided():
    """Test binomial exact test raises error for invalid sided argument."""
    with pytest.raises(ValueError, match="sided must be"):
        binomial_exact_test(10, 5, 0.5, "invalid")


# --- Test chi-squared test ---


def test_chi2_goodness_of_fit_perfect():
    """Test chi-squared when observed equals expected."""
    observed = [20, 20, 20, 20, 20]
    expected = [20, 20, 20, 20, 20]

    chi2_stat, p_value = chi2_goodness_of_fit(list(observed), list(expected))

    assert chi2_stat == 0.0
    assert p_value > 0.99


@pytest.mark.parametrize(
    "observed,expected,match",
    [
        ([10, 20], [10, 20, 30], "same length"),
        ([10], [10], "at least 2 categories"),
        ([10, 20], [0, 20], "must be positive"),
        ([-5, 20], [10, 20], "must be non-negative"),
    ],
    ids=["length_mismatch", "too_few", "expected_zero", "observed_negative"],
)
def test_chi2_invalid_inputs(observed, expected, match):
    """Test chi-squared raises errors for invalid inputs."""
    with pytest.raises(ValueError, match=match):
        chi2_goodness_of_fit(observed, expected)


# --- Test t-test ---


def test_t_test_mean_basic():
    """Test one-sample t-test."""
    # Sample mean=105, std=12, n=25, testing against mu0=100
    t_stat, p_value = t_test_mean(105, 12, 25, 100, "two")

    assert t_stat > 0  # Positive because mean > mu0
    assert 0 < p_value < 1


def test_t_test_mean_one_sided():
    """Test one-sided t-tests."""
    # Test "less"
    t_stat_less, p_value_less = t_test_mean(95, 12, 25, 100, "less")
    assert p_value_less < 0.5

    # Test "greater"
    t_stat_greater, p_value_greater = t_test_mean(105, 12, 25, 100, "greater")
    assert p_value_greater < 0.5


@pytest.mark.parametrize(
    "mean,std,n,mu0,match",
    [
        (100, 10, 1, 100, "Sample size must be > 1"),
        (100, 10, 0, 100, "Sample size must be > 1"),
        (100, -5, 10, 100, "Standard deviation must be non-negative"),
    ],
    ids=["n=1", "n=0", "std<0"],
)
def test_t_test_invalid_inputs(mean, std, n, mu0, match):
    """Test t-test raises errors for invalid inputs."""
    with pytest.raises(ValueError, match=match):
        t_test_mean(mean, std, n, mu0)


def test_t_test_invalid_sided():
    """Test t-test raises error for invalid sided argument."""
    with pytest.raises(ValueError, match="sided must be"):
        t_test_mean(105, 12, 25, 100, "invalid")


def test_t_test_zero_std():
    """Test t-test handles zero standard deviation."""
    # When std=0 and mean==mu0, p-value should be 1.0
    t_stat, p_value = t_test_mean(100, 0, 10, 100, "two")
    assert t_stat == 0.0
    assert p_value == 1.0

    # When std=0 and mean!=mu0, p-value should be 0.0
    t_stat2, p_value2 = t_test_mean(105, 0, 10, 100, "two")
    assert t_stat2 == float("inf")
    assert p_value2 == 0.0


# --- Test argument validation ---


def test_validate_args_z_test_invalid_n():
    """Test validation rejects invalid n."""
    original_argv = sys.argv
    original_stdout = sys.stdout
    try:
        sys.argv = ["pvalue", "--test", "z", "--n", "0", "--k", "5", "--p0", "0.5"]
        sys.stdout = io.StringIO()
        args = parse_args()
        result = validate_args(args)
        sys.stdout = original_stdout
        assert result == 2
    finally:
        sys.argv = original_argv
        sys.stdout = original_stdout


def test_validate_args_z_test_invalid_k():
    """Test validation rejects invalid k."""
    original_argv = sys.argv
    original_stdout = sys.stdout
    try:
        sys.argv = ["pvalue", "--test", "z", "--n", "10", "--k", "15", "--p0", "0.5"]
        sys.stdout = io.StringIO()
        args = parse_args()
        result = validate_args(args)
        sys.stdout = original_stdout
        assert result == 2
    finally:
        sys.argv = original_argv
        sys.stdout = original_stdout


def test_validate_args_z_test_invalid_p0():
    """Test validation rejects invalid p0."""
    original_argv = sys.argv
    original_stdout = sys.stdout
    try:
        sys.argv = ["pvalue", "--test", "z", "--n", "10", "--k", "5", "--p0", "1.5"]
        sys.stdout = io.StringIO()
        args = parse_args()
        result = validate_args(args)
        sys.stdout = original_stdout
        assert result == 2
    finally:
        sys.argv = original_argv
        sys.stdout = original_stdout


# def test_validate_args_t_test_valid():
#     """Test validation accepts valid t-test arguments."""
#     original_argv = sys.argv
#     try:
#         sys.argv = ["pvalue", "--test", "t", "--mean", "105", "--std", "12", "--n", "25", "--mu0", "100"]
#         args = parse_args()
#         assert validate_args(args) == 0
#     finally:
#         sys.argv = original_argv


def test_validate_args_t_test_missing():
    """Test validation rejects t-test with missing parameters."""
    original_argv = sys.argv
    original_stdout = sys.stdout
    try:
        sys.argv = ["pvalue", "--test", "t", "--mean", "105", "--std", "12"]
        sys.stdout = io.StringIO()
        args = parse_args()
        result = validate_args(args)
        sys.stdout = original_stdout
        assert result == 2
    finally:
        sys.argv = original_argv
        sys.stdout = original_stdout


def test_validate_args_t_test_invalid_n():
    """Test validation rejects t-test with n <= 1."""
    original_argv = sys.argv
    original_stdout = sys.stdout
    try:
        sys.argv = [
            "pvalue",
            "--test",
            "t",
            "--mean",
            "105",
            "--std",
            "12",
            "--n",
            "1",
            "--mu0",
            "100",
        ]
        sys.stdout = io.StringIO()
        args = parse_args()
        result = validate_args(args)
        sys.stdout = original_stdout
        assert result == 2
    finally:
        sys.argv = original_argv
        sys.stdout = original_stdout


def test_validate_args_t_test_invalid_std():
    """Test validation rejects negative std."""
    original_argv = sys.argv
    original_stdout = sys.stdout
    try:
        sys.argv = [
            "pvalue",
            "--test",
            "t",
            "--mean",
            "105",
            "--std",
            "-12",
            "--n",
            "25",
            "--mu0",
            "100",
        ]
        sys.stdout = io.StringIO()
        args = parse_args()
        result = validate_args(args)
        sys.stdout = original_stdout
        assert result == 2
    finally:
        sys.argv = original_argv
        sys.stdout = original_stdout


def test_validate_args_chi2_missing():
    """Test validation rejects chi2 with missing parameters."""
    original_argv = sys.argv
    original_stdout = sys.stdout
    try:
        sys.argv = ["pvalue", "--test", "chi2", "--observed", "18,22,20"]
        sys.stdout = io.StringIO()
        args = parse_args()
        result = validate_args(args)
        sys.stdout = original_stdout
        assert result == 2
    finally:
        sys.argv = original_argv
        sys.stdout = original_stdout


def test_validate_args_invalid_alpha():
    """Test validation rejects invalid alpha."""
    original_argv = sys.argv
    original_stdout = sys.stdout
    try:
        sys.argv = [
            "pvalue",
            "--test",
            "z",
            "--n",
            "10",
            "--k",
            "5",
            "--p0",
            "0.5",
            "--alpha",
            "1.5",
        ]
        sys.stdout = io.StringIO()
        args = parse_args()
        result = validate_args(args)
        sys.stdout = original_stdout
        assert result == 2
    finally:
        sys.argv = original_argv
        sys.stdout = original_stdout


def test_validate_args_sweep_invalid():
    """Test validation rejects invalid sweep-alpha."""
    original_argv = sys.argv
    original_stdout = sys.stdout
    try:
        sys.argv = [
            "pvalue",
            "--test",
            "z",
            "--n",
            "10",
            "--k",
            "5",
            "--p0",
            "0.5",
            "--sweep-alpha",
            "0.10",
            "0.01",
        ]
        sys.stdout = io.StringIO()
        args = parse_args()
        result = validate_args(args)
        sys.stdout = original_stdout
        assert result == 2
    finally:
        sys.argv = original_argv
        sys.stdout = original_stdout


def test_validate_args_sweep_invalid_values():
    """Test validation rejects out-of-range sweep-alpha values."""
    original_argv = sys.argv
    original_stdout = sys.stdout
    try:
        sys.argv = [
            "pvalue",
            "--test",
            "z",
            "--n",
            "10",
            "--k",
            "5",
            "--p0",
            "0.5",
            "--sweep-alpha",
            "0.01",
            "1.5",
        ]
        sys.stdout = io.StringIO()
        args = parse_args()
        result = validate_args(args)
        sys.stdout = original_stdout
        assert result == 2
    finally:
        sys.argv = original_argv
        sys.stdout = original_stdout


def test_validate_args_invalid_step():
    """Test validation rejects invalid step."""
    original_argv = sys.argv
    original_stdout = sys.stdout
    try:
        sys.argv = [
            "pvalue",
            "--test",
            "z",
            "--n",
            "10",
            "--k",
            "5",
            "--p0",
            "0.5",
            "--sweep-alpha",
            "0.01",
            "0.10",
            "--step",
            "-0.01",
        ]
        sys.stdout = io.StringIO()
        args = parse_args()
        result = validate_args(args)
        sys.stdout = original_stdout
        assert result == 2
    finally:
        sys.argv = original_argv
        sys.stdout = original_stdout


# --- Test main function ---


def test_main_binom_exact():
    """Test main() with binomial exact test."""
    original_argv = sys.argv
    original_stdout = sys.stdout

    try:
        sys.argv = [
            "pvalue",
            "--test",
            "binom-exact",
            "--n",
            "30",
            "--k",
            "22",
            "--p0",
            "0.60",
        ]
        sys.stdout = io.StringIO()

        result = main()
        output = sys.stdout.getvalue()

        sys.stdout = original_stdout

        assert result == 0
        assert "Binomial Exact Test" in output
        assert "p-value:" in output
    finally:
        sys.argv = original_argv
        sys.stdout = original_stdout


def test_main_t_test():
    """Test main() with t-test."""
    original_argv = sys.argv
    original_stdout = sys.stdout

    try:
        sys.argv = [
            "pvalue",
            "--test",
            "t",
            "--mean",
            "105",
            "--std",
            "12",
            "--n",
            "25",
            "--mu0",
            "100",
        ]
        sys.stdout = io.StringIO()

        result = main()
        output = sys.stdout.getvalue()

        sys.stdout = original_stdout

        assert result == 0
        assert "t-test" in output
        assert "p-value:" in output
        assert "Effect size:" in output
    finally:
        sys.argv = original_argv
        sys.stdout = original_stdout


def test_main_chi2():
    """Test main() with chi-squared test."""
    original_argv = sys.argv
    original_stdout = sys.stdout

    try:
        sys.argv = [
            "pvalue",
            "--test",
            "chi2",
            "--observed",
            "18,22,20,15,25",
            "--expected",
            "20,20,20,20,20",
        ]
        sys.stdout = io.StringIO()

        result = main()
        output = sys.stdout.getvalue()

        sys.stdout = original_stdout

        assert result == 0
        assert "Chi-squared" in output
        assert "p-value:" in output
        assert "df = 4" in output
    finally:
        sys.argv = original_argv
        sys.stdout = original_stdout


def test_main_one_sided_less():
    """Test main() with one-sided test (less)."""
    original_argv = sys.argv
    original_stdout = sys.stdout

    try:
        sys.argv = [
            "pvalue",
            "--test",
            "z",
            "--n",
            "1000",
            "--k",
            "480",
            "--p0",
            "0.5",
            "--sided",
            "less",
        ]
        sys.stdout = io.StringIO()

        result = main()
        output = sys.stdout.getvalue()

        sys.stdout = original_stdout

        assert result == 0
        assert "one-sided (less)" in output
    finally:
        sys.argv = original_argv
        sys.stdout = original_stdout


def test_main_sweep_alpha():
    """Test main() with sweep-alpha mode."""
    original_argv = sys.argv
    original_stdout = sys.stdout

    try:
        sys.argv = [
            "pvalue",
            "--test",
            "z",
            "--n",
            "1000",
            "--k",
            "480",
            "--p0",
            "0.5",
            "--sweep-alpha",
            "0.01",
            "0.10",
            "--step",
            "0.03",
        ]
        sys.stdout = io.StringIO()

        result = main()
        output = sys.stdout.getvalue()

        sys.stdout = original_stdout

        assert result == 0
        assert "Alpha Sweep" in output
        assert "Alpha" in output
        assert "Reject H0" in output
        # Should have multiple alpha values
        lines = output.split("\n")
        data_lines = [
            line for line in lines if line.strip() and line.strip()[0].isdigit()
        ]
        assert len(data_lines) >= 3
    finally:
        sys.argv = original_argv
        sys.stdout = original_stdout


def test_main_validation_error():
    """Test main() returns error code for invalid arguments."""
    original_argv = sys.argv
    original_stdout = sys.stdout

    try:
        sys.argv = ["pvalue", "--test", "z", "--n", "1000"]  # Missing --k and --p0
        sys.stdout = io.StringIO()

        result = main()

        sys.stdout = original_stdout

        assert result == 2
    finally:
        sys.argv = original_argv
        sys.stdout = original_stdout


def test_main_chi2_parsing_error():
    """Test main() handles chi2 parsing errors."""
    original_argv = sys.argv
    original_stdout = sys.stdout

    try:
        sys.argv = [
            "pvalue",
            "--test",
            "chi2",
            "--observed",
            "18,22",
            "--expected",
            "20,20,20",
        ]
        sys.stdout = io.StringIO()

        result = main()
        output = sys.stdout.getvalue()

        sys.stdout = original_stdout

        assert result == 2
        assert "Error:" in output
    finally:
        sys.argv = original_argv
        sys.stdout = original_stdout
