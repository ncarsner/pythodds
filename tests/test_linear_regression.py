"""Tests for linear regression functions and CLI."""

import math
import os
import tempfile

import pytest
from scipy import special, stats

import src.utils.linear_regression as linreg_module
from src.utils.linear_regression import (
    f_cdf,
    incomplete_beta,
    interpret_r_squared,
    inverse_t_cdf,
    linear_regression,
    main,
    mean,
    p_value_t,
    parse_csv_file,
    sum_of_squares,
    t_cdf,
)


def test_mean_empty_raises():
    """Test that mean of empty list raises ValueError."""
    with pytest.raises(ValueError):
        mean([])


def test_linear_regression_mismatched_lengths():
    """Test that mismatched x and y lengths raise ValueError."""
    x = [1, 2, 3]
    y = [2, 4]
    with pytest.raises(ValueError, match="same length"):
        linear_regression(x, y)


def test_linear_regression_too_few_points():
    """Test that fewer than 3 points raises ValueError."""
    x = [1, 2]
    y = [2, 4]
    with pytest.raises(ValueError, match="at least 3"):
        linear_regression(x, y)


def test_main_simple_case(capsys):
    """Test main with simple command-line arguments."""
    rc = main(["--x", "1,2,3,4,5", "--y", "2,4,6,8,10"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "LINEAR REGRESSION RESULTS" in out
    assert "Slope:" in out
    assert "Intercept:" in out
    assert "R²" in out
    assert "2.000000" in out  # slope should be 2


def test_main_with_prediction(capsys):
    """Test main with prediction."""
    rc = main(["--x", "1,2,3,4,5", "--y", "2,4,6,8,10", "--predict", "6"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Prediction at x = 6" in out
    assert "Predicted y:" in out
    assert "Confidence interval" in out
    assert "Prediction interval" in out


def test_main_too_few_points_returns_2(capsys):
    """Test that fewer than 3 points returns error code 2."""
    rc = main(["--x", "1,2", "--y", "2,4"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "at least 3" in err


def test_main_mismatched_lengths_returns_2(capsys):
    """Test that mismatched lengths returns error code 2."""
    rc = main(["--x", "1,2,3", "--y", "2,4"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "same number" in err


def test_main_zero_variance_returns_2(capsys):
    """Test that zero variance in x returns error code 2."""
    rc = main(["--x", "5,5,5,5,5", "--y", "2,4,6,8,10"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "zero variance" in err


def test_main_invalid_alpha_returns_2(capsys):
    """Test that invalid alpha returns error code 2."""
    rc = main(["--x", "1,2,3,4,5", "--y", "2,4,6,8,10", "--alpha", "1.5"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "alpha" in err.lower()


def test_main_missing_y_returns_error():
    """Test that missing --y with --x returns error."""
    with pytest.raises(SystemExit) as exc_info:
        main(["--x", "1,2,3"])
    assert exc_info.value.code == 2


def test_main_csv_file():
    """Test main with CSV file input."""
    # Create temporary CSV file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("height,weight\n")
        f.write("150,50\n")
        f.write("160,58\n")
        f.write("170,65\n")
        f.write("180,75\n")
        f.write("190,82\n")
        f.write("200,90\n")
        csv_path = f.name

    try:
        rc = main(["--file", csv_path, "--x-col", "height", "--y-col", "weight"])
        assert rc == 0
    finally:
        os.unlink(csv_path)


def test_main_csv_nonexistent_file_returns_2(capsys):
    """Test that nonexistent CSV file returns error code 2."""
    rc = main(["--file", "nonexistent.csv", "--x-col", "x", "--y-col", "y"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "not found" in err.lower()


def test_main_invalid_numeric_values_returns_2(capsys):
    """Test that invalid numeric values return error code 2."""
    rc = main(["--x", "1,2,abc", "--y", "2,4,6"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "Error" in err


# -------------------------------------------------------
# Tests for internal functions and edge cases
# -------------------------------------------------------


def test_sum_of_squares_without_mean():
    """Test sum_of_squares computes mean automatically when not provided."""
    values = [1, 2, 3, 4, 5]
    ss = sum_of_squares(values)  # mean_val not provided, should compute it
    expected = 10.0  # sum of (x - 3)^2 for x in [1,2,3,4,5]
    assert abs(ss - expected) < 1e-10


def test_t_cdf_invalid_df():
    """Test t_cdf raises error for invalid degrees of freedom."""
    with pytest.raises(ValueError, match="Degrees of freedom must be at least 1"):
        t_cdf(1.5, 0)


def test_t_cdf_infinite_t():
    """Test t_cdf handles infinite t values."""
    # Positive infinity
    result = t_cdf(float("inf"), 10)
    assert result == 1.0


def test_t_cdf_large_df():
    """t_cdf is exact above df = 30, where it used to fall back to the normal."""
    result = t_cdf(1.96, 100)
    assert result == pytest.approx(float(stats.t.cdf(1.96, 100)), abs=1e-9)
    # The normal CDF is a visibly different number at this df; the old
    # implementation returned it verbatim.
    assert abs(result - 0.5 * (1 + math.erf(1.96 / math.sqrt(2)))) > 1e-4


def test_incomplete_beta_edge_cases():
    """Test incomplete_beta handles edge cases correctly."""
    # x < 0 should return 0
    result = incomplete_beta(2.0, 3.0, -0.5)
    assert result == 0.0

    # x == 0 should return 0
    result = incomplete_beta(2.0, 3.0, 0.0)
    assert result == 0.0

    # x == 1 should return 1
    result = incomplete_beta(2.0, 3.0, 1.0)
    assert result == 1.0


def test_inverse_t_cdf_invalid_p():
    """Test inverse_t_cdf raises error for invalid p values."""
    with pytest.raises(ValueError, match="p must be between 0 and 1"):
        inverse_t_cdf(0.0, 10)

    # with pytest.raises(ValueError, match="p must be between 0 and 1"):
    #     inverse_t_cdf(1.0, 10)


def test_inverse_t_cdf_large_df():
    """inverse_t_cdf is exact above df = 30, not the normal quantile."""
    result = inverse_t_cdf(0.975, 100)
    assert result == pytest.approx(float(stats.t.ppf(0.975, 100)), abs=1e-8)
    assert abs(result - 1.959963985) > 1e-3


def test_f_cdf_edge_cases():
    """Test f_cdf handles edge cases."""
    # f <= 0 should return 0
    result = f_cdf(0.0, 1, 10)
    assert result == 0.0


def test_p_value_t_one_tailed():
    """Test p_value_t for one-tailed tests."""
    # Positive t, one-tailed
    p_val = p_value_t(2.0, 10, sided="one")
    assert 0 < p_val < 0.05

    # Negative t, one-tailed (should use complementary probability)
    p_val = p_value_t(-2.0, 10, sided="one")
    assert 0.95 < p_val < 1.0


def test_parse_csv_missing_x_column(capsys):
    """Test parse_csv_file with missing x column."""
    # Create CSV with only one column
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("weight\n")
        f.write("50\n")
        f.write("55\n")
        csv_path = f.name

    try:
        with pytest.raises(ValueError, match="Column 'height' not found"):
            parse_csv_file(csv_path, "height", "weight")
    finally:
        os.unlink(csv_path)


def test_parse_csv_missing_y_column(capsys):
    """Test parse_csv_file with missing y column."""
    # Create CSV with only one column
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("height\n")
        f.write("150\n")
        f.write("160\n")
        csv_path = f.name

    try:
        with pytest.raises(ValueError, match="Column 'weight' not found"):
            parse_csv_file(csv_path, "height", "weight")
    finally:
        os.unlink(csv_path)


def test_parse_csv_invalid_numeric(capsys):
    """Test parse_csv_file with invalid numeric values."""
    # Create CSV with non-numeric data
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("height,weight\n")
        f.write("150,50\n")
        f.write("abc,55\n")
        csv_path = f.name

    try:
        with pytest.raises(ValueError, match="Invalid numeric value"):
            parse_csv_file(csv_path, "height", "weight")
    finally:
        os.unlink(csv_path)


def test_parse_csv_no_data_rows(capsys):
    """Test parse_csv_file with no data rows."""
    # Create CSV with only headers
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("height,weight\n")
        csv_path = f.name

    try:
        with pytest.raises(ValueError, match="No data rows found"):
            parse_csv_file(csv_path, "height", "weight")
    finally:
        os.unlink(csv_path)


def test_main_missing_file_columns_returns_error():
    """Test that missing --x-col or --y-col with --file returns error."""
    # Missing both columns
    with pytest.raises(SystemExit) as exc_info:
        main(["--file", "data.csv"])
    assert exc_info.value.code == 2

    # Missing y-col
    with pytest.raises(SystemExit) as exc_info:
        main(["--file", "data.csv", "--x-col", "x"])
    assert exc_info.value.code == 2


def test_interpret_r_squared_weak():
    """Test interpret_r_squared for weak fit."""
    assert interpret_r_squared(0.2) == "very weak fit"
    assert interpret_r_squared(0.4) == "weak fit"
    assert interpret_r_squared(0.6) == "moderate fit"
    assert interpret_r_squared(0.8) == "strong fit"
    assert interpret_r_squared(0.95) == "excellent fit"


def test_incomplete_beta_lentz_floor_guards_hit(monkeypatch):
    """Force the near-zero denominator guards in the continued fraction.

    Real inputs never drive a Lentz denominator below 1e-30, so raise the floor
    until every denominator trips it.

    Asserting only that the result is finite would be worthless here: it holds
    whether or not the guards exist, so the test would still pass with all four
    deleted. Pin the clamped value instead. With every denominator forced to
    _TINY, `c * d` is exactly 1.0, the convergence check breaks on the first
    iteration, and the continued fraction collapses to its seed value of
    1 / _TINY. For (2, 3, 0.5) the symmetry relation routes this through
    (3, 2, 0.5), whose front factor is 0.125, giving 1 - 0.125e-5. Deleting or
    inverting any guard changes that number.
    """
    unclamped = incomplete_beta(2.0, 3.0, 0.5)
    monkeypatch.setattr(linreg_module, "_TINY", 1e5)
    clamped = incomplete_beta(2.0, 3.0, 0.5)

    assert math.isfinite(clamped)
    assert clamped == pytest.approx(1.0 - 0.125e-5)
    assert clamped != unclamped


def test_incomplete_beta_survives_subnormal_parameters():
    """Extreme a/b values must still yield a finite probability, not nan or a raise."""
    test_cases = [
        (1e-100, 1e-100, 0.5),
        (1e-150, 1e-150, 0.1),
        (1e-200, 1e-200, 0.9),
        (1e-250, 1e-250, 0.3),
        (1e-300, 1e-300, 0.7),
    ]

    for a, b, x in test_cases:
        result = incomplete_beta(a, b, x)
        assert math.isfinite(result), f"non-finite result for ({a}, {b}, {x})"
        assert 0 <= result <= 1, f"out of range result for ({a}, {b}, {x})"


# ---------------------------------------------------------------------------
# Distribution kernels against scipy
#
# The suite previously pinned only edge cases and the shape of the large-df
# shortcut, so a continued fraction that was seeded incorrectly -- dropping the
# leading term and returning 0.2285 for I_0.4(2,3) against a true 0.5248 --
# passed everything. These compare values, not shapes.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "a,b,x",
    [
        (2.0, 3.0, 0.4),
        (0.5, 0.5, 0.3),
        (5.0, 0.5, 0.5),
        (12.5, 0.5, 0.9259),
        (1.0, 20.0, 0.05),
        (50.0, 0.5, 0.99),
    ],
)
def test_incomplete_beta_matches_scipy(a, b, x):
    assert incomplete_beta(a, b, x) == pytest.approx(
        float(special.betainc(a, b, x)), abs=1e-9
    )


def test_incomplete_beta_known_closed_form():
    """I_0.4(2,3) = 0.5248 exactly; the seeding bug returned 0.2285."""
    assert incomplete_beta(2.0, 3.0, 0.4) == pytest.approx(0.5248, abs=1e-12)


@pytest.mark.parametrize("df", [1, 2, 5, 25, 30, 31, 40, 100, 500])
@pytest.mark.parametrize("t", [-3.0, -1.0, -0.25, 0.5, 2.0, 4.0])
def test_t_cdf_matches_scipy(t, df):
    assert t_cdf(t, df) == pytest.approx(float(stats.t.cdf(t, df)), abs=1e-9)


def test_t_cdf_is_continuous_across_the_old_df_threshold():
    """The old normal shortcut put a visible step at df = 30/31."""
    below, above = t_cdf(2.0, 30), t_cdf(2.0, 31)
    assert abs(below - above) < 1e-3


@pytest.mark.parametrize("df", [1, 5, 10, 30, 40, 100])
@pytest.mark.parametrize("p", [0.6, 0.9, 0.975, 0.995])
def test_inverse_t_cdf_matches_scipy(p, df):
    assert inverse_t_cdf(p, df) == pytest.approx(float(stats.t.ppf(p, df)), abs=1e-6)


def test_inverse_t_cdf_round_trips_with_t_cdf():
    assert t_cdf(inverse_t_cdf(0.9, 7), 7) == pytest.approx(0.9, abs=1e-8)


def test_inverse_t_cdf_rejects_bad_df():
    with pytest.raises(ValueError, match="Degrees of freedom must be at least 1"):
        inverse_t_cdf(0.5, 0)


@pytest.mark.parametrize(
    "f_stat,df1,df2",
    [(0.5, 1, 10), (2.0, 1, 10), (5.0, 2, 20), (12.0, 3, 30), (0.1, 4, 8)],
)
def test_f_cdf_matches_scipy(f_stat, df1, df2):
    assert f_cdf(f_stat, df1, df2) == pytest.approx(
        float(stats.f.cdf(f_stat, df1, df2)), abs=1e-9
    )


def test_regression_p_value_matches_scipy_end_to_end():
    """The whole path: fit, t-statistic, and the reported two-sided p-value."""
    x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    y = [2.3, 3.9, 6.4, 7.6, 10.5, 11.8, 14.6, 15.9]
    model = linear_regression(x, y)
    reference = stats.linregress(x, y)

    assert model.slope == pytest.approx(reference.slope, rel=1e-12)
    assert model.se_slope == pytest.approx(reference.stderr, rel=1e-10)
    expected_p = float(2 * stats.t.sf(abs(model.t_slope), model.df))
    assert p_value_t(model.t_slope, model.df) == pytest.approx(expected_p, rel=1e-6)


def test_confidence_interval_critical_value_matches_scipy():
    """t_crit drives every interval the CLI prints; it was 18% low at df = 5."""
    x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    y = [2.1, 4.2, 5.8, 8.1, 9.9, 12.2, 14.0]
    model = linear_regression(x, y)
    assert inverse_t_cdf(0.975, model.df) == pytest.approx(
        float(stats.t.ppf(0.975, model.df)), abs=1e-8
    )
