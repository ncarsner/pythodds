"""Tests for Spearman rank correlation functions."""

import math

import pytest

from src.utils.spearman_correlation import (
    correlation_confidence_interval,
    correlation_p_value,
    correlation_t_statistic,
    fisher_z_transform,
    interpret_correlation,
    inverse_fisher_z,
    inverse_normal_cdf,
    main,
    mean,
    parse_args,
    parse_csv_file,
    pearson_r_from_ranks,
    spearman_rho,
    standard_normal_cdf,
    t_cdf,
)


def test_mean_empty():
    """Test mean raises error for empty list."""
    with pytest.raises(ValueError, match="Cannot compute mean of empty list"):
        mean([])


def test_spearman_rho_unequal_lengths():
    """Test that unequal length inputs raise error."""
    x = [1, 2, 3]
    y = [1, 2]
    with pytest.raises(ValueError, match="same length"):
        spearman_rho(x, y)


def test_spearman_rho_insufficient_data():
    """Test that insufficient data raises error."""
    x = [1]
    y = [2]
    with pytest.raises(ValueError, match="at least 2 data points"):
        spearman_rho(x, y)


def test_correlation_t_statistic_insufficient_data():
    """Test that insufficient data raises error."""
    with pytest.raises(ValueError, match="at least 3 data points"):
        correlation_t_statistic(0.5, 2)


def test_inverse_normal_cdf_boundary():
    """Test inverse normal CDF at boundaries."""
    with pytest.raises(ValueError):
        inverse_normal_cdf(0.0)

    with pytest.raises(ValueError):
        inverse_normal_cdf(1.0)


def test_correlation_confidence_interval_small_n():
    """Test CI for very small sample returns [-1, 1]."""
    lower, upper = correlation_confidence_interval(0.5, 3, 0.05)
    assert lower == -1.0
    assert upper == 1.0


def test_correlation_p_value_one_tailed():
    """Test one-tailed p-value calculation."""
    rho = 0.5
    n = 20
    p_two = correlation_p_value(rho, n, "two")
    p_one = correlation_p_value(rho, n, "one")
    # One-tailed p-value is approximately half of two-tailed
    assert p_one < p_two


def test_parse_csv_file_missing_column(tmp_path):
    """Test CSV parsing with missing column."""
    csv_file = tmp_path / "test_data.csv"
    csv_file.write_text("x,y\n1,2\n")

    # Test missing x column
    with pytest.raises(ValueError, match="Column 'z' not found"):
        parse_csv_file(str(csv_file), "z", "y")

    # Test missing y column
    with pytest.raises(ValueError, match="Column 'z' not found"):
        parse_csv_file(str(csv_file), "x", "z")


def test_interpret_correlation():
    """Test correlation interpretation."""
    assert "very strong positive" in interpret_correlation(0.95)
    assert "strong positive" in interpret_correlation(0.75)
    assert "moderate positive" in interpret_correlation(0.55)
    assert "weak positive" in interpret_correlation(0.35)
    assert "very weak positive" in interpret_correlation(0.15)

    assert "very strong negative" in interpret_correlation(-0.95)
    assert "strong negative" in interpret_correlation(-0.75)
    assert "moderate negative" in interpret_correlation(-0.55)
    assert "weak negative" in interpret_correlation(-0.35)
    assert "very weak negative" in interpret_correlation(-0.15)


def test_parse_args_missing_y():
    """Test that missing --y with --x raises error."""
    with pytest.raises(SystemExit):
        parse_args(["--x", "1,2,3"])


def test_parse_args_missing_cols():
    """Test that missing column names with --file raises error."""
    with pytest.raises(SystemExit):
        parse_args(["--file", "data.csv"])


def test_main_with_show_ranks(capsys):
    """Test main function with --show-ranks."""
    ret = main(["--x", "1,2,3,4,5", "--y", "5,4,3,2,1", "--show-ranks"])
    assert ret == 0

    captured = capsys.readouterr()
    assert "Data and Ranks:" in captured.out
    assert "Rank(X)" in captured.out
    assert "Rank(Y)" in captured.out


def test_main_insufficient_data(capsys):
    """Test main with insufficient data."""
    ret = main(["--x", "1", "--y", "2"])
    assert ret == 2

    captured = capsys.readouterr()
    assert "Error:" in captured.err
    assert "at least 2 data points" in captured.err


def test_main_unequal_lengths(capsys):
    """Test main with unequal length inputs."""
    ret = main(["--x", "1,2,3", "--y", "4,5"])
    assert ret == 2

    captured = capsys.readouterr()
    assert "Error:" in captured.err
    assert "same number of values" in captured.err


def test_main_invalid_alpha(capsys):
    """Test main with invalid alpha."""
    ret = main(["--x", "1,2,3", "--y", "4,5,6", "--alpha", "1.5"])
    assert ret == 2

    captured = capsys.readouterr()
    assert "Error:" in captured.err
    assert "alpha must be between 0 and 1" in captured.err


def test_main_csv_file(tmp_path, capsys):
    """Test main with CSV file input."""
    csv_file = tmp_path / "data.csv"
    csv_file.write_text("satisfaction,loyalty\n5,4\n3,2\n4,3\n2,1\n1,0\n")

    ret = main(
        ["--file", str(csv_file), "--x-col", "satisfaction", "--y-col", "loyalty"]
    )
    assert ret == 0

    captured = capsys.readouterr()
    assert "Spearman's ρ:" in captured.out


def test_main_csv_not_found(capsys):
    """Test main with non-existent CSV file."""
    ret = main(["--file", "/nonexistent.csv", "--x-col", "x", "--y-col", "y"])
    assert ret == 2

    captured = capsys.readouterr()
    assert "Error:" in captured.err


def test_main_one_tailed(capsys):
    """Test main with one-tailed test."""
    ret = main(
        ["--x", "1,2,3,4,5", "--y", "2,3,5,6,20", "--alpha", "0.05", "--sided", "one"]
    )
    assert ret == 0

    captured = capsys.readouterr()
    assert "one-tailed" in captured.out


def test_pearson_r_from_ranks_unequal_lengths():
    """Test that unequal length rank inputs raise error."""
    rank_x = [1.0, 2.0, 3.0]
    rank_y = [1.0, 2.0]
    with pytest.raises(ValueError, match="same length"):
        pearson_r_from_ranks(rank_x, rank_y)


def test_pearson_r_from_ranks_insufficient_data():
    """Test that insufficient rank data raises error."""
    rank_x = [1.0]
    rank_y = [1.0]
    with pytest.raises(ValueError, match="at least 2 data points"):
        pearson_r_from_ranks(rank_x, rank_y)


def test_t_cdf_invalid_df():
    """Test t-CDF with invalid degrees of freedom."""
    with pytest.raises(ValueError, match="Degrees of freedom must be at least 1"):
        t_cdf(1.5, 0)


def test_t_cdf_large_df():
    """Test t-CDF with very large degrees of freedom (converges to normal)."""
    t_val = 1.96
    result = t_cdf(t_val, 1500)
    # Should be very close to normal CDF value at 1.96
    expected = standard_normal_cdf(t_val)
    assert abs(result - expected) < 1e-4


def test_fisher_z_transform_negative():
    """Test Fisher Z-transformation with negative correlation."""
    rho = -0.5
    z = fisher_z_transform(rho)
    # Z should be negative for negative rho
    assert z < 0

    # Test negative perfect correlation
    rho = -1.0
    z = fisher_z_transform(rho)
    assert math.isinf(z) and z < 0


def test_inverse_fisher_z_positive_inf():
    """Test inverse Fisher Z with positive infinity."""
    z = float("inf")
    rho = inverse_fisher_z(z)
    assert rho == 1.0


def test_correlation_p_value_one_tailed_negative():
    """Test one-tailed p-value with negative correlation."""
    rho = -0.5
    n = 20
    p_one = correlation_p_value(rho, n, "one")
    # For negative rho, one-tailed p-value (testing positive direction) should be large
    assert p_one > 0.5


def test_main_hypothesis_test_insufficient_data(capsys):
    """Test main with hypothesis test on insufficient data."""
    ret = main(["--x", "1,2", "--y", "3,4", "--alpha", "0.05"])
    assert ret == 2

    captured = capsys.readouterr()
    assert "Error:" in captured.err
    assert "at least 3 data points" in captured.err


def test_correlation_confidence_interval_negative_perfect():
    """Test confidence interval for near-perfect negative correlation."""
    rho = -0.9999
    n = 50
    alpha = 0.05

    lower, upper = correlation_confidence_interval(rho, n, alpha)

    # For near-perfect negative correlation, CI should be narrow and negative
    assert lower == -1.0
    assert upper == -0.9


def test_parse_csv_file_empty(tmp_path):
    """Test CSV parsing with empty file."""
    csv_file = tmp_path / "empty.csv"
    csv_file.write_text("")

    with pytest.raises(ValueError, match="empty or has no header"):
        parse_csv_file(str(csv_file), "x", "y")


def test_main_spearman_calculation_error(capsys, monkeypatch):
    """Test main when spearman_rho raises an error during calculation."""

    # Patch spearman_rho to raise a ValueError
    def mock_spearman_rho(x, y):
        raise ValueError("Calculation error")

    monkeypatch.setattr(
        "src.utils.spearman_correlation.spearman_rho", mock_spearman_rho
    )

    ret = main(["--x", "1,2,3", "--y", "4,5,6"])
    assert ret == 2


def test_main_not_significant_correlation(capsys):
    """Test main with weak correlation that is not significant."""
    # Use data with essentially no correlation
    ret = main(
        ["--x", "1,2,3,4,5,6,7,8,9,10", "--y", "5,4,6,3,7,2,8,1,9,0", "--alpha", "0.05"]
    )
    assert ret == 0

    captured = capsys.readouterr()
    # Should show "Fail to reject H₀"
    assert "Fail to reject H₀" in captured.out or "Not significant" in captured.out


def test_main_hypothesis_test_error(capsys, monkeypatch):
    """Test main when hypothesis test calculation raises an error."""

    # Valid data passes initial checks
    # Patch correlation_t_statistic to raise an error
    def mock_t_stat(rho, n):
        raise ValueError("t-statistic error")

    monkeypatch.setattr(
        "src.utils.spearman_correlation.correlation_t_statistic", mock_t_stat
    )

    ret = main(["--x", "1,2,3,4,5", "--y", "2,4,6,8,10", "--alpha", "0.05"])
    assert ret == 2

    captured = capsys.readouterr()
    assert "Error in hypothesis test:" in captured.err


def test_parse_csv_file_invalid_float(tmp_path):
    """Test CSV parsing with data that can't be converted to float."""
    csv_file = tmp_path / "bad_float.csv"
    csv_file.write_text("x,y\n1.0,2.0\ninvalid,3.0\n")

    with pytest.raises(ValueError, match="Failed to parse numeric value"):
        parse_csv_file(str(csv_file), "x", "y")


def test_main_with_constant_data(capsys):
    """Test main with data that results in calculation issues."""
    # Same values causes issues
    ret = main(["--x", "1,1,1,1,1", "--y", "2,2,2,2,2"])
    # Succeeds with rho=0 or fails gracefully
    assert ret in [0, 2]
