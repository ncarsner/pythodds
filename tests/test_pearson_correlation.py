"""Tests for Pearson correlation functions."""

import math
import os
import tempfile
from unittest.mock import patch

import pytest

from src.utils.pearson_correlation import (
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
    pearson_r,
    standard_normal_cdf,
    t_cdf,
)


def test_mean_empty():
    """Test mean raises error for empty list."""
    with pytest.raises(ValueError, match="Cannot compute mean of empty list"):
        mean([])


def test_pearson_r_constant_values():
    """Test with constant values (zero variance)."""
    x = [1, 1, 1, 1, 1]
    y = [2, 3, 4, 5, 6]
    r = pearson_r(x, y)
    assert r == 0.0


def test_pearson_r_unequal_lengths():
    """Test that unequal length inputs raise error."""
    x = [1, 2, 3]
    y = [1, 2]
    with pytest.raises(ValueError, match="same length"):
        pearson_r(x, y)


def test_pearson_r_insufficient_data():
    """Test that insufficient data raises error."""
    x = [1]
    y = [2]
    with pytest.raises(ValueError, match="at least 2 data points"):
        pearson_r(x, y)


def test_correlation_t_statistic_insufficient_data():
    """Test that insufficient data raises error."""
    with pytest.raises(ValueError, match="at least 3 data points"):
        correlation_t_statistic(0.5, 2)


def test_fisher_z_transform():
    """Test Fisher Z-transformation."""
    # Z(0) = 0
    assert abs(fisher_z_transform(0.0)) < 1e-10

    # Z(0.5) ≈ 0.549
    z = fisher_z_transform(0.5)
    assert abs(z - 0.549) < 0.01

    # Z(1) = inf
    assert math.isinf(fisher_z_transform(1.0))


def test_inverse_normal_cdf_boundary():
    """Test inverse normal CDF at boundaries."""
    with pytest.raises(ValueError):
        inverse_normal_cdf(0.0)

    with pytest.raises(ValueError):
        inverse_normal_cdf(1.0)


def test_correlation_confidence_interval():
    """Test confidence interval calculation."""
    r = 0.7
    n = 50
    alpha = 0.05

    lower, upper = correlation_confidence_interval(r, n, alpha)

    # CI should contain r
    assert lower < r < upper

    # Bounds should be in [-1, 1]
    assert -1.0 <= lower <= 1.0
    assert -1.0 <= upper <= 1.0

    # Lower should be less than upper
    assert lower < upper


def test_correlation_confidence_interval_small_sample():
    """Test CI for very small sample."""
    r = 0.5
    n = 3  # Very small sample
    alpha = 0.05

    lower, upper = correlation_confidence_interval(r, n, alpha)

    # Should return wide interval [-1, 1]
    assert lower == -1.0
    assert upper == 1.0


def test_interpret_correlation():
    """Test correlation interpretation."""
    assert "very strong" in interpret_correlation(0.95).lower()
    assert "strong" in interpret_correlation(0.75).lower()
    assert "moderate" in interpret_correlation(0.6).lower()
    assert "weak" in interpret_correlation(0.35).lower()
    assert "very weak" in interpret_correlation(0.1).lower()
    assert "positive" in interpret_correlation(0.5).lower()
    assert "negative" in interpret_correlation(-0.5).lower()


def test_parse_csv_file_missing_column():
    """Test CSV parsing with missing column."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
        f.write("height,weight\n")
        f.write("160,60\n")
        temp_path = f.name

    try:
        with pytest.raises(ValueError, match="not found in CSV"):
            parse_csv_file(temp_path, "height", "age")
    finally:
        os.unlink(temp_path)


def test_parse_csv_file_not_found():
    """Test CSV parsing with non-existent file."""
    with pytest.raises(ValueError, match="File not found"):
        parse_csv_file("nonexistent.csv", "x", "y")


def test_main_with_csv_file():
    """Test main function with CSV file."""
    # Create temporary CSV file
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
        f.write("x,y\n")
        f.write("1,2\n")
        f.write("2,4\n")
        f.write("3,6\n")
        temp_path = f.name

    try:
        exit_code = main(["--file", temp_path, "--x-col", "x", "--y-col", "y"])
        assert exit_code == 0
    finally:
        os.unlink(temp_path)


def test_main_insufficient_data():
    """Test main function with insufficient data."""
    exit_code = main(["--x", "1", "--y", "2"])
    assert exit_code == 2


def test_main_unequal_lengths():
    """Test main function with unequal length inputs."""
    exit_code = main(["--x", "1,2,3", "--y", "4,5"])
    assert exit_code == 2


def test_main_invalid_alpha():
    """Test main function with invalid alpha."""
    exit_code = main(["--x", "1,2,3,4,5", "--y", "2,4,6,8,10", "--alpha", "1.5"])
    assert exit_code == 2


def test_main_invalid_numeric_values():
    """Test main function with invalid numeric values."""
    exit_code = main(["--x", "1,2,abc", "--y", "4,5,6"])
    assert exit_code == 2


def test_main_one_tailed_test():
    """Test main function with one-tailed test."""
    exit_code = main(
        ["--x", "1,2,3,4,5", "--y", "2,4,6,8,10", "--alpha", "0.05", "--sided", "one"]
    )
    assert exit_code == 0


def test_t_cdf_invalid_df():
    """Test t_cdf raises error for invalid degrees of freedom."""
    with pytest.raises(ValueError, match="Degrees of freedom must be at least 1"):
        t_cdf(1.5, 0)


def test_t_cdf_infinite_t():
    """Test t_cdf with infinite t values."""
    # Positive infinity
    assert t_cdf(float("inf"), 10) == 1.0
    # Negative infinity
    assert t_cdf(float("-inf"), 10) == 0.0


def test_t_cdf_large_df():
    """Test t_cdf uses normal approximation for large df."""
    # With df > 30, should use normal approximation
    result = t_cdf(1.96, 50)
    # Should be close to normal CDF
    expected = standard_normal_cdf(1.96)
    assert abs(result - expected) < 1e-6


def test_t_cdf_negative_t():
    """Test t_cdf with negative t value and small df."""
    # Test with negative t and small df
    result = t_cdf(-1.5, 10)
    # Should be less than 0.5
    assert result < 0.5
    assert result > 0.0


def test_correlation_p_value_one_tailed_positive():
    """Test one-tailed p-value with positive correlation."""
    # Positive correlation, one-tailed test
    r = 0.6
    n = 25
    p = correlation_p_value(r, n, "one")

    # One-tailed should be smaller than two-tailed
    p_two = correlation_p_value(r, n, "two")
    assert p < p_two
    assert p > 0.0


def test_correlation_p_value_one_tailed_negative():
    """Test one-tailed p-value with negative correlation."""
    # Negative correlation, one-tailed test
    r = -0.6
    n = 25
    p = correlation_p_value(r, n, "one")

    # Should return 1 - p_upper for negative t
    assert p > 0.5
    assert p < 1.0


def test_inverse_fisher_z_positive_infinity():
    """Test inverse Fisher Z with positive infinity."""
    result = inverse_fisher_z(float("inf"))
    assert result == 1.0


def test_inverse_normal_cdf_lower_region():
    """Test inverse normal CDF in lower region (p < 0.02425)."""
    # Test values in the lower region
    p = 0.01  # Less than _P_LOW = 0.02425
    z = inverse_normal_cdf(p)
    # Should be a negative value
    assert z < 0
    # Check approximate correctness (p=0.01 should give z ≈ -2.326)
    assert abs(z - (-2.326)) < 0.01


def test_inverse_normal_cdf_upper_region():
    """Test inverse normal CDF in upper region (p > 0.97575)."""
    # Test values in the upper region
    p = 0.99  # Greater than _P_HIGH = 0.97575
    z = inverse_normal_cdf(p)
    # Should be a positive value
    assert z > 0
    # Check approximate correctness (p=0.99 should give z ≈ 2.326)
    assert abs(z - 2.326) < 0.01


def test_correlation_confidence_interval_near_perfect_negative():
    """Test CI for near-perfect negative correlation."""
    # Test with r very close to -1
    r = -0.9999
    n = 50
    alpha = 0.05

    lower, upper = correlation_confidence_interval(r, n, alpha)

    # Should return (-1.0, -0.9) for near-perfect negative
    assert lower == -1.0
    assert upper == -0.9


def test_parse_csv_file_empty():
    """Test CSV parsing with empty file or no header."""
    # Create an empty CSV file
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
        # Write nothing (empty file)
        temp_path = f.name

    try:
        with pytest.raises(ValueError, match="empty or has no header"):
            parse_csv_file(temp_path, "x", "y")
    finally:
        os.unlink(temp_path)


def test_parse_csv_file_x_column_not_found():
    """Test CSV parsing when x column is not found."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
        f.write("height,weight\n")
        f.write("160,60\n")
        temp_path = f.name

    try:
        with pytest.raises(ValueError, match="Column 'x' not found in CSV"):
            parse_csv_file(temp_path, "x", "weight")
    finally:
        os.unlink(temp_path)


def test_parse_csv_file_invalid_numeric():
    """Test CSV parsing with invalid numeric values."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
        f.write("x,y\n")
        f.write("1,2\n")
        f.write("not_a_number,4\n")  # Invalid numeric value
        temp_path = f.name

    try:
        with pytest.raises(ValueError, match="Failed to parse numeric value"):
            parse_csv_file(temp_path, "x", "y")
    finally:
        os.unlink(temp_path)


def test_parse_args_missing_y():
    """Test parse_args raises error when --x is provided without --y."""
    with pytest.raises(SystemExit):
        # parser.error() calls sys.exit(2) which raises SystemExit
        parse_args(["--x", "1,2,3"])


def test_parse_args_missing_file_columns():
    """Test parse_args raises error when --file is provided without columns."""
    # Missing both --x-col and --y-col
    with pytest.raises(SystemExit):
        parse_args(["--file", "data.csv"])

    # Missing --y-col
    with pytest.raises(SystemExit):
        parse_args(["--file", "data.csv", "--x-col", "x"])

    # Missing --x-col
    with pytest.raises(SystemExit):
        parse_args(["--file", "data.csv", "--y-col", "y"])


def test_main_pearson_r_error():
    """Test main handles ValueError from pearson_r computation (lines 499-501)."""
    # Mock pearson_r to raise ValueError
    with patch("src.utils.pearson_correlation.pearson_r") as mock_pearson:
        mock_pearson.side_effect = ValueError("Computation error")
        exit_code = main(["--x", "1,2,3", "--y", "4,5,6"])
        assert exit_code == 2


def test_main_hypothesis_test_insufficient_data():
    """Test main with alpha but insufficient data for hypothesis test (lines 518-519)."""
    # Only 2 data points, but alpha provided (need at least 3 for hypothesis test)
    exit_code = main(["--x", "1,2", "--y", "3,4", "--alpha", "0.05"])
    assert exit_code == 2


def test_main_non_significant_result():
    """Test main with non-significant correlation result (line 538)."""
    # Use data with very weak/no correlation to get non-significant result
    # x and y are essentially uncorrelated
    exit_code = main(["--x", "1,2,3,4,5", "--y", "5,3,7,2,6", "--alpha", "0.05"])
    assert exit_code == 0


def test_main_hypothesis_test_error():
    """Test main handles ValueError from hypothesis test computation (lines 546-548)."""
    # Mock correlation_t_statistic to raise ValueError
    with patch("src.utils.pearson_correlation.correlation_t_statistic") as mock_t_stat:
        mock_t_stat.side_effect = ValueError("Hypothesis test error")
        exit_code = main(["--x", "1,2,3,4", "--y", "5,6,7,8", "--alpha", "0.05"])
        assert exit_code == 2
