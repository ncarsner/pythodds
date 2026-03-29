"""Tests for bootstrap confidence intervals functions."""

import random
import statistics
from unittest.mock import patch

from src.utils.bootstrap_confidence_intervals import (
    bootstrap_resample,
    compute_confidence_interval,
    get_stat_function,
    main,
    read_stdin_data,
)


def test_bootstrap_resample_returns_correct_length():
    """Test that bootstrap_resample returns the correct number of statistics."""
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    n_bootstrap = 100

    def mean_func(x):
        return sum(x) / len(x)

    result = bootstrap_resample(data, n_bootstrap, mean_func)
    assert len(result) == n_bootstrap


def test_bootstrap_resample_with_seed():
    """Test that bootstrap_resample is reproducible with same seed."""
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    n_bootstrap = 50

    def mean_func(x):
        return sum(x) / len(x)

    # First run
    random.seed(42)
    result1 = bootstrap_resample(data, n_bootstrap, mean_func)

    # Second run with same seed
    random.seed(42)
    result2 = bootstrap_resample(data, n_bootstrap, mean_func)

    assert result1 == result2


def test_compute_confidence_interval_95_percent():
    """Test confidence interval computation with 95% confidence."""
    # Create a simple distribution
    bootstrap_stats = (*range(1, 101),)

    lower, upper = compute_confidence_interval(bootstrap_stats, 0.95)

    # For 95% CI, we expect roughly the 2.5th and 97.5th percentiles
    # With 100 values, that's around index 2 and 97
    assert lower <= 5
    assert upper >= 95


def test_compute_confidence_interval_90_percent():
    """Test confidence interval computation with 90% confidence."""
    bootstrap_stats = list(range(1, 101))

    lower, upper = compute_confidence_interval(bootstrap_stats, 0.90)

    # For 90% CI, we expect roughly the 5th and 95th percentiles
    assert lower <= 10
    assert upper >= 90


def test_compute_confidence_interval_bounds_clamping():
    """Test that index clamping works correctly."""
    # Small sample
    bootstrap_stats = [1.0, 2.0, 3.0]

    lower, upper = compute_confidence_interval(bootstrap_stats, 0.95)

    # Should not crash and should return valid bounds
    assert lower >= 1.0
    assert upper <= 3.0
    assert lower <= upper


def test_get_stat_function_mean():
    """Test that get_stat_function returns correct function for mean."""
    func = get_stat_function("mean")
    result = func([1, 2, 3, 4, 5])
    assert result == 3.0


def test_get_stat_function_median():
    """Test that get_stat_function returns correct function for median."""
    func = get_stat_function("median")
    result = func([1, 2, 3, 4, 5])
    assert result == 3


def test_get_stat_function_stdev():
    """Test that get_stat_function returns correct function for stdev."""
    func = get_stat_function("stdev")
    result = func([1, 2, 3, 4, 5])
    assert result > 0  # Standard deviation should be positive


def test_get_stat_function_invalid():
    """Test that get_stat_function raises error for invalid statistic."""
    try:
        get_stat_function("invalid_stat")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Unknown statistic" in str(e)


def test_main_basic_usage(capsys):
    """Test main function with basic arguments."""
    random.seed(42)
    result = main(
        [
            "--data",
            "10",
            "20",
            "30",
            "40",
            "50",
            "--stat",
            "mean",
            "--n-bootstrap",
            "100",
            "--seed",
            "42",
        ]
    )
    assert result == 0

    captured = capsys.readouterr()
    assert "Original mean:" in captured.out
    assert "Confidence Interval:" in captured.out


def test_main_median_statistic(capsys):
    """Test main function with median statistic."""
    random.seed(42)
    result = main(
        [
            "--data",
            "1",
            "2",
            "3",
            "4",
            "5",
            "--stat",
            "median",
            "--n-bootstrap",
            "100",
            "--seed",
            "42",
        ]
    )
    assert result == 0

    captured = capsys.readouterr()
    assert "median" in captured.out


def test_main_stdev_statistic(capsys):
    """Test main function with stdev statistic."""
    random.seed(42)
    result = main(
        [
            "--data",
            "1",
            "2",
            "3",
            "4",
            "5",
            "--stat",
            "stdev",
            "--n-bootstrap",
            "100",
            "--seed",
            "42",
        ]
    )
    assert result == 0

    captured = capsys.readouterr()
    assert "stdev" in captured.out


def test_main_custom_confidence(capsys):
    """Test main function with custom confidence level."""
    random.seed(42)
    result = main(
        [
            "--data",
            "10",
            "20",
            "30",
            "--stat",
            "mean",
            "--confidence",
            "0.90",
            "--seed",
            "42",
        ]
    )
    assert result == 0

    captured = capsys.readouterr()
    assert "90.0%" in captured.out


def test_main_insufficient_data():
    """Test main function with insufficient data."""
    result = main(["--data", "10", "--stat", "mean"])
    assert result == 2


def test_main_no_data():
    """Test main function with no data from stdin."""
    with patch("sys.stdin", iter([])):
        result = main(["--stat", "mean"])
        assert result == 2


def test_main_invalid_confidence_too_low():
    """Test main function with confidence level too low."""
    result = main(["--data", "10", "20", "30", "--stat", "mean", "--confidence", "0.0"])
    assert result == 2


def test_main_invalid_confidence_too_high():
    """Test main function with confidence level too high."""
    result = main(["--data", "10", "20", "30", "--stat", "mean", "--confidence", "1.0"])
    assert result == 2


def test_main_invalid_confidence_negative():
    """Test main function with negative confidence level."""
    result = main(
        ["--data", "10", "20", "30", "--stat", "mean", "--confidence", "-0.5"]
    )
    assert result == 2


def test_main_invalid_n_bootstrap():
    """Test main function with invalid n_bootstrap."""
    result = main(["--data", "10", "20", "30", "--stat", "mean", "--n-bootstrap", "0"])
    assert result == 2


def test_main_stdev_insufficient_data():
    """Test main function with stdev but only one data point."""
    result = main(["--data", "10", "--stat", "stdev"])
    assert result == 2


def test_main_precision_parameter(capsys):
    """Test main function with custom precision."""
    random.seed(42)
    result = main(
        [
            "--data",
            "10.123",
            "20.456",
            "30.789",
            "--stat",
            "mean",
            "--n-bootstrap",
            "100",
            "--precision",
            "2",
            "--seed",
            "42",
        ]
    )
    assert result == 0

    captured = capsys.readouterr()
    # Output should have 2 decimal places
    assert captured.out.count(".") >= 2


def test_read_stdin_data_single_line():
    """Test reading data from stdin (single line)."""
    with patch("sys.stdin", iter(["10 20 30 40 50\n"])):
        data = read_stdin_data()
        assert data == [10.0, 20.0, 30.0, 40.0, 50.0]


def test_read_stdin_data_multiple_lines():
    """Test reading data from stdin (multiple lines)."""
    with patch("sys.stdin", iter(["10 20\n", "30 40\n", "50\n"])):
        data = read_stdin_data()
        assert data == [10.0, 20.0, 30.0, 40.0, 50.0]


def test_read_stdin_data_empty_lines():
    """Test reading data from stdin with empty lines."""
    with patch("sys.stdin", iter(["10 20\n", "\n", "30 40\n"])):
        data = read_stdin_data()
        assert data == [10.0, 20.0, 30.0, 40.0]


def test_read_stdin_data_with_invalid_values(capsys):
    """Test reading data from stdin with invalid values."""
    with patch("sys.stdin", iter(["10 20 invalid 30\n"])):
        data = read_stdin_data()
        # Should skip invalid value
        assert data == [10.0, 20.0, 30.0]

        # Check warning was printed
        captured = capsys.readouterr()
        assert "Warning" in captured.err
        assert "invalid" in captured.err


def test_main_with_stdin_data(capsys):
    """Test main function reading from stdin."""
    random.seed(42)
    with patch("sys.stdin", iter(["10 20 30 40 50\n"])):
        result = main(["--stat", "mean", "--n-bootstrap", "100", "--seed", "42"])
        assert result == 0

        captured = capsys.readouterr()
        assert "Original mean:" in captured.out


def test_main_with_stdin_insufficient_data():
    """Test main function with stdin having insufficient data."""
    with patch("sys.stdin", iter(["10\n"])):
        result = main(["--stat", "mean"])
        assert result == 2


def test_bootstrap_resample_all_values_in_range():
    """Test that bootstrap resamples only contain values from original data."""
    data = [1.0, 2.0, 3.0]
    n_bootstrap = 50

    def identity(x):
        # Return first value to check resampling
        return x[0]

    result = bootstrap_resample(data, n_bootstrap, identity)

    # All results should be from original data
    for val in result:
        assert val in data


def test_compute_confidence_interval_sorted_list():
    """Test that confidence interval works with presorted list."""
    bootstrap_stats = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    lower, upper = compute_confidence_interval(bootstrap_stats, 0.80)

    # Should return valid interval
    assert lower < upper
    assert lower >= 1
    assert upper <= 10


def test_main_get_stat_function_error():
    """Test main handles ValueError from get_stat_function."""
    with patch(
        "src.utils.bootstrap_confidence_intervals.get_stat_function"
    ) as mock_get_stat:
        mock_get_stat.side_effect = ValueError("Invalid statistic")
        result = main(["--data", "10", "20", "30", "--stat", "mean"])
        assert result == 2


def test_main_statistics_error():
    """Test main handles StatisticsError when computing statistic."""
    with patch(
        "src.utils.bootstrap_confidence_intervals.get_stat_function"
    ) as mock_get_stat:
        # Create a function that raises StatisticsError
        def raise_error(data):
            raise statistics.StatisticsError("Cannot compute statistic")

        mock_get_stat.return_value = raise_error
        result = main(["--data", "10", "20", "30", "--stat", "mean"])
        assert result == 2
