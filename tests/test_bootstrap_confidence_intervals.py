"""Tests for bootstrap confidence intervals functions."""

import random
import statistics
from unittest.mock import patch

from src.utils.bootstrap_confidence_intervals import (
    get_stat_function,
    main,
    read_stdin_data,
)


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


def test_main_no_data():
    """Test main function with no data from stdin."""
    with patch("sys.stdin", iter([])):
        result = main(["--stat", "mean"])
        assert result == 2


def test_main_invalid_confidence_too_high():
    """Test main function with confidence level too high."""
    result = main(["--data", "10", "20", "30", "--stat", "mean", "--confidence", "1.0"])
    assert result == 2


def test_main_invalid_n_bootstrap():
    """Test main function with invalid n_bootstrap."""
    result = main(["--data", "10", "20", "30", "--stat", "mean", "--n-bootstrap", "0"])
    assert result == 2


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
