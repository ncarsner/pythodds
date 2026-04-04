"""Tests for forecast_time_series module."""

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.forecast_time_series import (
    backtest,
    calculate_metrics,
    double_exponential_smoothing,
    forecast_holt_winters,
    holt_winters,
    load_data,
    main,
    simple_exponential_smoothing,
)


class TestSimpleExponentialSmoothing:
    """Tests for simple exponential smoothing."""

    def test_simple_smoothing_empty_data(self):
        """Test with empty data."""
        with pytest.raises(ValueError, match="Data cannot be empty"):
            simple_exponential_smoothing([], alpha=0.3)


class TestDoubleExponentialSmoothing:
    """Tests for double exponential smoothing."""

    def test_double_smoothing_insufficient_data(self):
        """Test with insufficient data."""
        with pytest.raises(ValueError, match="at least 2 data points"):
            double_exponential_smoothing([10], alpha=0.3, beta=0.1)


class TestHoltWinters:
    """Tests for Holt-Winters exponential smoothing."""

    def test_holt_winters_multiplicative(self):
        """Test Holt-Winters with multiplicative seasonality."""
        data = [100, 150, 80, 120] * 5  # 20 points, period=4
        fitted, level, trend, seasonal = holt_winters(
            data,
            alpha=0.3,
            beta=0.1,
            gamma=0.1,
            seasonal_period=4,
            seasonal_type="multiplicative",
        )

        assert len(seasonal) == 4
        # Multiplicative seasonal components should average to approximately 1
        assert 0.5 < sum(seasonal) / len(seasonal) < 1.5

    def test_holt_winters_insufficient_data(self):
        """Test with insufficient data for seasonal model."""
        data = [10, 20, 15]
        with pytest.raises(ValueError, match="at least"):
            holt_winters(data, seasonal_period=4)


class TestForecasting:
    """Tests for forecast generation functions."""

    def test_forecast_holt_winters_multiplicative(self):
        """Test Holt-Winters forecast with multiplicative seasonality."""
        level = 100
        trend = 1
        seasonal = [1.1, 0.9, 1.05, 0.95]
        periods = 4
        forecasts = forecast_holt_winters(
            level,
            trend,
            seasonal,
            periods,
            seasonal_period=4,
            seasonal_type="multiplicative",
        )

        assert len(forecasts) == periods


class TestMetrics:
    """Tests for metrics and residual calculations."""

    def test_calculate_metrics_empty(self):
        """Test metrics with empty residuals."""
        metrics = calculate_metrics([])
        assert metrics["rmse"] == 0
        assert metrics["mae"] == 0
        assert metrics["std"] == 0


class TestDataLoading:
    """Tests for data loading functionality."""

    def test_load_data_csv_with_header(self):
        """Test loading CSV with non-numeric header."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("value\n100\n200\n300\n")
            temp_path = f.name

        try:
            data = load_data(temp_path)
            assert data == [100, 200, 300]
        finally:
            Path(temp_path).unlink()


class TestBacktest:
    """Tests for backtesting functionality."""

    def test_backtest_invalid_periods(self):
        """Test backtest with invalid periods."""
        data = [10, 20, 30]
        with pytest.raises(ValueError, match="must be less than"):
            backtest(data, "simple", periods=3)

    def test_backtest_holt_winters(self):
        """Test backtest with Holt-Winters method."""
        # Create seasonal data with at least 2*seasonal_period points
        data = [10, 20, 15, 25, 12, 22, 17, 27] * 3  # 24 points
        result = backtest(
            data,
            "holt-winters",
            periods=4,
            alpha=0.3,
            beta=0.1,
            gamma=0.1,
            seasonal_period=4,
            seasonal_type="additive",
        )

        assert len(result["forecasts"]) == 4
        assert len(result["actuals"]) == 4
        assert isinstance(result["rmse"], float)
        assert isinstance(result["mae"], float)


class TestCLI:
    """Tests for CLI functionality."""

    def test_main_double_method(self, capsys, monkeypatch):
        """Test CLI with double exponential smoothing."""
        args = [
            "forecast",
            "--data",
            "10,12,14,16,18,20",
            "--method",
            "double",
            "--periods",
            "2",
        ]
        monkeypatch.setattr(sys, "argv", args)

        result = main()
        assert result == 0

        captured = capsys.readouterr()
        assert "FORECAST" in captured.out

    def test_main_json_output(self, capsys, monkeypatch):
        """Test CLI with JSON output format."""
        args = [
            "forecast",
            "--data",
            "10,20,30,40",
            "--method",
            "simple",
            "--periods",
            "2",
            "--format",
            "json",
        ]
        monkeypatch.setattr(sys, "argv", args)

        result = main()
        assert result == 0

        captured = capsys.readouterr()
        output = json.loads(captured.out)
        assert "forecasts" in output
        assert "metrics" in output
        assert len(output["forecasts"]) == 2

    def test_main_csv_output(self, capsys, monkeypatch):
        """Test CLI with CSV output format."""
        args = [
            "forecast",
            "--data",
            "10,20,30",
            "--method",
            "simple",
            "--periods",
            "2",
            "--format",
            "csv",
        ]
        monkeypatch.setattr(sys, "argv", args)

        result = main()
        assert result == 0

        captured = capsys.readouterr()
        assert "period,type,value,lower,upper" in captured.out

    def test_main_backtest(self, capsys, monkeypatch):
        """Test CLI with backtest mode."""
        args = [
            "forecast",
            "--data",
            "10,12,14,16,18,20,22,24",
            "--method",
            "simple",
            "--periods",
            "2",
            "--backtest",
            "2",
        ]
        monkeypatch.setattr(sys, "argv", args)

        result = main()
        assert result == 0

        captured = capsys.readouterr()
        assert "BACKTEST" in captured.out
        assert "RMSE" in captured.out

    def test_main_backtest_json(self, capsys, monkeypatch):
        """Test CLI with backtest mode and JSON output."""
        args = [
            "forecast",
            "--data",
            "10,12,14,16,18,20,22,24",
            "--method",
            "double",
            "--periods",
            "3",
            "--backtest",
            "3",
            "--format",
            "json",
        ]
        monkeypatch.setattr(sys, "argv", args)

        result = main()
        assert result == 0

        captured = capsys.readouterr()
        output = json.loads(captured.out)
        assert "forecasts" in output
        assert "actuals" in output
        assert "rmse" in output
        assert "mae" in output
        assert len(output["forecasts"]) == 3
        assert len(output["actuals"]) == 3

    def test_main_invalid_alpha(self, capsys, monkeypatch):
        """Test CLI with invalid alpha parameter."""
        args = [
            "forecast",
            "--data",
            "10,20,30",
            "--method",
            "simple",
            "--periods",
            "1",
            "--alpha",
            "1.5",
        ]
        monkeypatch.setattr(sys, "argv", args)

        result = main()
        assert result == 1

        captured = capsys.readouterr()
        assert "alpha must be in" in captured.err

    def test_main_invalid_confidence(self, capsys, monkeypatch):
        """Test CLI with invalid confidence level."""
        args = [
            "forecast",
            "--data",
            "10,20,30",
            "--method",
            "simple",
            "--periods",
            "1",
            "--confidence",
            "1.5",
        ]
        monkeypatch.setattr(sys, "argv", args)

        result = main()
        assert result == 1

        captured = capsys.readouterr()
        assert "confidence must be in" in captured.err

    def test_main_invalid_beta(self, capsys, monkeypatch):
        """Test CLI with invalid beta parameter."""
        args = [
            "forecast",
            "--data",
            "10,20,30",
            "--method",
            "double",
            "--periods",
            "1",
            "--beta",
            "2.0",
        ]
        monkeypatch.setattr(sys, "argv", args)

        result = main()
        assert result == 1

        captured = capsys.readouterr()
        assert "beta must be in" in captured.err

    def test_main_invalid_gamma(self, capsys, monkeypatch):
        """Test CLI with invalid gamma parameter."""
        # Need enough data for Holt-Winters
        data = ",".join(str(x) for x in ([10, 20, 15, 25] * 6))
        args = [
            "forecast",
            "--data",
            data,
            "--method",
            "holt-winters",
            "--seasonal-period",
            "4",
            "--periods",
            "1",
            "--gamma",
            "0.0",
        ]
        monkeypatch.setattr(sys, "argv", args)

        result = main()
        assert result == 1

        captured = capsys.readouterr()
        assert "gamma must be in" in captured.err

    def test_main_no_data(self, capsys, monkeypatch):
        """Test CLI with empty data."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("")
            temp_path = f.name

        try:
            args = [
                "forecast",
                "--data",
                temp_path,
                "--method",
                "simple",
                "--periods",
                "1",
            ]
            monkeypatch.setattr(sys, "argv", args)

            result = main()
            assert result == 1

            captured = capsys.readouterr()
            assert "No data loaded" in captured.err
        finally:
            Path(temp_path).unlink()

    def test_main_file_not_found(self, capsys, monkeypatch):
        """Test CLI with non-existent CSV file that triggers FileNotFoundError."""
        # Create a CSV filename that looks like a path (not parseable as CSV values)
        # treat as a file to open
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=True) as f:
            # File is created with context manager, which deletes on exit
            temp_path = f.name

        # File does not exist but has .csv extension, so the script will try to open
        args = ["forecast", "--data", temp_path, "--method", "simple", "--periods", "1"]
        monkeypatch.setattr(sys, "argv", args)

        result = main()
        assert result == 1

        captured = capsys.readouterr()
        # Could be FileNotFoundError or wrapped in ValueError
        assert "Error" in captured.err

    def test_main_file_not_found_explicit(self, capsys, monkeypatch):
        """Test CLI with FileNotFoundError explicitly raised."""
        # Create a file that exists to pass the path.exists() check
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("10\n20\n30\n")
            temp_path = f.name

        try:
            args = [
                "forecast",
                "--data",
                temp_path,
                "--method",
                "simple",
                "--periods",
                "1",
            ]
            monkeypatch.setattr(sys, "argv", args)

            # Mock open to raise FileNotFoundError after path.exists() passes
            with patch("builtins.open", side_effect=FileNotFoundError("File deleted")):
                result = main()
                assert result == 1

                captured = capsys.readouterr()
                assert "File not found" in captured.err
        finally:
            Path(temp_path).unlink()

    def test_main_holt_winters(self, capsys, monkeypatch):
        """Test CLI with Holt-Winters method."""
        # Create data with at least 2*seasonal_period points
        data = ",".join(str(x) for x in ([10, 20, 15, 25] * 6))
        args = [
            "forecast",
            "--data",
            data,
            "--method",
            "holt-winters",
            "--seasonal-period",
            "4",
            "--periods",
            "4",
        ]
        monkeypatch.setattr(sys, "argv", args)

        result = main()
        assert result == 0

        captured = capsys.readouterr()
        assert "FORECAST" in captured.out

    def test_main_unexpected_error(self, capsys, monkeypatch):
        """Test CLI with unexpected error (generic Exception handler)."""
        args = [
            "forecast",
            "--data",
            "10,20,30",
            "--method",
            "simple",
            "--periods",
            "1",
        ]
        monkeypatch.setattr(sys, "argv", args)

        # Mock simple_exponential_smoothing to raise an unexpected exception
        with patch(
            "src.utils.forecast_time_series.simple_exponential_smoothing",
            side_effect=RuntimeError("Unexpected error"),
        ):
            result = main()
            assert result == 2  # Generic exception handler returns 2

            captured = capsys.readouterr()
            assert "Unexpected error" in captured.err


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
