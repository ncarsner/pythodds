"""Tests for forecast_time_series module."""

import json
import math
import random
import statistics
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
    calculate_residuals,
    centered_moving_average,
    double_exponential_smoothing,
    forecast_double,
    forecast_holt_winters,
    forecast_simple,
    holt_winters,
    horizon_std_multipliers,
    initial_components,
    load_data,
    main,
    prediction_intervals,
    residual_sigma,
    seasonal_indices,
    simple_exponential_smoothing,
)

SEASON = [-20.0, -10.0, 0.0, 5.0, 15.0, 25.0, 30.0, 10.0, -5.0, -15.0, -25.0, -10.0]
# Centered so the pattern is pure seasonality, which is what the indices recover.
SEASON = [s - statistics.fmean(SEASON) for s in SEASON]


def linear(n, base=100.0, slope=10.0):
    """Noiseless straight line: no seasonality, so every index must come out 0."""
    return [base + slope * t for t in range(n)]


def linear_plus_season(n, base=100.0, slope=2.0, season=None):
    """Noiseless line plus an additive seasonal pattern."""
    season = SEASON if season is None else season
    return [base + slope * t + season[t % len(season)] for t in range(n)]


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


class TestCenteredMovingAverage:
    """Tests for the centered moving average used to detrend before measuring seasonality."""

    def test_odd_period_averages_a_symmetric_window(self):
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = centered_moving_average(data, 3)

        assert result[0] is None
        assert result[-1] is None
        assert result[1:4] == pytest.approx([2.0, 3.0, 4.0])

    def test_even_period_half_weights_the_end_points(self):
        # 2 x 4 weighting: (0.5*a + b + c + d + 0.5*e) / 4
        data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        result = centered_moving_average(data, 4)

        assert result[:2] == [None, None]
        assert result[-2:] == [None, None]
        assert result[2] == pytest.approx((0.5 * 1 + 2 + 3 + 4 + 0.5 * 5) / 4)

    def test_recovers_a_line_exactly(self):
        """A centered average of a straight line is the line itself."""
        data = linear(30)
        result = centered_moving_average(data, 12)

        for t, value in enumerate(result):
            if value is not None:
                assert value == pytest.approx(data[t])

    def test_period_below_one_raises(self):
        with pytest.raises(ValueError, match="seasonal period must be >= 1"):
            centered_moving_average([1.0, 2.0], 0)


class TestSeasonalIndices:
    """Tests for seasonal index estimation (#73)."""

    def test_trending_series_without_seasonality_yields_zero_indices(self):
        """Regression test for #73.

        The previous initialization compared each phase's whole-sample average
        against the first cycle's mean, so on this series — a pure line with no
        seasonality at all — it produced indices [65, 75, ... 175] summing to
        1440 instead of zero, assigning the entire trend to the season.
        """
        indices = seasonal_indices(linear(36), 12)

        assert sum(indices) == pytest.approx(0.0, abs=1e-9)
        for index in indices:
            assert index == pytest.approx(0.0, abs=1e-9)

    def test_additive_indices_sum_to_zero_and_recover_the_pattern(self):
        indices = seasonal_indices(linear_plus_season(48), 12)

        assert sum(indices) == pytest.approx(0.0, abs=1e-9)
        assert indices == pytest.approx(SEASON, abs=1e-9)

    def test_multiplicative_indices_average_to_one(self):
        factors = [
            0.88,
            0.85,
            0.97,
            0.98,
            1.02,
            1.04,
            1.03,
            1.01,
            0.97,
            0.99,
            1.08,
            1.18,
        ]
        factors = [f / statistics.fmean(factors) for f in factors]
        data = [1000.0 * 1.004**t * factors[t % 12] for t in range(48)]

        indices = seasonal_indices(data, 12, "multiplicative")

        assert statistics.fmean(indices) == pytest.approx(1.0, abs=1e-9)
        # Ratio-to-moving-average is approximate on a compounding trend.
        assert indices == pytest.approx(factors, rel=0.01)

    def test_multiplicative_rejects_non_positive_trend(self):
        data = [10.0, -10.0, 10.0, -10.0, 10.0, -10.0, 10.0, -10.0]
        with pytest.raises(ValueError, match="strictly positive"):
            seasonal_indices(data, 4, "multiplicative")

    def test_period_below_two_raises(self):
        with pytest.raises(ValueError, match="must be >= 2"):
            seasonal_indices(linear(20), 1)

    def test_too_few_observations_raises(self):
        with pytest.raises(ValueError, match="at least 24 data points"):
            seasonal_indices(linear(23), 12)


class TestInitialComponents:
    """Tests for the Holt-Winters starting state."""

    def test_level_sits_one_step_before_the_first_observation(self):
        data = linear(36, base=100.0, slope=10.0)
        level, trend, seasonal = initial_components(data, 12)

        assert trend == pytest.approx(10.0)
        assert level == pytest.approx(90.0)
        # level + trend + seasonal[0] is the first fitted value, which must land
        # on the first observation for a noiseless line.
        assert level + trend + seasonal[0] == pytest.approx(data[0])

    def test_multiplicative_deseasonalizes_by_division(self):
        factors = [0.9, 1.1, 0.95, 1.05]
        factors = [f / statistics.fmean(factors) for f in factors]
        data = [100.0 * (1 + 0.01 * t) * factors[t % 4] for t in range(24)]

        level, trend, seasonal = initial_components(data, 4, "multiplicative")

        assert statistics.fmean(seasonal) == pytest.approx(1.0, abs=1e-9)
        assert trend == pytest.approx(1.0, rel=0.05)
        assert (level + trend) * seasonal[0] == pytest.approx(data[0], rel=0.02)


class TestExactRecovery:
    """A noiseless series the model can represent must be forecast exactly (#73)."""

    def test_double_smoothing_reproduces_a_line(self):
        data = linear(36)
        fitted, level, trend = double_exponential_smoothing(data, 0.3, 0.1)
        forecasts = forecast_double(level, trend, 6)

        assert fitted == pytest.approx(data, abs=1e-9)
        assert forecasts == pytest.approx(
            [100.0 + 10.0 * t for t in range(36, 42)], abs=1e-9
        )

    @pytest.mark.parametrize("seasonal_type", ["additive", "multiplicative"])
    def test_holt_winters_reproduces_a_line(self, seasonal_type):
        """With no seasonality present, the seasonal term must not distort the trend.

        Before #73 this returned 373.4 (additive) and 330.9 (multiplicative)
        against a true 460.0, because the trend had been folded into the
        seasonal indices.
        """
        data = linear(36)
        _, level, trend, seasonal = holt_winters(data, 0.3, 0.1, 0.1, 12, seasonal_type)
        forecasts = forecast_holt_winters(level, trend, seasonal, 12, 12, seasonal_type)

        assert forecasts == pytest.approx(
            [100.0 + 10.0 * t for t in range(36, 48)], rel=1e-9
        )

    def test_holt_winters_reproduces_trend_plus_seasonality(self):
        data = linear_plus_season(48)
        train, test = data[:36], data[36:]

        _, level, trend, seasonal = holt_winters(train, 0.3, 0.1, 0.1, 12, "additive")
        forecasts = forecast_holt_winters(level, trend, seasonal, 12, 12, "additive")

        assert forecasts == pytest.approx(test, abs=1e-9)

    def test_holt_winters_multiplicative_tracks_a_compounding_series(self):
        factors = [
            0.88,
            0.85,
            0.97,
            0.98,
            1.02,
            1.04,
            1.03,
            1.01,
            0.97,
            0.99,
            1.08,
            1.18,
        ]
        factors = [f / statistics.fmean(factors) for f in factors]
        data = [1000.0 * 1.004**t * factors[t % 12] for t in range(48)]
        train, test = data[:36], data[36:]

        _, level, trend, seasonal = holt_winters(
            train, 0.3, 0.1, 0.1, 12, "multiplicative"
        )
        forecasts = forecast_holt_winters(
            level, trend, seasonal, 12, 12, "multiplicative"
        )

        # An additive trend recursion against a compounding series, with
        # ratio-to-moving-average indices: close, not exact.
        assert forecasts == pytest.approx(test, rel=0.01)


class TestResidualSigma:
    """Tests for the one-step error estimate behind interval width."""

    def test_divides_by_the_remaining_degrees_of_freedom(self):
        residuals = [1.0, -1.0, 2.0, -2.0, 1.0]
        # sum of squares = 11, dof = 5 - 2
        assert residual_sigma(residuals, n_params=2) == pytest.approx(math.sqrt(11 / 3))

    def test_burn_in_drops_leading_residuals(self):
        residuals = [100.0, 1.0, -1.0]
        assert residual_sigma(residuals, n_params=0, burn_in=1) == pytest.approx(
            math.sqrt(2 / 2)
        )

    def test_empty_after_burn_in_raises(self):
        with pytest.raises(ValueError, match="no residuals left"):
            residual_sigma([1.0, 2.0], burn_in=2)

    def test_too_few_residuals_for_the_parameter_count_raises(self):
        with pytest.raises(ValueError, match="not enough residuals"):
            residual_sigma([1.0, 2.0], n_params=3)


class TestHorizonStdMultipliers:
    """Tests for how forecast uncertainty grows with the horizon."""

    def test_simple_matches_the_closed_form(self):
        alpha = 0.3
        multipliers = horizon_std_multipliers(5, "simple", alpha)

        expected = [math.sqrt(1 + (h - 1) * alpha**2) for h in range(1, 6)]
        assert multipliers == pytest.approx(expected)

    def test_double_accumulates_the_trend_term(self):
        alpha, beta = 0.3, 0.1
        multipliers = horizon_std_multipliers(4, "double", alpha, beta)

        expected = [
            math.sqrt(1 + sum((alpha * (1 + j * beta)) ** 2 for j in range(1, h)))
            for h in range(1, 5)
        ]
        assert multipliers == pytest.approx(expected)

    def test_holt_winters_adds_gamma_once_per_season(self):
        alpha, beta, gamma, period = 0.3, 0.1, 0.2, 4
        multipliers = horizon_std_multipliers(
            6, "holt-winters", alpha, beta, gamma, period
        )

        expected = []
        for h in range(1, 7):
            variance = 1.0
            for j in range(1, h):
                c = alpha * (1 + j * beta) + (gamma if j % period == 0 else 0.0)
                variance += c * c
            expected.append(math.sqrt(variance))
        assert multipliers == pytest.approx(expected)
        # The seasonal shock only reaches horizons a full season out.
        assert multipliers[4] > expected[3]

    def test_first_horizon_is_always_one(self):
        for method in ("simple", "double", "holt-winters"):
            assert horizon_std_multipliers(1, method, 0.3, 0.1, 0.1, 12) == [1.0]

    def test_zero_periods_is_empty(self):
        assert horizon_std_multipliers(0, "simple", 0.3) == []

    def test_negative_periods_raises(self):
        with pytest.raises(ValueError, match="periods must be >= 0"):
            horizon_std_multipliers(-1, "simple", 0.3)


class TestPredictionIntervals:
    """Tests for interval construction (#73)."""

    def test_uses_the_requested_confidence_level(self):
        """Regression test for #73.

        A four-entry lookup table previously returned z = 1.960 for any level it
        did not contain, so 0.85 and 0.50 both produced the 95% interval.
        """
        forecasts = [100.0]
        widths = {}
        for confidence in (0.5, 0.85, 0.95, 0.99):
            (lower, upper) = prediction_intervals(forecasts, 10.0, [1.0], confidence)[0]
            widths[confidence] = upper - lower

        assert len(set(widths.values())) == 4
        assert widths[0.5] < widths[0.85] < widths[0.95] < widths[0.99]
        # 95% two-sided uses the 97.5th percentile.
        assert widths[0.95] == pytest.approx(2 * 1.959963985 * 10.0, rel=1e-6)

    def test_centered_on_the_forecast(self):
        intervals = prediction_intervals([50.0, 60.0], 5.0, [1.0, 2.0], 0.95)

        for forecast, (lower, upper) in zip([50.0, 60.0], intervals):
            assert (lower + upper) / 2 == pytest.approx(forecast)
        assert (intervals[1][1] - intervals[1][0]) == pytest.approx(
            2 * (intervals[0][1] - intervals[0][0])
        )

    @pytest.mark.parametrize("confidence", [0.0, 1.0, -0.1, 1.5])
    def test_confidence_outside_the_unit_interval_raises(self, confidence):
        with pytest.raises(ValueError, match="confidence must be in"):
            prediction_intervals([1.0], 1.0, [1.0], confidence)

    def test_multiplier_count_must_match_the_forecasts(self):
        with pytest.raises(ValueError, match="horizon multipliers"):
            prediction_intervals([1.0, 2.0], 1.0, [1.0], 0.95)


class TestIntervalCoverage:
    """Intervals are only honest if they contain the actual value as often as claimed."""

    @staticmethod
    def _simulate_local_level(seed, alpha, n, horizon, sigma=5.0):
        """Draw from the model simple smoothing assumes: a level driven by noise."""
        rng = random.Random(seed)
        level = 100.0
        series = []
        for _ in range(n + horizon):
            error = rng.gauss(0, sigma)
            series.append(level + error)
            level += alpha * error
        return series[:n], series[n:]

    @pytest.mark.parametrize("confidence", [0.8, 0.95])
    def test_simple_coverage_is_close_to_nominal(self, confidence):
        alpha, horizon, inside, total = 0.3, 6, 0, 0

        for seed in range(200):
            train, test = self._simulate_local_level(seed, alpha, 120, horizon)
            fitted, level = simple_exponential_smoothing(train, alpha)
            forecasts = forecast_simple(level, horizon)
            sigma = residual_sigma(
                calculate_residuals(train, fitted), n_params=1, burn_in=1
            )
            intervals = prediction_intervals(
                forecasts,
                sigma,
                horizon_std_multipliers(horizon, "simple", alpha),
                confidence,
            )
            for actual, (lower, upper) in zip(test, intervals):
                total += 1
                inside += lower <= actual <= upper

        assert inside / total == pytest.approx(confidence, abs=0.03)

    def test_holt_winters_coverage_is_close_to_nominal(self):
        """Seasonal intervals are checked too, since the seasonal term feeds their width."""
        alpha, beta, gamma, period, horizon = 0.3, 0.1, 0.1, 12, 12
        inside = total = 0

        for seed in range(150):
            rng = random.Random(seed)
            level, trend, seasonal = 100.0, 0.5, list(SEASON)
            series = []
            for t in range(120 + horizon):
                phase = t % period
                error = rng.gauss(0, 5.0)
                series.append(level + trend + seasonal[phase] + error)
                new_level = level + trend + alpha * error
                trend += alpha * beta * error
                seasonal[phase] += gamma * error
                level = new_level
            train, test = series[:120], series[120:]

            fitted, lvl, trd, seas = holt_winters(
                train, alpha, beta, gamma, period, "additive"
            )
            forecasts = forecast_holt_winters(
                lvl, trd, seas, horizon, period, "additive"
            )
            sigma = residual_sigma(
                calculate_residuals(train, fitted), n_params=3, burn_in=period
            )
            intervals = prediction_intervals(
                forecasts,
                sigma,
                horizon_std_multipliers(
                    horizon, "holt-winters", alpha, beta, gamma, period
                ),
                0.95,
            )
            for actual, (lower, upper) in zip(test, intervals):
                total += 1
                inside += lower <= actual <= upper

        # Measured at 0.93 against a nominal 0.95: the level, trend, and seasonal
        # states are all estimated from the same history, which the variance
        # expression treats as known. Mildly optimistic, and bounded.
        assert 0.90 <= inside / total <= 0.97

    def test_sqrt_h_growth_would_overstate_the_spread(self):
        """The old rule scaled every method by sqrt(h), which only suits a random walk."""
        alpha, beta, horizon = 0.3, 0.1, 12
        fitted = horizon_std_multipliers(horizon, "double", alpha, beta)
        old = [math.sqrt(h) for h in range(1, horizon + 1)]

        assert fitted[-1] < old[-1]
        # With a fast-adapting model the same rule understates it instead, so no
        # single horizon rule can serve every parameter set.
        reactive = horizon_std_multipliers(horizon, "double", 0.8, 0.3)
        assert reactive[-1] > old[-1]


class TestConfidenceReachesTheCLI:
    """The interval defect in #73 was visible from the command line."""

    DATA = "120,135,148,130,142,155,160,151,149,163"

    def _run(self, capsys, monkeypatch, confidence):
        args = [
            "forecast",
            "--data",
            self.DATA,
            "--method",
            "simple",
            "--periods",
            "2",
            "--confidence",
            str(confidence),
            "--format",
            "json",
        ]
        monkeypatch.setattr(sys, "argv", args)
        assert main() == 0
        return json.loads(capsys.readouterr().out)

    def test_different_levels_give_different_widths(self, capsys, monkeypatch):
        """Before #73, 0.50 and 0.85 both printed the 95% interval."""
        widths = {}
        for confidence in (0.5, 0.85, 0.95):
            payload = self._run(capsys, monkeypatch, confidence)
            assert payload["confidence"] == confidence
            first = payload["prediction_intervals"][0]
            widths[confidence] = first["upper"] - first["lower"]

        assert widths[0.5] < widths[0.85] < widths[0.95]

    def test_table_output_labels_the_level(self, capsys, monkeypatch):
        args = [
            "forecast",
            "--data",
            self.DATA,
            "--method",
            "simple",
            "--periods",
            "1",
            "--confidence",
            "0.85",
        ]
        monkeypatch.setattr(sys, "argv", args)
        assert main() == 0

        assert "85% prediction intervals" in capsys.readouterr().out


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
