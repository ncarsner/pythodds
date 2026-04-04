#!/usr/bin/env python3
"""
Time Series Forecasting with Prediction Intervals

Fits exponential smoothing models to time series data and generates point forecasts
with prediction intervals. Supports simple, double, and Holt-Winters exponential smoothing.
"""

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path


def simple_exponential_smoothing(data, alpha=0.3):
    """
    Simple exponential smoothing (level only).

    Args:
        data: List of numeric values
        alpha: Smoothing parameter (0 < alpha <= 1)

    Returns:
        Tuple of (fitted_values, level)
    """
    if not data:
        raise ValueError("Data cannot be empty")

    fitted = []
    level = data[0]  # Initialize level with first observation

    for value in data:
        fitted.append(level)
        level = alpha * value + (1 - alpha) * level

    return fitted, level


def double_exponential_smoothing(data, alpha=0.3, beta=0.1):
    """
    Double exponential smoothing (Holt's method: level + trend).

    Args:
        data: List of numeric values
        alpha: Level smoothing parameter (0 < alpha <= 1)
        beta: Trend smoothing parameter (0 < beta <= 1)

    Returns:
        Tuple of (fitted_values, level, trend)
    """
    if len(data) < 2:
        raise ValueError("Double exponential smoothing requires at least 2 data points")

    fitted = []
    level = data[0]
    trend = data[1] - data[0] if len(data) > 1 else 0

    for value in data:
        fitted.append(level + trend)
        new_level = alpha * value + (1 - alpha) * (level + trend)
        trend = beta * (new_level - level) + (1 - beta) * trend
        level = new_level

    return fitted, level, trend


def holt_winters(
    data, alpha=0.3, beta=0.1, gamma=0.1, seasonal_period=12, seasonal_type="additive"
):
    """
    Holt-Winters exponential smoothing (level + trend + seasonal).

    Args:
        data: List of numeric values
        alpha: Level smoothing parameter (0 < alpha <= 1)
        beta: Trend smoothing parameter (0 < beta <= 1)
        gamma: Seasonal smoothing parameter (0 < gamma <= 1)
        seasonal_period: Number of periods in a season
        seasonal_type: 'additive' or 'multiplicative'

    Returns:
        Tuple of (fitted_values, level, trend, seasonal_components)
    """
    if len(data) < 2 * seasonal_period:
        raise ValueError(
            f"Holt-Winters requires at least {2 * seasonal_period} data points"
        )

    # Initialize components
    # Level: average of first seasonal period
    level = sum(data[:seasonal_period]) / seasonal_period

    # Trend: average change between first two seasonal periods
    trend = sum(
        (data[seasonal_period + i] - data[i]) for i in range(seasonal_period)
    ) / (seasonal_period**2)

    # Seasonal: initial seasonal indices
    seasonal = []
    for i in range(seasonal_period):
        season_avg = sum(
            data[j] for j in range(i, len(data), seasonal_period) if j < len(data)
        ) / len([j for j in range(i, len(data), seasonal_period) if j < len(data)])
        if seasonal_type == "additive":
            seasonal.append(season_avg - level)
        else:  # multiplicative
            seasonal.append(season_avg / level if level != 0 else 1)

    fitted = []

    for idx, value in enumerate(data):
        season_idx = idx % seasonal_period

        # Forecast
        if seasonal_type == "additive":
            forecast = level + trend + seasonal[season_idx]
        else:  # multiplicative
            forecast = (level + trend) * seasonal[season_idx]

        fitted.append(forecast)

        # Update components
        if seasonal_type == "additive":
            new_level = alpha * (value - seasonal[season_idx]) + (1 - alpha) * (
                level + trend
            )
            trend = beta * (new_level - level) + (1 - beta) * trend
            seasonal[season_idx] = (
                gamma * (value - new_level) + (1 - gamma) * seasonal[season_idx]
            )
        else:  # multiplicative
            new_level = alpha * (
                value / seasonal[season_idx] if seasonal[season_idx] != 0 else value
            ) + (1 - alpha) * (level + trend)
            trend = beta * (new_level - level) + (1 - beta) * trend
            seasonal[season_idx] = (
                gamma * (value / new_level if new_level != 0 else 1)
                + (1 - gamma) * seasonal[season_idx]
            )

        level = new_level

    return fitted, level, trend, seasonal


def forecast_simple(level, periods):
    """Generate forecasts using simple exponential smoothing."""
    return [level] * periods


def forecast_double(level, trend, periods):
    """Generate forecasts using double exponential smoothing."""
    return [level + (h + 1) * trend for h in range(periods)]


def forecast_holt_winters(
    level, trend, seasonal, periods, seasonal_period, seasonal_type="additive"
):
    """Generate forecasts using Holt-Winters."""
    forecasts = []
    for h in range(periods):
        season_idx = h % seasonal_period
        if seasonal_type == "additive":
            forecasts.append(level + (h + 1) * trend + seasonal[season_idx])
        else:  # multiplicative
            forecasts.append((level + (h + 1) * trend) * seasonal[season_idx])
    return forecasts


def calculate_residuals(data, fitted):
    """Calculate residuals between actual and fitted values."""
    return [actual - fit for actual, fit in zip(data, fitted)]


def calculate_metrics(residuals):
    """Calculate forecast accuracy metrics."""
    n = len(residuals)
    if n == 0:
        return {"rmse": 0, "mae": 0, "std": 0}

    mae = sum(abs(r) for r in residuals) / n
    mse = sum(r**2 for r in residuals) / n
    rmse = math.sqrt(mse)
    std = statistics.stdev(residuals) if n > 1 else 0

    return {"rmse": rmse, "mae": mae, "std": std}


def prediction_intervals(forecasts, residual_std, alpha=0.95):
    """
    Calculate prediction intervals for forecasts.

    Uses residual standard deviation to construct symmetric intervals.
    For multi-step forecasts, uncertainty increases with horizon.
    """
    # Z-score for confidence level using normal approximation
    z_scores = {0.80: 1.282, 0.90: 1.645, 0.95: 1.960, 0.99: 2.576}
    z = z_scores.get(alpha, 1.960)

    intervals = []
    for h, forecast in enumerate(forecasts, start=1):
        # Uncertainty grows with forecast horizon
        se = residual_std * math.sqrt(h)
        lower = forecast - z * se
        upper = forecast + z * se
        intervals.append((lower, upper))

    return intervals


def load_data(data_arg):
    """Load data from CSV file or comma-separated values."""
    path = Path(data_arg)

    if path.exists() and path.suffix == ".csv":
        # Load from CSV file (assume single column or first numeric column)
        with open(path, "r") as f:
            reader = csv.reader(f)
            data = []
            for row in reader:
                # Skip header if non-numeric
                try:
                    data.append(float(row[0]))
                except (ValueError, IndexError):
                    continue
            return data
    else:
        # Parse as comma-separated values
        try:
            return [float(x.strip()) for x in data_arg.split(",")]
        except ValueError as e:
            raise ValueError(f"Could not parse data: {e}")


def format_output(
    data, fitted, forecasts, intervals, metrics, format_type="table", precision=2
):
    """Format output in table, JSON, or CSV format."""
    if format_type == "json":
        output = {
            "fitted": [round(f, precision) for f in fitted],
            "forecasts": [round(f, precision) for f in forecasts],
            "prediction_intervals": [
                {"lower": round(x, precision), "upper": round(u, precision)}
                for x, u in intervals
            ],
            "metrics": {k: round(v, precision) for k, v in metrics.items()},
        }
        return json.dumps(output, indent=2)

    elif format_type == "csv":
        lines = ["period,type,value,lower,upper"]

        # Historical fitted values
        for i, (actual, fit) in enumerate(zip(data, fitted), start=1):
            lines.append(f"{i},fitted,{fit:.{precision}f},,")

        # Forecasts
        for i, (forecast, (lower, upper)) in enumerate(
            zip(forecasts, intervals), start=len(data) + 1
        ):
            lines.append(
                f"{i},forecast,{forecast:.{precision}f},{lower:.{precision}f},{upper:.{precision}f}"
            )

        return "\n".join(lines)

    else:  # table
        output = []
        output.append("\n" + "=" * 60)
        output.append("TIME SERIES FORECAST")
        output.append("=" * 60)

        # Metrics
        output.append("\nIn-Sample Metrics:")
        output.append(f"  RMSE: {metrics['rmse']:.{precision}f}")
        output.append(f"  MAE:  {metrics['mae']:.{precision}f}")
        output.append(f"  Residual Std: {metrics['std']:.{precision}f}")

        # Forecasts
        output.append(f"\nForecast ({len(forecasts)} periods):")
        output.append("-" * 60)
        output.append(f"{'Period':<8} {'Forecast':>12} {'Lower':>12} {'Upper':>12}")
        output.append("-" * 60)

        for i, (forecast, (lower, upper)) in enumerate(
            zip(forecasts, intervals), start=1
        ):
            output.append(
                f"{i:<8} {forecast:>12.{precision}f} {lower:>12.{precision}f} {upper:>12.{precision}f}"
            )

        output.append("=" * 60 + "\n")

        return "\n".join(output)


def backtest(data, method, periods, **kwargs):
    """
    Backtest by holding out last `periods` observations and comparing
    forecasts to actual values.
    """
    if periods >= len(data):
        raise ValueError("Backtest periods must be less than data length")

    train = data[:-periods]
    test = data[-periods:]

    # Fit on training data
    if method == "simple":
        alpha = kwargs.get("alpha", 0.3)
        fitted, level = simple_exponential_smoothing(train, alpha)
        forecasts = forecast_simple(level, periods)
    elif method == "double":
        alpha = kwargs.get("alpha", 0.3)
        beta = kwargs.get("beta", 0.1)
        fitted, level, trend = double_exponential_smoothing(train, alpha, beta)
        forecasts = forecast_double(level, trend, periods)
    else:  # holt-winters
        alpha = kwargs.get("alpha", 0.3)
        beta = kwargs.get("beta", 0.1)
        gamma = kwargs.get("gamma", 0.1)
        seasonal_period = kwargs.get("seasonal_period", 12)
        seasonal_type = kwargs.get("seasonal_type", "additive")
        fitted, level, trend, seasonal = holt_winters(
            train, alpha, beta, gamma, seasonal_period, seasonal_type
        )
        forecasts = forecast_holt_winters(
            level, trend, seasonal, periods, seasonal_period, seasonal_type
        )

    # Calculate errors on test set
    errors = [actual - forecast for actual, forecast in zip(test, forecasts)]
    rmse = math.sqrt(sum(e**2 for e in errors) / len(errors))
    mae = sum(abs(e) for e in errors) / len(errors)

    return {"forecasts": forecasts, "actuals": test, "rmse": rmse, "mae": mae}


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Time Series Forecasting with Prediction Intervals",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Simple exponential smoothing, 6-period forecast
  forecast --data 120,135,148,130,142,155,160 --method simple --periods 6

  # Double exponential smoothing from CSV file
  forecast --data sales.csv --method double --periods 12

  # Holt-Winters with weekly seasonality
  forecast --data data.csv --method holt-winters --seasonal-period 7 --periods 14

  # Backtest: evaluate accuracy on last 4 observations
  forecast --data data.csv --method double --periods 4 --backtest 4
        """,
    )

    parser.add_argument(
        "--data", required=True, help="CSV file path or comma-separated values"
    )
    parser.add_argument(
        "--method",
        choices=["simple", "double", "holt-winters"],
        default="simple",
        help="Forecasting method (default: simple)",
    )
    parser.add_argument(
        "--periods",
        type=int,
        default=1,
        help="Number of periods to forecast (default: 1)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.3,
        help="Level smoothing parameter (default: 0.3)",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.1,
        help="Trend smoothing parameter (default: 0.1)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.1,
        help="Seasonal smoothing parameter (default: 0.1)",
    )
    parser.add_argument(
        "--seasonal-period",
        type=int,
        default=12,
        help="Seasonal period for Holt-Winters (default: 12)",
    )
    parser.add_argument(
        "--seasonal-type",
        choices=["additive", "multiplicative"],
        default="additive",
        help="Seasonal component type (default: additive)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="Confidence level for prediction intervals (default: 0.95)",
    )
    parser.add_argument(
        "--format",
        choices=["table", "json", "csv"],
        default="table",
        dest="output_format",
        help="Output format (default: table)",
    )
    parser.add_argument(
        "--precision", type=int, default=2, help="Decimal precision (default: 2)"
    )
    parser.add_argument(
        "--backtest",
        type=int,
        metavar="K",
        help="Hold out last K observations for backtesting",
    )

    args = parser.parse_args()

    try:
        # Load data
        data = load_data(args.data)

        if not data:
            print("Error: No data loaded", file=sys.stderr)
            return 1

        # Validate parameters
        if not 0 < args.alpha <= 1:
            print("Error: alpha must be in (0, 1]", file=sys.stderr)
            return 1
        if not 0 < args.beta <= 1:
            print("Error: beta must be in (0, 1]", file=sys.stderr)
            return 1
        if not 0 < args.gamma <= 1:
            print("Error: gamma must be in (0, 1]", file=sys.stderr)
            return 1
        if args.confidence < 0.5 or args.confidence >= 1.0:
            print("Error: confidence must be in [0.5, 1.0)", file=sys.stderr)
            return 1

        # Backtest mode
        if args.backtest:
            result = backtest(
                data,
                args.method,
                args.backtest,
                alpha=args.alpha,
                beta=args.beta,
                gamma=args.gamma,
                seasonal_period=args.seasonal_period,
                seasonal_type=args.seasonal_type,
            )

            if args.output_format == "json":
                output = {
                    "forecasts": [
                        round(f, args.precision) for f in result["forecasts"]
                    ],
                    "actuals": [round(a, args.precision) for a in result["actuals"]],
                    "rmse": round(result["rmse"], args.precision),
                    "mae": round(result["mae"], args.precision),
                }
                print(json.dumps(output, indent=2))
            else:
                print("\n" + "=" * 60)
                print("BACKTEST RESULTS")
                print("=" * 60)
                print(f"\nHold-out period: last {args.backtest} observations")
                print(f"Method: {args.method}")
                print(f"\n{'Period':<8} {'Actual':>12} {'Forecast':>12} {'Error':>12}")
                print("-" * 60)
                for i, (actual, forecast) in enumerate(
                    zip(result["actuals"], result["forecasts"]), start=1
                ):
                    error = actual - forecast
                    print(
                        f"{i:<8} {actual:>12.{args.precision}f} {forecast:>12.{args.precision}f} {error:>12.{args.precision}f}"
                    )
                print("-" * 60)
                print(f"RMSE: {result['rmse']:.{args.precision}f}")
                print(f"MAE:  {result['mae']:.{args.precision}f}")
                print("=" * 60 + "\n")

            return 0

        # Fit model and forecast
        if args.method == "simple":
            fitted, level = simple_exponential_smoothing(data, args.alpha)
            forecasts = forecast_simple(level, args.periods)
        elif args.method == "double":
            fitted, level, trend = double_exponential_smoothing(
                data, args.alpha, args.beta
            )
            forecasts = forecast_double(level, trend, args.periods)
        else:  # holt-winters
            fitted, level, trend, seasonal = holt_winters(
                data,
                args.alpha,
                args.beta,
                args.gamma,
                args.seasonal_period,
                args.seasonal_type,
            )
            forecasts = forecast_holt_winters(
                level,
                trend,
                seasonal,
                args.periods,
                args.seasonal_period,
                args.seasonal_type,
            )

        # Calculate metrics and intervals
        residuals = calculate_residuals(data, fitted)
        metrics = calculate_metrics(residuals)
        intervals = prediction_intervals(forecasts, metrics["std"], args.confidence)

        # Output results
        output = format_output(
            data,
            fitted,
            forecasts,
            intervals,
            metrics,
            args.output_format,
            args.precision,
        )
        print(output)

        return 0

    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
