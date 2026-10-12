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

from src.utils.anova import f_sf
from src.utils.normal_gaussian import normal_ppf


def centered_moving_average(data, period):
    """
    Centered moving average of `data` over `period` observations.

    Returns a list the same length as `data`, with `None` where the window would
    run off either end. An even period uses the 2 x period weighting (half weight
    on the two end points) so the average stays centered on an observation
    rather than falling between two of them.

    A moving average one full season wide averages every phase of the cycle
    exactly once, so it removes seasonality and leaves trend. That is what makes
    it the right baseline to measure seasonal indices against: subtracting it
    cannot leave trend behind in the indices.
    """
    if period < 1:
        raise ValueError("seasonal period must be >= 1")

    n = len(data)
    half = period // 2
    result = [None] * n

    if period % 2 == 1:
        for t in range(half, n - half):
            result[t] = statistics.fmean(data[t - half : t + half + 1])
    else:
        for t in range(half, n - half):
            window = data[t - half : t + half + 1]
            interior = sum(window[1:-1])
            result[t] = (0.5 * window[0] + interior + 0.5 * window[-1]) / period

    return result


def trend_estimate(data, period):
    """
    Trend level at every observation, with no gaps.

    A centered moving average supplies the interior, where it tracks curvature
    the way a single fitted line cannot. It is undefined for the first and last
    half-season, so those are filled by extending a line fitted through the
    moving average itself.

    Dropping those ends instead would discard the most recent half-season from
    the seasonal estimate, which is the worst half to lose: when a seasonal
    shape is changing, that is the part describing its current form.
    """
    smoothed = centered_moving_average(data, period)
    defined = [t for t, level in enumerate(smoothed) if level is not None]

    slope, intercept = statistics.linear_regression(
        defined, [smoothed[t] for t in defined]
    )

    return [
        smoothed[t] if smoothed[t] is not None else intercept + slope * t
        for t in range(len(data))
    ]


def seasonal_indices(data, period, seasonal_type="additive"):
    """
    Estimate seasonal indices by classical decomposition.

    Each observation is compared against the centered moving average at the same
    point, so the trend is removed before the seasonal effect is measured. The
    per-phase averages are then normalized: additive indices sum to zero and
    multiplicative indices average to one, which is what makes them pure
    seasonal effects rather than a mixture of season and level.

    Returns a list of `period` indices, where element i applies to observations
    at positions i, i + period, i + 2 * period, ...
    """
    if period < 2:
        raise ValueError("seasonal period must be >= 2 to estimate seasonality")
    if len(data) < 2 * period:
        raise ValueError(
            f"estimating seasonality requires at least {2 * period} data points"
        )

    trend = trend_estimate(data, period)

    if seasonal_type == "multiplicative" and any(level <= 0 for level in trend):
        raise ValueError(
            "multiplicative seasonality requires strictly positive data; "
            "the detrended ratio is undefined where the trend is zero or negative"
        )

    by_phase = [[] for _ in range(period)]
    for t, (value, level) in enumerate(zip(data, trend)):
        if seasonal_type == "additive":
            by_phase[t % period].append(value - level)
        else:
            by_phase[t % period].append(value / level)

    raw = [statistics.fmean(phase) for phase in by_phase]

    # Normalize so the indices carry only seasonality: anything common to every
    # phase belongs to the level, not to the season.
    if seasonal_type == "additive":
        offset = statistics.fmean(raw)
        return [value - offset for value in raw]

    scale = statistics.fmean(raw)
    return [value / scale for value in raw]


def seasonality_p_value(data, period, seasonal_type="additive"):
    """
    Probability of seeing this much phase-to-phase variation with no seasonality.

    One-way ANOVA across the seasonal phases of the detrended series: if the
    position in the cycle carries no information, the phase means differ only by
    noise, and the ratio of between-phase to within-phase variance follows
    F(period - 1, n - period). A small p-value is evidence of real seasonality.

    The grouped values are residuals from a fitted trend rather than independent
    draws, so this is a screen for whether a seasonal term is worth carrying at
    all, not an exact test.
    """
    trend = trend_estimate(data, period)

    if seasonal_type == "additive":
        detrended = [value - level for value, level in zip(data, trend)]
    else:
        detrended = [value / level for value, level in zip(data, trend)]

    by_phase = [detrended[i::period] for i in range(period)]
    grand_mean = statistics.fmean(detrended)

    between = sum(
        len(phase) * (statistics.fmean(phase) - grand_mean) ** 2 for phase in by_phase
    )
    within = sum(
        (value - statistics.fmean(phase)) ** 2 for phase in by_phase for value in phase
    )

    df_between = period - 1
    df_within = len(detrended) - period

    if within == 0:
        # A perfectly repeating pattern: seasonal unless the series is constant.
        return 0.0 if between > 0 else 1.0

    f_stat = (between / df_between) / (within / df_within)
    return f_sf(f_stat, df_between, df_within)


def initial_components(data, period, seasonal_type="additive", seasonality_alpha=0.05):
    """
    Starting level, trend, and seasonal indices for Holt-Winters.

    The series is deseasonalized with the indices from `seasonal_indices`, and a
    least-squares line through the deseasonalized data supplies level and trend.
    Level is the fitted value one step *before* the first observation, which is
    where the smoothing recursion starts from.

    Seasonal indices are only estimated when `seasonality_p_value` clears
    `seasonality_alpha`. Without that gate, a series carrying no seasonality
    still gets indices fitted to its noise, and with only a handful of cycles to
    average they persist through the forecast: that cost 19% median accuracy on
    a series of level shifts. `seasonality_alpha=1` always estimates them and
    `0` never does.
    """
    if not 0 <= seasonality_alpha <= 1:
        raise ValueError(
            f"seasonality_alpha must be in [0, 1], got {seasonality_alpha}"
        )

    # The > 0 guard keeps seasonality_alpha=0 meaning "never", since a perfectly
    # repeating pattern has a p-value of exactly 0 and would otherwise pass.
    if (
        seasonality_alpha > 0
        and seasonality_p_value(data, period, seasonal_type) <= seasonality_alpha
    ):
        seasonal = seasonal_indices(data, period, seasonal_type)
    else:
        # Neutral indices leave the recursion as level plus trend, which is
        # Holt's method — the right model for a series with no seasonality.
        seasonal = [0.0] * period if seasonal_type == "additive" else [1.0] * period

    if seasonal_type == "additive":
        deseasonalized = [value - seasonal[t % period] for t, value in enumerate(data)]
    else:
        deseasonalized = [value / seasonal[t % period] for t, value in enumerate(data)]

    slope, intercept = statistics.linear_regression(range(len(data)), deseasonalized)

    # The recursion's first forecast is level + trend + seasonal[0], so level has
    # to sit at t = -1 for that to land on the fitted line at t = 0.
    return intercept - slope, slope, seasonal


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
    # A least-squares line through the whole series initializes level and trend.
    # Taking the trend from data[1] - data[0] instead would set it from two noisy
    # observations, and that error decays only over roughly 1 / (alpha * beta)
    # periods, inflating the residuals it is estimated from along the way.
    # Level sits one step before the first observation, since the recursion's
    # first forecast is level + trend.
    slope, intercept = statistics.linear_regression(range(len(data)), data)
    trend = slope
    level = intercept - slope

    for value in data:
        fitted.append(level + trend)
        new_level = alpha * value + (1 - alpha) * (level + trend)
        trend = beta * (new_level - level) + (1 - beta) * trend
        level = new_level

    return fitted, level, trend


def holt_winters(
    data,
    alpha=0.3,
    beta=0.1,
    gamma=0.1,
    seasonal_period=12,
    seasonal_type="additive",
    seasonality_alpha=0.05,
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
        seasonality_alpha: Significance level the seasonal screen must clear
            before indices are estimated; see `initial_components`

    Returns:
        Tuple of (fitted_values, level, trend, seasonal_components)
    """
    if len(data) < 2 * seasonal_period:
        raise ValueError(
            f"Holt-Winters requires at least {2 * seasonal_period} data points"
        )

    level, trend, seasonal = initial_components(
        data, seasonal_period, seasonal_type, seasonality_alpha
    )

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


def residual_sigma(residuals, n_params=0, burn_in=0):
    """
    One-step-ahead error standard deviation, corrected for fitted parameters.

    `burn_in` drops leading residuals produced while the smoothing recursion is
    still settling from its initial state; those reflect the initialization
    rather than the model's accuracy. `n_params` is the number of fitted
    quantities, removed from the denominator so the estimate is not biased low
    by the parameters that were chosen to make these residuals small.
    """
    usable = residuals[burn_in:]
    if not usable:
        raise ValueError("no residuals left after burn-in")

    dof = len(usable) - n_params
    if dof < 1:
        raise ValueError(
            f"not enough residuals ({len(usable)}) to estimate error "
            f"with {n_params} fitted parameters"
        )

    return math.sqrt(sum(r * r for r in usable) / dof)


def horizon_std_multipliers(
    periods, method, alpha, beta=0.0, gamma=0.0, seasonal_period=1
):
    """
    Per-horizon multipliers on the one-step error standard deviation.

    Forecast uncertainty grows with the horizon at a rate set by the model, not
    by a single universal rule. These are the standard additive-error
    exponential smoothing variance expressions: each future shock reaches the
    forecast through the level, and through the trend and seasonal terms where
    the model carries them.

        simple         var_h / sigma^2 = 1 + (h - 1) * alpha^2
        double         c_j = alpha * (1 + j * beta)
        holt-winters   c_j = alpha * (1 + j * beta) + gamma  (every seasonal_period steps)
                       var_h / sigma^2 = 1 + sum of c_j^2 for j = 1 .. h-1

    For a random walk these reduce to sqrt(h); for a model carrying a trend they
    grow faster, which is why one shared sqrt(h) rule understates the longer
    horizons of the trending and seasonal methods.

    Multiplicative seasonality has no closed form here, and reuses the
    additive-seasonal expression as an approximation.
    """
    if periods < 0:
        raise ValueError("periods must be >= 0")

    multipliers = []
    for h in range(1, periods + 1):
        variance_ratio = 1.0
        for j in range(1, h):
            if method == "simple":
                c = alpha
            elif method == "double":
                c = alpha * (1 + j * beta)
            else:  # holt-winters
                c = alpha * (1 + j * beta)
                if seasonal_period > 0 and j % seasonal_period == 0:
                    c += gamma
            variance_ratio += c * c
        multipliers.append(math.sqrt(variance_ratio))

    return multipliers


def prediction_intervals(forecasts, sigma, multipliers, confidence=0.95):
    """
    Symmetric prediction intervals around each forecast.

    `sigma` is the one-step error standard deviation and `multipliers` scales it
    to each horizon — see `horizon_std_multipliers`. Both are required rather
    than defaulted, since a wrong horizon rule is invisible in the output.
    """
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    if len(multipliers) != len(forecasts):
        raise ValueError(
            f"got {len(multipliers)} horizon multipliers for {len(forecasts)} forecasts"
        )

    # Two-sided interval: a 95% interval needs the 97.5th percentile.
    z = normal_ppf(1 - (1 - confidence) / 2)

    intervals = []
    for forecast, multiplier in zip(forecasts, multipliers):
        margin = z * sigma * multiplier
        intervals.append((forecast - margin, forecast + margin))

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
    data,
    fitted,
    forecasts,
    intervals,
    metrics,
    format_type="table",
    precision=2,
    confidence=0.95,
    seasonality=None,
):
    """Format output in table, JSON, or CSV format."""
    if format_type == "json":
        output = {
            "fitted": [round(f, precision) for f in fitted],
            "forecasts": [round(f, precision) for f in forecasts],
            "confidence": confidence,
            "prediction_intervals": [
                {"lower": round(x, precision), "upper": round(u, precision)}
                for x, u in intervals
            ],
            "metrics": {k: round(v, precision) for k, v in metrics.items()},
        }
        if seasonality is not None:
            output["seasonality"] = seasonality
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

        if seasonality is not None:
            if seasonality["detected"]:
                verdict = "detected"
            else:
                verdict = "not detected — forecasting level and trend only"
            output.append(
                f"\nSeasonality: {verdict} (p = {seasonality['p_value']:.4f})"
            )

        # Forecasts
        output.append(
            f"\nForecast ({len(forecasts)} periods), "
            f"{confidence:.0%} prediction intervals:"
        )
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
        seasonality_alpha = kwargs.get("seasonality_alpha", 0.05)
        fitted, level, trend, seasonal = holt_winters(
            train,
            alpha,
            beta,
            gamma,
            seasonal_period,
            seasonal_type,
            seasonality_alpha,
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
    parser.add_argument(
        "--seasonality-alpha",
        type=float,
        default=0.05,
        help=(
            "Significance level the seasonal screen must clear before "
            "Holt-Winters estimates seasonal indices; 1 always estimates them, "
            "0 never does (default: 0.05)"
        ),
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
        if not 0 < args.confidence < 1:
            print("Error: confidence must be in (0, 1)", file=sys.stderr)
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
                seasonality_alpha=args.seasonality_alpha,
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
                args.seasonality_alpha,
            )
            forecasts = forecast_holt_winters(
                level,
                trend,
                seasonal,
                args.periods,
                args.seasonal_period,
                args.seasonal_type,
            )

        # Calculate metrics and intervals. The leading residuals are dropped:
        # they measure the initialization settling in, not forecast accuracy.
        residuals = calculate_residuals(data, fitted)
        metrics = calculate_metrics(residuals)

        if args.method == "simple":
            burn_in, n_params = 1, 1
        elif args.method == "double":
            burn_in, n_params = 2, 2
        else:  # holt-winters
            burn_in, n_params = args.seasonal_period, 3

        sigma = residual_sigma(residuals, n_params=n_params, burn_in=burn_in)
        multipliers = horizon_std_multipliers(
            args.periods,
            args.method,
            args.alpha,
            args.beta,
            args.gamma,
            args.seasonal_period,
        )
        intervals = prediction_intervals(forecasts, sigma, multipliers, args.confidence)

        # Report whether the seasonal term earned its place, so a Holt-Winters
        # run that forecast level and trend alone does not look unexplained.
        seasonality = None
        if args.method == "holt-winters":
            p_value = seasonality_p_value(
                data, args.seasonal_period, args.seasonal_type
            )
            seasonality = {
                "p_value": p_value,
                "detected": p_value <= args.seasonality_alpha,
            }

        # Output results
        output = format_output(
            data,
            fitted,
            forecasts,
            intervals,
            metrics,
            args.output_format,
            args.precision,
            args.confidence,
            seasonality,
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
