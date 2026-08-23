#!/usr/bin/env python3
"""Command-line utility for multiple linear regression with prediction intervals.

Fits ordinary least squares across several predictors:

    y = b0 + b1*x1 + ... + bk*xk + e

and reports the full inference set -- coefficient estimates with standard
errors, t-statistics, p-values, and confidence intervals; model fit via R²,
adjusted R², residual standard error, and the overall F-test; optional variance
inflation factors; and predictions carrying both interval kinds.

The distinction between the two intervals is the point of the tool.  A
confidence interval covers the *mean* response at a set of predictor values; a
prediction interval covers a *single new observation* and is always wider,
because it must absorb the residual scatter as well as the uncertainty in the
fitted surface.

Extends ``linreg`` from one predictor to many.

Usage examples:
  # Fit from a CSV, every non-target column as a predictor
  mlreg --file housing.csv --target price

  # Chosen predictors, with multicollinearity diagnostics
  mlreg --file data.csv --target sales --features advertising headcount --vif

  # Predict new observations with 90% intervals
  mlreg --file train.csv --target output --predict-file new_inputs.csv --alpha 0.10
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from typing import Any

import numpy as np

# The exact regularised incomplete beta and its F/t tail functions already live
# in the ANOVA module, verified there against scipy.  They are imported rather
# than copied: the equivalents in linear_regression.py substitute a normal
# approximation above df = 30, which would quietly bias coefficient p-values on
# any dataset larger than a few dozen rows.
from src.utils.anova import regularized_incomplete_beta, t_sf_two_sided

# A design matrix whose condition number exceeds this is treated as rank
# deficient for reporting purposes even when numpy can still factor it.
_CONDITION_LIMIT = 1e12


# ---------------------------------------------------------------------------
# Distribution helpers
# ---------------------------------------------------------------------------


def f_sf(f_stat: float, df1: int, df2: int) -> float:
    """Upper-tail probability ``P(F > f_stat)`` for F(df1, df2).

    Evaluated straight from the incomplete-beta identity rather than as
    ``1 - cdf``.  The subtraction form cancels catastrophically in the far
    tail: for F = 1417 on (2, 37) df the CDF rounds to exactly 1.0, so
    ``1 - cdf`` reports a p-value of 0 where the true value is near 1e-35.
    Highly significant models are exactly where that matters.

    Args:
        f_stat: Observed F-statistic; must be >= 0.
        df1: Numerator degrees of freedom; must be >= 1.
        df2: Denominator degrees of freedom; must be >= 1.

    Returns:
        ``P(F > f_stat)`` in [0, 1].

    Raises:
        ValueError: If ``df1``/``df2`` < 1 or ``f_stat`` < 0.
    """
    if df1 < 1 or df2 < 1:
        raise ValueError(f"df1 and df2 must be >= 1, got df1={df1}, df2={df2}")
    if f_stat < 0:
        raise ValueError(f"f_stat must be >= 0, got {f_stat}")
    if f_stat == 0:
        return 1.0
    if math.isinf(f_stat):
        return 0.0
    return regularized_incomplete_beta(df2 / (df2 + df1 * f_stat), df2 / 2, df1 / 2)


def t_cdf(t_stat: float, df: float) -> float:
    """Cumulative distribution function of Student's t.

    Args:
        t_stat: Evaluation point.
        df: Degrees of freedom; must be > 0.

    Returns:
        ``P(T <= t_stat)`` in [0, 1].

    Raises:
        ValueError: If ``df`` <= 0.
    """
    if df <= 0:
        raise ValueError(f"df must be > 0, got {df}")
    tail = 0.5 * regularized_incomplete_beta(df / (df + t_stat * t_stat), df / 2, 0.5)
    return 1.0 - tail if t_stat > 0 else tail


def t_quantile(p: float, df: float) -> float:
    """Inverse CDF of Student's t, by bisection on :func:`t_cdf`.

    Args:
        p: Cumulative probability, strictly between 0 and 1.
        df: Degrees of freedom; must be > 0.

    Returns:
        The t with ``P(T <= t) = p``.

    Raises:
        ValueError: If ``p`` is not in (0, 1) or ``df`` <= 0.
    """
    if not 0.0 < p < 1.0:
        raise ValueError(f"probability must be strictly between 0 and 1, got {p}")
    if df <= 0:
        raise ValueError(f"df must be > 0, got {df}")

    low, high = -1e4, 1e4
    for _ in range(300):
        mid = (low + high) / 2.0
        if t_cdf(mid, df) < p:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0


# ---------------------------------------------------------------------------
# Core model
# ---------------------------------------------------------------------------


class MultipleRegression:
    """A fitted OLS model and everything derived from it."""

    def __init__(
        self,
        coefficients: list[float],
        names: list[str],
        xtx_inv: np.ndarray,
        residuals: np.ndarray,
        fitted: np.ndarray,
        observed: np.ndarray,
    ):
        """Derive inference statistics from a completed least-squares solve.

        Args:
            coefficients: Estimates, intercept first.
            names: Term names aligned with ``coefficients``.
            xtx_inv: Inverse of ``X'X`` for the design matrix with intercept.
            residuals: Observed minus fitted values.
            fitted: Fitted values.
            observed: Observed response values.
        """
        self.coefficients = coefficients
        self.names = names
        self.xtx_inv = xtx_inv
        self.residuals = residuals
        self.fitted = fitted
        self.observed = observed

        self.n = len(observed)
        self.k = len(coefficients)
        self.df = self.n - self.k

        self.sse = float(np.sum(residuals**2))
        self.sst = float(np.sum((observed - np.mean(observed)) ** 2))
        self.ssr = self.sst - self.sse

        self.mse = self.sse / self.df
        self.residual_std_error = math.sqrt(self.mse)

        self.std_errors = [
            math.sqrt(self.mse * max(value, 0.0)) for value in np.diag(xtx_inv)
        ]
        self.t_stats = [
            coef / se if se > 0 else math.inf
            for coef, se in zip(coefficients, self.std_errors)
        ]
        self.p_values = [t_sf_two_sided(t, self.df) for t in self.t_stats]

    @property
    def r_squared(self) -> float:
        """Fraction of the response variance the model explains."""
        return 1.0 - self.sse / self.sst

    @property
    def adj_r_squared(self) -> float:
        """R² penalised for the number of predictors."""
        return 1.0 - (1.0 - self.r_squared) * (self.n - 1) / self.df

    @property
    def df_model(self) -> int:
        """Numerator degrees of freedom for the overall F-test."""
        return self.k - 1

    @property
    def f_statistic(self) -> float:
        """Overall F-statistic against the intercept-only model.

        No zero-denominator guard is needed: ``fit`` rejects a model with no
        predictors and one whose response is constant, and normal-equation
        roundoff leaves the residual mean square strictly positive even on an
        exactly-linear response -- around 1e-30 rather than 0.
        """
        return (self.ssr / self.df_model) / self.mse

    @property
    def f_p_value(self) -> float:
        """p-value for the overall F-test."""
        return f_sf(self.f_statistic, self.df_model, self.df)

    def confidence_intervals(self, alpha: float) -> list[tuple[float, float]]:
        """Two-sided intervals for each coefficient.

        Args:
            alpha: Significance level, e.g. 0.05 for 95% intervals.

        Returns:
            ``(lower, upper)`` per coefficient.
        """
        crit = t_quantile(1.0 - alpha / 2.0, self.df)
        return [
            (coef - crit * se, coef + crit * se)
            for coef, se in zip(self.coefficients, self.std_errors)
        ]


def fit(
    features: list[list[float]],
    outcomes: list[float],
    names: list[str] | None = None,
) -> MultipleRegression:
    """Fit OLS with an automatically prepended intercept.

    Args:
        features: One row per observation, one value per predictor.
        outcomes: Response values, one per row.
        names: Predictor names; defaults to ``x1``..``xk``.

    Returns:
        The fitted :class:`MultipleRegression`.

    Raises:
        ValueError: If the data are empty or ragged, the row counts disagree,
            there are no residual degrees of freedom, or the predictors are
            collinear.
    """
    if not features:
        raise ValueError("no observations to fit")
    if len(features) != len(outcomes):
        raise ValueError(
            f"got {len(features)} feature rows but {len(outcomes)} responses"
        )

    width = len(features[0])
    if width == 0:
        raise ValueError("no predictor columns to fit")
    if any(len(row) != width for row in features):
        raise ValueError("all feature rows must have the same number of columns")

    design = np.column_stack(
        [np.ones(len(features)), np.asarray(features, dtype=float)]
    )
    y = np.asarray(outcomes, dtype=float)
    n, k = design.shape
    if n <= k:
        raise ValueError(
            f"need more observations than parameters, got n={n} and k={k}; "
            "with no residual degrees of freedom the fit cannot be tested"
        )

    if float(np.var(y)) == 0.0:
        raise ValueError(
            f"the response is constant at {y[0]:g}; there is no variance to "
            "explain and no F-test to run"
        )

    gram = design.T @ design
    if np.linalg.matrix_rank(gram) < k or np.linalg.cond(gram) > _CONDITION_LIMIT:
        raise ValueError(
            "the predictors are collinear: X'X is rank deficient, so the "
            "coefficients are not identified; drop or combine redundant columns"
        )

    xtx_inv = np.linalg.inv(gram)
    beta = xtx_inv @ design.T @ y
    fitted = design @ beta

    return MultipleRegression(
        coefficients=[float(v) for v in beta],
        names=["(intercept)"]
        + (list(names) if names else [f"x{i + 1}" for i in range(width)]),
        xtx_inv=xtx_inv,
        residuals=y - fitted,
        fitted=fitted,
        observed=y,
    )


def predict(
    model: MultipleRegression, row: list[float], alpha: float = 0.05
) -> dict[str, float]:
    """Point estimate with both interval kinds for one new observation.

    Args:
        model: A fitted :class:`MultipleRegression`.
        row: Predictor values, without the intercept.
        alpha: Significance level for the intervals.

    Returns:
        Mapping with ``fit``, ``ci_lower``, ``ci_upper``, ``pi_lower``,
        ``pi_upper``, and the two standard errors behind them.

    Raises:
        ValueError: If the row width does not match the model.
    """
    if len(row) != model.k - 1:
        raise ValueError(f"expected {model.k - 1} predictors, got {len(row)}")

    x0 = np.concatenate([[1.0], np.asarray(row, dtype=float)])
    point = float(x0 @ np.asarray(model.coefficients))

    leverage = float(x0 @ model.xtx_inv @ x0)
    se_mean = math.sqrt(model.mse * max(leverage, 0.0))
    se_single = math.sqrt(model.mse * (1.0 + max(leverage, 0.0)))
    crit = t_quantile(1.0 - alpha / 2.0, model.df)

    return {
        "fit": point,
        "se_mean": se_mean,
        "se_single": se_single,
        "ci_lower": point - crit * se_mean,
        "ci_upper": point + crit * se_mean,
        "pi_lower": point - crit * se_single,
        "pi_upper": point + crit * se_single,
    }


def variance_inflation_factors(features: list[list[float]]) -> list[float]:
    """Variance inflation factor for each predictor.

    Each VIF is ``1 / (1 - R²)`` from regressing that predictor on all the
    others.  Values above roughly 5 to 10 are the usual multicollinearity
    warning.

    Args:
        features: One row per observation, one value per predictor.

    Returns:
        One VIF per predictor column, ``inf`` where a predictor is an exact
        linear combination of the others.

    Raises:
        ValueError: If there are fewer than two predictors, which leaves
            nothing to regress against.
    """
    matrix = np.asarray(features, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] < 2:
        raise ValueError("variance inflation needs at least two predictors")

    factors: list[float] = []
    for index in range(matrix.shape[1]):
        target = matrix[:, index]
        others = np.delete(matrix, index, axis=1)
        design = np.column_stack([np.ones(len(others)), others])

        coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
        residuals = target - design @ coefficients
        total = float(np.sum((target - np.mean(target)) ** 2))
        if total == 0:
            factors.append(math.inf)
            continue
        r_squared = 1.0 - float(np.sum(residuals**2)) / total
        factors.append(math.inf if r_squared >= 1.0 else 1.0 / (1.0 - r_squared))
    return factors


# ---------------------------------------------------------------------------
# CSV input
# ---------------------------------------------------------------------------


def read_csv(path: str) -> tuple[list[str], list[dict[str, str]]]:
    """Read a CSV with a header row.

    Args:
        path: Path to the file.

    Returns:
        ``(column_names, rows)`` where each row maps column name to raw text.

    Raises:
        ValueError: If the file is missing, unreadable, or has no header.
    """
    try:
        with open(path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"{path} is empty; a header row is required")
            return list(reader.fieldnames), [dict(row) for row in reader]
    except OSError as exc:
        raise ValueError(f"could not read {path}: {exc}") from exc


def extract_columns(
    rows: list[dict[str, str]], columns: list[str]
) -> list[list[float]]:
    """Pull named columns from CSV rows as floats.

    Args:
        rows: Rows from :func:`read_csv`.
        columns: Column names to extract, in order.

    Returns:
        One list of floats per row.

    Raises:
        ValueError: If a column is missing or a value is not numeric.
    """
    table: list[list[float]] = []
    for index, row in enumerate(rows, start=2):  # line 1 is the header
        values: list[float] = []
        for column in columns:
            if column not in row or row[column] is None:
                raise ValueError(f"column {column!r} missing on line {index}")
            try:
                values.append(float(row[column]))
            except ValueError:
                raise ValueError(
                    f"column {column!r} on line {index} is not numeric: {row[column]!r}"
                ) from None
        table.append(values)
    return table


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Build and return the argument parser namespace.

    Args:
        argv: Argument list (uses ``sys.argv`` when ``None``).

    Returns:
        Parsed :class:`argparse.Namespace`.
    """
    parser = argparse.ArgumentParser(
        description="Multiple linear regression with confidence and prediction "
        "intervals.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  mlreg --file housing.csv --target price
  mlreg --file data.csv --target sales --features advertising headcount --vif
  mlreg --file train.csv --target output --predict-file new_inputs.csv --alpha 0.10
""",
    )
    parser.add_argument(
        "--file",
        required=True,
        metavar="CSV",
        help="training data with a header row",
    )
    parser.add_argument(
        "--target",
        required=True,
        metavar="COL",
        help="response column",
    )
    parser.add_argument(
        "--features",
        nargs="+",
        default=None,
        metavar="COL",
        help="predictor columns (default: every column except the target)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        metavar="F",
        help="significance level for all intervals (default: 0.05)",
    )
    parser.add_argument(
        "--predict-file",
        default=None,
        metavar="CSV",
        help="predict these observations, with both interval kinds",
    )
    parser.add_argument(
        "--vif",
        action="store_true",
        help="report variance inflation factors for multicollinearity",
    )
    parser.add_argument(
        "--format",
        choices=["table", "json", "csv"],
        default="table",
        help="output format (default: table)",
    )
    parser.add_argument(
        "--precision",
        "-P",
        type=int,
        default=4,
        metavar="PREC",
        help="decimal places for output (default: 4)",
    )
    return parser.parse_args(argv)


def validate(args: argparse.Namespace) -> str | None:
    """Return an error message string, or ``None`` if arguments are valid.

    Args:
        args: Parsed argument namespace from :func:`parse_args`.

    Returns:
        Error description string, or ``None`` when validation passes.
    """
    if not 0.0 < args.alpha < 1.0:
        return f"--alpha must be strictly between 0 and 1, got {args.alpha}"
    if args.precision < 0:
        return f"--precision must be non-negative, got {args.precision}"
    if args.features is not None and args.target in args.features:
        return f"the target column {args.target!r} cannot also be a feature"
    return None


# ---------------------------------------------------------------------------
# Result assembly
# ---------------------------------------------------------------------------


def resolve_features(columns: list[str], args: argparse.Namespace) -> list[str]:
    """Decide which columns are predictors.

    Args:
        columns: Every column in the training file.
        args: Parsed argument namespace.

    Returns:
        Predictor column names.

    Raises:
        ValueError: If the target is absent, a named feature is absent, or no
            predictors remain.
    """
    if args.target not in columns:
        raise ValueError(
            f"target column {args.target!r} not in file; available: "
            f"{', '.join(columns)}"
        )
    if args.features is None:
        features = [name for name in columns if name != args.target]
    else:
        missing = [name for name in args.features if name not in columns]
        if missing:
            raise ValueError(
                f"feature column(s) not in file: {', '.join(missing)}; "
                f"available: {', '.join(columns)}"
            )
        features = list(args.features)
    if not features:
        raise ValueError("no predictor columns left after removing the target")
    return features


def build_result(args: argparse.Namespace) -> dict[str, Any]:
    """Fit the model and assemble every reported quantity.

    Args:
        args: Validated argument namespace from :func:`parse_args`.

    Returns:
        Result mapping ready for formatting.

    Raises:
        ValueError: On unreadable input, bad column selection, or a design
            matrix that cannot be fitted.
    """
    columns, rows = read_csv(args.file)
    if not rows:
        raise ValueError(f"{args.file} has a header but no data rows")

    feature_names = resolve_features(columns, args)
    design = extract_columns(rows, feature_names)
    response = [values[0] for values in extract_columns(rows, [args.target])]

    model = fit(design, response, names=feature_names)
    intervals = model.confidence_intervals(args.alpha)

    result: dict[str, Any] = {
        "n": model.n,
        "target": args.target,
        "features": feature_names,
        "alpha": args.alpha,
        "coefficients": [
            {
                "term": name,
                "estimate": coef,
                "std_error": se,
                "t": t_stat,
                "p_value": p,
                "ci_lower": low,
                "ci_upper": high,
            }
            for name, coef, se, t_stat, p, (low, high) in zip(
                model.names,
                model.coefficients,
                model.std_errors,
                model.t_stats,
                model.p_values,
                intervals,
            )
        ],
        "fit": {
            "r_squared": model.r_squared,
            "adj_r_squared": model.adj_r_squared,
            "residual_std_error": model.residual_std_error,
            "df": model.df,
            "f_statistic": model.f_statistic,
            "f_df": [model.df_model, model.df],
            "f_p_value": model.f_p_value,
        },
    }

    if args.vif:
        if len(feature_names) < 2:
            raise ValueError(
                "--vif needs at least two predictors; with one there is nothing "
                "for it to be collinear with"
            )
        result["vif"] = [
            {"term": name, "vif": value}
            for name, value in zip(feature_names, variance_inflation_factors(design))
        ]

    if args.predict_file is not None:
        _, new_rows = read_csv(args.predict_file)
        if not new_rows:
            raise ValueError(f"{args.predict_file} has a header but no data rows")
        result["predictions"] = [
            {"row": index, **predict(model, row, args.alpha)}
            for index, row in enumerate(
                extract_columns(new_rows, feature_names), start=1
            )
        ]
    return result


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def _fmt(value: float, precision: int) -> str:
    """Format a float to fixed precision.

    Args:
        value: Number to render.
        precision: Decimal places.

    Returns:
        Formatted string, or ``inf`` for a non-finite value.
    """
    if not math.isfinite(value):
        return "inf" if value > 0 else "-inf"
    return f"{value:.{precision}f}"


def _stars(p_value: float) -> str:
    """Conventional significance markers for a p-value.

    Args:
        p_value: Two-sided p-value.

    Returns:
        ``***``, ``**``, ``*``, ``.``, or an empty string.
    """
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    if p_value < 0.1:
        return "."
    return ""


def format_table(result: dict[str, Any], precision: int) -> str:
    """Render the result as an aligned plain-text report.

    Args:
        result: Mapping from :func:`build_result`.
        precision: Decimal places for numeric output.

    Returns:
        Multi-line report string.
    """
    level = int(round((1.0 - result["alpha"]) * 100))
    width = max(len(row["term"]) for row in result["coefficients"])
    lines = [
        f"Multiple regression: {result['target']} ~ {' + '.join(result['features'])}",
        f"n = {result['n']}",
        "",
        "Coefficients",
        "------------",
        f"  {'term':<{width}}  {'estimate':>12}  {'std.err':>10}  {'t':>8}  "
        f"{'p':>10}  {f'{level}% CI':>26}",
    ]
    for row in result["coefficients"]:
        interval = (
            f"[{_fmt(row['ci_lower'], precision)}, {_fmt(row['ci_upper'], precision)}]"
        )
        lines.append(
            f"  {row['term']:<{width}}  {_fmt(row['estimate'], precision):>12}  "
            f"{_fmt(row['std_error'], precision):>10}  {_fmt(row['t'], 3):>8}  "
            f"{_fmt(row['p_value'], 4):>10}  {interval:>26} {_stars(row['p_value'])}"
        )
    lines.append("")
    lines.append("  Signif. codes: *** p<0.001, ** p<0.01, * p<0.05, . p<0.1")

    stats = result["fit"]
    lines += [
        "",
        "Model fit",
        "---------",
        f"  R²:                    {_fmt(stats['r_squared'], precision)}",
        f"  Adjusted R²:           {_fmt(stats['adj_r_squared'], precision)}",
        f"  Residual std error:    {_fmt(stats['residual_std_error'], precision)}"
        f" on {stats['df']} df",
        f"  F-statistic:           {_fmt(stats['f_statistic'], precision)}"
        f" on {stats['f_df'][0]} and {stats['f_df'][1]} df",
        f"  Overall p-value:       {_fmt(stats['f_p_value'], 6)}",
    ]

    if "vif" in result:
        lines += ["", "Variance inflation", "------------------"]
        for row in result["vif"]:
            flag = "  <- multicollinear" if row["vif"] > 10 else ""
            lines.append(
                f"  {row['term']:<{width}}  {_fmt(row['vif'], precision):>10}{flag}"
            )

    if "predictions" in result:
        lines += [
            "",
            f"Predictions ({level}% intervals)",
            "-" * len(f"Predictions ({level}% intervals)"),
            f"  {'row':>5}  {'fit':>12}  {'CI (mean)':>26}  {'PI (single)':>26}",
        ]
        for row in result["predictions"]:
            ci = f"[{_fmt(row['ci_lower'], precision)}, {_fmt(row['ci_upper'], precision)}]"
            pi = f"[{_fmt(row['pi_lower'], precision)}, {_fmt(row['pi_upper'], precision)}]"
            lines.append(
                f"  {row['row']:>5}  {_fmt(row['fit'], precision):>12}  "
                f"{ci:>26}  {pi:>26}"
            )
    return "\n".join(lines)


def format_csv(result: dict[str, Any], precision: int) -> str:
    """Render the coefficient table, and predictions if present, as CSV.

    Args:
        result: Mapping from :func:`build_result`.
        precision: Decimal places for numeric output.

    Returns:
        CSV document string.
    """
    lines = ["section,term,estimate,std_error,t,p_value,ci_lower,ci_upper"]
    for row in result["coefficients"]:
        lines.append(
            "coefficient,"
            f"{row['term']},{_fmt(row['estimate'], precision)},"
            f"{_fmt(row['std_error'], precision)},{_fmt(row['t'], precision)},"
            f"{_fmt(row['p_value'], precision)},{_fmt(row['ci_lower'], precision)},"
            f"{_fmt(row['ci_upper'], precision)}"
        )
    for row in result.get("predictions", []):
        lines.append(
            f"prediction,{row['row']},{_fmt(row['fit'], precision)},,,"
            f",{_fmt(row['pi_lower'], precision)},{_fmt(row['pi_upper'], precision)}"
        )
    return "\n".join(lines)


def format_json(result: dict[str, Any]) -> str:
    """Render the result as indented JSON.

    Args:
        result: Mapping from :func:`build_result`.

    Returns:
        JSON document string.
    """
    return json.dumps(result, indent=2)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Run the multiple regression CLI.

    Args:
        argv: Argument list override for testing (uses ``sys.argv`` when ``None``).

    Returns:
        0 on success, 2 on input or computation error.
    """
    args = parse_args(argv)

    error = validate(args)
    if error:
        print(f"Error: {error}", file=sys.stderr)
        return 2

    try:
        result = build_result(args)
        if args.format == "json":
            print(format_json(result))
        elif args.format == "csv":
            print(format_csv(result, args.precision))
        else:
            print(format_table(result, args.precision))
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
