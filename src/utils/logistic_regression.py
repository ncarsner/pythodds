#!/usr/bin/env python3
"""Command-line utility for binary logistic regression.

Models the probability of a binary outcome as a sigmoid of a linear predictor:

    P(y = 1 | x) = 1 / (1 + exp(-(b0 + b1*x1 + ... + bk*xk)))

Fitted by Newton-Raphson (iteratively reweighted least squares), which
converges in a handful of iterations on well-conditioned data and reports
coefficients as odds ratios -- the form that travels to non-technical readers.

The classification counterpart to ``linreg``: reach for this when the outcome
is binary rather than continuous.

Usage examples:
  # Fit from a CSV, all non-target columns as features
  logreg --file patient_data.csv --target disease

  # Chosen features, lower cutoff to favour recall
  logreg --file churn.csv --target churned --features tenure spend --threshold 0.3

  # Score new observations with the fitted model
  logreg --file train.csv --target clicked --predict-file new_users.csv

  # JSON for downstream processing
  logreg --file iris.csv --target setosa --format json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from typing import Any

import numpy as np

# Convergence is declared when the largest absolute coefficient change between
# Newton steps drops below this.
_DEFAULT_TOL = 1e-8

# Probabilities are clamped this far from 0 and 1 before taking logs, so a
# saturated fit yields a large finite log-likelihood instead of -inf.
_EPS = 1e-12


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def sigmoid(z: float) -> float:
    """Logistic function, evaluated without overflowing on large |z|.

    Args:
        z: Linear predictor value.

    Returns:
        ``1 / (1 + exp(-z))`` in [0, 1].
    """
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-z))
    exp_z = math.exp(z)
    return exp_z / (1.0 + exp_z)


def log_odds(p: float) -> float:
    """Logit of a probability -- the inverse of :func:`sigmoid`.

    Args:
        p: Probability, strictly between 0 and 1.

    Returns:
        ``log(p / (1 - p))``.

    Raises:
        ValueError: If ``p`` is outside the open interval (0, 1), where the
            logit is undefined.
    """
    if not 0.0 < p < 1.0:
        raise ValueError(f"probability must be strictly between 0 and 1, got {p}")
    return math.log(p / (1.0 - p))


def _sigmoid_vec(z: np.ndarray) -> np.ndarray:
    """Vectorised, overflow-safe sigmoid.

    Args:
        z: Array of linear predictor values.

    Returns:
        Array of probabilities the same shape as ``z``.
    """
    out = np.empty_like(z, dtype=float)
    positive = z >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    out[~positive] = exp_z / (1.0 + exp_z)
    return out


def log_likelihood(y: np.ndarray, probs: np.ndarray) -> float:
    """Bernoulli log-likelihood of the fitted probabilities.

    Args:
        y: Observed 0/1 outcomes.
        probs: Fitted probabilities, same length as ``y``.

    Returns:
        ``sum(y*log(p) + (1-y)*log(1-p))``, with probabilities clamped away
        from 0 and 1 so a saturated fit stays finite.
    """
    clipped = np.clip(probs, _EPS, 1.0 - _EPS)
    return float(np.sum(y * np.log(clipped) + (1.0 - y) * np.log(1.0 - clipped)))


def normal_sf(z: float) -> float:
    """Upper-tail probability of the standard normal distribution.

    Args:
        z: Standard normal deviate.

    Returns:
        ``P(Z > z)``.
    """
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def two_sided_p(z: float) -> float:
    """Two-sided normal p-value for a Wald z-statistic.

    Args:
        z: Wald statistic (coefficient divided by its standard error).

    Returns:
        ``2 * P(Z > |z|)``.
    """
    return 2.0 * normal_sf(abs(z))


def _solve_information(hessian: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Solve against the information matrix, naming the failure it can hit.

    Args:
        hessian: Observed information matrix ``X' W X``.
        rhs: Right-hand side -- the score vector for a Newton step, or the
            identity matrix to obtain the covariance.

    Returns:
        The solution array.

    Raises:
        ValueError: If the matrix is singular, which means the predictors are
            collinear or constant, or the classes are perfectly separated.
    """
    try:
        return np.linalg.solve(hessian, rhs)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "the information matrix is singular: predictors are collinear, "
            "constant, or the classes are perfectly separated"
        ) from exc


class LogisticModel:
    """Fitted binary logistic regression and its inference statistics."""

    def __init__(
        self,
        coefficients: list[float],
        std_errors: list[float],
        names: list[str],
        loglik: float,
        null_loglik: float,
        n: int,
        iterations: int,
    ):
        """Store a fit and derive its Wald statistics.

        Args:
            coefficients: Estimates, intercept first.
            std_errors: Standard errors aligned with ``coefficients``.
            names: Term names aligned with ``coefficients``.
            loglik: Log-likelihood at the fitted coefficients.
            null_loglik: Log-likelihood of the intercept-only model.
            n: Number of observations.
            iterations: Newton steps taken before convergence.
        """
        self.coefficients = coefficients
        self.std_errors = std_errors
        self.names = names
        self.loglik = loglik
        self.null_loglik = null_loglik
        self.n = n
        self.iterations = iterations

        self.z_stats = [
            coef / se if se > 0 else math.inf
            for coef, se in zip(coefficients, std_errors)
        ]
        self.p_values = [two_sided_p(z) for z in self.z_stats]
        self.odds_ratios = [math.exp(coef) for coef in coefficients]

    @property
    def k(self) -> int:
        """Number of estimated parameters, intercept included."""
        return len(self.coefficients)

    @property
    def aic(self) -> float:
        """Akaike information criterion, ``2k - 2*loglik``."""
        return 2.0 * self.k - 2.0 * self.loglik

    @property
    def bic(self) -> float:
        """Bayesian information criterion, ``k*log(n) - 2*loglik``."""
        return self.k * math.log(self.n) - 2.0 * self.loglik

    @property
    def pseudo_r2(self) -> float:
        """McFadden's pseudo-R², ``1 - loglik / null_loglik``."""
        if self.null_loglik == 0.0:
            return 0.0
        return 1.0 - self.loglik / self.null_loglik

    def confidence_intervals(self, alpha: float) -> list[tuple[float, float]]:
        """Wald confidence intervals for each coefficient.

        Args:
            alpha: Significance level, e.g. 0.05 for 95% intervals.

        Returns:
            ``(lower, upper)`` per coefficient, on the log-odds scale.
        """
        z_crit = normal_quantile(1.0 - alpha / 2.0)
        return [
            (coef - z_crit * se, coef + z_crit * se)
            for coef, se in zip(self.coefficients, self.std_errors)
        ]


def normal_quantile(p: float) -> float:
    """Inverse standard normal CDF via bisection on :func:`normal_sf`.

    Args:
        p: Cumulative probability, strictly between 0 and 1.

    Returns:
        The z with ``P(Z <= z) = p``.

    Raises:
        ValueError: If ``p`` is not in the open interval (0, 1).
    """
    if not 0.0 < p < 1.0:
        raise ValueError(f"probability must be strictly between 0 and 1, got {p}")
    low, high = -40.0, 40.0
    for _ in range(200):
        mid = (low + high) / 2.0
        if 1.0 - normal_sf(mid) < p:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0


def fit(
    features: list[list[float]],
    outcomes: list[float],
    names: list[str] | None = None,
    max_iter: int = 50,
    tol: float = _DEFAULT_TOL,
) -> LogisticModel:
    """Fit a binary logistic regression by Newton-Raphson (IRLS).

    An intercept column is prepended automatically.

    Args:
        features: One row per observation, one value per predictor.
        outcomes: Binary 0/1 outcomes, one per row.
        names: Predictor names; defaults to ``x1``..``xk``.
        max_iter: Maximum Newton steps before giving up.
        tol: Convergence threshold on the largest coefficient change.

    Returns:
        The fitted :class:`LogisticModel`.

    Raises:
        ValueError: If the data are empty or ragged, the outcome is not
            binary, there are fewer observations than parameters, the
            predictors are collinear or the data perfectly separable, or the
            fit fails to converge within ``max_iter``.
    """
    if not features:
        raise ValueError("no observations to fit")
    if len(features) != len(outcomes):
        raise ValueError(
            f"got {len(features)} feature rows but {len(outcomes)} outcomes"
        )

    width = len(features[0])
    if width == 0:
        raise ValueError("no predictor columns to fit")
    if any(len(row) != width for row in features):
        raise ValueError("all feature rows must have the same number of columns")

    distinct = sorted(set(outcomes))
    if distinct != [0.0, 1.0]:
        raise ValueError(
            "the target must be binary and take both values; got "
            f"{[f'{v:g}' for v in distinct]}"
        )

    design = np.column_stack(
        [np.ones(len(features)), np.asarray(features, dtype=float)]
    )
    y = np.asarray(outcomes, dtype=float)
    n, k = design.shape
    if n <= k:
        raise ValueError(
            f"need more observations than parameters, got n={n} and k={k}; "
            "the fit would be exactly determined or underdetermined"
        )

    beta = np.zeros(k)
    iterations = 0
    for iterations in range(1, max_iter + 1):
        eta = design @ beta
        probs = _sigmoid_vec(eta)
        weights = probs * (1.0 - probs)

        hessian = design.T @ (design * weights[:, None])
        gradient = design.T @ (y - probs)
        step = _solve_information(hessian, gradient)

        # No separate "coefficients diverged" guard here.  On separable data
        # the weights collapse toward zero and the information matrix goes
        # singular, or the step never shrinks below tol -- both already raise.
        # Traced over separations from adjacent to 1e6 apart, the coefficients
        # never ran away before one of those two fired.
        beta = beta + step
        if np.max(np.abs(step)) < tol:
            break
    else:
        raise ValueError(
            f"Newton-Raphson did not converge in {max_iter} iterations; "
            "raise --max-iter or check for near-separable data"
        )

    probs = _sigmoid_vec(design @ beta)
    weights = probs * (1.0 - probs)
    hessian = design.T @ (design * weights[:, None])
    covariance = _solve_information(hessian, np.eye(k))
    std_errors = np.sqrt(np.abs(np.diag(covariance)))

    base_rate = float(np.mean(y))
    null_ll = float(
        n
        * (base_rate * math.log(base_rate) + (1 - base_rate) * math.log(1 - base_rate))
    )
    term_names = ["(intercept)"] + (
        list(names) if names else [f"x{i + 1}" for i in range(width)]
    )
    return LogisticModel(
        coefficients=[float(v) for v in beta],
        std_errors=[float(v) for v in std_errors],
        names=term_names,
        loglik=log_likelihood(y, probs),
        null_loglik=null_ll,
        n=n,
        iterations=iterations,
    )


def predict_proba(coefficients: list[float], row: list[float]) -> float:
    """Predicted probability for one observation.

    Args:
        coefficients: Fitted coefficients, intercept first.
        row: Predictor values, without the intercept.

    Returns:
        ``P(y = 1 | row)``.

    Raises:
        ValueError: If the row length does not match the coefficients.
    """
    if len(row) != len(coefficients) - 1:
        raise ValueError(f"expected {len(coefficients) - 1} predictors, got {len(row)}")
    total = coefficients[0] + sum(c * v for c, v in zip(coefficients[1:], row))
    return sigmoid(total)


def predict_class(probability: float, threshold: float = 0.5) -> int:
    """Turn a probability into a 0/1 label.

    Args:
        probability: Predicted probability of the positive class.
        threshold: Cutoff at or above which the label is 1.

    Returns:
        1 when ``probability >= threshold``, else 0.
    """
    return int(probability >= threshold)


def confusion_matrix(actual: list[float], predicted: list[int]) -> dict[str, int]:
    """Count the four outcome cells.

    Args:
        actual: Observed 0/1 outcomes.
        predicted: Predicted 0/1 labels, same length.

    Returns:
        Mapping with ``tp``, ``fp``, ``tn``, and ``fn`` counts.

    Raises:
        ValueError: If the two sequences differ in length.
    """
    if len(actual) != len(predicted):
        raise ValueError(
            f"got {len(actual)} actual values but {len(predicted)} predictions"
        )
    cells = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    for truth, guess in zip(actual, predicted):
        if truth == 1 and guess == 1:
            cells["tp"] += 1
        elif truth == 0 and guess == 1:
            cells["fp"] += 1
        elif truth == 0 and guess == 0:
            cells["tn"] += 1
        else:
            cells["fn"] += 1
    return cells


def classification_metrics(cells: dict[str, int]) -> dict[str, float]:
    """Accuracy, precision, recall, and F1 from a confusion matrix.

    Precision, recall, and F1 are reported as 0.0 when their denominator is
    zero -- a model that never predicts the positive class has no precision to
    speak of, and saying 0 is clearer than refusing to answer.

    Args:
        cells: Mapping from :func:`confusion_matrix`.

    Returns:
        Mapping with ``accuracy``, ``precision``, ``recall``, and ``f1``.

    Raises:
        ValueError: If the matrix is empty.
    """
    tp, fp, tn, fn = cells["tp"], cells["fp"], cells["tn"], cells["fn"]
    total = tp + fp + tn + fn
    if total == 0:
        raise ValueError("confusion matrix is empty")

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": (tp + tn) / total,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


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


def encode_binary(values: list[float]) -> list[float]:
    """Map a two-valued column onto 0 and 1.

    Already-binary 0/1 columns pass through untouched; any other pair of values
    is mapped with the smaller to 0 and the larger to 1.

    Args:
        values: Raw numeric target column.

    Returns:
        The column encoded as 0.0 and 1.0.

    Raises:
        ValueError: If the column does not take exactly two distinct values.
    """
    distinct = sorted(set(values))
    if len(distinct) != 2:
        raise ValueError(
            "the target column must take exactly two distinct values, got "
            f"{len(distinct)}: {[f'{v:g}' for v in distinct[:5]]}"
        )
    low, high = distinct
    return [0.0 if value == low else 1.0 for value in values]


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
        description="Binary logistic regression with odds ratios and fit metrics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  logreg --file patient_data.csv --target disease
  logreg --file churn.csv --target churned --features tenure spend --threshold 0.3
  logreg --file train.csv --target clicked --predict-file new_users.csv
  logreg --file iris.csv --target setosa --format json
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
        help="binary outcome column",
    )
    parser.add_argument(
        "--features",
        nargs="+",
        default=None,
        metavar="COL",
        help="predictor columns (default: every column except the target)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        metavar="F",
        help="probability cutoff for the positive class (default: 0.5)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        metavar="F",
        help="significance level for coefficient intervals (default: 0.05)",
    )
    parser.add_argument(
        "--predict-file",
        default=None,
        metavar="CSV",
        help="score these observations with the fitted model",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=50,
        metavar="INT",
        help="maximum Newton-Raphson iterations (default: 50)",
    )
    parser.add_argument(
        "--format",
        choices=["table", "json"],
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
    if not 0.0 < args.threshold < 1.0:
        return f"--threshold must be strictly between 0 and 1, got {args.threshold}"
    if not 0.0 < args.alpha < 1.0:
        return f"--alpha must be strictly between 0 and 1, got {args.alpha}"
    if args.max_iter < 1:
        return f"--max-iter must be >= 1, got {args.max_iter}"
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
        ValueError: On unreadable input, bad column selection, or a fit that
            cannot converge.
    """
    columns, rows = read_csv(args.file)
    if not rows:
        raise ValueError(f"{args.file} has a header but no data rows")

    feature_names = resolve_features(columns, args)
    design = extract_columns(rows, feature_names)
    raw_target = [values[0] for values in extract_columns(rows, [args.target])]
    outcomes = encode_binary(raw_target)

    model = fit(design, outcomes, names=feature_names, max_iter=args.max_iter)
    intervals = model.confidence_intervals(args.alpha)

    probabilities = [predict_proba(model.coefficients, row) for row in design]
    labels = [predict_class(p, args.threshold) for p in probabilities]
    cells = confusion_matrix(outcomes, labels)

    result: dict[str, Any] = {
        "n": model.n,
        "features": feature_names,
        "target": args.target,
        "threshold": args.threshold,
        "iterations": model.iterations,
        "coefficients": [
            {
                "term": name,
                "estimate": coef,
                "std_error": se,
                "z": z,
                "p_value": p,
                "odds_ratio": odds,
                "ci_lower": lo,
                "ci_upper": hi,
            }
            for name, coef, se, z, p, odds, (lo, hi) in zip(
                model.names,
                model.coefficients,
                model.std_errors,
                model.z_stats,
                model.p_values,
                model.odds_ratios,
                intervals,
            )
        ],
        "fit": {
            "log_likelihood": model.loglik,
            "null_log_likelihood": model.null_loglik,
            "pseudo_r2": model.pseudo_r2,
            "aic": model.aic,
            "bic": model.bic,
        },
        "confusion": cells,
        "metrics": classification_metrics(cells),
    }

    if args.predict_file is not None:
        _, new_rows = read_csv(args.predict_file)
        if not new_rows:
            raise ValueError(f"{args.predict_file} has a header but no data rows")
        new_design = extract_columns(new_rows, feature_names)
        result["predictions"] = [
            {
                "row": index,
                "probability": (prob := predict_proba(model.coefficients, row)),
                "predicted": predict_class(prob, args.threshold),
            }
            for index, row in enumerate(new_design, start=1)
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
    width = max(len(row["term"]) for row in result["coefficients"])
    lines = [
        f"Logistic regression: {result['target']} ~ {' + '.join(result['features'])}",
        f"n = {result['n']}, converged in {result['iterations']} iterations",
        "",
        "Coefficients",
        "------------",
        f"  {'term':<{width}}  {'estimate':>12}  {'std.err':>10}  {'z':>8}  "
        f"{'p':>10}  {'odds ratio':>12}",
    ]
    for row in result["coefficients"]:
        lines.append(
            f"  {row['term']:<{width}}  {_fmt(row['estimate'], precision):>12}  "
            f"{_fmt(row['std_error'], precision):>10}  {_fmt(row['z'], 3):>8}  "
            f"{_fmt(row['p_value'], 4):>10}  "
            f"{_fmt(row['odds_ratio'], precision):>12} {_stars(row['p_value'])}"
        )
    lines.append("")
    lines.append("  Signif. codes: *** p<0.001, ** p<0.01, * p<0.05, . p<0.1")

    fit_stats = result["fit"]
    lines += [
        "",
        "Model fit",
        "---------",
        f"  Log-likelihood:      {_fmt(fit_stats['log_likelihood'], precision)}",
        f"  Null log-likelihood: {_fmt(fit_stats['null_log_likelihood'], precision)}",
        f"  McFadden pseudo-R²:  {_fmt(fit_stats['pseudo_r2'], precision)}",
        f"  AIC:                 {_fmt(fit_stats['aic'], precision)}",
        f"  BIC:                 {_fmt(fit_stats['bic'], precision)}",
    ]

    cells = result["confusion"]
    metrics = result["metrics"]
    lines += [
        "",
        f"Classification at threshold {result['threshold']:g}",
        "-" * (28 + len(f"{result['threshold']:g}")),
        f"  {'':<12}{'pred 0':>8}{'pred 1':>8}",
        f"  {'actual 0':<12}{cells['tn']:>8}{cells['fp']:>8}",
        f"  {'actual 1':<12}{cells['fn']:>8}{cells['tp']:>8}",
        "",
        f"  Accuracy:   {_fmt(metrics['accuracy'], precision)}",
        f"  Precision:  {_fmt(metrics['precision'], precision)}",
        f"  Recall:     {_fmt(metrics['recall'], precision)}",
        f"  F1:         {_fmt(metrics['f1'], precision)}",
    ]

    if "predictions" in result:
        lines += [
            "",
            "Predictions",
            "-----------",
            f"  {'row':>5}  {'probability':>12}  {'class':>5}",
        ]
        for row in result["predictions"]:
            lines.append(
                f"  {row['row']:>5}  {_fmt(row['probability'], precision):>12}  "
                f"{row['predicted']:>5}"
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
    """Run the logistic regression CLI.

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
        else:
            print(format_table(result, args.precision))
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
