"""Tests for the binary logistic regression utility."""

import argparse
import json
import math

import numpy as np
import pytest
from scipy.optimize import minimize

from src.utils.logistic_regression import (
    LogisticModel,
    build_result,
    classification_metrics,
    confusion_matrix,
    encode_binary,
    extract_columns,
    fit,
    format_json,
    format_table,
    log_likelihood,
    log_odds,
    main,
    normal_quantile,
    normal_sf,
    predict_class,
    predict_proba,
    read_csv,
    resolve_features,
    sigmoid,
    two_sided_p,
    validate,
)

# A deterministic dataset with genuine class overlap: the outcome rises with x1,
# but four labels near the boundary are flipped so the classes are not linearly
# separable.  Without that overlap the likelihood has no interior maximum and
# the information matrix goes singular, which fit() rejects by design.
_X = [[float(i), float((i * 7) % 5)] for i in range(20)]
_Y = [1.0 if i > 9 else 0.0 for i in range(20)]
for _flipped in (3, 8, 11, 15):
    _Y[_flipped] = 1.0 - _Y[_flipped]


def _args(**overrides):
    """Build a namespace with valid defaults, overridden per test."""
    base = dict(
        file="train.csv",
        target="y",
        features=None,
        threshold=0.5,
        alpha=0.05,
        predict_file=None,
        max_iter=50,
        format="table",
        precision=4,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def _write_csv(path, header, rows):
    """Write a CSV fixture and return its path as a string."""
    lines = [",".join(header)]
    lines += [",".join(str(value) for value in row) for row in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


@pytest.fixture
def train_csv(tmp_path):
    """Training file matching the module-level fixture data."""
    rows = [[x[0], x[1], int(y)] for x, y in zip(_X, _Y)]
    return _write_csv(tmp_path / "train.csv", ["x1", "x2", "y"], rows)


# ---------------------------------------------------------------------------
# sigmoid / log_odds
# ---------------------------------------------------------------------------


def test_sigmoid_at_zero_is_half():
    assert sigmoid(0.0) == 0.5


def test_sigmoid_is_symmetric():
    assert sigmoid(2.0) + sigmoid(-2.0) == pytest.approx(1.0)


@pytest.mark.parametrize("z", [800.0, -800.0])
def test_sigmoid_does_not_overflow(z):
    value = sigmoid(z)
    assert 0.0 <= value <= 1.0


def test_log_odds_inverts_sigmoid():
    assert sigmoid(log_odds(0.73)) == pytest.approx(0.73)


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
def test_log_odds_outside_open_interval_raises(bad):
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        log_odds(bad)


# ---------------------------------------------------------------------------
# normal helpers
# ---------------------------------------------------------------------------


def test_normal_sf_at_zero():
    assert normal_sf(0.0) == pytest.approx(0.5)


def test_normal_sf_matches_known_tail():
    assert normal_sf(1.959963985) == pytest.approx(0.025, abs=1e-9)


def test_two_sided_p_is_symmetric():
    assert two_sided_p(-1.5) == two_sided_p(1.5)


def test_normal_quantile_matches_known_critical_value():
    assert normal_quantile(0.975) == pytest.approx(1.959963985, abs=1e-6)


def test_normal_quantile_round_trips_with_sf():
    z = normal_quantile(0.9)
    assert 1.0 - normal_sf(z) == pytest.approx(0.9, abs=1e-6)


@pytest.mark.parametrize("bad", [0.0, 1.0])
def test_normal_quantile_rejects_endpoints(bad):
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        normal_quantile(bad)


# ---------------------------------------------------------------------------
# log_likelihood
# ---------------------------------------------------------------------------


def test_log_likelihood_perfect_fit_is_near_zero():
    y = np.array([1.0, 0.0])
    probs = np.array([1.0 - 1e-15, 1e-15])
    assert log_likelihood(y, probs) == pytest.approx(0.0, abs=1e-9)


def test_log_likelihood_clamps_instead_of_returning_negative_infinity():
    value = log_likelihood(np.array([1.0]), np.array([0.0]))
    assert math.isfinite(value)


def test_log_likelihood_coin_flip_baseline():
    y = np.array([1.0, 0.0, 1.0, 0.0])
    probs = np.full(4, 0.5)
    assert log_likelihood(y, probs) == pytest.approx(4 * math.log(0.5))


# ---------------------------------------------------------------------------
# fit
# ---------------------------------------------------------------------------


def test_fit_recovers_expected_sign_pattern():
    model = fit(_X, _Y, names=["x1", "x2"])
    assert model.coefficients[1] > 0
    assert model.coefficients[2] < 0


def test_fit_matches_scipy_maximum_likelihood():
    """scipy oracle: our Newton-Raphson must land on the same MLE."""
    model = fit(_X, _Y, names=["x1", "x2"])

    design = np.column_stack([np.ones(len(_X)), np.array(_X)])
    y = np.array(_Y)

    def negative_loglik(beta):
        z = design @ beta
        return float(np.sum(np.logaddexp(0.0, z) - y * z))

    opt = minimize(negative_loglik, np.zeros(3), method="BFGS", options={"gtol": 1e-12})
    assert model.coefficients == pytest.approx(list(opt.x), abs=1e-4)
    assert model.loglik == pytest.approx(-opt.fun, abs=1e-8)


def test_fit_standard_errors_match_numerical_hessian():
    """Wald SEs must equal the inverse observed information, not an approximation."""
    model = fit(_X, _Y, names=["x1", "x2"])
    design = np.column_stack([np.ones(len(_X)), np.array(_X)])
    y = np.array(_Y)

    def gradient(beta):
        probs = 1.0 / (1.0 + np.exp(-(design @ beta)))
        return -(design.T @ (y - probs))

    beta = np.array(model.coefficients)
    step = 1e-6
    hessian = np.column_stack(
        [
            (gradient(beta + step * unit) - gradient(beta - step * unit)) / (2 * step)
            for unit in np.eye(len(beta))
        ]
    )
    expected = np.sqrt(np.diag(np.linalg.inv(hessian)))
    assert model.std_errors == pytest.approx(list(expected), rel=1e-6)


def test_fit_gradient_is_zero_at_solution():
    model = fit(_X, _Y, names=["x1", "x2"])
    design = np.column_stack([np.ones(len(_X)), np.array(_X)])
    probs = 1.0 / (1.0 + np.exp(-(design @ np.array(model.coefficients))))
    assert np.max(np.abs(design.T @ (np.array(_Y) - probs))) == pytest.approx(
        0.0, abs=1e-7
    )


def test_fit_default_names_when_none_given():
    model = fit(_X, _Y)
    assert model.names == ["(intercept)", "x1", "x2"]


def test_fit_converges_quickly():
    assert fit(_X, _Y).iterations <= 10


def test_fit_empty_data_raises():
    with pytest.raises(ValueError, match="no observations"):
        fit([], [])


def test_fit_length_mismatch_raises():
    with pytest.raises(ValueError, match="but 1 outcomes"):
        fit(_X, [1.0])


def test_fit_no_columns_raises():
    with pytest.raises(ValueError, match="no predictor columns"):
        fit([[], []], [0.0, 1.0])


def test_fit_ragged_rows_raise():
    with pytest.raises(ValueError, match="same number of columns"):
        fit([[1.0, 2.0], [3.0]], [0.0, 1.0])


def test_fit_single_class_target_raises():
    with pytest.raises(ValueError, match="must be binary"):
        fit(_X, [1.0] * len(_X))


def test_fit_non_binary_target_raises():
    outcomes = [0.0, 1.0, 2.0] + [0.0] * (len(_X) - 3)
    with pytest.raises(ValueError, match="must be binary"):
        fit(_X, outcomes)


def test_fit_too_few_observations_raises():
    with pytest.raises(ValueError, match="more observations than parameters"):
        fit([[1.0, 2.0], [3.0, 4.0]], [0.0, 1.0])


def test_fit_collinear_predictors_raise():
    rows = [[float(i), float(2 * i)] for i in range(10)]
    outcomes = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    with pytest.raises(ValueError, match="singular|separ"):
        fit(rows, outcomes)


def test_fit_perfectly_separable_data_raises():
    """Separable classes collapse the weights, so the information matrix goes
    singular rather than the fit quietly returning a runaway answer."""
    rows = [[float(i)] for i in range(12)]
    outcomes = [0.0] * 6 + [1.0] * 6
    with pytest.raises(ValueError, match="perfectly separated"):
        fit(rows, outcomes)


def test_fit_widely_separated_classes_hit_the_iteration_cap():
    """The other separation path: coefficients creep up without converging."""
    rows = [[float(i)] for i in list(range(6)) + list(range(100, 106))]
    outcomes = [0.0] * 6 + [1.0] * 6
    with pytest.raises(ValueError, match="did not converge"):
        fit(rows, outcomes, max_iter=200)


def test_fit_non_convergence_raises_rather_than_returning_silently():
    with pytest.raises(ValueError, match="did not converge"):
        fit(_X, _Y, max_iter=1, tol=1e-300)


# ---------------------------------------------------------------------------
# LogisticModel statistics
# ---------------------------------------------------------------------------


def test_model_information_criteria():
    model = fit(_X, _Y, names=["x1", "x2"])
    assert model.aic == pytest.approx(2 * 3 - 2 * model.loglik)
    assert model.bic == pytest.approx(3 * math.log(model.n) - 2 * model.loglik)


def test_model_pseudo_r2_between_zero_and_one():
    model = fit(_X, _Y, names=["x1", "x2"])
    assert 0.0 < model.pseudo_r2 < 1.0


def test_model_pseudo_r2_zero_when_null_loglik_is_zero():
    model = LogisticModel([0.0], [1.0], ["(intercept)"], 0.0, 0.0, 10, 1)
    assert model.pseudo_r2 == 0.0


def test_model_odds_ratio_is_exponentiated_coefficient():
    model = fit(_X, _Y, names=["x1", "x2"])
    assert model.odds_ratios[1] == pytest.approx(math.exp(model.coefficients[1]))


def test_model_z_is_infinite_when_standard_error_is_zero():
    model = LogisticModel([1.0], [0.0], ["(intercept)"], -1.0, -2.0, 10, 1)
    assert math.isinf(model.z_stats[0])


def test_model_confidence_intervals_bracket_the_estimate():
    model = fit(_X, _Y, names=["x1", "x2"])
    for coef, (low, high) in zip(model.coefficients, model.confidence_intervals(0.05)):
        assert low < coef < high


def test_model_wider_interval_at_smaller_alpha():
    model = fit(_X, _Y, names=["x1", "x2"])
    narrow = model.confidence_intervals(0.05)[1]
    wide = model.confidence_intervals(0.01)[1]
    assert wide[1] - wide[0] > narrow[1] - narrow[0]


# ---------------------------------------------------------------------------
# prediction and metrics
# ---------------------------------------------------------------------------


def test_predict_proba_matches_manual_sigmoid():
    assert predict_proba([0.5, 2.0], [1.0]) == pytest.approx(sigmoid(2.5))


def test_predict_proba_wrong_width_raises():
    with pytest.raises(ValueError, match="expected 1 predictors"):
        predict_proba([0.5, 2.0], [1.0, 3.0])


@pytest.mark.parametrize(
    "prob,threshold,expected",
    [(0.7, 0.5, 1), (0.3, 0.5, 0), (0.5, 0.5, 1), (0.4, 0.3, 1), (0.2, 0.3, 0)],
)
def test_predict_class(prob, threshold, expected):
    assert predict_class(prob, threshold) == expected


def test_confusion_matrix_counts_all_four_cells():
    cells = confusion_matrix([1.0, 0.0, 1.0, 0.0], [1, 1, 0, 0])
    assert cells == {"tp": 1, "fp": 1, "tn": 1, "fn": 1}


def test_confusion_matrix_length_mismatch_raises():
    with pytest.raises(ValueError, match="but 1 predictions"):
        confusion_matrix([1.0, 0.0], [1])


def test_classification_metrics_perfect_classifier():
    metrics = classification_metrics({"tp": 5, "fp": 0, "tn": 5, "fn": 0})
    assert metrics == pytest.approx(
        {"accuracy": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0}
    )


def test_classification_metrics_never_predicting_positive():
    metrics = classification_metrics({"tp": 0, "fp": 0, "tn": 5, "fn": 5})
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0
    assert metrics["accuracy"] == 0.5


def test_classification_metrics_f1_is_harmonic_mean():
    metrics = classification_metrics({"tp": 3, "fp": 1, "tn": 4, "fn": 2})
    expected = 2 * 0.75 * 0.6 / (0.75 + 0.6)
    assert metrics["f1"] == pytest.approx(expected)


def test_classification_metrics_empty_matrix_raises():
    with pytest.raises(ValueError, match="empty"):
        classification_metrics({"tp": 0, "fp": 0, "tn": 0, "fn": 0})


# ---------------------------------------------------------------------------
# CSV input
# ---------------------------------------------------------------------------


def test_read_csv_returns_header_and_rows(train_csv):
    columns, rows = read_csv(train_csv)
    assert columns == ["x1", "x2", "y"]
    assert len(rows) == len(_X)


def test_read_csv_missing_file_raises():
    with pytest.raises(ValueError, match="could not read"):
        read_csv("/nonexistent/path/train.csv")


def test_read_csv_empty_file_raises(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="header row is required"):
        read_csv(str(path))


def test_extract_columns_reads_floats(train_csv):
    _, rows = read_csv(train_csv)
    assert extract_columns(rows, ["x1"])[0] == [0.0]


def test_extract_columns_missing_column_raises(train_csv):
    _, rows = read_csv(train_csv)
    with pytest.raises(ValueError, match="missing on line"):
        extract_columns(rows, ["nope"])


def test_extract_columns_non_numeric_raises(tmp_path):
    path = _write_csv(tmp_path / "bad.csv", ["a", "y"], [[1, 0], ["oops", 1]])
    _, rows = read_csv(path)
    with pytest.raises(ValueError, match="not numeric"):
        extract_columns(rows, ["a"])


def test_encode_binary_passes_through_zero_one():
    assert encode_binary([0.0, 1.0, 1.0]) == [0.0, 1.0, 1.0]


def test_encode_binary_maps_arbitrary_pair():
    assert encode_binary([2.0, 7.0, 7.0]) == [0.0, 1.0, 1.0]


@pytest.mark.parametrize("values", [[1.0, 1.0], [1.0, 2.0, 3.0]])
def test_encode_binary_rejects_wrong_cardinality(values):
    with pytest.raises(ValueError, match="exactly two distinct values"):
        encode_binary(values)


# ---------------------------------------------------------------------------
# validate / resolve_features
# ---------------------------------------------------------------------------


def test_validate_accepts_defaults():
    assert validate(_args()) is None


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.2, 1.4])
def test_validate_rejects_threshold_outside_interval(bad):
    assert "--threshold" in validate(_args(threshold=bad))


@pytest.mark.parametrize("bad", [0.0, 1.0])
def test_validate_rejects_alpha_outside_interval(bad):
    assert "--alpha" in validate(_args(alpha=bad))


def test_validate_rejects_zero_max_iter():
    assert "--max-iter" in validate(_args(max_iter=0))


def test_validate_rejects_negative_precision():
    assert "--precision" in validate(_args(precision=-1))


def test_validate_rejects_target_listed_as_feature():
    assert "cannot also be a feature" in validate(_args(features=["x1", "y"]))


def test_resolve_features_defaults_to_all_but_target():
    assert resolve_features(["x1", "x2", "y"], _args()) == ["x1", "x2"]


def test_resolve_features_honours_explicit_list():
    assert resolve_features(["x1", "x2", "y"], _args(features=["x2"])) == ["x2"]


def test_resolve_features_missing_target_raises():
    with pytest.raises(ValueError, match="target column"):
        resolve_features(["x1", "x2"], _args())


def test_resolve_features_missing_feature_raises():
    with pytest.raises(ValueError, match="feature column"):
        resolve_features(["x1", "y"], _args(features=["nope"]))


def test_resolve_features_no_predictors_left_raises():
    with pytest.raises(ValueError, match="no predictor columns left"):
        resolve_features(["y"], _args())


# ---------------------------------------------------------------------------
# build_result
# ---------------------------------------------------------------------------


def test_build_result_core_shape(train_csv):
    result = build_result(_args(file=train_csv))
    assert result["n"] == len(_X)
    assert result["features"] == ["x1", "x2"]
    assert [row["term"] for row in result["coefficients"]] == [
        "(intercept)",
        "x1",
        "x2",
    ]
    assert set(result["confusion"]) == {"tp", "fp", "tn", "fn"}


def test_build_result_confusion_totals_match_n(train_csv):
    result = build_result(_args(file=train_csv))
    assert sum(result["confusion"].values()) == result["n"]


def test_build_result_threshold_shifts_predictions(train_csv):
    low = build_result(_args(file=train_csv, threshold=0.1))["confusion"]
    high = build_result(_args(file=train_csv, threshold=0.9))["confusion"]
    assert low["tp"] + low["fp"] > high["tp"] + high["fp"]


def test_build_result_predict_file(tmp_path, train_csv):
    new = _write_csv(tmp_path / "new.csv", ["x1", "x2"], [[2, 5], [9, 1]])
    result = build_result(_args(file=train_csv, predict_file=new))
    assert [row["row"] for row in result["predictions"]] == [1, 2]
    assert (
        result["predictions"][1]["probability"]
        > result["predictions"][0]["probability"]
    )


def test_build_result_header_only_training_file_raises(tmp_path):
    path = _write_csv(tmp_path / "head.csv", ["x1", "y"], [])
    with pytest.raises(ValueError, match="no data rows"):
        build_result(_args(file=path))


def test_build_result_header_only_predict_file_raises(tmp_path, train_csv):
    path = _write_csv(tmp_path / "head.csv", ["x1", "x2"], [])
    with pytest.raises(ValueError, match="no data rows"):
        build_result(_args(file=train_csv, predict_file=path))


def test_build_result_accepts_non_numeric_free_target_labels(tmp_path):
    rows = [[x[0], x[1], 5 if y else 3] for x, y in zip(_X, _Y)]
    path = _write_csv(tmp_path / "labels.csv", ["x1", "x2", "y"], rows)
    result = build_result(_args(file=path))
    assert result["n"] == len(_X)


# ---------------------------------------------------------------------------
# formatting
# ---------------------------------------------------------------------------


def test_format_table_has_every_section(train_csv):
    text = format_table(build_result(_args(file=train_csv)), 4)
    for heading in (
        "Coefficients",
        "Model fit",
        "Classification at threshold",
        "Accuracy",
    ):
        assert heading in text


def test_format_table_includes_predictions_when_present(tmp_path, train_csv):
    new = _write_csv(tmp_path / "new.csv", ["x1", "x2"], [[2, 5]])
    text = format_table(build_result(_args(file=train_csv, predict_file=new)), 4)
    assert "Predictions" in text


def test_format_table_significance_stars(train_csv):
    text = format_table(build_result(_args(file=train_csv)), 4)
    assert "Signif. codes" in text


@pytest.mark.parametrize(
    "p,marker",
    [(0.0001, "***"), (0.005, "**"), (0.02, "*"), (0.07, "."), (0.5, "")],
)
def test_significance_markers(p, marker):
    from src.utils.logistic_regression import _stars

    assert _stars(p) == marker


def test_format_table_renders_infinite_z():
    from src.utils.logistic_regression import _fmt

    assert _fmt(math.inf, 2) == "inf"
    assert _fmt(-math.inf, 2) == "-inf"


def test_format_json_round_trips(train_csv):
    payload = json.loads(format_json(build_result(_args(file=train_csv))))
    assert payload["features"] == ["x1", "x2"]
    assert len(payload["coefficients"]) == 3


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def test_main_table_run(capsys, train_csv):
    assert main(["--file", train_csv, "--target", "y"]) == 0
    assert "Logistic regression" in capsys.readouterr().out


def test_main_json_run(capsys, train_csv):
    assert main(["--file", train_csv, "--target", "y", "--format", "json"]) == 0
    assert json.loads(capsys.readouterr().out)["n"] == len(_X)


def test_main_selected_features(capsys, train_csv):
    assert main(["--file", train_csv, "--target", "y", "--features", "x1"]) == 0
    assert "x2" not in capsys.readouterr().out


def test_main_validation_error_exits_two(capsys, train_csv):
    assert main(["--file", train_csv, "--target", "y", "--threshold", "0"]) == 2
    assert "--threshold" in capsys.readouterr().err


def test_main_missing_file_exits_two(capsys):
    assert main(["--file", "/nonexistent.csv", "--target", "y"]) == 2
    assert "could not read" in capsys.readouterr().err


def test_main_separable_data_exits_two(capsys, tmp_path):
    rows = [[float(i), 0.0 if i < 6 else 1.0] for i in range(12)]
    path = _write_csv(tmp_path / "sep.csv", ["x1", "y"], rows)
    assert main(["--file", path, "--target", "y"]) == 2
    assert "separa" in capsys.readouterr().err
