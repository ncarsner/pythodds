"""Tests for the multiple linear regression utility."""

import argparse
import json
import math

import numpy as np
import pytest
from scipy import stats

from src.utils.multiple_regression import (
    MultipleRegression,
    build_result,
    extract_columns,
    f_sf,
    fit,
    format_csv,
    format_json,
    format_table,
    main,
    predict,
    read_csv,
    resolve_features,
    t_cdf,
    t_quantile,
    validate,
    variance_inflation_factors,
)

# Deterministic design: two predictors that are not collinear, and a response
# built from them with a fixed residual pattern rather than random noise.
_X = [[float(i), float((i * 7) % 11)] for i in range(24)]
_WOBBLE = [1.0, -2.0, 0.5, 3.0, -1.5, 0.0]
_Y = [
    12.5 + 3.2 * row[0] + 8.1 * row[1] + _WOBBLE[i % len(_WOBBLE)]
    for i, row in enumerate(_X)
]


def _args(**overrides):
    """Build a namespace with valid defaults, overridden per test."""
    base = dict(
        file="train.csv",
        target="y",
        features=None,
        alpha=0.05,
        predict_file=None,
        vif=False,
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
    rows = [[x[0], x[1], y] for x, y in zip(_X, _Y)]
    return _write_csv(tmp_path / "train.csv", ["x1", "x2", "y"], rows)


def _ols_oracle(features, outcomes):
    """Independent OLS solution via numpy, for oracle comparisons."""
    design = np.column_stack([np.ones(len(features)), np.array(features, dtype=float)])
    y = np.array(outcomes, dtype=float)
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    n, k = design.shape
    df = n - k
    residuals = y - design @ beta
    mse = float(residuals @ residuals) / df
    std_errors = np.sqrt(mse * np.diag(np.linalg.inv(design.T @ design)))
    return design, y, beta, std_errors, df, mse


# ---------------------------------------------------------------------------
# Distribution helpers
# ---------------------------------------------------------------------------


def test_t_cdf_at_zero_is_half():
    assert t_cdf(0.0, 10) == pytest.approx(0.5)


@pytest.mark.parametrize("t_stat,df", [(-2.5, 7), (-0.4, 3), (1.1, 15), (3.3, 40)])
def test_t_cdf_matches_scipy(t_stat, df):
    assert t_cdf(t_stat, df) == pytest.approx(float(stats.t.cdf(t_stat, df)), abs=1e-12)


def test_t_cdf_rejects_zero_df():
    with pytest.raises(ValueError, match="df must be > 0"):
        t_cdf(1.0, 0)


@pytest.mark.parametrize("p,df", [(0.975, 12), (0.95, 5), (0.005, 30), (0.5, 8)])
def test_t_quantile_matches_scipy(p, df):
    assert t_quantile(p, df) == pytest.approx(float(stats.t.ppf(p, df)), abs=1e-6)


def test_t_quantile_inverts_t_cdf():
    assert t_cdf(t_quantile(0.9, 11), 11) == pytest.approx(0.9, abs=1e-6)


@pytest.mark.parametrize("bad", [0.0, 1.0])
def test_t_quantile_rejects_endpoints(bad):
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        t_quantile(bad, 10)


def test_t_quantile_rejects_zero_df():
    with pytest.raises(ValueError, match="df must be > 0"):
        t_quantile(0.5, 0)


@pytest.mark.parametrize(
    "f_stat,df1,df2", [(0.5, 3, 20), (2.0, 1, 10), (50.0, 4, 100), (1417.5, 2, 37)]
)
def test_f_sf_matches_scipy(f_stat, df1, df2):
    assert f_sf(f_stat, df1, df2) == pytest.approx(
        float(stats.f.sf(f_stat, df1, df2)), rel=1e-10
    )


def test_f_sf_stays_accurate_in_the_far_tail():
    """The 1 - cdf form collapses to exactly 0 here; the direct form must not."""
    value = f_sf(1417.5, 2, 37)
    assert value > 0.0
    assert value == pytest.approx(float(stats.f.sf(1417.5, 2, 37)), rel=1e-10)


def test_f_sf_at_zero_is_one():
    assert f_sf(0.0, 2, 10) == 1.0


def test_f_sf_at_infinity_is_zero():
    assert f_sf(math.inf, 2, 10) == 0.0


@pytest.mark.parametrize("df1,df2", [(0, 5), (5, 0)])
def test_f_sf_rejects_bad_df(df1, df2):
    with pytest.raises(ValueError, match="must be >= 1"):
        f_sf(1.0, df1, df2)


def test_f_sf_rejects_negative_statistic():
    with pytest.raises(ValueError, match="must be >= 0"):
        f_sf(-1.0, 2, 5)


# ---------------------------------------------------------------------------
# fit
# ---------------------------------------------------------------------------


def test_fit_recovers_known_coefficients():
    model = fit(_X, _Y, names=["x1", "x2"])
    assert model.coefficients[1] == pytest.approx(3.2, abs=0.2)
    assert model.coefficients[2] == pytest.approx(8.1, abs=0.2)


def test_fit_coefficients_match_numpy_lstsq():
    model = fit(_X, _Y, names=["x1", "x2"])
    _, _, beta, _, _, _ = _ols_oracle(_X, _Y)
    assert model.coefficients == pytest.approx(list(beta), rel=1e-10)


def test_fit_standard_errors_match_oracle():
    model = fit(_X, _Y, names=["x1", "x2"])
    _, _, _, std_errors, _, _ = _ols_oracle(_X, _Y)
    assert model.std_errors == pytest.approx(list(std_errors), rel=1e-10)


def test_fit_p_values_match_scipy():
    model = fit(_X, _Y, names=["x1", "x2"])
    _, _, beta, std_errors, df, _ = _ols_oracle(_X, _Y)
    expected = 2 * stats.t.sf(np.abs(beta / std_errors), df)
    assert model.p_values == pytest.approx(list(expected), abs=1e-12)


def test_fit_f_statistic_and_p_match_scipy():
    model = fit(_X, _Y, names=["x1", "x2"])
    expected = float(stats.f.sf(model.f_statistic, model.df_model, model.df))
    assert model.f_p_value == pytest.approx(expected, rel=1e-10)


def test_fit_residuals_are_orthogonal_to_predictors():
    model = fit(_X, _Y, names=["x1", "x2"])
    design = np.column_stack([np.ones(len(_X)), np.array(_X)])
    assert np.max(np.abs(design.T @ model.residuals)) == pytest.approx(0.0, abs=1e-8)


def test_fit_default_names():
    assert fit(_X, _Y).names == ["(intercept)", "x1", "x2"]


def test_fit_empty_data_raises():
    with pytest.raises(ValueError, match="no observations"):
        fit([], [])


def test_fit_length_mismatch_raises():
    with pytest.raises(ValueError, match="but 1 responses"):
        fit(_X, [1.0])


def test_fit_no_columns_raises():
    with pytest.raises(ValueError, match="no predictor columns"):
        fit([[], []], [1.0, 2.0])


def test_fit_ragged_rows_raise():
    with pytest.raises(ValueError, match="same number of columns"):
        fit([[1.0, 2.0], [3.0]], [1.0, 2.0])


def test_fit_no_residual_df_raises():
    with pytest.raises(ValueError, match="more observations than parameters"):
        fit([[1.0, 2.0], [3.0, 4.0]], [1.0, 2.0])


def test_fit_collinear_predictors_raise():
    rows = [[float(i), float(2 * i)] for i in range(12)]
    with pytest.raises(ValueError, match="collinear"):
        fit(rows, [float(i) for i in range(12)])


def test_fit_constant_predictor_raises():
    rows = [[float(i), 5.0] for i in range(12)]
    with pytest.raises(ValueError, match="collinear"):
        fit(rows, [float(i) for i in range(12)])


# ---------------------------------------------------------------------------
# Model statistics
# ---------------------------------------------------------------------------


def test_r_squared_matches_oracle():
    model = fit(_X, _Y, names=["x1", "x2"])
    y = np.array(_Y)
    expected = 1.0 - model.sse / float(np.sum((y - y.mean()) ** 2))
    assert model.r_squared == pytest.approx(expected)


def test_adjusted_r_squared_below_r_squared():
    model = fit(_X, _Y, names=["x1", "x2"])
    assert model.adj_r_squared < model.r_squared


def test_constant_response_is_rejected():
    """Zero response variance leaves nothing to explain, and would otherwise
    produce a negative F from roundoff-signed sums of squares."""
    rows = [[float(i), float((i * 3) % 5)] for i in range(10)]
    with pytest.raises(ValueError, match="response is constant"):
        fit(rows, [7.0] * 10)


def test_exactly_linear_response_gives_huge_finite_f():
    """Normal-equation roundoff keeps the residual mean square positive, so the
    F-statistic is enormous but finite. The direct-tail f_sf still resolves the
    p-value instead of collapsing it to zero the way 1 - cdf would."""
    rows = [[float(i), float((i * 3) % 5)] for i in range(10)]
    exact = [2.0 * row[0] + 3.0 * row[1] for row in rows]
    model = fit(rows, exact)
    assert model.r_squared == pytest.approx(1.0)
    assert model.f_statistic > 1e20
    assert math.isfinite(model.f_statistic)
    assert 0.0 < model.f_p_value < 1e-50


def test_residual_std_error_is_root_mse():
    model = fit(_X, _Y, names=["x1", "x2"])
    assert model.residual_std_error == pytest.approx(math.sqrt(model.mse))


def test_t_stat_infinite_when_standard_error_is_zero():
    model = MultipleRegression(
        coefficients=[1.0],
        names=["(intercept)"],
        xtx_inv=np.zeros((1, 1)),
        residuals=np.zeros(4),
        fitted=np.ones(4),
        observed=np.array([1.0, 2.0, 3.0, 4.0]),
    )
    assert math.isinf(model.t_stats[0])


def test_confidence_intervals_match_scipy():
    model = fit(_X, _Y, names=["x1", "x2"])
    crit = float(stats.t.ppf(0.975, model.df))
    for coef, se, (low, high) in zip(
        model.coefficients, model.std_errors, model.confidence_intervals(0.05)
    ):
        assert low == pytest.approx(coef - crit * se, rel=1e-8)
        assert high == pytest.approx(coef + crit * se, rel=1e-8)


def test_confidence_intervals_widen_at_smaller_alpha():
    model = fit(_X, _Y, names=["x1", "x2"])
    narrow = model.confidence_intervals(0.05)[1]
    wide = model.confidence_intervals(0.01)[1]
    assert wide[1] - wide[0] > narrow[1] - narrow[0]


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------


def test_predict_point_matches_manual_dot_product():
    model = fit(_X, _Y, names=["x1", "x2"])
    expected = (
        model.coefficients[0]
        + model.coefficients[1] * 12.0
        + model.coefficients[2] * 4.0
    )
    assert predict(model, [12.0, 4.0])["fit"] == pytest.approx(expected)


def test_prediction_interval_is_wider_than_confidence_interval():
    model = fit(_X, _Y, names=["x1", "x2"])
    out = predict(model, [12.0, 4.0])
    assert out["pi_lower"] < out["ci_lower"]
    assert out["pi_upper"] > out["ci_upper"]


def test_predict_intervals_match_oracle():
    model = fit(_X, _Y, names=["x1", "x2"])
    design, _, beta, _, df, mse = _ols_oracle(_X, _Y)
    x0 = np.array([1.0, 12.0, 4.0])
    leverage = float(x0 @ np.linalg.inv(design.T @ design) @ x0)
    crit = float(stats.t.ppf(0.975, df))
    point = float(x0 @ beta)

    out = predict(model, [12.0, 4.0], 0.05)
    assert out["ci_lower"] == pytest.approx(point - crit * math.sqrt(mse * leverage))
    assert out["pi_upper"] == pytest.approx(
        point + crit * math.sqrt(mse * (1 + leverage))
    )


def test_predict_interval_widens_away_from_the_data_centre():
    model = fit(_X, _Y, names=["x1", "x2"])
    near = predict(model, [12.0, 5.0])
    far = predict(model, [500.0, 5.0])
    assert far["ci_upper"] - far["ci_lower"] > near["ci_upper"] - near["ci_lower"]


def test_predict_wrong_width_raises():
    model = fit(_X, _Y, names=["x1", "x2"])
    with pytest.raises(ValueError, match="expected 2 predictors"):
        predict(model, [1.0])


# ---------------------------------------------------------------------------
# variance inflation
# ---------------------------------------------------------------------------


def test_vif_near_one_for_uncorrelated_predictors():
    factors = variance_inflation_factors(_X)
    assert all(1.0 <= value < 2.0 for value in factors)


def test_vif_matches_pairwise_correlation_formula():
    """With two predictors, VIF reduces to 1 / (1 - r²)."""
    matrix = np.array(_X)
    r = float(np.corrcoef(matrix[:, 0], matrix[:, 1])[0, 1])
    expected = 1.0 / (1.0 - r * r)
    assert variance_inflation_factors(_X) == pytest.approx([expected, expected])


def test_vif_infinite_for_exact_linear_dependence():
    rows = [[float(i), float(2 * i), float((i * 5) % 7)] for i in range(12)]
    factors = variance_inflation_factors(rows)
    assert math.isinf(factors[0])
    assert math.isinf(factors[1])


def test_vif_infinite_for_constant_predictor():
    rows = [[float(i), 3.0] for i in range(10)]
    assert math.isinf(variance_inflation_factors(rows)[1])


def test_vif_needs_two_predictors():
    with pytest.raises(ValueError, match="at least two predictors"):
        variance_inflation_factors([[1.0], [2.0], [3.0]])


# ---------------------------------------------------------------------------
# CSV input
# ---------------------------------------------------------------------------


def test_read_csv_returns_header_and_rows(train_csv):
    columns, rows = read_csv(train_csv)
    assert columns == ["x1", "x2", "y"]
    assert len(rows) == len(_X)


def test_read_csv_missing_file_raises():
    with pytest.raises(ValueError, match="could not read"):
        read_csv("/nonexistent/path/data.csv")


def test_read_csv_empty_file_raises(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="header row is required"):
        read_csv(str(path))


def test_extract_columns_missing_column_raises(train_csv):
    _, rows = read_csv(train_csv)
    with pytest.raises(ValueError, match="missing on line"):
        extract_columns(rows, ["nope"])


def test_extract_columns_non_numeric_raises(tmp_path):
    path = _write_csv(tmp_path / "bad.csv", ["a", "y"], [[1, 2], ["oops", 3]])
    _, rows = read_csv(path)
    with pytest.raises(ValueError, match="not numeric"):
        extract_columns(rows, ["a"])


# ---------------------------------------------------------------------------
# validate / resolve_features
# ---------------------------------------------------------------------------


def test_validate_accepts_defaults():
    assert validate(_args()) is None


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.5, 2.0])
def test_validate_rejects_alpha_outside_interval(bad):
    assert "--alpha" in validate(_args(alpha=bad))


def test_validate_rejects_negative_precision():
    assert "--precision" in validate(_args(precision=-1))


def test_validate_rejects_target_as_feature():
    assert "cannot also be a feature" in validate(_args(features=["x1", "y"]))


def test_resolve_features_defaults_to_all_but_target():
    assert resolve_features(["x1", "x2", "y"], _args()) == ["x1", "x2"]


def test_resolve_features_missing_target_raises():
    with pytest.raises(ValueError, match="target column"):
        resolve_features(["x1", "x2"], _args())


def test_resolve_features_missing_feature_raises():
    with pytest.raises(ValueError, match="feature column"):
        resolve_features(["x1", "y"], _args(features=["nope"]))


def test_resolve_features_none_left_raises():
    with pytest.raises(ValueError, match="no predictor columns left"):
        resolve_features(["y"], _args())


# ---------------------------------------------------------------------------
# build_result
# ---------------------------------------------------------------------------


def test_build_result_core_shape(train_csv):
    result = build_result(_args(file=train_csv))
    assert result["n"] == len(_X)
    assert [row["term"] for row in result["coefficients"]] == [
        "(intercept)",
        "x1",
        "x2",
    ]
    assert result["fit"]["df"] == len(_X) - 3


def test_build_result_includes_vif_when_requested(train_csv):
    result = build_result(_args(file=train_csv, vif=True))
    assert [row["term"] for row in result["vif"]] == ["x1", "x2"]


def test_build_result_vif_needs_two_features(train_csv):
    with pytest.raises(ValueError, match="at least two predictors"):
        build_result(_args(file=train_csv, features=["x1"], vif=True))


def test_build_result_predictions(tmp_path, train_csv):
    new = _write_csv(tmp_path / "new.csv", ["x1", "x2"], [[5, 3], [20, 9]])
    result = build_result(_args(file=train_csv, predict_file=new))
    assert [row["row"] for row in result["predictions"]] == [1, 2]
    assert result["predictions"][0]["pi_upper"] > result["predictions"][0]["ci_upper"]


def test_build_result_alpha_widens_intervals(train_csv):
    narrow = build_result(_args(file=train_csv))["coefficients"][1]
    wide = build_result(_args(file=train_csv, alpha=0.01))["coefficients"][1]
    assert wide["ci_upper"] - wide["ci_lower"] > narrow["ci_upper"] - narrow["ci_lower"]


def test_build_result_header_only_file_raises(tmp_path):
    path = _write_csv(tmp_path / "head.csv", ["x1", "y"], [])
    with pytest.raises(ValueError, match="no data rows"):
        build_result(_args(file=path))


def test_build_result_header_only_predict_file_raises(tmp_path, train_csv):
    path = _write_csv(tmp_path / "head.csv", ["x1", "x2"], [])
    with pytest.raises(ValueError, match="no data rows"):
        build_result(_args(file=train_csv, predict_file=path))


# ---------------------------------------------------------------------------
# formatting
# ---------------------------------------------------------------------------


def test_format_table_has_every_section(tmp_path, train_csv):
    new = _write_csv(tmp_path / "new.csv", ["x1", "x2"], [[5, 3]])
    result = build_result(_args(file=train_csv, vif=True, predict_file=new))
    text = format_table(result, 4)
    for heading in (
        "Multiple regression",
        "Coefficients",
        "Model fit",
        "Variance inflation",
        "Predictions",
    ):
        assert heading in text


def test_format_table_flags_multicollinearity(tmp_path):
    rows = [
        [float(i), float(2 * i + (i % 2) * 0.001), float(i) * 3.0 + (i % 3)]
        for i in range(20)
    ]
    path = _write_csv(tmp_path / "col.csv", ["x1", "x2", "y"], rows)
    text = format_table(build_result(_args(file=path, vif=True)), 4)
    assert "multicollinear" in text


def test_format_table_reports_interval_level(train_csv):
    text = format_table(build_result(_args(file=train_csv, alpha=0.10)), 4)
    assert "90% CI" in text


def test_format_table_renders_infinity():
    from src.utils.multiple_regression import _fmt

    assert _fmt(math.inf, 2) == "inf"
    assert _fmt(-math.inf, 2) == "-inf"


@pytest.mark.parametrize(
    "p,marker",
    [(0.0001, "***"), (0.005, "**"), (0.02, "*"), (0.07, "."), (0.5, "")],
)
def test_significance_markers(p, marker):
    from src.utils.multiple_regression import _stars

    assert _stars(p) == marker


def test_format_csv_has_a_row_per_coefficient(train_csv):
    text = format_csv(build_result(_args(file=train_csv)), 4)
    assert text.splitlines()[0].startswith("section,term")
    assert len(text.splitlines()) == 4


def test_format_csv_appends_predictions(tmp_path, train_csv):
    new = _write_csv(tmp_path / "new.csv", ["x1", "x2"], [[5, 3]])
    text = format_csv(build_result(_args(file=train_csv, predict_file=new)), 4)
    assert "prediction," in text


def test_format_json_round_trips(train_csv):
    payload = json.loads(format_json(build_result(_args(file=train_csv))))
    assert payload["features"] == ["x1", "x2"]
    assert len(payload["coefficients"]) == 3


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def test_main_table_run(capsys, train_csv):
    assert main(["--file", train_csv, "--target", "y"]) == 0
    assert "Multiple regression" in capsys.readouterr().out


def test_main_json_run(capsys, train_csv):
    assert main(["--file", train_csv, "--target", "y", "--format", "json"]) == 0
    assert json.loads(capsys.readouterr().out)["n"] == len(_X)


def test_main_csv_run(capsys, train_csv):
    assert main(["--file", train_csv, "--target", "y", "--format", "csv"]) == 0
    assert "section,term" in capsys.readouterr().out


def test_main_vif_flag(capsys, train_csv):
    assert main(["--file", train_csv, "--target", "y", "--vif"]) == 0
    assert "Variance inflation" in capsys.readouterr().out


def test_main_validation_error_exits_two(capsys, train_csv):
    assert main(["--file", train_csv, "--target", "y", "--alpha", "0"]) == 2
    assert "--alpha" in capsys.readouterr().err


def test_main_missing_file_exits_two(capsys):
    assert main(["--file", "/nonexistent.csv", "--target", "y"]) == 2
    assert "could not read" in capsys.readouterr().err


def test_main_collinear_data_exits_two(capsys, tmp_path):
    rows = [[float(i), float(2 * i), float(i) + 1.0] for i in range(12)]
    path = _write_csv(tmp_path / "col.csv", ["x1", "x2", "y"], rows)
    assert main(["--file", path, "--target", "y"]) == 2
    assert "collinear" in capsys.readouterr().err
