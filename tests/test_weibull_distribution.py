"""Tests for the Weibull distribution utility."""

import argparse
import json
import math

import pytest

import src.utils.weibull_distribution as weibull_module
from src.utils.weibull_distribution import (
    build_result,
    failure_mode,
    format_json,
    format_table,
    main,
    table_rows,
    validate,
    weibull_cdf,
    weibull_hazard,
    weibull_mean,
    weibull_median,
    weibull_pdf,
    weibull_quantile,
    weibull_survival,
    weibull_variance,
)


def _args(**overrides):
    """Build a namespace with valid defaults, overridden per test."""
    base = dict(
        x=500.0,
        k=2.0,
        lam=1000.0,
        quantile=None,
        survival=False,
        table=None,
        format="table",
        precision=4,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


# ---------------------------------------------------------------------------
# weibull_pdf
# ---------------------------------------------------------------------------


def test_weibull_pdf_matches_formula():
    expected = (2 / 1000) * (0.5**1) * math.exp(-0.25)
    assert weibull_pdf(500.0, 2.0, 1000.0) == pytest.approx(expected)


def test_weibull_pdf_reduces_to_exponential_when_k_is_one():
    assert weibull_pdf(500.0, 1.0, 1000.0) == pytest.approx(math.exp(-0.5) / 1000)


def test_weibull_pdf_negative_x_is_zero():
    assert weibull_pdf(-1.0, 2.0, 1000.0) == 0.0


def test_weibull_pdf_at_origin_wear_out_is_zero():
    assert weibull_pdf(0.0, 2.0, 1000.0) == 0.0


def test_weibull_pdf_at_origin_exponential_is_rate():
    assert weibull_pdf(0.0, 1.0, 1000.0) == pytest.approx(0.001)


def test_weibull_pdf_at_origin_infant_mortality_is_infinite():
    assert math.isinf(weibull_pdf(0.0, 0.5, 1000.0))


def test_weibull_pdf_invalid_shape_raises():
    with pytest.raises(ValueError, match="k must be > 0"):
        weibull_pdf(500.0, 0.0, 1000.0)


def test_weibull_pdf_invalid_scale_raises():
    with pytest.raises(ValueError, match="lambda must be > 0"):
        weibull_pdf(500.0, 2.0, 0.0)


# ---------------------------------------------------------------------------
# weibull_cdf / weibull_survival
# ---------------------------------------------------------------------------


def test_weibull_cdf_matches_formula():
    assert weibull_cdf(500.0, 2.0, 1000.0) == pytest.approx(1 - math.exp(-0.25))


def test_weibull_cdf_at_scale_is_one_minus_exp_neg_one():
    assert weibull_cdf(1000.0, 2.0, 1000.0) == pytest.approx(1 - math.exp(-1))


def test_weibull_cdf_at_or_below_zero_is_zero():
    assert weibull_cdf(0.0, 2.0, 1000.0) == 0.0
    assert weibull_cdf(-5.0, 2.0, 1000.0) == 0.0


def test_weibull_cdf_invalid_shape_raises():
    with pytest.raises(ValueError, match="k must be > 0"):
        weibull_cdf(500.0, -1.0, 1000.0)


def test_weibull_survival_matches_formula():
    assert weibull_survival(500.0, 2.0, 1000.0) == pytest.approx(math.exp(-0.25))


def test_weibull_survival_at_or_below_zero_is_one():
    assert weibull_survival(0.0, 2.0, 1000.0) == 1.0
    assert weibull_survival(-5.0, 2.0, 1000.0) == 1.0


def test_weibull_survival_complements_cdf():
    assert weibull_survival(750.0, 1.5, 900.0) + weibull_cdf(
        750.0, 1.5, 900.0
    ) == pytest.approx(1.0)


def test_weibull_survival_invalid_scale_raises():
    with pytest.raises(ValueError, match="lambda must be > 0"):
        weibull_survival(500.0, 2.0, -3.0)


# ---------------------------------------------------------------------------
# weibull_hazard
# ---------------------------------------------------------------------------


def test_weibull_hazard_equals_pdf_over_survival():
    x, k, lam = 500.0, 2.0, 1000.0
    assert weibull_hazard(x, k, lam) == pytest.approx(
        weibull_pdf(x, k, lam) / weibull_survival(x, k, lam)
    )


def test_weibull_hazard_is_constant_when_k_is_one():
    assert weibull_hazard(100.0, 1.0, 1000.0) == pytest.approx(0.001)
    assert weibull_hazard(9000.0, 1.0, 1000.0) == pytest.approx(0.001)


def test_weibull_hazard_increases_when_k_above_one():
    assert weibull_hazard(200.0, 2.5, 1000.0) < weibull_hazard(800.0, 2.5, 1000.0)


def test_weibull_hazard_decreases_when_k_below_one():
    assert weibull_hazard(200.0, 0.5, 1000.0) > weibull_hazard(800.0, 0.5, 1000.0)


def test_weibull_hazard_stays_finite_in_the_far_tail():
    """The closed form avoids the 0/0 that f(x)/S(x) hits far out in the tail."""
    assert math.isfinite(weibull_hazard(20000.0, 3.0, 1000.0))


def test_weibull_hazard_negative_x_is_zero():
    assert weibull_hazard(-1.0, 2.0, 1000.0) == 0.0


def test_weibull_hazard_at_origin_wear_out_is_zero():
    assert weibull_hazard(0.0, 2.0, 1000.0) == 0.0


def test_weibull_hazard_at_origin_exponential_is_rate():
    assert weibull_hazard(0.0, 1.0, 1000.0) == pytest.approx(0.001)


def test_weibull_hazard_at_origin_infant_mortality_is_infinite():
    assert math.isinf(weibull_hazard(0.0, 0.5, 1000.0))


def test_weibull_hazard_invalid_shape_raises():
    with pytest.raises(ValueError, match="k must be > 0"):
        weibull_hazard(500.0, 0.0, 1000.0)


# ---------------------------------------------------------------------------
# weibull_quantile
# ---------------------------------------------------------------------------


def test_weibull_quantile_matches_formula():
    assert weibull_quantile(0.05, 1.5, 800.0) == pytest.approx(
        800 * (-math.log(0.95)) ** (1 / 1.5)
    )


def test_weibull_quantile_inverts_cdf():
    x = weibull_quantile(0.3, 2.0, 1000.0)
    assert weibull_cdf(x, 2.0, 1000.0) == pytest.approx(0.3)


def test_weibull_quantile_at_zero_is_zero():
    assert weibull_quantile(0.0, 2.0, 1000.0) == 0.0


def test_weibull_quantile_at_one_minus_exp_neg_one_is_scale():
    assert weibull_quantile(1 - math.exp(-1), 2.0, 1000.0) == pytest.approx(1000.0)


@pytest.mark.parametrize("p", [-0.1, 1.0, 1.5])
def test_weibull_quantile_out_of_range_raises(p):
    with pytest.raises(ValueError, match=r"p must be in \[0, 1\)"):
        weibull_quantile(p, 2.0, 1000.0)


def test_weibull_quantile_invalid_shape_raises():
    with pytest.raises(ValueError, match="k must be > 0"):
        weibull_quantile(0.5, 0.0, 1000.0)


# ---------------------------------------------------------------------------
# weibull_mean / weibull_variance / weibull_median
# ---------------------------------------------------------------------------


def test_weibull_mean_matches_gamma_formula():
    assert weibull_mean(2.0, 1000.0) == pytest.approx(1000 * math.gamma(1.5))


def test_weibull_mean_reduces_to_scale_when_k_is_one():
    assert weibull_mean(1.0, 1000.0) == pytest.approx(1000.0)


def test_weibull_mean_invalid_scale_raises():
    with pytest.raises(ValueError, match="lambda must be > 0"):
        weibull_mean(2.0, 0.0)


def test_weibull_variance_matches_gamma_formula():
    expected = 1000**2 * (math.gamma(2.0) - math.gamma(1.5) ** 2)
    assert weibull_variance(2.0, 1000.0) == pytest.approx(expected)


def test_weibull_variance_of_exponential_is_scale_squared():
    assert weibull_variance(1.0, 1000.0) == pytest.approx(1000**2)


def test_weibull_variance_never_negative_for_large_shape():
    assert weibull_variance(200.0, 1000.0) >= 0.0


def test_weibull_variance_invalid_shape_raises():
    with pytest.raises(ValueError, match="k must be > 0"):
        weibull_variance(-1.0, 1000.0)


def test_weibull_median_matches_formula():
    assert weibull_median(2.0, 1000.0) == pytest.approx(1000 * math.log(2) ** 0.5)


def test_weibull_median_has_half_the_mass_below_it():
    assert weibull_cdf(weibull_median(1.7, 640.0), 1.7, 640.0) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# failure_mode
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("k", "fragment"),
    [(0.5, "infant mortality"), (1.0, "constant hazard"), (3.0, "wear-out")],
)
def test_failure_mode_by_shape(k, fragment):
    assert fragment in failure_mode(k)


def test_failure_mode_invalid_shape_raises():
    with pytest.raises(ValueError, match="k must be > 0"):
        failure_mode(0.0)


# ---------------------------------------------------------------------------
# table_rows
# ---------------------------------------------------------------------------


def test_table_rows_row_count_and_endpoints():
    rows = table_rows(0.0, 2000.0, 500.0, 2.5, 1200.0)
    assert len(rows) == 5
    assert rows[0][0] == pytest.approx(0.0)
    assert rows[-1][0] == pytest.approx(2000.0)


def test_table_rows_cdf_and_survival_complement():
    for _x, _pdf, cdf, surv, _hazard in table_rows(0.0, 1000.0, 250.0, 2.0, 800.0):
        assert cdf + surv == pytest.approx(1.0)


def test_table_rows_cdf_is_monotone():
    cdfs = [row[2] for row in table_rows(0.0, 2000.0, 200.0, 2.0, 1000.0)]
    assert cdfs == sorted(cdfs)


def test_table_rows_single_row_when_min_equals_max():
    assert len(table_rows(100.0, 100.0, 50.0, 2.0, 1000.0)) == 1


def test_table_rows_non_positive_step_raises():
    with pytest.raises(ValueError, match="step must be > 0"):
        table_rows(0.0, 100.0, 0.0, 2.0, 1000.0)


def test_table_rows_negative_min_raises():
    with pytest.raises(ValueError, match="min_x must be >= 0"):
        table_rows(-1.0, 100.0, 10.0, 2.0, 1000.0)


def test_table_rows_max_below_min_raises():
    with pytest.raises(ValueError, match="max_x must be >= min_x"):
        table_rows(100.0, 50.0, 10.0, 2.0, 1000.0)


def test_table_rows_invalid_shape_raises():
    with pytest.raises(ValueError, match="k must be > 0"):
        table_rows(0.0, 100.0, 10.0, 0.0, 1000.0)


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------


def test_validate_point_valid():
    assert validate(_args()) is None


def test_validate_non_positive_shape():
    result = validate(_args(k=0.0))
    assert result is not None and "-k must be > 0" in result


def test_validate_non_positive_scale():
    result = validate(_args(lam=0.0))
    assert result is not None and "--lambda must be > 0" in result


def test_validate_negative_precision():
    result = validate(_args(precision=-1))
    assert result is not None and "--precision" in result


def test_validate_table_valid():
    assert validate(_args(x=None, table=[0.0, 2000.0, 200.0])) is None


def test_validate_table_negative_min():
    result = validate(_args(table=[-1.0, 100.0, 10.0]))
    assert result is not None and "--table MIN" in result


def test_validate_table_max_below_min():
    result = validate(_args(table=[100.0, 50.0, 10.0]))
    assert result is not None and "--table MAX" in result


def test_validate_table_non_positive_step():
    result = validate(_args(table=[0.0, 100.0, 0.0]))
    assert result is not None and "--table STEP" in result


def test_validate_quantile_valid():
    assert validate(_args(x=None, quantile=0.05)) is None


def test_validate_quantile_out_of_range():
    result = validate(_args(quantile=1.0))
    assert result is not None and "--quantile must be in" in result


def test_validate_missing_x():
    result = validate(_args(x=None))
    assert result is not None and "-x is required" in result


def test_validate_negative_x():
    result = validate(_args(x=-1.0))
    assert result is not None and "-x must be >= 0" in result


# ---------------------------------------------------------------------------
# build_result
# ---------------------------------------------------------------------------


def test_build_result_point_section():
    result = build_result(_args())
    assert result["point"]["survival"] == pytest.approx(math.exp(-0.25))
    assert "table" not in result
    assert "quantile" not in result


def test_build_result_quantile_section():
    result = build_result(_args(x=None, quantile=0.05))
    assert result["quantile"]["x"] == pytest.approx(weibull_quantile(0.05, 2.0, 1000.0))
    assert "point" not in result


def test_build_result_table_section():
    result = build_result(_args(x=None, table=[0.0, 1000.0, 500.0]))
    assert len(result["table"]) == 3
    assert "point" not in result


def test_build_result_includes_moments_and_mode():
    result = build_result(_args())
    assert result["mean"] == pytest.approx(weibull_mean(2.0, 1000.0))
    assert "wear-out" in str(result["failure_mode"])


# ---------------------------------------------------------------------------
# format_table / format_json
# ---------------------------------------------------------------------------


def test_format_table_point_leads_with_cdf_by_default():
    out = format_table(build_result(_args()), 4, survival=False)
    lines = [line for line in out.splitlines() if "(x" in line]
    assert lines[1].strip().startswith("F(x")


def test_format_table_point_leads_with_survival_when_requested():
    out = format_table(build_result(_args()), 4, survival=True)
    lines = [line for line in out.splitlines() if "(x" in line]
    assert lines[1].strip().startswith("S(x")


def test_format_table_reports_failure_mode_and_moments():
    out = format_table(build_result(_args(k=0.7)), 4, survival=False)
    assert "infant mortality" in out
    assert "variance:" in out


def test_format_table_quantile_section():
    out = format_table(build_result(_args(x=None, quantile=0.05)), 4, survival=False)
    assert "x at p:" in out


def test_format_table_table_section():
    out = format_table(
        build_result(_args(x=None, table=[0.0, 1000.0, 500.0])), 4, survival=False
    )
    assert "h(x)" in out
    assert "1000.0000" in out


def test_format_table_renders_infinite_density_as_inf():
    out = format_table(build_result(_args(x=0.0, k=0.5)), 4, survival=False)
    assert "inf" in out


def test_format_json_point():
    data = json.loads(format_json(build_result(_args())))
    assert data["point"]["cdf"] == pytest.approx(1 - math.exp(-0.25))


def test_format_json_table_rows_become_objects():
    data = json.loads(
        format_json(build_result(_args(x=None, table=[0.0, 500.0, 500.0])))
    )
    assert data["table"][0]["x"] == pytest.approx(0.0)
    assert data["table"][1]["survival"] == pytest.approx(math.exp(-0.25))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def test_main_point_report(capsys):
    rc = main(["-x", "500", "-k", "2", "--lambda", "1000"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "0.7788" in out


def test_main_survival_flag(capsys):
    rc = main(["-x", "500", "-k", "2", "--lambda", "1000", "--survival"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "S(x >" in out


def test_main_quantile(capsys):
    rc = main(["--quantile", "0.05", "-k", "1.5", "--lambda", "800"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "110.4410" in out


def test_main_table(capsys):
    rc = main(["-k", "2.5", "--lambda", "1200", "--table", "0", "2000", "500"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "2000.0000" in out


def test_main_scale_alias_accepted(capsys):
    rc = main(["-x", "500", "--shape", "2", "--scale", "1000"])
    assert rc == 0
    assert "Weibull" in capsys.readouterr().out


def test_main_json_format(capsys):
    rc = main(["-x", "500", "-k", "2", "--lambda", "1000", "--format", "json"])
    out = capsys.readouterr().out
    assert rc == 0
    assert json.loads(out)["k"] == pytest.approx(2.0)


def test_main_missing_x_returns_2(capsys):
    assert main(["-k", "2", "--lambda", "1000"]) == 2
    assert "Error" in capsys.readouterr().err


def test_main_invalid_shape_returns_2(capsys):
    assert main(["-x", "500", "-k", "0", "--lambda", "1000"]) == 2


def test_main_computation_error_returns_2(monkeypatch, capsys):
    """Cover the ValueError branch in main when a core function raises."""

    def raise_value_error(*_args, **_kwargs):
        raise ValueError("forced weibull error")

    monkeypatch.setattr(weibull_module, "weibull_mean", raise_value_error)
    assert main(["-x", "500", "-k", "2", "--lambda", "1000"]) == 2
    assert "forced weibull error" in capsys.readouterr().err
