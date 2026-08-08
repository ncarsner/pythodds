"""Tests for the chi-square test calculator utility."""

import argparse
import math

import pytest

import src.utils.chi_squared as chisq_module
from src.utils.chi_squared import (
    chi2_cdf,
    chi2_sf,
    chisq_gof,
    chisq_independence,
    format_gof,
    format_independence,
    main,
    parse_number_list,
    regularized_gamma_p,
    validate,
)

# ---------------------------------------------------------------------------
# regularized_gamma_p / chi2_cdf / chi2_sf
# ---------------------------------------------------------------------------


def test_regularized_gamma_p_zero_x():
    assert regularized_gamma_p(2.0, 0.0) == 0.0


def test_regularized_gamma_p_series_branch():
    # x < a + 1 triggers the series expansion
    assert regularized_gamma_p(5.0, 2.0) == pytest.approx(0.0526530, abs=1e-6)


def test_regularized_gamma_p_cf_branch():
    # x >= a + 1 triggers the continued-fraction branch
    assert regularized_gamma_p(2.0, 10.0) == pytest.approx(0.9995006, abs=1e-6)


def test_regularized_gamma_p_bounds():
    assert 0.0 <= regularized_gamma_p(3.0, 3.0) <= 1.0


def test_gamma_cf_fpmin_guards_hit(monkeypatch):
    """Force the near-zero denominator safety branches in the continued fraction."""
    monkeypatch.setattr(chisq_module, "_FPMIN", 1e5)
    result = chisq_module._gamma_cf(2.0, 10.0)
    assert math.isfinite(result)


def test_regularized_gamma_p_a_non_positive_raises():
    with pytest.raises(ValueError, match="a must be > 0"):
        regularized_gamma_p(0.0, 1.0)


def test_regularized_gamma_p_x_negative_raises():
    with pytest.raises(ValueError, match="x must be >= 0"):
        regularized_gamma_p(1.0, -1.0)


def test_chi2_cdf_matches_scipy_oracle():
    scipy_stats = pytest.importorskip("scipy.stats")
    for df in (1, 2, 5, 10, 30, 100):
        for x in (0.5, 1.0, df * 0.5, df, df * 2.0, df * 5.0 + 10):
            got = chi2_cdf(x, df)
            want = scipy_stats.chi2.cdf(x, df)
            assert got == pytest.approx(want, abs=1e-9)


def test_chi2_cdf_invalid_df_raises():
    with pytest.raises(ValueError, match="df must be >= 1"):
        chi2_cdf(1.0, 0)


def test_chi2_sf_complements_cdf():
    assert chi2_cdf(5.0, 3) + chi2_sf(5.0, 3) == pytest.approx(1.0)


def test_chi2_sf_clips_to_unit_interval():
    assert 0.0 <= chi2_sf(0.0, 1) <= 1.0


# ---------------------------------------------------------------------------
# chisq_gof
# ---------------------------------------------------------------------------


def test_chisq_gof_matches_scipy_oracle():
    scipy_stats = pytest.importorskip("scipy.stats")
    observed = [18, 22, 17, 25, 19, 19]
    expected = [20, 20, 20, 20, 20, 20]
    result = chisq_gof(observed, expected)
    want = scipy_stats.chisquare(observed, f_exp=expected)
    assert result.statistic == pytest.approx(want.statistic)
    assert result.p_value == pytest.approx(want.pvalue)
    assert result.df == 5


def test_chisq_gof_contributions_sum_to_statistic():
    result = chisq_gof([52, 48], [50, 50])
    assert sum(result.contributions) == pytest.approx(result.statistic)


def test_chisq_gof_length_mismatch_raises():
    with pytest.raises(ValueError, match="same length"):
        chisq_gof([1, 2, 3], [1, 2])


def test_chisq_gof_too_few_categories_raises():
    with pytest.raises(ValueError, match="at least 2"):
        chisq_gof([10], [10])


def test_chisq_gof_negative_observed_raises():
    with pytest.raises(ValueError, match="non-negative"):
        chisq_gof([-1, 5], [2, 2])


def test_chisq_gof_non_positive_expected_raises():
    with pytest.raises(ValueError, match="strictly positive"):
        chisq_gof([1, 5], [0, 2])


def test_chisq_gof_invalid_alpha_raises():
    with pytest.raises(ValueError, match="alpha must be"):
        chisq_gof([1, 2], [1, 2], alpha=1.5)


# ---------------------------------------------------------------------------
# chisq_independence
# ---------------------------------------------------------------------------


def test_chisq_independence_matches_scipy_oracle():
    scipy_stats = pytest.importorskip("scipy.stats")
    table = [[40, 30, 20], [25, 45, 30]]
    result = chisq_independence(table)
    want = scipy_stats.chi2_contingency(table)
    assert result.statistic == pytest.approx(want.statistic)
    assert result.p_value == pytest.approx(want.pvalue)
    assert result.df == want.dof


def test_chisq_independence_expected_totals_match_observed():
    table = [[40, 30, 20], [25, 45, 30]]
    result = chisq_independence(table)
    assert sum(sum(row) for row in result.expected) == pytest.approx(
        sum(sum(row) for row in table)
    )


def test_chisq_independence_too_few_rows_raises():
    with pytest.raises(ValueError, match="at least 2 rows"):
        chisq_independence([[1, 2, 3]])


def test_chisq_independence_too_few_columns_raises():
    with pytest.raises(ValueError, match="at least 2 columns"):
        chisq_independence([[5], [6]])


def test_chisq_independence_ragged_rows_raises():
    with pytest.raises(ValueError, match="same number of columns"):
        chisq_independence([[1, 2, 3], [4, 5]])


def test_chisq_independence_negative_value_raises():
    with pytest.raises(ValueError, match="non-negative"):
        chisq_independence([[1, -2], [3, 4]])


def test_chisq_independence_zero_grand_total_raises():
    with pytest.raises(ValueError):
        chisq_independence([[0, 0], [0, 0]])


def test_chisq_independence_zero_row_total_raises():
    with pytest.raises(ValueError, match="row and column total"):
        chisq_independence([[0, 0], [3, 4]])


def test_chisq_independence_zero_col_total_raises():
    with pytest.raises(ValueError, match="row and column total"):
        chisq_independence([[0, 3], [0, 4]])


def test_chisq_independence_invalid_alpha_raises():
    with pytest.raises(ValueError, match="alpha must be"):
        chisq_independence([[1, 2], [3, 4]], alpha=0.0)


# ---------------------------------------------------------------------------
# parse_number_list
# ---------------------------------------------------------------------------


def test_parse_number_list_basic():
    assert parse_number_list("1,2.5,3") == [1.0, 2.5, 3.0]


def test_parse_number_list_empty_raises():
    with pytest.raises(ValueError, match="at least one value"):
        parse_number_list("")


def test_parse_number_list_non_finite_raises():
    with pytest.raises(ValueError, match="finite"):
        parse_number_list("1,nan,3")


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------


def _ns(**kwargs):
    defaults = dict(
        test="gof", observed=None, expected=None, table=None, alpha=0.05, precision=4
    )
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def test_validate_alpha_out_of_range():
    result = validate(_ns(alpha=0.0))
    assert result is not None and "--alpha" in result


def test_validate_precision_negative():
    result = validate(_ns(precision=-1))
    assert result is not None and "--precision" in result


def test_validate_gof_missing_observed():
    result = validate(_ns(test="gof", expected="1,2"))
    assert result is not None and "requires both" in result


def test_validate_gof_with_table_errors():
    result = validate(_ns(test="gof", observed="1,2", expected="1,2", table=["1,2"]))
    assert result is not None and "--table" in result


def test_validate_gof_valid():
    assert validate(_ns(test="gof", observed="1,2", expected="1,2")) is None


def test_validate_independence_missing_table():
    result = validate(_ns(test="independence", table=None))
    assert result is not None and "at least two" in result


def test_validate_independence_single_row():
    result = validate(_ns(test="independence", table=["1,2"]))
    assert result is not None and "at least two" in result


def test_validate_independence_with_observed_errors():
    result = validate(_ns(test="independence", table=["1,2", "3,4"], observed="1,2"))
    assert result is not None and "not used" in result


def test_validate_independence_valid():
    assert validate(_ns(test="independence", table=["1,2", "3,4"])) is None


# ---------------------------------------------------------------------------
# format_gof / format_independence
# ---------------------------------------------------------------------------


def test_format_gof_contains_statistic():
    result = chisq_gof([18, 22, 17, 25, 19, 19], [20, 20, 20, 20, 20, 20])
    out = format_gof(result, 4)
    assert "χ² statistic" in out
    assert "Fail to reject" in out


def test_format_gof_reject_decision():
    result = chisq_gof([90, 10], [50, 50])
    out = format_gof(result, 4)
    assert "Reject H" in out


def test_format_independence_contains_tables():
    result = chisq_independence([[40, 30, 20], [25, 45, 30]])
    out = format_independence(result, 4)
    assert "Observed:" in out
    assert "Expected:" in out
    assert "Reject H" in out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def test_main_gof_success(capsys):
    rc = main(
        [
            "--test",
            "gof",
            "--observed",
            "18,22,17,25,19,19",
            "--expected",
            "20,20,20,20,20,20",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "χ² statistic" in out


def test_main_independence_success(capsys):
    rc = main(["--test", "independence", "--table", "40,30,20", "--table", "25,45,30"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Observed:" in out


def test_main_alpha_sweep(capsys):
    rc = main(
        [
            "--test",
            "gof",
            "--observed",
            "52,48",
            "--expected",
            "50,50",
            "--alpha",
            "0.10",
        ]
    )
    assert rc == 0


def test_main_validate_error_returns_2(capsys):
    assert main(["--test", "gof"]) == 2
    err = capsys.readouterr().err
    assert "Error" in err


def test_main_computation_error_returns_2(monkeypatch, capsys):
    """Cover the ValueError branch in main when a core function raises."""

    def raise_value_error(*_args, **_kwargs):
        raise ValueError("forced chisq error")

    monkeypatch.setattr(chisq_module, "chisq_gof", raise_value_error)
    assert main(["--test", "gof", "--observed", "1,2", "--expected", "1,2"]) == 2
    err = capsys.readouterr().err
    assert "forced chisq error" in err


def test_main_computation_error_independence(monkeypatch, capsys):
    """Cover the ValueError branch in main for the independence path."""

    def raise_value_error(*_args, **_kwargs):
        raise ValueError("forced independence error")

    monkeypatch.setattr(chisq_module, "chisq_independence", raise_value_error)
    assert main(["--test", "independence", "--table", "1,2", "--table", "3,4"]) == 2
    err = capsys.readouterr().err
    assert "forced independence error" in err
