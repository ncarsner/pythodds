"""Tests for the geometric distribution utility."""

import argparse

import pytest

import src.utils.geometric_distribution as geometric_module
from src.utils.geometric_distribution import (
    format_single,
    format_table,
    geo_cdf,
    geo_mean,
    geo_pmf,
    geo_survival,
    geo_variance,
    main,
    validate,
)

# ---------------------------------------------------------------------------
# geo_pmf
# ---------------------------------------------------------------------------


def test_geo_pmf_first_trial():
    assert geo_pmf(1, 0.3) == pytest.approx(0.3)


def test_geo_pmf_matches_formula():
    assert geo_pmf(5, 0.3) == pytest.approx((0.7**4) * 0.3)


def test_geo_pmf_p_equals_one():
    assert geo_pmf(1, 1.0) == pytest.approx(1.0)
    assert geo_pmf(2, 1.0) == pytest.approx(0.0)


def test_geo_pmf_k_below_one_raises():
    with pytest.raises(ValueError, match=">= 1"):
        geo_pmf(0, 0.5)


def test_geo_pmf_p_zero_raises():
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        geo_pmf(1, 0.0)


def test_geo_pmf_p_above_one_raises():
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        geo_pmf(1, 1.5)


# ---------------------------------------------------------------------------
# geo_cdf
# ---------------------------------------------------------------------------


def test_geo_cdf_matches_formula():
    assert geo_cdf(5, 0.3) == pytest.approx(1 - 0.7**5)


def test_geo_cdf_k_below_one_is_zero():
    assert geo_cdf(0, 0.5) == 0.0
    assert geo_cdf(-3, 0.5) == 0.0


def test_geo_cdf_invalid_p_raises():
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        geo_cdf(5, 0.0)


# ---------------------------------------------------------------------------
# geo_survival
# ---------------------------------------------------------------------------


def test_geo_survival_matches_formula():
    assert geo_survival(10, 0.2) == pytest.approx(0.8**10)


def test_geo_survival_k_negative_is_one():
    assert geo_survival(-1, 0.5) == 1.0


def test_geo_survival_invalid_p_raises():
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        geo_survival(1, 0.0)


def test_geo_survival_and_cdf_complement():
    assert geo_cdf(7, 0.25) + geo_survival(7, 0.25) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# geo_mean / geo_variance
# ---------------------------------------------------------------------------


def test_geo_mean():
    assert geo_mean(0.2) == pytest.approx(5.0)


def test_geo_mean_invalid_p_raises():
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        geo_mean(0.0)


def test_geo_variance():
    assert geo_variance(0.2) == pytest.approx(0.8 / 0.04)


def test_geo_variance_p_one_is_zero():
    assert geo_variance(1.0) == pytest.approx(0.0)


def test_geo_variance_invalid_p_raises():
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        geo_variance(0.0)


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------


def test_validate_p_out_of_range():
    args = argparse.Namespace(k=1, p=0.0, table=None, precision=4)
    result = validate(args)
    assert result is not None and "-p" in result


def test_validate_precision_negative():
    args = argparse.Namespace(k=1, p=0.5, table=None, precision=-1)
    result = validate(args)
    assert result is not None and "--precision" in result


def test_validate_table_min_below_one():
    args = argparse.Namespace(k=None, p=0.5, table=[0, 5], precision=4)
    result = validate(args)
    assert result is not None and "MIN" in result


def test_validate_table_max_below_min():
    args = argparse.Namespace(k=None, p=0.5, table=[5, 2], precision=4)
    result = validate(args)
    assert result is not None and "MAX" in result


def test_validate_table_valid():
    args = argparse.Namespace(k=None, p=0.5, table=[1, 5], precision=4)
    assert validate(args) is None


def test_validate_k_missing_without_table():
    args = argparse.Namespace(k=None, p=0.5, table=None, precision=4)
    result = validate(args)
    assert result is not None and "-k is required" in result


def test_validate_k_below_one():
    args = argparse.Namespace(k=0, p=0.5, table=None, precision=4)
    result = validate(args)
    assert result is not None and "-k must be" in result


def test_validate_valid_single():
    args = argparse.Namespace(k=5, p=0.5, table=None, precision=4)
    assert validate(args) is None


# ---------------------------------------------------------------------------
# format_single / format_table
# ---------------------------------------------------------------------------


def test_format_single_cdf_mode():
    out = format_single(5, 0.3, survival=False, precision=4)
    assert "CDF" in out
    assert "P(X > 5)" not in out


def test_format_single_survival_mode():
    out = format_single(5, 0.3, survival=True, precision=4)
    assert "P(X > 5)" in out
    assert "CDF" not in out


def test_format_table_cdf_mode():
    out = format_table(1, 3, 0.25, survival=False, precision=4)
    assert "P(X <= k)" in out
    assert "3" in out


def test_format_table_survival_mode():
    out = format_table(1, 3, 0.25, survival=True, precision=4)
    assert "P(X > k)" in out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def test_main_single_report(capsys):
    rc = main(["-k", "5", "-p", "0.3"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "mean" in out


def test_main_survival_flag(capsys):
    rc = main(["-k", "10", "-p", "0.2", "--survival"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "P(X > 10)" in out


def test_main_table(capsys):
    rc = main(["-p", "0.25", "--table", "1", "15"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "15" in out


def test_main_missing_k_returns_2(capsys):
    assert main(["-p", "0.5"]) == 2
    err = capsys.readouterr().err
    assert "Error" in err


def test_main_invalid_p_returns_2(capsys):
    assert main(["-k", "1", "-p", "0"]) == 2


def test_main_computation_error_returns_2(monkeypatch, capsys):
    """Cover the ValueError branch in main when a core function raises."""

    def raise_value_error(*_args, **_kwargs):
        raise ValueError("forced geometric error")

    monkeypatch.setattr(geometric_module, "geo_pmf", raise_value_error)
    assert main(["-k", "5", "-p", "0.3"]) == 2
    err = capsys.readouterr().err
    assert "forced geometric error" in err
