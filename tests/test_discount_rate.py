"""Tests for the discount rate utility."""

import argparse
import json

import pytest

import src.utils.discount_rate as discount_module
from src.utils.discount_rate import (
    build_result,
    discount_factor,
    format_json,
    format_table,
    main,
    nominal_rate,
    npv,
    payback_period,
    present_value,
    real_npv,
    real_rate,
    resolve_rates,
    validate,
)

# A conventional project: up-front cost then three years of inflows.
FLOWS = [-1000.0, 300.0, 400.0, 500.0]


def _args(**overrides):
    """Build a namespace with valid defaults, overridden per test."""
    base = dict(
        nominal=0.08,
        real=None,
        inflation=None,
        fv=None,
        periods=1.0,
        cashflows=None,
        format="table",
        precision=4,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


# ---------------------------------------------------------------------------
# real_rate / nominal_rate
# ---------------------------------------------------------------------------


def test_real_rate_matches_fisher_equation():
    assert real_rate(0.08, 0.03) == pytest.approx(1.08 / 1.03 - 1)


def test_real_rate_with_zero_inflation_equals_nominal():
    assert real_rate(0.08, 0.0) == pytest.approx(0.08)


def test_real_rate_is_negative_when_inflation_outruns_nominal():
    assert real_rate(0.02, 0.05) < 0


def test_real_rate_invalid_nominal_raises():
    with pytest.raises(ValueError, match="nominal must be > -1"):
        real_rate(-1.0, 0.03)


def test_real_rate_invalid_inflation_raises():
    with pytest.raises(ValueError, match="inflation must be > -1"):
        real_rate(0.08, -1.5)


def test_nominal_rate_matches_fisher_equation():
    assert nominal_rate(0.05, 0.03) == pytest.approx(1.05 * 1.03 - 1)


def test_nominal_rate_inverts_real_rate():
    assert nominal_rate(real_rate(0.08, 0.03), 0.03) == pytest.approx(0.08)


def test_nominal_rate_invalid_real_raises():
    with pytest.raises(ValueError, match="real must be > -1"):
        nominal_rate(-1.0, 0.03)


def test_nominal_rate_invalid_inflation_raises():
    with pytest.raises(ValueError, match="inflation must be > -1"):
        nominal_rate(0.05, -1.0)


# ---------------------------------------------------------------------------
# discount_factor / present_value
# ---------------------------------------------------------------------------


def test_discount_factor_matches_formula():
    assert discount_factor(0.08, 1) == pytest.approx(1 / 1.08)


def test_discount_factor_at_zero_periods_is_one():
    assert discount_factor(0.08, 0) == pytest.approx(1.0)


def test_discount_factor_with_zero_rate_is_one():
    assert discount_factor(0.0, 10) == pytest.approx(1.0)


def test_discount_factor_negative_periods_raises():
    with pytest.raises(ValueError, match="periods must be >= 0"):
        discount_factor(0.08, -1)


def test_discount_factor_invalid_rate_raises():
    with pytest.raises(ValueError, match="rate must be > -1"):
        discount_factor(-1.0, 5)


def test_present_value_matches_formula():
    assert present_value(10000.0, 0.08, 5) == pytest.approx(10000 / 1.08**5)


def test_present_value_at_zero_periods_is_face_value():
    assert present_value(10000.0, 0.08, 0) == pytest.approx(10000.0)


def test_present_value_invalid_rate_raises():
    with pytest.raises(ValueError, match="rate must be > -1"):
        present_value(10000.0, -2.0, 5)


# ---------------------------------------------------------------------------
# npv / real_npv
# ---------------------------------------------------------------------------


def test_npv_matches_manual_sum():
    expected = sum(cf / 1.08**t for t, cf in enumerate(FLOWS))
    assert npv(0.08, FLOWS) == pytest.approx(expected)


def test_npv_at_zero_rate_is_the_plain_sum():
    assert npv(0.0, FLOWS) == pytest.approx(sum(FLOWS))


def test_npv_first_flow_is_undiscounted():
    assert npv(0.5, [-100.0]) == pytest.approx(-100.0)


def test_npv_falls_as_the_discount_rate_rises():
    assert npv(0.15, FLOWS) < npv(0.05, FLOWS)


def test_npv_empty_flows_raises():
    with pytest.raises(ValueError, match="at least one value"):
        npv(0.08, [])


def test_npv_invalid_rate_raises():
    with pytest.raises(ValueError, match="rate must be > -1"):
        npv(-1.0, FLOWS)


def test_real_npv_discounts_at_the_real_rate():
    assert real_npv(0.08, 0.03, FLOWS) == pytest.approx(
        npv(real_rate(0.08, 0.03), FLOWS)
    )


def test_real_npv_with_zero_inflation_matches_nominal_npv():
    assert real_npv(0.08, 0.0, FLOWS) == pytest.approx(npv(0.08, FLOWS))


def test_real_npv_invalid_inflation_raises():
    with pytest.raises(ValueError, match="inflation must be > -1"):
        real_npv(0.08, -1.0, FLOWS)


# ---------------------------------------------------------------------------
# payback_period
# ---------------------------------------------------------------------------


def test_payback_period_interpolates_within_the_crossing_period():
    # Cumulative discounted flow is -379.28 after period 2 and +17.63 after 3.
    assert payback_period(FLOWS, 0.08) == pytest.approx(2.9556, abs=1e-4)


def test_payback_period_zero_when_first_flow_is_already_positive():
    assert payback_period([100.0, -50.0], 0.08) == 0.0


def test_payback_period_none_when_never_recovered():
    assert payback_period([-1000.0, 100.0, 100.0], 0.08) is None


def test_payback_period_empty_flows_raises():
    with pytest.raises(ValueError, match="at least one value"):
        payback_period([], 0.08)


def test_payback_period_invalid_rate_raises():
    with pytest.raises(ValueError, match="rate must be > -1"):
        payback_period(FLOWS, -1.0)


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------


def test_validate_nominal_only():
    assert validate(_args()) is None


def test_validate_requires_a_rate():
    result = validate(_args(nominal=None))
    assert result is not None and "one of --nominal or --real" in result


def test_validate_real_without_inflation():
    result = validate(_args(nominal=None, real=0.05))
    assert result is not None and "--real requires --inflation" in result


def test_validate_real_with_inflation():
    assert validate(_args(nominal=None, real=0.05, inflation=0.03)) is None


def test_validate_nominal_at_or_below_negative_one():
    result = validate(_args(nominal=-1.0))
    assert result is not None and "--nominal must be > -1" in result


def test_validate_real_at_or_below_negative_one():
    result = validate(_args(nominal=None, real=-1.0, inflation=0.03))
    assert result is not None and "--real must be > -1" in result


def test_validate_inflation_at_or_below_negative_one():
    result = validate(_args(inflation=-1.0))
    assert result is not None and "--inflation must be > -1" in result


def test_validate_negative_periods():
    result = validate(_args(periods=-1.0))
    assert result is not None and "--periods" in result


def test_validate_negative_precision():
    result = validate(_args(precision=-1))
    assert result is not None and "--precision" in result


def test_validate_empty_cashflows():
    result = validate(_args(cashflows=[]))
    assert result is not None and "--cashflows" in result


# ---------------------------------------------------------------------------
# resolve_rates
# ---------------------------------------------------------------------------


def test_resolve_rates_nominal_only_leaves_real_unset():
    nominal, real, inflation = resolve_rates(_args())
    assert nominal == pytest.approx(0.08)
    assert real is None
    assert inflation is None


def test_resolve_rates_derives_real_from_nominal_and_inflation():
    nominal, real, inflation = resolve_rates(_args(inflation=0.03))
    assert nominal == pytest.approx(0.08)
    assert real == pytest.approx(real_rate(0.08, 0.03))
    assert inflation == pytest.approx(0.03)


def test_resolve_rates_derives_nominal_from_real_and_inflation():
    nominal, real, _inflation = resolve_rates(
        _args(nominal=None, real=0.05, inflation=0.03)
    )
    assert nominal == pytest.approx(nominal_rate(0.05, 0.03))
    assert real == pytest.approx(0.05)


def test_resolve_rates_prefers_fisher_value_over_supplied_real():
    """--nominal wins: the reported real rate is the Fisher-derived one."""
    _nominal, real, _inflation = resolve_rates(
        _args(nominal=0.08, real=0.99, inflation=0.03)
    )
    assert real == pytest.approx(real_rate(0.08, 0.03))


# ---------------------------------------------------------------------------
# build_result
# ---------------------------------------------------------------------------


def test_build_result_rates_only():
    result = build_result(_args())
    assert result["real_rate"] is None
    assert result["discount_factor_real"] is None
    assert "lump_sum" not in result
    assert "npv" not in result


def test_build_result_with_inflation():
    result = build_result(_args(inflation=0.03))
    assert result["real_rate"] == pytest.approx(real_rate(0.08, 0.03))
    assert result["discount_factor_real"] == pytest.approx(
        discount_factor(real_rate(0.08, 0.03), 1.0)
    )


def test_build_result_lump_sum_without_inflation():
    result = build_result(_args(fv=10000.0, periods=5.0))
    lump = result["lump_sum"]
    assert lump["present_value_nominal"] == pytest.approx(10000 / 1.08**5)
    assert lump["present_value_real"] is None


def test_build_result_lump_sum_with_inflation():
    result = build_result(_args(fv=10000.0, periods=5.0, inflation=0.03))
    assert result["lump_sum"]["present_value_real"] == pytest.approx(
        present_value(10000.0, real_rate(0.08, 0.03), 5.0)
    )


def test_build_result_npv_without_inflation():
    result = build_result(_args(cashflows=FLOWS))
    assert result["npv"]["nominal"] == pytest.approx(npv(0.08, FLOWS))
    assert result["npv"]["real"] is None


def test_build_result_npv_with_inflation():
    result = build_result(_args(cashflows=FLOWS, inflation=0.03))
    assert result["npv"]["real"] == pytest.approx(real_npv(0.08, 0.03, FLOWS))
    assert result["npv"]["payback_period"] == pytest.approx(2.9556, abs=1e-4)


# ---------------------------------------------------------------------------
# format_table / format_json
# ---------------------------------------------------------------------------


def test_format_table_rates_only_omits_inflation_rows():
    out = format_table(build_result(_args()), 4)
    assert "Nominal rate" in out
    assert "Inflation:" not in out
    assert "Discount factor (real)" not in out


def test_format_table_with_inflation():
    out = format_table(build_result(_args(inflation=0.03)), 4)
    assert "Real rate (Fisher)" in out
    assert "4.8544%" in out


def test_format_table_lump_sum_section():
    out = format_table(build_result(_args(fv=10000.0, periods=5.0)), 4)
    assert "Present value (nominal)" in out
    assert "Present value (real)" not in out


def test_format_table_lump_sum_real_row_with_inflation():
    out = format_table(build_result(_args(fv=10000.0, periods=5.0, inflation=0.03)), 4)
    assert "Present value (real)" in out


def test_format_table_npv_section():
    out = format_table(build_result(_args(cashflows=FLOWS, inflation=0.03)), 4)
    assert "NPV (nominal)" in out
    assert "NPV (real)" in out
    assert "purchasing power" in out


def test_format_table_npv_without_inflation_omits_real_row():
    out = format_table(build_result(_args(cashflows=FLOWS)), 4)
    assert "NPV (real)" not in out


def test_format_table_reports_never_when_payback_never_arrives():
    out = format_table(build_result(_args(cashflows=[-1000.0, 100.0])), 4)
    assert "never" in out


def test_format_json_round_trips():
    data = json.loads(format_json(build_result(_args(inflation=0.03))))
    assert data["real_rate"] == pytest.approx(real_rate(0.08, 0.03))
    assert data["inflation"] == pytest.approx(0.03)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def test_main_rate_report(capsys):
    rc = main(["--nominal", "0.08", "--inflation", "0.03"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "4.8544%" in out


def test_main_real_rate_input(capsys):
    rc = main(["--real", "0.05", "--inflation", "0.03"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "8.1500%" in out


def test_main_lump_sum(capsys):
    rc = main(["--nominal", "0.08", "--fv", "10000", "--periods", "5"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "6,805.8320" in out


def test_main_npv(capsys):
    rc = main(["--nominal", "0.08", "--cashflows", "-1000", "300", "400", "500"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "NPV (nominal)" in out


def test_main_json_format(capsys):
    rc = main(["--nominal", "0.08", "--inflation", "0.03", "--format", "json"])
    out = capsys.readouterr().out
    assert rc == 0
    assert json.loads(out)["nominal_rate"] == pytest.approx(0.08)


def test_main_missing_rate_returns_2(capsys):
    assert main([]) == 2
    assert "Error" in capsys.readouterr().err


def test_main_real_without_inflation_returns_2(capsys):
    assert main(["--real", "0.05"]) == 2
    assert "--real requires --inflation" in capsys.readouterr().err


def test_main_computation_error_returns_2(monkeypatch, capsys):
    """Cover the ValueError branch in main when a core function raises."""

    def raise_value_error(*_args, **_kwargs):
        raise ValueError("forced discount error")

    monkeypatch.setattr(discount_module, "discount_factor", raise_value_error)
    assert main(["--nominal", "0.08"]) == 2
    assert "forced discount error" in capsys.readouterr().err
