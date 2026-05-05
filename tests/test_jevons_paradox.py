"""Tests for Jevons' Paradox (rebound effect) analysis."""

from __future__ import annotations

import argparse
import json
import math

import pytest

import src.utils.jevons_paradox as jp_module
from src.utils.jevons_paradox import (
    _sweep_values,
    actual_savings,
    analyze,
    expected_savings,
    format_analysis,
    format_sweep_table,
    is_backfire,
    main,
    net_consumption,
    outcome_label,
    rebound_consumption,
    rebound_rate,
    validate,
)

# ---------------------------------------------------------------------------
# expected_savings
# ---------------------------------------------------------------------------


def test_expected_savings_basic():
    assert math.isclose(expected_savings(100.0, 0.3), 30.0)


def test_expected_savings_full_range():
    assert math.isclose(expected_savings(200.0, 0.5), 100.0)


def test_expected_savings_invalid_baseline_raises():
    with pytest.raises(ValueError, match="baseline must be positive"):
        expected_savings(0.0, 0.3)


def test_expected_savings_negative_baseline_raises():
    with pytest.raises(ValueError, match="baseline must be positive"):
        expected_savings(-10.0, 0.3)


def test_expected_savings_eta_zero_raises():
    with pytest.raises(ValueError, match="eta.*must be in"):
        expected_savings(100.0, 0.0)


def test_expected_savings_eta_one_raises():
    with pytest.raises(ValueError, match="eta.*must be in"):
        expected_savings(100.0, 1.0)


# ---------------------------------------------------------------------------
# rebound_consumption
# ---------------------------------------------------------------------------


def test_rebound_consumption_known_values():
    # baseline=1, eta=0.30, epsilon=0.5 → 1 × 0.5 × 0.30 × 0.70 = 0.105
    assert math.isclose(rebound_consumption(1.0, 0.30, 0.5), 0.105)


def test_rebound_consumption_zero_elasticity():
    assert rebound_consumption(100.0, 0.3, 0.0) == 0.0


def test_rebound_consumption_invalid_baseline_raises():
    with pytest.raises(ValueError, match="baseline must be positive"):
        rebound_consumption(0.0, 0.3, 0.5)


def test_rebound_consumption_invalid_eta_raises():
    with pytest.raises(ValueError, match="eta.*must be in"):
        rebound_consumption(1.0, 1.0, 0.5)


def test_rebound_consumption_negative_epsilon_raises():
    with pytest.raises(ValueError, match="elasticity.*non-negative"):
        rebound_consumption(1.0, 0.3, -0.1)


# ---------------------------------------------------------------------------
# net_consumption
# ---------------------------------------------------------------------------


def test_net_consumption_known_values():
    # (1 + 0.5×0.30) × (1−0.30) = 1.15 × 0.70 = 0.805
    assert math.isclose(net_consumption(1.0, 0.30, 0.5), 0.805)


def test_net_consumption_zero_elasticity_equals_no_rebound():
    # With zero elasticity, net = baseline × (1 - eta)
    assert math.isclose(net_consumption(100.0, 0.4, 0.0), 60.0)


def test_net_consumption_invalid_baseline_raises():
    with pytest.raises(ValueError, match="baseline must be positive"):
        net_consumption(-1.0, 0.3, 0.5)


def test_net_consumption_invalid_eta_raises():
    with pytest.raises(ValueError, match="eta.*must be in"):
        net_consumption(1.0, 0.0, 0.5)


def test_net_consumption_negative_epsilon_raises():
    with pytest.raises(ValueError, match="elasticity.*non-negative"):
        net_consumption(1.0, 0.3, -1.0)


# ---------------------------------------------------------------------------
# rebound_rate
# ---------------------------------------------------------------------------


def test_rebound_rate_known_values():
    # 0.5 × (1 − 0.30) = 0.35
    assert math.isclose(rebound_rate(0.30, 0.5), 0.35)


def test_rebound_rate_zero_elasticity_gives_zero():
    assert rebound_rate(0.3, 0.0) == 0.0


def test_rebound_rate_backfire_threshold():
    # eta=0.30, epsilon=1/(1-0.30)=1/0.70 ≈ 1.4286 → rate = 1.0
    eps = 1.0 / 0.70
    assert math.isclose(rebound_rate(0.30, eps), 1.0)


def test_rebound_rate_invalid_eta_raises():
    with pytest.raises(ValueError, match="eta.*must be in"):
        rebound_rate(0.0, 0.5)


def test_rebound_rate_negative_epsilon_raises():
    with pytest.raises(ValueError, match="elasticity.*non-negative"):
        rebound_rate(0.3, -0.1)


# ---------------------------------------------------------------------------
# actual_savings
# ---------------------------------------------------------------------------


def test_actual_savings_known_values():
    # expected=0.30, rebound=0.105 → 0.195
    assert math.isclose(actual_savings(1.0, 0.30, 0.5), 0.195)


def test_actual_savings_negative_for_backfire():
    # rebound_rate = 1.5 × 0.70 = 1.05 > 1 → actual savings negative
    assert actual_savings(1.0, 0.30, 1.5) < 0


# ---------------------------------------------------------------------------
# is_backfire
# ---------------------------------------------------------------------------


def test_is_backfire_false_for_low_elasticity():
    assert not is_backfire(0.30, 0.5)


def test_is_backfire_true_for_high_elasticity():
    assert is_backfire(0.30, 1.5)


def test_is_backfire_false_at_exact_boundary():
    # rate == 1.0 is NOT backfire (consumption does not exceed baseline)
    eps = 1.0 / (1.0 - 0.30)
    assert not is_backfire(0.30, eps)


# ---------------------------------------------------------------------------
# outcome_label
# ---------------------------------------------------------------------------


def test_outcome_label_zero_rate():
    assert "Full conservation" in outcome_label(0.0)


def test_outcome_label_weak_rebound():
    assert "Weak rebound" in outcome_label(0.3)


def test_outcome_label_strong_rebound():
    assert "Strong rebound" in outcome_label(0.7)


def test_outcome_label_full_rebound():
    assert "Full rebound" in outcome_label(1.0)


def test_outcome_label_backfire():
    assert "BACKFIRE" in outcome_label(1.5)


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------


def test_analyze_returns_all_keys():
    result = analyze(1.0, 0.30, 0.5)
    expected_keys = {
        "baseline",
        "eta",
        "epsilon",
        "expected_savings",
        "rebound_consumption",
        "actual_savings",
        "net_consumption",
        "rebound_rate",
        "rebound_pct",
        "backfire",
    }
    assert set(result.keys()) == expected_keys


def test_analyze_known_values():
    result = analyze(1.0, 0.30, 0.5)
    assert math.isclose(float(result["expected_savings"]), 0.30)
    assert math.isclose(float(result["rebound_consumption"]), 0.105)
    assert math.isclose(float(result["actual_savings"]), 0.195)
    assert math.isclose(float(result["net_consumption"]), 0.805)
    assert math.isclose(float(result["rebound_rate"]), 0.35)
    assert math.isclose(float(result["rebound_pct"]), 35.0)
    assert result["backfire"] is False


def test_analyze_backfire_scenario():
    result = analyze(1.0, 0.30, 1.5)
    assert result["backfire"] is True
    assert float(result["net_consumption"]) > 1.0


# ---------------------------------------------------------------------------
# _sweep_values
# ---------------------------------------------------------------------------


def test_sweep_values_basic():
    vals = _sweep_values(0.1, 0.3, 0.1)
    assert len(vals) == 3
    assert math.isclose(vals[0], 0.1)
    assert math.isclose(vals[1], 0.2)
    assert math.isclose(vals[2], 0.3)


def test_sweep_values_single_step():
    vals = _sweep_values(0.5, 0.5, 0.1)
    assert len(vals) == 1
    assert math.isclose(vals[0], 0.5)


# ---------------------------------------------------------------------------
# format_analysis
# ---------------------------------------------------------------------------


def test_format_analysis_contains_header():
    result = analyze(100.0, 0.30, 0.5)
    output = format_analysis(result, "coal", 4)
    assert "Jevons' Paradox Analysis" in output


def test_format_analysis_contains_efficiency():
    result = analyze(100.0, 0.30, 0.5)
    output = format_analysis(result, "units", 2)
    assert "30.00%" in output


def test_format_analysis_contains_outcome():
    result = analyze(100.0, 0.30, 0.5)
    output = format_analysis(result, "units", 4)
    assert "Outcome:" in output


def test_format_analysis_backfire_label():
    result = analyze(100.0, 0.30, 2.0)
    output = format_analysis(result, "units", 4)
    assert "BACKFIRE" in output


def test_format_analysis_resource_label():
    result = analyze(100.0, 0.30, 0.5)
    output = format_analysis(result, "barrels", 2)
    assert "barrels" in output


# ---------------------------------------------------------------------------
# format_sweep_table
# ---------------------------------------------------------------------------


def test_format_sweep_table_has_header_and_rows():
    rows = [analyze(1.0, 0.2, 0.5), analyze(1.0, 0.4, 0.5)]
    output = format_sweep_table(rows, "eta", 2)
    lines = output.splitlines()
    # header + separator + 2 data rows
    assert len(lines) == 4


def test_format_sweep_table_backfire_label():
    rows = [analyze(1.0, 0.3, 2.0)]
    output = format_sweep_table(rows, "epsilon", 2)
    assert "YES" in output


def test_format_sweep_table_no_backfire_label():
    rows = [analyze(1.0, 0.3, 0.5)]
    output = format_sweep_table(rows, "eta", 2)
    assert "no" in output


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------


def test_validate_valid_single():
    args = argparse.Namespace(
        efficiency=0.30,
        elasticity=0.5,
        baseline=1.0,
        step=0.05,
        precision=4,
        sweep_efficiency=None,
        sweep_elasticity=None,
        resource="units",
        format="table",
    )
    assert validate(args) is None


def test_validate_missing_efficiency():
    args = argparse.Namespace(
        efficiency=None,
        elasticity=0.5,
        baseline=1.0,
        step=0.05,
        precision=4,
        sweep_efficiency=None,
        sweep_elasticity=None,
        resource="units",
        format="table",
    )
    assert validate(args) == "--efficiency is required"


def test_validate_missing_elasticity():
    args = argparse.Namespace(
        efficiency=0.3,
        elasticity=None,
        baseline=1.0,
        step=0.05,
        precision=4,
        sweep_efficiency=None,
        sweep_elasticity=None,
        resource="units",
        format="table",
    )
    assert validate(args) == "--elasticity is required"


def test_validate_efficiency_out_of_range():
    args = argparse.Namespace(
        efficiency=1.5,
        elasticity=0.5,
        baseline=1.0,
        step=0.05,
        precision=4,
        sweep_efficiency=None,
        sweep_elasticity=None,
        resource="units",
        format="table",
    )
    result = validate(args)
    assert result is not None and "--efficiency must be in (0, 1)" in result


def test_validate_negative_elasticity():
    args = argparse.Namespace(
        efficiency=0.3,
        elasticity=-0.1,
        baseline=1.0,
        step=0.05,
        precision=4,
        sweep_efficiency=None,
        sweep_elasticity=None,
        resource="units",
        format="table",
    )
    result = validate(args)
    assert result is not None and "--elasticity must be non-negative" in result


def test_validate_negative_precision():
    args = argparse.Namespace(
        efficiency=0.3,
        elasticity=0.5,
        baseline=1.0,
        step=0.05,
        precision=-1,
        sweep_efficiency=None,
        sweep_elasticity=None,
        resource="units",
        format="table",
    )
    result = validate(args)
    assert result is not None and "--precision must be non-negative" in result


def test_validate_non_positive_baseline():
    args = argparse.Namespace(
        efficiency=0.3,
        elasticity=0.5,
        baseline=0.0,
        step=0.05,
        precision=4,
        sweep_efficiency=None,
        sweep_elasticity=None,
        resource="units",
        format="table",
    )
    result = validate(args)
    assert result is not None and "--baseline must be positive" in result


def test_validate_non_positive_step():
    args = argparse.Namespace(
        efficiency=0.3,
        elasticity=0.5,
        baseline=1.0,
        step=0.0,
        precision=4,
        sweep_efficiency=None,
        sweep_elasticity=None,
        resource="units",
        format="table",
    )
    result = validate(args)
    assert result is not None and "--step must be positive" in result


def test_validate_sweep_efficiency_missing_elasticity():
    args = argparse.Namespace(
        efficiency=None,
        elasticity=None,
        baseline=1.0,
        step=0.05,
        precision=4,
        sweep_efficiency=[0.1, 0.5],
        sweep_elasticity=None,
        resource="units",
        format="table",
    )
    result = validate(args)
    assert result is not None and "--elasticity is required" in result


def test_validate_sweep_efficiency_invalid_min():
    args = argparse.Namespace(
        efficiency=None,
        elasticity=0.5,
        baseline=1.0,
        step=0.05,
        precision=4,
        sweep_efficiency=[0.0, 0.5],
        sweep_elasticity=None,
        resource="units",
        format="table",
    )
    result = validate(args)
    assert result is not None and "MIN must be in (0, 1)" in result


def test_validate_sweep_efficiency_invalid_max():
    args = argparse.Namespace(
        efficiency=None,
        elasticity=0.5,
        baseline=1.0,
        step=0.05,
        precision=4,
        sweep_efficiency=[0.1, 1.0],
        sweep_elasticity=None,
        resource="units",
        format="table",
    )
    result = validate(args)
    assert result is not None and "MAX must be in (0, 1)" in result


def test_validate_sweep_efficiency_min_gte_max():
    args = argparse.Namespace(
        efficiency=None,
        elasticity=0.5,
        baseline=1.0,
        step=0.05,
        precision=4,
        sweep_efficiency=[0.5, 0.3],
        sweep_elasticity=None,
        resource="units",
        format="table",
    )
    result = validate(args)
    assert result is not None and "MIN must be less than MAX" in result


def test_validate_sweep_efficiency_negative_elasticity():
    args = argparse.Namespace(
        efficiency=None,
        elasticity=-0.5,
        baseline=1.0,
        step=0.05,
        precision=4,
        sweep_efficiency=[0.1, 0.5],
        sweep_elasticity=None,
        resource="units",
        format="table",
    )
    result = validate(args)
    assert result is not None and "--elasticity must be non-negative" in result


def test_validate_sweep_elasticity_missing_efficiency():
    args = argparse.Namespace(
        efficiency=None,
        elasticity=None,
        baseline=1.0,
        step=0.05,
        precision=4,
        sweep_efficiency=None,
        sweep_elasticity=[0.0, 1.0],
        resource="units",
        format="table",
    )
    result = validate(args)
    assert result is not None and "--efficiency is required" in result


def test_validate_sweep_elasticity_invalid_efficiency():
    args = argparse.Namespace(
        efficiency=0.0,
        elasticity=None,
        baseline=1.0,
        step=0.05,
        precision=4,
        sweep_efficiency=None,
        sweep_elasticity=[0.0, 1.0],
        resource="units",
        format="table",
    )
    result = validate(args)
    assert result is not None and "--efficiency must be in (0, 1)" in result


def test_validate_sweep_elasticity_negative_min():
    args = argparse.Namespace(
        efficiency=0.3,
        elasticity=None,
        baseline=1.0,
        step=0.05,
        precision=4,
        sweep_efficiency=None,
        sweep_elasticity=[-0.1, 1.0],
        resource="units",
        format="table",
    )
    result = validate(args)
    assert result is not None and "MIN must be non-negative" in result


def test_validate_sweep_elasticity_negative_max():
    args = argparse.Namespace(
        efficiency=0.3,
        elasticity=None,
        baseline=1.0,
        step=0.05,
        precision=4,
        sweep_efficiency=None,
        sweep_elasticity=[0.0, -1.0],
        resource="units",
        format="table",
    )
    result = validate(args)
    assert result is not None and "MAX must be non-negative" in result


def test_validate_sweep_elasticity_min_gte_max():
    args = argparse.Namespace(
        efficiency=0.3,
        elasticity=None,
        baseline=1.0,
        step=0.05,
        precision=4,
        sweep_efficiency=None,
        sweep_elasticity=[1.0, 0.5],
        resource="units",
        format="table",
    )
    result = validate(args)
    assert result is not None and "MIN must be less than MAX" in result


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def test_main_single_analysis_success(capsys):
    rc = main(["--efficiency", "0.30", "--elasticity", "0.5"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Jevons' Paradox Analysis" in out


def test_main_single_analysis_json(capsys):
    rc = main(["--efficiency", "0.30", "--elasticity", "0.5", "--format", "json"])
    out = capsys.readouterr().out
    assert rc == 0
    data = json.loads(out)
    assert math.isclose(data["rebound_pct"], 35.0)


def test_main_single_analysis_with_baseline_and_resource(capsys):
    rc = main(
        [
            "--efficiency",
            "0.30",
            "--elasticity",
            "0.5",
            "--baseline",
            "1000",
            "--resource",
            "barrels",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "barrels" in out


def test_main_backfire_output(capsys):
    rc = main(["--efficiency", "0.30", "--elasticity", "2.0"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "BACKFIRE" in out


def test_main_sweep_efficiency_table(capsys):
    rc = main(
        [
            "--sweep-efficiency",
            "0.1",
            "0.3",
            "--elasticity",
            "0.5",
            "--step",
            "0.1",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    lines = out.strip().splitlines()
    # header + separator + 3 data rows (0.1, 0.2, 0.3)
    assert len(lines) == 5


def test_main_sweep_efficiency_json(capsys):
    rc = main(
        [
            "--sweep-efficiency",
            "0.1",
            "0.2",
            "--elasticity",
            "0.5",
            "--step",
            "0.1",
            "--format",
            "json",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    data = json.loads(out)
    assert isinstance(data, list)
    assert len(data) == 2


def test_main_sweep_elasticity_table(capsys):
    rc = main(
        [
            "--efficiency",
            "0.30",
            "--sweep-elasticity",
            "0.0",
            "1.0",
            "--step",
            "0.5",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "Elasticity" in out


def test_main_sweep_elasticity_json(capsys):
    rc = main(
        [
            "--efficiency",
            "0.30",
            "--sweep-elasticity",
            "0.0",
            "1.0",
            "--step",
            "0.5",
            "--format",
            "json",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    data = json.loads(out)
    assert isinstance(data, list)


def test_main_missing_efficiency_returns_2(capsys):
    assert main(["--elasticity", "0.5"]) == 2


def test_main_missing_elasticity_returns_2(capsys):
    assert main(["--efficiency", "0.3"]) == 2


def test_main_efficiency_out_of_range_returns_2(capsys):
    assert main(["--efficiency", "1.5", "--elasticity", "0.5"]) == 2


def test_main_negative_elasticity_returns_2(capsys):
    assert main(["--efficiency", "0.3", "--elasticity", "-0.5"]) == 2


def test_main_negative_baseline_returns_2(capsys):
    assert main(["--efficiency", "0.3", "--elasticity", "0.5", "--baseline", "-1"]) == 2


def test_main_calculation_error_returns_2(monkeypatch, capsys):
    def raise_value_error(*_args, **_kwargs):
        raise ValueError("calculation failed")

    monkeypatch.setattr(jp_module, "analyze", raise_value_error)
    assert main(["--efficiency", "0.3", "--elasticity", "0.5"]) == 2
    err = capsys.readouterr().err
    assert "Error: calculation failed" in err
