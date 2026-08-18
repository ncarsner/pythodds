"""Tests for the break-even analysis utility."""

import argparse
import json

import pytest

import src.utils.break_even as break_even_module
from src.utils.break_even import (
    breakeven_revenue,
    breakeven_units,
    build_result,
    contribution_margin,
    contribution_margin_ratio,
    format_chart,
    format_csv,
    format_json,
    format_table,
    main,
    margin_of_safety,
    profit_at,
    sweep_rows,
    target_profit_units,
    validate,
)


def _args(**overrides):
    """Build a namespace with valid defaults, overridden per test."""
    base = dict(
        fixed=50000.0,
        price=25.0,
        variable=10.0,
        target_profit=None,
        actual_units=None,
        sweep=None,
        chart=False,
        format="table",
        precision=2,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


# ---------------------------------------------------------------------------
# contribution_margin / contribution_margin_ratio
# ---------------------------------------------------------------------------


def test_contribution_margin_matches_formula():
    assert contribution_margin(25.0, 10.0) == pytest.approx(15.0)


def test_contribution_margin_ratio_matches_formula():
    assert contribution_margin_ratio(25.0, 10.0) == pytest.approx(0.6)


def test_contribution_margin_zero_variable_cost():
    assert contribution_margin_ratio(25.0, 0.0) == pytest.approx(1.0)


def test_contribution_margin_price_not_positive_raises():
    with pytest.raises(ValueError, match="price must be > 0"):
        contribution_margin(0.0, 10.0)


def test_contribution_margin_negative_variable_raises():
    with pytest.raises(ValueError, match="variable must be >= 0"):
        contribution_margin(25.0, -1.0)


def test_contribution_margin_ratio_invalid_price_raises():
    with pytest.raises(ValueError, match="price must be > 0"):
        contribution_margin_ratio(-5.0, 1.0)


# ---------------------------------------------------------------------------
# breakeven_units / breakeven_revenue
# ---------------------------------------------------------------------------


def test_breakeven_units_matches_formula():
    assert breakeven_units(50000.0, 25.0, 10.0) == pytest.approx(50000 / 15)


def test_breakeven_units_zero_fixed_costs():
    assert breakeven_units(0.0, 25.0, 10.0) == pytest.approx(0.0)


def test_breakeven_units_no_margin_raises():
    with pytest.raises(ValueError, match="price must exceed variable cost"):
        breakeven_units(50000.0, 10.0, 10.0)


def test_breakeven_units_negative_fixed_raises():
    with pytest.raises(ValueError, match="fixed must be >= 0"):
        breakeven_units(-1.0, 25.0, 10.0)


def test_breakeven_revenue_matches_units_times_price():
    units = breakeven_units(50000.0, 25.0, 10.0)
    ratio = contribution_margin_ratio(25.0, 10.0)
    assert breakeven_revenue(50000.0, ratio) == pytest.approx(units * 25.0)


def test_breakeven_revenue_negative_fixed_raises():
    with pytest.raises(ValueError, match="fixed must be >= 0"):
        breakeven_revenue(-1.0, 0.5)


@pytest.mark.parametrize("ratio", [0.0, -0.2, 1.5])
def test_breakeven_revenue_invalid_ratio_raises(ratio):
    with pytest.raises(ValueError, match=r"margin_ratio must be in \(0, 1\]"):
        breakeven_revenue(50000.0, ratio)


# ---------------------------------------------------------------------------
# margin_of_safety
# ---------------------------------------------------------------------------


def test_margin_of_safety_above_breakeven():
    assert margin_of_safety(5000.0, 4000.0) == pytest.approx(0.2)


def test_margin_of_safety_below_breakeven_is_negative():
    assert margin_of_safety(1000.0, 2000.0) == pytest.approx(-1.0)


def test_margin_of_safety_at_breakeven_is_zero():
    assert margin_of_safety(3000.0, 3000.0) == pytest.approx(0.0)


def test_margin_of_safety_non_positive_actual_raises():
    with pytest.raises(ValueError, match="actual_units must be > 0"):
        margin_of_safety(0.0, 100.0)


def test_margin_of_safety_negative_breakeven_raises():
    with pytest.raises(ValueError, match="be_units must be >= 0"):
        margin_of_safety(100.0, -1.0)


# ---------------------------------------------------------------------------
# target_profit_units / profit_at
# ---------------------------------------------------------------------------


def test_target_profit_units_matches_formula():
    assert target_profit_units(50000.0, 25.0, 10.0, 20000.0) == pytest.approx(
        70000 / 15
    )


def test_target_profit_zero_equals_breakeven():
    assert target_profit_units(50000.0, 25.0, 10.0, 0.0) == pytest.approx(
        breakeven_units(50000.0, 25.0, 10.0)
    )


def test_target_profit_units_accepts_negative_target():
    assert target_profit_units(50000.0, 25.0, 10.0, -15000.0) == pytest.approx(
        35000 / 15
    )


def test_target_profit_units_no_margin_raises():
    with pytest.raises(ValueError, match="price must exceed variable cost"):
        target_profit_units(50000.0, 10.0, 12.0, 100.0)


def test_target_profit_units_negative_fixed_raises():
    with pytest.raises(ValueError, match="fixed must be >= 0"):
        target_profit_units(-1.0, 25.0, 10.0, 100.0)


def test_profit_at_breakeven_volume_is_zero():
    units = breakeven_units(50000.0, 25.0, 10.0)
    assert profit_at(units, 50000.0, 25.0, 10.0) == pytest.approx(0.0)


def test_profit_at_zero_units_is_negative_fixed():
    assert profit_at(0.0, 50000.0, 25.0, 10.0) == pytest.approx(-50000.0)


def test_profit_at_negative_units_raises():
    with pytest.raises(ValueError, match="units must be >= 0"):
        profit_at(-1.0, 50000.0, 25.0, 10.0)


def test_profit_at_negative_fixed_raises():
    with pytest.raises(ValueError, match="fixed must be >= 0"):
        profit_at(10.0, -1.0, 25.0, 10.0)


# ---------------------------------------------------------------------------
# sweep_rows
# ---------------------------------------------------------------------------


def test_sweep_rows_row_count_and_endpoints():
    rows = sweep_rows(0.0, 8000.0, 2000.0, 50000.0, 25.0, 10.0)
    assert len(rows) == 5
    assert rows[0][0] == pytest.approx(0.0)
    assert rows[-1][0] == pytest.approx(8000.0)


def test_sweep_rows_profit_is_revenue_minus_cost():
    rows = sweep_rows(0.0, 4000.0, 2000.0, 50000.0, 25.0, 10.0)
    for _units, revenue, cost, profit in rows:
        assert profit == pytest.approx(revenue - cost)


def test_sweep_rows_crosses_zero_at_breakeven():
    rows = sweep_rows(0.0, 8000.0, 1000.0, 50000.0, 25.0, 10.0)
    profits = [row[3] for row in rows]
    assert profits[0] < 0 < profits[-1]


def test_sweep_rows_single_row_when_min_equals_max():
    assert len(sweep_rows(100.0, 100.0, 50.0, 1000.0, 25.0, 10.0)) == 1


def test_sweep_rows_non_positive_step_raises():
    with pytest.raises(ValueError, match="step must be > 0"):
        sweep_rows(0.0, 100.0, 0.0, 1000.0, 25.0, 10.0)


def test_sweep_rows_negative_min_raises():
    with pytest.raises(ValueError, match="min_units must be >= 0"):
        sweep_rows(-1.0, 100.0, 10.0, 1000.0, 25.0, 10.0)


def test_sweep_rows_max_below_min_raises():
    with pytest.raises(ValueError, match="max_units must be >= min_units"):
        sweep_rows(100.0, 50.0, 10.0, 1000.0, 25.0, 10.0)


def test_sweep_rows_negative_fixed_raises():
    with pytest.raises(ValueError, match="fixed must be >= 0"):
        sweep_rows(0.0, 100.0, 10.0, -1.0, 25.0, 10.0)


def test_sweep_rows_invalid_price_raises():
    with pytest.raises(ValueError, match="price must be > 0"):
        sweep_rows(0.0, 100.0, 10.0, 1000.0, 0.0, 10.0)


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------


def test_validate_accepts_minimal_args():
    assert validate(_args()) is None


def test_validate_negative_fixed():
    result = validate(_args(fixed=-1.0))
    assert result is not None and "--fixed" in result


def test_validate_non_positive_price():
    result = validate(_args(price=0.0))
    assert result is not None and "--price must be > 0" in result


def test_validate_negative_variable():
    result = validate(_args(variable=-2.0))
    assert result is not None and "--variable" in result


def test_validate_price_not_above_variable():
    result = validate(_args(price=10.0, variable=10.0))
    assert result is not None and "--price must exceed --variable" in result


def test_validate_negative_precision():
    result = validate(_args(precision=-1))
    assert result is not None and "--precision" in result


def test_validate_non_positive_actual_units():
    result = validate(_args(actual_units=0.0))
    assert result is not None and "--actual-units" in result


def test_validate_sweep_negative_min():
    result = validate(_args(sweep=[-1.0, 100.0, 10.0]))
    assert result is not None and "--sweep MIN" in result


def test_validate_sweep_max_below_min():
    result = validate(_args(sweep=[100.0, 50.0, 10.0]))
    assert result is not None and "--sweep MAX" in result


def test_validate_sweep_non_positive_step():
    result = validate(_args(sweep=[0.0, 100.0, 0.0]))
    assert result is not None and "--sweep STEP" in result


def test_validate_sweep_valid():
    assert validate(_args(sweep=[0.0, 100.0, 10.0], chart=True)) is None


def test_validate_chart_without_sweep():
    result = validate(_args(chart=True))
    assert result is not None and "--chart requires --sweep" in result


# ---------------------------------------------------------------------------
# build_result
# ---------------------------------------------------------------------------


def test_build_result_core_keys_only():
    result = build_result(_args())
    assert result["breakeven_units"] == pytest.approx(50000 / 15)
    assert "target_profit" not in result
    assert "margin_of_safety" not in result
    assert "sweep" not in result


def test_build_result_includes_target_profit():
    result = build_result(_args(target_profit=20000.0))
    target = result["target_profit"]
    assert target["units"] == pytest.approx(70000 / 15)
    assert target["revenue"] == pytest.approx(70000 / 15 * 25.0)


def test_build_result_includes_margin_of_safety():
    result = build_result(_args(actual_units=5000.0))
    mos = result["margin_of_safety"]
    assert mos["units"] == pytest.approx(5000 - 50000 / 15)
    assert mos["profit"] == pytest.approx(25000.0)


def test_build_result_includes_sweep():
    result = build_result(_args(sweep=[0.0, 4000.0, 2000.0]))
    assert len(result["sweep"]) == 3


# ---------------------------------------------------------------------------
# format_chart
# ---------------------------------------------------------------------------


def test_format_chart_empty_rows_is_empty_string():
    assert format_chart([]) == ""


def test_format_chart_losses_left_profits_right():
    rows = sweep_rows(0.0, 8000.0, 4000.0, 50000.0, 25.0, 10.0)
    out = format_chart(rows)
    loss_line, profit_line = out.splitlines()[3], out.splitlines()[5]
    assert loss_line.index("#") < loss_line.index("|")
    assert profit_line.index("#") > profit_line.index("|")


def test_format_chart_all_zero_profit_draws_no_bars():
    # A price equal to variable cost plus zero fixed costs makes every row 0.
    rows = sweep_rows(0.0, 2.0, 1.0, 0.0, 10.0, 10.0)
    assert "#" not in format_chart(rows)


# ---------------------------------------------------------------------------
# format_table / format_csv / format_json
# ---------------------------------------------------------------------------


def test_format_table_core_only():
    out = format_table(build_result(_args()), 2, chart=False)
    assert "Break-even units" in out
    assert "Target profit" not in out
    assert "Margin of safety" not in out


def test_format_table_with_optional_sections():
    result = build_result(_args(target_profit=20000.0, actual_units=5000.0))
    out = format_table(result, 2, chart=False)
    assert "Target profit" in out
    assert "Margin of safety" in out


def test_format_table_sweep_without_chart():
    result = build_result(_args(sweep=[0.0, 4000.0, 2000.0]))
    out = format_table(result, 2, chart=False)
    assert "profit" in out
    assert "Profit / loss chart" not in out


def test_format_table_sweep_with_chart():
    result = build_result(_args(sweep=[0.0, 4000.0, 2000.0]))
    out = format_table(result, 2, chart=True)
    assert "Profit / loss chart" in out


def test_format_csv_core_only():
    out = format_csv(build_result(_args()), 2)
    assert out.splitlines()[0] == "metric,value"
    assert "breakeven_units,3333.33" in out


def test_format_csv_with_optional_sections():
    result = build_result(
        _args(target_profit=20000.0, actual_units=5000.0, sweep=[0.0, 2000.0, 1000.0])
    )
    out = format_csv(result, 2)
    assert "target_profit_units," in out
    assert "margin_of_safety_ratio," in out
    assert "units,revenue,cost,profit" in out


def test_format_json_core_only():
    data = json.loads(format_json(build_result(_args())))
    assert data["breakeven_units"] == pytest.approx(50000 / 15)
    assert "sweep" not in data


def test_format_json_sweep_rows_become_objects():
    data = json.loads(format_json(build_result(_args(sweep=[0.0, 2000.0, 1000.0]))))
    assert data["sweep"][0]["units"] == pytest.approx(0.0)
    assert data["sweep"][0]["profit"] == pytest.approx(-50000.0)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def test_main_basic_report(capsys):
    rc = main(["--fixed", "50000", "--price", "25", "--variable", "10"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Break-even units" in out


def test_main_json_format(capsys):
    rc = main(
        ["--fixed", "50000", "--price", "25", "--variable", "10", "--format", "json"]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert json.loads(out)["contribution_margin"] == pytest.approx(15.0)


def test_main_csv_format(capsys):
    rc = main(
        ["--fixed", "50000", "--price", "25", "--variable", "10", "--format", "csv"]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert out.startswith("metric,value")


def test_main_sweep_with_chart(capsys):
    rc = main(
        [
            "--fixed",
            "50000",
            "--price",
            "25",
            "--variable",
            "10",
            "--sweep",
            "0",
            "8000",
            "2000",
            "--chart",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "Profit / loss chart" in out


def test_main_invalid_margin_returns_2(capsys):
    assert main(["--fixed", "50000", "--price", "10", "--variable", "10"]) == 2
    assert "Error" in capsys.readouterr().err


def test_main_chart_without_sweep_returns_2(capsys):
    rc = main(["--fixed", "50000", "--price", "25", "--variable", "10", "--chart"])
    assert rc == 2
    assert "--chart requires --sweep" in capsys.readouterr().err


def test_main_computation_error_returns_2(monkeypatch, capsys):
    """Cover the ValueError branch in main when a core function raises."""

    def raise_value_error(*_args, **_kwargs):
        raise ValueError("forced break-even error")

    monkeypatch.setattr(break_even_module, "breakeven_units", raise_value_error)
    assert main(["--fixed", "50000", "--price", "25", "--variable", "10"]) == 2
    assert "forced break-even error" in capsys.readouterr().err
