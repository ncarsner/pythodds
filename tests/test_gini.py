"""Tests for Gini coefficient utility."""

import argparse
import json

import pytest

import src.utils.gini as gini_module
from src.utils.gini import (
    _gini_trapezoid,
    _weighted_mean,
    format_comparison,
    format_grouped,
    format_lorenz,
    format_single,
    gini_coefficient,
    gini_grouped,
    lorenz_curve,
    main,
    relative_mad,
    validate,
)

# ---------------------------------------------------------------------------
# _gini_trapezoid
# ---------------------------------------------------------------------------


def test_gini_trapezoid_known():
    vals = [1.0, 2.0, 3.0, 4.0, 5.0]
    w = [1.0] * 5
    assert _gini_trapezoid(vals, w) == pytest.approx(4 / 15, abs=1e-12)


def test_gini_trapezoid_zero_total():
    assert _gini_trapezoid([0.0, 0.0], [1.0, 1.0]) == 0.0


# ---------------------------------------------------------------------------
# gini_coefficient
# ---------------------------------------------------------------------------


def test_gini_coefficient_known():
    assert gini_coefficient([1, 2, 3, 4, 5]) == pytest.approx(4 / 15, abs=1e-12)


def test_gini_coefficient_perfect_equality():
    assert gini_coefficient([3.0, 3.0, 3.0]) == pytest.approx(0.0, abs=1e-12)


def test_gini_coefficient_max_inequality():
    assert gini_coefficient([0.0, 0.0, 0.0, 1.0]) == pytest.approx(0.75, abs=1e-12)


def test_gini_coefficient_corrected():
    g = gini_coefficient([1, 2, 3, 4, 5], corrected=True)
    assert g == pytest.approx(4 / 15 * 5 / 4, abs=1e-12)


def test_gini_coefficient_corrected_n2():
    g = gini_coefficient([1.0, 3.0], corrected=True)
    g_raw = gini_coefficient([1.0, 3.0], corrected=False)
    assert g == pytest.approx(g_raw * 2 / 1, abs=1e-12)


def test_gini_coefficient_corrected_n1():
    # n=1: correction skipped (n-1 = 0)
    assert gini_coefficient([5.0], corrected=True) == pytest.approx(0.0, abs=1e-12)


def test_gini_coefficient_weighted():
    # Equal data but unequal weights → non-zero Gini
    g = gini_coefficient([1.0, 3.0], weights=[3.0, 1.0])
    assert 0.0 < g < 1.0


def test_gini_coefficient_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        gini_coefficient([])


def test_gini_coefficient_negative_raises():
    with pytest.raises(ValueError, match="non-negative"):
        gini_coefficient([-1.0, 2.0, 3.0])


def test_gini_coefficient_all_zero_raises():
    with pytest.raises(ValueError, match="positive"):
        gini_coefficient([0.0, 0.0, 0.0])


def test_gini_coefficient_weights_wrong_length():
    with pytest.raises(ValueError, match="length"):
        gini_coefficient([1.0, 2.0], weights=[1.0])


def test_gini_coefficient_weights_nonpositive():
    with pytest.raises(ValueError, match="positive"):
        gini_coefficient([1.0, 2.0], weights=[0.0, 1.0])


# ---------------------------------------------------------------------------
# lorenz_curve
# ---------------------------------------------------------------------------


def test_lorenz_curve_starts_at_origin():
    pts = lorenz_curve([1, 2, 3, 4, 5])
    assert pts[0] == (0.0, 0.0)


def test_lorenz_curve_ends_at_one():
    pts = lorenz_curve([1, 2, 3, 4, 5])
    assert pts[-1][0] == pytest.approx(1.0, abs=1e-12)
    assert pts[-1][1] == pytest.approx(1.0, abs=1e-12)


def test_lorenz_curve_length():
    data = [1, 2, 3, 4, 5]
    pts = lorenz_curve(data)
    assert len(pts) == len(data) + 1


def test_lorenz_curve_monotone():
    pts = lorenz_curve([1, 2, 3, 4, 5])
    pops = [p for p, _ in pts]
    incs = [i for _, i in pts]
    assert all(pops[j] <= pops[j + 1] for j in range(len(pops) - 1))
    assert all(incs[j] <= incs[j + 1] for j in range(len(incs) - 1))


def test_lorenz_curve_perfect_equality():
    pts = lorenz_curve([3.0, 3.0, 3.0])
    for pop, inc in pts[1:]:
        assert pop == pytest.approx(inc, abs=1e-12)


def test_lorenz_curve_weighted():
    pts = lorenz_curve([1.0, 2.0], weights=[2.0, 1.0])
    assert pts[0] == (0.0, 0.0)
    assert len(pts) == 3


def test_lorenz_curve_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        lorenz_curve([])


def test_lorenz_curve_negative_raises():
    with pytest.raises(ValueError, match="non-negative"):
        lorenz_curve([-1.0, 2.0])


def test_lorenz_curve_all_zero_raises():
    with pytest.raises(ValueError, match="positive"):
        lorenz_curve([0.0, 0.0])


def test_lorenz_curve_weights_wrong_length():
    with pytest.raises(ValueError, match="length"):
        lorenz_curve([1.0, 2.0], weights=[1.0])


def test_lorenz_curve_weights_nonpositive():
    with pytest.raises(ValueError, match="positive"):
        lorenz_curve([1.0, 2.0], weights=[0.0, 1.0])


# ---------------------------------------------------------------------------
# relative_mad
# ---------------------------------------------------------------------------


def test_relative_mad_equals_two_times_gini():
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert relative_mad(data) == pytest.approx(2 * gini_coefficient(data), abs=1e-12)


def test_relative_mad_weighted():
    data = [1.0, 3.0]
    w = [2.0, 1.0]
    assert relative_mad(data, weights=w) == pytest.approx(
        2 * gini_coefficient(data, weights=w), abs=1e-12
    )


# ---------------------------------------------------------------------------
# gini_grouped
# ---------------------------------------------------------------------------


def test_gini_grouped_perfect_equality():
    groups = [(1 / 3, 1 / 3), (1 / 3, 1 / 3), (1 / 3, 1 / 3)]
    assert gini_grouped(groups) == pytest.approx(0.0, abs=1e-10)


def test_gini_grouped_two_groups_known():
    # Bottom 40% earns 10%, top 60% earns 90%
    groups = [(0.4, 0.1), (0.6, 0.9)]
    g = gini_grouped(groups)
    assert 0.0 < g < 1.0


def test_gini_grouped_sorted_by_per_capita():
    # Same groups in different order should yield same result
    groups_a = [(0.4, 0.1), (0.6, 0.9)]
    groups_b = [(0.6, 0.9), (0.4, 0.1)]
    assert gini_grouped(groups_a) == pytest.approx(gini_grouped(groups_b), abs=1e-12)


def test_gini_grouped_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        gini_grouped([])


def test_gini_grouped_negative_pop_raises():
    with pytest.raises(ValueError, match="positive"):
        gini_grouped([(-0.1, 0.5), (1.1, 0.5)])


def test_gini_grouped_negative_inc_raises():
    with pytest.raises(ValueError, match="non-negative"):
        gini_grouped([(0.5, -0.1), (0.5, 1.1)])


def test_gini_grouped_pop_sum_not_one_raises():
    with pytest.raises(ValueError, match="sum to 1"):
        gini_grouped([(0.3, 0.5), (0.3, 0.5)])


def test_gini_grouped_inc_sum_not_one_raises():
    with pytest.raises(ValueError, match="sum to 1"):
        gini_grouped([(0.5, 0.3), (0.5, 0.3)])


# ---------------------------------------------------------------------------
# _weighted_mean
# ---------------------------------------------------------------------------


def test_weighted_mean_equal_weights():
    assert _weighted_mean([1.0, 2.0, 3.0], [1.0, 1.0, 1.0]) == pytest.approx(2.0)


def test_weighted_mean_unequal_weights():
    # Mean of [1, 3] with weights [3, 1] = (3+3)/4 = 1.5
    assert _weighted_mean([1.0, 3.0], [3.0, 1.0]) == pytest.approx(1.5)


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------


def test_validate_no_mode():
    args = argparse.Namespace(
        data=None,
        groups=None,
        weights=None,
        lorenz=False,
        correct=False,
        precision=6,
        format="table",
    )
    assert "required" in validate(args)


def test_validate_negative_precision():
    args = argparse.Namespace(
        data=[[1.0, 2.0]],
        groups=None,
        weights=None,
        lorenz=False,
        correct=False,
        precision=-1,
        format="table",
    )
    assert "--precision" in validate(args)


def test_validate_data_and_groups_exclusive():
    args = argparse.Namespace(
        data=[[1.0, 2.0]],
        groups=[0.5, 0.5, 0.5, 0.5],
        weights=None,
        lorenz=False,
        correct=False,
        precision=6,
        format="table",
    )
    assert "mutually exclusive" in validate(args)


def test_validate_groups_odd_count():
    args = argparse.Namespace(
        data=None,
        groups=[0.5, 0.5, 0.5],
        weights=None,
        lorenz=False,
        correct=False,
        precision=6,
        format="table",
    )
    assert "--groups" in validate(args)


def test_validate_groups_single_value():
    args = argparse.Namespace(
        data=None,
        groups=[0.5],
        weights=None,
        lorenz=False,
        correct=False,
        precision=6,
        format="table",
    )
    assert "--groups" in validate(args)


def test_validate_weights_multi_dataset():
    args = argparse.Namespace(
        data=[[1.0, 2.0], [3.0, 4.0]],
        groups=None,
        weights=[1.0, 1.0],
        lorenz=False,
        correct=False,
        precision=6,
        format="table",
    )
    assert "--weights" in validate(args)


def test_validate_weights_wrong_length():
    args = argparse.Namespace(
        data=[[1.0, 2.0, 3.0]],
        groups=None,
        weights=[1.0, 1.0],
        lorenz=False,
        correct=False,
        precision=6,
        format="table",
    )
    assert "--weights" in validate(args)


def test_validate_weights_nonpositive():
    args = argparse.Namespace(
        data=[[1.0, 2.0]],
        groups=None,
        weights=[0.0, 1.0],
        lorenz=False,
        correct=False,
        precision=6,
        format="table",
    )
    assert "--weights" in validate(args)


def test_validate_lorenz_multi_dataset():
    args = argparse.Namespace(
        data=[[1.0, 2.0], [3.0, 4.0]],
        groups=None,
        weights=None,
        lorenz=True,
        correct=False,
        precision=6,
        format="table",
    )
    assert "--lorenz" in validate(args)


def test_validate_valid_single_data():
    args = argparse.Namespace(
        data=[[1.0, 2.0, 3.0]],
        groups=None,
        weights=None,
        lorenz=False,
        correct=False,
        precision=6,
        format="table",
    )
    assert validate(args) is None


def test_validate_valid_groups():
    args = argparse.Namespace(
        data=None,
        groups=[0.5, 0.3, 0.5, 0.7],
        weights=None,
        lorenz=False,
        correct=False,
        precision=6,
        format="table",
    )
    assert validate(args) is None


def test_validate_valid_multi_data():
    args = argparse.Namespace(
        data=[[1.0, 2.0], [3.0, 4.0]],
        groups=None,
        weights=None,
        lorenz=False,
        correct=False,
        precision=6,
        format="table",
    )
    assert validate(args) is None


def test_validate_valid_data_with_weights():
    args = argparse.Namespace(
        data=[[1.0, 2.0, 3.0]],
        groups=None,
        weights=[1.0, 2.0, 1.0],
        lorenz=False,
        correct=True,
        precision=6,
        format="table",
    )
    assert validate(args) is None


def test_validate_valid_lorenz_single():
    args = argparse.Namespace(
        data=[[1.0, 2.0, 3.0]],
        groups=None,
        weights=None,
        lorenz=True,
        correct=False,
        precision=6,
        format="table",
    )
    assert validate(args) is None


# ---------------------------------------------------------------------------
# format_single
# ---------------------------------------------------------------------------


def test_format_single_no_corrected():
    out = format_single(5, 3.0, 0.2667, 0.5333, None, 4)
    assert "0.2667" in out
    assert "Corrected" not in out
    assert "5" in out


def test_format_single_with_corrected():
    out = format_single(5, 3.0, 0.2667, 0.5333, 0.3333, 4)
    assert "0.3333" in out
    assert "Corrected" in out


# ---------------------------------------------------------------------------
# format_lorenz
# ---------------------------------------------------------------------------


def test_format_lorenz_contains_header():
    points = [(0.0, 0.0), (0.5, 0.3), (1.0, 1.0)]
    out = format_lorenz(points, 0.2667, 4)
    assert "Population" in out
    assert "Income" in out
    assert "0.2667" in out


def test_format_lorenz_contains_rows():
    points = [(0.0, 0.0), (1.0, 1.0)]
    out = format_lorenz(points, 0.0, 4)
    assert "0.0000" in out
    assert "1.0000" in out


# ---------------------------------------------------------------------------
# format_grouped
# ---------------------------------------------------------------------------


def test_format_grouped_contains_gini():
    out = format_grouped(0.3333, 3, 4)
    assert "0.3333" in out
    assert "3" in out


def test_format_grouped_contains_rmad():
    out = format_grouped(0.25, 2, 4)
    rmad = 2 * 0.25
    assert f"{rmad:.4f}" in out


# ---------------------------------------------------------------------------
# format_comparison
# ---------------------------------------------------------------------------


def test_format_comparison_no_corrected():
    results = [
        {"index": 1, "n": 5, "mean": 3.0, "gini": 0.27, "corrected": 0.33, "rank": 1},
        {"index": 2, "n": 4, "mean": 2.5, "gini": 0.35, "corrected": 0.47, "rank": 2},
    ]
    out = format_comparison(results, corrected=False, precision=4)
    assert "Gini" in out
    assert "Corrected" not in out
    assert "0.2700" in out


def test_format_comparison_with_corrected():
    results = [
        {"index": 1, "n": 5, "mean": 3.0, "gini": 0.27, "corrected": 0.33, "rank": 1},
    ]
    out = format_comparison(results, corrected=True, precision=4)
    assert "Corrected" in out
    assert "0.3300" in out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def test_main_no_args_returns_2(capsys):
    assert main([]) == 2


def test_main_negative_precision_returns_2(capsys):
    assert main(["--data", "1", "2", "3", "--precision", "-1"]) == 2


def test_main_single_data_table(capsys):
    rc = main(["--data", "1", "2", "3", "4", "5", "--precision", "4"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "0.2667" in out


def test_main_single_data_json(capsys):
    rc = main(["--data", "1", "2", "3", "4", "5", "--format", "json"])
    out = capsys.readouterr().out
    assert rc == 0
    data = json.loads(out)
    assert data["mode"] == "single"
    assert abs(data["gini"] - 4 / 15) < 1e-4


def test_main_single_data_json_with_correct(capsys):
    rc = main(["--data", "1", "2", "3", "4", "5", "--format", "json", "--correct"])
    out = capsys.readouterr().out
    assert rc == 0
    data = json.loads(out)
    assert "corrected_gini" in data


def test_main_single_data_lorenz_table(capsys):
    rc = main(["--data", "1", "2", "3", "4", "5", "--lorenz", "--precision", "4"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Population" in out


def test_main_single_data_lorenz_json(capsys):
    rc = main(["--data", "1", "2", "3", "--lorenz", "--format", "json"])
    out = capsys.readouterr().out
    assert rc == 0
    data = json.loads(out)
    assert data["mode"] == "lorenz"
    assert "lorenz_curve" in data


def test_main_single_data_lorenz_json_with_correct(capsys):
    rc = main(["--data", "1", "2", "3", "--lorenz", "--format", "json", "--correct"])
    out = capsys.readouterr().out
    assert rc == 0
    data = json.loads(out)
    assert "corrected_gini" in data


def test_main_single_data_with_weights(capsys):
    rc = main(["--data", "1", "2", "3", "--weights", "2", "1", "1", "--precision", "4"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Gini" in out


def test_main_single_data_with_correct(capsys):
    rc = main(["--data", "1", "2", "3", "4", "5", "--correct", "--precision", "4"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Corrected" in out


def test_main_groups_table(capsys):
    rc = main(
        ["--groups", "0.2", "0.05", "0.3", "0.15", "0.5", "0.80", "--precision", "4"]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "Gini" in out


def test_main_groups_json(capsys):
    rc = main(["--groups", "0.4", "0.1", "0.6", "0.9", "--format", "json"])
    out = capsys.readouterr().out
    assert rc == 0
    data = json.loads(out)
    assert data["mode"] == "grouped"
    assert "gini" in data


def test_main_groups_invalid_returns_2(capsys):
    # Two groups but pop sums to 0.5 (not 1.0) → gini_grouped raises
    assert main(["--groups", "0.2", "0.05", "0.3", "0.15"]) == 2


def test_main_multi_data_table(capsys):
    rc = main(["--data", "1", "2", "3", "--data", "4", "5", "6", "--precision", "4"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Dataset" in out


def test_main_multi_data_json(capsys):
    rc = main(["--data", "1", "2", "3", "--data", "4", "5", "6", "--format", "json"])
    out = capsys.readouterr().out
    assert rc == 0
    data = json.loads(out)
    assert data["mode"] == "compare"
    assert len(data["datasets"]) == 2


def test_main_multi_data_with_correct(capsys):
    rc = main(
        [
            "--data",
            "1",
            "2",
            "3",
            "--data",
            "4",
            "5",
            "6",
            "--correct",
            "--precision",
            "4",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "Corrected" in out


def test_main_single_data_invalid_returns_2(capsys):
    # Negative value passes validate but fails gini_coefficient
    assert main(["--data", "-1", "2", "3"]) == 2


def test_main_multi_data_invalid_returns_2(capsys):
    # Negative in first dataset → ValueError in multi-dataset loop
    assert main(["--data", "-1", "2", "3", "--data", "4", "5", "6"]) == 2


def test_main_lorenz_error_path(monkeypatch, capsys):
    """Cover the lorenz_curve ValueError branch in main."""

    def raise_value_error(*_args, **_kwargs):
        raise ValueError("forced lorenz error")

    monkeypatch.setattr(gini_module, "lorenz_curve", raise_value_error)
    assert main(["--data", "1", "2", "3", "--lorenz"]) == 2
    err = capsys.readouterr().err
    assert "forced lorenz error" in err
