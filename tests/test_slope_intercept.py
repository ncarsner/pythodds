"""Tests for the slope-intercept line utility."""

import argparse
import json
import math

import pytest

from src.utils.slope_intercept import (
    angle_of_inclination,
    build_result,
    distance_to_point,
    evaluate,
    foot_of_perpendicular,
    format_equation,
    format_json,
    format_standard,
    format_table,
    intersection,
    is_parallel,
    is_perpendicular,
    line_from_point_slope,
    line_from_points,
    line_from_standard,
    main,
    parse_point,
    perpendicular_slope,
    relationship,
    resolve_line,
    slope,
    solve_for_x,
    table_rows,
    to_standard,
    validate,
    x_intercept,
    y_intercept,
)


def _args(**overrides):
    """Build a namespace with a valid default line, overridden per test."""
    base = dict(
        points=None,
        slope=2.0,
        intercept=1.0,
        point=None,
        standard=None,
        at=None,
        solve=None,
        intersect=None,
        perpendicular=False,
        parallel=False,
        distance=False,
        table=None,
        format="table",
        precision=4,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


# ---------------------------------------------------------------------------
# slope / line construction
# ---------------------------------------------------------------------------


def test_slope_matches_rise_over_run():
    assert slope((0.0, 32.0), (100.0, 212.0)) == pytest.approx(1.8)


def test_slope_negative_when_falling():
    assert slope((0.0, 30000.0), (5.0, 5000.0)) == pytest.approx(-5000.0)


def test_slope_zero_for_horizontal():
    assert slope((1.0, 7.0), (9.0, 7.0)) == 0.0


def test_slope_vertical_raises():
    with pytest.raises(ValueError, match="vertical line"):
        slope((3.0, 1.0), (3.0, 5.0))


def test_slope_identical_points_raises():
    with pytest.raises(ValueError, match="identical"):
        slope((3.0, 1.0), (3.0, 1.0))


def test_line_from_points_celsius_to_fahrenheit():
    m, b = line_from_points((0.0, 32.0), (100.0, 212.0))
    assert m == pytest.approx(1.8)
    assert b == pytest.approx(32.0)


def test_line_from_points_intercept_from_second_point():
    m, b = line_from_points((2.0, 5.0), (4.0, 11.0))
    assert (m, b) == pytest.approx((3.0, -1.0))


def test_line_from_point_slope_recovers_intercept():
    assert line_from_point_slope((4.0, 3.0), -0.5) == pytest.approx((-0.5, 5.0))


def test_line_from_standard_matches_algebra():
    assert line_from_standard(3.0, -2.0, 6.0) == pytest.approx((1.5, -3.0))


def test_line_from_standard_zero_b_is_vertical():
    with pytest.raises(ValueError, match="vertical line"):
        line_from_standard(2.0, 0.0, 6.0)


def test_line_from_standard_all_zero_is_not_a_line():
    with pytest.raises(ValueError, match="not a line"):
        line_from_standard(0.0, 0.0, 6.0)


# ---------------------------------------------------------------------------
# to_standard
# ---------------------------------------------------------------------------


def test_to_standard_clears_fractional_slope():
    assert to_standard(1.8, 32.0) == (9, -5, -160)


def test_to_standard_reduces_by_gcd():
    assert to_standard(2.0, 4.0) == (2, -1, -4)


def test_to_standard_leading_coefficient_positive():
    a, _, _ = to_standard(-3.0, 2.0)
    assert a > 0


def test_to_standard_horizontal_line_has_zero_a():
    assert to_standard(0.0, 5.0) == (0, 1, 5)


def test_to_standard_round_trips_through_line_from_standard():
    a, b, c = to_standard(0.25, -1.5)
    assert line_from_standard(a, b, c) == pytest.approx((0.25, -1.5))


@pytest.mark.parametrize("bad", [math.inf, -math.inf, math.nan])
def test_to_standard_non_finite_raises(bad):
    with pytest.raises(ValueError, match="finite"):
        to_standard(bad, 1.0)


# ---------------------------------------------------------------------------
# evaluate / solve / intercepts
# ---------------------------------------------------------------------------


def test_evaluate_celsius_to_fahrenheit():
    assert evaluate(1.8, 32.0, 37.0) == pytest.approx(98.6)


def test_solve_for_x_inverts_evaluate():
    assert solve_for_x(1.8, 32.0, 98.6) == pytest.approx(37.0)


def test_solve_for_x_horizontal_raises():
    with pytest.raises(ValueError, match="horizontal line"):
        solve_for_x(0.0, 5.0, 2.0)


def test_x_intercept_matches_negative_b_over_m():
    assert x_intercept(2.0, 1.0) == pytest.approx(-0.5)


def test_x_intercept_horizontal_raises():
    with pytest.raises(ValueError, match="horizontal line"):
        x_intercept(0.0, 5.0)


def test_y_intercept_returns_b():
    assert y_intercept(2.0, 1.0) == 1.0


# ---------------------------------------------------------------------------
# parallel / perpendicular / intersection
# ---------------------------------------------------------------------------


def test_is_parallel_exact_match():
    assert is_parallel(2.0, 2.0)


def test_is_parallel_within_tolerance():
    assert is_parallel(2.0, 2.0 + 1e-12)


def test_is_parallel_outside_tolerance():
    assert not is_parallel(2.0, 2.5)


def test_is_perpendicular_true_for_negative_reciprocal():
    assert is_perpendicular(2.0, -0.5)


def test_is_perpendicular_survives_float_drift():
    assert is_perpendicular(3.0, -1.0 / 3.0)


def test_is_perpendicular_false_for_parallel():
    assert not is_perpendicular(2.0, 2.0)


def test_perpendicular_slope_is_negative_reciprocal():
    assert perpendicular_slope(2.0) == pytest.approx(-0.5)


def test_perpendicular_slope_horizontal_raises():
    with pytest.raises(ValueError, match="vertical"):
        perpendicular_slope(0.0)


def test_intersection_breakeven_point():
    point = intersection((10.0, 50000.0), (25.0, 0.0))
    assert point == pytest.approx((3333.3333333, 83333.3333333))


def test_intersection_parallel_returns_none():
    assert intersection((2.0, 1.0), (2.0, 9.0)) is None


def test_intersection_coincident_returns_none():
    assert intersection((2.0, 1.0), (2.0, 1.0)) is None


def test_intersection_point_lies_on_both_lines():
    x, y = intersection((3.0, -1.0), (-1.0, 7.0))
    assert evaluate(3.0, -1.0, x) == pytest.approx(y)
    assert evaluate(-1.0, 7.0, x) == pytest.approx(y)


@pytest.mark.parametrize(
    "line2,expected",
    [
        ((2.0, 1.0), "coincident"),
        ((2.0, 9.0), "parallel"),
        ((-0.5, 3.0), "perpendicular"),
        ((5.0, 3.0), "intersecting"),
    ],
)
def test_relationship_classifies(line2, expected):
    assert relationship((2.0, 1.0), line2) == expected


# ---------------------------------------------------------------------------
# distance / projection / angle
# ---------------------------------------------------------------------------


def test_distance_to_point_matches_formula():
    assert distance_to_point(2.0, 1.0, (4.0, 3.0)) == pytest.approx(
        6.0 / math.sqrt(5.0)
    )


def test_distance_zero_for_point_on_line():
    assert distance_to_point(2.0, 1.0, (3.0, 7.0)) == pytest.approx(0.0)


def test_foot_of_perpendicular_lies_on_line():
    fx, fy = foot_of_perpendicular(2.0, 1.0, (4.0, 3.0))
    assert evaluate(2.0, 1.0, fx) == pytest.approx(fy)


def test_foot_distance_equals_distance_to_point():
    point = (4.0, 3.0)
    foot = foot_of_perpendicular(2.0, 1.0, point)
    assert math.dist(point, foot) == pytest.approx(distance_to_point(2.0, 1.0, point))


def test_angle_of_inclination_45_degrees():
    assert angle_of_inclination(1.0) == pytest.approx(45.0)


def test_angle_of_inclination_horizontal_is_zero():
    assert angle_of_inclination(0.0) == 0.0


def test_angle_of_inclination_negative_slope_is_obtuse():
    assert angle_of_inclination(-1.0) == pytest.approx(135.0)


# ---------------------------------------------------------------------------
# table_rows
# ---------------------------------------------------------------------------


def test_table_rows_endpoints_and_count():
    rows = table_rows(-5000.0, 30000.0, 0.0, 5.0, 1.0)
    assert len(rows) == 6
    assert rows[0] == (0.0, 30000.0)
    assert rows[-1] == pytest.approx((5.0, 5000.0))


def test_table_rows_step_not_landing_on_stop():
    rows = table_rows(1.0, 0.0, 0.0, 1.0, 0.4)
    assert [x for x, _ in rows] == pytest.approx([0.0, 0.4, 0.8])


def test_table_rows_bad_step_raises():
    with pytest.raises(ValueError, match="step must be > 0"):
        table_rows(1.0, 0.0, 0.0, 5.0, 0.0)


def test_table_rows_reversed_range_raises():
    with pytest.raises(ValueError, match="stop must be >= start"):
        table_rows(1.0, 0.0, 5.0, 0.0, 1.0)


# ---------------------------------------------------------------------------
# parse_point
# ---------------------------------------------------------------------------


def test_parse_point_reads_pair():
    assert parse_point("4,3") == (4.0, 3.0)


def test_parse_point_accepts_negatives_and_decimals():
    assert parse_point("-1.5,2.25") == (-1.5, 2.25)


@pytest.mark.parametrize("bad", ["4", "4,3,2", "a,3", "4,b", ""])
def test_parse_point_rejects_malformed(bad):
    with pytest.raises(argparse.ArgumentTypeError, match="expected X,Y"):
        parse_point(bad)


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------


def test_validate_accepts_slope_intercept():
    assert validate(_args()) is None


def test_validate_requires_a_line():
    assert "no line given" in validate(_args(slope=None, intercept=None))


def test_validate_rejects_two_line_sources():
    msg = validate(_args(points=[(0.0, 0.0), (1.0, 1.0)]))
    assert "mutually exclusive" in msg


def test_validate_slope_alone_is_underdetermined():
    assert "needs either --intercept" in validate(_args(intercept=None))


def test_validate_intercept_without_slope():
    assert "--intercept needs --slope" in validate(_args(slope=None))


def test_validate_negative_precision():
    assert "--precision" in validate(_args(precision=-1))


@pytest.mark.parametrize("flag", ["distance", "perpendicular", "parallel"])
def test_validate_geometry_flags_need_point(flag):
    msg = validate(_args(**{flag: True}))
    assert "requires --point" in msg


@pytest.mark.parametrize("flag", ["distance", "perpendicular", "parallel"])
def test_validate_geometry_flags_reject_point_slope_anchor(flag):
    msg = validate(_args(intercept=None, point=(4.0, 3.0), **{flag: True}))
    assert "already" in msg


def test_validate_table_step_must_be_positive():
    assert "STEP must be > 0" in validate(_args(table=[0.0, 5.0, 0.0]))


def test_validate_table_reversed_range():
    assert "MAX must be >= MIN" in validate(_args(table=[5.0, 0.0, 1.0]))


def test_validate_accepts_full_table():
    assert validate(_args(table=[0.0, 5.0, 1.0])) is None


# ---------------------------------------------------------------------------
# resolve_line / build_result
# ---------------------------------------------------------------------------


def test_resolve_line_from_points():
    assert resolve_line(
        _args(slope=None, intercept=None, points=[(0.0, 32.0), (100.0, 212.0)])
    ) == pytest.approx((1.8, 32.0))


def test_resolve_line_from_standard():
    assert resolve_line(
        _args(slope=None, intercept=None, standard=[3.0, -2.0, 6.0])
    ) == pytest.approx((1.5, -3.0))


def test_resolve_line_from_point_slope():
    assert resolve_line(
        _args(slope=-0.5, intercept=None, point=(4.0, 3.0))
    ) == pytest.approx((-0.5, 5.0))


def test_build_result_core_fields():
    result = build_result(_args())
    assert result["slope"] == 2.0
    assert result["equation"] == "y = 2x + 1"
    assert result["standard"] == {"a": 2, "b": -1, "c": -1}
    assert result["x_intercept"] == pytest.approx(-0.5)
    assert result["horizontal"] is False


def test_build_result_horizontal_has_no_x_intercept():
    result = build_result(_args(slope=0.0, intercept=5.0))
    assert result["x_intercept"] is None
    assert result["horizontal"] is True


def test_build_result_evaluate_and_solve():
    result = build_result(_args(slope=1.8, intercept=32.0, at=37.0, solve=98.6))
    assert result["evaluate"]["y"] == pytest.approx(98.6)
    assert result["solve"]["x"] == pytest.approx(37.0)


def test_build_result_solve_on_horizontal_raises():
    with pytest.raises(ValueError, match="horizontal line"):
        build_result(_args(slope=0.0, intercept=5.0, solve=2.0))


def test_build_result_intersection():
    result = build_result(_args(slope=10.0, intercept=50000.0, intersect=(25.0, 0.0)))
    assert result["intersect"]["relationship"] == "intersecting"
    assert result["intersect"]["point"]["x"] == pytest.approx(3333.3333333)


def test_build_result_parallel_intersection_has_no_point():
    result = build_result(_args(intersect=(2.0, 9.0)))
    assert result["intersect"]["point"] is None
    assert result["intersect"]["relationship"] == "parallel"


def test_build_result_perpendicular_and_parallel_lines():
    result = build_result(_args(point=(4.0, 3.0), perpendicular=True, parallel=True))
    assert result["perpendicular"]["equation"] == "y = -0.5x + 5"
    assert result["parallel"]["slope"] == 2.0
    assert result["parallel"]["intercept"] == pytest.approx(-5.0)


def test_build_result_perpendicular_to_horizontal_raises():
    with pytest.raises(ValueError, match="vertical"):
        build_result(
            _args(slope=0.0, intercept=5.0, point=(4.0, 3.0), perpendicular=True)
        )


def test_build_result_distance_block():
    result = build_result(_args(point=(4.0, 3.0), distance=True))
    assert result["distance"]["distance"] == pytest.approx(6.0 / math.sqrt(5.0))
    assert result["distance"]["foot"]["x"] == pytest.approx(1.6)


def test_build_result_table():
    result = build_result(
        _args(slope=-5000.0, intercept=30000.0, table=[0.0, 5.0, 1.0])
    )
    assert len(result["table"]) == 6
    assert result["table"][-1]["y"] == pytest.approx(5000.0)


# ---------------------------------------------------------------------------
# formatting
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "m,b,expected",
    [
        (2.0, 1.0, "y = 2x + 1"),
        (2.0, -1.0, "y = 2x - 1"),
        (2.0, 0.0, "y = 2x"),
        (1.0, 3.0, "y = x + 3"),
        (-1.0, 3.0, "y = -x + 3"),
        (0.0, 5.0, "y = 5"),
        (-0.5, 5.0, "y = -0.5x + 5"),
    ],
)
def test_format_equation(m, b, expected):
    assert format_equation(m, b) == expected


@pytest.mark.parametrize(
    "coeffs,expected",
    [
        ((2, -1, -1), "2x - y = -1"),
        ((9, -5, -160), "9x - 5y = -160"),
        ((0, 1, 5), "y = 5"),
        ((1, 1, 4), "x + y = 4"),
        ((-1, 1, 4), "-x + y = 4"),
        ((1, -1, 0), "x - y = 0"),
        ((0, -1, 5), "-y = 5"),
    ],
)
def test_format_standard(coeffs, expected):
    assert format_standard(*coeffs) == expected


def test_format_table_includes_every_requested_block():
    args = _args(
        at=1.0,
        solve=5.0,
        intersect=(25.0, 0.0),
        point=(4.0, 3.0),
        perpendicular=True,
        parallel=True,
        distance=True,
        table=[0.0, 2.0, 1.0],
    )
    text = format_table(build_result(args), 4)
    for heading in (
        "Line",
        "Evaluate",
        "Solve",
        "Intersection",
        "Perpendicular through point",
        "Parallel through point",
        "Distance to point",
        "Table",
    ):
        assert heading in text


def test_format_table_horizontal_reports_no_x_intercept():
    text = format_table(build_result(_args(slope=0.0, intercept=5.0)), 4)
    assert "none (horizontal line)" in text


def test_format_table_parallel_intersection_note():
    text = format_table(build_result(_args(intersect=(2.0, 9.0))), 4)
    assert "parallel lines never meet" in text


def test_format_table_coincident_intersection_note():
    text = format_table(build_result(_args(intersect=(2.0, 1.0))), 4)
    assert "every point" in text


def test_format_table_avoids_negative_zero():
    text = format_table(build_result(_args(slope=1.0, intercept=-1e-9)), 2)
    assert "-0.00" not in text


def test_format_json_round_trips():
    payload = json.loads(format_json(build_result(_args())))
    assert payload["slope"] == 2.0
    assert payload["standard"]["a"] == 2


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def test_main_two_point_run(capsys):
    assert main(["--points", "0,32", "100,212"]) == 0
    assert "y = 1.8x + 32" in capsys.readouterr().out


def test_main_json_output(capsys):
    assert main(["--slope", "2", "--intercept", "1", "--format", "json"]) == 0
    assert json.loads(capsys.readouterr().out)["intercept"] == 1.0


def test_main_evaluate_matches_documented_example(capsys):
    assert main(["--slope", "1.8", "--intercept", "32", "--at", "37", "-P", "2"]) == 0
    assert "y(37.00) = 98.60" in capsys.readouterr().out


def test_main_validation_error_exits_two(capsys):
    assert main(["--slope", "2"]) == 2
    assert "Error:" in capsys.readouterr().err


def test_main_vertical_line_exits_two(capsys):
    assert main(["--points", "3,1", "3,5"]) == 2
    assert "vertical line" in capsys.readouterr().err


def test_main_horizontal_solve_exits_two(capsys):
    assert main(["--slope", "0", "--intercept", "5", "--solve", "2"]) == 2
    assert "horizontal line" in capsys.readouterr().err


def test_main_standard_form_entry(capsys):
    assert main(["--standard", "3", "-2", "6"]) == 0
    assert "y = 1.5x - 3" in capsys.readouterr().out
