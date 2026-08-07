"""Tests for the life-in-weeks grid visualizer utility."""

import argparse
import json
from datetime import date

import pytest

from src.utils.life_in_weeks import (
    _ELAPSED,
    _MILESTONE,
    _PHASES,
    _REMAINING,
    _paint,
    _paint_row,
    _parse_date,
    _phase_for_age,
    _row_label,
    _use_color,
    compute_grid,
    format_grid,
    format_json,
    format_reference,
    format_reference_json,
    grid_rows,
    main,
    milestone_units,
    reference_rows,
    units_elapsed,
    validate,
)

# ---------------------------------------------------------------------------
# units_elapsed
# ---------------------------------------------------------------------------


def test_units_elapsed_weeks_counts_whole_weeks():
    assert units_elapsed(date(2020, 1, 1), date(2020, 1, 22), "weeks") == 3
    assert units_elapsed(date(2020, 1, 1), date(2020, 1, 7), "weeks") == 0


def test_units_elapsed_months_counts_whole_months():
    assert units_elapsed(date(2020, 1, 15), date(2021, 1, 15), "months") == 12
    assert units_elapsed(date(2020, 1, 15), date(2020, 3, 14), "months") == 1


def test_units_elapsed_months_day_of_month_not_yet_reached():
    assert units_elapsed(date(2020, 1, 31), date(2020, 2, 28), "months") == 0


def test_units_elapsed_years_counts_whole_years():
    assert units_elapsed(date(1985, 3, 14), date(2026, 3, 14), "years") == 41
    assert units_elapsed(date(1985, 3, 14), date(2026, 3, 13), "years") == 40


def test_units_elapsed_leap_day_birthdate_in_non_leap_year():
    assert units_elapsed(date(2000, 2, 29), date(2001, 2, 28), "years") == 0
    assert units_elapsed(date(2000, 2, 29), date(2001, 3, 1), "years") == 1


def test_units_elapsed_same_day_is_zero():
    assert units_elapsed(date(2020, 6, 1), date(2020, 6, 1), "weeks") == 0


def test_units_elapsed_unknown_mode_raises():
    with pytest.raises(ValueError, match="unknown mode"):
        units_elapsed(date(2020, 1, 1), date(2020, 2, 1), "decades")


def test_units_elapsed_as_of_before_birth_raises():
    with pytest.raises(ValueError, match="is after as-of date"):
        units_elapsed(date(2030, 1, 1), date(2026, 8, 7), "weeks")


# ---------------------------------------------------------------------------
# compute_grid
# ---------------------------------------------------------------------------


def test_compute_grid_weeks_defaults():
    grid = compute_grid(date(1985, 3, 14), date(2026, 8, 7))
    assert grid.mode == "weeks"
    assert grid.total_units == 4680
    assert grid.cols == 52
    assert grid.elapsed_units + grid.remaining_units == grid.total_units


def test_compute_grid_months_and_years_totals():
    months = compute_grid(date(1985, 3, 14), date(2026, 8, 7), "months")
    years = compute_grid(date(1985, 3, 14), date(2026, 8, 7), "years")
    assert (months.total_units, months.cols) == (1080, 12)
    assert (years.total_units, years.cols) == (90, 10)
    assert years.elapsed_units == 41


def test_compute_grid_custom_lifespan():
    grid = compute_grid(date(1985, 3, 14), date(2026, 8, 7), "years", lifespan=100)
    assert grid.total_units == 100
    assert grid.remaining_units == 59


def test_compute_grid_clamps_elapsed_at_grid_capacity():
    # 52 weeks/year understates a real 90-year span by ~15 weeks; the clamp
    # keeps elapsed from exceeding the grid.
    grid = compute_grid(date(1900, 1, 1), date(2026, 8, 7))
    assert grid.elapsed_units == grid.total_units == 4680
    assert grid.remaining_units == 0


def test_compute_grid_life_exactly_at_lifespan_is_full():
    grid = compute_grid(date(1936, 8, 7), date(2026, 8, 7), "years")
    assert grid.elapsed_units == 90
    assert grid.remaining_units == 0


def test_compute_grid_unknown_mode_raises():
    with pytest.raises(ValueError, match="unknown mode"):
        compute_grid(date(1985, 3, 14), date(2026, 8, 7), "decades")


def test_compute_grid_non_positive_lifespan_raises():
    with pytest.raises(ValueError, match="lifespan must be positive"):
        compute_grid(date(1985, 3, 14), date(2026, 8, 7), "weeks", lifespan=0)


# ---------------------------------------------------------------------------
# grid_rows
# ---------------------------------------------------------------------------


def test_grid_rows_marks_leading_cells_as_elapsed():
    rows = grid_rows(total=8, elapsed=3, cols=4)
    assert rows == [_ELAPSED * 3 + _REMAINING, _REMAINING * 4]


def test_grid_rows_final_row_may_be_short():
    rows = grid_rows(total=7, elapsed=0, cols=4)
    assert [len(row) for row in rows] == [4, 3]


def test_grid_rows_all_elapsed():
    assert grid_rows(total=4, elapsed=4, cols=4) == [_ELAPSED * 4]


# ---------------------------------------------------------------------------
# Reference grid
# ---------------------------------------------------------------------------


def test_phase_for_age_boundaries_are_half_open():
    assert _phase_for_age(0.0).label.startswith("Childhood")
    assert _phase_for_age(17.99).label.startswith("Childhood")
    assert _phase_for_age(18.0).label.startswith("Higher education")
    assert _phase_for_age(22.0).label == "Working years"
    assert _phase_for_age(62.0).label == "Retirement"
    assert _phase_for_age(80.0).label.startswith("Beyond")


def test_reference_rows_shape_matches_mode():
    rows = reference_rows("weeks", 90)
    assert len(rows) == 90
    assert all(len(row) == 52 for row in rows)


def test_reference_rows_overlay_milestone_glyphs():
    rows = reference_rows("years", 90)
    assert "".join(rows).count(_MILESTONE) == 3


def test_reference_rows_skips_milestones_beyond_lifespan():
    rows = reference_rows("years", 10)
    assert _MILESTONE not in "".join(rows)


def test_reference_rows_uses_every_phase_glyph():
    grid = "".join(reference_rows("weeks", 90))
    for phase in _PHASES:
        assert phase.glyph in grid


def test_reference_rows_unknown_mode_raises():
    with pytest.raises(ValueError, match="unknown mode"):
        reference_rows("decades", 90)


def test_reference_rows_non_positive_lifespan_raises():
    with pytest.raises(ValueError, match="lifespan must be positive"):
        reference_rows("weeks", 0)


def test_milestone_units_scales_with_mode():
    assert milestone_units("years")[2] == (
        "average actual retirement age (Gallup 2023)",
        62,
    )
    assert milestone_units("weeks")[2][1] == 62 * 52


# ---------------------------------------------------------------------------
# Validation and color detection
# ---------------------------------------------------------------------------


def _ns(**overrides) -> argparse.Namespace:
    defaults = {
        "birthdate": "1985-03-14",
        "mode": "weeks",
        "lifespan": 90,
        "as_of": "2026-08-07",
        "reference": False,
        "no_color": False,
        "format": "table",
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_validate_accepts_normal_arguments():
    assert validate(_ns()) is None


def test_validate_requires_birthdate_without_reference():
    assert (
        validate(_ns(birthdate=None))
        == "BIRTHDATE is required unless --reference is used"
    )


def test_validate_allows_missing_birthdate_with_reference():
    assert validate(_ns(birthdate=None, reference=True)) is None


def test_validate_rejects_out_of_range_lifespan():
    assert validate(_ns(lifespan=0)) == "lifespan must be between 1 and 150, got 0"
    assert validate(_ns(lifespan=151)) == "lifespan must be between 1 and 150, got 151"


def test_parse_date_accepts_iso():
    assert _parse_date("2026-08-07") == date(2026, 8, 7)


def test_parse_date_rejects_garbage():
    with pytest.raises(ValueError, match="invalid date"):
        _parse_date("07/08/2026")


class _Stream:
    def __init__(self, tty: bool):
        self._tty = tty

    def isatty(self) -> bool:
        return self._tty


def test_use_color_true_for_tty_without_no_color():
    assert _use_color(_ns(), env={}, stream=_Stream(True)) is True


def test_use_color_false_for_non_tty():
    assert _use_color(_ns(), env={}, stream=_Stream(False)) is False


def test_use_color_false_when_flag_set():
    assert _use_color(_ns(no_color=True), env={}, stream=_Stream(True)) is False


def test_use_color_false_when_no_color_env_set():
    assert _use_color(_ns(), env={"NO_COLOR": "1"}, stream=_Stream(True)) is False


def test_use_color_false_when_stream_lacks_isatty():
    assert _use_color(_ns(), env={}, stream=object()) is False


def test_use_color_reads_process_environment_by_default(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    assert _use_color(_ns(), stream=_Stream(True)) is False


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def test_paint_is_a_noop_without_color():
    assert _paint(_ELAPSED, "cyan", False) == _ELAPSED
    assert "\033[36m" in _paint(_ELAPSED, "cyan", True)


def test_paint_row_groups_runs_of_identical_glyphs():
    row = _ELAPSED * 2 + _REMAINING * 2
    painted = _paint_row(row, True)
    assert painted.count("\033[0m") == 2
    assert _paint_row(row, False) == row


def test_row_label_uses_decades_for_years_mode():
    assert _row_label("years", 3) == " 30 │ "
    assert _row_label("weeks", 3) == "  3 │ "


def test_format_grid_contains_header_legend_and_summary():
    grid = compute_grid(date(1985, 3, 14), date(2026, 8, 7), "years")
    out = format_grid(grid)
    assert "Your Life in Years" in out
    assert "Birthdate:  1985-03-14" in out
    assert "Lived:      41 years (45.6%)" in out
    assert "Remaining:  49 years (54.4%)" in out
    assert "\033[" not in out


def test_format_grid_emits_ansi_when_colored():
    grid = compute_grid(date(1985, 3, 14), date(2026, 8, 7), "years")
    assert "\033[" in format_grid(grid, use_color=True)


def test_format_grid_row_count_matches_grid():
    grid = compute_grid(date(1985, 3, 14), date(2026, 8, 7))
    body = [line for line in format_grid(grid).splitlines() if "│" in line]
    assert len(body) == 90


def test_format_reference_lists_every_phase_and_milestone():
    out = format_reference("years", 90)
    assert "The Life of a Typical American" in out
    for phase in _PHASES:
        assert phase.label in out
    assert out.count("— year ") == 3
    assert "(78.4-onward)" in out
    assert "\033[" not in out


def test_format_reference_emits_ansi_when_colored():
    assert "\033[" in format_reference("years", 90, use_color=True)


def test_format_json_round_trips():
    grid = compute_grid(date(1985, 3, 14), date(2026, 8, 7))
    data = json.loads(format_json(grid))
    assert data["mode"] == "weeks"
    assert data["birthdate"] == "1985-03-14"
    assert data["elapsed_units"] + data["remaining_units"] == data["total_units"]
    assert data["percent_lived"] == pytest.approx(
        data["elapsed_units"] / data["total_units"] * 100, abs=5e-5
    )


def test_format_reference_json_reports_phase_unit_counts():
    data = json.loads(format_reference_json("years", 90))
    assert data["total_units"] == 90
    assert sum(phase["units"] for phase in data["phases"]) == 90
    assert len(data["milestones"]) == 3


def test_format_reference_json_zeroes_phases_beyond_lifespan():
    data = json.loads(format_reference_json("years", 10))
    beyond = [phase for phase in data["phases"] if phase["start_year"] >= 10]
    assert beyond and all(phase["units"] == 0 for phase in beyond)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def test_main_renders_personal_grid(capsys):
    rc = main(["1985-03-14", "--mode", "years", "--as-of", "2026-08-07"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Your Life in Years" in out
    assert "\033[" not in out


def test_main_json_output(capsys):
    rc = main(["1985-03-14", "--as-of", "2026-08-07", "--format", "json"])
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert data["total_units"] == 4680


def test_main_reference_needs_no_birthdate(capsys):
    rc = main(["--reference", "--mode", "years"])
    assert rc == 0
    assert "The Life of a Typical American" in capsys.readouterr().out


def test_main_reference_json_output(capsys):
    rc = main(["--reference", "--mode", "years", "--format", "json"])
    assert rc == 0
    assert json.loads(capsys.readouterr().out)["lifespan_years"] == 90


def test_main_defaults_as_of_to_today(capsys):
    rc = main(["1985-03-14", "--mode", "years", "--format", "json"])
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert data["as_of"] == date.today().isoformat()


def test_main_no_color_flag_suppresses_ansi(capsys):
    rc = main(["1985-03-14", "--mode", "years", "--as-of", "2026-08-07", "--no-color"])
    assert rc == 0
    assert "\033[" not in capsys.readouterr().out


def test_main_missing_birthdate_returns_2(capsys):
    rc = main([])
    assert rc == 2
    assert "Error" in capsys.readouterr().err


def test_main_invalid_lifespan_returns_2(capsys):
    rc = main(["1985-03-14", "--lifespan", "0"])
    assert rc == 2
    assert "lifespan must be between 1 and 150" in capsys.readouterr().err


def test_main_future_birthdate_returns_2(capsys):
    rc = main(["2030-01-01", "--as-of", "2026-08-07"])
    assert rc == 2
    assert "is after as-of date" in capsys.readouterr().err


def test_main_malformed_date_returns_2(capsys):
    rc = main(["not-a-date", "--as-of", "2026-08-07"])
    assert rc == 2
    assert "invalid date" in capsys.readouterr().err


def test_main_malformed_as_of_returns_2(capsys):
    rc = main(["1985-03-14", "--as-of", "yesterday"])
    assert rc == 2
    assert "invalid date" in capsys.readouterr().err
