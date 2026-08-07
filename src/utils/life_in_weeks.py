#!/usr/bin/env python3
"""Command-line utility for rendering a human life as a grid of time units.

Inspired by Wait But Why's "Your Life in Weeks" (2014), this renders an ANSI
grid where each cell is one week, month, or year of a nominal 90-year lifespan,
color-coded to distinguish elapsed units from remaining ones.

The --reference view renders "The Life of a Typical American": the same grid
shaded by life phase, with milestone markers. Phase boundaries and milestone
ages are approximate US averages drawn from public sources:

  - Median age at first marriage: US Census Bureau (2024) — 30.2 (men),
    28.6 (women); the grid uses the 29.4 midpoint.
  - Mean age at first birth: CDC/NCHS (2023) — 27.5.
  - Average actual retirement age: Gallup (2023) — 62.
  - Life expectancy at birth: CDC/NCHS (2023) — 78.4.

Usage examples:
  life 1985-03-14
  life 1985-03-14 --mode months
  life 1985-03-14 --mode years --lifespan 100
  life --reference
  life 1985-03-14 --as-of 2030-01-01 --format json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date
from typing import IO, Any, Mapping, NamedTuple

# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

_UNITS_PER_YEAR = {"weeks": 52, "months": 12, "years": 1}
_COLS = {"weeks": 52, "months": 12, "years": 10}

_ELAPSED = "█"
_REMAINING = "░"
_MILESTONE = "◆"

_RESET = "\033[0m"
_COLORS = {
    "cyan": "\033[36m",
    "blue": "\033[34m",
    "yellow": "\033[33m",
    "green": "\033[32m",
    "magenta": "\033[35m",
    "dim": "\033[90m",
}


class LifeGrid(NamedTuple):
    mode: str
    birthdate: date
    as_of: date
    lifespan: int
    total_units: int
    elapsed_units: int
    remaining_units: int
    cols: int


class Phase(NamedTuple):
    label: str
    start_year: float
    end_year: float
    color: str
    glyph: str


_PHASES: tuple[Phase, ...] = (
    Phase("Childhood and K-12 school", 0.0, 18.0, "cyan", "▁"),
    Phase("Higher education / early adulthood", 18.0, 22.0, "blue", "▃"),
    Phase("Working years", 22.0, 62.0, "yellow", "▅"),
    Phase("Retirement", 62.0, 78.4, "green", "▇"),
    Phase("Beyond US life expectancy", 78.4, float("inf"), "dim", "░"),
)

_MILESTONES: tuple[tuple[float, str], ...] = (
    (27.5, "mean age at first birth (NCHS 2023)"),
    (29.4, "median age at first marriage (Census 2024)"),
    (62.0, "average actual retirement age (Gallup 2023)"),
)


def units_elapsed(birth: date, as_of: date, mode: str) -> int:
    """Return whole time units elapsed between two dates for a display mode.

    The month and year comparisons use a literal day-of-month test rather than
    end-of-month clamping, so a birthdate on the 29th-31st advances only once
    the same day number is reached in a later month.

    Args:
        birth: Birthdate.
        as_of: Date to measure to.
        mode: One of "weeks", "months", or "years".

    Returns:
        Count of completed units, never negative.

    Raises:
        ValueError: If mode is unknown or as_of precedes birth.
    """
    if mode not in _UNITS_PER_YEAR:
        raise ValueError(f"unknown mode: {mode!r}")
    if as_of < birth:
        raise ValueError(
            f"birthdate {birth.isoformat()} is after as-of date {as_of.isoformat()}"
        )

    if mode == "weeks":
        return (as_of - birth).days // 7

    if mode == "months":
        months = (as_of.year - birth.year) * 12 + (as_of.month - birth.month)
        if as_of.day < birth.day:
            months -= 1
        return months

    years = as_of.year - birth.year
    if (as_of.month, as_of.day) < (birth.month, birth.day):
        years -= 1
    return years


def compute_grid(
    birth: date, as_of: date, mode: str = "weeks", lifespan: int = 90
) -> LifeGrid:
    """Compute grid dimensions and elapsed/remaining counts for a life.

    Args:
        birth: Birthdate.
        as_of: Date the grid is current as of.
        mode: One of "weeks", "months", or "years".
        lifespan: Nominal lifespan in years spanned by the grid.

    Returns:
        LifeGrid with totals; elapsed is clamped to the grid capacity so a
        life longer than the nominal lifespan renders as fully elapsed.

    Raises:
        ValueError: If mode is unknown, lifespan is not positive, or as_of
            precedes birth.
    """
    if mode not in _UNITS_PER_YEAR:
        raise ValueError(f"unknown mode: {mode!r}")
    if lifespan <= 0:
        raise ValueError(f"lifespan must be positive, got {lifespan}")

    total = lifespan * _UNITS_PER_YEAR[mode]
    elapsed = min(units_elapsed(birth, as_of, mode), total)

    return LifeGrid(
        mode=mode,
        birthdate=birth,
        as_of=as_of,
        lifespan=lifespan,
        total_units=total,
        elapsed_units=elapsed,
        remaining_units=total - elapsed,
        cols=_COLS[mode],
    )


def grid_rows(total: int, elapsed: int, cols: int) -> list[str]:
    """Return glyph rows for a grid, without any ANSI escapes.

    Args:
        total: Total cells in the grid.
        elapsed: Leading cells to mark as elapsed.
        cols: Cells per row.

    Returns:
        List of glyph strings, one per row; the last row may be short.
    """
    return [
        "".join(
            _ELAPSED if i < elapsed else _REMAINING
            for i in range(start, min(start + cols, total))
        )
        for start in range(0, total, cols)
    ]


def _phase_for_age(age: float) -> Phase:
    """Return the life phase containing a given age in years."""
    for phase in _PHASES[:-1]:
        if phase.start_year <= age < phase.end_year:
            return phase
    return _PHASES[-1]


def reference_rows(mode: str = "weeks", lifespan: int = 90) -> list[str]:
    """Return glyph rows for the typical-American reference grid.

    Each cell carries its life-phase glyph; cells containing a milestone age
    are overlaid with the milestone glyph.

    Args:
        mode: One of "weeks", "months", or "years".
        lifespan: Nominal lifespan in years spanned by the grid.

    Returns:
        List of glyph strings, one per row.

    Raises:
        ValueError: If mode is unknown or lifespan is not positive.
    """
    if mode not in _UNITS_PER_YEAR:
        raise ValueError(f"unknown mode: {mode!r}")
    if lifespan <= 0:
        raise ValueError(f"lifespan must be positive, got {lifespan}")

    per_year = _UNITS_PER_YEAR[mode]
    total = lifespan * per_year
    cells = [_phase_for_age(i / per_year).glyph for i in range(total)]

    for age, _ in _MILESTONES:
        index = int(age * per_year)
        if index < total:
            cells[index] = _MILESTONE

    cols = _COLS[mode]
    return ["".join(cells[start : start + cols]) for start in range(0, total, cols)]


def milestone_units(mode: str = "weeks") -> list[tuple[str, int]]:
    """Return each milestone label paired with its unit index for a mode."""
    per_year = _UNITS_PER_YEAR[mode]
    return [(label, int(age * per_year)) for age, label in _MILESTONES]


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Life in weeks: an ANSI grid of a human life, elapsed vs remaining.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  life 1985-03-14
  life 1985-03-14 --mode months
  life 1985-03-14 --mode years --lifespan 100
  life --reference
  life 1985-03-14 --as-of 2030-01-01 --format json
""",
    )
    parser.add_argument(
        "birthdate",
        metavar="BIRTHDATE",
        nargs="?",
        help="birthdate in ISO format (YYYY-MM-DD); required unless --reference is used",
    )
    parser.add_argument(
        "--mode",
        "-m",
        choices=["weeks", "months", "years"],
        default="weeks",
        help="grid granularity: one cell per unit (default: weeks)",
    )
    parser.add_argument(
        "--lifespan",
        "-l",
        type=int,
        default=90,
        metavar="YEARS",
        help="nominal lifespan spanned by the grid, 1-150 (default: 90)",
    )
    parser.add_argument(
        "--as-of",
        metavar="DATE",
        help="ISO date to measure elapsed time to (default: today)",
    )
    parser.add_argument(
        "--reference",
        "-r",
        action="store_true",
        help="render the typical-American reference grid instead of a personal one",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="disable ANSI color; glyphs alone distinguish the cells",
    )
    parser.add_argument(
        "--format",
        "-f",
        choices=["table", "json"],
        default="table",
        help="output format: table (default) or json",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate(args: argparse.Namespace) -> str | None:
    """Return an error message for invalid argument combinations, else None."""
    if args.birthdate is None and not args.reference:
        return "BIRTHDATE is required unless --reference is used"
    if not 1 <= args.lifespan <= 150:
        return f"lifespan must be between 1 and 150, got {args.lifespan}"
    return None


def _parse_date(raw: str) -> date:
    """Parse an ISO date string, raising ValueError with a uniform message."""
    try:
        return date.fromisoformat(raw)
    except ValueError as exc:
        raise ValueError(f"invalid date: {raw!r} (expected YYYY-MM-DD)") from exc


def _use_color(
    args: argparse.Namespace,
    env: Mapping[str, str] | None = None,
    stream: IO[str] | None = None,
) -> bool:
    """Return whether ANSI color should be emitted.

    Color is suppressed by --no-color, by a non-empty NO_COLOR environment
    variable, or when the output stream is not a terminal.
    """
    if args.no_color:
        return False
    env = os.environ if env is None else env
    if env.get("NO_COLOR"):
        return False
    stream = sys.stdout if stream is None else stream
    return bool(getattr(stream, "isatty", lambda: False)())


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def _paint(text: str, color: str, use_color: bool) -> str:
    """Wrap text in an ANSI color when coloring is enabled."""
    if not use_color:
        return text
    return f"{_COLORS[color]}{text}{_RESET}"


_GLYPH_COLORS = {phase.glyph: phase.color for phase in _PHASES} | {
    _ELAPSED: "cyan",
    _REMAINING: "dim",
    _MILESTONE: "magenta",
}


def _paint_row(row: str, use_color: bool) -> str:
    """Color a glyph row by grouping runs of identical glyphs."""
    if not use_color:
        return row

    painted: list[str] = []
    run_start = 0
    for i in range(1, len(row) + 1):
        if i == len(row) or row[i] != row[run_start]:
            glyph = row[run_start]
            painted.append(_paint(row[run_start:i], _GLYPH_COLORS[glyph], True))
            run_start = i
    return "".join(painted)


def _row_label(mode: str, row_index: int) -> str:
    """Return the left-margin age label for a grid row."""
    age = row_index * 10 if mode == "years" else row_index
    return f"{age:>3} │ "


def _render_grid(rows: list[str], mode: str, use_color: bool) -> list[str]:
    """Return labelled, optionally colored grid lines."""
    return [
        _row_label(mode, i) + _paint_row(row, use_color) for i, row in enumerate(rows)
    ]


def _pct(part: int, whole: int) -> float:
    """Return part as a percentage of whole."""
    return part / whole * 100.0


def format_grid(grid: LifeGrid, use_color: bool = False) -> str:
    """Return the rendered personal life grid with legend and summary."""
    unit = grid.mode
    lines = [
        f"Your Life in {unit.capitalize()}",
        "=" * 40,
        f"Birthdate:  {grid.birthdate.isoformat()}",
        f"As of:      {grid.as_of.isoformat()}",
        f"Lifespan:   {grid.lifespan} years ({grid.total_units:,} {unit})",
        "",
    ]
    lines.extend(
        _render_grid(
            grid_rows(grid.total_units, grid.elapsed_units, grid.cols),
            grid.mode,
            use_color,
        )
    )
    lines.extend(
        [
            "",
            f"{_paint(_ELAPSED, 'cyan', use_color)} lived    {_paint(_REMAINING, 'dim', use_color)} remaining",
            "",
            f"Lived:      {grid.elapsed_units:,} {unit} ({_pct(grid.elapsed_units, grid.total_units):.1f}%)",
            f"Remaining:  {grid.remaining_units:,} {unit} ({_pct(grid.remaining_units, grid.total_units):.1f}%)",
        ]
    )
    return "\n".join(lines)


def format_reference(
    mode: str = "weeks", lifespan: int = 90, use_color: bool = False
) -> str:
    """Return the rendered typical-American reference grid with legend."""
    total = lifespan * _UNITS_PER_YEAR[mode]
    lines = [
        "The Life of a Typical American",
        "=" * 40,
        f"Lifespan:   {lifespan} years ({total:,} {mode})",
        "Phase boundaries are approximate US averages; see module docstring",
        "for sources (Census, NCHS, Gallup).",
        "",
    ]
    lines.extend(_render_grid(reference_rows(mode, lifespan), mode, use_color))
    lines.append("")

    for phase in _PHASES:
        end = "onward" if phase.end_year == float("inf") else f"{phase.end_year:g}"
        lines.append(
            f"{_paint(phase.glyph, phase.color, use_color)} {phase.label} ({phase.start_year:g}-{end})"
        )

    lines.append("")
    for label, index in milestone_units(mode):
        lines.append(
            f"{_paint(_MILESTONE, 'magenta', use_color)} {label} — {mode[:-1]} {index:,}"
        )

    return "\n".join(lines)


def format_json(grid: LifeGrid) -> str:
    """Return the personal grid summary as JSON."""
    return json.dumps(
        {
            "mode": grid.mode,
            "birthdate": grid.birthdate.isoformat(),
            "as_of": grid.as_of.isoformat(),
            "lifespan_years": grid.lifespan,
            "total_units": grid.total_units,
            "elapsed_units": grid.elapsed_units,
            "remaining_units": grid.remaining_units,
            "percent_lived": round(_pct(grid.elapsed_units, grid.total_units), 4),
            "cols": grid.cols,
        },
        indent=2,
    )


def format_reference_json(mode: str = "weeks", lifespan: int = 90) -> str:
    """Return the reference grid's phases and milestones as JSON."""
    per_year = _UNITS_PER_YEAR[mode]
    phases: list[dict[str, Any]] = []
    for phase in _PHASES:
        end_year = min(phase.end_year, float(lifespan))
        phases.append(
            {
                "label": phase.label,
                "start_year": phase.start_year,
                "end_year": end_year,
                "units": max(
                    0, int(end_year * per_year) - int(phase.start_year * per_year)
                ),
            }
        )

    return json.dumps(
        {
            "mode": mode,
            "lifespan_years": lifespan,
            "total_units": lifespan * per_year,
            "phases": phases,
            "milestones": [
                {"label": label, "unit_index": index}
                for label, index in milestone_units(mode)
            ],
        },
        indent=2,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    error = validate(args)
    if error is not None:
        print(f"Error: {error}", file=sys.stderr)
        return 2

    use_color = _use_color(args)

    if args.reference:
        if args.format == "json":
            print(format_reference_json(args.mode, args.lifespan))
        else:
            print(format_reference(args.mode, args.lifespan, use_color))
        return 0

    try:
        as_of = _parse_date(args.as_of) if args.as_of else date.today()
        birth = _parse_date(args.birthdate)
        grid = compute_grid(birth, as_of, args.mode, args.lifespan)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    if args.format == "json":
        print(format_json(grid))
    else:
        print(format_grid(grid, use_color))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
