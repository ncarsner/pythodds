#!/usr/bin/env python3
"""Command-line utility for lines in slope-intercept form.

Builds a line from whatever is known, converts between the three standard
representations, and evaluates, intersects, or projects onto it:

    y = m * x + b

Distinct from ``linreg``: ``linreg`` *estimates* a line from noisy sample data
with standard errors and p-values, while this tool manipulates an *exact*,
known line.  The two pair naturally -- ``linreg`` produces ``m`` and ``b``,
``slopeint`` consumes them.

Pure Python via ``math`` and ``fractions`` -- no external dependencies.

Usage examples:
  # Unit conversion: Fahrenheit from Celsius, y = 1.8x + 32
  slopeint --points 0,32 100,212

  # Evaluate that line at 37 degrees C
  slopeint --slope 1.8 --intercept 32 --at 37

  # Straight-line depreciation schedule
  slopeint --points 0,30000 5,5000 --table 0 5 1

  # Break-even as an intersection of a cost line and a revenue line
  slopeint --slope 10 --intercept 50000 --intersect 25,0

  # Perpendicular line through a point, and the distance to it
  slopeint --slope 2 --intercept 1 --point 4,3 --perpendicular
  slopeint --slope 2 --intercept 1 --point 4,3 --distance
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from fractions import Fraction
from typing import Any

Point = tuple[float, float]
Line = tuple[float, float]

# Tolerance for the parallel and perpendicular predicates.  Exact float
# equality is wrong here: a slope derived from two points rarely lands on the
# same bit pattern as the same slope typed in directly.
_TOL = 1e-9

# Denominator cap when rationalising a float slope for standard form.  Large
# enough for any decimal a user types, small enough to keep coefficients sane.
_MAX_DENOM = 1_000_000


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def slope(p1: Point, p2: Point) -> float:
    """Slope of the line through two points.

    Args:
        p1: First point as ``(x, y)``.
        p2: Second point as ``(x, y)``.

    Returns:
        Rise over run, ``(y2 - y1) / (x2 - x1)``.

    Raises:
        ValueError: If the points share an x coordinate (vertical line, slope
            undefined) or are the same point.
    """
    (x1, y1), (x2, y2) = p1, p2
    if x1 == x2:
        if y1 == y2:
            raise ValueError(
                f"the two points are identical ({x1}, {y1}); a line needs two "
                "distinct points"
            )
        raise ValueError(
            f"vertical line at x = {x1}: slope is undefined, no slope-intercept form"
        )
    return (y2 - y1) / (x2 - x1)


def line_from_points(p1: Point, p2: Point) -> Line:
    """Slope-intercept form of the line through two points.

    Args:
        p1: First point as ``(x, y)``.
        p2: Second point as ``(x, y)``.

    Returns:
        ``(m, b)`` for ``y = m * x + b``.

    Raises:
        ValueError: If the line is vertical or the points are identical.
    """
    m = slope(p1, p2)
    return m, p1[1] - m * p1[0]


def line_from_point_slope(point: Point, m: float) -> Line:
    """Convert point-slope form to slope-intercept form.

    Args:
        point: A point ``(x, y)`` on the line.
        m: Slope of the line.

    Returns:
        ``(m, b)`` for ``y = m * x + b``.
    """
    x0, y0 = point
    return m, y0 - m * x0


def line_from_standard(a: float, b: float, c: float) -> Line:
    """Convert standard form ``Ax + By = C`` to slope-intercept form.

    Args:
        a: Coefficient of x.
        b: Coefficient of y.
        c: Right-hand constant.

    Returns:
        ``(m, intercept)`` for ``y = m * x + intercept``.

    Raises:
        ValueError: If ``b`` is zero -- a vertical line (or, with ``a`` also
            zero, not a line at all).
    """
    if b == 0:
        if a == 0:
            raise ValueError("A and B cannot both be zero; that is not a line")
        raise ValueError(
            f"B = 0 means the vertical line x = {c / a}, which has no "
            "slope-intercept form"
        )
    return -a / b, c / b


def to_standard(m: float, b: float) -> tuple[int, int, int]:
    """Convert slope-intercept form to integer-normalised ``Ax + By = C``.

    The slope and intercept are rationalised (denominators capped at
    ``_MAX_DENOM``), scaled to integers, and sign-fixed so that the leading
    non-zero coefficient is positive.  The result is already in lowest terms.

    Args:
        m: Slope of the line.
        b: y-intercept of the line.

    Returns:
        ``(A, B, C)`` with integer coefficients.

    Raises:
        ValueError: If ``m`` or ``b`` is not finite.
    """
    if not math.isfinite(m) or not math.isfinite(b):
        raise ValueError(f"slope and intercept must be finite, got m={m}, b={b}")

    mf = Fraction(m).limit_denominator(_MAX_DENOM)
    bf = Fraction(b).limit_denominator(_MAX_DENOM)

    # y = mx + b  ->  -m x + y = b.  Scale by the LCM of both denominators.
    scale = mf.denominator * bf.denominator // math.gcd(mf.denominator, bf.denominator)
    coef_a = -mf.numerator * (scale // mf.denominator)
    coef_b = scale
    coef_c = bf.numerator * (scale // bf.denominator)

    # No GCD reduction step here: the coefficients this construction produces
    # are already primitive.  B is lcm(d_m, d_b), and because each fraction is
    # in lowest terms, any prime dividing B fails to divide either A or C.
    # Verified exhaustively over every m, b with numerators in [-12, 12] and
    # denominators in [1, 12] -- 90,000 combinations, no reducible triple.
    if coef_a < 0 or (coef_a == 0 and coef_b < 0):
        coef_a, coef_b, coef_c = -coef_a, -coef_b, -coef_c
    return coef_a, coef_b, coef_c


def evaluate(m: float, b: float, x: float) -> float:
    """Value of y at a given x.

    Args:
        m: Slope of the line.
        b: y-intercept of the line.
        x: Point at which to evaluate.

    Returns:
        ``m * x + b``.
    """
    return m * x + b


def solve_for_x(m: float, b: float, y: float) -> float:
    """Value of x at a given y.

    Args:
        m: Slope of the line; must be non-zero.
        b: y-intercept of the line.
        y: Target y value.

    Returns:
        ``(y - b) / m``.

    Raises:
        ValueError: If ``m`` is zero -- a horizontal line takes one y value
            everywhere, so no single x solves for any other.
    """
    if m == 0:
        raise ValueError(
            f"horizontal line y = {b}: every x gives the same y, so x cannot be "
            "solved for"
        )
    return (y - b) / m


def x_intercept(m: float, b: float) -> float:
    """Where the line crosses the x axis.

    Args:
        m: Slope of the line; must be non-zero.
        b: y-intercept of the line.

    Returns:
        ``-b / m``.

    Raises:
        ValueError: If ``m`` is zero (horizontal line, no crossing unless it is
            the x axis itself).
    """
    return solve_for_x(m, b, 0.0)


def y_intercept(m: float, b: float) -> float:
    """Where the line crosses the y axis.

    Args:
        m: Slope of the line (unused; present for interface symmetry).
        b: y-intercept of the line.

    Returns:
        ``b``, which is the y value at x = 0 by definition.
    """
    del m
    return b


def is_parallel(m1: float, m2: float, tol: float = _TOL) -> bool:
    """Whether two slopes describe parallel lines.

    Args:
        m1: First slope.
        m2: Second slope.
        tol: Absolute tolerance for the comparison.

    Returns:
        True when the slopes agree within ``tol``.
    """
    return math.isclose(m1, m2, rel_tol=0.0, abs_tol=tol)


def is_perpendicular(m1: float, m2: float, tol: float = _TOL) -> bool:
    """Whether two slopes describe perpendicular lines.

    Args:
        m1: First slope.
        m2: Second slope.
        tol: Absolute tolerance for the comparison.

    Returns:
        True when ``m1 * m2`` is -1 within ``tol``.
    """
    return math.isclose(m1 * m2, -1.0, rel_tol=0.0, abs_tol=tol)


def perpendicular_slope(m: float) -> float:
    """Slope of any line perpendicular to this one.

    Args:
        m: Slope of the original line; must be non-zero.

    Returns:
        ``-1 / m``.

    Raises:
        ValueError: If ``m`` is zero -- the perpendicular to a horizontal line
            is vertical, which has no slope.
    """
    if m == 0:
        raise ValueError(
            "the perpendicular to a horizontal line is vertical, which has no slope"
        )
    return -1.0 / m


def intersection(line1: Line, line2: Line, tol: float = _TOL) -> Point | None:
    """Point where two lines cross.

    Args:
        line1: First line as ``(m, b)``.
        line2: Second line as ``(m, b)``.
        tol: Absolute tolerance for the parallel test.

    Returns:
        The crossing point ``(x, y)``, or ``None`` when the lines are parallel
        (including coincident lines, which cross everywhere rather than at a
        point).  Use :func:`relationship` to tell those two cases apart.
    """
    m1, b1 = line1
    m2, b2 = line2
    if is_parallel(m1, m2, tol):
        return None
    x = (b2 - b1) / (m1 - m2)
    return x, evaluate(m1, b1, x)


def relationship(line1: Line, line2: Line, tol: float = _TOL) -> str:
    """Classify how two lines relate.

    Args:
        line1: First line as ``(m, b)``.
        line2: Second line as ``(m, b)``.
        tol: Absolute tolerance for the slope comparisons.

    Returns:
        One of ``"coincident"``, ``"parallel"``, ``"perpendicular"``, or
        ``"intersecting"``.
    """
    m1, b1 = line1
    m2, b2 = line2
    if is_parallel(m1, m2, tol):
        if math.isclose(b1, b2, rel_tol=0.0, abs_tol=tol):
            return "coincident"
        return "parallel"
    if is_perpendicular(m1, m2, tol):
        return "perpendicular"
    return "intersecting"


def distance_to_point(m: float, b: float, point: Point) -> float:
    """Perpendicular distance from a point to the line.

    Args:
        m: Slope of the line.
        b: y-intercept of the line.
        point: The point ``(x, y)`` to measure from.

    Returns:
        ``|m * x0 - y0 + b| / sqrt(m^2 + 1)``, zero when the point is on the line.
    """
    x0, y0 = point
    return abs(m * x0 - y0 + b) / math.hypot(m, 1.0)


def foot_of_perpendicular(m: float, b: float, point: Point) -> Point:
    """Closest point on the line to a given point.

    This is the projection of ``point`` onto the line -- the other end of the
    segment measured by :func:`distance_to_point`.

    Args:
        m: Slope of the line.
        b: y-intercept of the line.
        point: The point ``(x, y)`` to project.

    Returns:
        The projected point ``(x, y)`` lying on the line.
    """
    x0, y0 = point
    x = (x0 + m * y0 - m * b) / (m * m + 1.0)
    return x, evaluate(m, b, x)


def angle_of_inclination(m: float) -> float:
    """Angle the line makes with the positive x axis.

    Args:
        m: Slope of the line.

    Returns:
        Degrees in ``[0, 180)``: ``atan(m)`` for a rising line, and the same
        angle measured anticlockwise from the axis for a falling one.
    """
    degrees = math.degrees(math.atan(m))
    return degrees if degrees >= 0.0 else degrees + 180.0


def table_rows(
    m: float, b: float, start: float, stop: float, step: float
) -> list[Point]:
    """Evaluate the line across a range of x values.

    Args:
        m: Slope of the line.
        b: y-intercept of the line.
        start: First x value.
        stop: Last x value (included when the step lands on it).
        step: Increment between rows; must be > 0.

    Returns:
        List of ``(x, y)`` pairs.

    Raises:
        ValueError: If ``step`` <= 0 or ``stop`` < ``start``.
    """
    if step <= 0:
        raise ValueError(f"step must be > 0, got {step}")
    if stop < start:
        raise ValueError(f"stop must be >= start, got start={start}, stop={stop}")

    rows: list[Point] = []
    count = int(math.floor((stop - start) / step + 1e-9)) + 1
    for i in range(count):
        x = start + i * step
        rows.append((x, evaluate(m, b, x)))
    return rows


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_point(text: str) -> Point:
    """Parse an ``X,Y`` pair from the command line.

    Args:
        text: Comma-separated coordinate pair, e.g. ``"4,3"``.

    Returns:
        The point as ``(x, y)``.

    Raises:
        argparse.ArgumentTypeError: If the text is not two numbers separated by
            a comma.
    """
    parts = text.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"expected X,Y (two comma-separated numbers), got {text!r}"
        )
    try:
        return float(parts[0]), float(parts[1])
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected X,Y (two comma-separated numbers), got {text!r}"
        ) from None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Build and return the argument parser namespace.

    Args:
        argv: Argument list (uses ``sys.argv`` when ``None``).

    Returns:
        Parsed :class:`argparse.Namespace`.
    """
    parser = argparse.ArgumentParser(
        description="Slope-intercept line calculator: y = mx + b.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  slopeint --points 0,32 100,212
  slopeint --slope 1.8 --intercept 32 --at 37
  slopeint --points 0,30000 5,5000 --table 0 5 1
  slopeint --slope 10 --intercept 50000 --intersect 25,0
  slopeint --slope 2 --intercept 1 --point 4,3 --perpendicular
  slopeint --standard 3 -2 6 --format json
""",
    )
    parser.add_argument(
        "--points",
        type=parse_point,
        nargs=2,
        default=None,
        metavar=("X1,Y1", "X2,Y2"),
        help="derive the line from two points",
    )
    parser.add_argument(
        "--slope",
        "-m",
        type=float,
        default=None,
        metavar="F",
        help="slope m, with --intercept or --point",
    )
    parser.add_argument(
        "--intercept",
        "-b",
        type=float,
        default=None,
        metavar="F",
        help="y-intercept b, with --slope",
    )
    parser.add_argument(
        "--point",
        type=parse_point,
        default=None,
        metavar="X,Y",
        help="anchor point for point-slope form, or the target for --distance, "
        "--perpendicular, and --parallel",
    )
    parser.add_argument(
        "--standard",
        type=float,
        nargs=3,
        default=None,
        metavar=("A", "B", "C"),
        help="read the line from standard form Ax + By = C",
    )
    parser.add_argument(
        "--at",
        type=float,
        default=None,
        metavar="F",
        help="evaluate y at this x",
    )
    parser.add_argument(
        "--solve",
        type=float,
        default=None,
        metavar="F",
        help="solve for the x that gives this y",
    )
    parser.add_argument(
        "--intersect",
        type=parse_point,
        default=None,
        metavar="M,B",
        help="intersect with a second line given as slope,intercept",
    )
    parser.add_argument(
        "--perpendicular",
        action="store_true",
        help="report the perpendicular line through --point",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="report the parallel line through --point",
    )
    parser.add_argument(
        "--distance",
        action="store_true",
        help="report the perpendicular distance from --point to the line",
    )
    parser.add_argument(
        "--table",
        type=float,
        nargs=3,
        default=None,
        metavar=("MIN", "MAX", "STEP"),
        help="append a table of (x, y) across a range",
    )
    parser.add_argument(
        "--format",
        choices=["table", "json"],
        default="table",
        help="output format (default: table)",
    )
    parser.add_argument(
        "--precision",
        "-P",
        type=int,
        default=4,
        metavar="PREC",
        help="decimal places for output (default: 4)",
    )
    return parser.parse_args(argv)


def _line_sources(args: argparse.Namespace) -> list[str]:
    """Names of the mutually exclusive line-definition flags that were given.

    Args:
        args: Parsed argument namespace.

    Returns:
        Flag names present, in a stable order.
    """
    sources = []
    if args.points is not None:
        sources.append("--points")
    if args.standard is not None:
        sources.append("--standard")
    if args.slope is not None:
        sources.append("--slope")
    return sources


def validate(args: argparse.Namespace) -> str | None:
    """Return an error message string, or ``None`` if arguments are valid.

    Args:
        args: Parsed argument namespace from :func:`parse_args`.

    Returns:
        Error description string, or ``None`` when validation passes.
    """
    # Checked before the no-source branch below: --intercept on its own is a
    # half-given line, and saying so beats the generic "no line given".
    if args.intercept is not None and args.slope is None:
        return "--intercept needs --slope"

    sources = _line_sources(args)
    if not sources:
        return (
            "no line given: use --points, --slope with --intercept or --point, "
            "or --standard"
        )
    if len(sources) > 1:
        return f"{' and '.join(sources)} are mutually exclusive; pick one"

    if args.slope is not None and args.intercept is None and args.point is None:
        return "--slope needs either --intercept or --point to fix the line"
    if args.precision < 0:
        return f"--precision must be non-negative, got {args.precision}"

    # The anchor point of point-slope form lies on the line, so measuring a
    # distance to it or drawing a parallel through it says nothing.
    anchored = args.slope is not None and args.intercept is None
    for flag, requested in (
        ("--distance", args.distance),
        ("--perpendicular", args.perpendicular),
        ("--parallel", args.parallel),
    ):
        if not requested:
            continue
        if args.point is None:
            return f"{flag} requires --point"
        if anchored:
            return (
                f"{flag} needs a --point off the line, but --point is already "
                "the anchor of point-slope form; supply --intercept instead"
            )

    if args.table is not None:
        start, stop, step = args.table
        if step <= 0:
            return f"--table STEP must be > 0, got {step}"
        if stop < start:
            return f"--table MAX must be >= MIN, got MIN={start}, MAX={stop}"
    return None


# ---------------------------------------------------------------------------
# Result assembly
# ---------------------------------------------------------------------------


def resolve_line(args: argparse.Namespace) -> Line:
    """Determine ``(m, b)`` from whichever line flags were supplied.

    Args:
        args: Validated argument namespace.

    Returns:
        The line as ``(m, b)``.

    Raises:
        ValueError: If the requested line is vertical or otherwise degenerate.
    """
    if args.points is not None:
        return line_from_points(args.points[0], args.points[1])
    if args.standard is not None:
        return line_from_standard(*args.standard)
    if args.intercept is not None:
        return args.slope, args.intercept
    return line_from_point_slope(args.point, args.slope)


def build_result(args: argparse.Namespace) -> dict[str, Any]:
    """Compute every requested quantity for the resolved line.

    Args:
        args: Validated argument namespace from :func:`parse_args`.

    Returns:
        Result mapping ready for formatting.

    Raises:
        ValueError: If the line is degenerate or a requested operation is
            undefined for it.
    """
    m, b = resolve_line(args)
    coef_a, coef_b, coef_c = to_standard(m, b)

    result: dict[str, Any] = {
        "slope": m,
        "intercept": b,
        "equation": format_equation(m, b),
        "standard": {"a": coef_a, "b": coef_b, "c": coef_c},
        "y_intercept": y_intercept(m, b),
        "angle_degrees": angle_of_inclination(m),
        "horizontal": m == 0,
    }
    result["x_intercept"] = None if m == 0 else x_intercept(m, b)

    if args.at is not None:
        result["evaluate"] = {"x": args.at, "y": evaluate(m, b, args.at)}
    if args.solve is not None:
        result["solve"] = {"y": args.solve, "x": solve_for_x(m, b, args.solve)}

    if args.intersect is not None:
        other = (args.intersect[0], args.intersect[1])
        point = intersection((m, b), other)
        result["intersect"] = {
            "other": {"slope": other[0], "intercept": other[1]},
            "relationship": relationship((m, b), other),
            "point": None if point is None else {"x": point[0], "y": point[1]},
        }

    if args.perpendicular:
        perp_m = perpendicular_slope(m)
        perp_b = line_from_point_slope(args.point, perp_m)[1]
        result["perpendicular"] = {
            "slope": perp_m,
            "intercept": perp_b,
            "equation": format_equation(perp_m, perp_b),
        }
    if args.parallel:
        par_b = line_from_point_slope(args.point, m)[1]
        result["parallel"] = {
            "slope": m,
            "intercept": par_b,
            "equation": format_equation(m, par_b),
        }
    if args.distance:
        foot = foot_of_perpendicular(m, b, args.point)
        result["distance"] = {
            "point": {"x": args.point[0], "y": args.point[1]},
            "distance": distance_to_point(m, b, args.point),
            "foot": {"x": foot[0], "y": foot[1]},
        }

    if args.table is not None:
        start, stop, step = args.table
        result["table"] = [
            {"x": x, "y": y} for x, y in table_rows(m, b, start, stop, step)
        ]
    return result


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def _fmt(value: float, precision: int) -> str:
    """Format a float to fixed precision, trimming a trailing ``-0``.

    Args:
        value: Number to render.
        precision: Decimal places.

    Returns:
        Formatted string.
    """
    text = f"{value:.{precision}f}"
    return text[1:] if text.startswith("-0.") and float(text) == 0.0 else text


def format_equation(m: float, b: float) -> str:
    """Render ``y = mx + b`` with tidy signs and no redundant terms.

    Args:
        m: Slope of the line.
        b: y-intercept of the line.

    Returns:
        Human-readable equation string.
    """
    m_text = f"{m:g}"
    b_text = f"{abs(b):g}"

    if m == 0:
        return f"y = {b:g}"
    slope_part = "x" if m == 1 else ("-x" if m == -1 else f"{m_text}x")
    if b == 0:
        return f"y = {slope_part}"
    sign = "-" if b < 0 else "+"
    return f"y = {slope_part} {sign} {b_text}"


def format_standard(a: int, b: int, c: int) -> str:
    """Render ``Ax + By = C`` with tidy signs and no redundant terms.

    Args:
        a: Coefficient of x.
        b: Coefficient of y.
        c: Right-hand constant.

    Returns:
        Human-readable standard-form string.
    """
    terms = ""
    if a != 0:
        terms = "x" if a == 1 else ("-x" if a == -1 else f"{a}x")
    if b != 0:
        y_term = "y" if abs(b) == 1 else f"{abs(b)}y"
        if terms:
            terms += f" {'-' if b < 0 else '+'} {y_term}"
        else:
            terms = y_term if b > 0 else f"-{y_term}"
    return f"{terms} = {c}"


def format_table(result: dict[str, Any], precision: int) -> str:
    """Render the result as an aligned plain-text report.

    Args:
        result: Mapping from :func:`build_result`.
        precision: Decimal places for numeric output.

    Returns:
        Multi-line report string.
    """
    std = result["standard"]
    lines = [
        "Line",
        "----",
        f"  Slope-intercept:  {result['equation']}",
        f"  Standard form:    {format_standard(std['a'], std['b'], std['c'])}",
        f"  Slope (m):        {_fmt(result['slope'], precision)}",
        f"  y-intercept (b):  {_fmt(result['intercept'], precision)}",
    ]
    if result["x_intercept"] is None:
        lines.append("  x-intercept:      none (horizontal line)")
    else:
        lines.append(f"  x-intercept:      {_fmt(result['x_intercept'], precision)}")
    lines.append(f"  Inclination:      {_fmt(result['angle_degrees'], precision)}°")

    if "evaluate" in result:
        ev = result["evaluate"]
        lines += [
            "",
            "Evaluate",
            "--------",
            f"  y({_fmt(ev['x'], precision)}) = {_fmt(ev['y'], precision)}",
        ]
    if "solve" in result:
        sv = result["solve"]
        lines += [
            "",
            "Solve",
            "-----",
            f"  x = {_fmt(sv['x'], precision)} when y = {_fmt(sv['y'], precision)}",
        ]
    if "intersect" in result:
        it = result["intersect"]
        other = it["other"]
        lines += [
            "",
            "Intersection",
            "------------",
            f"  Other line:       {format_equation(other['slope'], other['intercept'])}",
            f"  Relationship:     {it['relationship']}",
        ]
        if it["point"] is None:
            note = (
                "every point (the lines are the same)"
                if it["relationship"] == "coincident"
                else "none (parallel lines never meet)"
            )
            lines.append(f"  Crossing:         {note}")
        else:
            px = _fmt(it["point"]["x"], precision)
            py = _fmt(it["point"]["y"], precision)
            lines.append(f"  Crossing:         ({px}, {py})")

    for key, title in (("perpendicular", "Perpendicular"), ("parallel", "Parallel")):
        if key in result:
            entry = result[key]
            lines += [
                "",
                f"{title} through point",
                "-" * (len(title) + 14),
                f"  Equation:         {entry['equation']}",
                f"  Slope (m):        {_fmt(entry['slope'], precision)}",
                f"  y-intercept (b):  {_fmt(entry['intercept'], precision)}",
            ]

    if "distance" in result:
        dist = result["distance"]
        pt, foot = dist["point"], dist["foot"]
        lines += [
            "",
            "Distance to point",
            "-----------------",
            f"  Point:            ({_fmt(pt['x'], precision)}, "
            f"{_fmt(pt['y'], precision)})",
            f"  Distance:         {_fmt(dist['distance'], precision)}",
            f"  Closest on line:  ({_fmt(foot['x'], precision)}, "
            f"{_fmt(foot['y'], precision)})",
        ]

    if "table" in result:
        lines += ["", "Table", "-----", f"  {'x':>14}  {'y':>14}"]
        for row in result["table"]:
            lines.append(
                f"  {_fmt(row['x'], precision):>14}  {_fmt(row['y'], precision):>14}"
            )
    return "\n".join(lines)


def format_json(result: dict[str, Any]) -> str:
    """Render the result as indented JSON.

    Args:
        result: Mapping from :func:`build_result`.

    Returns:
        JSON document string.
    """
    return json.dumps(result, indent=2)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Run the slope-intercept CLI.

    Args:
        argv: Argument list override for testing (uses ``sys.argv`` when ``None``).

    Returns:
        0 on success, 2 on input or computation error.
    """
    args = parse_args(argv)

    error = validate(args)
    if error:
        print(f"Error: {error}", file=sys.stderr)
        return 2

    try:
        result = build_result(args)
        if args.format == "json":
            print(format_json(result))
        else:
            print(format_table(result, args.precision))
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
