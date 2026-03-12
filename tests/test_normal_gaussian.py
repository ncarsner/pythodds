"""Tests for Gaussian (normal) distribution functions."""

import pytest

from src.utils.normal_gaussian import (
    main,
    normal_pdf,
    normal_ppf,
)

# ---------------------------------------------------------------------------
# normal_pdf
# ---------------------------------------------------------------------------


def test_pdf_larger_sigma_shorter_peak():
    """Wider distribution has lower peak density."""
    assert normal_pdf(0.0, 0, 1) > normal_pdf(0.0, 0, 2)


# ---------------------------------------------------------------------------
# normal_ppf
# ---------------------------------------------------------------------------


def test_ppf_boundary_raises_value_error():
    """normal_ppf at p=0 or p=1 raises ValueError (erfinv domain guard)."""
    with pytest.raises(ValueError):
        normal_ppf(0.0)
    with pytest.raises(ValueError):
        normal_ppf(1.0)


# ---------------------------------------------------------------------------
# main — value mode
# ---------------------------------------------------------------------------


def test_main_value_mode_returns_zero(capsys):
    rc = main(["-x", "1.96", "-m", "0", "-s", "1"])
    assert rc == 0


# ---------------------------------------------------------------------------
# main — between mode
# ---------------------------------------------------------------------------


def test_main_between_output_contains_bounds(capsys):
    main(["--between", "-1.0", "1.0", "-m", "0", "-s", "1"])
    out = capsys.readouterr().out
    assert "Lower bound" in out
    assert "Upper bound" in out
    assert "68." in out  # ~68.27% probability


# ---------------------------------------------------------------------------
# main — quantile mode
# ---------------------------------------------------------------------------


def test_main_quantile_returns_zero(capsys):
    rc = main(["--quantile", "0.975", "-m", "0", "-s", "1"])
    assert rc == 0


def test_main_quantile_output_contains_x(capsys):
    main(["-q", "0.5", "-m", "0", "-s", "1"])
    out = capsys.readouterr().out
    # x at p=0.5 for standard normal is 0
    assert "0.000000" in out


# ---------------------------------------------------------------------------
# main — validation errors
# ---------------------------------------------------------------------------


def test_main_missing_mode_returns_2(capsys):
    rc = main(["-m", "0", "-s", "1"])
    assert rc == 2


def test_main_invalid_std_returns_2(capsys):
    rc = main(["-x", "1.0", "-m", "0", "-s", "-1"])
    assert rc == 2


def test_main_between_reversed_bounds_returns_2(capsys):
    rc = main(["--between", "2.0", "-1.0"])
    assert rc == 2


def test_main_quantile_zero_returns_2(capsys):
    rc = main(["-q", "0.0"])
    assert rc == 2
