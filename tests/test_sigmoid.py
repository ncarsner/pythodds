"""Tests for sigmoid function utility."""

import argparse
import math

import pytest

import src.utils.sigmoid as sigmoid_module
from src.utils.sigmoid import (
    _frange,
    format_inverse,
    format_single,
    format_table,
    inverse_logit,
    main,
    sigmoid,
    sigmoid_derivative,
    sparkline,
    validate,
)

# ---------------------------------------------------------------------------
# sigmoid
# ---------------------------------------------------------------------------


def test_sigmoid_at_zero():
    assert sigmoid(0.0) == 0.5


def test_sigmoid_positive_x():
    result = sigmoid(2.0)
    assert abs(result - 1.0 / (1.0 + math.exp(-2.0))) < 1e-12


def test_sigmoid_negative_x():
    result = sigmoid(-3.0)
    expected = math.exp(-3.0) / (1.0 + math.exp(-3.0))
    assert abs(result - expected) < 1e-12


def test_sigmoid_large_positive():
    assert sigmoid(1000.0) == pytest.approx(1.0, abs=1e-10)


def test_sigmoid_large_negative():
    assert sigmoid(-1000.0) == pytest.approx(0.0, abs=1e-10)


def test_sigmoid_output_in_zero_one():
    for x in [-1.0, 0.0, 1.0]:
        assert 0.0 < sigmoid(x) < 1.0


# ---------------------------------------------------------------------------
# sigmoid_derivative
# ---------------------------------------------------------------------------


def test_derivative_at_zero():
    assert sigmoid_derivative(0.0) == pytest.approx(0.25, abs=1e-12)


def test_derivative_is_s_times_one_minus_s():
    for x in [-2.0, 0.0, 1.5]:
        s = sigmoid(x)
        assert sigmoid_derivative(x) == pytest.approx(s * (1.0 - s), abs=1e-12)


def test_derivative_symmetry():
    assert sigmoid_derivative(3.0) == pytest.approx(sigmoid_derivative(-3.0), abs=1e-12)


# ---------------------------------------------------------------------------
# inverse_logit
# ---------------------------------------------------------------------------


def test_inverse_logit_half():
    assert inverse_logit(0.5) == pytest.approx(0.0, abs=1e-12)


def test_inverse_logit_roundtrip():
    for p in [0.1, 0.3, 0.75, 0.9]:
        assert sigmoid(inverse_logit(p)) == pytest.approx(p, abs=1e-12)


def test_inverse_logit_zero_raises():
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        inverse_logit(0.0)


def test_inverse_logit_one_raises():
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        inverse_logit(1.0)


def test_inverse_logit_negative_raises():
    with pytest.raises(ValueError):
        inverse_logit(-0.1)


# ---------------------------------------------------------------------------
# _frange
# ---------------------------------------------------------------------------


def test_frange_integers():
    assert _frange(0.0, 3.0, 1.0) == [0.0, 1.0, 2.0, 3.0]


def test_frange_includes_stop():
    result = _frange(-5.0, 5.0, 1.0)
    assert result[0] == -5.0
    assert result[-1] == pytest.approx(5.0, abs=1e-9)
    assert len(result) == 11


def test_frange_fractional_step():
    result = _frange(0.0, 1.0, 0.5)
    assert len(result) == 3
    assert result == pytest.approx([0.0, 0.5, 1.0])


# ---------------------------------------------------------------------------
# sparkline
# ---------------------------------------------------------------------------


def test_sparkline_empty():
    assert sparkline([]) == ""


def test_sparkline_zero():
    result = sparkline([0.0])
    assert result == "▁"


def test_sparkline_one():
    result = sparkline([1.0])
    assert result == "█"


def test_sparkline_length():
    result = sparkline([0.0, 0.25, 0.5, 0.75, 1.0])
    assert len(result) == 5


def test_sparkline_increasing():
    values = [0.0, 0.2, 0.5, 0.8, 1.0]
    result = sparkline(values)
    assert len(result) == 5
    assert all(c in "▁▂▃▄▅▆▇█" for c in result)


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------


def test_validate_negative_precision():
    args = argparse.Namespace(
        value=None,
        range=None,
        inverse=False,
        prob=None,
        derivative=False,
        plot=False,
        precision=-1,
    )
    assert validate(args) == "--precision must be non-negative"


def test_validate_no_mode():
    args = argparse.Namespace(
        value=None,
        range=None,
        inverse=False,
        prob=None,
        derivative=False,
        plot=False,
        precision=6,
    )
    assert validate(args) == "one of --value, --range, or --inverse is required"


def test_validate_inverse_without_prob():
    args = argparse.Namespace(
        value=None,
        range=None,
        inverse=True,
        prob=None,
        derivative=False,
        plot=False,
        precision=6,
    )
    assert validate(args) == "--prob is required with --inverse"


def test_validate_inverse_prob_out_of_range_high():
    args = argparse.Namespace(
        value=None,
        range=None,
        inverse=True,
        prob=1.0,
        derivative=False,
        plot=False,
        precision=6,
    )
    assert validate(args) == "--prob must be strictly between 0 and 1"


def test_validate_inverse_prob_out_of_range_low():
    args = argparse.Namespace(
        value=None,
        range=None,
        inverse=True,
        prob=0.0,
        derivative=False,
        plot=False,
        precision=6,
    )
    assert validate(args) == "--prob must be strictly between 0 and 1"


def test_validate_inverse_valid():
    args = argparse.Namespace(
        value=None,
        range=None,
        inverse=True,
        prob=0.75,
        derivative=False,
        plot=False,
        precision=6,
    )
    assert validate(args) is None


def test_validate_range_zero_step():
    args = argparse.Namespace(
        value=None,
        range=[-5.0, 5.0, 0.0],
        inverse=False,
        prob=None,
        derivative=False,
        plot=False,
        precision=6,
    )
    assert validate(args) == "STEP must be greater than 0"


def test_validate_range_negative_step():
    args = argparse.Namespace(
        value=None,
        range=[-5.0, 5.0, -1.0],
        inverse=False,
        prob=None,
        derivative=False,
        plot=False,
        precision=6,
    )
    assert validate(args) == "STEP must be greater than 0"


def test_validate_range_min_ge_max():
    args = argparse.Namespace(
        value=None,
        range=[5.0, -5.0, 1.0],
        inverse=False,
        prob=None,
        derivative=False,
        plot=False,
        precision=6,
    )
    assert validate(args) == "MIN must be less than MAX"


def test_validate_range_equal_min_max():
    args = argparse.Namespace(
        value=None,
        range=[3.0, 3.0, 1.0],
        inverse=False,
        prob=None,
        derivative=False,
        plot=False,
        precision=6,
    )
    assert validate(args) == "MIN must be less than MAX"


def test_validate_value_valid():
    args = argparse.Namespace(
        value=1.0,
        range=None,
        inverse=False,
        prob=None,
        derivative=False,
        plot=False,
        precision=6,
    )
    assert validate(args) is None


def test_validate_range_valid():
    args = argparse.Namespace(
        value=None,
        range=[-5.0, 5.0, 1.0],
        inverse=False,
        prob=None,
        derivative=False,
        plot=False,
        precision=6,
    )
    assert validate(args) is None


# ---------------------------------------------------------------------------
# format_single
# ---------------------------------------------------------------------------


def test_format_single_contains_value_and_sigmoid():
    out = format_single(0.0, 0.5, 0.25, 4, show_deriv=False)
    assert "0.0000" in out
    assert "0.5000" in out
    assert "σ'(x)" not in out


def test_format_single_with_derivative():
    out = format_single(0.0, 0.5, 0.25, 4, show_deriv=True)
    assert "0.2500" in out
    assert "σ'(x)" in out


# ---------------------------------------------------------------------------
# format_table
# ---------------------------------------------------------------------------


def test_format_table_no_deriv_no_plot():
    xs = [-1.0, 0.0, 1.0]
    sigs = [sigmoid(x) for x in xs]
    dsigs = [sigmoid_derivative(x) for x in xs]
    out = format_table(xs, sigs, dsigs, precision=4, show_deriv=False, show_plot=False)
    assert "σ(x)" in out
    assert "σ'(x)" not in out
    assert "▁" not in out


def test_format_table_with_deriv():
    xs = [0.0]
    sigs = [0.5]
    dsigs = [0.25]
    out = format_table(xs, sigs, dsigs, precision=4, show_deriv=True, show_plot=False)
    assert "σ'(x)" in out
    assert "0.2500" in out


def test_format_table_with_plot():
    xs = [-2.0, 0.0, 2.0]
    sigs = [sigmoid(x) for x in xs]
    dsigs = [sigmoid_derivative(x) for x in xs]
    out = format_table(xs, sigs, dsigs, precision=4, show_deriv=False, show_plot=True)
    assert any(c in out for c in "▁▂▃▄▅▆▇█")


# ---------------------------------------------------------------------------
# format_inverse
# ---------------------------------------------------------------------------


def test_format_inverse_contains_prob_and_logit():
    x = inverse_logit(0.75)
    out = format_inverse(0.75, x, precision=4)
    assert "0.7500" in out
    assert "logit" in out.lower() or "log" in out.lower()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def test_main_no_args_returns_2(capsys):
    assert main([]) == 2


def test_main_invalid_precision_returns_2(capsys):
    assert main(["--value", "0", "--precision", "-1"]) == 2


def test_main_single_value(capsys):
    rc = main(["-x", "0", "--precision", "4"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "0.5000" in out


def test_main_single_value_with_derivative(capsys):
    rc = main(["-x", "0", "--derivative", "--precision", "4"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "0.2500" in out


def test_main_single_value_with_plot(capsys):
    rc = main(["-x", "0", "--plot"])
    out = capsys.readouterr().out
    assert rc == 0
    assert any(c in out for c in "▁▂▃▄▅▆▇█")


def test_main_range(capsys):
    rc = main(["--range", "-2", "2", "1", "--precision", "4"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "σ(x)" in out


def test_main_range_with_derivative(capsys):
    rc = main(["--range", "-1", "1", "1", "--derivative", "--precision", "4"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "σ'(x)" in out


def test_main_range_with_plot(capsys):
    rc = main(["--range", "-3", "3", "1", "--plot"])
    out = capsys.readouterr().out
    assert rc == 0
    assert any(c in out for c in "▁▂▃▄▅▆▇█")


def test_main_inverse(capsys):
    rc = main(["--inverse", "--prob", "0.75", "--precision", "4"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "0.7500" in out


def test_main_inverse_missing_prob_returns_2(capsys):
    assert main(["--inverse"]) == 2


def test_main_inverse_invalid_prob_returns_2(capsys):
    assert main(["--inverse", "--prob", "1.5"]) == 2


def test_main_inverse_error_path(monkeypatch, capsys):
    """Cover the try/except branch in main for inverse_logit failure."""

    def raise_value_error(_p: float) -> float:
        raise ValueError("forced error")

    monkeypatch.setattr(sigmoid_module, "inverse_logit", raise_value_error)
    assert main(["--inverse", "--prob", "0.5"]) == 2
    err = capsys.readouterr().err
    assert "Error: forced error" in err


def test_main_range_invalid_step_returns_2(capsys):
    assert main(["--range", "-5", "5", "0"]) == 2


def test_main_range_invalid_bounds_returns_2(capsys):
    assert main(["--range", "5", "-5", "1"]) == 2
