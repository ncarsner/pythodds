"""Tests for Euler's number and related functions."""

import argparse
import json
import math

import pytest

import src.utils.euler as euler_module
from src.utils.euler import (
    _EULER_MASCHERONI_KNOWN,
    _approx_table,
    e_approx,
    euler_identity,
    euler_mascheroni,
    exp_series,
    format_approx,
    format_exp,
    format_identity,
    format_ln,
    format_mascheroni,
    main,
    natural_log,
    validate,
)

# ---------------------------------------------------------------------------
# e_approx
# ---------------------------------------------------------------------------


def test_e_approx_one():
    assert e_approx(1) == 2.0


def test_e_approx_large_n_close_to_e():
    assert abs(e_approx(1_000_000) - math.e) < 1e-5


def test_e_approx_monotone_increasing():
    prev = e_approx(1)
    for n in [10, 100, 1000, 10000]:
        curr = e_approx(n)
        assert curr > prev
        prev = curr


def test_e_approx_zero_raises():
    with pytest.raises(ValueError, match="at least 1"):
        e_approx(0)


def test_e_approx_negative_raises():
    with pytest.raises(ValueError):
        e_approx(-5)


# ---------------------------------------------------------------------------
# exp_series
# ---------------------------------------------------------------------------


def test_exp_series_at_zero():
    assert exp_series(0.0, 10) == pytest.approx(1.0, abs=1e-12)


def test_exp_series_at_one():
    assert exp_series(1.0, 50) == pytest.approx(math.e, abs=1e-10)


def test_exp_series_negative():
    assert exp_series(-1.0, 50) == pytest.approx(math.exp(-1.0), abs=1e-10)


def test_exp_series_higher_order_more_accurate():
    err_low = abs(exp_series(2.0, 5) - math.exp(2.0))
    err_high = abs(exp_series(2.0, 20) - math.exp(2.0))
    assert err_high < err_low


def test_exp_series_zero_order_raises():
    with pytest.raises(ValueError, match="at least 1"):
        exp_series(1.0, 0)


# ---------------------------------------------------------------------------
# natural_log
# ---------------------------------------------------------------------------


def test_natural_log_at_one():
    assert natural_log(1.0, 50) == pytest.approx(0.0, abs=1e-12)


def test_natural_log_at_e():
    assert natural_log(math.e, 100) == pytest.approx(1.0, abs=1e-6)


def test_natural_log_at_two():
    assert natural_log(2.0, 100) == pytest.approx(math.log(2.0), abs=1e-6)


def test_natural_log_higher_order_more_accurate():
    err_low = abs(natural_log(10.0, 5) - math.log(10.0))
    err_high = abs(natural_log(10.0, 50) - math.log(10.0))
    assert err_high < err_low


def test_natural_log_zero_raises():
    with pytest.raises(ValueError, match="positive"):
        natural_log(0.0, 10)


def test_natural_log_negative_raises():
    with pytest.raises(ValueError):
        natural_log(-1.0, 10)


def test_natural_log_zero_order_raises():
    with pytest.raises(ValueError, match="at least 1"):
        natural_log(2.0, 0)


# ---------------------------------------------------------------------------
# euler_identity
# ---------------------------------------------------------------------------


def test_euler_identity_real_is_minus_one():
    real, _, _ = euler_identity()
    assert real == pytest.approx(-1.0, abs=1e-12)


def test_euler_identity_imaginary_near_zero():
    _, imag, _ = euler_identity()
    assert abs(imag) < 1e-15


def test_euler_identity_sum_near_zero():
    _, _, total = euler_identity()
    assert abs(total) < 1e-15


# ---------------------------------------------------------------------------
# euler_mascheroni
# ---------------------------------------------------------------------------


def test_euler_mascheroni_accuracy():
    gamma = euler_mascheroni(100_000)
    assert abs(gamma - _EULER_MASCHERONI_KNOWN) < 0.001


def test_euler_mascheroni_default_close():
    gamma = euler_mascheroni(1000)
    assert abs(gamma - _EULER_MASCHERONI_KNOWN) < 0.01


# ---------------------------------------------------------------------------
# _approx_table
# ---------------------------------------------------------------------------


def test_approx_table_last_row_is_n():
    table = _approx_table(500)
    assert table[-1][0] == 500


def test_approx_table_includes_powers_of_ten():
    table = _approx_table(1000)
    ns = [row[0] for row in table]
    assert 1 in ns
    assert 10 in ns
    assert 100 in ns
    assert 1000 in ns


def test_approx_table_no_duplicate_n():
    table = _approx_table(1000)
    ns = [row[0] for row in table]
    assert len(ns) == len(set(ns))


def test_approx_table_n_equals_one():
    table = _approx_table(1)
    assert len(table) == 1
    assert table[0][0] == 1


def test_approx_table_errors_decrease():
    table = _approx_table(10_000)
    errors = [row[2] for row in table]
    for i in range(1, len(errors)):
        assert errors[i] < errors[i - 1]


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------


def test_validate_no_mode():
    args = argparse.Namespace(
        approx=None,
        exp=None,
        identity=False,
        ln=None,
        mascheroni=False,
        order=20,
        compare=False,
        precision=10,
        format="table",
    )
    result = validate(args)
    assert result is not None and "required" in result


def test_validate_negative_precision():
    args = argparse.Namespace(
        approx=None,
        exp=None,
        identity=True,
        ln=None,
        mascheroni=False,
        order=20,
        compare=False,
        precision=-1,
        format="table",
    )
    result = validate(args)
    assert result is not None and "--precision" in result


def test_validate_approx_zero():
    args = argparse.Namespace(
        approx=0,
        exp=None,
        identity=False,
        ln=None,
        mascheroni=False,
        order=20,
        compare=False,
        precision=10,
        format="table",
    )
    result = validate(args)
    assert result is not None and "--approx" in result


def test_validate_exp_bad_order():
    args = argparse.Namespace(
        approx=None,
        exp=2.0,
        identity=False,
        ln=None,
        mascheroni=False,
        order=0,
        compare=False,
        precision=10,
        format="table",
    )
    result = validate(args)
    assert result is not None and "--order" in result


def test_validate_ln_nonpositive():
    args = argparse.Namespace(
        approx=None,
        exp=None,
        identity=False,
        ln=0.0,
        mascheroni=False,
        order=20,
        compare=False,
        precision=10,
        format="table",
    )
    result = validate(args)
    assert result is not None and "--ln" in result


def test_validate_ln_negative():
    args = argparse.Namespace(
        approx=None,
        exp=None,
        identity=False,
        ln=-1.0,
        mascheroni=False,
        order=20,
        compare=False,
        precision=10,
        format="table",
    )
    result = validate(args)
    assert result is not None and "--ln" in result


def test_validate_ln_bad_order():
    args = argparse.Namespace(
        approx=None,
        exp=None,
        identity=False,
        ln=2.0,
        mascheroni=False,
        order=0,
        compare=False,
        precision=10,
        format="table",
    )
    result = validate(args)
    assert result is not None and "--order" in result


def test_validate_identity_valid():
    args = argparse.Namespace(
        approx=None,
        exp=None,
        identity=True,
        ln=None,
        mascheroni=False,
        order=20,
        compare=False,
        precision=10,
        format="table",
    )
    assert validate(args) is None


def test_validate_mascheroni_valid():
    args = argparse.Namespace(
        approx=None,
        exp=None,
        identity=False,
        ln=None,
        mascheroni=True,
        order=20,
        compare=False,
        precision=10,
        format="table",
    )
    assert validate(args) is None


def test_validate_approx_valid():
    args = argparse.Namespace(
        approx=1000,
        exp=None,
        identity=False,
        ln=None,
        mascheroni=False,
        order=20,
        compare=False,
        precision=10,
        format="table",
    )
    assert validate(args) is None


def test_validate_exp_valid():
    args = argparse.Namespace(
        approx=None,
        exp=2.0,
        identity=False,
        ln=None,
        mascheroni=False,
        order=20,
        compare=False,
        precision=10,
        format="table",
    )
    assert validate(args) is None


def test_validate_ln_valid():
    args = argparse.Namespace(
        approx=None,
        exp=None,
        identity=False,
        ln=2.0,
        mascheroni=False,
        order=20,
        compare=False,
        precision=10,
        format="table",
    )
    assert validate(args) is None


# ---------------------------------------------------------------------------
# format_approx
# ---------------------------------------------------------------------------


def test_format_approx_contains_header():
    table = _approx_table(100)
    out = format_approx(table, precision=6)
    assert "(1+1/n)^n" in out
    assert "Error vs e" in out


def test_format_approx_contains_math_e():
    table = _approx_table(100)
    out = format_approx(table, precision=6)
    assert "math.e" in out


# ---------------------------------------------------------------------------
# format_exp
# ---------------------------------------------------------------------------


def test_format_exp_no_compare():
    out = format_exp(1.0, 20, math.e, precision=6, compare=False)
    assert "e^x (series)" in out
    assert "math.exp" not in out


def test_format_exp_with_compare():
    result = exp_series(1.0, 50)
    out = format_exp(1.0, 50, result, precision=6, compare=True)
    assert "math.exp" in out
    assert "Absolute error" in out


# ---------------------------------------------------------------------------
# format_identity
# ---------------------------------------------------------------------------


def test_format_identity_contains_identity_label():
    real, imag, total = euler_identity()
    out = format_identity(real, imag, total, precision=6)
    assert "Euler" in out
    assert "e^(iπ)" in out


# ---------------------------------------------------------------------------
# format_ln
# ---------------------------------------------------------------------------


def test_format_ln_no_compare():
    out = format_ln(2.0, 20, math.log(2.0), precision=6, compare=False)
    assert "ln(x) (series)" in out
    assert "math.log" not in out


def test_format_ln_with_compare():
    out = format_ln(2.0, 50, natural_log(2.0, 50), precision=6, compare=True)
    assert "math.log" in out
    assert "Absolute error" in out


# ---------------------------------------------------------------------------
# format_mascheroni
# ---------------------------------------------------------------------------


def test_format_mascheroni_contains_gamma():
    out = format_mascheroni(0.5772, precision=6)
    assert "γ" in out or "Euler" in out


def test_format_mascheroni_contains_known_value():
    out = format_mascheroni(0.5772156649, precision=10)
    assert "0.5772156649" in out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def test_main_no_args_returns_2(capsys):
    assert main([]) == 2


def test_main_invalid_precision_returns_2(capsys):
    assert main(["--identity", "--precision", "-1"]) == 2


def test_main_approx_table(capsys):
    rc = main(["--approx", "1000", "--precision", "6"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "(1+1/n)^n" in out


def test_main_approx_json(capsys):
    rc = main(["--approx", "100", "--format", "json"])
    out = capsys.readouterr().out
    assert rc == 0
    data = json.loads(out)
    assert data["mode"] == "approx"
    assert "table" in data


def test_main_exp_table(capsys):
    rc = main(["--exp", "1", "--order", "20", "--precision", "6"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "e^x (series)" in out


def test_main_exp_table_compare(capsys):
    rc = main(["--exp", "1", "--compare", "--precision", "6"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "math.exp" in out


def test_main_exp_json(capsys):
    rc = main(["--exp", "2", "--format", "json"])
    out = capsys.readouterr().out
    assert rc == 0
    data = json.loads(out)
    assert data["mode"] == "exp"


def test_main_exp_json_compare(capsys):
    rc = main(["--exp", "2", "--format", "json", "--compare"])
    out = capsys.readouterr().out
    assert rc == 0
    data = json.loads(out)
    assert "math_exp" in data


def test_main_identity_table(capsys):
    rc = main(["--identity", "--precision", "6"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Euler" in out


def test_main_identity_json(capsys):
    rc = main(["--identity", "--format", "json"])
    out = capsys.readouterr().out
    assert rc == 0
    data = json.loads(out)
    assert data["mode"] == "identity"
    assert abs(data["real"] - (-1.0)) < 1e-12


def test_main_ln_table(capsys):
    rc = main(["--ln", "2", "--precision", "6"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "ln(x)" in out


def test_main_ln_table_compare(capsys):
    rc = main(["--ln", "2", "--compare", "--precision", "6"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "math.log" in out


def test_main_ln_json(capsys):
    rc = main(["--ln", "2", "--format", "json"])
    out = capsys.readouterr().out
    assert rc == 0
    data = json.loads(out)
    assert data["mode"] == "ln"


def test_main_ln_json_compare(capsys):
    rc = main(["--ln", "2", "--format", "json", "--compare"])
    out = capsys.readouterr().out
    assert rc == 0
    data = json.loads(out)
    assert "math_log" in data


def test_main_ln_invalid_x_returns_2(capsys):
    assert main(["--ln", "0"]) == 2


def test_main_mascheroni_table(monkeypatch, capsys):
    monkeypatch.setattr(
        euler_module, "euler_mascheroni", lambda: _EULER_MASCHERONI_KNOWN
    )
    rc = main(["--mascheroni", "--precision", "6"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Euler" in out


def test_main_mascheroni_json(monkeypatch, capsys):
    monkeypatch.setattr(
        euler_module, "euler_mascheroni", lambda: _EULER_MASCHERONI_KNOWN
    )
    rc = main(["--mascheroni", "--format", "json"])
    out = capsys.readouterr().out
    assert rc == 0
    data = json.loads(out)
    assert data["mode"] == "mascheroni"


def test_main_approx_bad_n_returns_2(capsys):
    assert main(["--approx", "0"]) == 2


def test_main_exp_bad_order_returns_2(capsys):
    assert main(["--exp", "1", "--order", "0"]) == 2
