"""Tests for Sunzi's Theorem (CRT) utility."""

import argparse
import json

import pytest

import src.utils.crt as crt_module
from src.utils.crt import (
    crt,
    extended_gcd,
    format_json,
    format_solution,
    main,
    mod_inverse,
    validate,
)

# ---------------------------------------------------------------------------
# extended_gcd
# ---------------------------------------------------------------------------


def test_extended_gcd_basic():
    g, s, t = extended_gcd(3, 5)
    assert g == 1
    assert 3 * s + 5 * t == 1


def test_extended_gcd_gcd_twelve_eight():
    g, s, t = extended_gcd(12, 8)
    assert g == 4
    assert 12 * s + 8 * t == 4


def test_extended_gcd_zero_b():
    g, s, t = extended_gcd(7, 0)
    assert g == 7
    assert s == 1
    assert t == 0


def test_extended_gcd_both_equal():
    g, s, t = extended_gcd(5, 5)
    assert g == 5
    assert 5 * s + 5 * t == 5


def test_extended_gcd_identity():
    for a, b in [(35, 15), (100, 37), (13, 5)]:
        g, s, t = extended_gcd(a, b)
        assert a * s + b * t == g


# ---------------------------------------------------------------------------
# mod_inverse
# ---------------------------------------------------------------------------


def test_mod_inverse_basic():
    assert mod_inverse(3, 7) == 5  # 3*5 = 15 ≡ 1 (mod 7)


def test_mod_inverse_result_in_range():
    x = mod_inverse(4, 9)
    assert 0 <= x < 9
    assert (4 * x) % 9 == 1


def test_mod_inverse_no_inverse_raises():
    with pytest.raises(ValueError, match="no inverse"):
        mod_inverse(4, 6)  # gcd(4,6) = 2 ≠ 1


def test_mod_inverse_modulus_too_small_raises():
    with pytest.raises(ValueError, match="at least 2"):
        mod_inverse(3, 1)


def test_mod_inverse_modulus_zero_raises():
    with pytest.raises(ValueError, match="at least 2"):
        mod_inverse(3, 0)


# ---------------------------------------------------------------------------
# crt
# ---------------------------------------------------------------------------


def test_crt_sunzi_classic():
    # x≡2(3), x≡3(5), x≡2(7) → 23 (mod 105)
    x, N = crt([2, 3, 2], [3, 5, 7])
    assert x == 23
    assert N == 105


def test_crt_single_congruence():
    x, N = crt([3], [7])
    assert x == 3
    assert N == 7


def test_crt_solution_satisfies_all():
    remainders = [2, 3, 2]
    moduli = [3, 5, 7]
    x, N = crt(remainders, moduli)
    for a, n in zip(remainders, moduli):
        assert x % n == a % n


def test_crt_negative_remainder_normalized():
    x, N = crt([-1, 0], [3, 5])
    assert x % 3 == 2  # -1 mod 3 = 2
    assert x % 5 == 0


def test_crt_large_remainders_normalized():
    x, N = crt([11, 13], [3, 5])
    assert x % 3 == 11 % 3
    assert x % 5 == 13 % 5


def test_crt_non_coprime_compatible():
    # x≡0(4), x≡3(6) → gcd(4,6)=2; (3-0)%2=1 ≠ 0 → inconsistent
    # x≡0(4), x≡2(6) → gcd(4,6)=2; (2-0)%2=0 ✓ → solution exists
    x, N = crt([0, 2], [4, 6])
    assert x % 4 == 0
    assert x % 6 == 2


def test_crt_non_coprime_inconsistent_raises():
    with pytest.raises(ValueError, match="inconsistent"):
        crt([0, 3], [4, 6])


def test_crt_length_mismatch_raises():
    with pytest.raises(ValueError, match="same length"):
        crt([1, 2], [3])


def test_crt_empty_raises():
    with pytest.raises(ValueError, match="at least one"):
        crt([], [])


def test_crt_modulus_zero_raises():
    with pytest.raises(ValueError, match=">= 1"):
        crt([1, 2], [3, 0])


def test_crt_modulus_negative_raises():
    with pytest.raises(ValueError, match=">= 1"):
        crt([1, 2], [3, -2])


def test_crt_result_in_range():
    x, N = crt([2, 3, 2], [3, 5, 7])
    assert 0 <= x < N


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------


def test_validate_no_solve():
    args = argparse.Namespace(solve=None, all=None, format="table")
    result = validate(args)
    assert result is not None and "--solve" in result


def test_validate_single_value():
    args = argparse.Namespace(solve=[3], all=None, format="table")
    result = validate(args)
    assert result is not None and "pair" in result


def test_validate_odd_values():
    args = argparse.Namespace(solve=[2, 3, 4], all=None, format="table")
    result = validate(args)
    assert result is not None and "even" in result


def test_validate_modulus_zero():
    # Moduli are at odd indices; index 1 is the first modulus
    args = argparse.Namespace(solve=[2, 0, 3, 5], all=None, format="table")
    result = validate(args)
    assert result is not None and ">= 1" in result


def test_validate_modulus_negative():
    args = argparse.Namespace(solve=[2, 3, 1, -5], all=None, format="table")
    result = validate(args)
    assert result is not None and ">= 1" in result


def test_validate_all_negative():
    args = argparse.Namespace(solve=[2, 3, 3, 5], all=-1, format="table")
    result = validate(args)
    assert result is not None and "--all" in result


def test_validate_valid():
    args = argparse.Namespace(solve=[2, 3, 3, 5, 2, 7], all=None, format="table")
    assert validate(args) is None


def test_validate_valid_with_all():
    args = argparse.Namespace(solve=[2, 3, 3, 5], all=100, format="table")
    assert validate(args) is None


# ---------------------------------------------------------------------------
# format_solution
# ---------------------------------------------------------------------------


def test_format_solution_contains_congruences():
    out = format_solution([2, 3, 2], [3, 5, 7], 23, 105, None)
    assert "x ≡ 2 (mod 3)" in out
    assert "x ≡ 3 (mod 5)" in out
    assert "x ≡ 2 (mod 7)" in out


def test_format_solution_shows_result():
    out = format_solution([2, 3, 2], [3, 5, 7], 23, 105, None)
    assert "23" in out
    assert "105" in out


def test_format_solution_verification_marks():
    out = format_solution([2, 3, 2], [3, 5, 7], 23, 105, None)
    assert "✓" in out


def test_format_solution_with_all():
    out = format_solution([2, 3], [3, 5], 2, 15, 50)
    assert "All solutions" in out
    assert "2" in out
    assert "17" in out
    assert "32" in out
    assert "47" in out


def test_format_solution_no_all():
    out = format_solution([2, 3], [3, 5], 2, 15, None)
    assert "All solutions" not in out


# ---------------------------------------------------------------------------
# format_json
# ---------------------------------------------------------------------------


def test_format_json_basic():
    out = format_json([2, 3, 2], [3, 5, 7], 23, 105, None)
    data = json.loads(out)
    assert data["solution"] == 23
    assert data["modulus"] == 105
    assert len(data["congruences"]) == 3
    assert all(c["valid"] for c in data["congruences"])


def test_format_json_with_all():
    out = format_json([2, 3], [3, 5], 2, 15, 50)
    data = json.loads(out)
    assert "all_solutions" in data
    assert 2 in data["all_solutions"]
    assert 17 in data["all_solutions"]


def test_format_json_no_all():
    out = format_json([2, 3, 2], [3, 5, 7], 23, 105, None)
    data = json.loads(out)
    assert "all_solutions" not in data


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def test_main_no_args_returns_2(capsys):
    assert main([]) == 2


def test_main_odd_values_returns_2(capsys):
    assert main(["--solve", "2", "3", "4"]) == 2


def test_main_bad_modulus_returns_2(capsys):
    assert main(["--solve", "2", "0"]) == 2


def test_main_sunzi_table(capsys):
    rc = main(["--solve", "2", "3", "3", "5", "2", "7"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "23" in out
    assert "105" in out


def test_main_sunzi_json(capsys):
    rc = main(["--solve", "2", "3", "3", "5", "2", "7", "--format", "json"])
    out = capsys.readouterr().out
    assert rc == 0
    data = json.loads(out)
    assert data["solution"] == 23


def test_main_with_all_table(capsys):
    rc = main(["--solve", "2", "3", "3", "5", "--all", "100"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "All solutions" in out


def test_main_with_all_json(capsys):
    rc = main(["--solve", "2", "3", "3", "5", "--all", "50", "--format", "json"])
    out = capsys.readouterr().out
    assert rc == 0
    data = json.loads(out)
    assert "all_solutions" in data


def test_main_all_negative_returns_2(capsys):
    assert main(["--solve", "2", "3", "3", "5", "--all", "-1"]) == 2


def test_main_inconsistent_system_returns_2(capsys):
    # x≡0(4) and x≡3(6) are inconsistent
    assert main(["--solve", "0", "4", "3", "6"]) == 2
    err = capsys.readouterr().err
    assert "Error" in err


def test_main_crt_error_path(monkeypatch, capsys):
    """Cover the ValueError branch in main when crt() raises."""

    def raise_value_error(*_args, **_kwargs):
        raise ValueError("forced crt error")

    monkeypatch.setattr(crt_module, "crt", raise_value_error)
    assert main(["--solve", "2", "3", "3", "5"]) == 2
    err = capsys.readouterr().err
    assert "forced crt error" in err
