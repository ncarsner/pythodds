"""Tests for Poisson distribution functions."""

import json

from src.utils.poisson_distribution import (
    format_csv_output,
    main,
    min_k_for_prob,
    poisson_cdf_ge,
    poisson_cdf_le,
    poisson_pmf,
    prob_table,
)

# ---------------------------------------------------------------------------
# poisson_pmf
# ---------------------------------------------------------------------------


def test_pmf_negative_k_zero():
    """P(X=k) for k < 0 is 0."""
    assert poisson_pmf(-1, 3.0) == 0.0


def test_pmf_lam_zero_k_positive():
    """P(X=k | λ=0) = 0 for k > 0."""
    assert poisson_pmf(3, 0.0) == 0.0


# ---------------------------------------------------------------------------
# poisson_cdf_le
# ---------------------------------------------------------------------------


def test_cdf_le_negative_k():
    """P(X ≤ k) for k < 0 is 0."""
    assert poisson_cdf_le(-1, 3.0) == 0.0


# ---------------------------------------------------------------------------
# min_k_for_prob
# ---------------------------------------------------------------------------


def test_min_k_target_zero_returns_zero():
    assert min_k_for_prob(0.0, 3.0) == 0


# ---------------------------------------------------------------------------
# prob_table
# ---------------------------------------------------------------------------


def test_prob_table_cdf_ge_matches_function():
    lam = 2.0
    rows = prob_table(lam, 0, 12)
    for r in rows:
        assert abs(r["cdf_ge"] - poisson_cdf_ge(r["k"], lam)) < 1e-10


# ---------------------------------------------------------------------------
# main — single event mode
# ---------------------------------------------------------------------------


def test_main_single_event_with_min_prob_fails(capsys):
    main(["-l", "3.0", "-k", "5", "--target", "10", "--min-prob", "0.99"])
    out = capsys.readouterr().out
    assert "False" in out


# ---------------------------------------------------------------------------
# main — target-prob mode
# ---------------------------------------------------------------------------


def test_main_target_prob_output_contains_k(capsys):
    main(["-l", "3.0", "-t", "0.95"])
    out = capsys.readouterr().out
    # Min k for P(X<=k)>=0.95 with λ=3 is 6
    assert "6" in out


# ---------------------------------------------------------------------------
# main — range table mode
# ---------------------------------------------------------------------------


def test_main_range_returns_zero(capsys):
    rc = main(["-l", "3.0", "-r", "0", "10"])
    assert rc == 0


def test_main_range_table_has_header(capsys):
    main(["-l", "3.0", "-r", "0", "5"])
    out = capsys.readouterr().out
    assert "P(X=k)" in out
    assert "P(X≤k)" in out


def test_main_range_json_structure(capsys):
    main(["-l", "3.0", "-r", "0", "5", "-f", "json"])
    out = capsys.readouterr().out
    data = json.loads(out)
    assert len(data) == 6
    assert all(
        "k" in r and "pmf" in r and "cdf_le" in r and "cdf_ge" in r for r in data
    )


def test_main_range_csv_has_header(capsys):
    main(["-l", "3.0", "-r", "0", "5", "-f", "csv"])
    out = capsys.readouterr().out
    assert out.splitlines()[0] == "k,pmf,cdf_le,cdf_ge"


# ---------------------------------------------------------------------------
# main — validation errors
# ---------------------------------------------------------------------------


def test_main_rate_negative_returns_two(capsys):
    rc = main(["-l", "-1.0", "-k", "3"])
    assert rc == 2


def test_main_no_mode_returns_two(capsys):
    rc = main(["-l", "3.0"])
    assert rc == 2


def test_main_negative_events_returns_two(capsys):
    rc = main(["-l", "3.0", "-k", "-1"])
    assert rc == 2


def test_main_invalid_target_prob_returns_two(capsys):
    rc = main(["-l", "3.0", "-t", "1.5"])
    assert rc == 2


def test_main_range_inverted_returns_two(capsys):
    rc = main(["-l", "3.0", "-r", "10", "5"])
    assert rc == 2


def test_main_target_without_events_returns_two(capsys):
    rc = main(["-l", "3.0", "-t", "0.95", "--target", "5"])
    assert rc == 2


def test_main_min_prob_without_target_returns_two(capsys):
    rc = main(["-l", "3.0", "-k", "5", "--min-prob", "0.5"])
    assert rc == 2


def test_min_k_for_prob_returns_none_when_max_k_too_small():
    """min_k_for_prob returns None when max_k is too small to reach the target (line 70)."""
    result = min_k_for_prob(0.9999, 3.0, max_k=2)
    assert result is None


def test_format_csv_output_empty_rows_returns_empty_string():
    """format_csv_output returns an empty string for an empty row list (line 160)."""
    assert format_csv_output([]) == ""


def test_main_range_negative_min_k_returns_two():
    """A negative MIN_K is rejected by validate (line 245)."""
    assert main(["-l", "3.0", "-r", "-1", "10"]) == 2


def test_main_range_span_exceeds_limit_returns_two():
    """A range span greater than 100,000 is rejected by validate (line 249)."""
    assert main(["-l", "3.0", "-r", "1", "100002"]) == 2


def test_main_min_prob_out_of_range_returns_two():
    """--min-prob outside [0, 1] is rejected by validate (line 258)."""
    assert main(["-l", "3.0", "-k", "5", "--target", "3", "--min-prob", "1.5"]) == 2


def test_main_target_prob_unreachable_returns_one(capsys):
    """When min_k_for_prob returns None, main prints to stderr and returns 1 (lines 292-296)."""
    from unittest.mock import patch

    with patch("src.utils.poisson_distribution.min_k_for_prob", return_value=None):
        result = main(["-l", "3.0", "-t", "0.95"])
    assert result == 1
    assert "reaches" in capsys.readouterr().err
