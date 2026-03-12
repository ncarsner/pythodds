"""Tests for expected value and discrete distribution functions."""

import json
import math
import os
import tempfile

from src.utils.expected_value import (
    entropy,
    expected_value,
    load_file,
    main,
    mgf,
    std_dev,
    validate,
    validate_distribution,
    variance,
)

# ---------------------------------------------------------------------------
# expected_value
# ---------------------------------------------------------------------------


def test_ev_fair_die():
    """E[X] for a fair six-sided die is 3.5."""
    outcomes = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    probs = [1 / 6] * 6
    assert abs(expected_value(outcomes, probs) - 3.5) < 1e-10


# ---------------------------------------------------------------------------
# variance
# ---------------------------------------------------------------------------


def test_variance_constant():
    """Variance of a degenerate distribution is 0."""
    assert variance([5.0], [1.0]) == 0.0


# ---------------------------------------------------------------------------
# std_dev
# ---------------------------------------------------------------------------


def test_std_dev_is_sqrt_variance():
    outcomes = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    probs = [1 / 6] * 6
    assert abs(std_dev(outcomes, probs) - math.sqrt(variance(outcomes, probs))) < 1e-15


# ---------------------------------------------------------------------------
# entropy
# ---------------------------------------------------------------------------


def test_entropy_uniform_binary():
    """H(fair coin) = 1 bit."""
    assert abs(entropy([0.5, 0.5]) - 1.0) < 1e-15


# ---------------------------------------------------------------------------
# mgf
# ---------------------------------------------------------------------------


def test_mgf_bernoulli():
    """M_X(t) = (1-p) + p*e^t for Bernoulli(p)."""
    p = 0.4
    t = 1.0
    expected = (1 - p) + p * math.exp(t)
    assert abs(mgf([0.0, 1.0], [1 - p, p], t) - expected) < 1e-12


# ---------------------------------------------------------------------------
# load_file — CSV
# ---------------------------------------------------------------------------


def _write_tmp(content: str, suffix: str) -> str:
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.write(fd, content.encode())
    os.close(fd)
    return path


def test_load_csv_headerless():
    path = _write_tmp("0,0.6\n1,0.4\n", ".csv")
    try:
        outcomes, probs = load_file(path)
        assert outcomes == [0.0, 1.0]
        assert probs == [0.6, 0.4]
    finally:
        os.unlink(path)


def test_load_json_list_format():
    data = [{"outcome": 1, "prob": 0.3}, {"outcome": 2, "prob": 0.7}]
    path = _write_tmp(json.dumps(data), ".json")
    try:
        outcomes, probs = load_file(path)
        assert outcomes == [1.0, 2.0]
        assert abs(probs[0] - 0.3) < 1e-15
    finally:
        os.unlink(path)


def test_load_json_dict_probabilities_key():
    """JSON dict using 'probabilities' key (line 124) is accepted."""
    data = {"outcomes": [0, 1], "probabilities": [0.4, 0.6]}
    path = _write_tmp(json.dumps(data), ".json")
    try:
        outcomes, probs = load_file(path)
        assert outcomes == [0.0, 1.0]
        assert abs(probs[0] - 0.4) < 1e-15
        assert abs(probs[1] - 0.6) < 1e-15
    finally:
        os.unlink(path)


def test_load_csv_unknown_header_falls_back_to_columns():
    """CSV with unrecognised header names falls back to col 0/1 (lines 79-80)."""
    path = _write_tmp("label,weight\n5,0.3\n10,0.7\n", ".csv")
    try:
        outcomes, probs = load_file(path)
        assert outcomes == [5.0, 10.0]
        assert abs(probs[0] - 0.3) < 1e-15
    finally:
        os.unlink(path)


def test_load_csv_skips_empty_rows():
    """Empty rows embedded in a CSV are silently skipped (line 85)."""
    path = _write_tmp("outcome,prob\n1,0.5\n\n2,0.5\n", ".csv")
    try:
        outcomes, probs = load_file(path)
        assert outcomes == [1.0, 2.0]
        assert probs == [0.5, 0.5]
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# main — outcomes mode
# ---------------------------------------------------------------------------


def test_main_basic_returns_zero(capsys):
    rc = main(["--outcomes", "0,1,5,10", "--probs", "0.50,0.25,0.15,0.10"])
    assert rc == 0


def test_main_fair_coin_ev(capsys):
    main(["--outcomes", "0,1", "--probs", "0.5,0.5"])
    out = capsys.readouterr().out
    assert "E[X]:              0.500000" in out


def test_main_mgf_output(capsys):
    main(["--outcomes", "0,1", "--probs", "0.5,0.5", "--mgf", "0.0"])
    out = capsys.readouterr().out
    assert "MGF" in out
    assert "1.000000" in out


# ---------------------------------------------------------------------------
# main — file mode
# ---------------------------------------------------------------------------


def test_main_missing_file_returns_2(capsys):
    rc = main(["--file", "/nonexistent/path/file.csv"])
    assert rc == 2


# ---------------------------------------------------------------------------
# main — validation errors
# ---------------------------------------------------------------------------


def test_main_no_arguments_returns_2(capsys):
    rc = main([])
    assert rc == 2


def test_main_outcomes_without_probs_returns_2(capsys):
    rc = main(["--outcomes", "1,2,3"])
    assert rc == 2


def test_main_mismatched_lengths_returns_2(capsys):
    rc = main(["--outcomes", "1,2,3", "--probs", "0.5,0.5"])
    assert rc == 2


def test_validate_distribution_empty_outcomes():
    """Empty outcomes list returns the line-124 error message."""
    result = validate_distribution([], [])
    assert result == "at least one outcome/probability pair is required"


def test_validate_both_file_and_outcomes():
    """validate() returns the line-139 message when both file and outcomes are set."""
    import argparse

    ns = argparse.Namespace(
        file="dummy.csv", outcomes="1,2", probs="0.5,0.5", mgf_t=None
    )
    result = validate(ns)
    assert result == "--outcomes and --file are mutually exclusive"


def test_main_probs_not_summing_to_one_returns_2(capsys):
    rc = main(["--outcomes", "1,2", "--probs", "0.4,0.4"])
    assert rc == 2


def test_main_negative_prob_returns_2(capsys):
    # Use = form so argparse doesn't interpret the leading - as a flag
    rc = main(["--outcomes", "1,2", "--probs=-0.1,1.1"])
    assert rc == 2


def test_main_infinite_mgf_returns_2(capsys):
    """--mgf inf is rejected by validate (line 143)."""
    rc = main(["--outcomes", "0,1", "--probs", "0.5,0.5", "--mgf", "inf"])
    assert rc == 2


def test_main_non_numeric_probs_returns_2(capsys):
    """Non-numeric probability string triggers ValueError handler (lines 262-264)."""
    rc = main(["--outcomes", "1,2", "--probs", "0.5,x"])
    assert rc == 2
