"""Tests for birthday problem collision probability functions."""

import argparse
import json
import math
from unittest.mock import patch
import pytest

from src.utils.birthday_problem import (
    _parse_weights,
    collision_prob_nonuniform,
    collision_prob_uniform,
    effective_pool_size,
    expected_duplicate_pairs,
    format_csv_output,
    main,
    min_group_for_prob,
    prob_table,
)


# ---------------------------------------------------------------------------
# collision_prob_uniform
# ---------------------------------------------------------------------------


def test_collision_prob_one_when_n_exceeds_d():
    """Pigeonhole: n > d guarantees a duplicate."""
    assert collision_prob_uniform(366, 365) == 1.0


# ---------------------------------------------------------------------------
# collision_prob_nonuniform
# ---------------------------------------------------------------------------


def test_nonuniform_one_person_zero():
    """One person cannot produce a duplicate."""
    assert collision_prob_nonuniform(1, [0.5, 0.3, 0.2]) == 0.0


def test_nonuniform_raises_on_all_zero_weights():
    """All-zero weights should raise ValueError."""
    with pytest.raises(ValueError):
        collision_prob_nonuniform(5, [0.0, 0.0])


# ---------------------------------------------------------------------------
# effective_pool_size
# ---------------------------------------------------------------------------


def test_effective_pool_size_raises_on_all_zero_weights():
    """All-zero weights should raise ValueError."""
    with pytest.raises(ValueError):
        effective_pool_size([0.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# expected_duplicate_pairs
# ---------------------------------------------------------------------------


def test_expected_pairs_formula():
    """E[pairs] = n*(n-1)/2 / d."""
    assert abs(expected_duplicate_pairs(23, 365) - 23 * 22 / 2 / 365) < 1e-12


def test_expected_pairs_one_person():
    """No pairs possible with one person."""
    assert expected_duplicate_pairs(1, 365) == 0.0


# ---------------------------------------------------------------------------
# min_group_for_prob
# ---------------------------------------------------------------------------


def test_min_group_target_zero_returns_one():
    assert min_group_for_prob(0.0, 365) == 1


def test_min_group_target_one_returns_d_plus_one():
    assert min_group_for_prob(1.0, 365) == 366


def test_min_group_returns_none_when_max_n_too_small():
    """None is returned when max_n is too small to reach the target probability."""
    result = min_group_for_prob(0.5, 365, max_n=5)
    assert result is None


# ---------------------------------------------------------------------------
# prob_table
# ---------------------------------------------------------------------------


def test_prob_table_probability_monotone():
    rows = prob_table(365, 1, 40)
    probs = [r["probability"] for r in rows]
    assert all(probs[i] <= probs[i + 1] for i in range(len(probs) - 1))


# ---------------------------------------------------------------------------
# main() CLI
# ---------------------------------------------------------------------------


def test_main_single_group_returns_zero():
    assert main(["--pool-size", "365", "--group-size", "23"]) == 0


def test_main_range_table_returns_zero():
    assert main(["--pool-size", "365", "--range", "1", "30"]) == 0


def test_main_invalid_target_prob_returns_two():
    assert main(["--pool-size", "365", "--target-prob", "1.5"]) == 2


def test_main_no_mode_returns_two():
    assert main(["--pool-size", "365"]) == 2


# ---------------------------------------------------------------------------
# format_csv_output
# ---------------------------------------------------------------------------


def test_format_csv_output_empty_rows_returns_empty_string():
    """format_csv_output returns an empty string when given an empty list."""
    assert format_csv_output([]) == ""


# ---------------------------------------------------------------------------
# _parse_weights
# ---------------------------------------------------------------------------


def test_parse_weights_non_numeric_raises():
    """Non-numeric tokens raise ArgumentTypeError (lines 193-194)."""
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_weights("a,b,c")


def test_parse_weights_negative_raises():
    """A negative weight raises ArgumentTypeError (line 198)."""
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_weights("-1,0.5,0.5")


def test_parse_weights_all_zero_raises():
    """All-zero weights raise ArgumentTypeError (line 200)."""
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_weights("0,0,0")


# ---------------------------------------------------------------------------
# validate — pool_size < 1
# ---------------------------------------------------------------------------


def test_main_pool_size_below_one_returns_two():
    """A pool size less than 1 is rejected by validate (line 260)."""
    assert main(["--pool-size", "0.5", "--group-size", "10"]) == 2


def test_main_range_inverted_returns_two():
    assert main(["--pool-size", "365", "--range", "50", "10"]) == 2


def test_main_target_prob_output_contains_23(capsys):
    """--target-prob 0.5 should report minimum group size of 23 for 365-day pool."""
    main(["--pool-size", "365", "--target-prob", "0.5"])
    captured = capsys.readouterr()
    assert "23" in captured.out


def test_main_json_range_structure(capsys):
    """JSON range output contains list of objects with expected keys."""
    main(["--pool-size", "365", "--range", "1", "5", "--format", "json"])
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert isinstance(data, list)
    assert len(data) == 5
    assert "n" in data[0]
    assert "probability" in data[0]


def test_main_csv_range_has_header(capsys):
    """CSV output includes column headers."""
    main(["--pool-size", "365", "--range", "1", "5", "--format", "csv"])
    captured = capsys.readouterr()
    assert "probability" in captured.out
    assert "expected_duplicate_pairs" in captured.out


def test_main_nonuniform_weights_returns_zero():
    """--weights mode with --group-size should succeed."""
    assert main(["--weights", "0.10,0.15,0.20,0.30,0.25", "--group-size", "10"]) == 0


def test_main_nonuniform_output_shows_effective_pool(capsys):
    """Non-uniform output should mention effective pool size."""
    main(["--weights", "1,1,1,1", "--group-size", "4"])
    captured = capsys.readouterr()
    assert "Effective pool size" in captured.out


def test_main_weights_with_target_prob_returns_two():
    """--weights + --target-prob (no --group-size) is rejected by validate (lines 264-265)."""
    assert main(["--weights", "0.1,0.2,0.7", "--target-prob", "0.5"]) == 2


def test_main_weights_with_range_returns_two():
    """--weights + --range (no --group-size) is rejected by validate (lines 266-267)."""
    assert main(["--weights", "0.1,0.2,0.7", "--range", "1", "10"]) == 2


def test_main_group_size_below_one_returns_two():
    """--group-size of 0 is rejected by validate (line 270)."""
    assert main(["--pool-size", "365", "--group-size", "0"]) == 2


def test_main_range_min_n_below_one_returns_two():
    """A range MIN_N of 0 is rejected by validate (line 278)."""
    assert main(["--pool-size", "365", "--range", "0", "10"]) == 2


def test_main_range_span_exceeds_limit_returns_two():
    """A range span greater than 100,000 is rejected by validate (line 282)."""
    assert main(["--pool-size", "365", "--range", "1", "100002"]) == 2


def test_main_target_prob_unreachable_returns_one(capsys):
    """When min_group_for_prob returns None, main() prints to stderr and returns 1 (lines 323-327)."""
    with patch("src.utils.birthday_problem.min_group_for_prob", return_value=None):
        result = main(["--pool-size", "365", "--target-prob", "0.5"])
    assert result == 1
    captured = capsys.readouterr()
    assert "reaches" in captured.err
