"""Tests for z-score functions."""

import argparse
import math

import pytest

import src.utils.z_score as z_score_module
from src.utils.z_score import (
    format_dataset,
    format_single,
    main,
    mean,
    parse_number_list,
    std_dev,
    validate,
    variance,
    z_score,
    z_scores,
)

# ---------------------------------------------------------------------------
# mean / variance / standard deviation
# ---------------------------------------------------------------------------


def test_mean_basic():
    assert mean([2.0, 4.0, 6.0]) == 4.0


def test_mean_empty_raises_value_error():
    with pytest.raises(ValueError, match="empty list"):
        mean([])


def test_population_variance_known_dataset():
    values = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
    assert variance(values) == 4.0


def test_sample_variance_known_dataset():
    values = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
    assert abs(variance(values, sample=True) - (32.0 / 7.0)) < 1e-12


def test_variance_empty_raises_value_error():
    with pytest.raises(ValueError, match="empty list"):
        variance([])


def test_sample_variance_one_value_raises_value_error():
    with pytest.raises(ValueError, match="at least 2 values"):
        variance([1.0], sample=True)


def test_std_dev_is_sqrt_variance():
    values = [1.0, 2.0, 3.0]
    assert std_dev(values) == math.sqrt(variance(values))


# ---------------------------------------------------------------------------
# z_score / z_scores
# ---------------------------------------------------------------------------


def test_z_score_above_mean():
    assert z_score(85.0, 70.0, 10.0) == 1.5


def test_z_score_below_mean():
    assert z_score(55.0, 70.0, 10.0) == -1.5


def test_z_score_zero_std_raises_value_error():
    with pytest.raises(ValueError, match="greater than 0"):
        z_score(1.0, 0.0, 0.0)


def test_z_score_negative_std_raises_value_error():
    with pytest.raises(ValueError, match="greater than 0"):
        z_score(1.0, 0.0, -1.0)


def test_z_scores_known_dataset_population():
    values = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
    assert z_scores(values) == [-1.5, -0.5, -0.5, -0.5, 0.0, 0.0, 1.0, 2.0]


def test_z_scores_known_dataset_sample():
    values = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
    scores = z_scores(values, sample=True)
    assert abs(scores[0] - (-3.0 / math.sqrt(32.0 / 7.0))) < 1e-12


def test_z_scores_constant_values_raises_value_error():
    with pytest.raises(ValueError, match="greater than 0"):
        z_scores([3.0, 3.0, 3.0])


# ---------------------------------------------------------------------------
# parse_number_list
# ---------------------------------------------------------------------------


def test_parse_number_list_trims_spaces_and_empty_parts():
    assert parse_number_list(" 1, 2, , 3 ") == [1.0, 2.0, 3.0]


def test_parse_number_list_empty_raises_value_error():
    with pytest.raises(ValueError, match="at least one value"):
        parse_number_list(" , ")


def test_parse_number_list_non_numeric_raises_value_error():
    with pytest.raises(ValueError):
        parse_number_list("1,two,3")


def test_parse_number_list_infinite_raises_value_error():
    with pytest.raises(ValueError, match="finite"):
        parse_number_list("1,inf,3")


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


def test_validate_requires_a_mode():
    args = argparse.Namespace(
        value=None, values=None, mean=None, std=None, precision=6, sample=False
    )
    assert validate(args) == "one of --value or --values is required"


def test_validate_single_requires_mean_and_std():
    args = argparse.Namespace(
        value=1.0, values=None, mean=None, std=1.0, precision=6, sample=False
    )
    assert validate(args) == "--mean and --std are required when --value is provided"


def test_validate_single_rejects_infinite_values():
    args = argparse.Namespace(
        value=float("inf"), values=None, mean=0.0, std=1.0, precision=6, sample=False
    )
    assert validate(args) == "--value, --mean, and --std must be finite"


def test_validate_single_rejects_non_positive_std():
    args = argparse.Namespace(
        value=1.0, values=None, mean=0.0, std=0.0, precision=6, sample=False
    )
    assert validate(args) == "--std must be greater than 0"


def test_validate_rejects_negative_precision():
    args = argparse.Namespace(
        value=None, values="1,2,3", mean=None, std=None, precision=-1, sample=False
    )
    assert validate(args) == "--precision must be non-negative"


# ---------------------------------------------------------------------------
# output formatting
# ---------------------------------------------------------------------------


def test_format_single_contains_z_score():
    output = format_single(85.0, 70.0, 10.0, 1.5, 2)
    assert "Z-score:           1.50" in output


def test_format_dataset_contains_population_label_and_scores():
    output = format_dataset([2.0, 4.0], 3.0, 1.0, [-1.0, 1.0], False, 1)
    assert "Std dev (population): 1.0" in output
    assert "2.0 -> -1.0" in output


def test_format_dataset_contains_sample_label():
    output = format_dataset([2.0, 4.0], 3.0, math.sqrt(2.0), [-0.707, 0.707], True, 3)
    assert "Std dev (sample):" in output


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def test_main_single_value_output(capsys):
    rc = main(["-x", "85", "-m", "70", "-s", "10", "--precision", "2"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Z-score:           1.50" in out


def test_main_values_output_contains_known_z_score(capsys):
    rc = main(["--values", "2,4,4,4,5,5,7,9", "--precision", "1"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "2.0 -> -1.5" in out
    assert "9.0 -> 2.0" in out


def test_main_sample_values_output(capsys):
    main(["--values", "1,2,3", "--sample"])
    out = capsys.readouterr().out
    assert "Std dev (sample)" in out


def test_main_missing_arguments_returns_2(capsys):
    assert main([]) == 2


def test_main_single_missing_mean_returns_2(capsys):
    assert main(["--value", "10", "--std", "2"]) == 2


def test_main_single_invalid_std_returns_2(capsys):
    assert main(["--value", "10", "--mean", "5", "--std", "0"]) == 2


def test_main_single_calculation_error_returns_2(monkeypatch, capsys):
    def raise_value_error(_value, _mean, _std):
        raise ValueError("calculation failed")

    monkeypatch.setattr(z_score_module, "z_score", raise_value_error)

    assert main(["--value", "10", "--mean", "5", "--std", "2"]) == 2
    err = capsys.readouterr().err
    assert "Error: calculation failed" in err


def test_main_invalid_values_returns_2(capsys):
    assert main(["--values", "1,not-a-number,3"]) == 2


def test_main_constant_values_returns_2(capsys):
    assert main(["--values", "4,4,4"]) == 2
