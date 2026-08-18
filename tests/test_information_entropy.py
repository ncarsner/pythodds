"""Tests for the information entropy utility."""

import argparse
import json
import math

import pytest

import src.utils.information_entropy as entropy_module
from src.utils.information_entropy import (
    base_value,
    build_result,
    conditional_entropy,
    cross_entropy,
    format_json,
    format_table,
    joint_entropy,
    kl_divergence,
    main,
    marginals,
    max_entropy,
    mutual_information,
    normalize,
    normalize_joint,
    parse_values,
    shannon_entropy,
    validate,
)

# A joint table for two perfectly correlated binary variables.
PERFECT = [[0.5, 0.0], [0.0, 0.5]]
# A joint table for two independent binary variables.
INDEPENDENT = [[0.25, 0.25], [0.25, 0.25]]


def _args(**overrides):
    """Build a namespace with valid defaults, overridden per test."""
    base = dict(
        probs="0.5,0.5",
        measure="entropy",
        base="2",
        probs_p=None,
        probs_q=None,
        joint=None,
        format="table",
        precision=4,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


# ---------------------------------------------------------------------------
# normalize
# ---------------------------------------------------------------------------


def test_normalize_scales_counts_to_probabilities():
    assert normalize([3, 1]) == pytest.approx([0.75, 0.25])


def test_normalize_leaves_a_distribution_unchanged():
    assert normalize([0.2, 0.8]) == pytest.approx([0.2, 0.8])


def test_normalize_empty_raises():
    with pytest.raises(ValueError, match="at least one outcome"):
        normalize([])


def test_normalize_negative_raises():
    with pytest.raises(ValueError, match="must be >= 0"):
        normalize([0.5, -0.1])


def test_normalize_zero_total_raises():
    with pytest.raises(ValueError, match="positive total"):
        normalize([0.0, 0.0])


# ---------------------------------------------------------------------------
# shannon_entropy / max_entropy
# ---------------------------------------------------------------------------


def test_shannon_entropy_fair_coin_is_one_bit():
    assert shannon_entropy([0.5, 0.5]) == pytest.approx(1.0)


def test_shannon_entropy_fair_die_is_log2_six():
    assert shannon_entropy([1] * 6) == pytest.approx(math.log2(6))


def test_shannon_entropy_certain_outcome_is_zero():
    assert shannon_entropy([1.0, 0.0]) == pytest.approx(0.0)


def test_shannon_entropy_in_nats():
    assert shannon_entropy([0.5, 0.5], math.e) == pytest.approx(math.log(2))


def test_shannon_entropy_in_hartleys():
    assert shannon_entropy([0.1] * 10, 10.0) == pytest.approx(1.0)


def test_shannon_entropy_accepts_unnormalized_counts():
    assert shannon_entropy([2, 2]) == pytest.approx(1.0)


def test_shannon_entropy_invalid_base_raises():
    with pytest.raises(ValueError, match="base must be > 1"):
        shannon_entropy([0.5, 0.5], 1.0)


def test_max_entropy_matches_log():
    assert max_entropy(8) == pytest.approx(3.0)


def test_max_entropy_single_outcome_is_zero():
    assert max_entropy(1) == pytest.approx(0.0)


def test_max_entropy_zero_outcomes_raises():
    with pytest.raises(ValueError, match="outcomes must be >= 1"):
        max_entropy(0)


def test_max_entropy_invalid_base_raises():
    with pytest.raises(ValueError, match="base must be > 1"):
        max_entropy(4, 0.5)


def test_entropy_never_exceeds_max_entropy():
    probs = [0.5, 0.3, 0.15, 0.05]
    assert shannon_entropy(probs) <= max_entropy(len(probs))


# ---------------------------------------------------------------------------
# kl_divergence / cross_entropy
# ---------------------------------------------------------------------------


def test_kl_divergence_identical_distributions_is_zero():
    assert kl_divergence([0.7, 0.3], [0.7, 0.3]) == pytest.approx(0.0)


def test_kl_divergence_known_value():
    expected = 0.7 * math.log2(0.7 / 0.5) + 0.3 * math.log2(0.3 / 0.5)
    assert kl_divergence([0.7, 0.3], [0.5, 0.5]) == pytest.approx(expected)


def test_kl_divergence_is_asymmetric():
    forward = kl_divergence([0.7, 0.3], [0.5, 0.5])
    reverse = kl_divergence([0.5, 0.5], [0.7, 0.3])
    assert forward != pytest.approx(reverse)


def test_kl_divergence_skips_zero_probability_in_p():
    assert kl_divergence([1.0, 0.0], [0.5, 0.5]) == pytest.approx(1.0)


def test_kl_divergence_zero_in_q_raises():
    with pytest.raises(ValueError, match="KL divergence is infinite"):
        kl_divergence([0.5, 0.5], [1.0, 0.0])


def test_kl_divergence_length_mismatch_raises():
    with pytest.raises(ValueError, match="same length"):
        kl_divergence([0.5, 0.5], [1.0])


def test_kl_divergence_invalid_base_raises():
    with pytest.raises(ValueError, match="base must be > 1"):
        kl_divergence([0.5, 0.5], [0.5, 0.5], 1.0)


def test_cross_entropy_equals_entropy_plus_kl():
    p, q = [0.7, 0.3], [0.5, 0.5]
    assert cross_entropy(p, q) == pytest.approx(
        shannon_entropy(p) + kl_divergence(p, q)
    )


def test_cross_entropy_with_identical_distributions_is_entropy():
    assert cross_entropy([0.7, 0.3], [0.7, 0.3]) == pytest.approx(
        shannon_entropy([0.7, 0.3])
    )


def test_cross_entropy_skips_zero_probability_in_p():
    assert cross_entropy([1.0, 0.0], [0.5, 0.5]) == pytest.approx(1.0)


def test_cross_entropy_zero_in_q_raises():
    with pytest.raises(ValueError, match="cross-entropy is infinite"):
        cross_entropy([0.5, 0.5], [1.0, 0.0])


def test_cross_entropy_length_mismatch_raises():
    with pytest.raises(ValueError, match="same length"):
        cross_entropy([0.5, 0.5], [1.0])


def test_cross_entropy_invalid_base_raises():
    with pytest.raises(ValueError, match="base must be > 1"):
        cross_entropy([0.5, 0.5], [0.5, 0.5], 1.0)


# ---------------------------------------------------------------------------
# normalize_joint / marginals
# ---------------------------------------------------------------------------


def test_normalize_joint_scales_counts():
    assert normalize_joint([[1, 1], [1, 1]]) == [[0.25, 0.25], [0.25, 0.25]]


def test_normalize_joint_empty_raises():
    with pytest.raises(ValueError, match="at least one row"):
        normalize_joint([])


def test_normalize_joint_empty_row_raises():
    with pytest.raises(ValueError, match="at least one column"):
        normalize_joint([[]])


def test_normalize_joint_ragged_raises():
    with pytest.raises(ValueError, match="same length"):
        normalize_joint([[0.5, 0.5], [1.0]])


def test_marginals_of_independent_table():
    px, py = marginals(INDEPENDENT)
    assert px == pytest.approx([0.5, 0.5])
    assert py == pytest.approx([0.5, 0.5])


def test_marginals_of_asymmetric_table():
    px, py = marginals([[0.4, 0.2], [0.1, 0.3]])
    assert px == pytest.approx([0.6, 0.4])
    assert py == pytest.approx([0.5, 0.5])


# ---------------------------------------------------------------------------
# joint_entropy / mutual_information / conditional_entropy
# ---------------------------------------------------------------------------


def test_joint_entropy_of_independent_table_is_two_bits():
    assert joint_entropy(INDEPENDENT) == pytest.approx(2.0)


def test_mutual_information_of_independent_table_is_zero():
    assert mutual_information(INDEPENDENT) == pytest.approx(0.0)


def test_mutual_information_of_perfect_dependence_is_one_bit():
    assert mutual_information(PERFECT) == pytest.approx(1.0)


def test_mutual_information_is_never_negative():
    assert mutual_information([[0.3, 0.2], [0.2, 0.3]]) >= 0.0


def test_conditional_entropy_of_independent_table_is_marginal_entropy():
    assert conditional_entropy(INDEPENDENT) == pytest.approx(1.0)


def test_conditional_entropy_of_perfect_dependence_is_zero():
    assert conditional_entropy(PERFECT) == pytest.approx(0.0)


def test_chain_rule_holds():
    joint = [[0.4, 0.2], [0.1, 0.3]]
    px, _py = marginals(joint)
    assert joint_entropy(joint) == pytest.approx(
        shannon_entropy(px) + conditional_entropy(joint)
    )


# ---------------------------------------------------------------------------
# parse_values / base_value
# ---------------------------------------------------------------------------


def test_parse_values_comma_separated():
    assert parse_values("0.7,0.3") == [0.7, 0.3]


def test_parse_values_whitespace_separated():
    assert parse_values("3 1 1") == [3.0, 1.0, 1.0]


def test_parse_values_empty_raises():
    with pytest.raises(ValueError, match="no numeric values"):
        parse_values("  ")


def test_parse_values_non_numeric_raises():
    with pytest.raises(ValueError, match="could not parse"):
        parse_values("0.5,abc")


@pytest.mark.parametrize(
    ("text", "expected"), [("2", 2.0), ("10", 10.0), ("e", math.e)]
)
def test_base_value_choices(text, expected):
    assert base_value(text) == pytest.approx(expected)


def test_base_value_unsupported_raises():
    with pytest.raises(ValueError, match="base must be one of"):
        base_value("7")


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------


def test_validate_entropy_valid():
    assert validate(_args()) is None


def test_validate_negative_precision():
    result = validate(_args(precision=-1))
    assert result is not None and "--precision" in result


def test_validate_entropy_missing_probs():
    result = validate(_args(probs=None))
    assert result is not None and "--probs is required" in result


def test_validate_kl_missing_vectors():
    result = validate(_args(measure="kl", probs_p="0.5,0.5"))
    assert result is not None and "--probs-p and --probs-q" in result


def test_validate_kl_valid():
    assert validate(_args(measure="kl", probs_p="0.7,0.3", probs_q="0.5,0.5")) is None


def test_validate_mi_missing_joint():
    result = validate(_args(measure="mi"))
    assert result is not None and "--joint is required" in result


def test_validate_conditional_valid():
    assert validate(_args(measure="conditional", joint=["0.5,0", "0,0.5"])) is None


# ---------------------------------------------------------------------------
# build_result
# ---------------------------------------------------------------------------


def test_build_result_entropy_keys():
    result = build_result(_args(probs="1,1,1,1"))
    assert result["outcomes"] == 4
    assert result["entropy"] == pytest.approx(2.0)
    assert result["efficiency"] == pytest.approx(1.0)


def test_build_result_entropy_single_outcome_efficiency_is_one():
    result = build_result(_args(probs="1"))
    assert result["max_entropy"] == pytest.approx(0.0)
    assert result["efficiency"] == pytest.approx(1.0)


def test_build_result_entropy_respects_base():
    result = build_result(_args(probs="0.5,0.5", base="e"))
    assert result["unit"] == "nats"
    assert result["entropy"] == pytest.approx(math.log(2))


def test_build_result_kl_keys():
    result = build_result(_args(measure="kl", probs_p="0.7,0.3", probs_q="0.5,0.5"))
    assert result["kl_divergence"] == pytest.approx(
        kl_divergence([0.7, 0.3], [0.5, 0.5])
    )
    assert result["kl_reverse"] != pytest.approx(result["kl_divergence"])


def test_build_result_cross_keys():
    result = build_result(_args(measure="cross", probs_p="1,0", probs_q="0.9,0.1"))
    assert result["cross_entropy"] == pytest.approx(-math.log2(0.9))


def test_build_result_reports_infinite_reverse_kl_as_none():
    """A zero in P makes D(Q||P) infinite; the requested measure still runs."""
    result = build_result(_args(measure="cross", probs_p="1,0", probs_q="0.9,0.1"))
    assert result["kl_reverse"] is None


def test_kl_or_none_returns_value_when_finite():
    assert entropy_module._kl_or_none([0.7, 0.3], [0.5, 0.5], 2.0) == pytest.approx(
        kl_divergence([0.7, 0.3], [0.5, 0.5])
    )


def test_kl_or_none_reraises_unrelated_value_error():
    with pytest.raises(ValueError, match="same length"):
        entropy_module._kl_or_none([0.5, 0.5], [1.0], 2.0)


def test_build_result_mi_keys():
    result = build_result(_args(measure="mi", joint=["0.25,0.25", "0.25,0.25"]))
    assert result["mutual_information"] == pytest.approx(0.0)
    assert result["joint_entropy"] == pytest.approx(2.0)


def test_build_result_conditional_both_directions():
    result = build_result(_args(measure="conditional", joint=["0.4,0.2", "0.1,0.3"]))
    assert result["conditional_entropy_y_given_x"] == pytest.approx(
        conditional_entropy([[0.4, 0.2], [0.1, 0.3]])
    )
    assert result["conditional_entropy_x_given_y"] == pytest.approx(
        conditional_entropy([[0.4, 0.1], [0.2, 0.3]])
    )


# ---------------------------------------------------------------------------
# format_table / format_json
# ---------------------------------------------------------------------------


def test_format_table_entropy():
    out = format_table(build_result(_args()), 4)
    assert "Shannon entropy" in out
    assert "Efficiency" in out


def test_format_table_kl():
    result = build_result(_args(measure="kl", probs_p="0.7,0.3", probs_q="0.5,0.5"))
    out = format_table(result, 4)
    assert "KL divergence" in out
    assert "asymmetric" in out


def test_format_table_shows_infinite_reverse_kl():
    result = build_result(_args(measure="kl", probs_p="1,0", probs_q="0.9,0.1"))
    assert "infinite" in format_table(result, 4)


def test_format_table_cross_uses_cross_title():
    result = build_result(_args(measure="cross", probs_p="0.7,0.3", probs_q="0.5,0.5"))
    out = format_table(result, 4)
    assert out.startswith("Cross-entropy")


def test_format_table_mi():
    result = build_result(_args(measure="mi", joint=["0.25,0.25", "0.25,0.25"]))
    out = format_table(result, 4)
    assert out.startswith("Mutual information")
    assert "I(X;Y)" in out


def test_format_table_conditional_uses_conditional_title():
    result = build_result(_args(measure="conditional", joint=["0.5,0", "0,0.5"]))
    assert format_table(result, 4).startswith("Conditional entropy")


def test_format_json_round_trips():
    data = json.loads(format_json(build_result(_args())))
    assert data["measure"] == "entropy"
    assert data["entropy"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def test_main_entropy_report(capsys):
    rc = main(["--probs", "0.167,0.167,0.167,0.167,0.167,0.167"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "2.5850" in out


def test_main_kl_report(capsys):
    rc = main(["--measure", "kl", "--probs-p", "0.7,0.3", "--probs-q", "0.5,0.5"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "0.1187" in out


def test_main_mi_report(capsys):
    rc = main(["--measure", "mi", "--joint", "0.25,0.25", "--joint", "0.25,0.25"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Mutual information" in out


def test_main_json_format(capsys):
    rc = main(["--probs", "0.5,0.5", "--format", "json"])
    out = capsys.readouterr().out
    assert rc == 0
    assert json.loads(out)["entropy"] == pytest.approx(1.0)


def test_main_missing_probs_returns_2(capsys):
    assert main([]) == 2
    assert "Error" in capsys.readouterr().err


def test_main_infinite_kl_returns_2(capsys):
    rc = main(["--measure", "kl", "--probs-p", "0.5,0.5", "--probs-q", "1,0"])
    assert rc == 2
    assert "infinite" in capsys.readouterr().err


def test_main_unparseable_values_returns_2(capsys):
    assert main(["--probs", "0.5,abc"]) == 2
    assert "could not parse" in capsys.readouterr().err


def test_main_computation_error_returns_2(monkeypatch, capsys):
    """Cover the ValueError branch in main when a core function raises."""

    def raise_value_error(*_args, **_kwargs):
        raise ValueError("forced entropy error")

    monkeypatch.setattr(entropy_module, "shannon_entropy", raise_value_error)
    assert main(["--probs", "0.5,0.5"]) == 2
    assert "forced entropy error" in capsys.readouterr().err
