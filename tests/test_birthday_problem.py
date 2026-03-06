"""Tests for birthday problem collision probability functions."""

import json
import math
import pytest

from src.utils.birthday_problem import (
    collision_prob_nonuniform,
    collision_prob_uniform,
    effective_pool_size,
    expected_duplicate_pairs,
    main,
    min_group_for_prob,
    prob_table,
)


# ---------------------------------------------------------------------------
# collision_prob_uniform
# ---------------------------------------------------------------------------

def test_collision_prob_zero_for_one_person():
    """A group of 1 cannot have a duplicate."""
    assert collision_prob_uniform(1, 365) == 0.0


def test_collision_prob_zero_for_zero_people():
    """A group of 0 cannot have a duplicate."""
    assert collision_prob_uniform(0, 365) == 0.0


def test_collision_prob_one_when_n_exceeds_d():
    """Pigeonhole: n > d guarantees a duplicate."""
    assert collision_prob_uniform(366, 365) == 1.0


def test_collision_prob_one_when_n_equals_d_plus_one():
    """n = d + 1 must produce probability 1.0 (pigeonhole)."""
    assert collision_prob_uniform(4, 3) == 1.0


def test_collision_prob_classic_23_in_365():
    """Classic result: ~50.7% probability for 23 people and 365 days."""
    prob = collision_prob_uniform(23, 365)
    assert abs(prob - 0.5072972) < 1e-5


def test_collision_prob_increases_with_group_size():
    """Probability is strictly increasing in n."""
    probs = [collision_prob_uniform(n, 365) for n in range(2, 30)]
    assert all(probs[i] < probs[i + 1] for i in range(len(probs) - 1))


def test_collision_prob_small_pool():
    """For d=2, P(collision) with n=2 is 0.5."""
    prob = collision_prob_uniform(2, 2)
    assert abs(prob - 0.5) < 1e-12


def test_collision_prob_large_pool_low_group():
    """Very large pool with tiny group should give near-zero probability."""
    prob = collision_prob_uniform(2, 1_000_000)
    assert prob < 1e-4


# ---------------------------------------------------------------------------
# collision_prob_nonuniform
# ---------------------------------------------------------------------------

def test_nonuniform_one_person_zero():
    """One person cannot produce a duplicate."""
    assert collision_prob_nonuniform(1, [0.5, 0.3, 0.2]) == 0.0


def test_nonuniform_uniform_weights_close_to_uniform_formula():
    """Equal weights should closely match the Poisson approximation of uniform."""
    weights = [1.0] * 365
    prob_nu = collision_prob_nonuniform(23, weights)
    # Poisson approx: 1 - exp(-C(23,2)/365)
    expected_approx = -math.expm1(-23 * 22 / 2 / 365)
    assert abs(prob_nu - expected_approx) < 1e-6


def test_nonuniform_skewed_distribution_higher_collision():
    """Concentrating weight in fewer categories raises collision probability."""
    # concentrated: most mass on 2 categories out of 365
    uniform_weights = [1.0] * 365
    skewed_weights = [180.0, 180.0] + [0.1] * 363
    prob_uniform_nu = collision_prob_nonuniform(23, uniform_weights)
    prob_skewed_nu = collision_prob_nonuniform(23, skewed_weights)
    assert prob_skewed_nu > prob_uniform_nu


def test_nonuniform_all_same_weight_equals_one_category():
    """Weights that are all equal to each other should behave consistently."""
    weights = [2.0] * 10
    prob = collision_prob_nonuniform(5, weights)
    # sum(p_i^2) = 10 * (0.1)^2 = 0.1
    expected = -math.expm1(-5 * 4 / 2 * 0.1)
    assert abs(prob - expected) < 1e-12


def test_nonuniform_normalises_weights():
    """Scaling all weights by a constant should not change the result."""
    weights_a = [1.0, 2.0, 3.0]
    weights_b = [10.0, 20.0, 30.0]
    assert abs(
        collision_prob_nonuniform(5, weights_a) - collision_prob_nonuniform(5, weights_b)
    ) < 1e-12


def test_nonuniform_raises_on_all_zero_weights():
    """All-zero weights should raise ValueError."""
    with pytest.raises(ValueError):
        collision_prob_nonuniform(5, [0.0, 0.0])


# ---------------------------------------------------------------------------
# effective_pool_size
# ---------------------------------------------------------------------------

def test_effective_pool_size_uniform():
    """For d equal weights the effective size equals d."""
    d = 50
    weights = [1.0] * d
    assert abs(effective_pool_size(weights) - d) < 1e-10


def test_effective_pool_size_one_category():
    """A single category has effective pool size of 1."""
    assert abs(effective_pool_size([1.0]) - 1.0) < 1e-12


def test_effective_pool_size_skewed_less_than_d():
    """Skewed distribution reduces effective pool size below d."""
    d = 10
    uniform = [1.0] * d
    skewed = [5.0] + [1.0] * (d - 1)
    assert effective_pool_size(skewed) < effective_pool_size(uniform)


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

def test_min_group_classic_50pct():
    """Minimum group for 50% collision probability on 365-day calendar is 23."""
    assert min_group_for_prob(0.50, 365) == 23


def test_min_group_target_zero_returns_one():
    assert min_group_for_prob(0.0, 365) == 1


def test_min_group_target_one_returns_d_plus_one():
    assert min_group_for_prob(1.0, 365) == 366


def test_min_group_result_satisfies_target():
    """The returned n must actually reach the target probability."""
    target = 0.75
    d = 365
    n = min_group_for_prob(target, d)
    assert n is not None
    assert collision_prob_uniform(n, d) >= target
    if n > 2:
        assert collision_prob_uniform(n - 1, d) < target


# ---------------------------------------------------------------------------
# prob_table
# ---------------------------------------------------------------------------

def test_prob_table_length():
    rows = prob_table(365, 1, 30)
    assert len(rows) == 30


def test_prob_table_n_values():
    rows = prob_table(365, 5, 10)
    assert [r["n"] for r in rows] == list(range(5, 11))


def test_prob_table_probability_monotone():
    rows = prob_table(365, 1, 40)
    probs = [r["probability"] for r in rows]
    assert all(probs[i] <= probs[i + 1] for i in range(len(probs) - 1))


# ---------------------------------------------------------------------------
# main() CLI
# ---------------------------------------------------------------------------

def test_main_single_group_returns_zero():
    assert main(["--pool-size", "365", "--group-size", "23"]) == 0


def test_main_target_prob_returns_zero():
    assert main(["--pool-size", "365", "--target-prob", "0.5"]) == 0


def test_main_range_table_returns_zero():
    assert main(["--pool-size", "365", "--range", "1", "30"]) == 0


def test_main_range_json_returns_zero():
    assert main(["--pool-size", "365", "--range", "1", "10", "--format", "json"]) == 0


def test_main_range_csv_returns_zero():
    assert main(["--pool-size", "365", "--range", "1", "10", "--format", "csv"]) == 0


def test_main_pool_size_zero_returns_two():
    assert main(["--pool-size", "0", "--group-size", "10"]) == 2


def test_main_invalid_target_prob_returns_two():
    assert main(["--pool-size", "365", "--target-prob", "1.5"]) == 2


def test_main_no_mode_returns_two():
    assert main(["--pool-size", "365"]) == 2


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
