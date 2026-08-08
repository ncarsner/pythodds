import argparse
import json
import math
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.utils.monte_carlo import (
    _is_extreme,
    _t_pvalue,
    _t_stat_one_sample,
    _t_stat_welch,
    analytical_value,
    main,
    run_experiment,
    simulate_bayes,
    standard_error,
    trials_for_scale,
    validate,
)

# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------


def test_standard_error_zero_n():
    assert standard_error(0.5, 0) == 0.0


def test_trials_for_scale():
    assert trials_for_scale(0.05) == 100  # ceil(0.25 / 0.05**2)


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------


def _ns(**kwargs):
    defaults = {
        "experiment": "binomial",
        "params": ["n=10", "k=5", "p=0.4"],
        "trials": 1000,
        "scale": None,
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def test_validate_invalid_kv_format():
    assert validate(_ns(params=["n10"])) is not None


def test_validate_binomial_bad_n():
    assert validate(_ns(params=["n=0", "k=5", "p=0.4"])) == "param n must be >= 1"


def test_validate_binomial_bad_k():
    assert validate(_ns(params=["n=10", "k=-1", "p=0.4"])) == "param k must be >= 0"


def test_validate_binomial_bad_p():
    assert (
        validate(_ns(params=["n=10", "k=5", "p=2.0"]))
        == "param p must be between 0 and 1"
    )


def test_validate_birthday_bad_pool():
    assert (
        validate(_ns(experiment="birthday", params=["pool=0", "group=23"]))
        == "param pool must be >= 1"
    )


def test_validate_birthday_bad_group():
    assert (
        validate(_ns(experiment="birthday", params=["pool=365", "group=0"]))
        == "param group must be >= 1"
    )


def test_validate_poisson_bad_lam():
    assert (
        validate(_ns(experiment="poisson", params=["lam=0.0", "k=3"]))
        == "param lam must be > 0"
    )


def test_validate_poisson_bad_k():
    assert (
        validate(_ns(experiment="poisson", params=["lam=3.0", "k=-1"]))
        == "param k must be >= 0"
    )


def test_validate_bad_param_value():
    error = validate(_ns(params=["n=abc", "k=5", "p=0.4"]))
    assert error is not None
    assert "invalid param value" in error


def test_validate_trials_too_small():
    assert validate(_ns(trials=0)) == "--trials must be >= 1"


def test_validate_scale_not_positive():
    assert validate(_ns(scale=-0.1)) == "--scale must be > 0"


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def test_main_returns_2_on_invalid(capsys):
    rc = main(["--experiment", "binomial", "--params", "n=10", "k=5"])
    assert rc == 2
    assert "Error" in capsys.readouterr().err


def test_main_binomial_json_confidence(capsys):
    rc = main(
        [
            "--experiment",
            "binomial",
            "--params",
            "n=10",
            "k=5",
            "p=0.4",
            "--trials",
            "500",
            "--seed",
            "42",
            "--confidence",
            "--format",
            "json",
        ]
    )
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert "ci_lower" in data
    assert "analytical_value" in data


def test_main_streak_table(capsys):
    rc = main(
        [
            "--experiment",
            "streak",
            "--params",
            "n=10",
            "k=2",
            "p=0.5",
            "--trials",
            "200",
            "--seed",
            "42",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "Estimated probability" in out
    assert "Analytical" not in out


def test_main_birthday_table_confidence(capsys):
    rc = main(
        [
            "--experiment",
            "birthday",
            "--params",
            "pool=365",
            "group=23",
            "--trials",
            "500",
            "--seed",
            "42",
            "--confidence",
        ]
    )
    assert rc == 0
    assert "Analytical value" in capsys.readouterr().out


def test_main_poisson_json(capsys):
    rc = main(
        [
            "--experiment",
            "poisson",
            "--params",
            "lam=3.0",
            "k=5",
            "--trials",
            "500",
            "--seed",
            "42",
            "--format",
            "json",
        ]
    )
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert "analytical_value" in data
    assert "ci_lower" not in data


def test_main_dump(capsys):
    rc = main(
        [
            "--experiment",
            "binomial",
            "--params",
            "n=10",
            "k=5",
            "p=0.4",
            "--trials",
            "5",
            "--seed",
            "42",
            "--dump",
        ]
    )
    assert rc == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[0] == "trial,outcome"
    assert len(lines) == 6  # header + 5 trials


# ---------------------------------------------------------------------------
# validate — power
# ---------------------------------------------------------------------------


def test_validate_power_bad_type():
    assert validate(_ns(experiment="power", params=["type=bad"])) is not None


def test_validate_power_mean_missing_n():
    assert (
        validate(_ns(experiment="power", params=["type=mean", "sigma=10", "delta=5"]))
        is not None
    )


def test_validate_power_mean_n_too_small():
    assert (
        validate(
            _ns(experiment="power", params=["type=mean", "n=1", "sigma=10", "delta=5"])
        )
        == "power param n must be >= 2"
    )


def test_validate_power_mean_sigma_zero():
    assert (
        validate(
            _ns(experiment="power", params=["type=mean", "n=30", "sigma=0", "delta=5"])
        )
        == "power param sigma must be > 0"
    )


def test_validate_power_mean_delta_zero():
    assert (
        validate(
            _ns(experiment="power", params=["type=mean", "n=30", "sigma=10", "delta=0"])
        )
        == "power param delta must be nonzero"
    )


def test_validate_power_bad_alpha():
    assert (
        validate(
            _ns(
                experiment="power",
                params=["type=mean", "n=30", "sigma=10", "delta=5", "alpha=1.5"],
            )
        )
        == "power param alpha must be in (0, 1)"
    )


def test_validate_power_comparison_p1_equals_p2():
    assert (
        validate(
            _ns(
                experiment="power",
                params=["type=comparison", "n=30", "p1=0.5", "p2=0.5"],
            )
        )
        == "power params p1 and p2 must differ"
    )


def test_validate_power_comparison_p1_out_of_range():
    assert (
        validate(
            _ns(
                experiment="power",
                params=["type=comparison", "n=30", "p1=1.5", "p2=0.5"],
            )
        )
        == "power param p1 must be in (0, 1)"
    )


# ---------------------------------------------------------------------------
# validate — permutation
# ---------------------------------------------------------------------------


def test_validate_permutation_bad_type():
    assert validate(_ns(experiment="permutation", params=["type=bad"])) is not None


def test_validate_permutation_one_missing_mu0():
    assert (
        validate(
            _ns(
                experiment="permutation",
                params=["type=one", "values=2.1,3.4,2.9"],
            )
        )
        is not None
    )


def test_validate_permutation_one_too_few_values():
    assert (
        validate(
            _ns(
                experiment="permutation",
                params=["type=one", "values=2.1,3.4", "mu0=3.0"],
            )
        )
        == "permutation type=one requires at least 3 values"
    )


def test_validate_permutation_paired_unequal_lengths():
    assert (
        validate(
            _ns(
                experiment="permutation",
                params=["type=paired", "values1=1,2,3", "values2=4,5"],
            )
        )
        == "permutation type=paired requires equal-length values1 and values2"
    )


def test_validate_permutation_two_too_few_per_group():
    assert (
        validate(
            _ns(
                experiment="permutation",
                params=["type=two", "values1=1", "values2=2,3"],
            )
        )
        == "permutation requires at least 2 values per group"
    )


# ---------------------------------------------------------------------------
# validate — bayes
# ---------------------------------------------------------------------------


def test_validate_bayes_prior_out_of_range():
    assert (
        validate(
            _ns(
                experiment="bayes",
                params=["prior=1.5", "likelihood=0.9", "fp=0.1"],
            )
        )
        == "bayes param prior must be in [0, 1]"
    )


def test_validate_bayes_likelihood_out_of_range():
    assert (
        validate(
            _ns(
                experiment="bayes",
                params=["prior=0.01", "likelihood=1.5", "fp=0.1"],
            )
        )
        == "bayes param likelihood must be in [0, 1]"
    )


def test_validate_bayes_fp_out_of_range():
    assert (
        validate(
            _ns(
                experiment="bayes",
                params=["prior=0.01", "likelihood=0.9", "fp=-0.1"],
            )
        )
        == "bayes param fp must be in [0, 1]"
    )


def test_validate_bayes_zero_evidence():
    assert (
        validate(
            _ns(
                experiment="bayes",
                params=["prior=0.0", "likelihood=0.99", "fp=0.0"],
            )
        )
        == "bayes evidence P(B) = 0: choose nonzero likelihood or fp"
    )


# ---------------------------------------------------------------------------
# validate — season
# ---------------------------------------------------------------------------


def test_validate_season_bad_win_pct():
    assert (
        validate(_ns(experiment="season", params=["win_pct=1.5"]))
        == "season param win_pct must be in (0, 1)"
    )


def test_validate_season_bad_games():
    assert (
        validate(_ns(experiment="season", params=["win_pct=0.5", "games=0"]))
        == "season param games must be >= 1"
    )


def test_validate_season_wins_ge_out_of_range():
    assert (
        validate(
            _ns(experiment="season", params=["win_pct=0.5", "games=10", "wins_ge=15"])
        )
        == "season param wins_ge must be in [0, 10]"
    )


# ---------------------------------------------------------------------------
# validate — linboot
# ---------------------------------------------------------------------------


def test_validate_linboot_unequal_lengths():
    assert (
        validate(_ns(experiment="linboot", params=["x=1,2,3", "y=1,2"]))
        == "linboot params x and y must have equal length"
    )


def test_validate_linboot_too_few_points():
    assert (
        validate(_ns(experiment="linboot", params=["x=1,2", "y=1,2"]))
        == "linboot requires at least 3 data points"
    )


def test_validate_linboot_zero_variance():
    assert (
        validate(_ns(experiment="linboot", params=["x=1,1,1", "y=2,3,4"]))
        == "linboot param x has zero variance (all x values identical)"
    )


# ---------------------------------------------------------------------------
# main — power
# ---------------------------------------------------------------------------


def test_main_power_mean_json_analytical(capsys):
    rc = main(
        [
            "--experiment",
            "power",
            "--params",
            "type=mean",
            "n=30",
            "sigma=10",
            "delta=5",
            "--trials",
            "500",
            "--seed",
            "42",
            "--format",
            "json",
        ]
    )
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert "analytical_value" in data
    assert 0.0 < data["analytical_value"] < 1.0


def test_main_power_comparison_json(capsys):
    rc = main(
        [
            "--experiment",
            "power",
            "--params",
            "type=comparison",
            "n=200",
            "p1=0.5",
            "p2=0.6",
            "--trials",
            "500",
            "--seed",
            "42",
            "--format",
            "json",
        ]
    )
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert "estimated_probability" in data
    assert "analytical_value" in data


# ---------------------------------------------------------------------------
# main — permutation
# ---------------------------------------------------------------------------


def test_main_permutation_one_table(capsys):
    rc = main(
        [
            "--experiment",
            "permutation",
            "--params",
            "type=one",
            "values=2.1,3.4,2.9,3.1,2.8",
            "mu0=3.0",
            "--trials",
            "500",
            "--seed",
            "42",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "Empirical p-value" in out
    assert "Parametric p-value" in out


def test_main_permutation_two_table(capsys):
    rc = main(
        [
            "--experiment",
            "permutation",
            "--params",
            "type=two",
            "values1=2.1,3.4,2.9,3.1,2.8",
            "values2=4.1,3.8,4.5,4.2,3.9",
            "--trials",
            "500",
            "--seed",
            "42",
        ]
    )
    assert rc == 0
    assert "Empirical p-value" in capsys.readouterr().out


def test_main_permutation_paired_table(capsys):
    rc = main(
        [
            "--experiment",
            "permutation",
            "--params",
            "type=paired",
            "values1=2.1,3.4,2.9,3.1,2.8",
            "values2=2.5,3.7,3.2,3.5,3.1",
            "--trials",
            "500",
            "--seed",
            "42",
        ]
    )
    assert rc == 0
    assert "Empirical p-value" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# main — bayes
# ---------------------------------------------------------------------------


def test_main_bayes_json(capsys):
    rc = main(
        [
            "--experiment",
            "bayes",
            "--params",
            "prior=0.01",
            "likelihood=0.99",
            "fp=0.05",
            "--trials",
            "500",
            "--seed",
            "42",
            "--format",
            "json",
        ]
    )
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert "analytical_value" in data
    assert 0.0 < data["analytical_value"] < 1.0


# ---------------------------------------------------------------------------
# main — season
# ---------------------------------------------------------------------------


def test_main_season_distribution_table(capsys):
    rc = main(
        [
            "--experiment",
            "season",
            "--params",
            "win_pct=0.58",
            "games=162",
            "--trials",
            "500",
            "--seed",
            "42",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "Mean wins" in out
    assert "Percentiles" in out


def test_main_season_distribution_json(capsys):
    rc = main(
        [
            "--experiment",
            "season",
            "--params",
            "win_pct=0.58",
            "games=162",
            "--trials",
            "200",
            "--seed",
            "42",
            "--format",
            "json",
        ]
    )
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert "mean_wins" in data
    assert "p50" in data


def test_main_season_wins_ge_json(capsys):
    rc = main(
        [
            "--experiment",
            "season",
            "--params",
            "win_pct=0.58",
            "games=162",
            "wins_ge=90",
            "--trials",
            "500",
            "--seed",
            "42",
            "--format",
            "json",
        ]
    )
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert "analytical_value" in data


def test_main_season_distribution_dump(capsys):
    rc = main(
        [
            "--experiment",
            "season",
            "--params",
            "win_pct=0.5",
            "games=10",
            "--trials",
            "5",
            "--seed",
            "42",
            "--dump",
        ]
    )
    assert rc == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[0] == "trial,wins"
    assert len(lines) == 6  # header + 5 trials


# ---------------------------------------------------------------------------
# main — linboot
# ---------------------------------------------------------------------------


def test_main_linboot_table(capsys):
    rc = main(
        [
            "--experiment",
            "linboot",
            "--params",
            "x=1,2,3,4,5",
            "y=2.1,3.9,6.2,7.8,10.1",
            "predict=6",
            "--trials",
            "200",
            "--seed",
            "42",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "Bootstrap Linear Regression" in out
    assert "Slope" in out
    assert "Intercept" in out
    assert "Prediction at x=6" in out


def test_main_linboot_json(capsys):
    rc = main(
        [
            "--experiment",
            "linboot",
            "--params",
            "x=1,2,3,4,5",
            "y=2.1,3.9,6.2,7.8,10.1",
            "predict=6",
            "--trials",
            "200",
            "--seed",
            "42",
            "--format",
            "json",
        ]
    )
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert "slope" in data
    assert "intercept" in data
    assert "prediction" in data
    assert "ci_lower" in data["slope"]


def test_main_linboot_no_predict(capsys):
    rc = main(
        [
            "--experiment",
            "linboot",
            "--params",
            "x=1,2,3,4,5",
            "y=2.1,3.9,6.2,7.8,10.1",
            "--trials",
            "200",
            "--seed",
            "42",
            "--format",
            "json",
        ]
    )
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert "slope" in data
    assert "prediction" not in data


def test_main_linboot_dump(capsys):
    rc = main(
        [
            "--experiment",
            "linboot",
            "--params",
            "x=1,2,3,4,5",
            "y=2.1,3.9,6.2,7.8,10.1",
            "predict=6",
            "--trials",
            "5",
            "--seed",
            "42",
            "--dump",
        ]
    )
    assert rc == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[0] == "trial,slope,intercept,prediction"
    assert len(lines) == 6  # header + 5 trials


# ---------------------------------------------------------------------------
# Private helpers — edge cases
# ---------------------------------------------------------------------------


def test_t_stat_one_sample_single_value():
    # n < 2 → 0.0
    assert _t_stat_one_sample([5.0], 3.0) == 0.0


def test_t_stat_one_sample_zero_se_away_from_mu0():
    # all values equal but != mu0, se == 0, mean != mu0 → ±inf
    result = _t_stat_one_sample([5.0, 5.0, 5.0], 3.0)
    assert math.isinf(result) and result > 0


def test_t_stat_welch_single_group():
    # n1 < 2 → 0.0
    assert _t_stat_welch([1.0], [2.0, 3.0]) == 0.0


def test_t_stat_welch_zero_se():
    # identical values in both groups → se == 0 → 0.0
    assert _t_stat_welch([2.0, 2.0, 2.0], [2.0, 2.0, 2.0]) == 0.0


def test_is_extreme_less_is_extreme():
    assert _is_extreme(0.5, 1.0, "less") == 1  # 0.5 <= 1.0


def test_is_extreme_greater_is_extreme():
    assert _is_extreme(1.5, 1.0, "greater") == 1  # 1.5 >= 1.0


def test_t_pvalue_less_greater_complement():
    # P(T <= t) + P(T > t) == 1
    p_less = _t_pvalue(1.5, 10, "less")
    p_greater = _t_pvalue(1.5, 10, "greater")
    assert abs(p_less + p_greater - 1.0) < 1e-12


# ---------------------------------------------------------------------------
# analytical_value — exception handlers
# ---------------------------------------------------------------------------


def test_analytical_value_power_invalid_n():
    # int("abc") raises ValueError → caught, returns None
    result = analytical_value(
        "power", {"type": "mean", "n": "abc", "sigma": "10", "delta": "5"}
    )
    assert result is None


def test_analytical_value_permutation_invalid_values():
    # float("bad") raises ValueError → caught, returns None
    result = analytical_value(
        "permutation", {"type": "one", "values": "bad", "mu0": "3.0"}
    )
    assert result is None


# ---------------------------------------------------------------------------
# validate — power comparison missing branches
# ---------------------------------------------------------------------------


def test_validate_power_comparison_missing_n():
    assert (
        validate(
            _ns(experiment="power", params=["type=comparison", "p1=0.5", "p2=0.6"])
        )
        == "power type=comparison requires param: n"
    )


def test_validate_power_comparison_n_too_small():
    assert (
        validate(
            _ns(
                experiment="power",
                params=["type=comparison", "n=1", "p1=0.5", "p2=0.6"],
            )
        )
        == "power param n must be >= 2"
    )


def test_validate_power_comparison_p2_out_of_range():
    assert (
        validate(
            _ns(
                experiment="power",
                params=["type=comparison", "n=30", "p1=0.5", "p2=1.5"],
            )
        )
        == "power param p2 must be in (0, 1)"
    )


# ---------------------------------------------------------------------------
# validate — permutation missing-param branches
# ---------------------------------------------------------------------------


def test_validate_permutation_one_missing_values():
    assert (
        validate(_ns(experiment="permutation", params=["type=one", "mu0=3.0"]))
        == "permutation type=one requires param: values"
    )


def test_validate_permutation_two_missing_values2():
    assert (
        validate(_ns(experiment="permutation", params=["type=two", "values1=1,2,3"]))
        == "permutation type=two requires param: values2"
    )


# ---------------------------------------------------------------------------
# main — power one-sided branches
# ---------------------------------------------------------------------------


def test_main_power_mean_one_sided(capsys):
    rc = main(
        [
            "--experiment",
            "power",
            "--params",
            "type=mean",
            "n=30",
            "sigma=10",
            "delta=5",
            "sided=one",
            "--trials",
            "300",
            "--seed",
            "42",
        ]
    )
    assert rc == 0
    assert "Estimated power" in capsys.readouterr().out


def test_main_power_comparison_one_sided_p1_gt_p2(capsys):
    # p1 >= p2 branch inside simulate_power comparison
    rc = main(
        [
            "--experiment",
            "power",
            "--params",
            "type=comparison",
            "n=100",
            "p1=0.6",
            "p2=0.5",
            "sided=one",
            "--trials",
            "300",
            "--seed",
            "42",
            "--format",
            "json",
        ]
    )
    assert rc == 0
    assert "estimated_probability" in json.loads(capsys.readouterr().out)


def test_main_power_comparison_one_sided_p1_lt_p2(capsys):
    # p1 < p2 branch inside simulate_power comparison
    rc = main(
        [
            "--experiment",
            "power",
            "--params",
            "type=comparison",
            "n=100",
            "p1=0.4",
            "p2=0.5",
            "sided=one",
            "--trials",
            "300",
            "--seed",
            "42",
            "--format",
            "json",
        ]
    )
    assert rc == 0
    assert "estimated_probability" in json.loads(capsys.readouterr().out)


# ---------------------------------------------------------------------------
# main — linboot dump without predict (no-predictions CSV branch)
# ---------------------------------------------------------------------------


def test_main_linboot_dump_no_predict(capsys):
    rc = main(
        [
            "--experiment",
            "linboot",
            "--params",
            "x=1,2,3,4,5",
            "y=2.1,3.9,6.2,7.8,10.1",
            "--trials",
            "5",
            "--seed",
            "42",
            "--dump",
        ]
    )
    assert rc == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[0] == "trial,slope,intercept"
    assert len(lines) == 6  # header + 5 trials


# ---------------------------------------------------------------------------
# Mocked paths
# ---------------------------------------------------------------------------


def test_simulate_bayes_padding_path():
    # Force the padding branch: mock the batch rng to return all-ones so
    # no B=1 events appear, then let the padding rng use real randomness.
    mock_batch_rng = MagicMock()
    mock_batch_rng.random.return_value = np.ones(100)

    real_pad_rng = np.random.default_rng(1)

    with patch("numpy.random.default_rng", side_effect=[mock_batch_rng, real_pad_rng]):
        result = simulate_bayes(
            {"prior": "0.5", "likelihood": "0.9", "fp": "0.1"},
            trials=5,
            seed=0,
        )

    assert len(result) == 5
    assert all(v in (0, 1) for v in result)


def test_main_linboot_runtime_error(capsys, monkeypatch):
    import src.utils.monte_carlo as mc

    monkeypatch.setattr(mc, "_linear_regression", None)
    rc = main(
        [
            "--experiment",
            "linboot",
            "--params",
            "x=1,2,3",
            "y=1,2,3",
            "--trials",
            "5",
        ]
    )
    assert rc == 2
    assert "Error" in capsys.readouterr().err


def test_main_linboot_no_resamples(capsys, monkeypatch):
    import src.utils.monte_carlo as mc
    from src.utils.linear_regression import linear_regression as real_lr

    call_count = [0]

    def mock_lr(x, y):
        call_count[0] += 1
        if call_count[0] == 1:
            return real_lr(x, y)
        raise ValueError("always fail")

    monkeypatch.setattr(mc, "_linear_regression", mock_lr)
    rc = main(
        [
            "--experiment",
            "linboot",
            "--params",
            "x=1,2,3,4,5",
            "y=2.1,3.9,6.2,7.8,10.1",
            "--trials",
            "5",
            "--seed",
            "42",
        ]
    )
    assert rc == 2
    assert "no successful bootstrap resamples" in capsys.readouterr().err


def test_run_experiment_unknown_name_raises():
    """--experiment is gated by argparse choices, but the function is public."""
    with pytest.raises(ValueError, match="Unknown experiment"):
        run_experiment("bogus", {}, trials=5, seed=0)
