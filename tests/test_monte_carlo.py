import argparse
import json

from src.utils.monte_carlo import (
    main,
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
