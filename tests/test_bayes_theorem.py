"""Tests for Bayes theorem functions and CLI."""

from unittest.mock import patch

import pytest

from src.utils.bayes_theorem import (
    bayes_posterior,
    main,
)


def test_bayes_posterior_raises_on_non_positive_evidence():
    with pytest.raises(ValueError):
        bayes_posterior(0.1, 0.8, 0.0)


# def test_evidence_from_false_positive_known_case():
#     evidence = evidence_from_false_positive(0.01, 0.99, 0.05)
#     assert abs(evidence - 0.0594) < 1e-12


def test_main_with_false_positive_mode(capsys):
    rc = main(["-p", "0.01", "-l", "0.99", "-f", "0.05"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Posterior P(A|B):" in out
    assert "16.666" in out


def test_main_with_evidence_mode(capsys):
    rc = main(["-p", "0.2", "-l", "0.8", "-e", "0.5"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Evidence P(B):" in out
    assert "Posterior P(A|B):" in out


def test_main_missing_evidence_source_returns_2(capsys):
    rc = main(["-p", "0.2", "-l", "0.8"])
    assert rc == 2


def test_main_invalid_prior_returns_2(capsys):
    rc = main(["-p", "1.2", "-l", "0.8", "-e", "0.5"])
    assert rc == 2


def test_main_invalid_likelihood_returns_2(capsys):
    rc = main(["-p", "0.2", "-l", "-0.1", "-e", "0.5"])
    assert rc == 2


def test_main_invalid_false_positive_returns_2(capsys):
    rc = main(["-p", "0.2", "-l", "0.8", "-f", "1.1"])
    assert rc == 2


def test_main_zero_evidence_returns_2(capsys):
    rc = main(["-p", "0.2", "-l", "0.8", "-e", "0"])
    assert rc == 2


def test_main_negative_precision_returns_2(capsys):
    """Covers validate() precision guard (line 114)."""
    rc = main(["-p", "0.2", "-l", "0.8", "-e", "0.5", "-P", "-1"])
    assert rc == 2


def test_main_non_finite_derived_evidence_returns_2(capsys):
    """Covers main() finite-evidence guard (lines 169-170)."""
    with patch(
        "src.utils.bayes_theorem.evidence_from_false_positive",
        return_value=float("inf"),
    ):
        rc = main(["-p", "0.2", "-l", "0.8", "-f", "0.1"])
    assert rc == 2
    assert "derived evidence" in capsys.readouterr().err
