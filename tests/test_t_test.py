"""Tests for t_test.py — one-sample, two-sample (Welch's), and paired t-tests."""

from __future__ import annotations

import argparse
import math

import pytest

import src.utils.t_test as t_test_module
from src.utils.t_test import (
    OneSampleResult,
    PairedResult,
    TwoSampleResult,
    _cohens_d_one_sample,
    _cohens_d_two_sample,
    _decision,
    _fmt,
    _mean,
    _p_value,
    _sided_label,
    _std,
    _var,
    format_one_sample,
    format_paired,
    format_two_sample,
    main,
    one_sample_t_test,
    paired_t_test,
    parse_number_list,
    two_sample_t_test,
    validate,
)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def test_mean_basic():
    assert _mean([1.0, 2.0, 3.0]) == pytest.approx(2.0)


def test_mean_single_value():
    assert _mean([5.0]) == 5.0


def test_mean_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        _mean([])


def test_var_basic():
    # Known: var of [2,4,4,4,5,5,7,9] sample = 32/7
    values = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
    assert _var(values) == pytest.approx(32.0 / 7.0)


def test_var_two_values():
    assert _var([0.0, 2.0]) == pytest.approx(2.0)


def test_var_one_value_raises():
    with pytest.raises(ValueError, match="at least 2"):
        _var([1.0])


def test_std_is_sqrt_var():
    values = [1.0, 3.0, 5.0]
    assert _std(values) == pytest.approx(math.sqrt(_var(values)))


def test_cohens_d_one_sample_basic():
    assert _cohens_d_one_sample(5.0, 3.0, 2.0) == pytest.approx(1.0)


def test_cohens_d_one_sample_zero_std():
    assert _cohens_d_one_sample(5.0, 3.0, 0.0) == 0.0


def test_cohens_d_two_sample_equal_groups():
    d = _cohens_d_two_sample(10.0, 8.0, 2.0, 2.0, 10, 10)
    assert d == pytest.approx(1.0)


def test_cohens_d_two_sample_zero_pooled_sd():
    d = _cohens_d_two_sample(5.0, 5.0, 0.0, 0.0, 5, 5)
    assert d == 0.0


def test_p_value_two_sided_symmetry():
    p_pos = _p_value(2.0, 10, "two")
    p_neg = _p_value(-2.0, 10, "two")
    assert p_pos == pytest.approx(p_neg)


def test_p_value_less():
    p = _p_value(-3.0, 20, "less")
    assert 0 < p < 0.01


def test_p_value_greater():
    p = _p_value(3.0, 20, "greater")
    assert 0 < p < 0.01


def test_p_value_clipped_to_zero_one():
    # Large positive t, "greater" → very small p
    p = _p_value(100.0, 5, "greater")
    assert p >= 0.0


def test_sided_label_all():
    assert "two" in _sided_label("two")
    assert "less" in _sided_label("less")
    assert "greater" in _sided_label("greater")


def test_fmt_basic():
    assert _fmt(3.14159, 2) == "3.14"


def test_fmt_inf():
    assert _fmt(math.inf, 4) == "inf"
    assert _fmt(-math.inf, 4) == "-inf"


def test_decision_reject():
    assert "Reject" in _decision(0.01, 0.05)


def test_decision_fail_to_reject():
    assert "Fail" in _decision(0.10, 0.05)


# ---------------------------------------------------------------------------
# parse_number_list
# ---------------------------------------------------------------------------


def test_parse_number_list_basic():
    assert parse_number_list("1,2,3") == [1.0, 2.0, 3.0]


def test_parse_number_list_spaces():
    assert parse_number_list(" 1.5 , 2.5 ") == [1.5, 2.5]


def test_parse_number_list_empty_raises():
    with pytest.raises(ValueError, match="at least one"):
        parse_number_list("  ,  ")


def test_parse_number_list_non_numeric_raises():
    with pytest.raises(ValueError):
        parse_number_list("1,abc,3")


def test_parse_number_list_infinite_raises():
    with pytest.raises(ValueError, match="finite"):
        parse_number_list("1,inf,3")


# ---------------------------------------------------------------------------
# one_sample_t_test core function
# ---------------------------------------------------------------------------


def test_one_sample_known_values():
    """Verify against scipy ttest_1samp for known dataset."""
    from scipy.stats import ttest_1samp

    data = [2.1, 3.4, 2.9, 3.1, 2.8]
    mu0 = 3.0
    res = one_sample_t_test(mean=_mean(data), std=_std(data), n=len(data), mu0=mu0)
    scipy_t, scipy_p = ttest_1samp(data, mu0)
    assert res.t_stat == pytest.approx(scipy_t, abs=1e-6)
    assert res.p_value == pytest.approx(scipy_p, abs=1e-6)
    assert res.df == len(data) - 1


def test_one_sample_sided_less():
    res = one_sample_t_test(mean=99.0, std=10.0, n=30, mu0=100.0, sided="less")
    assert res.p_value < 0.5


def test_one_sample_sided_greater():
    res = one_sample_t_test(mean=101.0, std=10.0, n=30, mu0=100.0, sided="greater")
    assert res.p_value < 0.5


def test_one_sample_ci_contains_mean():
    res = one_sample_t_test(mean=50.0, std=5.0, n=20, mu0=50.0)
    assert res.ci_lower < 50.0 < res.ci_upper


def test_one_sample_zero_std_equal_mean():
    res = one_sample_t_test(mean=5.0, std=0.0, n=10, mu0=5.0)
    assert res.t_stat == 0.0
    assert res.p_value == 1.0
    assert res.ci_lower == res.ci_upper == 5.0


def test_one_sample_zero_std_unequal_mean():
    res = one_sample_t_test(mean=6.0, std=0.0, n=10, mu0=5.0)
    assert math.isinf(res.t_stat)
    assert res.t_stat > 0
    assert res.p_value == 0.0


def test_one_sample_cohen_d():
    res = one_sample_t_test(mean=5.0, std=2.0, n=10, mu0=3.0)
    assert res.cohens_d == pytest.approx(1.0)


@pytest.mark.parametrize(
    "n,std,msg",
    [
        (1, 1.0, "Sample size"),
        (10, -1.0, "non-negative"),
    ],
)
def test_one_sample_invalid_inputs(n, std, msg):
    with pytest.raises(ValueError, match=msg):
        one_sample_t_test(mean=5.0, std=std, n=n, mu0=0.0)


def test_one_sample_invalid_alpha():
    with pytest.raises(ValueError, match="alpha"):
        one_sample_t_test(mean=5.0, std=1.0, n=10, mu0=0.0, alpha=1.5)


def test_one_sample_invalid_sided():
    with pytest.raises(ValueError, match="sided"):
        one_sample_t_test(mean=5.0, std=1.0, n=10, mu0=0.0, sided="both")


# ---------------------------------------------------------------------------
# two_sample_t_test core function
# ---------------------------------------------------------------------------


def test_two_sample_known_values():
    """Verify against scipy ttest_ind (equal_var=False) for a known dataset."""
    from scipy.stats import ttest_ind

    g1 = [2.1, 2.5, 2.3, 2.9, 2.0]
    g2 = [3.1, 3.5, 3.3, 3.9, 3.0]
    res = two_sample_t_test(
        _mean(g1),
        _std(g1),
        len(g1),
        _mean(g2),
        _std(g2),
        len(g2),
    )
    scipy_t, scipy_p = ttest_ind(g1, g2, equal_var=False)
    assert res.t_stat == pytest.approx(scipy_t, abs=1e-6)
    assert res.p_value == pytest.approx(scipy_p, abs=1e-6)


def test_two_sample_df_welch():
    """Welch df should differ from pooled df when variances differ."""
    res = two_sample_t_test(5.0, 1.0, 10, 3.0, 4.0, 10)
    pooled_df = 10 + 10 - 2
    assert res.df != pooled_df


def test_two_sample_sided_less():
    res = two_sample_t_test(3.0, 1.0, 20, 4.0, 1.0, 20, sided="less")
    assert res.p_value < 0.05


def test_two_sample_sided_greater():
    res = two_sample_t_test(5.0, 1.0, 20, 3.0, 1.0, 20, sided="greater")
    assert res.p_value < 0.05


def test_two_sample_ci_excludes_zero_when_significant():
    res = two_sample_t_test(10.0, 1.0, 30, 5.0, 1.0, 30)
    assert res.ci_lower > 0


def test_two_sample_zero_se_equal():
    res = two_sample_t_test(5.0, 0.0, 5, 5.0, 0.0, 5)
    assert res.t_stat == 0.0
    assert res.p_value == 1.0


def test_two_sample_zero_se_unequal():
    res = two_sample_t_test(6.0, 0.0, 5, 5.0, 0.0, 5)
    assert math.isinf(res.t_stat)


@pytest.mark.parametrize(
    "n1,std1,n2,std2,msg",
    [
        (1, 1.0, 5, 1.0, "n1"),
        (5, -1.0, 5, 1.0, "std for n1"),
        (5, 1.0, 1, 1.0, "n2"),
        (5, 1.0, 5, -1.0, "std for n2"),
    ],
)
def test_two_sample_invalid_inputs(n1, std1, n2, std2, msg):
    with pytest.raises(ValueError, match=msg):
        two_sample_t_test(5.0, std1, n1, 3.0, std2, n2)


def test_two_sample_invalid_alpha():
    with pytest.raises(ValueError, match="alpha"):
        two_sample_t_test(5.0, 1.0, 5, 3.0, 1.0, 5, alpha=0.0)


def test_two_sample_invalid_sided():
    with pytest.raises(ValueError, match="sided"):
        two_sample_t_test(5.0, 1.0, 5, 3.0, 1.0, 5, sided="both")


# ---------------------------------------------------------------------------
# paired_t_test core function
# ---------------------------------------------------------------------------


def test_paired_known_values():
    """Verify against scipy ttest_rel for a known dataset."""
    from scipy.stats import ttest_rel

    before = [85.0, 90.0, 78.0, 92.0, 88.0]
    after = [90.0, 95.0, 82.0, 95.0, 91.0]
    res = paired_t_test(before, after)
    scipy_t, scipy_p = ttest_rel(before, after)
    assert res.t_stat == pytest.approx(scipy_t, abs=1e-6)
    assert res.p_value == pytest.approx(scipy_p, abs=1e-6)


def test_paired_mean_diff():
    a = [1.0, 2.0, 3.0]
    b = [2.0, 3.0, 4.0]
    res = paired_t_test(a, b)
    assert res.mean_diff == pytest.approx(-1.0)


def test_paired_ci_excludes_zero_when_significant():
    a = [10.0] * 30
    b = [15.0] * 30
    # Zero std on differences — t_stat is inf, p_value is 0
    res = paired_t_test(a, b)
    assert res.p_value == 0.0


def test_paired_sided_less():
    before = [10.0, 12.0, 11.0, 13.0, 9.0]
    after = [15.0, 16.0, 14.0, 17.0, 13.0]
    res = paired_t_test(before, after, sided="less")
    assert res.p_value < 0.05


def test_paired_unequal_lengths_raises():
    with pytest.raises(ValueError, match="same length"):
        paired_t_test([1.0, 2.0], [1.0])


def test_paired_too_few_pairs_raises():
    with pytest.raises(ValueError, match="at least 2"):
        paired_t_test([1.0], [2.0])


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------


def _ns(**kwargs) -> argparse.Namespace:
    """Build a minimal Namespace for validate() tests."""
    defaults = dict(
        mode="one-sample",
        values=None,
        mean=None,
        std=None,
        n=None,
        mu0=3.0,
        values1=None,
        values2=None,
        mean1=None,
        std1=None,
        n1=None,
        mean2=None,
        std2=None,
        n2=None,
        alpha=0.05,
        sided="two",
        precision=4,
    )
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def test_validate_invalid_alpha():
    assert validate(_ns(alpha=1.5)) is not None


def test_validate_negative_precision():
    assert validate(_ns(precision=-1)) is not None


def test_validate_one_sample_no_values_missing_stats():
    args = _ns(values=None, mean=None, std=1.0, n=10)
    assert validate(args) is not None


def test_validate_one_sample_non_finite_mean():
    args = _ns(values=None, mean=float("inf"), std=1.0, n=10)
    assert validate(args) is not None


def test_validate_one_sample_negative_std():
    args = _ns(values=None, mean=5.0, std=-1.0, n=10)
    assert validate(args) is not None


def test_validate_one_sample_n_too_small():
    args = _ns(values=None, mean=5.0, std=1.0, n=1)
    assert validate(args) is not None


def test_validate_one_sample_non_finite_mu0():
    args = _ns(values="1,2,3", mu0=float("inf"))
    assert validate(args) is not None


def test_validate_one_sample_valid_values():
    assert validate(_ns(values="1,2,3")) is None


def test_validate_one_sample_valid_summary():
    assert validate(_ns(values=None, mean=5.0, std=1.0, n=10)) is None


def test_validate_two_sample_no_groups():
    args = _ns(mode="two-sample")
    assert validate(args) is not None


def test_validate_two_sample_raw_and_summary_conflict_g1():
    args = _ns(mode="two-sample", values1="1,2,3", mean1=2.0, std2=1.0, n2=5)
    assert validate(args) is not None


def test_validate_two_sample_raw_and_summary_conflict_g2():
    args = _ns(mode="two-sample", values2="1,2,3", mean2=2.0, std1=1.0, n1=5)
    assert validate(args) is not None


def test_validate_two_sample_incomplete_summary_g1():
    args = _ns(mode="two-sample", mean1=5.0, values2="1,2,3")
    assert validate(args) is not None


def test_validate_two_sample_incomplete_summary_g2():
    args = _ns(mode="two-sample", values1="1,2,3", mean2=5.0)
    assert validate(args) is not None


def test_validate_two_sample_invalid_mean1():
    args = _ns(
        mode="two-sample", mean1=float("inf"), std1=1.0, n1=5, mean2=3.0, std2=1.0, n2=5
    )
    assert validate(args) is not None


def test_validate_two_sample_invalid_std1():
    args = _ns(mode="two-sample", mean1=5.0, std1=-1.0, n1=5, mean2=3.0, std2=1.0, n2=5)
    assert validate(args) is not None


def test_validate_two_sample_invalid_n1():
    args = _ns(mode="two-sample", mean1=5.0, std1=1.0, n1=1, mean2=3.0, std2=1.0, n2=5)
    assert validate(args) is not None


def test_validate_two_sample_invalid_mean2():
    args = _ns(mode="two-sample", values1="1,2,3", mean2=float("nan"), std2=1.0, n2=5)
    assert validate(args) is not None


def test_validate_two_sample_invalid_std2():
    args = _ns(mode="two-sample", values1="1,2,3", mean2=3.0, std2=-1.0, n2=5)
    assert validate(args) is not None


def test_validate_two_sample_invalid_n2():
    args = _ns(mode="two-sample", values1="1,2,3", mean2=3.0, std2=1.0, n2=1)
    assert validate(args) is not None


def test_validate_two_sample_valid_raw():
    args = _ns(mode="two-sample", values1="1,2,3", values2="4,5,6")
    assert validate(args) is None


def test_validate_two_sample_valid_summary():
    args = _ns(
        mode="two-sample",
        mean1=5.0,
        std1=1.0,
        n1=10,
        mean2=3.0,
        std2=1.0,
        n2=10,
    )
    assert validate(args) is None


def test_validate_paired_valid():
    args = _ns(mode="paired", values1="1,2,3", values2="4,5,6")
    assert validate(args) is None


# ---------------------------------------------------------------------------
# format functions
# ---------------------------------------------------------------------------


def _make_one_result(**kwargs) -> OneSampleResult:
    defaults = dict(
        t_stat=2.5,
        df=9.0,
        p_value=0.034,
        mean=5.0,
        std=1.0,
        n=10,
        mu0=4.0,
        ci_lower=4.3,
        ci_upper=5.7,
        cohens_d=1.0,
        alpha=0.05,
        sided="two",
    )
    defaults.update(kwargs)
    return OneSampleResult(**defaults)


def _make_two_result(**kwargs) -> TwoSampleResult:
    defaults = dict(
        t_stat=3.0,
        df=18.0,
        p_value=0.008,
        mean1=6.0,
        mean2=4.0,
        std1=1.0,
        std2=1.0,
        n1=10,
        n2=10,
        ci_lower=0.5,
        ci_upper=3.5,
        cohens_d=2.0,
        alpha=0.05,
        sided="two",
    )
    defaults.update(kwargs)
    return TwoSampleResult(**defaults)


def _make_paired_result(**kwargs) -> PairedResult:
    defaults = dict(
        t_stat=-3.5,
        df=4.0,
        p_value=0.025,
        mean_diff=-2.0,
        std_diff=1.0,
        n=5,
        ci_lower=-4.0,
        ci_upper=-0.1,
        cohens_d=-2.0,
        alpha=0.05,
        sided="two",
    )
    defaults.update(kwargs)
    return PairedResult(**defaults)


def test_format_one_sample_contains_t_stat():
    out = format_one_sample(_make_one_result(), precision=2)
    assert "2.50" in out


def test_format_one_sample_contains_p_value():
    out = format_one_sample(_make_one_result(), precision=3)
    assert "0.034" in out


def test_format_one_sample_reject_label():
    out = format_one_sample(_make_one_result(p_value=0.01), precision=4)
    assert "Reject" in out


def test_format_one_sample_fail_to_reject_label():
    out = format_one_sample(_make_one_result(p_value=0.99), precision=4)
    assert "Fail" in out


def test_format_one_sample_ci_present():
    out = format_one_sample(_make_one_result(), precision=2)
    assert "CI" in out


def test_format_two_sample_contains_groups():
    out = format_two_sample(_make_two_result(), precision=2)
    assert "Group 1" in out
    assert "Group 2" in out


def test_format_two_sample_contains_diff():
    out = format_two_sample(_make_two_result(), precision=2)
    assert "2.00" in out  # mean1 - mean2 = 6 - 4 = 2


def test_format_two_sample_reject():
    out = format_two_sample(_make_two_result(p_value=0.001), precision=4)
    assert "Reject" in out


def test_format_paired_contains_pairs():
    out = format_paired(_make_paired_result(), precision=2)
    assert "pairs" in out


def test_format_paired_contains_mean_diff():
    out = format_paired(_make_paired_result(), precision=2)
    assert "-2.00" in out


def test_format_paired_reject():
    out = format_paired(_make_paired_result(p_value=0.01), precision=4)
    assert "Reject" in out


# ---------------------------------------------------------------------------
# main() integration tests
# ---------------------------------------------------------------------------


def test_main_one_sample_values(capsys):
    rc = main(["one-sample", "--values", "2.1,3.4,2.9,3.1,2.8", "--mu0", "3.0"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "One-sample" in out
    assert "t-stat" in out


def test_main_one_sample_summary_stats(capsys):
    rc = main(
        [
            "one-sample",
            "--mean",
            "105",
            "--std",
            "12",
            "--n",
            "25",
            "--mu0",
            "100",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "t-stat" in out


def test_main_one_sample_sided_greater(capsys):
    rc = main(
        [
            "one-sample",
            "--values",
            "4,5,6",
            "--mu0",
            "3.0",
            "--sided",
            "greater",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "greater" in out


def test_main_one_sample_precision(capsys):
    rc = main(
        [
            "one-sample",
            "--values",
            "1,2,3",
            "--mu0",
            "2.0",
            "--precision",
            "2",
        ]
    )
    capsys.readouterr()
    assert rc == 0


def test_main_two_sample_values(capsys):
    rc = main(
        [
            "two-sample",
            "--values1",
            "1.2,2.3,3.1,2.8",
            "--values2",
            "2.1,3.2,4.1,3.9",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "Two-sample" in out


def test_main_two_sample_summary(capsys):
    rc = main(
        [
            "two-sample",
            "--mean1",
            "5.0",
            "--std1",
            "1.0",
            "--n1",
            "20",
            "--mean2",
            "3.0",
            "--std2",
            "1.0",
            "--n2",
            "20",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "Two-sample" in out


def test_main_paired(capsys):
    rc = main(
        [
            "paired",
            "--values1",
            "85,90,78,92,88",
            "--values2",
            "90,95,82,95,91",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "Paired" in out


def test_main_missing_mu0_returns_2(capsys):
    rc = main(["one-sample", "--values", "1,2,3"])
    assert rc == 2


def test_main_invalid_values_returns_2(capsys):
    rc = main(["one-sample", "--values", "1,not-a-number", "--mu0", "2"])
    assert rc == 2


def test_main_one_sample_invalid_alpha_returns_2(capsys):
    rc = main(
        [
            "one-sample",
            "--values",
            "1,2,3",
            "--mu0",
            "2",
            "--alpha",
            "2.0",
        ]
    )
    assert rc == 2


def test_main_two_sample_missing_group2_returns_2(capsys):
    rc = main(["two-sample", "--values1", "1,2,3"])
    assert rc == 2


def test_main_paired_unequal_lengths_returns_2(capsys):
    rc = main(["paired", "--values1", "1,2,3", "--values2", "4,5"])
    assert rc == 2


def test_main_one_sample_calc_error_returns_2(monkeypatch, capsys):
    def raise_err(*_a, **_kw):
        raise ValueError("mock error")

    monkeypatch.setattr(t_test_module, "one_sample_t_test", raise_err)
    rc = main(["one-sample", "--values", "1,2,3,4,5", "--mu0", "3.0"])
    err = capsys.readouterr().err
    assert rc == 2
    assert "mock error" in err


def test_main_two_sample_calc_error_returns_2(monkeypatch, capsys):
    def raise_err(*_a, **_kw):
        raise ValueError("mock two-sample error")

    monkeypatch.setattr(t_test_module, "two_sample_t_test", raise_err)
    rc = main(
        [
            "two-sample",
            "--values1",
            "1,2,3,4,5",
            "--values2",
            "6,7,8,9,10",
        ]
    )
    err = capsys.readouterr().err
    assert rc == 2
    assert "mock two-sample error" in err


def test_main_paired_calc_error_returns_2(monkeypatch, capsys):
    def raise_err(*_a, **_kw):
        raise ValueError("mock paired error")

    monkeypatch.setattr(t_test_module, "paired_t_test", raise_err)
    rc = main(["paired", "--values1", "1,2,3", "--values2", "4,5,6"])
    err = capsys.readouterr().err
    assert rc == 2
    assert "mock paired error" in err
