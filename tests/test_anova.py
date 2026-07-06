"""Tests for the one-way ANOVA utility."""

import argparse
import json
import math

import pytest

import src.utils.anova as anova_module
from src.utils.anova import (
    anova_one_way,
    bonferroni,
    f_cdf,
    f_sf,
    format_json,
    format_table,
    main,
    parse_number_list,
    read_csv_groups,
    regularized_incomplete_beta,
    studentized_range_cdf,
    studentized_range_ppf,
    studentized_range_sf,
    t_sf_two_sided,
    tukey_hsd,
    validate,
)

G1 = [12.1, 11.8, 12.5, 11.9]
G2 = [9.8, 10.3, 10.1, 9.7]
G3 = [15.2, 14.9, 15.5, 16.0]

# ---------------------------------------------------------------------------
# regularized_incomplete_beta / f_cdf / f_sf / t_sf_two_sided
# ---------------------------------------------------------------------------


def test_regularized_incomplete_beta_endpoints():
    assert regularized_incomplete_beta(0.0, 2.0, 3.0) == 0.0
    assert regularized_incomplete_beta(1.0, 2.0, 3.0) == 1.0


def test_regularized_incomplete_beta_series_and_symmetric_branches():
    # x < (a+1)/(a+b+2) triggers the direct branch; the complementary
    # region (x closer to 1) exercises the symmetry-transformed branch.
    lo = regularized_incomplete_beta(0.1, 2.0, 5.0)
    hi = regularized_incomplete_beta(0.9, 2.0, 5.0)
    assert 0.0 <= lo <= 1.0
    assert 0.0 <= hi <= 1.0


def test_regularized_incomplete_beta_bounds_error():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        regularized_incomplete_beta(-0.1, 1.0, 1.0)


def test_regularized_incomplete_beta_shape_error():
    with pytest.raises(ValueError, match="a and b must be"):
        regularized_incomplete_beta(0.5, 0.0, 1.0)


def test_beta_cf_fpmin_guards_hit(monkeypatch):
    """Force the near-zero denominator safety branches in the continued fraction."""
    monkeypatch.setattr(anova_module, "_FPMIN", 1e5)
    result = anova_module._beta_cf(2.0, 5.0, 0.1)
    assert math.isfinite(result)


def test_f_cdf_matches_scipy_oracle():
    scipy_stats = pytest.importorskip("scipy.stats")
    for d1 in (1, 2, 5, 20):
        for d2 in (1, 5, 30, 100):
            for x in (0.1, 1.0, 5.0, 20.0):
                got = f_cdf(x, d1, d2)
                want = scipy_stats.f.cdf(x, d1, d2)
                assert got == pytest.approx(want, abs=1e-7)


def test_f_cdf_zero_is_zero():
    assert f_cdf(0.0, 2, 5) == 0.0


def test_f_cdf_infinite_is_one():
    assert f_cdf(math.inf, 2, 5) == 1.0


def test_f_cdf_invalid_df_raises():
    with pytest.raises(ValueError, match="df1 and df2"):
        f_cdf(1.0, 0, 5)


def test_f_cdf_negative_f_raises():
    with pytest.raises(ValueError, match="f_stat must be"):
        f_cdf(-1.0, 2, 5)


def test_f_sf_complements_cdf():
    assert f_cdf(3.0, 2, 10) + f_sf(3.0, 2, 10) == pytest.approx(1.0)


def test_t_sf_two_sided_matches_scipy_oracle():
    scipy_stats = pytest.importorskip("scipy.stats")
    for df in (1, 2, 10, 50, 200):
        for t in (0.0, 0.5, 1.0, 2.5, 5.0):
            got = t_sf_two_sided(t, df)
            want = 2 * scipy_stats.t.sf(abs(t), df)
            assert got == pytest.approx(want, abs=1e-7)


def test_t_sf_two_sided_zero_t_is_one():
    assert t_sf_two_sided(0.0, 10) == 1.0


def test_t_sf_two_sided_invalid_df_raises():
    with pytest.raises(ValueError, match="df must be > 0"):
        t_sf_two_sided(1.0, 0)


# ---------------------------------------------------------------------------
# anova_one_way
# ---------------------------------------------------------------------------


def test_anova_one_way_matches_scipy_oracle():
    scipy_stats = pytest.importorskip("scipy.stats")
    result = anova_one_way([G1, G2, G3])
    want = scipy_stats.f_oneway(G1, G2, G3)
    assert result.f_stat == pytest.approx(want.statistic)
    assert result.p_value == pytest.approx(want.pvalue)
    assert result.df_between == 2
    assert result.df_within == 9


def test_anova_one_way_group_means():
    result = anova_one_way([G1, G2, G3])
    assert result.group_means[0] == pytest.approx(sum(G1) / len(G1))
    assert result.group_sizes == [4, 4, 4]


def test_anova_one_way_too_few_groups_raises():
    with pytest.raises(ValueError, match="at least 2 groups"):
        anova_one_way([G1])


def test_anova_one_way_empty_group_raises():
    with pytest.raises(ValueError, match="at least 1 observation"):
        anova_one_way([G1, []])


def test_anova_one_way_insufficient_df_within_raises():
    with pytest.raises(ValueError, match="df_within"):
        anova_one_way([[1.0], [2.0]])


def test_anova_one_way_invalid_alpha_raises():
    with pytest.raises(ValueError, match="alpha must be"):
        anova_one_way([G1, G2], alpha=1.5)


def test_anova_one_way_zero_within_variance_equal_means():
    result = anova_one_way([[5.0, 5.0], [5.0, 5.0]])
    assert result.f_stat == 0.0
    assert result.p_value == 1.0


def test_anova_one_way_zero_within_variance_different_means():
    result = anova_one_way([[1.0, 1.0], [5.0, 5.0]])
    assert math.isinf(result.f_stat)
    assert result.p_value == 0.0


# ---------------------------------------------------------------------------
# bonferroni
# ---------------------------------------------------------------------------


def test_bonferroni_pair_count():
    result = anova_one_way([G1, G2, G3])
    comparisons = bonferroni([G1, G2, G3], result)
    assert len(comparisons) == 3  # C(3, 2)


def test_bonferroni_matches_manual_two_group_case():
    groups = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    result = anova_one_way(groups)
    comparisons = bonferroni(groups, result)
    assert len(comparisons) == 1
    c = comparisons[0]
    # For 2 groups, t^2 == F
    assert c.t_stat**2 == pytest.approx(result.f_stat)
    assert c.p_adj == pytest.approx(c.p_raw)  # single comparison, no correction


def test_bonferroni_adjusted_pvalue_capped_at_one():
    groups = [[1.0, 1.1, 0.9], [1.05, 0.95, 1.0], [1.02, 0.98, 1.0]]
    result = anova_one_way(groups)
    comparisons = bonferroni(groups, result)
    assert all(0.0 <= c.p_adj <= 1.0 for c in comparisons)


def test_bonferroni_zero_se_equal_means():
    # All groups have zero within-group variance, so ms_within == 0 and se == 0
    # for every pair; groups 0 and 1 additionally share the same mean.
    groups = [[5.0, 5.0], [5.0, 5.0], [3.0, 3.0]]
    result = anova_one_way(groups)
    comparisons = bonferroni(groups, result)
    zero_se_pair = next(c for c in comparisons if c.i == 0 and c.j == 1)
    assert zero_se_pair.t_stat == 0.0
    assert zero_se_pair.p_raw == 1.0


def test_bonferroni_zero_se_different_means():
    groups = [[1.0, 1.0], [9.0, 9.0], [3.0, 3.0]]
    result = anova_one_way(groups)
    comparisons = bonferroni(groups, result)
    pair = next(c for c in comparisons if c.i == 0 and c.j == 1)
    assert math.isinf(pair.t_stat)
    assert pair.p_raw == 0.0


def test_bonferroni_invalid_alpha_raises():
    result = anova_one_way([G1, G2])
    with pytest.raises(ValueError, match="alpha must be"):
        bonferroni([G1, G2], result, alpha=0.0)


def test_bonferroni_significance_flag():
    result = anova_one_way([G1, G2, G3])
    comparisons = bonferroni([G1, G2, G3], result)
    assert all(c.significant for c in comparisons)


# ---------------------------------------------------------------------------
# _simpson / _range_cdf_known_variance (private numerical helpers)
# ---------------------------------------------------------------------------


def test_simpson_empty_interval_is_zero():
    assert anova_module._simpson(lambda x: x, 5.0, 5.0, 10) == 0.0
    assert anova_module._simpson(lambda x: x, 5.0, 2.0, 10) == 0.0


def test_simpson_odd_n_rounds_up():
    # n=3 (odd) should be bumped to 4 internally without raising.
    result = anova_module._simpson(lambda x: x * x, 0.0, 1.0, 3)
    assert result == pytest.approx(1.0 / 3.0, abs=1e-6)


def test_range_cdf_known_variance_non_positive_x_is_zero():
    assert anova_module._range_cdf_known_variance(0.0, 3) == 0.0
    assert anova_module._range_cdf_known_variance(-1.0, 3) == 0.0


# ---------------------------------------------------------------------------
# studentized_range_cdf / studentized_range_sf / studentized_range_ppf
# ---------------------------------------------------------------------------


def test_studentized_range_cdf_matches_scipy_oracle():
    scipy_stats = pytest.importorskip("scipy.stats")
    cases = [
        (3.5, 3, 10),
        (2.5, 4, 20),
        (4.0, 5, 8),
        (3.0, 3, 200),
        (2.0, 6, 30),
        (3.0, 2, 5),
        (2.5, 3, 1),
        (1.5, 8, 15),
        (4.5, 10, 50),
    ]
    for q, k, df in cases:
        got = studentized_range_cdf(q, k, df)
        want = scipy_stats.studentized_range.cdf(q, k, df)
        assert got == pytest.approx(want, abs=1e-6)


def test_studentized_range_cdf_zero_q_is_zero():
    assert studentized_range_cdf(0.0, 3, 10) == 0.0


def test_studentized_range_cdf_invalid_k_raises():
    with pytest.raises(ValueError, match="k must be >= 2"):
        studentized_range_cdf(1.0, 1, 10)


def test_studentized_range_cdf_invalid_df_raises():
    with pytest.raises(ValueError, match="df must be >= 1"):
        studentized_range_cdf(1.0, 3, 0)


def test_studentized_range_cdf_negative_q_raises():
    with pytest.raises(ValueError, match="q must be >= 0"):
        studentized_range_cdf(-1.0, 3, 10)


def test_studentized_range_sf_complements_cdf():
    total = studentized_range_cdf(3.0, 3, 10) + studentized_range_sf(3.0, 3, 10)
    assert total == pytest.approx(1.0)


def test_studentized_range_ppf_matches_scipy_oracle():
    scipy_stats = pytest.importorskip("scipy.stats")
    for k, df in [(3, 10), (4, 20), (2, 5)]:
        got = studentized_range_ppf(0.95, k, df)
        want = scipy_stats.studentized_range.ppf(0.95, k, df)
        assert got == pytest.approx(want, abs=1e-4)


def test_studentized_range_ppf_roundtrips_cdf():
    q = studentized_range_ppf(0.9, 4, 15)
    assert studentized_range_cdf(q, 4, 15) == pytest.approx(0.9, abs=1e-4)


def test_studentized_range_ppf_invalid_p_raises():
    with pytest.raises(ValueError, match="p must be in"):
        studentized_range_ppf(0.0, 3, 10)
    with pytest.raises(ValueError, match="p must be in"):
        studentized_range_ppf(1.0, 3, 10)


# ---------------------------------------------------------------------------
# tukey_hsd
# ---------------------------------------------------------------------------


def test_tukey_hsd_matches_scipy_oracle():
    scipy_stats = pytest.importorskip("scipy.stats")
    result = anova_one_way([G1, G2, G3])
    tukey_result = tukey_hsd([G1, G2, G3], result)
    want = scipy_stats.tukey_hsd(G1, G2, G3)
    for c in tukey_result.comparisons:
        assert c.p_value == pytest.approx(want.pvalue[c.i][c.j], abs=1e-6)


def test_tukey_hsd_pair_count():
    result = anova_one_way([G1, G2, G3])
    tukey_result = tukey_hsd([G1, G2, G3], result)
    assert len(tukey_result.comparisons) == 3


def test_tukey_hsd_two_group_matches_t_test():
    # For k=2, the Tukey p-value equals the pooled two-sided t-test p-value.
    groups = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    result = anova_one_way(groups)
    tukey_result = tukey_hsd(groups, result)
    n_i, n_j = result.group_sizes[0], result.group_sizes[1]
    se = math.sqrt(result.ms_within * (1 / n_i + 1 / n_j))
    t_stat = (result.group_means[0] - result.group_means[1]) / se
    want_p = t_sf_two_sided(t_stat, result.df_within)
    assert tukey_result.comparisons[0].p_value == pytest.approx(want_p, abs=1e-6)


def test_tukey_hsd_invalid_alpha_raises():
    result = anova_one_way([G1, G2])
    with pytest.raises(ValueError, match="alpha must be"):
        tukey_hsd([G1, G2], result, alpha=0.0)


def test_tukey_hsd_zero_se_equal_means():
    groups = [[5.0, 5.0], [5.0, 5.0], [3.0, 3.0]]
    result = anova_one_way(groups)
    tukey_result = tukey_hsd(groups, result)
    pair = next(c for c in tukey_result.comparisons if c.i == 0 and c.j == 1)
    assert pair.q_stat == 0.0
    assert pair.p_value == 1.0


def test_tukey_hsd_zero_se_different_means():
    groups = [[1.0, 1.0], [9.0, 9.0], [3.0, 3.0]]
    result = anova_one_way(groups)
    tukey_result = tukey_hsd(groups, result)
    pair = next(c for c in tukey_result.comparisons if c.i == 0 and c.j == 1)
    assert math.isinf(pair.q_stat)
    assert pair.p_value == 0.0


def test_tukey_hsd_significance_flag():
    result = anova_one_way([G1, G2, G3])
    tukey_result = tukey_hsd([G1, G2, G3], result)
    assert all(c.significant for c in tukey_result.comparisons)


# ---------------------------------------------------------------------------
# read_csv_groups
# ---------------------------------------------------------------------------


def test_read_csv_groups_basic(tmp_path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text(
        "treatment,response\nA,1.0\nA,2.0\nB,3.0\nB,4.0\n", encoding="utf-8"
    )
    groups = read_csv_groups(str(csv_path), "treatment", "response")
    assert groups == {"A": [1.0, 2.0], "B": [3.0, 4.0]}


def test_read_csv_groups_missing_column_raises(tmp_path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("group,val\nA,1.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must contain columns"):
        read_csv_groups(str(csv_path), "treatment", "response")


def test_read_csv_groups_non_numeric_value_raises(tmp_path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("treatment,response\nA,oops\n", encoding="utf-8")
    with pytest.raises(ValueError, match="non-numeric"):
        read_csv_groups(str(csv_path), "treatment", "response")


def test_read_csv_groups_non_finite_value_raises(tmp_path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("treatment,response\nA,nan\n", encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite"):
        read_csv_groups(str(csv_path), "treatment", "response")


# ---------------------------------------------------------------------------
# parse_number_list
# ---------------------------------------------------------------------------


def test_parse_number_list_basic():
    assert parse_number_list("1,2.5,3") == [1.0, 2.5, 3.0]


def test_parse_number_list_empty_raises():
    with pytest.raises(ValueError, match="at least one value"):
        parse_number_list("")


def test_parse_number_list_non_finite_raises():
    with pytest.raises(ValueError, match="finite"):
        parse_number_list("1,inf,3")


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------


def _ns(**kwargs):
    defaults = dict(
        data=None,
        file=None,
        group_col=None,
        value_col=None,
        alpha=0.05,
        posthoc="none",
        format="table",
        precision=4,
    )
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def test_validate_alpha_out_of_range():
    assert "--alpha" in validate(_ns(alpha=0.0))


def test_validate_precision_negative():
    assert "--precision" in validate(_ns(precision=-1))


def test_validate_no_input_source():
    assert "provide either" in validate(_ns())


def test_validate_both_input_sources():
    assert "not both" in validate(
        _ns(data=["1,2", "3,4"], file="x.csv", group_col="g", value_col="v")
    )


def test_validate_data_too_few_groups():
    assert "at least 2 groups" in validate(_ns(data=["1,2,3"]))


def test_validate_file_missing_columns():
    assert "requires both" in validate(_ns(file="x.csv"))


def test_validate_valid_data_tukey():
    assert validate(_ns(data=["1,2", "3,4"], posthoc="tukey")) is None


def test_validate_valid_data():
    assert validate(_ns(data=["1,2", "3,4"])) is None


def test_validate_valid_file():
    assert validate(_ns(file="x.csv", group_col="g", value_col="v")) is None


# ---------------------------------------------------------------------------
# format_table / format_json
# ---------------------------------------------------------------------------


def test_format_table_no_posthoc():
    result = anova_one_way([G1, G2, G3])
    out = format_table(result, "none", None, 4)
    assert "Reject H" in out
    assert "Bonferroni" not in out
    assert "Tukey" not in out


def test_format_table_fail_to_reject():
    result = anova_one_way([[1.0, 1.1, 0.9], [1.05, 0.95, 1.0]])
    out = format_table(result, "none", None, 4)
    assert "Fail to reject" in out


def test_format_table_infinite_f_stat():
    result = anova_one_way([[1.0, 1.0], [9.0, 9.0]])
    out = format_table(result, "none", None, 4)
    assert "inf" in out


def test_format_table_with_bonferroni():
    result = anova_one_way([G1, G2, G3])
    comparisons = bonferroni([G1, G2, G3], result)
    out = format_table(result, "bonferroni", comparisons, 4)
    assert "Bonferroni pairwise comparisons" in out
    assert "*" in out


def test_format_table_with_tukey():
    result = anova_one_way([G1, G2, G3])
    tukey_result = tukey_hsd([G1, G2, G3], result)
    out = format_table(result, "tukey", tukey_result, 4)
    assert "Tukey HSD pairwise comparisons" in out
    assert "q_crit" in out
    assert "*" in out


def test_format_json_no_posthoc():
    result = anova_one_way([G1, G2, G3])
    out = format_json(result, "none", None)
    data = json.loads(out)
    assert "bonferroni" not in data
    assert "tukey" not in data
    assert data["reject_null"] is True


def test_format_json_with_bonferroni():
    result = anova_one_way([G1, G2, G3])
    comparisons = bonferroni([G1, G2, G3], result)
    out = format_json(result, "bonferroni", comparisons)
    data = json.loads(out)
    assert len(data["bonferroni"]) == 3


def test_format_json_with_tukey():
    result = anova_one_way([G1, G2, G3])
    tukey_result = tukey_hsd([G1, G2, G3], result)
    out = format_json(result, "tukey", tukey_result)
    data = json.loads(out)
    assert len(data["tukey"]["comparisons"]) == 3
    assert "q_crit" in data["tukey"]


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def test_main_data_table(capsys):
    rc = main(
        ["--data", "12.1,11.8,12.5,11.9", "9.8,10.3,10.1,9.7", "15.2,14.9,15.5,16.0"]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "Reject H" in out


def test_main_data_bonferroni(capsys):
    rc = main(
        [
            "--data",
            "12.1,11.8,12.5",
            "9.8,10.3,10.1",
            "15.2,14.9,15.5",
            "--posthoc",
            "bonferroni",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "Bonferroni" in out


def test_main_data_json(capsys):
    rc = main(["--data", "1,2,3", "4,5,6", "--format", "json"])
    out = capsys.readouterr().out
    assert rc == 0
    data = json.loads(out)
    assert "f_stat" in data


def test_main_file_input(tmp_path, capsys):
    csv_path = tmp_path / "experiment.csv"
    csv_path.write_text(
        "treatment,response\n"
        "A,12.1\nA,11.8\nA,12.5\nA,11.9\n"
        "B,9.8\nB,10.3\nB,10.1\nB,9.7\n"
        "C,15.2\nC,14.9\nC,15.5\nC,16.0\n",
        encoding="utf-8",
    )
    rc = main(
        [
            "--file",
            str(csv_path),
            "--group-col",
            "treatment",
            "--value-col",
            "response",
            "--alpha",
            "0.01",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "0.01" in out


def test_main_data_tukey(capsys):
    rc = main(
        [
            "--data",
            "12.1,11.8,12.5",
            "9.8,10.3,10.1",
            "15.2,14.9,15.5",
            "--posthoc",
            "tukey",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "Tukey HSD" in out


def test_main_data_tukey_json(capsys):
    rc = main(["--data", "1,2,3", "4,5,6", "--posthoc", "tukey", "--format", "json"])
    out = capsys.readouterr().out
    assert rc == 0
    data = json.loads(out)
    assert "tukey" in data


def test_main_no_input_returns_2(capsys):
    assert main([]) == 2
    err = capsys.readouterr().err
    assert "Error" in err


def test_main_missing_csv_file_returns_2(capsys):
    assert (
        main(
            [
                "--file",
                "/nonexistent/path/data.csv",
                "--group-col",
                "g",
                "--value-col",
                "v",
            ]
        )
        == 2
    )
    err = capsys.readouterr().err
    assert "Error" in err


def test_main_computation_error_returns_2(monkeypatch, capsys):
    """Cover the ValueError branch in main when a core function raises."""

    def raise_value_error(*_args, **_kwargs):
        raise ValueError("forced anova error")

    monkeypatch.setattr(anova_module, "anova_one_way", raise_value_error)
    assert main(["--data", "1,2,3", "4,5,6"]) == 2
    err = capsys.readouterr().err
    assert "forced anova error" in err
