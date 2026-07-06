#!/usr/bin/env python3
"""Command-line utility for one-way analysis of variance (ANOVA).

Tests whether the means of three or more independent groups are equal — the
natural generalization of the two-sample t-test to multiple groups. Supports
Tukey HSD and Bonferroni-corrected pairwise post-hoc comparisons.

The F-distribution CDF (and the pairwise t-test p-values used by the
Bonferroni post-hoc) are computed via the regularised incomplete beta
function (`math.lgamma`). Tukey HSD uses the studentized range distribution,
computed via numerical integration (nested Simpson's rule over the normal
and scaled chi densities) — pure Python, no external dependencies.

Usage examples:
  anova --data "12.1,11.8,12.5,11.9" "9.8,10.3,10.1,9.7" "15.2,14.9,15.5,16.0"
  anova --data "12.1,11.8,12.5" "9.8,10.3,10.1" "15.2,14.9,15.5" --posthoc tukey
  anova --file experiment.csv --group-col treatment --value-col response --alpha 0.01
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from typing import Callable, Sequence

# ---------------------------------------------------------------------------
# Regularised incomplete beta function (Numerical Recipes betai/betacf)
# ---------------------------------------------------------------------------

_ITMAX = 200
_EPS = 1e-14
_FPMIN = 1e-300


def _beta_cf(a: float, b: float, x: float) -> float:
    """Continued fraction used by :func:`regularized_incomplete_beta`.

    Args:
        a: First shape parameter; must be > 0.
        b: Second shape parameter; must be > 0.
        x: Evaluation point in [0, 1).

    Returns:
        Value of the continued fraction (not yet scaled by the leading term).
    """
    qab = a + b
    qap = a + 1
    qam = a - 1
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < _FPMIN:
        d = _FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, _ITMAX + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = 1.0 + aa / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = 1.0 + aa / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _EPS:
            break
    return h


def regularized_incomplete_beta(x: float, a: float, b: float) -> float:
    """Regularised incomplete beta function I_x(a, b).

    Args:
        x: Evaluation point; must be in [0, 1].
        a: First shape parameter; must be > 0.
        b: Second shape parameter; must be > 0.

    Returns:
        I_x(a, b) in [0, 1].

    Raises:
        ValueError: If ``x`` is not in [0, 1], or ``a``/``b`` are not > 0.
    """
    if not (0 <= x <= 1):
        raise ValueError(f"x must be in [0, 1], got {x}")
    if a <= 0 or b <= 0:
        raise ValueError(f"a and b must be > 0, got a={a}, b={b}")

    if x == 0 or x == 1:
        return x

    log_bt = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log(1 - x)
    )
    bt = math.exp(log_bt)

    if x < (a + 1) / (a + b + 2):
        return bt * _beta_cf(a, b, x) / a
    return 1.0 - bt * _beta_cf(b, a, 1 - x) / b


def f_cdf(f_stat: float, df1: int, df2: int) -> float:
    """Cumulative distribution function P(F <= f_stat) for F(df1, df2).

    Args:
        f_stat: Observed F-statistic; must be >= 0.
        df1: Numerator degrees of freedom; must be >= 1.
        df2: Denominator degrees of freedom; must be >= 1.

    Returns:
        P(F <= f_stat) in [0, 1].

    Raises:
        ValueError: If ``df1``/``df2`` < 1 or ``f_stat`` < 0.
    """
    if df1 < 1 or df2 < 1:
        raise ValueError(f"df1 and df2 must be >= 1, got df1={df1}, df2={df2}")
    if f_stat < 0:
        raise ValueError(f"f_stat must be >= 0, got {f_stat}")
    if f_stat == 0:
        return 0.0
    if math.isinf(f_stat):
        return 1.0
    x = df1 * f_stat / (df1 * f_stat + df2)
    return regularized_incomplete_beta(x, df1 / 2, df2 / 2)


def f_sf(f_stat: float, df1: int, df2: int) -> float:
    """Survival function (upper tail p-value) for F(df1, df2).

    Args:
        f_stat: Observed F-statistic; must be >= 0.
        df1: Numerator degrees of freedom; must be >= 1.
        df2: Denominator degrees of freedom; must be >= 1.

    Returns:
        P(F > f_stat), clipped to [0, 1].
    """
    return max(0.0, min(1.0, 1.0 - f_cdf(f_stat, df1, df2)))


def t_sf_two_sided(t_stat: float, df: float) -> float:
    """Two-sided p-value P(|T| > |t_stat|) for Student's t(df).

    Computed via the incomplete-beta identity for the t-distribution, so no
    separate t-distribution implementation is needed.

    Args:
        t_stat: Observed t-statistic.
        df: Degrees of freedom; must be > 0.

    Returns:
        Two-sided p-value in [0, 1].

    Raises:
        ValueError: If ``df`` <= 0.
    """
    if df <= 0:
        raise ValueError(f"df must be > 0, got {df}")
    if t_stat == 0:
        return 1.0
    x = df / (df + t_stat * t_stat)
    return max(0.0, min(1.0, regularized_incomplete_beta(x, df / 2, 0.5)))


# ---------------------------------------------------------------------------
# Studentized range distribution (for Tukey HSD)
#
# P(Q <= q | k groups, df) is computed as a nested numerical integral:
#   P(Q <= q | k, df) = ∫₀^∞ h(u; df) · R(q·u, k) du
#   R(x, k) = k · ∫_{-∞}^{∞} φ(z) · [Φ(z) − Φ(z − x)]^(k−1) dz   (known-variance
#             range CDF for k standard normal variates; φ/Φ are the standard
#             normal PDF/CDF)
#   h(u; df) = 2·(df/2)^(df/2) / Γ(df/2) · u^(df−1) · exp(−df·u²/2)   (density
#             of U = S/σ, the scaled sample standard deviation)
#
# Both integrals use a fixed-step Simpson's rule; the outer (u) integral's
# bounds are centered on U's exact mean/std so the peak stays resolved even
# when it becomes very narrow at large df.
# ---------------------------------------------------------------------------

_SQRT2 = math.sqrt(2.0)
_SQRT2PI = math.sqrt(2.0 * math.pi)


def _standard_normal_pdf(z: float) -> float:
    """Standard normal probability density function φ(z)."""
    return math.exp(-z * z / 2.0) / _SQRT2PI


def _standard_normal_cdf(z: float) -> float:
    """Standard normal cumulative distribution function Φ(z)."""
    return 0.5 * (1.0 + math.erf(z / _SQRT2))


def _simpson(f: Callable[[float], float], a: float, b: float, n: int) -> float:
    """Composite Simpson's rule for ∫ f over [a, b] using n subintervals.

    Args:
        f: Integrand.
        a: Lower bound.
        b: Upper bound.
        n: Number of subintervals (rounded up to even).

    Returns:
        Approximate value of the integral; 0.0 if ``b <= a``.
    """
    if b <= a:
        return 0.0
    if n % 2:
        n += 1
    h = (b - a) / n
    total = f(a) + f(b)
    for i in range(1, n):
        x = a + i * h
        total += (4 if i % 2 else 2) * f(x)
    return total * h / 3.0


def _range_cdf_known_variance(
    x: float, k: int, n_z: int = 100, zlim: float = 8.0
) -> float:
    """CDF of the range of k standard normal variates (known-variance case).

    Args:
        x: Range value; must be >= 0.
        k: Number of groups; must be >= 2.
        n_z: Number of Simpson subintervals for the z-integral.
        zlim: Integration bound for z (the standard normal tail beyond this
            is negligible).

    Returns:
        P(range <= x) in [0, 1].
    """
    if x <= 0:
        return 0.0

    def integrand(z: float) -> float:
        return _standard_normal_pdf(z) * (
            _standard_normal_cdf(z) - _standard_normal_cdf(z - x)
        ) ** (k - 1)

    integral = _simpson(integrand, -zlim, zlim, n_z)
    return max(0.0, min(1.0, k * integral))


def _u_mean_std(df: float) -> tuple[float, float]:
    """Exact mean and standard deviation of U = S/σ (U²·df ~ chi-square(df)).

    Args:
        df: Degrees of freedom; must be > 0.

    Returns:
        Tuple (mean, std) of U.
    """
    mean_u = math.sqrt(2.0 / df) * math.exp(
        math.lgamma((df + 1) / 2.0) - math.lgamma(df / 2.0)
    )
    var_u = max(1e-12, 1.0 - mean_u * mean_u)
    return mean_u, math.sqrt(var_u)


def studentized_range_cdf(
    q: float,
    k: int,
    df: float,
    n_z: int = 100,
    n_u: int = 200,
    width_sigmas: float = 12.0,
) -> float:
    """Cumulative distribution function P(Q <= q) for the studentized range.

    Args:
        q: Studentized range value; must be >= 0.
        k: Number of groups (means being compared); must be >= 2.
        df: Degrees of freedom of the variance estimate; must be >= 1.
        n_z: Simpson subintervals for the inner (known-variance) integral.
        n_u: Simpson subintervals for the outer (variance-mixing) integral.
        width_sigmas: Half-width of the outer integration window, in standard
            deviations of U, centered on U's exact mean.

    Returns:
        P(Q <= q) in [0, 1].

    Raises:
        ValueError: If ``k`` < 2, ``df`` < 1, or ``q`` < 0.
    """
    if k < 2:
        raise ValueError(f"k must be >= 2, got {k}")
    if df < 1:
        raise ValueError(f"df must be >= 1, got {df}")
    if q < 0:
        raise ValueError(f"q must be >= 0, got {q}")
    if q == 0:
        return 0.0

    log_const = math.log(2.0) + (df / 2.0) * math.log(df / 2.0) - math.lgamma(df / 2.0)

    def integrand(u: float) -> float:
        # u > 0 always holds here: the outer Simpson call below is bounded
        # below by `lo`, which is clamped to >= 1e-10.
        log_h = log_const + (df - 1) * math.log(u) - df * u * u / 2.0
        return math.exp(log_h) * _range_cdf_known_variance(q * u, k, n_z=n_z)

    mean_u, std_u = _u_mean_std(df)
    lo = max(1e-10, mean_u - width_sigmas * std_u)
    hi = mean_u + width_sigmas * std_u

    return max(0.0, min(1.0, _simpson(integrand, lo, hi, n_u)))


def studentized_range_sf(q: float, k: int, df: float) -> float:
    """Survival function (upper tail p-value) for the studentized range.

    Args:
        q: Studentized range value; must be >= 0.
        k: Number of groups; must be >= 2.
        df: Degrees of freedom; must be >= 1.

    Returns:
        P(Q > q), clipped to [0, 1].
    """
    return max(0.0, min(1.0, 1.0 - studentized_range_cdf(q, k, df)))


def studentized_range_ppf(
    p: float, k: int, df: float, tol: float = 1e-6, max_iter: int = 60
) -> float:
    """Quantile (inverse CDF) of the studentized range via bisection.

    Args:
        p: Target cumulative probability; must be in (0, 1).
        k: Number of groups; must be >= 2.
        df: Degrees of freedom; must be >= 1.
        tol: Bisection stops early once |CDF(mid) - p| < tol.
        max_iter: Maximum bisection iterations.

    Returns:
        q such that P(Q <= q) ≈ p.

    Raises:
        ValueError: If ``p`` is not in (0, 1).
    """
    if not (0 < p < 1):
        raise ValueError(f"p must be in (0, 1), got {p}")

    lo, hi = 0.0, 60.0
    mid = hi / 2.0
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        c = studentized_range_cdf(mid, k, df)
        if abs(c - p) < tol:
            break
        if c < p:
            lo = mid
        else:
            hi = mid
    return mid


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


class AnovaResult:
    """Result of a one-way ANOVA F-test."""

    def __init__(
        self,
        f_stat: float,
        p_value: float,
        df_between: int,
        df_within: int,
        ss_between: float,
        ss_within: float,
        ms_between: float,
        ms_within: float,
        group_means: list[float],
        group_sizes: list[int],
        grand_mean: float,
        alpha: float,
    ) -> None:
        """Store all computed quantities."""
        self.f_stat = f_stat
        self.p_value = p_value
        self.df_between = df_between
        self.df_within = df_within
        self.ss_between = ss_between
        self.ss_within = ss_within
        self.ms_between = ms_between
        self.ms_within = ms_within
        self.group_means = group_means
        self.group_sizes = group_sizes
        self.grand_mean = grand_mean
        self.alpha = alpha


class PairwiseComparison:
    """A single pairwise post-hoc comparison between two groups."""

    def __init__(
        self,
        i: int,
        j: int,
        mean_diff: float,
        t_stat: float,
        p_raw: float,
        p_adj: float,
        significant: bool,
    ) -> None:
        """Store all computed quantities for one pairwise comparison."""
        self.i = i
        self.j = j
        self.mean_diff = mean_diff
        self.t_stat = t_stat
        self.p_raw = p_raw
        self.p_adj = p_adj
        self.significant = significant


class TukeyComparison:
    """A single pairwise Tukey HSD comparison between two groups."""

    def __init__(
        self,
        i: int,
        j: int,
        mean_diff: float,
        q_stat: float,
        p_value: float,
        significant: bool,
    ) -> None:
        """Store all computed quantities for one pairwise comparison."""
        self.i = i
        self.j = j
        self.mean_diff = mean_diff
        self.q_stat = q_stat
        self.p_value = p_value
        self.significant = significant


class TukeyResult:
    """Result of a full Tukey HSD post-hoc analysis."""

    def __init__(self, comparisons: list[TukeyComparison], q_crit: float) -> None:
        """Store the pairwise comparisons and the family-wise critical value."""
        self.comparisons = comparisons
        self.q_crit = q_crit


# ---------------------------------------------------------------------------
# Core statistical functions
# ---------------------------------------------------------------------------


def _mean(values: Sequence[float]) -> float:
    """Compute the arithmetic mean of a non-empty sequence."""
    return sum(values) / len(values)


def anova_one_way(
    groups: Sequence[Sequence[float]], alpha: float = 0.05
) -> AnovaResult:
    """Perform a one-way ANOVA F-test.

    Tests H₀: μ₁ = μ₂ = ... = μₖ across k independent groups.

    Args:
        groups: At least 2 groups, each with at least 1 observation; the
            total number of observations must exceed the number of groups.
        alpha: Significance level (default 0.05).

    Returns:
        :class:`AnovaResult` with the ANOVA table and per-group summaries.

    Raises:
        ValueError: If fewer than 2 groups are given, any group is empty,
            df_within is not positive, or alpha is not in (0, 1).
    """
    if len(groups) < 2:
        raise ValueError(f"need at least 2 groups, got {len(groups)}")
    if any(len(g) == 0 for g in groups):
        raise ValueError("every group must have at least 1 observation")
    if not (0 < alpha < 1):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    k = len(groups)
    n_total = sum(len(g) for g in groups)
    df_between = k - 1
    df_within = n_total - k
    if df_within < 1:
        raise ValueError(
            f"df_within must be >= 1 (need more observations than groups), "
            f"got {df_within}"
        )

    group_means = [_mean(g) for g in groups]
    group_sizes = [len(g) for g in groups]
    grand_mean = sum(sum(g) for g in groups) / n_total

    ss_between = sum(
        n * (m - grand_mean) ** 2 for n, m in zip(group_sizes, group_means)
    )
    ss_within = sum(sum((x - m) ** 2 for x in g) for g, m in zip(groups, group_means))

    ms_between = ss_between / df_between
    ms_within = ss_within / df_within

    if ms_within == 0:
        f_stat = 0.0 if ms_between == 0 else math.inf
        p_value = 1.0 if ms_between == 0 else 0.0
    else:
        f_stat = ms_between / ms_within
        p_value = f_sf(f_stat, df_between, df_within)

    return AnovaResult(
        f_stat=f_stat,
        p_value=p_value,
        df_between=df_between,
        df_within=df_within,
        ss_between=ss_between,
        ss_within=ss_within,
        ms_between=ms_between,
        ms_within=ms_within,
        group_means=group_means,
        group_sizes=group_sizes,
        grand_mean=grand_mean,
        alpha=alpha,
    )


def bonferroni(
    groups: Sequence[Sequence[float]], result: AnovaResult, alpha: float = 0.05
) -> list[PairwiseComparison]:
    """Bonferroni-corrected pairwise post-hoc comparisons.

    Each pair uses a t-test against the pooled within-group variance
    (MS_within) from the ANOVA, with df_within degrees of freedom. The raw
    p-value is multiplied by the number of comparisons and clipped to 1.0.

    Args:
        groups: The same groups passed to :func:`anova_one_way`.
        result: The :class:`AnovaResult` computed from ``groups``.
        alpha: Family-wise significance level (default 0.05).

    Returns:
        List of :class:`PairwiseComparison`, one per unordered group pair.

    Raises:
        ValueError: If ``alpha`` is not in (0, 1).
    """
    if not (0 < alpha < 1):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    k = len(groups)
    n_pairs = k * (k - 1) // 2
    comparisons: list[PairwiseComparison] = []

    for i in range(k):
        for j in range(i + 1, k):
            mean_diff = result.group_means[i] - result.group_means[j]
            n_i, n_j = result.group_sizes[i], result.group_sizes[j]
            se = math.sqrt(result.ms_within * (1 / n_i + 1 / n_j))

            if se == 0:
                t_stat = 0.0 if mean_diff == 0 else math.copysign(math.inf, mean_diff)
                p_raw = 1.0 if mean_diff == 0 else 0.0
            else:
                t_stat = mean_diff / se
                p_raw = t_sf_two_sided(t_stat, result.df_within)

            p_adj = min(1.0, p_raw * n_pairs)
            comparisons.append(
                PairwiseComparison(
                    i=i,
                    j=j,
                    mean_diff=mean_diff,
                    t_stat=t_stat,
                    p_raw=p_raw,
                    p_adj=p_adj,
                    significant=p_adj < alpha,
                )
            )

    return comparisons


def tukey_hsd(
    groups: Sequence[Sequence[float]], result: AnovaResult, alpha: float = 0.05
) -> TukeyResult:
    """Tukey HSD (Tukey-Kramer) pairwise post-hoc comparisons.

    Each pair uses the studentized range statistic
    ``q = |mean_i - mean_j| / sqrt(MS_within / 2 * (1/n_i + 1/n_j))``
    (the Tukey-Kramer generalization for unequal group sizes), tested against
    the studentized range distribution with ``k`` groups and ``df_within``
    degrees of freedom. Unlike Bonferroni, the resulting p-value is already
    family-wise adjusted.

    Args:
        groups: The same groups passed to :func:`anova_one_way`.
        result: The :class:`AnovaResult` computed from ``groups``.
        alpha: Family-wise significance level (default 0.05).

    Returns:
        :class:`TukeyResult` with the pairwise comparisons and the critical
        studentized range value q_crit at the given alpha.

    Raises:
        ValueError: If ``alpha`` is not in (0, 1).
    """
    if not (0 < alpha < 1):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    k = len(groups)
    q_crit = studentized_range_ppf(1 - alpha, k, result.df_within)
    comparisons: list[TukeyComparison] = []

    for i in range(k):
        for j in range(i + 1, k):
            mean_diff = result.group_means[i] - result.group_means[j]
            n_i, n_j = result.group_sizes[i], result.group_sizes[j]
            se = math.sqrt(result.ms_within / 2 * (1 / n_i + 1 / n_j))

            if se == 0:
                q_stat = 0.0 if mean_diff == 0 else math.inf
                p_value = 1.0 if mean_diff == 0 else 0.0
            else:
                q_stat = abs(mean_diff) / se
                p_value = studentized_range_sf(q_stat, k, result.df_within)

            comparisons.append(
                TukeyComparison(
                    i=i,
                    j=j,
                    mean_diff=mean_diff,
                    q_stat=q_stat,
                    p_value=p_value,
                    significant=p_value < alpha,
                )
            )

    return TukeyResult(comparisons=comparisons, q_crit=q_crit)


def read_csv_groups(
    path: str, group_col: str, value_col: str
) -> dict[str, list[float]]:
    """Load grouped observations from a CSV file.

    Args:
        path: Path to a CSV file with a header row.
        group_col: Name of the column holding the group label.
        value_col: Name of the column holding the numeric observation.

    Returns:
        Mapping of group label to its list of observed values, in the order
        groups first appear in the file.

    Raises:
        ValueError: If either column is missing from the header, or a value
            cannot be parsed as a finite float.
    """
    groups: dict[str, list[float]] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None or (
            group_col not in reader.fieldnames or value_col not in reader.fieldnames
        ):
            raise ValueError(
                f"CSV must contain columns {group_col!r} and {value_col!r}"
            )
        for row in reader:
            label = row[group_col]
            try:
                value = float(row[value_col])
            except ValueError as exc:
                raise ValueError(
                    f"non-numeric value {row[value_col]!r} in column {value_col!r}"
                ) from exc
            if not math.isfinite(value):
                raise ValueError(f"non-finite value in column {value_col!r}")
            groups.setdefault(label, []).append(value)
    return groups


def parse_number_list(raw: str) -> list[float]:
    """Parse a comma-separated string of finite numbers.

    Args:
        raw: Comma-separated numeric string (e.g. ``"12.1,11.8,12.5"``).

    Returns:
        List of parsed floats.

    Raises:
        ValueError: If the string is empty, non-numeric, or contains non-finite values.
    """
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        raise ValueError("at least one value is required")
    values = [float(p) for p in parts]
    if any(not math.isfinite(v) for v in values):
        raise ValueError("all values must be finite")
    return values


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Build and return the argument parser namespace.

    Args:
        argv: Argument list (uses ``sys.argv`` when ``None``).

    Returns:
        Parsed :class:`argparse.Namespace`.
    """
    parser = argparse.ArgumentParser(
        description="One-way ANOVA calculator with Tukey HSD / Bonferroni post-hoc comparisons.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  anova --data "12.1,11.8,12.5,11.9" "9.8,10.3,10.1,9.7" "15.2,14.9,15.5,16.0"
  anova --data "12.1,11.8,12.5" "9.8,10.3,10.1" "15.2,14.9,15.5" --posthoc tukey
  anova --file experiment.csv --group-col treatment --value-col response --alpha 0.01
""",
    )
    parser.add_argument(
        "--data",
        nargs="+",
        metavar="GROUP",
        help="one comma-separated list of values per group (2 or more groups)",
    )
    parser.add_argument(
        "--file",
        type=str,
        metavar="CSV",
        help="path to a CSV file (use with --group-col and --value-col)",
    )
    parser.add_argument(
        "--group-col",
        type=str,
        metavar="COL",
        help="CSV column name holding the group label",
    )
    parser.add_argument(
        "--value-col",
        type=str,
        metavar="COL",
        help="CSV column name holding the numeric observation",
    )
    parser.add_argument(
        "--alpha",
        "-a",
        type=float,
        default=0.05,
        metavar="F",
        help="significance level (default: 0.05)",
    )
    parser.add_argument(
        "--posthoc",
        choices=["tukey", "bonferroni", "none"],
        default="none",
        help="post-hoc pairwise comparison method (default: none)",
    )
    parser.add_argument(
        "--format",
        choices=["table", "json"],
        default="table",
        help="output format (default: table)",
    )
    parser.add_argument(
        "--precision",
        "-P",
        type=int,
        default=4,
        metavar="PREC",
        help="decimal places for table output (default: 4)",
    )
    return parser.parse_args(argv)


def validate(args: argparse.Namespace) -> str | None:
    """Return an error message string, or ``None`` if arguments are valid.

    Args:
        args: Parsed argument namespace from :func:`parse_args`.

    Returns:
        Error description string, or ``None`` when validation passes.
    """
    if not (0 < args.alpha < 1):
        return f"--alpha must be in (0, 1), got {args.alpha}"
    if args.precision < 0:
        return "--precision must be non-negative"

    using_data = args.data is not None
    using_file = (
        args.file is not None
        or args.group_col is not None
        or args.value_col is not None
    )

    if using_data and using_file:
        return "provide either --data or --file/--group-col/--value-col, not both"
    if not using_data and not using_file:
        return "provide either --data or --file/--group-col/--value-col"

    if using_data and len(args.data) < 2:
        return "--data requires at least 2 groups"

    if using_file:
        if args.file is None or args.group_col is None or args.value_col is None:
            return "--file requires both --group-col and --value-col"

    return None


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def _fmt(value: float, precision: int) -> str:
    """Format a float to the requested number of decimal places."""
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return f"{value:.{precision}f}"


def _decision(p_value: float, alpha: float) -> str:
    """Return reject / fail-to-reject decision string."""
    if p_value < alpha:
        return f"Reject H₀  (p < α = {alpha})"
    return f"Fail to reject H₀  (p ≥ α = {alpha})"


def format_table(
    result: AnovaResult,
    posthoc: str,
    comparisons: list[PairwiseComparison] | TukeyResult | None,
    precision: int,
) -> str:
    """Format the ANOVA table (and optional post-hoc comparisons) for display.

    Args:
        result: Computed :class:`AnovaResult`.
        posthoc: One of ``"none"``, ``"bonferroni"``, or ``"tukey"``.
        comparisons: ``None`` for no post-hoc; a list of
            :class:`PairwiseComparison` for Bonferroni; a :class:`TukeyResult`
            for Tukey HSD.
        precision: Decimal places for all floating-point output.

    Returns:
        Multi-line string ready to print.
    """
    f = lambda v: _fmt(v, precision)  # noqa: E731
    lines = [
        "One-way ANOVA",
        "H₀: all group means are equal",
        "",
        f"  {'source':>10}  {'SS':>10}  {'df':>6}  {'MS':>10}  {'F':>10}  {'p-value':>10}",
        f"  {'between':>10}  {f(result.ss_between):>10}  {result.df_between:>6}  "
        f"{f(result.ms_between):>10}  {f(result.f_stat):>10}  {f(result.p_value):>10}",
        f"  {'within':>10}  {f(result.ss_within):>10}  {result.df_within:>6}  "
        f"{f(result.ms_within):>10}",
        "",
        f"  {_decision(result.p_value, result.alpha)}",
        "",
        "  Group means:",
    ]
    for idx, (mean, size) in enumerate(zip(result.group_means, result.group_sizes)):
        lines.append(f"    group {idx + 1}:  n={size},  mean={f(mean)}")

    if posthoc == "bonferroni" and isinstance(comparisons, list):
        lines += [
            "",
            "  Bonferroni pairwise comparisons:",
            f"    {'pair':>10}  {'mean diff':>10}  {'t-stat':>10}  "
            f"{'p (raw)':>10}  {'p (adj)':>10}  sig",
        ]
        for c in comparisons:
            sig = "*" if c.significant else ""
            lines.append(
                f"    {c.i + 1} vs {c.j + 1:<4}  {f(c.mean_diff):>10}  {f(c.t_stat):>10}  "
                f"{f(c.p_raw):>10}  {f(c.p_adj):>10}  {sig}"
            )
    elif posthoc == "tukey" and isinstance(comparisons, TukeyResult):
        lines += [
            "",
            f"  Tukey HSD pairwise comparisons  (q_crit={f(comparisons.q_crit)}):",
            f"    {'pair':>10}  {'mean diff':>10}  {'q-stat':>10}  "
            f"{'p (adj)':>10}  sig",
        ]
        for tc in comparisons.comparisons:
            sig = "*" if tc.significant else ""
            lines.append(
                f"    {tc.i + 1} vs {tc.j + 1:<4}  {f(tc.mean_diff):>10}  {f(tc.q_stat):>10}  "
                f"{f(tc.p_value):>10}  {sig}"
            )

    return "\n".join(lines)


def format_json(
    result: AnovaResult,
    posthoc: str,
    comparisons: list[PairwiseComparison] | TukeyResult | None,
) -> str:
    """Format the ANOVA result (and optional post-hoc comparisons) as JSON.

    Args:
        result: Computed :class:`AnovaResult`.
        posthoc: One of ``"none"``, ``"bonferroni"``, or ``"tukey"``.
        comparisons: ``None`` for no post-hoc; a list of
            :class:`PairwiseComparison` for Bonferroni; a :class:`TukeyResult`
            for Tukey HSD.

    Returns:
        JSON string.
    """
    data: dict[str, object] = {
        "f_stat": result.f_stat,
        "p_value": result.p_value,
        "df_between": result.df_between,
        "df_within": result.df_within,
        "ss_between": result.ss_between,
        "ss_within": result.ss_within,
        "ms_between": result.ms_between,
        "ms_within": result.ms_within,
        "group_means": result.group_means,
        "group_sizes": result.group_sizes,
        "alpha": result.alpha,
        "reject_null": result.p_value < result.alpha,
    }
    if posthoc == "bonferroni" and isinstance(comparisons, list):
        data["bonferroni"] = [
            {
                "i": c.i,
                "j": c.j,
                "mean_diff": c.mean_diff,
                "t_stat": c.t_stat,
                "p_raw": c.p_raw,
                "p_adj": c.p_adj,
                "significant": c.significant,
            }
            for c in comparisons
        ]
    elif posthoc == "tukey" and isinstance(comparisons, TukeyResult):
        data["tukey"] = {
            "q_crit": comparisons.q_crit,
            "comparisons": [
                {
                    "i": c.i,
                    "j": c.j,
                    "mean_diff": c.mean_diff,
                    "q_stat": c.q_stat,
                    "p_value": c.p_value,
                    "significant": c.significant,
                }
                for c in comparisons.comparisons
            ],
        }
    return json.dumps(data, indent=2)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Run the ANOVA CLI.

    Args:
        argv: Argument list override for testing (uses ``sys.argv`` when ``None``).

    Returns:
        0 on success, 2 on input or computation error.
    """
    args = parse_args(argv)

    error = validate(args)
    if error:
        print(f"Error: {error}", file=sys.stderr)
        return 2

    try:
        if args.data is not None:
            groups = [parse_number_list(g) for g in args.data]
        else:
            grouped = read_csv_groups(args.file, args.group_col, args.value_col)
            groups = list(grouped.values())

        result = anova_one_way(groups, args.alpha)
        comparisons: list[PairwiseComparison] | TukeyResult | None
        if args.posthoc == "bonferroni":
            comparisons = bonferroni(groups, result, args.alpha)
        elif args.posthoc == "tukey":
            comparisons = tukey_hsd(groups, result, args.alpha)
        else:
            comparisons = None

        if args.format == "json":
            print(format_json(result, args.posthoc, comparisons))
        else:
            print(format_table(result, args.posthoc, comparisons, args.precision))
    except (ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
