#!/usr/bin/env python3
"""Command-line utility for t-test statistics.

Supports one-sample, two-sample (Welch's), and paired t-tests.

Usage examples:
  ttest one-sample --values 2.1,3.4,2.9,3.1,2.8 --mu0 3.0
  ttest one-sample --mean 105 --std 12 --n 25 --mu0 100
  ttest two-sample --values1 1.2,2.3,3.1,2.8 --values2 2.1,3.2,4.1,3.9
  ttest paired --values1 85,90,78,92,88 --values2 90,95,82,95,91
  ttest one-sample --values 2.1,3.4,2.9 --mu0 3.0 --sided greater
"""

from __future__ import annotations

import argparse
import math
import sys
from typing import Sequence

from scipy.stats import t as t_dist

# ---------------------------------------------------------------------------
# Core statistical functions
# ---------------------------------------------------------------------------


def _mean(values: Sequence[float]) -> float:
    """Compute the arithmetic mean of a sequence.

    Args:
        values: Non-empty sequence of finite floats.

    Returns:
        Arithmetic mean.

    Raises:
        ValueError: If ``values`` is empty.
    """
    if not values:
        raise ValueError("Cannot compute mean of an empty sequence")
    return sum(values) / len(values)


def _var(values: Sequence[float]) -> float:
    """Compute the sample variance (divides by n-1).

    Args:
        values: Sequence with at least 2 elements.

    Returns:
        Sample variance.

    Raises:
        ValueError: If fewer than 2 values are provided.
    """
    n = len(values)
    if n < 2:
        raise ValueError(f"Need at least 2 values for sample variance, got {n}")
    mu = _mean(values)
    return sum((x - mu) ** 2 for x in values) / (n - 1)


def _std(values: Sequence[float]) -> float:
    """Compute sample standard deviation."""
    return math.sqrt(_var(values))


def _cohens_d_one_sample(mean: float, mu0: float, std: float) -> float:
    """Cohen's d effect size for a one-sample t-test.

    Args:
        mean: Sample mean.
        mu0: Hypothesised population mean.
        std: Sample standard deviation.

    Returns:
        Cohen's d, or 0.0 when std is zero.
    """
    return (mean - mu0) / std if std > 0 else 0.0


def _cohens_d_two_sample(
    mean1: float,
    mean2: float,
    std1: float,
    std2: float,
    n1: int,
    n2: int,
) -> float:
    """Pooled Cohen's d for a two-sample test.

    Args:
        mean1: Mean of group 1.
        mean2: Mean of group 2.
        std1: Std dev of group 1.
        std2: Std dev of group 2.
        n1: Size of group 1.
        n2: Size of group 2.

    Returns:
        Cohen's d using pooled standard deviation, or 0.0 if pooled sd is zero.
    """
    pooled_var = ((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2)
    pooled_sd = math.sqrt(pooled_var)
    return (mean1 - mean2) / pooled_sd if pooled_sd > 0 else 0.0


# ---------------------------------------------------------------------------
# One-sample t-test
# ---------------------------------------------------------------------------


class OneSampleResult:
    """Result container for a one-sample t-test."""

    def __init__(
        self,
        t_stat: float,
        df: float,
        p_value: float,
        mean: float,
        std: float,
        n: int,
        mu0: float,
        ci_lower: float,
        ci_upper: float,
        cohens_d: float,
        alpha: float,
        sided: str,
    ) -> None:
        """Store all computed quantities."""
        self.t_stat = t_stat
        self.df = df
        self.p_value = p_value
        self.mean = mean
        self.std = std
        self.n = n
        self.mu0 = mu0
        self.ci_lower = ci_lower
        self.ci_upper = ci_upper
        self.cohens_d = cohens_d
        self.alpha = alpha
        self.sided = sided


def one_sample_t_test(
    mean: float,
    std: float,
    n: int,
    mu0: float,
    alpha: float = 0.05,
    sided: str = "two",
) -> OneSampleResult:
    """Perform a one-sample t-test.

    Tests H₀: μ = mu0 against the alternative specified by ``sided``.

    Args:
        mean: Sample mean.
        std: Sample standard deviation (must be >= 0).
        n: Sample size (must be >= 2).
        mu0: Hypothesised population mean.
        alpha: Significance level for confidence interval (default 0.05).
        sided: ``"two"``, ``"less"``, or ``"greater"``.

    Returns:
        :class:`OneSampleResult` with t-statistic, df, p-value, CI, and Cohen's d.

    Raises:
        ValueError: On invalid inputs or unsupported ``sided`` value.
    """
    if n < 2:
        raise ValueError(f"Sample size must be >= 2, got {n}")
    if std < 0:
        raise ValueError(f"Standard deviation must be non-negative, got {std}")
    if not (0 < alpha < 1):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if sided not in ("two", "less", "greater"):
        raise ValueError(f"sided must be 'two', 'less', or 'greater', got {sided!r}")

    df = n - 1

    if std == 0:
        t_stat = 0.0 if mean == mu0 else math.copysign(math.inf, mean - mu0)
        p_value = 1.0 if mean == mu0 else 0.0
        ci_lower = ci_upper = mean
    else:
        se = std / math.sqrt(n)
        t_stat = (mean - mu0) / se
        p_value = _p_value(t_stat, df, sided)
        # CI is always two-sided at the given alpha
        t_crit = float(t_dist.ppf(1 - alpha / 2, df))
        ci_lower = mean - t_crit * se
        ci_upper = mean + t_crit * se

    cohens_d = _cohens_d_one_sample(mean, mu0, std)

    return OneSampleResult(
        t_stat=t_stat,
        df=df,
        p_value=p_value,
        mean=mean,
        std=std,
        n=n,
        mu0=mu0,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        cohens_d=cohens_d,
        alpha=alpha,
        sided=sided,
    )


# ---------------------------------------------------------------------------
# Two-sample t-test (Welch's)
# ---------------------------------------------------------------------------


class TwoSampleResult:
    """Result container for a two-sample Welch's t-test."""

    def __init__(
        self,
        t_stat: float,
        df: float,
        p_value: float,
        mean1: float,
        mean2: float,
        std1: float,
        std2: float,
        n1: int,
        n2: int,
        ci_lower: float,
        ci_upper: float,
        cohens_d: float,
        alpha: float,
        sided: str,
    ) -> None:
        """Store all computed quantities."""
        self.t_stat = t_stat
        self.df = df
        self.p_value = p_value
        self.mean1 = mean1
        self.mean2 = mean2
        self.std1 = std1
        self.std2 = std2
        self.n1 = n1
        self.n2 = n2
        self.ci_lower = ci_lower
        self.ci_upper = ci_upper
        self.cohens_d = cohens_d
        self.alpha = alpha
        self.sided = sided


def two_sample_t_test(
    mean1: float,
    std1: float,
    n1: int,
    mean2: float,
    std2: float,
    n2: int,
    alpha: float = 0.05,
    sided: str = "two",
) -> TwoSampleResult:
    """Perform a two-sample Welch's t-test.

    Tests H₀: μ₁ = μ₂.  Uses Welch-Satterthwaite degrees of freedom so equal
    variances are not assumed.

    Args:
        mean1: Mean of sample 1.
        std1: Std dev of sample 1 (>= 0).
        n1: Size of sample 1 (>= 2).
        mean2: Mean of sample 2.
        std2: Std dev of sample 2 (>= 0).
        n2: Size of sample 2 (>= 2).
        alpha: Significance level (default 0.05).
        sided: ``"two"``, ``"less"``, or ``"greater"``.

    Returns:
        :class:`TwoSampleResult` with t-statistic, Welch df, p-value, CI, Cohen's d.

    Raises:
        ValueError: On invalid inputs.
    """
    for label, n, std in (("n1", n1, std1), ("n2", n2, std2)):
        if n < 2:
            raise ValueError(f"{label} must be >= 2, got {n}")
        if std < 0:
            raise ValueError(f"std for {label} must be non-negative, got {std}")
    if not (0 < alpha < 1):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if sided not in ("two", "less", "greater"):
        raise ValueError(f"sided must be 'two', 'less', or 'greater', got {sided!r}")

    diff = mean1 - mean2
    var1_n = std1**2 / n1
    var2_n = std2**2 / n2
    se = math.sqrt(var1_n + var2_n)

    if se == 0:
        t_stat = 0.0 if diff == 0 else math.copysign(math.inf, diff)
        df = float(n1 + n2 - 2)
        p_value = 1.0 if diff == 0 else 0.0
        ci_lower = ci_upper = diff
    else:
        t_stat = diff / se
        # Welch-Satterthwaite degrees of freedom
        df = (var1_n + var2_n) ** 2 / (var1_n**2 / (n1 - 1) + var2_n**2 / (n2 - 1))
        p_value = _p_value(t_stat, df, sided)
        t_crit = float(t_dist.ppf(1 - alpha / 2, df))
        ci_lower = diff - t_crit * se
        ci_upper = diff + t_crit * se

    cohens_d = _cohens_d_two_sample(mean1, mean2, std1, std2, n1, n2)

    return TwoSampleResult(
        t_stat=t_stat,
        df=df,
        p_value=p_value,
        mean1=mean1,
        mean2=mean2,
        std1=std1,
        std2=std2,
        n1=n1,
        n2=n2,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        cohens_d=cohens_d,
        alpha=alpha,
        sided=sided,
    )


# ---------------------------------------------------------------------------
# Paired t-test
# ---------------------------------------------------------------------------


class PairedResult:
    """Result container for a paired t-test."""

    def __init__(
        self,
        t_stat: float,
        df: float,
        p_value: float,
        mean_diff: float,
        std_diff: float,
        n: int,
        ci_lower: float,
        ci_upper: float,
        cohens_d: float,
        alpha: float,
        sided: str,
    ) -> None:
        """Store all computed quantities."""
        self.t_stat = t_stat
        self.df = df
        self.p_value = p_value
        self.mean_diff = mean_diff
        self.std_diff = std_diff
        self.n = n
        self.ci_lower = ci_lower
        self.ci_upper = ci_upper
        self.cohens_d = cohens_d
        self.alpha = alpha
        self.sided = sided


def paired_t_test(
    values1: Sequence[float],
    values2: Sequence[float],
    alpha: float = 0.05,
    sided: str = "two",
) -> PairedResult:
    """Perform a paired t-test on matched observations.

    Reduces to a one-sample t-test on the pairwise differences (d = x1 - x2),
    testing H₀: μ_d = 0.

    Args:
        values1: First sequence of observations.
        values2: Second sequence of observations (same length as ``values1``).
        alpha: Significance level (default 0.05).
        sided: ``"two"``, ``"less"``, or ``"greater"``.

    Returns:
        :class:`PairedResult` computed on the differences.

    Raises:
        ValueError: If sequences have different lengths or fewer than 2 pairs.
    """
    if len(values1) != len(values2):
        raise ValueError(
            f"values1 and values2 must have the same length, "
            f"got {len(values1)} and {len(values2)}"
        )
    if len(values1) < 2:
        raise ValueError(f"Need at least 2 pairs, got {len(values1)}")

    diffs = [a - b for a, b in zip(values1, values2)]
    mean_diff = _mean(diffs)
    std_diff = _std(diffs)
    n = len(diffs)

    result = one_sample_t_test(
        mean_diff, std_diff, n, mu0=0.0, alpha=alpha, sided=sided
    )

    return PairedResult(
        t_stat=result.t_stat,
        df=result.df,
        p_value=result.p_value,
        mean_diff=mean_diff,
        std_diff=std_diff,
        n=n,
        ci_lower=result.ci_lower,
        ci_upper=result.ci_upper,
        cohens_d=result.cohens_d,
        alpha=alpha,
        sided=sided,
    )


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _p_value(t_stat: float, df: float, sided: str) -> float:
    """Compute the p-value for a t-statistic.

    Args:
        t_stat: Observed t-statistic.
        df: Degrees of freedom.
        sided: ``"two"``, ``"less"``, or ``"greater"``.

    Returns:
        p-value clipped to [0, 1].
    """
    if sided == "two":
        p = 2 * float(t_dist.sf(abs(t_stat), df))
    elif sided == "less":
        p = float(t_dist.cdf(t_stat, df))
    else:
        p = float(t_dist.sf(t_stat, df))
    return max(0.0, min(1.0, p))


def parse_number_list(raw: str) -> list[float]:
    """Parse a comma-separated string of finite numbers.

    Args:
        raw: Comma-separated numeric string (e.g. ``"1.0,2.5,3.1"``).

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
        description="t-test calculator (one-sample, two-sample, paired).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  ttest one-sample --values 2.1,3.4,2.9,3.1,2.8 --mu0 3.0
  ttest one-sample --mean 105 --std 12 --n 25 --mu0 100
  ttest two-sample --values1 1.2,2.3,3.1,2.8 --values2 2.1,3.2,4.1,3.9
  ttest paired --values1 85,90,78,92,88 --values2 90,95,82,95,91
""",
    )

    sub = parser.add_subparsers(dest="mode", metavar="MODE")
    sub.required = True

    # ---- one-sample ----
    p1 = sub.add_parser("one-sample", help="one-sample t-test (H₀: μ = mu0)")
    _add_single_group(p1)
    p1.add_argument(
        "--mu0",
        type=float,
        default=None,
        metavar="MU0",
        help="hypothesised population mean",
    )
    _add_common(p1)

    # ---- two-sample ----
    p2 = sub.add_parser("two-sample", help="two-sample Welch's t-test (H₀: μ₁ = μ₂)")
    p2.add_argument(
        "--values1",
        "-v1",
        type=str,
        metavar="X1,X2,...",
        help="comma-separated values for group 1",
    )
    p2.add_argument(
        "--values2",
        "-v2",
        type=str,
        metavar="X1,X2,...",
        help="comma-separated values for group 2",
    )
    p2.add_argument(
        "--mean1",
        type=float,
        metavar="MEAN1",
        help="mean of group 1 (use with --std1 --n1)",
    )
    p2.add_argument("--std1", type=float, metavar="STD1", help="std dev of group 1")
    p2.add_argument("--n1", type=int, metavar="N1", help="size of group 1")
    p2.add_argument(
        "--mean2",
        type=float,
        metavar="MEAN2",
        help="mean of group 2 (use with --std2 --n2)",
    )
    p2.add_argument("--std2", type=float, metavar="STD2", help="std dev of group 2")
    p2.add_argument("--n2", type=int, metavar="N2", help="size of group 2")
    _add_common(p2)

    # ---- paired ----
    p3 = sub.add_parser("paired", help="paired t-test (H₀: μ_d = 0)")
    p3.add_argument(
        "--values1",
        "-v1",
        type=str,
        required=True,
        metavar="X1,X2,...",
        help="comma-separated pre/group-1 values",
    )
    p3.add_argument(
        "--values2",
        "-v2",
        type=str,
        required=True,
        metavar="X1,X2,...",
        help="comma-separated post/group-2 values",
    )
    _add_common(p3)

    return parser.parse_args(argv)


def _add_single_group(p: argparse.ArgumentParser) -> None:
    """Attach raw-values or summary-stats arguments to a subparser."""
    group = p.add_mutually_exclusive_group()
    group.add_argument(
        "--values",
        "-v",
        type=str,
        metavar="X1,X2,...",
        help="comma-separated sample values",
    )
    group.add_argument(
        "--mean",
        type=float,
        metavar="MEAN",
        help="sample mean (use with --std and --n)",
    )
    p.add_argument(
        "--std",
        type=float,
        metavar="STD",
        help="sample standard deviation (use with --mean and --n)",
    )
    p.add_argument(
        "--n", type=int, metavar="N", help="sample size (use with --mean and --std)"
    )


def _add_common(p: argparse.ArgumentParser) -> None:
    """Attach shared alpha / sided / precision arguments."""
    p.add_argument(
        "--alpha",
        "-a",
        type=float,
        default=0.05,
        metavar="ALPHA",
        help="significance level (default: 0.05)",
    )
    p.add_argument(
        "--sided",
        "-S",
        choices=["two", "less", "greater"],
        default="two",
        help="alternative hypothesis (default: two)",
    )
    p.add_argument(
        "--precision",
        "-P",
        type=int,
        default=4,
        metavar="PREC",
        help="decimal places for output (default: 4)",
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


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

    if args.mode == "one-sample":
        return _validate_one_sample(args)

    if args.mode == "two-sample":
        return _validate_two_sample(args)

    # paired: argparse already enforces --values1/--values2 as required
    return None


def _validate_one_sample(args: argparse.Namespace) -> str | None:
    """Validate one-sample subcommand arguments."""
    if args.mu0 is None:
        return "--mu0 is required for one-sample t-test"
    if args.values is None:
        # Summary stats path
        if args.mean is None or args.std is None or args.n is None:
            return "provide either --values or all of --mean, --std, and --n"
        if not math.isfinite(args.mean):
            return "--mean must be finite"
        if args.std < 0:
            return "--std must be non-negative"
        if args.n < 2:
            return "--n must be >= 2"
    if not math.isfinite(args.mu0):
        return "--mu0 must be finite"
    return None


def _validate_two_sample(args: argparse.Namespace) -> str | None:
    """Validate two-sample subcommand arguments."""
    using_raw1 = args.values1 is not None
    using_raw2 = args.values2 is not None
    using_summary1 = (
        args.mean1 is not None or args.std1 is not None or args.n1 is not None
    )
    using_summary2 = (
        args.mean2 is not None or args.std2 is not None or args.n2 is not None
    )

    if using_raw1 and using_summary1:
        return "provide either --values1 or (--mean1, --std1, --n1), not both"
    if using_raw2 and using_summary2:
        return "provide either --values2 or (--mean2, --std2, --n2), not both"

    if not using_raw1 and not using_summary1:
        return "group 1: provide --values1 or all of --mean1, --std1, --n1"
    if not using_raw2 and not using_summary2:
        return "group 2: provide --values2 or all of --mean2, --std2, --n2"

    if using_summary1 and not (
        args.mean1 is not None and args.std1 is not None and args.n1 is not None
    ):
        return "group 1 summary stats: provide all of --mean1, --std1, --n1"
    if using_summary2 and not (
        args.mean2 is not None and args.std2 is not None and args.n2 is not None
    ):
        return "group 2 summary stats: provide all of --mean2, --std2, --n2"

    if using_summary1:
        if not math.isfinite(args.mean1):
            return "--mean1 must be finite"
        if args.std1 < 0:
            return "--std1 must be non-negative"
        if args.n1 < 2:
            return "--n1 must be >= 2"
    if using_summary2:
        if not math.isfinite(args.mean2):
            return "--mean2 must be finite"
        if args.std2 < 0:
            return "--std2 must be non-negative"
        if args.n2 < 2:
            return "--n2 must be >= 2"

    return None


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def _fmt(value: float, precision: int) -> str:
    """Format a float to the requested number of decimal places."""
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return f"{value:.{precision}f}"


def _sided_label(sided: str) -> str:
    """Return a human-readable label for the test direction."""
    return {
        "two": "two-sided",
        "less": "one-sided (less)",
        "greater": "one-sided (greater)",
    }[sided]


def _decision(p_value: float, alpha: float) -> str:
    """Return reject / fail-to-reject decision string."""
    if p_value < alpha:
        return f"Reject H₀  (p < α = {alpha})"
    return f"Fail to reject H₀  (p ≥ α = {alpha})"


def format_one_sample(result: OneSampleResult, precision: int) -> str:
    """Format one-sample t-test result for display.

    Args:
        result: Computed :class:`OneSampleResult`.
        precision: Decimal places for all floating-point output.

    Returns:
        Multi-line string ready to print.
    """
    f = lambda v: _fmt(v, precision)  # noqa: E731
    lines = [
        f"One-sample t-test  ({_sided_label(result.sided)})",
        f"H₀: μ = {f(result.mu0)}",
        "",
        f"  n:          {result.n}",
        f"  mean:       {f(result.mean)}",
        f"  std dev:    {f(result.std)}",
        "",
        f"  t-stat:     {f(result.t_stat)}",
        f"  df:         {f(result.df)}",
        f"  p-value:    {f(result.p_value)}",
        f"  {int((1 - result.alpha) * 100)}% CI:    [{f(result.ci_lower)}, {f(result.ci_upper)}]",
        f"  Cohen's d:  {f(result.cohens_d)}",
        "",
        f"  {_decision(result.p_value, result.alpha)}",
    ]
    return "\n".join(lines)


def format_two_sample(result: TwoSampleResult, precision: int) -> str:
    """Format two-sample Welch's t-test result for display.

    Args:
        result: Computed :class:`TwoSampleResult`.
        precision: Decimal places.

    Returns:
        Multi-line string ready to print.
    """
    f = lambda v: _fmt(v, precision)  # noqa: E731
    lines = [
        f"Two-sample Welch's t-test  ({_sided_label(result.sided)})",
        "H₀: μ₁ = μ₂",
        "",
        f"  Group 1:  n={result.n1},  mean={f(result.mean1)},  std={f(result.std1)}",
        f"  Group 2:  n={result.n2},  mean={f(result.mean2)},  std={f(result.std2)}",
        f"  diff (μ₁ − μ₂):  {f(result.mean1 - result.mean2)}",
        "",
        f"  t-stat:     {f(result.t_stat)}",
        f"  df:         {f(result.df)}",
        f"  p-value:    {f(result.p_value)}",
        f"  {int((1 - result.alpha) * 100)}% CI:    [{f(result.ci_lower)}, {f(result.ci_upper)}]",
        f"  Cohen's d:  {f(result.cohens_d)}",
        "",
        f"  {_decision(result.p_value, result.alpha)}",
    ]
    return "\n".join(lines)


def format_paired(result: PairedResult, precision: int) -> str:
    """Format paired t-test result for display.

    Args:
        result: Computed :class:`PairedResult`.
        precision: Decimal places.

    Returns:
        Multi-line string ready to print.
    """
    f = lambda v: _fmt(v, precision)  # noqa: E731
    lines = [
        f"Paired t-test  ({_sided_label(result.sided)})",
        "H₀: μ_d = 0  (d = x₁ − x₂)",
        "",
        f"  pairs:         {result.n}",
        f"  mean diff:     {f(result.mean_diff)}",
        f"  std diff:      {f(result.std_diff)}",
        "",
        f"  t-stat:        {f(result.t_stat)}",
        f"  df:            {f(result.df)}",
        f"  p-value:       {f(result.p_value)}",
        f"  {int((1 - result.alpha) * 100)}% CI (diff): [{f(result.ci_lower)}, {f(result.ci_upper)}]",
        f"  Cohen's d:     {f(result.cohens_d)}",
        "",
        f"  {_decision(result.p_value, result.alpha)}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Run the t-test CLI.

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

    precision = args.precision

    try:
        if args.mode == "one-sample":
            if args.values is not None:
                vals = parse_number_list(args.values)
                mean_v = _mean(vals)
                std_v = _std(vals)
                n_v = len(vals)
            else:
                mean_v, std_v, n_v = args.mean, args.std, args.n

            result_1 = one_sample_t_test(
                mean_v, std_v, n_v, args.mu0, args.alpha, args.sided
            )
            print(format_one_sample(result_1, precision))

        elif args.mode == "two-sample":
            if args.values1 is not None:
                v1 = parse_number_list(args.values1)
                m1, s1, n1 = _mean(v1), _std(v1), len(v1)
            else:
                m1, s1, n1 = args.mean1, args.std1, args.n1

            if args.values2 is not None:
                v2 = parse_number_list(args.values2)
                m2, s2, n2 = _mean(v2), _std(v2), len(v2)
            else:
                m2, s2, n2 = args.mean2, args.std2, args.n2

            result_2 = two_sample_t_test(m1, s1, n1, m2, s2, n2, args.alpha, args.sided)
            print(format_two_sample(result_2, precision))

        else:  # paired
            v1 = parse_number_list(args.values1)
            v2 = parse_number_list(args.values2)
            result_p = paired_t_test(v1, v2, args.alpha, args.sided)
            print(format_paired(result_p, precision))

    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
