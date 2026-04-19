"""
probability_values.py

p-value and hypothesis test calculator for statistical inference.

Supports:
- Z-test (one-sample proportion, known variance)
- Binomial exact test (proportion, small samples)
- Chi-squared goodness-of-fit test
- t-test (mean with unknown variance)

Usage:
    pvalue --test z --n 1000 --k 480 --p0 0.5
    pvalue --test binom-exact --n 30 --k 22 --p0 0.60
    pvalue --test chi2 --observed 18,22,20,15,25 --expected 20,20,20,20,20
    pvalue --test t --mean 105 --std 12 --n 25 --mu0 100
    pvalue --test z --n 1000 --k 480 --p0 0.5 --sweep-alpha 0.01 0.10 --step 0.01
"""

from __future__ import annotations

import argparse
import math
import sys
from typing import Tuple


def erfinv_approx(x: float) -> float:
    """
    Approximate the inverse error function using a rational approximation.
    Valid for |x| < 1.
    """
    if abs(x) >= 1:
        raise ValueError(f"erfinv input must be in (-1, 1), got {x}")

    a = 0.147
    ln_term = math.log(1 - x * x)
    first_term = 2 / (math.pi * a) + ln_term / 2

    sign = 1 if x >= 0 else -1
    result = sign * math.sqrt(
        math.sqrt(first_term * first_term - ln_term / a) - first_term
    )

    return result


def normal_cdf(z: float) -> float:
    """
    Cumulative distribution function for standard normal distribution.
    P(Z <= z) using error function.
    """
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def normal_quantile(p: float) -> float:
    """
    Approximate the standard normal quantile (inverse CDF) for probability p.
    """
    if p <= 0 or p >= 1:
        raise ValueError(f"Probability must be in (0, 1), got {p}")

    if p < 0.00001:
        return -6.0
    if p > 0.99999:
        return 6.0

    try:
        return math.sqrt(2) * erfinv_approx(2 * p - 1)
    except ValueError:
        return -5.0 if p < 0.5 else 5.0


def t_quantile_approx(p: float, df: int) -> float:
    """
    Approximate the t-distribution quantile.
    """
    if df <= 0:
        raise ValueError(f"Degrees of freedom must be positive, got {df}")

    if df > 1000:
        return normal_quantile(p)

    z = normal_quantile(p)

    # Hill's approximation for t quantile
    g1 = (z**3 + z) / 4
    g2 = (5 * z**5 + 16 * z**3 + 3 * z) / 96
    g3 = (3 * z**7 + 19 * z**5 + 17 * z**3 - 15 * z) / 384
    g4 = (79 * z**9 + 776 * z**7 + 1482 * z**5 - 1920 * z**3 - 945 * z) / 92160

    t = z + g1 / df + g2 / (df**2) + g3 / (df**3) + g4 / (df**4)
    return t


def chi2_cdf_approx(x: float, df: int) -> float:
    """
    Approximate chi-squared CDF using Wilson-Hilferty transformation.
    """
    if x <= 0:
        return 0.0
    if df <= 0:
        raise ValueError(f"Degrees of freedom must be positive, got {df}")

    # Wilson-Hilferty transformation
    w = ((x / df) ** (1 / 3) - (1 - 2 / (9 * df))) / math.sqrt(2 / (9 * df))
    return normal_cdf(w)


def binomial_pmf(n: int, k: int, p: float) -> float:
    """
    Probability mass function P(X = k) for Binomial(n, p).
    """
    if k < 0 or k > n:
        return 0.0
    if p == 0.0:
        return 1.0 if k == 0 else 0.0
    if p == 1.0:
        return 1.0 if k == n else 0.0

    # Log-space calculation
    log_prob = math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
    if k > 0:
        log_prob += k * math.log(p)
    if k < n:
        log_prob += (n - k) * math.log(1 - p)

    if log_prob < -700:
        return 0.0
    elif log_prob > 700:
        return 1.0

    return math.exp(log_prob)


def binomial_cdf_le(n: int, k: int, p: float) -> float:
    """Cumulative distribution P(X <= k) for Binomial(n, p)."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    return sum(binomial_pmf(n, i, p) for i in range(0, k + 1))


def z_test_proportion(
    n: int, k: int, p0: float, sided: str = "two"
) -> Tuple[float, float]:
    """
    Z-test for a single proportion.

    Args:
        n: Sample size
        k: Number of successes
        p0: Null hypothesis proportion
        sided: "two", "less", or "greater"

    Returns:
        (z_statistic, p_value)
    """
    if n <= 0:
        raise ValueError(f"Sample size must be positive, got {n}")
    if k < 0 or k > n:
        raise ValueError(f"Successes k must be in [0, n], got k={k}, n={n}")
    if p0 <= 0 or p0 >= 1:
        raise ValueError(f"Null proportion must be in (0, 1), got {p0}")

    p_hat = k / n
    se = math.sqrt(p0 * (1 - p0) / n)

    if se == 0:
        z_stat = 0.0 if p_hat == p0 else float("inf")
    else:
        z_stat = (p_hat - p0) / se

    # Calculate p-value based on sidedness
    if sided == "two":
        p_value = 2 * (1 - normal_cdf(abs(z_stat)))
    elif sided == "less":
        p_value = normal_cdf(z_stat)
    elif sided == "greater":
        p_value = 1 - normal_cdf(z_stat)
    else:
        raise ValueError(f"sided must be 'two', 'less', or 'greater', got {sided}")

    return z_stat, p_value


def binomial_exact_test(
    n: int, k: int, p0: float, sided: str = "two"
) -> Tuple[float, float]:
    """
    Exact binomial test for a proportion.

    Args:
        n: Sample size
        k: Number of successes
        p0: Null hypothesis proportion
        sided: "two", "less", or "greater"

    Returns:
        (test_statistic, p_value)
        Note: test_statistic is k for consistency with other tests
    """
    if n <= 0:
        raise ValueError(f"Sample size must be positive, got {n}")
    if k < 0 or k > n:
        raise ValueError(f"Successes k must be in [0, n], got k={k}, n={n}")
    if p0 <= 0 or p0 >= 1:
        raise ValueError(f"Null proportion must be in (0, 1), got {p0}")

    if sided == "less":
        p_value = binomial_cdf_le(n, k, p0)
    elif sided == "greater":
        p_value = 1 - binomial_cdf_le(n, k - 1, p0) if k > 0 else 1.0
    elif sided == "two":
        # Two-tailed: sum probabilities of outcomes as or more extreme than observed
        p_k = binomial_pmf(n, k, p0)
        p_value = sum(
            binomial_pmf(n, i, p0)
            for i in range(n + 1)
            if binomial_pmf(n, i, p0) <= p_k + 1e-10
        )
    else:
        raise ValueError(f"sided must be 'two', 'less', or 'greater', got {sided}")

    return float(k), p_value


def chi2_goodness_of_fit(
    observed: list[float], expected: list[float]
) -> Tuple[float, float]:
    """
    Chi-squared goodness-of-fit test.

    Args:
        observed: Observed frequencies
        expected: Expected frequencies under null hypothesis

    Returns:
        (chi2_statistic, p_value)
    """
    if len(observed) != len(expected):
        raise ValueError("Observed and expected must have same length")
    if len(observed) < 2:
        raise ValueError(f"Need at least 2 categories, got {len(observed)}")
    if any(e <= 0 for e in expected):
        raise ValueError("All expected frequencies must be positive")
    if any(o < 0 for o in observed):
        raise ValueError("All observed frequencies must be non-negative")

    chi2_stat = sum((o - e) ** 2 / e for o, e in zip(observed, expected))
    df = len(observed) - 1
    p_value = 1 - chi2_cdf_approx(chi2_stat, df)

    return chi2_stat, p_value


def t_test_mean(
    mean: float, std: float, n: int, mu0: float, sided: str = "two"
) -> Tuple[float, float]:
    """
    One-sample t-test for a mean.

    Args:
        mean: Sample mean
        std: Sample standard deviation
        n: Sample size
        mu0: Null hypothesis mean
        sided: "two", "less", or "greater"

    Returns:
        (t_statistic, p_value)
    """
    if n <= 1:
        raise ValueError(f"Sample size must be > 1, got {n}")
    if std < 0:
        raise ValueError(f"Standard deviation must be non-negative, got {std}")

    if std == 0:
        # With zero variance, test is deterministic
        if mean == mu0:
            t_stat = 0.0
            p_value = 1.0  # Always fail to reject
        else:
            t_stat = float("inf") if mean > mu0 else float("-inf")
            p_value = 0.0  # Always reject
    else:
        se = std / math.sqrt(n)
        t_stat = (mean - mu0) / se

        # df = n - 1

        # Calculate p-value
        if sided == "two":
            # Two-tailed: P(|T| >= |t|)
            p_value = 2 * (1 - normal_cdf(abs(t_stat)))
        elif sided == "less":
            p_value = normal_cdf(t_stat)
        elif sided == "greater":
            p_value = 1 - normal_cdf(t_stat)
        else:
            raise ValueError(f"sided must be 'two', 'less', or 'greater', got {sided}")

    return t_stat, p_value


def cohen_d(p_hat: float, p0: float) -> float:
    """
    Cohen's h effect size for proportions.
    h = 2 * (arcsin(sqrt(p_hat)) - arcsin(sqrt(p0)))
    """
    return 2 * (math.asin(math.sqrt(p_hat)) - math.asin(math.sqrt(p0)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="p-value and hypothesis test calculator"
    )

    parser.add_argument(
        "--test",
        choices=["z", "binom-exact", "chi2", "t"],
        required=True,
        help="Type of hypothesis test",
    )

    # Common parameters
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level (default: 0.05)",
    )

    parser.add_argument(
        "--sided",
        choices=["two", "less", "greater"],
        default="two",
        help="Alternative hypothesis (default: two)",
    )

    # Proportion test parameters (z-test and binomial exact)
    parser.add_argument(
        "--n",
        type=int,
        help="Sample size (for proportion or t-test)",
    )

    parser.add_argument(
        "--k",
        type=int,
        help="Number of successes (for proportion tests)",
    )

    parser.add_argument(
        "--p0",
        type=float,
        help="Null hypothesis proportion (0 < p0 < 1)",
    )

    # t-test parameters
    parser.add_argument(
        "--mean",
        type=float,
        help="Sample mean (for t-test)",
    )

    parser.add_argument(
        "--std",
        type=float,
        help="Sample standard deviation (for t-test)",
    )

    parser.add_argument(
        "--mu0",
        type=float,
        help="Null hypothesis mean (for t-test)",
    )

    # Chi-squared test parameters
    parser.add_argument(
        "--observed",
        type=str,
        help="Observed frequencies (comma-separated, e.g., '18,22,20,15,25')",
    )

    parser.add_argument(
        "--expected",
        type=str,
        help="Expected frequencies (comma-separated, e.g., '20,20,20,20,20')",
    )

    # Sweep mode
    parser.add_argument(
        "--sweep-alpha",
        type=float,
        nargs=2,
        metavar=("START", "END"),
        help="Sweep alpha from START to END",
    )

    parser.add_argument(
        "--step",
        type=float,
        default=0.01,
        help="Step size for alpha sweep (default: 0.01)",
    )

    # Output formatting
    parser.add_argument(
        "--precision",
        type=int,
        default=4,
        help="Decimal places for output (default: 4)",
    )

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> int:
    """Validate command-line arguments. Returns 0 if valid, error code otherwise."""

    if args.test in ["z", "binom-exact"]:
        if args.n is None or args.k is None or args.p0 is None:
            print(f"Error: {args.test} test requires --n, --k, and --p0")
            return 2
        if args.n <= 0:
            print(f"Error: --n must be positive, got {args.n}")
            return 2
        if args.k < 0 or args.k > args.n:
            print(f"Error: --k must be in [0, n], got k={args.k}, n={args.n}")
            return 2
        if args.p0 <= 0 or args.p0 >= 1:
            print(f"Error: --p0 must be in (0, 1), got {args.p0}")
            return 2

    elif args.test == "t":
        if args.n is None or args.mean is None or args.std is None or args.mu0 is None:
            print("Error: t-test requires --n, --mean, --std, and --mu0")
            return 2
        if args.n <= 1:
            print(f"Error: t-test requires n > 1, got n={args.n}")
            return 2
        if args.std < 0:
            print(f"Error: --std must be non-negative, got {args.std}")
            return 2

    elif args.test == "chi2":
        if args.observed is None or args.expected is None:
            print("Error: chi2 test requires --observed and --expected")
            return 2

    if args.alpha <= 0 or args.alpha >= 1:
        print(f"Error: --alpha must be in (0, 1), got {args.alpha}")
        return 2

    if args.sweep_alpha:
        start, end = args.sweep_alpha
        if start <= 0 or start >= 1 or end <= 0 or end >= 1:
            print("Error: sweep-alpha values must be in (0, 1)")
            return 2
        if start >= end:
            print("Error: sweep-alpha START must be < END")
            return 2
        if args.step <= 0:
            print(f"Error: --step must be positive, got {args.step}")
            return 2

    return 0


def main() -> int:
    args = parse_args()

    # Validate arguments
    error_code = validate_args(args)
    if error_code != 0:
        return error_code

    # Initialize variables
    observed = []
    expected = []
    test_name = ""
    stat_name = ""
    test_stat = 0.0
    p_value = 0.0
    effect_size = None

    # Run the test
    try:
        if args.test == "z":
            test_stat, p_value = z_test_proportion(args.n, args.k, args.p0, args.sided)
            test_name = "Z-test"
            stat_name = "z"

            # Calculate effect size
            p_hat = args.k / args.n
            effect_size = cohen_d(p_hat, args.p0)

        elif args.test == "binom-exact":
            test_stat, p_value = binomial_exact_test(
                args.n, args.k, args.p0, args.sided
            )
            test_name = "Binomial Exact Test"
            stat_name = "k"

            # Calculate effect size
            p_hat = args.k / args.n
            effect_size = cohen_d(p_hat, args.p0)

        elif args.test == "t":
            test_stat, p_value = t_test_mean(
                args.mean, args.std, args.n, args.mu0, args.sided
            )
            test_name = "One-sample t-test"
            stat_name = "t"

            # Calculate effect size (Cohen's d for means)
            effect_size = (args.mean - args.mu0) / args.std if args.std > 0 else 0.0

        elif args.test == "chi2":
            # Parse comma-separated values
            observed = [float(x.strip()) for x in args.observed.split(",")]
            expected = [float(x.strip()) for x in args.expected.split(",")]

            test_stat, p_value = chi2_goodness_of_fit(observed, expected)
            test_name = "Chi-squared Goodness-of-Fit"
            stat_name = "χ²"
            effect_size = None  # No standard effect size for chi-squared

    except ValueError as e:
        print(f"Error: {e}")
        return 2

    # Sweep mode
    if args.sweep_alpha:
        start, end = args.sweep_alpha
        print(f"{test_name} - Alpha Sweep")
        print(f"Test statistic ({stat_name}): {test_stat:.{args.precision}f}")
        print(f"p-value: {p_value:.{args.precision}f}")
        print()
        print(f"{'Alpha':>8}  {'Reject H0':>10}")
        print("-" * 20)

        alpha = start
        while alpha <= end + 1e-9:  # Small epsilon for floating point
            reject = "Yes" if p_value < alpha else "No"
            print(f"{alpha:8.{args.precision}f}  {reject:>10}")
            alpha += args.step

        return 0

    # Standard output
    sided_desc = {
        "two": "two-sided",
        "less": "one-sided (less)",
        "greater": "one-sided (greater)",
    }

    print(f"{test_name} ({sided_desc[args.sided]})")
    print(f"Significance level: α = {args.alpha}")
    print()

    # Display inputs based on test type
    if args.test in ["z", "binom-exact"]:
        p_hat = args.k / args.n
        print(f"Sample: n = {args.n}, k = {args.k}, p̂ = {p_hat:.{args.precision}f}")
        print(f"Null hypothesis: H₀: p = {args.p0}")
    elif args.test == "t":
        print(
            f"Sample: n = {args.n}, mean = {args.mean:.{args.precision}f}, std = {args.std:.{args.precision}f}"
        )
        print(f"Null hypothesis: H₀: μ = {args.mu0}")
    elif args.test == "chi2":
        print(f"Observed:  {', '.join(f'{x:.1f}' for x in observed)}")
        print(f"Expected:  {', '.join(f'{x:.1f}' for x in expected)}")
        print(f"df = {len(observed) - 1}")

    print()
    print(f"Test statistic: {stat_name} = {test_stat:.{args.precision}f}")
    print(f"p-value: {p_value:.{args.precision}f}")

    # Display effect size if available
    if effect_size is not None:
        print(f"Effect size: {effect_size:.{args.precision}f}")

    print()

    # Decision
    if p_value < args.alpha:
        print("Decision: Reject H₀ (p < α)")
    else:
        print("Decision: Fail to reject H₀ (p ≥ α)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
