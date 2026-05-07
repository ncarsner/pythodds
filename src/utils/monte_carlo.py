#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import random
import sys
from typing import Optional

import numpy as np
from scipy.special import ndtri as _norm_ppf
from scipy.stats import t as _t_dist

"""Command-line Monte Carlo probability simulator.

Runs repeated random experiments to estimate probabilities empirically,
cross-validating analytical results from binom, birthday, poisson, bayes,
sample_size, t_test, linreg, and pythagorean_record.

Supported experiments:
  binomial    — P(X ≥ k) for Binomial(n, p)
  birthday    — P(at least one collision) for group items drawn from pool
  streak      — P(at least one run of k consecutive successes in n Bernoulli trials)
  poisson     — P(X ≥ k) for Poisson(λ)
  power       — Empirical statistical power for mean or two-proportion tests
  permutation — Distribution-free p-value via sign-flip or shuffle permutation test
  bayes       — Empirical posterior P(A|B) via rejection sampling
  season      — Season win-total distribution or P(wins ≥ k) for a given win probability
  linboot     — Residual bootstrap confidence intervals for linear regression

Usage examples:
  simulate --experiment binomial --params n=10 k=5 p=0.4 --trials 100000
  simulate --experiment birthday --params pool=365 group=23 --confidence
  simulate --experiment streak --params n=100 k=5 p=0.5 --trials 50000
  simulate --experiment poisson --params lam=3.0 k=7 --seed 42
  simulate --experiment power --params type=mean n=30 sigma=10 delta=5
  simulate --experiment permutation --params type=one values=2.1,3.4,2.9,3.1,2.8 mu0=3.0
  simulate --experiment bayes --params prior=0.01 likelihood=0.99 fp=0.05
  simulate --experiment season --params win_pct=0.58 games=162 wins_ge=90
  simulate --experiment linboot --params x=1,2,3,4,5 y=2.1,3.9,6.2,7.8,10.1 predict=6
"""

HAS_NUMPY = True

# ---------------------------------------------------------------------------
# Analytical comparison imports — degrade gracefully if unavailable
# ---------------------------------------------------------------------------

_binom_cdf_ge = None
try:
    from src.utils.binomial_distribution import binomial_cdf_ge as _binom_cdf_ge
except ImportError:  # pragma: no cover
    pass

_collision_prob_uniform = None
try:
    from src.utils.birthday_problem import (
        collision_prob_uniform as _collision_prob_uniform,
    )
except ImportError:  # pragma: no cover
    pass

_poisson_cdf_ge = None
try:
    from src.utils.poisson_distribution import poisson_cdf_ge as _poisson_cdf_ge
except ImportError:  # pragma: no cover
    pass

_achieved_power_mean = None
_achieved_power_comparison = None
try:
    from src.utils.sample_size import (
        achieved_power_comparison as _achieved_power_comparison,
    )
    from src.utils.sample_size import (
        achieved_power_mean as _achieved_power_mean,
    )
except ImportError:  # pragma: no cover
    pass

_linear_regression = None
try:
    from src.utils.linear_regression import linear_regression as _linear_regression
except ImportError:  # pragma: no cover
    pass


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------


def standard_error(p_hat: float, n: int) -> float:
    """SE of a sample proportion."""
    if n <= 0:
        return 0.0
    return math.sqrt(p_hat * (1.0 - p_hat) / n)


def wilson_ci(p_hat: float, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score confidence interval for a proportion at significance level alpha."""
    z = _norm_ppf(1.0 - alpha / 2.0)
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (p_hat + z2 / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(p_hat * (1.0 - p_hat) / n + z2 / (4 * n * n))
    return max(0.0, centre - margin), min(1.0, centre + margin)


def trials_for_scale(target_se: float) -> int:
    """Minimum trial count to achieve worst-case SE ≤ target_se (p=0.5 is worst case)."""
    return math.ceil(0.25 / (target_se**2))


# ---------------------------------------------------------------------------
# Private helpers shared by simulation engines
# ---------------------------------------------------------------------------


def _t_stat_one_sample(values: list[float], mu0: float) -> float:
    """One-sample t-statistic for testing H₀: μ = mu0."""
    n = len(values)
    if n < 2:
        return 0.0
    mean_v = sum(values) / n
    var_v = sum((x - mean_v) ** 2 for x in values) / (n - 1)
    se = math.sqrt(var_v) / math.sqrt(n)
    if se == 0:
        return 0.0 if mean_v == mu0 else math.copysign(math.inf, mean_v - mu0)
    return (mean_v - mu0) / se


def _t_stat_welch(v1: list[float], v2: list[float]) -> float:
    """Welch two-sample t-statistic for testing H₀: μ₁ = μ₂."""
    n1, n2 = len(v1), len(v2)
    if n1 < 2 or n2 < 2:
        return 0.0
    m1 = sum(v1) / n1
    m2 = sum(v2) / n2
    var1 = sum((x - m1) ** 2 for x in v1) / (n1 - 1)
    var2 = sum((x - m2) ** 2 for x in v2) / (n2 - 1)
    se = math.sqrt(var1 / n1 + var2 / n2)
    if se == 0:
        return 0.0
    return (m1 - m2) / se


def _is_extreme(perm_stat: float, obs_stat: float, sided: str) -> int:
    """Return 1 if perm_stat is at least as extreme as obs_stat under H₀."""
    if sided == "two":
        return 1 if abs(perm_stat) >= abs(obs_stat) else 0
    if sided == "less":
        return 1 if perm_stat <= obs_stat else 0
    return 1 if perm_stat >= obs_stat else 0  # "greater"


def _t_pvalue(t_stat: float, df: float, sided: str) -> float:
    """p-value for a t-statistic under Student's t(df)."""
    if sided == "two":
        return float(2 * _t_dist.sf(abs(t_stat), df))
    if sided == "less":
        return float(_t_dist.cdf(t_stat, df))
    return float(_t_dist.sf(t_stat, df))


def _bootstrap_ci(vals: list[float], alpha: float) -> tuple[float, float]:
    """Percentile bootstrap CI from sorted bootstrap distribution."""
    sv = sorted(vals)
    n = len(sv)
    lo = int(alpha / 2 * n)
    hi = max(0, int((1 - alpha / 2) * n) - 1)
    return sv[lo], sv[hi]


# ---------------------------------------------------------------------------
# Simulation engines — original four
# ---------------------------------------------------------------------------


def simulate_binomial(
    n: int, k: int, p: float, trials: int, seed: Optional[int]
) -> list[int]:
    """Per-trial outcomes (1/0) for P(X ≥ k) where X ~ Binomial(n, p)."""
    if HAS_NUMPY:
        rng = np.random.default_rng(seed)
        counts = rng.binomial(n, p, size=trials)
        return (counts >= k).astype(int).tolist()
    else:  # pragma: no cover
        rng = random.Random(seed)
        return [
            1 if sum(1 for _ in range(n) if rng.random() < p) >= k else 0
            for _ in range(trials)
        ]


def simulate_birthday(
    pool: int, group: int, trials: int, seed: Optional[int]
) -> list[int]:
    """Per-trial collision indicators for the birthday problem."""
    if HAS_NUMPY:
        rng = np.random.default_rng(seed)
        samples = rng.integers(0, pool, size=(trials, group))
        sorted_s = np.sort(samples, axis=1)
        collisions = np.any(np.diff(sorted_s, axis=1) == 0, axis=1)
        return collisions.astype(int).tolist()
    else:  # pragma: no cover
        rng = random.Random(seed)
        results = []
        for _ in range(trials):
            draws = [rng.randrange(pool) for _ in range(group)]
            results.append(1 if len(set(draws)) < group else 0)
        return results


def simulate_streak(
    n: int, k: int, p: float, trials: int, seed: Optional[int]
) -> list[int]:
    """Per-trial outcomes (1/0) for P(at least one run of k consecutive successes in n Bernoulli(p) trials)."""
    rng = random.Random(seed)
    results = []
    for _ in range(trials):
        run = 0
        found = False
        for _ in range(n):
            if rng.random() < p:
                run += 1
                if run >= k:
                    found = True
                    break
            else:
                run = 0
        results.append(1 if found else 0)
    return results


def simulate_poisson(lam: float, k: int, trials: int, seed: Optional[int]) -> list[int]:
    """Per-trial outcomes (1/0) for P(X ≥ k) where X ~ Poisson(λ)."""
    if HAS_NUMPY:
        rng = np.random.default_rng(seed)
        counts = rng.poisson(lam, size=trials)
        return (counts >= k).astype(int).tolist()
    else:  # pragma: no cover
        rng = random.Random(seed)
        L = math.exp(-lam)
        results = []
        for _ in range(trials):
            count = 0
            prod = 1.0
            while True:
                prod *= rng.random()
                if prod < L:
                    break
                count += 1
            results.append(1 if count >= k else 0)
        return results


# ---------------------------------------------------------------------------
# Simulation engines — five new experiments
# ---------------------------------------------------------------------------


def simulate_power(
    params: dict[str, str], trials: int, seed: Optional[int]
) -> list[int]:
    """Empirical statistical power: fraction of simulated experiments that reject H₀.

    type=mean        — one-sample t-test: H₀: μ = mu0, H₁: μ = mu0+delta
    type=comparison  — two-proportion z-test: H₀: p₁ = p₂
    """
    test_type = params["type"]
    alpha = float(params.get("alpha", "0.05"))
    sided = params.get("sided", "two")
    rng = np.random.default_rng(seed)

    if test_type == "mean":
        n = int(params["n"])
        sigma = float(params["sigma"])
        delta = float(params["delta"])
        mu0 = float(params.get("mu0", "0"))
        mu1 = mu0 + delta
        df = n - 1

        samples = rng.normal(mu1, sigma, size=(trials, n))
        means = samples.mean(axis=1)
        stds = samples.std(axis=1, ddof=1)
        se = stds / math.sqrt(n)
        t_stats = np.where(se > 0, (means - mu0) / se, 0.0)

        if sided == "two":
            t_crit = _t_dist.ppf(1 - alpha / 2, df)
            rejections = np.abs(t_stats) > t_crit
        else:  # one-sided — reject in the direction of delta
            t_crit = _t_dist.ppf(1 - alpha, df)
            rejections = t_stats > t_crit if delta >= 0 else t_stats < -t_crit

        return rejections.astype(int).tolist()

    # test_type == "comparison"
    n = int(params["n"])
    p1 = float(params["p1"])
    p2 = float(params["p2"])

    counts1 = rng.binomial(n, p1, size=trials)
    counts2 = rng.binomial(n, p2, size=trials)
    p_hat1 = counts1 / n
    p_hat2 = counts2 / n
    p_pool = (counts1 + counts2) / (2 * n)
    se = np.sqrt(p_pool * (1 - p_pool) * (2.0 / n))
    z_stats = np.where(se > 0, (p_hat1 - p_hat2) / se, 0.0)

    z_crit_two = float(_norm_ppf(1 - alpha / 2))
    z_crit_one = float(_norm_ppf(1 - alpha))

    if sided == "two":
        rejections = np.abs(z_stats) > z_crit_two
    elif p1 >= p2:
        rejections = z_stats > z_crit_one
    else:
        rejections = z_stats < -z_crit_one

    return rejections.astype(int).tolist()


def simulate_permutation(
    params: dict[str, str], trials: int, seed: Optional[int]
) -> list[int]:
    """Empirical p-value via permutation test (sign-flip or group shuffle).

    type=one    — sign-flip one-sample test against mu0
    type=two    — pool-and-shuffle two-sample Welch test
    type=paired — sign-flip paired-differences test
    """
    perm_type = params["type"]
    sided = params.get("sided", "two")
    rng = random.Random(seed)

    if perm_type == "one":
        values = [float(v) for v in params["values"].split(",")]
        mu0 = float(params["mu0"])
        obs_t = _t_stat_one_sample(values, mu0)
        diffs = [x - mu0 for x in values]
        n = len(diffs)

        results = []
        for _ in range(trials):
            signs = [rng.choice((-1, 1)) for _ in range(n)]
            perm_vals = [mu0 + d * s for d, s in zip(diffs, signs)]
            perm_t = _t_stat_one_sample(perm_vals, mu0)
            results.append(_is_extreme(perm_t, obs_t, sided))
        return results

    if perm_type == "paired":
        v1 = [float(v) for v in params["values1"].split(",")]
        v2 = [float(v) for v in params["values2"].split(",")]
        diffs = [a - b for a, b in zip(v1, v2)]
        obs_t = _t_stat_one_sample(diffs, 0.0)
        n = len(diffs)

        results = []
        for _ in range(trials):
            signs = [rng.choice((-1, 1)) for _ in range(n)]
            perm_diffs = [d * s for d, s in zip(diffs, signs)]
            perm_t = _t_stat_one_sample(perm_diffs, 0.0)
            results.append(_is_extreme(perm_t, obs_t, sided))
        return results

    # perm_type == "two"
    v1 = [float(v) for v in params["values1"].split(",")]
    v2 = [float(v) for v in params["values2"].split(",")]
    obs_t = _t_stat_welch(v1, v2)
    pooled = v1 + v2
    n1 = len(v1)

    results = []
    for _ in range(trials):
        shuffled = pooled[:]
        rng.shuffle(shuffled)
        perm_t = _t_stat_welch(shuffled[:n1], shuffled[n1:])
        results.append(_is_extreme(perm_t, obs_t, sided))
    return results


def simulate_bayes(
    params: dict[str, str], trials: int, seed: Optional[int]
) -> list[int]:
    """Estimate P(A=1|B=1) by rejection-sampling joint (A, B) draws.

    Generates a large batch of (A, B) pairs; retains only the B=1 rows;
    returns the first `trials` A values conditioned on B=1.
    """
    prior = float(params["prior"])
    likelihood = float(params["likelihood"])
    fp = float(params["fp"])

    p_b = likelihood * prior + fp * (1.0 - prior)

    # Generate ~10× expected draws needed to almost certainly collect `trials` B=1 events.
    n_raw = min(max(int(math.ceil(trials / p_b * 10)), trials * 10), 10_000_000)
    rng = np.random.default_rng(seed)

    a = rng.random(n_raw) < prior
    b_given_a1 = rng.random(n_raw) < likelihood
    b_given_a0 = rng.random(n_raw) < fp
    b = np.where(a, b_given_a1, b_given_a0)

    a_given_b = a[b].astype(int)

    if len(a_given_b) >= trials:
        return a_given_b[:trials].tolist()

    # Pad with analytically-derived draws when the batch was insufficient.
    post = (likelihood * prior) / p_b
    pad_rng = np.random.default_rng((seed + 1) if seed is not None else None)
    pad = (pad_rng.random(trials - len(a_given_b)) < post).astype(int)
    return a_given_b.tolist() + pad.tolist()


def simulate_season(
    params: dict[str, str], trials: int, seed: Optional[int]
) -> list[int]:
    """Simulate season win totals for a team with a fixed per-game win probability.

    Returns binary list (wins ≥ wins_ge?) when wins_ge is provided, or raw
    win-count integers otherwise.
    """
    win_pct = float(params["win_pct"])
    games = int(params.get("games", "162"))
    wins_ge = params.get("wins_ge")

    if HAS_NUMPY:
        rng = np.random.default_rng(seed)
        win_counts = rng.binomial(games, win_pct, size=trials)
        if wins_ge is not None:
            return (win_counts >= int(wins_ge)).astype(int).tolist()
        return win_counts.tolist()
    else:  # pragma: no cover
        rng = random.Random(seed)
        win_counts = [
            sum(1 for _ in range(games) if rng.random() < win_pct)
            for _ in range(trials)
        ]
        if wins_ge is not None:
            k = int(wins_ge)
            return [1 if w >= k else 0 for w in win_counts]
        return win_counts


def simulate_linboot(
    params: dict[str, str],
    trials: int,
    seed: Optional[int],
) -> tuple[list[float], list[float], list[float]]:
    """Residual bootstrap for linear regression.

    Resamples OLS residuals with replacement, refits the model on each
    bootstrap dataset, and collects the distribution of slope, intercept,
    and (optionally) the prediction at a given x.

    Returns:
        (slopes, intercepts, predictions) — lists of length ≤ trials.
    """
    if _linear_regression is None:  # pragma: no cover
        raise RuntimeError("linear_regression module not available")

    x = [float(v) for v in params["x"].split(",")]
    y = [float(v) for v in params["y"].split(",")]
    predict_x = float(params["predict"]) if "predict" in params else None

    model = _linear_regression(x, y)
    fitted = [model.slope * xi + model.intercept for xi in x]
    residuals = [yi - fi for yi, fi in zip(y, fitted)]

    rng = random.Random(seed)
    slopes: list[float] = []
    intercepts: list[float] = []
    predictions: list[float] = []

    for _ in range(trials):
        resampled = [rng.choice(residuals) for _ in range(len(residuals))]
        y_boot = [fi + ri for fi, ri in zip(fitted, resampled)]
        try:
            boot = _linear_regression(x, y_boot)
        except ValueError:
            continue
        slopes.append(boot.slope)
        intercepts.append(boot.intercept)
        if predict_x is not None:
            predictions.append(boot.slope * predict_x + boot.intercept)

    return slopes, intercepts, predictions


# ---------------------------------------------------------------------------
# Dispatch and analytical comparison
# ---------------------------------------------------------------------------

REQUIRED_PARAMS: dict[str, list[str]] = {
    "binomial": ["n", "k", "p"],
    "birthday": ["pool", "group"],
    "streak": ["n", "k", "p"],
    "poisson": ["lam", "k"],
    "power": ["type"],
    "permutation": ["type"],
    "bayes": ["prior", "likelihood", "fp"],
    "season": ["win_pct"],
    "linboot": ["x", "y"],
}


def run_experiment(
    experiment: str, params: dict[str, str], trials: int, seed: Optional[int]
) -> list[int]:
    if experiment == "binomial":
        return simulate_binomial(
            int(params["n"]), int(params["k"]), float(params["p"]), trials, seed
        )
    if experiment == "birthday":
        return simulate_birthday(
            int(params["pool"]), int(params["group"]), trials, seed
        )
    if experiment == "streak":
        return simulate_streak(
            int(params["n"]), int(params["k"]), float(params["p"]), trials, seed
        )
    if experiment == "poisson":
        return simulate_poisson(float(params["lam"]), int(params["k"]), trials, seed)
    if experiment == "power":
        return simulate_power(params, trials, seed)
    if experiment == "permutation":
        return simulate_permutation(params, trials, seed)
    if experiment == "bayes":
        return simulate_bayes(params, trials, seed)
    if experiment == "season":
        return simulate_season(params, trials, seed)
    raise ValueError(f"Unknown experiment: {experiment!r}")  # pragma: no cover


def analytical_value(experiment: str, params: dict[str, str]) -> Optional[float]:
    """Return the exact analytical value for comparison, or None."""
    if experiment == "binomial" and _binom_cdf_ge is not None:
        return _binom_cdf_ge(int(params["n"]), int(params["k"]), float(params["p"]))
    if experiment == "birthday" and _collision_prob_uniform is not None:
        return _collision_prob_uniform(int(params["group"]), int(params["pool"]))
    if experiment == "poisson" and _poisson_cdf_ge is not None:
        return _poisson_cdf_ge(int(params["k"]), float(params["lam"]))

    if experiment == "season" and "wins_ge" in params and _binom_cdf_ge is not None:
        games = int(params.get("games", "162"))
        return _binom_cdf_ge(games, int(params["wins_ge"]), float(params["win_pct"]))

    if experiment == "bayes":
        prior = float(params["prior"])
        likelihood = float(params["likelihood"])
        fp = float(params["fp"])
        p_b = likelihood * prior + fp * (1.0 - prior)
        if p_b > 0:
            return (likelihood * prior) / p_b

    if experiment == "power":
        test_type = params.get("type", "")
        alpha = float(params.get("alpha", "0.05"))
        sided = params.get("sided", "two")
        try:
            if test_type == "mean" and _achieved_power_mean is not None:
                return _achieved_power_mean(
                    int(params["n"]),
                    float(params["sigma"]),
                    float(params["delta"]),
                    alpha,
                    sided,
                )
            if test_type == "comparison" and _achieved_power_comparison is not None:
                return _achieved_power_comparison(
                    int(params["n"]),
                    float(params["p1"]),
                    float(params["p2"]),
                    alpha,
                    sided,
                )
        except (KeyError, ValueError):
            pass

    if experiment == "permutation":
        perm_type = params.get("type", "")
        sided = params.get("sided", "two")
        try:
            if perm_type == "one":
                values = [float(v) for v in params["values"].split(",")]
                mu0 = float(params["mu0"])
                obs_t = _t_stat_one_sample(values, mu0)
                return _t_pvalue(obs_t, len(values) - 1, sided)
            if perm_type == "two":
                v1 = [float(v) for v in params["values1"].split(",")]
                v2 = [float(v) for v in params["values2"].split(",")]
                obs_t = _t_stat_welch(v1, v2)
                n1, n2 = len(v1), len(v2)
                m1 = sum(v1) / n1
                m2 = sum(v2) / n2
                var1 = sum((x - m1) ** 2 for x in v1) / (n1 - 1) if n1 > 1 else 0.0
                var2 = sum((x - m2) ** 2 for x in v2) / (n2 - 1) if n2 > 1 else 0.0
                vn1, vn2 = var1 / n1, var2 / n2
                denom = vn1**2 / (n1 - 1) + vn2**2 / (n2 - 1)
                df = (vn1 + vn2) ** 2 / denom if denom > 0 else n1 + n2 - 2
                return _t_pvalue(obs_t, df, sided)
            if perm_type == "paired":
                v1 = [float(v) for v in params["values1"].split(",")]
                v2 = [float(v) for v in params["values2"].split(",")]
                diffs = [a - b for a, b in zip(v1, v2)]
                obs_t = _t_stat_one_sample(diffs, 0.0)
                return _t_pvalue(obs_t, len(diffs) - 1, sided)
        except (KeyError, ValueError, ZeroDivisionError):
            pass

    return None


# ---------------------------------------------------------------------------
# Output formatters
# ---------------------------------------------------------------------------


def _fmt(v: float, precision: int) -> str:
    return f"{v:.{precision}f}"


def format_table(
    experiment: str,
    params: dict[str, str],
    trials: int,
    p_hat: float,
    se: float,
    ci: Optional[tuple[float, float]],
    analytical: Optional[float],
    precision: int,
    *,
    prob_label: str = "Estimated probability",
    analytical_label: str = "Analytical value",
) -> str:
    lines = [
        f"Experiment:              {experiment}",
        f"Parameters:              {', '.join(f'{k}={v}' for k, v in params.items())}",
        f"Trials:                  {trials:,}",
        f"{prob_label}:   {_fmt(p_hat, precision)}",
        f"Standard error:          {_fmt(se, precision)}",
    ]
    if ci is not None:
        lines.append(
            f"95% confidence interval: [{_fmt(ci[0], precision)}, {_fmt(ci[1], precision)}]"
        )
    if analytical is not None:
        lines.append(f"{analytical_label}:        {_fmt(analytical, precision)}")
        lines.append(f"Difference (sim-exact):  {p_hat - analytical:+.{precision}f}")
    return "\n".join(lines)


def format_dump_csv(results: list[int]) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["trial", "outcome"])
    for i, v in enumerate(results, 1):
        writer.writerow([i, v])
    return buf.getvalue()


def format_season_summary(
    params: dict[str, str],
    win_counts: list[int],
    trials: int,
    precision: int,
) -> str:
    """Distribution summary for season win-count simulation (no wins_ge threshold)."""
    win_pct = float(params["win_pct"])
    games = int(params.get("games", "162"))
    n = len(win_counts)

    mean_wins = sum(win_counts) / n
    var_wins = sum((w - mean_wins) ** 2 for w in win_counts) / n
    std_wins = math.sqrt(var_wins)
    sorted_c = sorted(win_counts)

    def pctile(p: float) -> float:
        idx = p * (n - 1)
        lo = int(idx)
        hi = min(lo + 1, n - 1)
        return sorted_c[lo] + (idx - lo) * (sorted_c[hi] - sorted_c[lo])

    lines = [
        "Experiment:          season",
        f"Win probability:     {_fmt(win_pct, precision)}",
        f"Games per season:    {games}",
        f"Trials:              {trials:,}",
        "",
        f"Mean wins:           {_fmt(mean_wins, precision)}",
        f"Std dev:             {_fmt(std_wins, precision)}",
        "",
        "Percentiles:",
        f"  10th: {pctile(0.10):.1f}",
        f"  25th: {pctile(0.25):.1f}",
        f"  50th (median): {pctile(0.50):.1f}",
        f"  75th: {pctile(0.75):.1f}",
        f"  90th: {pctile(0.90):.1f}",
    ]
    return "\n".join(lines)


def _season_dump_csv(win_counts: list[int]) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["trial", "wins"])
    for i, w in enumerate(win_counts, 1):
        writer.writerow([i, w])
    return buf.getvalue()


def _season_json(
    params: dict[str, str],
    win_counts: list[int],
    trials: int,
    precision: int,
) -> dict:
    win_pct = float(params["win_pct"])
    games = int(params.get("games", "162"))
    n = len(win_counts)
    mean_wins = sum(win_counts) / n
    var_wins = sum((w - mean_wins) ** 2 for w in win_counts) / n
    sorted_c = sorted(win_counts)

    def pctile(p: float) -> float:
        idx = p * (n - 1)
        lo = int(idx)
        hi = min(lo + 1, n - 1)
        return sorted_c[lo] + (idx - lo) * (sorted_c[hi] - sorted_c[lo])

    return {
        "experiment": "season",
        "win_pct": win_pct,
        "games": games,
        "trials": trials,
        "mean_wins": round(mean_wins, precision),
        "std_wins": round(math.sqrt(var_wins), precision),
        "p10": round(pctile(0.10), 1),
        "p25": round(pctile(0.25), 1),
        "p50": round(pctile(0.50), 1),
        "p75": round(pctile(0.75), 1),
        "p90": round(pctile(0.90), 1),
    }


def format_linboot_summary(
    params: dict[str, str],
    slopes: list[float],
    intercepts: list[float],
    predictions: list[float],
    precision: int,
) -> str:
    """Residual-bootstrap CI summary for linear regression."""
    alpha = float(params.get("alpha", "0.05"))
    conf_pct = int(round((1 - alpha) * 100))
    n = len(slopes)

    slope_lo, slope_hi = _bootstrap_ci(slopes, alpha)
    int_lo, int_hi = _bootstrap_ci(intercepts, alpha)
    slope_mean = sum(slopes) / n
    int_mean = sum(intercepts) / n

    lines = [
        "Bootstrap Linear Regression (residual bootstrap)",
        f"Trials:              {n:,}",
        "",
        "Slope",
        f"  Bootstrap mean:    {_fmt(slope_mean, precision)}",
        f"  {conf_pct}% CI:          [{_fmt(slope_lo, precision)}, {_fmt(slope_hi, precision)}]",
        "",
        "Intercept",
        f"  Bootstrap mean:    {_fmt(int_mean, precision)}",
        f"  {conf_pct}% CI:          [{_fmt(int_lo, precision)}, {_fmt(int_hi, precision)}]",
    ]

    if predictions:
        pred_lo, pred_hi = _bootstrap_ci(predictions, alpha)
        pred_mean = sum(predictions) / len(predictions)
        x_pred = params.get("predict", "?")
        lines += [
            "",
            f"Prediction at x={x_pred}",
            f"  Bootstrap mean:    {_fmt(pred_mean, precision)}",
            f"  {conf_pct}% CI:          [{_fmt(pred_lo, precision)}, {_fmt(pred_hi, precision)}]",
        ]

    return "\n".join(lines)


def _linboot_dump_csv(
    slopes: list[float],
    intercepts: list[float],
    predictions: list[float],
) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf)
    if predictions:
        writer.writerow(["trial", "slope", "intercept", "prediction"])
        for i, (s, ic, p) in enumerate(zip(slopes, intercepts, predictions), 1):
            writer.writerow([i, s, ic, p])
    else:
        writer.writerow(["trial", "slope", "intercept"])
        for i, (s, ic) in enumerate(zip(slopes, intercepts), 1):
            writer.writerow([i, s, ic])
    return buf.getvalue()


def _linboot_json(
    params: dict[str, str],
    slopes: list[float],
    intercepts: list[float],
    predictions: list[float],
    precision: int,
) -> dict:
    alpha = float(params.get("alpha", "0.05"))
    n = len(slopes)
    slope_lo, slope_hi = _bootstrap_ci(slopes, alpha)
    int_lo, int_hi = _bootstrap_ci(intercepts, alpha)

    out: dict = {
        "experiment": "linboot",
        "trials": n,
        "alpha": alpha,
        "slope": {
            "bootstrap_mean": round(sum(slopes) / n, precision),
            "ci_lower": round(slope_lo, precision),
            "ci_upper": round(slope_hi, precision),
        },
        "intercept": {
            "bootstrap_mean": round(sum(intercepts) / n, precision),
            "ci_lower": round(int_lo, precision),
            "ci_upper": round(int_hi, precision),
        },
    }

    if predictions:
        pred_lo, pred_hi = _bootstrap_ci(predictions, alpha)
        out["prediction"] = {
            "x": float(params.get("predict", "nan")),
            "bootstrap_mean": round(sum(predictions) / len(predictions), precision),
            "ci_lower": round(pred_lo, precision),
            "ci_upper": round(pred_hi, precision),
        }

    return out


# Experiment-specific display labels for format_table
_PROB_LABELS: dict[str, str] = {
    "power": "Estimated power",
    "permutation": "Empirical p-value",
    "bayes": "Estimated posterior",
}
_ANALYTICAL_LABELS: dict[str, str] = {
    "power": "Analytical power",
    "permutation": "Parametric p-value",
    "bayes": "Exact posterior",
    "season": "Analytical P(wins≥k)",
}

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _parse_kv_params(raw: list[str]) -> dict[str, str]:
    params: dict[str, str] = {}
    for item in raw:
        if "=" not in item:
            raise argparse.ArgumentTypeError(
                f"param {item!r} must be in KEY=VALUE format (e.g. n=10)"
            )
        key, _, val = item.partition("=")
        params[key.strip()] = val.strip()
    return params


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Monte Carlo probability simulator.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Experiments and required params:
  binomial    n=INT k=INT p=FLOAT         P(X >= k) for Binomial(n, p)
  birthday    pool=INT group=INT          P(at least one collision)
  streak      n=INT k=INT p=FLOAT         P(run of >= k successes in n trials)
  poisson     lam=FLOAT k=INT             P(X >= k) for Poisson(lambda)
  power       type=mean|comparison        Empirical statistical power
              mean:       n sigma delta [mu0 alpha sided]
              comparison: n p1 p2 [alpha sided]
  permutation type=one|two|paired         Distribution-free p-value
              one:    values mu0 [sided]
              two:    values1 values2 [sided]
              paired: values1 values2 [sided]
  bayes       prior likelihood fp         Empirical P(A|B) via rejection sampling
  season      win_pct [games wins_ge]     Season win distribution or P(wins >= k)
  linboot     x y [predict alpha]         Residual bootstrap for linear regression

Examples:
  simulate --experiment binomial --params n=10 k=5 p=0.4 --trials 100000
  simulate --experiment birthday --params pool=365 group=23 --confidence
  simulate --experiment power --params type=mean n=30 sigma=10 delta=5
  simulate --experiment permutation --params type=one values=2.1,3.4,2.9 mu0=3.0
  simulate --experiment bayes --params prior=0.01 likelihood=0.99 fp=0.05
  simulate --experiment season --params win_pct=0.58 wins_ge=90
  simulate --experiment linboot --params x=1,2,3,4,5 y=2.1,3.9,6.2,7.8,10.1 predict=6
""",
    )
    parser.add_argument(
        "--experiment",
        "-e",
        choices=list(REQUIRED_PARAMS),
        required=True,
        help="experiment type to simulate",
    )
    parser.add_argument(
        "--params",
        "-p",
        nargs="+",
        default=[],
        metavar="KEY=VALUE",
        help="experiment parameters (e.g. --params n=10 k=5 p=0.4)",
    )
    parser.add_argument(
        "--trials",
        "-t",
        type=int,
        default=10_000,
        metavar="N",
        help="number of simulation trials (default: 10,000; overridden by --scale)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=None,
        metavar="SE",
        help="target standard error; auto-computes --trials for worst-case p=0.5",
    )
    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=None,
        metavar="INT",
        help="random seed for reproducibility",
    )
    parser.add_argument(
        "--confidence",
        "-c",
        action="store_true",
        help="print 95%% Wilson confidence interval",
    )
    parser.add_argument(
        "--dump",
        action="store_true",
        help="output per-trial results as CSV instead of summary",
    )
    parser.add_argument(
        "--format",
        "-f",
        choices=["table", "json"],
        default="table",
        help="summary output format (default: table; ignored with --dump)",
    )
    parser.add_argument(
        "--precision",
        "-P",
        type=int,
        default=6,
        help="decimal places for printed probabilities (default: 6)",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate(args: argparse.Namespace) -> Optional[str]:
    try:
        params = _parse_kv_params(args.params)
    except argparse.ArgumentTypeError as exc:
        return str(exc)

    missing = [r for r in REQUIRED_PARAMS[args.experiment] if r not in params]
    if missing:
        return f"experiment '{args.experiment}' requires params: " + ", ".join(
            f"{m}=..." for m in missing
        )

    try:
        if args.experiment in ("binomial", "streak"):
            n, k, p = int(params["n"]), int(params["k"]), float(params["p"])
            if n < 1:
                return "param n must be >= 1"
            if k < 0:
                return "param k must be >= 0"
            if not (0.0 <= p <= 1.0):
                return "param p must be between 0 and 1"

        elif args.experiment == "birthday":
            pool, group = int(params["pool"]), int(params["group"])
            if pool < 1:
                return "param pool must be >= 1"
            if group < 1:
                return "param group must be >= 1"

        elif args.experiment == "poisson":
            lam, k = float(params["lam"]), int(params["k"])
            if lam <= 0.0:
                return "param lam must be > 0"
            if k < 0:
                return "param k must be >= 0"

        elif args.experiment == "power":
            ptype = params.get("type", "")
            if ptype not in ("mean", "comparison"):
                return "power param type must be 'mean' or 'comparison'"
            alpha = float(params.get("alpha", "0.05"))
            if not (0 < alpha < 1):
                return "power param alpha must be in (0, 1)"
            if ptype == "mean":
                for key in ("n", "sigma", "delta"):
                    if key not in params:
                        return f"power type=mean requires param: {key}"
                if int(params["n"]) < 2:
                    return "power param n must be >= 2"
                if float(params["sigma"]) <= 0:
                    return "power param sigma must be > 0"
                if float(params["delta"]) == 0:
                    return "power param delta must be nonzero"
            else:
                for key in ("n", "p1", "p2"):
                    if key not in params:
                        return f"power type=comparison requires param: {key}"
                if int(params["n"]) < 2:
                    return "power param n must be >= 2"
                p1, p2 = float(params["p1"]), float(params["p2"])
                if not (0 < p1 < 1):
                    return "power param p1 must be in (0, 1)"
                if not (0 < p2 < 1):
                    return "power param p2 must be in (0, 1)"
                if p1 == p2:
                    return "power params p1 and p2 must differ"

        elif args.experiment == "permutation":
            ptype = params.get("type", "")
            if ptype not in ("one", "two", "paired"):
                return "permutation param type must be 'one', 'two', or 'paired'"
            if ptype == "one":
                if "values" not in params:
                    return "permutation type=one requires param: values"
                if "mu0" not in params:
                    return "permutation type=one requires param: mu0"
                vals = [float(v) for v in params["values"].split(",")]
                if len(vals) < 3:
                    return "permutation type=one requires at least 3 values"
            else:
                for key in ("values1", "values2"):
                    if key not in params:
                        return f"permutation type={ptype} requires param: {key}"
                v1 = [float(v) for v in params["values1"].split(",")]
                v2 = [float(v) for v in params["values2"].split(",")]
                if len(v1) < 2 or len(v2) < 2:
                    return "permutation requires at least 2 values per group"
                if ptype == "paired" and len(v1) != len(v2):
                    return "permutation type=paired requires equal-length values1 and values2"

        elif args.experiment == "bayes":
            prior = float(params["prior"])
            likelihood = float(params["likelihood"])
            fp = float(params["fp"])
            if not (0 <= prior <= 1):
                return "bayes param prior must be in [0, 1]"
            if not (0 <= likelihood <= 1):
                return "bayes param likelihood must be in [0, 1]"
            if not (0 <= fp <= 1):
                return "bayes param fp must be in [0, 1]"
            if likelihood * prior + fp * (1.0 - prior) <= 0:
                return "bayes evidence P(B) = 0: choose nonzero likelihood or fp"

        elif args.experiment == "season":
            win_pct = float(params["win_pct"])
            if not (0 < win_pct < 1):
                return "season param win_pct must be in (0, 1)"
            games = int(params.get("games", "162"))
            if games < 1:
                return "season param games must be >= 1"
            if "wins_ge" in params:
                k = int(params["wins_ge"])
                if k < 0 or k > games:
                    return f"season param wins_ge must be in [0, {games}]"

        elif args.experiment == "linboot":
            x = [float(v) for v in params["x"].split(",")]
            y = [float(v) for v in params["y"].split(",")]
            if len(x) != len(y):
                return "linboot params x and y must have equal length"
            if len(x) < 3:
                return "linboot requires at least 3 data points"
            if len(set(x)) == 1:
                return "linboot param x has zero variance (all x values identical)"

    except (ValueError, KeyError) as exc:
        return f"invalid param value: {exc}"

    if args.trials < 1:
        return "--trials must be >= 1"
    if args.scale is not None and args.scale <= 0.0:
        return "--scale must be > 0"

    return None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    error = validate(args)
    if error:
        print(f"Error: {error}", file=sys.stderr)
        return 2

    params = _parse_kv_params(args.params)
    precision = args.precision
    trials = trials_for_scale(args.scale) if args.scale is not None else args.trials

    # ---- linboot: non-binary results, handled entirely here ----
    if args.experiment == "linboot":
        try:
            slopes, intercepts, predictions = simulate_linboot(
                params, trials, args.seed
            )
        except (ValueError, RuntimeError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 2
        if not slopes:
            print("Error: no successful bootstrap resamples", file=sys.stderr)
            return 2
        if args.dump:
            print(_linboot_dump_csv(slopes, intercepts, predictions), end="")
        elif args.format == "json":
            print(
                json.dumps(
                    _linboot_json(params, slopes, intercepts, predictions, precision),
                    indent=2,
                )
            )
        else:
            print(
                format_linboot_summary(
                    params, slopes, intercepts, predictions, precision
                )
            )
        return 0

    results = run_experiment(args.experiment, params, trials, args.seed)

    # ---- season without wins_ge: raw win counts, not binary ----
    if args.experiment == "season" and "wins_ge" not in params:
        if args.dump:
            print(_season_dump_csv(results), end="")
        elif args.format == "json":
            print(
                json.dumps(_season_json(params, results, trials, precision), indent=2)
            )
        else:
            print(format_season_summary(params, results, trials, precision))
        return 0

    # ---- standard binary-outcome flow ----
    if args.dump:
        print(format_dump_csv(results), end="")
        return 0

    hits = sum(results)
    p_hat = hits / trials
    se = standard_error(p_hat, trials)
    ci = wilson_ci(p_hat, trials) if args.confidence else None
    analytical = analytical_value(args.experiment, params)

    prob_label = _PROB_LABELS.get(args.experiment, "Estimated probability")
    analytical_label = _ANALYTICAL_LABELS.get(args.experiment, "Analytical value")

    if args.format == "json":
        output: dict = {
            "experiment": args.experiment,
            "params": params,
            "trials": trials,
            "hits": hits,
            "estimated_probability": round(p_hat, precision),
            "standard_error": round(se, precision),
        }
        if ci is not None:
            output["ci_lower"] = round(ci[0], precision)
            output["ci_upper"] = round(ci[1], precision)
        if analytical is not None:
            output["analytical_value"] = round(analytical, precision)
            output["difference"] = round(p_hat - analytical, precision)
        print(json.dumps(output, indent=2))
    else:
        print(
            format_table(
                args.experiment,
                params,
                trials,
                p_hat,
                se,
                ci,
                analytical,
                precision,
                prob_label=prob_label,
                analytical_label=analytical_label,
            )
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
