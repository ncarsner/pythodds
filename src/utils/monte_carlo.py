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

"""Command-line Monte Carlo probability simulator.

Runs repeated random experiments to estimate probabilities empirically,
cross-validating analytical results from binom, birthday, and poisson.

Supported experiments:
  binomial  — P(X ≥ k) for Binomial(n, p)
  birthday  — P(at least one collision) for group items drawn from pool
  streak    — P(at least one run of k consecutive successes in n Bernoulli trials)
  poisson   — P(X ≥ k) for Poisson(λ)

Usage examples:
  simulate --experiment binomial --params n=10 k=5 p=0.4 --trials 100000
  simulate --experiment birthday --params pool=365 group=23 --confidence
  simulate --experiment streak --params n=100 k=5 p=0.5 --trials 50000
  simulate --experiment poisson --params lam=3.0 k=7 --seed 42
  simulate --experiment binomial --params n=20 k=8 p=0.5 --scale 0.005 --seed 42
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
# Simulation engines
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
# Dispatch and analytical comparison
# ---------------------------------------------------------------------------

REQUIRED_PARAMS: dict[str, list[str]] = {
    "binomial": ["n", "k", "p"],
    "birthday": ["pool", "group"],
    "streak": ["n", "k", "p"],
    "poisson": ["lam", "k"],
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
    raise ValueError(f"Unknown experiment: {experiment!r}")  # pragma: no cover


def analytical_value(experiment: str, params: dict[str, str]) -> Optional[float]:
    """Return the exact analytical probability for comparison, or None."""
    if experiment == "binomial" and _binom_cdf_ge is not None:
        return _binom_cdf_ge(int(params["n"]), int(params["k"]), float(params["p"]))
    if experiment == "birthday" and _collision_prob_uniform is not None:
        return _collision_prob_uniform(int(params["group"]), int(params["pool"]))
    if experiment == "poisson" and _poisson_cdf_ge is not None:
        return _poisson_cdf_ge(int(params["k"]), float(params["lam"]))
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
) -> str:
    lines = [
        f"Experiment:              {experiment}",
        f"Parameters:              {', '.join(f'{k}={v}' for k, v in params.items())}",
        f"Trials:                  {trials:,}",
        f"Estimated probability:   {_fmt(p_hat, precision)}",
        f"Standard error:          {_fmt(se, precision)}",
    ]
    if ci is not None:
        lines.append(
            f"95% confidence interval: [{_fmt(ci[0], precision)}, {_fmt(ci[1], precision)}]"
        )
    if analytical is not None:
        lines.append(f"Analytical value:        {_fmt(analytical, precision)}")
        lines.append(f"Difference (sim-exact):  {p_hat - analytical:+.{precision}f}")
    return "\n".join(lines)


def format_dump_csv(results: list[int]) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["trial", "outcome"])
    for i, v in enumerate(results, 1):
        writer.writerow([i, v])
    return buf.getvalue()


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
  binomial  n=INT k=INT p=FLOAT   P(X >= k) for Binomial(n, p)
  birthday  pool=INT group=INT    P(at least one collision)
  streak    n=INT k=INT p=FLOAT   P(run of >= k successes in n trials)
  poisson   lam=FLOAT k=INT       P(X >= k) for Poisson(lambda)

Examples:
  simulate --experiment binomial --params n=10 k=5 p=0.4 --trials 100000
  simulate --experiment birthday --params pool=365 group=23 --confidence
  simulate --experiment streak --params n=100 k=5 p=0.5 --trials 50000
  simulate --experiment poisson --params lam=3.0 k=7 --seed 42
  simulate --experiment binomial --params n=20 k=8 p=0.5 --scale 0.005 --seed 42
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
        help="output per-trial results as CSV (trial, outcome) instead of summary",
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

    results = run_experiment(args.experiment, params, trials, args.seed)

    if args.dump:
        print(format_dump_csv(results), end="")
        return 0

    hits = sum(results)
    p_hat = hits / trials
    se = standard_error(p_hat, trials)
    ci = wilson_ci(p_hat, trials) if args.confidence else None
    analytical = analytical_value(args.experiment, params)

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
                args.experiment, params, trials, p_hat, se, ci, analytical, precision
            )
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
