# pythodds

[![PyPI version](https://badge.fury.io/py/pythodds.svg)](https://pypi.org/project/pythodds/)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/ncarsner/pythodds/blob/main/LICENSE)
[![Tests](https://github.com/ncarsner/pythodds/actions/workflows/tests.yml/badge.svg)](https://github.com/ncarsner/pythodds/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/ncarsner/pythodds/branch/main/graph/badge.svg)](https://codecov.io/gh/ncarsner/pythodds)

A command-line utility and Python library for calculating statistics, odds, and probabilities.

## Features

- **Binomial Distribution**: Calculate PMF, CDF, and survival functions for binomial distributions
- **Birthday Problem**: Compute collision probabilities for uniform and non-uniform pools, find minimum group sizes, and generate probability tables
- **Normal Distribution**: Compute PDF, CDF, survival probabilities, interval probabilities, and the inverse CDF (percent-point function) for a Gaussian N(μ, σ²) distribution
- **Expected Value**: Compute E[X], Var(X), SD(X), Shannon entropy, and the moment generating function for discrete probability distributions; supports inline input or CSV/JSON files
- **Poisson Distribution**: Compute PMF, CDF, and survival probabilities, find minimum event counts for a target cumulative probability, and generate full probability tables
- **Streak Probability**: Compute the probability of at least one consecutive run of successes and the expected length of the longest streak
- **Monte Carlo Simulator**: Empirically estimate probabilities for binomial, birthday, streak, and Poisson experiments with confidence intervals and analytical comparison
- **Command-line Interface**: Easy-to-use CLI tools (`binom`, `birthday`, `normal`, `expected`, `poisson`, `streak`, and `simulate` commands)
- **Pure Python**: No external dependencies required for core calculations

## Installation

Install from PyPI:

```bash
pip install pythodds
```

Or install from source:

```bash
git clone https://github.com/ncarsner/pythodds.git
cd pythodds
pip install -e .
```

## Usage

### Command Line

#### `binom` — Binomial Distribution

```bash
# Calculate binomial distribution probabilities
binom -n 10 -k 3 -p 0.4

# Specify a target and minimum probability threshold
binom -n 100 -k 30 -p 0.35 --target 40 --min-prob 0.05
```

#### `birthday` — Birthday Problem Collision Probability

Computes the probability that at least two items in a group share the same value when drawn from a pool of equally-likely possibilities. Defaults to a pool size of 365.25 (calendar days).

```bash
# P(duplicate birthday) in a group of 23 people
birthday -n 23

# Find the minimum group size to reach 50% collision probability
birthday --target-prob 0.50

# Print a probability table for group sizes 1–40
birthday --range 1 40

# Custom pool size (e.g. 7-digit phone numbers)
birthday -p 10_000_000 -n 1180

# Non-uniform pool via relative weights
birthday --group-size 30 --weights 0.10,0.15,0.20,0.30,0.25

# Output as JSON or CSV
birthday --range 1 60 --format <json|csv>
```

**Options:**

| Flag | Long form | Description |
|------|-----------|-------------|
| `-p` | `--pool-size` | Pool size — number of equally-likely outcomes (default: `365.25`) |
| `-n` | `--group-size` | Compute collision probability for exactly this group size |
| `-t` | `--target-prob` | Find the minimum group size reaching this probability |
| `-r` | `--range MIN MAX` | Print a probability table for group sizes MIN through MAX |
| `-w` | `--weights` | Comma-separated relative frequencies for a non-uniform pool |
| `-f` | `--format` | Output format: `table` (default), `json`, or `csv` |
| `-P` | `--precision` | Decimal places for printed probabilities (default: `6`) |

#### `normal` — Normal (Gaussian) Distribution

Computes PDF, CDF, survival probabilities, interval probabilities, and the inverse CDF (percent-point function) for a N(μ, σ²) distribution. Uses only the Python standard library.

```bash
# PDF, P(X ≤ 1.96), and P(X ≥ 1.96) for the standard normal
normal -x 1.96 -m 0 -s 1

# Same calculation for a custom distribution
normal -x 75 -m 70 -s 5

# P(−1.96 ≤ X ≤ 1.96)
normal --between -1.96 1.96 -m 0 -s 1

# Find the value x such that P(X ≤ x) = 0.975 (inverse CDF)
normal --quantile 0.975 -m 0 -s 1
```

**Options:**

| Flag | Long form | Description |
|------|-----------|-------------|
| `-x` | `--value` | Compute PDF, P(X ≤ x), and P(X ≥ x) for this value |
| | `--between LOW HIGH` | Compute P(LOW ≤ X ≤ HIGH) |
| `-q` | `--quantile` | Find x such that P(X ≤ x) = P (inverse CDF) |
| `-m` | `--mean` | Distribution mean μ (default: `0`) |
| `-s` | `--std` | Distribution standard deviation σ (default: `1`) |
| `-P` | `--precision` | Decimal places for printed values (default: `6`) |

> `-x/--value`, `--between`, and `-q/--quantile` are mutually exclusive; one is required.

#### `expected` — Expected Value & Discrete Distribution Statistics

Computes E[X], Var(X), SD(X), Shannon entropy, and optionally the moment generating function (MGF) for a discrete probability distribution supplied inline or via a CSV/JSON file.

```bash
# E[X] and statistics for a simple discrete distribution
expected --outcomes 0,1,5,10 --probs 0.50,0.25,0.15,0.10

# Non-uniform six-sided die
expected --outcomes 1,2,3,4,5,6 --probs 0.1,0.2,0.3,0.2,0.1,0.1

# Load distribution from a CSV or JSON file
expected --file payouts.csv

# Also compute the MGF at t=0.5
expected --outcomes 0,1 --probs 0.3,0.7 --mgf 0.5
```

**Options:**

| Flag | Long form | Description |
|------|-----------|-------------|
| `-o` | `--outcomes` | Comma-separated outcome values |
| `-f` | `--file` | CSV or JSON file with outcomes and probabilities |
| `-p` | `--probs` | Comma-separated probabilities (required with `--outcomes`) |
| | `--mgf T` | Also compute the moment generating function M_X(t) at t=T |
| `-P` | `--precision` | Decimal places for printed values (default: `6`) |

> `--outcomes` and `--file` are mutually exclusive; one is required. `--probs` is required when using `--outcomes`.

#### `poisson` — Poisson Distribution

Computes PMF, CDF, and survival probabilities for a Poisson(λ) distribution. Models rare, independent events occurring at a known average rate — server errors per hour, calls per minute, defects per batch, and so on.

```bash
# P(X=7), P(X≤7), and P(X≥7) for λ=3.0
poisson -l 3.0 -k 7

# Find the minimum k such that P(X ≤ k) >= 0.95
poisson -l 3.0 -t 0.95

# Print a probability table for k = 0 through 15
poisson -l 3.0 -r 0 15

# Also show P(X ≥ 5) and whether it meets a 1% threshold
poisson -l 0.5 -k 2 --target 5 --min-prob 0.01

# Output as JSON or CSV
poisson -l 3.0 -r 0 20 -f json
poisson -l 3.0 -r 0 20 -f csv
```

**Options:**

| Flag | Long form | Description |
|------|-----------|-------------|
| `-l` | `--rate` | Average event rate λ (required, must be > 0) |
| `-k` | `--events` | Compute PMF and CDF for exactly this event count |
| `-t` | `--target-prob` | Find the minimum k such that P(X ≤ k) ≥ PROB |
| `-r` | `--range MIN MAX` | Print a probability table for event counts MIN through MAX |
| | `--target` | With `-k`: also print P(X ≥ T) for this target count |
| | `--min-prob` | With `--target`: report whether P(X ≥ T) meets this threshold |
| `-f` | `--format` | Output format: `table` (default), `json`, or `csv` |
| `-P` | `--precision` | Decimal places for printed probabilities (default: `6`) |

#### `streak` — Streak / Consecutive Run Probability

Computes the exact probability of at least one run of k consecutive successes in n independent Bernoulli trials, and the expected length of the longest run. Uses dynamic programming for exact O(n·k) computation.

```bash
# P(at least one run of 5+ heads in 100 fair coin flips)
streak -n 100 -k 5 -p 0.5

# P(at least one hitting streak of 10+ games over a 162-game season at .320)
streak -n 162 -k 10 -p 0.32

# Expected length of the longest win streak in 50 trials at 40% success rate
streak -n 50 -p 0.40 --longest
```

**Options:**

| Flag | Long form | Description |
|------|-----------|-------------|
| `-n` | `--trials` | Total number of independent trials (required) |
| `-p` | `--prob` | Success probability per trial, 0–1 (required) |
| `-k` | `--streak-length` | Compute P(at least one run of K consecutive successes) |
| | `--longest` | Compute E[length of longest run of consecutive successes] |
| `-P` | `--precision` | Decimal places for printed probabilities (default: `6`) |

> `-k/--streak-length` and `--longest` are mutually exclusive; one is required.

#### `simulate` — Monte Carlo Probability Simulator

Runs repeated random experiments to estimate probabilities empirically, with optional confidence intervals and analytical comparison against `binom`, `birthday`, `poisson`, and `streak`.

```bash
# Estimate P(X >= 5) for Binomial(n=10, p=0.4) over 100,000 trials
simulate --experiment binomial --params n=10 k=5 p=0.4 --trials 100000

# Birthday collision probability for a group of 23 with a 95% confidence interval
simulate --experiment birthday --params pool=365 group=23 --confidence

# Streak probability: P(run of 5+ successes in 100 trials, p=0.5)
simulate --experiment streak --params n=100 k=5 p=0.5 --trials 50000

# Poisson: P(X >= 7) for λ=3.0 with a fixed seed
simulate --experiment poisson --params lam=3.0 k=7 --seed 42

# Auto-size trial count to achieve a target standard error of 0.005
simulate --experiment binomial --params n=20 k=8 p=0.5 --scale 0.005
```

**Options:**

| Flag | Long form | Description |
|------|-----------|-------------|
| `-e` | `--experiment` | Experiment type: `binomial`, `birthday`, `streak`, or `poisson` (required) |
| `-p` | `--params` | Space-separated `KEY=VALUE` experiment parameters (see below) |
| `-t` | `--trials` | Number of simulation trials (default: 10,000) |
| | `--scale` | Target standard error; auto-computes `--trials` (overrides `-t`) |
| `-s` | `--seed` | Random seed for reproducibility |
| `-c` | `--confidence` | Print 95% Wilson confidence interval |
| | `--dump` | Output per-trial results as CSV instead of summary |
| `-f` | `--format` | Summary output format: `table` (default) or `json` |
| `-P` | `--precision` | Decimal places for printed probabilities (default: `6`) |

**Required params by experiment:**

| Experiment | Required params |
|------------|-----------------|
| `binomial` | `n=INT k=INT p=FLOAT` |
| `birthday` | `pool=INT group=INT` |
| `streak` | `n=INT k=INT p=FLOAT` |
| `poisson` | `lam=FLOAT k=INT` |

### Python Library

#### Binomial Distribution

```python
from src.utils.binomial_distribution import binomial_pmf, binomial_cdf_le, binomial_cdf_ge

# P(X = 3) for Binomial(n=10, p=0.4)
pmf = binomial_pmf(10, 3, 0.4)

# P(X <= 3) for Binomial(n=10, p=0.4)
cdf = binomial_cdf_le(10, 3, 0.4)

# P(X >= 3) for Binomial(n=10, p=0.4)
survival = binomial_cdf_ge(10, 3, 0.4)
```

#### Birthday Problem

```python
from src.utils.birthday_problem import (
    collision_prob_uniform,
    collision_prob_nonuniform,
    min_group_for_prob,
    expected_duplicate_pairs,
)

# P(duplicate) for 23 people in a pool of 365.25
prob = collision_prob_uniform(23, 365.25)

# Minimum group size to reach 50% collision probability
n = min_group_for_prob(0.50, 365.25)

# P(duplicate) with a non-uniform pool
prob_nu = collision_prob_nonuniform(30, [0.10, 0.15, 0.20, 0.30, 0.25])

# Expected number of duplicate pairs
pairs = expected_duplicate_pairs(23, 365.25)
```

#### Poisson Distribution

```python
from src.utils.poisson_distribution import (
    poisson_pmf,
    poisson_cdf_le,
    poisson_cdf_ge,
    min_k_for_prob,
)

# P(X = 7) for Poisson(λ=3.0)
pmf = poisson_pmf(7, 3.0)

# P(X ≤ 7) for Poisson(λ=3.0)
cdf = poisson_cdf_le(7, 3.0)

# P(X ≥ 7) for Poisson(λ=3.0)
survival = poisson_cdf_ge(7, 3.0)

# Minimum k such that P(X ≤ k) >= 0.95
k = min_k_for_prob(0.95, 3.0)
```

#### Streak Probability

```python
from src.utils.streak_probability import (
    prob_at_least_one_streak,
    expected_longest_streak,
)

# P(at least one run of 5 consecutive heads in 100 fair coin flips)
p = prob_at_least_one_streak(100, 5, 0.5)

# Expected length of the longest run of successes in 162 trials at .300
e = expected_longest_streak(162, 0.300)
```

#### Normal Distribution

```python
from src.utils.normal_gaussian import (
    normal_pdf,
    normal_cdf,
    normal_ppf,
    normal_prob_between,
)

# PDF value at x=1.96 for the standard normal
pdf = normal_pdf(1.96, mu=0.0, sigma=1.0)

# P(X ≤ 1.96)
cdf = normal_cdf(1.96, mu=0.0, sigma=1.0)

# P(X ≥ 1.96)
survival = 1.0 - normal_cdf(1.96, mu=0.0, sigma=1.0)

# P(−1.96 ≤ X ≤ 1.96)
prob = normal_prob_between(-1.96, 1.96, mu=0.0, sigma=1.0)

# Find x such that P(X ≤ x) = 0.975 (inverse CDF)
x = normal_ppf(0.975, mu=0.0, sigma=1.0)
```

#### Expected Value

```python
from src.utils.expected_value import (
    expected_value,
    variance,
    std_dev,
    entropy,
    mgf,
    load_file,
)

outcomes = [0, 1, 5, 10]
probs    = [0.50, 0.25, 0.15, 0.10]

# E[X]
ev = expected_value(outcomes, probs)

# Var(X) and SD(X)
var = variance(outcomes, probs)
sd  = std_dev(outcomes, probs)

# Shannon entropy (bits)
H = entropy(probs)

# Moment generating function M_X(t) at t=0.5
M = mgf(outcomes, probs, t=0.5)

# Load a distribution from a CSV or JSON file
outcomes, probs = load_file("payouts.csv")
```

#### Monte Carlo Simulator

```python
from src.utils.monte_carlo import (
    simulate_binomial,
    simulate_birthday,
    simulate_streak,
    simulate_poisson,
    wilson_ci,
    standard_error,
)

# Simulate P(X >= 5) for Binomial(10, 0.4) over 100,000 trials
results = simulate_binomial(n=10, k=5, p=0.4, trials=100_000, seed=42)
p_hat = sum(results) / len(results)
se = standard_error(p_hat, len(results))
ci = wilson_ci(p_hat, len(results))
```

## Development

Clone the repository and install in editable mode:

```bash
git clone https://github.com/ncarsner/pythodds.git
cd pythodds
pip install -e .
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

[Nicholas Carsner](https://github.com/ncarsner)
