# pythodds

[![PyPI version](https://badge.fury.io/py/pythodds.svg)](https://pypi.org/project/pythodds/)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/ncarsner/pythodds/blob/main/LICENSE)

A command-line utility and Python library for calculating statistics, odds, and probabilities.

## Features

- **Binomial Distribution**: Calculate PMF, CDF, and survival functions for binomial distributions
- **Birthday Problem**: Compute collision probabilities for uniform and non-uniform pools, find minimum group sizes, and generate probability tables
- **Poisson Distribution**: Compute PMF, CDF, and survival probabilities, find minimum event counts for a target cumulative probability, and generate full probability tables
- **Command-line Interface**: Easy-to-use CLI tools (`binom`, `birthday`, and `poisson` commands)
- **Pure Python**: No external dependencies

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
birthday --range 1 60 --format json
birthday --range 1 60 --format csv
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
