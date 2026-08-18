# pythodds

[![PyPI version](https://badge.fury.io/py/pythodds.svg)](https://pypi.org/project/pythodds/)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/ncarsner/pythodds/blob/main/LICENSE)
[![Tests](https://github.com/ncarsner/pythodds/actions/workflows/tests.yml/badge.svg)](https://github.com/ncarsner/pythodds/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/ncarsner/pythodds/branch/main/graph/badge.svg)](https://codecov.io/gh/ncarsner/pythodds)

A command-line utility and Python library for calculating statistics, odds, and probabilities.

## Features

| | |
|-|-|
| **Binomial Distribution** |  Calculate PMF, CDF, and survival functions for binomial distributions |
| **Bayes' Theorem** | Compute posterior probabilities from priors, likelihoods, and either direct evidence or a false-positive rate |
| **Birthday Problem** | Compute collision probabilities for uniform and non-uniform pools, find minimum group sizes, and generate probability tables |
| **Normal Distribution** | Compute PDF, CDF, survival probabilities, interval probabilities, and the inverse CDF (percent-point function) for a Gaussian N(μ, σ²) distribution |
| **Z-Scores** | Compute standardized z-scores for a single value with known mean/std dev, or standardize an inline dataset using population or sample standard deviation |
| **Expected Value** | Compute E[X], Var(X), SD(X), Shannon entropy, and the moment generating function for discrete probability distributions; supports inline input or CSV/JSON files |
| **Poisson Distribution** | Compute PMF, CDF, and survival probabilities, find minimum event counts for a target cumulative probability, and generate full probability tables |
| **Prime Numbers** | Check primality, find the nth prime, count primes up to a limit (π function), list primes in a range, and compute prime factorizations |
| **Streak Probability** | Compute the probability of at least one consecutive run of successes and the expected length of the longest streak |
| **Pythagorean Record** | Calculate team winning percentage expectations using Bill James' Pythagorean formula or the SABR linear formula; project in-progress season records and compare actual vs. expected performance |
| **Pearson Correlation** | Compute Pearson's r, r², t-statistic, p-value, and confidence intervals; test for linear relationships between two continuous variables; supports inline data or CSV input |
| **Spearman Correlation** | Compute Spearman's ρ (rank correlation), ρ², t-statistic, p-value, and confidence intervals; test for monotonic relationships; robust to outliers and suitable for ordinal data; includes rank display for tie inspection |
| **Linear Regression** | Perform ordinary least squares (OLS) regression with full statistical inference: coefficients, standard errors, R², F-statistic, t-tests, confidence intervals, and predictions with confidence/prediction intervals |
| **Sample Size Calculator** | Determine minimum sample sizes for proportion estimation, mean difference detection, and two-proportion comparisons; includes power analysis sweeps |
| **Bootstrap Confidence Intervals** | Compute non-parametric confidence intervals for statistics (mean, median, standard deviation) using bootstrap resampling; distribution-free alternative requiring no normality assumptions |
| **Confidence Intervals (Parametric)** | Calculate confidence intervals using formula-based methods for proportions (Wilson, Clopper-Pearson, normal), means (t-interval), and count data (Poisson); works with summary statistics |
| **Hypothesis Testing & p-values** | Perform statistical hypothesis tests: z-tests and binomial exact tests (proportions), t-tests (means), and chi-squared goodness-of-fit tests; includes alpha sensitivity analysis |
| **T-Test** | Perform one-sample, two-sample (Welch's), and paired t-tests; compute t-statistics, degrees of freedom, p-values, confidence intervals, and Cohen's d effect size; accepts raw data or summary statistics (mean, std, n); supports one-sided and two-sided alternatives |
| **Time Series Forecasting** | Forecast future values with prediction intervals using simple, double (Holt's), or Holt-Winters exponential smoothing; supports backtesting and multiple output formats |
| **Collatz Conjecture** | Evaluate the Collatz conjecture (3n+1 problem) for positive integers up to n; track which numbers reach 1 and report the largest consecutive verified sequence |
| **Jevons' Paradox** | Quantify the rebound effect and backfire condition for resource efficiency improvements; compute expected savings, rebound consumption, actual savings, and net consumption change given an efficiency gain and price elasticity of demand; support sweeps over efficiency or elasticity ranges |
| **Sigmoid Function** | Evaluate σ(x) = 1/(1+e^(−x)), its derivative, and the inverse logit; range tables and Unicode sparkline |
| **Euler's Number** | Explore e via limit convergence, Taylor series for e^x and ln(x), Euler's identity, and the Euler-Mascheroni constant |
| **Monte Carlo Simulator** | Empirically estimate probabilities for `binomial`, `birthday`, `streak`, `poisson`, `power`, `permutation`, `bayes`, `season`, `linboot` experiments with confidence intervals and analytical comparison |
| **Subnet Mask Calculator** | Compute network address, broadcast address, first/last usable IP, subnet mask, host count, and classful network count from an IPv4 address and CIDR prefix |
| **Geometric Distribution** | Compute PMF, CDF, survival, mean, and variance for the number of trials until the first success; supports single-k reports and full probability tables |
| **Chi-Square Tests** | Goodness-of-fit and independence (contingency table) chi-square tests via the exact regularised incomplete gamma function; per-cell contributions for residual diagnostics |
| **One-Way ANOVA** | Test whether the means of three or more independent groups are equal; F-statistic via the regularised incomplete beta function, with Tukey HSD or Bonferroni-corrected pairwise post-hoc comparisons |
| **Life in Weeks** | Render a human life as an ANSI grid of weeks, months, or years, color-coded elapsed vs remaining from a birthdate; discrete shape glyphs with quarter and decade grouping keep individual cells countable, and a "Life of a Typical American" reference grid shades each life phase with milestone markers |
| **Break-Even Analysis** | Cost-volume-profit analysis: break-even units and revenue, contribution margin and ratio, margin of safety, and the volume needed to hit a target profit; optional profit/loss sweep across a unit range with a text bar chart |
| **Information Entropy** | Shannon entropy, KL divergence, cross-entropy, mutual information, and conditional entropy in bits, nats, or hartleys; accepts raw counts as well as probabilities |
| **Weibull Distribution** | PDF, CDF, survival, hazard rate, quantiles, mean, median, and variance for Weibull(k, λ), with the shape parameter's failure mode named (infant mortality, constant hazard, or wear-out) |
| **Discount Rates & NPV** | Real and nominal discount rates via the Fisher equation, discount factors, present value of a lump sum, and nominal and inflation-adjusted NPV with discounted payback period |
| **Command-line Interface** | `binom`, `bayes`, `birthday`, `normal`, `zscore`, `expected`, `poisson`, `prime`, `streak`, `pythag`, `pearson`, `spearman`, `linreg`, `sample`, `bootci`, `confint`, `pvalue`, `ttest`, `forecast`, `collatz`, `jevons`, `simulate`, `sigmoid`, `euler`, `gini`, `crt`, `subnet`, `geometric`, `chisq`, `anova`, `life`, `breakeven`, `entropy`, `weibull`, and `discount` commands |
| **Minimal Dependencies** | Core calculations use pure Python; Spearman correlation and Monte Carlo simulation use scipy/numpy for numerical robustness |


## Installation

Install from PyPI:

```bash
pip install pythodds
```
or
```
uv add pythodds
```

Or install from source:

```bash
git clone https://github.com/ncarsner/pythodds.git
cd pythodds
pip install -e .
```

## Command Line Usage

| Command | Description |
|---------|-------------|
| `binom` | Binomial PMF, CDF, and survival for Binomial(n, p) |
| `bayes` | Posterior probability from prior, likelihood, and evidence or false-positive rate |
| `birthday` | Collision probability for uniform and non-uniform pools |
| `normal` | PDF, CDF, survival, interval probabilities, and inverse CDF for Gaussian N(μ, σ²) |
| `zscore` | Standardized z-scores for a single value or an inline dataset |
| `expected` | E[X], Var(X), SD(X), entropy, and MGF for discrete distributions |
| `poisson` | PMF, CDF, survival, and minimum k for Poisson(λ) |
| `prime` | Primality testing, nth prime, prime counting, ranges, and factorization |
| `streak` | Consecutive run probability and expected longest streak |
| `pythag` | Win percentage and season projection (Pythagorean or linear formula) |
| `pearson` | Pearson r, r², t-statistic, p-value, and confidence intervals |
| `spearman` | Spearman ρ, rank correlation, hypothesis testing, and rank display |
| `linreg` | OLS regression with inference, R², F-statistic, and prediction intervals |
| `sample` | Minimum sample sizes for proportions, means, and two-proportion comparisons |
| `bootci` | Non-parametric confidence intervals via bootstrap resampling |
| `confint` | Parametric CIs: Wilson, Clopper-Pearson, normal, t-interval, Poisson |
| `pvalue` | z-test, binomial exact, one-sample t-test, and chi-squared goodness-of-fit |
| `ttest` | One-sample, two-sample (Welch's), and paired t-tests with Cohen's d |
| `forecast` | Exponential smoothing: simple, Holt's method, and Holt-Winters |
| `collatz` | Collatz conjecture (3n+1) verification for integers 1 through n |
| `jevons` | Rebound effect and backfire analysis for resource efficiency improvements |
| `sigmoid` | Sigmoid σ(x), derivative σ'(x), inverse logit, range tables, and Unicode sparkline |
| `euler` | Euler's number via limit/series, e^x Taylor series, ln(x) series, Euler's identity, and γ constant |
| `gini` | Gini coefficient and Lorenz curve from raw data, weighted samples, or grouped shares |
| `crt` | Sunzi's Theorem solver for systems of simultaneous congruences |
| `subnet` | Network address, broadcast, first/last usable IP, subnet mask, host count, and classful network count from an IPv4 CIDR block |
| `geometric` | PMF, CDF, survival, mean, and variance for the geometric distribution (trials until first success) |
| `chisq` | Chi-square goodness-of-fit and independence tests with per-cell contributions |
| `anova` | One-way ANOVA F-test with optional Tukey HSD or Bonferroni-corrected pairwise post-hoc comparisons |
| `life` | ANSI grid of a human life in weeks, months, or years, elapsed vs remaining, with a typical-American reference grid |
| `breakeven` | Break-even units and revenue, contribution margin, margin of safety, target-profit volume, and profit/loss sweep |
| `entropy` | Shannon entropy, KL divergence, cross-entropy, mutual information, and conditional entropy in bits, nats, or hartleys |
| `weibull` | PDF, CDF, survival, hazard, quantiles, and moments for the Weibull(k, λ) reliability distribution |
| `discount` | Real and nominal discount rates (Fisher), discount factors, present value, and inflation-adjusted NPV |
| `simulate` | Monte Carlo estimation with confidence intervals and analytical comparison |

---

<details>
<summary><strong><code>binom</code></strong> — Binomial Distribution</summary>

Computes exact, cumulative, and survival probabilities for a Binomial(n, p) distribution, and renders a color-coded stacked progress bar showing the share of mass below `k`, at `k`, and above `k`.

```bash
# Calculate binomial distribution probabilities
binom -n 10 -k 3 -p 0.4

# Specify a target and minimum probability threshold
binom -n 100 -k 30 -p 0.35 --target 40 --min-prob 0.05
```

Typical output includes a stacked terminal bar like this:

```text
n=10, k=3, p=0.4
P(X = 3):  0.214991 (21.499100%)
P(X <= 3): 0.382281 (38.228100%)
P(X >= 3): 0.832710 (83.271000%)
[stacked ANSI bar for <k | =k | >k]
```

</details>

---

<details>
<summary><strong><code>bayes</code></strong> — Bayes' Theorem Posterior Probability</summary>

Computes posterior probability $P(A\mid B)$ from a prior probability, likelihood, and either direct evidence $P(B)$ or a false-positive rate $P(B\mid \neg A)$.

```bash
# Medical test example: prevalence 1%, sensitivity 99%, false-positive rate 5%
bayes -p 0.01 -l 0.99 -f 0.05

# Provide evidence directly instead of a false-positive rate
bayes -p 0.2 -l 0.8 -e 0.5
```

**Options:**

| Flag | Long form | Description |
|------|-----------|-------------|
| `-p` | `--prior` | Prior probability $P(A)$, between `0` and `1` |
| `-l` | `--likelihood` | Likelihood $P(B\mid A)$, between `0` and `1` |
| `-e` | `--evidence` | Evidence probability $P(B)$, between `0` and `1` |
| `-f` | `--false-positive` | False-positive rate $P(B\mid \neg A)$, between `0` and `1` |
| `-P` | `--precision` | Decimal places for printed values (default: `6`) |

> `-e/--evidence` and `-f/--false-positive` are mutually exclusive; one is required.

</details>

---

<details>
<summary><strong><code>birthday</code></strong> — Birthday Problem Collision Probability</summary>

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

</details>

---

<details>
<summary><strong><code>normal</code></strong> — Normal (Gaussian) Distribution</summary>

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

</details>

---

<details>
<summary><strong><code>zscore</code></strong> — Z-Score Calculator</summary>

Computes standardized scores using `z = (x - mean) / standard deviation`. Use it with a single value and known distribution parameters, or provide an inline dataset to compute mean, standard deviation, and z-scores for every value.

```bash
# Standardize one value with a known mean and standard deviation
zscore --value 85 --mean 70 --std 10

# Compute population z-scores for an inline dataset
zscore --values 2,4,4,4,5,5,7,9

# Use sample standard deviation for the inline dataset
zscore --values 2,4,4,4,5,5,7,9 --sample

# Custom precision
zscore --value 85 --mean 70 --std 10 --precision 2
```

**Options:**

| Flag | Long form | Description |
|------|-----------|-------------|
| `-x` | `--value` | Single value to standardize |
| `-v` | `--values` | Comma-separated values to standardize as a dataset |
| `-m` | `--mean` | Mean to use with `--value` |
| `-s` | `--std` | Standard deviation to use with `--value` |
| | `--sample` | Use sample standard deviation for `--values` |
| `-P` | `--precision` | Decimal places for printed values (default: `6`) |

> `-x/--value` and `-v/--values` are mutually exclusive; one is required.<br>
> `--mean` and `--std` are required when using `--value`.

</details>

---

<details>
<summary><strong><code>expected</code></strong> — Expected Value & Discrete Distribution Statistics</summary>

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

> `--outcomes` and `--file` are mutually exclusive; one is required<br>
> `--probs` is required when using `--outcomes`

</details>

---

<details>
<summary><strong><code>poisson</code></strong> — Poisson Distribution</summary>

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

</details>

---

<details>
<summary><strong><code>prime</code></strong> — Prime Number Operations</summary>

Provides tools for working with prime numbers: primality testing, finding the nth prime, counting primes up to a limit (prime counting function π), listing primes in a range, and computing prime factorizations.

```bash
# Check if a number is prime
prime -is 97
prime --check 100

# Find the nth prime number (1-indexed)
prime -n 100
prime --nth 10000

# Count primes up to a limit (π function)
prime -C 1000
prime --count 1000000

# List all primes in a range [start, end]
prime -R 50 100
prime --range 1 50

# Compute prime factorization
prime -F 360
prime --factorize 2024

# Output as JSON
prime --check 97 --format json
prime --range 1 100 --format json
prime --factorize 360 --format json
```

**Options:**

| Flag | Long form | Description |
|------|-----------|-------------|
| `-is` | `--check` | Check if N is prime |
| `-n` | `--nth` | Find the Nth prime number (1-indexed) |
| `-C` | `--count` | Count primes up to and including LIMIT (π function) |
| `-R` | `--range START END` | List all primes in the range [START, END] inclusive |
| `-F` | `--factorize` | Compute prime factorization of N |
| | `--format` | Output format: `text` (default) or `json` |

> Operation flags are mutually exclusive; exactly one is required.

**Implementation details:**
- **Primality testing**: Uses trial division with 6k±1 optimization
- **Nth prime**: Uses Sieve of Eratosthenes with prime counting estimation
- **Prime counting**: Implements the prime counting function π(n)
- **Range listing**: Efficient sieve-based generation
- **Factorization**: Trial division returning `{prime: exponent}` dict

**Example outputs:**

```bash
$ prime --check 97
97 is prime

$ prime --nth 100
The 100th prime number is 541

$ prime --count 100
π(100) = 25 (there are 25 primes ≤ 100)

$ prime --factorize 360
Prime factorization of 360:
  360 = 2³ × 3² × 5

Factor breakdown:
  2^3 = 8
  3^2 = 9
  5^1 = 5
```

</details>

---

<details>
<summary><strong><code>streak</code></strong> — Streak / Consecutive Run Probability</summary>

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

</details>

---

<details>
<summary><strong><code>pythag</code></strong> — Pythagorean Record / Win Expectation</summary>

Calculates expected winning percentage for sports teams based on runs/points scored and allowed. Supports both the traditional Pythagorean formula (Bill James) and the newer linear formula from SABR research (Rothman, 2014). Can project final season records for teams in progress.

```bash
# MLB team using linear formula (default)
pythag --scored 800 --allowed 650

# Compare both linear and Pythagorean methods
pythag --scored 800 --allowed 650 --method both

# Use traditional Pythagorean formula with custom exponent
pythag --scored 800 --allowed 650 --method pythagorean --exponent 1.83

# NFL team projection
pythag --scored 420 --allowed 300 --sport nfl

# NBA team projection
pythag --scored 8500 --allowed 8200 --sport nba

# In-progress season: team is 45-37 after 82 games (shows projection)
pythag --scored 550 --allowed 490 --current-wins 45 --games-played 82
```

**Options:**

| Flag | Long form | Description |
|------|-----------|-------------|
| `-s` | `--scored` | Runs/points scored by the team (required) |
| `-a` | `--allowed` | Runs/points allowed by the team (required) |
| | `--sport` | Sport/league: `mlb` (default), `nfl`, or `nba` |
| `-m` | `--method` | Calculation method: `linear` (default), `pythagorean`, or `both` |
| `-e` | `--exponent` | Exponent for Pythagorean formula (default: `2.0`; optimal ~1.83 for baseball) |
| `-g` | `--games` | Total games in season (default: 162 for mlb, 17 for nfl, 82 for nba) |
| `-w` | `--current-wins` | Current wins (for in-progress season projection) |
| `-p` | `--games-played` | Games already played (for in-progress season projection) |
| `-P` | `--precision` | Decimal places for percentages (default: `2`) |

> `--current-wins` and `--games-played` must be used together for in-progress projections.

**Formulas:**
- **Pythagorean (James):** `EXP(W%) = RS^exp / (RS^exp + RA^exp)`
- **Linear (Rothman, 2014):**
  - MLB: `EXP(W%) = 0.000683(RS - RA) + 0.50`
  - NFL: `EXP(W%) = 0.001538(PS - PA) + 0.50`
  - NBA: `EXP(W%) = 0.000351(PS - PA) + 0.50`

</details>

---

<details>
<summary><strong><code>pearson</code></strong> — Pearson Correlation Coefficient</summary>

Computes the Pearson correlation coefficient (r) to measure the linear relationship between two continuous variables. Values range from -1 (perfect negative correlation) to 1 (perfect positive correlation), with 0 indicating no linear relationship. Includes hypothesis testing, p-values, and confidence intervals using Fisher's Z-transformation.

```bash
# Compute correlation from command-line values
pearson --x 1,2,3,4,5 --y 2.1,3.8,6.2,7.9,10.1

# Load data from CSV file
pearson --file data.csv --x-col height --y-col weight

# Include hypothesis test at α=0.05 significance level
pearson --x 10,20,30,40,50 --y 15,28,41,55,68 --alpha 0.05

# One-tailed test for positive correlation
pearson --x 1,2,3,4,5 --y 2,4,5,4,5 --alpha 0.05 --sided one

# Custom precision for output
pearson --x 1,2,3,4 --y 2,4,6,8 --precision 4
```

**Options:**

| Flag | Long form | Description |
|------|-----------|-------------|
| | `--x` | Comma-separated x values (use with `--y`) |
| | `--y` | Comma-separated y values (required with `--x`) |
| | `--file` | CSV file path (use with `--x-col` and `--y-col`) |
| | `--x-col` | Column name for x values in CSV (required with `--file`) |
| | `--y-col` | Column name for y values in CSV (required with `--file`) |
| | `--alpha` | Significance level for hypothesis test and CI (e.g., `0.05`) |
| | `--sided` | Hypothesis test type: `one` or `two` (default: `two`) |
| `-P` | `--precision` | Decimal places for printed values (default: `6`) |

> `--x` and `--file` are mutually exclusive; one is required.<br>
> When using `--x`, must also provide `--y`.<br>
> When using `--file`, must also provide `--x-col` and `--y-col`

**Output includes:**
- **Pearson's r**: Correlation coefficient measuring linear relationship strength
- **r²**: Coefficient of determination (proportion of variance explained)
- **Interpretation**: Qualitative description of correlation strength and direction
- **Hypothesis test** (if `--alpha` provided): t-statistic, p-value, significance result
- **Confidence interval** (if `--alpha` provided): CI for population correlation ρ using Fisher Z-transformation

</details>

---

<details>
<summary><strong><code>spearman</code></strong> — Spearman Rank Correlation Coefficient</summary>

Computes the Spearman rank correlation coefficient (ρ) to measure monotonic relationships between two variables. Unlike Pearson, Spearman evaluates correlation based on ranked data, making it robust to outliers and suitable for ordinal data or non-linear but monotonic relationships. Values range from -1 to 1, with the same interpretation as Pearson correlation.

**Implementation**: Uses scipy for numerically robust t-distribution CDF and inverse normal CDF calculations.

```bash
# Compute rank correlation from command-line values
spearman --x 1,2,3,4,10 --y 2,3,5,6,20

# Load data from CSV file (e.g., survey Likert scales)
spearman --file survey.csv --x-col satisfaction --y-col loyalty

# Include hypothesis test at α=0.01 significance level
spearman --x 100,150,120,180,200 --y 5,3,4,2,1 --alpha 0.01

# One-tailed test with rank display for diagnostic inspection
spearman --x 1,2,3,4,5 --y 2,4,5,4,5 --alpha 0.05 --sided one --show-ranks

# Analyze ordinal data with tied values
spearman --x 1,2,2,3,4 --y 1,2,3,4,5 --show-ranks --precision 3
```

**Options:**

| Flag | Long form | Description |
|------|-----------|-------------|
| | `--x` | Comma-separated x values (use with `--y`) |
| | `--y` | Comma-separated y values (required with `--x`) |
| | `--file` | CSV file path (use with `--x-col` and `--y-col`) |
| | `--x-col` | Column name for x values in CSV (required with `--file`) |
| | `--y-col` | Column name for y values in CSV (required with `--file`) |
| | `--alpha` | Significance level for hypothesis test and CI (e.g., `0.05`) |
| | `--sided` | Hypothesis test type: `one` or `two` (default: `two`) |
| | `--show-ranks` | Display ranked data table for diagnostic inspection |
| `-P` | `--precision` | Decimal places for printed values (default: `6`) |

> `--x` and `--file` are mutually exclusive; one is required.<br>
> When using `--x`, must also provide `--y`.<br>
> When using `--file`, must also provide `--x-col` and `--y-col`<br>
> **Use `--show-ranks` to inspect how ties are handled** (tied values receive average rank)

**Output includes:**
- **Spearman's ρ**: Rank correlation coefficient measuring monotonic relationship strength
- **ρ²**: Coefficient of determination for ranks
- **Interpretation**: Qualitative description of correlation strength and direction
- **Rank table** (if `--show-ranks` provided): Original values and their assigned ranks
- **Hypothesis test** (if `--alpha` provided): t-statistic, p-value, significance result
- **Confidence interval** (if `--alpha` provided): CI for population correlation ρ using Fisher Z-transformation

**When to use Spearman vs. Pearson:**
- **Spearman**: Ordinal data, non-linear but monotonic relationships, presence of outliers, or when distribution assumptions are violated
- **Pearson**: Continuous data with linear relationships and approximate normality

</details>

---

<details>
<summary><strong><code>linreg</code></strong> — Linear Regression (OLS)</summary>

Performs simple linear regression using ordinary least squares (OLS) to fit a line `y = slope * x + intercept`. Provides comprehensive statistical inference including coefficient tests, model fit statistics, and predictions with confidence and prediction intervals.

```bash
# Fit a line to comma-separated x, y values
linreg --x 1,2,3,4,5 --y 2.1,3.9,6.2,7.8,10.1

# Load data from CSV file
linreg --file data.csv --x-col height --y-col weight

# With prediction at x=6
linreg --x 1,2,3,4,5 --y 2,4,6,8,10 --predict 6

# With 90% confidence intervals (α=0.10)
linreg --x 10,20,30,40,50 --y 15,28,41,55,68 --alpha 0.10 --predict 60

# Custom precision
linreg --x 1,2,3,4 --y 2.1,4.3,5.8,8.2 --precision 4
```

**Options:**

| Flag | Long form | Description |
|------|-----------|-------------|
| | `--x` | Comma-separated x values (use with `--y`) |
| | `--y` | Comma-separated y values (required with `--x`) |
| | `--file` | CSV file path (use with `--x-col` and `--y-col`) |
| | `--x-col` | Column name for x values in CSV (required with `--file`) |
| | `--y-col` | Column name for y values in CSV (required with `--file`) |
| | `--predict` | x value for prediction with confidence/prediction intervals |
| | `--alpha` | Significance level for confidence intervals (default: `0.05`) |
| `-P` | `--precision` | Decimal places for printed values (default: `6`) |

> `--x` and `--file` are mutually exclusive; one is required<br>
> When using `--x`, must also provide `--y`<br>
> When using `--file`, must also provide `--x-col` and `--y-col`

**Output includes:**
- **Model equation**: Fitted line `y = slope * x + intercept`
- **Coefficients**: Slope and intercept with standard errors, t-statistics, p-values, and confidence intervals
- **Model fit**: R² (coefficient of determination), residual standard error, F-statistic, overall model significance
- **Predictions** (if `--predict` specified):
  - Point estimate at the specified x value
  - Confidence interval for the mean response (where we expect the average y)
  - Prediction interval for an individual observation (wider, accounts for individual variability)

**Statistical notes:**
- **R²**: Proportion of variance in y explained by x (0 to 1; higher is better)
- **Confidence interval**: Uncertainty in estimating the mean response
- **Prediction interval**: Uncertainty for a single new observation (always wider than confidence interval)
- **t-tests**: Test whether each coefficient is significantly different from zero
- **F-statistic**: Tests whether the overall model is significant (better than just predicting the mean)

</details>

---

<details>
<summary><strong><code>sample</code></strong> — Sample Size Calculator</summary>

Calculates the minimum sample size needed for statistical studies. Supports proportion estimation within a margin of error, mean difference detection with specified power, and two-proportion comparisons. Includes power analysis sweeps to show achieved power across a range of sample sizes.

```bash
# Minimum n to estimate a proportion within ±3% at 95% confidence
sample --type proportion --prop 0.5 --margin 0.03

# Minimum n to detect a mean shift of 5 units (σ=12) with 80% power
sample --type mean --delta 5 --std 12 --power 0.80

# Two-proportion comparison: detect difference between 40% and 50%
sample --type comparison --p1 0.40 --p2 0.50 --alpha 0.05 --power 0.80

# Power analysis sweep: show achieved power for n = 50 to 300
sample --type mean --delta 5 --std 12 --sweep 50 300 --step 25

# One-sided test with 90% power
sample --type mean --delta 3 --std 8 --power 0.90 --sided one
```

**Options:**

| Flag | Long form | Description |
|------|-----------|-------------|
| | `--type` | Calculation type: `proportion`, `mean`, or `comparison` _(required)_ |
| | `--prop` | Expected proportion for proportion estimation (0 to 1) |
| | `--margin` | Desired margin of error for proportion _(e.g., `0.03` for ±3%)_ |
| | `--std`, `--sigma` | Population standard deviation for mean calculations |
| | `--delta` | Minimum detectable effect size (mean difference) |
| | `--p1` | Proportion in group 1 for comparison |
| | `--p2` | Proportion in group 2 for comparison |
| | `--alpha` | Significance level (default: `0.05`) |
| | `--power` | Statistical power (1-β) for mean/comparison _(default: `0.80`)_ |
| | `--sided` | Test type: `one` or `two` _(default: `two`)_ |
| | `--sweep MIN MAX` | Show power across range of sample sizes |
| | `--step` | Step size for sweep (default: `10`) |
| `-P` | `--precision` | Decimal places for printed values (default: `4`) |

**Calculation types:**
- **Proportion**: Determines sample size to estimate a single proportion within a specified margin of error at a given confidence level
- **Mean**: Determines sample size to detect a mean difference (effect size) with specified statistical power
- **Comparison**: Determines sample size per group to detect a difference between two proportions with specified power

</details>

---

<details>
<summary><strong><code>bootci</code></strong> — Bootstrap Confidence Intervals</summary>

Computes non-parametric confidence intervals for a statistic (mean, median, or standard deviation) using bootstrap resampling. This distribution-free method makes no assumptions about the underlying population distribution, making it ideal when normality assumptions are questionable or for small sample sizes.

```bash
# Compute 95% CI for the mean of a dataset
bootci --data 10 20 30 40 50 --stat mean

# Compute 90% CI for the median with 50,000 bootstrap resamples
bootci --data 14.2 13.8 15.1 14.5 13.9 15.3 --stat median --n-bootstrap 50000 --confidence 0.90

# Compute CI for standard deviation with custom precision
bootci --data 1.2 3.4 5.6 7.8 9.1 --stat stdev --precision 3

# Use a random seed for reproducibility
bootci --data 5.5 6.2 7.1 5.8 6.5 --stat mean --seed 42
```

**Options:**

| Flag | Long form | Description |
|------|-----------|-------------|
| | `--data` | Space-separated data points. If not provided, reads from stdin |
| | `--stat` | Statistic to compute: `mean`, `median`, or `stdev` (required) |
| | `--n-bootstrap` | Number of bootstrap resamples (default: `10000`) |
| | `--confidence` | Confidence level, e.g., `0.95` for 95% CI (default: `0.95`) |
| | `--seed` | Random seed for reproducibility |
| `-P` | `--precision` | Decimal places for printed values (default: `4`) |

**Method:**
- Uses the **percentile method**: generates bootstrap resamples by sampling with replacement from the original data
- Each resample has the same size as the original dataset
- Computes the statistic for each resample
- Confidence interval is derived from the appropriate percentiles of the bootstrap distribution

**When to use bootstrap:**
- **Small sample sizes** where asymptotic normality assumptions are unreliable
- **Non-normal distributions** or skewed data
- **Robust statistics** like median or standard deviation
- **No parametric assumptions** available or appropriate

**Example output:**
```text
Data: n=6, statistic=mean
Original mean: 15.4833
Bootstrap samples: 10000
95.0% Confidence Interval: [13.7000, 17.1167]
```

**When to use `bootci` vs `confint`:**
- Use **`bootci`** when you have raw data and want distribution-free intervals (no normality assumptions required)
- Use **`confint`** when you have summary statistics (n, k, mean, std) or need specific parametric methods (Wilson, Clopper-Pearson, t-interval, Poisson)
- See the `confint` section below for parametric alternatives

</details>

---

<details>
<summary><strong><code>confint</code></strong> — Confidence Intervals (Parametric Methods)</summary>

Calculates confidence intervals using formula-based parametric methods for proportions, means, and count data. Unlike bootstrap (which requires raw data), these methods work with summary statistics and use established statistical formulas. Includes support for sample size sweeps.

```bash
# Wilson score interval for proportion (47 successes out of 120 trials)
confint --method wilson --n 120 --k 47

# Clopper-Pearson exact interval (conservative)
confint --method clopper-pearson --n 30 --k 22

# Normal approximation for proportion (requires large n)
confint --method normal --n 1000 --k 480

# t-interval for mean (sample size 15, mean 23.4, std 4.1)
confint --method t --n 15 --mean 23.4 --std 4.1

# Poisson interval for count data (observed 23 events)
confint --method poisson --count 23

# 90% confidence level instead of default 95%
confint --method wilson --n 100 --k 35 --alpha 0.10

# Sweep sample sizes to see how interval width changes
confint --method wilson --p 0.4 --sweep 50 500 --step 50
```

**Options:**

| Flag | Long form | Description |
|------|-----------|-------------|
| | `--method` | Interval method: `wilson` (default), `clopper-pearson`, `normal`, `t`, or `poisson` |
| | `--alpha` | Significance level (default: `0.05` for 95% CI) |
| | `--n` | Sample size (for proportion or t-interval) |
| | `--k` | Number of successes (for proportion interval) |
| | `--p` | Proportion (alternative to --k; used with --sweep) |
| | `--mean` | Sample mean (for t-interval) |
| | `--std` | Sample standard deviation (for t-interval) |
| | `--count` | Observed count (for Poisson interval) |
| | `--sweep START END` | Sweep sample sizes from START to END |
| | `--step` | Step size for sweep (default: `10`) |
| `-P` | `--precision` | Decimal places for output (default: `4`) |

**Methods explained:**

| Method | Use case | Advantages |
|--------|----------|------------|
| **wilson** | Proportions (default) | More reliable than normal approximation for small n or extreme p; recommended for general use |
| **clopper-pearson** | Proportions (exact) | Conservative exact intervals; guaranteed coverage; good for small samples |
| **normal** | Proportions (large n) | Simple formula but only appropriate when n·p ≥ 5 and n·(1-p) ≥ 5 |
| **t** | Population means | Standard approach for means when population std is unknown; uses t-distribution |
| **poisson** | Count/rate data | Exact intervals for Poisson processes (events per time, defects per unit, etc.) |

**Example output:**
```text
95.0% Confidence Interval for Proportion
Method: wilson
Sample: n = 120, k = 47, p̂ = 0.3917
Interval: [0.3087, 0.4809]
Width: 0.1722
Margin: ±0.0861
```

**`bootci` vs `confint` comparison:**

| Aspect | `bootci` | `confint` |
|--------|----------|-----------|
| **Input** | Raw data points | Summary statistics (n, k, mean, std) |
| **Approach** | Non-parametric resampling | Parametric formulas |
| **Assumptions** | Minimal (distribution-free) | Varies by method (normality for t, etc.) |
| **Speed** | Slower (10,000+ resamples) | Fast (direct calculation) |
| **Statistics** | Any (mean, median, stdev, custom) | Specific: proportions, means, counts |
| **Best for** | Small samples, non-normal data, median | Standard inferential statistics, summary data |

</details>

---

<details>
<summary><strong><code>pvalue</code></strong> — Hypothesis Testing & p-values</summary>

Computes p-values for statistical hypothesis tests on proportions, means, and categorical data. Supports z-tests, binomial exact tests, chi-squared goodness-of-fit, and t-tests. Includes alpha sensitivity analysis with sweep mode to visualize how rejection decisions change at different significance levels.

```bash
# Z-test for proportion: 480 successes out of 1000 trials vs p₀ = 0.5
pvalue --test z --n 1000 --k 480 --p0 0.5

# One-sided z-test (testing if proportion is less than 0.5)
pvalue --test z --n 1000 --k 480 --p0 0.5 --sided less

# Binomial exact test (better for small samples)
pvalue --test binom-exact --n 30 --k 22 --p0 0.60

# One-sample t-test for mean
pvalue --test t --n 30 --mean 105 --std 15 --mu0 100

# Chi-squared goodness-of-fit test
pvalue --test chi2 --observed "30,20,50" --expected "33.33,33.33,33.33"

# Alpha sweep: see how decision changes from α=0.01 to α=0.10
pvalue --test t --n 30 --mean 105 --std 15 --mu0 100 --sweep-alpha 0.01 0.10 --step 0.01

# Custom significance level (α=0.01 instead of default 0.05)
pvalue --test z --n 500 --k 310 --p0 0.5 --alpha 0.01
```

**Options:**

| Flag | Long form | Description |
|------|-----------|-------------|
| | `--test` | Test type: `z`, `binom-exact`, `t`, or `chi2` (required) |
| | `--alpha` | Significance level (default: `0.05`) |
| | `--sided` | Two-sided, less, or greater: `two`, `less`, `greater` (default: `two`) |
| | `--n` | Sample size (for z, binom-exact, or t) |
| | `--k` | Number of successes (for z or binom-exact) |
| | `--p0` | Null hypothesis proportion (for z or binom-exact) |
| | `--mean` | Sample mean (for t-test) |
| | `--std` | Sample standard deviation (for t-test) |
| | `--mu0` | Null hypothesis mean (for t-test) |
| | `--observed` | Comma-separated observed frequencies (for chi2) |
| | `--expected` | Comma-separated expected frequencies (for chi2) |
| | `--sweep-alpha START END` | Show rejection decisions from START to END |
| | `--step` | Step size for alpha sweep (default: `0.01`) |
| `-P` | `--precision` | Decimal places for output (default: `4`) |

**Tests explained:**

| Test | Use case | Null hypothesis |
|------|----------|-----------------|
| **z** | Single proportion, large n | Population proportion equals p₀ |
| **binom-exact** | Single proportion, any n | Population proportion equals p₀ (exact, no approximation) |
| **t** | Single mean, unknown σ | Population mean equals μ₀ |
| **chi2** | Categorical data, goodness-of-fit | Observed frequencies match expected distribution |

**Example output:**
```text
One-sample t-test (two-sided)
Significance level: α = 0.05

Sample: n = 30, mean = 105.0000, std = 15.0000
Null hypothesis: H₀: μ = 100

Test statistic: t = 1.8257
p-value: 0.0783
Effect size: 0.3333

Decision: Fail to reject H₀ (p ≥ α)
```

**Alpha sweep output example:**
```text
One-sample t-test - Alpha Sweep
Test statistic (t): 1.8257
p-value: 0.0783

   Alpha   Reject H0
--------------------
  0.0100          No
  0.0200          No
  0.0300          No
  0.0400          No
  0.0500          No
  0.0600          No
  0.0700         Yes
  0.0800         Yes
  0.0900         Yes
  0.1000         Yes
```

</details>

---

<details>
<summary><strong><code>ttest</code></strong> — T-Test (One-Sample, Two-Sample, Paired)</summary>

Performs one-sample, two-sample (Welch's), and paired t-tests with full statistical output: t-statistic, degrees of freedom, p-value, confidence interval for the mean (or mean difference), and Cohen's d effect size. Accepts either raw comma-separated values or pre-computed summary statistics (mean, std, n). Supports two-sided and one-sided alternatives.

```bash
# One-sample: test if a dataset's mean equals 3.0
ttest one-sample --values 2.1,3.4,2.9,3.1,2.8 --mu0 3.0

# One-sample from summary statistics (mean=105, std=12, n=25)
ttest one-sample --mean 105 --std 12 --n 25 --mu0 100

# Two-sample (Welch's): compare two independent groups
ttest two-sample --values1 1.2,2.3,3.1,2.8 --values2 2.1,3.2,4.1,3.9

# Two-sample from summary statistics
ttest two-sample --mean1 12.4 --std1 2.1 --n1 30 --mean2 10.8 --std2 1.9 --n2 28

# Paired: test pre/post measurements on the same subjects
ttest paired --values1 85,90,78,92,88 --values2 90,95,82,95,91

# One-sided test (greater)
ttest one-sample --values 2.1,3.4,2.9 --mu0 2.5 --sided greater

# Custom significance level and precision
ttest two-sample --values1 10,12,14,11 --values2 8,9,11,10 --alpha 0.01 --precision 6
```

**Subcommands:**

| Subcommand | Null hypothesis | Use case |
|---|---|---|
| `one-sample` | H₀: μ = μ₀ | Test if a sample mean differs from a known reference value |
| `two-sample` | H₀: μ₁ = μ₂ | Compare means of two independent groups (unequal variances OK) |
| `paired` | H₀: μ_d = 0 | Compare matched before/after or paired observations |

**Options (shared across subcommands):**

| Flag | Long form | Description |
|------|-----------|-------------|
| `-a` | `--alpha` | Significance level (default: `0.05`) |
| `-S` | `--sided` | Alternative hypothesis: `two` (default), `less`, or `greater` |
| `-P` | `--precision` | Decimal places for output (default: `4`) |

**`one-sample` options:**

| Flag | Long form | Description |
|------|-----------|-------------|
| `-v` | `--values` | Comma-separated raw sample values |
| | `--mean` | Sample mean (use with `--std` and `--n`) |
| | `--std` | Sample standard deviation |
| | `--n` | Sample size |
| | `--mu0` | Hypothesised population mean _(required)_ |

> Provide either `--values` or all of `--mean`, `--std`, `--n`.

**`two-sample` options:**

| Flag | Long form | Description |
|------|-----------|-------------|
| `-v1` | `--values1` | Comma-separated values for group 1 |
| `-v2` | `--values2` | Comma-separated values for group 2 |
| | `--mean1`, `--std1`, `--n1` | Summary statistics for group 1 |
| | `--mean2`, `--std2`, `--n2` | Summary statistics for group 2 |

> Each group takes either raw values or summary stats; groups may mix the two forms.

**`paired` options:**

| Flag | Long form | Description |
|------|-----------|-------------|
| `-v1` | `--values1` | Comma-separated pre/group-1 values _(required)_ |
| `-v2` | `--values2` | Comma-separated post/group-2 values _(required)_ |

**Output includes:**
- **t-statistic** and **degrees of freedom** (Welch-Satterthwaite for two-sample)
- **p-value** adjusted for the chosen alternative hypothesis
- **Confidence interval** for the mean (or mean difference)
- **Cohen's d** effect size
- **Reject / fail-to-reject decision** at the specified α

**Example output:**
```text
One-sample t-test  (two-sided)
H₀: μ = 3.0000

  n:          5
  mean:       2.8600
  std dev:    0.4827

  t-stat:     -0.6479
  df:         4.0000
  p-value:    0.5514
  95% CI:    [2.2605, 3.4595]
  Cohen's d:  -0.2900

  Fail to reject H₀  (p ≥ α = 0.05)
```

</details>

---

<details>
<summary><strong><code>forecast</code></strong> — Time Series Forecasting with Prediction Intervals</summary>

Fits exponential smoothing models to time series data and generates point forecasts with prediction intervals. Supports simple exponential smoothing (level only), double exponential smoothing (Holt's method: level + trend), and Holt-Winters exponential smoothing (level + trend + seasonal). Useful for sales forecasting, demand planning, and trend analysis.

```bash
# Simple exponential smoothing: 6-period forecast from inline data
forecast --data 120,135,148,130,142,155,160 --method simple --periods 6

# Double exponential smoothing (Holt's method) from CSV file
forecast --data sales.csv --method double --periods 12

# Holt-Winters with weekly seasonality (7-day cycle)
forecast --data data.csv --method holt-winters --seasonal-period 7 --periods 14

# Holt-Winters with monthly seasonality from CSV file
forecast --data monthly_sales.csv --method holt-winters --seasonal-period 12 --periods 6

# Backtest: hold out last 4 observations to evaluate accuracy
forecast --data data.csv --method double --periods 4 --backtest 4

# Custom smoothing parameters with 90% prediction intervals
forecast --data sales.csv --method double --alpha 0.4 --beta 0.2 --periods 10 --confidence 0.90

# Output as JSON
forecast --data 100,110,120,130,140 --method simple --periods 3 --format json
```

**Options:**

| Flag | Long form | Description |
|------|-----------|-------------|
| | `--data` | CSV file path or comma-separated values (required) |
| | `--method` | Forecasting method: `simple`, `double`, or `holt-winters` (default: `simple`) |
| | `--periods` | Number of periods to forecast (default: `1`) |
| | `--alpha` | Level smoothing parameter, 0 < α ≤ 1 (default: `0.3`) |
| | `--beta` | Trend smoothing parameter, 0 < β ≤ 1 (default: `0.1`) |
| | `--gamma` | Seasonal smoothing parameter, 0 < γ ≤ 1 (default: `0.1`) |
| | `--seasonal-period` | Seasonal period for Holt-Winters (default: `12`) |
| | `--seasonal-type` | Seasonal component: `additive` or `multiplicative` (default: `additive`) |
| | `--confidence` | Confidence level for prediction intervals, ≥0.5 and <1.0 (default: `0.95`) |
| | `--format` | Output format: `table` (default), `json`, or `csv` |
| | `--precision` | Decimal precision for output (default: `2`) |
| | `--backtest K` | Hold out last K observations for backtesting and compute RMSE/MAE |

**Forecasting methods:**
- **Simple exponential smoothing**: Best for data with no clear trend or seasonality; smooths level only
- **Double exponential smoothing (Holt's method)**: Captures level and linear trend; suitable for trending data without seasonality
- **Holt-Winters**: Captures level, trend, and seasonal patterns; ideal for data with recurring seasonal cycles (weekly, monthly, quarterly, etc.)

**Smoothing parameters:**
- **α (alpha)**: Controls how quickly the model adapts to changes in the level
- **β (beta)**: Controls how quickly the model adapts to changes in the trend
- **γ (gamma)**: Controls how quickly the model adapts to changes in the seasonal pattern

Higher values (closer to 1) place more weight on recent observations; lower values smooth out noise but respond more slowly to changes.

**Example output:**
```text
==============================================================
FORECAST
==============================================================
Method: double
Periods ahead: 6
Confidence level: 95.0%

Period   Forecast   Lower (95%)   Upper (95%)
------   --------   -----------   -----------
1        168.00     162.34        173.66
2        176.00     168.21        183.79
3        184.00     173.95        194.05
4        192.00     179.58        204.42
5        200.00     185.13        214.87
6        208.00     190.60        225.40
==============================================================
Model Statistics:
  RMSE: 5.12
  MAE:  4.23
```

</details>

---

<details>
<summary><strong><code>collatz</code></strong> — Collatz Conjecture (3n+1 Problem)</summary>

Evaluates the Collatz conjecture for positive integers up to n. For each starting value, follows the Collatz sequence (if even, divide by 2; if odd, multiply by 3 and add 1) to determine whether it eventually reaches 1. Reports the largest consecutive integer k such that all integers from 1 to k are verified to reach 1.

```bash
# Check integers from 1 to 10000
collatz --n 10000

# Check with verbose progress output every 1000 integers
collatz --n 100000 --interval 1000 --verbose

# Check a large range with less frequent progress updates
collatz --n 1000000 --interval 10000
```

**Options:**

| Flag | Long form | Description |
|------|-----------|-------------|
| | `--n` | Upper bound (inclusive) of starting integers to check (required) |
| | `--interval` | How often (in starts) to update/print progress (default: `1000`) |
| | `--verbose` | Print progress messages during computation |

**Background:**
The Collatz conjecture (also known as the 3n+1 problem) is an unsolved problem in mathematics. The conjecture states that for any positive integer, repeatedly applying these rules will eventually reach 1:
- If the number is even, divide it by 2
- If the number is odd, multiply by 3 and add 1

For example, starting with 6: 6 → 3 → 10 → 5 → 16 → 8 → 4 → 2 → 1

**Example output:**
```bash
$ collatz --n 10000
max_valid = 10000
```

This indicates that all integers from 1 to 10000 have been verified to eventually reach 1.

</details>

---

<details>
<summary><strong><code>jevons</code></strong> — Jevons' Paradox Rebound Effect Analysis</summary>

Quantifies the direct rebound effect arising from resource efficiency improvements. When technology lowers the effective cost per unit of an energy service, demand rises — partially or fully offsetting the expected savings. In extreme cases (backfire), net consumption exceeds the original baseline despite efficiency gains.

```bash
# Standard analysis: 30% efficiency gain with elasticity 0.5
jevons --efficiency 0.30 --elasticity 0.5

# With absolute baseline and resource label
jevons -e 0.30 -d 0.5 --baseline 1000 --resource coal

# Sweep efficiency from 10% to 90% at fixed elasticity
jevons --sweep-efficiency 0.1 0.9 --elasticity 0.7 --step 0.1

# Sweep elasticity from 0 to 2.0 at fixed efficiency
jevons --efficiency 0.30 --sweep-elasticity 0.0 2.0 --step 0.5

# JSON output
jevons --efficiency 0.30 --elasticity 0.5 --format json
```

**Options:**

| Flag | Long form | Description |
|------|-----------|-------------|
| `-e` | `--efficiency` | Efficiency improvement fraction (0 < η < 1), e.g., `0.30` for 30% |
| `-d` | `--elasticity` | Absolute price elasticity of demand (ε ≥ 0), e.g., `0.5` |
| `-b` | `--baseline` | Baseline resource consumption (default: `1.0`) |
| `-r` | `--resource` | Resource label for display (default: `units`) |
| | `--sweep-efficiency MIN MAX` | Sweep efficiency from MIN to MAX (requires `--elasticity`) |
| | `--sweep-elasticity MIN MAX` | Sweep elasticity from MIN to MAX (requires `--efficiency`) |
| | `--step` | Step size for sweep (default: `0.05`) |
| `-f` | `--format` | Output format: `table` (default) or `json` |
| `-P` | `--precision` | Decimal places for printed values (default: `4`) |

> `--sweep-efficiency` and `--sweep-elasticity` are mutually exclusive.

**Key formulas:**
- **Expected savings** (no rebound): `baseline × η`
- **Rebound consumption**: `baseline × ε × η × (1 − η)`
- **Net consumption**: `baseline × (1 + ε × η) × (1 − η)`
- **Rebound rate**: `ε × (1 − η)` as a fraction of expected savings
- **Backfire condition**: rebound rate > 100% (net consumption exceeds baseline)

**Rebound outcomes:**

| Rebound rate | Outcome |
|---|---|
| 0% | Full conservation — all efficiency gains become savings |
| 1–49% | Weak rebound — most efficiency gains translate to savings |
| 50–99% | Strong rebound — efficiency gains are significantly offset |
| 100% | Full rebound — efficiency gains are exactly cancelled |
| > 100% | **BACKFIRE** — consumption increases despite efficiency improvement |

**Example output:**
```text
Jevons' Paradox Analysis
============================================
Efficiency improvement:  30.0000%
Price elasticity:        0.5000
Baseline consumption:    1.0000 units

Expected savings (no rebound):  0.3000 units (30.0000%)
Rebound consumption:            0.1050 units
Actual savings (after rebound): 0.1950 units (19.5000%)
Net consumption:                0.8050 units (80.5000%)
Rebound rate:                   35.0000% of expected savings

Outcome: Strong rebound — efficiency gains are significantly offset
```

</details>

---

<details>
<summary><strong><code>sigmoid</code></strong> — Sigmoid Function</summary>

Evaluates σ(x) = 1/(1+e^(−x)), its derivative σ'(x), and the inverse logit. Supports single-value evaluation, range tables, and a Unicode sparkline of the sigmoid curve.

```bash
# Sigmoid value at x = 2.0
sigmoid -x 2.0

# Sigmoid and its derivative at x = 0
sigmoid -x 0 --derivative

# Range table from -5 to 5 in steps of 1
sigmoid --range -5 5 1

# Range table with derivative column and sparkline
sigmoid --range -4 4 0.5 --derivative --plot

# Inverse logit: find x such that σ(x) = 0.75
sigmoid --inverse --prob 0.75
```

**Options:**

| Flag | Long form | Description |
|------|-----------|-------------|
| `-x` | `--value` | Evaluate sigmoid at a single value |
| | `--range MIN MAX STEP` | Print a table over [MIN, MAX] in steps of STEP |
| | `--inverse` | Compute the inverse logit (logit function) |
| `-p` | `--prob` | Probability in (0, 1) for `--inverse` mode |
| `-d` | `--derivative` | Also show σ'(x) |
| | `--plot` | Append a Unicode sparkline of the sigmoid curve |
| `-P` | `--precision` | Decimal places (default: `6`) |

> `-x/--value`, `--range`, and `--inverse` are mutually exclusive; one is required.

</details>

---

<details>
<summary><strong><code>euler</code></strong> — Euler's Number</summary>

Explores Euler's number e ≈ 2.71828 through multiple lenses: the classic limit definition (1 + 1/n)^n, Taylor series for e^x and ln(x), Euler's identity e^(iπ) + 1 = 0, and the Euler-Mascheroni constant γ ≈ 0.5772. Supports table and JSON output.

```bash
# Convergence table for (1+1/n)^n up to n = 1,000,000
euler --approx 1000000

# e^2 via 10th-order Taylor series, compared to math.exp
euler --exp 2 --order 10 --compare

# Display Euler's identity at 15 decimal places
euler --identity --precision 15

# ln(10) via arctanh series with comparison
euler --ln 10 --compare

# Euler-Mascheroni constant γ
euler --mascheroni

# JSON output for downstream processing
euler --exp 1 --compare --format json
```

**Options:**

| Flag | Long form | Description |
|------|-----------|-------------|
| | `--approx N` | Show convergence of (1+1/N)^N to e at logarithmic steps |
| | `--exp X` | Compute e^X via Taylor series |
| | `--identity` | Display Euler's identity: e^(iπ) + 1 = 0 |
| | `--ln X` | Approximate ln(X) via arctanh series |
| | `--mascheroni` | Display the Euler-Mascheroni constant γ |
| | `--order N` | Number of series terms for `--exp` / `--ln` (default: `20`) |
| | `--compare` | Compare series result to `math.exp` / `math.log` reference |
| `-P` | `--precision` | Decimal places (default: `10`) |
| `-f` | `--format` | Output format: `table` (default) or `json` |

> `--approx`, `--exp`, `--identity`, `--ln`, and `--mascheroni` are mutually exclusive; one is required.

</details>

---

<details>
<summary><strong><code>gini</code></strong> — Gini Coefficient and Lorenz Curve</summary>

Computes the Gini coefficient and Lorenz curve for inequality measurement. Supports raw observations, optional positive weights for weighted samples, and pre-aggregated population/income share pairs (grouped mode). Groups are sorted internally by ascending per-capita income before constructing the Lorenz curve. Includes relative mean absolute difference (RMAD = 2 × Gini) and an optional n/(n−1) small-sample bias correction. Multi-dataset mode ranks all datasets by Gini from most equal to least equal.

```bash
# Single dataset — point estimate
gini --data 1 2 3 4 5

# Weighted samples
gini --data 10 30 50 80 --weights 4 3 2 1

# Lorenz curve coordinates
gini --data 1 2 3 4 5 --lorenz --precision 4

# With bias correction
gini --data 1 2 3 4 5 --correct

# Grouped (population share / income share pairs)
gini --groups 0.2 0.05 0.3 0.15 0.5 0.80

# Compare multiple datasets
gini --data 1 2 3 --data 5 5 90 --data 30 35 35

# JSON output
gini --data 1 2 3 4 5 --format json
```

**Options:**

| Flag | Long form | Description |
|------|-----------|-------------|
| | `--data` | Space-separated observation values; repeat flag for multiple datasets |
| | `--groups` | Alternating population and income shares: `POP INC POP INC ...` |
| | `--weights` | Space-separated positive weights parallel to `--data` (single dataset only) |
| | `--lorenz` | Display Lorenz curve coordinates (single dataset only) |
| | `--correct` | Apply n/(n−1) small-sample bias correction |
| `-f` | `--format` | Output format: `table` (default) or `json` |
| `-P` | `--precision` | Decimal places for printed values (default: `6`) |

> `--data` and `--groups` are mutually exclusive; one is required.<br>
> `--weights` and `--lorenz` require a single `--data` group.

</details>

---

<details>
<summary><strong><code>crt</code></strong> — Sunzi's Theorem Solver</summary>

Solves systems of simultaneous congruences using Sunzi's Theorem (general CRT). Handles both coprime and non-coprime moduli via pairwise merging (solution exists iff aᵢ ≡ aⱼ mod gcd(nᵢ, nⱼ) for all pairs). Remainders are normalized to [0, n) automatically. Returns the unique solution x in [0, N) and the combined modulus N = lcm(n₁, ..., nₖ). Supports listing all solutions up to a user-specified bound.

```bash
# Solve x≡2(mod 3), x≡3(mod 5), x≡2(mod 7) → 23 (mod 105)
crt --solve 2 3 3 5 2 7

# Non-coprime moduli (x≡0(mod 4), x≡2(mod 6))
crt --solve 0 4 2 6

# List all solutions up to 300
crt --solve 2 3 3 5 2 7 --all 300

# JSON output
crt --solve 2 3 3 5 2 7 --format json
```

**Options:**

| Flag | Long form | Description |
|------|-----------|-------------|
| | `--solve A₁ N₁ [A₂ N₂ ...]` | Alternating remainder/modulus pairs. `Aᵢ` is the remainder (residue); `Nᵢ` is the modulus (divisor). Each pair encodes one congruence x ≡ Aᵢ (mod Nᵢ). Requires ≥ 1 pair. |
| | `--all MAX` | List all solutions in [0, MAX] |
| `-f` | `--format` | Output format: `table` (default) or `json` |

> **Pair order:** Always remainder then modulus. `--solve 2 3` → x ≡ 2 (mod 3). `--solve 2 3 3 5` → x ≡ 2 (mod 3) **and** x ≡ 3 (mod 5).

</details>

---

<details>
<summary><strong><code>subnet</code></strong> — Subnet Mask Calculator</summary>

Computes network address, broadcast address, first/last usable IP, subnet mask, usable host count, and classful network count from an IPv4 address with optional CIDR prefix. Handles `/32` host routes (single host, no network/broadcast distinction) and `/31` point-to-point links (RFC 3021, both addresses usable). Default prefix is `/32` when omitted. Classful network count uses `2^(prefix − classful_prefix)` for subnets; supernets return `1`.

```bash
# Address with embedded CIDR
subnet 192.168.1.50/24

# CIDR as separate flag (overrides any embedded prefix)
subnet 10.0.0.1 --cidr 8

# Supernet example
subnet 172.16.5.100/20

# JSON output
subnet 192.168.1.50/24 --format json
```

**Options:**

| Flag | Long form | Description |
|------|-----------|-------------|
| | `ADDRESS` | IPv4 address with optional CIDR notation (e.g. `192.168.1.50/24`) |
| `-c` | `--cidr PREFIX` | Prefix length 0–32; overrides any CIDR embedded in ADDRESS (default: 32) |
| `-f` | `--format` | Output format: `table` (default) or `json` |

</details>

---

<details>
<summary><strong><code>geometric</code></strong> — Geometric Distribution</summary>

Computes PMF, CDF, survival, mean, and variance for the geometric distribution — the number of independent Bernoulli trials until the first success (support k = 1, 2, 3, ...). Pure Python, no external dependencies.

```bash
# P(first success on exactly the 5th trial, p=0.3)
geometric -k 5 -p 0.3

# P(needing more than 10 calls to close a sale with 20% close rate)
geometric -k 10 -p 0.2 --survival

# Range table: probability distribution for k = 1 to 15
geometric -p 0.25 --table 1 15
```

**Options:**

| Flag | Long form | Description |
|------|-----------|-------------|
| `-k` | | Trial number of the first success (required unless `--table` is used) |
| `-p` | | Success probability per trial, in (0, 1] |
| | `--survival` | Report P(X > k) instead of the cumulative P(X <= k) |
| | `--table MIN MAX` | Print PMF/CDF/survival for k = MIN..MAX instead of a single report |
| `-P` | `--precision` | Decimal places for output (default: 4) |

</details>

---

<details>
<summary><strong><code>chisq</code></strong> — Chi-Square Test Calculator</summary>

Computes chi-square goodness-of-fit (do observed categorical frequencies match expected ones?) and independence (are two categorical variables associated, from a contingency table?) tests. The chi-square CDF is computed exactly via the regularised incomplete gamma function (series expansion / continued fraction) — pure Python, no external dependencies. Output includes per-cell χ² contributions for residual diagnostics.

```bash
# Goodness-of-fit: are die rolls uniformly distributed?
chisq --test gof --observed 18,22,17,25,19,19 --expected 20,20,20,20,20,20

# Independence: is product preference associated with age group? (2×3 table)
chisq --test independence --table "40,30,20" --table "25,45,30"

# With explicit significance level
chisq --test gof --observed 52,48 --expected 50,50 --alpha 0.10
```

**Options:**

| Flag | Long form | Description |
|------|-----------|-------------|
| | `--test {gof,independence}` | Test type (required) |
| | `--observed O1,O2,...` | Comma-separated observed frequencies (gof mode) |
| | `--expected E1,E2,...` | Comma-separated expected frequencies (gof mode) |
| | `--table R1,R2,...` | One contingency table row; repeat once per row (independence mode) |
| `-a` | `--alpha` | Significance level (default: 0.05) |
| `-P` | `--precision` | Decimal places for output (default: 4) |

</details>

---

<details>
<summary><strong><code>anova</code></strong> — One-Way Analysis of Variance</summary>

Tests whether the means of three or more independent groups are equal — the natural generalization of the two-sample t-test to multiple groups. The F-distribution CDF (and the pairwise t-tests used by the Bonferroni post-hoc) are computed via the regularised incomplete beta function. Tukey HSD uses the studentized range distribution, computed via nested numerical integration (Simpson's rule over the normal and scaled chi densities) — pure Python, no external dependencies. Accepts inline comma-separated groups or a CSV file with group/value columns.

```bash
# One-way ANOVA across three treatment groups
anova --data "12.1,11.8,12.5,11.9" "9.8,10.3,10.1,9.7" "15.2,14.9,15.5,16.0"

# With Tukey HSD post-hoc comparisons
anova --data "12.1,11.8,12.5" "9.8,10.3,10.1" "15.2,14.9,15.5" --posthoc tukey

# With Bonferroni-corrected post-hoc comparisons
anova --data "12.1,11.8,12.5" "9.8,10.3,10.1" "15.2,14.9,15.5" --posthoc bonferroni

# From CSV with group and value columns
anova --file experiment.csv --group-col treatment --value-col response --alpha 0.01
```

**Options:**

| Flag | Long form | Description |
|------|-----------|-------------|
| | `--data GROUP [GROUP ...]` | One comma-separated list of values per group (2 or more groups) |
| | `--file CSV` | Path to a CSV file (use with `--group-col` and `--value-col`) |
| | `--group-col COL` | CSV column name holding the group label |
| | `--value-col COL` | CSV column name holding the numeric observation |
| `-a` | `--alpha` | Significance level (default: 0.05) |
| | `--posthoc {tukey,bonferroni,none}` | Post-hoc pairwise comparison method (default: none) |
| | `--format {table,json}` | Output format (default: table) |
| `-P` | `--precision` | Decimal places for table output (default: 4) |

</details>

---

<details>
<summary><strong><code>life</code></strong> — Life in Weeks</summary>

Renders a human life as a grid of time units — one cell per week, month, or year of a nominal 90-year lifespan — color-coded to distinguish elapsed units from remaining ones. Inspired by Wait But Why's [Your Life in Weeks](https://waitbutwhy.com/2014/05/life-weeks.html).

Cells are discrete shape glyphs (`■` lived, `□` remaining) rather than solid blocks, separated into groups of 13 weeks (or 3 months, 5 years) with a blank line between decades, so individual units stay countable instead of merging into a bar. `--group` overrides the group size; `--group 1` puts a space between every cell and `--group 0` renders an unbroken row.

`--reference` renders "The Life of a Typical American": the same grid with a distinct shape per life phase — `●` childhood/school, `▲` higher education, `■` working years, `◆` retirement, `○` beyond life expectancy — and `★` milestone markers. Phase boundaries and milestone ages are approximate US averages from public sources (US Census Bureau median age at first marriage; CDC/NCHS mean age at first birth and life expectancy; Gallup average actual retirement age) — see the module docstring for figures and years.

`--as-of` measures elapsed time to a date other than today, which makes output reproducible and supports what-if projections. Color is suppressed automatically when output is not a terminal, when `NO_COLOR` is set, or with `--no-color`; the block glyphs alone remain readable.

```bash
# Weeks grid (default), elapsed vs remaining
life 1985-03-14

# One cell per month
life 1985-03-14 --mode months

# One cell per year, over a 100-year lifespan
life 1985-03-14 --mode years --lifespan 100

# Typical-American reference grid (no birthdate needed)
life --reference

# Tighter cell grouping (one space between every cell)
life 1985-03-14 --group 1

# Reproducible / projected output
life 1985-03-14 --as-of 2030-01-01 --format json
```

**Options:**

| Flag | Long form | Description |
|------|-----------|-------------|
| | `BIRTHDATE` | Birthdate in ISO format (`YYYY-MM-DD`); required unless `--reference` is used |
| `-m` | `--mode {weeks,months,years}` | Grid granularity, one cell per unit (default: weeks) |
| `-l` | `--lifespan YEARS` | Nominal lifespan spanned by the grid, 1–150 (default: 90) |
| | `--as-of DATE` | ISO date to measure elapsed time to (default: today) |
| `-g` | `--group CELLS` | Cells per visual group, `0` to disable (default: 13 weeks, 3 months, 5 years) |
| `-r` | `--reference` | Render the typical-American reference grid instead of a personal one |
| | `--no-color` | Disable ANSI color; glyphs alone distinguish the cells |
| `-f` | `--format {table,json}` | Output format (default: table) |

</details>

---

<details>
<summary><strong><code>breakeven</code></strong> — Break-Even Analysis</summary>

Answers the foundational viability question: at what unit volume does revenue cover total cost? Reports break-even units and revenue, contribution margin and margin ratio, the volume needed to reach a target profit, and the margin of safety at an actual or projected volume. `--sweep` adds a profit/loss table across a unit range, and `--chart` renders that table as a text bar chart with losses left of the break-even axis and profits right of it — no plotting library required.

```bash
# Basic break-even: fixed costs 50,000, price 25, variable cost 10 per unit
breakeven --fixed 50000 --price 25 --variable 10

# How many units to hit a 20,000 profit?
breakeven --fixed 50000 --price 25 --variable 10 --target-profit 20000

# Margin of safety at a projected 5,000 units
breakeven --fixed 50000 --price 25 --variable 10 --actual-units 5000

# Profit/loss table from 0 to 8,000 units in steps of 500, with a chart
breakeven --fixed 50000 --price 25 --variable 10 --sweep 0 8000 500 --chart
```

**Options:**

| Flag | Long form | Description |
|------|-----------|-------------|
| | `--fixed F` | Total fixed costs (required) |
| | `--price F` | Selling price per unit (required) |
| | `--variable F` | Variable cost per unit (required) |
| | `--target-profit F` | Also report the unit volume needed for this profit |
| | `--actual-units F` | Actual or projected volume, for the margin of safety |
| | `--sweep MIN MAX STEP` | Append a profit/loss table across a unit range |
| | `--chart` | Render the sweep profit column as a text bar chart (needs `--sweep`) |
| | `--format {table,json,csv}` | Output format (default: table) |
| `-P` | `--precision` | Decimal places for output (default: 2) |

</details>

---

<details>
<summary><strong><code>entropy</code></strong> — Information Entropy</summary>

Computes the core information-theoretic measures: Shannon entropy, KL divergence, cross-entropy, mutual information, and conditional entropy. Pure Python via `math.log`, reported in bits (base 2), nats (base e), or hartleys (base 10). Input vectors are normalized to sum to 1, so raw counts work as well as probabilities. The entropy report also names the maximum entropy for that number of outcomes and the resulting efficiency; the divergence report includes the reverse KL, since D(P‖Q) is not symmetric.

```bash
# Shannon entropy of a fair six-sided die (2.585 bits)
entropy --probs 0.167,0.167,0.167,0.167,0.167,0.167

# Raw counts work too — normalized internally
entropy --probs "42 17 8 3"

# KL divergence between a biased coin (0.7/0.3) and a fair coin
entropy --measure kl --probs-p 0.7,0.3 --probs-q 0.5,0.5

# Cross-entropy loss of a prediction, in nats
entropy --measure cross --probs-p 1,0 --probs-q 0.9,0.1 --base e

# Mutual information from a 2x2 joint distribution
entropy --measure mi --joint 0.25,0.25 --joint 0.25,0.25
```

**Options:**

| Flag | Long form | Description |
|------|-----------|-------------|
| | `--probs VALUES` | Distribution for the `entropy` measure (comma or space separated) |
| | `--measure {entropy,kl,cross,mi,conditional}` | Quantity to compute (default: entropy) |
| | `--base {2,e,10}` | Logarithm base: 2=bits, e=nats, 10=hartleys (default: 2) |
| | `--probs-p VALUES` | Distribution P for the `kl` and `cross` measures |
| | `--probs-q VALUES` | Distribution Q for the `kl` and `cross` measures |
| | `--joint ROW` | One row of the joint table for `mi` and `conditional`; repeat per row |
| | `--format {table,json}` | Output format (default: table) |
| `-P` | `--precision` | Decimal places for output (default: 4) |

</details>

---

<details>
<summary><strong><code>weibull</code></strong> — Weibull Distribution</summary>

The standard model for component lifetimes and failure analysis. The shape parameter `k` sets the failure mode, which the report names outright: `k < 1` is a decreasing hazard (infant mortality), `k = 1` is the constant-hazard exponential case, and `k > 1` is an increasing hazard (wear-out). The hazard rate is computed in closed form rather than as `f(x)/S(x)`, so it stays finite far out in the tail where that ratio would divide zero by zero. Moments use `math.lgamma`, so large parameters do not overflow.

```bash
# Survival probability at t=500 hours, shape=2, scale=1000
weibull -x 500 -k 2 --lambda 1000 --survival

# 5th percentile of failure time — when have 5% of units failed?
weibull --quantile 0.05 -k 1.5 --lambda 800

# Range table from 0 to 2000 in steps of 200
weibull -k 2.5 --lambda 1200 --table 0 2000 200
```

**Options:**

| Flag | Long form | Description |
|------|-----------|-------------|
| `-x` | | Evaluation point (required unless `--quantile` or `--table` is used) |
| `-k` | `--shape F` | Shape parameter k; <1 infant mortality, 1 constant, >1 wear-out (required) |
| | `--lambda F` / `--scale F` | Scale parameter λ, the characteristic life (required) |
| | `--quantile F` | Report the x at this cumulative probability |
| | `--survival` | Lead the point report with S(x) = 1 − F(x) |
| | `--table MIN MAX STEP` | Print a table over x = MIN..MAX |
| | `--format {table,json}` | Output format (default: table) |
| `-P` | `--precision` | Decimal places for output (default: 4) |

</details>

---

<details>
<summary><strong><code>discount</code></strong> — Discount Rate & NPV</summary>

Relates nominal rates, real rates, and inflation through the Fisher equation — `real = (1 + nominal) / (1 + inflation) − 1` — and applies the result to lump sums and cash-flow series. Give either `--nominal` or `--real` (the latter with `--inflation`) and the other is derived. Cash flows start at period 0, so an up-front investment is the negative first element. The real NPV assumes the cash flows are stated in today's purchasing power and discounts them at the real rate; the discounted payback period interpolates within the period where cumulative cash flow crosses zero.

```bash
# Real rate and discount factor from a nominal rate and inflation
discount --nominal 0.08 --inflation 0.03

# Nominal rate implied by a real rate and inflation
discount --real 0.05 --inflation 0.03

# Present value of a 10,000 lump sum received in 5 periods
discount --nominal 0.08 --fv 10000 --periods 5

# Nominal and inflation-adjusted NPV of a project
discount --nominal 0.08 --inflation 0.03 --cashflows -1000 300 400 500
```

**Options:**

| Flag | Long form | Description |
|------|-----------|-------------|
| | `--nominal F` | Nominal discount rate per period, e.g. `0.08` for 8% |
| | `--real F` | Real discount rate per period (requires `--inflation`) |
| | `--inflation F` | Inflation rate per period, e.g. `0.03` for 3% |
| | `--fv F` | Future value of a lump sum to discount |
| | `--periods F` | Periods for the lump sum and discount factor (default: 1) |
| | `--cashflows F [F ...]` | Cash flow series starting at period 0, for NPV |
| | `--format {table,json}` | Output format (default: table) |
| `-P` | `--precision` | Decimal places for output (default: 4) |

</details>

---

<details>
<summary><strong><code>simulate</code></strong> — Monte Carlo Probability Simulator</summary>

Runs repeated random experiments to estimate probabilities empirically, with optional confidence intervals and analytical comparison against `binomial`, `birthday`, `streak`, `poisson`, `power`, `permutation`, `bayes`, `season`, and `linboot`.

**Implementation**: Uses numpy for efficient random number generation and scipy for statistical functions (Wilson confidence intervals).

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

# Statistical power: empirical power for detecting a mean shift of 5 (σ=10, n=30)
simulate --experiment power --params type=mean n=30 sigma=10 delta=5

# Permutation test: one-sample sign-flip against mu0=3.0
simulate --experiment permutation --params type=one values=2.1,3.4,2.9,3.1,2.8 mu0=3.0

# Bayes: empirical P(A|B) via rejection sampling
simulate --experiment bayes --params prior=0.01 likelihood=0.99 fp=0.05

# Season simulation: P(team wins ≥ 90 games) with 58% win probability
simulate --experiment season --params win_pct=0.58 games=162 wins_ge=90

# Linear regression bootstrap: slope/intercept CIs via residual resampling
simulate --experiment linboot --params x=1,2,3,4,5 y=2.1,3.9,6.2,7.8,10.1 predict=6
```

**Options:**

| Flag | Long form | Description |
|------|-----------|-------------|
| `-e` | `--experiment` | Experiment type: `binomial`, `birthday`, `streak`, `poisson`, `power`, `permutation`, `bayes`, `season`, or `linboot` (required) |
| `-p` | `--params` | Space-separated `KEY=VALUE` experiment parameters (see below) |
| `-t` | `--trials` | Number of simulation trials (default: 10,000) |
| | `--scale` | Target standard error; auto-computes `--trials` (overrides `-t`) |
| `-s` | `--seed` | Random seed for reproducibility |
| `-c` | `--confidence` | Print 95% Wilson confidence interval |
| | `--dump` | Output per-trial results as CSV instead of summary |
| `-f` | `--format` | Summary output format: `table` (default) or `json` |
| `-P` | `--precision` | Decimal places for printed probabilities (default: `6`) |

**Required params by experiment:**

| Experiment | Required params | Description |
|------------|-----------------|-------------|
| `binomial` | `n=INT k=INT p=FLOAT` | P(X ≥ k) for Binomial(n, p) |
| `birthday` | `pool=INT group=INT` | P(at least one collision) |
| `streak` | `n=INT k=INT p=FLOAT` | P(run of ≥ k successes in n trials) |
| `poisson` | `lam=FLOAT k=INT` | P(X ≥ k) for Poisson(λ) |
| `power` | `type=mean\|comparison` | Empirical statistical power |
| `permutation` | `type=one\|two\|paired` | Distribution-free p-value via permutation test |
| `bayes` | `prior=FLOAT likelihood=FLOAT fp=FLOAT` | Empirical P(A\|B) via rejection sampling |
| `season` | `win_pct=FLOAT [games wins_ge]` | Season win distribution or P(wins ≥ k) |
| `linboot` | `x=CSV y=CSV [predict alpha]` | Residual bootstrap for linear regression coefficients |

</details>

---

### 🐍 Python Library

| Module | Import path | Key functions |
|--------|-------------|---------------|
| Binomial | `src.utils.binomial_distribution` | `binomial_pmf`, `binomial_cdf_le`, `binomial_cdf_ge` |
| Bayes | `src.utils.bayes_theorem` | `bayes_posterior`, `evidence_from_false_positive` |
| Birthday | `src.utils.birthday_problem` | `collision_prob_uniform`, `collision_prob_nonuniform`, `min_group_for_prob` |
| Normal | `src.utils.normal_gaussian` | `normal_pdf`, `normal_cdf`, `normal_ppf`, `normal_prob_between` |
| Z-Score | `src.utils.z_score` | `z_score`, `z_scores`, `mean`, `std_dev` |
| Expected Value | `src.utils.expected_value` | `expected_value`, `variance`, `std_dev`, `entropy`, `mgf`, `load_file` |
| Poisson | `src.utils.poisson_distribution` | `poisson_pmf`, `poisson_cdf_le`, `poisson_cdf_ge`, `min_k_for_prob` |
| Primes | `src.utils.prime_numbers` | `is_prime`, `nth_prime`, `count_primes`, `primes_in_range`, `prime_factorization` |
| Streak | `src.utils.streak_probability` | `prob_at_least_one_streak`, `expected_longest_streak` |
| Pythagorean | `src.utils.pythagorean_record` | `pythagorean_expectation`, `linear_expectation`, `expected_wins` |
| Pearson | `src.utils.pearson_correlation` | `pearson_r`, `correlation_t_statistic`, `correlation_p_value`, `correlation_confidence_interval` |
| Spearman | `src.utils.spearman_correlation` | `spearman_rho`, `rank_data`, `correlation_t_statistic`, `correlation_p_value` |
| Linear Reg. | `src.utils.linear_regression` | `linear_regression`, `predict`, `mean` |
| Sample Size | `src.utils.sample_size` | `sample_size_proportion`, `sample_size_mean`, `sample_size_comparison` |
| Bootstrap | `src.utils.bootstrap_confidence_intervals` | `bootstrap_resample`, `compute_confidence_interval`, `get_stat_function` |
| Conf. Intervals | `src.utils.confidence_intervals` | `wilson_interval`, `clopper_pearson_interval`, `t_interval`, `poisson_interval` |
| Hypothesis Test | `src.utils.probability_values` | `z_test_proportion`, `binomial_exact_test`, `t_test_mean`, `chi2_goodness_of_fit` |
| T-Test | `src.utils.t_test` | `one_sample_t_test`, `two_sample_t_test`, `paired_t_test` |
| Forecast | `src.utils.forecast_time_series` | `simple_exponential_smoothing`, `double_exponential_smoothing`, `holt_winters` |
| Collatz | `src.utils.collatz_conjecture` | `CollatzChecker`, `collatz_next` |
| Jevons | `src.utils.jevons_paradox` | `analyze`, `expected_savings`, `rebound_consumption`, `net_consumption`, `is_backfire` |
| Gini | `src.utils.gini` | `gini_coefficient`, `lorenz_curve`, `relative_mad`, `gini_grouped` |
| CRT | `src.utils.crt` | `extended_gcd`, `mod_inverse`, `crt` |
| Subnet | `src.utils.subnet` | `compute_subnet`, `_classful_prefix` |
| Geometric | `src.utils.geometric_distribution` | `geo_pmf`, `geo_cdf`, `geo_survival`, `geo_mean`, `geo_variance` |
| Chi-Square | `src.utils.chi_squared` | `chisq_gof`, `chisq_independence`, `chi2_cdf`, `regularized_gamma_p` |
| ANOVA | `src.utils.anova` | `anova_one_way`, `tukey_hsd`, `bonferroni`, `f_cdf`, `studentized_range_cdf` |
| Life in Weeks | `src.utils.life_in_weeks` | `compute_grid`, `units_elapsed`, `grid_rows`, `reference_rows`, `milestone_units` |
| Break-Even | `src.utils.break_even` | `breakeven_units`, `breakeven_revenue`, `contribution_margin`, `margin_of_safety`, `target_profit_units`, `sweep_rows` |
| Entropy | `src.utils.information_entropy` | `shannon_entropy`, `kl_divergence`, `cross_entropy`, `mutual_information`, `conditional_entropy`, `joint_entropy` |
| Weibull | `src.utils.weibull_distribution` | `weibull_pdf`, `weibull_cdf`, `weibull_survival`, `weibull_hazard`, `weibull_quantile`, `weibull_mean` |
| Discount Rate | `src.utils.discount_rate` | `real_rate`, `nominal_rate`, `discount_factor`, `present_value`, `npv`, `real_npv`, `payback_period` |
| Monte Carlo | `src.utils.monte_carlo` | `simulate_binomial`, `simulate_birthday`, `simulate_streak`, `simulate_poisson`, `simulate_power`, `simulate_permutation`, `simulate_bayes`, `simulate_season`, `simulate_linboot` |

---

<details>
<summary><strong>Binomial Distribution</strong></summary>

```python
from src.utils.binomial_distribution import binomial_pmf, binomial_cdf_le, binomial_cdf_ge

# P(X = 3) for Binomial(n=10, p=0.4)
pmf = binomial_pmf(10, 3, 0.4)

# P(X <= 3) for Binomial(n=10, p=0.4)
cdf = binomial_cdf_le(10, 3, 0.4)

# P(X >= 3) for Binomial(n=10, p=0.4)
survival = binomial_cdf_ge(10, 3, 0.4)
```

</details>

<details>
<summary><strong>Bayes' Theorem</strong></summary>

```python
from src.utils.bayes_theorem import bayes_posterior, evidence_from_false_positive

# Derive P(B) from a prior, likelihood, and false-positive rate
evidence = evidence_from_false_positive(0.01, 0.99, 0.05)

# Posterior P(A|B)
posterior = bayes_posterior(0.01, 0.99, evidence)
```

</details>

<details>
<summary><strong>Birthday Problem</strong></summary>

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

</details>

<details>
<summary><strong>Normal Distribution</strong></summary>

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

</details>

<details>
<summary><strong>Z-Scores</strong></summary>

```python
from src.utils.z_score import mean, std_dev, z_score, z_scores

# Standardize one value with known distribution parameters
z = z_score(85.0, mu=70.0, sigma=10.0)

# Compute population z-scores for an inline dataset
values = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
scores = z_scores(values)

# Use sample standard deviation
sample_scores = z_scores(values, sample=True)

# Summary statistics are available directly
mu = mean(values)
sigma = std_dev(values)
sample_sigma = std_dev(values, sample=True)
```

</details>

<details>
<summary><strong>Expected Value</strong></summary>

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

</details>

<details>
<summary><strong>Poisson Distribution</strong></summary>

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

</details>

<details>
<summary><strong>Prime Numbers</strong></summary>

```python
from src.utils.prime_numbers import (
    is_prime,
    nth_prime,
    count_primes,
    primes_in_range,
    prime_factorization,
    sieve_of_eratosthenes,
    format_factorization,
)

# Check if a number is prime
if is_prime(97):
    print("97 is prime")

# Find the 100th prime number
p = nth_prime(100)  # Returns 541

# Count primes up to 1000 (π function)
count = count_primes(1000)  # Returns 168

# Get all primes in a range
primes = primes_in_range(50, 100)  # [53, 59, 61, 67, 71, 73, 79, 83, 89, 97]

# Generate all primes up to a limit using Sieve of Eratosthenes
all_primes = sieve_of_eratosthenes(100)  # [2, 3, 5, 7, 11, ..., 97]

# Prime factorization
factors = prime_factorization(360)  # {2: 3, 3: 2, 5: 1} → 2³ × 3² × 5
formatted = format_factorization(factors)  # "2³ × 3² × 5"
```

</details>

<details>
<summary><strong>Streak Probability</strong></summary>

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

</details>

<details>
<summary><strong>Pythagorean Record</strong></summary>

```python
from src.utils.pythagorean_record import (
    pythagorean_expectation,
    linear_expectation,
    expected_wins,
)

# Traditional Pythagorean formula: expected win % for 800 RS, 650 RA
win_pct = pythagorean_expectation(800, 650, exponent=2.0)

# Linear formula (SABR 2014): expected win % for MLB
win_pct = linear_expectation(800, 650, sport="mlb")

# Linear formula for NFL
win_pct = linear_expectation(420, 300, sport="nfl")

# Linear formula for NBA
win_pct = linear_expectation(8500, 8200, sport="nba")

# Convert win percentage to expected wins
wins = expected_wins(win_pct, games=162)
```

</details>

<details>
<summary><strong>Pearson Correlation</strong></summary>

```python
from src.utils.pearson_correlation import (
    pearson_r,
    correlation_t_statistic,
    correlation_p_value,
    correlation_confidence_interval,
)

x = [1, 2, 3, 4, 5]
y = [2.1, 3.8, 6.2, 7.9, 10.1]

# Compute Pearson's r
r = pearson_r(x, y)

# r² (coefficient of determination)
r_squared = r ** 2

# t-statistic for testing H₀: ρ = 0
t_stat = correlation_t_statistic(r, n=len(x))

# p-value for two-tailed test
p_value = correlation_p_value(r, n=len(x), sided="two")

# 95% confidence interval for population correlation ρ
ci_lower, ci_upper = correlation_confidence_interval(r, n=len(x), alpha=0.05)
```

</details>

<details>
<summary><strong>Spearman Correlation</strong></summary>

```python
from src.utils.spearman_correlation import (
    spearman_rho,
    rank_data,
    correlation_t_statistic,
    correlation_p_value,
    correlation_confidence_interval,
)

x = [1, 2, 3, 4, 10]
y = [2, 3, 5, 6, 20]

# Compute Spearman's ρ (rank correlation)
rho = spearman_rho(x, y)

# ρ² (coefficient of determination for ranks)
rho_squared = rho ** 2

# Get ranks for inspection (handles ties by averaging)
rank_x = rank_data(x)
rank_y = rank_data(y)

# t-statistic for testing H₀: ρ = 0
t_stat = correlation_t_statistic(rho, n=len(x))

# p-value for two-tailed test
p_value = correlation_p_value(rho, n=len(x), sided="two")

# 99% confidence interval for population correlation ρ
ci_lower, ci_upper = correlation_confidence_interval(rho, n=len(x), alpha=0.01)
```

</details>

<details>
<summary><strong>Linear Regression</strong></summary>

```python
from src.utils.linear_regression import (
    linear_regression,
    predict,
    mean,
)

x = [1, 2, 3, 4, 5]
y = [2.1, 3.9, 6.2, 7.8, 10.1]

# Perform linear regression
model = linear_regression(x, y)

# Access model attributes
print(f"Slope: {model.slope}")
print(f"Intercept: {model.intercept}")
print(f"R²: {model.r_squared}")
print(f"Residual SE: {model.residual_std_error}")
print(f"t-statistic (slope): {model.t_slope}")

# Make a prediction with confidence and prediction intervals
x_new = 6.0
prediction, conf_lower, conf_upper, pred_lower, pred_upper = predict(
    model, x_new, alpha=0.05
)

print(f"Prediction at x={x_new}: {prediction}")
print(f"95% Confidence interval: [{conf_lower}, {conf_upper}]")
print(f"95% Prediction interval: [{pred_lower}, {pred_upper}]")
```

</details>

<details>
<summary><strong>Sample Size Calculator</strong></summary>

```python
from src.utils.sample_size import (
    sample_size_proportion,
    sample_size_mean,
    sample_size_comparison,
    achieved_power_mean,
    achieved_power_comparison,
)

# Minimum sample size to estimate a proportion within ±3% at 95% confidence
n = sample_size_proportion(p=0.5, margin=0.03, alpha=0.05)

# Minimum sample size to detect a mean difference of 5 with σ=12 at 80% power
n = sample_size_mean(sigma=12, delta=5, alpha=0.05, power=0.80, sided="two")

# Sample sizes for two-proportion comparison (detect 0.40 vs 0.50 with 80% power)
n1, n2 = sample_size_comparison(p1=0.40, p2=0.50, alpha=0.05, power=0.80, sided="two")

# Achieved power for a given sample size
power = achieved_power_mean(n=100, sigma=12, delta=5, alpha=0.05, sided="two")

# Achieved power for two-proportion comparison
power = achieved_power_comparison(n_per_group=150, p1=0.40, p2=0.50, alpha=0.05, sided="two")
```

</details>

<details>
<summary><strong>Bootstrap Confidence Intervals</strong></summary>

```python
from src.utils.bootstrap_confidence_intervals import (
    bootstrap_resample,
    compute_confidence_interval,
    get_stat_function,
)
import statistics
import random

# Sample data
data = [14.2, 13.8, 15.1, 14.5, 13.9, 15.3, 14.7, 14.1]

# Set random seed for reproducibility
random.seed(42)

# Get the statistic function (mean, median, or stdev)
stat_func = get_stat_function("mean")

# Compute original statistic
original_stat = stat_func(data)

# Generate bootstrap resamples
n_bootstrap = 10_000
bootstrap_stats = bootstrap_resample(data, n_bootstrap, stat_func)

# Compute 95% confidence interval using percentile method
ci_lower, ci_upper = compute_confidence_interval(bootstrap_stats, confidence=0.95)

print(f"Original mean: {original_stat:.4f}")
print(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")

# For median
median_func = get_stat_function("median")
bootstrap_medians = bootstrap_resample(data, n_bootstrap, median_func)
median_ci = compute_confidence_interval(bootstrap_medians, 0.90)
print(f"90% CI for median: [{median_ci[0]:.4f}, {median_ci[1]:.4f}]")
```

</details>

<details>
<summary><strong>Confidence Intervals (Parametric Methods)</strong></summary>

```python
from src.utils.confidence_intervals import (
    wilson_interval,
    clopper_pearson_interval,
    normal_proportion_interval,
    t_interval,
    poisson_interval,
)

# Wilson score interval for a proportion (recommended for general use)
# 47 successes out of 120 trials at 95% confidence
lower, upper = wilson_interval(n=120, k=47, alpha=0.05)
print(f"Wilson 95% CI: [{lower:.4f}, {upper:.4f}]")

# Clopper-Pearson exact interval (conservative)
lower, upper = clopper_pearson_interval(n=30, k=22, alpha=0.05)
print(f"Clopper-Pearson 95% CI: [{lower:.4f}, {upper:.4f}]")

# Normal approximation for proportion (requires large n)
lower, upper = normal_proportion_interval(n=1000, k=480, alpha=0.05)
print(f"Normal approximation 95% CI: [{lower:.4f}, {upper:.4f}]")

# t-interval for population mean (unknown variance)
# Sample: n=15, mean=23.4, std=4.1
lower, upper = t_interval(n=15, mean=23.4, std=4.1, alpha=0.05)
print(f"t-interval 95% CI: [{lower:.4f}, {upper:.4f}]")

# 90% confidence interval (α=0.10)
lower, upper = t_interval(n=25, mean=100.5, std=12.3, alpha=0.10)
print(f"t-interval 90% CI: [{lower:.4f}, {upper:.4f}]")

# Poisson interval for count/rate data
# Observed 23 events
lower, upper = poisson_interval(k=23, alpha=0.05)
print(f"Poisson 95% CI for λ: [{lower:.4f}, {upper:.4f}]")

# Different confidence level
lower, upper = poisson_interval(k=10, alpha=0.01)
print(f"Poisson 99% CI for λ: [{lower:.4f}, {upper:.4f}]")
```

</details>

<details>
<summary><strong>Hypothesis Testing & p-values</strong></summary>

```python
from src.utils.probability_values import (
    z_test_proportion,
    binomial_exact_test,
    t_test_mean,
    chi2_goodness_of_fit,
    cohen_d,
)

# Z-test for a single proportion
# H₀: p = 0.5 vs H₁: p ≠ 0.5
# Sample: 480 successes out of 1000 trials
z_stat, p_value = z_test_proportion(n=1000, k=480, p0=0.5, sided="two")
print(f"Z-test: z = {z_stat:.4f}, p-value = {p_value:.4f}")

# One-sided z-test (testing if proportion is less than 0.5)
z_stat, p_value = z_test_proportion(n=1000, k=450, p0=0.5, sided="less")
print(f"One-sided Z-test: z = {z_stat:.4f}, p-value = {p_value:.4f}")

# One-sided z-test (testing if proportion is greater than 0.5)
z_stat, p_value = z_test_proportion(n=1000, k=550, p0=0.5, sided="greater")
print(f"One-sided Z-test: z = {z_stat:.4f}, p-value = {p_value:.4f}")

# Binomial exact test (better for small samples)
# H₀: p = 0.60
# Sample: 22 successes out of 30 trials
test_stat, p_value = binomial_exact_test(n=30, k=22, p0=0.60, sided="two")
print(f"Binomial exact test: k = {test_stat}, p-value = {p_value:.4f}")

# One-sample t-test for mean
# H₀: μ = 100 vs H₁: μ ≠ 100
# Sample: n=30, mean=105, std=15
t_stat, p_value = t_test_mean(mean=105, std=15, n=30, mu0=100, sided="two")
print(f"t-test: t = {t_stat:.4f}, p-value = {p_value:.4f}")

# One-sided t-test
t_stat, p_value = t_test_mean(mean=105, std=15, n=30, mu0=100, sided="greater")
print(f"One-sided t-test: t = {t_stat:.4f}, p-value = {p_value:.4f}")

# Chi-squared goodness-of-fit test
# H₀: observed frequencies match expected distribution
observed = [30, 20, 50]
expected = [33.33, 33.33, 33.33]
chi2_stat, p_value = chi2_goodness_of_fit(observed, expected)
print(f"Chi-squared test: χ² = {chi2_stat:.4f}, p-value = {p_value:.4f}")

# Effect size (Cohen's d) for proportions
p_hat = 0.48  # Sample proportion
p0 = 0.50     # Null hypothesis proportion
effect = cohen_d(p_hat, p0)
print(f"Effect size (Cohen's d): {effect:.4f}")

# Make a decision at α=0.05
alpha = 0.05
if p_value < alpha:
    print(f"Reject H₀ (p = {p_value:.4f} < α = {alpha})")
else:
    print(f"Fail to reject H₀ (p = {p_value:.4f} ≥ α = {alpha})")
```

</details>

<details>
<summary><strong>T-Test</strong></summary>

```python
from src.utils.t_test import (
    one_sample_t_test,
    two_sample_t_test,
    paired_t_test,
)

# One-sample t-test from summary statistics
# H₀: μ = 100 vs H₁: μ ≠ 100
result = one_sample_t_test(mean=105, std=12, n=25, mu0=100, alpha=0.05, sided="two")
print(f"t = {result.t_stat:.4f}, df = {result.df:.4f}, p = {result.p_value:.4f}")
print(f"95% CI: [{result.ci_lower:.4f}, {result.ci_upper:.4f}]")
print(f"Cohen's d: {result.cohens_d:.4f}")

# One-sided test (greater)
result_gt = one_sample_t_test(mean=105, std=12, n=25, mu0=100, alpha=0.05, sided="greater")

# Two-sample Welch's t-test from summary statistics
# H₀: μ₁ = μ₂ (unequal variances assumed)
result2 = two_sample_t_test(
    mean1=12.4, std1=2.1, n1=30,
    mean2=10.8, std2=1.9, n2=28,
    alpha=0.05, sided="two",
)
print(f"Welch df = {result2.df:.4f}, p = {result2.p_value:.4f}")
print(f"95% CI (μ₁−μ₂): [{result2.ci_lower:.4f}, {result2.ci_upper:.4f}]")

# Two-sample from raw values (compute summary stats first or pass directly)
from src.utils.t_test import _mean, _std
values1 = [1.2, 2.3, 3.1, 2.8]
values2 = [2.1, 3.2, 4.1, 3.9]
result2_raw = two_sample_t_test(
    mean1=_mean(values1), std1=_std(values1), n1=len(values1),
    mean2=_mean(values2), std2=_std(values2), n2=len(values2),
)

# Paired t-test on matched observations
# H₀: μ_d = 0  (d = x₁ − x₂)
pre  = [85, 90, 78, 92, 88]
post = [90, 95, 82, 95, 91]
result_p = paired_t_test(pre, post, alpha=0.05, sided="two")
print(f"Mean difference: {result_p.mean_diff:.4f}")
print(f"t = {result_p.t_stat:.4f}, p = {result_p.p_value:.4f}")
print(f"Cohen's d: {result_p.cohens_d:.4f}")
```

</details>

<details>
<summary><strong>Time Series Forecasting</strong></summary>

```python
from src.utils.forecast_time_series import (
    simple_exponential_smoothing,
    double_exponential_smoothing,
    holt_winters,
    forecast_simple,
    forecast_double,
    forecast_holt_winters,
)

# Time series data
sales_data = [120, 135, 148, 130, 142, 155, 160, 165, 172, 180]

# Simple exponential smoothing (level only)
fitted_simple, level = simple_exponential_smoothing(sales_data, alpha=0.3)
forecasts_simple = forecast_simple(level, periods=3)
print(f"Simple ES forecasts: {forecasts_simple}")

# Double exponential smoothing (Holt's method: level + trend)
fitted_double, level, trend = double_exponential_smoothing(sales_data, alpha=0.3, beta=0.1)
forecasts_double = forecast_double(level, trend, periods=6)
print(f"Double ES forecasts: {forecasts_double}")

# Holt-Winters (level + trend + seasonal)
# Requires at least 2 * seasonal_period data points
monthly_data = [100, 110, 95, 105, 120, 115, 130, 140, 125, 135, 150,
                145, 105, 115, 100, 110, 125, 120, 135, 145, 130, 140, 155, 150]
fitted_hw, level, trend, seasonal = holt_winters(
    monthly_data,
    alpha=0.3,
    beta=0.1,
    gamma=0.1,
    seasonal_period=12,
    seasonal_type='additive'
)
forecasts_hw = forecast_holt_winters(
    level, trend, seasonal, periods=6,
    seasonal_period=12, seasonal_type='additive'
)
print(f"Holt-Winters forecasts: {[f'{x:.2f}' for x in forecasts_hw]}")
```

</details>

<details>
<summary><strong>Collatz Conjecture</strong></summary>

```python
from src.utils.collatz_conjecture import CollatzChecker, collatz_next

# Create a Collatz checker instance
checker = CollatzChecker()

# Test the Collatz sequence for a single number
n = 27
sequence = [n]
while n != 1:
    n = collatz_next(n)
    sequence.append(n)
print(f"Collatz sequence for 27: {sequence[:10]}...")  # First 10 steps

# Check all integers from 1 to 10000
checker.ensure_up_to(10_000, check_interval=1000, verbose=False)
print(f"Largest consecutive verified integer: {checker.max_valid}")

# Check if specific numbers are proven to reach 1
print(f"Is 9999 proven? {checker.proven(9999)}")
print(f"Is 10000 proven? {checker.proven(10000)}")

# Access the steps histogram (steps required to reach 1)
print(f"Number of integers requiring exactly 10 steps: {checker.steps_histogram.get(10, 0)}")

# Verify a larger range
large_checker = CollatzChecker()
large_checker.ensure_up_to(100_000, check_interval=10_000, verbose=False)
print(f"Largest consecutive verified (up to 100k): {large_checker.max_valid}")
print(f"Total numbers resolved: {len(large_checker.resolved)}")
```

</details>

<details>
<summary><strong>Jevons' Paradox</strong></summary>

```python
from src.utils.jevons_paradox import (
    analyze,
    expected_savings,
    rebound_consumption,
    rebound_rate,
    net_consumption,
    actual_savings,
    is_backfire,
    outcome_label,
)

# 30% efficiency improvement, price elasticity 0.5, baseline 1000 units
eta = 0.30
epsilon = 0.5
baseline = 1000.0

# Expected resource savings without any demand response
exp_sav = expected_savings(baseline, eta)          # 300.0 units

# Additional consumption caused by demand rebound
reb_con = rebound_consumption(baseline, eta, epsilon)  # 105.0 units

# Net consumption after efficiency gain and rebound
net_con = net_consumption(baseline, eta, epsilon)  # 805.0 units

# Actual savings after rebound
act_sav = actual_savings(baseline, eta, epsilon)   # 195.0 units

# Rebound rate as a fraction of expected savings (0.35 = 35%)
rate = rebound_rate(eta, epsilon)

# Check if backfire occurs (net consumption > baseline)
if is_backfire(eta, epsilon):
    print("Backfire: consumption increased despite efficiency gains")

# Qualitative description of the outcome
print(outcome_label(rate))  # "Strong rebound — efficiency gains are significantly offset"

# Full analysis dict
result = analyze(baseline, eta, epsilon)
print(f"Rebound rate: {result['rebound_pct']:.1f}%")
print(f"Net consumption: {result['net_consumption']:.1f} units")
```
</details>


<details>
<summary><strong>Sigmoid Function and Logistic Curve</strong></summary>

```python
from src.utils.sigmoid import (
    sigmoid,
    sigmoid_derivative,
    inverse_logit,
)

sigma = sigmoid(2.0)          # 0.8807970779778823
dsigma = sigmoid_derivative(0.0)  # 0.25
x = inverse_logit(0.75)       # 1.0986122886681098
```

</details>

<details>
<summary><strong>Euler's Number</strong></summary>

```python
from src.utils.euler import (
    e_approx,
    exp_series,
    natural_log,
    euler_identity,
)

approx = e_approx(1_000_000)        # 2.718280469...
ex = exp_series(1.0, 50)            # ≈ math.e to 15+ decimal places
ln2 = natural_log(2.0, 100)         # ≈ 0.693147...
real, imag, total = euler_identity() # (-1.0, ~0.0, ~0.0)
```

</details>


<details>
<summary><strong>Gini Coefficient and Lorenz Curve</strong></summary>

```python
from src.utils.gini import (
    gini_coefficient,
    lorenz_curve,
    relative_mad,
    gini_grouped,
)

data = [1.0, 2.0, 3.0, 4.0, 5.0]

# Gini coefficient (0 = perfect equality, approaching 1 = maximum inequality)
g = gini_coefficient(data)

# With positive weights
g_w = gini_coefficient([10.0, 30.0, 50.0, 80.0], weights=[4.0, 3.0, 2.0, 1.0])

# With small-sample bias correction
g_corr = gini_coefficient(data, corrected=True)

# Relative mean absolute difference (= 2 × Gini)
rmad = relative_mad(data)

# Lorenz curve coordinates: list of (population_share, income_share) tuples
points = lorenz_curve(data)

# Gini from pre-aggregated (population_share, income_share) pairs
groups = [(0.2, 0.05), (0.3, 0.15), (0.5, 0.80)]
g_grouped = gini_grouped(groups)
```

</details>

<details>
<summary><strong>Sunzi's Theorem Solver (CRT)</strong></summary>

```python
from src.utils.crt import (
    extended_gcd,
    mod_inverse,
    crt,
)

# Extended GCD: returns (g, s, t) such that a·s + b·t = g = gcd(a, b)
g, s, t = extended_gcd(35, 15)  # (5, 1, -2)

# Modular inverse: x such that a·x ≡ 1 (mod m)
x = mod_inverse(3, 7)           # 5 (since 3×5 = 15 ≡ 1 (mod 7))

# Solve x≡2(mod 3), x≡3(mod 5), x≡2(mod 7) → (23, 105)
solution, modulus = crt([2, 3, 2], [3, 5, 7])

# Non-coprime moduli (x≡0(mod 4), x≡2(mod 6))
solution, modulus = crt([0, 2], [4, 6])

# List all solutions up to a bound
x0, N = crt([2, 3, 2], [3, 5, 7])
all_solutions = list(range(x0, 300 + 1, N))
```

</details>

<details>
<summary><strong>Subnet Mask Calculator</strong></summary>

```python
from src.utils.subnet import compute_subnet, _classful_prefix

# Basic subnet calculation (CIDR embedded in address)
result = compute_subnet("192.168.1.50/24")
print(result.network_address)   # 192.168.1.0
print(result.broadcast_address) # 192.168.1.255
print(result.first_ip)          # 192.168.1.1
print(result.last_ip)           # 192.168.1.254
print(result.subnet_mask)       # 255.255.255.0
print(result.hosts)             # 254
print(result.networks)          # 1

# CIDR passed separately (overrides any embedded prefix)
result = compute_subnet("10.0.0.1", cidr=8)

# /31 point-to-point link (RFC 3021): both addresses usable
result = compute_subnet("192.168.1.0/31")
# result.hosts == 2; first_ip == "192.168.1.0"; last_ip == "192.168.1.1"

# /32 host route: single address
result = compute_subnet("10.0.0.5/32")
# result.hosts == 1; first_ip == last_ip == "10.0.0.5"

# Classful prefix boundary
prefix = _classful_prefix(result.network_address)  # 8 (Class A)
```

</details>

<details>
<summary><strong>Geometric Distribution</strong></summary>

```python
from src.utils.geometric_distribution import (
    geo_cdf,
    geo_mean,
    geo_pmf,
    geo_survival,
    geo_variance,
)

# P(first success on exactly the 5th trial, p=0.3)
geo_pmf(5, 0.3)       # 0.07203

# P(first success within the first 5 trials)
geo_cdf(5, 0.3)       # 0.83193

# P(needing more than 10 trials, p=0.2)
geo_survival(10, 0.2) # 0.10737

geo_mean(0.2)         # 5.0
geo_variance(0.2)     # 20.0
```

</details>

<details>
<summary><strong>Chi-Square Test Calculator</strong></summary>

```python
from src.utils.chi_squared import chisq_gof, chisq_independence

# Goodness-of-fit
result = chisq_gof(
    observed=[18, 22, 17, 25, 19, 19],
    expected=[20, 20, 20, 20, 20, 20],
)
print(result.statistic)  # 2.2
print(result.df)         # 5
print(result.p_value)    # 0.8208...

# Independence (contingency table)
result = chisq_independence([[40, 30, 20], [25, 45, 30]])
print(result.statistic)  # 7.9573...
print(result.p_value)    # 0.0187...
```

</details>

<details>
<summary><strong>One-Way ANOVA</strong></summary>

```python
from src.utils.anova import anova_one_way, bonferroni, tukey_hsd

groups = [
    [12.1, 11.8, 12.5, 11.9],
    [9.8, 10.3, 10.1, 9.7],
    [15.2, 14.9, 15.5, 16.0],
]

result = anova_one_way(groups)
print(result.f_stat)   # 229.2574...
print(result.p_value)  # 1.9055e-08...

# Tukey HSD pairwise comparisons
tukey_result = tukey_hsd(groups, result)
print(tukey_result.q_crit)
for c in tukey_result.comparisons:
    print(c.i, c.j, c.mean_diff, c.p_value, c.significant)

# Bonferroni-corrected pairwise comparisons
for c in bonferroni(groups, result):
    print(c.i, c.j, c.mean_diff, c.p_adj, c.significant)
```

</details>

<details>
<summary><strong>Life in Weeks</strong></summary>

```python
from datetime import date

from src.utils.life_in_weeks import (
    compute_grid,
    grid_rows,
    milestone_units,
    reference_rows,
    units_elapsed,
)

# Grid totals for a 90-year life measured to a fixed date
grid = compute_grid(date(1985, 3, 14), date(2026, 8, 7))
print(grid.total_units)      # 4680
print(grid.elapsed_units)    # 2160
print(grid.remaining_units)  # 2520

# Elapsed units in any mode
print(units_elapsed(date(1985, 3, 14), date(2026, 8, 7), "years"))  # 41

# Glyph rows, one string per row, no ANSI escapes
rows = grid_rows(grid.total_units, grid.elapsed_units, grid.cols)
print(len(rows))  # 90

# Typical-American reference grid: phase glyphs with milestone overlays
print(reference_rows("years", 90)[2])  # ▲▲■■■■■★■★

# Milestone ages converted to unit indices
print(milestone_units("weeks")[0])  # ('mean age at first birth (NCHS 2023)', 1430)
```

</details>

<details>
<summary><strong>Break-Even Analysis</strong></summary>

```python
from src.utils.break_even import (
    breakeven_units,
    contribution_margin,
    margin_of_safety,
    target_profit_units,
)

# Units needed to cover 50,000 of fixed cost at a 15 per-unit margin
units = breakeven_units(50_000, price=25, variable=10)

# Per-unit contribution toward fixed costs
margin = contribution_margin(25, 10)

# Volume required to earn a 20,000 profit
target = target_profit_units(50_000, 25, 10, profit=20_000)

# Fraction volume may fall before the business hits break-even
safety = margin_of_safety(actual_units=5_000, be_units=units)
```

</details>

<details>
<summary><strong>Information Entropy</strong></summary>

```python
from src.utils.information_entropy import (
    conditional_entropy,
    cross_entropy,
    kl_divergence,
    mutual_information,
    shannon_entropy,
)

# Entropy of a fair six-sided die, in bits (2.585)
h = shannon_entropy([1] * 6)

# Raw counts are normalized internally
h_counts = shannon_entropy([42, 17, 8, 3])

# Extra bits from coding a biased coin with a fair-coin code
kl = kl_divergence([0.7, 0.3], [0.5, 0.5])

# Cross-entropy loss, in nats
import math
loss = cross_entropy([1, 0], [0.9, 0.1], base=math.e)

# Shared information and residual uncertainty in a joint distribution
mi = mutual_information([[0.4, 0.2], [0.1, 0.3]])
h_y_given_x = conditional_entropy([[0.4, 0.2], [0.1, 0.3]])
```

</details>

<details>
<summary><strong>Weibull Distribution</strong></summary>

```python
from src.utils.weibull_distribution import (
    failure_mode,
    weibull_hazard,
    weibull_mean,
    weibull_quantile,
    weibull_survival,
)

# P(component survives past 500 hours) for Weibull(k=2, lambda=1000)
s = weibull_survival(500, 2, 1000)

# Instantaneous failure rate of a unit that reached 500 hours
h = weibull_hazard(500, 2, 1000)

# Time by which 5% of units have failed
t05 = weibull_quantile(0.05, 1.5, 800)

# Mean time to failure, and what the shape parameter implies
mttf = weibull_mean(2, 1000)
mode = failure_mode(2)  # 'wear-out (increasing hazard)'
```

</details>

<details>
<summary><strong>Discount Rate & NPV</strong></summary>

```python
from src.utils.discount_rate import (
    discount_factor,
    npv,
    payback_period,
    present_value,
    real_npv,
    real_rate,
)

# Fisher-equation real rate from an 8% nominal rate and 3% inflation
r = real_rate(0.08, 0.03)

# Present-value factor and the PV of a future lump sum
factor = discount_factor(0.08, periods=5)
pv = present_value(10_000, 0.08, periods=5)

# NPV of a project, nominal and in today's purchasing power
flows = [-1000, 300, 400, 500]
value = npv(0.08, flows)
value_real = real_npv(0.08, 0.03, flows)

# Period at which discounted cumulative cash flow turns positive
payback = payback_period(flows, 0.08)
```

</details>

<details>
<summary><strong>Monte Carlo Simulator</strong></summary>

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

</details>

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
