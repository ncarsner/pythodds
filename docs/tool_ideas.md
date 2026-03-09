# Tool Ideas for `pythodds`

This document outlines candidate tools for future addition to `pythodds`. Each entry describes the proposed command name, its statistical/mathematical architecture, practical application, and target user base — contextualised against the existing `binom` and `birthday` tools.

---

## Existing Tools (Reference Baseline)

| Command    | What it does | Core user |
|------------|--------------|-----------| 
| `binom`    | PMF, CDF, and survival function for a Binomial(n, p) distribution | Analysts testing pass/fail rates, A/B testers, QA engineers |
| `birthday` | Collision probability for uniform and non-uniform pools; range tables; JSON/CSV output | Security researchers, data engineers checking ID uniqueness, statisticians |

All existing tools are **pure-Python, zero-dependency, CLI-first**, with optional use as an importable library. Proposed tools follow the same constraints and conventions.

---

## 1. `poisson` — Distribution Calculator

### Architecture
- **Core functions:** `poisson_pmf(k, lam)`, `poisson_cdf_le(k, lam)`, `poisson_cdf_ge(k, lam)`
- Uses `math.lgamma` for numerical stability at large `k`
- CLI flags mirror `binom`: `-k`/`--events`, `-l`/`--rate`, `--target`, `--min-prob`, `--precision`
- Output: PMF, CDF ≤ k, survival ≥ k; optional threshold check

### Application
The Poisson distribution models the probability of a given number of rare, independent events occurring in a fixed interval of time or space. Natural companion to `binom` for low-probability / high-trial-count scenarios (e.g. "given 3 server errors per hour on average, what is the probability of seeing 7 or more in the next hour?").

```bash
poisson -k 7 -l 3.0
poisson -k 5 -l 2.5 --target 8 --min-prob 0.05
```

### Target User Base
- **DevOps / SREs** monitoring error rates, alert thresholds, and incident frequency
- **Actuaries and risk analysts** modelling claim arrival rates
- **Scientists and researchers** already using `binom` for discrete probability problems
- Directly adjacent to the existing audience: anyone comfortable with `binom -n 100 -k 3 -p 0.03` will immediately understand `poisson -k 3 -l 3.0`

---

## 2. `normal` — Gaussian Distribution Calculator

### Architecture
- **Core functions:** `normal_pdf(x, mu, sigma)`, `normal_cdf(x, mu, sigma)`, `normal_ppf(p, mu, sigma)` (percent-point / quantile function)
- Implemented via `math.erf` — no external dependencies
- CLI flags: `-x`/`--value`, `-m`/`--mean`, `-s`/`--std`, `--between LOW HIGH`, `--quantile P`, `--precision`
- Output: PDF at x, P(X ≤ x), P(X ≥ x), optionally P(LOW ≤ X ≤ HIGH) or the x-value at quantile p

### Application
The normal distribution is the most widely used continuous distribution. Covers z-score calculations, confidence interval boundary checks, standardisation questions, and quality control (Six Sigma-style defect rate estimation).

```bash
normal -x 1.96 -m 0 -s 1
normal --between -1.96 1.96 -m 0 -s 1
normal --quantile 0.975 -m 0 -s 1
```

### Target User Base
- **Students and educators** verifying z-table lookups without needing scipy
- **QA engineers and manufacturing analysts** checking whether process measurements fall within tolerance bands
- **Data scientists** who want a quick sanity-check tool without spinning up a Python REPL
- Broader than `binom`/`birthday` — the most universally accessible addition to the suite

---

## 3. `zscore` — Z-Score Calculator

### Architecture
- **Core functions:** `zscore(x, mu, sigma)`, `zscore_to_prob(z, tail)`, `prob_to_zscore(p, tail)`
- Implemented via `math.erf` — no external dependencies; shares internals with `normal`
- CLI flags: `-x`/`--value`, `-m`/`--mean`, `-s`/`--std`, `--tail {lower,upper,two}`, `--reverse` (given a probability, return the critical z-score), `--precision`
- Output: z-score, corresponding tail probability, and percentile rank

### Application
Standardises a raw observation to its distance from the mean in standard-deviation units, and converts between z-scores and tail probabilities. Ubiquitous in quality control (Six Sigma), academic grading, financial return normalisation, and any scenario where "how unusual is this value?" is the core question.

```bash
# Z-score for a value of 75 from a distribution with mean 70, std 5
zscore -x 75 -m 70 -s 5

# Two-tailed probability for z = 1.96
zscore -x 1.96 -m 0 -s 1 --tail two

# Find the critical z-score at the 97.5th percentile
zscore --reverse --prob 0.975 --tail lower
```

### Target User Base
- **Students and educators** standardising exam scores or verifying z-table lookups
- **QA engineers** computing how many standard deviations a measurement falls from spec
- **Financial analysts** normalising asset returns for cross-security comparison
- A focused companion to `normal` — where `normal` computes full PDF/CDF/quantile for arbitrary distributions, `zscore` answers the single question: "how many sigmas away is this value, and how unusual is that?"

---

## 4. `sample` — Sample Size Calculator

### Architecture
- **Core functions:** `sample_size_proportion(p, margin, alpha)`, `sample_size_mean(sigma, delta, alpha, power)`, `sample_size_comparison(p1, p2, alpha, power)`
- z/t quantiles via `math.erf` — no external dependencies
- CLI flags: `--type {proportion,mean,comparison}`, `-p`/`--prop`, `--delta`, `--std`, `--margin`, `--alpha`, `--power`, `--sided {one,two}`, `--sweep` (table of n vs. achieved power)
- Output: minimum sample size, achieved power or margin of error at that n

### Application
Answers "how many observations do I need?" before running a study, experiment, or audit. Whether sizing an A/B test, planning a clinical trial, or determining how many items to inspect in a batch, sample size calculation is prerequisite to every inferential analysis.

```bash
# Minimum n to estimate a proportion within ±3% at 95% confidence
sample --type proportion --p 0.5 --margin 0.03

# Minimum n to detect a mean shift of 5 units (std=12) with 80% power
sample --type mean --delta 5 --std 12 --power 0.80

# Sweep: show achieved power across n = 50–300 for a two-proportion comparison
sample --type comparison --p1 0.40 --p2 0.50 --alpha 0.05 --sweep 50 300
```

### Target User Base
- **A/B testers and product analysts** sizing experiments before launch
- **Clinical researchers** meeting pre-specified sample size requirements
- **Auditors** determining how many records to sample for a given detection sensitivity
- Natural prerequisite companion to `confint` and `pvalue` — the "before" step to those tools' "after" analyses

---

## 5. `ttest` — t-Test Calculator

### Architecture
- **Core functions:** `one_sample_t(mean, mu0, std, n)`, `two_sample_t(mean1, mean2, std1, std2, n1, n2, equal_var)`, `paired_t(differences)`
- t-distribution CDF approximated via regularised incomplete beta (`math.lgamma`) — no external dependencies
- CLI flags: `--type {one-sample,two-sample,paired}`, `-n`/`--n1`/`--n2`, `--mean`/`--mean1`/`--mean2`, `--std`/`--std1`/`--std2`, `--mu0`, `--alpha`, `--sided {one,two}`, `--equal-var`
- Output: t-statistic, degrees of freedom, p-value, decision at `--alpha`, confidence interval for the mean difference

### Application
The workhorse of small-sample inference. One-tailed tests answer directional questions ("is the treated group *faster*?"); two-tailed tests answer symmetric questions ("is there *any* difference?"). Covers one-sample, independent two-sample (Welch and equal-variance), and paired designs.

```bash
# One-sample, two-tailed: is the mean different from 100? (n=20, mean=97.3, std=8.1)
ttest --type one-sample -n 20 --mean 97.3 --std 8.1 --mu0 100 --sided two

# One-tailed: does the treatment group score higher than control?
ttest --type two-sample --n1 25 --mean1 54.2 --std1 9.3 --n2 28 --mean2 49.8 --std2 11.1 --sided one

# Paired t-test from before/after differences
ttest --type paired --diffs 2.1,-0.3,4.5,1.8,3.2,-1.1,2.9
```

### Target User Base
- **Students and educators** working through hypothesis-testing exercises
- **Product and UX researchers** comparing metric means between two variants
- **Lab scientists** analysing small-n experiments where z-tests are inappropriate
- The "small-sample" counterpart to `zscore` and `normal` — users who know their n is small will reach for `ttest` first

---

## 6. `chisq` — Chi-Square Test Calculator

### Architecture
- **Core functions:** `chisq_gof(observed, expected)`, `chisq_independence(table)`, `chisq_cdf(x, df)`
- Chi-square CDF via regularised incomplete gamma (`math.lgamma`) — no external dependencies
- CLI flags: `--test {gof,independence}`, `--observed`, `--expected`, `--table` (repeated flag, one row per call), `--alpha`, `--precision`
- Output: χ² statistic, degrees of freedom, p-value, decision; per-cell contributions to χ² for residual diagnostics

### Application
Tests whether observed categorical frequencies match expected ones (goodness-of-fit) or whether two categorical variables are independent (contingency table). Widely used for survey analysis, genetics (Hardy–Weinberg equilibrium), market research, and categorical A/B test evaluation.

```bash
# Goodness-of-fit: are die rolls uniformly distributed?
chisq --test gof --observed 18,22,17,25,19,19 --expected 20,20,20,20,20,20

# Independence: is product preference associated with age group? (2×3 table)
chisq --test independence --table "40,30,20" --table "25,45,30"

# With explicit significance level
chisq --test gof --observed 52,48 --expected 50,50 --alpha 0.10
```

### Target User Base
- **Survey analysts and market researchers** testing whether response distributions fit expectations
- **Biologists and geneticists** checking population allele frequency assumptions
- **A/B testers** comparing multi-category outcome distributions between variants
- Pairs naturally with `binom` (binary outcomes) and `pvalue` — the categorical-data generalisation of binary proportion tests

---

## 7. `linreg` — Simple Linear Regression

### Architecture
- **Core functions:** `linreg(x, y)` → slope, intercept, R², SE(slope), SE(intercept), t-statistics and p-values; `predict(x_new, model, alpha)` → point estimate with confidence and prediction intervals
- Closed-form OLS via `math` and `statistics` — no external dependencies
- CLI flags: `--x`/`--y` (comma-separated values), `--file CSV`, `--predict X`, `--alpha`, `--format {table,json,csv}`, `--precision`
- Output: slope, intercept, R², F-statistic, overall p-value, coefficient CIs; optional residual table and prediction interval

### Application
Quantifies the linear relationship between two continuous variables and makes predictions with uncertainty bounds. Applicable across domains: calibration curves in lab science, price-versus-demand modelling, performance-versus-load curves in engineering, and trend lines in operations.

```bash
# Fit a line to paired x, y values
linreg --x 1,2,3,4,5 --y 2.1,3.9,6.2,7.8,10.1

# Fit from a CSV file and predict at x=6 with 95% confidence interval
linreg --file data.csv --predict 6 --alpha 0.05

# JSON output for downstream processing
linreg --x 10,20,30,40,50 --y 15,28,41,55,68 --format json
```

### Target User Base
- **Scientists and engineers** fitting calibration curves or modelling physical relationships
- **Economists and analysts** estimating demand or trend lines from tabular data
- **Students** learning regression who want a no-setup CLI alternative to Excel or a notebook
- Extends the suite into predictive modelling — the natural destination after summarising data with `expected` and testing with `ttest` or `chisq`

---

## 8. `expected` — Expected Value and Variance Calculator

### Architecture
- Accepts a discrete probability distribution as paired lists or a CSV/JSON file: `--outcomes 1,2,3,4,5,6 --probs 0.1,0.2,0.3,0.2,0.1,0.1`
- **Core functions:** `expected_value(outcomes, probs)`, `variance(outcomes, probs)`, `std_dev(outcomes, probs)`, `entropy(probs)`
- Optional: moment generating function value at a given `t` (`--mgf T`)
- Output: E[X], Var(X), SD(X), Shannon entropy

### Application
A general-purpose tool for computing summary statistics of any user-defined discrete distribution. Useful for analysing custom payout tables, lottery structures, card game odds, or any scenario where the user supplies raw outcomes and weights — complementing `birthday`'s `--weights` flag philosophy.

```bash
expected --outcomes 0,1,5,10 --probs 0.50,0.25,0.15,0.10
expected --file payouts.csv
```

### Target User Base
- **Game designers and odds analysts** modelling prize tables or betting structures
- **Educators** teaching probability theory who want a quick computation tool
- **Researchers** exploring custom distributions before committing to a full scipy/numpy setup
- Power users of `birthday --weights` who want deeper summary statistics on their custom pools

---

## 9. `hypergeo` — Hypergeometric Distribution Calculator

### Architecture
- **Core functions:** `hypergeo_pmf(k, N, K, n)`, `hypergeo_cdf_le(k, N, K, n)`, `hypergeo_cdf_ge(k, N, K, n)`
- Uses `math.comb` (already used in `binom`) for exact computation
- CLI flags: `-N`/`--population`, `-K`/`--successes-in-pop`, `-n`/`--draws`, `-k`/`--observed`, `--precision`
- Output: PMF, CDF ≤ k, CDF ≥ k

### Application
Models drawing without replacement from a finite population — the key distinction from the binomial. Classic applications include quality control sampling (defective units in a batch), card game probability (probability of drawing exactly 2 aces from a 5-card hand), and audit sampling.

```bash
# P(exactly 2 aces in a 5-card hand from a standard 52-card deck)
hypergeo -N 52 -K 4 -n 5 -k 2

# Audit: 10 defective items in a batch of 100; sample 15 — P(catching ≥ 2)
hypergeo -N 100 -K 10 -n 15 -k 2
```

### Target User Base
- **Auditors and compliance analysts** sizing samples to detect defects with known confidence
- **Card game / tabletop RPG designers** computing draw probabilities
- **Scientists** running enrichment analyses (e.g. gene-set overlap)
- Direct conceptual neighbour of `binom` — the natural "sampling without replacement" counterpart; existing `binom` users will find the interface familiar

---

## 10. `streak` — Consecutive Success/Failure Streak Probability

### Architecture
- **Core functions:**
  - `prob_at_least_one_streak(n, k, p)` — probability of at least one run of `k` consecutive successes in `n` independent Bernoulli trials
  - `expected_longest_streak(n, p)` — expected length of the longest run
- Uses dynamic programming (DP table) for exact computation; O(n·k) time and O(k) space
- CLI flags: `-n`/`--trials`, `-k`/`--streak-length`, `-p`/`--prob`, `--longest`, `--precision`

### Application
Answers questions like "In 162 baseball games with a 0.300 batting average, what is the probability of a hitting streak of at least 20 games?" or "In 50 sales calls with a 10% close rate, what is the probability of getting 3 consecutive closes?". Streak/run probability is a common question in sports analytics, quality control (consecutive defects), and trading (win/loss streaks).

```bash
# P(at least one streak of 5+ heads in 100 fair coin flips)
streak -n 100 -k 5 -p 0.5

# Expected longest run of successes
streak -n 162 -p 0.300 --longest
```

### Target User Base
- **Sports analysts and bettors** evaluating hot/cold streaks
- **Traders and quants** assessing drawdown streaks in a strategy's win/loss record
- **QA engineers** monitoring consecutive failures in automated test suites
- Users already comfortable with `binom` who want to reason about sequential structure, not just aggregate counts

---

## 11. `bayes` — Bayesian Probability Updater

### Architecture
- **Core functions:** `posterior(prior, likelihood, evidence)`, `update(prior, likelihood_hit, likelihood_miss)` (sequential updates)
- Accepts a prior, a likelihood of evidence given hypothesis, and a likelihood of evidence given ¬hypothesis
- CLI supports `--prior`, `--likelihood-pos`, `--likelihood-neg`, and `--iterations` for repeated updating (e.g. multiple test results)
- Output: posterior probability after each update step; table or single value

### Application
Bayesian updating underpins medical testing (sensitivity/specificity), spam filtering, and iterative belief revision. This tool provides an accessible, step-by-step command-line interface for P(H|E) calculations without requiring a full probabilistic programming library.

```bash
# Medical test: disease prevalence 1%, test sensitivity 99%, false positive rate 5%
bayes --prior 0.01 --likelihood-pos 0.99 --likelihood-neg 0.05

# Two sequential positive tests
bayes --prior 0.01 --likelihood-pos 0.99 --likelihood-neg 0.05 --iterations 2
```

### Target User Base
- **Medical and public health analysts** interpreting diagnostic test results
- **Security analysts** updating threat probability as evidence accumulates
- **Students** learning Bayesian reasoning who need a tactile, command-line walkthrough
- A natural complement to the frequentist tools (`binom`, `poisson`) already in the suite — giving `pythodds` both frequentist and Bayesian perspectives

---

## ⚙️ Extended Tool Ideas (Dependency-Optional, Dynamic Input)

The tools below may introduce optional or required third-party dependencies — primarily for visualisation (`matplotlib`, `rich`) or numerical computation (`numpy`). Where dependencies are optional, tools degrade gracefully to plain-text output when the library is not installed. All tools continue to accept dynamic user-supplied variables to scale computation to real input.

---

## 12. `plotdist` — Distribution Visualiser

### Dependencies
- **Required:** `matplotlib` (plot rendering)
- **Optional:** `numpy` (faster linspace/meshgrid for large ranges; falls back to `range` + `math`)

### Architecture
- Accepts a distribution name and its parameters via flags; renders a PMF bar chart or PDF line plot to screen or saves to a file
- Supported distributions (initial): `binomial`, `poisson`, `normal`, `hypergeometric`
- CLI flags: `--dist DIST`, `--params KEY=VALUE [...]`, `--range MIN MAX`, `--output FILE`, `--title STR`, `--style {bar,line,step}`, `--dpi INT`
- Falls back to a Unicode block-character histogram in stdout if `matplotlib` is not installed (`--text` flag or auto-detected)
- Dynamic scaling: `--range` adjusts x-axis automatically; `--params` can be passed multiple times to overlay distributions on one plot

```bash
# Bar chart of Binomial(20, 0.4) PMF
plotdist --dist binomial --params n=20 p=0.4

# Overlay two Poisson distributions
plotdist --dist poisson --params lam=2 --params lam=6 --range 0 20

# Save a Normal PDF to a file
plotdist --dist normal --params mu=0 sigma=1 --output normal_curve.png

# Text fallback histogram (no matplotlib needed)
plotdist --dist binomial --params n=10 p=0.3 --text
```

### Target User Base
- **Educators and students** who want a visual companion to the existing numeric tools
- **Analysts** building quick presentation-ready charts from the CLI without opening a notebook
- **Existing `binom` and `birthday` users** who want to "see" the distributions they're already querying numerically
- The `--text` fallback makes this useful even in headless / SSH environments

---

## 13. `simulate` — Monte Carlo Probability Simulator

### Dependencies
- **Optional:** `numpy` (vectorised sampling, significantly faster for large `--trials`; pure `random` module used as fallback)

### Architecture
- Runs repeated random experiments to estimate probabilities empirically, cross-validating analytical results from `binom`, `birthday`, etc.
- Modes: `--experiment {binomial,birthday,streak,custom}`
- User-supplied variables: `--trials N` (number of simulations), `--params KEY=VALUE [...]`, `--seed INT` (reproducibility), `--confidence` (prints 95% CI around the estimate)
- Output: estimated probability, standard error, optional comparison to analytical value, optional CSV of per-trial results (`--dump`)
- `--scale` flag dynamically adjusts trial count based on desired precision: `--scale 0.001` runs enough trials to achieve ±0.1% standard error

```bash
# Empirically estimate P(X >= 5) for Binomial(10, 0.4) using 100,000 simulations
simulate --experiment binomial --params n=10 k=5 p=0.4 --trials 100000

# Birthday problem simulation for a pool of 365, group of 23
simulate --experiment birthday --params pool=365 group=23 --trials 50000 --confidence

# Auto-scale trials for ±0.01 standard error
simulate --experiment binomial --params n=20 k=8 p=0.5 --scale 0.01 --seed 42
```

### Target User Base
- **Students and educators** verifying analytical results through simulation
- **Researchers** stress-testing edge cases where closed-form approximations may lose accuracy
- **Power users** of existing tools who want to validate outputs empirically
- The `--scale` flag is especially valuable for users who don't know how many trials are "enough" for a given precision target

---

## 14. `oddsconv` — Odds Format Converter

### Dependencies
- None (pure Python)

### Architecture
- Converts between all major odds formats: **decimal**, **fractional**, **American (moneyline)**, **implied probability**, and **Hong Kong / Malay / Indonesian** odds
- Dynamic: accepts any one format as input and outputs all others simultaneously
- CLI flags: `--decimal F`, `--fractional N/D`, `--american INT`, `--prob F`, `--hk F`, `--malay F`, `--indo F`
- Optional: `--vig` to back-calculate overround/vig from a set of implied probabilities (`--prob 0.52 0.51` → prints book margin)
- Output: table of all equivalent representations, with implied probability and fair value

```bash
# Convert decimal odds to all formats
oddsconv --decimal 2.50

# Convert American moneyline to all formats
oddsconv --american -150

# Compute vig/overround from a two-outcome market
oddsconv --vig --prob 0.526 0.526
```

### Target User Base
- **Sports bettors and traders** working across platforms that use different odds formats
- **Quantitative analysts** building pricing models who need a fast reference tool
- **Educators** teaching probability through real-world gambling/markets examples
- Complements `binom` and `expected` for users doing sports analytics or betting modelling end-to-end

---

## 15. `confint` — Confidence Interval Calculator

### Dependencies
- None (pure Python via `math.erf` and lookup tables for t-distribution)

### Architecture
- Computes confidence intervals for proportions, means, and count data
- Modes: `--method {wilson,clopper-pearson,normal,t,poisson}`
- Dynamic user variables: `--n` (sample size), `--k` or `--p` (successes or proportion), `--mean`, `--std`, `--alpha` (significance level, default 0.05), `--sided {one,two}`
- Includes a `--sweep` flag to print a table of intervals across a range of sample sizes (e.g. `--sweep 10 500 --step 10`), useful for study design / power planning
- Output: lower bound, upper bound, width, midpoint; optionally formatted as `[LB, UB]` or `midpoint ± margin`

```bash
# Wilson confidence interval for 47 successes in 120 trials
confint --method wilson --n 120 --k 47

# t-interval for a small sample mean
confint --method t --n 15 --mean 23.4 --std 4.1

# Sweep: how does interval width shrink as n grows from 50 to 500?
confint --method wilson --p 0.4 --sweep 50 500 --step 50
```

### Target User Base
- **A/B testers and product analysts** reporting conversion rate confidence intervals
- **Clinical and public health researchers** computing prevalence estimates
- **Students** learning inferential statistics who want a CLI alternative to lookup tables
- The `--sweep` flag is particularly useful for researchers planning sample sizes — a direct extension of `birthday`'s range-table philosophy applied to inference

---

## 16. `pvalue` — p-value and Hypothesis Test Calculator

### Dependencies
- None (pure Python; t, chi-squared, and F quantiles via numerical approximation using `math`)

### Architecture
- Supports one-sample and two-sample tests: z-test, t-test, chi-squared goodness-of-fit, binomial exact test
- Dynamic inputs: `--test {z,t,chi2,binom-exact}`, `--stat VALUE` (observed test statistic) or raw data flags (`--n`, `--k`, `--p0`, `--mean`, `--std`), `--alpha`, `--sided {one,two}`
- Output: test statistic, p-value, decision (reject / fail to reject at `--alpha`), effect size where applicable
- Optional `--sweep-alpha` to print decision boundaries across a range of α values

```bash
# One-sample z-test: is a coin fair? (480 heads in 1000 flips)
pvalue --test z --n 1000 --k 480 --p0 0.5

# Two-sided binomial exact test
pvalue --test binom-exact --n 30 --k 22 --p0 0.60

# Chi-squared goodness of fit
pvalue --test chi2 --observed 18,22,20,15,25 --expected 20,20,20,20,20
```

### Target User Base
- **Data analysts and scientists** conducting quick hypothesis tests from summary statistics
- **Students** in introductory statistics courses who want a CLI calculator for assignments
- **QA/software teams** running statistical tests on experiment results
- Natural follow-on to `binom` for users who reach "I have a result — is it significant?" — the logical next step after PMF/CDF queries

---

## 17. `sensitivity` — Parameter Sensitivity / Tornado Chart

### Dependencies
- **Optional:** `matplotlib` (tornado/bar chart output); degrades to a ranked plain-text table
- None for core computation

### Architecture
- Takes a target formula or pythodds function (`--func {binom-pmf,poisson-pmf,normal-cdf,...}`) and a set of base-case parameters, then sweeps each parameter independently across a user-specified range
- Dynamically scales: `--range-pct P` sweeps each parameter ±P% from its base value; `--range-abs` allows per-parameter absolute ranges
- Output: ranked table or tornado chart showing which parameter has the greatest impact on the output value
- Supports custom expressions via `--expr "binom_pmf(n, k, p)"` for power users

```bash
# How sensitive is P(X=3 | n=10) to ±20% changes in each of n, k, p?
sensitivity --func binom-pmf --params n=10 k=3 p=0.4 --range-pct 20

# Sensitivity of Poisson PMF to ±1 unit changes in lambda
sensitivity --func poisson-pmf --params k=5 lam=3.0 --range-abs lam=1.0

# Save a tornado chart
sensitivity --func normal-cdf --params x=1.5 mu=0 sigma=1 --range-pct 30 --output tornado.png
```

### Target User Base
- **Risk analysts and quants** who need to know which inputs drive a probability estimate
- **Researchers** presenting results who want to show robustness (or fragility) of a finding
- **Advanced users** of the existing tools who want to understand how outputs change as their assumptions change
- The dynamic `--range-pct` flag makes this especially accessible: users don't need to specify exact ranges, just a percentage tolerance

---

## 18. `randforest` — Random Forest Classifier / Regressor

### Dependencies
- **Required:** `scikit-learn` (decision tree and ensemble fitting, feature importances)
- **Optional:** `numpy` (faster array handling; falls back to standard lists for small datasets), `pandas` (CSV ingestion with named columns; falls back to `csv` module)

### Architecture
- Wraps `sklearn.ensemble.RandomForestClassifier` / `RandomForestRegressor` behind a consistent CLI interface, keeping the same data-in → metrics-out philosophy as the rest of the suite
- Detects task type automatically from `--target-type {auto,classify,regress}` (default `auto`: classifies if the target column has ≤ 20 unique values)
- User-supplied variables: `--file CSV`, `--target COLUMN`, `--features COL [...]` (default: all non-target columns), `--trees N` (default 100), `--max-depth INT`, `--test-size F` (train/test split fraction, default 0.2), `--seed INT`, `--cv K` (k-fold cross-validation folds, default disabled)
- Output:
  - **Classification:** accuracy, precision, recall, F1, confusion matrix (plain-text), top-N feature importances
  - **Regression:** RMSE, MAE, R², top-N feature importances
  - `--format {table,json,csv}` for importances and metrics
  - `--predict-file CSV` to score new observations after fitting

```bash
# Classify from a CSV file, auto-detect task type
randforest --file data.csv --target label

# Regression with 200 trees, max depth 5, reproducible seed
randforest --file housing.csv --target price --trees 200 --max-depth 5 --seed 42 --target-type regress

# 5-fold cross-validation, JSON output of metrics and importances
randforest --file iris.csv --target species --cv 5 --format json

# Score new data after fitting
randforest --file train.csv --target outcome --predict-file new_obs.csv
```

### Target User Base
- **Data analysts and data scientists** who want a quick model baseline from the command line without writing boilerplate notebook code
- **Researchers** doing exploratory feature importance analysis on tabular datasets before committing to a full modelling pipeline
- **Students** learning ensemble methods who want a tactile CLI interface to complement sklearn tutorials
- Power users of `linreg` who need a non-linear, multi-feature model with built-in feature importance — the natural "what if the relationship isn't linear?" follow-on

---

## Summary Table

| Command | Distribution / Concept | Deps (optional*) | Zero-dep fallback? | Closest existing tool |
|---|---|---|---|---|
| `poisson`     | Poisson                              | None                      | N/A                | `binom`                       |
| `normal`      | Normal / Gaussian                    | None                      | N/A                | `binom`                       |
| `zscore`      | Z-score standardisation              | None                      | N/A                | `normal`                      |
| `sample`      | Sample size calculation              | None                      | N/A                | `confint`                     |
| `ttest`       | 1- & 2-sample t-tests                | None                      | N/A                | `pvalue`                      |
| `chisq`       | Chi-square tests                     | None                      | N/A                | `pvalue`                      |
| `linreg`      | Simple linear regression             | None                      | N/A                | `expected`                    |
| `expected`    | Discrete EV / variance               | None                      | N/A                | `birthday --weights`          |
| `hypergeo`    | Hypergeometric                       | None                      | N/A                | `binom`                       |
| `streak`      | Run / streak probability             | None                      | N/A                | `binom`                       |
| `bayes`       | Bayesian posterior update            | None                      | N/A                | `birthday`                    |
| `plotdist`    | Distribution visualiser              | `matplotlib`, `numpy`*    | ✅ Unicode text    | `binom` / `birthday`          |
| `simulate`    | Monte Carlo simulator                | `numpy`*                  | ✅ `random` module | `binom` / `birthday`          |
| `oddsconv`    | Odds format converter + vig calc     | None                      | N/A                | `expected`                    |
| `confint`     | Confidence interval calculator       | None                      | N/A                | `binom` / `normal`            |
| `pvalue`      | p-value and hypothesis test          | None                      | N/A                | `binom`                       |
| `sensitivity` | Parameter sensitivity / tornado      | `matplotlib`*             | ✅ ranked table    | all tools                     |
| `randforest`  | Random forest classifier / regressor | `scikit-learn`, `numpy`*, `pandas`* | ✅ numpy/pandas | `linreg`          |

\* Optional dependency: tool functions without it but with reduced output capability.