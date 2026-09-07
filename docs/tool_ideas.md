# Tool Ideas for `pythodds`

This document tracks **proposed tools not yet implemented** in `pythodds`. Each entry describes the proposed command name, its mathematical architecture, practical application, and target user base.

For the full list of implemented tools and their CLI entry points, see `README.md` and the `[project.scripts]` section in `pyproject.toml`.

---

## Implemented Tools (Reference)

| Command | Description |
|---------|-------------|
| `anova` | One-way ANOVA with Tukey HSD and Bonferroni post-hoc tests |
| `bayes` | Bayesian posterior update |
| `binom` | PMF, CDF, and survival function for Binomial(n, p) |
| `birthday` | Collision probability for uniform and non-uniform ID pools |
| `bootci` | Bootstrap confidence intervals |
| `breakeven` | Break-even (cost-volume-profit) analysis |
| `chisq` | Chi-square goodness-of-fit and independence tests |
| `collatz` | Collatz conjecture / hailstone sequences |
| `confint` | Confidence interval calculator |
| `crt` | Sunzi's Theorem (CRT) solver |
| `discount` | Real and nominal discount rates, PV, and inflation-adjusted NPV |
| `entropy` | Shannon entropy, KL divergence, and mutual information |
| `expected` | Expected value and variance for discrete distributions |
| `forecast` | Time series forecasting with prediction intervals |
| `geometric` | Geometric distribution PMF, CDF, and survival |
| `gini` | Gini coefficient and Lorenz curve |
| `jevons` | Jevons paradox / rebound effect modeling |
| `logreg` | Binary logistic regression with odds ratios and classification metrics |
| `life` | Life-in-weeks grid visualiser |
| `linreg` | Simple linear regression |
| `mlreg` | Multiple linear regression with confidence and prediction intervals |
| `normal` | Gaussian PDF, CDF, and quantile |
| `pearson` | Pearson correlation coefficient |
| `poisson` | Poisson PMF, CDF, and survival |
| `prime` | Prime number tools and factorization |
| `pvalue` | p-value and hypothesis test calculator |
| `pythag` | Pythagorean win expectation |
| `sample` | Sample size calculator |
| `slopeint` | Slope-intercept line algebra, conversion, intersection, and projection |
| `simulate` | Monte Carlo probability simulator |
| `sigmoid` | Sigmoid function σ(x), derivative, inverse logit, and Unicode sparkline |
| `spearman` | Spearman rank correlation |
| `streak` | Consecutive success/failure streak probability |
| `subnet` | IPv4 subnet mask, CIDR, and host range calculator |
| `ttest` | One- and two-sample t-tests |
| `euler` | Euler's number via limit/series, e^x, ln(x), identity, and γ constant |
| `weibull` | Weibull distribution PDF, CDF, survival, and hazard |
| `zscore` | Z-score calculator |

---

## Proposed Tools

All tools below are pure-Python unless a `Dependencies` section is noted. Dependency-optional tools degrade gracefully to plain-text output when the library is not installed.

---

## 1. `hypergeo` — Hypergeometric Distribution Calculator

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
- Auditors and compliance analysts: _sizing samples to detect defects with known confidence_
- Card game / tabletop RPG designers: _computing draw probabilities_
- Scientists: _running enrichment analyses (e.g. gene-set overlap)_
- Direct conceptual neighbour of `binom` — the natural "sampling without replacement" counterpart; existing `binom` users will find the interface familiar

---

## 2. `plotdist` — Distribution Visualiser

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
- Educators and students: _who want a visual companion to the existing numeric tools_
- Analysts: _building quick presentation-ready charts from the CLI without opening a notebook_
- Existing `binom` and `birthday` users: _who want to "see" the distributions they're already querying numerically_
- The `--text` fallback makes this useful even in headless / SSH environments

---

## 3. `oddsconv` — Odds Format Converter

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
- Sports bettors and traders: _working across platforms that use different odds formats_
- Quantitative analysts: _building pricing models who need a fast reference tool_
- Educators: _teaching probability through real-world gambling/markets examples_
- Complements `binom` and `expected` for users doing sports analytics or betting modeling end-to-end

---

## 4. `sensitivity` — Parameter Sensitivity / Tornado Chart

### Dependencies
- **Optional:** `matplotlib` (tornado/bar chart output); degrades to a ranked plain-text table

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
- Risk analysts and quants: _who need to know which inputs drive a probability estimate_
- Researchers: _presenting results who want to show robustness (or fragility) of a finding_
- Advanced users: _of existing tools who want to understand how outputs change as their assumptions change_
- The dynamic `--range-pct` flag makes this especially accessible: users don't need to specify exact ranges, just a percentage tolerance

---

## 5. `randforest` — Random Forest Classifier / Regressor

### Dependencies
- **Required:** `scikit-learn` (decision tree and ensemble fitting, feature importances)
- **Optional:** `numpy` (faster array handling; falls back to standard lists for small datasets), `pandas` (CSV ingestion with named columns; falls back to `csv` module)

### Architecture
- Wraps `sklearn.ensemble.RandomForestClassifier` / `RandomForestRegressor` behind a consistent CLI interface, keeping the same data-in → metrics-out philosophy as the rest of the suite
- Detects task type automatically from `--target-type {auto,classify,regress}` (default `auto`: classifies if the target column has ≤ 20 unique values)
- User-supplied variables: `--file CSV`, `--target COLUMN`, `--features COL [...]` (default: all non-target columns), `--trees N` (default 100), `--max-depth INT`, `--test-size F` (train/test split fraction, default 0.2), `--seed INT`, `--cv K` (k-fold cross-validation folds, default disabled)
- Output: Classification: accuracy, precision, recall, F1, confusion matrix, top-N feature importances; Regression: RMSE, MAE, R², top-N feature importances
- `--format {table,json,csv}` for importances and metrics; `--predict-file CSV` to score new observations after fitting

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
- Data analysts and data scientists: _who want a quick model baseline from the command line without writing boilerplate notebook code_
- Researchers: _doing exploratory feature importance analysis on tabular datasets before committing to a full modeling pipeline_
- Students: _learning ensemble methods who want a tactile CLI interface to complement sklearn tutorials_
- Power users of `linreg` who need a non-linear, multi-feature model with built-in feature importance — the natural "what if the relationship isn't linear?" follow-on

---

## 6. `ewma` — Exponentially Weighted Moving Average & Control Limits

### Architecture
- Computes an EWMA (exponentially weighted moving average) of a series and derives upper/lower control limits (UCL/LCL) from the rolling variance estimate — the statistical basis of real-time anomaly detection and EWMA control charts
- **Core functions:** `ewma(data, lam)` → smoothed series; `ewma_variance(data, lam)` → rolling variance; `control_limits(data, lam, k)` → UCL and LCL at ±k sigma
- CLI flags: `--data CSV_OR_VALUES`, `--lambda F` (smoothing parameter 0 < λ ≤ 1, default 0.2), `--k F` (sigma multiplier for limits, default 3.0), `--format {table,json,csv}`
- Output: original values, EWMA values, rolling variance, UCL, LCL, and a boolean `out_of_control` flag per row

```bash
# EWMA chart with 3-sigma control limits (λ=0.2)
ewma --data 10.1,9.8,10.3,10.0,9.7,11.2,10.1,10.4 --lambda 0.2 --k 3.0

# Tighter smoothing (λ=0.1) for slow-moving processes
ewma --data metrics.csv --lambda 0.1 --k 2.5 --format csv

# JSON output for piping to plotdist or downstream alerting
ewma --data error_counts.csv --lambda 0.3 --format json
```

### Target User Base
- DevOps / SREs and platform engineers: _building statistical process control charts for service metrics_
- Manufacturing and QA engineers: _running EWMA control charts on production measurements_
- Analysts: _needing a lightweight alternative to full SPC software for monitoring KPIs_
- Direct companion to `forecast` — `forecast` projects future values, `ewma` monitors current values for deviation from expected behaviour

---

## 7. `vartest` — Variance Equality Tests

### Architecture
- Tests whether two or more samples have equal variances — a critical prerequisite for `ttest --equal-var` and many ANOVA-based analyses
- Tests: `--test {f,levene,bartlett}` — F-test (two samples), Levene (robust, 2+ samples), Bartlett (2+ samples, assumes normality)
- CLI flags: `--data GROUP1 GROUP2 [...]` (comma-separated values per group), `--file CSV --group-col COL --value-col COL`, `--alpha F`, `--sided {one,two}`
- Output: test statistic, degrees of freedom, p-value, decision; sample variances and ratio for the F-test
- Pure Python via `math.lgamma` for F and chi-square CDFs

```bash
# F-test for equality of variances between two groups
vartest --test f --data "12.1,11.8,12.5,11.9" "9.8,10.3,10.1,9.7,10.5"

# Levene's test across three groups from a CSV
vartest --test levene --file experiment.csv --group-col treatment --value-col response

# Bartlett's test with explicit significance level
vartest --test bartlett --data "1.2,1.5,1.3" "2.1,2.4,2.2,2.0" --alpha 0.01
```

### Target User Base
- Researchers and analysts: _validating the equal-variance assumption before running a two-sample t-test_
- QA and manufacturing engineers: _comparing process variability across production lines or shifts_
- Students: _learning applied statistics who need to check assumptions, not just run tests_
- A natural pre-flight check for `ttest` — the question "can I use `--equal-var`?" is answered directly by `vartest`

---

## 8. `taylor` — Taylor Series Approximation

### Architecture
- **Core functions:** `taylor_series(func, a, x, n)` → approximation value and coefficients; `taylor_error(func, a, x, n)` → actual value, approximation, absolute and relative error
- Computes the n-th order Taylor series expansion of common mathematical functions around a point `a` and evaluates at `x`
- Supported functions: `exp`, `sin`, `cos`, `tan`, `ln`, `sqrt`, `sinh`, `cosh`, and user-defined custom functions via `--custom "f(x) = ..."`
- CLI flags: `--func {exp,sin,cos,ln,sqrt,sinh,cosh,custom}`, `--center A` (expansion point), `--eval X` (evaluation point), `--order N`, `--terms` (show individual terms), `--compare` (compare to true value), `--precision INT`, `--format {table,json}`
- Output: series coefficients, partial sums (1st through n-th order), final approximation, comparison to actual function value with error metrics
- Pure Python — no external dependencies

### Application
Taylor series are fundamental for numerical approximation, understanding function behavior near a point, and deriving efficient computational methods. Applications include numerical analysis (algorithm design for sin/cos/exp in calculators), physics (linearizing equations of motion), signal processing (filter design), and teaching calculus concepts tactilely.

```bash
# Approximate e^x at x=1 using 5th-order Taylor series around x=0
taylor --func exp --center 0 --eval 1 --order 5 --compare

# Show individual terms of sin(x) expansion at x=π/4 around 0
taylor --func sin --center 0 --eval 0.7854 --order 7 --terms

# Compare increasing order approximations: sweep orders 1–10
taylor --func ln --center 1 --eval 1.5 --order 10 --compare --format json
```

### Target User Base
- Students and educators: _visualizing Taylor series convergence and approximation quality_
- Numerical analysts and engineers: _designing or debugging custom function approximations_
- Scientists: _deriving linearized models (1st-order) or quadratic approximations (2nd-order) of complex equations_
- The first truly "pure math" tool in the suite — where other tools focus on probability/statistics, `taylor` serves users building computational methods or learning analysis

---

## 9. `compound` — Compound Interest & Time Value of Money

### Architecture
- **Core functions:** `future_value(pv, r, n, t)`, `present_value(fv, r, n, t)`, `annuity(pmt, r, n, t)`, `pmt_from_pv(pv, r, n, t)`, `effective_rate(nom_rate, n)`, `continuous_compound(pv, r, t)`
- Computes compound interest, annuities, payment schedules, and effective annual rates
- Modes: `--mode {fv,pv,annuity,payment,effective,continuous}`
- CLI flags: `--pv F` (present value), `--fv F` (future value), `--rate F` (interest rate per period), `--periods INT` (compounding periods per year), `--time F` (years), `--pmt F` (payment per period), `--precision INT`, `--format {table,json,csv}`, `--schedule` (amortization table)
- Output: computed value, total interest earned/paid, effective annual rate; optional payment schedule with per-period interest/principal breakdown
- Pure Python — no external dependencies

### Application
Fundamental to personal finance (savings, loans, mortgages), investment analysis (NPV, IRR prerequisites), retirement planning (annuity valuation), and business finance (capital budgeting). Answers "how much will I have in 30 years?", "what monthly payment fits my budget?", "what is the effective APR given monthly compounding?".

```bash
# Future value: $10,000 at 5% annual interest, compounded monthly, for 10 years
compound --mode fv --pv 10000 --rate 0.05 --periods 12 --time 10

# Monthly payment on a $300,000 loan at 4% over 30 years
compound --mode payment --pv 300000 --rate 0.04 --periods 12 --time 30

# Effective annual rate for 6% nominal rate, compounded daily
compound --mode effective --rate 0.06 --periods 365

# Amortization schedule for a loan
compound --mode payment --pv 50000 --rate 0.06 --periods 12 --time 5 --schedule
```

### Target User Base
- Individuals: _planning savings, comparing loan offers, or evaluating investment returns_
- Financial analysts: _computing NPV inputs, comparing financing options, or teaching time value of money_
- Students: _learning finance who need a CLI calculator for homework or exam prep_
- The most immediately practical "real-life math" tool — every adult with a bank account or loan can use this daily

---

## 10. `matrix` — Matrix Operations & Linear Algebra

### Dependencies
- **Optional:** `numpy` (efficient operations on large matrices, eigenvalues, SVD); falls back to pure-Python nested lists for small matrices

### Architecture
- **Core functions:** `add`, `multiply`, `transpose`, `inverse`, `determinant`, `trace`, `rank`, `eigenvalues`, `eigenvectors`, `svd`, `solve` (linear system Ax = b)
- Accepts matrix input via `--matrix "[[1,2],[3,4]]"` or `--file CSV`; multiple matrices for operations via `--matrix-a`, `--matrix-b`
- CLI flags: `--op {add,multiply,transpose,inverse,det,trace,rank,eigen,svd,solve}`, `--matrix`/`--matrix-a`/`--matrix-b`, `--vector` (for solve mode), `--precision INT`, `--format {table,json,latex}`, `--show-steps` (for Gaussian elimination in solve/inverse)
- Output: result matrix/value, condition number (for inverse/solve), optional step-by-step algorithm trace

### Application
Linear algebra underpins machine learning (PCA, neural network backprop), computer graphics (3D transformations), physics (quantum mechanics, classical mechanics matrix formulations), economics (input-output models, Markov chains), and engineering (control theory, structural analysis). CLI access without MATLAB/Octave.

```bash
# Matrix multiplication
matrix --op multiply --matrix-a "[[1,2],[3,4]]" --matrix-b "[[5,6],[7,8]]"

# Determinant and inverse
matrix --op det --matrix "[[2,1],[5,3]]"
matrix --op inverse --matrix "[[4,7],[2,6]]" --precision 4

# Solve Ax = b
matrix --op solve --matrix "[[3,1],[-1,2]]" --vector "[9,8]" --show-steps

# Eigenvalues and eigenvectors (numpy required for large matrices)
matrix --op eigen --matrix "[[6,-1],[2,3]]" --format json
```

### Target User Base
- Students: _verifying homework solutions for linear algebra courses_
- Data scientists: _inspecting matrix properties (rank, condition number) before regression/PCA_
- Engineers and physicists: _solving small linear systems or transformation matrices_
- Complements `mlreg` (which uses matrix algebra internally) — power users who want to inspect the $(X^TX)^{-1}$ matrix or check for multicollinearity via determinant/eigenvalues

---

## 11. `fibonacci` — Fibonacci Sequence & Golden Ratio

### Architecture
- **Core functions:** `fib(n)` (n-th Fibonacci number), `fib_seq(n)` (first n terms), `golden_ratio()`, `fib_ratio(n)` (ratio F(n)/F(n-1) approaching φ), `lucas(n)` (Lucas numbers), `binet_formula(n)` (closed-form calculation)
- Uses matrix exponentiation for large n (O(log n) time), memoization for sequences
- CLI flags: `--nth`, `--seq`, `--ratio`, `--lucas`, `--golden`, `--approx`, `--properties`, `--precision INT`, `--format {table,json,csv}`
- Output: Fibonacci number(s), ratio, φ constant, comparison table (showing convergence of F(n)/F(n-1) → φ)
- Pure Python — no external dependencies

### Application
The Fibonacci sequence appears in nature (phyllotaxis, branching, spirals), art (golden rectangle, golden spiral in composition), finance (Fibonacci retracements in technical analysis), and computer science (algorithm analysis, data structure performance). Practical uses include algorithm complexity analysis (Fibonacci heap), teaching recursion/DP, and exploring growth patterns.

```bash
# The 50th Fibonacci number
fibonacci --nth 50

# First 20 terms of the sequence
fibonacci --seq 20 --format csv

# Show convergence of ratio to golden ratio
fibonacci --ratio 30

# Compare iterative vs. Binet formula accuracy
fibonacci --approx 40
```

### Target User Base
- Students and educators: _teaching recursion, dynamic programming, or mathematical sequences_
- Designers and artists: _using the golden ratio for layout, composition, or proportion calculations_
- Traders: _applying Fibonacci retracement levels in technical analysis_
- Computer scientists: _analyzing algorithm complexity or Fibonacci heap performance_

---

## 12. `grover` — Grover's Quantum Search Algorithm Simulator

### Architecture
- **Core functions:** `optimal_iterations(n)` → `floor(π/4 · √n)`; `success_probability(n, k, iterations)` → analytical amplitude calculation; `amplitude_evolution(n, k, t)` → probability at each step; `speedup_ratio(n)` → classical vs quantum step ratio
- Pure Python — no external dependencies (quantum amplitudes computed analytically from trigonometry via `math`)
- CLI flags: `-n`/`--items INT` (search space size), `-k`/`--targets INT` (number of marked items, default 1), `--iterations INT` (override optimal; default = optimal), `--compare` (side-by-side classical O(n) vs quantum O(√n) step counts), `--sweep` (table of success probability across iteration counts), `--format {table,json}`
- Output: optimal iteration count, success probability at optimal iterations, amplitude evolution table, classical vs quantum step comparison

### Application
Grover's algorithm provides a provable quadratic speedup over classical unstructured search, finding a marked item in N elements with O(√N) oracle queries versus O(N) classically. A cornerstone result in quantum computing and cryptanalysis (implications for symmetric key security). This tool provides an analytical simulator — computing exact probabilities from amplitude equations — making the algorithm accessible without quantum hardware.

```bash
# Optimal Grover search over 1,024 items (1 target)
grover -n 1024

# Search with 3 marked targets in a space of 256
grover -n 256 -k 3

# Compare classical vs quantum steps across space sizes 2^4 through 2^20
grover --compare --sweep 16 1048576
```

### Target User Base
- CS and physics students: _visualizing quantum amplitude amplification and understanding the quadratic speedup_
- Quantum computing researchers and educators: _demonstrating Grover's oracle complexity interactively_
- Security engineers: _understanding quantum threats to symmetric cryptographic primitives_
- The most conceptually distinctive tool in the suite — bridging classical probability and quantum information theory

---

## 13. `exponential` — Exponential Distribution Calculator

### Architecture
- **Core functions:** `exp_pdf(x, lam)`, `exp_cdf(x, lam)`, `exp_survival(x, lam)`, `exp_hazard(x, lam)`, `exp_quantile(p, lam)`
- Pure Python via `math.exp` — no external dependencies
- CLI flags: `-x F` (value), `--lambda F`/`-l F` (rate parameter), `--quantile F` (return x at given CDF probability), `--survival` (return 1 - CDF), `--table MIN MAX STEP`, `--mean` (print expected waiting time 1/λ), `--precision INT`
- Output: PDF, CDF, survival probability, hazard rate, optional quantile or range table

```bash
# PDF and CDF at x=2 for lambda=0.5
exponential -x 2.0 --lambda 0.5

# 95th percentile of inter-arrival time (lambda=1/3 arrivals per minute)
exponential --quantile 0.95 --lambda 0.333

# Survival table from 0 to 10 in steps of 1
exponential --lambda 0.5 --table 0 10 1 --survival
```

### Application
The continuous analog of the geometric distribution; models waiting times between independent events in a Poisson process. Used for inter-arrival times (customer arrivals, network packets), component lifetimes (memoryless failure), and queueing theory. The unique "memoryless" property — knowing a component has survived to time t gives no information about its remaining lifetime — makes it the baseline model before considering `weibull` for wear-out effects.

### Target User Base
- DevOps / SREs: _modeling inter-request intervals and timeout thresholds_
- Reliability engineers: _baseline failure time modeling before fitting Weibull_
- Operations researchers and queueing theorists: _arrival and service time distributions_
- A continuous-distribution companion to `poisson` and `geometric` — completing the Poisson process family

---

## 14. `describe` — Descriptive Statistics

### Architecture
- **Core functions:** `describe(data)` → n, mean, median, mode, std, variance, min, max, q1, q3, iqr, skewness, kurtosis, range, cv (coefficient of variation)
- Pure Python via `statistics` and `math` — no external dependencies
- CLI flags: `--data CSV_OR_VALUES`, `--file CSV --col COL`, `--percentiles P [...]` (additional percentiles beyond quartiles), `--precision INT`, `--format {table,json,csv}`
- Output: full summary statistics table with location, spread, and shape measures

```bash
# Full summary for a list of values
describe --data 12.1,11.8,13.4,12.9,11.5,14.2,12.7

# From CSV, specific column with additional percentiles
describe --file sales.csv --col revenue --percentiles 10 25 75 90 99

# JSON output for downstream processing
describe --file measurements.csv --col weight --format json
```

### Application
Produces a one-shot summary of a dataset's location, spread, shape, and outlier potential. Eliminates the need to open a notebook or import pandas for a quick look at raw data. The natural first step before running any inferential analysis — `describe` before you test or model.

### Target User Base
- Data analysts and scientists: _initial EDA before choosing a test or model_
- QA engineers: _summarizing measurement distributions from automated test runs_
- Students: _learning exploratory data analysis without needing pandas_
- The universal entry point for any user with a column of numbers — the tool most likely to be reached for first

---

## 15. `effect` — Effect Size Calculator

### Architecture
- **Core functions:** `cohens_d(mean1, mean2, std1, std2, n1, n2)` → d and pooled SE; `eta_squared(ss_between, ss_total)` → η²; `omega_squared(ss_between, ms_within, k, n)` → ω²; `odds_ratio(a, b, c, d)` → OR and 95% CI; `risk_ratio(a, b, c, d)` → RR; `r_from_t(t, df)` → Pearson r
- Pure Python — no external dependencies
- CLI flags: `--measure {d,eta2,omega2,or,rr,r}`, `--mean1 F`, `--mean2 F`, `--std1 F`, `--std2 F`, `--n1 INT`, `--n2 INT`, `--ss-between F`, `--ss-total F`, `--ms-within F`, `--k INT`, `--table "A,B,C,D"`, `--t-stat F`, `--df INT`, `--interpret` (print Cohen's small/medium/large benchmarks), `--format {table,json}`
- Output: effect size value with confidence interval where applicable, interpretation label when `--interpret` is passed

```bash
# Cohen's d for two group means
effect --measure d --mean1 54.2 --std1 9.3 --n1 25 --mean2 49.8 --std2 11.1 --n2 28 --interpret

# Eta-squared from ANOVA output
effect --measure eta2 --ss-between 48.3 --ss-total 312.7

# Odds ratio from a 2x2 contingency table (a, b, c, d)
effect --measure or --table "40,20,30,50"
```

### Application
p-values tell you whether an effect exists; effect sizes tell you how large it is. Cohen's d, eta-squared, and odds ratios are the standard language of practical significance in clinical research, social science, and A/B testing. Essential for power analysis, meta-analyses, and communicating results to stakeholders who need to know whether a statistically significant finding is also meaningfully large.

### Target User Base
- Clinical and social science researchers: _reporting standardised effect sizes alongside p-values_
- Product analysts: _quantifying the magnitude of A/B test wins, not just their significance_
- Students: _learning the distinction between statistical and practical significance_
- A natural complement to `ttest`, `anova`, and `chisq` — the "how big?" follow-up to "is it real?"

---

## 16. `combinatorics` — Permutations, Combinations & Counting

### Architecture
- **Core functions:** `permutations(n, r)`, `combinations(n, r)` (via `math.comb`), `multinomial(n, *ks)`, `derangements(n)`, `catalan(n)`, `stirling2(n, k)` (Stirling numbers, second kind), `bell(n)` (Bell numbers)
- Pure Python — no external dependencies; `math.comb` and `math.factorial` for exact integer arithmetic
- CLI flags: `--func {perm,comb,multi,derange,catalan,stirling2,bell}`, `-n INT`, `-r INT`, `--ks INT [...]` (for multinomial), `--table N` (print first N values of the sequence), `--format {table,json}`
- Output: exact integer result; optional sequence table

```bash
# How many ways to arrange 5 items chosen from 10?
combinatorics --func perm -n 10 -r 5

# How many ways to choose a committee of 4 from 12?
combinatorics --func comb -n 12 -r 4

# Multinomial: ways to assign 12 people to groups of 3, 4, and 5
combinatorics --func multi -n 12 --ks 3 4 5

# Number of derangements of 8 items (permutations with no fixed points)
combinatorics --func derange -n 8

# First 10 Catalan numbers
combinatorics --func catalan --table 10
```

### Application
Counting functions underpin probability calculations everywhere — the denominator in hypergeometric and binomial probabilities, the foundation of combinatorial proofs, and practical tools for scheduling, tournament bracket design, and resource allocation. A direct complement to `hypergeo` and `binom`, providing the explicit counting tools those distributions use internally.

### Target User Base
- Students: _computing combinatorial quantities for probability homework or math competitions_
- Statisticians: _verifying denominators in hypergeometric and multinomial probability calculations_
- Software engineers: _counting configurations, arrangements, or partitions in algorithm design_
- Puzzle and game designers: _evaluating search spaces and strategy complexity_

---

## 17. `liquidity` — Liquidity & Short-Term Solvency Ratios

### Architecture
- **Core functions:** `current_ratio(ca, cl)`, `quick_ratio(cash, securities, receivables, cl)`, `cash_ratio(cash, securities, cl)`, `net_working_capital(ca, cl)`, `working_capital_ratio(nwc, total_assets)`, `defensive_interval(liquid_assets, daily_expenses)`, `operating_cash_flow_ratio(ocf, cl)`
- Quick ratio computed either additively (cash + marketable securities + receivables) or subtractively (current assets − inventory − prepaid expenses); the tool accepts whichever inputs are available and reports which formula it used
- Modes: `--mode {current,quick,cash,nwc,interval,ocf,all}`
- CLI flags: `--current-assets F`, `--current-liabilities F`, `--inventory F`, `--prepaid F`, `--cash F`, `--securities F`, `--receivables F`, `--ocf F`, `--daily-expenses F`, `--benchmark F` (compare against a target or industry norm), `--precision INT`, `--format {table,json,csv}`
- Output: each ratio with its interpretation band (adequate / tight / distressed), net working capital in currency units, and the shortfall or surplus against `--benchmark`
- Pure Python — no external dependencies

### Application
Short-term solvency asks a single question: can the entity cover obligations coming due within a year? The current ratio is the blunt instrument; the quick and cash ratios strip out progressively less-liquid assets to test the same question under stress. Net working capital converts the ratio back into currency so the answer is a dollar figure rather than a multiple. The defensive interval reframes it as time — how many days of operating expenses the liquid assets cover with no further revenue.

```bash
# Current and quick ratios from a balance sheet
liquidity --mode all --current-assets 250000 --current-liabilities 120000 --inventory 60000 --prepaid 5000

# Acid test from component assets
liquidity --mode quick --cash 40000 --securities 15000 --receivables 85000 --current-liabilities 120000

# Days of runway at $3,200/day of operating expense
liquidity --mode interval --cash 40000 --securities 15000 --receivables 85000 --daily-expenses 3200

# Test against a 2.0 covenant floor, JSON for downstream alerting
liquidity --mode current --current-assets 250000 --current-liabilities 120000 --benchmark 2.0 --format json
```

### Target User Base
- Small business owners: _checking whether next quarter's payables are covered before taking on new commitments_
- Credit analysts and lenders: _screening borrowers against covenant thresholds_
- Accounting students: _working balance-sheet problems without a spreadsheet_
- Anyone reading a 10-Q: _turning the balance sheet into the three ratios that actually matter_

---

## 18. `solvency` — Leverage & Long-Term Solvency Ratios

### Architecture
- **Core functions:** `debt_to_equity(debt, equity)`, `debt_to_assets(debt, assets)`, `equity_multiplier(assets, equity)`, `times_interest_earned(ebit, interest)`, `ebitda_coverage(ebitda, interest)`, `debt_service_coverage(ebitda, interest, principal)`, `fixed_charge_coverage(ebit, lease, interest, principal, tax_rate)`, `lt_debt_to_capitalization(ltd, ltd_plus_equity)`, `net_debt_to_ebitda(debt, cash, ebitda)`
- Modes: `--mode {de,da,tie,dscr,fcc,ltd,netdebt,all}`
- CLI flags: `--total-debt F`, `--long-term-debt F`, `--total-equity F`, `--total-assets F`, `--ebit F`, `--ebitda F`, `--interest F`, `--principal F`, `--lease F`, `--cash F`, `--tax-rate F`, `--covenant F`, `--precision INT`, `--format {table,json,csv}`
- `--covenant` reports headroom: how far EBIT (or EBITDA) can fall in percentage terms before the stated coverage threshold is breached — the number a borrower actually needs
- Output: each ratio, its coverage interpretation, and for coverage modes the implied breach point
- Pure Python — no external dependencies

### Application
Where `liquidity` tests the next twelve months, leverage ratios test the capital structure itself. Times interest earned and its siblings (DSCR, fixed-charge coverage) ask how many times over current earnings cover mandatory payments; debt-to-equity and debt-to-assets ask how much of the entity someone else owns. Together they are the standard covenant package in a loan agreement, and the headroom calculation turns a passing ratio into a margin of safety.

```bash
# Full leverage picture
solvency --mode all --total-debt 850000 --total-equity 1200000 --total-assets 2400000 --ebit 310000 --interest 74000

# Times interest earned with a 3.0x covenant — how far can EBIT fall?
solvency --mode tie --ebit 310000 --interest 74000 --covenant 3.0

# Debt service coverage including principal amortization
solvency --mode dscr --ebitda 420000 --interest 74000 --principal 96000

# Net debt / EBITDA, the leverage multiple used in credit agreements
solvency --mode netdebt --total-debt 850000 --cash 130000 --ebitda 420000
```

### Target User Base
- Business owners and CFOs: _monitoring covenant headroom between reporting periods_
- Credit and equity analysts: _comparing capital structures across an industry_
- Commercial real estate investors: _DSCR is the single number underwriting turns on_
- Finance students: _leverage ratio homework with the formulas spelled out_

---

## 19. `altman` — Altman Z-Score Bankruptcy Prediction

### Architecture
- **Core functions:** `z_score_public(x1..x5)`, `z_score_private(x1..x5)`, `z_score_nonmanufacturing(x1..x4)`, `component_ratios(...)`, `classify(z, model)`
- Three published variants selected by `--model`:
  - `public` (Altman 1968): `Z = 1.2·X1 + 1.4·X2 + 3.3·X3 + 0.6·X4 + 1.0·X5`, where X4 uses **market** value of equity. Zones: distress `< 1.81`, grey `1.81–2.99`, safe `> 2.99`
  - `private` (Z′, 1983): `Z′ = 0.717·X1 + 0.847·X2 + 3.107·X3 + 0.420·X4 + 0.998·X5`, X4 uses **book** value of equity. Zones: distress `< 1.23`, grey `1.23–2.90`, safe `> 2.90`
  - `nonmanufacturing` (Z″): `Z″ = 6.56·X1 + 3.26·X2 + 6.72·X3 + 1.05·X4`, dropping asset turnover; `--emerging` adds the +3.25 constant for the emerging-market credit score. Zones: distress `< 1.10`, grey `1.10–2.60`, safe `> 2.60`
- Components: `X1` working capital / total assets, `X2` retained earnings / total assets, `X3` EBIT / total assets, `X4` equity / total liabilities, `X5` sales / total assets
- CLI flags: `--model {public,private,nonmanufacturing}`, `--emerging`, `--working-capital F`, `--retained-earnings F`, `--ebit F`, `--market-equity F`, `--book-equity F`, `--sales F`, `--total-assets F`, `--total-liabilities F`, `--current-assets F` / `--current-liabilities F` (derive working capital), `--contributions`, `--precision INT`, `--format {table,json,csv}`
- `--contributions` prints each weighted term's share of the total score, showing which ratio is dragging the firm toward distress — the same per-component breakdown `chisq` gives for cell residuals
- Output: Z value, named zone, distance to the nearest zone boundary, and the component table
- Pure Python — no external dependencies

### Application
The Z-score is the canonical composite solvency indicator: five balance-sheet and income-statement ratios collapsed into one number with empirically fitted weights and published cutoffs. Its value is less the score than the decomposition — a firm can sit in the grey zone because of thin retained earnings (a young company) or because of collapsing EBIT (a dying one), and the contribution breakdown separates those cases. Pairs naturally with `sensitivity` (#21) for sweeping an input, and with `liquidity` (#61) and `solvency` (#62), which compute several of the same underlying ratios.

```bash
# Classic Z-score for a public manufacturer
altman --model public --working-capital 130000 --retained-earnings 480000 --ebit 310000 \
       --market-equity 1800000 --sales 2900000 --total-assets 2400000 --total-liabilities 850000

# Private-firm Z' using book equity, with the component breakdown
altman --model private --current-assets 250000 --current-liabilities 120000 --retained-earnings 480000 \
       --ebit 310000 --book-equity 1200000 --sales 2900000 --total-assets 2400000 \
       --total-liabilities 850000 --contributions

# Z'' for a service business (no asset turnover term)
altman --model nonmanufacturing --working-capital 130000 --retained-earnings 480000 --ebit 310000 \
       --book-equity 1200000 --total-assets 2400000 --total-liabilities 850000

# Emerging-market credit score variant
altman --model nonmanufacturing --emerging --working-capital 130000 --retained-earnings 480000 \
       --ebit 310000 --book-equity 1200000 --total-assets 2400000 --total-liabilities 850000
```

### Target User Base
- Credit analysts: _screening a portfolio of counterparties for distress risk_
- Investors: _a fast first-pass filter before committing to deeper diligence_
- Auditors: _evidence for the going-concern assessment_
- Finance students: _the most-taught multivariate scoring model in corporate finance_

---

## 20. `profit` — Profitability, Return & Per-Share Metrics

### Architecture
- **Core functions:** `gross_margin`, `operating_margin`, `net_profit_margin`, `ebitda_margin`, `return_on_assets`, `return_on_equity`, `return_on_invested_capital`, `eps_basic(net_income, preferred_dividends, weighted_shares)`, `eps_diluted(..., dilutive_shares)`, `pe_ratio(price, eps)`, `payout_ratio`, `retention_ratio`, `dupont_three`, `dupont_five`
- Modes: `--mode {margin,returns,pershare,dupont,all}`
- DuPont decomposition: 3-step `ROE = net margin × asset turnover × equity multiplier`; 5-step splits net margin into `tax burden × interest burden × operating margin`, isolating whether a return is earned operationally or manufactured by leverage and tax treatment
- CLI flags: `--revenue F`, `--cogs F`, `--operating-income F`, `--ebitda F`, `--net-income F`, `--pretax-income F`, `--ebit F`, `--total-assets F`, `--total-equity F`, `--invested-capital F`, `--shares F`, `--dilutive-shares F`, `--preferred-dividends F`, `--dividends F`, `--price F`, `--precision INT`, `--format {table,json,csv}`
- Output: each metric as a percentage or per-share figure, with the DuPont chain printed as a multiplicative decomposition that reconciles to ROE
- Pure Python — no external dependencies

### Application
Solvency ratios say whether an entity survives; profitability ratios say whether it is worth keeping alive. EPS and the P/E built on it are the most-quoted numbers in equity markets, and the DuPont identity is the standard method for attributing a headline ROE to its operating, efficiency, and leverage sources — a 20% ROE from thin margins and heavy debt is a different business from a 20% ROE from pricing power. The trailing P/E computed here is the direct counterpart to the cyclically-adjusted CAPE proposed in #41.

```bash
# Every margin from an income statement
profit --mode margin --revenue 2900000 --cogs 1740000 --operating-income 380000 --net-income 244000

# Returns on assets, equity, and invested capital
profit --mode returns --net-income 244000 --total-assets 2400000 --total-equity 1200000 --invested-capital 2050000

# Basic and diluted EPS, payout ratio, and trailing P/E
profit --mode pershare --net-income 244000 --preferred-dividends 12000 --shares 500000 \
       --dilutive-shares 35000 --dividends 60000 --price 18.40

# Five-step DuPont: where does the ROE actually come from?
profit --mode dupont --revenue 2900000 --ebit 380000 --pretax-income 306000 --net-income 244000 \
       --total-assets 2400000 --total-equity 1200000
```

### Target User Base
- Investors: _computing EPS, P/E, and payout from a filing without a terminal subscription_
- Business owners: _tracking margin compression month over month_
- Analysts: _DuPont attribution when comparing two firms with identical ROE_
- Students: _the ratio set every introductory corporate finance course examines_

---

## 21. `turnover` — Activity Ratios & the Cash Conversion Cycle

### Architecture
- **Core functions:** `inventory_turnover(cogs, avg_inventory)`, `days_inventory_outstanding`, `receivables_turnover(revenue, avg_receivables)`, `days_sales_outstanding`, `payables_turnover(cogs, avg_payables)`, `days_payables_outstanding`, `asset_turnover(revenue, avg_assets)`, `working_capital_turnover`, `cash_conversion_cycle(dio, dso, dpo)`
- `CCC = DIO + DSO − DPO` — the days of operations the entity must finance out of its own pocket
- Modes: `--mode {inventory,receivables,payables,asset,ccc,all}`
- CLI flags: `--revenue F`, `--cogs F`, `--inventory F`, `--receivables F`, `--payables F`, `--total-assets F`, `--beginning-* F` / `--ending-* F` (average the two balances automatically), `--days INT` (period length, default 365), `--precision INT`, `--format {table,json,csv}`
- Output: turnover multiples and their day-equivalents side by side, plus the CCC with each component's contribution signed
- Pure Python — no external dependencies

### Application
Activity ratios are the leading indicator that liquidity ratios lag. A current ratio can hold steady while receivables quietly age and inventory stops moving; the cash conversion cycle catches that months earlier, because it measures the gap between paying suppliers and collecting from customers. A negative CCC — customers pay before suppliers are due — means the working capital cycle funds itself, which is why the metric matters as much to a small distributor as to a retailer.

```bash
# Full working capital cycle from period-average balances
turnover --mode all --revenue 2900000 --cogs 1740000 --inventory 320000 --receivables 410000 --payables 260000

# Average beginning and ending balances automatically
turnover --mode inventory --cogs 1740000 --beginning-inventory 290000 --ending-inventory 350000

# Cash conversion cycle on a 90-day quarter
turnover --mode ccc --revenue 725000 --cogs 435000 --inventory 320000 --receivables 410000 \
         --payables 260000 --days 90

# Days sales outstanding alone, JSON for a monthly dashboard
turnover --mode receivables --revenue 2900000 --receivables 410000 --format json
```

### Target User Base
- Operations and finance managers: _spotting a lengthening collection cycle before it becomes a cash crisis_
- Small business owners: _deciding whether to tighten credit terms or stretch payables_
- Credit analysts: _distinguishing a liquidity problem from a profitability problem_
- Students: _working capital management coursework_

---

## 22. `finhealth` — Personal Financial Health Ratios

### Architecture
- **Core functions:** `net_worth(assets, liabilities)`, `debt_to_income(monthly_debt, gross_monthly_income)`, `housing_ratio(housing_payment, gross_monthly_income)`, `savings_rate(saved, income)`, `emergency_fund_months(liquid_assets, monthly_essential_expenses)`, `personal_liquidity_ratio`, `solvency_ratio(net_worth, total_assets)`, `debt_to_asset_ratio`, `fi_ratio(net_worth, annual_expenses, swr)`, `years_to_fi(...)`
- Front-end (housing only) and back-end (all debt service) DTI reported separately, with the conventional 28/36 underwriting guideline and the 43% qualified-mortgage ceiling shown as reference bands
- FI ratio uses a configurable safe withdrawal rate (`--swr`, default 0.04), so the target is `annual_expenses / swr` rather than a hardcoded 25×
- Modes: `--mode {networth,dti,housing,savings,emergency,solvency,fi,all}`
- CLI flags: `--gross-income F`, `--net-income F`, `--housing-payment F`, `--other-debt-payments F`, `--monthly-expenses F`, `--essential-expenses F`, `--liquid-assets F`, `--total-assets F`, `--total-liabilities F`, `--annual-savings F`, `--swr F`, `--real-return F`, `--monthly` / `--annual` (input period), `--precision INT`, `--format {table,json,csv}`
- Output: each ratio with a plain-language band (e.g. DTI `≤ 36%` comfortable, `36–43%` stretched, `> 43%` above the QM limit), net worth in currency, emergency fund expressed in months, and years-to-FI given a real return assumption
- Pure Python — no external dependencies

### Application
The same solvency questions asked of a business apply to a household, with different names and different thresholds. Debt-to-income is what a mortgage underwriter computes; the emergency fund runway is the personal defensive interval; net worth over total assets is the personal solvency ratio. The tool exists because these are individually trivial and collectively never calculated — nobody opens a spreadsheet to divide two numbers, so the numbers go unchecked until a lender computes them instead. Complements `compound` (#24) for the projection side and `discount` for inflation-adjusted planning.

```bash
# Full personal snapshot
finhealth --mode all --gross-income 7500 --housing-payment 1950 --other-debt-payments 640 \
          --essential-expenses 4100 --liquid-assets 22000 --total-assets 415000 --total-liabilities 268000

# What a mortgage underwriter sees: front-end and back-end DTI
finhealth --mode dti --gross-income 7500 --housing-payment 1950 --other-debt-payments 640

# Months of runway if income stops tomorrow
finhealth --mode emergency --liquid-assets 22000 --essential-expenses 4100

# Progress toward financial independence at a 3.5% withdrawal rate
finhealth --mode fi --total-assets 415000 --total-liabilities 268000 --monthly-expenses 5200 \
          --swr 0.035 --annual-savings 28000 --real-return 0.05
```

### Target User Base
- Individuals: _checking DTI before applying for a mortgage, or runway before changing jobs_
- Households: _an annual net worth and savings rate review_
- Financial coaches and planners: _computing a client's baseline ratios in one command_
- FI/RE community: _the ratio set that defines the target and the timeline_
---

## Summary Table

| Command | Distribution / Concept | Deps (optional*) | Zero-dep fallback? | Closest existing tool | Issue |
|---|---|---|---|---|---|
| `hypergeo`     | Hypergeometric                               | None                               | N/A                | `binom`                   | #18 |
| `plotdist`     | Distribution visualiser                      | `matplotlib`, `numpy`*             | ✅ Unicode text    | `binom` / `birthday`     | #7  |
| `oddsconv`     | Odds format converter + vig calc             | None                               | N/A                | `expected`                | #19 |
| `sensitivity`  | Parameter sensitivity / tornado              | `matplotlib`*                      | ✅ ranked table    | all tools                 | #21 |
| `randforest`   | Random forest classifier / regressor         | `scikit-learn`, `numpy`*, `pandas`*| ✅ numpy/pandas    | `linreg`                  | #20 |
| `ewma`         | EWMA control chart + variance limits         | None                               | N/A                | `forecast`                | #22 |
| `vartest`      | Variance equality tests (F, Levene, Bartlett)| None                               | N/A                | `ttest`                   | #23 |
| `taylor`       | Taylor series approximation                  | None                               | N/A                | N/A                       | #25 |
| `compound`     | Compound interest & time value of money      | None                               | N/A                | `expected`                | #24 |
| `matrix`       | Matrix operations & linear algebra           | `numpy`*                           | ✅ nested lists    | `mlreg` (implemented)     | #26 |
| `fibonacci`    | Fibonacci sequence & golden ratio            | None                               | N/A                | N/A                       | #27 |
| `grover`       | Grover's quantum search algorithm simulator  | None                               | N/A                | `prime` / `fibonacci`     | #16 |
| `exponential`  | Exponential distribution                     | None                               | N/A                | `poisson` / `geometric`   | #31 |
| `describe`     | Descriptive statistics summary               | None                               | N/A                | all tools                 | #30 |
| `effect`       | Effect size (Cohen's d, eta², odds ratio)    | None                               | N/A                | `ttest` / `chisq`         | #33 |
| `combinatorics`| Permutations, combinations, counting         | None                               | N/A                | `hypergeo` / `binom`      | #35 |
| `liquidity`    | Current, quick, cash ratios & working capital| None                               | N/A                | `breakeven`               | #61 |
| `solvency`     | Leverage & coverage ratios (D/E, TIE, DSCR)  | None                               | N/A                | `breakeven` / `discount`  | #62 |
| `altman`       | Altman Z-score bankruptcy prediction         | None                               | N/A                | `logreg`                  | #63 |
| `profit`       | Margins, ROA/ROE/ROIC, EPS, DuPont           | None                               | N/A                | `breakeven`               | #64 |
| `turnover`     | Activity ratios & cash conversion cycle      | None                               | N/A                | `liquidity`               | #65 |
| `finhealth`    | Personal DTI, savings rate, emergency fund   | None                               | N/A                | `compound` / `discount`   | #66 |

\* _Optional dependency: functionality exists but reduced output capability without the package._
