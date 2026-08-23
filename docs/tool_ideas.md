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
| `life` | Life-in-weeks grid visualiser |
| `linreg` | Simple linear regression |
| `normal` | Gaussian PDF, CDF, and quantile |
| `pearson` | Pearson correlation coefficient |
| `poisson` | Poisson PMF, CDF, and survival |
| `prime` | Prime number tools and factorization |
| `pvalue` | p-value and hypothesis test calculator |
| `pythag` | Pythagorean win expectation |
| `sample` | Sample size calculator |
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

## 8. `mlreg` — Multiple Linear Regression with Prediction Intervals

### Dependencies
- **Required:** `numpy` (matrix algebra for OLS: $(X^TX)^{-1}X^Ty$)
- **Optional:** `pandas` (named-column CSV ingestion; falls back to `csv` module with positional columns)

### Architecture
- Fits OLS multiple regression and produces full inference output including individual and joint prediction intervals, driven entirely by user-supplied data
- **Core functions:** `fit(X, y)` → coefficients, SE, t-stats, p-values, R², adjusted R², F-stat; `predict(X_new, model, alpha)` → point estimate, confidence interval (mean response), prediction interval (individual response)
- CLI flags: `--file CSV`, `--target COL`, `--features COL [...]` (default: all non-target numeric columns), `--alpha F`, `--predict-file CSV`, `--vif` (variance inflation factors for multicollinearity), `--format {table,json,csv}`, `--precision INT`
- Output: coefficient table (estimate, SE, t, p, 95% CI), model summary (R², adjusted R², RMSE, F-stat, overall p), optional prediction table with PI bounds

```bash
# Fit a multiple regression from a CSV file
mlreg --file housing.csv --target price

# Include only selected features and compute VIF
mlreg --file data.csv --target sales --features advertising headcount --vif

# Predict new observations with 90% prediction intervals
mlreg --file train.csv --target output --predict-file new_inputs.csv --alpha 0.10 --format json
```

### Target User Base
- Analysts and data scientists: _who need a CLI multiple regression tool without opening a notebook or statistical package_
- Researchers: _reporting coefficient estimates with standard errors and prediction intervals_
- Engineers: _modeling a response variable (yield, latency, defect rate) as a function of multiple controllable inputs_
- Extends `linreg` to multiple predictors and adds the critical distinction between confidence intervals (mean response variance) and prediction intervals (individual response variance)

---

## 9. `taylor` — Taylor Series Approximation

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

## 10. `compound` — Compound Interest & Time Value of Money

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

## 11. `matrix` — Matrix Operations & Linear Algebra

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

## 12. `fibonacci` — Fibonacci Sequence & Golden Ratio

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

## 13. `logreg` — Logistic Regression

### Architecture
- **Core functions:** `fit(X, y)` → coefficients, SE, z-stats, p-values, log-likelihood; `predict_proba(X, coeffs)` → probability; `predict_class(X, coeffs, threshold)` → binary label; `log_odds(p)` → logit
- Gradient-descent or Newton-Raphson fitting via `math` — no required external dependencies; optional `numpy` for matrix operations on larger datasets
- CLI flags: `--file CSV`, `--target COL`, `--features COL [...]` (default: all non-target), `--threshold F` (classification cutoff, default 0.5), `--alpha F` (significance level), `--predict-file CSV`, `--format {table,json,csv}`, `--max-iter INT`, `--precision INT`
- Output: coefficient table (estimate, SE, z-stat, p-value, OR = exp(coeff)), model summary (log-likelihood, AIC, BIC, pseudo-R²), confusion matrix, accuracy/precision/recall/F1

### Application
Logistic regression is the workhorse binary classifier, directly modeling the probability of a binary outcome as a sigmoid function of linear predictors. Underpins medical diagnosis (disease yes/no), marketing (click/no-click), credit scoring (default/no-default), and any domain where the outcome is binary and interpretability matters. Coefficients as odds ratios make results directly communicable to non-technical stakeholders.

```bash
# Fit logistic regression from CSV; default threshold 0.5
logreg --file patient_data.csv --target disease

# Specify features and lower classification threshold (favor recall)
logreg --file churn.csv --target churned --features tenure spend logins --threshold 0.3

# Score new observations
logreg --file train.csv --target clicked --predict-file new_users.csv --format json
```

### Target User Base
- Data scientists and analysts: _baseline binary classifier before trying more complex models_
- Medical and public health researchers: _odds ratio estimation and risk factor analysis_
- Marketing and product analysts: _conversion and churn modeling_
- The "classification" counterpart to `linreg` — users who reach "my outcome is binary" will reach for `logreg` just as `linreg` users reach for it when the outcome is continuous

---

## 14. `grover` — Grover's Quantum Search Algorithm Simulator

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

## 15. `exponential` — Exponential Distribution Calculator

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

## 16. `describe` — Descriptive Statistics

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

## 17. `effect` — Effect Size Calculator

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

## 18. `combinatorics` — Permutations, Combinations & Counting

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

## 19. `slopeint` — Slope-Intercept Line Calculator

### Architecture
- **Core functions:** `slope(p1, p2)`, `line_from_points(p1, p2)`, `line_from_point_slope(point, m)`, `line_from_standard(a, b, c)`, `to_standard(m, b)`, `evaluate(m, b, x)`, `solve_for_x(m, b, y)`, `x_intercept(m, b)`, `y_intercept(m, b)`, `intersection(line1, line2)`, `is_parallel(m1, m2)`, `is_perpendicular(m1, m2)`, `perpendicular_slope(m)`, `distance_to_point(m, b, point)`, `angle_of_inclination(m)`
- Pure Python via `math` — no external dependencies
- CLI flags: `--points X1,Y1 X2,Y2`, `--slope F`, `--intercept F`, `--point X,Y`, `--standard A B C`, `--at F` (evaluate y at x), `--solve F` (solve x for y), `--intersect M,B`, `--perpendicular`, `--parallel`, `--distance`, `--table MIN MAX STEP`, `--format {table,json}`, `--precision INT`
- Output: the fitted equation in slope-intercept and standard form, both intercepts, angle of inclination; optional evaluation, intersection, perpendicular distance, or range table
- Edge cases handled explicitly: vertical line (`x1 == x2`) reports `x = c` rather than infinite slope; horizontal line (`m == 0`) raises on `--solve` instead of dividing by zero; `--intersect` distinguishes parallel from coincident; parallel/perpendicular predicates use float tolerance, not exact `==`

### Application
Lines in `y = mx + b` form are the most-used model in applied math, and the arithmetic around them — two-point construction, conversion to and from `Ax + By = C`, intersections, perpendicular projection — is exactly the kind of error-prone algebra worth a CLI. Distinct from `linreg`: `linreg` *estimates* a line from noisy sample data with standard errors and p-values, while `slopeint` manipulates an *exact*, known line. The two pair naturally — `linreg` produces `m` and `b`, `slopeint` consumes them for prediction, intersection, and projection. Break-even geometry is the same operation as intersecting a cost line with a revenue line, giving a cross-check against `breakeven`.

```bash
# Unit conversion: Fahrenheit from Celsius, y = 1.8x + 32
slopeint --points 0,32 100,212

# Straight-line depreciation: $30k asset, $5k salvage, 5-year life
slopeint --points 0,30000 5,5000 --table 0 5 1

# Cost model $50k fixed + $10/unit against revenue $25/unit — break-even as an intersection
slopeint --slope 10 --intercept 50000 --intersect 25,0

# Perpendicular line through a point, and the distance to it
slopeint --slope 2 --intercept 1 --point 4,3 --perpendicular
slopeint --slope 2 --intercept 1 --point 4,3 --distance
```

### Target User Base
- Students and educators: _working through algebra and precalculus where the line is given rather than fitted_
- Analysts: _evaluating, intersecting, or projecting a line already fitted by `linreg` without writing code_
- Engineers and lab technicians: _calibration curves, unit conversion, and quick projection math_
- Finance and operations users: _straight-line depreciation schedules and linear cost/revenue models, complementing `breakeven`_

---

## Summary Table

| Command | Distribution / Concept | Deps (optional*) | Zero-dep fallback? | Closest existing tool | Issue |
|---|---|---|---|---|---|
| `hypergeo`     | Hypergeometric                               | None                               | N/A                | `binom`                   | #18 |
| `plotdist`     | Distribution visualiser                      | `matplotlib`, `numpy`*             | ✅ Unicode text    | `binom` / `birthday`     | #7  |
| `oddsconv`     | Odds format converter + vig calc             | None                               | N/A                | `expected`                | #19 |
| `sensitivity`  | Parameter sensitivity / tornado              | `matplotlib`*                      | ✅ ranked table    | all tools                 | #20 |
| `randforest`   | Random forest classifier / regressor         | `scikit-learn`, `numpy`*, `pandas`*| ✅ numpy/pandas    | `linreg`                  | #21 |
| `ewma`         | EWMA control chart + variance limits         | None                               | N/A                | `forecast`                | #22 |
| `vartest`      | Variance equality tests (F, Levene, Bartlett)| None                               | N/A                | `ttest`                   | #23 |
| `mlreg`        | Multiple linear regression + pred. intervals | `numpy`, `pandas`*                 | N/A                | `linreg`                  | #3  |
| `taylor`       | Taylor series approximation                  | None                               | N/A                | N/A                       | #24 |
| `compound`     | Compound interest & time value of money      | None                               | N/A                | `expected`                | #25 |
| `matrix`       | Matrix operations & linear algebra           | `numpy`*                           | ✅ nested lists    | `mlreg`                   | #26 |
| `fibonacci`    | Fibonacci sequence & golden ratio            | None                               | N/A                | N/A                       | #27 |
| `logreg`       | Logistic regression (binary classifier)      | `numpy`*                           | ✅ gradient descent| `linreg`                  | #10 |
| `grover`       | Grover's quantum search algorithm simulator  | None                               | N/A                | `prime` / `fibonacci`     | #16 |
| `exponential`  | Exponential distribution                     | None                               | N/A                | `poisson` / `geometric`   | #31 |
| `describe`     | Descriptive statistics summary               | None                               | N/A                | all tools                 | #30 |
| `effect`       | Effect size (Cohen's d, eta², odds ratio)    | None                               | N/A                | `ttest` / `chisq`         | #33 |
| `combinatorics`| Permutations, combinations, counting         | None                               | N/A                | `hypergeo` / `binom`      | #35 |
| `slopeint`     | Slope-intercept line algebra & geometry      | None                               | N/A                | `linreg`                  | #55 |

\* _Optional dependency: functionality exists but reduced output capability without the package._
