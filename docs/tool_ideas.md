# Tool Ideas for `pythodds`

This document tracks **proposed tools not yet implemented** in `pythodds`. Each entry describes the proposed command name, its mathematical architecture, practical application, and target user base.

For the full list of implemented tools and their CLI entry points, see `README.md` and the `[project.scripts]` section in `pyproject.toml`.

---

## Implemented Tools (Reference)

| Command | Description |
|---------|-------------|
| `bayes` | Bayesian posterior update |
| `binom` | PMF, CDF, and survival function for Binomial(n, p) |
| `birthday` | Collision probability for uniform and non-uniform ID pools |
| `bootci` | Bootstrap confidence intervals |
| `collatz` | Collatz conjecture / hailstone sequences |
| `confint` | Confidence interval calculator |
| `crt` | Sunzi's Theorem (CRT) solver |
| `expected` | Expected value and variance for discrete distributions |
| `forecast` | Time series forecasting with prediction intervals |
| `gini` | Gini coefficient and Lorenz curve |
| `jevons` | Jevons paradox / rebound effect modeling |
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
| `ttest` | One- and two-sample t-tests |
| `euler` | Euler's number via limit/series, e^x, ln(x), identity, and γ constant |
| `zscore` | Z-score calculator |

---

## Proposed Tools

All tools below are pure-Python unless a `Dependencies` section is noted. Dependency-optional tools degrade gracefully to plain-text output when the library is not installed.

---

## 1. `chisq` — Chi-Square Test Calculator

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
- Survey analysts and market researchers: _testing whether response distributions fit expectations_
- Biologists and geneticists: _checking population allele frequency assumptions_
- A/B testers: _comparing multi-category outcome distributions between variants_
- Pairs naturally with `binom` (binary outcomes) and `pvalue` — the categorical-data generalisation of binary proportion tests

---

## 2. `hypergeo` — Hypergeometric Distribution Calculator

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

## 3. `plotdist` — Distribution Visualiser

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

## 4. `oddsconv` — Odds Format Converter

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

## 5. `sensitivity` — Parameter Sensitivity / Tornado Chart

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

## 6. `randforest` — Random Forest Classifier / Regressor

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

## 7. `ewma` — Exponentially Weighted Moving Average & Control Limits

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

## 8. `vartest` — Variance Equality Tests

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

## 9. `mlreg` — Multiple Linear Regression with Prediction Intervals

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

## 10. `taylor` — Taylor Series Approximation

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

## 11. `compound` — Compound Interest & Time Value of Money

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

## 12. `matrix` — Matrix Operations & Linear Algebra

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

## 13. `fibonacci` — Fibonacci Sequence & Golden Ratio

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

## 14. `logreg` — Logistic Regression

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

## 15. `breakeven` — Break-Even Analysis

### Architecture
- **Core functions:** `breakeven_units(fixed, price, variable)`, `breakeven_revenue(fixed, margin)`, `margin_of_safety(actual_units, breakeven_units)`, `target_profit_units(fixed, price, variable, profit)`
- Pure Python — no external dependencies; optional `matplotlib` for a revenue/cost curve chart
- CLI flags: `--fixed F` (total fixed costs), `--price F` (selling price per unit), `--variable F` (variable cost per unit), `--target-profit F` (optional: units needed for a profit target), `--margin` (compute contribution margin and ratio), `--sweep MIN MAX STEP` (table across price or unit range), `--chart` (text bar chart or matplotlib curve if available), `--format {table,json,csv}`, `--precision`
- Output: break-even units, break-even revenue, contribution margin, margin of safety; optional profit/loss table across unit range

### Application
Break-even analysis is the foundational tool for business viability assessment: at what volume does a product, project, or business become profitable? Used in startup planning, product launches, pricing decisions, event budgeting, and investment threshold analysis. The `--sweep` flag and optional chart make it especially actionable for scenario planning and presentations.

```bash
# Basic break-even: fixed costs $50k, price $25, variable cost $10 per unit
breakeven --fixed 50000 --price 25 --variable 10

# How many units to hit $20,000 profit?
breakeven --fixed 50000 --price 25 --variable 10 --target-profit 20000

# Sweep: profit/loss table from 0 to 8,000 units in steps of 500
breakeven --fixed 50000 --price 25 --variable 10 --sweep 0 8000 500
```

### Target User Base
- Entrepreneurs and small business owners: _validating whether a product can be profitable before launch_
- Finance and business analysts: _building pricing models and scenario analyses_
- Students: _learning managerial accounting and cost-volume-profit relationships_
- A practical "real-world math" companion to `compound` and `expected` — the most immediately applicable tool for anyone with a revenue model

---

## 16. `grover` — Grover's Quantum Search Algorithm Simulator

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

## 17. `anova` — One-Way Analysis of Variance

### Architecture
- **Core functions:** `anova_one_way(groups)` → F-stat, p-value, df_between, df_within, SS_between, SS_within, MS values; `tukey_hsd(groups)` → pairwise mean differences with adjusted p-values; `bonferroni(groups, alpha)` → corrected per-comparison α
- F-distribution CDF via regularised incomplete beta (`math.lgamma`) — no external dependencies
- CLI flags: `--data GROUP1 GROUP2 [...]` (comma-separated values per group), `--file CSV --group-col COL --value-col COL`, `--alpha F` (default 0.05), `--posthoc {tukey,bonferroni,none}`, `--format {table,json}`
- Output: ANOVA table (SS, df, MS, F, p), decision at α; post-hoc pairwise comparison table with adjusted p-values and significance indicators

```bash
# One-way ANOVA across three treatment groups
anova --data "12.1,11.8,12.5,11.9" "9.8,10.3,10.1,9.7" "15.2,14.9,15.5,16.0"

# With Tukey HSD post-hoc comparisons
anova --data "12.1,11.8,12.5" "9.8,10.3,10.1" "15.2,14.9,15.5" --posthoc tukey

# From CSV with group and value columns
anova --file experiment.csv --group-col treatment --value-col response --alpha 0.01
```

### Application
Tests whether the means of three or more independent groups are equal — the natural generalization of the two-sample t-test to multiple groups. Used in clinical trials (multiple treatment arms), product testing (A/B/C/D experiments), and any designed experiment where more than two conditions are compared simultaneously, avoiding the inflated Type I error of running multiple pairwise t-tests.

### Target User Base
- Researchers and statisticians: _comparing means across multiple experimental conditions_
- Product and UX analysts: _running multi-variant experiments with more than two arms_
- Students: _learning the F-distribution and the relationship between ANOVA and t-tests_
- The natural next step after `ttest` — users who outgrow two-group comparisons will reach for `anova` first

---

## 18. `geometric` — Geometric Distribution Calculator

### Architecture
- **Core functions:** `geo_pmf(k, p)`, `geo_cdf(k, p)`, `geo_survival(k, p)`, `geo_mean(p)`, `geo_variance(p)`
- Pure Python via `math` — no external dependencies
- CLI flags: `-k INT` (trial number), `-p F` (success probability per trial), `--survival` (return P(X > k) instead of CDF), `--table MIN MAX` (range of PMF/CDF values), `--precision INT`
- Output: PMF at k, CDF ≤ k, survival P(X > k), mean, variance

```bash
# P(first success on exactly the 5th trial, p=0.3)
geometric -k 5 -p 0.3

# P(needing more than 10 calls to close a sale with 20% close rate)
geometric -k 10 -p 0.2 --survival

# Range table: probability distribution for k = 1 to 15
geometric -p 0.25 --table 1 15
```

### Application
Models "how many independent trials until the first success?" — the discrete-time analog of the exponential distribution. Common in quality control (how many items tested until first defect), sales (calls until first close), reliability (components tested until first failure), and network retransmission modeling.

### Target User Base
- QA and reliability engineers: _modeling failure and defect detection rates_
- Sales and operations analysts: _estimating conversion trial counts_
- Students: _learning discrete distributions adjacent to `binom` and `poisson`_
- A natural sibling to `binom` and `poisson` — completing the classic trio of discrete distributions for count data

---

## 19. `exponential` — Exponential Distribution Calculator

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

## 20. `describe` — Descriptive Statistics

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

## 21. `entropy` — Information Entropy

### Architecture
- **Core functions:** `shannon_entropy(probs, base)` → bits (base 2) or nats (base e); `kl_divergence(p, q)` → KL(P‖Q); `cross_entropy(p, q)`; `mutual_information(joint)`; `conditional_entropy(joint)`
- Pure Python via `math.log` — no external dependencies
- CLI flags: `--probs CSV_OR_VALUES`, `--measure {entropy,kl,cross,mi,conditional}`, `--base {2,e,10}`, `--probs-p`, `--probs-q`, `--joint ROW [...]` (repeated flag for joint distribution rows), `--precision INT`, `--format {table,json}`
- Output: entropy/divergence value with units (bits/nats/hartleys), interpretation note for KL divergence direction

```bash
# Shannon entropy of a six-sided die (should be ~2.585 bits)
entropy --probs 0.167,0.167,0.167,0.167,0.167,0.167

# KL divergence between a biased coin (0.7/0.3) and a fair coin (0.5/0.5)
entropy --measure kl --probs-p 0.7,0.3 --probs-q 0.5,0.5

# Mutual information from a 2x2 joint distribution
entropy --measure mi --joint "0.25,0.25" --joint "0.25,0.25"
```

### Application
Shannon entropy quantifies the uncertainty or information content of a probability distribution — the fundamental measure of information theory. Used in data compression, ML model evaluation (cross-entropy loss), feature selection (information gain / mutual information), communications engineering, and any domain where "how much information does this tell me?" is the core question.

### Target User Base
- ML engineers and data scientists: _feature selection via mutual information, model evaluation via cross-entropy_
- Information theorists and communications engineers: _encoding efficiency and channel capacity analysis_
- Students: _learning information theory concepts complementary to probability distributions_
- A natural bridge from the probability tools to information-theoretic applications of those distributions

---

## 22. `effect` — Effect Size Calculator

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

## 23. `combinatorics` — Permutations, Combinations & Counting

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

## 24. `weibull` — Weibull Distribution Calculator

### Architecture
- **Core functions:** `weibull_pdf(x, k, lam)`, `weibull_cdf(x, k, lam)`, `weibull_survival(x, k, lam)`, `weibull_hazard(x, k, lam)`, `weibull_quantile(p, k, lam)`, `weibull_mean(k, lam)` (via `math.lgamma`)
- Pure Python — no external dependencies
- CLI flags: `-x F` (evaluation point), `-k F`/`--shape F` (shape parameter), `--lambda F`/`--scale F` (scale parameter), `--quantile F`, `--survival` (return 1 - CDF), `--table MIN MAX STEP`, `--precision INT`, `--format {table,json}`
- Output: PDF, CDF, survival probability, hazard rate at x; optional quantile or range table

### Application
The Weibull distribution is the standard model for component lifetimes and failure analysis. The shape parameter k controls failure behaviour: k < 1 (decreasing hazard — infant mortality / early failure), k = 1 (constant hazard = exponential distribution), k > 1 (increasing hazard — wear-out). Used in reliability engineering, warranty data analysis, survival analysis, and materials science. A direct extension of `exponential` for non-constant hazard rates.

```bash
# Survival probability at t=500 hours, shape=2, scale=1000
weibull -x 500 -k 2 --lambda 1000 --survival

# 5th percentile of failure time (when have 5% of units failed?)
weibull --quantile 0.05 -k 1.5 --lambda 800

# Range table: CDF and survival from 0 to 2000 in steps of 200
weibull -k 2.5 --lambda 1200 --table 0 2000 200
```

### Target User Base
- Reliability engineers and QA analysts: _modeling component lifetime distributions and warranty periods_
- Materials scientists and physicists: _fitting failure time data with non-constant hazard_
- Actuaries: _survival analysis beyond the constant-hazard exponential assumption_
- The natural follow-on to `exponential` — when "memoryless" is too simple and wear-out or burn-in effects matter

---

## Summary Table

| Command | Distribution / Concept | Deps (optional*) | Zero-dep fallback? | Closest existing tool | Issue |
|---|---|---|---|---|---|
| `chisq`        | Chi-square tests                             | None                               | N/A                | `pvalue`                  | #6  |
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
| `breakeven`    | Break-even analysis and cost-volume-profit   | `matplotlib`*                      | ✅ text table      | `compound` / `expected`   | #14 |
| `grover`       | Grover's quantum search algorithm simulator  | None                               | N/A                | `prime` / `fibonacci`     | #16 |
| `anova`        | One-way ANOVA + post-hoc tests               | None                               | N/A                | `ttest`                   | #28 |
| `geometric`    | Geometric distribution                       | None                               | N/A                | `binom` / `poisson`       | #29 |
| `exponential`  | Exponential distribution                     | None                               | N/A                | `poisson` / `geometric`   | #30 |
| `describe`     | Descriptive statistics summary               | None                               | N/A                | all tools                 | #31 |
| `entropy`      | Information entropy and KL divergence        | None                               | N/A                | `birthday`                | #32 |
| `effect`       | Effect size (Cohen's d, eta², odds ratio)    | None                               | N/A                | `ttest` / `chisq`         | #33 |
| `combinatorics`| Permutations, combinations, counting         | None                               | N/A                | `hypergeo` / `binom`      | #35 |
| `weibull`      | Weibull distribution (reliability/survival)  | None                               | N/A                | `exponential`             | #34 |

\* _Optional dependency: functionality exists but reduced output capability without the package._
