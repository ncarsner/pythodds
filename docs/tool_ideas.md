---

## Extended Tool Ideas (Dependency-Optional, Dynamic Input)

The tools below may introduce optional or required third-party dependencies — primarily for visualisation (`matplotlib`, `rich`) or numerical computation (`numpy`). Where dependencies are optional, tools degrade gracefully to plain-text output when the library is not installed. All tools continue to accept dynamic user-supplied variables to scale computation to real input.

---

## 7. `plotdist` — Distribution Visualiser

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

## 8. `simulate` — Monte Carlo Probability Simulator

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

## 9. `oddsconv` — Odds Format Converter

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

## 10. `confint` — Confidence Interval Calculator

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

## 11. `pvalue` — p-value and Hypothesis Test Calculator

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

## 12. `sensitivity` — Parameter Sensitivity / Tornado Chart

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

## Updated Summary Table

| Command       | Distribution / Concept               | Deps (optional*)          | Zero-dep fallback? | Closest existing tool         |
|---------------|--------------------------------------|---------------------------|--------------------|-------------------------------|
| `poisson`     | Poisson                              | None                      | N/A                | `binom`                       |
| `normal`      | Normal / Gaussian                    | None                      | N/A                | `binom`                       |
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

\* Optional dependency: tool functions without it but with reduced output capability.