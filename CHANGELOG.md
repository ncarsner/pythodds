# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.24.0] — 2026-08-23

### Added
- Multiple linear regression tool (`mlreg`) — OLS across several predictors with coefficient estimates, standard errors, t-statistics, p-values, and confidence intervals (#3)
  - Predictions carry both interval kinds: a confidence interval for the mean response and the always-wider prediction interval for a single new observation
  - Model fit reports R², adjusted R², residual standard error, and the overall F-test; `--vif` adds variance inflation factors, flagging any predictor above 10
  - The overall p-value is evaluated from the incomplete-beta tail directly instead of as `1 - cdf`, which cancels to exactly 0 for a strongly significant model where the true value is near 1e-35
  - Collinear or constant predictors and a constant response are rejected with a named error rather than producing unidentified coefficients or a negative F-statistic
  - Verified against numpy and scipy oracles: coefficients, standard errors, R², and both interval kinds to 1e-10 or better
- Binary logistic regression tool (`logreg`) — Newton-Raphson (IRLS) fit reporting coefficients as log-odds and odds ratios, with Wald standard errors, z-statistics, p-values, and confidence intervals (#10)
  - Model fit: log-likelihood against the intercept-only null, McFadden pseudo-R², AIC, and BIC; classification: confusion matrix with accuracy, precision, recall, and F1 at a configurable `--threshold`
  - Standard errors come from the exact inverse observed information rather than a quasi-Newton approximation, verified against a numerical Hessian to 1e-9 and the coefficients against a scipy optimiser to 1e-8
  - Degenerate fits raise instead of returning quietly: perfectly separable classes have no finite MLE and are reported as such, and a non-converged run raises rather than handing back the last iterate
  - `--predict-file` scores new observations; any two distinct target values are accepted, not just 0/1
- Slope-intercept line tool (`slopeint`) — construct a line from two points, point-slope, or standard form `Ax + By = C`, and report it in every representation (#55)
  - Standard form is normalised to primitive integer coefficients through exact rational arithmetic (`fractions.Fraction`), so `y = 1.8x + 32` reports as `9x - 5y = -160` rather than as rounded floats
  - Evaluates y at an x, solves x for a y, intersects a second line, projects a point onto the line with the perpendicular distance and the closest point, and emits the perpendicular or parallel line through a point
  - Degenerate cases are named rather than returned as infinities: a vertical line reports having no slope-intercept form, a horizontal line reports that solving for x has no answer, and parallel lines are distinguished from coincident ones
  - Complements `linreg`, which estimates a line from noisy data; `slopeint` operates on an exact, known one

### Fixed
> **Upgrade note:** `linreg` now reports different p-values and interval bounds than 0.23.0 did. The previous values were wrong, not merely less precise; confidence and prediction intervals in particular were materially too narrow. Any result carried forward from an earlier version should be recomputed.

- `linreg` distribution kernels were producing wrong p-values and interval bounds (issue #59)
  - `incomplete_beta` seeded its continued fraction at `d = 0`, dropping the leading term of the Lentz recurrence. It returned 0.2285 for `I_0.4(2,3)` against a true 0.5248, and the error reached every t- and F-based p-value in the module. Now seeded at `d = 1 / (1 - (a+b)x/(a+1))` and agrees with `scipy.special.betainc` to 1e-9
  - `t_cdf` substituted a normal approximation above `df = 30`, capping accuracy at ~3e-3 relative and putting a visible step in the reported p-value at the df 30/31 boundary. Now exact at every df
  - `inverse_t_cdf` used a third-order Cornish-Fisher expansion below `df = 30` and a normal quantile above it, off by 18% at df=5, p=0.975 and 29% at p=0.995. This is the t-critical value behind every confidence and prediction interval `linreg` prints, so those bounds were materially too narrow — at df=5 the 95% critical value was 1.9837 where the true value is 2.5706. Now obtained by bisecting the exact CDF, matching scipy to 1e-8
  - `f_cdf` was computed as `1 - I_x(...)`, and its caller then subtracted that from 1 again, losing significant digits twice. Now evaluated in direct form
  - Added scipy oracle tests across all four kernels. The previous tests asserted only edge cases and the *shape* of the large-df shortcut, which is why a continued fraction seeded incorrectly passed everything

### Removed
- `standard_normal_cdf` and `inverse_normal_cdf` from `src.utils.linear_regression`, unused once the normal-approximation shortcuts were dropped. Neither was part of the module's documented surface; equivalents remain in `pearson_correlation` and `normal_gaussian`

---

## [0.23.0] — 2026-08-18

### Added
- Break-even analysis tool (`breakeven`) — cost-volume-profit analysis reporting break-even units and revenue, contribution margin and margin ratio, target-profit volume, and margin of safety (#14)
  - `--sweep MIN MAX STEP` appends a profit/loss table across a unit range; `--chart` renders that column as a text bar chart with losses left of the break-even axis and profits right of it, so the crossing is visible without a plotting dependency
  - `--format` supports `table`, `json`, and `csv`
- Information entropy tool (`entropy`) — Shannon entropy, KL divergence, cross-entropy, mutual information, and conditional entropy in bits, nats, or hartleys (#32)
  - Input vectors are normalized to sum to 1, so raw counts are accepted alongside probabilities
  - The entropy report names the maximum entropy for that number of outcomes and the resulting efficiency; the divergence report adds the reverse KL, since D(P‖Q) is asymmetric, and reports it as infinite rather than failing when the reverse direction is undefined
- Weibull distribution tool (`weibull`) — PDF, CDF, survival, hazard, quantile, mean, median, and variance for Weibull(k, λ) (#34)
  - Names the failure mode implied by the shape parameter: infant mortality (k<1), constant hazard (k=1), or wear-out (k>1)
  - The hazard rate uses the closed form rather than f(x)/S(x), so it stays finite in the far tail where that ratio would divide zero by zero; moments go through `math.lgamma` so large parameters do not overflow
- Discount rate tool (`discount`) — real and nominal rates via the Fisher equation, discount factors, present value of a lump sum, and nominal and inflation-adjusted NPV (#42)
  - Supplying either `--nominal` or `--real` (the latter with `--inflation`) derives the other
  - Reports the discounted payback period, interpolated within the period where cumulative discounted cash flow crosses zero

---

## [0.22.0] — 2026-08-07

### Added
- Life in weeks tool (`life`) — renders a human life as an ANSI grid of weeks, months, or years against a nominal 90-year lifespan, color-coded elapsed vs remaining (#46, #48)
  - Cells are discrete shape glyphs (`■` lived, `□` remaining) grouped into quarters with a blank line between decades, so individual units stay countable instead of merging into a solid bar
  - `--reference` renders "The Life of a Typical American": the same grid shaded by life phase, each phase with its own silhouette (`● ▲ ■ ◆ ○`) and `★` milestone markers, from cited US averages (Census, CDC/NCHS, Gallup)
  - `--as-of` accepts an explicit date so output is deterministic and what-if dates can be projected; `--group` overrides the per-mode group size; `--lifespan`, `--mode`, `--no-color`, and `--format json` round out the interface
  - Honors the `NO_COLOR` environment variable and disables color automatically when stdout is not a TTY

### Changed
- Test suite runtime cut from 8.81s to 2.75s across 1288 tests by switching the coverage backend to `sys.monitoring` (`core = "sysmon"`), since `--cov` runs on every invocation via `addopts` (#50)
- ruff pre-commit hook updated `v0.6.4` → `v0.16.1`, and `id: ruff` → `id: ruff-check` (the bare id is a deprecated alias as of 0.16)
- Lint rule set now pinned explicitly as `select = ["E4", "E7", "E9", "F", "I"]` in `pyproject.toml`, so a ruff upgrade cannot silently widen what is enforced
- Dependency updates gated behind a 72-hour cooling-off period via a new `renovate.json` (`minimumReleaseAge`, `internalChecksFilter: "strict"`) as a supply-chain safeguard
- `_use_color` in `life_in_weeks` now types its `stream` parameter as `object`, matching the `getattr`-with-fallback TTY check that accepts any value

### Removed
- Test-only `_test_force_tiny` parameter from `incomplete_beta` in `linear_regression`; the continued-fraction floor is now the module constant `_TINY`, reachable from tests by monkeypatch without a flag in the production signature
- Unreachable numpy fallback branches and the always-true `HAS_NUMPY` constant from `monte_carlo`, along with the `pragma: no cover` directives that had been holding the dead code at 100% coverage

### Fixed
- `test_incomplete_beta_lentz_floor_guards_hit` asserted only that the result was finite, which held whether or not the four Lentz guards existed — the test passed with all four deleted. It now pins the clamped value, so any change to a guard fails it
- Replaced a `try/except: pass` test in `linear_regression` that could not fail with one asserting finite, in-range results for subnormal parameters
- Type-narrowing errors at 41 `validate()` assertion sites across six test files, where `str | None` was passed directly to the `in` operator

---

## [0.21.0] — 2026-07-05

### Added
- Geometric distribution tool (`geometric`) — PMF, CDF, survival, mean, and variance for the number of trials until the first success
- Chi-square test calculator (`chisq`) — goodness-of-fit and independence (contingency table) tests with per-cell contributions; exact CDF via a regularised incomplete gamma function
- One-way ANOVA tool (`anova`) — F-test via a regularised incomplete beta function, with Tukey HSD and Bonferroni-corrected pairwise post-hoc comparisons; Tukey HSD uses the studentized range distribution computed via nested numerical integration

---

## [0.20.0] — 2026-06-30

### Added
- Subnet mask calculator (`subnet`) — network address, broadcast address, first/last usable IP, subnet mask, host count, and classful network count from an IPv4 address with optional CIDR notation

---

## [0.19.1] — 2026-06-17

### Changed
- Improved `crt --help` output: expanded description shows the full congruence system; `--solve` metavar changed to `A N` and help text explicitly names the remainder (residue) and modulus (divisor) in each pair; epilog examples annotated with `remainder=` / `mod=` labels and expected output
- Updated README `crt` section: `--solve` option row names each position; added blockquote clarifying pair order with minimal inline examples

---

## [0.19.0] — 2026-06-12

### Added
- Sigmoid function tool (`sigmoid`) — σ(x), derivative, inverse logit, and Unicode sparkline
- Euler's number tool (`euler`) — limit convergence, e^x Taylor series, ln(x) series, Euler's identity, and Euler-Mascheroni constant
- Gini coefficient tool (`gini`) — inequality measurement from raw data, weighted samples, grouped shares, Lorenz curve, and multi-dataset comparison
- Sunzi's Theorem tool (`crt`) — simultaneous congruence solver supporting coprime and non-coprime moduli

---

## [0.18.0] — 2026-05-11

### Added
- Jevons Paradox model (`jevons`) — efficiency-induced demand rebound simulation
- T-test tool (`ttest`) — one-sample and two-sample hypothesis testing
- Monte Carlo simulator expanded to 9 experiments: added `power`, `permutation`, `bayes`, `season`, and `linboot`

---

## [0.17.0] — 2026-05-03

### Added
- Z-score calculator (`zscore`) — standardized score with optional lookup

### Removed
- Cleaned up unused dependencies

---

## [0.16.0] — 2026-04-19

### Added
- Confidence intervals tool (`confint`) — mean confidence intervals from sample data
- P-value tool (`pvalue`) — one- and two-tailed probability values

---

## [0.15.2] — 2026-04-05

### Fixed
- Patch version bump following Collatz refinements

## [0.15.1] — 2026-04-05

### Changed
- Collatz conjecture (`collatz`) updated to include step-by-step sequence tracing
- Refined associated tests

## [0.15.0] — 2026-04-04

### Added
- Bootstrap confidence intervals (`bootci`) — resampling-based interval estimation
- Collatz conjecture (`collatz`) — sequence length and path visualization
- Time series forecaster (`forecast`) — trend/seasonal decomposition and projection

---

## [0.14.0] — 2026-03-28

### Added
- Bootstrap confidence intervals (`bootci`) — initial implementation

---

## [0.13.0] — 2026-03-26

### Added
- Prime number utilities (`prime`) — primality test, factorization, nth prime

---

## [0.12.0] — 2026-03-22

### Added
- Spearman rank correlation (`spearman`) — non-parametric monotonic relationship test

---

## [0.11.0] — 2026-03-21

### Added
- Sample size calculator (`sample`) — minimum n for margin of error / power targets

---

## [0.10.0] — 2026-03-19

### Added
- Linear regression (`linreg`) — OLS with slope, intercept, and R² output

---

## [0.9.1] — 2026-03-17

### Fixed
- Resolved integer overflow error in probability calculations; updated tests and docs

## [0.9.0] — 2026-03-16

### Added
- Pearson correlation coefficient (`pearson`) — linear relationship strength and direction

---

## [0.8.0] — 2026-03-15

### Added
- Pythagorean win record (`pythag`) — expected W-L from runs/points scored and allowed

---

## [0.7.3] — 2026-03-14

### Changed
- Minor internal updates

## [0.7.2] — 2026-03-14

### Changed
- Added Codecov token to CI workflow

## [0.7.1] — 2026-03-14

### Fixed
- Patch version bump

## [0.7.0] — 2026-03-14

### Added
- Bayes theorem calculator (`bayes`) — prior/likelihood/posterior probability
- Binomial distribution bar chart output

---

## [0.6.0] — 2026-03-12

### Added
- Expected value calculator (`expected`) — weighted outcome average
- Normal distribution tool (`normal`) — PDF/CDF with z-score lookup

---

## [0.5.2] — 2026-03-11

### Changed
- Added code coverage reporting (pytest-cov + Codecov integration)

## [0.5.1] — 2026-03-11

### Changed
- Installed pre-commit hooks; minor project maintenance

## [0.5.0] — 2026-03-10

### Added
- Monte Carlo simulator (`simulate`) — initial support for `binomial`, `birthday`, `streak`, and `poisson` experiments
- Streak probability tool (`streak`) — consecutive event probability over N trials

---

## [0.4.1] — 2026-03-08

### Fixed
- Version bump patch

## [0.4.0] — 2026-03-08

### Added
- Poisson distribution tool (`poisson`) — event rate / arrival probability

---

## [0.3.1] — 2026-03-05

### Changed
- README updates

## [0.3.0] — 2026-03-05

### Added
- Birthday problem tool (`birthday`) — collision probability for a group of size n

---

## [0.2.1] — 2026-03-05

### Changed
- README updates

## [0.2.0] — 2026-03-04

### Added
- Binomial distribution tool (`binom`) — n/k/p probability calculator with optional chart

---

## [0.1.8] — 2026-03-04

### Fixed
- CI publish workflow: use `--token` flag instead of `--username`/`--password`

## [0.1.7] — 2026-03-04

### Fixed
- Version bump

## [0.1.6] — 2026-03-04

### Changed
- Stabilized CI workflows; prepared PyPI publish pipeline

## [0.1.5] — 2026-03-04

### Fixed
- GitHub Actions: replaced `uv python install` with standard `setup-python` action

## [0.1.4] — 2026-03-04

### Fixed
- Publish workflow corrected for PyPI upload URL

## [0.1.3] — 2026-03-04

### Fixed
- Additional publish workflow corrections

## [0.1.2] — 2026-03-04

### Fixed
- GitHub Actions: corrected pytest configuration in test workflow

## [0.1.1] — 2026-03-04

### Fixed
- Entry point and import path corrections for installed package

## [0.1.0] — 2026-03-04

### Added
- Initial release: project scaffold, GitHub Actions CI/CD, PyPI packaging via Hatchling
- `uv` as package manager and build tool
