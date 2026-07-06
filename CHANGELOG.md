# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
