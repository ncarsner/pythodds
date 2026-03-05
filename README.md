# pythodds

[![PyPI version](https://badge.fury.io/py/pythodds.svg)](https://pypi.org/project/pythodds/)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/yourusername/pythodds/blob/main/LICENSE)

A command-line utility and Python library for calculating statistics, odds, and probabilities.

## Features

- **Binomial Distribution**: Calculate PMF, CDF, and survival functions for binomial distributions
- **Command-line Interface**: Easy-to-use CLI tool (`binom` command)
- **Pure Python**: No external dependencies

## Installation

Install from PyPI:

```bash
pip install pythodds
```

Or install from source:

```bash
git clone https://github.com/yourusername/pythodds.git
cd pythodds
pip install -e .
```

## Usage

### Command Line

```bash
# Calculate binomial distribution probabilities
binom -n 10 -k 3 -p 0.4

# Specify a target and minimum probability threshold
binom -n 100 -k 30 -p 0.35 --target 40 --min-prob 0.05
```

### Python Library

```python
from utils.binomial_distribution import binomial_pmf, binomial_cdf_le, binomial_cdf_ge

# P(X = 3) for Binomial(n=10, p=0.4)
pmf = binomial_pmf(10, 3, 0.4)

# P(X <= 3) for Binomial(n=10, p=0.4)
cdf = binomial_cdf_le(10, 3, 0.4)

# P(X >= 3) for Binomial(n=10, p=0.4)
survival = binomial_cdf_ge(10, 3, 0.4)
```

## Development

Clone the repository and install in editable mode:

```bash
git clone https://github.com/yourusername/pythodds.git
cd pythodds
pip install -e .
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

Your Name - [@yourusername](https://github.com/yourusername)
