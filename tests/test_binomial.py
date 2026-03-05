"""Tests for binomial distribution functions."""

import pytest
from utils.binomial_distribution import binomial_pmf, binomial_cdf_le, binomial_cdf_ge


def test_binomial_pmf_basic():
    """Test PMF with simple values."""
    # P(X=0) for Binomial(n=1, p=0.5) should be 0.5
    result = binomial_pmf(1, 0, 0.5)
    assert abs(result - 0.5) < 1e-10
    
    # P(X=1) for Binomial(n=1, p=0.5) should be 0.5
    result = binomial_pmf(1, 1, 0.5)
    assert abs(result - 0.5) < 1e-10


def test_binomial_pmf_boundary():
    """Test PMF boundary conditions."""
    # P(X=k) when k < 0 should be 0
    assert binomial_pmf(10, -1, 0.5) == 0.0
    
    # P(X=k) when k > n should be 0
    assert binomial_pmf(10, 11, 0.5) == 0.0


def test_binomial_cdf_le():
    """Test CDF (less than or equal)."""
    # P(X <= 0) for Binomial(n=1, p=0.5) should be 0.5
    result = binomial_cdf_le(1, 0, 0.5)
    assert abs(result - 0.5) < 1e-10
    
    # P(X <= n) should be 1.0
    assert binomial_cdf_le(10, 10, 0.5) == 1.0
    
    # P(X <= k) when k < 0 should be 0
    assert binomial_cdf_le(10, -1, 0.5) == 0.0


def test_binomial_cdf_ge():
    """Test survival function (greater than or equal)."""
    # P(X >= 1) for Binomial(n=1, p=0.5) should be 0.5
    result = binomial_cdf_ge(1, 1, 0.5)
    assert abs(result - 0.5) < 1e-10
    
    # P(X >= 0) should be 1.0
    assert binomial_cdf_ge(10, 0, 0.5) == 1.0
    
    # P(X >= k) when k > n should be 0
    assert binomial_cdf_ge(10, 11, 0.5) == 0.0


def test_complementary_probabilities():
    """Test that CDF and survival function are complementary."""
    n, p = 10, 0.4
    for k in range(n + 1):
        cdf = binomial_cdf_le(n, k, p)
        sf = binomial_cdf_ge(n, k + 1, p)
        # cdf + sf should equal 1.0
        assert abs(cdf + sf - 1.0) < 1e-10
