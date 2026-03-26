#!/usr/bin/env python3
from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from src.utils.prime_numbers import (
    format_factorization,
    is_prime,
    main,
    nth_prime,
    prime_factorization,
    primes_in_range,
    sieve_of_eratosthenes,
)

# ---------------------------------------------------------------------------
# Tests for is_prime
# ---------------------------------------------------------------------------


def test_is_prime_small_composites():
    """Test primality check for small composite numbers."""
    composites = [4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 22, 24, 25]
    for c in composites:
        assert not is_prime(c), f"{c} should not be prime"


# ---------------------------------------------------------------------------
# Tests for nth_prime
# ---------------------------------------------------------------------------


def test_nth_prime_first_ten():
    """Test finding the first ten prime numbers."""
    expected = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    for i, expected_prime in enumerate(expected, start=1):
        assert nth_prime(i) == expected_prime


def test_nth_prime_invalid():
    """Test that nth_prime raises ValueError for invalid input."""
    with pytest.raises(ValueError):
        nth_prime(0)
    with pytest.raises(ValueError):
        nth_prime(-1)


# ---------------------------------------------------------------------------
# Tests for sieve_of_eratosthenes
# ---------------------------------------------------------------------------


def test_sieve_small_limits():
    """Test sieve for small limits."""
    assert sieve_of_eratosthenes(0) == []
    assert sieve_of_eratosthenes(1) == []
    assert sieve_of_eratosthenes(2) == [2]
    assert sieve_of_eratosthenes(10) == [2, 3, 5, 7]
    assert sieve_of_eratosthenes(20) == [2, 3, 5, 7, 11, 13, 17, 19]


# ---------------------------------------------------------------------------
# Tests for primes_in_range
# ---------------------------------------------------------------------------


def test_primes_in_range_edge_cases():
    """Test edge cases for primes_in_range."""
    # Range starting below 2
    assert primes_in_range(0, 5) == [2, 3, 5]
    assert primes_in_range(-10, 5) == [2, 3, 5]

    # Empty range (start > end)
    assert primes_in_range(10, 5) == []

    # Range with only 2
    assert primes_in_range(2, 2) == [2]


# ---------------------------------------------------------------------------
# Tests for prime_factorization
# ---------------------------------------------------------------------------


def test_prime_factorization_invalid():
    """Test that factorization raises ValueError for invalid input."""
    with pytest.raises(ValueError):
        prime_factorization(0)
    with pytest.raises(ValueError):
        prime_factorization(1)
    with pytest.raises(ValueError):
        prime_factorization(-5)


# ---------------------------------------------------------------------------
# Tests for format_factorization
# ---------------------------------------------------------------------------


def test_format_factorization_empty():
    """Test formatting of empty factorization."""
    assert format_factorization({}) == "1"


# ---------------------------------------------------------------------------
# Tests for main CLI
# ---------------------------------------------------------------------------


def test_main_check_edge_cases():
    """Test --check flag with edge cases."""
    assert main(["--check", "0"]) == 0
    assert main(["--check", "1"]) == 0
    assert main(["--check", "2"]) == 0


def test_main_check_negative():
    """Test --check flag with negative number."""
    assert main(["--check", "-5"]) == 2


def test_main_nth_invalid():
    """Test --nth flag with invalid input."""
    assert main(["--nth", "0"]) == 2
    assert main(["--nth", "-1"]) == 2


def test_main_count():
    """Test --count flag."""
    assert main(["--count", "0"]) == 0
    assert main(["--count", "10"]) == 0
    assert main(["--count", "100"]) == 0


def test_main_count_negative():
    """Test --count flag with negative limit."""
    assert main(["--count", "-10"]) == 2


def test_main_range_invalid():
    """Test --range flag with invalid input."""
    assert main(["--range", "-5", "10"]) == 2
    assert main(["--range", "20", "10"]) == 2


def test_main_factorize_invalid():
    """Test --factorize flag with invalid input."""
    assert main(["--factorize", "0"]) == 2
    assert main(["--factorize", "1"]) == 2


# ---------------------------------------------------------------------------
# Integration tests with output validation
# ---------------------------------------------------------------------------


def test_check_json_output(capsys):
    """Test --check JSON output."""
    main(["--check", "7", "--format", "json"])
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["n"] == 7
    assert data["is_prime"] is True


def test_nth_output(capsys):
    """Test --nth output."""
    main(["--nth", "10"])
    captured = capsys.readouterr()
    assert "10th prime number is 29" in captured.out


def test_nth_json_output(capsys):
    """Test --nth JSON output."""
    main(["--nth", "10", "--format", "json"])
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["n"] == 10
    assert data["nth_prime"] == 29


def test_count_json_output(capsys):
    """Test --count JSON output."""
    main(["--count", "10", "--format", "json"])
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["limit"] == 10
    assert data["prime_count"] == 4


def test_range_output(capsys):
    """Test --range output."""
    main(["--range", "10", "20"])
    captured = capsys.readouterr()
    assert "11" in captured.out
    assert "13" in captured.out
    assert "17" in captured.out
    assert "19" in captured.out


def test_range_json_output(capsys):
    """Test --range JSON output."""
    main(["--range", "10", "20", "--format", "json"])
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["start"] == 10
    assert data["end"] == 20
    assert data["primes"] == [11, 13, 17, 19]


def test_factorize_output(capsys):
    """Test --factorize output."""
    main(["--factorize", "12"])
    captured = capsys.readouterr()
    assert "12" in captured.out
    assert "2²" in captured.out or "2^2" in captured.out


def test_factorize_json_output(capsys):
    """Test --factorize JSON output."""
    main(["--factorize", "12", "--format", "json"])
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["n"] == 12
    assert data["factors"] == {"2": 2, "3": 1}


# ---------------------------------------------------------------------------
# Performance/stress tests
# ---------------------------------------------------------------------------


def test_large_range():
    """Test finding primes in a large range."""
    primes = primes_in_range(10000, 10100)
    assert len(primes) > 0
    assert all(is_prime(p) for p in primes)


def test_large_factorization():
    """Test factorization of a large number."""
    # 2^10 * 3^5 = 1024 * 243 = 248832
    factors = prime_factorization(248832)
    assert factors == {2: 10, 3: 5}


# ---------------------------------------------------------------------------
# Coverage tests for edge cases
# ---------------------------------------------------------------------------


def test_nth_prime_requires_estimate_expansion():
    """Test that nth_prime expands estimate when needed (covers line 81)."""
    # Mock sieve_of_eratosthenes to return fewer primes initially,
    # forcing the estimate expansion loop to execute
    original_sieve = sieve_of_eratosthenes
    call_count = [0]

    def mock_sieve(limit):
        call_count[0] += 1
        if call_count[0] == 1:
            # First call: return only 5 primes (not enough for 10th)
            return [2, 3, 5, 7, 11]
        else:
            # Subsequent calls: return actual sieve result
            return original_sieve(limit)

    with patch("src.utils.prime_numbers.sieve_of_eratosthenes", side_effect=mock_sieve):
        result = nth_prime(10)
        assert result == 29
        assert call_count[0] >= 2  # Should have called sieve multiple times


def test_main_value_error_handling(capsys):
    """Test ValueError exception handling in main (covers lines 356-357)."""
    # Mock a function to raise ValueError after validation passes
    with patch(
        "src.utils.prime_numbers.is_prime", side_effect=ValueError("Test error")
    ):
        exit_code = main(["--check", "7"])
        assert exit_code == 2
        captured = capsys.readouterr()
        assert "Error: Test error" in captured.err


def test_main_keyboard_interrupt_handling(capsys):
    """Test KeyboardInterrupt handling in main (covers lines 359-360)."""
    # Mock a function to raise KeyboardInterrupt
    with patch("src.utils.prime_numbers.is_prime", side_effect=KeyboardInterrupt()):
        exit_code = main(["--check", "7"])
        assert exit_code == 130
        captured = capsys.readouterr()
        assert "Interrupted" in captured.err
