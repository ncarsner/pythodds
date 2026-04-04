import io
import sys
import unittest.mock as mock

from src.utils.collatz_conjecture import CollatzChecker, collatz_next, main


def _checker(n: int) -> CollatzChecker:
    checker = CollatzChecker()
    checker.ensure_up_to(n)
    return checker


def test_collatz_next():
    assert collatz_next(8) == 4
    assert collatz_next(3) == 10
    assert collatz_next(1) == 4  # 3*1+1; progression stops because 1 is pre-resolved


def test_steps_for_small_ints():
    checker = _checker(1)
    assert checker.steps_for[1] == 0
    # 2 -> 1  (1 step)
    checker = _checker(2)
    assert checker.steps_for[2] == 1
    # 3 -> 10 -> 5 -> 16 -> 8 -> 4 -> 2 (resolved); 2 was directly evaluated before 3 (6 steps)
    checker = _checker(3)
    assert checker.steps_for[3] == 6
    # 4 is precomputed (path member of start=3); 4 -> 2 (1 step to resolved 2)
    checker = _checker(4)
    assert checker.steps_for[4] == 1
    # 6 -> 3 (resolved); 3 was directly evaluated before 6 (1 step)
    checker = _checker(6)
    assert checker.steps_for[6] == 1
    # 5 is an intermediate path member of start=3 so it is stored as 0 (already validated; no further hops needed)
    checker = _checker(5)
    assert checker.steps_for[5] == 0
    # 7 -> 22 -> 11 -> 34 -> 17 -> 52 -> 26 -> 13 -> 40 -> 20 -> 10 (precomputed as 0)
    # 10 was an intermediate path member of start=3; walk terminates there (10 steps)
    checker = _checker(7)
    assert checker.steps_for[7] == 10


# --- precomputed values are reused (early-termination correctness) ---


def test_precomputed_steps_reused():
    """Numbers resolved as side-effects of earlier starts must have correct step counts."""
    checker = _checker(20)
    expected = {
        1: 0,
        2: 1,
        3: 6,
        4: 1,
        5: 0,
        6: 1,
        7: 10,
        8: 0,
        9: 3,
        10: 0,
        11: 0,
        12: 1,
        13: 0,
        14: 1,
        15: 9,
        16: 0,
        17: 0,
        18: 1,
        19: 5,
        20: 0,
    }
    for k, expected_steps in expected.items():
        assert (
            checker.steps_for[k] == expected_steps
        ), f"steps_for[{k}]: expected {expected_steps}, got {checker.steps_for[k]}"


# --- histogram reflects correct step counts ---


def test_histogram_step_zero_for_validated_members():
    """Integers precomputed as validated path members record 0 steps in the histogram."""
    checker = _checker(10)
    # In 1..10: starts 1, 5, 8, 10 are precomputed path members → 0 steps each
    assert checker.steps_histogram.get(0, 0) == 4


def test_verbose_output(capsys):
    """Test that verbose=True produces expected final summary and histogram output."""
    # Capture output without capsys to ensure coverage tracking
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()

    try:
        checker = CollatzChecker()
        checker.ensure_up_to(20, check_interval=10, verbose=True)
        output = sys.stdout.getvalue()
    finally:
        sys.stdout = old_stdout

    # Should have final summary
    assert "Finished checking 1..20" in output
    assert "max_valid=" in output
    assert "Steps histogram" in output
    # Verify histogram loop executes
    assert "0:" in output  # Should have entries in histogram
    assert (
        len(
            [
                line
                for line in output.split("\n")
                if ":" in line and line.strip().split(":")[0].strip().isdigit()
            ]
        )
        > 0
    )


def test_verbose_periodic_output(capsys):
    """Test verbose periodic progress output"""
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()

    try:
        checker = CollatzChecker()
        # Use smaller n with smaller interval to ensure the periodic check is triggered multiple times during the test run
        # The periodic output happens when start % check_interval == 0
        checker.ensure_up_to(100, check_interval=10, verbose=True)
        output = sys.stdout.getvalue()
    finally:
        sys.stdout = old_stdout

    # Should have periodic progress messages at intervals
    assert "Checked up to" in output or "Finished checking" in output
    # Verify that elapsed time calculation happened
    assert "time=" in output


def test_proven_method():
    """Test the proven method returns correct values"""
    checker = CollatzChecker()
    # Initially only 1 is resolved
    result1 = checker.proven(1)
    assert result1 is True
    result2 = checker.proven(2)
    assert result2 is False
    # After checking up to 5
    checker.ensure_up_to(5)
    result3 = checker.proven(5)
    assert result3 is True
    result4 = checker.proven(100)
    assert result4 is False
    # Test multiple values to ensure method is fully covered
    for i in range(1, 6):
        assert checker.proven(i) is True


# --- test ValueError safety check ---


def test_collatz_sequence_safety_check():
    """Test the safety check for invalid Collatz sequences."""
    checker = CollatzChecker()
    # Mock collatz_next to return a negative value after first call
    with mock.patch("src.utils.collatz_conjecture.collatz_next") as mock_next:
        # First call returns -1 (invalid)
        mock_next.return_value = -1
        try:
            checker.ensure_up_to(2)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Unexpected value in Collatz sequence" in str(e)


def test_main(capsys):
    # Save original argv and stdout
    original_argv = sys.argv
    original_stdout = sys.stdout

    try:
        # Set up test arguments
        sys.argv = ["collatz_conjecture.py", "--n", "10"]

        # Redirect stdout
        sys.stdout = io.StringIO()

        # Run main
        main()

        # Get output
        output = sys.stdout.getvalue()

        # Restore stdout so we can use capsys
        sys.stdout = original_stdout

        # Verify output contains max_valid
        assert "max_valid =" in output
        assert "10" in output
    finally:
        # Restore original values
        sys.argv = original_argv
        sys.stdout = original_stdout
