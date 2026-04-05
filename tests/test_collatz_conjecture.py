import io
import sys
import unittest.mock as mock

from src.utils.collatz_conjecture import (
    CollatzChecker,
    collatz_next,
    main,
    trace_collatz_sequence,
)


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


def test_trace_collatz_sequence_basic():
    """Test trace_collatz_sequence returns correct sequence and step count."""
    # Test with 1
    sequence, steps = trace_collatz_sequence(1, show_steps=False)
    assert sequence == [1]
    assert steps == 0

    # Test with 2
    sequence, steps = trace_collatz_sequence(2, show_steps=False)
    assert sequence == [2, 1]
    assert steps == 1

    # Test with 10
    sequence, steps = trace_collatz_sequence(10, show_steps=False)
    assert sequence == [10, 5, 16, 8, 4, 2, 1]
    assert steps == 6

    # Test with 3 (should have sequence 3 -> 10 -> 5 -> 16 -> 8 -> 4 -> 2 -> 1)
    sequence, steps = trace_collatz_sequence(3, show_steps=False)
    assert sequence == [3, 10, 5, 16, 8, 4, 2, 1]
    assert steps == 7


def test_trace_collatz_sequence_with_output():
    """Test that trace_collatz_sequence produces expected output when show_steps=True."""
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()

    try:
        sequence, steps = trace_collatz_sequence(10, show_steps=True)
        output = sys.stdout.getvalue()
    finally:
        sys.stdout = old_stdout

    # Verify output contains expected elements
    assert "Tracing Collatz sequence for 10:" in output
    assert "Step 0: 10" in output
    assert "Step 1: 10 → 5 (÷ 2)" in output
    assert "Step 2: 5 → 16 (× 3 + 1)" in output
    assert "Reached 1 in 6 steps" in output
    assert "Max value in sequence: 16" in output
    assert "Sequence length: 7" in output

    # Verify sequence and steps are correct
    assert sequence == [10, 5, 16, 8, 4, 2, 1]
    assert steps == 6


def test_main_no_arguments():
    """Test that main() returns error code 2 when no mode is specified."""
    original_argv = sys.argv
    original_stdout = sys.stdout

    try:
        sys.argv = ["collatz_conjecture.py"]
        sys.stdout = io.StringIO()

        result = main()
        output = sys.stdout.getvalue()

        sys.stdout = original_stdout

        assert result == 2
        assert "Error: Must specify one of --n, --trace, or --trace-range" in output
    finally:
        sys.argv = original_argv
        sys.stdout = original_stdout


def test_main_trace_mode_summary_only():
    """Test main() with --trace and --summary-only flags."""
    original_argv = sys.argv
    original_stdout = sys.stdout

    try:
        sys.argv = ["collatz_conjecture.py", "--trace", "27", "--summary-only"]
        sys.stdout = io.StringIO()

        result = main()
        output = sys.stdout.getvalue()

        sys.stdout = original_stdout

        assert result == 0
        assert "Number: 27" in output
        assert "Steps to reach 1: 111" in output
        assert "Max value in sequence: 9232" in output
        assert "Sequence length: 112" in output
        # Should NOT have step-by-step output
        assert "Step 1:" not in output
    finally:
        sys.argv = original_argv
        sys.stdout = original_stdout


def test_main_trace_mode_invalid():
    """Test main() with --trace flag and invalid (non-positive) number."""
    original_argv = sys.argv
    original_stdout = sys.stdout

    try:
        # Using -5 instead of 0 because 0 is falsy and gets caught by the mode validation first
        sys.argv = ["collatz_conjecture.py", "--trace", "-5"]
        sys.stdout = io.StringIO()

        result = main()
        output = sys.stdout.getvalue()

        sys.stdout = original_stdout

        assert result == 1
        assert "Error:" in output
        assert "Starting value must be positive" in output
    finally:
        sys.argv = original_argv
        sys.stdout = original_stdout


def test_main_trace_range_summary_only():
    """TODO: reduce assertions to just key summary elements and absence of step-by-step output."""
    """Test main() with --trace-range and --summary-only flags."""
    original_argv = sys.argv
    original_stdout = sys.stdout

    try:
        sys.argv = [
            "collatz_conjecture.py",
            "--trace-range",
            "5",
            "7",
            "--summary-only",
        ]
        sys.stdout = io.StringIO()

        result = main()
        output = sys.stdout.getvalue()

        sys.stdout = original_stdout

        assert result == 0
        assert "Tracing Collatz sequences for range [5, 7]:" in output
        assert "5: 5 steps, max=16, length=6" in output
        assert "6: 8 steps, max=16, length=9" in output
        assert "7: 16 steps, max=52, length=17" in output
        assert "Summary for range [5, 7]:" in output
        assert "Numbers tested: 3" in output
        # Should NOT have step-by-step output
        assert "Step 1:" not in output
    finally:
        sys.argv = original_argv
        sys.stdout = original_stdout


def test_main_trace_range_invalid_start():
    """Test main() with --trace-range flag and invalid start value."""
    original_argv = sys.argv
    original_stdout = sys.stdout

    try:
        sys.argv = ["collatz_conjecture.py", "--trace-range", "0", "5"]
        sys.stdout = io.StringIO()

        result = main()
        output = sys.stdout.getvalue()

        sys.stdout = original_stdout

        assert result == 1
        assert "Error: Range values must be positive integers" in output
    finally:
        sys.argv = original_argv
        sys.stdout = original_stdout


def test_main_trace_range_start_greater_than_end():
    """Test main() with --trace-range flag where START > END."""
    original_argv = sys.argv
    original_stdout = sys.stdout

    try:
        sys.argv = ["collatz_conjecture.py", "--trace-range", "10", "5"]
        sys.stdout = io.StringIO()

        result = main()
        output = sys.stdout.getvalue()

        sys.stdout = original_stdout

        assert result == 1
        assert "Error: START must be less than or equal to END" in output
    finally:
        sys.argv = original_argv
        sys.stdout = original_stdout
