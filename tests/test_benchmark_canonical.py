import numpy as np
import pytest

from benchmark.main import make_inputs, parse_args, validate_result


def test_inputs_are_full_slot_finite_complex_vectors():
    left, right, weights = make_inputs()
    assert left.shape == right.shape == weights.shape == (2**15,)
    assert left.dtype == right.dtype == np.complex128
    assert weights.dtype == np.float64
    assert np.isfinite(left).all()
    assert np.isfinite(right).all()
    assert np.isfinite(weights).all()


def test_cli_accepts_only_valid_run_counts():
    args = parse_args(["--warmup", "0", "--runs", "2"])
    assert args.warmup == 0
    assert args.runs == 2

    with pytest.raises(SystemExit):
        parse_args(["--warmup", "-1"])
    with pytest.raises(SystemExit):
        parse_args(["--runs", "0"])


def test_result_validation_fails_loudly():
    expected = np.asarray([1.0, 2.0])
    assert validate_result("add", expected, expected, tolerance=1e-9) == 0.0

    with pytest.raises(FloatingPointError):
        validate_result(
            "add",
            np.asarray([np.nan, 2.0]),
            expected,
            tolerance=1e-9,
        )
    with pytest.raises(AssertionError):
        validate_result(
            "add",
            np.asarray([1.1, 2.0]),
            expected,
            tolerance=1e-9,
        )
