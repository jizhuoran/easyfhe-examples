import numpy as np
import pytest

from bootstrap.main import make_values, parse_args


def test_full_slot_input_has_finite_real_and_imaginary_data():
    values = make_values(2**15)

    assert values.shape == (2**15,)
    assert values.dtype == np.complex128
    assert np.isfinite(values).all()
    assert np.all(values.real != 0.0)
    assert np.all(values.imag != 0.0)


def test_cli_only_accepts_nonnegative_warmup_and_positive_runs():
    args = parse_args(["--warmup", "0", "--runs", "2"])
    assert args.warmup == 0
    assert args.runs == 2

    with pytest.raises(SystemExit):
        parse_args(["--warmup", "-1"])
    with pytest.raises(SystemExit):
        parse_args(["--runs", "0"])
