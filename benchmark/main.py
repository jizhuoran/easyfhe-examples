"""Benchmark representative public EasyFHE u64 operations on CUDA."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

import easyfhe
import easyfhe.bs.openfhe as bs
import easyfhe.fhe as fhe


LOG_N = 16
SLOTS = 2**15
WORK_LIMBS = 6
BOOTSTRAP_INPUT_LIMBS = 3
SECRET_KEY_DIST = "SPARSE_TERNARY"


@dataclass(frozen=True)
class Operation:
    name: str
    run: Callable[[], object]
    decode: Callable[[object], np.ndarray]
    expected: np.ndarray
    tolerance: float


@dataclass(frozen=True)
class Result:
    name: str
    average_ms: float
    minimum_ms: float
    maximum_ms: float
    max_error: float


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--runs", type=int, default=10)
    args = parser.parse_args(argv)
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.runs <= 0:
        parser.error("--runs must be positive")
    return args


def make_inputs(slots: int = SLOTS) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    angle = 2.0 * np.pi * np.arange(slots, dtype=np.float64) / slots
    left = 0.03 + 0.01 * np.sin(angle) + 1j * (0.02 + 0.005 * np.cos(angle))
    right = 0.4 + 0.1 * np.cos(2 * angle) + 1j * 0.05 * np.sin(angle)
    weights = 0.5 + 0.1 * np.sin(3 * angle)
    return (
        left.astype(np.complex128),
        right.astype(np.complex128),
        weights.astype(np.float64),
    )


def bootstrap_spec() -> bs.BootstrapSpec:
    return bs.BootstrapSpec(
        log_slots=15,
        level_budget=(4, 4),
        output_levels=2,
        strategy="normal_giant",
        mode="modraise_first",
    )


def create_runtime():
    spec = bootstrap_spec()
    requirements = bs.requirements(
        spec,
        log_n=LOG_N,
        secret_key_dist=SECRET_KEY_DIST,
    )
    rotations = tuple(dict.fromkeys((1, *requirements.rotations)))
    client, context = fhe.generate_client_context(
        fhe.CKKSContextSpec(
            depth=requirements.context_depth,
            log_n=LOG_N,
            dnum=3,
            dcrt_bits=59,
            first_mod=60,
            secret_key_dist=SECRET_KEY_DIST,
            scale_mode="fixed",
            rescale_policy="manual",
            rotations=rotations,
        ),
        device="cuda",
    )
    return client, context, bs.generate(context, spec), requirements


def build_operations(client, context, program) -> tuple[Operation, ...]:
    left, right, weights = make_inputs()
    left_cipher = client.encrypt(left, slots=SLOTS, cur_limbs=WORK_LIMBS)
    right_cipher = client.encrypt(right, slots=SLOTS, cur_limbs=WORK_LIMBS)
    bootstrap_cipher = client.encrypt(
        left,
        slots=SLOTS,
        cur_limbs=BOOTSTRAP_INPUT_LIMBS,
    )

    # This is the public on-GPU path for already slot-ready constants.
    packed_weights = easyfhe.as_tensor(
        weights,
        dtype=easyfhe.float64,
        device="cuda",
    )
    constants = fhe.ConstantBundle(
        vectors={"weights": fhe.PackedRaw(packed_weights)},
        cache_mode="middle",
    )
    plaintext = constants.plaintext(
        "weights",
        state=left_cipher.state,
        slots=SLOTS,
        context=context,
    )

    decrypt_cipher = lambda value: _to_numpy(
        client.decrypt(value, complex_output=True)
    )[:SLOTS]
    decode_values = lambda value: _to_numpy(value)[:SLOTS]
    return (
        Operation(
            "encrypt",
            lambda: client.encrypt(left, slots=SLOTS, cur_limbs=WORK_LIMBS),
            decrypt_cipher,
            left,
            1e-6,
        ),
        Operation(
            "decrypt",
            lambda: client.decrypt(left_cipher, complex_output=True),
            decode_values,
            left,
            1e-6,
        ),
        Operation(
            "add_cipher",
            lambda: fhe.homo_add(left_cipher, right_cipher, context),
            decrypt_cipher,
            left + right,
            1e-6,
        ),
        Operation(
            "multiply_plain_rescale",
            lambda: fhe.homo_mul_pt_rescale(left_cipher, plaintext, context),
            decrypt_cipher,
            left * weights,
            1e-6,
        ),
        Operation(
            "multiply_cipher_rescale",
            lambda: fhe.homo_mul_relin_rescale_postop(
                left_cipher,
                right_cipher,
                context,
            ),
            decrypt_cipher,
            left * right,
            1e-6,
        ),
        Operation(
            "rotate",
            lambda: fhe.homo_rotate(left_cipher, 1, context),
            decrypt_cipher,
            np.roll(left, -1),
            1e-6,
        ),
        Operation(
            "bootstrap",
            lambda: bs.bootstrap(bootstrap_cipher, context, program),
            decrypt_cipher,
            left,
            5e-3,
        ),
    )


def run_operation(operation: Operation, *, warmup: int, runs: int) -> Result:
    for _ in range(warmup):
        operation.run()
    synchronize()

    elapsed = []
    output = None
    for _ in range(runs):
        start = time.perf_counter()
        output = operation.run()
        synchronize()
        elapsed.append(time.perf_counter() - start)

    actual = operation.decode(output)
    max_error = validate_result(
        operation.name,
        actual,
        operation.expected,
        tolerance=operation.tolerance,
    )
    milliseconds = np.asarray(elapsed, dtype=np.float64) * 1000.0
    return Result(
        name=operation.name,
        average_ms=float(milliseconds.mean()),
        minimum_ms=float(milliseconds.min()),
        maximum_ms=float(milliseconds.max()),
        max_error=max_error,
    )


def validate_result(
    name: str,
    actual: np.ndarray,
    expected: np.ndarray,
    *,
    tolerance: float,
) -> float:
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    if actual.shape != expected.shape:
        raise ValueError(
            f"{name}: output shape {actual.shape} does not match {expected.shape}"
        )
    if not np.isfinite(actual).all():
        raise FloatingPointError(f"{name}: output contains non-finite values")
    max_error = float(np.max(np.abs(actual - expected)))
    if max_error > tolerance:
        raise AssertionError(
            f"{name}: max error {max_error:.6e} exceeds {tolerance:.6e}"
        )
    return max_error


def synchronize():
    easyfhe.cuda.synchronize()


def _to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def main(argv=None):
    args = parse_args(argv)
    client, context, program, requirements = create_runtime()
    operations = build_operations(client, context, program)

    print("\n=== canonical EasyFHE u64 latency benchmark ===")
    print(
        f"N={context.ring_dim} slots={SLOTS} depth={requirements.context_depth} "
        f"rotations={len(tuple(dict.fromkeys((1, *requirements.rotations))))}"
    )
    print(f"warmup={args.warmup} measured_runs={args.runs}")
    print(f"{'operation':28} {'avg ms':>11} {'min ms':>11} {'max ms':>11} {'max error':>12}")

    results = []
    for operation in operations:
        result = run_operation(operation, warmup=args.warmup, runs=args.runs)
        results.append(result)
        print(
            f"{result.name:28} {result.average_ms:11.3f} "
            f"{result.minimum_ms:11.3f} {result.maximum_ms:11.3f} "
            f"{result.max_error:12.3e}"
        )
    return tuple(results)


if __name__ == "__main__":
    main()
