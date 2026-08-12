"""Run the canonical u64 full-slot complex CKKS bootstrap."""

import argparse
import time

import numpy as np

import easyfhe
import easyfhe.bs.openfhe as bs
import easyfhe.fhe as fhe


LOG_N = 16
INPUT_LIMBS = 3
SECRET_KEY_DIST = "SPARSE_TERNARY"
ERROR_TOLERANCE = 5e-3


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=1)
    args = parser.parse_args(argv)
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.runs <= 0:
        parser.error("--runs must be positive")
    return args


def make_values(slots: int) -> np.ndarray:
    angle = 2.0 * np.pi * np.arange(slots, dtype=np.float64) / slots
    real = 0.04 + 0.01 * np.sin(angle)
    imag = -0.03 + 0.008 * np.cos(angle)
    return (real + 1j * imag).astype(np.complex128)


def synchronize():
    easyfhe.cuda.synchronize()


def main(argv=None):
    args = parse_args(argv)

    bootstrap_spec = bs.BootstrapSpec(
        log_slots=15,
        level_budget=(4, 4),
        output_levels=2,
        strategy="normal_giant",
        mode="modraise_first",
    )
    requirements = bs.requirements(
        bootstrap_spec,
        log_n=LOG_N,
        secret_key_dist=SECRET_KEY_DIST,
    )

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
            rotations=requirements.rotations,
        ),
        device="cuda",
    )
    program = bs.generate(context, bootstrap_spec)

    values = make_values(bootstrap_spec.slots)
    cipher = client.encrypt(
        values,
        slots=bootstrap_spec.slots,
        cur_limbs=INPUT_LIMBS,
    )

    print("\n=== u64 full-slot complex bootstrap ===")
    print(
        f"N={context.ring_dim} slots={bootstrap_spec.slots} "
        f"strategy={bootstrap_spec.strategy} mode={bootstrap_spec.mode}"
    )
    print(
        f"context_depth={requirements.context_depth} "
        f"bootstrap_depth={requirements.bootstrap_depth} "
        f"rotation_keys={len(requirements.rotations)}"
    )
    print(f"input_state={cipher.state}")

    for _ in range(args.warmup):
        bs.bootstrap(cipher, context, program)

    # Exclude warmup and any earlier queued CUDA work from measured execution.
    synchronize()

    elapsed = []
    result = None
    for _ in range(args.runs):
        start = time.perf_counter()
        result = bs.bootstrap(cipher, context, program)
        synchronize()
        elapsed.append(time.perf_counter() - start)

    if result.state != program.output_state:
        raise AssertionError(
            f"bootstrap returned {result.state}, expected {program.output_state}"
        )

    decoded = (
        client.decrypt(result, complex_output=True)
        .cpu()
        .numpy()[: bootstrap_spec.slots]
    )
    if not np.isfinite(decoded).all():
        raise AssertionError("bootstrap produced non-finite output")

    max_error = float(np.max(np.abs(decoded - values)))

    print(f"output_state={result.state}")
    print("input[:4] =", np.array2string(values[:4], precision=8, separator=", "))
    print("output[:4] =", np.array2string(decoded[:4], precision=8, separator=", "))
    print(f"max_complex_abs_error={max_error:.6e}")
    print(
        f"bootstrap_seconds avg={sum(elapsed) / len(elapsed):.6f} "
        f"min={min(elapsed):.6f} max={max(elapsed):.6f}"
    )

    if max_error > ERROR_TOLERANCE:
        raise AssertionError(
            f"bootstrap error {max_error:.6e} exceeds tolerance {ERROR_TOLERANCE:.6e}"
        )


if __name__ == "__main__":
    main()
