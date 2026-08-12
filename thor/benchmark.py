"""Synchronized warmup, measurement, and correctness reporting for THOR."""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

import easyfhe as torch

from .config import (
    BOOTSTRAP_LEVEL_BUDGET,
    BOOTSTRAP_OUTPUT_LEVELS,
    DEPTH,
    DEVICE,
    INPUT_LIMBS,
    PREDICTION_REL_L2_TOLERANCE,
    RunConfig,
    SLOTS,
)
from .model import infer_encrypted
from .reference import PreparedSample, ReferenceAssets, prepare_sample
from .runtime import ThorRuntime


@dataclass(frozen=True)
class SampleResult:
    index: int
    label: int
    reference_prediction: int
    prediction: int
    reference_logits: np.ndarray
    logits: np.ndarray
    relative_l2_error: float
    encrypt_seconds: float
    inference_seconds: float
    decrypt_seconds: float


def validate_benchmark_request(config: RunConfig, assets: ReferenceAssets) -> None:
    stop = config.start_index + config.runs
    if stop > len(assets.split):
        raise ValueError(
            f"requested samples [{config.start_index}, {stop}) exceed "
            f"the {config.split!r} split of length {len(assets.split)}"
        )


def run_benchmark(
    client,
    runtime: ThorRuntime,
    assets: ReferenceAssets,
    config: RunConfig,
):
    """Run correctness-checked warmups and measured encrypted inferences."""

    validate_benchmark_request(config, assets)
    stop = config.start_index + config.runs

    print(
        "u64 THOR context:",
        f"depth={DEPTH}",
        f"limbs={runtime.context.max_limbs}",
        f"input_limbs={INPUT_LIMBS}",
        f"planned_rotations={len(runtime.plan.rotations)}",
    )
    print(
        "bootstrap:",
        f"budget={BOOTSTRAP_LEVEL_BUDGET}",
        f"output_levels={BOOTSTRAP_OUTPUT_LEVELS}",
        f"required_depth={runtime.plan.bootstrap_requirements.context_depth}",
    )
    print(
        f"dataset: split={config.split} samples={len(assets.split)}; "
        f"warmup={config.warmup}; measured_runs={config.runs}"
    )

    if config.warmup:
        warmup_sample = prepare_sample(config.start_index, assets)
        for run in range(config.warmup):
            result = _run_sample(client, runtime, warmup_sample)
            print(_result_line(result, f"warmup {run + 1}/{config.warmup}"))
        warmup_sample.attention_mask_bundle.clear_cache()
        del warmup_sample

    results = []
    for run, index in enumerate(range(config.start_index, stop), start=1):
        sample = prepare_sample(index, assets)
        try:
            result = _run_sample(client, runtime, sample)
        finally:
            # Per-sample masks are no longer needed after this inference.
            sample.attention_mask_bundle.clear_cache()
        results.append(result)
        print(_result_line(result, f"run {run}/{config.runs}"))
        print(
            "    "
            f"encrypt={_seconds(result.encrypt_seconds)} "
            f"infer={_seconds(result.inference_seconds)} "
            f"decrypt={_seconds(result.decrypt_seconds)} "
            f"rel_l2={result.relative_l2_error:.3e} "
            f"logits={np.array2string(result.logits, precision=6, separator=', ')}"
        )

    correct = sum(result.prediction == result.label for result in results)
    inference_times = [result.inference_seconds for result in results]
    print(f"accuracy: {correct}/{len(results)} ({100 * correct / len(results):.2f}%)")
    print(
        "inference time after warmup: "
        f"avg={_seconds(sum(inference_times) / len(inference_times))} "
        f"min={_seconds(min(inference_times))} "
        f"max={_seconds(max(inference_times))}"
    )
    return tuple(results)


def _run_sample(client, runtime: ThorRuntime, sample: PreparedSample) -> SampleResult:
    _synchronize()
    start = time.perf_counter()
    input_ciphers = [
        client.encrypt(
            sample.packed_input[index],
            slots=SLOTS,
            device=DEVICE,
            cur_limbs=INPUT_LIMBS,
        )
        for index in range(4)
    ]
    _synchronize()
    encrypt_seconds = time.perf_counter() - start

    start = time.perf_counter()
    output_ciphers = infer_encrypted(
        runtime,
        input_ciphers,
        sample.attention_mask_bundle,
    )
    _synchronize()
    inference_seconds = time.perf_counter() - start

    start = time.perf_counter()
    logits = np.asarray(
        [
            _as_numpy(client.decrypt(cipher)).reshape(-1)[0]
            for cipher in output_ciphers
        ],
        dtype=np.float64,
    )
    _synchronize()
    decrypt_seconds = time.perf_counter() - start

    relative_l2_error = _relative_l2(logits, sample.reference_logits)
    prediction = int(np.argmax(logits))
    _validate_result(
        sample,
        logits,
        prediction,
        relative_l2_error,
        tolerance=PREDICTION_REL_L2_TOLERANCE,
    )
    return SampleResult(
        index=sample.index,
        label=sample.label,
        reference_prediction=sample.reference_prediction,
        prediction=prediction,
        reference_logits=sample.reference_logits,
        logits=logits,
        relative_l2_error=relative_l2_error,
        encrypt_seconds=encrypt_seconds,
        inference_seconds=inference_seconds,
        decrypt_seconds=decrypt_seconds,
    )


def _validate_result(
    sample: PreparedSample,
    logits: np.ndarray,
    prediction: int,
    relative_l2_error: float,
    *,
    tolerance: float,
):
    if logits.shape != sample.reference_logits.shape:
        raise ValueError(
            f"sample {sample.index}: decrypted logits have shape {logits.shape}, "
            f"expected {sample.reference_logits.shape}"
        )
    if not np.isfinite(logits).all():
        raise FloatingPointError(
            f"sample {sample.index}: decrypted logits are non-finite: {logits}"
        )
    if not np.isfinite(sample.reference_logits).all():
        raise FloatingPointError(
            f"sample {sample.index}: reference logits are non-finite: "
            f"{sample.reference_logits}"
        )
    if not np.isfinite(relative_l2_error):
        raise FloatingPointError(
            f"sample {sample.index}: relative L2 error is non-finite"
        )
    if prediction != sample.reference_prediction:
        raise AssertionError(
            f"sample {sample.index}: encrypted prediction {prediction} differs "
            f"from reference prediction {sample.reference_prediction}; "
            f"encrypted={logits}, reference={sample.reference_logits}"
        )
    if relative_l2_error > tolerance:
        raise AssertionError(
            f"sample {sample.index}: relative L2 error {relative_l2_error:.6e} "
            f"exceeds tolerance {tolerance:.6e}"
        )


def _relative_l2(actual: np.ndarray, expected: np.ndarray) -> float:
    difference = np.asarray(actual) - np.asarray(expected)
    denominator = float(np.linalg.norm(np.asarray(expected).reshape(-1))) + 1e-12
    return float(np.linalg.norm(difference.reshape(-1)) / denominator)


def _as_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _synchronize():
    torch.cuda.synchronize()


def _result_line(result: SampleResult, prefix: str) -> str:
    status = "correct" if result.prediction == result.label else "wrong"
    return (
        f"[{prefix}] index={result.index} label={result.label} "
        f"prediction={result.prediction} reference={result.reference_prediction} "
        f"{status}"
    )


def _seconds(value: float) -> str:
    return f"{value:.3f}s"
