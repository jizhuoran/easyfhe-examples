"""Synchronized warmup, measurement, and correctness reporting."""

from dataclasses import dataclass
import time

import numpy as np

import easyfhe
import easyfhe.fhe as fhe

from .config import INPUT_LIMBS, INPUT_SLOTS, RunConfig
from .model import infer_encrypted
from .runtime import ResNet20Runtime


@dataclass(frozen=True)
class SampleResult:
    index: int
    label: int
    logits: np.ndarray
    prediction: int
    encrypt_seconds: float
    infer_seconds: float
    decrypt_seconds: float


def run_dataset(client: fhe.Client, runtime: ResNet20Runtime, config: RunConfig):
    images, labels = _load_dataset(config.dataset_path)
    print(
        f"dataset: {len(images)} samples; "
        f"warmup={config.warmup}; measured_runs={config.runs}"
    )

    for index in range(config.warmup):
        result = _run_sample(client, runtime, images, labels, index)
        print(_result_line(result, f"warmup {index + 1}/{config.warmup}"))

    results = [
        _run_sample(client, runtime, images, labels, index)
        for index in range(config.runs)
    ]
    for run, result in enumerate(results, start=1):
        print(_result_line(result, f"run {run}/{config.runs}"))
        print(
            "    "
            f"encrypt={_seconds(result.encrypt_seconds)} "
            f"infer={_seconds(result.infer_seconds)} "
            f"decrypt={_seconds(result.decrypt_seconds)} "
            f"logits={np.array2string(result.logits, precision=6, separator=', ')}"
        )

    correct = sum(result.label == result.prediction for result in results)
    infer_times = [result.infer_seconds for result in results]
    print(f"accuracy: {_accuracy(correct, config.runs)}")
    if infer_times:
        print(
            "inference time: "
            f"avg={_seconds(sum(infer_times) / len(infer_times))} "
            f"min={_seconds(min(infer_times))} "
            f"max={_seconds(max(infer_times))}"
        )


def _load_dataset(path):
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"images", "labels"}:
            raise ValueError(
                f"dataset must contain exactly images and labels, got {archive.files}"
            )
        images = np.asarray(archive["images"])
        labels = np.asarray(archive["labels"])

    if images.ndim != 2 or images.shape[1] != 3 * 32 * 32:
        raise ValueError(f"expected images shaped [N, 3072], got {images.shape}")
    if images.dtype != np.float64:
        raise ValueError(f"expected float64 images, got {images.dtype}")
    if labels.shape != (images.shape[0],):
        raise ValueError(
            f"expected labels shaped [{images.shape[0]}], got {labels.shape}"
        )
    if labels.dtype != np.int64:
        raise ValueError(f"expected int64 labels, got {labels.dtype}")
    if not len(images):
        raise ValueError("dataset must contain at least one sample")
    if not np.isfinite(images).all():
        raise ValueError("dataset contains non-finite image values")
    if np.any((labels < 0) | (labels >= 10)):
        raise ValueError("dataset labels must be integers in [0, 9]")
    return images, labels


def _run_sample(client, runtime, images, labels, index):
    sample_index = index % len(images)

    _sync()
    start = time.perf_counter()
    input_cipher = client.encrypt(
        images[sample_index],
        device="cuda",
        slots=INPUT_SLOTS,
        cur_limbs=INPUT_LIMBS,
    )
    _sync()
    encrypt_seconds = time.perf_counter() - start

    start = time.perf_counter()
    output_cipher = infer_encrypted(input_cipher, runtime)
    _sync()
    infer_seconds = time.perf_counter() - start

    start = time.perf_counter()
    logits = client.decrypt(output_cipher).cpu().numpy().reshape(-1)[:10]
    decrypt_seconds = time.perf_counter() - start
    if logits.shape != (10,):
        raise ValueError(f"decryption produced {logits.size} logits; expected 10")
    if not np.isfinite(logits).all():
        raise FloatingPointError(f"decryption produced non-finite logits: {logits}")

    return SampleResult(
        index=sample_index,
        label=int(labels[sample_index]),
        logits=logits,
        prediction=int(np.argmax(logits)),
        encrypt_seconds=encrypt_seconds,
        infer_seconds=infer_seconds,
        decrypt_seconds=decrypt_seconds,
    )


def _sync():
    easyfhe.cuda.synchronize()


def _result_line(result, prefix):
    status = "correct" if result.label == result.prediction else "wrong"
    return (
        f"[{prefix}] index={result.index} label={result.label} "
        f"prediction={result.prediction} {status}"
    )


def _accuracy(correct, total):
    percent = 100.0 * correct / total if total else 0.0
    return f"{correct}/{total} ({percent:.2f}%)"


def _seconds(value):
    return f"{value:.3f}s"
