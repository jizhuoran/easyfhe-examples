import argparse
import os
import time
from dataclasses import dataclass

import numpy as np

import easyfhe as torch
import easyfhe.fhe as fhe

from .model import AespaRuntime, infer_one
from .utils import (
    DEFAULT_DATA_DIR,
    DEFAULT_WEIGHTS_PATH,
    WeightPack,
    read_image,
    resolve_test_batch_path,
)

ROTATION_INDICES = (
    -8192,
    -4096,
    -1024,
    -768,
    -256,
    -192,
    -64,
    -32,
    -16,
    -15,
    -8,
    -1,
    1,
    2,
    4,
    8,
    16,
    24,
    32,
    48,
    64,
    128,
    256,
    512,
    1024,
    2048,
    12288,
    24576,
)
# These parameters describe the encrypted program shape: available rotations,
# bootstrap budget, and CKKS modulus chain. They are kept near the entrypoint so
# the demo shows exactly what an EasyFHE application must choose before running.
MAX_LEVELS_REMAINING = 12
LOG_BS_SLOTS = (14,)
LOG_N = 16
DCRT_BITS = 52
FIRST_MOD = 55
LEVEL_BUDGETS = ((4, 4),)
SECRET_KEY_DIST = "SPARSE_TERNARY"
RESCALE_TECH = "FIXEDMANUAL"


def _positive_int(value):
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


@dataclass(frozen=True)
class AespaConfig:
    total: int
    data_dir: str
    weights_path: str
    rotate_indices: tuple[int, ...]
    max_levels_remaining: int
    log_bs_slots: tuple[int, ...]
    log_n: int
    dnum: int
    dcrt_bits: int
    first_mod: int
    level_budgets: tuple[tuple[int, int], ...]
    secret_key_dist: str
    rescale_tech: str
    device: str
    weight_cache_mode: str


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Run CIFAR-10 ResNet-20 encrypted inference with EasyFHE."
    )
    fhe.add_runtime_args(parser, default_device=os.environ.get("EASYFHE_DEVICE", "cpu"))
    fhe.add_output_args(parser)
    parser.add_argument(
        "--total",
        type=_positive_int,
        default=os.environ.get("EASYFHE_TOTAL", "1"),
        help="number of CIFAR-10 test images to run; defaults to EASYFHE_TOTAL or 1",
    )
    return parser.parse_args()


def build_config(args):
    data_dir = os.environ.get("EASYFHE_RESNET20_AESPA_DATA_DIR", str(DEFAULT_DATA_DIR))
    weights_path = os.environ.get("EASYFHE_RESNET20_AESPA_WEIGHTS", str(DEFAULT_WEIGHTS_PATH))

    return AespaConfig(
        total=args.total,
        data_dir=data_dir,
        weights_path=weights_path,
        rotate_indices=ROTATION_INDICES,
        max_levels_remaining=MAX_LEVELS_REMAINING,
        log_bs_slots=LOG_BS_SLOTS,
        log_n=LOG_N,
        dnum=int(os.environ.get("EASYFHE_DNUM", "3")),
        dcrt_bits=DCRT_BITS,
        first_mod=FIRST_MOD,
        level_budgets=LEVEL_BUDGETS,
        secret_key_dist=SECRET_KEY_DIST,
        rescale_tech=RESCALE_TECH,
        device=args.device,
        weight_cache_mode=os.environ.get("EASYFHE_WEIGHT_CACHE_MODE", "plain"),
    )


def _print_config(config):
    print("rotate_index_list: ", list(config.rotate_indices))
    print("maxLevelsRemaining: ", config.max_levels_remaining)
    print("logBsSlots_list: ", list(config.log_bs_slots))
    print("logN: ", config.log_n)
    print("dnum: ", config.dnum)
    print("dcrtBits: ", config.dcrt_bits)
    print("firstMod: ", config.first_mod)
    print("levelBudget_list: ", [list(level_budget) for level_budget in config.level_budgets])
    print("secretKeyDist: ", config.secret_key_dist)
    print("rescaleTech: ", config.rescale_tech)
    print("weightCacheMode: ", config.weight_cache_mode)
    print("\n\n")
    print("device: ", config.device)
    print("data_dir=", config.data_dir)
    print("test_batch=", resolve_test_batch_path(config.data_dir))
    print("weights_path=", config.weights_path)


def _decrypt_prediction(final_res, runtime):
    try:
        clear_result = runtime.ctx.decrypt(final_res).cpu().numpy().reshape(-1)[:10]
        return clear_result, np.argmax(clear_result)
    except RuntimeError as e:
        print(f"Decryption failed: {e}")
        return None, 11


def _sync_device(runtime):
    if runtime.ctx.device == "cuda":
        torch.cuda.synchronize()


def _format_seconds(seconds):
    return f"{seconds:.3f}s"


def _format_accuracy(correct, total):
    percent = 100.0 * correct / total if total else 0.0
    return f"{correct}/{total} ({percent:.2f}%)"


def _format_bytes(num_bytes):
    units = ("B", "KiB", "MiB", "GiB")
    value = float(num_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.1f}{unit}" if unit != "B" else f"{int(value)}B"
        value /= 1024.0


def _print_timing_summary(infer_times, total_seconds, correct, total):
    print("\n================ dataset summary ================")
    print(f"accuracy: {_format_accuracy(correct, total)}")
    print(f"wall time: {_format_seconds(total_seconds)}")
    if not infer_times:
        return

    avg = sum(infer_times) / len(infer_times)
    print(
        "inference time:",
        f"avg={_format_seconds(avg)}",
        f"min={_format_seconds(min(infer_times))}",
        f"max={_format_seconds(max(infer_times))}",
    )
    if len(infer_times) > 1:
        warm_avg = sum(infer_times[1:]) / (len(infer_times) - 1)
        print(f"inference time excluding first image: avg={_format_seconds(warm_avg)}")


def _print_weight_cache_summary(weights):
    if not hasattr(weights, "cache_info"):
        return
    info = weights.cache_info()
    print(
        "weight cache:",
        f"mode={info['mode']}",
        f"middle={info['middle_entries']}({_format_bytes(info['middle_bytes'])})",
        f"plain={info['plain_entries']}({_format_bytes(info['plain_bytes'])})",
        f"plain_hits={info['plain_hits']}",
        f"plain_misses={info['plain_misses']}",
        f"middle_hits={info['middle_hits']}",
        f"middle_misses={info['middle_misses']}",
    )


def _bootstrap_specs(config):
    return tuple(
        fhe.BootstrapSpec(log_bs_slots, tuple(level_budget))
        for log_bs_slots, level_budget in zip(config.log_bs_slots, config.level_budgets)
    )


def _create_crypto_context(config, args):
    bootstrap_specs = _bootstrap_specs(config)
    options = fhe.runtime_options_from_args(args)
    return fhe.generate_context(
        fhe.CKKSContextSpec(
            depth=fhe.bootstrap_depth(
                config.max_levels_remaining,
                bootstrap_specs,
                config.secret_key_dist,
            ),
            log_n=config.log_n,
            dnum=config.dnum,
            dcrt_bits=config.dcrt_bits,
            first_mod=config.first_mod,
            secret_key_dist=config.secret_key_dist,
            rescale_tech=config.rescale_tech,
            rotations=config.rotate_indices,
        ),
        device=config.device,
        options=options,
    )


def _create_bootstrap_constants(config, crypto_context):
    bootstrap_constants = {}
    for log_bs_slots, level_budget in zip(config.log_bs_slots, config.level_budgets):
        bootstrap_constants[int(log_bs_slots)] = fhe.generate_bootstrap_constants(
            crypto_context,
            log_bs_slots,
            level_budget,
            config.max_levels_remaining,
        )
    return bootstrap_constants


def _validate_inputs(config):
    test_batch_path = resolve_test_batch_path(config.data_dir)
    if not test_batch_path.exists():
        raise ValueError(f"CIFAR-10 test batch {test_batch_path} does not exist!")


def build_runtime(config, args):
    _print_config(config)
    crypto_context = _create_crypto_context(config, args)
    bootstrap_constants = _create_bootstrap_constants(config, crypto_context)
    print("crypto_context: ", crypto_context)

    # WeightPack caches encoded plaintexts by level/slots/context so later
    # images can reuse weights prepared during earlier encrypted inference.
    weights = WeightPack.from_npz(config.weights_path, cache_mode=config.weight_cache_mode)
    print("weights loaded:", len(weights))
    return AespaRuntime(crypto_context, weights, config, bootstrap_constants)


def run_dataset(runtime):
    total = runtime.config.total
    infer_times = []
    correct = 0
    dataset_start = time.perf_counter()

    print("\n================ run dataset ================")
    print(f"images: {total}")
    print(f"device: {runtime.ctx.device}")

    for i in range(total):
        image_vector, label, index = read_image(i, data_dir=runtime.config.data_dir)
        item_start = time.perf_counter()

        _sync_device(runtime)
        infer_start = time.perf_counter()
        final_res = infer_one(image_vector, runtime)
        _sync_device(runtime)
        infer_seconds = time.perf_counter() - infer_start
        infer_times.append(infer_seconds)

        decrypt_start = time.perf_counter()
        logits, max_element_idx = _decrypt_prediction(final_res, runtime)
        decrypt_seconds = time.perf_counter() - decrypt_start
        item_seconds = time.perf_counter() - item_start

        is_correct = label == max_element_idx
        if is_correct:
            correct += 1
        status = "correct" if is_correct else "wrong"

        print(
            f"[{i + 1}/{total}] index={index} label={label} "
            f"prediction={max_element_idx} {status}"
        )
        print(
            "    "
            f"infer={_format_seconds(infer_seconds)} "
            f"decrypt={_format_seconds(decrypt_seconds)} "
            f"total={_format_seconds(item_seconds)} "
            f"accuracy={_format_accuracy(correct, i + 1)}"
        )
        if logits is not None:
            print("    logits=", np.array2string(logits, precision=6, separator=", "))

    total_seconds = time.perf_counter() - dataset_start
    _print_timing_summary(infer_times, total_seconds, correct, total)
    _print_weight_cache_summary(runtime.weights)


def main(config=None, args=None):
    if args is None:
        args = _parse_args()
    if config is None:
        config = build_config(args)

    _validate_inputs(config)
    run_dataset(build_runtime(config, args))


if __name__ == "__main__":
    main()
