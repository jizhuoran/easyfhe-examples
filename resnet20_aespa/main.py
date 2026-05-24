import argparse
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import easyfhe as torch
import easyfhe.bs.openfhe as bs
import easyfhe.fhe as fhe
from .model import AespaRuntime, encrypt_input, infer_encrypted

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_PATH = SCRIPT_DIR / "data" / "cifar10" / "test_batch.npz"
DATASET_PATH = os.environ.get("EASYFHE_RESNET20_AESPA_DATASET", str(DEFAULT_DATASET_PATH))
WEIGHTS_PATH = os.environ.get(
    "EASYFHE_RESNET20_AESPA_WEIGHTS",
    str(SCRIPT_DIR / "resnet20_aespa_weights.npz"),
)
WARMUP_ITERS = 1
WEIGHT_CACHE_RESERVED_GB = 24.0


@dataclass(frozen=True)
class AespaConfig:
    total: int
    dataset_path: str
    weights_path: str
    input_level: int
    rotate_indices: tuple[int, ...]
    post_bootstrap_levels: int
    log_bs_slots: tuple[int, ...]
    log_n: int
    dnum: int
    dcrt_bits: int
    first_mod: int
    level_budgets: tuple[tuple[int, int], ...]
    bootstrap_strategy: str
    bootstrap_mode: str
    secret_key_dist: str
    scale_mode: str
    rescale_policy: str
    device: str
    weight_cache_mode: str
    weight_plain_cache_limit_gb: float | None
    weight_plain_cache_policy: str


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default=os.environ.get("EASYFHE_DEVICE", "cuda"))
    key_group = parser.add_mutually_exclusive_group()
    key_group.add_argument("--auto-load-keys", dest="auto_load_keys", action="store_true", default=None)
    key_group.add_argument("--no-auto-load-keys", dest="auto_load_keys", action="store_false")
    parser.add_argument(
        "--rotation-random-mode",
        choices=("fresh", "reuse_by_shape"),
        default="fresh",
    )
    parser.add_argument(
        "--rot-key-limb-limit",
        action="append",
        default=[],
        metavar="ROT:LIMBS",
    )
    parser.add_argument("--save-middle", action="store_true")
    parser.add_argument("--save-end", action="store_true")
    parser.add_argument("--total", type=int, default=int(os.environ.get("EASYFHE_TOTAL", "1")))
    parser.add_argument(
        "--bootstrap-strategy",
        choices=("double_hoist", "normal_giant", "normal_bsgs"),
        default=os.environ.get("EASYFHE_BOOTSTRAP_STRATEGY", "normal_giant"),
    )
    parser.add_argument(
        "--bootstrap-mode",
        choices=("classic", "modraise_first", "slots_first", "stc_first"),
        default=os.environ.get("EASYFHE_BOOTSTRAP_MODE", "modraise_first"),
    )
    parser.add_argument(
        "--secret-key-dist",
        choices=("SPARSE_TERNARY", "UNIFORM_TERNARY"),
        default=os.environ.get("EASYFHE_SECRET_KEY_DIST", "SPARSE_TERNARY"),
    )
    return parser.parse_known_args()[0]


def _optional_float_env(name):
    value = os.environ.get(name)
    if value is None or value == "":
        return None
    return float(value)


def _selected_cuda_device():
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible:
        return visible.split(",", 1)[0].strip()
    return "0"


def _default_weight_cache_limit_gb(device):
    if device != "cuda":
        return None
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.total",
                "--format=csv,noheader,nounits",
                "-i",
                _selected_cuda_device(),
            ],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        total_mib = float(result.stdout.strip().splitlines()[0])
    except (OSError, subprocess.CalledProcessError, IndexError, ValueError):
        return None
    return max(total_mib / 1024.0 - WEIGHT_CACHE_RESERVED_GB, 0.0)


def _parse_rotation_key_limb_limits(values):
    limits = {}
    for value in values or ():
        try:
            rotation, limbs = str(value).split(":", 1)
            limits[int(rotation)] = int(limbs)
        except ValueError as exc:
            raise ValueError(
                f"invalid --rot-key-limb-limit {value!r}; expected ROT:LIMBS"
            ) from exc
    return limits


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


def _format_seconds(seconds):
    return f"{seconds:.3f}s"


def _load_weights(config):
    weight_path = Path(config.weights_path)
    if not weight_path.exists():
        raise ValueError(f"Weight npz {weight_path} does not exist!")
    with np.load(weight_path) as weights:
        scalars = {
            name: float(np.asarray(weights[name]))
            for name in weights.files
            if name.startswith("scale.")
        }
        vectors = {
            name: np.asarray(weights[name], dtype=np.double)
            for name in weights.files
            if not name.startswith("scale.")
        }
    return fhe.ConstantBundle(
        scalars=scalars,
        vectors=vectors,
        cache_mode=config.weight_cache_mode,
        plain_cache_limit_gb=config.weight_plain_cache_limit_gb,
        plain_cache_policy=config.weight_plain_cache_policy,
    )


def build_config(args):
    secret_key_dist = str(
        getattr(args, "secret_key_dist", os.environ.get("EASYFHE_SECRET_KEY_DIST", "SPARSE_TERNARY"))
    ).upper()
    weight_plain_cache_limit_gb = _optional_float_env("EASYFHE_WEIGHT_PLAIN_CACHE_GB")
    if weight_plain_cache_limit_gb is None:
        weight_plain_cache_limit_gb = _default_weight_cache_limit_gb(args.device)
    return AespaConfig(
        total=args.total,
        dataset_path=DATASET_PATH,
        weights_path=WEIGHTS_PATH,
        input_level=int(os.environ.get("EASYFHE_INPUT_LEVEL", "13")),
        rotate_indices=(
            -8192, -4096, -1024, -768, -256, -192, -64, -33, -32, -31, -17, -16,
            -15, -9, -8, -7, -1, 1, 2, 4, 7, 8, 9, 15, 16, 17, 24, 31, 32, 33,
            48, 64, 128, 256, 512, 1024, 2048, 12288, 24576,
        ),
        post_bootstrap_levels=int(os.environ.get("EASYFHE_POST_BOOTSTRAP_LEVELS", "11")),
        log_bs_slots=(14,),
        log_n=16,
        dnum=int(os.environ.get("EASYFHE_DNUM", "3")),
        dcrt_bits=int(os.environ.get("EASYFHE_DCRT_BITS", "59")),
        first_mod=int(os.environ.get("EASYFHE_FIRST_MOD", "60")),
        level_budgets=((4, 4),),
        bootstrap_strategy=getattr(
            args,
            "bootstrap_strategy",
            os.environ.get("EASYFHE_BOOTSTRAP_STRATEGY", "normal_giant"),
        ),
        bootstrap_mode=getattr(
            args,
            "bootstrap_mode",
            os.environ.get("EASYFHE_BOOTSTRAP_MODE", "modraise_first"),
        ),
        secret_key_dist=secret_key_dist,
        scale_mode="fixed",
        rescale_policy="manual",
        device=args.device,
        weight_cache_mode=os.environ.get("EASYFHE_WEIGHT_CACHE_MODE", "mix"),
        weight_plain_cache_limit_gb=weight_plain_cache_limit_gb,
        weight_plain_cache_policy=os.environ.get("EASYFHE_WEIGHT_PLAIN_CACHE_POLICY", "small_first"),
    )


def _print_config(config):
    print("rotate_index_list: ", list(config.rotate_indices))
    print("postBootstrapLevels: ", config.post_bootstrap_levels)
    print("logBsSlots_list: ", list(config.log_bs_slots))
    print("logN: ", config.log_n)
    print("dnum: ", config.dnum)
    print("dcrtBits: ", config.dcrt_bits)
    print("firstMod: ", config.first_mod)
    print("levelBudget_list: ", [list(level_budget) for level_budget in config.level_budgets])
    print("bootstrapStrategy: ", config.bootstrap_strategy)
    print("bootstrapMode: ", config.bootstrap_mode)
    print("secretKeyDist: ", config.secret_key_dist)
    print("scaleMode: ", config.scale_mode)
    print("rescalePolicy: ", config.rescale_policy)
    print("inputLevel: ", config.input_level)
    print("weightCacheMode: ", config.weight_cache_mode)
    print("weightPlainCacheLimitGB: ", config.weight_plain_cache_limit_gb)
    print("weightPlainCachePolicy: ", config.weight_plain_cache_policy)
    print("\n\n")
    print("device: ", config.device)
    print("dataset_path=", config.dataset_path)
    print("weights_path=", config.weights_path)


def _decrypt_prediction(final_res, rt):
    try:
        clear_result = rt.client.decrypt(final_res).cpu().numpy().reshape(-1)[:10]
        return clear_result, np.argmax(clear_result)
    except RuntimeError as e:
        print(f"Decryption failed: {e}")
        return None, 11


def _sync_device(rt):
    if rt.ctx.device == "cuda":
        torch.cuda.synchronize()


def _print_time_series(label, times):
    if not times:
        return
    avg = sum(times) / len(times)
    print(
        f"{label}:",
        f"avg={_format_seconds(avg)}",
        f"min={_format_seconds(min(times))}",
        f"max={_format_seconds(max(times))}",
    )


def _print_timing_summary(encrypt_times, infer_times, correct, total):
    print("\n================ dataset summary ================")
    print(f"accuracy: {_format_accuracy(correct, total)}")
    print(f"e2e no encrypt/decrypt: {_format_seconds(sum(infer_times))}")
    _print_time_series("encrypt time", encrypt_times)
    _print_time_series("inference time", infer_times)


def _print_weight_cache_summary(weights):
    if not hasattr(weights, "cache_info"):
        return
    info = weights.cache_info()
    print(
        "weight cache:",
        f"mode={info['mode']}",
        f"plain_policy={info.get('plain_cache_policy', 'first_fit')}",
        f"middle={info['middle_entries']}({_format_bytes(info['middle_bytes'])})",
        f"plain={info['plain_entries']}({_format_bytes(info['plain_bytes'])})",
        f"scalar={info.get('scalar_entries', 0)}({_format_bytes(info.get('scalar_bytes', 0))})",
        f"total={_format_bytes(info.get('total_bytes', 0))}",
        f"plain_limit={_format_optional_bytes(info.get('plain_cache_limit_bytes'))}",
        f"plain_remaining={_format_optional_bytes(info.get('plain_cache_remaining_bytes'))}",
        f"plain_skips={info.get('plain_cache_skips', 0)}",
        f"plain_evictions={info.get('plain_cache_evictions', 0)}",
        f"plain_hits={info['plain_hits']}",
        f"plain_misses={info['plain_misses']}",
        f"scalar_hits={info.get('scalar_hits', 0)}",
        f"scalar_misses={info.get('scalar_misses', 0)}",
        f"middle_hits={info['middle_hits']}",
        f"middle_misses={info['middle_misses']}",
    )


def _format_optional_bytes(num_bytes):
    return "unlimited" if num_bytes is None else _format_bytes(num_bytes)


def _run_sample(rt, dataset, index, *, timed):
    sample_index = index % len(dataset)
    image_vector, label = dataset[sample_index]

    if timed:
        _sync_device(rt)
        encrypt_start = time.perf_counter()
        input_cipher = encrypt_input(image_vector, rt)
        _sync_device(rt)
        encrypt_seconds = time.perf_counter() - encrypt_start

        infer_start = time.perf_counter()
        final_res = infer_encrypted(input_cipher, rt)
        _sync_device(rt)
        infer_seconds = time.perf_counter() - infer_start

        decrypt_start = time.perf_counter()
        logits, max_element_idx = _decrypt_prediction(final_res, rt)
        decrypt_seconds = time.perf_counter() - decrypt_start
    else:
        _sync_device(rt)
        input_cipher = encrypt_input(image_vector, rt)
        final_res = infer_encrypted(input_cipher, rt)
        _sync_device(rt)
        logits, max_element_idx = _decrypt_prediction(final_res, rt)
        encrypt_seconds = infer_seconds = decrypt_seconds = None

    return {
        "index": sample_index,
        "label": label,
        "logits": logits,
        "prediction": max_element_idx,
        "encrypt_seconds": encrypt_seconds,
        "infer_seconds": infer_seconds,
        "decrypt_seconds": decrypt_seconds,
    }


def run_dataset(rt):
    dataset = np.load(rt.config.dataset_path, allow_pickle=True)["samples"]
    total = rt.config.total
    encrypt_times = []
    infer_times = []
    correct = 0

    print("\n================ run dataset ================")
    print(f"warmup iterations: {WARMUP_ITERS}")
    print(f"measured iterations: {total}")
    print(f"device: {rt.ctx.device}")

    for i in range(WARMUP_ITERS):
        result = _run_sample(rt, dataset, i, timed=False)
        is_correct = result["label"] == result["prediction"]
        status = "correct" if is_correct else "wrong"
        print(
            f"[warmup {i + 1}/{WARMUP_ITERS}] index={result['index']} "
            f"label={result['label']} prediction={result['prediction']} {status}"
        )

    for i in range(total):
        result = _run_sample(rt, dataset, i, timed=True)
        encrypt_times.append(result["encrypt_seconds"])
        infer_times.append(result["infer_seconds"])

        is_correct = result["label"] == result["prediction"]
        if is_correct:
            correct += 1
        status = "correct" if is_correct else "wrong"

        print(
            f"[{i + 1}/{total}] index={result['index']} label={result['label']} "
            f"prediction={result['prediction']} {status}"
        )
        print(
            "    "
            f"encrypt={_format_seconds(result['encrypt_seconds'])} "
            f"infer={_format_seconds(result['infer_seconds'])} "
            f"decrypt={_format_seconds(result['decrypt_seconds'])} "
            f"e2e_no_encrypt_decrypt={_format_seconds(result['infer_seconds'])} "
            f"accuracy={_format_accuracy(correct, i + 1)}"
        )
        if result["logits"] is not None:
            print("    logits=", np.array2string(result["logits"], precision=6, separator=", "))

    _print_timing_summary(encrypt_times, infer_times, correct, total)
    _print_weight_cache_summary(rt.weights)


def resnet20(config=None, args=None):
    if args is None:
        args = _parse_args()
    if config is None:
        config = build_config(args)

    _print_config(config)
    bootstrap_extra_depth = bs.depth(
        log_bs_slots=config.log_bs_slots,
        level_budget=config.level_budgets,
        secret_key_dist=config.secret_key_dist,
    )
    bootstrap_rotations = bs.plan_rot_keys(
        log_n=config.log_n,
        log_bs_slots=config.log_bs_slots,
        level_budget=config.level_budgets,
        strategy=config.bootstrap_strategy,
    )
    rotations = tuple(dict.fromkeys([*config.rotate_indices, *bootstrap_rotations]))
    client, cryptoContext = fhe.generate_client_context(
        fhe.CKKSContextSpec(
            depth=config.post_bootstrap_levels + bootstrap_extra_depth,
            log_n=config.log_n,
            dnum=config.dnum,
            dcrt_bits=config.dcrt_bits,
            first_mod=config.first_mod,
            secret_key_dist=config.secret_key_dist,
            scale_mode=config.scale_mode,
            rescale_policy=config.rescale_policy,
            rotations=rotations,
            auto_load_keys=args.auto_load_keys,
            rotation_random_mode=str(args.rotation_random_mode),
            rotation_key_limb_limits=_parse_rotation_key_limb_limits(args.rot_key_limb_limit),
        ),
        device=config.device,
    )
    bootstrap_material = {}
    for log_bs_slots, level_budget in zip(config.log_bs_slots, config.level_budgets):
        constants, plan = bs.generate(
            cryptoContext,
            log_bs_slots=log_bs_slots,
            level_budget=level_budget,
            post_bootstrap_levels=config.post_bootstrap_levels,
            strategy=config.bootstrap_strategy,
        )
        bootstrap_material[int(log_bs_slots)] = (constants, plan)
    print("cryptoContext: ", cryptoContext)
    weights = _load_weights(config)
    print("weights loaded:", len(weights))

    run_dataset(AespaRuntime(cryptoContext, client, weights, config, bootstrap_material))


if __name__ == "__main__":
    resnet20()
