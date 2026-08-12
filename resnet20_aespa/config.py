"""Fixed u64 ResNet20 design and the small benchmark CLI."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import easyfhe.bs.openfhe as bs


PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_ASSET_DIR = PACKAGE_DIR / "assets"
DEFAULT_DATASET_PATH = DEFAULT_ASSET_DIR / "test_batch.npz"
DEFAULT_WEIGHTS_PATH = DEFAULT_ASSET_DIR / "resnet20_aespa_weights.npz"

# This example is one tested circuit, not a parameter sweep.  Keeping its
# cryptographic design fixed makes the context construction below reproducible.
LOG_N = 16
DNUM = 3
LOG_BOOTSTRAP_SLOTS = 14
BOOTSTRAP_LEVEL_BUDGET = (4, 4)
BOOTSTRAP_OUTPUT_LEVELS = (5, 9, 11, 12)
FIRST_PRIME_BITS = 60
RESCALE_PRIME_BITS = 59
SECRET_KEY_DIST = "SPARSE_TERNARY"
BOOTSTRAP_STRATEGY = "normal_giant"
BOOTSTRAP_MODE = "modraise_first"
INPUT_LIMBS = 18
INPUT_SLOTS = 16 * 32 * 32

# Application rotations only. Bootstrap rotations are supplied by
# bs.requirements() and are unioned with these in runtime.py.
NETWORK_ROTATIONS = (
    -8192,
    -4096,
    -1024,
    -768,
    -256,
    -192,
    -64,
    -33,
    -32,
    -31,
    -17,
    -16,
    -15,
    -9,
    -8,
    -7,
    -1,
    1,
    2,
    4,
    7,
    8,
    9,
    15,
    16,
    17,
    24,
    31,
    32,
    33,
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


@dataclass(frozen=True)
class RunConfig:
    """Dataset and measurement options; the FHE circuit is fixed above."""

    runs: int = 1
    warmup: int = 1
    dataset_path: Path = DEFAULT_DATASET_PATH
    weights_path: Path = DEFAULT_WEIGHTS_PATH


def bootstrap_specs() -> tuple[bs.BootstrapSpec, ...]:
    """Return the four bootstrap contracts used by the network graph."""

    return tuple(
        bs.BootstrapSpec(
            log_slots=LOG_BOOTSTRAP_SLOTS,
            level_budget=BOOTSTRAP_LEVEL_BUDGET,
            output_levels=output_levels,
            strategy=BOOTSTRAP_STRATEGY,
            mode=BOOTSTRAP_MODE,
        )
        for output_levels in BOOTSTRAP_OUTPUT_LEVELS
    )


def parse_args(argv=None) -> RunConfig:
    parser = argparse.ArgumentParser(
        description="Canonical u64 CKKS ResNet20 AESPA inference"
    )
    parser.add_argument(
        "--runs", type=_positive_int, default=1, help="measured inferences"
    )
    parser.add_argument(
        "--warmup",
        type=_nonnegative_int,
        default=1,
        help="unmeasured warmup inferences",
    )
    parser.add_argument(
        "--dataset", type=Path, default=DEFAULT_DATASET_PATH, help="numeric NPZ"
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_WEIGHTS_PATH,
        help="compact runtime-packed NPZ",
    )
    args = parser.parse_args(argv)
    return RunConfig(
        runs=args.runs,
        warmup=args.warmup,
        dataset_path=args.dataset,
        weights_path=args.weights,
    )


def _nonnegative_int(value):
    value = int(value)
    if value < 0:
        raise argparse.ArgumentTypeError("expected a non-negative integer")
    return value


def _positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return value
