"""Configuration for the canonical u64 THOR example."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_ASSET_DIR = PACKAGE_DIR / "assets"
DEFAULT_DATASET_PATH = DEFAULT_ASSET_DIR / "mrpc"
DEFAULT_MODEL_DIR = DEFAULT_ASSET_DIR / "model"
DEFAULT_WEIGHTS_PATH = DEFAULT_MODEL_DIR / "model.safetensors"
DEFAULT_TOKENIZER_PATH = DEFAULT_MODEL_DIR


@dataclass(frozen=True)
class LayerPlan:
    """The fixed, validated arithmetic choices for one BERT encoder layer."""

    index: int
    attention_key_scale: float
    softmax_variant: str
    softmax_epsilon: float
    refine_softmax_inverse: bool
    force_softmax_bootstrap: bool
    ff_layernorm_min_var: float
    ff_layernorm_max_var: float

    @classmethod
    def for_layer(cls, index: int) -> "LayerPlan":
        index = int(index)
        if not 0 <= index < 12:
            raise ValueError(f"THOR layer index must be in [0, 11], got {index}")
        uses_softmax2 = index == 2
        uses_layernorm3_bounds = index in (9, 10)
        return cls(
            index=index,
            attention_key_scale=1 / 1024 if uses_softmax2 else 1 / 512,
            softmax_variant="softmax2" if uses_softmax2 else "softmax1",
            softmax_epsilon=2 ** (-18) if uses_softmax2 else 2 ** (-11),
            refine_softmax_inverse=uses_softmax2,
            force_softmax_bootstrap=uses_softmax2,
            ff_layernorm_min_var=0.75 if uses_layernorm3_bounds else 0.2,
            ff_layernorm_max_var=2500.0 if uses_layernorm3_bounds else 150.0,
        )


@dataclass(frozen=True)
class RunConfig:
    """Client-side assets and benchmark options."""

    dataset_path: Path
    weights_path: Path
    tokenizer_path: Path
    split: str = "train"
    warmup: int = 1
    runs: int = 1
    start_index: int = 1


# The application exposes one validated circuit, not a cryptographic tuning
# surface. These constants are consumed by runtime/model code, while RunConfig
# remains strictly client-side.
DEVICE = "cuda"
LOG_N = 16
LOG_SLOTS = 15
SLOTS = 2**LOG_SLOTS
DEPTH = 30
DNUM = 3
FIRST_PRIME_BITS = 60
RESCALE_PRIME_BITS = 59
SECRET_KEY_DIST = "SPARSE_TERNARY"
SCALE_MODE = "fixed"
RESCALE_POLICY = "manual"
BOOTSTRAP_LEVEL_BUDGET = (3, 3)
BOOTSTRAP_OUTPUT_LEVELS = 14
BOOTSTRAP_STRATEGY = "normal_giant"
BOOTSTRAP_MODE = "modraise_first"
INPUT_LIMBS = 10
NUM_LAYERS = 12
SEQUENCE_LENGTH = 128
HIDDEN_SIZE = 768
PREDICTION_REL_L2_TOLERANCE = 5e-2


def layer_plans() -> tuple[LayerPlan, ...]:
    """Return the fixed arithmetic schedule for all encoder layers."""

    return tuple(LayerPlan.for_layer(index) for index in range(NUM_LAYERS))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Canonical u64 THOR BERT inference on CUDA"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="path to a Hugging Face dataset saved with save_to_disk",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_WEIGHTS_PATH,
        help="path to the THOR BERT model.safetensors file",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=DEFAULT_TOKENIZER_PATH,
        help="path to the local BERT tokenizer directory",
    )
    parser.add_argument(
        "--split",
        choices=("train", "validation", "test"),
        default="train",
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--start-index", type=int, default=1)
    args = parser.parse_args(argv)

    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.runs <= 0:
        parser.error("--runs must be positive")
    if args.start_index < 0:
        parser.error("--start-index must be non-negative")
    for option, path in (
        ("--dataset", args.dataset),
        ("--weights", args.weights),
        ("--tokenizer", args.tokenizer),
    ):
        if not path.exists():
            parser.error(
                f"{option} does not exist: {path}; "
                "run `python -m thor.assets` first"
            )
    if not args.dataset.is_dir():
        parser.error(f"--dataset must be a directory: {args.dataset}")
    if not args.weights.is_file():
        parser.error(f"--weights must be a file: {args.weights}")
    if not args.tokenizer.is_dir():
        parser.error(f"--tokenizer must be a directory: {args.tokenizer}")
    return RunConfig(
        dataset_path=args.dataset.resolve(),
        weights_path=args.weights.resolve(),
        tokenizer_path=args.tokenizer.resolve(),
        split=args.split,
        warmup=args.warmup,
        runs=args.runs,
        start_index=args.start_index,
    )
