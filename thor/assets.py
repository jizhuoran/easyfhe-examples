"""Download the pinned model, tokenizer, and MRPC data for THOR."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from datasets import DatasetDict, load_dataset, load_from_disk
from huggingface_hub import snapshot_download

from .config import DEFAULT_ASSET_DIR


MODEL_REPO_ID = "jizhuoran/easyfhe-thor-mrpc"
MODEL_REVISION = "f3bf6809fe3cb7bc94897ef875257ab707d170a8"
MODEL_SHA256 = "1a9d5a7e7afc705b74820f47d9c1e884b266037ebaa891cfd2474758ce5ccc3d"
MODEL_FILES = (
    "config.json",
    "model.safetensors",
    "tokenizer_config.json",
    "vocab.txt",
)

DATASET_REPO_ID = "glue"
DATASET_CONFIG = "mrpc"
DATASET_REVISION = "bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c"
EXPECTED_SPLITS = ("train", "validation", "test")


def download_assets(output_dir=DEFAULT_ASSET_DIR) -> tuple[Path, Path]:
    """Materialize the exact assets expected by ``thor.main``."""

    output_dir = Path(output_dir)
    model_dir = output_dir / "model"
    dataset_dir = output_dir / "mrpc"
    output_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=MODEL_REPO_ID,
        revision=MODEL_REVISION,
        allow_patterns=MODEL_FILES,
        local_dir=model_dir,
    )
    _verify_sha256(model_dir / "model.safetensors", MODEL_SHA256)

    if dataset_dir.exists():
        dataset = load_from_disk(str(dataset_dir))
        _validate_dataset(dataset)
    else:
        dataset = load_dataset(
            DATASET_REPO_ID,
            DATASET_CONFIG,
            revision=DATASET_REVISION,
        )
        _validate_dataset(dataset)
        dataset.save_to_disk(str(dataset_dir))

    return model_dir.resolve(), dataset_dir.resolve()


def _validate_dataset(dataset: DatasetDict) -> None:
    if set(dataset) != set(EXPECTED_SPLITS):
        raise ValueError(
            f"expected MRPC splits {EXPECTED_SPLITS}, got {tuple(dataset)}"
        )
    required_columns = {"sentence1", "sentence2", "label"}
    for split in EXPECTED_SPLITS:
        missing = required_columns - set(dataset[split].column_names)
        if missing:
            raise ValueError(f"MRPC {split!r} split is missing columns {missing}")


def _verify_sha256(path: Path, expected: str) -> None:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected:
        raise ValueError(
            f"SHA-256 mismatch for {path}: expected {expected}, got {actual}"
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_ASSET_DIR)
    args = parser.parse_args(argv)
    model_dir, dataset_dir = download_assets(args.output)
    print(f"model and tokenizer: {model_dir}")
    print(f"MRPC dataset: {dataset_dir}")
    return model_dir, dataset_dir


if __name__ == "__main__":
    main()
