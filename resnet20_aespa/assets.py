"""Download pinned weights and test inputs for the ResNet20 example."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil

from huggingface_hub import hf_hub_download

from .config import DEFAULT_ASSET_DIR


DATASET_REPO_ID = "jizhuoran/easyfhe-resnet20-cifar10"
DATASET_REVISION = "5d3eb274a45479a9561bf83fc3354220bf5354b5"
DATASET_FILENAME = "test_batch.npz"
DATASET_SHA256 = "223647d4702cc5bd4630a40d712aac594f81cdada77aa82c09cd03f515902e7c"
WEIGHTS_REPO_ID = "jizhuoran/easyfhe-resnet20-aespa"
WEIGHTS_REVISION = "409132619e8dc013ea38b1b6021aca0d1fd7c5b8"
WEIGHTS_FILENAME = "resnet20_aespa_weights.npz"
WEIGHTS_SHA256 = "b85547e40e954561fe567471569244e8a8e0bf9cae8d626e863e49bf00ef2c7f"


def download_assets(output_dir=DEFAULT_ASSET_DIR) -> tuple[Path, Path]:
    """Download and verify the exact weights and 10,000-example archive."""

    dataset_cache = Path(
        hf_hub_download(
            repo_id=DATASET_REPO_ID,
            repo_type="dataset",
            revision=DATASET_REVISION,
            filename=DATASET_FILENAME,
        )
    )
    weights_cache = Path(
        hf_hub_download(
            repo_id=WEIGHTS_REPO_ID,
            revision=WEIGHTS_REVISION,
            filename=WEIGHTS_FILENAME,
        )
    )
    _verify_sha256(dataset_cache, DATASET_SHA256)
    _verify_sha256(weights_cache, WEIGHTS_SHA256)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / DATASET_FILENAME
    weights_path = output_dir / WEIGHTS_FILENAME
    shutil.copyfile(dataset_cache, dataset_path)
    shutil.copyfile(weights_cache, weights_path)
    _verify_sha256(dataset_path, DATASET_SHA256)
    _verify_sha256(weights_path, WEIGHTS_SHA256)
    return dataset_path.resolve(), weights_path.resolve()


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
    dataset_path, weights_path = download_assets(args.output)
    print(f"test inputs: {dataset_path}")
    print(f"compact weights: {weights_path}")
    return dataset_path, weights_path


if __name__ == "__main__":
    main()
