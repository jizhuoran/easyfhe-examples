"""Assemble and run the canonical u64 ResNet20 example."""

from pathlib import Path

from .benchmark import run_dataset
from .config import parse_args
from .runtime import create_runtime


def main(argv=None):
    run_config = parse_args(argv)
    _require_assets(run_config.dataset_path, run_config.weights_path)
    client, runtime = create_runtime(run_config.weights_path)
    run_dataset(client, runtime, run_config)


def _require_assets(dataset_path, weights_path):
    missing = [
        Path(path)
        for path in (dataset_path, weights_path)
        if not Path(path).is_file()
    ]
    if missing:
        formatted = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(
            f"missing ResNet20 assets: {formatted}; "
            "run `python -m resnet20_aespa.assets` first"
        )


if __name__ == "__main__":
    main()
